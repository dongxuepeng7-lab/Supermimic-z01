"""Finite, frozen-weight triage after diag01; never launches policy training."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import signal
import sys
import tarfile
import time

import numpy as np

from ral_eight_hour import ROOT, atomic_json, digest, execute, gpu_lock, output_size, read

CONFIG = ROOT/'configs/ral/mechanism_probe_v1.json'
PROTOCOL = ROOT/'release/ral/mechanism_probe_protocol_v1.json'
WORKER = ROOT/'tools/ral_mechanism_probe.py'


def verify_inputs(plan):
    from partymimic.experiments.protocol import code_hash, asset_hash
    frozen = read(PROTOCOL)
    if digest(CONFIG) != frozen['config_sha256']:
        raise ValueError('Diagnostic plan changed; freeze a new version')
    for path, expected in frozen['files'].items():
        if digest(ROOT/path) != expected:
            raise ValueError(f'Diagnostic dependency differs: {path}')
    for key, actual in [('code_sha256', code_hash()), ('asset_sha256', asset_hash()),
                        ('params_sha256', digest(ROOT/plan['params'])),
                        ('response_sha256', digest(ROOT/plan['responses']))]:
        if actual != plan['inputs'][key]:
            raise ValueError(f'Frozen diag01 input differs: {key}')


def storage_check(folder, estimated_bytes, min_free_gib):
    free = shutil.disk_usage(folder).free
    reserve = 256*2**20
    required = min_free_gib*2**30 + estimated_bytes + reserve
    return dict(free_bytes=free, session_bytes=output_size(folder),
        minimum_remaining_bytes=min_free_gib*2**30, next_extraction_bytes=estimated_bytes,
        temporary_reserve_bytes=reserve, required_free_bytes=required,
        reasons=[] if free >= required else ['insufficient_free_space_including_exact_extraction_and_256MiB_reserve'])


def verified_models(source, item, destination):
    import torch
    from ral_prior_diagnose import extract_verified
    from ral_recovery import check_run
    from ral_structure import module_hashes
    from ral_eight_hour_worker import finite_weights
    folder = source/item['condition_path']
    retention = read(folder/item['retention_file'])
    if retention != item['retention'] or digest(folder/'protocol.json') != item['condition_protocol_sha256']:
        raise ValueError('Model retention/condition protocol differs from audited selection')
    archive = folder/retention['archive']
    members = extract_verified(archive, retention['archive_sha256'], destination)
    if members != retention['members']:
        raise ValueError('Extracted members differ from retained manifest')
    protocol = read(folder/'protocol.json')
    paired = None
    runs = []
    for group in ('A+', 'B'):
        run = destination/'jobs'/f"s{item['seed']}_{group.replace('+','plus')}"/'run'
        ledger, index = check_run(run, protocol)
        if (ledger['training_seed'], ledger['group'], ledger['env_steps'], ledger['protocol_sha256']) != (
            item['seed'], group, 12865536, item['condition_protocol_sha256']):
            raise ValueError('Archived run identity/budget differs')
        for name, iteration in [('initial.pt', 0), ('ckpt_005000.pt', 5000)]:
            if digest(run/name) != index[name]['sha256']:
                raise ValueError('Checkpoint index mismatch')
            ck = torch.load(run/name, map_location='cpu', weights_only=False)
            finite_weights(ck, ('wm','pi','critic') if iteration == 0 else ('wm','pi','pi_live','critic'))
            if ck['args'] != ledger['args'] or ck['iteration'] != iteration or ck['sim_steps'] != (12865536 if iteration else 0):
                raise ValueError('Actual checkpoint content differs from ledger')
            del ck
        actual = module_hashes(run/'initial.pt')
        if actual != ledger['initial_module_sha256']:
            raise ValueError('Actual initialization hash differs')
        pair = {k: actual[k] for k in ('pi','critic')}
        if paired is not None and pair != paired:
            raise ValueError('A+/B actor/critic initializations differ')
        paired = pair
        runs.append((group, run/'ckpt_005000.pt'))
    return archive, members, runs


def summarize(folder):
    state = read(folder/'status.json')
    result = dict(status=state['status'], machine=state['machine'], C_authorized=False,
        new_training_runs=0, formal_results=False, jobs=[],
        next_action='Review completed mechanism evidence before prescribing one matched training intervention; no automatic promotion')
    for job in state['jobs']:
        summary = {k:job[k] for k in ('id','status')}
        path = folder/job['output']
        if job['status'] == 'completed':
            if digest(path) != job['sha256']: raise ValueError('Probe report changed')
            report = read(path)
            if report['status'] != 'completed': raise ValueError('Incomplete worker report')
            if report['task'] == 'backend_strata':
                summary['strata'] = report['strata']
            elif report['task'] == 'derivative':
                summary['by_split_precision'] = {}
                for split in ('fit','cal'):
                    for precision in ('float32','float64'):
                        rows = [r for r in report['blocks'] if r['split']==split and r['precision']==precision]
                        summary['by_split_precision'][split+'_'+precision] = dict(blocks=len(rows), ladder=[
                            dict(factor=rows[0]['epsilon_ladder'][i]['epsilon_factor'],
                                 median_rmse_by_horizon={h:float(np.median([r['epsilon_ladder'][i]['horizons'][h]['fd_vs_ad_scaled_rmse'] for r in rows]))
                                     for h in rows[0]['epsilon_ladder'][i]['horizons']}) for i in range(len(rows[0]['epsilon_ladder']))])
            else:
                summary['fixed_CPU_batch_clipping_not_online_rate'] = {}
                for split in ('fit','cal'):
                    rows = [r for r in report['blocks'] if r['split']==split]
                    summary['fixed_CPU_batch_clipping_not_online_rate'][split] = {str(h):{
                        name:dict(batches=len(rows), would_clip_fraction=float(np.mean([r['windows'][i][name]['would_clip_at_1'] for r in rows])),
                                  median_preclip_norm=float(np.median([r['windows'][i][name]['preclip_grad_norm'] for r in rows])))
                        for name in ('per_sequence_sum','divide_by_horizon','divide_by_valid_steps')}
                        for i,h in enumerate((8,32))}
        result['jobs'].append(summary)
    atomic_json(folder/'judgment.json', result)
    return result


def package(folder):
    target = folder.parent/(folder.name+'_review.tar.gz')
    with tarfile.open(target, 'x:gz') as tar:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and 'extracted' not in path.relative_to(folder).parts and path.suffix != '.tmp':
                tar.add(path, arcname=str(Path(folder.name)/path.relative_to(folder)), recursive=False)
    target.with_suffix(target.suffix+'.sha256').write_text(f'{digest(target)}  {target.name}\n')
    return target


def run(a):
    plan = read(CONFIG)
    verify_inputs(plan)
    source = a.source or ROOT/'logs/ral_9h'/f'diag01_{a.machine}'
    if not source.is_dir():
        raise FileNotFoundError(f'缺少原实验目录：{source}；用 --source 指向原始 diag01 目录')
    # Check ALL required model archives before any work, not after a partial pair.
    if a.machine == '3090' and digest(source/'status.json') != plan['machines']['3090']['source_status_sha256']:
        raise ValueError('3090 原始会话与已审阅 diag01 不同')
    items = plan['machines'][a.machine]['archives']
    for item in items:
        archive = source/item['condition_path']/item['retention']['archive']
        if not archive.is_file():
            raise FileNotFoundError(f'缺少模型归档：{archive}；review 包不含权重，请在原训练机运行')
    a.out.mkdir(parents=True, exist_ok=False)
    cutoff = time.monotonic() + a.minutes*60
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    state = dict(status='running', machine=a.machine, source=str(source), started_unix=time.time(),
        minutes_max=a.minutes, jobs=[], protocol_sha256=digest(PROTOCOL), new_training_runs=0)
    atomic_json(a.out/'protocol.json', read(PROTOCOL))
    save = lambda: atomic_json(a.out/'status.json', state)
    save()
    lock = ROOT/'logs/ral_8h'/('.gpu_'+hashlib.sha256(os.environ.get('CUDA_VISIBLE_DEVICES','0').encode()).hexdigest()[:12]+'.lock')
    if a.machine == '3090': lock = a.out/'analysis.lock'
    try:
        with gpu_lock(lock) as fd:
            def worker(name, command):
                if cancelled[0] or time.monotonic() >= cutoff:
                    raise TimeoutError('任务截止；已有产物保留')
                job = dict(id=name, status='running', output=name+'.json')
                state['jobs'].append(job); save()
                print(f"[{a.machine}] {name} 开始（只诊断，不训练）", flush=True)
                status = execute([sys.executable, '-u', str(WORKER), *map(str, command), '--out', str(a.out/job['output'])],
                                 a.out/(name+'.log'), cutoff, cancelled, fd)
                job['status'] = status
                if status == 'completed':
                    report = read(a.out/job['output'])
                    if report.get('status') != 'completed':
                        job['status'] = status = 'failed'
                    else:
                        job['sha256'] = digest(a.out/job['output'])
                save()
                if status != 'completed': raise RuntimeError(f'{name}: {status}；查看对应 .log')
                summarize(a.out)
            if a.machine == '3090':
                state['storage'] = storage_check(a.out, 0, plan['min_free_gib']); save()
                if state['storage']['reasons']: raise RuntimeError(state['storage']['reasons'][0])
                worker('backend_strata', ['--task','backend','--session',source])
            for item in items:
                archive = source/item['condition_path']/item['retention']['archive']
                if digest(archive) != item['retention']['archive_sha256']:
                    raise ValueError('Archive hash differs from audited diag01')
                with tarfile.open(archive) as tar: needed = sum(m.size for m in tar.getmembers())
                state['storage'] = storage_check(a.out, needed, plan['min_free_gib']); save()
                if state['storage']['reasons']: raise RuntimeError(state['storage']['reasons'][0])
                dest = a.out/'extracted'/item['name']
                archive, members, models = verified_models(source, item, dest)
                atomic_json(a.out/(item['name']+'_verified_members.json'), members)
                try:
                    for group, checkpoint in models:
                        worker(item['name']+'_'+group.replace('+','plus'), ['--task',plan['machines'][a.machine]['task'],
                            '--checkpoint',checkpoint, '--params',ROOT/plan['params'], '--responses',ROOT/plan['responses'],
                            '--blocks-per-split',plan['blocks_per_split']])
                finally:
                    # Only byte-verified extracted duplicates of this new run may be removed.
                    if digest(archive) == item['retention']['archive_sha256']:
                        for name, sha in members.items():
                            path = dest/name
                            if path.suffix == '.pt' and path.is_file() and digest(path) == sha: path.unlink()
            verify_inputs(plan)
            state['status'] = 'completed'
    except Exception as exc:
        state.update(status='stopped', reason=str(exc))
        print(f'停止：{exc}', flush=True)
    finally:
        state['finished_unix'] = time.time(); save()
        try:
            summarize(a.out)
        except Exception as exc:
            state.update(status='stopped', judgment_error=str(exc)); save()
        print(f'回传文件：{package(a.out)}', flush=True)
    return 0 if state['status'] == 'completed' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--machine', choices=('3090','4090','5080'), required=True)
    r.add_argument('--source', type=Path)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--minutes', type=float, default=120)
    j = sub.add_parser('judge'); j.add_argument('--session', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'judge':
        if not (a.session/'status.json').is_file(): p.error(f'未找到 {a.session}/status.json；先确认启动日志和实际输出目录')
        result = summarize(a.session)
        print(f"{result['machine']}: {result['status']}，完成 {sum(j['status']=='completed' for j in result['jobs'])}/{len(result['jobs'])} 项；详见 judgment.json")
        return 0
    if not np.isfinite(a.minutes) or a.minutes <= 0: p.error('Positive finite --minutes required')
    return run(a)


if __name__ == '__main__':
    raise SystemExit(main())
