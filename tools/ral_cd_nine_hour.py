"""Nine-hour C/D intervention, with a GPU-free predecessor wait.

Only additive files are used; the running cd01 source/contract is unchanged.
Each machine owns its labels, matched training and evaluation. No cross-device
pooling, relaxed calibration, optimizer-resume fiction or repeated filler jobs.
"""
import argparse
from contextlib import ExitStack
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time
from types import SimpleNamespace

import ral_cd as base
from ral_eight_hour import ROOT, atomic_json, digest, read, output_size

CONFIG = ROOT / 'configs/ral/cd_nine_hour_v1.json'
BASE_CONTRACT = base.contract
ADDON_FILES = ('tools/ral_cd_nine_hour.py', 'configs/ral/cd_nine_hour_v1.json',
               'scripts/ral_3090_cd9.sh', 'scripts/ral_5080_cd9.sh')
TERMINAL_OK = {'completed', 'completed_with_blocked_windows'}


class BudgetFinished(RuntimeError):
    pass


def predecessor_gate(state):
    if state is None:
        return 'waiting_predecessor'
    if state.get('machine') != '3090':
        raise ValueError('Predecessor must be the 3090 session')
    if state.get('status') in TERMINAL_OK:
        return 'ready'
    if state.get('status') == 'running':
        return 'waiting_predecessor'
    return 'blocked_by_predecessor'


def addon_hashes():
    return {n: digest(ROOT / n) for n in ADDON_FILES}


def frozen_contract(params, motions):
    return dict(BASE_CONTRACT(params, motions), cd9_files=addon_hashes())


def training_plan(plan, candidate):
    p = copy.deepcopy(plan)
    p['training_overrides']['response_horizon'] = candidate['response_horizon']
    p['C_response_weight'] = candidate['response_weight']
    return p


def has_mixed_bins(cal):
    assigned = {r['horizon'] for r in cal['bins']}
    return cal['short'] != cal['long'] and {cal['short'], cal['long']} <= assigned


class Queue(base.Queue):
    def __init__(self, a, plan, fd, acquired):
        old = read(a.out / 'status.json') if (a.out / 'status.json').is_file() else {}
        self.previous_used = float(old.get('budget_used_seconds', 0))
        self.active_start = acquired
        self.total_seconds = plan['hours'] * 3600
        self.reserve = plan['shutdown_reserve_seconds']
        self.last_size_check = 0.
        if old and old.get('source_inbox') != str(a.inbox):
            raise ValueError('Continuation inbox differs')
        super().__init__(a, plan, fd)
        self.state.update(source_inbox=str(a.inbox), predecessor=str(a.after) if a.after else None,
                          budget_hours=plan['hours'],
                          budget_semantics='active wall time after predecessor success and GPU lock; waiting excluded; reserve included')
        for key in ('models', 'evaluations', 'windows', 'calibration', 'responses', 'training_diagnostics'):
            self.state.setdefault(key, {})
        self.save()

    def used(self):
        return self.previous_used + time.monotonic() - self.active_start

    def save(self):
        self.state['budget_used_seconds'] = self.used()
        super().save()

    def guard(self, _pid=None):
        super().guard(_pid)
        if self.used() >= self.total_seconds - self.reserve:
            raise BudgetFinished('九小时预算进入收尾阶段；完整任务保留，未完成任务不计入配对结果')
        if time.monotonic() - self.last_size_check >= 30:
            self.last_size_check = time.monotonic()
            size = output_size(self.a.out)
            self.state['output_gib'] = size / 2**30
            if size >= self.plan['max_output_gib'] * 2**30:
                raise RuntimeError('Output cap reached; completed models retained')
            self.save()

    def job(self, name, build, check=None, allow_blocked=False):
        self.guard()
        normal = self.plan['job_timeout_hours']
        remaining = self.total_seconds - self.reserve - self.used()
        self.plan['job_timeout_hours'] = min(normal, max(0., remaining) / 3600)
        try:
            return super().job(name, build, check, allow_blocked)
        except RuntimeError:
            if self.used() >= self.total_seconds - self.reserve - 2:
                raise BudgetFinished('九小时预算截止，最后任务可能只留下中间检查点') from None
            raise
        finally:
            self.plan['job_timeout_hours'] = normal


def import_seed(q, seed):
    archive = q.a.inbox / f"{q.plan['source_session']}_s{seed}.tar.gz"
    expected = q.plan['source_archive_sha256'][str(seed)]
    if digest(archive) != expected:
        raise ValueError(f'Original cd01 handoff differs: {archive}')
    # The input handoff carries the original contract, before additive cd9 tools.
    old_contract = {k: v for k, v in q.state['contract'].items() if k != 'cd9_files'}
    proxy = SimpleNamespace(a=SimpleNamespace(out=q.a.out, session=q.plan['source_session']),
                            state={'contract': old_contract})
    return base.unpack_handoff(proxy, archive, seed)


def checkpoint(run, iterations=6000):
    path = run / f'ckpt_{iterations:06d}.pt'
    return path if path.is_file() else run / 'last.pt'


def evaluate(q, seed, variant, scenario):
    run = q.a.out / q.state['models'][str(seed)][variant]
    ck = checkpoint(run, q.plan['training_overrides']['iterations'])
    name = f's{seed}_eval_{scenario}_{variant}'
    def check(path):
        result = read(path)
        if len(result['episodes']) != len(q.state['contract']['motions']):
            raise ValueError('Incomplete reference evaluation')
        if any(e['nonfinite_steps'] for e in result['episodes']):
            raise ValueError('Nonfinite evaluation')
        result.update(stage=q.plan['stage'], variant=variant, training_seed=seed,
                      scope=q.plan['evaluation']['scope'],
                      checkpoint_weights='live' if variant in ('D', 'C_perm', 'C_fixed8', 'C_fixed32') else 'EMA')
        atomic_json(path, result)
    path = q.job(name, lambda folder: (
        [sys.executable, '-u', ROOT / 'tools/ral_stress_eval.py', '--run', run,
         '--checkpoint', ck, '--params', q.a.params, '--motions', q.a.motions,
         '--scenario', scenario, '--out', folder / 'evaluation.json'],
        folder / 'evaluation.json'), check=check)
    q.state['evaluations'][name] = dict(seed=seed, variant=variant, scenario=scenario,
                                      result=str(path.relative_to(q.a.out)))
    q.save()


def response_job(q, seed, variant, pairs):
    run = q.a.out / q.state['models'][str(seed)][variant]
    path = q.job(f's{seed}_{variant}_cal_response', lambda folder: q.worker(
        'responses', checkpoint(run), seed, folder, folder / 'response.json', responses=pairs / 'manifest.json'))
    q.state['responses'][f's{seed}_{variant}'] = str(path.relative_to(q.a.out))
    q.save()



def training_diagnostics(run):
    args = read(run / 'args.json')
    with (run / 'native_response_metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    result = dict(response_updates=len(rows), fit_eligible=args['response_fit_eligible_blocks'],
                  fit_total=args['response_fit_total_blocks'],
                  tail_response_huber=statistics.mean(float(r['response_loss']) for r in rows[-500:]),
                  scope='last 500 changing-checkpoint fit batches; not final calibration or task success')
    if args['experiment_group'] == 'C':
        ratios = [args['wm_window'] * float(r['effective_weight']) * float(r['response_grad_norm']) /
                  max(float(r['forward_grad_norm']), 1e-30) for r in rows]
        result.update(weighted_response_gradient_ratio_median=statistics.median(ratios),
                      weighted_response_gradient_ratio_max=max(ratios),
                      cap_active_updates=sum(float(r['effective_weight']) < .999 * float(r['scheduled_weight']) for r in rows),
                      gradient_scope='before common clipping/Muon; logged effective weight multiplied by WM window')
    return result


def train_seed(q, seed):
    data = import_seed(q, seed)
    parent = data / 'models/B_parent/model.pt'
    # Generate on the actual GPU. The old RTX 5080 labels are not training labels on 3090.
    pairs = q.job(f's{seed}_native_pairs', lambda folder: q.worker(
        'pairs', parent, seed, folder, folder / 'pairs', split_manifest=data / 'pairs/manifest.json'))
    profile = q.plan['machines'][q.a.machine]
    candidates = profile['candidates']
    shift = q.plan['seeds'].index(seed) % len(candidates)
    ordered = candidates[shift:] + candidates[:shift]
    models = q.state['models'].setdefault(str(seed), {})
    initial = None
    for candidate in ordered:
        p = training_plan(q.plan, candidate)
        group, variant = candidate['group'], candidate['name']
        run = q.job(f's{seed}_train_{variant}', lambda folder: (
            base.train_command(parent, group, pairs / 'manifest.json', folder / 'run', q.a.params, q.a.motions, p),
            folder / 'run'), check=lambda run: base.verify_training(run, parent, p, group))
        ledger = read(run / 'run.json')
        modules = ledger['initial_module_sha256']
        if initial is not None and modules != initial:
            raise ValueError('Within-seed initial WM/actor/critic mismatch')
        initial = modules
        models[variant] = str(run.relative_to(q.a.out))
        q.state['training_diagnostics'][f's{seed}_{variant}'] = training_diagnostics(run)
        q.save()
    for c in candidates:
        response_job(q, seed, c['name'], pairs)
    lead = profile['window_parent']
    C = checkpoint(q.a.out / models[lead])
    cal = q.job(f's{seed}_calibration', lambda folder: q.worker(
        'calibrate', C, seed, folder, folder / 'calibration.json', responses=pairs / 'manifest.json'))
    q.state['calibration'][str(seed)] = str(cal.relative_to(q.a.out))
    q.state['seeds'][str(seed)] = dict(status='C_trained_and_calibrated', window_parent=lead)
    q.save()
    # Obtain complete paired outcomes early, before committing the whole night to long training.
    for scenario in q.plan['evaluation']['priority_scenarios']:
        for c in candidates[shift:] + candidates[:shift]:
            evaluate(q, seed, c['name'], scenario)
    summarize(q)


def verify_windows(q, seed):
    variants = ('D', 'C_perm', 'C_fixed8', 'C_fixed32')
    runs = {n: q.a.out / q.state['models'][str(seed)][n] for n in variants}
    records = {n: read(r / 'run.json') for n, r in runs.items()}
    for key in ('init_sha256', 'start_schedule_sha256'):
        if len({r[key] for r in records.values()}) != 1:
            raise ValueError(f'Window control pairing differs: {key}')
    for key in ('assigned_imagined_steps', 'computed_imagined_steps'):
        if records['D'][key] != records['C_perm'][key]:
            raise ValueError(f'D/permuted budget differs: {key}')
    histograms = []
    for name in ('D', 'C_perm'):
        with (runs[name] / 'train_metrics.csv').open() as f:
            histograms.append([tuple(r[k] for k in ('iteration', 'assigned_steps', 'computed_steps', 'short_count'))
                               for r in csv.DictReader(f)])
    if histograms[0] != histograms[1]:
        raise ValueError('D/permuted per-update histogram differs')


def window_seed(q, seed):
    cal = q.a.out / q.state['calibration'][str(seed)]
    if not has_mixed_bins(read(cal)):
        q.state['windows'][str(seed)] = dict(status='blocked_degenerate_calibration',
            reason='No calibrated mix of 8/32 bins; D, permutation and redundant fixed controls not launched')
        q.save()
        print(f'[{q.a.machine}] s{seed} D：校准没有长短窗口变化，保留 blocked，不放宽阈值', flush=True)
        return
    lead = q.plan['machines'][q.a.machine]['window_parent']
    C = checkpoint(q.a.out / q.state['models'][str(seed)][lead])
    pairs = q.a.out / q.state['jobs'][f's{seed}_native_pairs']['result']
    starts = q.job(f's{seed}_starts', lambda folder: q.worker(
        'starts', C, seed, folder, folder / 'starts', responses=pairs / 'manifest.json'))
    def window(name, group, fixed=None):
        opts = dict(calibration=cal, starts=starts / 'manifest.json', group=group)
        if fixed is not None:
            opts['fixed_horizon'] = fixed
        def check(run):
            d = read(run / 'run.json')
            if d['status'] != 'completed' or d['iterations'] != q.plan['windows']['iterations']:
                raise ValueError('Incomplete window training')
            if not d['frozen_world_model'] or not d['frozen_critic']:
                raise ValueError('Window model/critic changed')
        run = q.job(f's{seed}_window_{name}', lambda folder: q.worker(
            'window', C, seed, folder, folder / 'run', **opts), check=check, allow_blocked=True)
        if run is not None:
            q.state['models'][str(seed)][name] = str(run.relative_to(q.a.out))
            q.save()
        return run
    if window('D', 'D') is None:
        q.state['windows'][str(seed)] = dict(status='blocked_degenerate_calibration',
            reason='No short/long mixture on actual fit starts; other window branches skipped')
        q.save()
        return
    controls = [('C_perm', 'C-perm', None), ('C_fixed8', 'D', 8), ('C_fixed32', 'D', 32)]
    shift = q.plan['seeds'].index(seed) % len(controls)
    for args in controls[shift:] + controls[:shift]:
        if window(*args) is None:
            raise ValueError('A paired control unexpectedly blocked after D completed')
    verify_windows(q, seed)
    q.state['windows'][str(seed)] = dict(status='completed', pairing_verified=True, parent=lead)
    q.save()
    for scenario in q.plan['evaluation']['priority_scenarios']:
        for variant in ('D', 'C_perm', 'C_fixed32', 'C_fixed8'):
            evaluate(q, seed, variant, scenario)
    summarize(q)


def summarize(q):
    cells = {}
    for e in q.state['evaluations'].values():
        d = read(q.a.out / e['result'])
        value = 100 * sum(bool(r['complete']) for r in d['episodes']) / len(d['episodes'])
        cells.setdefault(str(e['seed']), {}).setdefault(e['variant'], {})[e['scenario']] = value
    candidates = q.plan['machines'][q.a.machine]['candidates']
    comparisons = [(r['name'], 'B') for r in candidates if r['group'] == 'C']
    comparisons += [('D', 'C_perm'), ('D', 'C_fixed32'), ('D', 'C_fixed8')]
    full = {'nominal', *q.plan['evaluation']['remaining_scenarios']}
    result = dict(status=q.state['status'], machine=q.a.machine, stage=q.plan['stage'],
                  budget_hours=q.plan['hours'], active_seconds=q.used(), formal_results=False,
                  response_test_accessed=False, completion_percent=cells, windows=q.state['windows'],
                  training_diagnostics=q.state['training_diagnostics'],
                  cal_responses={k:read(q.a.out / v)['means'] for k,v in q.state['responses'].items()},
                  paired_effects={}, interpretation='development; no cross-machine pooling; missing/partial is not zero')
    for a,b in comparisons:
        summaries = {}
        for label,scenarios in [('identified',{'identified'}), ('nominal',{'nominal'}), ('five_nonreference_mean',full)]:
            values = []
            for seed,c in cells.items():
                if scenarios <= set(c.get(a, {})) and scenarios <= set(c.get(b, {})):
                    values.append(dict(seed=int(seed), difference_pp=statistics.mean(c[a][s]-c[b][s] for s in scenarios)))
            diffs = [v['difference_pp'] for v in values]
            summaries[label] = dict(n=len(values), seeds=values,
                mean_pp=statistics.mean(diffs) if diffs else None,
                sample_sd_pp=statistics.stdev(diffs) if len(diffs)>1 else None)
        result['paired_effects'][a+'_minus_'+b] = summaries
    atomic_json(q.a.out / 'summary.json', result)


def run_work(q):
    for seed in q.plan['seeds']:
        train_seed(q, seed)
    for seed in q.plan['seeds']:
        window_seed(q, seed)
    for scenario in q.plan['evaluation']['remaining_scenarios']:
        for seed in q.plan['seeds']:
            variants = [c['name'] for c in q.plan['machines'][q.a.machine]['candidates']]
            if q.state['windows'].get(str(seed), {}).get('status') == 'completed':
                variants += ['D', 'C_perm', 'C_fixed32', 'C_fixed8']
            for variant in variants:
                evaluate(q, seed, variant, scenario)
            summarize(q)
    q.state['status'] = 'completed_with_blocked_windows' if any(
        r['status'].startswith('blocked') for r in q.state['windows'].values()) else 'completed'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--machine', required=True, choices=('3090','5080'))
    p.add_argument('--session', default='cd02')
    p.add_argument('--inbox', type=Path)
    p.add_argument('--after', type=Path, help='3090 predecessor directory or status.json; waiting excluded from budget')
    p.add_argument('--continue-session', action='store_true')
    p.add_argument('--status', action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.session):
        p.error('Unsafe session name')
    os.chdir(ROOT)
    a.out=ROOT/'logs/ral_cd9'/f'{a.session}_{a.machine}'
    a.inbox=(a.inbox or (ROOT/'data/ral_cd_inbox' if a.machine=='3090' else ROOT/'logs/ral_cd/cd01_5080/handoff')).resolve()
    if a.machine=='3090':
        a.after=(a.after or ROOT/'logs/ral_cd/cd01_3090/status.json').resolve()
        if a.after.is_dir(): a.after=a.after/'status.json'
    elif a.after:
        p.error('--after is only for the 3090 predecessor')
    a.params=ROOT/'release/ral/params.csv'
    a.motions=ROOT/'data/motions/z01_lafan_v2_clean'
    launch_dir=ROOT/'logs/ral_cd9/launchers'
    launch_file=launch_dir/f'{a.session}_{a.machine}.json'
    if a.status:
        for path in (launch_file,a.out/'status.json',a.out/'summary.json'):
            if path.is_file():
                d=read(path)
                print(path)
                print(json.dumps({k:v for k,v in d.items() if k not in ('contract','jobs','models','evaluations','handoffs','completion_percent')},ensure_ascii=False,indent=2))
        return
    plan=read(CONFIG)
    if plan['hours']!=9 or plan['shutdown_reserve_seconds']<30:
        raise ValueError('Expected fixed nine-hour plan and shutdown reserve')
    if a.out.exists() and not a.continue_session:
        p.error('Session exists; use --continue-session to use its remaining active budget')
    launch_dir.mkdir(parents=True,exist_ok=True)
    with (launch_dir/f'{a.session}_{a.machine}.lock').open('a+') as launcher_lock:
        try:fcntl.flock(launcher_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('This launcher is already running; do not duplicate it')
        cancel=[False]
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:cancel.__setitem__(0,True))
        frozen_addon=addon_hashes()
        launch=dict(status='preparing',pid=os.getpid(),machine=a.machine,session=a.session,
                    predecessor=str(a.after) if a.after else None,waiting_counts_toward_budget=False,
                    created_unix=time.time(),command=sys.argv,addon_files=frozen_addon)
        def update(status,**extra):
            changed=launch.get('status')!=status
            launch.update(status=status,updated_unix=time.time(),**extra)
            atomic_json(launch_file,launch)
            if changed:print(f'[{a.machine}] {status}',flush=True)
        try:
            while a.after:
                if cancel[0]:raise InterruptedError('Cancelled while waiting')
                state=read(a.after) if a.after.is_file() else None
                gate=predecessor_gate(state)
                if gate=='ready':break
                update(gate,predecessor_status=state.get('status') if state else None)
                time.sleep(15)
            visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
            gpu_path=ROOT/'logs/ral_8h'/('.gpu_'+hashlib.sha256(visible.encode()).hexdigest()[:12]+'.lock')
            with ExitStack() as stack:
                while True:
                    if cancel[0]:raise InterruptedError('Cancelled before GPU acquisition')
                    try:fd=stack.enter_context(base.gpu_lock(gpu_path));break
                    except RuntimeError:
                        update('waiting_gpu_lock');time.sleep(15)
                acquired=time.monotonic()
                if addon_hashes()!=frozen_addon:raise ValueError('Add-on changed while waiting; restart the launcher')
                update('starting',gpu_acquired_unix=time.time())
                import torch
                if not torch.cuda.is_available() or a.machine not in torch.cuda.get_device_name():
                    raise ValueError(f'Requires RTX {a.machine}')
                base.CONFIG=CONFIG
                base.contract=frozen_contract
                q=Queue(a,plan,fd,acquired)
                update('running',queue_status=str(a.out/'status.json'))
                try:
                    run_work(q)
                except BudgetFinished as exc:
                    q.state.update(status='budget_exhausted',reason=str(exc))
                except BaseException as exc:
                    q.state.update(status='stopped',reason=str(exc))
                    raise
                finally:
                    q.state['finished_unix']=time.time();q.save()
                    try:
                        summarize(q);base.review_package(q)
                    except Exception as exc:
                        q.state['review_package_error']=str(exc);q.save()
                    update(q.state['status'],reason=q.state.get('reason'),active_seconds=q.used())
        except BaseException as exc:
            update('stopped',reason=str(exc));raise
    print(f'[{a.machine}] {a.out}/status.json',flush=True)


if __name__=='__main__':
    main()
