"""Read-only audit of the returned three-hour intervention and local backend logs.

Never changes the frozen protocol, imports weights, launches training, or opens
the response test split. Review archives contain metadata, not model tensors.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import t

from ral_eight_hour import ROOT, atomic_json, digest, read
from ral_intervention import block_result, blocks, condition_config, dependencies
from ral_recovery import check_run, current_inputs, evaluation_summary


def require(value, message):
    if not value:
        raise ValueError(message)


def summary(values):
    values = list(map(float, values))
    require(values and all(math.isfinite(v) for v in values), 'Invalid summary input')
    return dict(n=len(values), mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                values=values)


def interval(values):
    s = summary(values)
    margin = float(t.ppf(.975, s['n']-1))*s['sample_sd']/math.sqrt(s['n'])
    s['approximate_95pct_t_interval'] = [s['mean']-margin, s['mean']+margin]
    s['interval_scope'] = 'Exploratory paired-seed t interval; five training seeds, no clip-level pseudoreplication or multiplicity adjustment'
    return s


def local_time(unix):
    return datetime.fromtimestamp(unix, ZoneInfo('Asia/Shanghai')).isoformat()


def audit_archive(archive, folder):
    expected=archive.with_suffix(archive.suffix+'.sha256').read_text().split()
    require(len(expected)==2 and expected[1]==archive.name and digest(archive)==expected[0], 'Archive checksum mismatch')
    seen=set()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            p=PurePosixPath(member.name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts
                    and p.parts[0]==folder.name and member.name not in seen, 'Unsafe archive member')
            seen.add(member.name)
            h=hashlib.sha256()
            with tar.extractfile(member) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):
                    h.update(chunk)
            require(digest(folder.parent/member.name)==h.hexdigest(), 'Extracted archive member changed')
    return dict(path=str(archive),sha256=expected[0],bytes=archive.stat().st_size,
                verified_regular_members=len(seen))


def audit_5080(folder):
    from partymimic.experiments.initialization import stream_seeds
    from ral_backend_diagnose import select_blocks
    from ral_mechanism_probe import FACTORS, HORIZONS
    state, plan, protocol = (read(folder/name) for name in ('status.json', 'config.json', 'protocol.json'))
    require(state['machine']=='5080' and not state['smoke_only'], 'Not the intended full run')
    require(plan==read(ROOT/'configs/ral/intervention_three_hour_v1.json'), 'Plan changed')
    require(protocol==read(ROOT/'release/ral/intervention_three_hour_protocol_v1.json'), 'Protocol changed')
    require(protocol['tools']==dependencies(), 'Frozen tool inventory changed')
    frozen = read(folder/'inputs.json')
    require(frozen==current_inputs(ROOT/'data/motions/z01_lafan_v2_clean', ROOT/'release/ral/params.csv',
                                  ROOT/'data/ral_pairs_v1/manifest.json'), 'Local input inventory differs')
    expected_blocks = {}
    manifest = read(ROOT/'data/ral_pairs_v1/manifest.json')
    for split in ('fit', 'cal'):
        entry = manifest['splits'][split]
        p = ROOT/'data/ral_pairs_v1'/entry['file']
        require(digest(p)==entry['sha256'], 'Response split hash changed')
        with np.load(p, allow_pickle=False) as z:
            expected_blocks[split] = {str(v) for v in z['block_id'][select_blocks(z['block_id'],24)]}
    result = dict(status=state['status'], started=local_time(state['started_unix']),
                  finished=local_time(state['finished_unix']), wall_seconds=state['finished_unix']-state['started_unix'],
                  formal_results=False, response_test_accessed=False, C_authorized=False,
                  model_tensor_bytes_verified_locally=False,
                  model_verification_scope='Returned metadata and retained-member hashes checked; full model archives remain on 5080 and are absent from this review package',
                  scope='Final EMA, identified reference plant, all 163 training-library clips, repeat 0 primary; repeat 1 reproducibility only',
                  complete_seeds=[], models=[], partial=[], checks={})
    expected_schedule = list(blocks(plan))
    compared_retained_files = 0
    for index, block in enumerate(state['blocks']):
        expected = expected_schedule[index]
        require(block['seed']==expected['seed'], 'Seed order changed')
        require([c['id'] for c in block['conditions']]==[c['id'] for c in expected['conditions']], 'Variant order changed')
        if block['status']!='completed':
            for c in block['conditions']:
                partial = dict(seed=block['seed'], variant=c['id'], status=c['status'], included_in_primary=False)
                if 'folder' in c:
                    p = folder/c['folder']/f"jobs/s{block['seed']}_{c['id']}/run/run.json"
                    if p.exists():
                        d=read(p)
                        partial.update({k:d.get(k) for k in ('status','wm_updates','policy_updates','env_steps')})
                result['partial'].append(partial)
            continue
        result['complete_seeds'].append(block['seed'])
        require(block_result(block) is not None, 'Incomplete primary triplet')
        for c in block['conditions']:
            cell = folder/c['folder']
            p = read(cell/'protocol.json')
            require(read(cell/'config.json')==condition_config(plan,c), 'Condition config mismatch')
            expected_p = dict(protocol, training=condition_config(plan,c)['training'],
                              training_seeds=[block['seed']], config_sha256=digest(cell/'config.json'), variant=c['id'])
            require(p==expected_p, 'Condition protocol mismatch')
            run = cell/f"jobs/s{block['seed']}_{c['id']}/run"
            ledger, checkpoints = check_run(run,p)
            require((ledger['group'],ledger['training_seed'],ledger['env_steps'])==
                    (c['group'],block['seed'],12865536), 'Wrong training identity/budget')
            require(ledger['protocol_sha256']==digest(cell/'protocol.json'), 'Ledger protocol hash differs')
            require(ledger['initial_module_sha256']==c['initial_module_sha256']==block['paired_initial_module_sha256'], 'Unmatched initial tensors')
            require(ledger['random_streams']['seeds']==stream_seeds(block['seed']), 'Random streams differ')
            require(ledger['args']['experiment_model']==('matched' if c['id']=='Aplus' else 'actuator'), 'Unexpected model implementation')
            require(ledger['args']['friction_speed']==c['friction_speed'], 'Wrong smoothing')
            require(ledger['args']['resume'] is None and ledger['args']['init'] is None, 'Not a fresh run')
            retention = read(cell/(c['id']+'_retention.json'))
            require(retention==c['retention'] and retention['verified_before_cleanup'], 'Retention metadata mismatch')
            ckname = f"jobs/s{block['seed']}_{c['id']}/run/ckpt_005000.pt"
            require(retention['members'][ckname]==c['checkpoint_sha256']==checkpoints['ckpt_005000.pt']['sha256'], 'Final checkpoint hash mismatch')
            for name, fingerprint in retention['members'].items():
                if Path(name).suffix!='.pt':
                    require(digest(cell/name)==fingerprint, f'Retained metadata changed: {name}')
                    compared_retained_files+=1
            with (run/'train_metrics.csv').open() as f:
                metrics=list(csv.DictReader(f))
            require([int(r['iteration']) for r in metrics]==list(range(5001)), 'Missing iterations')
            require(all(math.isfinite(float(r[k])) for r in metrics[1:] for k in ('wm_loss','policy_loss')), 'Nonfinite loss')
            require([int(metrics[-1][k]) for k in ('policy_updates','wm_updates','env_steps')]==[5000,25000,12865536], 'CSV budget differs')
            with (run/'train_log.csv').open() as f:
                logs=list(csv.DictReader(f))
            require(all(float(r['nonfinite'])==0 for r in logs), 'Skipped nonfinite training updates')
            require(all(math.isfinite(float(r[k])) for r in logs for k in ('wm_loss','pi_loss','v_loss','pi_gradnorm')), 'Nonfinite training log')
            rows=[]; termination=[]; errors=[]; episodes=[]
            for repeat in (0,1):
                ev=read(run.parent/f'eval_r{repeat}.json')
                provenance=read(run.parent/f'eval_r{repeat}.inputs.json')
                require(provenance['inputs']==frozen and provenance['checkpoint_sha256']==c['checkpoint_sha256'], 'Evaluation provenance mismatch')
                require((provenance['plant'],provenance['checkpoint_weights'],provenance['eval_seed'])==('identified','EMA',0), 'Evaluation setting differs')
                row=dict(evaluation_summary(ev,frozen['motions']),repeat_check=repeat)
                require(row==c['rows'][repeat] and ev['plant']=='identified', 'Episode totals disagree')
                rows.append(row)
                causes=read(run.parent/f'eval_r{repeat}.termination.json')
                require(sum(causes['counts'].values())==163 and causes['counts']['reference_complete']==row['completed'], 'Termination counts disagree')
                termination.append(causes['counts'])
                episodes.append(ev['episodes'])
                errors.append(dict(q_rmse_rad=math.sqrt(sum(e['q_sqerr_sum_rad2'] for e in ev['episodes'])/sum(e['q_valid_count'] for e in ev['episodes'])),
                                   norm_return=ev['norm_return']))
            diagnostics={}
            for kind in ('predictions','derivative'):
                record=c[kind]; require(digest(folder/record['path'])==record['sha256'], 'Diagnostic hash mismatch')
                d=read(folder/record['path'])
                require(d['checkpoint_sha256']==c['checkpoint_sha256'] and not d['response_test_accessed'], 'Diagnostic provenance differs')
                diagnostics[kind]=d
            pred=diagnostics['predictions']; deriv=diagnostics['derivative']
            require(all(x['valid'] for x in pred['splits'].values()), 'Invalid forward diagnostic')
            require(deriv['status']=='completed' and deriv['optimizer_steps']==0 and deriv['weights_and_normalizers_unchanged'], 'Invalid derivative diagnostic')
            require(len(deriv['blocks'])==96, 'Wrong derivative coverage')
            ds={}
            for split in ('fit','cal'):
                for precision in ('float32','float64'):
                    rr=[r for r in deriv['blocks'] if r['split']==split and r['precision']==precision]
                    require(len(rr)==24 and {r['block_ids'][0] for r in rr}==expected_blocks[split], 'Derivative sample selection differs')
                    for r in rr:
                        require(r['actual_parameter_dtype']=='torch.'+precision, 'Wrong arithmetic dtype')
                        require(tuple(v['epsilon_factor'] for v in r['epsilon_ladder'])==FACTORS, 'Amplitude ladder differs')
                        require(all(set(v['horizons'])=={str(h) for h in HORIZONS} for v in r['epsilon_ladder']), 'Horizons differ')
                    ds[split+'_'+precision]={}
                    for h in ('8','32'):
                        a=np.array([[v['horizons'][h]['fd_vs_ad_scaled_rmse'] for v in r['epsilon_ladder']] for r in rr])
                        require(np.isfinite(a).all(), 'Nonfinite derivative data')
                        ds[split+'_'+precision][h]=dict(median_rmse_by_factor=np.median(a,axis=0).tolist(),
                            final_lower_count=int((a[:,-1]<a[:,0]).sum()), final_max=float(a[:,-1].max()),
                            median_relative_error_factor1=statistics.median(r['epsilon_ladder'][0]['horizons'][h]['relative_error'] for r in rr))
            result['models'].append(dict(seed=block['seed'],variant=c['id'],checkpoint_sha256=c['checkpoint_sha256'],
                training_seconds=ledger['finished_unix']-ledger['started_unix'], rows=rows,
                repeat_completion_spread_pp=100*abs(rows[1]['completion']-rows[0]['completion']),
                termination=termination,valid_frame_metrics=errors,predictions=pred['splits'],derivative=ds,
                primary_failed_clips=[e['motion_id'] for e in episodes[0] if not e['complete']]))
    primary=[block_result(b) for b in state['blocks'] if block_result(b) is not None]
    judgment=read(folder/'judgment.json')
    require(primary==judgment['results'], 'Stored paired results disagree')
    result['groups']={}
    for variant in ('Aplus','B_ref','B_v005'):
        rr=[m for m in result['models'] if m['variant']==variant]
        counts=Counter(k for m in rr for k in m['primary_failed_clips'])
        result['groups'][variant]=dict(completion_pct=summary(m['rows'][0]['completion']*100 for m in rr),
            repeat1_completion_pct=summary(m['rows'][1]['completion']*100 for m in rr),
            persistent_failed_clips=[k for k,v in sorted(counts.items()) if v==len(rr)],
            termination_counts=dict(sum((Counter(m['termination'][0]) for m in rr),Counter())),
            cal_q_rmse_mrad={h:summary(1000*m['predictions']['cal']['q_rmse_rad'][h] for m in rr) for h in ('8','32')},
            cal_fp64_derivative_median_factor1={h:summary(m['derivative']['cal_float64'][h]['median_rmse_by_factor'][0] for m in rr) for h in ('8','32')})
    result['paired_effects_pp']={}
    for name in ('B_v005_minus_B_ref_pp','B_ref_minus_Aplus_pp','B_v005_minus_Aplus_pp'):
        values=[r[name] for r in primary]
        require(abs(statistics.mean(values)-judgment['effects'][name]['mean'])<1e-10, 'Stored effect differs')
        result['paired_effects_pp'][name]=dict(interval(values),seeds=result['complete_seeds'],
            positive_seeds=[b['seed'] for b in primary if b[name]>1e-12],
            negative_seeds=[b['seed'] for b in primary if b[name]<-1e-12],
            tied_seeds=[b['seed'] for b in primary if abs(b[name])<=1e-12])
    result['checks']=dict(frozen_protocol_and_tools_match=True,current_inputs_match=True,
        matched_initial_wm_actor_critic=True,full_training_and_finite_logs=True,
        checked_retained_metadata_files=compared_retained_files,checked_models=len(result['models']),
        checked_evaluations=2*len(result['models']),all_163_clips=True,
        max_repeat_completion_spread_pp=max(m['repeat_completion_spread_pp'] for m in result['models']))
    return result


def audit_3090(folder):
    state=read(folder/'status.json')
    require(state['machine']=='3090' and not state['smoke_only'], 'Not the full 3090 run')
    require(read(folder/'protocol.json')==read(ROOT/'release/ral/intervention_three_hour_protocol_v1.json'), '3090 protocol differs')
    selection=read(ROOT/'release/ral/precision_selection_v1.json')
    expected={(s,r['block_id']) for s in ('fit','cal') for r in selection[s]}
    result=dict(status=state['status'], started=local_time(state['started_unix']), finished=local_time(state['finished_unix']),
        wall_seconds=state['finished_unix']-state['started_unix'],formal_results=False,C_authorized=False,
        scope='48 failure-enriched blocks reused across directions and processes; not population failure rates or independent training samples',
        completed_cells=[],incomplete_cells=[],verified_raw_original_blocks=0)
    for cell in state['backend']:
        row=dict(id=cell['id'],status=cell['status'],processes=[])
        for job in cell['jobs']:
            p=folder/job['folder']/'summary.json'
            d=read(p)
            if job['status']!='completed':
                row['processes'].append(dict(repeat=job['repeat'],status=job['status'],finished_blocks=len(d['blocks'])))
                continue
            require(digest(p)==job['sha256'] and d['status']=='completed', '3090 completed report changed')
            require({(b['split'],b['block_id']) for b in d['blocks']}==expected and len(d['blocks'])==48, 'Backend selection mismatch')
            require(d['selection_sha256']==digest(ROOT/'release/ral/precision_selection_v1.json'), 'Selection checksum mismatch')
            require(not d['response_test_accessed'] and not d['C_authorized'], 'Unexpected backend scope')
            failures={case:sum(any(not x['passed'] for x in b['comparisons'][case]) for b in d['blocks'])
                      for case in ('native','round_state','round_ctrl','round_both')}
            require(failures==job['compared_case_failures'], 'Stored backend counts disagree')
            raw_trace=[]
            if cell['id']=='original' and job['repeat']==0:
                files={digest(p):p for p in p.parent.glob('*_trajectories.npz')}
                for b in d['blocks']:
                    require(b['trajectory_sha256'] in files, 'Missing/changed original raw trajectory')
                    with np.load(files[b['trajectory_sha256']],allow_pickle=False) as z:
                        error=np.sqrt(np.mean(((z['gpu_substeps'][0]-z['native_substeps'][0])/z['scales'])**2,axis=-1))
                        contacts=np.any(z['native_active_contacts'][0]!=z['gpu_active_contacts'][0],axis=-1)
                        constraint=z['native_stats'][0,:,0]!=z['gpu_stats'][0,:,0]
                        def first(v):
                            hits=np.flatnonzero(v)
                            return int(hits[0]+1)*5 if len(hits) else None
                        raw_trace.append(dict(block_id=b['block_id'],split=b['split'],baseline_first_scaled_forward_over_point1_ms=first(error>.1),
                            baseline_first_active_geom_pair_mismatch_ms=first(contacts),baseline_first_constraint_count_mismatch_ms=first(constraint)))
                result['verified_raw_original_blocks']=len(raw_trace)
                result['original_unperturbed_branch_substep_diagnostics']=raw_trace
                result['substep_scope']='5ms sampling; q/qd are postintegration, contacts computed before integration. Timing co-occurrence is not causal attribution. Only branch 0, original direction/process 0.'
            row['processes'].append(dict(repeat=job['repeat'],status='completed',finished_blocks=48,
                failure_counts=failures,failures_by_horizon={str(h):sum(any(not x['passed'] for x in b['comparisons']['native'] if x['horizon']==h) for b in d['blocks']) for h in (1,2,4,8,16,32)},
                max_cpu_original_replay=max(b['cpu_original_replay_max_abs'] for b in d['blocks']),
                max_transfer_vs_rounded_cpu=max(b['gpu_transfer_vs_rounded_cpu_max_abs'] for b in d['blocks']),
                max_initial_state_transfer=max(b['initial_state_transfer_max_abs'] for b in d['blocks']),
                overflow_blocks=sum(b['gpu_overflow_any'] for b in d['blocks']),
                active_contact_mismatch_blocks=sum(b['contact_mismatch_fraction']['native']>0 for b in d['blocks']),
                constraint_mismatch_blocks=sum(b['constraint_count_mismatch_fraction']['native']>0 for b in d['blocks'])))
        if cell['status']=='completed':
            p=folder/'backend'/cell['id']/'cross_process.json'
            require(digest(p)==cell['cross_process']['sha256'], 'Cross-process report hash differs')
            cross=read(p);failed=[b for b in cross['blocks'] if not b['passed']]
            require(len(cross['blocks'])==48 and len(failed)==cell['cross_process']['failed'], 'Cross-process counts differ')
            row['cross_process_failed_blocks']=[b['block_id'] for b in failed]
            result['completed_cells'].append(row)
        else:
            result['incomplete_cells'].append(row)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,default=ROOT/'logs/ral_3h/import/intervention01_5080')
    parser.add_argument('--backend',type=Path,default=ROOT/'logs/ral_3h/precision01_3090')
    parser.add_argument('--archive',type=Path,default=Path.home()/'Downloads/intervention01_5080_review.tar.gz')
    parser.add_argument('--out',type=Path,default=ROOT/'release/ral/intervention01_review_20261125.json')
    args=parser.parse_args()
    r=dict(audit_time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),audit_script_sha256=digest(__file__),
           archive=audit_archive(args.archive,args.session),
           training_5080=audit_5080(args.session),backend_3090=audit_3090(args.backend),
           interpretation=[
               'A+ and original B have identical mean reference completion in these five seeds; this is not statistical equivalence.',
               'B_v005 improves model-own finite-difference consistency, but its small mean policy benefit has a paired-seed interval spanning zero.',
               'Input rounding does not remove the CPU/GPU discrepancy; do not enable C, D, or C-perm response supervision.',
               'No unseen actuator or hardware superiority claim; historical failures remain separate and unchanged.'])
    atomic_json(args.out,r)
    print(args.out)


if __name__=='__main__':
    main()
