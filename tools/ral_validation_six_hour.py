"""Six-hour frozen-actor stress and CPU/GPU constraint-isolation queues."""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import statistics
import sys
import time

from ral_eight_hour import ROOT, atomic_json, check_disk, digest, execute, gpu_lock, read
from ral_intervention import dependencies
from ral_online_night import review_package, compare_backend_processes
from ral_recovery import current_inputs, evaluation_summary, check_run

CONFIG=ROOT/'configs/ral/validation_six_hour_v1.json'
PROTOCOL=ROOT/'release/ral/validation_six_hour_protocol_v1.json'
MODEL_SELECTION=ROOT/'release/ral/stress_models_v1.json'
BACKEND_SELECTION=ROOT/'release/ral/constraint_sources_v1.json'


def files():
    names=['tools/ral_validation_six_hour.py','tools/ral_stress_eval.py','tools/ral_constraint_probe.py',
           'tools/ral_prior_diagnose.py','release/ral/stress_models_v1.json','release/ral/constraint_sources_v1.json',
           'release/ral/intervention01_review_20261125.json']
    return dict(dependencies(),**{n:digest(ROOT/n) for n in names})


def inputs():
    return current_inputs(ROOT/'data/motions/z01_lafan_v2_clean',ROOT/'release/ral/params.csv',ROOT/'data/ral_pairs_v1/manifest.json')


def verify_release():
    frozen=read(PROTOCOL)
    if digest(CONFIG)!=frozen['config_sha256'] or files()!=frozen['files'] or inputs()!=frozen['inputs']:
        raise ValueError('Six-hour code/data differs from the frozen release; synchronize the complete package')
    return frozen


def model_selection(source,smoke=False):
    if not smoke:
        selected=read(MODEL_SELECTION)
        if digest(source/'status.json')!=selected['source_status_sha256']:raise ValueError('5080 source session differs')
        return selected['models']
    state=read(source/'status.json')
    if not state.get('smoke_only'):raise ValueError('Smoke requires a separate tiny integration fixture')
    models=[]
    for b in state['blocks']:
        if b['status']!='completed':continue
        for c in b['conditions']:
            models.append(dict(seed=b['seed'],variant=c['id'],group=c['group'],folder=c['folder'],retention=c['retention'],
                protocol_sha256=digest(source/c['folder']/'protocol.json'),initial_module_sha256=c['initial_module_sha256'],
                checkpoint_sha256=c['checkpoint_sha256'],final_iteration=6))
    return models


def verify_sources(machine,source,smoke=False):
    if machine=='5080':
        models=model_selection(source,smoke)
        for m in models:
            folder=source/m['folder']
            if digest(folder/'protocol.json')!=m['protocol_sha256'] or read(folder/(m['variant']+'_retention.json'))!=m['retention']:
                raise ValueError('Model source metadata changed')
            if digest(folder/m['retention']['archive'])!=m['retention']['archive_sha256']:
                raise ValueError('Model archive missing/changed; keep the original models directory on 5080')
        return dict(models=len(models),verified_archives=True)
    selected=read(BACKEND_SELECTION)
    if digest(source/'status.json')!=selected['source_status_sha256']:raise ValueError('3090 source session differs')
    count=0
    for direction,entry in selected['directions'].items():
        folder=source/'backend'/direction/'r0'
        if digest(folder/'summary.json')!=entry['summary_sha256']:raise ValueError('Backend source summary changed')
        for row in entry['blocks']:
            if digest(folder/row['file'])!=row['trajectory_sha256']:raise ValueError('Missing/changed 3090 raw trajectory')
            count+=1
    return dict(verified_raw_blocks=count)


def evaluation_jobs(plan,models,smoke=False):
    scenarios=plan['scenarios'][:2] if smoke else plan['scenarios']
    seeds=sorted({m['seed'] for m in models})
    variants=plan['variants']
    # Finish all primary measurements first; rotation balances runtime ordering.
    for repeat in ([0] if smoke else plan['repeats']):
        for ci,scenario in enumerate(scenarios):
            for si,seed in enumerate(seeds):
                order=variants[(si+ci)%3:]+variants[:(si+ci)%3]
                for variant in order:
                    yield dict(id=f"r{repeat}_{scenario['id']}_s{seed}_{variant}",kind='evaluation',
                        seed=seed,variant=variant,scenario=scenario['id'],repeat=repeat,
                        record=repeat==0 and (seed==plan['record_primary_seed'] or smoke))


def backend_jobs(plan,smoke=False):
    directions=plan['backend']['directions'][:1] if smoke else plan['backend']['directions']
    profiles=['native','constraints_off'] if smoke else plan['backend']['profiles']
    for direction in directions:
        for profile in profiles:
            for repeat in range(2):
                yield dict(id=f'{direction}_{profile}_r{repeat}',kind='backend',direction=direction,profile=profile,repeat=repeat)


def prepare_model(source,item,out):
    import torch
    from ral_prior_diagnose import extract_verified
    from ral_structure import module_hashes
    from ral_eight_hour_worker import finite_weights
    folder=source/item['folder'];destination=out/'model_cache'/f"s{item['seed']}_{item['variant']}"
    members=extract_verified(folder/item['retention']['archive'],item['retention']['archive_sha256'],destination)
    if members!=item['retention']['members']:raise ValueError('Archived member manifest differs')
    run=destination/f"jobs/s{item['seed']}_{item['variant']}/run"
    p=read(folder/'protocol.json');ledger,index=check_run(run,p)
    if (ledger['training_seed'],ledger['group'],ledger['protocol_sha256'])!=(item['seed'],item['group'],item['protocol_sha256']):
        raise ValueError('Actual archived ledger differs')
    n=item['final_iteration']
    ck=run/f'ckpt_{n:06d}.pt'
    for path,it in ((run/'initial.pt',0),(ck,n)):
        if digest(path)!=index[path.name]['sha256']:raise ValueError('Actual tensor file differs')
        d=torch.load(path,map_location='cpu',weights_only=False)
        finite_weights(d,('wm','pi','critic') if it==0 else ('wm','pi','pi_live','critic'))
        if d['args']!=ledger['args'] or d['iteration']!=it or d['sim_steps']!=(ledger['env_steps'] if it else 0):
            raise ValueError('Checkpoint budget/arguments differ')
        del d
    if digest(ck)!=item['checkpoint_sha256'] or module_hashes(run/'initial.pt')!=item['initial_module_sha256']:
        raise ValueError('Actual paired initial/final weights differ')
    atomic_json(destination/'verified_members.json',members)
    return run,ck


def aggregate(state,plan):
    out=dict(status=state['status'],machine=state['machine'],smoke_only=state['smoke_only'],formal_results=False,
             C_authorized=False,response_test_accessed=False,new_training_runs=0,
             planned_jobs=len(state['jobs']),
             completed_jobs=sum(j['status']=='completed' for j in state['jobs']),
             unstarted_jobs=[j['id'] for j in state['jobs'] if j['status']=='pending'],
             incomplete_jobs=[j['id'] for j in state['jobs'] if j['status']!='completed'])
    if state['machine']=='3090':
        out['cells']=state.get('cells',[])
        out['scope']=plan['backend']['scope']
        return out
    rows=[j for j in state['jobs'] if j['status']=='completed']
    cells=[]
    keys=sorted({(j['repeat'],j['scenario'],j['seed']) for j in rows})
    for repeat,scenario,seed in keys:
        rr=[j for j in rows if (j['repeat'],j['scenario'],j['seed'])==(repeat,scenario,seed)]
        if len(rr)!=3 or {j['variant'] for j in rr}!=set(plan['variants']):continue
        if len({(j['initial_state_sha256'],j['effective_parameters_sha256']) for j in rr})!=1:
            raise ValueError('Unmatched initial state/plant across policies')
        values={j['variant']:j['metrics']['completion'] for j in rr}
        cells.append(dict(seed=seed,scenario=scenario,repeat=repeat,completion=values,
            B_v005_minus_B_ref_pp=100*(values['B_v005']-values['B_ref']),
            B_ref_minus_Aplus_pp=100*(values['B_ref']-values['Aplus']),
            B_v005_minus_Aplus_pp=100*(values['B_v005']-values['Aplus'])))
    out.update(complete_condition_triplets=cells,scope=plan['scope'],primary_definition=plan['primary'])
    needed={c['id'] for c in plan['scenarios'] if c['id']!='identified'}
    effects=[]
    for seed in sorted({j['seed'] for j in rows}):
        rr=[c for c in cells if c['seed']==seed and c['repeat']==0 and c['scenario'] in needed]
        if {c['scenario'] for c in rr}!=needed:continue
        effects.append(dict(seed=seed,mean_B_v005_minus_B_ref_pp=statistics.mean(c['B_v005_minus_B_ref_pp'] for c in rr)))
    values=[e['mean_B_v005_minus_B_ref_pp'] for e in effects]
    out['primary_complete_seed_effects']=effects
    seeds=sorted({j['seed'] for j in state['jobs']})
    out['primary_incomplete_seeds']=[dict(seed=seed,missing_scenarios=sorted(needed-
        {c['scenario'] for c in cells if c['seed']==seed and c['repeat']==0}))
        for seed in seeds if seed not in {e['seed'] for e in effects}]
    out['primary_summary']=dict(n_training_seeds=len(values),mean_pp=statistics.mean(values) if values else None,
                               sample_sd_pp=statistics.stdev(values) if len(values)>1 else None,
                               all_five_seeds_complete=len(values)==5 and not state['smoke_only'])
    repeat_spreads=[]
    for j in rows:
        if j['repeat']!=0:continue
        mates=[k for k in rows if (k['repeat'],k['scenario'],k['seed'],k['variant'])==(1,j['scenario'],j['seed'],j['variant'])]
        if mates:repeat_spreads.append(dict(id=j['id'],spread_pp=100*abs(j['metrics']['completion']-mates[0]['metrics']['completion'])))
    out['repeat_checks']=repeat_spreads
    out['repeat_spread_over_3pp']=[r for r in repeat_spreads if r['spread_pp']>3]
    out['interpretation']='All complete primary cells included; repeated evaluation does not increase training-seed n. Negative outcomes do not stop the queue. Stress values are not hardware uncertainty bounds.'
    return out


def judge(out):
    state=read(out/'status.json');plan=read(out/'config.json')
    reports={}
    for job in state['jobs']:
        if job['status']=='completed' and digest(out/job['output'])!=job['sha256']:raise ValueError('Completed result changed')
        if job['status']=='completed':
            report=read(out/job['output']);reports[job['id']]=report
            if state['machine']=='5080':
                motions=read(out/'protocol.json')['inputs']['motions']
                if state['smoke_only']:motions={k:motions[k] for k in sorted(motions)[:3]}
                if evaluation_summary(report,motions)!=job['metrics']:raise ValueError('Stored episode summary disagrees')
                if report['checkpoint_sha256']!=job['checkpoint_sha256']:raise ValueError('Evaluation checkpoint disagrees')
                if report['plant']!=job['scenario']:raise ValueError('Evaluation scenario disagrees')
                for key in ('initial_state_sha256','effective_parameters_sha256'):
                    if report['scenario_readback'][key]!=job[key]:raise ValueError('Scenario readback disagrees')
    for cell in state.get('cells',[]):
        if digest(out/cell['output'])!=cell['sha256']:raise ValueError('Cross-process result changed')
    result=aggregate(state,plan)
    if state['machine']=='3090':
        import numpy as np
        comparisons=[]
        for direction in plan['backend']['directions']:
            for repeat in range(2):
                baseline=reports.get(f'{direction}_native_r{repeat}')
                if baseline is None:continue
                def probes(d):return {(b['block_id'],p['branch'],p['pre_step_index']):p for b in d['blocks'] for p in b['shared_state_probes']}
                base=probes(baseline)
                for profile in plan['backend']['profiles']:
                    d=reports.get(f'{direction}_{profile}_r{repeat}')
                    if d is None:continue
                    ps=probes(d)
                    if ps.keys()!=base.keys() or any(ps[k]['shared_state_sha256']!=base[k]['shared_state_sha256'] for k in base):
                        raise ValueError('Counterfactual probes did not use the same native CPU states/controls')
                    values=np.array([p['scaled_state_rmse'] for p in ps.values()])
                    comparisons.append(dict(direction=direction,profile=profile,repeat=repeat,
                        shared_snapshots=len(values),median_shared_step_scaled_rmse=float(np.median(values)),
                        p95_shared_step_scaled_rmse=float(np.quantile(values,.95)),
                        native_median_shared_step_scaled_rmse=float(np.median([p['scaled_state_rmse'] for p in base.values()])),
                        type_count_mismatch_snapshots=sum(p['cpu']['type_counts']!=p['gpu']['type_counts'] for p in ps.values()),
                        free_rollout_failed_blocks=sum(not b['passed'] for b in d['blocks'])))
        result['counterfactual_summaries']=comparisons
        result['interpretation']='Same archived CPU states and commands verified across interventions. Changed physics that improves numerical agreement is a localization clue, never a repair or permission for C. Only complete two-process cells enter repeatability conclusions.'
    atomic_json(out/'judgment.json',result);return result


class StopQueue(RuntimeError):pass


def run(a,plan):
    if a.out.exists():raise ValueError('Use a fresh output directory; old attempts remain intact')
    a.out.mkdir(parents=True)
    state=dict(status='preparing',machine=a.machine,smoke_only=a.smoke,jobs=[],cells=[],started_unix=time.time(),
               formal_results=False,C_authorized=False,new_training_runs=0,source=str(a.source))
    atomic_json(a.out/'config.json',plan);atomic_json(a.out/'status.json',state)
    cancelled=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:cancelled.__setitem__(0,True))
    cutoff=time.monotonic()+(1800 if a.smoke else plan['hours']*3600)-plan['shutdown_reserve_seconds']
    state['deadline_unix']=state['started_unix']+(1800 if a.smoke else plan['hours']*3600)
    def save():
        state['updated_unix']=time.time();atomic_json(a.out/'status.json',state)
    def guard():
        if cancelled[0]:raise StopQueue('cancelled')
        if time.monotonic()>=cutoff:raise StopQueue('time_budget_reached')
        disk=check_disk(a.out,plan);state['disk_check']=disk
        if disk['reasons']:save();raise StopQueue('disk_limit_reached')
    try:
        frozen=verify_release();atomic_json(a.out/'protocol.json',frozen)
        atomic_json(a.out/'source_verification.json',verify_sources(a.machine,a.source,a.smoke))
        guard()
        if state['disk_check']['free_bytes']<(plan['min_free_gib']+plan['start_reserve_gib'])*2**30:
            raise StopQueue('insufficient_start_reserve')
        key=hashlib.sha256(os.environ.get('CUDA_VISIBLE_DEVICES','0').encode()).hexdigest()[:12]
        with gpu_lock(ROOT/'logs/ral_8h'/('.gpu_'+key+'.lock')) as fd:
            models=model_selection(a.source,a.smoke) if a.machine=='5080' else []
            prepared={}
            tasks=list(evaluation_jobs(plan,models,a.smoke) if a.machine=='5080' else backend_jobs(plan,a.smoke))
            state['jobs']=[dict(task,status='pending') for task in tasks]
            state['status']='running';save()
            for task in state['jobs']:
                guard();task.update(status='running',started_unix=time.time());state['current']=task['id'];save()
                try:
                    if a.machine=='5080':
                        key=(task['seed'],task['variant'])
                        if key not in prepared:
                            item=next(m for m in models if (m['seed'],m['variant'])==key)
                            prepared[key]=prepare_model(a.source,item,a.out)
                        folder,ck=prepared[key];fingerprint=digest(ck)
                        output=a.out/'evaluations'/(task['id']+'.json');output.parent.mkdir(exist_ok=True)
                        cmd=[sys.executable,'-u',ROOT/'tools/ral_stress_eval.py','--run',folder,'--checkpoint',ck,
                             '--params',ROOT/'release/ral/params.csv','--motions',ROOT/'data/motions/z01_lafan_v2_clean',
                             '--out',output,'--scenario',task['scenario']]
                        if task['record']:cmd+=['--record']
                    else:
                        output=a.out/'backend'/task['direction']/task['profile']/f"r{task['repeat']}"/'summary.json'
                        output.parent.parent.mkdir(parents=True,exist_ok=True)
                        cmd=[sys.executable,'-u',ROOT/'tools/ral_constraint_probe.py','--source',a.source,'--out',output.parent,
                             '--direction',task['direction'],'--profile',task['profile']]
                    if a.smoke:cmd+=['--smoke']
                    log=output.with_suffix('.console.log') if a.machine=='5080' else output.parent.with_suffix('.log')
                    atomic_json(log.with_suffix('.command.json'),list(map(str,cmd)))
                    guard()
                    if files()!=frozen['files'] or inputs()!=frozen['inputs']:raise ValueError('Inputs/code changed')
                    print(f"[{a.machine}] {task['id']} 开始；剩余 {(cutoff-time.monotonic())/3600:.2f} 小时",flush=True)
                    outcome=execute(list(map(str,cmd)),log,min(cutoff,time.monotonic()+plan['job_timeout_minutes']*60),cancelled,fd,heartbeat=lambda _:guard())
                    if outcome!='completed':
                        guard();raise RuntimeError(f'{outcome}: {log}')
                    result=read(output)
                    if files()!=frozen['files'] or inputs()!=frozen['inputs']:raise ValueError('Inputs/code changed during job')
                    if a.machine=='5080':
                        if digest(ck)!=fingerprint or result['checkpoint_sha256']!=fingerprint:raise ValueError('Checkpoint changed')
                        motions=frozen['inputs']['motions']
                        if a.smoke:motions={k:motions[k] for k in sorted(motions)[:3]}
                        task['metrics']=evaluation_summary(result,motions)
                        if result['plant']!=task['scenario'] or result['smoke_only']!=a.smoke:raise ValueError('Wrong scenario/scope')
                        task.update({k:result['scenario_readback'][k] for k in ('initial_state_sha256','effective_parameters_sha256')})
                        task['checkpoint_sha256']=fingerprint
                    else:
                        if result['status']!='completed' or len(result['blocks'])!=(2 if a.smoke else 48):raise ValueError('Incomplete diagnostic coverage')
                        task.update(failed_blocks=sum(not b['passed'] for b in result['blocks']),checked_blocks=len(result['blocks']),
                            shared_state_probes=sum(len(b['shared_state_probes']) for b in result['blocks']))
                        if task['repeat']==1 and not a.smoke:
                            cross=compare_backend_processes(output.parent.parent/'r0',output.parent)
                            path=output.parent.parent/'cross_process.json';atomic_json(path,cross)
                            state['cells'].append(dict(direction=task['direction'],profile=task['profile'],
                                checked=len(cross['blocks']),failed=sum(not r['passed'] for r in cross['blocks']),
                                output=str(path.relative_to(a.out)),sha256=digest(path)))
                    task.update(status='completed',finished_unix=time.time(),output=str(output.relative_to(a.out)),sha256=digest(output));save();judge(a.out)
                    print(f"[{a.machine}] {task['id']} 完成",flush=True)
                finally:
                    if task['status']=='running':task['status']='incomplete'
                    save()
            state['status']='planned_work_completed'
    except StopQueue as exc:state['status']=str(exc)
    except Exception as exc:state.update(status='error',reason=str(exc));print(f'停止：{exc}',flush=True)
    finally:
        state['finished_unix']=time.time();save()
        try:judge(a.out)
        except Exception as exc:state.update(status='error',judgment_error=str(exc));save()
        review_package(a.out)
        package=a.out.parent/(a.out.name+'_review.tar.gz')
        package.with_suffix(package.suffix+'.sha256').write_text(f'{digest(package)}  {package.name}\n')
        print(f"[{a.machine}] {state['status']}",flush=True)
    return 1 if state['status']=='error' else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('freeze','preflight','run','judge'))
    p.add_argument('--machine',choices=('5080','3090'))
    p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();plan=read(CONFIG)
    if a.mode=='freeze':
        if PROTOCOL.exists():p.error('Frozen protocol already exists; use a new version for revisions')
        atomic_json(PROTOCOL,dict(stage=plan['stage'],config_sha256=digest(CONFIG),files=files(),inputs=inputs(),
                                 formal_results=False,C_authorized=False,new_training_runs=0))
        return 0
    if a.mode=='judge':
        if not a.out or not (a.out/'status.json').exists():p.error('Missing status.json: check the output path and console log')
        d=judge(a.out);print(f"{d['machine']}: {d['status']}; completed jobs={d['completed_jobs']}; {a.out}/judgment.json");return 0
    if not a.machine:p.error('--machine required')
    a.source=(a.source or ROOT/plan['source_'+a.machine]).resolve()
    if a.mode=='preflight':
        verify_release();value=verify_sources(a.machine,a.source,a.smoke)
        print(dict(machine=a.machine,ready=True,source=str(a.source),**value));return 0
    if not a.out:p.error('--out required')
    a.out=a.out.resolve()
    return run(a,plan)


if __name__=='__main__':raise SystemExit(main())
