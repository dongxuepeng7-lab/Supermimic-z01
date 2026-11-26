"""Nine-hour complete-matrix C/empirical-D campaign; reuse verified parents, no hidden promotion."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics as st
import sys
import time

import ral_cd as base
import ral_cd_nine_hour as nine
from ral_eight_hour import ROOT,atomic_json,digest,read

CONFIG=ROOT/'configs/ral/cd_full_night_v1.json'
EXTRA_WORKER=ROOT/'tools/ral_cd_full_worker.py'
ADDITIONS=('tools/ral_cd_full_night.py','tools/ral_cd_full_worker.py',
           'configs/ral/cd_full_night_v1.json','scripts/ral_3090_cd_full.sh','scripts/ral_5080_cd_full.sh')
BASE_CONTRACT=base.contract


def validate_plan(plan):
    if plan['hours']!=9 or plan['seeds']!=[501,505,509,513,517]:raise ValueError('Fixed nine-hour/five-seed design required')
    t=plan['training_overrides']
    for k,v in dict(iterations=2000,wm_only_iters=100,response_horizon=4,response_grad_ratio=.1).items():
        if v is not None and t[k]!=v:raise ValueError('Training plan differs: '+k)
    if plan['windows']['iterations']!=100 or plan['windows']['lr']!=1e-5:raise ValueError('Conservative windows changed')
    if (plan['windows']['short'],plan['windows']['long'],plan['windows']['tolerance'])!=(8,32,.5):raise ValueError('Original calibration gate changed')
    if (plan['empirical_windows']['short'],plan['empirical_windows']['long'])!=(16,32):raise ValueError('Empirical horizons changed')
    expected={'3090':[('B_refresh','B',0),('C_h4_w05','C',.05)],
              '5080':[('B_refresh','B',0),('C_h4_w01','C',.01),('C_h4_w05','C',.05)]}
    for machine,candidates in expected.items():
        if [(c['name'],c['group'],c['response_weight']) for c in plan['machines'][machine]['candidates']]!=candidates:raise ValueError('Candidate matrix changed')
    if plan['window_parent']!='C_h4_w05':raise ValueError('Parent must be fixed before outcomes')
    if plan['response_test_accessed'] or plan['formal_results']:raise ValueError('Development protocol only')


def check_inputs(a,plan):
    validate_plan(plan)
    for name,h in plan['required_existing_files_sha256'].items():
        if not (ROOT/name).is_file() or digest(ROOT/name)!=h:raise ValueError('Dependency missing/changed: '+name)
    contract=BASE_CONTRACT(a.params,a.motions)
    if contract['code_sha256']!=plan['required_code_sha256']:raise ValueError('Training source differs')
    source=a.source/'status.json';profile=plan['sources'][a.machine]
    if not source.is_file():raise FileNotFoundError(f'需要完整旧模型目录（review 包不含权重）：{source}')
    if digest(source)!=profile['status_sha256']:raise ValueError('Reviewed source status differs; do not silently substitute another session')
    s=read(source)
    if s['machine']!=a.machine or s['status'] not in ('completed','completed_with_blocked_windows','budget_exhausted'):raise ValueError('Source must be stopped and from this machine')
    for k in ('code_sha256','asset_sha256','params_sha256','motions','files','versions'):
        if s['contract'][k]!=contract[k]:raise ValueError('Source/current contract differs: '+k)
    pins={'status.json':digest(source)}
    for seed,entry in profile['parents'].items():
        for rel,h in [(entry['run']+'/ckpt_006000.pt',entry['checkpoint_sha256']),
                      (entry['split_manifest'],entry['split_manifest_sha256'])]:
            if not (a.source/rel).is_file() or digest(a.source/rel)!=h:raise ValueError('Missing/changed source input: '+str(a.source/rel))
            pins[rel]=h
        args=read(a.source/entry['run']/'args.json')
        if args['seed']!=int(seed) or args['experiment_group']!='B':raise ValueError('Source B seed differs')
    legacy=[]
    if plan['machines'][a.machine]['legacy_completion']:
        for seed in plan['seeds']:
            for variant in ('B','C_h32'):
                run=s['models'][str(seed)][variant];j=s['jobs'][f's{seed}_train_{variant}']
                rel=run+'/ckpt_006000.pt';h=j['hashes']['ckpt_006000.pt']
                if digest(a.source/rel)!=h:raise ValueError('Legacy model hash differs')
                pins[rel]=h
        for name,e in s['evaluations'].items():
            j=s['jobs'][name];path=a.source/e['result']
            if j['status']!='completed' or digest(path)!=j['hashes'][path.name]:raise ValueError('Old evaluation changed')
            ck=a.source/s['models'][str(e['seed'])][e['variant']]/'ckpt_006000.pt'
            validate_evaluation(read(path),e['scenario'],pins[str(ck.relative_to(a.source))],contract['motions'])
            legacy.append(dict(e,job_id=name,sha256=digest(path)))
    return dict(contract=contract,source=s,pins=pins,legacy=legacy,
                legacy_missing=60-len(legacy) if a.machine=='3090' else 0)


def training_plan(plan,candidate):
    p=copy.deepcopy(plan);p['C_response_weight']=candidate['response_weight'];return p


def validate_evaluation(result,scenario,checkpoint_hash,motions):
    ep=result['episodes']
    if len(ep)!=len(motions) or {x['motion_id'] for x in ep}!={Path(n).stem for n in motions}:raise ValueError('Incomplete/duplicate episode matrix')
    if result['checkpoint_sha256']!=checkpoint_hash or result['scenario']['id']!=scenario:raise ValueError('Evaluation source differs')
    if any(e.get('eval_seed',0)!=0 or e.get('repeat',0)!=0 or e['complete'] not in (True,False,0,1) for e in ep):raise ValueError('Evaluation seed/repeat/completion differs')
    if any(e['nonfinite_steps'] or bool(e['complete'])!=(e['termination_reason']=='reference_complete') for e in ep):raise ValueError('Invalid/nonfinite episode')
    if any(not math.isfinite(v) for e in ep for v in e.values() if isinstance(v,float)):raise ValueError('Nonfinite episode metric')
    if result.get('smoke_only') or result.get('response_test_accessed'):raise ValueError('Unexpected evaluation protocol')


def candidate_names(plan,machine):
    names=[c['name'] for c in plan['machines'][machine]['candidates']]
    if plan['machines'][machine]['window_experiment']:names+=['C_parent_live',*plan['window_variants']]
    return names


def expected_cells(plan,machine,legacy=False):
    variants=['B','C_h32'] if legacy else candidate_names(plan,machine)
    return [(str(seed),v,sc) for seed in plan['seeds'] for v in variants for sc in plan['evaluation']['scenarios']]


def matrix_report(plan,machine,cells,legacy=False):
    missing=[dict(seed=int(s),variant=v,scenario=sc) for s,v,sc in expected_cells(plan,machine,legacy) if sc not in cells.get(s,{}).get(v,{})]
    n=len(expected_cells(plan,machine,legacy))
    return dict(expected=n,completed=n-len(missing),complete=not missing,missing=missing)


def paired_summary(cells,a,b,scenarios,seeds):
    values=[]
    for seed in seeds:
        c=cells.get(str(seed),{})
        if all(sc in c.get(v,{}) for v in (a,b) for sc in scenarios):
            values.append(dict(seed=seed,difference_pp=st.mean(c[a][sc]-c[b][sc] for sc in scenarios)))
    diffs=[v['difference_pp'] for v in values]
    result=dict(n=len(diffs),per_seed=values,mean_pp=st.mean(diffs) if diffs else None,sample_sd_pp=st.stdev(diffs) if len(diffs)>1 else None)
    if len(diffs)==5:
        half=2.7764451051977987*st.stdev(diffs)/math.sqrt(5)
        result['exploratory_paired_t95_pp']=[st.mean(diffs)-half,st.mean(diffs)+half]
    return result


def summarize(q):
    def cells_from(entries):
        cells={}
        for e in entries.values():
            cells.setdefault(str(e['seed']),{}).setdefault(e['variant'],{})[e['scenario']]=e['completion_percent']
        return cells
    cells=cells_from(q.state['evaluations']);legacy=cells_from(q.state['legacy_evaluations'])
    contrasts=[(c['name'],'B_refresh') for c in q.plan['machines'][q.a.machine]['candidates'] if c['group']=='C']
    if q.a.machine=='5080':contrasts += [('C_h4_w05','C_h4_w01')]+[('D_emp',x) for x in ['C_perm_emp','C_fixed16','C_fixed32','C_parent_live']]
    matrix=matrix_report(q.plan,q.a.machine,cells)
    prior=matrix_report(q.plan,q.a.machine,legacy,True) if q.a.machine=='3090' else None
    diag=q.state['training_diagnostics']
    result=dict(status=q.state['status'],machine=q.a.machine,stage=q.plan['stage'],active_seconds=q.used(),budget_hours=9,
        completion_percent=cells,matrix=matrix,legacy_matrix=prior,legacy_completion_percent=legacy,
        paired_effects={a+'_minus_'+b:paired_summary(cells,a,b,q.plan['evaluation']['primary_scenarios'],q.plan['seeds']) for a,b in contrasts},
        legacy_effects=paired_summary(legacy,'C_h32','B',q.plan['evaluation']['primary_scenarios'],q.plan['seeds']) if prior else None,
        windows=q.state['windows'],training_diagnostics=diag,
        cal_responses={k:read(q.a.out/v)['means'] for k,v in q.state['responses'].items()},
        failures=q.state['independent_failures'],formal_results=False,response_test_accessed=False,
        all_policy_cells_complete=matrix['complete'] and (prior is None or prior['complete']),
        interpretation='Separate GPU replications. Cal-risk ranking is empirical, not original certified D. Every planned cell is listed; missing is never zero. Paired t intervals exploratory, n=5, no multiplicity correction.')
    atomic_json(q.a.out/'summary.json',result)
    return result


class CampaignQueue(nine.Queue):
    def __init__(self,a,plan,fd,acquired,inputs):
        self.inputs=inputs
        super().__init__(a,plan,fd,acquired)
        self.state.setdefault('legacy_evaluations',{})
        self.state.setdefault('independent_failures',{})
        self.state['input_pins']=inputs['pins']
        self.state['source_root']=str(a.source)
        self.save()

    def isolated(self,name,fn):
        try:
            value=fn();self.state['independent_failures'].pop(name,None);return value
        except RuntimeError as exc:
            # A normal child failure is isolated; disk/integrity/budget/cancel errors propagate.
            if ': failed，' not in str(exc):raise
            current=self.state['jobs'].get(self.state.get('current'),{})
            logfile=self.a.out/current.get('log','__missing__')
            tail=logfile.read_text(errors='replace')[-12000:] if logfile.is_file() else ''
            if 'FloatingPointError' in tail or 'Nonfinite' in tail:raise
            self.state['independent_failures'][name]=str(exc);self.save()
            print(f'[{self.a.machine}] {name} 失败，保留记录，继续独立任务：{exc}',flush=True)
            return None

    def extra(self,kind,checkpoint,seed,folder,result,**options):
        command=[sys.executable,'-u',EXTRA_WORKER,kind,'--checkpoint',checkpoint,'--params',self.a.params,
                 '--motions',self.a.motions,'--config',CONFIG,'--seed',seed,'--out',result]
        for k,v in options.items():command+=['--'+k.replace('_','-'),v]
        return command,result


def evaluate(q,seed,variant,scenario,run,checkpoint,weights='EMA',legacy=False):
    name=f"{'legacy_' if legacy else ''}s{seed}_eval_{scenario}_{variant}"
    h=digest(checkpoint)
    if legacy and h!=q.inputs['pins'][str(checkpoint.relative_to(q.a.source))]:raise ValueError('Legacy checkpoint changed after preflight')
    def check(path):
        result=read(path);validate_evaluation(result,scenario,h,q.state['contract']['motions'])
        initial=result['scenario_readback']['initial_state_sha256']
        if q.state.setdefault('common_eval_initial_state',initial)!=initial:raise ValueError('Evaluation initial state differs')
        actual=result['scenario_readback']['effective_parameters']
        if q.state.setdefault('scenario_parameter_readbacks',{}).setdefault(scenario,actual)!=actual:raise ValueError('Evaluation parameters differ within condition')
        result.update(stage=q.plan['stage'],variant=variant,training_seed=seed,checkpoint_weights=weights,
                      legacy_completion=legacy,original_certified_D=False if variant in q.plan['window_variants'] else None)
        atomic_json(path,result)
    path=q.job(name,lambda folder:([sys.executable,'-u',ROOT/'tools/ral_stress_eval.py','--run',run,
        '--checkpoint',checkpoint,'--params',q.a.params,'--motions',q.a.motions,'--scenario',scenario,'--out',folder/'evaluation.json'],folder/'evaluation.json'),check=check)
    result=read(path)
    record=dict(seed=seed,variant=variant,scenario=scenario,result=str(path.relative_to(q.a.out)),
                checkpoint_sha256=h,weights=weights,sha256=digest(path),
                completion_percent=100*sum(bool(e['complete']) for e in result['episodes'])/len(result['episodes']))
    q.state['legacy_evaluations' if legacy else 'evaluations'][name]=record
    q.save();summarize(q)


def complete_legacy(q):
    s=q.inputs['source']
    for e in q.inputs['legacy']:
        if digest(q.a.source/e['result'])!=e['sha256']:raise ValueError('Legacy evaluation changed after preflight')
        r=read(q.a.source/e['result'])
        q.state['legacy_evaluations'][e['job_id']]=dict(e,external_source=True,completion_percent=100*sum(bool(x['complete']) for x in r['episodes'])/len(r['episodes']))
    q.save();summarize(q)
    for seed,v,sc in expected_cells(q.plan,q.a.machine,True):
        oldname=f's{seed}_eval_{sc}_{v}'
        if oldname in s['evaluations']:continue
        run=q.a.source/s['models'][seed][v]
        q.isolated('legacy_'+oldname,lambda seed=int(seed),v=v,sc=sc,run=run:evaluate(q,seed,v,sc,run,run/'ckpt_006000.pt',legacy=True))


def verify_train(run,parent,p,group):
    result=base.verify_training(run,parent,p,group);args=read(run/'args.json')
    for k in ('response_horizon','response_grad_ratio','response_loss','response_huber_delta'):
        if args[k]!=p['training_overrides'][k]:raise ValueError('Executed training setting differs: '+k)
    if args['response_weight']!=(p['C_response_weight'] if group=='C' else 0):raise ValueError('Response weight differs')
    return result


def train_seed(q,seed):
    source=q.plan['sources'][q.a.machine]['parents'][str(seed)]
    parent=q.a.source/source['run']/'ckpt_006000.pt'
    if digest(parent)!=source['checkpoint_sha256'] or digest(q.a.source/source['split_manifest'])!=source['split_manifest_sha256']:raise ValueError('Source parent/split changed after preflight')
    pairs=q.job(f's{seed}_native_pairs',lambda folder:q.worker('pairs',parent,seed,folder,folder/'pairs',split_manifest=q.a.source/source['split_manifest']))
    models=q.state['models'].setdefault(str(seed),{})
    candidates=q.plan['machines'][q.a.machine]['candidates'];shift=q.plan['seeds'].index(seed)%len(candidates)
    initial=None
    for c in candidates[shift:]+candidates[:shift]:
        p=training_plan(q.plan,c);v=c['name'];group=c['group']
        run=q.isolated(f's{seed}_train_{v}',lambda p=p,group=group,v=v:q.job(f's{seed}_train_{v}',lambda folder:(base.train_command(parent,group,pairs/'manifest.json',folder/'run',q.a.params,q.a.motions,p),folder/'run'),check=lambda run:verify_train(run,parent,p,group)))
        if run is None:continue
        modules=read(run/'run.json')['initial_module_sha256']
        if initial is not None and initial!=modules:raise ValueError('Within-seed initialization differs')
        initial=modules;models[v]=str(run.relative_to(q.a.out))
        q.state['training_diagnostics'][f's{seed}_{v}']=nine.training_diagnostics(run);q.save()
    for c in candidates:
        v=c['name']
        if v not in models:continue
        ck=q.a.out/models[v]/'ckpt_002000.pt'
        path=q.isolated(f's{seed}_{v}_cal_response',lambda v=v,ck=ck:q.job(f's{seed}_{v}_cal_response',lambda folder:q.worker('responses',ck,seed,folder,folder/'response.json',responses=pairs/'manifest.json')))
        if path is not None:q.state['responses'][f's{seed}_{v}']=str(path.relative_to(q.a.out));q.save()
    lead=q.plan['window_parent']
    if lead in models:
        ck=q.a.out/models[lead]/'ckpt_002000.pt'
        cal=q.isolated(f's{seed}_strict_calibration',lambda:q.job(f's{seed}_strict_calibration',lambda folder:q.worker('calibrate',ck,seed,folder,folder/'calibration.json',responses=pairs/'manifest.json')))
        if cal is not None:
            q.state['calibration'][str(seed)]=str(cal.relative_to(q.a.out))
            raw=read(cal);q.state['windows'][str(seed)]=dict(strict_short_validated=sum(x['short_validated'] for x in raw['bins']),strict_long_validated=sum(x['long_validated'] for x in raw['bins']),strict_horizons=[8,32],strict_tolerance=.5)
            q.save()
    summarize(q)


def verify_window_pairing(q,seed):
    runs={v:q.a.out/q.state['models'][str(seed)][v] for v in q.plan['window_variants']}
    records={v:read(p/'run.json') for v,p in runs.items()}
    for k in ('init_sha256','start_schedule_sha256'):
        if len({r[k] for r in records.values()})!=1:raise ValueError('Window pairing differs: '+k)
    for k in ('assigned_imagined_steps','computed_imagined_steps'):
        if records['D_emp'][k]!=records['C_perm_emp'][k]:raise ValueError('Empirical/permuted budget differs')
    hist=[]
    for v in ('D_emp','C_perm_emp'):
        with (runs[v]/'train_metrics.csv').open() as f:hist.append([tuple(r[k] for k in ('iteration','assigned_steps','computed_steps','short_count')) for r in csv.DictReader(f)])
    if hist[0]!=hist[1]:raise ValueError('Per-update horizon histograms differ')
    return records['D_emp']['empirical_assignment_identifiable']


def window_seed(q,seed):
    models=q.state['models'].get(str(seed),{});lead=q.plan['window_parent']
    if lead not in models or str(seed) not in q.state['calibration']:return
    ck=q.a.out/models[lead]/'ckpt_002000.pt';strict=q.a.out/q.state['calibration'][str(seed)]
    pairs=q.a.out/q.state['jobs'][f's{seed}_native_pairs']['result']/'manifest.json'
    risk=q.job(f's{seed}_empirical_map',lambda folder:q.extra('risk',ck,seed,folder,folder/'empirical.json',calibration=strict))
    starts=q.job(f's{seed}_starts',lambda folder:q.worker('starts',ck,seed,folder,folder/'starts',responses=pairs))
    live=q.job(f's{seed}_live_parent',lambda folder:q.extra('live',ck,seed,folder,folder/'run'))
    models['C_parent_live']=str(live.relative_to(q.a.out));q.save()
    variants=q.plan['window_variants'];shift=q.plan['seeds'].index(seed)%len(variants)
    def check(run):
        r=read(run/'run.json')
        if r['status']!='completed' or r['iterations']!=100 or r['lr']!=1e-5:raise ValueError('Incomplete window run')
        if not r['frozen_world_model'] or not r['frozen_critic'] or r['original_certified_D']:raise ValueError('Window protocol differs')
    for v in variants[shift:]+variants[:shift]:
        run=q.isolated(f's{seed}_window_{v}',lambda v=v:q.job(f's{seed}_window_{v}',lambda folder:q.extra('window',ck,seed,folder,folder/'run',calibration=risk,starts=starts/'manifest.json',variant=v),check=check))
        if run is not None:models[v]=str(run.relative_to(q.a.out));q.save()
    if all(v in models for v in variants):
        mixed=verify_window_pairing(q,seed)
        q.state['windows'][str(seed)].update(empirical_status='completed',pairing_verified=True,empirical_assignment_identifiable=mixed,empirical_mapping=str(risk.relative_to(q.a.out)),original_certified_D=False)
        q.save()
    summarize(q)


def run_work(q):
    if q.plan['machines'][q.a.machine]['legacy_completion']:complete_legacy(q)
    # Produce all declared seed/model branches before the long evaluation matrix.
    for seed in q.plan['seeds']:q.isolated(f's{seed}_joint_phase',lambda seed=seed:train_seed(q,seed))
    if q.plan['machines'][q.a.machine]['window_experiment']:
        for seed in q.plan['seeds']:q.isolated(f's{seed}_window_phase',lambda seed=seed:window_seed(q,seed))
    for scenario in q.plan['evaluation']['scenarios']:
        for seed in q.plan['seeds']:
            for v in candidate_names(q.plan,q.a.machine):
                rel=q.state['models'].get(str(seed),{}).get(v)
                if not rel:continue
                run=q.a.out/rel;is_live=v=='C_parent_live' or v in q.plan['window_variants']
                ck=run/('last.pt' if is_live else 'ckpt_002000.pt')
                q.isolated(f's{seed}_eval_{scenario}_{v}',lambda seed=seed,v=v,run=run,ck=ck,scenario=scenario,is_live=is_live:evaluate(q,seed,v,scenario,run,ck,'live' if is_live else 'EMA'))
    result=summarize(q)
    required_responses=len(q.plan['seeds'])*len(q.plan['machines'][q.a.machine]['candidates'])
    all_windows=not q.plan['machines'][q.a.machine]['window_experiment'] or all(q.state['windows'].get(str(s),{}).get('pairing_verified') for s in q.plan['seeds'])
    complete=(result['all_policy_cells_complete'] and len(q.state['responses'])==required_responses and len(q.state['calibration'])==5 and all_windows and not q.state['independent_failures'])
    q.state['status']='completed' if complete else 'completed_with_missing_or_failed_cells'
    q.state['complete_for_declared_campaign']=complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--machine',choices=('3090','5080'),required=True)
    p.add_argument('--session',default='cd04_full')
    p.add_argument('--source',type=Path)
    p.add_argument('--check-inputs',action='store_true')
    p.add_argument('--status',action='store_true')
    p.add_argument('--continue-session',action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.session):p.error('Unsafe session name')
    os.chdir(ROOT);a.source=(a.source or ROOT/'logs/ral_cd9'/f'cd02_{a.machine}').resolve()
    a.out=ROOT/'logs/ral_cd9'/f'{a.session}_{a.machine}';a.params=ROOT/'release/ral/params.csv';a.motions=ROOT/'data/motions/z01_lafan_v2_clean';a.inbox=a.source;a.after=None
    if a.status:
        if not (a.out/'status.json').is_file():p.error('Session not started: '+str(a.out))
        s=read(a.out/'status.json');print(json.dumps({k:s.get(k) for k in ['status','current','budget_used_seconds','reason','complete_for_declared_campaign']},ensure_ascii=False,indent=2))
        if (a.out/'summary.json').is_file():
            d=read(a.out/'summary.json')
            brief={k:d[k] for k in ['all_policy_cells_complete','failures']}
            for k in ['matrix','legacy_matrix']:
                v=d[k];brief[k]=None if v is None else {**{a:b for a,b in v.items() if a!='missing'},'first_missing':v['missing'][:6]}
            print(json.dumps(brief,ensure_ascii=False,indent=2))
        return
    plan=read(CONFIG);inputs=check_inputs(a,plan)
    if a.check_inputs:
        print(json.dumps(dict(ready=True,machine=a.machine,source=str(a.source),verified_input_files=len(inputs['pins']),legacy_missing=inputs['legacy_missing'],new_policy_cells=len(expected_cells(plan,a.machine)),estimate=plan['budget_estimates'][a.machine],gpu_work_started=False),ensure_ascii=False,indent=2));return
    if a.out.exists() and not a.continue_session:p.error('Session exists; use --continue-session for remaining original budget')
    launchdir=ROOT/'logs/ral_cd9/launchers';launchdir.mkdir(parents=True,exist_ok=True)
    with (launchdir/f'{a.session}_{a.machine}.lock').open('a+') as launchlock:
        try:fcntl.flock(launchlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('This queue is already running')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','0');lock=ROOT/'logs/ral_8h'/('.gpu_'+hashlib.sha256(visible.encode()).hexdigest()[:12]+'.lock')
        with base.gpu_lock(lock) as fd:
            import torch
            if not torch.cuda.is_available() or a.machine not in torch.cuda.get_device_name():raise ValueError(f'Requires RTX {a.machine}')
            def contract(params,motions):
                return dict(BASE_CONTRACT(params,motions),cd_full_files={n:digest(ROOT/n) for n in ADDITIONS},source_status_sha256=digest(a.source/'status.json'))
            base.contract=contract;base.CONFIG=CONFIG
            q=CampaignQueue(a,plan,fd,time.monotonic(),inputs)
            launchfile=launchdir/f'{a.session}_{a.machine}.json'
            atomic_json(launchfile,dict(status='running',pid=os.getpid(),machine=a.machine,session=a.session,started_unix=time.time(),queue_status=str(a.out/'status.json')))
            try:run_work(q)
            except nine.BudgetFinished as exc:q.state.update(status='budget_exhausted',reason=str(exc))
            except BaseException as exc:q.state.update(status='stopped',reason=str(exc));raise
            finally:
                q.state['finished_unix']=time.time();q.save()
                try:summarize(q);base.review_package(q)
                except Exception as exc:q.state['review_package_error']=str(exc);q.save()
                atomic_json(launchfile,dict(status=q.state['status'],pid=os.getpid(),machine=a.machine,session=a.session,finished_unix=time.time(),queue_status=str(a.out/'status.json')))
    print('Final status: '+q.state['status'],flush=True)


if __name__=='__main__':main()
