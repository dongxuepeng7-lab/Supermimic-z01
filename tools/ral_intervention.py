"""Three-hour queues: paired smoothing ablation on 5080, precision isolation on 3090.

No 4090 dependency, no locked test, no automatic C or favorable-result stopping.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time
from types import SimpleNamespace

from ral_eight_hour import ROOT, atomic_json, read, digest, check_disk, execute, gpu_lock
from ral_recovery import current_inputs, evaluate_checkpoint
from ral_online_night import verify_training, archive_pair, review_package, compare_backend_processes

CONFIG=ROOT/'configs/ral/intervention_three_hour_v1.json'
PROTOCOL=ROOT/'release/ral/intervention_three_hour_protocol_v1.json'
SELECTION=ROOT/'release/ral/precision_selection_v1.json'


class StopQueue(RuntimeError): pass


def dependencies():
    from ral_nine_hour import inventory
    names=['tools/ral_intervention.py','tools/ral_backend_precision.py','tools/ral_mechanism_probe.py',
           'tools/ral_mechanism_review.py','release/ral/mechanism01_5080_review_20261124.json',
           str(SELECTION.relative_to(ROOT))]
    return dict(inventory(),**{n:digest(ROOT/n) for n in names})


def validate(plan):
    old=read(ROOT/'configs/ral/diagnose_nine_hour_v1.json')
    if plan['training']!=old['training'] or plan['hours']!=3 or plan['C_authorized'] or plan['locked_response_test_access']:
        raise ValueError('Base training or diagnostic constraints changed')
    variants=plan['machines']['5080']['variants']
    if [(v['id'],v['group'],v['friction_speed']) for v in variants]!=[
        ('Aplus','A+',.01),('B_ref','B',.01),('B_v005','B',.05)]:raise ValueError('Prespecified three-model contrast changed')
    if plan['machines']['5080']['seeds']!=[501+4*i for i in range(16)]:raise ValueError('Prespecified seed order changed')


def blocks(plan):
    variants=plan['machines']['5080']['variants']
    for index,seed in enumerate(plan['machines']['5080']['seeds']):
        order=variants[index%3:]+variants[:index%3]
        yield dict(id=f's{seed}',seed=seed,status='pending',conditions=[dict(v,status='pending') for v in order])


def condition_config(plan,variant):
    cfg=copy.deepcopy(plan);cfg['training']['friction_speed']=variant['friction_speed']
    return cfg


def block_result(block):
    if block['status']!='completed' or len(block['conditions'])!=3 or any(c['status']!='completed' for c in block['conditions']):return None
    if {c['id'] for c in block['conditions']}!={'Aplus','B_ref','B_v005'}:raise ValueError('Missing/duplicate variant')
    values={}
    for c in block['conditions']:
        if len(c['rows'])!=2 or {r['repeat_check'] for r in c['rows']}!={0,1}:raise ValueError('Two evaluations required')
        values[c['id']]=next(r for r in c['rows'] if r['repeat_check']==0)
    return dict(seed=block['seed'],primary=values,
        B_v005_minus_B_ref_pp=100*(values['B_v005']['completion']-values['B_ref']['completion']),
        B_ref_minus_Aplus_pp=100*(values['B_ref']['completion']-values['Aplus']['completion']),
        B_v005_minus_Aplus_pp=100*(values['B_v005']['completion']-values['Aplus']['completion']))


def freeze(a):
    from partymimic.experiments.protocol import freeze as base_freeze
    plan=read(CONFIG);validate(plan)
    p=base_freeze(CONFIG,a.params,a.motions,a.responses,PROTOCOL)
    p.update(stage=plan['stage'],groups=['A+','B'],training_seeds=plan['machines']['5080']['seeds'],
        expected_environment_transitions=plan['expected_environment_transitions'],tools=dependencies(),
        design=plan['machines'],selection=read(SELECTION),response_labels_consumed=False,C_authorized=False)
    atomic_json(PROTOCOL,p)


def judge(folder,verify=True):
    state=read(folder/'status.json')
    result=dict(status=state['status'],machine=state['machine'],formal_results=False,C_authorized=False,
        response_test_accessed=False,smoke_only=state.get('smoke_only',False))
    if state['machine']=='5080':
        if verify:
            for b in state['blocks']:
                for c in b['conditions']:
                    for kind in ('predictions','derivative'):
                        record=c.get(kind)
                        if record and digest(folder/record['path'])!=record['sha256']:raise ValueError('Diagnostic changed')
                    if c.get('status')=='completed':
                        record=c['retention']
                        if digest(folder/c['folder']/record['archive'])!=record['archive_sha256']:raise ValueError('Model archive changed')
        values=[v for b in state['blocks'] if (v:=block_result(b)) is not None]
        keys=('B_v005_minus_B_ref_pp','B_ref_minus_Aplus_pp','B_v005_minus_Aplus_pp')
        result.update(complete_triplets=len(values),partial_triplets=sum(b['status']!='completed' for b in state['blocks']),
            results=values,effects={k:dict(n=len(values),mean=statistics.mean([v[k] for v in values]) if values else None,
                sample_sd=statistics.stdev([v[k] for v in values]) if len(values)>1 else None,
                negative_seeds=[v['seed'] for v in values if v[k]<0]) for k in keys},
            interpretation='Primary final EMA repeat 0, paired seeds on 5080. All complete triplets retained. No heldout/hardware/superiority conclusion or outcome-dependent stopping.')
    else:
        if verify:
            for cell in state['backend']:
                for job in cell['jobs']:
                    if job.get('status')=='completed' and digest(folder/job['folder']/'summary.json')!=job['sha256']:raise ValueError('Backend report changed')
        result.update(complete_cells=sum(c['status']=='completed' for c in state['backend']),cells=state['backend'],
            interpretation='48 failure-enriched blocks, repeated directions/processes, not independent training seeds. Rounded-input comparisons never replace native reference failures.')
    atomic_json(folder/'judgment.json',result)
    return result


class Runner:
    def __init__(self,a,plan,state,cutoff,fd,base):
        self.a,self.plan,self.state,self.cutoff,self.fd,self.base=a,plan,state,cutoff,fd,base
        self.cancelled=[False];self.tools=dependencies()
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:self.cancelled.__setitem__(0,True))
    def save(self):
        self.state['updated_unix']=time.time();atomic_json(self.a.out/'status.json',self.state)
    def guard(self,reserve=False):
        if self.cancelled[0]:raise StopQueue('cancelled')
        if time.monotonic()>=self.cutoff:raise StopQueue('time_budget_reached')
        disk=check_disk(self.a.out,self.plan);slack=self.plan['start_reserve_gib']*2**30 if reserve else 0
        if disk['free_bytes']<disk['min_free_bytes']+slack:disk['reasons'].append('free_space_including_start_reserve')
        if disk['output_bytes']+slack>=disk['max_output_bytes']:disk['reasons'].append('output_including_start_reserve')
        disk['start_reserve_bytes']=slack;self.state['disk_check']=disk
        if disk['reasons']:self.save();raise StopQueue('disk_limit_reached: '+','.join(disk['reasons']))
    def command(self,cmd,log):
        self.guard()
        if dependencies()!=self.tools or current_inputs(self.a.motions,self.a.params,self.a.responses)!=self.base:raise ValueError('Inputs/code changed')
        outcome=execute(list(map(str,cmd)),log,min(self.cutoff,time.monotonic()+60*self.plan['job_timeout_minutes']),self.cancelled,self.fd,heartbeat=lambda _:self.guard())
        if outcome!='completed':
            self.guard();raise RuntimeError(f'{outcome}: {log}')
        if dependencies()!=self.tools or current_inputs(self.a.motions,self.a.params,self.a.responses)!=self.base:raise ValueError('Inputs/code changed during job')
    def evaluate(self,run,ck,out):
        self.guard()
        try:return evaluate_checkpoint(run,ck,out,self.a,self.base,self.cutoff,self.cancelled,self.fd,True)
        except RuntimeError:
            self.guard();raise


def train_queue(r,protocol):
    from partymimic.experiments.cli import training_command
    plan=r.plan;n=plan['training']['iterations']
    for b in blocks(plan):
        r.guard(True);b['status']='running';r.state['blocks'].append(b);r.state['current']=b['id'];r.save()
        canonical=r.a.out/'blocks'/b['id']/'paired_initial.pt';hashes=None
        try:
            for c in b['conditions']:
                r.guard(True);folder=canonical.parent/c['id'];folder.mkdir(parents=True)
                cfg=condition_config(plan,c);atomic_json(folder/'config.json',cfg)
                p=copy.deepcopy(protocol);p.update(training=cfg['training'],training_seeds=[b['seed']],config_sha256=digest(folder/'config.json'),variant=c['id'])
                atomic_json(folder/'protocol.json',p)
                job=dict(id=f"s{b['seed']}_{c['id']}",seed=b['seed'],group=c['group'])
                c.update(status='running',rows=[],folder=str(folder.relative_to(r.a.out)))
                run=folder/'jobs'/job['id']/'run';run.parent.mkdir(parents=True)
                try:
                    cmd=training_command(SimpleNamespace(config=folder/'config.json',params=r.a.params,motion_dir=r.a.motions,
                        responses=r.a.responses,seed=b['seed'],group=c['group'],protocol=folder/'protocol.json',out=run))
                    if canonical.exists():cmd+=['--paired_initial_reference',str(canonical)]
                    atomic_json(run.parent/'command.json',cmd)
                    print(f"[5080] {b['id']}/{c['id']} 开始，剩余 {(r.cutoff-time.monotonic())/3600:.2f} 小时",flush=True)
                    r.command(cmd,run.parent/'console.log')
                    ledger=verify_training(run,p,job);job['validated']=True
                    actual=ledger['initial_module_sha256']
                    if hashes is None:hashes=actual;shutil.copyfile(run/'initial.pt',canonical)
                    elif actual!=hashes:raise ValueError('Initial WM/actor/critic tensors differ across triplet')
                    c['initial_module_sha256']=actual;ck=run/f'ckpt_{n:06d}.pt'
                    c['checkpoint_sha256']=digest(ck)
                    for repeat in range(2):
                        row=r.evaluate(run,ck,run.parent/f'eval_r{repeat}.json')
                        c['rows'].append(dict(row,repeat_check=repeat));r.save()
                    for kind,tool,extra in [('predictions','ral_structure_metrics.py',[]),
                        ('derivative','ral_mechanism_probe.py',['--task','derivative','--blocks-per-split',str(plan['machines']['5080']['precision_blocks_per_split'])])]:
                        output=run.parent/(kind+'.json')
                        r.command([sys.executable,'-u',ROOT/'tools'/tool,'--checkpoint',ck,'--params',r.a.params,'--responses',r.a.responses,'--out',output,*extra],run.parent/(kind+'.log'))
                        report=read(output)
                        valid=all(x['valid'] for x in report['splits'].values()) if kind=='predictions' else report['status']=='completed' and report['weights_and_normalizers_unchanged']
                        if not valid:raise ValueError('Invalid diagnostic')
                        c[kind]=dict(path=str(output.relative_to(r.a.out)),sha256=digest(output));r.save()
                    c['status']='completed'
                finally:
                    if c['status']!='completed':c['status']='incomplete'
                    if run.exists():c['retention']=archive_pair(folder,dict(id=c['id'],jobs=[job]),n)
                    r.save()
            b.update(status='completed',paired_initial_module_sha256=hashes)
            canonical.unlink();r.save();judge(r.a.out)
        finally:
            if b['status']!='completed':b['status']='incomplete'
            r.save()


def backend_queue(r):
    source=r.a.backend_source or ROOT/r.plan['machines']['3090']['source']
    selection=read(SELECTION)
    if digest(source/'summary.json')!=selection['source_summary_sha256']:raise ValueError('Original 3090 source differs')
    for split in ('fit','cal'):
        for row in selection[split]:
            if digest(source/row['file'])!=row['trajectory_sha256']:raise ValueError('Missing/changed source raw block')
    for seed in r.plan['machines']['3090']['direction_seeds']:
        r.guard(True);cell=dict(id='original' if seed is None else f'direction{seed}',status='running',jobs=[])
        r.state['backend'].append(cell);folder=r.a.out/'backend'/cell['id'];folder.mkdir(parents=True)
        try:
            for repeat in range(2):
                out=folder/f'r{repeat}';job=dict(repeat=repeat,status='running',folder=str(out.relative_to(r.a.out)))
                cell['jobs'].append(job);r.state['current']=cell['id']+f'/r{repeat}';r.save()
                cmd=[sys.executable,'-u',ROOT/'tools/ral_backend_precision.py','--source',source,'--out',out,
                     '--selection',SELECTION,'--params',r.a.params,'--responses',r.a.responses]
                if seed is not None:cmd+=['--direction-seed',seed]
                if r.a.smoke:cmd+=['--smoke']
                print(f"[3090] {cell['id']}/r{repeat} 开始，剩余 {(r.cutoff-time.monotonic())/3600:.2f} 小时",flush=True)
                r.command(cmd,out.with_suffix('.log'))
                result=read(out/'summary.json');expected=2 if r.a.smoke else 48
                if result['status']!='completed' or len(result['blocks'])!=expected:raise ValueError('Incomplete backend coverage')
                job.update(status='completed',sha256=digest(out/'summary.json'),checked=expected,
                    native_failed=sum(not b['passed'] for b in result['blocks']),
                    compared_case_failures={case:sum(any(not x['passed'] for x in b['comparisons'][case]) for b in result['blocks']) for case in ('native','round_state','round_ctrl','round_both')})
                r.save()
            if not r.a.smoke:
                cross=compare_backend_processes(folder/'r0',folder/'r1');atomic_json(folder/'cross_process.json',cross)
                cell['cross_process']=dict(checked=len(cross['blocks']),failed=sum(not x['passed'] for x in cross['blocks']),sha256=digest(folder/'cross_process.json'))
            cell['status']='completed';r.save();judge(r.a.out)
        finally:
            if cell['status']!='completed':cell['status']='incomplete'
            r.save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','freeze','judge'))
    parser.add_argument('--machine',choices=('5080','3090'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--motions',type=Path,default=ROOT/'data/motions/z01_lafan_v2_clean')
    parser.add_argument('--params',type=Path,default=ROOT/'release/ral/params.csv')
    parser.add_argument('--responses',type=Path,default=ROOT/'data/ral_pairs_v1/manifest.json')
    parser.add_argument('--backend-source',type=Path)
    parser.add_argument('--smoke',action='store_true',help='Tiny integration test only; never paper results')
    a=parser.parse_args()
    if a.mode=='freeze':freeze(a);return 0
    if a.out is None:parser.error('--out required')
    if a.mode=='judge':
        if not (a.out/'status.json').is_file():parser.error(f'未找到 {a.out}/status.json；检查启动日志和输出目录')
        value=judge(a.out);print(f"{value['machine']}: {value['status']}; {a.out}/judgment.json");return 0
    if a.machine is None:parser.error('--machine required')
    start=time.monotonic();plan=read(CONFIG);validate(plan);protocol=read(PROTOCOL)
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    state=dict(status='preparing',machine=a.machine,started_unix=time.time(),blocks=[],backend=[],smoke_only=a.smoke,
               formal_results=False,C_authorized=False,deadline_unix=time.time()+10800)
    atomic_json(a.out/'status.json',state)
    try:
        base=current_inputs(a.motions,a.params,a.responses)
        if protocol['tools']!=dependencies() or protocol['config_sha256']!=digest(CONFIG):raise ValueError('Synchronize frozen three-hour code/protocol')
        for key in ('code_sha256','asset_sha256','params_sha256','motion_sha256','response_sha256'):
            if base[key]!=protocol[key]:raise ValueError(f'Frozen input changed: {key}')
        if a.smoke:
            plan['hours']=1/3
            plan['training'].update(iterations=6,num_envs=8,buffer_steps=80,pi_batch=16,wm_batch=16,wm_batches=2,
                wm_hidden='32,32',enc_hidden='32,32',dec_hidden='32,32',v_gcons_n=16,ckpt_every=6,log_every=1)
            plan['machines']['5080'].update(seeds=[901],precision_blocks_per_split=1)
            plan['machines']['3090']['direction_seeds']=[None]
            protocol=copy.deepcopy(protocol);protocol.update(training=plan['training'],training_seeds=[901],expected_environment_transitions=632,smoke_only=True)
        atomic_json(a.out/'config.json',plan);atomic_json(a.out/'protocol.json',protocol);atomic_json(a.out/'inputs.json',base)
        state.update(status='running',deadline_unix=state['started_unix']+plan['hours']*3600,storage_estimate=plan['storage_estimate'])
        lock=ROOT/'logs/ral_8h'/('.gpu_'+hashlib.sha256(os.environ.get('CUDA_VISIBLE_DEVICES','0').encode()).hexdigest()[:12]+'.lock')
        with gpu_lock(lock) as fd:
            r=Runner(a,plan,state,start+plan['hours']*3600-plan['shutdown_reserve_seconds'],fd,base);r.save()
            if a.machine=='5080':train_queue(r,protocol)
            else:backend_queue(r)
        state['status']='planned_work_completed'
    except StopQueue as exc:state['status']=str(exc)
    except Exception as exc:
        state.update(status='error',reason=str(exc));print(f'停止：{exc}',flush=True)
    finally:
        state['finished_unix']=time.time();atomic_json(a.out/'status.json',state)
        try:judge(a.out)
        except Exception as exc:
            state.update(status='error',judgment_error=str(exc));atomic_json(a.out/'status.json',state)
        review_package(a.out)
        package=a.out.parent/(a.out.name+'_review.tar.gz')
        package.with_suffix(package.suffix+'.sha256').write_text(f'{digest(package)}  {package.name}\n')
        print(f"[{a.machine}] {state['status']}",flush=True)
    return 1 if state['status']=='error' else 0


if __name__=='__main__':raise SystemExit(main())
