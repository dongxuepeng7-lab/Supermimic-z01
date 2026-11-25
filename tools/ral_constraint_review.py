"""Audit completed local constraint diagnostics without changing source results."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import mujoco

from ral_eight_hour import ROOT, atomic_json, digest, read
from ral_intervention_review import audit_archive, local_time, require
from ral_backend_diagnose import compare
from ral_online_night import compare_backend_processes
from ral_validation_six_hour import backend_jobs, verify_release, verify_sources


def audit(folder):
    state,plan,protocol=(read(folder/n) for n in ('status.json','config.json','protocol.json'))
    require(state['machine']=='3090' and state['status']=='planned_work_completed' and not state['smoke_only'], 'Wrong/incomplete run')
    require(protocol==verify_release() and plan==read(ROOT/'configs/ral/validation_six_hour_v1.json'), 'Changed frozen release')
    require(verify_sources('3090',Path(state['source']))==dict(verified_raw_blocks=96), 'Source records changed')
    selection=read(ROOT/'release/ral/constraint_sources_v1.json')['directions']
    require(len(state['jobs'])==28,'Wrong job count')
    judgment=read(folder/'judgment.json');reports={};snapshots={};summaries=[];raw_count=0
    require(judgment['completed_jobs']==judgment['planned_jobs']==28 and not judgment['incomplete_jobs'],'Stored completion differs')
    flags={'cold_start':'mjDSBL_WARMSTART','frictionloss_off':'mjDSBL_FRICTIONLOSS','limits_off':'mjDSBL_LIMIT',
           'contacts_off':'mjDSBL_CONTACT','constraints_off':'mjDSBL_CONSTRAINT'}
    for job,expected in zip(state['jobs'],backend_jobs(plan),strict=True):
        require(all(job[k]==v for k,v in expected.items()) and job['status']=='completed','Changed schedule')
        p=folder/job['output'];require(digest(p)==job['sha256'],'Summary hash changed');d=read(p)
        require(d['status']=='completed' and not d['smoke_only'] and not d['response_test_accessed'] and len(d['blocks'])==48,'Wrong diagnostic coverage/scope')
        require(d['direction']==job['direction'] and d['profile']==job['profile'],'Wrong condition')
        reports[job['id']]=d
        if job['profile']!='native':
            opts=reports[f"{job['direction']}_native_r{job['repeat']}"]['model_options'].copy()
            if job['profile']=='tight_solver':opts.update(iterations=200,ls_iterations=100,tolerance=1e-9)
            else:opts['disableflags']|=int(getattr(mujoco.mjtDisableBit,flags[job['profile']]))
            require(d['model_options']==opts,'Wrong physical intervention')
        files={digest(f):f for f in p.parent.glob('*_trajectories.npz')};require(len(files)==48,'Missing raw files')
        selected={r['block_id']:r for r in selection[job['direction']]['blocks']}
        require({b['block_id'] for b in d['blocks']}==selected.keys(),'Different selected blocks')
        ps={};failures=[]
        for b in d['blocks']:
            require(b['source_sha256']==selected[b['block_id']]['trajectory_sha256'] and b['split']==selected[b['block_id']]['split'],'Source block changed')
            with np.load(files[b['trajectory_sha256']],allow_pickle=False) as raw:
                recomputed=compare(raw['train_cpu'],raw['train_gpu'],raw['scales'],raw['epsilon'],d['config'])
                require(recomputed==b['backend'] and b['passed']==all(r['passed'] for r in recomputed),'Free-rollout comparison differs')
                require(len(raw['probe_states'])==len(raw['probe_controls'])==len(b['shared_state_probes']),'Probe count differs')
                for index,probe in enumerate(b['shared_state_probes']):
                    h=hashlib.sha256(raw['probe_states'][index].tobytes()+raw['probe_controls'][index].tobytes()).hexdigest()
                    require(h==probe['shared_state_sha256'],'Shared-state bytes differ')
                    key=(b['block_id'],probe['branch'],probe['pre_step_index']);require(key not in ps,'Repeated probe')
                    ps[key]=probe
                    delta=(np.asarray(probe['gpu']['joint_state'])-probe['cpu']['joint_state'])/raw['scales']
                    require(float(np.sqrt(np.square(delta).mean()))==probe['scaled_state_rmse'],'Single-step metric differs')
            raw_count+=1
            if not b['passed']:failures.append(b['block_id'])
        snapshots[job['id']]=ps
        native=snapshots[f"{job['direction']}_native_r{job['repeat']}"]
        require(ps.keys()==native.keys() and all(ps[k]['shared_state_sha256']==native[k]['shared_state_sha256'] for k in ps),'Unmatched intervention states')
        values=np.array([v['scaled_state_rmse'] for v in ps.values()])
        entry=dict(direction=job['direction'],profile=job['profile'],repeat=job['repeat'],shared_snapshots=len(values),
            median_shared_step_scaled_rmse=float(np.median(values)),p95_shared_step_scaled_rmse=float(np.quantile(values,.95)),
            native_median_shared_step_scaled_rmse=float(np.median([v['scaled_state_rmse'] for v in native.values()])),
            type_count_mismatch_snapshots=sum(v['cpu']['type_counts']!=v['gpu']['type_counts'] for v in ps.values()),
            free_rollout_failed_blocks=len(failures))
        require(entry in judgment['counterfactual_summaries'],'Stored condition summary differs')
        entry.update(failed_block_ids=failures,shared_error_above_point1=sum(values>.1).item())
        summaries.append(entry)
    cross=[]
    for c in state['cells']:
        p=folder/c['output'];require(digest(p)==c['sha256'],'Cross-process bytes changed')
        recomputed=compare_backend_processes(p.parent/'r0',p.parent/'r1')
        require(recomputed==read(p),'Cross-process comparison differs')
        cross.append(dict(direction=c['direction'],profile=c['profile'],failed_blocks=[b['block_id'] for b in recomputed['blocks'] if not b['passed']]))
    require(len(cross)==14,'Incomplete cross-process checks')
    return dict(status=state['status'],started=local_time(state['started_unix']),finished=local_time(state['finished_unix']),
        wall_seconds=state['finished_unix']-state['started_unix'],jobs=28,raw_trajectories_recomputed=raw_count,
        summaries=summaries,cross_process=cross,protocol_sha256=digest(folder/'protocol.json'),
        formal_results=False,response_test_accessed=False,C_authorized=False,
        scope='Two fixed perturbation directions on 48 fault-enriched fit/cal blocks each. Counts are diagnostic, not population failure estimates.',
        interpretation='Contact-disabled counterfactuals remove almost all free-rollout failures and the large shared-state error tail. Evidence localizes a major discrepancy to the contact-related path, not a specific implementation bug or a deployable repair. Remaining no-contact failure and changed physical systems must be retained.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=ROOT/'logs/ral_6h/constraint01_3090')
    p.add_argument('--out',type=Path,default=ROOT/'release/ral/constraint01_3090_review_20261125.json')
    a=p.parse_args();result=audit(a.folder)
    result['archive']=audit_archive(a.folder.parent/(a.folder.name+'_review.tar.gz'),a.folder)
    result['audit_tool_sha256']=digest(__file__)
    atomic_json(a.out,result)
    print('Verified 28 jobs, 1344 raw trajectories, all single-step metrics and 14 cross-process comparisons.')
    print('Audit:',a.out)


if __name__=='__main__':main()
