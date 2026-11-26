"""Read-only audit of the frozen 5080 parameter-mismatch evaluation return.

No simulator runs, checkpoint loading, protocol edits, or response-test access.
Inference uses paired training seeds; clips, conditions and repeats are not seeds.
"""
import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import statistics

from ral_eight_hour import ROOT, atomic_json, digest, read
from ral_intervention_review import audit_archive, interval, local_time, require, summary
from ral_recovery import check_run, evaluation_summary
from ral_stress_eval import stress_values
from ral_validation_six_hour import aggregate, evaluation_jobs, verify_release
from partymimic.experiments.runtime import object_hash


def audit(folder):
    state, plan, protocol = (read(folder/n) for n in ('status.json', 'config.json', 'protocol.json'))
    require(state['machine']=='5080' and not state['smoke_only'], 'Wrong machine/smoke scope')
    require(state['status']=='planned_work_completed', 'This audit expects a complete prespecified run')
    require(state['new_training_runs']==0, 'Unexpected new training')
    require(protocol==verify_release(), 'Frozen protocol differs')
    require(plan==read(ROOT/'configs/ral/validation_six_hour_v1.json'), 'Plan changed')
    selected=read(ROOT/'release/ral/stress_models_v1.json')['models']
    expected=list(evaluation_jobs(plan, selected))
    require(len(state['jobs'])==len(expected)==180, 'Wrong job coverage')
    require(read(folder/'source_verification.json')==dict(models=15,verified_archives=True), 'Source check record differs')
    require(aggregate(state,plan)==read(folder/'judgment.json'), 'Stored aggregate disagrees')
    previous=ROOT/'logs/ral_3h/import/intervention01_5080'
    metadata_count=0
    model_map={(m['seed'],m['variant']):m for m in selected}
    for item in selected:
        cache=folder/'model_cache'/f"s{item['seed']}_{item['variant']}"
        members=read(cache/'verified_members.json')
        require(members==item['retention']['members'], 'Retained member manifest differs')
        for name, fingerprint in members.items():
            if Path(name).suffix!='.pt':
                require(digest(cache/name)==fingerprint, 'Retained training metadata changed')
                metadata_count+=1
        source=previous/item['folder']
        p=read(source/'protocol.json')
        require(digest(source/'protocol.json')==item['protocol_sha256'], 'Training protocol changed')
        run=cache/f"jobs/s{item['seed']}_{item['variant']}/run"
        ledger,index=check_run(run,p)
        require((ledger['training_seed'],ledger['group'],ledger['policy_updates'],ledger['wm_updates'],ledger['env_steps'])==
                (item['seed'],item['group'],5000,25000,12865536), 'Model identity/budget changed')
        require(ledger['initial_module_sha256']==item['initial_module_sha256'], 'Initial module metadata changed')
        require(ledger['protocol_sha256']==item['protocol_sha256'], 'Wrong ledger protocol')
        require(ledger['args']['friction_speed']==(.05 if item['variant']=='B_v005' else .01), 'Wrong smoothing condition')
        require(index['ckpt_005000.pt']['sha256']==item['checkpoint_sha256'], 'Wrong final checkpoint')
    for seed in plan['training_seeds']:
        initial=[model_map[(seed,v)]['initial_module_sha256'] for v in plan['variants']]
        require(initial[0]==initial[1]==initial[2], 'Unpaired initial networks')
    with (ROOT/'release/ral/params.csv').open() as f:params=list(csv.DictReader(f))
    scenarios={s['id']:s for s in plan['scenarios']}
    rows=[]; reports={}; shared=set(); terminations=[]
    for job,expected_job in zip(state['jobs'],expected,strict=True):
        require(all(job[k]==v for k,v in expected_job.items()) and job['status']=='completed', 'Schedule/completion differs')
        path=folder/job['output'];require(digest(path)==job['sha256'], 'Evaluation bytes changed')
        r=read(path);reports[job['id']]=r
        require(not r['smoke_only'] and not r['formal_results'] and not r['response_test_accessed'], 'Evaluation scope differs')
        require(r['checkpoint_sha256']==job['checkpoint_sha256']==model_map[(job['seed'],job['variant'])]['checkpoint_sha256'], 'Wrong evaluation weights')
        require(r['checkpoint_weights']=='EMA' and r['plant']==job['scenario'] and r['scenario']==scenarios[job['scenario']], 'Wrong evaluation settings')
        require(r['worker_sha256']==protocol['files']['tools/ral_stress_eval.py'] and r['evaluator_sha256']==protocol['inputs']['evaluator_sha256'], 'Wrong worker/evaluator')
        require(evaluation_summary(r,protocol['inputs']['motions'])==job['metrics'], 'Episode summary differs')
        effective=stress_values(params,r['scenario']['identified_fraction']);rb=r['scenario_readback']
        require(rb['effective_parameters']==effective and rb['effective_parameters_sha256']==object_hash(effective)==job['effective_parameters_sha256'], 'Actual parameter readback differs')
        require(rb['initial_state_sha256']==job['initial_state_sha256'] and rb['clips']==163, 'Initial scene differs')
        shared.add(rb['initial_state_sha256'])
        episodes=r['episodes'];failed={e['motion_id'] for e in episodes if not e['complete']}
        require(set(r['failed'])==failed and math.isclose(r['fail_rate'],len(failed)/163,abs_tol=1e-7), 'Failure list disagrees')
        for e in episodes:
            for sq,count,rmse in [('q_sqerr_sum_rad2','q_valid_count','q_rmse_rad'),('v_sqerr_sum','v_valid_count','v_rmse_m_s')]:
                require(e[count]>0 and math.isclose(math.sqrt(e[sq]/e[count]),e[rmse],rel_tol=2e-6,abs_tol=1e-8), 'RMSE arithmetic differs')
            require((e['termination_reason']=='reference_complete')==bool(e['complete']), 'Termination reason disagrees')
        if job['record']:
            cause=read(path.with_suffix('.termination.json'))
            require(sum(cause['counts'].values())==163 and cause['counts']['reference_complete']==job['metrics']['completed'], 'Termination totals disagree')
            require({e['motion_id'] for e in cause['episodes']}=={e['motion_id'] for e in episodes}, 'Termination coverage differs')
            require(dict(Counter(e['cause'] for e in cause['episodes']))=={k:v for k,v in cause['counts'].items() if v}, 'Termination labels disagree')
            require({e['motion_id'] for e in cause['episodes'] if e['cause']!='reference_complete'}==failed, 'Termination failure IDs disagree')
            terminations.append(dict(seed=job['seed'],variant=job['variant'],scenario=job['scenario'],counts=cause['counts']))
        rows.append(dict(id=job['id'],seed=job['seed'],variant=job['variant'],scenario=job['scenario'],repeat=job['repeat'],
            completed=job['metrics']['completed'],completion_pct=100*job['metrics']['completion'],
            valid_duration_fraction=sum(e['valid_s'] for e in episodes)/sum(e['planned_s'] for e in episodes),
            valid_frame_q_rmse_mrad=1000*math.sqrt(sum(e['q_sqerr_sum_rad2'] for e in episodes)/sum(e['q_valid_count'] for e in episodes)),
            norm_return=r['norm_return'],failed_clips=sorted(failed)))
    require(len(shared)==1, 'Unmatched initial states')
    require(len(terminations)==18, 'Recorded diagnostic coverage differs')
    by_key={(r['repeat'],r['scenario'],r['seed'],r['variant']):r for r in rows}
    def value(repeat,scenario,seed,variant):return by_key[(repeat,scenario,seed,variant)]['completion_pct']
    per_condition=[]
    for scenario in scenarios:
        group={v:summary(value(0,scenario,s,v) for s in plan['training_seeds']) for v in plan['variants']}
        effects={label:interval(value(0,scenario,s,a)-value(0,scenario,s,b) for s in plan['training_seeds'])
                 for label,a,b in [('B_v005_minus_B_ref','B_v005','B_ref'),('B_ref_minus_Aplus','B_ref','Aplus'),('B_v005_minus_Aplus','B_v005','Aplus')]}
        per_condition.append(dict(scenario=scenario,completion_pct=group,paired_differences_pp=effects))
    stress=[s for s in scenarios if s!='identified'];primary=[];reference_adjusted=[]
    for seed in plan['training_seeds']:
        diffs=[value(0,s,seed,'B_v005')-value(0,s,seed,'B_ref') for s in stress]
        effect=statistics.mean(diffs)
        primary.append(dict(seed=seed,mean_difference_pp=effect,positive_conditions=sum(v>0 for v in diffs),negative_conditions=sum(v<0 for v in diffs)))
        reference_adjusted.append(effect-(value(0,'identified',seed,'B_v005')-value(0,'identified',seed,'B_ref')))
    repetitions=[]
    for r in rows:
        if r['repeat']!=0:continue
        other=by_key[(1,r['scenario'],r['seed'],r['variant'])]
        rr=reports[r['id']]['episodes'];ss=reports[other['id']]['episodes'];em={e['motion_id']:e for e in ss}
        repetitions.append(dict(id=r['id'],spread_pp=abs(r['completion_pct']-other['completion_pct']),
            clips_switching_completion=sum(e['complete']!=em[e['motion_id']]['complete'] for e in rr)))
    old_reference=[]
    for item in selected:
        path=previous/item['folder']/f"jobs/s{item['seed']}_{item['variant']}/eval_r0.json"
        old=evaluation_summary(read(path),protocol['inputs']['motions'])
        old_reference.append(dict(seed=item['seed'],variant=item['variant'],previous_pct=old['completion']*100,
            current_pct=value(0,'identified',item['seed'],item['variant'])))
    result=dict(status=state['status'],started=local_time(state['started_unix']),finished=local_time(state['finished_unix']),
        wall_seconds=state['finished_unix']-state['started_unix'],models=15,jobs=180,episode_evaluations=180*163,
        training_seeds=plan['training_seeds'],new_training_runs=0,formal_results=False,response_test_accessed=False,C_authorized=False,
        checks=dict(protocol_and_current_inputs_match=True,model_metadata_members_verified=metadata_count,
            all_180_episode_summaries_recomputed=True,all_parameter_readbacks_match=True,common_initial_state_sha256=next(iter(shared)),
            all_model_initializations_paired=True,actual_tensor_bytes_verified_locally=False,nonfinite_evaluation_steps=0),
        scope=plan['scope'],primary_definition=plan['primary'],primary_seed_effects=primary,
        primary_difference_pp=interval(p['mean_difference_pp'] for p in primary),
        stress_completion_pct={v:summary(statistics.mean(value(0,c,s,v) for c in stress) for s in plan['training_seeds']) for v in plan['variants']},
        per_condition=per_condition,rows=rows,repetition_checks=repetitions,
        repeat_max_spread_pp=max(r['spread_pp'] for r in repetitions),repeat_cases_above_3pp=sum(r['spread_pp']>3 for r in repetitions),
        reference_adjusted_stress_difference_pp=interval(reference_adjusted),
        previous_reference_comparison=old_reference,recorded_seed501_termination_counts=terminations,
        tensor_scope='Model retention manifests, ledgers and checkpoint fingerprints verified locally; actual tensor bytes and raw trajectory NPZ are absent from the review package.',
        interpretation='All six condition means favor B_v005 over B_ref, but the prespecified paired-seed primary interval includes zero. This is a small positive point estimate, not established robustness superiority. Same reference clips and known parameter endpoints; no hardware/unseen-motion claim.',
        auxiliary_metric_caution='Valid-frame error can be biased by early termination; do not use it alone to establish control improvement. All condition-wise/reference-adjusted comparisons are exploratory.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=ROOT/'logs/ral_6h/import/stress01_5080')
    p.add_argument('--archive',type=Path,default=Path('/home/zoyi/Downloads/stress01_5080_review.tar.gz'))
    p.add_argument('--out',type=Path,default=ROOT/'release/ral/stress01_5080_review_20261125.json')
    a=p.parse_args();archive=audit_archive(a.archive,a.folder);result=audit(a.folder)
    result['archive']=archive;result['audit_tool_sha256']=digest(__file__)
    atomic_json(a.out,result)
    print('Verified 180 jobs, 29340 episode evaluations, 15 model metadata sets.')
    print('Primary paired difference (percentage points):',result['primary_difference_pp'])
    print('Audit:',a.out)


if __name__=='__main__':main()
