"""Scientific-design invariants and the complete-pair reporting boundary."""
import copy
import csv
from pathlib import Path
import sys
import unittest

import numpy as np
import mujoco

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ral_eight_hour import ROOT,read
from ral_validation_six_hour import CONFIG,aggregate,evaluation_jobs,backend_jobs
from ral_stress_eval import stress_values
from ral_constraint_probe import configure,probe_steps


class SixHourTests(unittest.TestCase):
    def setUp(self):self.plan=read(CONFIG)

    def test_pressure_endpoints_and_zero_parameters(self):
        with (ROOT/'release/ral/params.csv').open() as f:rows=list(csv.DictReader(f))
        allv={c['id']:stress_values(rows,c['identified_fraction']) for c in self.plan['scenarios']}
        for r in rows:
            key=r['joint_id']+':'+r['param_id']
            if r['enabled']=='0':self.assertNotIn(key,allv['identified']);continue
            self.assertEqual(allv['identified'][key],float(r['effective_A']))
            self.assertEqual(allv['nominal'][key],float(r['effective_A0']))
            self.assertAlmostEqual(allv['midpoint'][key],(allv['identified'][key]+allv['nominal'][key])/2)
            for family in ('armature','damping','frictionloss'):
                self.assertEqual(allv[family+'_nominal'][key],allv['nominal' if family==r['param_id'] else 'identified'][key])
        self.assertTrue(all(v>=0 for values in allv.values() for v in values.values()))

    def test_all_primary_before_repeats_and_counterbalanced_order(self):
        models=[dict(seed=s,variant=v) for s in self.plan['training_seeds'] for v in self.plan['variants']]
        jobs=list(evaluation_jobs(self.plan,models))
        self.assertEqual(len(jobs),180)
        self.assertEqual({j['repeat'] for j in jobs[:90]},{0})
        self.assertEqual({j['repeat'] for j in jobs[90:]},{1})
        self.assertEqual(len({j['id'] for j in jobs}),180)
        self.assertEqual(sum(j['record'] for j in jobs),18)
        for position in range(3):self.assertEqual({jobs[i*3+position]['variant'] for i in range(3)},set(self.plan['variants']))

    def state(self):
        models=[dict(seed=s,variant=v) for s in self.plan['training_seeds'] for v in self.plan['variants']]
        jobs=list(evaluation_jobs(self.plan,models))
        for j in jobs:
            j.update(status='completed',initial_state_sha256='same',effective_parameters_sha256=j['scenario'],
                     metrics=dict(completion=dict(Aplus=.8,B_ref=.9,B_v005=.85)[j['variant']]))
        return dict(machine='5080',status='planned_work_completed',smoke_only=False,jobs=jobs)

    def test_repeats_not_extra_training_seeds_negative_results_retained(self):
        state=self.state();result=aggregate(state,self.plan)
        self.assertEqual(result['primary_summary']['n_training_seeds'],5)
        self.assertAlmostEqual(result['primary_summary']['mean_pp'],-5)
        for j in state['jobs']:
            if j['repeat']==1 and j['variant']=='B_v005':j['metrics']['completion']=1.
        self.assertAlmostEqual(aggregate(state,self.plan)['primary_summary']['mean_pp'],-5)
        state['jobs'][15]['status']='incomplete' # first non-reference cell
        state['jobs'][-1]['status']='pending'
        result=aggregate(state,self.plan)
        self.assertEqual(result['primary_summary']['n_training_seeds'],4)
        self.assertEqual(result['planned_jobs'],180)
        self.assertEqual(result['unstarted_jobs'],[state['jobs'][-1]['id']])
        self.assertEqual(result['primary_incomplete_seeds'],[dict(seed=501,missing_scenarios=['nominal'])])

    def test_unmatched_physics_must_not_enter_paired_summary(self):
        state=self.state();state['jobs'][0]['effective_parameters_sha256']='wrong'
        with self.assertRaises(ValueError):aggregate(state,self.plan)

    def test_constraint_interventions_are_scoped_and_do_not_mutate_original(self):
        original=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
        for profile,flag in [('cold_start','mjDSBL_WARMSTART'),('frictionloss_off','mjDSBL_FRICTIONLOSS'),
                             ('limits_off','mjDSBL_LIMIT'),('contacts_off','mjDSBL_CONTACT'),('constraints_off','mjDSBL_CONSTRAINT')]:
            m=copy.copy(original);configure(m,profile)
            self.assertEqual(m.opt.disableflags,int(getattr(mujoco.mjtDisableBit,flag)))
            self.assertEqual(original.opt.disableflags,0)
        m=copy.copy(original);configure(m,'tight_solver');self.assertEqual(m.opt.iterations,200)
        self.assertEqual(m.opt.disableflags,0)
        jobs=list(backend_jobs(self.plan));self.assertEqual(len(jobs),28)
        self.assertEqual([j['repeat'] for j in jobs[:2]],[0,1])

    def test_probe_selection_uses_archived_first_deviation_and_keeps_controls(self):
        old=dict(gpu_substeps=np.zeros((1,128,2)),native_substeps=np.zeros((1,128,2)),
                 scales=np.ones(2),gpu_stats=np.zeros((1,128,5)),native_stats=np.zeros((1,128,5)))
        self.assertEqual(probe_steps(old,0),[0,127])
        old['gpu_substeps'][0,20:]=1;old['gpu_stats'][0,12:,0]=2
        self.assertEqual(probe_steps(old,0),[0,11,12,13,19,20,21,127])


if __name__=='__main__':unittest.main()
