"""CPU checks for scientific labels, complete matrices, and exhausted-budget safety."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import ral_cd_full_night as full
from ral_cd_full_worker import empirical_calibration


class FullNightChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=json.loads((ROOT/'configs/ral/cd_full_night_v1.json').read_text())

    def calibration(self):
        rows=[dict(bin=i,eligible_blocks=20,independent_sessions=3,prefix_error=[.2]*15+[r]*17,
                   horizon=8,short_validated=False,long_validated=False) for i,r in enumerate([.7,.8,.9,1.0])]
        return dict(short=8,long=32,min_blocks=10,tolerance=.5,bins=rows)

    def test_empirical_assignment_does_not_promote_failed_calibration(self):
        strict=self.calibration();before=copy.deepcopy(strict)
        result=empirical_calibration(strict,self.plan['empirical_windows'])
        self.assertEqual(strict,before)
        self.assertEqual([r['horizon'] for r in result['bins']],[32,32,16,16])
        self.assertTrue(all(not r['short_validated'] and not r['long_validated'] and not r['assignment_is_certified'] for r in result['bins']))
        self.assertFalse(result['certified_policy_return_bound'])

    def test_unsupported_and_noisy_bins_cannot_receive_long_assignment(self):
        strict=self.calibration();strict['bins'][0]['eligible_blocks']=9
        strict['bins'][1]['prefix_error'][2]=1e30
        strict['bins'][2]['independent_sessions']=1
        result=empirical_calibration(strict,self.plan['empirical_windows'])
        self.assertTrue(all(r['horizon']==16 for r in result['bins']))
        self.assertEqual(result['empirical_long_bins'],[])

    def test_prefix_violation_not_hidden_by_final_error(self):
        strict=self.calibration()
        strict['bins'][0]['prefix_error']=[.1]*32;strict['bins'][0]['prefix_error'][3]=.6
        r=empirical_calibration(strict,self.plan['empirical_windows'])['bins'][0]
        self.assertFalse(r['short_validated']);self.assertFalse(r['long_validated'])

    def test_incomplete_pairs_are_never_zero_filled(self):
        scenarios=self.plan['evaluation']['primary_scenarios'];cells={'501':{'C':dict.fromkeys(scenarios,95.),'B':dict.fromkeys(scenarios,94.)}}
        cells['505']={'C':{'nominal':99.},'B':dict.fromkeys(scenarios,10.)}
        r=full.paired_summary(cells,'C','B',scenarios,self.plan['seeds'])
        self.assertEqual(r['n'],1);self.assertEqual(r['mean_pp'],1.)
        self.assertNotIn('exploratory_paired_t95_pp',r)
        self.assertIsNone(full.paired_summary({},'C','B',scenarios,self.plan['seeds'])['mean_pp'])

    def test_all_expected_cells_have_five_seeds_six_conditions(self):
        for machine,n in [('3090',60),('5080',240)]:
            cells=full.expected_cells(self.plan,machine)
            self.assertEqual(len(cells),n);self.assertEqual(len(set(cells)),n)
            report=full.matrix_report(self.plan,machine,{})
            self.assertFalse(report['complete']);self.assertEqual(report['completed'],0)
        self.assertEqual(len(full.expected_cells(self.plan,'3090',True)),60)

    def test_budget_resume_keeps_spent_time(self):
        q=object.__new__(full.CampaignQueue)
        q.previous_used=32281.;q.active_start=100.;q.total_seconds=32400.;q.reserve=120.
        with patch.object(full.nine.time,'monotonic',return_value=101.),patch.object(full.base.Queue,'guard',return_value=None):
            self.assertEqual(q.used(),32282.)
            with self.assertRaises(full.nine.BudgetFinished):q.guard()

    def test_evaluation_rejects_wrong_model_or_missing_motion(self):
        e={'motion_id':'a','complete':True,'termination_reason':'reference_complete','nonfinite_steps':0}
        r={'episodes':[e],'checkpoint_sha256':'good','scenario':{'id':'nominal'}}
        full.validate_evaluation(r,'nominal','good',{'a.npz':'h'})
        with self.assertRaises(ValueError):full.validate_evaluation(r,'nominal','other',{'a.npz':'h'})
        with self.assertRaises(ValueError):full.validate_evaluation(r,'nominal','good',{'a.npz':'h','b.npz':'h'})

    def test_configuration_keeps_certified_and_empirical_protocols_separate(self):
        full.validate_plan(self.plan)
        changed=copy.deepcopy(self.plan);changed['windows']['tolerance']=1.1
        with self.assertRaises(ValueError):full.validate_plan(changed)
        self.assertEqual(self.plan['window_parent'],'C_h4_w05')
        self.assertFalse(self.plan['machines']['3090']['window_experiment'])


if __name__=='__main__':unittest.main()
