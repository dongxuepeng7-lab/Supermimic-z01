"""Analytic correctness and data-scope checks for the frozen diagnostic probes."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from ral_mechanism_probe import (canonical_inputs, constraint_features, derivative_probe,
                                 loss_statistics, loss_probe)
from ral_mechanism import storage_check


class LinearWorld(torch.nn.Module):
    """Known linear q/qd transition: finite differences must equal JVP at all h."""
    def __init__(self):
        super().__init__()
        self.num_j = 1
        self.register_buffer('q0', torch.tensor([0.], dtype=torch.float64))
        for name, value in [('kp',1.), ('kd',.1), ('limit',100.), ('action_scale',1.), ('friction',.1)]:
            self.register_buffer(name, torch.tensor([value], dtype=torch.float64))
        self.friction_speed = .01
        self.offset_slice = slice(0,1)
    def step(self, s, a, previous, context):
        return (*s[:4], s[4]+.02*s[5]+.01*a, s[5]+.03*a)


class MechanismTests(unittest.TestCase):
    def test_loss_gradient_rescaling_with_mask(self):
        parameter = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        err = parameter.square()*torch.tensor([[1.,2.,3.],[4.,5.,6.]], dtype=torch.float64)
        mask = torch.tensor([[1.,1.,0.],[1.,0.,0.]], dtype=torch.float64)
        summed = (err*mask).sum()/2
        g, = torch.autograd.grad(summed, parameter, retain_graph=True)
        mean = (err*mask).sum()/mask.sum()
        gmean, = torch.autograd.grad(mean, parameter)
        result = loss_statistics(err.detach(), mask, float(g.abs()))
        self.assertAlmostEqual(result['divide_by_valid_steps']['preclip_grad_norm'], float(gmean.abs()))
        self.assertEqual(result['valid_steps'], 3)
        self.assertEqual(result['valid_fraction'], .5)
        self.assertNotEqual(result['divide_by_horizon']['loss'], result['divide_by_valid_steps']['loss'])
        with self.assertRaises(ValueError): loss_statistics(err.detach(), torch.zeros_like(mask), 1.)

    def test_multistep_fd_jvp_agree_for_known_linear_model(self):
        torch.manual_seed(5)
        d = dict(base=torch.randn(1,32,1,dtype=torch.float64),
                 direction=torch.randn(1,32,1,dtype=torch.float64),
                 s0=torch.tensor([[0.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],dtype=torch.float64),
                 a_prev=torch.zeros(1,1,dtype=torch.float64),xi=torch.zeros(1,1,dtype=torch.float64),
                 epsilon=torch.tensor([.1],dtype=torch.float64))
        d['direction'] /= d['direction'].norm()
        result = derivative_probe(LinearWorld(), SimpleNamespace(normalize=lambda x:x), d, .05, .5)
        self.assertEqual(len(result['epsilon_ladder']),8)
        for row in result['epsilon_ladder']:
            for val in row['horizons'].values(): self.assertLess(val['fd_vs_ad_scaled_rmse'], 1e-10)

    def test_precision_inputs_share_direction_without_prefix_renorm(self):
        plus, minus = torch.ones(2,32,1)+.01, torch.ones(2,32,1)-.01
        ds=SimpleNamespace(data=dict(a_plus=plus,a_minus=minus,epsilon=torch.ones(2)*.2))
        a=canonical_inputs(ds,[1],'cpu',torch.float64)
        b=canonical_inputs(ds,[1],'cpu',torch.float32)
        self.assertTrue(torch.equal(a['direction'].float(),b['direction']))
        self.assertLess(a['direction'][:,:8].norm(),a['direction'].norm())

    def test_constraints_are_not_assumed_to_be_contact_labels(self):
        n=np.zeros((14,8),dtype=int); it=np.ones((14,8),dtype=int)
        a=constraint_features(n,it,2,7,8)
        self.assertFalse(a['any_constraints'])
        n[3,4:]=2;it[3,5]=100
        b=constraint_features(n,it,2,7,8)
        self.assertTrue(b['branch_constraint_count_differs'])
        self.assertTrue(b['constraint_count_changes_in_time'])
        self.assertTrue(b['solver_reached_100'])

    def test_storage_reason_includes_extraction_and_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('ral_mechanism.shutil.disk_usage', return_value=SimpleNamespace(free=16*2**30)):
                a=storage_check(Path(tmp),2**30,15)
                self.assertTrue(a['reasons'])
                self.assertGreater(a['required_free_bytes'],a['free_bytes'])
                b=storage_check(Path(tmp),2**29,15)
                self.assertFalse(b['reasons'])


if __name__ == '__main__':
    unittest.main()
