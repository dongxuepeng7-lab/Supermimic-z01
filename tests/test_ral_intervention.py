"""Prespecified contrast, partial-cell accounting, and precision/contact arithmetic."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ral_eight_hour import read
from ral_intervention import CONFIG,blocks,condition_config,block_result,validate
from ral_backend_precision import round32,contact_bits_cpu,contact_bits_gpu


class InterventionTests(unittest.TestCase):
    def test_only_smoothing_changes_training_factor(self):
        plan=read(CONFIG);validate(plan)
        configs=[condition_config(plan,c)['training'] for c in plan['machines']['5080']['variants']]
        self.assertEqual(configs[0],configs[1])
        self.assertEqual({k for k in configs[1] if configs[1][k]!=configs[2][k]}, {'friction_speed'})
        self.assertEqual(configs[2]['horizon'],32);self.assertEqual(configs[2]['wm_window'],8)
        self.assertEqual(configs[2]['response_weight'],0.)
        bad=copy.deepcopy(plan);bad['training']['wm_window']=32
        with self.assertRaises(ValueError):validate(bad)

    def test_seeds_and_counterbalance_prespecified(self):
        cells=list(blocks(read(CONFIG)))
        self.assertEqual(len({b['seed'] for b in cells}),16)
        self.assertEqual([b['seed'] for b in cells[:3]],[501,505,509])
        for pos in range(3):self.assertEqual({b['conditions'][pos]['id'] for b in cells[:3]},{'Aplus','B_ref','B_v005'})

    def test_partial_triplets_excluded_repeats_not_extra_seeds(self):
        b=next(blocks(read(CONFIG)));self.assertIsNone(block_result(b));b['status']='completed'
        for c,value in zip(b['conditions'],(.9,.8,.85)):
            c.update(status='completed',rows=[dict(repeat_check=0,completion=value),dict(repeat_check=1,completion=.99)])
        v=block_result(b);self.assertAlmostEqual(v['B_v005_minus_B_ref_pp'],5.)
        self.assertAlmostEqual(v['B_v005_minus_Aplus_pp'],-5.)
        b['conditions'][2]['status']='incomplete';self.assertIsNone(block_result(b))
        b['conditions'][2]['status']='completed';b['conditions'][0]['rows'].pop()
        with self.assertRaises(ValueError):block_result(b)

    def test_rounding_preserves_cpu_arithmetic_dtype_and_inputs(self):
        x=np.array([1.+2**-30,.1,1e-9],dtype=np.float64);saved=x.copy();y=round32(x)
        self.assertEqual(y.dtype,np.float64);self.assertTrue(np.array_equal(x,saved))
        self.assertEqual(y[0],1.);self.assertTrue(np.array_equal(y.astype(np.float32),x.astype(np.float32)))
        self.assertTrue(np.array_equal(round32(y),y))

    def test_contact_sets_ignore_order_duplicates_inactive_and_invalid(self):
        geoms=np.array([[2,1],[1,2],[0,2],[-1,2],[0,4]])
        active=np.array([True,True,False,True,True])
        x=contact_bits_cpu(geoms,active,3)
        y=contact_bits_gpu(torch.tensor(geoms),torch.tensor(active),3).numpy()
        self.assertTrue(np.array_equal(x,y))
        self.assertEqual(np.unpackbits(x,bitorder='little').sum(),1)
        self.assertEqual(np.unpackbits(x,bitorder='little')[1*3+2],1)

    def test_friction_smoothing_changes_velocity_slope_not_direct_action_gain(self):
        from partymimic.experiments.models import ActuatorWorldModel
        slopes=[];gains=[]
        for speed in (.01,.05):
            t=lambda x:torch.tensor([x],dtype=torch.float64)
            model=SimpleNamespace(xi_scale=t(1.),xi_center=t(0.),q0=t(0.),action_scale=t(1.),
                offset_slice=slice(0,1),kp=t(3.),kd=t(2.),limit=t(100.),damping=t(.1),friction=t(.5),friction_speed=speed)
            a=t(0.).requires_grad_();v=t(0.).requires_grad_();S=(None,None,None,None,t(0.),v)
            out=ActuatorWorldModel.effort(model,S,a,t(0.))
            ga,gv=torch.autograd.grad(out.sum(),(a,v));gains.append(float(ga));slopes.append(-float(gv)-2.1)
        self.assertEqual(gains,[3.,3.]);self.assertAlmostEqual(slopes[0]/slopes[1],5.)


if __name__=='__main__':unittest.main()
