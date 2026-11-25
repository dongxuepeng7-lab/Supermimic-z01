"""Frozen-actor parameter-mismatch evaluation through the unchanged evaluator."""
import argparse
import dataclasses
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

from ral_eight_hour import ROOT, atomic_json, digest, read


def stress_values(rows, alpha):
    """Known nominal/identified endpoints; NOT identification confidence limits."""
    return {r['joint_id']+':'+r['param_id']:
            (1-alpha[r['param_id']])*float(r['effective_A0'])+alpha[r['param_id']]*float(r['effective_A'])
            for r in rows if r['enabled']=='1'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','checkpoint','params','motions','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scenario',required=True)
    p.add_argument('--record',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.out.exists():p.error('Use a new output file')
    import numpy as np
    import torch
    from partymimic.env import TrackingSim
    from partymimic.experiments import parameters
    from partymimic.experiments.runtime import object_hash
    from ral_eight_hour_worker import state_digest
    from ral_policy_diagnose import trajectory_causes
    plan=read(ROOT/'configs/ral/validation_six_hour_v1.json')
    scenario=next(c for c in plan['scenarios'] if c['id']==a.scenario)
    original_parameters,original_reset=parameters.parameterized_robot,TrackingSim.reset
    expected={};snapshots=[]

    def configured(robot,path,plant):
        if Path(path).resolve()!=a.params.resolve() or plant!='identified':
            raise ValueError('Unexpected evaluator plant request')
        rows=parameters.read_manifest(path,robot)
        expected.update(stress_values(rows,scenario['identified_fraction']))
        original_cfg=robot.robot_cfg
        def robot_cfg():
            cfg=original_cfg();original_spec=cfg.spec_fn
            def spec_fn():
                spec=original_spec()
                for key,value in expected.items():
                    joint,param=key.split(':')
                    setattr(spec.joint(joint),param,[value,0.,0.] if param=='damping' else value)
                return spec
            actuators=tuple(dataclasses.replace(act,armature=expected.get(name+':armature',act.armature))
                            for name,act in zip(robot.joint_names,cfg.articulation.actuators,strict=True))
            return dataclasses.replace(cfg,spec_fn=spec_fn,
                articulation=dataclasses.replace(cfg.articulation,actuators=actuators))
        return dataclasses.replace(robot,robot_cfg=robot_cfg)

    def reset(env,ids,frames=None):
        value=original_reset(env,ids,frames)
        if frames is not None:
            if env.randomize or not env.eval_mode or len(ids)!=env.N:
                raise ValueError('Expected full deterministic evaluation reset')
            m=env.sim.mj_model;actual={}
            for key,want in expected.items():
                joint,param=key.split(':');adr=m.jnt_dofadr[m.joint('robot/'+joint).id]
                raw=np.asarray(getattr(m,'dof_'+param)[adr]).reshape(-1)
                got=float(raw[0]);actual[key]=got
                if abs(got-want)>1e-9:raise ValueError(f'Actual plant differs for {key}: {got} != {want}')
                field=getattr(env.sim.model,'dof_'+param)
                gpu=field.detach().cpu().numpy()
                gpu_value=np.asarray(gpu[...,adr,:] if param=='damping' and gpu.ndim==3 else gpu[...,adr]).reshape(-1)
                if abs(float(gpu_value[0])-want)>1e-5:raise ValueError('GPU plant differs from scenario')
            tensors={f'state_{i}':x.detach().cpu().clone() for i,x in enumerate(env.state())}
            tensors.update(frames=frames.detach().cpu().clone(),xi=env.xi.detach().cpu().clone())
            snapshots.append(dict(initial_state_sha256=state_digest({'scenario':tensors},('scenario',)),
                                  effective_parameters=actual,effective_parameters_sha256=object_hash(actual),clips=env.N))
        return value

    fingerprint=digest(a.checkpoint)
    parameters.parameterized_robot,TrackingSim.reset=configured,reset
    try:
        with tempfile.TemporaryDirectory(prefix='stress_eval_',dir=a.out.parent) as tmp:
            tmp=Path(tmp);ck=tmp/a.checkpoint.name;ck.symlink_to(a.checkpoint.resolve())
            shutil.copyfile(a.run/'args.json',tmp/'args.json')
            motions=a.motions
            if a.smoke:
                motions=tmp/'motions';motions.mkdir()
                for source in sorted(a.motions.glob('*.npz'))[:3]:
                    (motions/source.name).symlink_to(source.resolve())
            sys.argv=['partymimic.evaluate','--run_dir',str(tmp),'--ckpt',str(ck),
                '--motion_dir',str(motions),'--parameter_manifest',str(a.params),'--plant','identified',
                '--seed','0','--out',str(a.out)]
            if a.record:sys.argv+=['--record',str(a.out.with_suffix('.trajectory.npz'))]
            try:runpy.run_module('partymimic.evaluate',run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (None,0):raise
        if len(snapshots)!=1 or digest(a.checkpoint)!=fingerprint:raise ValueError('Unexpected reset/checkpoint mutation')
        result=read(a.out);result['plant']=a.scenario
        result.update(scenario=scenario,checkpoint_sha256=fingerprint,checkpoint_weights='EMA',
            scenario_readback=snapshots[0],smoke_only=a.smoke,formal_results=False,
            response_test_accessed=False,scope=plan['scope'],gpu=torch.cuda.get_device_name(),
            evaluator_sha256=digest(ROOT/'src/partymimic/evaluate.py'),worker_sha256=digest(__file__))
        atomic_json(a.out,result)
        if a.record:atomic_json(a.out.with_suffix('.termination.json'),trajectory_causes(a.out.with_suffix('.trajectory.npz'),result['episodes']))
    finally:
        parameters.parameterized_robot,TrackingSim.reset=original_parameters,original_reset


if __name__=='__main__':main()
