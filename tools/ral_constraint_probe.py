"""Constraint isolation: shared-state single steps plus free paired rollouts.

Disabled forces/constraints are counterfactual diagnostic plants, not proposed
training fixes. The historical labels and passing thresholds are unchanged.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from ral_eight_hour import ROOT, atomic_json, digest, read
from ral_backend_diagnose import compare

PROFILES=('native','cold_start','frictionloss_off','limits_off','contacts_off','constraints_off','tight_solver')


def configure(model,profile):
    import mujoco
    bits={'cold_start':'mjDSBL_WARMSTART','frictionloss_off':'mjDSBL_FRICTIONLOSS',
          'limits_off':'mjDSBL_LIMIT','contacts_off':'mjDSBL_CONTACT','constraints_off':'mjDSBL_CONSTRAINT'}
    if profile in bits:model.opt.disableflags |= int(getattr(mujoco.mjtDisableBit,bits[profile]))
    elif profile=='tight_solver':
        model.opt.tolerance=1e-9;model.opt.iterations=200;model.opt.ls_iterations=100
    elif profile!='native':raise ValueError(profile)


def probe_steps(old,branch):
    """Fixed points around first archived deviation, plus first/last step."""
    error=np.sqrt(np.square((old['gpu_substeps'][branch]-old['native_substeps'][branch])/old['scales']).mean(-1))
    nefc=old['gpu_stats'][branch,:,0]!=old['native_stats'][branch,:,0]
    points={0,len(error)-1}
    for hits in (np.flatnonzero(error>.1),np.flatnonzero(nefc)):
        if len(hits):points.update(max(0,int(hits[0])+offset) for offset in (-1,0,1) if int(hits[0])+offset<len(error))
    return sorted(points)


def type_counts(values):
    return {str(k):v for k,v in sorted(Counter(map(int,values)).items())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--direction',choices=('original','direction2001'),required=True)
    p.add_argument('--profile',choices=PROFILES,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import mujoco
    import mujoco_warp as mw
    import warp as wp
    import torch
    from mjlab.scene import Scene
    from partymimic.env import make_scene_cfg,make_sim
    from partymimic.experiments.cli import common_robot
    from partymimic.experiments.response import ResponseData
    from partymimic.dr import XiLayout
    selection=read(ROOT/'release/ral/constraint_sources_v1.json')['directions'][a.direction]
    source=a.source/'backend'/a.direction/'r0'
    if digest(source/'summary.json')!=selection['summary_sha256']:raise ValueError('Original precision report changed')
    robot=common_robot(ROOT/'release/ral/params.csv');layout=XiLayout(robot,'cpu')
    scene=Scene(make_scene_cfg(robot,1),device='cuda:0');native=scene.compile();make_sim(robot).mujoco.apply(native)
    import copy
    model=copy.copy(native);configure(model,a.profile)
    dt=mujoco.mjtState.mjSTATE_INTEGRATION
    jj=[native.joint('robot/'+n).id for n in robot.joint_names]
    qa,va=native.jnt_qposadr[jj],native.jnt_dofadr[jj]
    aids=np.array([native.actuator('robot/'+n).id for n in robot.joint_names])
    body=native.body('robot/'+robot.com_body).id;com0=native.body_ipos[body].copy()
    default,scale=np.array(robot.default_joint_pos_list()),np.array(robot.action_scale_list())
    datasets={s:ResponseData(ROOT/'data/ral_pairs_v1/manifest.json',s,robot,layout,digest(ROOT/'release/ral/params.csv'),'cpu').arrays for s in ('fit','cal')}
    lookups={s:{str(v):i for i,v in enumerate(z['block_id'])} for s,z in datasets.items()}
    source_report=read(source/'summary.json');cfg=source_report['config']
    report=dict(status='running',profile=a.profile,direction=a.direction,config=cfg,smoke_only=a.smoke,
        response_test_accessed=False,C_authorized=False,formal_results=False,blocks=[],
        gpu=torch.cuda.get_device_name(),mujoco_version=mujoco.__version__,
        scope='Counterfactual constraint diagnosis; shared native CPU states, then each condition free rollout; not response-label approval',
        geom_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)],
        object_id_scope='Contact constraint IDs are backend-local contact indices; compare contact geometry/positions, not raw IDs across backends.',
        constraint_types={str(int(getattr(mujoco.mjtConstraint,n))):n for n in dir(mujoco.mjtConstraint) if n.startswith('mjCNSTR_')},
        model_options=dict(disableflags=int(model.opt.disableflags),iterations=int(model.opt.iterations),tolerance=float(model.opt.tolerance),ls_iterations=int(model.opt.ls_iterations)))
    atomic_json(a.out/'summary.json',report)
    rows=selection['blocks']
    if a.smoke:rows=[next(r for r in rows if r['split']==s) for s in ('fit','cal')]
    for number,row in enumerate(rows):
        file=source/row['file']
        if digest(file)!=row['trajectory_sha256']:raise ValueError('Original raw trajectory changed')
        with np.load(file,allow_pickle=False) as f:old={k:f[k].copy() for k in f.files}
        z=datasets[row['split']];i=lookups[row['split']][row['block_id']];xi=z['xi'][i]
        initial=old['initial_train_integration'];controls=default+scale*old['actions']+xi[layout.slices['joint_offset']]
        for m in (native,model):
            m.geom_friction[m.geom_bodyid!=0,0]=xi[layout.slices['friction']][0]
            m.body_ipos[body]=com0+xi[layout.slices['com']]
            mujoco.mj_setConst(m,mujoco.MjData(m))
        gm=mw.put_model(model)
        def cpu_reset(m,state):
            d=mujoco.MjData(m);mujoco.mj_setState(m,d,state,dt);mujoco.mj_forward(m,d);return d
        def gpu_reset(d):
            return mw.put_data(model,d,nworld=1,nconmax=robot.nconmax,njmax=robot.njmax)
        def arr(value):return value.numpy().copy()
        def snapshot(d,gpu):
            if gpu:
                n=int(arr(d.nefc)[0]);types=arr(d.efc.type)[0,:n];ids=arr(d.efc.id)[0,:n]
                force=arr(d.efc.force)[0,:n];jac=arr(d.efc.J)[0,:n,:model.nv]
                qacc=arr(d.qacc)[0];constraint=arr(d.qfrc_constraint)[0];passive=arr(d.qfrc_passive)[0];actuator=arr(d.qfrc_actuator)[0]
                it=int(arr(d.solver_niter)[0]);overflow=int(arr(d.overflow)[0])
                q,qd=arr(d.qpos)[0],arr(d.qvel)[0]
                nc=int(arr(d.nacon)[0]);contact=d.contact
                contacts=dict(geom=arr(contact.geom)[:nc].tolist(),pos=arr(contact.pos)[:nc].tolist(),
                    dist=arr(contact.dist)[:nc].tolist(),efc_address=arr(contact.efc_address)[:nc].tolist())
            else:
                n=d.nefc;types=d.efc_type[:n].copy();ids=d.efc_id[:n].copy();force=d.efc_force[:n].copy()
                if mujoco.mj_isSparse(model):
                    jac=np.zeros((n,model.nv))
                    for j in range(n):
                        adr=d.efc_J_rowadr[j];nnz=d.efc_J_rownnz[j]
                        jac[j,d.efc_J_colind[adr:adr+nnz]]=d.efc_J[adr:adr+nnz]
                else:jac=d.efc_J.reshape(n,model.nv).copy()
                qacc=d.qacc.copy();constraint=d.qfrc_constraint.copy();passive=d.qfrc_passive.copy();actuator=d.qfrc_actuator.copy()
                it=int(d.solver_niter.max());overflow=0;q,qd=d.qpos.copy(),d.qvel.copy()
                nc=d.ncon;contacts=dict(geom=d.contact.geom[:nc].tolist(),pos=d.contact.pos[:nc].tolist(),
                    dist=d.contact.dist[:nc].tolist(),efc_address=d.contact.efc_address[:nc].tolist())
            if overflow:raise ValueError('GPU overflow')
            vectors=dict(qacc=qacc,qfrc_constraint=constraint,qfrc_passive=passive,qfrc_actuator=actuator,
                joint_state=np.r_[q[qa],qd[va]])
            if not all(np.isfinite(v).all() for v in vectors.values()):raise ValueError('Nonfinite single-step result')
            return dict(nefc=int(n),type_counts=type_counts(types),object_ids_by_type={str(k):sorted(map(int,ids[types==k])) for k in np.unique(types)},
                constraint_generalized_force_by_type={str(k): (jac[types==k].T@force[types==k]).tolist() for k in np.unique(types)},
                solver_iterations=it,contacts=contacts,**{k:v.tolist() for k,v in vectors.items()})
        probes=[];probe_states=[];probe_controls=[]
        # Snapshot native CPU integration state BEFORE selected steps; never
        # feed the GPU's already-diverged state into the shared-state comparison.
        for branch in (0,3,4):
            nd=cpu_reset(native,initial);points=set(probe_steps(old,branch));flat=np.repeat(controls[branch],4,axis=0)
            for step,u in enumerate(flat):
                nd.ctrl[aids]=u
                if step in points:
                    state=np.zeros(mujoco.mj_stateSize(native,dt));mujoco.mj_getState(native,nd,state,dt)
                    probe_states.append(state.copy());probe_controls.append(u.copy())
                    cd=cpu_reset(model,state);cd.ctrl[aids]=u;gd=gpu_reset(cd)
                    wp.to_torch(gd.ctrl)[0,aids]=torch.as_tensor(u,dtype=torch.float32,device='cuda:0')
                    transfer=max(float(np.abs(arr(gd.qpos)[0]-cd.qpos).max()),float(np.abs(arr(gd.qvel)[0]-cd.qvel).max()))
                    if transfer>cfg['initial_state_transfer_max_abs']:raise ValueError('Shared-state transfer exceeds frozen limit')
                    mujoco.mj_step(model,cd);mw.step(gm,gd)
                    cpu,gpu=snapshot(cd,False),snapshot(gd,True)
                    delta=(np.array(gpu['joint_state'])-cpu['joint_state'])/old['scales']
                    probes.append(dict(branch=branch,pre_step_index=step,post_step_ms=(step+1)*5,initial_transfer_max_abs=transfer,
                        shared_state_sha256=hashlib.sha256(state.tobytes()+u.tobytes()).hexdigest(),
                        scaled_state_rmse=float(np.sqrt(np.square(delta).mean())),cpu=cpu,gpu=gpu))
                mujoco.mj_step(native,nd)
                err=np.max(np.abs(np.r_[nd.qpos[qa],nd.qvel[va]]-old['native_substeps'][branch,step]))
                if err>cfg['cpu_reconstruction_max_abs']:raise ValueError('Native substep reconstruction mismatch')
        cpu_values=[];gpu_values=[]
        for ctrl in controls:
            cd=cpu_reset(model,initial);gd=gpu_reset(cd)
            vals=[];gvals=[]
            q,v,u=wp.to_torch(gd.qpos),wp.to_torch(gd.qvel),wp.to_torch(gd.ctrl)
            tensor_ctrl=torch.tensor(ctrl,dtype=u.dtype,device='cuda:0')
            for index,command in enumerate(tensor_ctrl):
                cd.ctrl[aids]=ctrl[index];u[0,aids]=command
                for _ in range(4):
                    mujoco.mj_step(model,cd);mw.step(gm,gd)
                vals.append(np.r_[cd.qpos[qa],cd.qvel[va]])
                gvals.append(torch.cat((q[0,qa],v[0,va])).clone())
            if int(arr(gd.overflow)[0]):raise ValueError('Free-rollout capacity overflow')
            cpu_values.append(vals);gpu_values.append(torch.stack(gvals).cpu().numpy())
        cpu=np.array(cpu_values).reshape(2,7,32,-1);gpu=np.array(gpu_values).reshape(2,7,32,-1)
        comparisons=compare(cpu,gpu,old['scales'],old['epsilon'],cfg)
        raw=a.out/f'{row["split"]}_{number:04d}_trajectories.npz'
        np.savez_compressed(raw,train_cpu=cpu,train_gpu=gpu,actions=old['actions'],epsilon=old['epsilon'],scales=old['scales'],initial_train_integration=initial,
                            probe_states=np.array(probe_states),probe_controls=np.array(probe_controls))
        report['blocks'].append(dict(split=row['split'],block_id=row['block_id'],source_sha256=row['trajectory_sha256'],
            backend=comparisons,passed=all(x['passed'] for x in comparisons),shared_state_probes=probes,
            trajectory_sha256=digest(raw)))
        atomic_json(a.out/'summary.json',report)
        print(f"{a.profile} {a.direction} {number+1}/{len(rows)} {row['block_id']}",flush=True)
    report['status']='completed';atomic_json(a.out/'summary.json',report)


if __name__=='__main__':main()
