"""Single-world CPU/GPU input-rounding factorial with every physics substep recorded.

CPU arithmetic and model parameters stay float64. Rounding the inputs does NOT
turn MuJoCo CPU into a float32 simulator or validate response training labels.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from ral_eight_hour import ROOT, atomic_json, digest, read
from ral_backend_diagnose import compare, diagnostic_direction

CPU_CASES = ('native', 'round_state', 'round_ctrl', 'round_both')


def round32(x):
    return np.asarray(x).astype(np.float32).astype(np.float64)


def contact_bits_cpu(geoms, active, ngeom):
    bits=np.zeros(ngeom*ngeom,dtype=np.uint8)
    geoms=np.asarray(geoms,dtype=int).reshape(-1,2)
    if len(geoms):
        g=np.sort(geoms,axis=-1)
        mask=np.asarray(active,dtype=bool)&(g[:,0]>=0)&(g[:,1]<ngeom)
        bits[g[mask,0]*ngeom+g[mask,1]]=1
    return np.packbits(bits,bitorder='little')


def contact_bits_gpu(geoms, active, ngeom):
    lo,hi=geoms.min(-1).values,geoms.max(-1).values
    valid=active&(lo>=0)&(hi<ngeom)
    ids=(lo*ngeom+hi).clamp(0,ngeom*ngeom-1).long()
    size=((ngeom*ngeom+7)//8)*8
    counts=torch.zeros(size,device=geoms.device,dtype=torch.int32)
    counts.scatter_add_(0,ids,valid.to(torch.int32))
    weights=2**torch.arange(8,device=geoms.device)
    return ((counts>0).reshape(-1,8)*weights).sum(-1).to(torch.uint8)


def main():
    import mujoco
    import mujoco_warp as mw
    import warp as wp
    from mjlab.scene import Scene
    from partymimic.env import make_scene_cfg,make_sim
    from partymimic.experiments.cli import common_robot
    from partymimic.experiments.response import ResponseData
    from partymimic.dr import XiLayout
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','out','selection','params','responses'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--direction-seed',type=int)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    selection=read(a.selection)
    if digest(a.source/'summary.json')!=selection['source_summary_sha256']:
        raise ValueError('Archived single-world source changed')
    cfg=read(ROOT/'configs/ral/backend_check_v1.json')
    cfg.update(horizons=[1,2,4,8,16,32],blocks_per_split=24,selection='Fixed 12 failed + 12 passing per split from diag01')
    robot=common_robot(a.params);layout=XiLayout(robot,'cpu')
    scene=Scene(make_scene_cfg(robot,1),device='cuda:0');model=scene.compile();make_sim(robot).mujoco.apply(model)
    data=mujoco.MjData(model);state_type=mujoco.mjtState.mjSTATE_INTEGRATION
    jj=[model.joint('robot/'+n).id for n in robot.joint_names]
    qa,va=model.jnt_qposadr[jj],model.jnt_dofadr[jj]
    aids=np.array([model.actuator('robot/'+n).id for n in robot.joint_names])
    body=model.body('robot/'+robot.com_body).id;com0=model.body_ipos[body].copy()
    q0,scale=np.asarray(robot.default_joint_pos_list()),np.asarray(robot.action_scale_list())
    report=dict(status='running',config=cfg,smoke_only=a.smoke,direction_seed=a.direction_seed,
        source_summary_sha256=selection['source_summary_sha256'],selection_sha256=digest(a.selection),
        response_test_accessed=False,formal_results=False,C_authorized=False,blocks=[],
        gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__),mujoco_version=mujoco.__version__,
        scope='Input-rounding factorial only; same native physical model, no float32 CPU arithmetic. Counts/forces are not solver residual tolerances.',
        trace_time='Each 5ms substep: q/qd after integration; contacts/constraints are the values computed by that substep before integration. Geom-pair bitsets discard contact multiplicity; counts retain it.',
        geom_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)])
    atomic_json(a.out/'protocol.json',report)
    for split in ('fit','cal'):
        ds=ResponseData(a.responses,split,robot,layout,digest(a.params),'cpu')
        z=ds.arrays;lookup={str(x):i for i,x in enumerate(z['block_id'])}
        selected=selection[split][:1] if a.smoke else selection[split]
        for selected_row in selected:
            block=selected_row['block_id'];i=lookup[block]
            source=a.source/selected_row['file']
            if digest(source)!=selected_row['trajectory_sha256']:raise ValueError('Archived raw block changed')
            with np.load(source,allow_pickle=False) as old:
                initial=old['initial_train_integration'].copy();old_cpu=old['train_cpu'].copy();old_actions=old['actions'].copy()
            if len(initial)!=mujoco.mj_stateSize(model,state_type):raise ValueError('Integration-state layout differs')
            xi=z['xi'][i]
            model.geom_friction[model.geom_bodyid!=0,0]=xi[layout.slices['friction']][0]
            model.body_ipos[body]=com0+xi[layout.slices['com']];mujoco.mj_setConst(model,data)
            base=(z['a_plus'][i]+z['a_minus'][i])/2
            delta=(z['a_plus'][i]-z['a_minus'][i])/2
            native_actions=np.stack([base+f*delta for _ in range(2) for f in (0.,.5,-.5,1.,-1.,2.,-2.)])
            if not np.array_equal(native_actions[:,:old_actions.shape[1]],old_actions):raise ValueError('Original actions differ')
            if a.direction_seed is not None:
                delta=z['epsilon'][i]*diagnostic_direction(a.direction_seed,block,base.shape)/scale
            actions=np.stack([base+f*delta for _ in range(2) for f in (0.,.5,-.5,1.,-1.,2.,-2.)])
            controls=q0+scale*actions+xi[layout.slices['joint_offset']]
            H=actions.shape[1];traces={};trajectories={}
            def cpu_run(init,ctrl):
                values=[];stats=[];detected=[];active=[]
                for branch in ctrl:
                    mujoco.mj_setState(model,data,init,state_type);mujoco.mj_forward(model,data)
                    vv=[];ss=[];dd=[];aa=[]
                    for command in branch:
                        data.ctrl[aids]=command
                        for _ in range(4):
                            mujoco.mj_step(model,data)
                            vv.append(np.r_[data.qpos[qa],data.qvel[va]])
                            geoms=data.contact.geom[:data.ncon]
                            dd.append(contact_bits_cpu(geoms,np.ones(data.ncon,dtype=bool),model.ngeom))
                            aa.append(contact_bits_cpu(geoms,data.contact.efc_address[:data.ncon]>=0,model.ngeom))
                            ss.append([data.nefc,data.ncon,int(data.solver_niter.max()),float(np.linalg.norm(data.qfrc_constraint)),0])
                    values.append(vv);stats.append(ss);detected.append(dd);active.append(aa)
                return np.asarray(values),np.asarray(stats),np.asarray(detected),np.asarray(active)
            for case in CPU_CASES:
                init=round32(initial) if case in ('round_state','round_both') else initial
                ctrl=round32(controls) if case in ('round_ctrl','round_both') else controls
                values,stats,detected,active=cpu_run(init,ctrl)
                traces[case+'_substeps']=values
                traces[case+'_stats']=stats
                traces[case+'_contacts']=detected;traces[case+'_active_contacts']=active
                trajectories[case]=values[:,3::4].reshape(2,7,H,-1)
            # Check original native replay even when a new perturbation direction is used.
            if a.direction_seed is None:original=trajectories['native'][:,:,:old_cpu.shape[2]]
            else:
                vals,*_=cpu_run(initial,q0+scale*native_actions[:,:old_cpu.shape[2]]+xi[layout.slices['joint_offset']])
                original=vals[:,3::4].reshape(old_cpu.shape)
            replay=float(np.abs(original-old_cpu).max())
            if replay>cfg['cpu_reconstruction_max_abs']:raise ValueError(f'Native CPU replay differs: {block} {replay}')
            mujoco.mj_setState(model,data,initial,state_type);mujoco.mj_forward(model,data)
            gpu_values=[];gpu_stats=[];gpu_contacts=[];gpu_active=[];transfer=0.;raw_state_transfer=0.
            with wp.ScopedDevice('cuda:0'):
                gm=mw.put_model(model)
                for ctrl in controls:
                    gd=mw.put_data(model,data,nworld=1,nconmax=robot.nconmax,njmax=robot.njmax)
                    q,v,u=wp.to_torch(gd.qpos),wp.to_torch(gd.qvel),wp.to_torch(gd.ctrl)
                    raw_state_transfer=max(raw_state_transfer,float(np.abs(q.cpu().numpy()-data.qpos).max()),float(np.abs(v.cpu().numpy()-data.qvel).max()))
                    for tensor,expected in ((q,data.qpos),(v,data.qvel),(wp.to_torch(gd.qacc_warmstart),data.qacc_warmstart)):
                        transfer=max(transfer,float(np.abs(tensor.cpu().numpy()-round32(expected)).max()))
                    tensor_ctrl=torch.tensor(ctrl,dtype=u.dtype,device='cuda:0')
                    contact_index=torch.arange(gd.naconmax,device='cuda:0')
                    values=[];stats=[];detected=[];active=[]
                    for command in tensor_ctrl:
                        u[:,aids]=command
                        for _ in range(4):
                            mw.step(gm,gd)
                            values.append(torch.cat((q[0,qa],v[0,va])).clone())
                            ncon=wp.to_torch(gd.nacon)[0]
                            valid=(contact_index<ncon)&(wp.to_torch(gd.contact.worldid)==0)
                            geoms=wp.to_torch(gd.contact.geom)
                            detected.append(contact_bits_gpu(geoms,valid,model.ngeom))
                            active.append(contact_bits_gpu(geoms,valid&(wp.to_torch(gd.contact.efc_address)>=0).any(-1),model.ngeom))
                            stats.append(torch.stack((wp.to_torch(gd.nefc)[0].float(),ncon.float(),
                                wp.to_torch(gd.solver_niter)[0].float(),wp.to_torch(gd.qfrc_constraint)[0].norm(),
                                wp.to_torch(gd.overflow)[0].float())).clone())
                    gpu_values.append(torch.stack(values).cpu().numpy());gpu_stats.append(torch.stack(stats).cpu().numpy())
                    gpu_contacts.append(torch.stack(detected).cpu().numpy());gpu_active.append(torch.stack(active).cpu().numpy())
            traces['gpu_substeps']=np.array(gpu_values);traces['gpu_stats']=np.array(gpu_stats)
            traces['gpu_contacts']=np.array(gpu_contacts);traces['gpu_active_contacts']=np.array(gpu_active)
            gpu=traces['gpu_substeps'][:,3::4].reshape(2,7,H,-1)
            scales=np.r_[np.full(robot.num_joints,ds.q_scale),np.full(robot.num_joints,ds.v_scale)]
            if not all(np.isfinite(x).all() for x in traces.values()):raise ValueError('Nonfinite traces')
            comparisons={case:compare(trajectories[case],gpu,scales,z['epsilon'][i],cfg) for case in CPU_CASES}
            row=dict(split=split,block_id=block,previous_overall_passed=selected_row['passed'],
                cpu_original_replay_max_abs=replay,gpu_transfer_vs_rounded_cpu_max_abs=transfer,
                initial_state_transfer_max_abs=raw_state_transfer,
                comparisons=comparisons,backend=comparisons['native'],
                rounding_effect_scope='Changed reference inputs; passing a rounded comparison does not override original native failures',
                substep_stats_columns=['nefc','detected_ncon','solver_niter','qfrc_constraint_l2','overflow'],
                contact_mismatch_fraction={case:float(np.any(traces[case+'_active_contacts']!=traces['gpu_active_contacts'],axis=-1).mean()) for case in CPU_CASES},
                constraint_count_mismatch_fraction={case:float((traces[case+'_stats'][...,0]!=traces['gpu_stats'][...,0]).mean()) for case in CPU_CASES},
                gpu_solver_iterations_max=int(traces['gpu_stats'][...,2].max()),
                gpu_overflow_any=bool(traces['gpu_stats'][...,4].any()))
            row['passed']=all(x['passed'] for x in comparisons['native']) and not row['gpu_overflow_any'] and transfer==0. and raw_state_transfer<=cfg['initial_state_transfer_max_abs']
            filename=f'{split}_{i:04d}_trajectories.npz'
            np.savez_compressed(a.out/filename,**traces,**{'cpu_'+k:v for k,v in trajectories.items()},
                train_cpu=trajectories['native'],train_gpu=gpu,actions=actions,epsilon=z['epsilon'][i],scales=scales,initial_train_integration=initial)
            row['trajectory_sha256']=digest(a.out/filename);report['blocks'].append(row)
            atomic_json(a.out/'summary.json',report)
            print(f'precision backend {split} {block}: native_pass={row["passed"]}',flush=True)
            if row['gpu_overflow_any'] or transfer!=0.:raise ValueError('Overflow or unexpected transfer: retain diagnostics')
    report.update(status='completed',passed=all(r['passed'] for r in report['blocks']))
    atomic_json(a.out/'summary.json',report)


if __name__=='__main__':main()
