"""Isolate graph capture and world packing without changing historical labels or tolerances."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import signal
import sys
import tarfile
import time

import numpy as np
import torch

from ral_eight_hour import ROOT, atomic_json, digest, execute, gpu_lock, read

CONFIG = ROOT / "configs/ral/backend_check_v1.json"


def select_blocks(ids, count):
    order = np.argsort(ids.astype(str), kind="stable")
    return order[np.unique(np.linspace(0, len(order)-1, min(count, len(order))).astype(int))]


def rms(a):
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite backend trajectory")
    return float(np.sqrt(np.square(a).mean()))


def diagnostic_direction(seed, block, shape):
    """One target-radian direction over the full sequence, common to all profiles."""
    value = int(hashlib.sha256(f"{seed}:{block}".encode()).hexdigest()[:16], 16)
    direction = np.random.default_rng(value).normal(size=shape)
    return direction / np.linalg.norm(direction)


def compare(cpu, gpu, scales, epsilon, cfg):
    """Arrays: repeats x (base, plus/minus per amplitude) x time x (q,qd)."""
    rows = []
    for k, factor in enumerate(cfg["epsilon_factors"]):
        plus, minus = 1+2*k, 2+2*k
        cr = (cpu[:, plus]-cpu[:, minus])/(2*epsilon*factor)/scales
        gr = (gpu[:, plus]-gpu[:, minus])/(2*epsilon*factor)/scales
        for h in cfg["horizons"]:
            signal_rms = rms(cr[0, :h])
            error = rms(gr[:, :h]-cr[:, :h])
            repeat = rms(gr[0, :h]-gr[1, :h])
            forward = rms((gpu[:, [0, plus, minus], :h]-cpu[:, [0, plus, minus], :h])/scales)
            limit = max(cfg["response_absolute_scaled_rmse_floor"], cfg["response_relative_rmse_max"]*signal_rms)
            repeat_limit = max(cfg["gpu_repeat_absolute_scaled_rmse_floor"], cfg["gpu_repeat_relative_rmse_max"]*signal_rms)
            rows.append(dict(factor=factor, horizon=h, cpu_response_rms=signal_rms,
                zero_response_rmse=signal_rms, cross_response_rmse=error,
                relative_response_rmse=error/signal_rms if signal_rms > 0 else None,
                gpu_repeat_response_rmse=repeat, forward_scaled_rmse=forward,
                response_limit=limit, repeat_limit=repeat_limit,
                passed=error <= limit and repeat <= repeat_limit and forward <= cfg["forward_scaled_rmse_max"]))
    return rows


def run_check(a):
    import mujoco
    import mujoco_warp as mw
    import warp as wp
    from mjlab.scene import Scene
    from partymimic.dr import XiLayout
    from partymimic.env import make_scene_cfg, make_sim
    from partymimic.experiments.cli import common_robot
    from partymimic.experiments.response import ResponseData
    from partymimic.kinematics import Kinematics
    from partymimic.motion import MotionLib
    from partymimic.rotations import mat_to_quat, quat_to_mat
    from partymimic.mdp import flatten_state
    from ral_structure import shared_inputs
    cfg = read(a.config)
    a.out.mkdir(parents=True, exist_ok=False)
    frozen = shared_inputs(a)
    report = dict(status="running", config=cfg, config_sha256=digest(a.config), inputs=frozen,
                  profile=a.profile, direction_seed=a.direction_seed,
                  gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                  mujoco_warp_version=importlib.metadata.version("mujoco-warp"),
                  warp_version=wp.__version__, mjlab_version=importlib.metadata.version("mjlab"),
                  mujoco_version=mujoco.__version__, response_test_accessed=False,
                  formal_results=False, C_authorized=False, smoke_only=a.smoke, blocks=[])
    atomic_json(a.out / "protocol.json", report)
    robot = common_robot(a.params)
    layout, kin = XiLayout(robot, "cpu"), Kinematics(robot, "cpu")
    q0, scale = np.asarray(robot.default_joint_pos_list()), np.asarray(robot.action_scale_list())
    spec = robot.robot_cfg().build().spec.copy()
    spec.worldbody.add_geom(name="pair_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                           size=[0, 0, .05], friction=[.3, .005, .0001])
    pair = spec.compile()
    pair.opt.timestep = .005
    pair.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    pair.opt.iterations, pair.opt.ls_iterations, pair.opt.tolerance = 100, 50, 1e-6
    scene = Scene(make_scene_cfg(robot, 1), device="cuda:0")
    train = scene.compile()
    make_sim(robot).mujoco.apply(train)
    def indices(model, prefix):
        joints = [model.joint(prefix+n).id for n in robot.joint_names]
        return model.jnt_qposadr[joints], model.jnt_dofadr[joints], np.array([model.actuator(prefix+n).id for n in robot.joint_names])
    qa, va, aids = indices(pair, "")
    tq, tv, ta = indices(train, "robot/")
    if (pair.nq, pair.nv, pair.na) != (train.nq, train.nv, train.na) or not np.array_equal(qa, tq) or not np.array_equal(va, tv):
        raise ValueError("Integration-state mapping needs explicit remapping for this model")
    com, tcom = pair.body(robot.com_body).id, train.body("robot/"+robot.com_body).id
    com0, tcom0 = pair.body_ipos[com].copy(), train.body_ipos[tcom].copy()
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    pd, td = mujoco.MjData(pair), mujoco.MjData(train)
    branch_specs = [(0., "base")] + [(f*s, side) for f in cfg["epsilon_factors"] for s, side in ((1, "plus"), (-1, "minus"))]
    H, worlds = max(cfg["horizons"]), len(branch_specs)*cfg["repeats"]
    for split in cfg["splits"]:
        ds = ResponseData(a.responses, split, robot, layout, digest(a.params))
        z = ds.arrays
        scales = np.r_[np.full(robot.num_joints, ds.q_scale), np.full(robot.num_joints, ds.v_scale)]
        ids = select_blocks(z["block_id"], 1 if a.smoke else cfg["blocks_per_split"])
        if a.selection and not a.smoke:
            selected = set(read(a.selection)[split])
            if not selected.issubset(set(z["block_id"].astype(str))):
                raise ValueError("Diagnostic selection has unknown block IDs")
            ids = np.array([i for i in ids if str(z["block_id"][i]) in selected], dtype=int)
        for index in ids:
            block = str(z["block_id"][index])
            print(f"[3090 backend] {split}: {block}", flush=True)
            clip, _, start = block.rsplit(":", 2)
            lib = MotionLib([str(a.motions / (clip+".npz"))], kin, "cpu", robot.load_motion)
            xi = z["xi"][index]
            for model, data, body, original in ((pair, pd, com, com0), (train, td, tcom, tcom0)):
                model.geom_friction[model.geom_bodyid != 0, 0] = xi[layout.slices["friction"]][0]
                model.body_ipos[body] = original + xi[layout.slices["com"]]
                mujoco.mj_setConst(model, data)
                mujoco.mj_resetData(model, data)
            p, R, v, w, q, dq = (t[0].numpy() for t in lib.ref_state(torch.tensor([int(start)])))
            pd.qpos[:3], pd.qpos[3:7], pd.qpos[qa] = p, mat_to_quat(torch.tensor(R)).numpy(), q
            pd.qvel[:3], pd.qvel[3:6], pd.qvel[va] = v, R.T@w, dq
            def step(model, data, acts, qadr, vadr, action):
                data.ctrl[acts] = q0+scale*action+xi[layout.slices["joint_offset"]]
                for _ in range(4):
                    mujoco.mj_step(model, data)
                return np.r_[data.qpos[qadr], data.qvel[vadr]]
            for t in range(20):
                step(pair, pd, aids, qa, va, (lib.q[int(start)+t+1].numpy()-q0)/scale)
            R0 = quat_to_mat(torch.tensor(pd.qpos[3:7]))
            s0 = flatten_state((torch.tensor(pd.qpos[:3]), R0, torch.tensor(pd.qvel[:3]),
                 R0@torch.tensor(pd.qvel[3:6]), torch.tensor(pd.qpos[qa]), torch.tensor(pd.qvel[va]))).numpy()
            initial_error = float(np.abs(s0-z["s0"][index]).max())
            # Map complete integration state; only world body names differ. Applied forces
            # and mocap/user inputs must be zero in these archived position-control pairs.
            for name in ("act", "qpos", "qvel", "qacc_warmstart", "qfrc_applied"):
                getattr(td, name)[:] = getattr(pd, name)
            td.time = pd.time
            td.ctrl[ta] = pd.ctrl[aids]
            if any(np.any(getattr(pd, name)) for name in ("xfrc_applied", "mocap_pos", "userdata")) or pair.nplugin:
                raise ValueError("Unsupported nonzero external/mocap/plugin integration state")
            saved = {}
            for name, model, data in (("pair", pair, pd), ("train", train, td)):
                saved[name] = np.empty(mujoco.mj_stateSize(model, state_type))
                mujoco.mj_getState(model, data, saved[name], state_type)
            base = (z["a_plus"][index, :H]+z["a_minus"][index, :H])/2
            delta = (z["a_plus"][index, :H]-z["a_minus"][index, :H])/2
            stored_delta = delta.copy()
            if a.direction_seed is not None:
                direction = diagnostic_direction(a.direction_seed, block, z["a_plus"][index].shape)
                delta = z["epsilon"][index] * direction[:H] / scale
            actions = np.stack([base+factor*delta for _ in range(2) for factor, _ in branch_specs])
            trajectories = {}
            for name, model, data, aa, qq, vv in (("pair_cpu", pair, pd, aids, qa, va), ("train_cpu", train, td, ta, tq, tv)):
                sims = []
                for action in actions:
                    mujoco.mj_setState(model, data, saved["pair" if model is pair else "train"], state_type)
                    mujoco.mj_forward(model, data)
                    sims.append(np.stack([step(model, data, aa, qq, vv, act) for act in action]))
                trajectories[name] = np.stack(sims).reshape(2, len(branch_specs), H, -1)
            mujoco.mj_setState(train, td, saved["train"], state_type)
            mujoco.mj_forward(train, td)
            # Restore from one common CPU integration state for every launch batch.
            batches = [actions] if a.profile != "eager_single" else [x[None] for x in actions]
            gpu_values, solver_traces, constraint_traces, overflow_traces = [], [], [], []
            capture_mutations, warmstart_errors = [], []
            transfer_error = 0.
            with wp.ScopedDevice("cuda:0"):
                gm = mw.put_model(train)
                for batch in batches:
                    gd = mw.put_data(train, td, nworld=len(batch), nconmax=robot.nconmax, njmax=robot.njmax)
                    qgpu, vgpu, ctrl = wp.to_torch(gd.qpos), wp.to_torch(gd.qvel), wp.to_torch(gd.ctrl)
                    transfer_error = max(transfer_error, float(np.abs(qgpu.cpu().numpy()-td.qpos).max()),
                                         float(np.abs(vgpu.cpu().numpy()-td.qvel).max()))
                    warmstart_errors.append(float(np.abs(wp.to_torch(gd.qacc_warmstart).cpu().numpy()-td.qacc_warmstart).max()))
                    controls = torch.tensor(q0+scale*batch+xi[layout.slices["joint_offset"]], device="cuda:0", dtype=ctrl.dtype)
                    q_before, v_before = qgpu.clone(), vgpu.clone()
                    if a.profile == "graph_batch":
                        with wp.ScopedCapture() as capture:
                            mw.step(gm, gd)
                        capture_mutations.append(max(float((qgpu-q_before).abs().max()), float((vgpu-v_before).abs().max())))
                    values, niters, nefcs, overflows = [], [], [], []
                    for t in range(H):
                        ctrl[:, ta] = controls[:, t]
                        for _ in range(4):
                            if a.profile == "graph_batch":
                                wp.capture_launch(capture.graph)
                            else:
                                mw.step(gm, gd)
                        values.append(torch.cat((qgpu[:, tq], vgpu[:, tv]), -1).clone())
                        niters.append(wp.to_torch(gd.solver_niter).clone())
                        nefcs.append(wp.to_torch(gd.nefc).clone())
                        overflows.append(wp.to_torch(gd.overflow).clone())
                    gpu_values.append(torch.stack(values, 1).cpu().numpy())
                    solver_traces.append(torch.stack(niters, 1).cpu().numpy())
                    constraint_traces.append(torch.stack(nefcs, 1).cpu().numpy())
                    overflow_traces.append(torch.stack(overflows, 1).cpu().numpy())
            trajectories["train_gpu"] = np.concatenate(gpu_values).reshape(2, len(branch_specs), H, -1)
            diagnostics = dict(gpu_solver_niter=np.concatenate(solver_traces),
                gpu_nefc=np.concatenate(constraint_traces), gpu_overflow=np.concatenate(overflow_traces))
            pair_cpu, train_cpu, train_gpu = (trajectories[k] for k in ("pair_cpu", "train_cpu", "train_gpu"))
            if a.direction_seed is None:
                original_plus, original_minus = pair_cpu[0, 3], pair_cpu[0, 4]
            else:
                original = []
                for action in (base+stored_delta, base-stored_delta):
                    mujoco.mj_setState(pair, pd, saved["pair"], state_type)
                    mujoco.mj_forward(pair, pd)
                    original.append(np.stack([step(pair, pd, aids, qa, va, act) for act in action]))
                original_plus, original_minus = original
            replay_error = max(float(np.abs(original_plus-z["s_plus"][index, :H, 13:]).max()),
                               float(np.abs(original_minus-z["s_minus"][index, :H, 13:]).max()))
            cpu_repeat = max(float(np.abs(t[0]-t[1]).max()) for t in (pair_cpu, train_cpu))
            backend = compare(train_cpu, train_gpu, scales, z["epsilon"][index], cfg)
            end_to_end = compare(pair_cpu, train_gpu, scales, z["epsilon"][index], cfg)
            row = dict(split=split, block_id=block, profile=a.profile, direction_seed=a.direction_seed,
                capture_state_mutation_max_abs=max(capture_mutations, default=0.),
                warmstart_transfer_max_abs=max(warmstart_errors),
                gpu_solver_iterations_max=int(diagnostics["gpu_solver_niter"].max()),
                gpu_constraints_max=int(diagnostics["gpu_nefc"].max()),
                gpu_overflow_any=bool(diagnostics["gpu_overflow"].any()), initial_reconstruction_max_abs=initial_error,
                saved_pair_max_abs=replay_error, cpu_repeat_max_abs=cpu_repeat,
                initial_state_transfer_max_abs=transfer_error, backend=backend, archived_to_training=end_to_end,
                cpu_scene_scaled_rmse=rms((pair_cpu-train_cpu)/scales))
            row["passed"] = (max(initial_error, replay_error) <= cfg["cpu_reconstruction_max_abs"]
                and cpu_repeat <= cfg["cpu_repeat_max_abs"] and transfer_error <= cfg["initial_state_transfer_max_abs"]
                and all(r["passed"] for r in backend+end_to_end)
                and not row["gpu_overflow_any"] and row["capture_state_mutation_max_abs"] == 0.)
            filename = f"{split}_{int(index):04d}_trajectories.npz"
            np.savez_compressed(a.out / filename, **trajectories, **diagnostics, actions=actions, epsilon=z["epsilon"][index],
                                scales=scales, initial_pair_integration=saved["pair"], initial_train_integration=saved["train"])
            row["trajectory_sha256"] = digest(a.out / filename)
            report["blocks"].append(row)
            atomic_json(a.out / "summary.json", report)
    if shared_inputs(a) != frozen:
        raise ValueError("Inputs changed during backend check")
    report.update(status="completed", passed=all(r["passed"] for r in report["blocks"]),
                  full_prespecified_sample=not a.smoke and not a.selection,
                  interpretation="Development sample only; runtime uses training scene/options and the Warp backend. No formal response equivalence or automatic C; altered directions are diagnostic only.")
    atomic_json(a.out / "summary.json", report)
    print(f"Backend screen passed={report['passed']}; blocks={len(report['blocks'])}; smoke={a.smoke}", flush=True)
    return 0



def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "motions", "params", "responses", "config"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--profile", choices=("graph_batch", "eager_batch", "eager_single"), required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--direction-seed", type=int)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if read(a.config)["splits"] != ["fit", "cal"]:
        raise ValueError("Only fit/cal diagnostics are allowed")
    cfg, frozen = read(a.config), read(CONFIG)
    if any(cfg[k] != frozen[k] for k in frozen if k not in ("horizons", "blocks_per_split", "selection")):
        raise ValueError("Backend amplitudes, repeats and original thresholds must remain frozen")
    if not cfg["horizons"] or any(type(h) is not int or h < 1 or h > 32 for h in cfg["horizons"]):
        raise ValueError("Diagnostic horizons must stay within the recorded 32 steps")
    return run_check(a)


if __name__ == "__main__":
    raise SystemExit(main())
