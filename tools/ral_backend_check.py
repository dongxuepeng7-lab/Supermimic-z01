"""Reconstruct archived CPU pairs, then compare the training scene on CPU/Warp GPU."""
import argparse
import hashlib
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
    cfg = read(CONFIG)
    a.out.mkdir(parents=True, exist_ok=False)
    frozen = shared_inputs(a)
    report = dict(status="running", config=cfg, config_sha256=digest(CONFIG), inputs=frozen,
                  gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
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
            with wp.ScopedDevice("cuda:0"):
                gm = mw.put_model(train)
                gd = mw.put_data(train, td, nworld=worlds, nconmax=robot.nconmax, njmax=robot.njmax)
                qgpu, vgpu, ctrl = wp.to_torch(gd.qpos), wp.to_torch(gd.qvel), wp.to_torch(gd.ctrl)
                transfer_error = max(float(np.abs(qgpu.cpu().numpy()-td.qpos).max()),
                                     float(np.abs(vgpu.cpu().numpy()-td.qvel).max()))
                controls = torch.tensor(q0+scale*actions+xi[layout.slices["joint_offset"]], device="cuda:0", dtype=ctrl.dtype)
                # Capture one physics substep, with persistent control/state buffers.
                with wp.ScopedCapture() as capture:
                    mw.step(gm, gd)
                values = []
                for t in range(H):
                    ctrl[:, ta] = controls[:, t]
                    for _ in range(4):
                        wp.capture_launch(capture.graph)
                    values.append(torch.cat((qgpu[:, tq], vgpu[:, tv]), -1).clone())
                trajectories["train_gpu"] = torch.stack(values, 1).cpu().numpy().reshape(2, len(branch_specs), H, -1)
            pair_cpu, train_cpu, train_gpu = (trajectories[k] for k in ("pair_cpu", "train_cpu", "train_gpu"))
            replay_error = max(float(np.abs(pair_cpu[0, 3]-z["s_plus"][index, :H, 13:]).max()),
                               float(np.abs(pair_cpu[0, 4]-z["s_minus"][index, :H, 13:]).max()))
            cpu_repeat = max(float(np.abs(t[0]-t[1]).max()) for t in (pair_cpu, train_cpu))
            backend = compare(train_cpu, train_gpu, scales, z["epsilon"][index], cfg)
            end_to_end = compare(pair_cpu, train_gpu, scales, z["epsilon"][index], cfg)
            row = dict(split=split, block_id=block, initial_reconstruction_max_abs=initial_error,
                saved_pair_max_abs=replay_error, cpu_repeat_max_abs=cpu_repeat,
                initial_state_transfer_max_abs=transfer_error, backend=backend, archived_to_training=end_to_end,
                cpu_scene_scaled_rmse=rms((pair_cpu-train_cpu)/scales))
            row["passed"] = (max(initial_error, replay_error) <= cfg["cpu_reconstruction_max_abs"]
                and cpu_repeat <= cfg["cpu_repeat_max_abs"] and transfer_error <= cfg["initial_state_transfer_max_abs"]
                and all(r["passed"] for r in backend+end_to_end))
            filename = f"{split}_{int(index):04d}_trajectories.npz"
            np.savez_compressed(a.out / filename, **trajectories, actions=actions, epsilon=z["epsilon"][index],
                                scales=scales, initial_pair_integration=saved["pair"], initial_train_integration=saved["train"])
            row["trajectory_sha256"] = digest(a.out / filename)
            report["blocks"].append(row)
            atomic_json(a.out / "summary.json", report)
    if shared_inputs(a) != frozen:
        raise ValueError("Inputs changed during backend check")
    report.update(status="completed", passed=all(r["passed"] for r in report["blocks"]),
                  full_prespecified_sample=not a.smoke,
                  interpretation="Development sample only; runtime uses training scene/options and the Warp backend. No formal response equivalence or automatic C.")
    atomic_json(a.out / "summary.json", report)
    print(f"Backend screen passed={report['passed']}; blocks={len(report['blocks'])}; smoke={a.smoke}", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--motions", type=Path, default=ROOT / "data/motions/z01_lafan_v2_clean")
    p.add_argument("--params", type=Path, default=ROOT / "release/ral/params.csv")
    p.add_argument("--responses", type=Path, default=ROOT / "data/ral_pairs_v1/manifest.json")
    p.add_argument("--smoke", action="store_true", help="One block per split; validates execution only")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    os.chdir(ROOT)
    for key in ("out", "motions", "params", "responses"):
        setattr(a, key, getattr(a, key).resolve())
    if a.out.exists() or a.out.parent.joinpath(a.out.name+"_review.tar.gz").exists():
        p.error("Use a new output directory")
    if not torch.cuda.is_available() or "3090" not in torch.cuda.get_device_name():
        p.error("Backend validation is assigned to RTX 3090")
    if a.worker:
        return run_check(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock = ROOT / "logs/ral_8h" / (".gpu_"+hashlib.sha256(visible.encode()).hexdigest()[:12]+".lock")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    with gpu_lock(lock) as fd:
        outcome = execute(command, a.out.parent / (a.out.name+"_worker.log"), time.monotonic()+7200, cancelled, fd)
    if outcome != "completed":
        if a.out.exists():
            atomic_json(a.out / "failure.json", dict(status=outcome, reason="Inspect sibling worker log"))
            summary_path = a.out / "summary.json"
            summary = read(summary_path) if summary_path.exists() else {}
            summary.update(status="stopped", passed=False, reason=f"Worker {outcome}; inspect sibling worker log",
                           formal_results=False, C_authorized=False)
            atomic_json(summary_path, summary)
        raise RuntimeError(f"Backend worker {outcome}: {a.out.parent / (a.out.name+'_worker.log')}")
    archive = a.out.parent / (a.out.name+"_review.tar.gz")
    with tarfile.open(archive, "x:gz", compresslevel=1) as tar:
        tar.add(a.out, arcname=a.out.name)
    print(f"检查完成，passed={read(a.out / 'summary.json')['passed']}；回传 {archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
