"""Isolated GPU jobs used by ral_eight_hour.py; never opens locked test arrays."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from ral_eight_hour import ROOT, atomic_json, digest, read, validate_plan
from partymimic.experiments.cli import common_robot, load_context, preflight, training_command
from partymimic.experiments.protocol import asset_hash, code_hash
from partymimic.experiments.runtime import object_hash, state_hash

STOP = False


def stop_requested(*_):
    global STOP
    STOP = True


def torch_save(path, value):
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def state_digest(ck, names):
    h = hashlib.sha256()
    for name in sorted(names):
        for key, value in sorted(ck[name].items()):
            value = value.detach().cpu().contiguous()
            h.update(f"{name}/{key}:{value.dtype}:{list(value.shape)}".encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tools_inventory():
    return {str(p.relative_to(ROOT)): digest(p) for p in
            (ROOT / "tools/ral_eight_hour.py", ROOT / "tools/ral_eight_hour_worker.py")}


def prepare(a):
    session = Path(a.session)
    plan = read(session / "plan.json")
    validate_plan(plan)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    gpu = torch.cuda.get_device_name()
    if a.machine not in gpu and not a.allow_gpu_mismatch:
        raise RuntimeError(f"Requested {a.machine}, detected {gpu}; use the matching script")
    ready = preflight(a.params, a.motions, "C", a.responses)
    if not ready["ready"]:
        raise RuntimeError(json.dumps(ready, ensure_ascii=False))
    meta = read(a.responses)
    if meta["source_type"] != "physics_paired":
        raise ValueError("Eight-hour research queues require actual physics_paired data")
    if any(meta["motion_hashes"].get(k) != v for k, v in ready["motions"].items()) or meta["motion_hashes"] != ready["motions"]:
        raise ValueError("Motion library differs from the paired-response source")
    base = read(ROOT / plan["base_config"])
    if base["training"]["wm_amp"] != "none":
        raise ValueError("These comparisons require FP32 world models")
    atomic_json(session / "base_config.json", base)
    shared = dict(code_sha256=code_hash(), asset_sha256=asset_hash(), params_sha256=digest(a.params),
                  response_sha256=digest(a.responses), motion_sha256=object_hash(ready["motions"]))
    provenance = dict(schema_version=1, gpu=gpu, requested_machine=a.machine,
                      gpu_mismatch_allowed=a.allow_gpu_mismatch,
                      torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      tools=tools_inventory(), plan_sha256=digest(session / "plan.json"),
                      base_config_sha256=digest(session / "base_config.json"),
                      paths=dict(params=a.params, motions=a.motions, responses=a.responses),
                      motions=ready["motions"], shared=shared, formal_results=False,
                      test_outcomes_inspected=False, backend_validation="CPU/GPU response equivalence still pending")
    atomic_json(session / "provenance.json", provenance)
    print(f"Preflight passed: {gpu}, {ready['motion_count']} real motions, frozen fit/cal inputs", flush=True)


def verify_inputs(session, provenance):
    if tools_inventory() != provenance["tools"]:
        raise ValueError("Queue tools changed during the session")
    if digest(session / "plan.json") != provenance["plan_sha256"] or digest(session / "base_config.json") != provenance["base_config_sha256"]:
        raise ValueError("Frozen session plan/config changed")
    paths = provenance["paths"]
    actual = dict(code_sha256=code_hash(), asset_sha256=asset_hash(), params_sha256=digest(paths["params"]),
                  response_sha256=digest(paths["responses"]),
                  motion_sha256=object_hash({p.name: digest(p) for p in sorted(Path(paths["motions"]).glob("*.npz"))}))
    if actual != provenance["shared"]:
        raise ValueError("Frozen source/assets/data changed during the session")


def datasets(robot, layout, provenance):
    from partymimic.experiments.response import ResponseData
    p = provenance["paths"]
    return [ResponseData(p["responses"], split, robot, layout, provenance["shared"]["params_sha256"], "cuda:0")
            for split in ("fit", "cal")]


@torch.no_grad()
def evaluate(wm, ds, layout):
    from partymimic.experiments.response import paired_predictions
    from partymimic.mdp import unflatten_state
    forward, response, signal_energy, qerror = [], [], [], []
    for i in range(0, ds.count, 128):
        batch = {k: v[i:i+128] for k, v in ds.data.items()}
        plus, minus = paired_predictions(wm, batch, layout.normalize)
        tp, tm = unflatten_state(batch["s_plus"]), unflatten_state(batch["s_minus"])
        if not all(bool(torch.isfinite(v).all()) for state in (plus, minus) for v in state):
            raise FloatingPointError("Nonfinite forward prediction")
        scale = wm.inc_norm.std().clamp(min=0.001)
        forward.append((((wm.state_error(plus, tp)/scale).square().mean(-1)
                         + (wm.state_error(minus, tm)/scale).square().mean(-1))/2).cpu())
        eps = 2 * batch["epsilon"][:, None, None]
        target = torch.cat(((tp[4]-tm[4])/eps/ds.q_scale, (tp[5]-tm[5])/eps/ds.v_scale), -1)
        pred = torch.cat(((plus[4]-minus[4])/eps/ds.q_scale, (plus[5]-minus[5])/eps/ds.v_scale), -1)
        response.append((pred-target).square().mean(-1).cpu())
        signal_energy.append(target.square().mean(-1).cpu())
        qerror.append((((plus[4]-tp[4]).square().mean(-1)+(minus[4]-tm[4]).square().mean(-1))/2).cpu())
    f, r, sig, qe = [torch.cat(x).double().numpy() for x in (forward, response, signal_energy, qerror)]
    sessions = ds.arrays["session_id"].astype(str)
    if ds.horizon != 32:
        raise ValueError("This frozen eight-hour plan expects full 32-step response arrays")
    return dict(blocks=ds.count, horizon=ds.horizon, forward_mse=float(f.mean()),
                q_rmse32_rad=float(np.sqrt(qe.mean())), response_rmse4=float(np.sqrt(r[:,:4].mean())),
                response_rmse8=float(np.sqrt(r[:,:8].mean())), response_rmse32=float(np.sqrt(r.mean())),
                zero_response_rmse8=float(np.sqrt(sig[:,:8].mean())),
                zero_response_rmse32=float(np.sqrt(sig.mean())),
                sessions={s: dict(blocks=int((sessions == s).sum()), response_rmse8=float(np.sqrt(r[sessions == s,:8].mean()))) for s in sorted(set(sessions))},
                response_rmse_by_horizon=np.sqrt(r.mean(0)).tolist())


def finite_weights(ck, names):
    if not all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for k in names for v in ck[k].values()):
        raise FloatingPointError("Nonfinite checkpoint weights")


def comparison_key(provenance, kind, settings):
    # Absolute paths, seed, response objective and machine label are excluded.
    return object_hash(dict(shared=provenance["shared"], tools=provenance["tools"], kind=kind,
                            torch_version=provenance["torch_version"], settings=settings))


def train_world_model(session, directory, job, plan, provenance):
    from partymimic.dr import XiLayout
    from partymimic.experiments.models import make_world_model
    from partymimic.experiments.response import initialize_pair_normalizers, paired_losses
    from partymimic.optim import MuonAdam
    import partymimic.networks as networks

    settings = plan["world_model"]
    robot = common_robot(provenance["paths"]["params"])
    layout = XiLayout(robot, "cuda:0")
    fit, cal = datasets(robot, layout, provenance)
    networks.RunningNorm.no_sync = True
    networks.AMP_DTYPE = None
    torch.manual_seed(job["seed"])
    args = dict(experiment_model="actuator", experiment_group=job["group"], seed=job["seed"],
                parameter_manifest_sha256=provenance["shared"]["params_sha256"],
                wm_hidden=settings["wm_hidden"], wm_structured=True, wm_amp="none", friction_speed=0.01,
                amp="none", **{k: v for k, v in job["response"].items() if k != "name"})
    wm = make_world_model(robot, layout, args).cuda()
    initialize_pair_normalizers(wm, fit, layout.normalize)
    initial_hash = state_hash({"wm": wm})
    run = directory / "run"
    run.mkdir()
    torch_save(run / "initial.pt", dict(wm=wm.state_dict(), args=args, diagnostic_only=True))
    generator = torch.Generator().manual_seed(job["seed"] + 7301)
    schedule = torch.randint(fit.count, (settings["updates"], settings["batch_size"]), generator=generator)
    np.save(run / "sampling.npy", schedule.numpy())
    schedule = schedule.cuda()
    opt = MuonAdam(wm, [wm.net], lr=settings["lr"], betas=(.9, .999), fused=True, capturable=True, flat_adam=True)
    weight = args["response_weight"] if job["group"] == "C" else 0.
    curves = []
    update = 0
    with (run / "train_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("update", "forward_loss", "response_loss", "weighted_response_loss", "grad_norm"))
        writer.writeheader()
        for update, ids in enumerate(schedule, 1):
            batch = {k: v[ids] for k, v in fit.data.items()}
            forward, response, errors = paired_losses(wm, batch, layout.normalize, fit.q_scale, fit.v_scale,
                **{k: args[k] for k in ("response_horizon", "response_loss", "response_huber_delta")})
            loss = forward + weight * response
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training objective")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            writer.writerow(dict(update=update, forward_loss=float(forward.detach()), response_loss=float(response.detach()),
                                 weighted_response_loss=float(response.detach())*weight, grad_norm=float(norm)))
            f.flush()
            if update % settings["checkpoint_every"] == 0 or update == settings["updates"] or STOP:
                torch_save(run / "last.pt", dict(wm=wm.state_dict(), args=args, updates=update,
                                                 opt_wm=opt.state_dict(), diagnostic_only=True))
                torch_save(run / f"ckpt_{update:06d}.pt", dict(wm=wm.state_dict(), args=args,
                                                             updates=update, diagnostic_only=True))
                if not STOP:
                    curves.append(dict(update=update, fit=evaluate(wm, fit, layout), cal=evaluate(wm, cal, layout)))
                    atomic_json(run / "learning_curve.json", curves)
                print(f"{job['id']}: WM update {update}/{settings['updates']}", flush=True)
            if STOP:
                atomic_json(directory / "analysis.json", dict(complete=False, updates=update, reason="interrupted"))
                return 2
    final = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    finite_weights(final, ("wm",))
    result = dict(kind="world_model", group=job["group"], candidate=job["candidate"], seed=job["seed"],
                  complete=True, diagnostic_only=True, policy_updates=0,
                  budget=dict(wm_updates=update, batch_size=settings["batch_size"]),
                  init_sha256=initial_hash, comparison_sha256=comparison_key(provenance, "world_model", settings),
                  sampling_sha256=digest(run / "sampling.npy"), checkpoint_sha256=digest(run / "last.pt"),
                  fit=curves[-1]["fit"], cal=curves[-1]["cal"], formal_results=False, test_outcomes_inspected=False)
    atomic_json(directory / "analysis.json", result)
    return 0


def train_policy(session, directory, job, plan, provenance):
    training = dict(read(session / "base_config.json")["training"], **plan["policy"])
    training.update({k: v for k, v in job["response"].items() if k != "name"})
    # Time is a truncation guard, never the matched experimental budget.
    training.update(wm_amp="none", max_hours=plan["job_timeout_minutes"] / 60 * .85)
    config = directory / "config.json"
    protocol = directory / "protocol.json"
    atomic_json(config, dict(schema_version=1, formal_results=False, training=training))
    frozen = dict(schema_version=1, implementation="PartyMimic Z01 local fork", groups=[job["group"]],
                  training_seeds=[job["seed"]], training=training, window_candidates=dict(short=4, long=8),
                  config_sha256=digest(config), motions=provenance["motions"], **provenance["shared"],
                  formal_results=False, purpose="Fresh development attempt; no test evaluation authorized")
    atomic_json(protocol, frozen)
    p = provenance["paths"]
    args = SimpleNamespace(config=str(config), params=p["params"], motion_dir=p["motions"],
                           group=job["group"], seed=job["seed"], out=str(directory / "run"),
                           responses=p["responses"] if job["group"] not in ("A0", "A") else None,
                           protocol=str(protocol))
    command = training_command(args)
    atomic_json(directory / "command.json", command)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode or STOP:
        return result.returncode or 2
    run = directory / "run"
    ledger = read(run / "run.json")
    if ledger["status"] != "completed":
        atomic_json(directory / "analysis.json", dict(complete=False, reason=ledger["status"]))
        return 2
    for key, value in provenance["shared"].items():
        expected = None if key == "response_sha256" and job["group"] in ("A0", "A") else value
        if ledger[key] != expected:
            raise ValueError(f"Run provenance mismatch: {key}")
    if ledger["protocol_sha256"] != digest(protocol):
        raise ValueError("Run protocol mismatch")
    entries = list(csv.DictReader((run / "checkpoints.csv").open()))
    if not {"initial.pt", "last.pt"}.issubset({row["path"] for row in entries}):
        raise ValueError("Incomplete checkpoint index")
    for row in entries:
        path = Path(row["path"])
        if path.name != str(path) or digest(run / path) != row["sha256"]:
            raise ValueError("Checkpoint index/hash mismatch")
    initial = torch.load(run / "initial.pt", weights_only=False, map_location="cpu")
    if state_digest(initial, ("wm", "pi", "critic")) != ledger["init_sha256"]:
        raise ValueError("Initial state hash mismatch")
    ck = torch.load(run / "last.pt", weights_only=False, map_location="cpu")
    finite_weights(ck, ("wm", "pi", "pi_live", "critic"))
    for key in ("wm", "pi"):
        if not any("weight" in k and not torch.equal(v, ck[key][k]) for k, v in initial[key].items()):
            raise ValueError(f"{key} did not update")
    del initial
    if ck["iteration"] != training["iterations"] or ck["args"] != ledger["args"] or ck["sim_steps"] != ledger["env_steps"]:
        raise ValueError("Incomplete/different training budget")
    metrics = list(csv.DictReader((run / "train_metrics.csv").open()))
    if [int(m["iteration"]) for m in metrics] != list(range(training["iterations"]+1)):
        raise ValueError("Missing training iterations")
    for m in metrics[1:]:
        if not all(math.isfinite(float(v)) for v in m.values() if v):
            raise FloatingPointError("Nonfinite training metrics")
        if job["group"] not in ("A0", "A") and not math.isclose(float(m["forward_loss"])+float(m["weighted_response_loss"]), float(m["wm_loss"]), rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Loss decomposition mismatch")
    train_log = list(csv.DictReader((run / "train_log.csv").open()))
    if any(float(row.get("nonfinite", 0)) > 0 for row in train_log):
        raise FloatingPointError("Trainer recorded nonfinite updates")
    common = {k: v for k, v in training.items() if not k.startswith("response_")}
    analysis = dict(kind="policy", group=job["group"], candidate=job["candidate"], seed=job["seed"],
                    complete=True, diagnostic_only=False, closed_loop_evaluation_performed=False,
                    budget=dict(iterations=ck["iteration"], wm_updates=ledger["wm_updates"],
                                policy_updates=ledger["policy_updates"], env_steps=ledger["env_steps"], wm_batch=training["wm_batch"]),
                    init_sha256=ledger["init_sha256"], comparison_sha256=comparison_key(provenance, "policy", common),
                    sampling_sha256=None, checkpoint_sha256=digest(run / "last.pt"),
                    checkpoint_hashes_verified=len(entries), formal_results=False, test_outcomes_inspected=False,
                    final_training_log=train_log[-1] if train_log else None, fit=None, cal=None)
    del ck
    if job["group"] not in ("A0", "A"):
        ck, robot, layout, wm = load_context(run / "last.pt", p["params"], device="cuda:0")
        del ck
        fit, cal = datasets(robot, layout, provenance)
        analysis.update(fit=evaluate(wm, fit, layout), cal=evaluate(wm, cal, layout))
    atomic_json(directory / "analysis.json", analysis)
    return 0


def amplitude_probe(directory, job, provenance):
    """Stratified fit-only CPU replay; records sensitivity without inventing a tolerance."""
    import mujoco
    from partymimic.dr import XiLayout
    from partymimic.kinematics import Kinematics
    from partymimic.motion import MotionLib
    from partymimic.rotations import mat_to_quat
    p = provenance["paths"]
    meta = read(p["responses"])
    path = Path(p["responses"]).parent / meta["splits"]["fit"]["file"]
    if digest(path) != meta["splits"]["fit"]["sha256"]:
        raise ValueError("Fit response archive changed")
    with np.load(path, allow_pickle=False) as z:
        data_arrays = {k: z[k] for k in z.files}
    z = data_arrays
    robot = common_robot(p["params"])
    layout, kin = XiLayout(robot, "cpu"), Kinematics(robot, "cpu")
    J = robot.num_joints
    scales = np.r_[np.full(J, meta["output_scales"]["q_rad"]), np.full(J, meta["output_scales"]["qd_rad_s"])]
    ref = (z["s_plus"][:,:,13:]-z["s_minus"][:,:,13:])/(2*z["epsilon"][:,None,None])/scales
    order = np.argsort(np.square(ref[:,:8]).mean((1,2)))
    ids = order[np.unique(np.linspace(0, len(order)-1, min(job["blocks"], len(order))).astype(int))]
    spec = robot.robot_cfg().build().spec.copy()
    spec.worldbody.add_geom(name="pair_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0,0,.05], friction=[.3,.005,.0001])
    model = spec.compile()
    model.opt.timestep = .005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations, model.opt.ls_iterations, model.opt.tolerance = 100, 50, 1e-6
    joints = [model.joint(n).id for n in robot.joint_names]
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    aids = [model.actuator(n).id for n in robot.joint_names]
    d = mujoco.MjData(model)
    q0, scale = np.asarray(robot.default_joint_pos_list()), np.asarray(robot.action_scale_list())
    com = model.body(robot.com_body).id
    default_com = model.body_ipos[com].copy()
    rows = []
    for index in ids:
        if STOP:
            return 2
        clip, _, start = str(z["block_id"][index]).rsplit(":", 2)
        start = int(start)
        lib = MotionLib([str(Path(p["motions"])/(clip+".npz"))], kin, "cpu", robot.load_motion)
        xi = z["xi"][index]
        model.geom_friction[model.geom_bodyid != 0,0] = xi[layout.slices["friction"]][0]
        model.body_ipos[com] = default_com + xi[layout.slices["com"]]
        mujoco.mj_setConst(model,d)
        mujoco.mj_resetData(model,d)
        pos,R,v,w,q,dq = (a[0].numpy() for a in lib.ref_state(torch.tensor([start])))
        d.qpos[:3],d.qpos[3:7],d.qpos[qa] = pos,mat_to_quat(torch.tensor(R)).numpy(),q
        d.qvel[:3],d.qvel[3:6],d.qvel[va] = v,R.T@w,dq
        def step(a):
            d.ctrl[aids] = q0+scale*a+xi[layout.slices["joint_offset"]]
            for _ in range(4):
                mujoco.mj_step(model,d)
            return np.r_[d.qpos[qa],d.qvel[va]]
        for t in range(20):
            step((lib.q[start+t+1].numpy()-q0)/scale)
        initial_error = float(np.max(np.abs(np.r_[d.qpos[qa],d.qvel[va]]-z["s0"][index,13:])))
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        saved = np.empty(mujoco.mj_stateSize(model,state_type))
        mujoco.mj_getState(model,d,saved,state_type)
        def rollout(actions):
            mujoco.mj_setState(model,d,saved,state_type)
            mujoco.mj_forward(model,d)
            return np.stack([step(a) for a in actions])
        base = (z["a_plus"][index]+z["a_minus"][index])/2
        delta = (z["a_plus"][index]-z["a_minus"][index])/2
        row = dict(block_id=str(z["block_id"][index]), initial_state_max_error=initial_error,
                   repeat_max_error=float(np.abs(rollout(base)-rollout(base)).max()), amplitudes=[])
        for factor in (.25,.5,1.,2.):
            plus,minus = rollout(base+factor*delta),rollout(base-factor*delta)
            response = (plus-minus)/(2*z["epsilon"][index]*factor)/scales
            item = dict(factor=factor, epsilon_rad=float(z["epsilon"][index]*factor), prefixes={})
            for h in (4,8,32):
                x,y = response[:h].ravel(),ref[index,:h].ravel()
                nx,ny = float(np.linalg.norm(x)),float(np.linalg.norm(y))
                item["prefixes"][str(h)] = dict(relative_difference=float(np.linalg.norm(x-y)/ny) if ny>1e-12 else None,
                                               cosine=float(np.dot(x,y)/(nx*ny)) if nx*ny>1e-12 else None)
            if factor == 1.:
                item["saved_pair_max_error"] = float(max(np.abs(plus-z["s_plus"][index,:,13:]).max(),np.abs(minus-z["s_minus"][index,:,13:]).max()))
            row["amplitudes"].append(item)
        rows.append(row)
        atomic_json(directory / "amplitudes.json", dict(blocks=rows, selection="Even quantiles of fit 8-step response energy", formal_results=False,
                                                        test_outcomes_inspected=False, tolerance_selected=False, backend_validation=False))
    replay_ok = all(row["initial_state_max_error"] <= 1e-8 and row["repeat_max_error"] <= 1e-10
                    and all(item.get("saved_pair_max_error", 0.) <= 1e-8 for item in row["amplitudes"]) for row in rows)
    atomic_json(directory / "analysis.json", dict(complete=replay_ok, kind="amplitude", blocks=len(rows), replay_integrity_passed=replay_ok,
                                                   report_sha256=digest(directory / "amplitudes.json"),
                                                   formal_results=False, test_outcomes_inspected=False,
                                                   interpretation="CPU replay and amplitude sensitivity only; CPU/GPU equivalence remains pending"))
    return 0 if replay_ok else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--session", required=True)
    prep.add_argument("--machine", required=True)
    prep.add_argument("--params", required=True)
    prep.add_argument("--motions", required=True)
    prep.add_argument("--responses", required=True)
    prep.add_argument("--allow-gpu-mismatch", action="store_true")
    job_parser = sub.add_parser("job")
    job_parser.add_argument("--session", required=True)
    job_parser.add_argument("--job", required=True)
    a = p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.command == "prepare":
        prepare(a)
        return 0
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_requested)
    session = Path(a.session)
    provenance = read(session / "provenance.json")
    verify_inputs(session, provenance)
    directory = session / "jobs" / a.job
    job, plan = read(directory / "job.json"), read(session / "plan.json")
    if job["kind"] == "world_model":
        return train_world_model(session, directory, job, plan, provenance)
    if job["kind"] == "policy":
        return train_policy(session, directory, job, plan, provenance)
    if job["kind"] == "amplitude":
        return amplitude_probe(directory, job, provenance)
    raise ValueError("Unknown job type")


if __name__ == "__main__":
    raise SystemExit(main())
