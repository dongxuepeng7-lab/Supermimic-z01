"""GPU-native finite-amplitude response labels for matched online B/C continuation.

The full training scene, domain randomization, world count and control graph are
used. Both branches restore the same integration state in the same world slots.
The locked CPU test archive is never opened or reused as a GPU ground truth.
"""
from pathlib import Path
import importlib.metadata
import json
import time

import numpy as np
import torch

from .parameters import sha256


class NativeStepper:
    """Capture the same four write/step/update operations used by TrackingSim.

    No reset, reward, policy or random draws occur inside a paired response.
    Training's outer collection graph additionally performs those operations.
    """
    def __init__(self, env):
        import warp as wp
        from mjlab.sim.sim_data import TorchArray
        from partymimic.env import DECIMATION, PHYSICS_DT
        from partymimic.mdp import flatten_state

        self.env = env
        self.action = torch.zeros_like(env.a_prev)

        def core():
            env.robot.set_joint_position_target(env.default_q + env.xi[:, 1:1 + env.J] + env.alpha * self.action)
            for _ in range(DECIMATION):
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(PHYSICS_DT)
            return flatten_state(env.state())

        # Compilation warmup is counted separately and discarded by a subsequent reset.
        for _ in range(3):
            core()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        setitem, use_graph = TorchArray.__setitem__, env.sim.use_cuda_graph
        TorchArray.__setitem__ = lambda obj, idx, value: obj._tensor.__setitem__(idx, value)
        env.sim.use_cuda_graph = False
        try:
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                ws = wp.stream_from_torch(torch.cuda.current_stream())
                with wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
                    wp.capture_begin(stream=ws, external=True)
                    try:
                        self.out = core()
                    finally:
                        self.wp_graph = wp.capture_end(stream=ws)
        finally:
            TorchArray.__setitem__, env.sim.use_cuda_graph = setitem, use_graph

    def __call__(self, action):
        self.action.copy_(action)
        self.graph.replay()
        return self.out.clone()


def integration_snapshot(env):
    import warp as wp
    # MuJoCo integration inputs, including solver warm start and delay history.
    # Extra acceleration/sleep inputs are retained; derived contact/FK buffers are
    # recomputed by the next physics step. Z01 has no external stateful actuators.
    names = ("time", "qpos", "qvel", "act", "history", "qacc_warmstart", "ctrl",
             "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
             "userdata", "qacc", "act_dot", "tree_asleep")
    values = {}
    for name in names:
        raw = getattr(env.sim.wp_data, name, None)
        if raw is not None and raw.size:
            tensor = wp.to_torch(raw)
            values[name] = (tensor, tensor.clone())
    if any(name not in values for name in ("qpos", "qvel", "qacc_warmstart", "ctrl")):
        raise ValueError("GPU runtime lacks required integration-state fields")
    return values


def restore(snapshot):
    for tensor, saved in snapshot.values():
        tensor.copy_(saved)


@torch.no_grad()
def build_native_pairs(checkpoint, params, motions, split_manifest, output, *,
                       horizon=32, blocks_per_clip=8, epsilon=.01, pre_roll=20, seed=73):
    from partymimic.env import TrackingSim, make_sim
    from partymimic.mdp import flatten_state
    from partymimic.networks import Policy, policy_kwargs
    import partymimic.networks as networks
    import partymimic.rotations as rotations
    from .cli import load_context
    from .protocol import code_hash, asset_hash

    if min(horizon, blocks_per_clip, pre_roll) < 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive finite GPU-pair dimensions/amplitude required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"GPU pairs already exist; never overwrite: {output}")
    split = json.loads(Path(split_manifest).read_text())
    groups = {s: set(split["splits"][s]["session_ids"]) for s in ("fit", "cal", "test")}
    if any(groups[x] & groups[y] for x, y in (("fit", "cal"), ("fit", "test"), ("cal", "test"))):
        raise ValueError("Source session split leaks")
    torch.manual_seed(seed)
    rotations.ELEMENTWISE = True
    networks.RunningNorm.no_sync = True
    ck, robot, layout, wm, motion, mdp = load_context(checkpoint, params, motions, "cuda:0")
    a = ck["args"]
    if a.get("experiment_group") != "B" or a.get("wm_fit_source") != "online" or a.get("wm_amp") != "none":
        raise ValueError("Native pair teacher must be the frozen online FP32 B parent")
    if split["params_sha256"] != sha256(params) or split["motion_hashes"] != motion.input_hashes:
        raise ValueError("Parent motions/parameters differ from original session split")
    N, J = a["num_envs"], robot.num_joints
    widths = lambda key: tuple(map(int, a[key].split(",")))
    pi = Policy(mdp.obs_dim, motion.goal_dim(a["goal_frames"]), J, hist=a["hist"],
                enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"),
                **policy_kwargs(a, robot)).to("cuda:0")
    pi.load_state_dict(ck["pi_live"])
    pi.eval()
    del wm, ck
    env = TrackingSim(make_sim(robot), N, robot, motion, mdp, "cuda:0", gamma=a["gamma"],
                      randomize=True, reset_noise=True, fail_reset=a["fail_reset"],
                      div_k=a["div_k"], div_tau=a["div_tau"])
    if env.pipe.stateful or env.push_enabled or robot.has_action_delay:
        raise ValueError("Native pair v1 supports stateless sensing/actuators without pushes only")
    step = NativeStepper(env)
    rng = np.random.default_rng(seed)
    jobs = []
    for i, name in enumerate(motion.names):
        session = name.split("__seg")[0]
        part = next((s for s in ("fit", "cal") if session in groups[s]), None)
        if part is None:
            continue
        lo, hi = int(motion.clip_start[i]), int(motion.clip_end[i]) - pre_roll - horizon - 1
        if hi <= lo:
            raise ValueError(f"Source clip too short for frozen protocol: {name}")
        for b in range(blocks_per_clip):
            jobs.append(dict(split=part, session_id=session, block_id=f"{name}:{b}",
                             frame=int(rng.integers(lo, hi))))
    if not jobs or any(len({j["session_id"] for j in jobs if j["split"] == s}) < 2 for s in ("fit", "cal")):
        raise ValueError("Need at least two source sessions per fit/cal split")
    output.mkdir(parents=True)
    start_time = time.time()
    meta = dict(schema_version=1, source_type="physics_paired",
                backend="TrackingSim/mujoco_warp/native_pairs_v1", status="generating",
                teacher_checkpoint_sha256=sha256(checkpoint), teacher_weights="pi_live",
                source_split_manifest_sha256=sha256(split_manifest), joint_names=list(robot.joint_names),
                params_sha256=sha256(params), motion_hashes=motion.input_hashes,
                control_dt=.02, physics_dt=.005, output_scales=dict(q_rad=.01, qd_rad_s=.1),
                seed=seed, horizon=horizon, pre_roll=pre_roll, blocks_per_clip=blocks_per_clip,
                epsilon_joint_target_rad=epsilon, response_test_accessed=False, formal_results=False,
                runtime=dict(num_envs=N, control_dt=.02, physics_dt=.005,
                             gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                             code_sha256=code_hash(), asset_sha256=asset_hash(),
                             **{p: importlib.metadata.version(p) for p in ("mujoco", "mujoco-warp", "mjlab", "warp-lang")}),
                execution="same training physics operations in outer CUDA graph; fixed world slots; no auto-reset within pairs",
                split_scope="held-out response sessions, NOT unseen policy-training motions", splits={})
    (output / "generation.json").write_text(json.dumps(meta, indent=2))
    buckets = {s: [] for s in ("fit", "cal")}
    ids = torch.arange(N, device="cuda:0")
    scales = torch.tensor([.01] * J + [.1] * J, device="cuda:0")
    batch_records = []
    for offset in range(0, len(jobs), N):
        selected = jobs[offset:offset + N]
        # Padding retains training world packing; padded rows never enter statistics.
        padded = selected + [selected[0]] * (N - len(selected))
        frames = torch.tensor([j["frame"] for j in padded], device="cuda:0")
        env.reset(ids, frames)
        hist = None

        def teacher_step():
            nonlocal hist
            env._obs = None
            observation = env.observe()
            hist = (observation[:, None].repeat(1, a["hist"], 1) if hist is None else
                    torch.cat((hist[:, 1:], observation[:, None]), 1))
            action = pi(hist, motion.goal(env.f, a["goal_frames"])).clamp(-20, 20)
            state = step(action)
            env.a_prev.copy_(action)
            env.f.add_(1)
            return action.clone(), state

        for _ in range(pre_roll):
            teacher_step()
        s0, previous, xi = flatten_state(env.state()).clone(), env.a_prev.clone(), env.xi.clone()
        snap = integration_snapshot(env)
        source_frame = env.f.clone()
        terminated_at_start = mdp.termination(env.state(), source_frame)
        actions, base = zip(*(teacher_step() for _ in range(horizon)))
        actions, base = torch.stack(actions, 1), torch.stack(base, 1)
        direction = rng.normal(size=(N, horizon, J))
        direction /= np.linalg.norm(direction.reshape(N, -1), axis=1)[:, None, None]
        delta = torch.tensor(direction * epsilon, dtype=torch.float32, device="cuda:0") / env.alpha
        sides = {}
        for name, command in (("base_repeat", actions), ("plus", actions + delta), ("minus", actions - delta)):
            restore(snap)
            sides[name] = torch.stack([step(command[:, t]) for t in range(horizon)], 1)
        signal = (sides["plus"][:, :, 13:] - sides["minus"][:, :, 13:]) / (2 * epsilon) / scales
        noise = (base[:, :, 13:] - sides["base_repeat"][:, :, 13:]) / (2 * epsilon) / scales
        def prefix(x):
            return (x.square().mean(-1).cumsum(-1) / torch.arange(1, horizon + 1, device=x.device)).sqrt()
        arrays = dict(s0=s0, a_prev=previous, xi=xi, a_plus=actions + delta, a_minus=actions - delta,
                      s_plus=sides["plus"], s_minus=sides["minus"], signal_prefix=prefix(signal),
                      noise_prefix=prefix(noise), terminated_at_start=terminated_at_start,
                      source_frame=source_frame)
        for key, tensor in arrays.items():
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"Nonfinite GPU labels: {key}; generation retained but not admitted")
        cpu = {key: tensor.cpu().numpy() for key, tensor in arrays.items()}
        for i, job in enumerate(selected):
            buckets[job["split"]].append(dict(
                **{k: v[i] for k, v in cpu.items()}, epsilon=np.float32(epsilon),
                session_id=job["session_id"], block_id=job["block_id"]))
        # Keep exact integration inputs for every world in the batch, including padding.
        fname = f"integration_{offset // N:03d}.npz"
        np.savez_compressed(output / fname, **{k: v.cpu().numpy() for k, (_, v) in snap.items()},
                            xi=xi.cpu().numpy(), source_frame=source_frame.cpu().numpy())
        batch_records.append(dict(file=fname, sha256=sha256(output / fname), valid_rows=len(selected),
                                  block_ids=[j["block_id"] for j in selected]))
        print(f"GPU response pairs: {min(offset + N, len(jobs))}/{len(jobs)}", flush=True)
    for part, rows in buckets.items():
        path = output / f"{part}.npz"
        np.savez_compressed(path, **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})
        meta["splits"][part] = dict(file=path.name, sha256=sha256(path), blocks=len(rows),
                                    session_ids=sorted({r["session_id"] for r in rows}))
    meta["splits"]["test"] = dict(session_ids=sorted(groups["test"]), status="locked_not_generated")
    meta.update(status="completed", integration_batches=batch_records,
                generation_env_transitions=3 * N + len(batch_records) * N * (pre_roll + 4 * horizon),
                generation_wall_seconds=time.time() - start_time,
                interpretation="GPU simulation labels; base replay estimates numerical repeat noise. No hardware truth or backend equivalence claim.")
    (output / "manifest.json").write_text(json.dumps(meta, indent=2))
    (output / "generation.json").write_text(json.dumps(meta, indent=2))
    return meta
