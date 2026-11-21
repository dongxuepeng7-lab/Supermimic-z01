"""Frozen-model actor window ablation; equal starts and exact per-update horizon histograms."""

from pathlib import Path
import json

import numpy as np
import torch

from partymimic.mdp import unflatten_state
from .parameters import sha256
from .runtime import object_hash, state_hash


def variable_objective(rewards, dones, values, horizons, gamma):
    """Differentiable terminal bootstrap at each sample's selected H (never at Hmax)."""
    H, B = rewards.shape
    alive = torch.ones(B, device=rewards.device)
    total = torch.zeros_like(alive)
    for t in range(H):
        active = (t < horizons).to(rewards.dtype)
        total = total + gamma**t * alive * active * rewards[t]
        alive = alive * torch.where(
            t < horizons, 1 - dones[t], torch.ones_like(dones[t])
        )
        total = total + gamma ** (t + 1) * alive * (horizons == t + 1) * values[t]
    return total


def speed_bins(state, edges):
    # Bin definitions are fixed before calibration and contain no test information.
    return torch.bucketize(unflatten_state(state)[5].square().mean(-1).sqrt(), edges)


def calibrate_windows(
    wm,
    dataset,
    normalize,
    checkpoint,
    short,
    long,
    edges,
    tolerance,
    quantile=0.9,
    min_blocks=10,
    signal_floor=0.01,
):
    from .response import paired_losses

    if (
        not 1 <= short <= long <= dataset.horizon
        or not 0 < quantile <= 1
        or min_blocks < 1
        or tolerance <= 0
    ):
        raise ValueError(
            "Invalid calibration horizon, quantile, tolerance or minimum block count"
        )
    if sorted(set(edges)) != edges or any(x <= 0 for x in edges):
        raise ValueError(
            "Speed bin edges must be strictly increasing positive rad/s values"
        )
    device = dataset.data["s0"].device
    bins = speed_bins(dataset.data["s0"], torch.tensor(edges, device=device))
    errors = []
    with torch.no_grad():
        for i in range(0, dataset.count, 128):
            batch = {k: v[i : i + 128] for k, v in dataset.data.items()}
            errors.append(
                paired_losses(wm, batch, normalize, dataset.q_scale, dataset.v_scale)[2]
            )
    errors = torch.cat(errors).cpu().numpy()
    target_plus, target_minus = (
        unflatten_state(dataset.data["s_plus"]),
        unflatten_state(dataset.data["s_minus"]),
    )
    eps = 2 * dataset.data["epsilon"][:, None, None]
    signal = torch.cat(
        (
            (target_plus[4] - target_minus[4]) / eps / dataset.q_scale,
            (target_plus[5] - target_minus[5]) / eps / dataset.v_scale,
        ),
        -1,
    )
    signal = signal.square().mean(-1).sqrt().amax(-1).cpu().numpy()
    rows = []
    for b in range(len(edges) + 1):
        mask = (bins.cpu().numpy() == b) & (signal >= signal_floor)
        n = int(mask.sum())
        # Aggregate within sessions first; many windows from one record are not independent evidence.
        session_errors = []
        for session in sorted(set(dataset.arrays["session_id"].astype(str)[mask])):
            sm = mask & (dataset.arrays["session_id"].astype(str) == session)
            session_errors.append(np.quantile(errors[sm], quantile, axis=0))
        q = (
            np.quantile(session_errors, quantile, axis=0)
            if session_errors
            else np.full(dataset.horizon, np.inf)
        )
        enough = n >= min_blocks and len(session_errors) >= 2
        valid_short = enough and bool(np.all(q[:short] <= tolerance))
        valid_long = enough and bool(np.all(q[:long] <= tolerance))
        rows.append(
            dict(
                bin=b,
                eligible_blocks=n,
                independent_sessions=len(session_errors),
                horizon=long if valid_long else short,
                short_validated=valid_short,
                prefix_error=[float(x) if np.isfinite(x) else None for x in q],
            )
        )
    return dict(
        schema_version=1,
        checkpoint_sha256=sha256(checkpoint),
        response_sha256=sha256(dataset.path),
        split="cal",
        speed_edges_rad_s=edges,
        short=short,
        long=long,
        tolerance=tolerance,
        quantile=quantile,
        min_blocks=min_blocks,
        signal_floor=signal_floor,
        bins=rows,
    )


def train_windows(
    checkpoint,
    calibration,
    starts_path,
    output,
    group,
    robot,
    motion,
    mdp,
    layout,
    iterations=100,
    batch_size=64,
    seed=1,
    lr=1e-4,
    device="cuda:0",
):
    """D/C-perm update only the actor using the same frozen C model/critic and start pool."""
    from partymimic.networks import Policy, Critic, policy_kwargs
    import partymimic.networks as networks
    from partymimic.mdp import inv_dim
    from .models import make_world_model

    if group not in ("D", "C-perm") or min(iterations, batch_size) < 1:
        raise ValueError("Window ablation requires D/C-perm and positive budget")
    cal = json.loads(Path(calibration).read_text())
    if cal.get("checkpoint_sha256") != sha256(checkpoint) or cal.get("split") != "cal":
        raise ValueError(
            "Calibration must come from this exact frozen C checkpoint and the cal split"
        )
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    args = ck["args"]
    if args.get("experiment_group") != "C" or "critic" not in ck:
        raise ValueError("A complete C checkpoint including critic is required")
    response_hash = args.get("response_manifest_sha256") or sha256(
        args["response_manifest"]
    )
    if cal["response_sha256"] != response_hash:
        raise ValueError("Calibration response manifest differs from C training")
    manifest_path = Path(starts_path)
    meta = json.loads(manifest_path.read_text())
    path = manifest_path.parent / meta["file"]
    if (
        meta.get("checkpoint_sha256") != sha256(checkpoint)
        or meta.get("split") != "fit"
    ):
        raise ValueError("Actor starts must be captured from C on fit motions")
    if sha256(path) != meta["sha256"]:
        raise ValueError("Actor-start archive changed")
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    required = ("s", "a_prev", "xi", "f", "obs_hist")
    if any(k not in raw for k in required) or len({len(raw[k]) for k in required}) != 1:
        raise ValueError("Malformed actor start pool")
    if any(not np.isfinite(raw[k]).all() for k in required):
        raise ValueError("Nonfinite actor start pool")
    pool = {
        k: torch.as_tensor(
            raw[k], device=device, dtype=torch.long if k == "f" else torch.float32
        )
        for k in required
    }
    if (
        pool["s"].shape[-1] != 13 + 2 * robot.num_joints
        or pool["xi"].shape[-1] != layout.dim
    ):
        raise ValueError("Actor start dimensions mismatch")
    H, K = cal["long"], args["goal_frames"]
    frames = pool["f"]
    if not len(frames) or bool(((frames < 0) | (frames >= motion.num_frames)).any()):
        raise ValueError("Invalid actor-start reference frames")
    if bool((frames + H >= motion.clip_end[motion.clip_id[frames]]).any()):
        raise ValueError("Actor starts cross a reference clip boundary")
    if meta.get("motion_names") != motion.names:
        raise ValueError("Actor-start motion order differs")
    if meta.get("motion_hashes") != getattr(motion, "input_hashes", None):
        raise ValueError("Actor-start motion content differs")
    widths = lambda key: tuple(int(x) for x in args[key].split(","))
    networks.AMP_DTYPE = torch.bfloat16 if args.get("amp") == "bf16" else None
    wm = make_world_model(robot, layout, args).to(device)
    wm.load_state_dict(ck["wm"])
    pi = Policy(
        mdp.obs_dim,
        motion.goal_dim(K),
        robot.num_joints,
        hist=args["hist"],
        enc_hidden=widths("enc_hidden"),
        dec_hidden=widths("dec_hidden"),
        **policy_kwargs(args, robot),
    ).to(device)
    pi.load_state_dict(ck.get("pi_live", ck["pi"]))
    critic = Critic(
        inv_dim(robot.num_joints) + mdp.critic_extra + motion.goal_dim(K)
    ).to(device)
    critic.load_state_dict(ck["critic"])
    for m in (wm, critic):
        m.eval()
        m.requires_grad_(False)
    frozen_hash = state_hash({"wm": wm, "critic": critic})
    initial_hash = state_hash({"pi": pi, "wm": wm, "critic": critic})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    args = dict(args, experiment_group=group)
    (output / "args.json").write_text(json.dumps(args, indent=2))
    run = dict(
        group=group,
        status="running",
        checkpoint_sha256=sha256(checkpoint),
        calibration_sha256=sha256(calibration),
        starts_sha256=sha256(starts_path),
        init_sha256=initial_hash,
        training_seed=seed,
        iterations=iterations,
        batch_size=batch_size,
        lr=lr,
        frozen_world_model=True,
        frozen_critic=True,
        formal_results=False,
        fallback_bins=[r["bin"] for r in cal["bins"] if not r["short_validated"]],
    )
    from .protocol import code_hash, asset_hash

    run.update(
        code_sha256=code_hash(),
        asset_sha256=asset_hash(),
        params_sha256=args.get("parameter_manifest_sha256"),
        response_sha256=response_hash,
        motion_sha256=object_hash(motion.input_hashes),
        inherited_env_steps=ck["sim_steps"],
        new_env_steps=0,
        implementation="PartyMimic Z01 frozen-C actor window ablation",
    )
    (output / "run.json").write_text(json.dumps(run, indent=2))
    optimizer = torch.optim.Adam(pi.parameters(), lr=lr)
    sample_rng = torch.Generator(device=device).manual_seed(seed)
    perm_rng = torch.Generator(device=device).manual_seed(seed + 9001)
    edges = torch.tensor(cal["speed_edges_rad_s"], device=device)
    lookup = torch.tensor([r["horizon"] for r in cal["bins"]], device=device)
    import csv

    rows, sampling = [], []
    try:
        for it in range(iterations):
            ids = torch.randint(
                len(frames), (batch_size,), generator=sample_rng, device=device
            )
            data = {k: v[ids] for k, v in pool.items()}
            horizons = lookup[speed_bins(data["s"], edges)]
            if group == "C-perm":
                horizons = horizons[
                    torch.randperm(batch_size, generator=perm_rng, device=device)
                ]
            S, previous, f, hist = (
                unflatten_state(data["s"]),
                data["a_prev"],
                data["f"],
                data["obs_hist"],
            )
            xi, xi_n = data["xi"], layout.normalize(data["xi"])
            rewards, dones, values = [], [], []
            for t in range(H):
                action = pi(hist, motion.goal(f, K))
                active = t < horizons
                prediction = wm.step(S, action, previous, xi_n)
                S = tuple(
                    torch.where(active.reshape(-1, *([1] * (x.ndim - 1))), y, x)
                    for x, y in zip(S, prediction)
                )
                f = f + active.long()
                reward = mdp.reward(
                    S,
                    f,
                    action,
                    previous,
                    w_contact=args["contact_reward"],
                    w_joint_velocity=args["joint_velocity_penalty"],
                    velocity_margin=args["joint_velocity_margin"],
                )
                rewards.append(reward)
                dones.append(mdp.soft_termination(S, f))
                values.append(
                    critic(
                        torch.cat(
                            (mdp.critic_features(S, f, xi_n), motion.goal(f, K)), -1
                        )
                    )
                )
                hist = torch.cat(
                    (hist[:, 1:], mdp.obs(S, action, xi, f).unsqueeze(1)), 1
                )
                previous = action
            loss = -variable_objective(
                torch.stack(rewards),
                torch.stack(dones),
                torch.stack(values),
                horizons,
                args["gamma"],
            ).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite frozen-window actor objective")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                pi.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            rows.append(
                dict(
                    iteration=it + 1,
                    loss=float(loss.detach()),
                    assigned_steps=int(horizons.sum()),
                    computed_steps=H * batch_size,
                    short_count=int((horizons == cal["short"]).sum()),
                )
            )
            sampling.append(ids.cpu().numpy())
        if state_hash({"wm": wm, "critic": critic}) != frozen_hash:
            raise RuntimeError("Frozen model/critic changed")
        torch.save(
            dict(
                pi=pi.state_dict(),
                wm=wm.state_dict(),
                critic=critic.state_dict(),
                args=args,
                iteration=iterations,
                sim_steps=ck["sim_steps"],
            ),
            output / "last.pt",
        )
        with (output / "checkpoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=("path", "sha256", "env_steps", "policy_updates")
            )
            w.writeheader()
            w.writerow(
                dict(
                    path="last.pt",
                    sha256=sha256(output / "last.pt"),
                    env_steps=ck["sim_steps"],
                    policy_updates=iterations,
                )
            )
        run.update(
            status="completed",
            start_schedule_sha256=object_hash(np.asarray(sampling).tolist()),
            assigned_imagined_steps=sum(r["assigned_steps"] for r in rows),
            computed_imagined_steps=sum(r["computed_steps"] for r in rows),
        )
    except BaseException:
        run["status"] = "interrupted_or_failed"
        raise
    finally:
        (output / "run.json").write_text(json.dumps(run, indent=2))
        with (output / "train_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=(
                    "iteration",
                    "loss",
                    "assigned_steps",
                    "computed_steps",
                    "short_count",
                ),
            )
            w.writeheader()
            w.writerows(rows)
        np.save(output / "start_schedule.npy", np.asarray(sampling))
