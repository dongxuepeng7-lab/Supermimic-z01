"""Capture an immutable shared actor-start pool from the frozen C policy."""

from pathlib import Path
import json
import numpy as np
import torch

from .parameters import sha256


@torch.no_grad()
def capture_starts(
    ck, robot, layout, wm, motion, mdp, checkpoint, output, steps, horizon, device,
    native=False,
):
    from partymimic.env import TrackingSim, make_sim
    from partymimic.networks import Policy, policy_kwargs
    from partymimic.mdp import flatten_state

    a = ck["args"]
    if a.get("experiment_group") != "C":
        raise ValueError("Capture requires a frozen C policy")
    response = json.loads(Path(a["response_manifest"]).read_text())
    fit = set(response["splits"]["fit"]["session_ids"])
    clips = [i for i, n in enumerate(motion.names) if n.split("__seg")[0] in fit]
    if not clips:
        raise ValueError("No fit-split motions for actor starts")
    N = a["num_envs"] if native else len(clips)
    K = a["goal_frames"]
    hist_len = a["hist"]
    widths = lambda k: tuple(int(x) for x in a[k].split(","))
    pi = Policy(
        mdp.obs_dim,
        motion.goal_dim(K),
        robot.num_joints,
        hist=hist_len,
        enc_hidden=widths("enc_hidden"),
        dec_hidden=widths("dec_hidden"),
        **policy_kwargs(a, robot),
    ).to(device)
    pi.load_state_dict(ck.get("pi_live", ck["pi"]))
    pi.eval()
    env = TrackingSim(
        make_sim(robot), N, robot, motion, mdp, device, eval_mode=True, randomize=native
    )
    ids = torch.arange(N, device=device)
    clip_ids = torch.tensor([clips[i % len(clips)] for i in range(N)], device=device)
    if native:
        env.reset_noise = True
    env.reset(ids, motion.clip_start[clip_ids])
    hist = None
    stored = []
    for t in range(steps):
        observation = env.observe()
        hist = (
            observation[:, None].repeat(1, hist_len, 1)
            if hist is None
            else torch.cat((hist[:, 1:], observation[:, None]), 1)
        )
        good = (~env.done_eval) & (env.f + horizon < motion.clip_end[clip_ids])
        if t % 4 == 0 and bool(good.any()):
            stored.append(
                {
                    k: v[good].cpu().numpy()
                    for k, v in dict(
                        s=flatten_state(env.state()),
                        a_prev=env.a_prev,
                        xi=env.xi,
                        f=env.f,
                        obs_hist=hist,
                    ).items()
                }
            )
        action = pi(hist, motion.goal(env.f, K))
        env.step(action)
        env.f = torch.minimum(env.f, motion.clip_end[clip_ids] - 1)
        if bool(env.done_eval.all()):
            break
    if not stored:
        raise ValueError("No valid start windows before termination")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "starts.npz"
    np.savez_compressed(
        path, **{k: np.concatenate([r[k] for r in stored]) for k in stored[0]}
    )
    meta = dict(
        schema_version=1,
        file=path.name,
        sha256=sha256(path),
        split="fit",
        checkpoint_sha256=sha256(checkpoint),
        motion_names=motion.names,
        motion_hashes=motion.input_hashes,
        session_ids=sorted(fit),
        horizon=horizon,
        num_envs=N,
        randomized=native,
        gpu=torch.cuda.get_device_name() if str(device).startswith("cuda") else "cpu",
    )
    (output / "manifest.json").write_text(json.dumps(meta, indent=2))
