"""Matched, weight-only B -> B/C continuation (never an exact optimizer resume)."""
from pathlib import Path
import json

import torch

from .parameters import sha256


def validate_parent(path, args):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    old = ck["args"]
    if old.get("experiment_group") != "B" or old.get("wm_fit_source") != "online":
        raise ValueError("Matched continuation requires a complete online B checkpoint")
    if any(k not in ck for k in ("wm", "pi", "pi_live", "critic", "iteration", "sim_steps")):
        raise ValueError("Parent lacks WM/live actor/EMA actor/critic or budget metadata")
    keys = ("robot", "seed", "num_envs", "interface", "hist", "goal_frames", "wm_hidden",
            "wm_structured", "friction_speed", "wm_amp", "amp", "enc_hidden", "dec_hidden",
            "pi_ref_skip", "ref_enc", "xattn_dim", "xattn_heads", "kp_skip", "kp_aux",
            "horizon", "wm_window", "std_scale", "contact_reward", "joint_velocity_penalty",
            "joint_velocity_margin", "gamma", "pi_ema", "wm_opt", "pi_opt")
    for key in keys:
        if old.get(key) != getattr(args, key):
            raise ValueError(f"Continuation changed parent setting {key}: {old.get(key)} != {getattr(args, key)}")
    if old.get("parameter_manifest_sha256") != sha256(args.parameter_manifest):
        raise ValueError("Parent parameter manifest differs")
    meta = json.loads(Path(args.response_manifest).read_text())
    if meta.get("teacher_checkpoint_sha256") != sha256(path):
        raise ValueError("GPU pairs were not generated from this paired parent")
    motions = {p.name: sha256(p) for p in sorted(Path(args.motion_dir).glob("*.npz"))}
    if meta.get("motion_hashes") != motions:
        raise ValueError("Continuation motion library differs from GPU pairs")
    return ck


def validate_native_pairs(meta, args):
    import importlib.metadata
    from .protocol import code_hash, asset_hash
    expected = dict(num_envs=args.num_envs, control_dt=0.02, physics_dt=0.005,
                    gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                    code_sha256=code_hash(), asset_sha256=asset_hash())
    for package in ("mujoco", "mujoco-warp", "mjlab", "warp-lang"):
        expected[package] = importlib.metadata.version(package)
    if meta.get("backend") != "TrackingSim/mujoco_warp/native_pairs_v1":
        raise ValueError("Online response supervision requires native GPU pairs, not old CPU labels")
    for key, value in expected.items():
        if meta.get("runtime", {}).get(key) != value:
            raise ValueError(f"GPU pair generation/training mismatch: {key}")


def bounded_response_backward(forward, response, parameters, weight, max_ratio):
    """Detached per-update cap on response gradient norm relative to forward gradient.

    This is an engineering safeguard, not a monotonic-improvement guarantee. The
    optimizer (e.g. Muon) can transform the summed gradient after this operation.
    """
    parameters = list(parameters)
    gf = torch.autograd.grad(forward, parameters, allow_unused=True)
    gr = torch.autograd.grad(response, parameters, allow_unused=True)
    zero = forward.new_zeros(())
    nf = sum((g.float().square().sum() for g in gf if g is not None), zero).sqrt()
    nr = sum((g.float().square().sum() for g in gr if g is not None), zero).sqrt()
    effective = torch.minimum(forward.new_tensor(weight), max_ratio * nf / nr.clamp_min(1e-12)).detach()
    for p, f, r in zip(parameters, gf, gr):
        combined = None if f is None and r is None else (
            (torch.zeros_like(p) if f is None else f) +
            effective * (torch.zeros_like(p) if r is None else r))
        # MuonAdam's FlatAdam parameters retain views into one gradient buffer.
        # Replacing those views would silently make the Adam part use stale zeros.
        if p.grad is not None:
            p.grad.zero_() if combined is None else p.grad.copy_(combined)
        else:
            p.grad = combined
    return effective, nf.detach(), nr.detach()
