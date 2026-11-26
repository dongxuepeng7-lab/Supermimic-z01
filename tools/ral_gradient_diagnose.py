"""Frozen-model directional derivatives and actuator diagnostics on fit/cal only.

This measures open-loop joint sensitivity, not the complete actor/critic gradient.
CPU labels are a diagnostic reference; passing this does not validate GPU response labels.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from ral_eight_hour import atomic_json, digest


def rollout(wm, state, previous, context, actions):
    from partymimic.mdp import unflatten_state
    current = unflatten_state(state)
    output = []
    for action in actions.unbind(1):
        current = wm.step(current, action, previous, context)
        previous = action
        output.append(torch.cat((current[4], current[5]), -1))
    return torch.stack(output, 1)


def per_block(wm, layout, batch, q_scale, v_scale):
    from partymimic.mdp import unflatten_state
    base = (batch["a_plus"] + batch["a_minus"]) / 2
    eps = batch["epsilon"][:, None, None]
    direction = (batch["a_plus"] - batch["a_minus"]) / (2 * eps)
    context = layout.normalize(batch["xi"])
    scales = base.new_tensor([q_scale] * base.shape[-1] + [v_scale] * base.shape[-1])
    fn = lambda u: rollout(wm, batch["s0"], batch["a_prev"], context, u)
    # The stored direction is normalized in target radians over the full 32-step sequence.
    # Its action-coordinate derivative is D/action_scale; do not renormalize a prefix.
    _, tangent = torch.autograd.functional.jvp(fn, base, direction, create_graph=False)
    with torch.no_grad():
        plus, minus = fn(batch["a_plus"]), fn(batch["a_minus"])
        finite = (plus - minus) / (2 * eps)
        target = (batch["s_plus"][..., 13:] - batch["s_minus"][..., 13:]) / (2 * eps)
        derivative, central, reference = [x / scales for x in (tangent, finite, target)]
        s = unflatten_state(batch["s0"])
        previous, saturated, slopes = batch["a_prev"], [], []
        for action in base.unbind(1):
            target_q = wm.q0 + wm.action_scale * action + batch["xi"][..., wm.offset_slice]
            raw = wm.kp * (target_q - s[4]) - wm.kd * s[5]
            saturated.append((raw.abs() >= wm.limit).float())
            slopes.append(wm.friction / wm.friction_speed * (1 - torch.tanh(s[5] / wm.friction_speed).square()))
            s = wm.step(s, action, previous, context)
            previous = action
        sat, slope = torch.stack(saturated, 1), torch.stack(slopes, 1)
        results = []
        for i in range(base.shape[0]):
            arrays = [derivative[i], central[i], reference[i], sat[i], slope[i], plus[i], minus[i]]
            if not all(bool(torch.isfinite(x).all()) for x in arrays):
                results.append(dict(valid=False, reason="Nonfinite diagnostic; never omit from aggregate silently"))
                continue
            item = dict(valid=True, horizons={})
            for h in (1, 2, 4, 8, 16, 32):
                def rms(x):
                    return float(x[:h].double().square().mean().sqrt())
                x, y = derivative[i, :h].flatten().double(), reference[i, :h].flatten().double()
                norm = float(x.norm() * y.norm())
                item["horizons"][str(h)] = dict(
                    autodiff_response_rms=rms(derivative[i]), physics_response_rms=rms(reference[i]),
                    model_fd_vs_autodiff_rmse=rms(central[i] - derivative[i]),
                    model_fd_vs_cpu_rmse=rms(central[i] - reference[i]),
                    autodiff_vs_cpu_cosine=float(x.dot(y)) / norm if norm > 1e-12 else None,
                    saturation_fraction=float(sat[i, :h].mean()),
                    friction_slope_max_Nms_rad=float(slope[i, :h].max()))
            results.append(item)
    return results


def main():
    from partymimic.experiments.cli import load_context
    from partymimic.experiments.response import ResponseData
    from ral_backend_check import select_blocks
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "params", "responses", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--blocks-per-split", type=int, default=24)
    a = p.parse_args()
    if a.out.exists() or a.blocks_per_split < 1:
        p.error("Fresh output and positive diagnostic count required")
    fingerprint = digest(a.checkpoint)
    _, robot, layout, wm = load_context(a.checkpoint, a.params, device=a.device)
    if wm.mode not in ("matched", "actuator"):
        raise ValueError("Requires A+/B models")
    wm.requires_grad_(False)
    report = dict(checkpoint_sha256=fingerprint, model=wm.mode, response_test_accessed=False,
        formal_results=False, scope="Open-loop joint sensitivity, not full actor gradient",
        saturation_scope="Common PD proxy for both models; A+ does not route its learned input through this clamp",
        difference_scope="Finite-amplitude central differences include nonlinear/contact effects; disagreement alone is not an autodiff bug",
        blocks=[])
    for split in ("fit", "cal"):
        ds = ResponseData(a.responses, split, robot, layout, digest(a.params), a.device)
        ids = select_blocks(ds.arrays["block_id"], a.blocks_per_split)
        for index in ids:
            batch = {k: v[int(index):int(index)+1] for k, v in ds.data.items()}
            row = per_block(wm, layout, batch, ds.q_scale, ds.v_scale)[0]
            report["blocks"].append(dict(row, split=split, block_id=str(ds.arrays["block_id"][index])))
            atomic_json(a.out, report)
    if digest(a.checkpoint) != fingerprint:
        raise ValueError("Checkpoint changed during diagnosis")
    report.update(status="completed", all_finite=all(r["valid"] for r in report["blocks"]))
    atomic_json(a.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
