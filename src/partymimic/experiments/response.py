"""Frozen paired-perturbation data and multi-step directional response supervision."""

from pathlib import Path
import json
import math

import numpy as np
import torch

from partymimic.mdp import unflatten_state
from .parameters import sha256


class ResponseData:
    """NPZ block arrays; each row is a complete pair, never a resampled time point.

    Manifest splits contain disjoint session IDs and separately hashed NPZ files.
    All model variants see the same forward labels; only C adds the response loss.
    """

    def __init__(
        self,
        manifest,
        split,
        robot,
        layout,
        params_hash,
        device="cpu",
        allow_test=False,
    ):
        if split == "test" and not allow_test:
            raise ValueError(
                "Locked test responses cannot enter training or calibration"
            )
        self.path = Path(manifest).resolve()
        meta = self.meta = json.loads(self.path.read_text())
        if meta.get("schema_version") != 1 or meta.get("source_type") not in (
            "physics_paired",
            "hardware_paired",
            "synthetic_test",
        ):
            raise ValueError("Unknown response data schema/source")
        if (
            meta.get("joint_names") != list(robot.joint_names)
            or meta.get("control_dt") != 0.02
        ):
            raise ValueError("Response joint order or control period mismatch")
        if meta.get("params_sha256") != params_hash:
            raise ValueError("Response dataset uses different frozen plant parameters")
        sessions = [
            set(meta["splits"][s]["session_ids"]) for s in ("fit", "cal", "test")
        ]
        if any(sessions[i] & sessions[j] for i in range(3) for j in range(i)):
            raise ValueError("Session leakage across fit/cal/test splits")
        entry = meta["splits"][split]
        path = self.path.parent / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Response archive hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        J, X = robot.num_joints, layout.dim
        B, H, _ = arrays["a_plus"].shape
        expected = {
            "s0": (B, 13 + 2 * J),
            "a_prev": (B, J),
            "xi": (B, X),
            "a_plus": (B, H, J),
            "a_minus": (B, H, J),
            "s_plus": (B, H, 13 + 2 * J),
            "s_minus": (B, H, 13 + 2 * J),
            "epsilon": (B,),
            "session_id": (B,),
            "block_id": (B,),
        }
        if B < 1 or H < 1:
            raise ValueError("Empty response dataset")
        for k, shape in expected.items():
            if k not in arrays or arrays[k].shape != shape:
                raise ValueError(f"Response array {k} must have shape {shape}")
        if set(arrays["session_id"].astype(str)) != set(entry["session_ids"]):
            raise ValueError("Response sessions disagree with manifest")
        if len(set(arrays["block_id"].astype(str))) != B:
            raise ValueError("Repeated response block IDs")
        for k, arr in arrays.items():
            if arr.dtype.kind in "fiu" and not np.isfinite(arr).all():
                raise ValueError(f"Nonfinite response data: {k}")
        if np.any(arrays["epsilon"] <= 0):
            raise ValueError("Response epsilon must be positive (joint-target radians)")
        # Perturbations are normalized over the complete joint-target sequence.
        delta = (arrays["a_plus"] - arrays["a_minus"]) * np.asarray(
            robot.action_scale_list()
        )
        norm = np.linalg.norm(delta.reshape(B, -1), axis=1)
        if not np.allclose(norm, 2 * arrays["epsilon"], rtol=0.002, atol=1e-7):
            raise ValueError(
                "Paired actions do not match recorded perturbation norm/epsilon"
            )
        for k in ("s0", "s_plus", "s_minus"):
            if not np.allclose(
                np.linalg.norm(arrays[k][..., 3:7], axis=-1), 1, atol=1e-3
            ):
                raise ValueError("Response root quaternions must be normalized wxyz")
        self.arrays, self.horizon, self.count = arrays, H, B
        self.data = {
            k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device)
            for k in expected
            if k not in ("session_id", "block_id")
        }
        scales = meta.get("output_scales", {})
        self.q_scale, self.v_scale = float(scales["q_rad"]), float(scales["qd_rad_s"])
        if min(self.q_scale, self.v_scale) <= 0:
            raise ValueError("Positive response output scales required")
        if not np.isfinite([self.q_scale, self.v_scale]).all():
            raise ValueError("Finite response output scales required")

    def sample(self, count, generator=None):
        ids = torch.randint(
            self.count, (count,), device=self.data["s0"].device, generator=generator
        )
        return {k: v[ids] for k, v in self.data.items()}


def paired_predictions(wm, batch, normalize):
    predictions = []
    xi_n = normalize(batch["xi"])
    for side in ("plus", "minus"):
        S, previous = unflatten_state(batch["s0"]), batch["a_prev"]
        states = []
        for action in batch["a_" + side].unbind(1):
            S = wm.step(S, action, previous, xi_n)
            previous = action
            states.append(S)
        predictions.append(
            tuple(torch.stack([s[i] for s in states], 1) for i in range(6))
        )
    return predictions


def validate_response_objective(horizon, loss, huber_delta, available_horizon):
    if not isinstance(horizon, int) or not 0 <= horizon <= available_horizon:
        raise ValueError("response_horizon must be 0 (all) or within the recorded horizon")
    if loss not in ("mse", "huber"):
        raise ValueError("response_loss must be mse or huber")
    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("response_huber_delta must be finite and positive")


@torch.no_grad()
def initialize_pair_normalizers(wm, dataset, normalize):
    """Common fit-only statistics for A+/B/C, independent of response horizon/loss."""
    d = dataset.data
    for side in ("plus", "minus"):
        current = torch.cat((d["s0"][:, None], d["s_" + side][:, :-1]), 1)
        previous = torch.cat((d["a_prev"][:, None], d["a_" + side][:, :-1]), 1)
        xi = d["xi"][:, None].expand(-1, dataset.horizon, -1)
        for start in range(0, dataset.count, 256):
            sl = slice(start, start + 256)
            S0, S1 = unflatten_state(current[sl]), unflatten_state(d["s_" + side][sl])
            wm.in_norm.update(wm.inputs(S0, d["a_" + side][sl], previous[sl], normalize(xi[sl])))
            wm.out_norm.update(wm.targets(S0, S1))
            wm.inc_norm.update(wm.increments(S0, S1))


def paired_losses(wm, batch, normalize, q_scale, v_scale, *,
                  response_horizon=0, response_loss="mse", response_huber_delta=1.0):
    """Full-span common forward loss, configurable response prefix, raw full-span errors.

    The diagnostic errors are never clipped/robustified. A shorter response prefix
    does not remove later forward labels or change the normalization statistics.
    Defaults exactly retain the pilot01 objective.
    """
    validate_response_objective(response_horizon, response_loss, response_huber_delta,
                                batch["a_plus"].shape[1])
    plus, minus = paired_predictions(wm, batch, normalize)
    target_plus, target_minus = (
        unflatten_state(batch["s_plus"]),
        unflatten_state(batch["s_minus"]),
    )
    # Same full-state forward objective and same paired samples for A+, B and C.
    scale = wm.inc_norm.std().clamp(min=0.001)
    forward = (
        sum(
            ((wm.state_error(pred, target) / scale) ** 2).mean()
            for pred, target in ((plus, target_plus), (minus, target_minus))
        )
        / 2
    )
    eps = 2 * batch["epsilon"][:, None, None]
    errors = torch.cat(
        (
            ((plus[4] - minus[4]) - (target_plus[4] - target_minus[4])) / eps / q_scale,
            ((plus[5] - minus[5]) - (target_plus[5] - target_minus[5])) / eps / v_scale,
        ),
        -1,
    )
    supervised = errors[:, :response_horizon] if response_horizon else errors
    if response_loss == "huber":
        objective = torch.nn.functional.huber_loss(
            supervised, torch.zeros_like(supervised), delta=response_huber_delta)
    else:
        objective = supervised.square().mean()
    return forward, objective, errors.square().mean(-1).sqrt()
