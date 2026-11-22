"""Common fit/cal forward and directional-response diagnostics; never opens test."""
import argparse
from pathlib import Path

import torch

from ral_eight_hour import atomic_json, digest


@torch.no_grad()
def metrics(wm, ds, layout):
    from partymimic.experiments.response import paired_predictions
    from partymimic.mdp import unflatten_state
    qe, re, ze = [], [], []
    for i in range(0, ds.count, 128):
        b = {k: v[i:i+128] for k, v in ds.data.items()}
        plus, minus = paired_predictions(wm, b, layout.normalize)
        tp, tm = unflatten_state(b["s_plus"]), unflatten_state(b["s_minus"])
        eps = 2 * b["epsilon"][:, None, None]
        target = torch.cat(((tp[4]-tm[4])/ds.q_scale, (tp[5]-tm[5])/ds.v_scale), -1)/eps
        pred = torch.cat(((plus[4]-minus[4])/ds.q_scale, (plus[5]-minus[5])/ds.v_scale), -1)/eps
        qe.append(((plus[4]-tp[4]).square().mean(-1)+(minus[4]-tm[4]).square().mean(-1)).cpu()/2)
        re.append((pred-target).square().mean(-1).cpu())
        ze.append(target.square().mean(-1).cpu())
    q, r, z = (torch.cat(x).double() for x in (qe, re, ze))
    if ds.horizon < 32:
        raise ValueError("Insufficient response horizons")
    if any(not bool(torch.isfinite(x).all()) for x in (q, r, z)):
        return dict(valid=False, blocks=ds.count, reason="Nonfinite development prediction; no finite-only averaging",
                    nonfinite_values={name: int((~torch.isfinite(x)).sum()) for name, x in (("q", q), ("response", r), ("zero", z))})
    return dict(valid=True, blocks=ds.count,
        q_rmse_rad={str(h): float(q[:, :h].mean().sqrt()) for h in (8, 32)},
        response_rmse={str(h): float(r[:, :h].mean().sqrt()) for h in (4, 8)},
        zero_response_rmse={str(h): float(z[:, :h].mean().sqrt()) for h in (4, 8)})


def main():
    from partymimic.experiments.cli import load_context
    from partymimic.experiments.response import ResponseData
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "params", "responses", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("Use a fresh output file")
    fingerprint = digest(a.checkpoint)
    _, robot, layout, wm = load_context(a.checkpoint, a.params, device="cuda:0")
    result = dict(checkpoint_sha256=fingerprint, response_manifest_sha256=digest(a.responses),
                  formal_results=False, response_test_accessed=False,
                  interpretation="CPU pair labels; diagnostic until backend consistency is established", splits={})
    for split in ("fit", "cal"):
        ds = ResponseData(a.responses, split, robot, layout, digest(a.params), "cuda:0")
        result["splits"][split] = metrics(wm, ds, layout)
    if digest(a.checkpoint) != fingerprint:
        raise ValueError("Checkpoint changed during evaluation")
    atomic_json(a.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
