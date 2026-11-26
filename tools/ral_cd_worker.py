"""Single isolated job for the native-response C/D continuation pipeline."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("pairs", "calibrate", "starts", "window", "responses"))
    for name in ("checkpoint", "params", "motions", "out", "config"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("responses", "split-manifest", "calibration", "starts"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--group", choices=("D", "C-perm"), default="D")
    p.add_argument("--fixed-horizon", type=int)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    import numpy as np
    import torch
    from partymimic.experiments.parameters import sha256
    from partymimic.experiments.cli import load_context
    from partymimic.experiments.response import ResponseData, paired_losses, paired_predictions
    from partymimic.mdp import unflatten_state
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.kind == "pairs":
        from partymimic.experiments.native_pairs import build_native_pairs
        build_native_pairs(a.checkpoint, a.params, a.motions, a.split_manifest, a.out,
                           seed=a.seed + 7301, **cfg["pairs"])
        return
    context = load_context(a.checkpoint, a.params, a.motions, "cuda:0")
    ck, robot, layout, wm, motion, mdp = context
    if a.kind in ("calibrate", "responses"):
        ds = ResponseData(a.responses, "cal", robot, layout, sha256(a.params), "cuda:0")
    if a.kind == "calibrate":
        from partymimic.experiments.windows import calibrate_windows
        w = cfg["windows"]
        result = calibrate_windows(wm, ds, layout.normalize, a.checkpoint,
            w["short"], w["long"], w["edges"], w["tolerance"], w["quantile"], w["min_blocks"],
            w["signal_floor"], binning=w["binning"], kin=mdp.kin, feet=mdp.feet, relative=w["relative"])
        result.update(config_sha256=sha256(a.config), formal_results=False,
                      response_test_accessed=False, gpu=torch.cuda.get_device_name())
        with a.out.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
    elif a.kind == "starts":
        from partymimic.experiments.starts import capture_starts
        if sha256(a.responses) != ck["args"]["response_manifest_sha256"]:
            raise ValueError("Relocated response manifest differs")
        ck["args"]["response_manifest"] = str(a.responses)
        capture_starts(*context, a.checkpoint, a.out, cfg["windows"]["capture_steps"],
                       cfg["windows"]["long"], "cuda:0", native=True)
    elif a.kind == "window":
        from partymimic.experiments.windows import train_windows
        w = cfg["windows"]
        # Release the context's duplicate WM before train_windows constructs its trainable graph.
        del context, wm
        train_windows(a.checkpoint, a.calibration, a.starts, a.out, a.group,
                      robot, motion, mdp, layout, w["iterations"], w["batch_size"], a.seed,
                      w["lr"], "cuda:0", fixed_horizon=a.fixed_horizon)
    elif a.kind == "responses":
        results = {h: [] for h in (4, 8, 32)}
        qerr, verr = [], []
        with torch.no_grad():
            for start in range(0, ds.count, 64):
                batch = {k: v[start:start + 64] for k, v in ds.data.items()}
                _, _, errors = paired_losses(wm, batch, layout.normalize, ds.q_scale, ds.v_scale)
                pred = paired_predictions(wm, batch, layout.normalize)
                truth = [unflatten_state(batch["s_" + s]) for s in ("plus", "minus")]
                for h in results:
                    results[h].extend(errors[:, :h].square().mean(-1).sqrt().cpu().tolist())
                qerr.extend(((pred[0][4] - truth[0][4]).square().mean((1, 2)) / 2 +
                             (pred[1][4] - truth[1][4]).square().mean((1, 2)) / 2).sqrt().cpu().tolist())
                verr.extend(((pred[0][5] - truth[0][5]).square().mean((1, 2)) / 2 +
                             (pred[1][5] - truth[1][5]).square().mean((1, 2)) / 2).sqrt().cpu().tolist())
        sessions = ds.arrays["session_id"].astype(str)
        rows = []
        for session in sorted(set(sessions)):
            mask = sessions == session
            rows.append(dict(session=session, blocks=int(mask.sum()),
                q_rmse32_rad=float(np.mean(np.asarray(qerr)[mask])),
                qd_rmse32_rad_s=float(np.mean(np.asarray(verr)[mask])),
                zero_response_rmse4=float(ds.arrays["signal_prefix"][mask, 3].mean()),
                **{f"response_rmse{h}": float(np.mean(np.asarray(v)[mask])) for h, v in results.items()}))
        result = dict(checkpoint_sha256=sha256(a.checkpoint), response_sha256=sha256(a.responses),
                      split="cal", rows=rows, formal_results=False, response_test_accessed=False,
                      aggregation="RMSE per block, equal block mean within session, equal session mean",
                      means={k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("session", "blocks")})
        with a.out.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
