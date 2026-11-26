"""Run the frozen evaluator, recording matched initial scenarios in a fresh folder."""
import argparse
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

from ral_eight_hour import atomic_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run_dir", "ckpt", "motion_dir", "parameter_manifest", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--plant", choices=("identified", "nominal"), required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    from partymimic.env import TrackingSim
    from ral_eight_hour_worker import state_digest
    original = TrackingSim.reset
    snapshots = []

    def record_reset(env, ids, frames=None):
        result = original(env, ids, frames)
        if frames is not None:
            if not env.eval_mode or not env.randomize or len(ids) != env.N:
                raise ValueError("Expected one explicit full-library randomized evaluation start")
            tensors = dict(xi=env.xi.detach().cpu().clone(), frames=frames.detach().cpu().clone(),
                           observation_noise_half_widths=env.mdp.obs_noise.detach().cpu().clone(),
                           cuda_rng=torch.cuda.get_rng_state().cpu())
            tensors.update({f"state_{i}": t.detach().cpu().clone() for i, t in enumerate(env.state())})
            snapshots.append(tensors)
        return result

    TrackingSim.reset = record_reset
    try:
        # The legacy evaluator writes a DR sidecar next to its checkpoint. Give it
        # a temporary link so source/transfer directories remain read-only.
        with tempfile.TemporaryDirectory(prefix="evaluation_", dir=a.out.parent) as folder:
            folder = Path(folder)
            checkpoint = folder / a.ckpt.name
            checkpoint.symlink_to(a.ckpt.resolve())
            shutil.copyfile(a.run_dir / "args.json", folder / "args.json")
            sys.argv = ["partymimic.evaluate", "--run_dir", str(folder), "--ckpt", str(checkpoint),
                "--motion_dir", str(a.motion_dir), "--parameter_manifest", str(a.parameter_manifest),
                "--plant", a.plant, "--dr_from", "self", "--dr_parts", "dynamics",
                "--seed", str(a.seed), "--out", str(a.out)]
            try:
                runpy.run_module("partymimic.evaluate", run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise
        if len(snapshots) != 1:
            raise ValueError("Missing or repeated explicit evaluation reset")
        state = snapshots[0]
        data = a.out.with_suffix(".scenario.npz")
        np.savez_compressed(data, **{k: v.numpy() for k, v in state.items()})
        atomic_json(a.out.with_suffix(".scenario.json"), dict(
            state_sha256=state_digest({"scenario": state}, ("scenario",)),
            arrays_sha256=digest(data), plant=a.plant, evaluation_seed=a.seed,
            clips=len(state["frames"]), fields=list(state),
            gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda, formal_results=False))
    finally:
        TrackingSim.reset = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
