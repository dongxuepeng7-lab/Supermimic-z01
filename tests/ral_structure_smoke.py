"""Opt-in real GPU integration with tiny budgets. NOT a research/control experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ral_eight_hour import atomic_json, read
from ral_structure import make_protocol, verify_pair
from ral_recovery import check_learning
from partymimic.experiments.cli import training_command


def main():
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        raise ValueError("Use a new smoke output")
    a = SimpleNamespace(params=ROOT / "release/ral/params.csv",
        motions=ROOT / "data/motions/z01_lafan_v2_clean",
        responses=ROOT / "data/ral_pairs_v1/manifest.json")
    cfg = read(ROOT / "configs/ral/structure_online_v1.json")
    cfg["training"].update(iterations=6, num_envs=8, buffer_steps=80, pi_batch=16, wm_batch=16,
        wm_batches=2, wm_hidden="32,32", enc_hidden="32,32", dec_hidden="32,32",
        v_gcons_n=16, ckpt_every=2, log_every=1)
    cfg.update(expected_environment_transitions=(64+15)*8, smoke_only=True)
    protocol = make_protocol(a, cfg, out)
    jobs = [dict(id="s2_Aplus", seed=2, group="A+"), dict(id="s2_B", seed=2, group="B")]
    for job in jobs:
        run = out / "jobs" / job["id"] / "run"
        run.parent.mkdir(parents=True)
        cmd = training_command(SimpleNamespace(config=out / "config.json", params=a.params,
            motion_dir=a.motions, responses=a.responses, group=job["group"], seed=2,
            protocol=out / "protocol.json", out=run))
        if job["group"] == "B":
            cmd += ["--paired_initial_reference", str(out / "jobs/s2_Aplus/run/initial.pt")]
        atomic_json(run.parent / "command.json", cmd)
        print("Smoke training", job["group"], flush=True)
        with (run.parent / "console.log").open("w") as log:
            subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="4"),
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        check_learning(run)
    report = verify_pair(out, protocol, jobs)
    report.update(smoke_only=True, research_training_completed=False, iterations_per_model=6)
    atomic_json(out / "validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
