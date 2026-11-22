"""Read-only closed-loop diagnosis using the unchanged checkpoint evaluator.

Compare initial / final EMA / final live policies at identical reference starts,
plus a reference-PD control (zero learned residual). No training or response-test
access. This is reference-library development diagnosis, not formal evaluation.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def trajectory_causes(path, episodes):
    """The first non-alive sample is the post-termination state, when recorded.

The original evaluator records pre-step states, so the last world to terminate
has no terminal sample. Keep that case explicitly unknown instead of guessing.
"""
    import numpy as np
    from partymimic.mdp import RewardCfg

    counts = dict(height_only=0, tilt_only=0, height_and_tilt=0,
                  neither_threshold=0, terminal_sample_unavailable=0)
    details = []
    cfg = RewardCfg()
    with np.load(path, allow_pickle=False) as archive:
        d = {key: archive[key] for key in ("names", "alive", "f", "root_quat", "root_p", "ref_root_p", "ref_root_R")}
        for c, name in enumerate(d["names"]):
            if episodes[c]["motion_id"] != str(name):
                raise ValueError("Trajectory/result motion ordering differs")
            if episodes[c]["complete"]:
                counts["reference_complete"] = counts.get("reference_complete", 0) + 1
                details.append(dict(motion_id=str(name), cause="reference_complete"))
                continue
            ends = np.flatnonzero(~d["alive"][:, c])
            if not len(ends):
                counts["terminal_sample_unavailable"] += 1
                details.append(dict(motion_id=str(name), cause="terminal_sample_unavailable"))
                continue
            t = int(ends[0])
            f = int(d["f"][t, c])
            q = d["root_quat"][t, c]  # wxyz
            rzz = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
            height = float(abs(d["root_p"][t, c, 2] - d["ref_root_p"][f, 2]))
            tilt = float(abs(rzz - d["ref_root_R"][f, 2, 2]))
            if not np.isfinite([height, tilt]).all():
                cause = "nonfinite"
                counts[cause] = counts.get(cause, 0) + 1
            else:
                h, r = height > cfg.term_height, tilt > cfg.term_tilt
                cause = "height_and_tilt" if h and r else "height_only" if h else "tilt_only" if r else "neither_threshold"
                counts[cause] += 1
            details.append(dict(motion_id=str(name), cause=cause,
                                height_error_m=height if np.isfinite(height) else None,
                                tilt_error=tilt if np.isfinite(tilt) else None))
    return dict(counts=counts, episodes=details,
                note="Threshold diagnostics, not a causal explanation of the fall; terminal sample unavailable for last terminating worlds.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, default=ROOT / "logs/ral_verification/night01_5080_review_import/night01_5080")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--candidates", nargs="+", choices=["B", "C_reference", "C_huber_1e2"],
                    default=["B", "C_reference", "C_huber_1e2"])
    ap.add_argument("--plants", nargs="+", choices=["nominal", "identified"], default=["identified"])
    ap.add_argument("--motions", type=Path, default=ROOT / "data/motions/z01_lafan_v2_clean")
    ap.add_argument("--params", type=Path, default=ROOT / "release/ral/params.csv")
    ap.add_argument("--dry-run", action="store_true", help="Validate file layout and print tasks without loading CUDA")
    a = ap.parse_args()
    if (len(set(a.seeds)) != len(a.seeds) or len(set(a.candidates)) != len(a.candidates)
            or len(set(a.plants)) != len(a.plants) or min(a.seeds) < 0):
        ap.error("Seeds, candidates and plants must be unique; seeds must be nonnegative")
    a.session, a.motions, a.params = a.session.resolve(), a.motions.resolve(), a.params.resolve()
    output = (a.out or ROOT / "logs/ral_diagnostics" / time.strftime("policy_3090_%Y%m%d_%H%M%S")).resolve()
    if output.exists():
        ap.error(f"Choose a new output directory: {output}")
    runs = [a.session / "jobs" / f"s{seed}_{candidate}" / "run"
            for seed in a.seeds for candidate in a.candidates]
    required = [a.params] + [r / name for r in runs
                            for name in ("initial.pt", "last.pt", "args.json", "run.json", "checkpoints.csv")]
    missing = [str(p) for p in required if not p.is_file()]
    if not a.motions.is_dir() or not list(a.motions.glob("*.npz")):
        missing.append(str(a.motions))
    if missing:
        ap.error("Missing inputs; nothing started:\n" + "\n".join(missing))
    tasks = []
    for plant in a.plants:
        for run in runs:
            for variant in ("initial", "final_ema", "final_live"):
                tasks.append(dict(name=f"{run.parent.name}_{variant}_{plant}", run=str(run),
                                  variant=variant, plant=plant, record=run == runs[0]))
        tasks.append(dict(name=f"reference_pd_{plant}", run=str(runs[0]), variant="reference_pd",
                          plant=plant, record=True))
    if a.dry_run:
        print(json.dumps(dict(output=str(output), task_count=len(tasks), tasks=tasks), indent=2))
        return 0

    import torch
    from partymimic.experiments.parameters import sha256
    from partymimic.experiments.protocol import asset_hash, code_hash
    from partymimic.experiments.runtime import object_hash

    actual = dict(code_sha256=code_hash(), asset_sha256=asset_hash(), params_sha256=sha256(a.params),
                  motion_sha256=object_hash({p.name: sha256(p) for p in sorted(a.motions.glob("*.npz"))}))
    provenance = {}
    for run in runs:
        ledger, config = read(run / "run.json"), read(run / "args.json")
        if ledger["status"] != "completed" or ledger["args"] != config:
            raise ValueError(f"Incomplete run or mismatched arguments: {run}")
        for key, value in actual.items():
            if ledger[key] != value:
                raise ValueError(f"Frozen input mismatch ({key}): {run}")
        if config.get("interface", "joint") != "joint" or not config.get("pi_ref_skip") or config.get("kp_skip"):
            raise ValueError("This diagnostic requires joint interface with pi_ref_skip and without kp_skip")
        with (run / "checkpoints.csv").open() as f:
            indexed = {row["path"]: row["sha256"] for row in csv.DictReader(f)}
        checkpoints = {}
        for name in ("initial.pt", "last.pt"):
            digest = sha256(run / name)
            if indexed.get(name) != digest:
                raise ValueError(f"Checkpoint checksum mismatch: {run / name}")
            ck = torch.load(run / name, map_location="cpu", weights_only=True)
            if ck["args"] != config:
                raise ValueError(f"Checkpoint arguments mismatch: {run / name}")
            keys = ("pi", "pi_live") if name == "last.pt" else ("pi",)
            for key in keys:
                if key not in ck or not all(torch.isfinite(v).all().item() for v in ck[key].values()):
                    raise ValueError(f"Missing/nonfinite {key}: {run / name}")
            checkpoints[name] = digest
            del ck
        provenance[str(run)] = dict(checkpoints=checkpoints, run_json_sha256=sha256(run / "run.json"))
    output.mkdir(parents=True)
    report = dict(schema_version=1, purpose="closed-loop development diagnosis", formal_results=False,
                  response_test_accessed=False, scope="All reference clips, first-frame starts, evaluation seed 0, no DR",
                  limitations=["Not unseen-motion generalization", "Reference PD is an open-loop setpoint control, not a stability oracle",
                               "EMA/live comparison does not change the prespecified official checkpoint",
                               "Full-clip completion and valid-duration errors must be interpreted together"],
                  inputs=actual, provenance=provenance, tool_sha256=sha256(__file__),
                  gpu=torch.cuda.get_device_name(), tasks=tasks, results=[], status="running")
    write(output / "summary.json", report)
    print(f"Output: {output}\nTasks: {len(tasks)}; no training", flush=True)
    started = time.monotonic()
    for index, task in enumerate(tasks, 1):
        run = Path(task["run"])
        directory = output / task["name"]
        directory.mkdir()
        variant = task["variant"]
        source = run / ("initial.pt" if variant in ("initial", "reference_pd") else "last.pt")
        checkpoint = source
        if variant in ("final_live", "reference_pd"):
            ck = torch.load(source, map_location="cpu", weights_only=True)
            policy = ck["pi_live"] if variant == "final_live" else ck["pi"]
            if variant == "reference_pd":
                keys = [key for key in policy if re.fullmatch(r"dec\.\d+\.weight", key)]
                if not keys:
                    raise ValueError("Unsupported policy decoder")
                key = max(keys, key=lambda x: int(x.split(".")[1]))
                policy[key] = torch.zeros_like(policy[key])
                policy[key.replace("weight", "bias")] = torch.zeros_like(policy[key.replace("weight", "bias")])
            checkpoint = directory / "diagnostic_policy.pt"
            torch.save(dict(pi=policy, args=ck["args"]), checkpoint)
            write(directory / "args.json", ck["args"])
            del ck, policy
        command = [sys.executable, "-u", "-m", "partymimic.evaluate", "--run_dir", str(run),
                   "--ckpt", str(checkpoint), "--motion_dir", str(a.motions), "--parameter_manifest", str(a.params),
                   "--plant", task["plant"], "--seed", "0", "--out", str(directory / "evaluation.json")]
        if task["record"]:
            command += ["--record", str(directory / "trajectory.npz")]
        write(directory / "command.json", command)
        before = time.monotonic()
        print(f"[{index}/{len(tasks)}] {task['name']} started", flush=True)
        with (directory / "console.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - before
        write(directory / "timing.json", dict(elapsed_s=elapsed, returncode=process.returncode))
        if process.returncode:
            report.update(status="failed", failed_task=task["name"], elapsed_s=time.monotonic() - started)
            write(output / "summary.json", report)
            print(f"Failed; inspect {directory / 'console.log'}", file=sys.stderr)
            return 1
        result = read(directory / "evaluation.json")
        episodes = result["episodes"]
        seconds = [e["valid_s"] for e in episodes]
        row = dict(**task, source_checkpoint=str(source), evaluated_checkpoint_sha256=sha256(checkpoint),
                   elapsed_s=elapsed, episode_count=len(episodes), completed=sum(e["complete"] for e in episodes),
                   failed=result["n_failed"], norm_return=result["norm_return"],
                   median_valid_s=statistics.median(seconds), mean_valid_s=statistics.mean(seconds),
                   max_valid_s=max(seconds), nonfinite_steps=sum(e["nonfinite_steps"] for e in episodes))
        if task["record"]:
            causes = trajectory_causes(directory / "trajectory.npz", episodes)
            write(directory / "termination_details.json", causes)
            row["recorded_terminal_causes"] = causes["counts"]
        report["results"].append(row)
        report["elapsed_s"] = time.monotonic() - started
        write(output / "summary.json", report)
        print(f"  completed {row['completed']}/{row['episode_count']}; median {row['median_valid_s']:.2f}s; wall {elapsed:.1f}s", flush=True)
    report.update(status="completed", elapsed_s=time.monotonic() - started)
    write(output / "summary.json", report)
    print(f"Diagnostic complete: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
