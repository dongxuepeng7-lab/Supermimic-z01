"""Bounded A/A+ recovery on 5080 and independent checkpoint evaluation on 3090.

Development only: no C sweep, locked response-test access or automatic promotion.
The old frozen-pair protocol is unchanged; online fitting is a new intervention.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import tarfile
import time
from types import SimpleNamespace

from ral_eight_hour import ROOT, atomic_json, digest, read, execute, gpu_lock, check_disk


def jobs_for(config):
    seeds = config["training_seeds"]
    if len(seeds) != 2 or len(set(seeds)) != 2 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Recovery requires exactly two distinct nonnegative seeds")
    if config["groups"] != ["A", "A+"] or config["training"]["wm_fit_source"] != "online":
        raise ValueError("This recovery stage only permits online A/A+")
    if config["training"]["wm_window"] != 8 or config["training"]["horizon"] != 32:
        raise ValueError("Match the base 8-step forward objective and 32-step policy horizon")
    for i, seed in enumerate(seeds):
        for group in (["A", "A+"] if i == 0 else ["A+", "A"]):
            yield dict(id=f"s{seed}_{group.replace('+', 'plus')}", group=group, seed=seed)


def evaluation_summary(result, motions):
    episodes = result["episodes"]
    # MotionLib uses stems as clip IDs; the frozen inventory uses NPZ filenames.
    expected = {Path(name).stem for name in motions}
    if len(episodes) != len(motions) or {e["motion_id"] for e in episodes} != expected:
        raise ValueError("Evaluation motion inventory differs")
    if any(e["repeat"] != 0 or e["eval_seed"] != 0 for e in episodes):
        raise ValueError("Expected one deterministic-start evaluation per motion, seed 0")
    for e in episodes:
        if e["complete"] not in (0, 1) or e["nonfinite_steps"] != 0:
            raise ValueError("Nonfinite or invalid evaluation episode")
        if not all(math.isfinite(e[k]) for k in ("valid_s", "planned_s")) or not 0 < e["valid_s"] <= e["planned_s"] + .021:
            raise ValueError("Invalid evaluation duration")
        if e["complete"] and abs(e["valid_s"] - e["planned_s"]) > .021:
            raise ValueError("Complete episode has inconsistent duration")
        for key in ("q_sqerr_sum_rad2", "v_sqerr_sum", "q_rmse_rad", "v_rmse_m_s"):
            if not math.isfinite(e[key]) or e[key] < 0:
                raise ValueError(f"Invalid evaluation error metric: {key}")
    completed = sum(e["complete"] for e in episodes)
    if result["n_failed"] != len(episodes) - completed:
        raise ValueError("Evaluation totals disagree")
    return dict(completed=completed, clips=len(episodes), completion=completed / len(episodes),
                median_valid_s=statistics.median(e["valid_s"] for e in episodes), nonfinite_steps=0)


def recovery_gate(rows, config, repeats):
    """Conservative operational screen, never a significance/equivalence claim."""
    rule, reasons, pairs = config["screening"], [], []
    for seed in config["training_seeds"]:
        by_group = {}
        for group in ("A", "A+"):
            matches = [r for r in rows if r["seed"] == seed and r["group"] == group and r["final"]]
            if len(matches) != repeats or {r["repeat_check"] for r in matches} != set(range(repeats)):
                reasons.append(f"s{seed}_{group}: missing or duplicate final evaluations")
                continue
            if any(r["nonfinite_steps"] for r in matches):
                reasons.append(f"s{seed}_{group}: nonfinite evaluation")
            values = [r["completion"] for r in matches]
            if max(values) - min(values) > rule["repeat_completion_spread_max"]:
                reasons.append(f"s{seed}_{group}: repeat variation exceeds screen")
            by_group[group] = matches
        if len(by_group) != 2:
            continue
        a, ap = by_group["A"], by_group["A+"]
        if min(r["completion"] for r in a) < rule["baseline_completion_min"]:
            reasons.append(f"s{seed}: baseline A not restored")
        # Use the conservative extrema of repeat checks, not the best observed run.
        drop = max(r["completion"] for r in a) - min(r["completion"] for r in ap)
        ratio = min(r["median_valid_s"] for r in ap) / max(r["median_valid_s"] for r in a)
        if drop > rule["matched_completion_drop_max"] + 1e-12:
            reasons.append(f"s{seed}: A+ completion remains below A")
        if ratio < rule["matched_median_duration_ratio_min"]:
            reasons.append(f"s{seed}: A+ valid duration remains below A")
        pairs.append(dict(seed=seed, worst_completion_drop_pp=100 * drop, worst_duration_ratio=ratio))
    return dict(passed=not reasons, reasons=reasons, pairs=pairs, formal_results=False,
                meaning="Development readiness for a separately designed B comparison; not proof of benefit or equivalence.",
                next_action="Review an online A+/B comparison" if not reasons else "Diagnose the failed checks before B/C",
                automatic_training_authorized=False)


def current_inputs(motions, params, responses):
    from partymimic.experiments.protocol import asset_hash, code_hash
    from partymimic.experiments.runtime import object_hash
    inventory = {p.name: digest(p) for p in sorted(Path(motions).glob("*.npz"))}
    if not inventory:
        raise ValueError("Real motion NPZ files are missing")
    return dict(code_sha256=code_hash(), asset_sha256=asset_hash(), params_sha256=digest(params),
                response_sha256=digest(responses), motion_sha256=object_hash(inventory), motions=inventory,
                evaluator_sha256=digest(ROOT / "src/partymimic/evaluate.py"),
                runner_sha256=digest(__file__))


def check_run(run, protocol, complete=True):
    ledger, args = read(run / "run.json"), read(run / "args.json")
    if ledger["args"] != args or args.get("wm_fit_source") != "online":
        raise ValueError(f"Run arguments differ: {run}")
    for key in ("code_sha256", "asset_sha256", "params_sha256", "motion_sha256", "response_sha256"):
        if ledger[key] != protocol[key]:
            raise ValueError(f"Run input mismatch ({key}): {run}")
    for key, value in protocol.get("training", {}).items():
        if args.get(key) != value:
            raise ValueError(f"Training budget/configuration mismatch ({key}): {run}")
    if complete and (ledger["status"] != "completed" or ledger["policy_updates"] != args["iterations"]
                     or ledger["wm_updates"] != args["iterations"] * args["wm_batches"]):
        raise ValueError(f"Incomplete training budget: {run}")
    with (run / "checkpoints.csv").open() as f:
        index = {r["path"]: r for r in csv.DictReader(f)}
    return ledger, index


def check_learning(run):
    import torch
    initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=False)
    final = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    for module in ("wm", "pi", "pi_live", "critic"):
        if any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in final[module].values()):
            raise ValueError(f"Nonfinite {module} weights: {run}")
    for module in ("wm", "pi"):
        if not any(not torch.equal(v, final[module][k]) for k, v in initial[module].items() if "weight" in k):
            raise ValueError(f"{module} weights did not update: {run}")
    with (run / "train_log.csv").open() as f:
        rows = list(csv.DictReader(f))
    if not rows or any(float(r["nonfinite"]) != 0 for r in rows):
        raise ValueError("Skipped nonfinite training updates")
    for row in rows:
        for key in ("wm_loss", "pi_loss", "v_loss", "pi_gradnorm"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite training metric: {key}")


def evaluate_checkpoint(run, checkpoint, output, a, frozen, cutoff, cancelled, lock_fd, record=False):
    import torch
    if current_inputs(a.motions, a.params, a.responses) != frozen:
        raise ValueError("Evaluation inputs/code changed; do not mix revisions")
    ledger, index = check_run(run, frozen)
    entry = index[checkpoint.name]
    fingerprint = digest(checkpoint)
    if fingerprint != entry["sha256"]:
        raise ValueError(f"Checkpoint checksum differs: {checkpoint}")
    command = [sys.executable, "-u", "-m", "partymimic.evaluate", "--run_dir", str(run),
               "--ckpt", str(checkpoint), "--motion_dir", str(a.motions), "--parameter_manifest", str(a.params),
               "--plant", "identified", "--seed", "0", "--out", str(output)]
    if record:
        command += ["--record", str(output.with_suffix(".trajectory.npz"))]
    atomic_json(output.with_suffix(".inputs.json"), dict(checkpoint_sha256=fingerprint,
                training_code_sha256=ledger["code_sha256"], inputs=frozen, command=command,
                plant="identified", eval_seed=0, checkpoint_weights="EMA", formal_results=False,
                gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__), cuda_version=torch.version.cuda))
    status = execute(command, output.with_suffix(".console.log"), cutoff, cancelled, lock_fd)
    if status != "completed":
        raise RuntimeError(f"Evaluation {status}: {output.with_suffix('.console.log')}")
    if digest(checkpoint) != fingerprint or current_inputs(a.motions, a.params, a.responses) != frozen:
        raise ValueError("Evaluation inputs changed during execution")
    result = read(output)
    summary = evaluation_summary(result, frozen["motions"])
    if result["plant"] != "identified":
        raise ValueError("Unexpected evaluation plant")
    if record:
        from ral_policy_diagnose import trajectory_causes
        atomic_json(output.with_suffix(".termination.json"),
                    trajectory_causes(output.with_suffix(".trajectory.npz"), result["episodes"]))
    return summary


def transfer_bundle(session, config):
    """Keep full training outputs locally; send only selected checkpoints/metadata."""
    archive = session.parent / (session.name + "_transfer.tar.gz")
    points = {config["training"]["ckpt_every"], 2 * config["training"]["ckpt_every"], config["training"]["iterations"]}
    with tarfile.open(archive, "x:gz", compresslevel=1) as tar:
        for p in sorted(session.rglob("*")):
            if not p.is_file() or p.suffix == ".tmp":
                continue
            if p.suffix == ".pt" and (not p.name.startswith("ckpt_") or int(p.stem[5:]) not in points):
                continue
            tar.add(p, arcname=str(Path(session.name) / p.relative_to(session)), recursive=False)
    print(f"Transfer package: {archive} ({archive.stat().st_size / 2**20:.1f} MiB)", flush=True)


def train(a, cutoff, cancelled, lock_fd):
    from partymimic.experiments.protocol import freeze
    from partymimic.experiments.cli import training_command
    config = read(a.config)
    jobs = list(jobs_for(config))
    session = a.out
    session.mkdir(parents=True, exist_ok=False)
    atomic_json(session / "config.json", config)
    status = dict(status="preparing", stage="online_A_Aplus", jobs=jobs, formal_results=False)
    rows = []
    try:
        protocol = freeze(session / "config.json", a.params, a.motions, a.responses, session / "protocol.json")
        # These are new files, finalized before any job starts; historical protocols are never touched.
        protocol.update(training_seeds=config["training_seeds"], groups=config["groups"],
                        diagnostic="online_path_recovery", response_labels_consumed=False)
        atomic_json(session / "protocol.json", protocol)
        frozen = current_inputs(a.motions, a.params, a.responses)
        atomic_json(session / "inputs.json", frozen)
        for job in jobs:
            disk = check_disk(session, dict(min_free_gib=15, max_output_gib=20))
            if disk["reasons"]:
                raise RuntimeError(disk["message"])
            if time.monotonic() >= cutoff or cancelled[0]:
                raise RuntimeError("Recovery deadline/cancellation reached")
            run = session / "jobs" / job["id"] / "run"
            run.parent.mkdir(parents=True)
            job["status"], status["status"], status["current_job"] = "running", "running", job["id"]
            atomic_json(session / "status.json", status)
            command = training_command(SimpleNamespace(config=session / "config.json", params=a.params,
                        motion_dir=a.motions, group=job["group"], responses=a.responses, seed=job["seed"],
                        protocol=session / "protocol.json", out=run))
            atomic_json(run.parent / "command.json", command)
            print(f"[5080] {job['id']} start; online WM + policy + critic", flush=True)
            outcome = execute(command, run.parent / "console.log", min(cutoff, time.monotonic() + 3600), cancelled, lock_fd)
            if outcome != "completed":
                job["status"] = outcome
                raise RuntimeError(f"Training {outcome}: {run.parent / 'console.log'}")
            check_run(run, protocol)
            check_learning(run)
            checkpoint = run / f"ckpt_{config['training']['iterations']:06d}.pt"
            quick = evaluate_checkpoint(run, checkpoint, run.parent / "quick_eval.json", a, frozen,
                                        cutoff, cancelled, lock_fd)
            row = dict(**quick, group=job["group"], seed=job["seed"], final=True, repeat_check=0)
            rows.append(row)
            job.update(status="completed", quick_eval=quick)
            atomic_json(session / "status.json", status)
            print(f"[5080] {job['id']} completion {quick['completed']}/{quick['clips']}", flush=True)
            if job["group"] == "A" and quick["completion"] < config["screening"]["baseline_completion_min"]:
                raise RuntimeError("Baseline A failed the development screen; stop and diagnose before spending more training")
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="stopped", reason=str(exc))
        for job in jobs:
            if job.get("status") == "running":
                job.update(status="failed_validation", reason=str(exc))
        raise
    finally:
        status["current_job"] = None
        atomic_json(session / "status.json", status)
        atomic_json(session / "quick_judgment.json", dict(rows=rows, gate=recovery_gate(rows, config, 1),
                    independent_3090_review_required=True))
        transfer_bundle(session, config)


def evaluate(a, cutoff, cancelled, lock_fd):
    session, output = a.session, a.out
    status, config, frozen = read(session / "status.json"), read(session / "config.json"), read(session / "inputs.json")
    protocol = read(session / "protocol.json")
    if (digest(session / "config.json") != protocol["config_sha256"]
            or protocol["training"] != config["training"]
            or protocol["training_seeds"] != config["training_seeds"] or protocol["groups"] != config["groups"]):
        raise ValueError("Transferred configuration differs from the frozen protocol")
    expected = list(jobs_for(config))
    if status["status"] != "completed" or [j["id"] for j in status["jobs"]] != [j["id"] for j in expected]:
        raise ValueError("Need the complete four-run recovery session, not an old night01 or partial transfer")
    if current_inputs(a.motions, a.params, a.responses) != frozen:
        raise ValueError("3090 code/assets/data must match the new 5080 recovery session")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "source.json", dict(session=str(session), inputs=frozen,
                session_status_sha256=digest(session / "status.json"), config=config,
                limits="Training-reference development evaluation; shared identified plant only; no heldout claim."))
    rows = []
    report = dict(status="running", rows=rows, response_test_accessed=False, formal_results=False)
    try:
        for job in expected:
            run = session / "jobs" / job["id"] / "run"
            ledger, indexed = check_run(run, protocol)
            if ledger["protocol_sha256"] != digest(session / "protocol.json"):
                raise ValueError("Transferred run references a different protocol")
            if ledger["group"] != job["group"] or ledger["training_seed"] != job["seed"]:
                raise ValueError("Transferred run identity differs")
            final = config["training"]["iterations"]
            points = sorted({min(final, config["training"]["ckpt_every"]), min(final, 2 * config["training"]["ckpt_every"]), final})
            for iteration in points:
                for repeat in range(2 if iteration == final else 1):
                    name = f"{job['id']}_i{iteration:06d}_r{repeat}"
                    print(f"[3090] evaluating {name}", flush=True)
                    ck = run / f"ckpt_{iteration:06d}.pt"
                    result = evaluate_checkpoint(run, ck, output / f"{name}.json", a, frozen, cutoff, cancelled, lock_fd,
                                                  record=iteration == final and repeat == 0 and job["group"] == "A+")
                    rows.append(dict(**result, group=job["group"], seed=job["seed"], iteration=iteration,
                                     checkpoint_sha256=indexed[ck.name]["sha256"], final=iteration == final,
                                     repeat_check=repeat))
                    atomic_json(output / "summary.json", report)
        report.update(status="completed", gate=recovery_gate(rows, config, 2))
    except BaseException as exc:
        report.update(status="stopped", reason=str(exc))
        raise
    finally:
        atomic_json(output / "summary.json", report)
        review = output.parent / (output.name + "_review.tar.gz")
        with tarfile.open(review, "x:gz", compresslevel=1) as tar:
            tar.add(output, arcname=output.name)
        print(f"Review package: {review}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("train", "evaluate"))
    ap.add_argument("--out", type=Path, required=True, help="new directory; never overwrite old results")
    ap.add_argument("--session", type=Path, help="evaluate: extracted new 5080 recovery session")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/ral/recovery_online_v1.json")
    ap.add_argument("--motions", type=Path, default=ROOT / "data/motions/z01_lafan_v2_clean")
    ap.add_argument("--params", type=Path, default=ROOT / "release/ral/params.csv")
    ap.add_argument("--responses", type=Path, default=ROOT / "data/ral_pairs_v1/manifest.json")
    ap.add_argument("--hours", type=float, default=2, help="whole-stage wall-clock ceiling, not a target duration")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-gpu-mismatch", action="store_true", help="explicit integration test override")
    a = ap.parse_args()
    if not math.isfinite(a.hours) or not .02 <= a.hours <= 8:
        ap.error("--hours must be between .02 and 8")
    if a.mode == "evaluate" and not a.session:
        ap.error("evaluate requires --session")
    os.chdir(ROOT)
    for name in ("out", "session", "config", "motions", "params", "responses"):
        if getattr(a, name) is not None:
            setattr(a, name, getattr(a, name).resolve())
    config = read(a.config if a.mode == "train" else a.session / "config.json")
    jobs = list(jobs_for(config))
    if a.dry_run:
        print(json.dumps(dict(mode=a.mode, output=str(a.out), jobs=jobs, config=config,
                              note="Plan only; no GPU execution or claim that remote inputs are present"), indent=2, ensure_ascii=False))
        return 0
    if a.out.exists() or any(a.out.parent.glob(a.out.name + "_*.tar.gz")):
        ap.error("Output/archive already exists; use a new attempt name")
    import torch
    expected_gpu = "5080" if a.mode == "train" else "3090"
    if not torch.cuda.is_available() or (expected_gpu not in torch.cuda.get_device_name() and not a.allow_gpu_mismatch):
        ap.error(f"Expected {expected_gpu}; detected {torch.cuda.get_device_name() if torch.cuda.is_available() else 'no CUDA'}")
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    import hashlib
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock = ROOT / "logs/ral_8h" / (".gpu_" + hashlib.sha256(visible.encode()).hexdigest()[:12] + ".lock")
    with gpu_lock(lock) as lock_fd:
        (train if a.mode == "train" else evaluate)(a, time.monotonic() + a.hours * 3600 - 20, cancelled, lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
