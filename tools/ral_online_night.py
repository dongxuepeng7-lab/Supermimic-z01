"""Eight-hour online A+/B replication and independent 3090 development evaluation.

Separate protocol; legacy source, old protocols and historical results are preserved.
No locked test arrays, C training, warm resumes or result-dependent hyperparameters.
"""
import argparse
import csv
import hashlib
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

from ral_eight_hour import ROOT, atomic_json, check_disk, digest, execute, gpu_lock, read
from ral_recovery import check_learning, check_run, current_inputs, evaluation_summary, evaluate_checkpoint
from ral_structure import module_hashes, stage_gate, tools_inventory, verified_sessions

CONFIG = ROOT / "configs/ral/online_night_v1.json"
PROTOCOL = ROOT / "release/ral/online_night_protocol_v1.json"


def validate(plan):
    t = plan["training"]
    required = dict(iterations=5000, wm_batches=5, wm_fit_source="online", response_weight=0.,
                    wm_window=8, horizon=32, paired_initialization=True, ckpt_every=5000,
                    ckpt_minutes=0, num_envs=1024, utd=.4, warmup_steps=64, wm_amp="none")
    if plan["groups"] != ["A+", "B"] or any(t.get(k) != v for k, v in required.items()):
        raise ValueError("Use the frozen online A+/B 5000-update plan")
    if plan["C_authorized"] or plan["locked_response_test_access"] or plan["hours"] != 8:
        raise ValueError("This development queue excludes C/test and is bounded to eight hours")
    sets = []
    for machine in ("5080", "4090"):
        m = plan["machines"][machine]
        if any(type(m[k]) is not int or m[k] <= 0 for k in ("seed_start", "seed_stride", "max_pairs")):
            raise ValueError("Invalid deterministic seed allocation")
        sets.append({m["seed_start"]+i*m["seed_stride"] for i in range(m["max_pairs"])})
    if (sets[0] & sets[1] or (sets[0] | sets[1]) != set(plan["training_seeds"])
            or {2, 4} & (sets[0] | sets[1])):
        raise ValueError("Overlapping or incomplete seed allocation")
    if plan["expected_environment_transitions"] != (t["warmup_steps"]+int(t["iterations"]/t["utd"]))*t["num_envs"]:
        raise ValueError("Environment budget mismatch")
    if (plan["evaluation"] != dict(clips=163, plant="identified", seed=0,
            repeats_per_final=2, checkpoint_weights="EMA")
            or plan["evaluator_3090"]["backend_full_split_passes"] != 2
            or plan["evaluator_3090"]["conditions"] != ["identified_training_dr", "nominal_training_dr"]):
        raise ValueError("Evaluation coverage differs from the implemented paired protocol")


def pairs_for(plan, machine):
    validate(plan)
    m = plan["machines"][machine]
    for i in range(m["max_pairs"]):
        seed = m["seed_start"]+i*m["seed_stride"]
        yield dict(id=f"s{seed}", seed=seed, jobs=[
            dict(id=f"s{seed}_{g.replace('+', 'plus')}", group=g, seed=seed) for g in plan["groups"]])


def night_tools():
    return dict(tools_inventory(), **{str(p.relative_to(ROOT)): digest(p) for p in
        (Path(__file__), ROOT / "tools/ral_online_night_worker.py",
         ROOT / "tools/ral_online_eval_worker.py", CONFIG)})


def inputs(a):
    return dict(base=current_inputs(a.motions, a.params, a.responses), tools=night_tools())


def freeze(a, plan, out):
    from partymimic.experiments.protocol import freeze as base_freeze
    validate(plan)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "config.json", plan)
    p = base_freeze(out / "config.json", a.params, a.motions, a.responses, out / "protocol.json")
    p.update(groups=plan["groups"], training_seeds=plan["training_seeds"],
        stage=plan["stage"], tools=night_tools(), screening=plan["screening"],
        evaluation=plan["evaluation"], response_labels_consumed=False,
        expected_environment_transitions=plan["expected_environment_transitions"],
        formal_results=False, C_authorized=False)
    atomic_json(out / "protocol.json", p)
    return p


def assert_current(a, frozen):
    if inputs(a) != frozen:
        raise ValueError("Code, assets or data changed during the queue; stopped")


def verify_training(run, protocol, job):
    import torch
    ledger, index = check_run(run, protocol)
    if (ledger["group"], ledger["training_seed"], ledger["env_steps"]) != (
        job["group"], job["seed"], protocol["expected_environment_transitions"]):
        raise ValueError("Training identity/environment budget differs")
    if ledger["protocol_sha256"] != digest(run.parents[2] / "protocol.json"):
        raise ValueError("Training protocol differs")
    from partymimic.experiments.initialization import stream_seeds
    if ledger.get("random_streams", {}).get("seeds") != stream_seeds(job["seed"]):
        raise ValueError("Initialization stream provenance missing")
    n = protocol["training"]["iterations"]
    for name, iteration, steps in (("initial.pt", 0, 0), (f"ckpt_{n:06d}.pt", n, ledger["env_steps"])):
        if name not in index or digest(run / name) != index[name]["sha256"]:
            raise ValueError("Checkpoint missing or changed")
        ck = torch.load(run / name, map_location="cpu", weights_only=False)
        from ral_eight_hour_worker import finite_weights
        finite_weights(ck, ("wm", "pi", "critic") if iteration == 0 else ("wm", "pi", "pi_live", "critic"))
        if ck["iteration"] != iteration or ck["sim_steps"] != steps or ck["args"] != ledger["args"]:
            raise ValueError("Checkpoint contents do not match full-budget ledger")
    if module_hashes(run / "initial.pt") != ledger["initial_module_sha256"]:
        raise ValueError("Initial weights do not match ledger")
    check_learning(run)
    from ral_eight_hour_worker import state_digest
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    final = torch.load(run / f"ckpt_{n:06d}.pt", map_location="cpu", weights_only=False)
    if digest(run / "last.pt") != index["last.pt"]["sha256"] or state_digest(last, ("wm", "pi", "pi_live", "critic")) != state_digest(final, ("wm", "pi", "pi_live", "critic")):
        raise ValueError("Final snapshot differs from full optimizer checkpoint; do not prune")
    with (run / "train_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    if [int(r["iteration"]) for r in rows] != list(range(n+1)):
        raise ValueError("Missing/duplicate iteration records")
    if any(not math.isfinite(float(r[k])) for r in rows[1:] for k in ("wm_loss", "policy_loss")):
        raise ValueError("Nonfinite iteration loss")
    return ledger


def archive_pair(session, pair, final_iteration):
    """Remove ONLY this queue's generated .pt files, AFTER verifying archive bytes.

    Full optimizer checkpoints are redundant for this no-resume protocol. Keep the
    initial/final tensor snapshots and record every discarded filename/hash explicitly.
    """
    folder = session / "models"
    folder.mkdir(exist_ok=True)
    archive = folder / (pair["id"]+"_models.tar.gz")
    if archive.exists():
        raise FileExistsError(f"Refusing to replace model archive: {archive}")
    files, discarded = [], []
    for job in pair["jobs"]:
        run = session / "jobs" / job["id"] / "run"
        if not run.exists():
            continue
        # Partial/failed runs retain every available checkpoint for diagnosis.
        keep = {"initial.pt", f"ckpt_{final_iteration:06d}.pt"} if job.get("validated") else {p.name for p in run.glob("*.pt")}
        for p in run.iterdir():
            if p.is_symlink():
                raise ValueError("Generated training files must not be symlinks")
            if p.is_file() and (p.suffix != ".pt" or p.name in keep) and p.suffix != ".tmp":
                files.append(p)
            elif p.is_file() and p.suffix == ".pt":
                discarded.append(dict(path=str(p.relative_to(session)), sha256=digest(p),
                    reason="Verified final tensors retained; no optimizer-resume in this protocol"))
    checksums = {str(p.relative_to(session)): digest(p) for p in files}
    temp = archive.with_suffix(archive.suffix+".tmp")
    with tarfile.open(temp, "x:gz", compresslevel=1) as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(session)), recursive=False)
    with tarfile.open(temp) as tar:
        for name, expected in checksums.items():
            h = hashlib.sha256()
            with tar.extractfile(name) as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError("Archive verification failed; original checkpoints retained")
    temp.rename(archive)
    record = dict(archive=str(archive.relative_to(session)), archive_sha256=digest(archive),
                  members=checksums, discarded_optimizer_files=discarded, verified_before_cleanup=True)
    atomic_json(session / (pair["id"]+"_retention.json"), record)
    # Only explicitly scoped new job files, never anything in historical result folders.
    cleanup = {k: v for k, v in checksums.items() if Path(k).suffix == ".pt"}
    cleanup.update({r["path"]: r["sha256"] for r in discarded})
    if any(digest(session / name) != expected for name, expected in cleanup.items()):
        raise ValueError("Checkpoint changed during archiving; originals retained")
    for name in cleanup:
        (session / name).unlink()
    return record


def review_package(session):
    archive = session.parent / (session.name+"_review.tar.gz")
    with tarfile.open(archive, "x:gz", compresslevel=1) as tar:
        for p in sorted(session.rglob("*")):
            if (p.is_file() and "models" not in p.relative_to(session).parts
                    and p.suffix not in (".pt", ".npz", ".tmp")):
                tar.add(p, arcname=str(Path(session.name) / p.relative_to(session)), recursive=False)
    print(f"审阅包（不含模型）：{archive}", flush=True)


def training_queue(a, cutoff, cancelled, fd):
    from partymimic.experiments.cli import training_command
    plan = read(CONFIG)
    # Write status even if input checks fail, so a missing status never masks failure.
    a.out.mkdir(parents=True, exist_ok=False)
    status = dict(status="preparing", machine=a.machine, pairs=[], formal_results=False,
                  deadline_unix=time.time()+max(0, cutoff-time.monotonic()), C_authorized=False)
    atomic_json(a.out / "status.json", status)
    active = None
    try:
        import tempfile
        with tempfile.TemporaryDirectory(dir=a.out) as parent:
            candidate = Path(parent) / "freeze"
            protocol = freeze(a, plan, candidate)
            if protocol != read(PROTOCOL):
                raise ValueError("Synchronize the exact online-night code/data/protocol bundle")
            import shutil
            for name in ("protocol.json", "config.json"):
                shutil.copyfile(candidate / name, a.out / name)
        frozen = inputs(a)
        atomic_json(a.out / "inputs.json", frozen)
        for pair in pairs_for(plan, a.machine):
            if cancelled[0] or time.monotonic() >= cutoff:
                status["status"] = "cancelled" if cancelled[0] else "time_budget_reached"
                break
            disk = check_disk(a.out, plan)
            status["disk_check"] = disk
            print(disk["message"], flush=True)
            # Reserve two GiB for two raw runs and the archive before cleanup.
            if (disk["reasons"] or disk["free_bytes"] < (plan["min_free_gib"]+2)*2**30
                    or disk["output_bytes"]+2*2**30 > plan["max_output_gib"]*2**30):
                status["status"] = "disk_limit_reached"
                break
            active = pair
            status["pairs"].append(pair)
            pair.update(status="running", rows=[])
            status.update(status="running", current_pair=pair["id"])
            atomic_json(a.out / "status.json", status)
            for job in pair["jobs"]:
                assert_current(a, frozen)
                run = a.out / "jobs" / job["id"] / "run"
                run.parent.mkdir(parents=True)
                command = training_command(SimpleNamespace(config=a.out / "config.json", params=a.params,
                    motion_dir=a.motions, group=job["group"], responses=a.responses, seed=job["seed"],
                    protocol=a.out / "protocol.json", out=run))
                if job["group"] == "B":
                    command += ["--paired_initial_reference", str(a.out / "jobs" / pair["jobs"][0]["id"] / "run/initial.pt")]
                atomic_json(run.parent / "command.json", command)
                job["status"] = "training"
                atomic_json(a.out / "status.json", status)
                print(f"[{a.machine}] {job['id']} 开始；剩余 {(cutoff-time.monotonic())/3600:.2f} 小时", flush=True)
                outcome = execute(command, run.parent / "console.log",
                    min(cutoff, time.monotonic()+60*plan["job_timeout_minutes"]), cancelled, fd)
                if outcome != "completed":
                    job["status"] = outcome
                    raise RuntimeError(f"Training {outcome}: {job['id']}")
                ledger = verify_training(run, protocol, job)
                job.update(validated=True, initial_module_sha256=ledger["initial_module_sha256"])
                ck = run / "ckpt_005000.pt"
                for repeat in range(2):
                    result = evaluate_checkpoint(run, ck, run.parent / f"eval_r{repeat}.json", a,
                        frozen["base"], min(cutoff, time.monotonic()+60*plan["job_timeout_minutes"]), cancelled, fd)
                    pair["rows"].append(dict(**result, group=job["group"], seed=job["seed"], final=True, repeat_check=repeat))
                    atomic_json(a.out / "status.json", status)
                if job["group"] == "A+" and min(r["completion"] for r in pair["rows"]) < plan["screening"]["baseline_completion_min"]:
                    raise RuntimeError("A+ control below 70%; stop before B and preserve diagnostics")
                command = [sys.executable, "-u", str(ROOT / "tools/ral_structure_metrics.py"),
                    "--checkpoint", str(ck), "--params", str(a.params), "--responses", str(a.responses),
                    "--out", str(run.parent / "predictions.json")]
                outcome = execute(command, run.parent / "predictions.log",
                    min(cutoff, time.monotonic()+60*plan["job_timeout_minutes"]), cancelled, fd)
                if outcome != "completed":
                    raise RuntimeError(f"Prediction diagnostic {outcome}")
                job.update(status="completed", final_sha256=digest(ck), predictions=read(run.parent / "predictions.json"))
                print(f"[{a.machine}] {job['id']} 完成", flush=True)
            if any(pair["jobs"][0]["initial_module_sha256"][k] != pair["jobs"][1]["initial_module_sha256"][k] for k in ("pi", "critic")):
                raise ValueError("Paired initial policy/critic mismatch")
            pair.update(status="completed", gate=stage_gate(pair["rows"], plan, seeds=[pair["seed"]]))
            pair["retention"] = archive_pair(a.out, pair, 5000)
            active = None
            atomic_json(a.out / "status.json", status)
            assert_current(a, frozen)
            if not pair["gate"]["passed"]:
                status.update(status="control_screen_failed", reason="Preserve this completed negative pair; diagnose before more training")
                break
        else:
            status["status"] = "planned_queue_completed"
    except BaseException as exc:
        status.update(status="stopped", reason=str(exc))
        if active:
            active.update(status="incomplete", reason=str(exc))
        raise
    finally:
        if active:
            try:
                active["retention"] = archive_pair(a.out, active, 5000)
            except Exception as exc:
                status["archive_error"] = str(exc)
        status.update(current_pair=None, completed_pairs=sum(p["status"] == "completed" for p in status["pairs"]))
        atomic_json(a.out / "status.json", status)
        review_package(a.out)


def discover_sessions(paths, parent):
    if paths:
        candidates = [Path(p).resolve() for p in paths]
    else:
        candidates = [p for p in sorted(parent.iterdir()) if (p / "status.json").is_file()] if parent.is_dir() else []
    selected = {}
    for path in candidates:
        if not (path / "status.json").is_file():
            found = [p.name for p in path.parent.iterdir() if p.is_dir() and (p / "status.json").is_file()] if path.parent.exists() else []
            raise ValueError(f"Missing session {path}; available session directories: {found}")
        status = read(path / "status.json")
        if status.get("machine") not in ("5080", "4090") or "jobs" not in status:
            continue
        machine = status["machine"]
        if machine in selected:
            raise ValueError(f"Multiple {machine} sessions; use --sessions to explicitly choose the paired study")
        selected[machine] = path
    if set(selected) != {"5080", "4090"}:
        raise ValueError("Need one 5080 and one 4090 structure transfer; identify by internal status, not archive filename")
    return [selected[m] for m in ("5080", "4090")]


def evaluation_jobs(plan, sessions):
    for seed in plan["evaluator_3090"]["evaluation_seeds"]:
        for condition in plan["evaluator_3090"]["conditions"]:
            cell = f"{condition}_e{seed}"
            for machine in ("5080", "4090"):
                session, jobs = sessions[machine]
                order = jobs if seed % 2 else list(reversed(jobs))
                for job in order:
                    yield dict(id=f"{cell}_{job['id']}", cell=cell, eval_seed=seed,
                        condition=condition, plant=condition.split("_")[0], group=job["group"],
                        seed=job["seed"], run=str(session / "jobs" / job["id"] / "run"))


def summarize_randomized(result, motions, seed, plant):
    if result["plant"] != plant or any(e["eval_seed"] != seed for e in result["episodes"]):
        raise ValueError("Evaluation plant/seed differs from the assigned scenario")
    # Reuse episode arithmetic/finite checks; the original helper is seed-0-specific.
    copy = dict(result, episodes=[dict(e, eval_seed=0) for e in result["episodes"]])
    return evaluation_summary(copy, motions)


def completed_cells(rows):
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)
    expected = {(2, "A+"), (2, "B"), (4, "A+"), (4, "B")}
    out = []
    for cell, values in cells.items():
        if len(values) != 4 or {(r["seed"], r["group"]) for r in values} != expected:
            continue
        if (len({(r["condition"], r["eval_seed"], r["scenario_sha256"]) for r in values}) != 1
                or not values[0]["scenario_sha256"]):
            raise ValueError("Models were not evaluated under identical recorded initial scenarios")
        pairs = []
        for seed in (2, 4):
            a, b = [next(r for r in values if r["seed"] == seed and r["group"] == g) for g in ("A+", "B")]
            pairs.append(dict(training_seed=seed, B_minus_Aplus_completion_pp=100*(b["completion"]-a["completion"]),
                              Aplus_completed=a["completed"], B_completed=b["completed"]))
        out.append(dict(cell=cell, condition=values[0]["condition"], eval_seed=values[0]["eval_seed"], pairs=pairs))
    return out


def compare_backend_processes(first, second):
    import numpy as np
    from ral_backend_check import compare
    reports = [read(p / "summary.json") for p in (first, second)]
    if any(r["status"] != "completed" or r["smoke_only"] for r in reports):
        raise ValueError("Two complete full backend passes are required")
    maps = [{r["block_id"]: r for r in p["blocks"]} for p in reports]
    if maps[0].keys() != maps[1].keys() or reports[0]["config"] != reports[1]["config"]:
        raise ValueError("Backend repeats covered different blocks/configurations")
    files = [{digest(p): p for p in folder.glob("*_trajectories.npz")} for folder in (first, second)]
    rows = []
    for key, row in maps[0].items():
        with np.load(files[0][row["trajectory_sha256"]]) as x, np.load(files[1][maps[1][key]["trajectory_sha256"]]) as y:
            if any(not np.array_equal(x[k], y[k]) for k in ("actions", "epsilon", "scales", "initial_train_integration")):
                raise ValueError("Independent backend processes used different inputs")
            comparisons = compare(x["train_cpu"], np.stack((x["train_gpu"][0], y["train_gpu"][0])),
                                  x["scales"], x["epsilon"], reports[0]["config"])
        rows.append(dict(block_id=key, split=row["split"], comparisons=comparisons,
            passed=all(r["gpu_repeat_response_rmse"] <= r["repeat_limit"] for r in comparisons)))
    return dict(passed=all(r["passed"] for r in rows), blocks=rows, formal_results=False,
                meaning="Cross-process response repeatability only; does not override CPU/GPU discrepancy failures")


def evaluation_queue(a, cutoff, cancelled, fd):
    a.sessions = discover_sessions(a.sessions, a.imports)
    config, base_frozen, sessions = verified_sessions(a)
    plan = read(CONFIG)
    validate(plan)
    protocol = read(PROTOCOL)
    if protocol["tools"] != night_tools() or protocol["config_sha256"] != digest(CONFIG):
        raise ValueError("Night runner differs from frozen release")
    a.out.mkdir(parents=True, exist_ok=False)
    frozen = inputs(a)
    for key in ("code_sha256", "asset_sha256", "params_sha256", "motion_sha256", "response_sha256"):
        if protocol[key] != frozen["base"][key]:
            raise ValueError(f"Night protocol input differs: {key}")
    atomic_json(a.out / "inputs.json", frozen)
    atomic_json(a.out / "config.json", plan)
    atomic_json(a.out / "protocol.json", protocol)
    atomic_json(a.out / "source.json", dict(sessions=[str(s) for s in a.sessions],
        protocol_sha256=digest(PROTOCOL), training_budget=1000,
        source_training_protocol_sha256=digest(a.sessions[0] / "protocol.json"),
        warning="These four source policies have 1000 updates; never pool with the new 5000-update training queue"))
    rows = []
    checkpoint_hashes = {str(s / "jobs" / j["id"] / "run/ckpt_001000.pt"):
        digest(s / "jobs" / j["id"] / "run/ckpt_001000.pt") for s, jobs in sessions.values() for j in jobs}
    report = dict(status="running", machine="3090", rows=rows, backend=[], formal_results=False,
        response_test_accessed=False, C_authorized=False, training_seeds=[2, 4],
        interpretation="Repeated evaluation seeds are not additional training seeds. Uses training dynamics ranges and the default training observation noise; no heldout-hardware claim.")
    atomic_json(a.out / "status.json", report)
    try:
        for repeat in range(plan["evaluator_3090"]["backend_full_split_passes"]):
            assert_current(a, frozen)
            disk = check_disk(a.out, plan)
            if disk["reasons"]:
                raise RuntimeError(disk["message"])
            out = a.out / f"backend_full_r{repeat}"
            command = [sys.executable, "-u", str(ROOT / "tools/ral_online_night_worker.py"),
                "--out", str(out), "--motions", str(a.motions), "--params", str(a.params), "--responses", str(a.responses)]
            print(f"[3090] 全量 fit/cal 后端检查，独立进程 {repeat+1}/2", flush=True)
            outcome = execute(command, a.out / f"backend_full_r{repeat}.log",
                min(cutoff, time.monotonic()+60*plan["evaluator_3090"]["backend_job_timeout_minutes"]), cancelled, fd)
            if outcome != "completed":
                raise RuntimeError(f"Backend validation {outcome}")
            check = read(out / "summary.json")
            report["backend"].append(dict(repeat=repeat, blocks=len(check["blocks"]), passed=check["passed"],
                failed_blocks=[r["block_id"] for r in check["blocks"] if not r["passed"]], summary_sha256=digest(out / "summary.json")))
            atomic_json(a.out / "status.json", report)
        cross = compare_backend_processes(a.out / "backend_full_r0", a.out / "backend_full_r1")
        atomic_json(a.out / "backend_cross_process.json", cross)
        report["backend_cross_process"] = dict(passed=cross["passed"], blocks=len(cross["blocks"]),
            failed_blocks=[r["block_id"] for r in cross["blocks"] if not r["passed"]])
        atomic_json(a.out / "status.json", report)
        for job in evaluation_jobs(plan, sessions):
            if cancelled[0] or time.monotonic() >= cutoff:
                report["status"] = "cancelled" if cancelled[0] else "time_budget_reached"
                break
            disk = check_disk(a.out, plan)
            if disk["reasons"]:
                report.update(status="disk_limit_reached", disk_check=disk)
                break
            assert_current(a, frozen)
            run = Path(job["run"])
            ck = run / "ckpt_001000.pt"
            fingerprint = digest(ck)
            if fingerprint != checkpoint_hashes[str(ck)]:
                raise ValueError("Source checkpoint changed after initial verification")
            output = a.out / (job["id"]+".json")
            command = [sys.executable, "-u", str(ROOT / "tools/ral_online_eval_worker.py"), "--run_dir", str(run),
                "--ckpt", str(ck), "--motion_dir", str(a.motions), "--parameter_manifest", str(a.params),
                "--plant", job["plant"],
                "--seed", str(job["eval_seed"]), "--out", str(output)]
            atomic_json(output.with_suffix(".inputs.json"), dict(job=job, checkpoint_sha256=fingerprint,
                inputs=frozen, command=command, checkpoint_weights="EMA", formal_results=False))
            report["current_job"] = job["id"]
            atomic_json(a.out / "status.json", report)
            print(f"[3090] {job['id']}；剩余 {(cutoff-time.monotonic())/3600:.2f} 小时", flush=True)
            outcome = execute(command, output.with_suffix(".console.log"),
                min(cutoff, time.monotonic()+60*plan["job_timeout_minutes"]), cancelled, fd)
            if outcome != "completed":
                report.update(status=outcome, incomplete_job=job)
                break
            if digest(ck) != fingerprint:
                raise ValueError("Checkpoint changed during evaluation")
            result = summarize_randomized(read(output), frozen["base"]["motions"], job["eval_seed"], job["plant"])
            scenario = read(output.with_suffix(".scenario.json"))
            if (scenario["evaluation_seed"], scenario["plant"], scenario["clips"]) != (job["eval_seed"], job["plant"], 163):
                raise ValueError("Recorded scenario differs from the assigned evaluation")
            if any(r["scenario_sha256"] != scenario["state_sha256"] for r in rows if r["cell"] == job["cell"]):
                raise ValueError("Common randomized scenario check failed; do not compare unmatched evaluations")
            rows.append(dict(**job, **result, checkpoint_sha256=fingerprint, scenario_sha256=scenario["state_sha256"]))
            report["complete_cells"] = completed_cells(rows)
            atomic_json(a.out / "status.json", report)
        else:
            report["status"] = "planned_queue_completed"
        assert_current(a, frozen)
    except BaseException as exc:
        report.update(status="stopped", reason=str(exc))
        raise
    finally:
        report.update(current_job=None, complete_cells=completed_cells(rows))
        atomic_json(a.out / "status.json", report)
        review_package(a.out)


def judge(path):
    state = read(path / "status.json")
    if state["machine"] == "3090":
        cells = completed_cells(state["rows"])
        contrasts = []
        for condition in sorted({c["condition"] for c in cells}):
            subset = [c for c in cells if c["condition"] == condition]
            for seed in (2, 4):
                values = [next(p for p in c["pairs"] if p["training_seed"] == seed)["B_minus_Aplus_completion_pp"] for c in subset]
                contrasts.append(dict(condition=condition, training_seed=seed,
                    complete_evaluation_seeds=len(values), mean_B_minus_Aplus_completion_pp=statistics.mean(values),
                    difference_range_pp=[min(values), max(values)]))
        print(json.dumps(dict(status=state["status"], completed_evaluations=len(state["rows"]),
            complete_cells=len(cells), backend=state["backend"], training_seeds=[2,4],
            backend_cross_process=state.get("backend_cross_process"),
            descriptive_contrasts=contrasts, note="Evaluation repeats are not new training seeds; no formal superiority claim",
            formal_results=False, C_authorized=False), indent=2, ensure_ascii=False))
        return
    verified = 0
    for pair in state["pairs"]:
        retention = pair.get("retention")
        if pair["status"] == "completed" and not retention:
            raise ValueError("Completed pair is missing its verified model archive")
        if retention and digest(path / retention["archive"]) != retention["archive_sha256"]:
            raise ValueError("Model archive checksum changed")
        if pair["status"] == "completed":
            verified += 1
    print(json.dumps(dict(status=state["status"], complete_pairs=verified,
        seeds=[p["seed"] for p in state["pairs"] if p["status"] == "completed"],
        passing_pairs=sum(p.get("gate", {}).get("passed", False) for p in state["pairs"]),
        results=[pair_result(p) for p in state["pairs"] if p["status"] == "completed"],
        formal_results=False, C_authorized=False), indent=2))


def pair_result(pair):
    """Keep one primary result per training seed; repeats remain repeat checks."""
    result = dict(seed=pair["seed"], primary={}, repeat_ranges={}, prediction_diagnostics={})
    for group in ("A+", "B"):
        rows = [r for r in pair["rows"] if r["group"] == group]
        if len(rows) != 2 or {r["repeat_check"] for r in rows} != {0, 1}:
            raise ValueError("Completed pair has missing/duplicate final evaluations")
        primary = next(r for r in rows if r["repeat_check"] == 0)
        result["primary"][group] = {k: primary[k] for k in ("completed", "clips", "completion", "median_valid_s")}
        result["repeat_ranges"][group] = [min(r["completion"] for r in rows), max(r["completion"] for r in rows)]
    result["primary_B_minus_Aplus_completion_pp"] = 100*(result["primary"]["B"]["completion"]-result["primary"]["A+"]["completion"])
    jobs = {j["group"]: j for j in pair["jobs"]}
    for split in ("fit", "cal"):
        a, b = [jobs[g]["predictions"]["splits"][split] for g in ("A+", "B")]
        if not a["valid"] or not b["valid"]:
            result["prediction_diagnostics"][split] = dict(valid=False, reason="Nonfinite predictions retained")
            continue
        gain = lambda x, y: 100*(x-y)/x if x > 0 else None
        result["prediction_diagnostics"][split] = dict(valid=True,
            q32_B_reduction_percent=gain(a["q_rmse_rad"]["32"], b["q_rmse_rad"]["32"]),
            response8_B_reduction_percent=gain(a["response_rmse"]["8"], b["response_rmse"]["8"]),
            response8_B_reduction_vs_zero_percent=gain(b["zero_response_rmse"]["8"], b["response_rmse"]["8"]),
            interpretation="CPU pair diagnostics; response backend validity is unresolved")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "judge", "freeze"))
    p.add_argument("--machine", choices=("5080", "4090", "3090"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sessions", nargs="*", type=Path, default=[])
    p.add_argument("--imports", type=Path, default=ROOT / "logs/ral_structure/import")
    p.add_argument("--motions", type=Path, default=ROOT / "data/motions/z01_lafan_v2_clean")
    p.add_argument("--params", type=Path, default=ROOT / "release/ral/params.csv")
    p.add_argument("--responses", type=Path, default=ROOT / "data/ral_pairs_v1/manifest.json")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    os.chdir(ROOT)
    for name in ("out", "imports", "motions", "params", "responses"):
        setattr(a, name, getattr(a, name).resolve())
    if a.mode == "judge":
        judge(a.out)
        return 0
    plan = read(CONFIG)
    validate(plan)
    if a.mode == "run" and not a.machine:
        p.error("--machine required")
    if a.dry_run:
        print(json.dumps(dict(machine=a.machine, hours=8,
            first_pairs=list(pairs_for(plan, a.machine))[:3] if a.machine in ("5080", "4090") else [],
            training=plan["training"] if a.machine != "3090" else None,
            evaluator=plan["evaluator_3090"] if a.machine == "3090" else None,
            execution=False), indent=2))
        return 0
    if a.out.exists() or (a.out.parent / (a.out.name+"_review.tar.gz")).exists():
        p.error("Fresh output required; existing results are never overwritten")
    if a.mode == "freeze":
        freeze(a, plan, a.out)
        print(f"Frozen candidate: {a.out / 'protocol.json'}")
        return 0
    import torch
    if not torch.cuda.is_available() or a.machine not in torch.cuda.get_device_name():
        p.error(f"Expected RTX {a.machine}; use the matching machine script")
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock = ROOT / "logs/ral_8h" / (".gpu_"+hashlib.sha256(visible.encode()).hexdigest()[:12]+".lock")
    with gpu_lock(lock) as fd:
        cutoff = time.monotonic()+8*3600-plan["shutdown_reserve_seconds"]
        try:
            (evaluation_queue if a.machine == "3090" else training_queue)(a, cutoff, cancelled, fd)
        except BaseException as exc:
            if not (a.out / "status.json").exists():
                a.out.mkdir(parents=True, exist_ok=True)
                atomic_json(a.out / "status.json", dict(status="preflight_failed", machine=a.machine,
                    reason=str(exc), pairs=[], rows=[], backend=[], formal_results=False, C_authorized=False))
                review_package(a.out)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
