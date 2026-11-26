"""Bounded online A+/B development: 5080 seed 2, 4090 seed 4, 3090 review.

Fresh attempts only. Never reads locked response-test arrays or starts C automatically.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tarfile
import time
from types import SimpleNamespace

from ral_eight_hour import ROOT, atomic_json, check_disk, digest, execute, gpu_lock, read
from ral_recovery import check_learning, check_run, current_inputs, evaluate_checkpoint, recovery_gate

CONFIG = ROOT / "configs/ral/structure_online_v1.json"
PROTOCOL = ROOT / "release/ral/structure_protocol_v1.json"


def jobs_for(config, machine):
    if (config["groups"] != ["A+", "B"] or config["training_seeds"] != [2, 4]
            or config["assignments"] != {"5080": 2, "4090": 4}):
        raise ValueError("Structure stage requires the complete seed 2/4 A+/B pairs")
    t = config["training"]
    required = dict(wm_fit_source="online", paired_initialization=True, wm_window=8,
                    horizon=32, response_weight=0., wm_amp="none", iterations=1000,
                    wm_batches=5, num_envs=1024, warmup_steps=64, utd=.4, ckpt_every=250)
    if any(t.get(k) != v for k, v in required.items()):
        raise ValueError("Frozen structure-stage training settings differ")
    if machine not in config["assignments"]:
        raise ValueError("Training machine must be 5080 or 4090")
    seed = config["assignments"][machine]
    return [dict(id=f"s{seed}_{g.replace('+', 'plus')}", seed=seed, group=g) for g in config["groups"]]


def tools_inventory():
    names = ["tools/ral_structure.py", "tools/ral_structure_metrics.py", "tools/ral_backend_check.py",
             "tools/ral_recovery.py", "tools/ral_eight_hour.py", "tools/ral_eight_hour_worker.py",
             "tools/ral_policy_diagnose.py", "uv.lock", "pyproject.toml",
             "configs/ral/backend_check_v1.json"]
    return {name: digest(ROOT / name) for name in names}


def shared_inputs(a):
    return dict(base=current_inputs(a.motions, a.params, a.responses), tools=tools_inventory())


def make_protocol(a, config, folder):
    from partymimic.experiments.protocol import freeze
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "config.json", config)
    p = freeze(folder / "config.json", a.params, a.motions, a.responses, folder / "protocol.json")
    p.update(groups=config["groups"], training_seeds=config["training_seeds"],
             stage="online_Aplus_B", response_labels_consumed=False,
             screening=config["screening"], evaluation=config["evaluation"],
             tools=tools_inventory(), expected_environment_transitions=config["expected_environment_transitions"])
    atomic_json(folder / "protocol.json", p)
    return p


def stage_gate(rows, config, repeats=2, seeds=None):
    # Reuse the independently tested conservative calculation with explicit relabeling.
    translated = [dict(r, group={"A+": "A", "B": "A+"}[r["group"]]) for r in rows]
    cfg = dict(config, training_seeds=seeds or config["training_seeds"])
    result = recovery_gate(translated, cfg, repeats)
    result["reasons"] = [s.replace("A+", "candidate B").replace("baseline A", "control A+") for s in result["reasons"]]
    result.update(meaning="Control-retention screen only; not superiority, equivalence or a paper conclusion.",
                  next_action="Review predictions and response backend validity before designing C" if result["passed"]
                  else "Diagnose failed control/structure/repeatability checks; do not extend automatically",
                  C_authorized=False)
    return result


def module_hashes(path):
    import torch
    from ral_eight_hour_worker import finite_weights, state_digest
    ck = torch.load(path, map_location="cpu", weights_only=False)
    finite_weights(ck, ("wm", "pi", "critic"))
    return {name: state_digest(ck, (name,)) for name in ("wm", "pi", "critic")}


def verify_run(run, protocol, job):
    ledger, index = check_run(run, protocol)
    if ledger["protocol_sha256"] != digest(run.parents[2] / "protocol.json"):
        raise ValueError("Run protocol fingerprint differs")
    if ledger["group"] != job["group"] or ledger["training_seed"] != job["seed"]:
        raise ValueError("Run group/seed differs from assigned job")
    if ledger["env_steps"] != protocol["expected_environment_transitions"]:
        raise ValueError("Environment-transition budget incomplete")
    from partymimic.experiments.initialization import stream_seeds
    if ledger.get("random_streams", {}).get("seeds") != stream_seeds(job["seed"]):
        raise ValueError("Paired random-stream provenance missing")
    final_iteration = protocol["training"]["iterations"]
    every = protocol["training"]["ckpt_every"]
    names = ["initial.pt"] + [f"ckpt_{i:06d}.pt" for i in sorted({every, 2*every, final_iteration})]
    for name in names:
        path = run / name
        if name not in index or not path.is_file() or digest(path) != index[name]["sha256"]:
            raise ValueError(f"Missing or changed checkpoint: {path}")
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        from ral_eight_hour_worker import finite_weights
        finite_weights(ck, ("wm", "pi", "critic") if name == "initial.pt" else ("wm", "pi", "pi_live", "critic"))
        if ck["iteration"] != int(index[name]["iteration"]) or ck["sim_steps"] != int(index[name]["env_steps"]):
            raise ValueError("Checkpoint internal counters differ from its index")
        expected_iteration = 0 if name == "initial.pt" else int(Path(name).stem[5:])
        t = protocol["training"]
        expected_steps = 0 if expected_iteration == 0 else (
            t["warmup_steps"] + int(expected_iteration / t["utd"])) * t["num_envs"]
        if ck["iteration"] != expected_iteration or ck["sim_steps"] != expected_steps:
            raise ValueError("Checkpoint filename/budget does not match its contents")
        if ck["args"] != ledger["args"]:
            raise ValueError("Checkpoint arguments differ from run ledger")
    actual = module_hashes(run / "initial.pt")
    if actual != ledger["initial_module_sha256"]:
        raise ValueError("Initial tensor fingerprints differ from ledger")
    # The transfer intentionally omits last.pt and optimizer state; use the final snapshot.
    import torch
    final = torch.load(run / f"ckpt_{final_iteration:06d}.pt", map_location="cpu", weights_only=False)
    initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=False)
    for name in ("wm", "pi", "critic"):
        if not any(not torch.equal(v, final[name][k]) for k, v in initial[name].items() if "weight" in k):
            raise ValueError(f"No learned weight changed in {name}")
    import csv
    with (run / "train_metrics.csv").open() as f:
        metrics = list(csv.DictReader(f))
    if [int(r["iteration"]) for r in metrics] != list(range(final_iteration+1)):
        raise ValueError("Missing or duplicate per-iteration training records")
    for r in metrics[1:]:
        if any(not math.isfinite(float(r[k])) for k in ("wm_loss", "policy_loss")):
            raise ValueError("Nonfinite per-iteration loss")
    with (run / "train_log.csv").open() as f:
        for r in csv.DictReader(f):
            if float(r["nonfinite"]) != 0 or any(not math.isfinite(float(r[k])) for k in ("wm_loss", "pi_loss", "v_loss")):
                raise ValueError("Nonfinite or skipped training update")
    return ledger


def verify_pair(session, protocol, jobs):
    records = [verify_run(session / "jobs" / j["id"] / "run", protocol, j) for j in jobs]
    for name in ("pi", "critic"):
        if records[0]["initial_module_sha256"][name] != records[1]["initial_module_sha256"][name]:
            raise ValueError(f"A+/B initial {name} mismatch")
    return dict(passed=True, initial_module_sha256=[r["initial_module_sha256"] for r in records],
                identical_actor_and_critic=True, identical_training_labels=False)


def bundle(session):
    path = session.parent / (session.name + "_transfer.tar.gz")
    keep = {"initial.pt", "ckpt_000250.pt", "ckpt_000500.pt", "ckpt_001000.pt"}
    with tarfile.open(path, "x:gz", compresslevel=1) as tar:
        for p in sorted(session.rglob("*")):
            if p.is_file() and p.suffix != ".tmp" and (p.suffix != ".pt" or p.name in keep):
                tar.add(p, arcname=str(Path(session.name) / p.relative_to(session)), recursive=False)
    print(f"回传包：{path} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)


def train(a, cutoff, cancelled, lock_fd):
    from partymimic.experiments.cli import training_command
    config = read(CONFIG)
    jobs = jobs_for(config, a.machine)
    a.out.mkdir(parents=True, exist_ok=False)
    status = dict(status="preparing", machine=a.machine, jobs=jobs, formal_results=False)
    rows = []
    try:
        protocol = make_protocol(a, config, a.out)
        if protocol != read(PROTOCOL):
            raise ValueError("Code/data/config do not match release/ral/structure_protocol_v1.json; synchronize the code bundle")
        frozen = shared_inputs(a)
        atomic_json(a.out / "inputs.json", frozen)
        for job in jobs:
            disk = check_disk(a.out, dict(min_free_gib=15, max_output_gib=20))
            print(disk["message"], flush=True)
            if disk["reasons"]:
                raise RuntimeError(disk["message"])
            if cancelled[0] or time.monotonic() >= cutoff:
                raise RuntimeError("Deadline/cancellation reached")
            if shared_inputs(a) != frozen:
                raise ValueError("Inputs changed during training queue")
            run = a.out / "jobs" / job["id"] / "run"
            run.parent.mkdir(parents=True)
            job["status"] = "running"
            status.update(status="running", current_job=job["id"])
            atomic_json(a.out / "status.json", status)
            command = training_command(SimpleNamespace(config=a.out / "config.json", params=a.params,
                motion_dir=a.motions, group=job["group"], responses=a.responses, seed=job["seed"],
                protocol=a.out / "protocol.json", out=run))
            if job["group"] == "B":
                command += ["--paired_initial_reference", str(a.out / "jobs" / jobs[0]["id"] / "run/initial.pt")]
            atomic_json(run.parent / "command.json", command)
            print(f"[{a.machine}] {job['id']}：训练世界模型、策略和价值网络", flush=True)
            outcome = execute(command, run.parent / "console.log", min(cutoff, time.monotonic()+3600), cancelled, lock_fd)
            if outcome != "completed":
                job["status"] = outcome
                raise RuntimeError(f"Training {outcome}; inspect {run.parent / 'console.log'}")
            verify_run(run, protocol, job)
            check_learning(run)
            quick = evaluate_checkpoint(run, run / "ckpt_001000.pt", run.parent / "quick_eval.json",
                                         a, frozen["base"], cutoff, cancelled, lock_fd)
            rows.append(dict(**quick, group=job["group"], seed=job["seed"], final=True, repeat_check=0))
            job.update(status="completed", quick_eval=quick)
            atomic_json(a.out / "status.json", status)
            print(f"[{a.machine}] {job['id']}：完成 {quick['completed']}/{quick['clips']} 条动作", flush=True)
            if job["group"] == "A+" and quick["completion"] < config["screening"]["baseline_completion_min"]:
                raise RuntimeError("A+ control below 70%; stopped before B. Preserve results for diagnosis.")
        atomic_json(a.out / "paired_initialization.json", verify_pair(a.out, protocol, jobs))
        if shared_inputs(a) != frozen:
            raise ValueError("Inputs changed during training")
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="stopped", reason=str(exc))
        for job in jobs:
            if job.get("status") == "running":
                job.update(status="failed_validation", reason=str(exc))
        raise
    finally:
        status["current_job"] = None
        atomic_json(a.out / "status.json", status)
        atomic_json(a.out / "quick_judgment.json", dict(rows=rows,
            gate=stage_gate(rows, config, repeats=1, seeds=[jobs[0]["seed"]]), independent_3090_review_required=True))
        bundle(a.out)


def verified_sessions(a):
    config = read(CONFIG)
    frozen = shared_inputs(a)
    sessions = {}
    for session in a.sessions:
        status, protocol = read(session / "status.json"), read(session / "protocol.json")
        machine = status["machine"]
        jobs = jobs_for(config, machine)
        if machine in sessions or status["status"] != "completed":
            raise ValueError("Require one completed pair per machine, without duplicate sessions")
        if read(session / "config.json") != config or protocol != read(PROTOCOL):
            raise ValueError("Transferred configuration/protocol differs from frozen release")
        if digest(session / "config.json") != protocol["config_sha256"] or read(session / "inputs.json") != frozen:
            raise ValueError("Transferred code/data/config fingerprints differ")
        if [j["id"] for j in status["jobs"]] != [j["id"] for j in jobs] or any(j["status"] != "completed" for j in status["jobs"]):
            raise ValueError("Incomplete or unexpected jobs in transferred session")
        verify_pair(session, protocol, jobs)
        sessions[machine] = (session, jobs)
    if set(sessions) != {"5080", "4090"}:
        raise ValueError("Need both 5080 seed 2 and 4090 seed 4 transfer directories")
    return config, frozen, sessions


def evaluate(a, cutoff, cancelled, lock_fd):
    config, frozen, sessions = verified_sessions(a)
    a.out.mkdir(parents=True, exist_ok=False)
    atomic_json(a.out / "source.json", dict(inputs=frozen, config=config,
        sessions={m: str(s[0]) for m, s in sessions.items()}, protocol_sha256=digest(PROTOCOL)))
    rows = []
    predictions = {}
    report = dict(status="running", rows=rows, predictions=predictions, formal_results=False, response_test_accessed=False)
    try:
        for machine in ("5080", "4090"):
            session, jobs = sessions[machine]
            for job in jobs:
                run = session / "jobs" / job["id"] / "run"
                for repeat in range(2):
                    name = f"{job['id']}_i001000_r{repeat}"
                    print(f"[3090] 评估 {name}", flush=True)
                    disk = check_disk(a.out, dict(min_free_gib=15, max_output_gib=20))
                    if disk["reasons"]:
                        raise RuntimeError(disk["message"])
                    result = evaluate_checkpoint(run, run / "ckpt_001000.pt", a.out / f"{name}.json",
                        a, frozen["base"], cutoff, cancelled, lock_fd, record=repeat == 0 and job["group"] == "B")
                    rows.append(dict(**result, group=job["group"], seed=job["seed"], final=True,
                                     repeat_check=repeat, checkpoint_sha256=digest(run / "ckpt_001000.pt")))
                    atomic_json(a.out / "summary.json", report)
                command = [sys.executable, "-u", str(ROOT / "tools/ral_structure_metrics.py"),
                    "--checkpoint", str(run / "ckpt_001000.pt"), "--params", str(a.params),
                    "--responses", str(a.responses), "--out", str(a.out / f"{job['id']}_predictions.json")]
                outcome = execute(command, a.out / f"{job['id']}_predictions.log", cutoff, cancelled, lock_fd)
                if outcome != "completed":
                    raise RuntimeError(f"Development predictions {outcome}")
                predictions[job["id"]] = read(a.out / f"{job['id']}_predictions.json")
        if shared_inputs(a) != frozen:
            raise ValueError("Review inputs changed during execution")
        report.update(status="completed", control_gate=stage_gate(rows, config),
            prediction_finite=all(s["valid"] for p in predictions.values() for s in p["splits"].values()),
            response_metrics="CPU fit/cal diagnostics only until backend validation; no automatic C")
    except BaseException as exc:
        report.update(status="stopped", reason=str(exc))
        raise
    finally:
        atomic_json(a.out / "summary.json", report)
        archive = a.out.parent / (a.out.name + "_review.tar.gz")
        with tarfile.open(archive, "x:gz", compresslevel=1) as tar:
            tar.add(a.out, arcname=a.out.name)
        print(f"复核包：{archive}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("train", "evaluate", "judge", "freeze"))
    ap.add_argument("--machine", choices=("5080", "4090"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sessions", nargs="+", type=Path, default=[])
    ap.add_argument("--motions", type=Path, default=ROOT / "data/motions/z01_lafan_v2_clean")
    ap.add_argument("--params", type=Path, default=ROOT / "release/ral/params.csv")
    ap.add_argument("--responses", type=Path, default=ROOT / "data/ral_pairs_v1/manifest.json")
    ap.add_argument("--hours", type=float, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    if not math.isfinite(a.hours) or not .02 <= a.hours <= 8:
        ap.error("--hours must be .02 to 8; this is a ceiling, not a target")
    for name in ("out", "motions", "params", "responses"):
        if getattr(a, name) is not None:
            setattr(a, name, getattr(a, name).resolve())
    a.sessions = [p.resolve() for p in a.sessions]
    if a.mode == "train" and not a.machine:
        ap.error("train requires --machine")
    if a.mode in ("train", "evaluate", "freeze") and a.out is None:
        ap.error("--out is required (new directory)")
    if a.mode in ("evaluate", "judge") and len(a.sessions) != 2:
        ap.error("--sessions needs the extracted 5080 and 4090 directories")
    if a.dry_run:
        print(json.dumps(dict(mode=a.mode, jobs=[j for m in ([a.machine] if a.machine else ["5080", "4090"])
            for j in jobs_for(read(CONFIG), m)], out=str(a.out), execution=False), indent=2))
        return 0
    if a.mode == "judge":
        verified_sessions(a)
        print("四个模型的输入、预算、权重和配对初始化检查通过；闭环结果须另行评估。")
        return 0
    if a.out.exists() or any(a.out.parent.glob(a.out.name + "_*.tar.gz")):
        ap.error("Output/archive exists; use a fresh attempt name")
    if a.mode == "freeze":
        make_protocol(a, read(CONFIG), a.out)
        print(f"Frozen candidate protocol: {a.out / 'protocol.json'}")
        return 0
    import torch
    expected = a.machine if a.mode == "train" else "3090"
    if not torch.cuda.is_available() or expected not in torch.cuda.get_device_name():
        ap.error(f"This command requires an RTX {expected}")
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock = ROOT / "logs/ral_8h" / (".gpu_" + hashlib.sha256(visible.encode()).hexdigest()[:12] + ".lock")
    with gpu_lock(lock) as fd:
        (train if a.mode == "train" else evaluate)(a, time.monotonic()+a.hours*3600-20, cancelled, fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
