"""Bounded sequential experiment queues and read-only result judgment.

The wall deadline includes preparation, compilation, training, audit and judgment.
No trainer warm-resume, locked test evaluation or automatic window acceptance.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools/ral_eight_hour_worker.py"
TOKEN = re.compile(r"[A-Za-z0-9_-]+\Z")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    # Manual judgment can run concurrently with the queue's automatic judgment.
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(payload)
            f.flush()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def validate_plan(plan):
    if plan["schema_version"] != 1:
        raise ValueError("Unsupported plan schema")
    for key in ("hours", "shutdown_reserve_seconds", "job_timeout_minutes", "min_free_gib", "max_output_gib"):
        if not math.isfinite(plan[key]) or plan[key] <= 0:
            raise ValueError(f"Positive finite {key} required")
    for key in ("max_rounds", "max_consecutive_failures"):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(key)
    if plan["shutdown_reserve_seconds"] < 20:
        raise ValueError("At least 20 seconds required for bounded shutdown")
    names = set()
    for c in plan["candidates"]:
        if not TOKEN.fullmatch(c["name"]) or c["name"] in names or not c["name"].startswith("C_"):
            raise ValueError("Unique safe C_ candidate names required")
        names.add(c["name"])
        if (not math.isfinite(c["response_weight"]) or c["response_weight"] <= 0
                or type(c["response_horizon"]) is not int or not 1 <= c["response_horizon"] <= 32
                or c["response_loss"] not in ("mse", "huber")
                or not math.isfinite(c["response_huber_delta"]) or c["response_huber_delta"] <= 0):
            raise ValueError("Invalid response candidate")
    if not names:
        raise ValueError("At least one response candidate required")
    for machine, m in plan["machines"].items():
        if machine not in ("3090", "4090", "5080") or m["mode"] not in ("policy", "world_model"):
            raise ValueError("Unsupported machine/mode")
        if any(type(m[k]) is not int or m[k] < 1 for k in ("seed_start", "seed_stride")):
            raise ValueError("Positive integer seeds required")
        if m["mode"] == "policy" and (set(m["baseline_order"]) - {"A0", "A", "A+", "B"}
                                       or m["baseline_order"].count("B") != 1
                                       or len(set(m["baseline_order"])) != len(m["baseline_order"])):
            raise ValueError("Policy baseline order must include B exactly once")
    for section, keys in (("policy", ("iterations", "ckpt_every")),
                          ("world_model", ("updates", "checkpoint_every", "batch_size"))):
        for key in keys:
            if type(plan[section][key]) is not int or plan[section][key] < 1:
                raise ValueError(f"Invalid {section}/{key}")
    if set(plan["policy"]) & {"resume", "init", "protocol", "log_dir", "experiment_group", "parameter_manifest", "response_manifest", "motion_dir", "seed", "synthetic_test", "max_hours"}:
        raise ValueError("Plan cannot override run identity, data or wall-clock controls")
    rule = plan["judgment"]
    for k in ("forward_ratio_max", "q_rmse_ratio_max", "response8_ratio_max", "zero_response8_ratio_max", "response32_ratio_max"):
        if not math.isfinite(rule[k]) or rule[k] <= 0:
            raise ValueError(k)
    if not 0 < rule["session_win_fraction_min"] <= 1 or rule["minimum_confirming_seeds"] < 2:
        raise ValueError("Invalid confirmation rule")


def jobs_for(plan, machine):
    m = plan["machines"][machine]
    if m.get("amplitude_blocks", 0):
        yield dict(id="amplitude_fit", kind="amplitude", blocks=m["amplitude_blocks"])
    for round_id in range(plan["max_rounds"]):
        seed = m["seed_start"] + round_id * m["seed_stride"]
        baseline = m.get("baseline_order", ["B"])
        # Counterbalance candidate order across successive independent seeds.
        candidates = plan["candidates"] if round_id % 2 == 0 else list(reversed(plan["candidates"]))
        for name in baseline + [c["name"] for c in candidates]:
            group = "C" if name.startswith("C_") else name
            candidate = next((c for c in candidates if c["name"] == name), None)
            yield dict(id=f"s{seed}_{name.replace('+', 'plus')}", kind=m["mode"],
                       group=group, candidate=name, seed=seed, round=round_id,
                       response=candidate or dict(response_weight=0.0001, response_horizon=8,
                                                   response_loss="mse", response_huber_delta=1.0))


def compare_pair(baseline, candidate, rule):
    """Only equal-budget, identical-input, equal-initialization comparisons qualify."""
    reasons = []
    for key in ("kind", "seed", "budget", "init_sha256", "comparison_sha256", "sampling_sha256"):
        if baseline.get(key) != candidate.get(key):
            reasons.append(f"unmatched_{key}")
    if not baseline.get("complete") or not candidate.get("complete"):
        reasons.append("incomplete_run")
    if reasons:
        return dict(passed=False, reasons=reasons)
    b, c = baseline["cal"], candidate["cal"]
    checks = {
        "forward_degraded": (c["forward_mse"], b["forward_mse"] * rule["forward_ratio_max"]),
        "joint_prediction_degraded": (c["q_rmse32_rad"], b["q_rmse32_rad"] * rule["q_rmse_ratio_max"]),
        "response8_gain_insufficient": (c["response_rmse8"], b["response_rmse8"] * rule["response8_ratio_max"]),
        "zero_response_control_not_beaten": (c["response_rmse8"], c["zero_response_rmse8"] * rule["zero_response8_ratio_max"]),
        "response32_degraded": (c["response_rmse32"], b["response_rmse32"] * rule["response32_ratio_max"]),
    }
    for reason, (value, limit) in checks.items():
        if not math.isfinite(value) or not math.isfinite(limit) or value > limit:
            reasons.append(reason)
    bs, cs = b["sessions"], c["sessions"]
    if not bs or bs.keys() != cs.keys():
        reasons.append("session_set_mismatch")
        wins = 0.
    else:
        wins = sum(cs[s]["response_rmse8"] < bs[s]["response_rmse8"] for s in bs) / len(bs)
        if wins < rule["session_win_fraction_min"]:
            reasons.append("session_gain_insufficient")
    return dict(passed=not reasons, reasons=reasons, session_win_fraction=wins,
                response8_change_percent=100 * (c["response_rmse8"] / max(b["response_rmse8"], 1e-30) - 1),
                B_cal=b, C_cal=c)


def judge_sessions(paths, verify=True):
    pairs, runs, sessions, excluded, initializations = [], [], [], [], []
    for session in map(Path, paths):
        state, plan = read(session / "status.json"), read(session / "plan.json")
        validate_plan(plan)
        sessions.append(dict(path=str(session), machine=state["machine"], status=state["status"],
                             deadline_unix=state["deadline_unix"], current_job=state.get("current_job")))
        local = []
        for job in state["jobs"]:
            if job.get("status") != "completed" or job["kind"] == "amplitude":
                excluded.append(dict(session=str(session), job=job["id"], status=job.get("status")))
                continue
            path = session / "jobs" / job["id"] / "analysis.json"
            try:
                if digest(path) != job["analysis_sha256"]:
                    raise ValueError("Analysis hash mismatch")
                row = read(path)
                checkpoint = path.parent / "run" / "last.pt"
                if not checkpoint.is_file() or (verify and digest(checkpoint) != row["checkpoint_sha256"]):
                    raise ValueError("Checkpoint missing/changed")
                if not row["complete"]:
                    raise ValueError("Incomplete analysis")
                row.update(session=str(session), job=job["id"])
                local.append(row)
                runs.append(row)
            except (OSError, ValueError, KeyError) as exc:
                excluded.append(dict(session=str(session), job=job["id"], status="invalid", reason=str(exc)))
        baselines = {(x["kind"], x["seed"]): x for x in local if x["group"] == "B"}
        for seed in sorted({x["seed"] for x in local if x["kind"] == "policy"}):
            by_group = {x["group"]: x for x in local if x["kind"] == "policy" and x["seed"] == seed and x["group"] != "C"}
            for left, right in (("A0", "A"), ("A+", "B")):
                if left in by_group and right in by_group:
                    equal = all(by_group[left][k] == by_group[right][k] for k in ("init_sha256", "budget", "comparison_sha256"))
                    initializations.append(dict(session=str(session), seed=seed, groups=[left, right], matched=equal))
        for c in (x for x in local if x["group"] == "C"):
            b = baselines.get((c["kind"], c["seed"]))
            result = compare_pair(b, c, plan["judgment"]) if b else dict(passed=False, reasons=["missing_completed_B"])
            pairs.append(dict(session=str(session), kind=c["kind"], seed=c["seed"], candidate=c["candidate"],
                              comparison_sha256=c["comparison_sha256"], plan_sha256=digest(session / "plan.json"),
                              minimum_confirming_seeds=plan["judgment"]["minimum_confirming_seeds"], **result))
    # Never pool policy-trained and model-only runs, or different plans/input hashes.
    buckets = {}
    for p in pairs:
        key = (p["kind"], p["candidate"], p["comparison_sha256"], p["plan_sha256"])
        buckets.setdefault(key, []).append(p)
    decisions = []
    for key, values in buckets.items():
        seeds = sorted({p["seed"] for p in values})
        passing = sorted({p["seed"] for p in values if p["passed"]})
        accepted = len(seeds) >= values[0]["minimum_confirming_seeds"] and all(p["passed"] for p in values)
        decisions.append(dict(kind=key[0], candidate=key[1], seeds=seeds, passing_seeds=passing,
                              comparison_sha256=key[2],
                              decision="candidate_for_review" if accepted else "no_confirmed_gain",
                              automatic_promotion=False))
    return dict(schema_version=1, created_unix=time.time(), formal_results=False,
                test_outcomes_inspected=False, closed_loop_evaluation_performed=False,
                sessions=sessions, completed_training_runs=len(runs), excluded=excluded,
                paired_initialization_checks=initializations, paired_comparisons=pairs, decisions=decisions, windows_authorized=False,
                next_action="Continue remaining preregistered candidates/seeds within the wall budget; no automatic longer training or D/C-perm. Review pooled same-plan seeds after collection.")


@contextmanager
def gpu_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another eight-hour queue is using this visible GPU") from None
        yield f.fileno()


def stop_process(process):
    """Bound shutdown of the whole job process group, including trainer children."""
    for sig, seconds in ((signal.SIGINT, 8), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=seconds)
            # A parent exiting must not leave a child in this process group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        except subprocess.TimeoutExpired:
            pass


def execute(command, log, cutoff, cancelled, lock_fd, heartbeat=None):
    with Path(log).open("ab") as f:
        process = subprocess.Popen(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock_fd,),
                                   env=dict(os.environ, PYTHONUNBUFFERED="1"))
        try:
            last_heartbeat = 0.
            while process.poll() is None:
                now = time.monotonic()
                if cancelled[0] or now >= cutoff:
                    stop_process(process)
                    return "cancelled" if cancelled[0] else "timed_out"
                if heartbeat and now - last_heartbeat >= 10:
                    heartbeat(process.pid)
                    last_heartbeat = now
                time.sleep(min(0.5, max(0.01, cutoff - now)))
            return "completed" if process.returncode == 0 else "failed"
        finally:
            if process.poll() is None:
                stop_process(process)


def output_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def run_queue(a):
    os.chdir(ROOT)
    plan = read(a.plan)
    validate_plan(plan)
    hours = plan["hours"] if a.hours is None else a.hours
    if not math.isfinite(hours) or not 0 < hours <= 8:
        raise ValueError("Hours must be positive and at most 8 per session")
    if hours * 3600 <= plan["shutdown_reserve_seconds"] + 5:
        raise ValueError("Duration is shorter than the shutdown reserve")
    name = a.session or time.strftime("night_%Y%m%d_%H%M%S")
    if not TOKEN.fullmatch(name):
        raise ValueError("Session name must contain only letters, numbers, _ and -")
    session = Path(a.output).resolve() / f"{name}_{a.machine}"
    queue = list(jobs_for(plan, a.machine))
    if a.dry_run:
        print(json.dumps(dict(session=str(session), hours=hours, total_available_jobs=len(queue),
                              first_jobs=queue[:12], plan=plan), indent=2, ensure_ascii=False))
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock = ROOT / "logs/ral_8h" / (".gpu_" + hashlib.sha256(visible.encode()).hexdigest()[:12] + ".lock")
    with gpu_lock(lock) as lock_fd:
        session.mkdir(parents=True, exist_ok=False)
        (session / "jobs").mkdir()
        atomic_json(session / "plan.json", plan)
        state = dict(schema_version=1, machine=a.machine, status="preparing", session=str(session),
                     started_unix=time.time(), deadline_unix=time.time() + hours * 3600,
                     jobs=[], current_job=None, formal_results=False, test_outcomes_inspected=False)
        deadline = time.monotonic() + hours * 3600
        cutoff = deadline - plan["shutdown_reserve_seconds"]
        cancelled = [False]
        old_handlers = {sig: signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
                        for sig in (signal.SIGINT, signal.SIGTERM)}
        def save(pid=None):
            state["updated_unix"] = time.time()
            state["active_pid"] = pid
            atomic_json(session / "status.json", state)
        def judgment():
            atomic_json(session / "judgment.json", judge_sessions([session], verify=False))
        save()
        try:
            prepare = [sys.executable, "-u", str(WORKER), "prepare", "--session", str(session),
                       "--machine", a.machine, "--params", str(Path(a.params).resolve()),
                       "--motions", str(Path(a.motions).resolve()), "--responses", str(Path(a.responses).resolve())]
            if a.allow_gpu_mismatch:
                prepare.append("--allow-gpu-mismatch")
            status = execute(prepare, session / "prepare.log", cutoff, cancelled, lock_fd, save)
            if status != "completed":
                state["status"] = "preflight_" + status
                return 2
            failures = 0
            state["status"] = "running"
            for spec in queue:
                if cancelled[0] or time.monotonic() >= cutoff:
                    state["status"] = "cancelled" if cancelled[0] else "time_budget_reached"
                    break
                if a.max_jobs is not None and len(state["jobs"]) >= a.max_jobs:
                    state["status"] = "job_limit_reached"
                    break
                if shutil.disk_usage(session).free < plan["min_free_gib"] * 2**30 or output_size(session) >= plan["max_output_gib"] * 2**30:
                    state["status"] = "disk_limit_reached"
                    break
                job = dict(spec, status="running", started_unix=time.time())
                state["jobs"].append(job)
                state["current_job"] = job["id"]
                job_dir = session / "jobs" / job["id"]
                job_dir.mkdir()
                atomic_json(job_dir / "job.json", spec)
                save()
                # A missing or failed B cannot support any C comparison for that seed.
                if job.get("group") == "C" and not any(x.get("seed") == job["seed"] and x.get("group") == "B" and x["status"] == "completed" for x in state["jobs"]):
                    job["status"] = "skipped_missing_B"
                    job["finished_unix"] = time.time()
                    save()
                    continue
                print(f"[{a.machine}] {job['id']} 开始；剩余 {(deadline-time.monotonic())/3600:.2f} 小时", flush=True)
                command = [sys.executable, "-u", str(WORKER), "job", "--session", str(session), "--job", job["id"]]
                # Limit hung jobs as well as the complete session. A timeout is never a completed run.
                job_cutoff = min(cutoff, time.monotonic() + plan["job_timeout_minutes"] * 60)
                job["status"] = execute(command, job_dir / "console.log", job_cutoff, cancelled, lock_fd, save)
                if job["status"] == "completed":
                    analysis = job_dir / "analysis.json"
                    if not analysis.is_file() or not read(analysis).get("complete"):
                        job["status"] = "failed_missing_analysis"
                    else:
                        job["analysis_sha256"] = digest(analysis)
                job["finished_unix"] = time.time()
                failures = 0 if job["status"] == "completed" else failures + 1
                save()
                judgment()
                print(f"[{a.machine}] {job['id']} {job['status']}；判断已更新", flush=True)
                if job["kind"] == "amplitude" and job["status"] != "completed":
                    state["status"] = "cancelled" if cancelled[0] else ("time_budget_reached" if time.monotonic() >= cutoff else "data_replay_failed")
                    break
                if failures >= plan["max_consecutive_failures"]:
                    state["status"] = "repeated_failures"
                    break
            else:
                state["status"] = "queue_exhausted"
        except BaseException as exc:
            state["status"] = "runner_failed"
            state["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            state["finished_unix"] = time.time()
            state["current_job"] = None
            save()
            judgment()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            print(f"记录目录：{session}\n结束状态：{state['status']}", flush=True)
    return 0 if state["status"] in ("time_budget_reached", "job_limit_reached", "queue_exhausted") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--machine", choices=("3090", "4090", "5080"), required=True)
    run.add_argument("--plan", default=str(ROOT / "configs/ral/eight_hour.json"))
    run.add_argument("--session")
    run.add_argument("--hours", type=float)
    run.add_argument("--output", default=str(ROOT / "logs/ral_8h"))
    run.add_argument("--params", default="release/ral/params.csv")
    run.add_argument("--motions", default="data/motions/z01_lafan_v2_clean")
    run.add_argument("--responses", default="data/ral_pairs_v1/manifest.json")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--max-jobs", type=int, help="Short validation only; default has no job-count limit")
    run.add_argument("--allow-gpu-mismatch", action="store_true", help="Explicit local validation on another GPU; recorded")
    judge = sub.add_parser("judge")
    judge.add_argument("--sessions", nargs="+", required=True)
    judge.add_argument("--machine", choices=("3090", "4090", "5080"))
    judge.add_argument("--out", help="New report path; default updates judgment.json for a single session")
    judge.add_argument("--require-gain", action="store_true", help="Exit 3 if no candidate has confirmed development gains")
    a = parser.parse_args()
    if a.command == "run":
        if a.max_jobs is not None and a.max_jobs < 1:
            parser.error("--max-jobs must be positive")
        return run_queue(a)
    sessions = [Path(p).resolve() for p in a.sessions]
    if a.machine and any(read(p / "status.json")["machine"] != a.machine for p in sessions):
        parser.error("Judgment wrapper does not match session machine")
    if len(sessions) > 1 and not a.out:
        parser.error("Supply --out for a combined report")
    result = judge_sessions(sessions)
    out = Path(a.out).resolve() if a.out else sessions[0] / "judgment.json"
    if a.out and out.exists():
        parser.error("Choose a new combined report path; refusing overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, result)
    for s in result["sessions"]:
        print(f"{s['machine']}: {s['status']}；当前任务 {s['current_job'] or '无'}")
    print(f"已核验 {result['completed_training_runs']} 次训练，{len(result['paired_comparisons'])} 个 B/C 配对，{len(result['excluded'])} 项未计入完整训练。")
    for d in result["decisions"]:
        print(f"{d['kind']} / {d['candidate']}: {d['decision']}；通过种子 {d['passing_seeds']}，全部种子 {d['seeds']}")
    print(f"判断记录：{out}\n这些是 fit/cal 开发诊断；未自动授权 D/C-perm 或正式测试。")
    return 3 if a.require_gain and not any(d["decision"] == "candidate_for_review" for d in result["decisions"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
