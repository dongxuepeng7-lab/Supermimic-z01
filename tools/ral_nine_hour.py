"""Nine-hour diagnostic protocol: two factorial training queues and backend isolation.

Development evidence only. Frozen source and historical results are never edited.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

from ral_eight_hour import ROOT, atomic_json, check_disk, digest, execute, gpu_lock, read
from ral_online_night import (archive_pair, compare_backend_processes, pair_result,
                             review_package, verify_training)
from ral_recovery import current_inputs, evaluate_checkpoint
from ral_structure import stage_gate, tools_inventory

CONFIG = ROOT / "configs/ral/diagnose_nine_hour_v1.json"
PROTOCOL = ROOT / "release/ral/diagnose_nine_hour_protocol_v1.json"
AUDIT = ROOT / "release/ral/online_night01_review_20261123.json"


def validate(plan):
    old = read(ROOT / "configs/ral/online_night_v1.json")
    if (plan["training"] != old["training"] or plan["groups"] != ["A+", "B"]
            or plan["hours"] != 9 or plan["C_authorized"] or plan["locked_response_test_access"]
            or plan["evaluation"] != old["evaluation"]
            or plan["expected_environment_transitions"] != 12865536):
        raise ValueError("Use the frozen nine-hour A+/B diagnostic plan")
    seeds = []
    for machine, factor, levels in (("5080", "horizon", [32, 8]), ("4090", "wm_window", [8, 32])):
        m = plan["machines"][machine]
        if m["factor"] != factor or m["levels"] != levels:
            raise ValueError("Factor assignment changed")
        if any(type(m[k]) is not int or m[k] <= 0 for k in ("seed_start", "seed_stride", "max_blocks")):
            raise ValueError("Invalid seed allocation")
        seeds += [m["seed_start"]+i*m["seed_stride"] for i in range(m["max_blocks"])]
    if len(seeds) != len(set(seeds)) or min(seeds) <= 256:
        raise ValueError("New training seeds must be unique and separate from historical seeds")
    backend = plan["machines"]["3090"]
    if (backend["profiles"] != ["graph_batch", "eager_batch", "eager_single"]
            or backend["repeats"] != 2 or backend["full_horizons"] != [4, 8]
            or backend["direction_horizons"] != [1, 2, 4, 8, 16, 32]):
        raise ValueError("Backend coverage changed")


def blocks_for(plan, machine):
    m = plan["machines"][machine]
    for i in range(m["max_blocks"]):
        seed = m["seed_start"] + i*m["seed_stride"]
        levels = m["levels"][::1 if i % 2 == 0 else -1]
        groups = plan["groups"][::1 if (i//2) % 2 == 0 else -1]
        yield dict(id=f"s{seed}", seed=seed, factor=m["factor"], conditions=[
            dict(id=f"{m['factor']}{level}", level=level, seed=seed, jobs=[
                dict(id=f"s{seed}_{g.replace('+', 'plus')}", seed=seed, group=g) for g in groups])
            for level in levels])


def inventory():
    names = ["tools/ral_nine_hour.py", "tools/ral_backend_diagnose.py", "tools/ral_gradient_diagnose.py",
             "tools/ral_prior_diagnose.py", "tools/ral_online_night.py",
             "configs/ral/diagnose_nine_hour_v1.json", "configs/ral/online_night_v1.json",
             "release/ral/online_night01_review_20261123.json"]
    return dict(tools_inventory(), **{n: digest(ROOT/n) for n in names})


def inputs(a):
    return dict(base=current_inputs(a.motions, a.params, a.responses), tools=inventory())


def diagnostic_selection(a, plan):
    """Freeze a failure-enriched set using ONLY the preceding completed audit."""
    import numpy as np
    from partymimic.dr import XiLayout
    from partymimic.experiments.cli import common_robot
    from partymimic.experiments.response import ResponseData
    from ral_backend_check import select_blocks
    previous = read(AUDIT)["local_3090"]
    failed = set(x for p in previous["backend_passes"] for x in p["failed_blocks"])
    robot = common_robot(a.params)
    layout = XiLayout(robot, "cpu")
    cfg = plan["machines"]["3090"]
    result = dict(selection="Prespecified from previous full audit; failure-enriched, not population representative",
                  previous_audit_sha256=digest(AUDIT), response_test_accessed=False)
    for split in ("fit", "cal"):
        ds = ResponseData(a.responses, split, robot, layout, digest(a.params))
        names = sorted(ds.arrays["block_id"].astype(str))
        bad, good = [n for n in names if n in failed], [n for n in names if n not in failed]
        choose = lambda seq, n: [seq[j] for j in select_blocks(np.array(seq), n)] if seq else []
        result[split] = sorted(choose(bad, cfg["failed_blocks_per_split"]) + choose(good, cfg["passing_blocks_per_split"]))
    return result


def freeze(a, plan, folder):
    from partymimic.experiments.protocol import freeze as base_freeze
    validate(plan)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / "config.json", plan)
    p = base_freeze(folder / "config.json", a.params, a.motions, a.responses, folder / "protocol.json")
    p.update(groups=plan["groups"], training_seeds=[b["seed"] for m in ("5080", "4090") for b in blocks_for(plan, m)],
             stage=plan["stage"], tools=inventory(), diagnostic_selection=diagnostic_selection(a, plan),
             evaluation=plan["evaluation"], screening=plan["screening"], design=plan["machines"],
             expected_environment_transitions=plan["expected_environment_transitions"],
             response_labels_consumed=False, formal_results=False, C_authorized=False)
    atomic_json(folder / "protocol.json", p)
    return p


def condition_protocol(plan, protocol, block, condition, folder):
    folder.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(plan)
    cfg["training"][block["factor"]] = condition["level"]
    atomic_json(folder / "config.json", cfg)
    p = copy.deepcopy(protocol)
    p.update(training=cfg["training"], training_seeds=[block["seed"]],
             config_sha256=digest(folder / "config.json"), factor=block["factor"], level=condition["level"])
    atomic_json(folder / "protocol.json", p)
    return p


class QueueStop(RuntimeError):
    pass


class Runner:
    def __init__(self, a, plan, status, cutoff, cancelled, fd, frozen):
        self.a, self.plan, self.status = a, plan, status
        self.cutoff, self.cancelled, self.fd, self.frozen = cutoff, cancelled, fd, frozen

    def save(self):
        self.status["updated_unix"] = time.time()
        atomic_json(self.a.out / "status.json", self.status)

    def guard(self, reserve=False):
        if self.cancelled[0] or time.monotonic() >= self.cutoff:
            raise QueueStop("cancelled" if self.cancelled[0] else "time_budget_reached")
        disk = check_disk(self.a.out, self.plan)
        self.status["disk_check"] = disk
        slack = 2*2**30 if reserve else 0
        if (disk["reasons"] or disk["free_bytes"] < self.plan["min_free_gib"]*2**30+slack
                or disk["output_bytes"]+slack >= self.plan["max_output_gib"]*2**30):
            raise QueueStop("disk_limit_reached")

    def command(self, command, log, cutoff=None):
        self.guard()
        if inputs(self.a) != self.frozen:
            raise ValueError("Code/data changed during the queue")
        limit = min(self.cutoff, time.monotonic()+60*self.plan["job_timeout_minutes"], cutoff or self.cutoff)
        outcome = execute(command, log, limit, self.cancelled, self.fd, heartbeat=lambda _: self.guard())
        if outcome != "completed":
            if self.cancelled[0] or time.monotonic() >= self.cutoff:
                raise QueueStop("cancelled" if self.cancelled[0] else "time_budget_reached")
            raise RuntimeError(f"Process {outcome}: {log}")
        if inputs(self.a) != self.frozen:
            raise ValueError("Code/data changed during execution")

    def gradient(self, ck, folder):
        out = folder / "gradient.json"
        self.command([sys.executable, "-u", str(ROOT / "tools/ral_gradient_diagnose.py"),
            "--checkpoint", str(ck), "--params", str(self.a.params), "--responses", str(self.a.responses),
            "--out", str(out), "--blocks-per-split", str(self.plan["prior_diagnosis"]["gradient_blocks_per_split"])], folder / "gradient.log")
        report = read(out)
        if report.get("status") != "completed" or not report["all_finite"]:
            raise ValueError("Nonfinite/incomplete gradient diagnostics retained; stop")
        return dict(path=str(out.relative_to(self.a.out)), sha256=digest(out), all_finite=True)

    def evaluate(self, run, ck, output, record=False):
        self.guard()
        try:
            return evaluate_checkpoint(run, ck, output, self.a, self.frozen["base"],
                min(self.cutoff, time.monotonic()+60*self.plan["job_timeout_minutes"]), self.cancelled, self.fd, record)
        except RuntimeError:
            if self.cancelled[0] or time.monotonic() >= self.cutoff:
                raise QueueStop("cancelled" if self.cancelled[0] else "time_budget_reached") from None
            raise


def training_queue(r, protocol):
    from partymimic.experiments.cli import training_command
    from ral_prior_diagnose import diagnose_prior
    diagnose_prior(r)
    for block in blocks_for(r.plan, r.a.machine):
        r.guard(reserve=True)
        block.update(status="running")
        r.status["blocks"].append(block)
        r.status["current"] = block["id"]
        r.save()
        canonical = r.a.out / "blocks" / block["id"] / "paired_initial.pt"
        first_hashes = None
        try:
            for condition in block["conditions"]:
                r.guard(reserve=True)
                folder = canonical.parent / condition["id"]
                p = condition_protocol(r.plan, protocol, block, condition, folder)
                condition.update(status="running", rows=[], folder=str(folder.relative_to(r.a.out)))
                try:
                    for job in condition["jobs"]:
                        r.guard(reserve=True)
                        run = folder / "jobs" / job["id"] / "run"
                        run.parent.mkdir(parents=True)
                        cmd = training_command(SimpleNamespace(config=folder/"config.json", params=r.a.params,
                            motion_dir=r.a.motions, responses=r.a.responses, seed=job["seed"], group=job["group"],
                            protocol=folder/"protocol.json", out=run))
                        if canonical.exists():
                            cmd += ["--paired_initial_reference", str(canonical)]
                        atomic_json(run.parent / "command.json", cmd)
                        job.update(status="training", started_unix=time.time())
                        r.save()
                        print(f"[{r.a.machine}] {block['id']}/{condition['id']}/{job['group']} 开始；剩余 {(r.cutoff-time.monotonic())/3600:.2f} 小时", flush=True)
                        r.command(cmd, run.parent / "console.log")
                        ledger = verify_training(run, p, job)
                        job.update(validated=True, initial_module_sha256=ledger["initial_module_sha256"])
                        hashes = {k: job["initial_module_sha256"][k] for k in ("pi", "critic")}
                        if first_hashes is None:
                            first_hashes = hashes
                            shutil.copyfile(run / "initial.pt", canonical)
                        elif hashes != first_hashes:
                            raise ValueError("Four-model initial actor/critic mismatch")
                        ck = run / "ckpt_005000.pt"
                        for repeat in range(2):
                            row = r.evaluate(run, ck, run.parent / f"eval_r{repeat}.json", record=repeat == 0)
                            condition["rows"].append(dict(row, group=job["group"], seed=job["seed"], final=True, repeat_check=repeat))
                            r.save()
                        pred = run.parent / "predictions.json"
                        r.command([sys.executable, "-u", str(ROOT/"tools/ral_structure_metrics.py"),
                            "--checkpoint", str(ck), "--params", str(r.a.params), "--responses", str(r.a.responses),
                            "--out", str(pred)], run.parent / "predictions.log")
                        job["predictions"] = read(pred)
                        if not all(x["valid"] for x in job["predictions"]["splits"].values()):
                            raise ValueError("Nonfinite prediction diagnostics retained; stop")
                        job["gradient"] = r.gradient(ck, run.parent)
                        job.update(status="completed", final_sha256=digest(ck), finished_unix=time.time())
                        r.save()
                    condition.update(status="completed", gate=stage_gate(condition["rows"], r.plan, seeds=[block["seed"]]))
                finally:
                    if condition["status"] != "completed":
                        condition["status"] = "incomplete"
                    condition["retention"] = archive_pair(folder, condition, 5000)
                    r.save()
            block.update(status="completed", paired_initial_pi_critic=first_hashes)
            # Both verified archives retain the actual initialization; remove only our copied reference.
            canonical.unlink()
            r.save()
        finally:
            if block["status"] != "completed":
                block["status"] = "incomplete"
            r.save()


def backend_cells(plan):
    cfg = plan["machines"]["3090"]
    for direction in [None] + cfg["direction_seeds"]:
        yield dict(id="full_original" if direction is None else f"direction{direction}", direction_seed=direction,
            horizons=cfg["full_horizons"] if direction is None else cfg["direction_horizons"],
            jobs=[dict(profile=p, repeat=repeat) for p in cfg["profiles"] for repeat in range(2)])


def verify_profile_inputs(folders):
    """Make graph/batching the only intended backend intervention."""
    import numpy as np
    maps = []
    for folder in folders:
        report = read(folder/"summary.json")
        paths = {digest(p):p for p in folder.glob("*_trajectories.npz")}
        maps.append({row["block_id"]: paths[row["trajectory_sha256"]] for row in report["blocks"]})
    if any(m.keys() != maps[0].keys() for m in maps[1:]):
        raise ValueError("Backend profiles covered different blocks")
    for key, path in maps[0].items():
        with np.load(path) as reference:
            for other in maps[1:]:
                with np.load(other[key]) as candidate:
                    if any(not np.array_equal(reference[k], candidate[k]) for k in (
                            "actions", "epsilon", "scales", "initial_train_integration", "train_cpu")):
                        raise ValueError("Backend profiles used different input/CPU reference trajectories")
    return dict(blocks=len(maps[0]), profiles=len(folders), identical_inputs_and_cpu_reference=True)


def backend_queue(r, protocol):
    selected = r.a.out / "selection.json"
    atomic_json(selected, protocol["diagnostic_selection"])
    for cell in backend_cells(r.plan):
        r.guard(reserve=True)
        cell.update(status="running")
        r.status["backend"].append(cell)
        r.status["current"] = cell["id"]
        folder = r.a.out / "backend" / cell["id"]
        folder.mkdir(parents=True)
        cfg = read(ROOT / "configs/ral/backend_check_v1.json")
        cfg.update(blocks_per_split=1000000, horizons=cell["horizons"], selection="All fit/cal blocks" if cell["direction_seed"] is None else "Frozen previous-audit failure-enriched selection")
        atomic_json(folder / "config.json", cfg)
        try:
            for job in cell["jobs"]:
                job["status"] = "running"
                out = folder / f"{job['profile']}_r{job['repeat']}"
                job["folder"] = str(out.relative_to(r.a.out))
                r.save()
                cmd = [sys.executable, "-u", str(ROOT/"tools/ral_backend_diagnose.py"), "--out", str(out),
                    "--motions", str(r.a.motions), "--params", str(r.a.params), "--responses", str(r.a.responses),
                    "--config", str(folder/"config.json"), "--profile", job["profile"]]
                if cell["direction_seed"] is not None:
                    cmd += ["--selection", str(selected), "--direction-seed", str(cell["direction_seed"])]
                print(f"[3090] {cell['id']}/{job['profile']}/r{job['repeat']} 开始；剩余 {(r.cutoff-time.monotonic())/3600:.2f} 小时", flush=True)
                r.command(cmd, out.with_suffix(".log"))
                report = read(out / "summary.json")
                expected = 516 if cell["direction_seed"] is None else sum(len(protocol["diagnostic_selection"][s]) for s in ("fit", "cal"))
                if report["status"] != "completed" or len(report["blocks"]) != expected:
                    raise ValueError("Backend coverage incomplete")
                job.update(status="completed", checked=expected, failed=sum(not b["passed"] for b in report["blocks"]),
                           summary_sha256=digest(out/"summary.json"))
                r.save()
            cell["cross_process"] = {}
            cell["profile_inputs"] = verify_profile_inputs([folder/f"{p}_r0" for p in r.plan["machines"]["3090"]["profiles"]])
            for profile in r.plan["machines"]["3090"]["profiles"]:
                result = compare_backend_processes(folder/f"{profile}_r0", folder/f"{profile}_r1")
                atomic_json(folder/f"{profile}_cross_process.json", result)
                cell["cross_process"][profile] = dict(checked=len(result["blocks"]), failed=sum(not x["passed"] for x in result["blocks"]))
            cell["status"] = "completed"
        finally:
            if cell["status"] != "completed":
                cell["status"] = "incomplete"
            r.save()


def factorial_result(block, plan, machine):
    if block["status"] != "completed" or len(block["conditions"]) != 2 or any(c["status"] != "completed" for c in block["conditions"]):
        return None
    levels = plan["machines"][machine]["levels"]
    conditions = {c["level"]: pair_result(c) for c in block["conditions"]}
    baseline, intervention = [conditions[l] for l in levels]
    change = {g: 100*(intervention["primary"][g]["completion"] - baseline["primary"][g]["completion"]) for g in ("A+", "B")}
    return dict(seed=block["seed"], conditions=conditions, group_changes_pp=change,
                B_minus_Aplus_interaction_pp=change["B"]-change["A+"], primary_repeat=0)


def judge(path, verify=True):
    if not (path/"status.json").exists():
        raise ValueError(f"未找到任务状态：{path}/status.json。先检查启动命令和控制台日志；不要新建假状态文件。")
    state = read(path/"status.json")
    plan = read(CONFIG)
    if state["machine"] == "3090":
        if verify:
            for cell in state["backend"]:
                for job in cell["jobs"]:
                    if job.get("status") == "completed" and digest(path/job["folder"]/"summary.json") != job["summary_sha256"]:
                        raise ValueError("Backend summary changed")
        result = dict(status=state["status"], machine="3090", cells=state["backend"],
            complete_cells=sum(c["status"] == "completed" for c in state["backend"]),
            note="Original full coverage and enriched new-direction subsets are distinct. Incomplete cells excluded; negative finite results retained.")
    else:
        for block in state["blocks"]:
            for condition in block["conditions"]:
                retention = condition.get("retention")
                if condition.get("status") == "completed" and not retention:
                    raise ValueError("Completed condition has no verified model archive")
                if verify and retention and digest(path/condition["folder"]/retention["archive"]) != retention["archive_sha256"]:
                    raise ValueError("Model archive changed")
        values = [v for b in state["blocks"] if (v := factorial_result(b, plan, state["machine"])) is not None]
        effects = [v["B_minus_Aplus_interaction_pp"] for v in values]
        result = dict(status=state["status"], machine=state["machine"], complete_four_model_blocks=len(values),
            incomplete_blocks=sum(b["status"] != "completed" for b in state["blocks"]), results=values,
            descriptive_mean_interaction_pp=statistics.mean(effects) if effects else None,
            passing_condition_screens=sum(c.get("gate", {}).get("passed", False) for b in state["blocks"] for c in b["conditions"]),
            negative_interactions=sum(x < 0 for x in effects),
            note="One primary per training seed; separate GPU/factor cohorts. Equal optimizer updates/environment steps, unequal unroll compute. No superiority inference.")
    result.update(formal_results=False, C_authorized=False, response_test_accessed=False)
    atomic_json(path/"judgment.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "freeze", "judge"))
    p.add_argument("--machine", choices=("5080", "4090", "3090"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--motions", type=Path, default=ROOT/"data/motions/z01_lafan_v2_clean")
    p.add_argument("--params", type=Path, default=ROOT/"release/ral/params.csv")
    p.add_argument("--responses", type=Path, default=ROOT/"data/ral_pairs_v1/manifest.json")
    p.add_argument("--prior-session", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    os.chdir(ROOT)
    for key in ("out", "motions", "params", "responses", "prior_session"):
        if getattr(a, key) is not None:
            setattr(a, key, getattr(a, key).resolve())
    if a.mode == "judge":
        print(json.dumps(judge(a.out), indent=2, ensure_ascii=False))
        return 0
    plan = read(CONFIG)
    validate(plan)
    if a.mode == "run" and a.machine is None:
        p.error("--machine required")
    if a.dry_run:
        print(json.dumps(dict(machine=a.machine, hours=9, execution=False,
            first_blocks=list(blocks_for(plan, a.machine))[:2] if a.machine in ("5080", "4090") else [],
            backend=plan["machines"]["3090"] if a.machine == "3090" else None), indent=2))
        return 0
    if a.out.exists() or a.out.with_name(a.out.name+"_review.tar.gz").exists():
        p.error("Use a fresh session name; existing results are never overwritten")
    if a.mode == "freeze":
        freeze(a, plan, a.out)
        return 0
    # Status exists even when CUDA, lock acquisition or data preflight fails.
    a.out.mkdir(parents=True)
    state = dict(status="preparing", machine=a.machine, started_unix=time.time(),
                 blocks=[], backend=[], prior=[], formal_results=False, C_authorized=False)
    atomic_json(a.out/"status.json", state)
    cutoff = time.monotonic()+9*3600-plan["shutdown_reserve_seconds"]
    state["deadline_unix"] = time.time()+9*3600
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    exit_code = 0
    try:
        import torch
        if not torch.cuda.is_available() or a.machine not in torch.cuda.get_device_name():
            raise ValueError(f"Expected RTX {a.machine}; use the matching machine script")
        lock = ROOT/"logs/ral_8h"/(".gpu_"+hashlib.sha256(os.environ.get("CUDA_VISIBLE_DEVICES", "0").encode()).hexdigest()[:12]+".lock")
        with gpu_lock(lock) as fd:
            with tempfile.TemporaryDirectory(dir=a.out) as temp:
                candidate = Path(temp)/"freeze"
                protocol = freeze(a, plan, candidate)
                if protocol != read(PROTOCOL):
                    raise ValueError("Synchronize exact nine-hour code/data/protocol bundle")
                for name in ("protocol.json", "config.json"):
                    shutil.copyfile(candidate/name, a.out/name)
            frozen = inputs(a)
            atomic_json(a.out/"inputs.json", frozen)
            state.update(status="running", gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__))
            r = Runner(a, plan, state, cutoff, cancelled, fd, frozen)
            r.save()
            (backend_queue if a.machine == "3090" else training_queue)(r, protocol)
            state["status"] = "planned_queue_completed"
    except QueueStop as exc:
        state.update(status=str(exc), reason=str(exc))
    except BaseException as exc:
        state.update(status="stopped_error", reason=str(exc))
        print(f"停止：{exc}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        state.update(current=None, finished_unix=time.time())
        atomic_json(a.out/"status.json", state)
        try:
            judge(a.out)
        except Exception as exc:
            atomic_json(a.out/"judgment_error.json", dict(error=str(exc)))
            exit_code = 1
        review_package(a.out)
        print(f"任务状态：{state['status']}；审阅结果：{a.out/'judgment.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
