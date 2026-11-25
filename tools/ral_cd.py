"""Two-machine native-response C/D pipeline; no diagnostic sweeps or auto-promotion.

5080: verified archived B parents -> GPU pairs -> matched B/C continuation -> handoff.
3090: verified handoff -> calibration -> fixed/D/permuted actor branches -> common evaluation.
--continue-session reuses verified completed jobs, never pretends an interrupted trainer resumed exactly.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tarfile
import tempfile
import time

from ral_eight_hour import ROOT, atomic_json, digest, execute, gpu_lock, read

CONFIG = ROOT / "configs/ral/cd_native_v1.json"
SELECTION = ROOT / "release/ral/stress_models_v1.json"
WORKER = ROOT / "tools/ral_cd_worker.py"


def contract(params, motions):
    import importlib.metadata
    from partymimic.experiments.protocol import code_hash, asset_hash
    names = ("tools/ral_cd.py", "tools/ral_cd_worker.py", "tools/ral_stress_eval.py",
             "tools/ral_eight_hour.py", "tools/ral_eight_hour_worker.py", "tools/ral_policy_diagnose.py",
             "tools/ral_prior_diagnose.py", "tools/ral_structure.py", "tools/ral_recovery.py",
             "tools/ral_validation_six_hour.py", "tools/ral_intervention.py", "tools/ral_online_night.py",
             "configs/ral/cd_native_v1.json", "configs/ral/validation_six_hour_v1.json",
             "release/ral/stress_models_v1.json", "pyproject.toml", "uv.lock")
    files = {p.name: digest(p) for p in sorted(motions.glob("*.npz"))}
    if not files:
        raise FileNotFoundError(f"真实动作 NPZ 缺失：{motions}")
    return dict(code_sha256=code_hash(), asset_sha256=asset_hash(), params_sha256=digest(params),
                motions=files, files={n: digest(ROOT / n) for n in names},
                versions={p: importlib.metadata.version(p) for p in ("torch", "mujoco", "mujoco-warp", "mjlab", "warp-lang")})


def tree_hashes(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def train_command(parent, group, responses, out, params, motions, plan):
    import torch
    ck = torch.load(parent, map_location="cpu", weights_only=False)
    values = dict(ck["args"])
    # Only actual parser options are emitted; provenance fields in saved args are not CLI options.
    allowed = {}
    for node in ast.walk(ast.parse((ROOT / "src/partymimic/train.py").read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith("--"):
                action = next((kw.value for kw in node.keywords if kw.arg == "action"), None)
                allowed[node.args[0].value[2:]] = isinstance(action, ast.Name) and action.id == "ON"
    for key in ("resume", "init", "protocol", "paired_initial_reference", "max_hours"):
        values.pop(key, None)
    values.update(plan["training_overrides"])
    values.update(experiment_group=group, paired_parent=str(parent), log_dir=str(out),
                  motion_dir=str(motions), parameter_manifest=str(params), response_manifest=str(responses),
                  paired_initialization=True, wm_fit_source="online", pipeline=False,
                  synthetic_test=False, eager=False,
                  response_weight=plan["C_response_weight"] if group == "C" else 0.)
    command = [sys.executable, "-u", "-m", "partymimic.train"]
    for key, value in values.items():
        if key not in allowed or value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append("--" + key)
            elif allowed[key]:
                command.append("--no-" + key)
        else:
            command += ["--" + key, str(value)]
    return command


def verify_training(run, parent, plan, group):
    import torch
    from ral_eight_hour_worker import finite_weights
    state, args = read(run / "run.json"), read(run / "args.json")
    n, warm = plan["training_overrides"]["iterations"], plan["training_overrides"]["wm_only_iters"]
    if (state["status"] != "completed" or state["policy_updates"] != n - warm or
            state["wm_updates"] != n * args["wm_batches"] or args["experiment_group"] != group or
            args["parent_checkpoint_sha256"] != digest(parent)):
        raise ValueError("Incomplete or mismatched paired continuation")
    ck = torch.load(run / f"ckpt_{n:06d}.pt", map_location="cpu", weights_only=False)
    finite_weights(ck, ("wm", "pi", "pi_live", "critic"))
    if ck["args"] != args or ck["iteration"] != n or ck["sim_steps"] != state["env_steps"]:
        raise ValueError("Checkpoint metadata differs from completed ledger")
    with (run / "train_log.csv").open() as f:
        rows = list(csv.DictReader(f))
    if not rows or any(float(row["nonfinite"]) != 0 for row in rows):
        raise ValueError("Nonfinite/skipped policy updates; preserve attempt, do not promote")
    return state


class Queue:
    def __init__(self, a, plan, lock_fd):
        self.a, self.plan, self.fd = a, plan, lock_fd
        self.cancelled = [False]
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.cancelled.__setitem__(0, True))
        frozen = contract(a.params, a.motions)
        if a.out.exists():
            if not a.continue_session:
                raise FileExistsError(f"目录已存在：{a.out}。继续已完成阶段需加 --continue-session；旧训练不会伪装成精确续训。")
            self.state = read(a.out / "status.json")
            if self.state["contract"] != frozen or self.state["machine"] != a.machine:
                raise ValueError("Existing session code/data/machine differs; use a new session")
        else:
            a.out.mkdir(parents=True)
            self.state = dict(schema_version=1, machine=a.machine, stage=plan["stage"],
                              status="running", started_unix=time.time(), contract=frozen,
                              jobs={}, handoffs={}, seeds={}, formal_results=False, response_test_accessed=False)
            atomic_json(a.out / "plan.json", plan)
        self.state["status"] = "running"
        self.state.pop("reason", None)
        self.save()

    def save(self):
        self.state["updated_unix"] = time.time()
        atomic_json(self.a.out / "status.json", self.state)

    def guard(self, _pid=None):
        if self.cancelled[0]:
            raise InterruptedError("队列收到停止信号")
        free = shutil.disk_usage(self.a.out).free / 2**30
        if free < self.plan["min_free_gib"]:
            raise RuntimeError(f"可用空间 {free:.1f} GiB，低于 {self.plan['min_free_gib']} GiB；已停止，保留已完成产物")

    def job(self, name, build, check=None, allow_blocked=False):
        """build(folder) -> (argv, result path); result directories are immutable after completion."""
        entry = self.state["jobs"].get(name)
        if entry and entry["status"] in ("completed", "blocked_degenerate_calibration"):
            result = self.a.out / entry["result"]
            hashes = tree_hashes(result) if result.is_dir() else {result.name: digest(result)}
            if hashes != entry["hashes"]:
                raise ValueError(f"已完成产物被修改：{name}")
            return None if entry["status"] != "completed" else result
        self.guard()
        if contract(self.a.params, self.a.motions) != self.state["contract"]:
            raise ValueError("Code/data changed during queue")
        attempt = 1 if entry is None else entry["attempt"] + 1
        folder = self.a.out / "jobs" / name / f"attempt{attempt:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        command, result = build(folder)
        command = list(map(str, command))
        entry = dict(status="running", attempt=attempt, started_unix=time.time(),
                     result=str(result.relative_to(self.a.out)), log=str((folder / "console.log").relative_to(self.a.out)))
        self.state["jobs"][name] = entry
        self.state["current"] = name
        atomic_json(folder / "command.json", command)
        self.save()
        print(f"[{self.a.machine}] {name} 开始；日志 {folder / 'console.log'}", flush=True)
        try:
            outcome = execute(command, folder / "console.log", time.monotonic() + self.plan["job_timeout_hours"] * 3600,
                              self.cancelled, self.fd, heartbeat=self.guard)
            blocked = (allow_blocked and (result / "run.json").is_file() and
                       read(result / "run.json").get("status") == "blocked_degenerate_calibration")
            if outcome != "completed" and not blocked:
                raise RuntimeError(f"{name}: {outcome}，请查看 {folder / 'console.log'}")
            if check and not blocked:
                check(result)
            if contract(self.a.params, self.a.motions) != self.state["contract"]:
                raise ValueError("Code/data changed while a job was running")
            entry.update(status="blocked_degenerate_calibration" if blocked else "completed",
                         finished_unix=time.time(), hashes=tree_hashes(result) if result.is_dir() else {result.name: digest(result)})
            print(f"[{self.a.machine}] {name}: {entry['status']}", flush=True)
            return None if blocked else result
        except BaseException as exc:
            entry.update(status="failed_or_interrupted", reason=str(exc), finished_unix=time.time())
            raise
        finally:
            self.save()

    def worker(self, kind, checkpoint, seed, folder, result, **options):
        command = [sys.executable, "-u", WORKER, kind, "--checkpoint", checkpoint,
                   "--params", self.a.params, "--motions", self.a.motions,
                   "--config", CONFIG, "--seed", seed, "--out", result]
        for key, value in options.items():
            command += ["--" + key.replace("_", "-"), value]
        return command, result


def parent_for(q, item):
    from ral_prior_diagnose import extract_verified
    source = q.a.source / item["folder"]
    archive = source / item["retention"]["archive"]
    cache = q.a.out / "parents" / f"s{item['seed']}"
    run = cache / f"jobs/s{item['seed']}_B_v005/run"
    ck = run / f"ckpt_{item['final_iteration']:06d}.pt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".s{item['seed']}_", dir=cache.parent) as temp:
            unpacked = Path(temp) / "unpacked"
            members = extract_verified(archive, item["retention"]["archive_sha256"], unpacked)
            if members != item["retention"]["members"]:
                raise ValueError("Parent archive member hashes differ")
            unpacked.rename(cache)
    if tree_hashes(cache) != item["retention"]["members"]:
        raise ValueError(f"父模型缓存被修改：{cache}")
    if not ck.is_file() or digest(ck) != item["checkpoint_sha256"]:
        raise ValueError(f"父模型缺失或损坏：{ck}；需要原始 B_v005 models 归档，不是 review 包")
    parent_args = read(run / "args.json")
    if parent_args["seed"] != item["seed"] or parent_args["friction_speed"] != .05:
        raise ValueError("Parent seed/variant differs")
    # Verify original motion inventory, not just the new response manifest.
    if read(run / "input_inventory.json")["motions"] != q.state["contract"]["motions"]:
        raise ValueError("Current motions differ from the original B training library")
    return run, ck


def pack_seed(q, seed, parent, pairs, branches):
    handoff = q.a.out / "handoff"
    handoff.mkdir(exist_ok=True)
    target = handoff / f"{q.a.session}_s{seed}.tar.gz"
    if target.exists():
        expected = q.state["handoffs"].get(str(seed), {}).get("sha256")
        sidecar = target.with_suffix(target.suffix + ".sha256")
        if expected is None and sidecar.exists():
            expected = sidecar.read_text().split()[0]
        if expected != digest(target):
            raise ValueError("Existing handoff differs; preserve it and choose a new session")
        q.state["handoffs"][str(seed)] = dict(file=str(target.relative_to(q.a.out)), sha256=expected)
        q.save()
        return
    selected = {"params.csv": q.a.params, "config.json": CONFIG,
                "models/B_parent/model.pt": parent,
                "models/B_parent/args.json": parent.parent / "args.json"}
    for path in pairs.rglob("*"):
        if path.is_file():
            selected["pairs/" + str(path.relative_to(pairs))] = path
    n = q.plan["training_overrides"]["iterations"]
    for name, run in branches.items():
        for path in run.iterdir():
            if path.is_file() and (path.suffix in (".json", ".csv") or path.name == f"ckpt_{n:06d}.pt"):
                selected[f"models/{name}/" + ("model.pt" if path.suffix == ".pt" else path.name)] = path
    manifest = dict(schema_version=1, seed=seed, session=q.a.session, contract=q.state["contract"],
                    files={name: digest(path) for name, path in selected.items()},
                    parent_checkpoint_sha256=digest(parent), response_manifest_sha256=digest(pairs / "manifest.json"),
                    formal_results=False)
    manifest_path = handoff / f"s{seed}_manifest.json"
    atomic_json(manifest_path, manifest)
    selected["manifest.json"] = manifest_path
    temp = target.with_suffix(target.suffix + ".tmp")
    with tarfile.open(temp, "w:gz", compresslevel=1) as tar:
        for name, path in sorted(selected.items()):
            tar.add(path, arcname=f"s{seed}/{name}", recursive=False)
    fingerprint = digest(temp)
    target.with_suffix(target.suffix + ".sha256").write_text(f"{fingerprint}  {target.name}\n")
    temp.replace(target)
    q.state["handoffs"][str(seed)] = dict(file=str(target.relative_to(q.a.out)), sha256=fingerprint)
    q.save()
    print(f"[5080] s{seed} 可交接：{target} 及其 .sha256", flush=True)


def train_5080(q):
    selection = read(SELECTION)
    if digest(q.a.source / "status.json") != selection["source_status_sha256"]:
        raise ValueError("5080 父会话不同；默认需要 logs/ral_3h/intervention01_5080")
    items = {m["seed"]: m for m in selection["models"] if m["variant"] == q.plan["parent_variant"]}
    # Fail before a long queue if any later seed's required original model is missing.
    for seed in q.plan["seeds"]:
        item = items[seed]
        archive = q.a.source / item["folder"] / item["retention"]["archive"]
        if not archive.is_file() or digest(archive) != item["retention"]["archive_sha256"]:
            raise FileNotFoundError(f"缺少原始 B_v005 模型归档或校验不符：{archive}")
    for index, seed in enumerate(q.plan["seeds"]):
        _, parent = parent_for(q, items[seed])
        pairs = q.job(f"s{seed}_pairs", lambda folder: q.worker(
            "pairs", parent, seed, folder, folder / "pairs", split_manifest=q.a.responses))
        branches = {}
        order = ("B_continue", "C") if index % 2 == 0 else ("C", "B_continue")
        for name in order:
            group = "B" if name == "B_continue" else "C"
            branches[name] = q.job(f"s{seed}_{name}", lambda folder: (
                train_command(parent, group, pairs / "manifest.json", folder / "run", q.a.params, q.a.motions, q.plan),
                folder / "run"), check=lambda run: verify_training(run, parent, q.plan, group))
        ledgers = [read(branches[name] / "run.json") for name in ("B_continue", "C")]
        if ledgers[0]["initial_module_sha256"] != ledgers[1]["initial_module_sha256"]:
            raise ValueError("B/C paired initial WM/actor/critic differ")
        pack_seed(q, seed, parent, pairs, branches)
        q.state["seeds"][str(seed)] = dict(status="handoff_ready", paired_initialization_verified=True)
        q.save()


def unpack_handoff(q, archive, seed):
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    expected = sidecar.read_text().split()[0]
    if len(expected) != 64 or digest(archive) != expected:
        raise ValueError(f"交接包校验失败或复制未完成：{archive}")
    dest = q.a.out / "inputs" / f"s{seed}"
    if not dest.exists():
        staging = q.a.out / "inputs" / f".s{seed}.unpacking"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            with tarfile.open(archive, "r:gz") as tar:
                members = tar.getmembers()
                names = set()
                for member in members:
                    path = Path(member.name)
                    if (not member.isfile() or path.is_absolute() or ".." in path.parts or
                            not path.parts or path.parts[0] != f"s{seed}" or member.name in names):
                        raise ValueError("Unsafe/duplicate handoff archive member")
                    names.add(member.name)
                    target = staging / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
            (staging / f"s{seed}").rename(dest)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    manifest = read(dest / "manifest.json")
    if manifest["seed"] != seed or manifest["session"] != q.a.session or manifest["contract"] != q.state["contract"]:
        raise ValueError("Handoff session/seed/code/data/environment differs; synchronize the same code package")
    actual = tree_hashes(dest)
    actual.pop("manifest.json")
    if actual != manifest["files"]:
        raise ValueError("Extracted handoff inventory differs")
    return dest


def process_3090(q, seed, data):
    pairs = data / "pairs/manifest.json"
    models = {name: data / "models" / name for name in ("B_parent", "B_continue", "C")}
    C = models["C"] / "model.pt"
    for name in ("B_continue", "C"):
        q.job(f"s{seed}_{name}_response", lambda folder: q.worker(
            "responses", models[name] / "model.pt", seed, folder, folder / "response.json", responses=pairs))
    calibration = q.job(f"s{seed}_calibration", lambda folder: q.worker(
        "calibrate", C, seed, folder, folder / "calibration.json", responses=pairs))
    starts = q.job(f"s{seed}_starts", lambda folder: q.worker(
        "starts", C, seed, folder, folder / "starts", responses=pairs))
    blocked = []
    variants = [(f"C_fixed{h}", "D", h) for h in q.plan["windows"]["fixed_controls"]] + [("D", "D", None), ("C_perm", "C-perm", None)]
    # Rotation changes scheduling only. Every branch has identical sampling RNG and starting live actor.
    shift = q.plan["seeds"].index(seed) % len(variants)
    for name, group, fixed in variants[shift:] + variants[:shift]:
        options = dict(calibration=calibration, starts=starts / "manifest.json", group=group)
        if fixed is not None:
            options["fixed_horizon"] = fixed
        run = q.job(f"s{seed}_{name}", lambda folder: q.worker(
            "window", C, seed, folder, folder / "run", **options), allow_blocked=True)
        if run is None:
            blocked.append(name)
        else:
            models[name] = run
    if "D" in models and "C_perm" in models:
        d, p = [read(models[n] / "run.json") for n in ("D", "C_perm")]
        for key in ("init_sha256", "start_schedule_sha256", "assigned_imagined_steps", "computed_imagined_steps"):
            if d[key] != p[key]:
                raise ValueError(f"D/C-perm budget or pairing differs: {key}")
        histograms = []
        for name in ("D", "C_perm"):
            with (models[name] / "train_metrics.csv").open() as f:
                histograms.append([tuple(row[k] for k in ("iteration", "assigned_steps", "computed_steps", "short_count"))
                                   for row in csv.DictReader(f)])
        if histograms[0] != histograms[1]:
            raise ValueError("D/C-perm per-update horizon histograms differ")
    for scenario in q.plan["evaluation"]["scenarios"]:
        for name, run in models.items():
            checkpoint = run / ("model.pt" if (run / "model.pt").exists() else "last.pt")
            def admit_evaluation(path):
                result = read(path)
                if len(result["episodes"]) != len(q.state["contract"]["motions"]) or any(e["nonfinite_steps"] for e in result["episodes"]):
                    raise ValueError("Incomplete/nonfinite common evaluation")
                result.update(stage=q.plan["stage"], scope=q.plan["evaluation"]["scope"],
                              variant=name, training_seed=seed,
                              checkpoint_weights="live" if name in ("D", "C_perm", "C_fixed8", "C_fixed32") else "EMA")
                atomic_json(path, result)
            result = q.job(f"s{seed}_eval_{scenario}_{name}", lambda folder: (
                [sys.executable, "-u", ROOT / "tools/ral_stress_eval.py", "--run", run,
                 "--checkpoint", checkpoint, "--params", q.a.params, "--motions", q.a.motions,
                 "--scenario", scenario, "--out", folder / "evaluation.json"], folder / "evaluation.json"), check=admit_evaluation)
            q.state["seeds"].setdefault(str(seed), {}).setdefault("evaluation", []).append(
                dict(variant=name, scenario=scenario, result=str(result.relative_to(q.a.out)),
                     weights="live" if name in ("D", "C_perm", "C_fixed8", "C_fixed32") else "EMA"))
            q.save()
    q.state["seeds"][str(seed)]["status"] = "completed_with_blocked_windows" if blocked else "completed"
    q.state["seeds"][str(seed)]["blocked"] = blocked
    q.save()


def result_summary(q):
    """Only complete within-seed comparisons; preserve incomplete and negative outcomes."""
    import statistics
    summary = dict(status=q.state["status"], machine=q.a.machine, formal_results=False,
                   response_test_accessed=False, seeds=[], effects={},
                   interpretation="development continuation; no superiority guarantee; D/C-perm use dense Hmax computation")
    scenarios = set(q.plan["evaluation"]["scenarios"]) - {"identified"}
    contrasts = (("C", "B_continue"), ("D", "C_perm"), ("D", "C_fixed32"), ("D", "C_fixed8"))
    differences = {f"{a}_minus_{b}_pp": [] for a, b in contrasts}
    for seed, state in q.state["seeds"].items():
        cells = {}
        for row in state.get("evaluation", []):
            if row["scenario"] not in scenarios:
                continue
            result = read(q.a.out / row["result"])
            cells.setdefault(row["variant"], {})[row["scenario"]] = sum(e["complete"] for e in result["episodes"]) / len(result["episodes"])
        values = {name: statistics.mean(c.values()) for name, c in cells.items() if set(c) == scenarios}
        effects = {f"{a}_minus_{b}_pp": 100 * (values[a] - values[b]) for a, b in contrasts if a in values and b in values}
        for name, value in effects.items():
            differences[name].append(value)
        summary["seeds"].append(dict(seed=int(seed), status=state["status"], blocked=state.get("blocked", []),
                                     completion=values, paired_effects=effects))
    for name, values in differences.items():
        summary["effects"][name] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                                        sample_sd=statistics.stdev(values) if len(values) > 1 else None)
    atomic_json(q.a.out / "summary.json", summary)


def review_package(q):
    archive = q.a.out.with_name(q.a.out.name + "_review.tar.gz")
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
        for path in sorted(q.a.out.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".csv", ".log", ".txt"):
                tar.add(path, arcname=q.a.out.name + "/" + str(path.relative_to(q.a.out)), recursive=False)
    temporary.replace(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest(archive)}  {archive.name}\n")
    print(f"轻量回传包：{archive}", flush=True)


def train_3090(q):
    q.a.inbox.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + q.a.wait_hours * 3600
    pending = list(q.plan["seeds"])
    print(f"[3090] 等待 {q.a.inbox}/{q.a.session}_s*.tar.gz 及 .sha256；收到一个种子即可工作", flush=True)
    while pending:
        q.guard()
        ready = [s for s in pending if (q.a.inbox / f"{q.a.session}_s{s}.tar.gz.sha256").is_file()
                 and (q.a.inbox / f"{q.a.session}_s{s}.tar.gz").is_file()]
        if not ready:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待交接包超时，尚缺种子 {pending}；放入文件后使用 --continue-session 继续")
            time.sleep(10)
            continue
        for seed in ready:
            data = unpack_handoff(q, q.a.inbox / f"{q.a.session}_s{seed}.tar.gz", seed)
            # Build the per-seed summary anew when reusing verified completed jobs.
            q.state["seeds"][str(seed)] = dict(status="running", evaluation=[])
            process_3090(q, seed, data)
            pending.remove(seed)
            deadline = time.monotonic() + q.a.wait_hours * 3600


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--machine", choices=("5080", "3090"), required=True)
    p.add_argument("--session", default="cd01")
    p.add_argument("--out", type=Path)
    p.add_argument("--source", type=Path, default=ROOT / "logs/ral_3h/intervention01_5080")
    p.add_argument("--motions", type=Path, default=ROOT / "data/motions/z01_lafan_v2_clean")
    p.add_argument("--params", type=Path, default=ROOT / "release/ral/params.csv")
    p.add_argument("--responses", type=Path, default=ROOT / "data/ral_pairs_v1/manifest.json")
    p.add_argument("--inbox", type=Path, default=ROOT / "data/ral_cd_inbox")
    p.add_argument("--wait-hours", type=float, default=12)
    p.add_argument("--continue-session", action="store_true")
    p.add_argument("--status", action="store_true", help="read queue status without CUDA, training or evaluation")
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.session) or not 0 < a.wait_hours <= 72:
        p.error("session must be a safe identifier; wait-hours must be in (0,72]")
    os.chdir(ROOT)
    a.out = (a.out or ROOT / "logs/ral_cd" / f"{a.session}_{a.machine}").resolve()
    if a.status:
        path = a.out / "status.json"
        if not path.is_file():
            p.error(f"尚未创建该队列：{path}；请核对 --session / --out 和启动日志")
        state = read(path)
        print(json.dumps(dict(status=state["status"], current=state.get("current"), reason=state.get("reason"),
                              completed_jobs=sum(j["status"] == "completed" for j in state["jobs"].values()),
                              blocked_jobs=[n for n,j in state["jobs"].items() if j["status"] == "blocked_degenerate_calibration"],
                              seeds=state["seeds"], handoffs=state["handoffs"]), indent=2, ensure_ascii=False))
        return
    for name in ("source", "motions", "params", "responses", "inbox"):
        setattr(a, name, getattr(a, name).resolve())
    import torch
    if not torch.cuda.is_available() or a.machine not in torch.cuda.get_device_name():
        p.error(f"此命令应在 RTX {a.machine} 上执行")
    plan = read(CONFIG)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock = ROOT / "logs/ral_8h" / (".gpu_" + hashlib.sha256(visible.encode()).hexdigest()[:12] + ".lock")
    with gpu_lock(lock) as fd:
        q = Queue(a, plan, fd)
        try:
            train_5080(q) if a.machine == "5080" else train_3090(q)
            q.state["status"] = ("completed_with_blocked_windows" if any(s.get("blocked") for s in q.state["seeds"].values()) else "completed")
        except BaseException as exc:
            q.state.update(status="stopped", reason=str(exc))
            print(f"[{a.machine}] 停止：{exc}", flush=True)
            raise
        finally:
            q.state["finished_unix"] = time.time()
            q.save()
            try:
                result_summary(q)
                review_package(q)
            except Exception as exc:
                q.state["review_package_error"] = str(exc)
                q.save()
                print(f"回传包生成失败：{exc}；训练状态与原始产物仍保存在 {a.out}", flush=True)
    print(f"[{a.machine}] {q.state['status']}：{a.out / 'status.json'}", flush=True)


if __name__ == "__main__":
    main()
