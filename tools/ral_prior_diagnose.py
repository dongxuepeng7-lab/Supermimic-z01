"""Verify and diagnose the prespecified historical first/failed A+/B pairs locally."""
import hashlib
from pathlib import Path
import shutil
import tarfile
import time

from ral_eight_hour import ROOT, atomic_json, digest, read
from ral_recovery import check_run
from ral_structure import module_hashes


def extract_verified(archive, expected, destination):
    if digest(archive) != expected:
        raise ValueError(f"Historical model archive differs from audited hash: {archive}")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for m in members:
            if not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts:
                raise ValueError("Archive must contain only safe relative regular files")
        for m in members:
            target = destination / m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    return {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*") if p.is_file()}


def verify_archived_run(run, protocol, seed, group):
    import torch
    from ral_eight_hour_worker import finite_weights
    ledger, index = check_run(run, protocol)
    if (ledger["training_seed"], ledger["group"], ledger["env_steps"]) != (seed, group, 12865536):
        raise ValueError("Historical full training budget differs")
    if ledger["protocol_sha256"] != digest(ROOT/"release/ral/online_night_protocol_v1.json"):
        raise ValueError("Historical protocol differs")
    states = []
    for name, iteration in (("initial.pt", 0), ("ckpt_005000.pt", 5000)):
        if digest(run/name) != index[name]["sha256"]:
            raise ValueError("Historical checkpoint differs from index")
        ck = torch.load(run/name, map_location="cpu", weights_only=False)
        finite_weights(ck, ("wm", "pi", "critic") if iteration == 0 else ("wm", "pi", "pi_live", "critic"))
        if (ck["iteration"], ck["sim_steps"], ck["args"]) != (iteration, ledger["env_steps"] if iteration else 0, ledger["args"]):
            raise ValueError("Historical tensors do not match metadata")
        states.append(ck)
    actual = module_hashes(run/"initial.pt")
    if actual != ledger["initial_module_sha256"]:
        raise ValueError("Historical initial tensor hashes differ")
    for name in ("wm", "pi"):
        if not any(not torch.equal(v, states[1][name][k]) for k,v in states[0][name].items() if "weight" in k):
            raise ValueError("Historical model did not learn")
    return {k: actual[k] for k in ("pi", "critic")}


def diagnose_prior(r):
    from ral_nine_hour import AUDIT, QueueStop
    audit = read(AUDIT)["machines"][r.a.machine]
    previous = r.a.prior_session or ROOT/"logs/ral_online_8h"/f"online_night01_{r.a.machine}"
    original_cutoff = r.cutoff
    r.cutoff = min(original_cutoff, time.monotonic()+r.plan["prior_diagnosis"]["minutes_max"]*60)
    try:
        for seed in r.plan["machines"][r.a.machine]["prior_seeds"]:
            row = next(p for p in audit["pairs"] if p["seed"] == seed)
            archive = previous / row["model_archive"]
            item = dict(seed=seed, archive=str(archive), expected_sha256=row["model_archive_sha256"], jobs=[])
            r.status["prior"].append(item)
            if not archive.exists():
                item.update(status="skipped_missing_archive", actual_weights_verified=False)
                print(f"旧模型诊断跳过：缺少 {archive}；继续新的独立四模型对照。", flush=True)
                r.save()
                continue
            r.guard(reserve=True)
            folder = r.a.out/"prior"/f"s{seed}"
            item.update(status="verifying")
            r.save()
            members = extract_verified(archive, row["model_archive_sha256"], folder)
            atomic_json(folder/"verified_members.json", members)
            expected = None
            for group in ("A+", "B"):
                run = folder/"jobs"/f"s{seed}_{group.replace('+', 'plus')}"/"run"
                actual = verify_archived_run(run, read(ROOT/"release/ral/online_night_protocol_v1.json"), seed, group)
                if expected is not None and expected != actual:
                    raise ValueError("Historical actual initial actor/critic differ")
                expected = actual
            item.update(actual_weights_verified=True, paired_initial_pi_critic=expected)
            try:
                for group in ("A+", "B"):
                    run = folder/"jobs"/f"s{seed}_{group.replace('+', 'plus')}"/"run"
                    ck = run/"ckpt_005000.pt"
                    job = dict(group=group, rows=[], checkpoint_sha256=digest(ck))
                    item["jobs"].append(job)
                    for repeat in range(2):
                        job["rows"].append(dict(r.evaluate(run, ck, run.parent/f"reeval_r{repeat}.json", record=repeat == 0), repeat=repeat))
                        r.save()
                    job["gradient"] = r.gradient(ck, run.parent)
                    job["status"] = "completed"
                    r.save()
                item["status"] = "completed"
            finally:
                # The original verified archive stays untouched. Only extracted copies are removed.
                if digest(archive) != item["expected_sha256"]:
                    raise ValueError("Historical archive changed; extracted copies retained")
                for name, sha in members.items():
                    path = folder/name
                    if path.suffix == ".pt":
                        if digest(path) != sha:
                            raise ValueError("Extracted tensor changed; retain copies")
                        path.unlink()
                r.save()
    except (QueueStop, RuntimeError) as exc:
        if not r.cancelled[0] and time.monotonic() >= r.cutoff and r.cutoff < original_cutoff:
            if r.status["prior"]:
                r.status["prior"][-1].update(status="prelude_time_limit", reason=str(exc))
            print("旧模型诊断达到预定45分钟上限；保留已有结果，进入新实验。", flush=True)
        else:
            raise
    finally:
        r.cutoff = original_cutoff
        r.save()
