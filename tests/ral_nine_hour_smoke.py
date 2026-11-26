"""Opt-in six tiny real GPU runs + three backend modes. Never a paper experiment."""
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from ral_eight_hour import atomic_json, digest, execute, gpu_lock, read
from ral_nine_hour import CONFIG
from ral_online_night import verify_training, archive_pair
from partymimic.experiments.cli import training_command
from partymimic.experiments.protocol import freeze


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    a = SimpleNamespace(params=ROOT/"release/ral/params.csv", motions=ROOT/"data/motions/z01_lafan_v2_clean", responses=ROOT/"data/ral_pairs_v1/manifest.json")
    os.environ["OMP_NUM_THREADS"] = "4"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock = ROOT/"logs/ral_8h"/(".gpu_"+hashlib.sha256(visible.encode()).hexdigest()[:12]+".lock")
    report = dict(smoke_only=True, formal_results=False, training=[], backend=[], gradient=[])
    reference, initial_hashes = out/"paired_initial.pt", None
    with gpu_lock(lock) as fd:
        def run(cmd, log):
            result = execute(cmd, log, time.monotonic()+600, [False], fd)
            if result != "completed":
                raise RuntimeError(f"Smoke {result}: {log}")
        for name, horizon, window in (("baseline",32,8),("short_actor",8,8),("long_world",32,32)):
            folder = out/name
            folder.mkdir()
            cfg = read(CONFIG)
            cfg["training"].update(iterations=6, num_envs=8, buffer_steps=80, pi_batch=16, wm_batch=16,
                wm_batches=2, wm_hidden="32,32", enc_hidden="32,32", dec_hidden="32,32", v_gcons_n=16,
                ckpt_every=6, log_every=1, horizon=horizon, wm_window=window)
            atomic_json(folder/"config.json", cfg)
            protocol = freeze(folder/"config.json", a.params, a.motions, a.responses, folder/"protocol.json")
            protocol.update(groups=["A+","B"], training_seeds=[301], expected_environment_transitions=632, smoke_only=True)
            atomic_json(folder/"protocol.json", protocol)
            pair = dict(id=name, jobs=[])
            for group in ("A+","B"):
                job = dict(id="s301_"+group.replace("+","plus"), group=group, seed=301)
                pair["jobs"].append(job)
                directory = folder/"jobs"/job["id"]/"run"
                directory.parent.mkdir(parents=True)
                cmd = training_command(SimpleNamespace(config=folder/"config.json", params=a.params,
                    motion_dir=a.motions, responses=a.responses, group=group, seed=301,
                    protocol=folder/"protocol.json", out=directory))
                if reference.exists():
                    cmd += ["--paired_initial_reference",str(reference)]
                print("Smoke",name,group,flush=True)
                run(cmd, directory.parent/"console.log")
                ledger = verify_training(directory, protocol, job)
                job["validated"] = True
                hashes = {k:ledger["initial_module_sha256"][k] for k in ("pi","critic")}
                if initial_hashes is None:
                    initial_hashes = hashes
                    shutil.copyfile(directory/"initial.pt", reference)
                elif hashes != initial_hashes:
                    raise ValueError("Six-model pairing failed")
                report["training"].append(dict(condition=name, group=group, verified=True, updates=6, env_steps=632))
                if name == "baseline":
                    gradient = directory.parent/"gradient.json"
                    run([sys.executable,"-u",str(ROOT/"tools/ral_gradient_diagnose.py"),"--checkpoint",str(directory/"ckpt_000006.pt"),
                        "--params",str(a.params),"--responses",str(a.responses),"--out",str(gradient),"--blocks-per-split","1"], directory.parent/"gradient.log")
                    result = read(gradient)
                    if not result["all_finite"] or len(result["blocks"]) != 2:
                        raise ValueError("Gradient smoke failed")
                    report["gradient"].append(dict(group=group, blocks=2, all_finite=True))
                atomic_json(out/"validation.json", report)
            archive_pair(folder, pair, 6)
        cfg = read(ROOT/"configs/ral/backend_check_v1.json")
        cfg["horizons"] = [1,2,4,8,16,32]
        atomic_json(out/"backend_config.json",cfg)
        for profile in ("graph_batch","eager_batch","eager_single"):
            print("Smoke backend",profile,flush=True)
            folder=out/profile
            run([sys.executable,"-u",str(ROOT/"tools/ral_backend_diagnose.py"),"--profile",profile,
                "--config",str(out/"backend_config.json"),"--motions",str(a.motions),"--params",str(a.params),
                "--responses",str(a.responses),"--out",str(folder),"--direction-seed","1001","--smoke"],out/(profile+".log"))
            result=read(folder/"summary.json")
            if result["status"] != "completed" or len(result["blocks"]) != 2:
                raise ValueError("Backend smoke failed")
            report["backend"].append(dict(profile=profile, blocks=2, status="completed", screen_passed=result["passed"]))
            atomic_json(out/"validation.json",report)
    report.update(status="completed", initial_actor_critic_matched=True)
    atomic_json(out/"validation.json",report)
    print(report)


if __name__ == "__main__":
    main()
