"""Control-flow and scientific comparison safeguards for unattended queues."""
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ral_eight_hour as night


def result(group="C", seed=1):
    metrics = dict(forward_mse=1., q_rmse32_rad=.05, response_rmse8=9.,
                   response_rmse32=19., zero_response_rmse8=11.,
                   sessions={s: dict(response_rmse8=9.) for s in ("a", "b", "c")})
    if group == "B":
        metrics.update(response_rmse8=10., response_rmse32=20.,
                       sessions={s: dict(response_rmse8=10.) for s in ("a", "b", "c")})
    return dict(kind="policy", group=group, candidate="C_reference" if group == "C" else "B",
                seed=seed, budget=dict(iterations=1000), complete=True, init_sha256="same-init",
                sampling_sha256=None, comparison_sha256="same-inputs-settings", cal=metrics)


class QueueRules(unittest.TestCase):
    def setUp(self):
        self.plan = night.read(night.ROOT / "configs/ral/eight_hour.json")
        self.rule = self.plan["judgment"]

    def test_seeds_and_control_order(self):
        self.plan["max_rounds"] = 2
        seeds = {}
        for machine in ("3090", "4090", "5080"):
            jobs = list(night.jobs_for(self.plan, machine))
            seeds[machine] = {j["seed"] for j in jobs if "seed" in j}
            completed_controls = set()
            for j in jobs:
                if j.get("group") == "B":
                    completed_controls.add(j["seed"])
                if j.get("group") == "C":
                    self.assertIn(j["seed"], completed_controls)
        self.assertEqual(seeds["4090"], {1, 3})
        self.assertEqual(seeds["5080"], {2, 4})
        self.assertFalse(seeds["3090"] & (seeds["4090"] | seeds["5080"]))

    def test_bad_plans_rejected(self):
        for edit in (lambda p: p.update(hours=float("nan")),
                     lambda p: p["candidates"][0].update(name="../bad"),
                     lambda p: p["candidates"][0].update(response_weight=-1.),
                     lambda p: p["policy"].update(resume="old.pt"),
                     lambda p: p.update(shutdown_reserve_seconds=1)):
            p = copy.deepcopy(self.plan)
            edit(p)
            with self.assertRaises(ValueError):
                night.validate_plan(p)

    def test_good_pair_passes_only_with_response_gain(self):
        b, c = result("B"), result()
        self.assertTrue(night.compare_pair(b, c, self.rule)["passed"])
        c["cal"]["response_rmse8"] = 9.99
        self.assertIn("response8_gain_insufficient", night.compare_pair(b, c, self.rule)["reasons"])

    def test_mismatches_and_partial_budget_rejected(self):
        for key, value in (("seed", 2), ("kind", "world_model"), ("budget", {"iterations": 500}),
                           ("init_sha256", "other"), ("comparison_sha256", "other"),
                           ("sampling_sha256", "other"), ("complete", False)):
            c = result()
            c[key] = value
            self.assertFalse(night.compare_pair(result("B"), c, self.rule)["passed"], key)

    def test_forward_nan_zero_control_and_session_guards(self):
        for key, value, reason in (("forward_mse", 2., "forward_degraded"),
                                   ("q_rmse32_rad", float("nan"), "joint_prediction_degraded"),
                                   ("zero_response_rmse8", 8., "zero_response_control_not_beaten"),
                                   ("response_rmse32", 30., "response32_degraded")):
            c = result()
            c["cal"][key] = value
            self.assertIn(reason, night.compare_pair(result("B"), c, self.rule)["reasons"])
        c = result()
        c["cal"]["sessions"] = {s: dict(response_rmse8=12.) for s in ("a", "b", "c")}
        self.assertIn("session_gain_insufficient", night.compare_pair(result("B"), c, self.rule)["reasons"])

    def make_session(self, root, seed=1, mode="policy"):
        root.mkdir()
        night.atomic_json(root / "plan.json", self.plan)
        jobs = []
        for group in ("B", "C"):
            directory = root / "jobs" / group
            (directory / "run").mkdir(parents=True)
            ck = directory / "run/last.pt"
            ck.write_bytes(b"unit-test-checkpoint")
            row = result(group, seed)
            row.update(kind=mode, checkpoint_sha256=night.digest(ck))
            night.atomic_json(directory / "analysis.json", row)
            jobs.append(dict(id=group, kind=mode, status="completed", analysis_sha256=night.digest(directory / "analysis.json")))
        night.atomic_json(root / "status.json", dict(machine="4090", status="running", deadline_unix=1., jobs=jobs))
        return root

    def test_confirmation_requires_distinct_seeds_and_same_training_kind(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = self.make_session(root / "a", 1)
            repeat = self.make_session(root / "repeat", 1)
            b = self.make_session(root / "b", 2)
            wm = self.make_session(root / "wm", 2, "world_model")
            self.assertEqual(night.judge_sessions([a, repeat])["decisions"][0]["decision"], "no_confirmed_gain")
            self.assertEqual(night.judge_sessions([a, b])["decisions"][0]["decision"], "candidate_for_review")
            self.assertTrue(all(d["decision"] == "no_confirmed_gain" for d in night.judge_sessions([a, wm])["decisions"]))
            self.assertFalse(night.judge_sessions([a, b])["windows_authorized"])

    def test_tampered_or_incomplete_results_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.make_session(Path(temp) / "session")
            (root / "jobs/C/run/last.pt").write_bytes(b"changed")
            report = night.judge_sessions([root])
            self.assertEqual(report["completed_training_runs"], 1)
            self.assertEqual(report["excluded"][0]["status"], "invalid")
            state = night.read(root / "status.json")
            state["jobs"][0]["status"] = "timed_out"
            night.atomic_json(root / "status.json", state)
            self.assertEqual(night.judge_sessions([root])["completed_training_runs"], 0)

    def test_legacy_empty_disk_stop_has_actionable_judgment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            night.atomic_json(root / "plan.json", self.plan)
            night.atomic_json(root / "status.json", dict(machine="4090", status="disk_limit_reached",
                                                         deadline_unix=1., jobs=[]))
            report = night.judge_sessions([root])
            self.assertEqual(report["completed_training_runs"], 0)
            self.assertEqual(report["decisions"], [])
            self.assertIsNone(report["sessions"][0]["disk_check"])
            self.assertIn("旧记录未保存", report["next_action"])
            self.assertIn("--output", report["next_action"])

    def judge_cli(self, session):
        return subprocess.run([sys.executable, str(night.ROOT / "tools/ral_eight_hour.py"),
                               "judge", "--machine", "4090", "--sessions", str(session)],
                              capture_output=True, text=True, timeout=10)

    def test_missing_session_cli_suggests_existing_path_without_switching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = self.make_session(root / "night01_4090")
            missing = root / "night01_retry_4090"
            process = self.judge_cli(missing)
            self.assertEqual(process.returncode, 2)
            self.assertIn(str(missing / "status.json"), process.stderr)
            self.assertIn(str(existing), process.stderr)
            self.assertIn("不会启动训练", process.stderr)
            self.assertIn("--output", process.stderr)
            self.assertNotIn("Traceback", process.stderr)
            self.assertFalse(missing.exists())
            self.assertFalse((existing / "judgment.json").exists())

    def test_existing_session_without_status_is_not_reported_as_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            night.atomic_json(root / "plan.json", self.plan)
            process = self.judge_cli(root)
            self.assertEqual(process.returncode, 2)
            self.assertIn("刚启动可稍后重试", process.stderr)
            self.assertNotIn("Traceback", process.stderr)
            self.assertFalse((root / "status.json").exists())
            self.assertFalse((root / "judgment.json").exists())

    def test_preparing_session_can_be_judged_before_first_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            night.atomic_json(root / "plan.json", self.plan)
            night.atomic_json(root / "status.json", dict(machine="4090", status="preparing",
                                                         deadline_unix=time.time()+3600, jobs=[]))
            process = self.judge_cli(root)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = night.read(root / "judgment.json")
            self.assertEqual(report["completed_training_runs"], 0)
            self.assertIn("仍在运行", report["next_action"])

    def test_invalid_status_cli_reports_input_error_without_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            night.atomic_json(root / "plan.json", self.plan)
            (root / "status.json").write_text("invalid json")
            process = self.judge_cli(root)
            self.assertEqual(process.returncode, 2)
            self.assertIn("无法读取状态记录", process.stderr)
            self.assertNotIn("Traceback", process.stderr)
            self.assertFalse((root / "judgment.json").exists())


class DiskStops(unittest.TestCase):
    def run_with_disk(self, root, free_values, used=0):
        plan = night.read(night.ROOT / "configs/ral/eight_hour.json")
        plan_path = root / "plan.json"
        night.atomic_json(plan_path, plan)
        args = SimpleNamespace(plan=plan_path, hours=None, session="disk_test", machine="4090",
                               output=root / "results", dry_run=False, max_jobs=None,
                               params="params.csv", motions="motions", responses="manifest.json",
                               allow_gpu_mismatch=False)
        console = io.StringIO()
        previous = Path.cwd()
        try:
            with mock.patch.object(night, "ROOT", root), \
                    mock.patch.object(night.shutil, "disk_usage", side_effect=[SimpleNamespace(free=x) for x in free_values]), \
                    mock.patch.object(night, "output_size", return_value=used), \
                    mock.patch.object(night, "execute", return_value="completed") as execute, \
                    redirect_stdout(console):
                self.assertEqual(night.run_queue(args), 2)
                calls = execute.call_count
        finally:
            os.chdir(previous)
        session = root / "results/disk_test_4090"
        state = night.read(session / "status.json")
        report = night.read(session / "judgment.json")
        self.assertEqual(state["status"], "disk_limit_reached")
        self.assertEqual(state["jobs"], [])
        self.assertEqual(report["completed_training_runs"], 0)
        self.assertEqual(report["sessions"][0]["disk_check"], state["disk_check"])
        self.assertIn("完整训练结果：0", console.getvalue())
        self.assertIn("--session", report["next_action"])
        return state, calls, console.getvalue()

    def test_low_disk_stops_before_gpu_prepare_with_exact_reading(self):
        with tempfile.TemporaryDirectory() as temp:
            free = 14 * 2**30 + 123
            state, calls, output = self.run_with_disk(Path(temp), [free])
            self.assertEqual(calls, 0)
            self.assertEqual(state["disk_check"]["free_bytes"], free)
            self.assertEqual(state["disk_check"]["reasons"], ["insufficient_free_space"])
            self.assertIn("最低 15 GiB", output)

    def test_disk_rechecked_after_prepare_before_first_training(self):
        with tempfile.TemporaryDirectory() as temp:
            state, calls, _ = self.run_with_disk(Path(temp), [100 * 2**30, 14 * 2**30])
            self.assertEqual(calls, 1)  # prepare only; no training process started
            self.assertEqual(state["disk_check"]["free_bytes"], 14 * 2**30)

    def test_output_limit_distinguished_from_free_space(self):
        with tempfile.TemporaryDirectory() as temp:
            state, calls, output = self.run_with_disk(Path(temp), [200 * 2**30], used=120 * 2**30)
            self.assertEqual(calls, 0)
            self.assertEqual(state["disk_check"]["reasons"], ["session_output_limit"])
            self.assertIn("本队列产物达到上限", output)


class Lifecycle(unittest.TestCase):
    def test_concurrent_judgment_writes_are_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "judgment.json"
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: night.atomic_json(path, dict(writer=i, values=[i]*500)), range(40)))
            r = night.read(path)
            self.assertEqual(r["values"], [r["writer"]]*500)
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_gpu_lock_rejects_second_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gpu.lock"
            with night.gpu_lock(path):
                with self.assertRaises(RuntimeError):
                    with night.gpu_lock(path):
                        pass
            with night.gpu_lock(path):
                pass

    def test_process_timeout_kills_descendants_and_marks_truncated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # The descendant would leave a marker if the queue only stopped its parent.
            child = "import time,pathlib;time.sleep(2);pathlib.Path(" + repr(str(root / "leaked")) + ").touch()"
            parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c'," + repr(child) + "]);time.sleep(30)"
            with night.gpu_lock(root / "lock") as fd:
                before = time.monotonic()
                status = night.execute([sys.executable, "-c", parent], root / "log", before + .5, [False], fd)
            self.assertEqual(status, "timed_out")
            self.assertLess(time.monotonic() - before, 3)
            time.sleep(2)
            self.assertFalse((root / "leaked").exists())

    def test_exit_failure_and_cancel_are_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with night.gpu_lock(root / "lock") as fd:
                self.assertEqual(night.execute([sys.executable, "-c", "raise SystemExit(7)"], root / "failed.log",
                                                time.monotonic()+3, [False], fd), "failed")
                self.assertEqual(night.execute([sys.executable, "-c", "import time;time.sleep(30)"], root / "cancelled.log",
                                                time.monotonic()+3, [True], fd), "cancelled")


if __name__ == "__main__":
    unittest.main()
