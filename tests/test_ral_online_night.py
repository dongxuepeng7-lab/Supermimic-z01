import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ral_online_night import (archive_pair, compare_backend_processes, completed_cells, pair_result,
    discover_sessions, evaluation_jobs, judge, pairs_for, summarize_randomized, validate)
from ral_eight_hour import atomic_json, digest, read


class Night(unittest.TestCase):
    def setUp(self):
        self.plan = read(ROOT / "configs/ral/online_night_v1.json")

    def test_new_disjoint_balanced_seeds_and_fixed_budget(self):
        queues = [list(pairs_for(self.plan, m)) for m in ("5080", "4090")]
        self.assertEqual([q[0]["seed"] for q in queues], [6, 8])
        self.assertFalse({p["seed"] for p in queues[0]} & {p["seed"] for p in queues[1]})
        self.assertTrue(all([j["group"] for j in p["jobs"]] == ["A+", "B"] for q in queues for p in q))
        self.assertEqual(self.plan["expected_environment_transitions"], 12865536)
        for key, value in (("iterations", 1000), ("wm_fit_source", "legacy"), ("response_weight", .01)):
            bad = copy.deepcopy(self.plan)
            bad["training"][key] = value
            with self.assertRaises(ValueError):
                validate(bad)
        bad = copy.deepcopy(self.plan)
        bad["machines"]["4090"] = bad["machines"]["5080"]
        with self.assertRaises(ValueError):
            validate(bad)
        for key in ("C_authorized", "locked_response_test_access"):
            bad = copy.deepcopy(self.plan)
            bad[key] = True
            with self.assertRaises(ValueError):
                validate(bad)

    def test_discovery_uses_internal_machine_not_archive_name(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, machine in (("structure02_5080", "5080"), ("structure01_4090", "4090")):
                (root / name).mkdir()
                atomic_json(root / name / "status.json", dict(machine=machine, jobs=[]))
            self.assertEqual([p.name for p in discover_sessions([], root)], ["structure02_5080", "structure01_4090"])
            with self.assertRaisesRegex(ValueError, "structure02_5080"):
                discover_sessions([root / "structure01_5080"], root)
            (root / "another").mkdir()
            atomic_json(root / "another" / "status.json", dict(machine="5080", jobs=[]))
            with self.assertRaisesRegex(ValueError, "Multiple 5080"):
                discover_sessions([], root)

    def test_balanced_order_and_no_counting_partial_or_duplicate_cells(self):
        sessions = {m: (Path(m), [dict(id=f"s{s}_{g}", seed=s, group=g) for g in ("A+", "B")])
                    for m, s in (("5080", 2), ("4090", 4))}
        jobs = list(evaluation_jobs(self.plan, sessions))
        self.assertEqual(len({j["cell"] for j in jobs[:4]}), 1)
        self.assertEqual(jobs[0]["group"], "A+")
        self.assertEqual(jobs[8]["group"], "B")
        rows = [dict(j, completion=.8, completed=130, scenario_sha256="identical") for j in jobs[:4]]
        self.assertEqual(len(completed_cells(rows)), 1)
        self.assertEqual(completed_cells(rows[:3]), [])
        self.assertEqual(completed_cells(rows+[rows[0]]), [])
        rows[3]["scenario_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "identical"):
            completed_cells(rows)

    def test_randomized_summary_requires_requested_seed_plant_and_finite_episodes(self):
        e = dict(motion_id="a", repeat=0, eval_seed=101, complete=1, nonfinite_steps=0,
            valid_s=1., planned_s=1., q_sqerr_sum_rad2=0., v_sqerr_sum=0., q_rmse_rad=0., v_rmse_m_s=0.)
        result = dict(episodes=[e], plant="identified", n_failed=0)
        self.assertEqual(summarize_randomized(result, {"a.npz": "hash"}, 101, "identified")["completed"], 1)
        for seed, plant in ((0, "identified"), (101, "nominal")):
            with self.assertRaises(ValueError):
                summarize_randomized(result, {"a.npz": "hash"}, seed, plant)
        result["episodes"][0]["v_rmse_m_s"] = float("nan")
        with self.assertRaises(ValueError):
            summarize_randomized(result, {"a.npz": "hash"}, 101, "identified")

    def fixture_pair(self, root, validated=True):
        pair = dict(id="s6", jobs=[dict(id="s6_Aplus", validated=validated)])
        session = root / "new"
        run = session / "jobs/s6_Aplus/run"
        run.mkdir(parents=True)
        for name in ("initial.pt", "ckpt_005000.pt", "last.pt", "run.json"):
            (run / name).write_bytes((name+" data").encode())
        old = root / "historical.pt"
        old.write_bytes(b"untouched")
        return pair, session, run, old

    def test_archive_roundtrip_precedes_scoped_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            pair, session, run, old = self.fixture_pair(Path(d))
            original = {p.name: p.read_bytes() for p in run.iterdir()}
            record = archive_pair(session, pair, 5000)
            self.assertEqual(digest(session / record["archive"]), record["archive_sha256"])
            with tarfile.open(session / record["archive"]) as tar:
                self.assertEqual({Path(m).name for m in tar.getnames()}, set(original)-{"last.pt"})
                for name in tar.getnames():
                    self.assertEqual(tar.extractfile(name).read(), original[Path(name).name])
            self.assertEqual(list(run.glob("*.pt")), [])
            self.assertEqual(old.read_bytes(), b"untouched")
            self.assertEqual(record["discarded_optimizer_files"][0]["path"], "jobs/s6_Aplus/run/last.pt")
            with self.assertRaises(FileExistsError):
                archive_pair(session, pair, 5000)

    def test_failed_job_preserves_optimizer_and_corrupt_archive_preserves_originals(self):
        with tempfile.TemporaryDirectory() as d:
            pair, session, run, old = self.fixture_pair(Path(d), validated=False)
            record = archive_pair(session, pair, 5000)
            self.assertFalse(record["discarded_optimizer_files"])
            with tarfile.open(session / record["archive"]) as tar:
                self.assertIn("last.pt", [Path(n).name for n in tar.getnames()])
        with tempfile.TemporaryDirectory() as d:
            pair, session, run, old = self.fixture_pair(Path(d))
            with patch.object(tarfile.TarFile, "extractfile", return_value=io.BytesIO(b"corrupted")):
                with self.assertRaisesRegex(ValueError, "Archive verification failed"):
                    archive_pair(session, pair, 5000)
            self.assertEqual(len(list(run.glob("*.pt"))), 3)
            self.assertEqual(old.read_bytes(), b"untouched")

    def test_backend_compares_independent_processes(self):
        with tempfile.TemporaryDirectory() as d:
            roots = [Path(d) / name for name in ("first", "second")]
            cfg = read(ROOT / "configs/ral/backend_check_v1.json")
            cpu = np.zeros((2, 7, 8, 2))
            for root in roots:
                root.mkdir()
                raw = root / "fit_0000_trajectories.npz"
                gpu = cpu.copy()
                if root == roots[1]:
                    gpu[:, 1] = 1.
                np.savez(raw, train_cpu=cpu, train_gpu=gpu, actions=np.zeros((14,8,1)),
                         epsilon=np.array(.1), scales=np.ones(2), initial_train_integration=np.zeros(4))
                atomic_json(root / "summary.json", dict(status="completed", smoke_only=False, config=cfg,
                    blocks=[dict(block_id="a:1:2", split="fit", trajectory_sha256=digest(raw))]))
            report = compare_backend_processes(*roots)
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["blocks"]), 1)

    def test_pair_judgment_keeps_first_repeat_and_negative_prediction_results(self):
        rows = [dict(group=g, repeat_check=r, completed=n, clips=100, completion=n/100, median_valid_s=12.)
                for g, r, n in (("A+", 0, 80), ("A+", 1, 82), ("B", 0, 79), ("B", 1, 83))]
        jobs = [dict(group=g, predictions=dict(splits={s: dict(valid=True, q_rmse_rad={"32": q},
            response_rmse={"8": e}, zero_response_rmse={"8": 10.}) for s in ("fit", "cal")}))
                for g, q, e in (("A+", .05, 9.), ("B", .04, 9.5))]
        pair = dict(seed=6, rows=rows, jobs=jobs)
        result = pair_result(pair)
        self.assertAlmostEqual(result["primary_B_minus_Aplus_completion_pp"], -1.)
        self.assertAlmostEqual(result["prediction_diagnostics"]["cal"]["q32_B_reduction_percent"], 20.)
        self.assertLess(result["prediction_diagnostics"]["cal"]["response8_B_reduction_percent"], 0.)
        pair["rows"].append(rows[0])
        with self.assertRaises(ValueError):
            pair_result(pair)

    def test_judge_rejects_completed_status_without_model_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            atomic_json(root / "status.json", dict(machine="5080", status="completed",
                pairs=[dict(status="completed", seed=6)]))
            with self.assertRaisesRegex(ValueError, "model archive"):
                judge(root)


if __name__ == "__main__":
    unittest.main()
