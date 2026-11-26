import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ral_recovery import evaluation_summary, jobs_for, recovery_gate


class Recovery(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/ral/recovery_online_v1.json").read_text())

    def rows(self):
        return [dict(seed=seed, group=group, final=True, repeat_check=repeat,
                     completion=.80 if group == "A" else .78, median_valid_s=14.0,
                     nonfinite_steps=0)
                for seed in (2, 4) for group in ("A", "A+") for repeat in range(2)]

    def test_bounded_counterbalanced_queue(self):
        self.assertEqual([(j["seed"], j["group"]) for j in jobs_for(self.config)],
                         [(2, "A"), (2, "A+"), (4, "A+"), (4, "A")])
        self.config["groups"].append("C")
        with self.assertRaises(ValueError):
            list(jobs_for(self.config))

    def test_screen_requires_both_seeds_and_repeat_checks(self):
        rows = self.rows()
        self.assertTrue(recovery_gate(rows, self.config, 2)["passed"])
        self.assertFalse(recovery_gate(rows[:-1], self.config, 2)["passed"])
        self.assertFalse(recovery_gate(rows + [rows[0]], self.config, 2)["passed"])
        rows[-1]["final"] = False
        self.assertFalse(recovery_gate(rows, self.config, 2)["passed"])

    def test_no_best_repeat_cherry_picking(self):
        rows = self.rows()
        rows[2]["completion"] = .90
        self.assertFalse(recovery_gate(rows, self.config, 2)["passed"])
        rows = self.rows()
        rows[2]["median_valid_s"] = 1.0
        self.assertFalse(recovery_gate(rows, self.config, 2)["passed"])
        rows = self.rows()
        rows[0]["nonfinite_steps"] = 1
        self.assertFalse(recovery_gate(rows, self.config, 2)["passed"])

    def test_validates_episode_inventory_totals_and_nonfinite(self):
        e = dict(motion_id="a", repeat=0, eval_seed=0, complete=1, nonfinite_steps=0,
                 valid_s=1.0, planned_s=1.0, q_sqerr_sum_rad2=0., v_sqerr_sum=0.,
                 q_rmse_rad=0., v_rmse_m_s=0.)
        r = dict(episodes=[e], n_failed=0)
        self.assertEqual(evaluation_summary(r, {"a.npz": "hash"})["completion"], 1.0)
        for field, value in [("nonfinite_steps", 1), ("valid_s", float("nan")), ("motion_id", "other"),
                             ("q_sqerr_sum_rad2", float("inf")), ("valid_s", .2)]:
            invalid = copy.deepcopy(r)
            invalid["episodes"][0][field] = value
            with self.assertRaises(ValueError):
                evaluation_summary(invalid, {"a.npz": "hash"})
        r["n_failed"] = 1
        with self.assertRaises(ValueError):
            evaluation_summary(r, {"a.npz": "hash"})

    def test_unimplemented_online_c_fails_before_cuda_start(self):
        result = subprocess.run([sys.executable, "-m", "partymimic.train", "--experiment_group", "C",
                                 "--wm_fit_source", "online"], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertIn("C hybrid supervision is not implemented", result.stderr)


if __name__ == "__main__":
    unittest.main()
