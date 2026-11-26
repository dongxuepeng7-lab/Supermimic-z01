import copy
import json
from pathlib import Path
import sys
import tempfile
import tarfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ral_structure import bundle, jobs_for, stage_gate
from ral_backend_check import compare, select_blocks
from partymimic.experiments.initialization import check_initial_reference, model_rng
from partymimic.experiments.runtime import state_hash


class Structure(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/ral/structure_online_v1.json").read_text())

    def rows(self):
        return [dict(seed=s, group=g, final=True, repeat_check=r, completion=.8,
                     median_valid_s=14., nonfinite_steps=0)
                for s in (2, 4) for g in ("A+", "B") for r in (0, 1)]

    def test_machine_assignments_and_fixed_objective(self):
        self.assertEqual([(j["seed"], j["group"]) for j in jobs_for(self.config, "5080")], [(2, "A+"), (2, "B")])
        self.assertEqual([j["seed"] for j in jobs_for(self.config, "4090")], [4, 4])
        for k, v in (("wm_fit_source", "legacy"), ("paired_initialization", False), ("response_weight", .01)):
            c = copy.deepcopy(self.config)
            c["training"][k] = v
            with self.assertRaises(ValueError):
                jobs_for(c, "5080")

    def test_gates_require_both_seeds_and_conservative_repeats(self):
        rows = self.rows()
        self.assertTrue(stage_gate(rows, self.config)["passed"])
        self.assertFalse(stage_gate(rows[:-1], self.config)["passed"])
        self.assertFalse(stage_gate(rows+[rows[0]], self.config)["passed"])
        for index, key, value in ((0, "completion", .69), (2, "completion", .70),
                                  (2, "median_valid_s", 10.), (0, "completion", .9), (3, "nonfinite_steps", 1)):
            bad = copy.deepcopy(rows)
            bad[index][key] = value
            self.assertFalse(stage_gate(bad, self.config)["passed"])
        self.assertFalse(stage_gate(rows, self.config)["C_authorized"])

    def test_model_construction_does_not_shift_actor_or_runtime_rng(self):
        def build(width):
            torch.manual_seed(27)
            with model_rng(2, "wm", True):
                torch.nn.Linear(width, width)
            with model_rng(2, "pi", True):
                pi = torch.nn.Linear(7, 4)
            with model_rng(2, "critic", True):
                critic = torch.nn.Linear(7, 1)
            return state_hash({"pi": pi, "critic": critic}), torch.rand(10)
        a, ra = build(8)
        b, rb = build(64)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(ra, rb))
        torch.manual_seed(27)
        self.assertTrue(torch.equal(ra, torch.rand(10)))

    def test_initial_reference_rejects_different_policy(self):
        modules = {name: torch.nn.Linear(4, 2) for name in ("pi", "critic")}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "initial.pt"
            torch.save({k: m.state_dict() for k, m in modules.items()}, path)
            check_initial_reference(modules, path)
            with torch.no_grad():
                modules["pi"].weight.add_(1)
            with self.assertRaises(ValueError):
                check_initial_reference(modules, path)

    def test_response_gate_does_not_accept_zero_predictor(self):
        cfg = json.loads((ROOT / "configs/ral/backend_check_v1.json").read_text())
        cpu = np.zeros((2, 7, 8, 4))
        for k, factor in enumerate(cfg["epsilon_factors"]):
            cpu[:, 1+2*k] = factor
            cpu[:, 2+2*k] = -factor
        self.assertTrue(all(x["passed"] for x in compare(cpu, cpu.copy(), np.ones(4), .01, cfg)))
        self.assertFalse(any(x["passed"] for x in compare(cpu, np.zeros_like(cpu), np.ones(4), .01, cfg)))
        noisy = cpu.copy()
        noisy[1, 1::2] += .2
        self.assertFalse(all(x["passed"] for x in compare(cpu, noisy, np.ones(4), .01, cfg)))
        self.assertEqual(select_blocks(np.array(["c", "a", "b"]), 2).tolist(), [1, 0])

    def test_transfer_preserves_initialization_and_final_not_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            session = Path(d) / "pair"
            run = session / "jobs/s2_B/run"
            run.mkdir(parents=True)
            files = ("initial.pt", "ckpt_000250.pt", "ckpt_000500.pt", "ckpt_000750.pt",
                     "ckpt_001000.pt", "last.pt", "train_log.csv")
            for name in files:
                (run / name).write_text("fixture")
            bundle(session)
            with tarfile.open(Path(d) / "pair_transfer.tar.gz") as archive:
                names = {Path(p).name for p in archive.getnames()}
            self.assertEqual(names, set(files)-{"last.pt", "ckpt_000750.pt"})


if __name__ == "__main__":
    unittest.main()
