"""Diagnostic queue invariants; no long training or locked test data."""
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from ral_eight_hour import atomic_json, digest, read
from ral_nine_hour import CONFIG, backend_cells, blocks_for, condition_protocol, factorial_result, validate
from ral_prior_diagnose import extract_verified


class NineHourTests(unittest.TestCase):
    def setUp(self):
        self.plan = read(CONFIG)

    def test_seed_blocks_and_counterbalanced_factor(self):
        validate(self.plan)
        sets = []
        for machine in ("5080", "4090"):
            blocks = list(blocks_for(self.plan, machine))
            sets.append({b["seed"] for b in blocks})
            for b in blocks:
                self.assertEqual({(c["level"], j["group"]) for c in b["conditions"] for j in c["jobs"]},
                    {(level, group) for level in (8,32) for group in ("A+", "B")})
            self.assertEqual([c["level"] for c in blocks[0]["conditions"]], [c["level"] for c in blocks[1]["conditions"]][::-1])
            self.assertEqual([j["group"] for j in blocks[0]["conditions"][0]["jobs"]], [j["group"] for j in blocks[2]["conditions"][0]["jobs"]][::-1])
        self.assertFalse(sets[0] & sets[1])

    def test_only_assigned_training_factor_changes(self):
        for machine in ("5080", "4090"):
            block = next(blocks_for(self.plan, machine))
            condition = block["conditions"][1]
            with tempfile.TemporaryDirectory() as tmp:
                result = condition_protocol(self.plan, {"training":self.plan["training"]}, block, condition, Path(tmp)/"condition")
            diff = {k for k,v in result["training"].items() if self.plan["training"][k] != v}
            self.assertEqual(diff, {block["factor"]})

    def test_partial_block_excluded_and_first_repeat_primary(self):
        block = next(blocks_for(self.plan, "5080"))
        block["status"] = "incomplete"
        self.assertIsNone(factorial_result(block, self.plan, "5080"))
        block["status"] = "completed"
        for c, values in zip(block["conditions"], ((.8,.7),(.81,.75))):
            c.update(status="completed", rows=[])
            for j, completion in zip(c["jobs"], values):
                j["predictions"] = {"splits":{s:{"valid":False} for s in ("fit","cal")}}
                for repeat in (0,1):
                    value = completion if repeat == 0 else .99
                    c["rows"].append(dict(group=j["group"], repeat_check=repeat,
                        completion=value, completed=round(value*163), clips=163, median_valid_s=2))
        result = factorial_result(block, self.plan, "5080")
        self.assertAlmostEqual(result["B_minus_Aplus_interaction_pp"], 4.)
        self.assertEqual(result["conditions"][32]["primary"]["B"]["completion"], .7)
        block["conditions"][0]["rows"].pop()
        with self.assertRaises(ValueError):
            factorial_result(block, self.plan, "5080")

    def test_backend_design_full_before_new_directions(self):
        cells = list(backend_cells(self.plan))
        self.assertIsNone(cells[0]["direction_seed"])
        self.assertEqual(cells[0]["horizons"], [4,8])
        self.assertEqual(len(cells), 257)
        for cell in cells:
            self.assertEqual(len(cell["jobs"]), 6)
            self.assertEqual({(j["profile"],j["repeat"]) for j in cell["jobs"]},
                {(p,r) for p in ("graph_batch","eager_batch","eager_single") for r in (0,1)})

    def test_direction_reproducible_and_prefix_not_renormalized(self):
        import numpy as np
        from ral_backend_diagnose import diagnostic_direction
        x = diagnostic_direction(1001, "clip:0:20", (32,26))
        self.assertTrue(np.array_equal(x, diagnostic_direction(1001, "clip:0:20", (32,26))))
        self.assertAlmostEqual(float(np.linalg.norm(x)), 1.)
        self.assertLess(float(np.linalg.norm(x[:8])), 1.)
        self.assertFalse(np.array_equal(x, diagnostic_direction(1002, "clip:0:20", (32,26))))

    def test_archive_hash_and_traversal_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            archive = folder/"archive.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("../outside.pt"); info.size=3
                tar.addfile(info, io.BytesIO(b"bad"))
            with self.assertRaises(ValueError):
                extract_verified(archive, "wrong", folder/"bad_hash")
            self.assertFalse((folder/"bad_hash").exists())
            with self.assertRaises(ValueError):
                extract_verified(archive, digest(archive), folder/"traversal")
            self.assertFalse((folder/"outside.pt").exists())
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("jobs/model.pt"); info.size=2
                tar.addfile(info, io.BytesIO(b"ok"))
            result = extract_verified(archive, digest(archive), folder/"valid")
            self.assertEqual(result["jobs/model.pt"], digest(folder/"valid/jobs/model.pt"))


if __name__ == "__main__":
    unittest.main()
