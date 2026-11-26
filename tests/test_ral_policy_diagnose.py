"""Prevent terminal-record gaps or completed clips from becoming false failures."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ral_policy_diagnose as diagnostic


class RecordedTermination(unittest.TestCase):
    def test_success_and_missing_terminal_sample_are_not_guessed(self):
        names = np.array(["completed", "height", "last_world"])
        alive = np.array([[True, True, True], [False, False, True]])
        position = np.zeros((2, 3, 3))
        position[1, :, 2] = 1.0
        quaternion = np.zeros((2, 3, 4))
        quaternion[:, :, 0] = 1.0
        episodes = [dict(motion_id=str(name), complete=int(i == 0)) for i, name in enumerate(names)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez(path, names=names, alive=alive, root_p=position, root_quat=quaternion,
                     f=np.zeros((2, 3), dtype=int), ref_root_p=np.zeros((1, 3)), ref_root_R=np.eye(3)[None])
            result = diagnostic.trajectory_causes(path, episodes)
        self.assertEqual([x["cause"] for x in result["episodes"]],
                         ["reference_complete", "height_only", "terminal_sample_unavailable"])
        self.assertEqual(sum(result["counts"].values()), 3)

    def test_tilt_nonfinite_and_order_mismatch(self):
        names = np.array(["tilt", "nonfinite"])
        position = np.zeros((2, 2, 3))
        position[1, 1, 2] = np.nan
        quaternion = np.zeros((2, 2, 4))
        quaternion[:, :, 0] = 1.0
        quaternion[1, 0] = [2**-.5, 2**-.5, 0, 0]  # 90 degree roll
        episodes = [dict(motion_id=str(name), complete=0) for name in names]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez(path, names=names, alive=np.array([[True, True], [False, False]]),
                     root_p=position, root_quat=quaternion, f=np.zeros((2, 2), dtype=int),
                     ref_root_p=np.zeros((1, 3)), ref_root_R=np.eye(3)[None])
            result = diagnostic.trajectory_causes(path, episodes)
            with self.assertRaisesRegex(ValueError, "ordering"):
                diagnostic.trajectory_causes(path, list(reversed(episodes)))
        self.assertEqual([x["cause"] for x in result["episodes"]], ["tilt_only", "nonfinite"])
        self.assertIsNone(result["episodes"][1]["height_error_m"])


if __name__ == "__main__":
    unittest.main()
