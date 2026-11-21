"""Run with the project environment: uv run python -m unittest discover -s tests -p test_z01_condition.py."""
import unittest

import numpy as np

from tools.data.condition_z01 import fit_signal, valid_segments


class TrajectoryConstraints(unittest.TestCase):
    def test_step_has_no_causal_phase_lag_and_obeys_bounds(self):
        desired = np.r_[np.full(50, -.8), np.full(50, .8)]
        result = fit_signal(desired, -.6, .6, .03)
        self.assertLessEqual(np.max(np.abs(np.diff(result))), .030002)
        self.assertLessEqual(np.max(np.abs(result)), .600002)
        np.testing.assert_allclose(result, -result[::-1], atol=2e-5)

    def test_smooth_lift_stays_above_every_required_height(self):
        minimum = np.zeros(120)
        minimum[50:55] = .08
        result = fit_signal(np.zeros(120), minimum, np.inf, .01, smoothness=100.)
        self.assertTrue(np.all(result >= minimum - 2e-6))
        self.assertLessEqual(np.max(np.abs(np.diff(result))), .010002)
        self.assertLess(result[0], .001)

    def test_constant_pose_is_preserved(self):
        result = fit_signal(np.full(80, .3), -1., 1., .02)
        np.testing.assert_allclose(result, .3, atol=2e-6)

    def test_segments_remove_bad_frames_and_both_buffers(self):
        bad = np.zeros(1600, dtype=bool)
        bad[700] = True
        self.assertEqual(valid_segments(bad, 50.), [(0, 675), (726, 1600)])
        self.assertEqual(valid_segments(np.zeros(500, dtype=bool), 50.), [])


if __name__ == "__main__":
    unittest.main()
