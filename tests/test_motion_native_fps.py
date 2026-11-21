import unittest
from types import SimpleNamespace

import numpy as np

from partymimic.motion import MotionLib


class NativeMotionRate(unittest.TestCase):
    def test_long_50hz_clip_preserves_frames_and_values(self):
        n = 20000
        rng = np.random.default_rng(10)
        p = rng.normal(size=(n, 3)).astype(np.float32)
        q = rng.normal(size=(n, 2)).astype(np.float32)
        quat = np.tile(np.array([1., 0., 0., 0.], dtype=np.float32), (n, 1))
        kin = SimpleNamespace(forward=lambda p, R, v, w, q, qd: (p[:, None], R[:, None], v[:, None], w[:, None]))
        motion = MotionLib(["synthetic.npz"], kin, "cpu", lambda _: (p, quat, q, 50.))
        self.assertEqual(motion.num_frames, n)
        np.testing.assert_array_equal(motion.q.numpy(), q)
        np.testing.assert_array_equal(motion.root_p.numpy(), p)


if __name__ == "__main__":
    unittest.main()
