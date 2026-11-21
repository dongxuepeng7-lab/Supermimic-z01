"""Check the data boundaries retained after removing other robot pipelines."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from tools.data.hht_csv import G1_JOINTS, g1csv, load_hht_csv
from tools.data.prep_motions import load_lafan_csv, resample


class RetargetDataInterfaces(unittest.TestCase):
    def test_lafan_root_quaternion_and_50hz_timeline(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            # LAFAN1 stores xyzw; the internal source format must use wxyz.
            rows = np.zeros((31, 36))
            rows[:, 0] = np.linspace(0., 1., 31)
            rows[:, 6] = 1.
            rows[:, 7:] = np.arange(29) * .01
            np.savetxt(path, rows, delimiter=",")
            result = resample(load_lafan_csv(str(path)), 30., 50.)
            self.assertEqual(result.shape, (51, 36))
            np.testing.assert_allclose(result[:, 0], np.linspace(0., 1., 51))
            np.testing.assert_array_equal(result[:, 3:7], np.tile([1., 0., 0., 0.], (51, 1)))
            np.testing.assert_allclose(result[:, 7:], np.tile(rows[0, 7:], (51, 1)))

    def test_g1_source_csv_has_named_joints_and_pelvis_first(self):
        with TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "source", Path(tmp) / "shards"
            src.mkdir()
            p = np.array([[[1., 2., .97]], [[2., 3., .98]]])
            quat = np.tile([1., 0., 0., 0.], (2, 1, 1))
            q = np.tile(np.arange(29) * .01, (2, 1))
            np.savez(src / "walk.npz", fps=np.array([50.]), joint_pos=q, body_pos_w=p, body_quat_w=quat)
            g1csv(SimpleNamespace(src=src, out=out, shards=1, fps=50.))
            path = out / "shard_0/walk.csv"
            header = path.read_text().splitlines()[0].split(",")
            self.assertEqual(header[7:], [f"dof_{name}(rad)" for name in G1_JOINTS])
            rows = np.loadtxt(path, delimiter=",", skiprows=1)
            np.testing.assert_allclose(rows, np.concatenate((p[:, 0], quat[:, 0], q), axis=1))

    def test_hht_target_csv_uses_joint_names_not_column_order(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "retarget.csv"
            path.write_text(
                "# sample_rate: 50\n"
                "dof_right,root_qz,root_x,root_y,root_z,root_qw,root_qx,root_qy,dof_left\n"
                "0.2,0,1,2,0.97,-2,0,0,-0.1\n"
            )
            qpos, fps = load_hht_csv(path, ["left", "right"])
            self.assertEqual(fps, 50.)
            np.testing.assert_allclose(qpos, [[1., 2., .97, 1., 0., 0., 0., -.1, .2]])


if __name__ == "__main__":
    unittest.main()
