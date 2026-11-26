import csv
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile, ZipInfo

import numpy as np
import torch

from partymimic.experiments import GROUPS
from partymimic.experiments.parameters import (
    read_manifest,
    parameterized_robot,
    nominal_parameters,
    sha256,
)
from partymimic.experiments.models import make_world_model
from partymimic.experiments.response import ResponseData, paired_losses, validate_response_objective
from partymimic.experiments.windows import variable_objective
from partymimic.experiments.data import (
    inspect_motion_zip,
    import_motion_zip,
    build_response_pairs,
)
from partymimic.dr import XiLayout
from partymimic.mdp import unflatten_state
from partymimic.robot import get_robot

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "release/ral/params.csv"


def response_fixture(directory, robot, layout, param_path, horizon=3):
    """Deliberately synthetic test-only responses, with independent fake session IDs."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    B, J = 4, robot.num_joints
    meta = dict(
        schema_version=1,
        source_type="synthetic_test",
        joint_names=list(robot.joint_names),
        control_dt=0.02,
        params_sha256=sha256(param_path),
        output_scales=dict(q_rad=0.01, qd_rad_s=0.1),
        splits={},
    )
    for split in ("fit", "cal", "test"):
        s0 = np.zeros((B, 13 + 2 * J), np.float32)
        s0[:, 2] = 0.97
        s0[:, 3] = 1
        s0[:, 13 : 13 + J] = robot.default_joint_pos_list()
        epsilon = np.full(B, 0.01, np.float32)
        ap = np.zeros((B, horizon, J), np.float32)
        ap[:, :, 0] = 0.01 / (np.sqrt(horizon) * robot.action_scale_list()[0])
        states = np.repeat(s0[:, None], horizon, 1)
        plus, minus = states.copy(), states.copy()
        plus[:, :, 13] += 0.001 * np.arange(1, horizon + 1)
        minus[:, :, 13] -= 0.001 * np.arange(1, horizon + 1)
        session = np.array([f"{split}_s{i % 2}" for i in range(B)])
        path = root / f"{split}.npz"
        np.savez(
            path,
            s0=s0,
            a_prev=np.zeros((B, J)),
            xi=np.repeat(layout.center.numpy()[None], B, 0),
            a_plus=ap,
            a_minus=-ap,
            s_plus=plus,
            s_minus=minus,
            epsilon=epsilon,
            session_id=session,
            block_id=np.array([f"{split}_b{i}" for i in range(B)]),
        )
        meta["splits"][split] = dict(
            file=path.name, sha256=sha256(path), session_ids=sorted(set(session))
        )
    path = root / "manifest.json"
    path.write_text(json.dumps(meta))
    return path


class Parameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = get_robot("z01")

    def test_actual_compiled_parameters_and_unchanged_controller(self):
        a0 = parameterized_robot(self.robot, PARAMS, "nominal")
        a = parameterized_robot(self.robot, PARAMS, "identified")
        self.assertIs(a0, self.robot)
        orig = self.robot.robot_cfg().build().compile()
        changed = a.robot_cfg().build().compile()
        np.testing.assert_array_equal(orig.actuator_gainprm, changed.actuator_gainprm)
        np.testing.assert_array_equal(orig.actuator_biasprm, changed.actuator_biasprm)
        values = nominal_parameters(a)
        self.assertAlmostEqual(values["left_knee_pitch"]["armature"], 0.292)
        self.assertAlmostEqual(values["left_hip_pitch"]["damping"], 4.840868)
        self.assertEqual(
            values["left_ankle_roll"], nominal_parameters(self.robot)["left_ankle_roll"]
        )
        self.assertEqual(
            sum(r["enabled"] == "1" for r in read_manifest(PARAMS, self.robot)), 21
        )

    def test_invalid_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows = read_manifest(PARAMS, self.robot)
            rows[0]["unit"] = "motor_current"
            p = Path(d) / "p.csv"
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0])
                w.writeheader()
                w.writerows(rows)
            with self.assertRaises(ValueError):
                read_manifest(p, self.robot)

    def test_all_seven_groups_distinct(self):
        self.assertEqual(set(GROUPS), {"A0", "A", "A+", "B", "C", "C-perm", "D"})
        self.assertEqual(
            len({json.dumps(v, sort_keys=True) for v in GROUPS.values()}), 7
        )


class Responses(unittest.TestCase):
    def setUp(self):
        self.robot = parameterized_robot(get_robot("z01"), PARAMS, "identified")
        self.layout = XiLayout(self.robot, "cpu")

    def test_matched_capacity_and_response_gradients(self):
        with tempfile.TemporaryDirectory() as d:
            path = response_fixture(d, self.robot, self.layout, PARAMS)
            ds = ResponseData(path, "fit", self.robot, self.layout, sha256(PARAMS))
            args = dict(wm_hidden="16,16", experiment_model="matched")
            torch.manual_seed(1)
            matched = make_world_model(self.robot, self.layout, args)
            torch.manual_seed(1)
            act = make_world_model(
                self.robot, self.layout, dict(args, experiment_model="actuator")
            )
            self.assertEqual(
                sum(p.numel() for p in matched.parameters()),
                sum(p.numel() for p in act.parameters()),
            )
            for x, y in zip(matched.parameters(), act.parameters()):
                torch.testing.assert_close(x, y)
            b = ds.data
            loss, response, _ = paired_losses(
                act, b, self.layout.normalize, ds.q_scale, ds.v_scale
            )
            (loss + response).backward()
            self.assertTrue(torch.isfinite(loss + response))
            self.assertTrue(
                any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in act.parameters()
                )
            )
            S = unflatten_state(b["s0"])
            a = torch.zeros_like(b["a_prev"], requires_grad=True)
            effort = act.effort(S, a, self.layout.normalize(b["xi"]))
            self.assertTrue(
                torch.isfinite(torch.autograd.grad(effort.sum(), a)[0]).all()
            )

    def test_split_leakage_and_test_access_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = response_fixture(d, self.robot, self.layout, PARAMS)
            with self.assertRaises(ValueError):
                ResponseData(p, "test", self.robot, self.layout, sha256(PARAMS))
            m = json.loads(p.read_text())
            m["splits"]["cal"]["session_ids"] = m["splits"]["fit"]["session_ids"]
            p.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                ResponseData(p, "fit", self.robot, self.layout, sha256(PARAMS))

    def test_response_prefix_keeps_later_forward_labels_and_raw_errors(self):
        with tempfile.TemporaryDirectory() as d:
            path = response_fixture(d, self.robot, self.layout, PARAMS)
            ds = ResponseData(path, "fit", self.robot, self.layout, sha256(PARAMS))
            wm = make_world_model(self.robot, self.layout,
                                  dict(wm_hidden="16,16", experiment_model="actuator"))
            original = paired_losses(wm, ds.data, self.layout.normalize, .01, .1, response_horizon=1)
            changed = {k:v.clone() for k,v in ds.data.items()}
            changed['s_plus'][:, 2, 13] += 10.
            later = paired_losses(wm, changed, self.layout.normalize, .01, .1, response_horizon=1)
            torch.testing.assert_close(original[1], later[1])
            self.assertGreater(float(later[0]), float(original[0]))
            self.assertGreater(float(later[2][:,2].mean()), float(original[2][:,2].mean()))
            mse = paired_losses(wm, changed, self.layout.normalize, .01, .1)
            huber = paired_losses(wm, changed, self.layout.normalize, .01, .1, response_loss="huber")
            torch.testing.assert_close(mse[0], huber[0])
            torch.testing.assert_close(mse[2], huber[2])
            self.assertLess(float(huber[1]), float(mse[1]))
            huber[1].backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in wm.parameters()))

    def test_response_settings_reject_invalid_values(self):
        for h, loss, delta in [(-1,'mse',1.), (4,'mse',1.), (1,'bad',1.),
                               (1,'huber',0.), (1,'huber',float('nan'))]:
            with self.assertRaises(ValueError):
                validate_response_objective(h,loss,delta,3)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA precision control')
    def test_world_model_precision_override_does_not_change_policy_default(self):
        import partymimic.networks as networks
        old = networks.AMP_DTYPE
        try:
            networks.AMP_DTYPE = torch.bfloat16
            net = torch.nn.Linear(16,16).cuda()
            x = torch.randn(4,16,device='cuda')
            torch.testing.assert_close(networks.run_mlp(net,x,"none"), net(x), rtol=0, atol=0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                explicit_fp32 = networks.run_mlp(net,x,"none")
            self.assertEqual(explicit_fp32.dtype, torch.float32)
            torch.testing.assert_close(explicit_fp32, net(x), rtol=0, atol=0)
            self.assertIs(networks.AMP_DTYPE, torch.bfloat16)
            wm = make_world_model(self.robot, self.layout,
                                  dict(wm_hidden="16,16", experiment_model="actuator",wm_amp="none"))
            self.assertEqual(wm.mlp_amp,"none")
        finally:
            networks.AMP_DTYPE = old

    def test_terminal_bootstrap_and_gradient_at_selected_horizon(self):
        r = torch.ones(3, 2, requires_grad=True)
        d = torch.zeros_like(r)
        v = torch.full_like(r, 10.0, requires_grad=True)
        h = torch.tensor([1, 3])
        J = variable_objective(r, d, v, h, 0.5)
        torch.testing.assert_close(J, torch.tensor([6.0, 3.0]))
        J.sum().backward()
        torch.testing.assert_close(v.grad, torch.tensor([[0.5, 0], [0, 0], [0, 0.125]]))
        self.assertEqual(float(r.grad[2, 0]), 0.0)

    def test_permutation_preserves_budget(self):
        h = torch.tensor([2, 2, 4, 4, 4])
        p = h[torch.randperm(5)]
        self.assertEqual(h.sum(), p.sum())
        torch.testing.assert_close(h.sort().values, p.sort().values)


class MotionArchive(unittest.TestCase):
    def test_pair_generator_rejects_missing_inputs_before_building_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            motions = root / "motions"
            motions.mkdir()
            (motions / "missing.npz").symlink_to(root / "absent.npz")
            # No robot is needed: input validation must precede model construction.
            with self.assertRaisesRegex(
                FileNotFoundError, "missing files or broken symlinks"
            ):
                build_response_pairs(None, motions, PARAMS, root / "pairs")
            self.assertFalse((root / "pairs").exists())

    def test_missing_symlinks_not_imported_as_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "motions.zip"
            with ZipFile(p, "w") as z:
                i = ZipInfo("motions/fake.npz")
                i.create_system = 3
                i.external_attr = 0o120777 << 16
                z.writestr(i, "/nonexistent/motion.npz")
            self.assertEqual(inspect_motion_zip(p)["actual_npz"], 0)
            with self.assertRaises(ValueError):
                import_motion_zip(p, Path(d) / "out")
            self.assertFalse((Path(d) / "out").exists())


if __name__ == "__main__":
    unittest.main()
