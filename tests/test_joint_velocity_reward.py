import unittest
from types import SimpleNamespace

import torch

from partymimic.kinematics import Kinematics
from partymimic.mdp import MDP
from partymimic.motion import CONTROL_DT
from partymimic.robot import get_robot


class SpeedPenalty(unittest.TestCase):
    def test_excess_penalty_and_gradient(self):
        robot = get_robot("z01")
        kin = Kinematics(robot, "cpu")
        q = torch.tensor([robot.default_joint_pos_list()])
        p, R = torch.tensor([[0., 0., .97]]), torch.eye(3)[None]
        zeros = torch.zeros(1, 3)
        qd = torch.zeros_like(q, requires_grad=True)
        with torch.no_grad():
            qd[0, 14] = kin.qd_max[14] * 1.2
        state = p, R, zeros, zeros, q, qd
        bp, br, bv, bw = kin.forward(*state)
        motion = SimpleNamespace(interface="joint", root_p=p, root_R=R,
                                 body_p=bp.detach(), body_R=br.detach(), body_v=bv.detach(), body_w=bw.detach())
        mdp = MDP(kin, motion, q[0], 1)
        action, frame = torch.zeros_like(q), torch.tensor([0])
        base = mdp.reward(state, frame, action, action)
        penalized = mdp.reward(state, frame, action, action, w_joint_velocity=.5, velocity_margin=.9)
        torch.testing.assert_close(base - penalized, torch.tensor([.5 * .3**2 * CONTROL_DT]))
        grad, = torch.autograd.grad(penalized - base, qd)
        self.assertLess(grad[0, 14].item(), 0.)
        self.assertEqual(torch.count_nonzero(grad).item(), 1)


if __name__ == "__main__":
    unittest.main()
