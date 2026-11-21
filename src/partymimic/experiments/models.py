"""Matched-input and explicit effort world models for the supported instantaneous plant.

Delay, load bias and lag are deliberately not inferred from fixture-specific event replays.
The differentiable Coulomb term is an explicit approximation to MuJoCo frictionloss.
"""

import torch

from partymimic.networks import WorldModel


class ActuatorWorldModel(WorldModel):
    def __init__(
        self, robot, layout, mode, hidden, structured=True, dt=0.02, friction_speed=0.01
    ):
        if mode not in ("matched", "actuator") or friction_speed <= 0:
            raise ValueError("Invalid actuator model configuration")
        if robot.has_action_delay:
            raise ValueError(
                "This experiment implementation requires instantaneous position actuators"
            )
        J = robot.num_joints
        # A+ and B have identical input widths and parameter counts.
        super().__init__(
            J, layout.dim + 7 * J, hidden=hidden, structured=structured, dt=dt
        )
        self.mode, self.friction_speed = mode, friction_speed
        model = robot.robot_cfg().build().compile()
        values = [[] for _ in range(7)]
        for name in robot.joint_names:
            j, a = model.joint(name), model.actuator(name)
            d = model.jnt_dofadr[j.id]
            for out, value in zip(
                values,
                (
                    model.actuator_gainprm[a.id, 0],
                    -model.actuator_biasprm[a.id, 2],
                    model.jnt_actfrcrange[j.id, 1],
                    model.dof_armature[d],
                    model.dof_damping[d],
                    model.dof_frictionloss[d],
                    robot.action_scale[name],
                ),
            ):
                out.append(float(value))
        for name, value in zip(
            ("kp", "kd", "limit", "armature", "damping", "friction", "action_scale"),
            values,
        ):
            self.register_buffer(name, torch.tensor(value))
        self.register_buffer("q0", torch.tensor(robot.default_joint_pos_list()))
        self.register_buffer("xi_center", layout.center.clone())
        self.register_buffer("xi_scale", layout.scale.clone())
        self.offset_slice = layout.slices["joint_offset"]

    def effort(self, S, a, xi_n):
        xi = xi_n * self.xi_scale + self.xi_center
        target = self.q0 + self.action_scale * a + xi[..., self.offset_slice]
        raw = self.kp * (target - S[4]) - self.kd * S[5]
        return (
            raw.clamp(-self.limit, self.limit)
            - self.damping * S[5]
            - self.friction * torch.tanh(S[5] / self.friction_speed)
        )

    def inputs(self, S, a, a_prev, xi_n):
        # Only B/C route action dependence through explicit effort. A+ has the same
        # state, parameter context and capacity but learns its raw-action dependence.
        channel = self.effort(S, a, xi_n) / self.limit if self.mode == "actuator" else a
        context = torch.cat(
            (
                self.kp / 1000,
                self.kd / 100,
                self.limit / 330,
                self.armature,
                self.damping / 20,
                self.friction / 10,
                self.action_scale,
            )
        )
        context = context.expand(*a.shape[:-1], -1)
        return super().inputs(S, channel, a_prev, torch.cat((xi_n, context), -1))


def make_world_model(robot, layout, args):
    widths = tuple(int(x) for x in args.get("wm_hidden", "1024,1024,1024").split(","))
    mode = args.get("experiment_model", "base")
    if mode == "base":
        return WorldModel(
            robot.num_joints,
            layout.dim,
            use_prev_action=robot.has_action_delay,
            hidden=widths,
            structured=args.get("wm_structured", True),
            dt=0.02,
        )
    return ActuatorWorldModel(
        robot,
        layout,
        mode,
        widths,
        args.get("wm_structured", True),
        friction_speed=args.get("friction_speed", 0.01),
    )
