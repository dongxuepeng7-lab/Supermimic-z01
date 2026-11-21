"""Numerical original/A0 equivalence and identified-parameter wiring check."""

from pathlib import Path
import csv
import json

import mujoco
import numpy as np

from partymimic.robot import get_robot
from .parameters import parameterized_robot, nominal_parameters


def probe(params, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = get_robot("z01")
    robots = {
        "original": original,
        "A0": parameterized_robot(original, params, "nominal"),
        "A": parameterized_robot(original, params, "identified"),
    }
    trajectories = {}
    rows = []
    for group, robot in robots.items():
        m = robot.robot_cfg().build().compile()
        m.opt.timestep = 0.005
        m.opt.gravity[:] = 0
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        d = mujoco.MjData(m)
        ids = [m.joint(n).id for n in robot.joint_names]
        qa, va = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
        aids = [m.actuator(n).id for n in robot.joint_names]
        q0 = np.asarray(robot.default_joint_pos_list())
        d.qpos[qa] = q0
        values = []
        for t in range(100):
            target = q0 + 0.02 * np.sin(2 * np.pi * t * 0.005 / 0.2)
            d.ctrl[aids] = target
            mujoco.mj_step(m, d)
            values.append(
                np.concatenate(
                    (d.qpos[qa], d.qvel[va], d.qfrc_actuator[va], d.qfrc_passive[va])
                )
            )
            for i, n in enumerate(robot.joint_names):
                rows.append(
                    dict(
                        group=group,
                        joint=n,
                        time_s=(t + 1) * 0.005,
                        target_rad=target[i],
                        q_rad=d.qpos[qa[i]],
                        dq_rad_s=d.qvel[va[i]],
                        actuator_torque_Nm=d.qfrc_actuator[va[i]],
                        passive_torque_Nm=d.qfrc_passive[va[i]],
                    )
                )
        trajectories[group] = np.stack(values)
    error = float(np.max(np.abs(trajectories["A0"] - trajectories["original"])))
    change = float(np.max(np.abs(trajectories["A"] - trajectories["A0"])))
    if (
        not all(np.isfinite(v).all() for v in trajectories.values())
        or error > 1e-10
        or change < 1e-8
    ):
        raise AssertionError((error, change))
    report = dict(
        original_A0_max_abs=error,
        A_A0_max_abs=change,
        passed=True,
        scope="CPU free-body, zero gravity actuator wiring; not locomotion/hardware validation",
        effective_identified=nominal_parameters(robots["A"]),
    )
    with (output / "actuator_probe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    (output / "probe.json").write_text(json.dumps(report, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout="constrained")
    for group in ("A0", "A"):
        r = [r for r in rows if r["group"] == group and r["joint"] == "left_knee_pitch"]
        axs[0].plot([x["time_s"] for x in r], [x["q_rad"] for x in r], label=group)
        axs[1].plot(
            [x["time_s"] for x in r], [x["actuator_torque_Nm"] for x in r], label=group
        )
    axs[0].set_ylabel("Position (rad)")
    axs[1].set_ylabel("Actuator torque (N m)")
    axs[1].set_xlabel("Time (s)")
    axs[0].legend()
    fig.suptitle("Left knee: parameter wiring check (zero-gravity simulation)")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"fig_actuator_probe.{extension}", dpi=180)
    plt.close(fig)
    return report
