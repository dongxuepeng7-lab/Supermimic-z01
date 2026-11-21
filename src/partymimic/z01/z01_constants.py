"""Z01 v1.6: source dynamics with simulation-only PD settings in assets/z01/control.json.

Generate the training MJCF with `uv run python tools/data/z01_data.py prepare`.
The original standalone MJCF is preserved. Hardware-calibrated gains/inertias are not available.
"""

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from partymimic.motion import load_npz
from partymimic.robot import RobotSpec

REPO = Path(__file__).resolve().parents[3]
MJCF_PATH = REPO / "assets/z01/robot.xml"
CONTROL = json.loads((REPO / "assets/z01/control.json").read_text())
if not MJCF_PATH.is_file():
    raise FileNotFoundError("Z01 training MJCF is missing. Run: uv run python tools/data/z01_data.py prepare")
_M = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
JOINT_NAMES = tuple(_M.joint(j).name for j in range(_M.njnt) if _M.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE)
_URDF_SPEEDS = {j.get("name"): float(j.find("limit").get("velocity"))
                for j in ET.parse(REPO / "assets/hhtools_robots/z01/z01.urdf").findall("joint")
                if j.find("limit") is not None}
DEFAULT_JOINT_POS = CONTROL["default_joint_pos"]
TRACKED_BODIES = (
    "base_link", "left_hip_roll_link", "left_knee_pitch_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_pitch_link", "right_ankle_roll_link", "waist_pitch_link",
    "left_shoulder_roll_link", "left_elbow_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_pitch_link", "right_wrist_yaw_link", "head_pitch_link",
)


def actuator_of(name):
    groups = [group for group in CONTROL["actuators"] if re.fullmatch(group["pattern"], name)]
    if len(groups) != 1:
        raise ValueError(f"Z01 joint {name}: expected one control group, got {len(groups)}")
    return groups[0]


def validate_control():
    if set(DEFAULT_JOINT_POS) != set(JOINT_NAMES):
        raise ValueError("Z01 control.json must define every joint's standing angle")
    expected_pose = [0., 0., CONTROL["root_height"], 1., 0., 0., 0.,
                     *(DEFAULT_JOINT_POS[name] for name in JOINT_NAMES)]
    consistent = np.allclose(_M.key("standing").qpos, expected_pose)
    for name in JOINT_NAMES:
        group = actuator_of(name)
        aid, jid = _M.actuator(name).id, _M.joint(name).id
        consistent &= np.isclose(_M.actuator_gainprm[aid, 0], group["kp"])
        consistent &= np.isclose(-_M.actuator_biasprm[aid, 2], group["kd"])
        consistent &= np.isclose(_M.dof_armature[_M.jnt_dofadr[jid]], group["armature"])
    if not consistent:
        raise ValueError("Z01 controller and generated MJCF differ. Run: uv run python tools/data/z01_data.py prepare")


validate_control()


def mirror_signs():
    data = mujoco.MjData(_M)
    mujoco.mj_kinematics(_M, data)
    reflection = np.diag([1., -1., 1.])
    signs = []
    for name in JOINT_NAMES:
        partner = name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        axis = reflection @ data.xaxis[_M.joint(name).id]
        other = data.xaxis[_M.joint(partner).id]
        if np.allclose(other, -axis, atol=1e-5):
            signs.append(1.)
        elif np.allclose(other, axis, atol=1e-5):
            signs.append(-1.)
        else:
            raise ValueError(f"Non-mirrored Z01 axes: {name}, {partner}")
    return tuple(signs)


def get_spec():
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    for actuator in list(spec.actuators):
        spec.delete(actuator)  # mjlab creates the position actuators below
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def get_robot_cfg():
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(re.escape(name),), stiffness=actuator_of(name)["kp"],
            damping=actuator_of(name)["kd"], effort_limit=float(_M.jnt_actfrcrange[_M.joint(name).id, 1]),
            armature=actuator_of(name)["armature"],
        ) for name in JOINT_NAMES
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0., 0., CONTROL["root_height"]),
                                             joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.}),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=.9),
    )


SPEC = RobotSpec(
    name="z01", mjcf_path=str(MJCF_PATH), joint_names=JOINT_NAMES, tracked_bodies=TRACKED_BODIES,
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale={name: CONTROL["action_fraction_of_effort"] * float(_M.jnt_actfrcrange[_M.joint(name).id, 1]) /
                  actuator_of(name)["kp"] for name in JOINT_NAMES},
    com_body="waist_pitch_link", com_range=((-0.025, .025), (-.025, .025), (-.025, .025)),
    robot_cfg=get_robot_cfg, motion_dir=str(REPO / "data/motions/z01_lafan_v2_clean"), motion_glob="*.npz",
    load_motion=load_npz, mirror_sign=mirror_signs(),
    joint_velocity_limits=tuple(_URDF_SPEEDS[n] for n in JOINT_NAMES),
)
