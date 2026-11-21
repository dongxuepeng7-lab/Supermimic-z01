"""CPU checks for Z01: assets, actuator mapping, FK, mirror symmetry and PD standing.

    uv run python tools/check_z01.py
"""

import hashlib
import json

import mujoco
import numpy as np
import torch

from partymimic.kinematics import Kinematics
from partymimic.rotations import quat_to_mat
from partymimic.z01.z01_constants import CONTROL, MJCF_PATH, REPO, SPEC


def main():
    for path in ("urdf/Z01_v1.6.urdf", "mjcf/Z01_v1.6.urdf", "mjcf/Z01_v1.6.xml"):
        mujoco.MjModel.from_xml_path(str(REPO / "Z01_v1.6_whole_body" / path))
    entity = SPEC.robot_cfg().build()
    assert tuple(entity.joint_names) == SPEC.joint_names
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    compiled_entity = entity.compile()
    for field in ("body_mass", "body_inertia", "dof_armature", "dof_damping", "dof_frictionloss",
                  "actuator_gainprm", "actuator_biasprm", "jnt_actfrcrange"):
        assert np.allclose(getattr(model, field), getattr(compiled_entity, field)), field
    data = mujoco.MjData(model)
    kin = Kinematics(SPEC, "cpu")
    bodies = [model.body(name).id for name in SPEC.tracked_bodies]
    joints = [model.joint(name).id for name in SPEC.joint_names]
    assert model.nq == 33 and model.nu == 26
    assert set(CONTROL["default_joint_pos"]) == set(SPEC.joint_names)
    assert np.all(model.actuator_gainprm[:, 0] > 0)
    assert np.all(model.actuator_biasprm[:, 2] < 0)
    assert not any(model.geom_bodyid[g] == 0 for g in range(model.ngeom))
    for i, name in enumerate(SPEC.joint_names):
        lo, hi = model.jnt_range[model.joint(name).id]
        assert lo <= SPEC.default_joint_pos[name] <= hi
        assert model.actuator_trnid[i, 0] == model.joint(name).id
    rng = np.random.default_rng(0)
    worst_pos = worst_rot = worst_mirror = 0.
    reflection = np.diag([1., -1., 1.])
    permutation = [SPEC.joint_names.index(name.replace("left_", "@").replace("right_", "left_").replace("@", "right_"))
                   for name in SPEC.joint_names]
    mirrored_bodies = [model.body(name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")).id
                       for name in SPEC.tracked_bodies]
    for _ in range(12):
        qpos = model.qpos0.copy()
        qpos[2] = .97
        qpos[7:] = rng.uniform(model.jnt_range[joints, 0] * .8, model.jnt_range[joints, 1] * .8)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        p = torch.tensor(qpos[None, :3], dtype=torch.float32)
        R = quat_to_mat(torch.tensor(qpos[None, 3:7], dtype=torch.float32))
        q = torch.tensor(qpos[None, 7:], dtype=torch.float32)
        out = kin.forward(p, R, torch.zeros(1, 3), torch.zeros(1, 3), q, torch.zeros_like(q))
        worst_pos = max(worst_pos, float(np.abs(out[0].numpy()[0] - data.xpos[bodies]).max()))
        worst_rot = max(worst_rot, float(np.abs(out[1].numpy()[0] - data.xmat[bodies].reshape(-1, 3, 3)).max()))
        expected = data.xpos[mirrored_bodies].copy() @ reflection
        data.qpos[7:] = qpos[7:][permutation] * np.array(SPEC.mirror_sign)
        mujoco.mj_kinematics(model, data)
        worst_mirror = max(worst_mirror, float(np.abs(data.xpos[bodies] - expected).max()))
    assert worst_pos < 1e-5 and worst_rot < 1e-5
    assert worst_mirror < 1e-4, worst_mirror

    # Independently simulate the position actuators under gravity at the actual 5 ms training step.
    # Add only the ground that mjlab supplies in training; no support or root constraints are used.
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    spec.worldbody.add_geom(name="check_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .05], contype=1, conaffinity=1)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    initial_collisions = [(model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name, float(c.dist))
                          for c in data.contact if c.dist < -.001]
    assert not initial_collisions, initial_collisions
    for _ in range(round(5. / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    height, upright = float(data.qpos[2]), float(data.xmat[model.body("base_link").id].reshape(3, 3)[2, 2])
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert height > .75 and upright > .9, (height, upright)
    print(json.dumps({"robot": SPEC.name, "joints": SPEC.num_joints, "assets_and_actuators": "OK",
                      "model_sha256": hashlib.sha256(MJCF_PATH.read_bytes()).hexdigest(),
                      "control_sha256": hashlib.sha256((REPO / "assets/z01/control.json").read_bytes()).hexdigest(),
                      "fk_max_pos_m": worst_pos, "fk_max_rot": worst_rot, "mirror_max_pos_m": worst_mirror,
                      "standing_seconds": 5, "standing_root_height_m": height, "standing_upright": upright,
                      "parameter_status": CONTROL["parameter_status"]}, indent=2))


if __name__ == "__main__":
    main()
