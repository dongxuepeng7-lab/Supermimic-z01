"""Safe motion intake and deterministic CPU physics response pairs."""

from pathlib import Path, PurePosixPath
import json
import stat
from zipfile import ZipFile

import numpy as np

from .parameters import sha256


def inspect_motion_zip(path):
    records = []
    with ZipFile(path) as archive:
        for item in archive.infolist():
            p = PurePosixPath(item.filename)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Unsafe archive path: {p}")
            if item.filename.endswith(".npz"):
                link = stat.S_ISLNK(item.external_attr >> 16)
                records.append(
                    dict(
                        name=item.filename,
                        is_symlink=link,
                        target=archive.read(item).decode() if link else None,
                        bytes=item.file_size,
                    )
                )
    return dict(
        archive_sha256=sha256(path),
        files=len(records),
        actual_npz=sum(not r["is_symlink"] for r in records),
        missing_link_targets=[
            r["target"]
            for r in records
            if r["is_symlink"] and not Path(r["target"]).is_file()
        ],
        records=records,
    )


def import_motion_zip(path, destination):
    """Resolve links only if their original targets actually exist; never install broken .npz files."""
    info = inspect_motion_zip(path)
    if info["missing_link_targets"]:
        raise ValueError(
            f"Archive contains {len(info['missing_link_targets'])} missing symbolic-link targets and no usable copy of those motions"
        )
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Motion destination is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    import io

    with ZipFile(path) as archive:
        names = set()
        for r in info["records"]:
            name = Path(r["name"]).name
            if name in names:
                raise ValueError("Duplicate motion filenames")
            names.add(name)
            content = (
                Path(r["target"]).read_bytes()
                if r["is_symlink"]
                else archive.read(r["name"])
            )
            with np.load(io.BytesIO(content), allow_pickle=False) as z:
                if any(
                    k not in z
                    for k in ("joint_pos", "body_pos_w", "body_quat_w", "fps")
                ):
                    raise ValueError(f"Invalid motion file {name}")
            (destination / name).write_bytes(content)
    return info


def build_response_pairs(
    robot,
    motion_dir,
    params_path,
    output,
    horizon=10,
    blocks_per_clip=4,
    epsilon=0.01,
    seed=73,
):
    """CPU MuJoCo pairs from independent motion-clip sessions, no hardware claims.

    Both branches restore the same complete integration state after a shared reference
    pre-roll. The model/physics rate matches training; backend differences are recorded.
    """
    import mujoco
    import torch
    from partymimic.dr import XiLayout
    from partymimic.kinematics import Kinematics
    from partymimic.motion import MotionLib
    from partymimic.mdp import flatten_state
    from partymimic.rotations import quat_to_mat

    if min(horizon, blocks_per_clip) < 1 or epsilon <= 0:
        raise ValueError("Positive pair horizon/count/amplitude required")
    files = sorted(Path(motion_dir).glob("*.npz"))
    # Segment IDs from the same source recording stay together across all splits.
    sessions = sorted({p.stem.split("__seg")[0] for p in files})
    if len(sessions) < 6:
        raise ValueError(
            "At least six source motion sessions needed for fit/cal/test (two per split)"
        )
    rng = np.random.default_rng(seed)
    sessions = [sessions[i] for i in rng.permutation(len(sessions))]
    ncal = max(2, len(sessions) // 5)
    mapping = {
        s: "cal" if i < ncal else "test" if i < 2 * ncal else "fit"
        for i, s in enumerate(sessions)
    }
    kin = Kinematics(robot, "cpu")
    layout = XiLayout(robot, "cpu")
    # The compiled entity has the same actuator and joint parameters used by mjlab.
    entity = robot.robot_cfg().build()
    spec = entity.spec.copy()
    spec.worldbody.add_geom(
        name="pair_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        friction=[0.3, 0.005, 0.0001],
    )
    model = spec.compile()
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    model.opt.tolerance = 1e-6
    joints = [model.joint(n).id for n in robot.joint_names]
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    aids = [model.actuator(n).id for n in robot.joint_names]
    data = mujoco.MjData(model)
    q0, scale = (
        np.asarray(robot.default_joint_pos_list()),
        np.asarray(robot.action_scale_list()),
    )
    xi = layout.center.numpy().copy()
    xi[layout.slices["friction"]] = 1.0
    xi[layout.slices["joint_offset"]] = 0.0
    xi[layout.slices["com"]] = 0.0
    com_id = model.body(robot.com_body).id
    default_com = model.body_ipos[com_id].copy()
    robot_geoms = model.geom_bodyid != 0

    def state():
        R = quat_to_mat(torch.tensor(data.qpos[3:7], dtype=torch.float64))
        S = (
            torch.tensor(data.qpos[:3]),
            R,
            torch.tensor(data.qvel[:3]),
            R @ torch.tensor(data.qvel[3:6]),
            torch.tensor(data.qpos[qa]),
            torch.tensor(data.qvel[va]),
        )
        return flatten_state(S).numpy()

    def step(action):
        data.ctrl[aids] = q0 + scale * action + xi[layout.slices["joint_offset"]]
        for _ in range(4):
            mujoco.mj_step(model, data)
        return state()

    buckets = {s: [] for s in ("fit", "cal", "test")}
    for path in files:
        lib = MotionLib([str(path)], kin, "cpu", robot.load_motion)
        if lib.num_frames < horizon + 25:
            continue
        for b in range(blocks_per_clip):
            from partymimic.dr import _match

            xi[layout.slices["friction"]] = rng.uniform(*robot.dr.friction_range)
            xi[layout.slices["joint_offset"]] = rng.uniform(
                -1, 1, robot.num_joints
            ) * np.asarray(_match(robot.dr.joint_offset, robot.joint_names))
            xi[layout.slices["com"]] = [
                rng.uniform(lo, hi) for lo, hi in robot.com_range
            ]
            model.geom_friction[robot_geoms, 0] = xi[layout.slices["friction"]][0]
            model.body_ipos[com_id] = default_com + xi[layout.slices["com"]]
            mujoco.mj_setConst(model, data)
            start = int(rng.integers(0, lib.num_frames - horizon - 24))
            mujoco.mj_resetData(model, data)
            p, R, v, w, q, dq = (
                a[0].numpy() for a in lib.ref_state(torch.tensor([start]))
            )
            from partymimic.rotations import mat_to_quat

            data.qpos[:3] = p
            data.qpos[3:7] = mat_to_quat(torch.tensor(R)).numpy()
            data.qpos[qa] = q
            data.qvel[:3] = v
            data.qvel[3:6] = R.T @ w
            data.qvel[va] = dq
            previous = (q - q0) / scale
            for t in range(20):
                previous = (lib.q[start + t + 1].numpy() - q0) / scale
                step(previous)
            s0 = state()
            if not np.isfinite(s0).all():
                raise FloatingPointError(f"Nonfinite shared pre-roll: {path.name}")
            spec_state = mujoco.mjtState.mjSTATE_INTEGRATION
            saved = np.empty(mujoco.mj_stateSize(model, spec_state))
            mujoco.mj_getState(model, data, saved, spec_state)
            action = (lib.q[start + 21 : start + 21 + horizon].numpy() - q0) / scale
            direction = rng.normal(size=action.shape)
            direction /= np.linalg.norm(direction)
            row = dict(
                s0=s0,
                a_prev=previous.copy(),
                xi=xi.copy(),
                epsilon=epsilon,
                session_id=path.stem.split("__seg")[0],
                block_id=f"{path.stem}:{b}:{start}",
            )
            for side, sign in (("plus", 1), ("minus", -1)):
                mujoco.mj_setState(model, data, saved, spec_state)
                mujoco.mj_forward(model, data)
                actions = action + sign * epsilon * direction / scale
                row["a_" + side] = actions
                row["s_" + side] = np.stack([step(a) for a in actions])
            buckets[mapping[row["session_id"]]].append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=1,
        source_type="physics_paired",
        backend="CPU MuJoCo; cross-backend validation required",
        joint_names=list(robot.joint_names),
        control_dt=0.02,
        physics_dt=0.005,
        params_sha256=sha256(params_path),
        output_scales={"q_rad": 0.01, "qd_rad_s": 0.1},
        motion_hashes={p.name: sha256(p) for p in files},
        seed=seed,
        splits={},
    )
    for split, rows in buckets.items():
        if not rows:
            raise ValueError(f"No sufficiently long clips in {split}")
        file = output / f"{split}.npz"
        np.savez_compressed(
            file, **{k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        )
        manifest["splits"][split] = dict(
            file=file.name,
            sha256=sha256(file),
            session_ids=sorted({r["session_id"] for r in rows}),
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
