"""Prepare the G1 LAFAN1 source clips for Z01 retargeting (30 Hz CSV -> 50 Hz npz).

G1 body states are computed with mjlab's source model. Only root pose and joint
angles are used by HHT; full body states remain useful for source diagnostics.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    small = s < 1e-6
    w0 = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, s))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1, s))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample(qpos: np.ndarray, fps_in: float, fps: float) -> np.ndarray:
    n = qpos.shape[0]
    t = np.arange(0, (n - 1) / fps_in + 1e-9, 1 / fps)
    x = t * fps_in
    i0 = np.minimum(np.floor(x).astype(int), n - 2)
    a = (x - i0)[:, None]
    out = (1 - a) * qpos[i0] + a * qpos[i0 + 1]
    out[:, 3:7] = quat_slerp(qpos[i0, 3:7], qpos[i0 + 1, 3:7], a)
    return out


def quat_to_rotvec_diff(q_prev: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    """World-frame rotation vector of q_next * q_prev^-1."""
    out = np.zeros((q_prev.shape[0], 3))
    for i in range(q_prev.shape[0]):
        qi = np.zeros(4)
        mujoco.mju_negQuat(qi, q_prev[i])
        dq = np.zeros(4)
        mujoco.mju_mulQuat(dq, q_next[i], qi)
        mujoco.mju_quat2Vel(out[i], dq, 1.0)
    return out


def convert(qpos: np.ndarray, model: mujoco.MjModel, fps: float) -> dict[str, np.ndarray]:
    dt = 1 / fps
    T = qpos.shape[0]
    qvel = np.zeros((T, model.nv))
    pad = np.concatenate([qpos[:1], qpos, qpos[-1:]])
    qvel[:, :3] = (pad[2:, :3] - pad[:-2, :3]) / (2 * dt)
    qvel[:, 6:] = (pad[2:, 7:] - pad[:-2, 7:]) / (2 * dt)
    ang_w = quat_to_rotvec_diff(pad[:-2, 3:7], pad[2:, 3:7]) / (2 * dt)
    # free-joint angular velocity is expressed in the body frame
    for i in range(T):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, qpos[i, 3:7])
        qvel[i, 3:6] = R.reshape(3, 3).T @ ang_w[i]
    qvel[[0, -1]] = qvel[[1, -2]]
    data = mujoco.MjData(model)
    nb = model.nbody - 1
    body_pos = np.zeros((T, nb, 3))
    body_quat = np.zeros((T, nb, 4))
    body_lin = np.zeros((T, nb, 3))
    body_ang = np.zeros((T, nb, 3))
    vel = np.zeros(6)
    for i in range(T):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        body_pos[i] = data.xpos[1:]
        body_quat[i] = data.xquat[1:]
        for b in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)
            body_ang[i, b - 1] = vel[:3]
            body_lin[i, b - 1] = vel[3:]
    return {
        "fps": np.array([fps]),
        "joint_pos": qpos[:, 7:].astype(np.float32),
        "joint_vel": qvel[:, 6:].astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
    }


def load_lafan_csv(path: str) -> np.ndarray:
    """LAFAN1 G1 retarget csv (no header): root position (m), root quaternion xyzw, 29 joint angles (rad)."""
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return np.concatenate((a[:, :3], a[:, [6, 3, 4, 5]], a[:, 7:36]), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/lafan1_g1"))
    parser.add_argument("--out", type=Path, default=Path("data/motions/g1/lafan"))
    args = parser.parse_args()
    files = sorted(args.src.glob("*.csv"))
    if not files:
        parser.error(f"No LAFAN1 source CSVs in {args.src}; run tools/download_data.sh first")
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for file in files:
        raw = load_lafan_csv(str(file))
        if raw.shape[1] != model.nq or len(raw) < 2 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid G1 LAFAN1 source: {file}")
        qpos = resample(raw, 30., 50.)
        np.savez(args.out / (file.stem + ".npz"), **convert(qpos, model, 50.))
        index.append({"name": file.stem, "frames": len(qpos)})
        print(f"[{len(index)}/{len(files)}] {file.stem}", flush=True)
    report = {"fps": 50., "body_names": [model.body(i).name for i in range(1, model.nbody)], "clips": index}
    (args.out / "index.json").write_text(json.dumps(report, indent=2))
    print(f"Prepared {len(index)} source clips -> {args.out}")


if __name__ == "__main__":
    main()
