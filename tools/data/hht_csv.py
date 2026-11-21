"""CSV interchange for the G1 source and HHT's Z01 retargeting output.

G1 is retained only as the LAFAN1 retargeting source, not as a training robot.
"""

import argparse
from pathlib import Path

import numpy as np

OBJ_PREFIX = "x__"

G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def g1csv(a) -> None:
    files = sorted(Path(a.src).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No G1 source npz clips in {a.src}")
    header = ["root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)", "root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
    header += [f"dof_{n}(rad)" for n in G1_JOINTS]
    for i, f in enumerate(files):
        d = np.load(f)
        if float(d["fps"].reshape(-1)[0]) != a.fps:
            raise SystemExit(f"{f}: fps {d['fps']} != {a.fps}")
        rows = np.concatenate((d["body_pos_w"][:, 0], d["body_quat_w"][:, 0], d["joint_pos"]), -1)
        # R2R takes a folder holding an object_*.csv for an interaction scene and skips all of its clips
        stem = f"{OBJ_PREFIX}{f.stem}" if f.stem.startswith("object_") else f.stem
        out = Path(a.out) / f"shard_{i % a.shards}" / f"{stem}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, rows, delimiter=",", fmt="%.6f", header=",".join(header), comments="")
    print(len(files), "clips ->", a.out)


def load_hht_csv(path: Path, joints: list[str]) -> tuple[np.ndarray, float]:
    """qpos rows [root xyz, root wxyz, joints in MJCF order] and the sample rate of an HHT retargeted CSV."""
    fps = None
    with open(path) as fh:
        for n_meta, line in enumerate(fh):
            if not line.startswith("#"):
                head = line.strip().split(",")
                break
            if line.startswith("# sample_rate:"):
                fps = float(line.split(":")[1])
    raw = np.loadtxt(path, delimiter=",", skiprows=n_meta + 1, ndmin=2)
    col = {n: i for i, n in enumerate(head)}
    q = raw[:, [col[k] for k in ("root_qw", "root_qx", "root_qy", "root_qz")]]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    qpos = np.concatenate((raw[:, [col["root_x"], col["root_y"], col["root_z"]]], q,
                           raw[:, [col[f"dof_{n}"] for n in joints]]), -1)
    return qpos, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="G1 LAFAN1 npz directory")
    parser.add_argument("out", type=Path, help="HHT input CSV shards")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--fps", type=float, default=50.)
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    g1csv(args)


if __name__ == "__main__":
    main()
