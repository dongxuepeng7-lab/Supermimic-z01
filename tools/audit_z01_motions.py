"""Audit every frame of the Z01 library against the corresponding G1 references.

This is a kinematic diagnostic, not a dynamic feasibility certificate. A hand behind
an elbow can be intentional; the direction-mismatch flag also requires the source
forearm to point forward. Reports preserve per-side counts and example timestamps.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from partymimic.robot import get_robot


class PoseAudit:
    def __init__(self, robot_name):
        if robot_name == "z01":
            self.robot = get_robot("z01")
            self.model = m = mujoco.MjModel.from_xml_path(self.robot.mjcf_path)
            joint_names = self.robot.joint_names
        elif robot_name == "g1_source":
            # Source FK only: G1 is not registered as a trainable robot.
            from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

            self.model = m = mujoco.MjModel.from_xml_path(str(G1_XML))
            joint_names = [m.joint(j).name for j in range(m.njnt)
                           if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        else:
            raise ValueError(f"Unknown audit model: {robot_name}")
        self.data = mujoco.MjData(m)
        self.joint_adr = [m.joint(n).qposadr[0] for n in joint_names]
        self.root_adr = int(m.jnt_qposadr[np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]])
        self.torso = m.body("waist_pitch_link" if robot_name == "z01" else "torso_link").id
        self.arm_bodies = []
        for side in ("left", "right"):
            names = [side + "_shoulder_roll_link", side + ("_elbow_pitch_link" if robot_name == "z01" else "_elbow_link"),
                     side + "_wrist_yaw_link"]
            self.arm_bodies.append([m.body(n).id for n in names])
        self.foot_geoms, self.foot_hulls = [], []
        if robot_name == "z01":
            for side in ("left", "right"):
                c, v = [m.geom(side + "_ankle_roll_link_" + suffix).id for suffix in ("collision", "visual")]
                mid = m.geom_dataid[v]
                start, count = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                verts = m.mesh_vert[start:start + count].astype(np.float64)
                # A linear height minimum over the hull is also the exact minimum over all vertices.
                self.foot_hulls.append(verts[ConvexHull(verts).vertices])
                self.foot_geoms.append((c, v))

    def measure(self, p, quat, q):
        m, d = self.model, self.data
        n = len(q)
        direction, bend = np.empty((n, 2, 3)), np.empty((n, 2))
        sole, visual = np.empty((n, 2)), np.empty((n, 2))
        for i in range(n):
            a = self.root_adr
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = p[i], quat[i]
            d.qpos[self.joint_adr] = q[i]
            mujoco.mj_kinematics(m, d)
            torso_R = d.xmat[self.torso].reshape(3, 3)
            for side, (shoulder, elbow, wrist) in enumerate(self.arm_bodies):
                upper = d.xpos[elbow] - d.xpos[shoulder]
                fore = d.xpos[wrist] - d.xpos[elbow]
                fore /= np.linalg.norm(fore)
                direction[i, side] = fore @ torso_R
                bend[i, side] = np.degrees(np.arccos(np.clip(np.dot(upper / np.linalg.norm(upper), fore), -1, 1)))
            for side, ((c, v), hull) in enumerate(zip(self.foot_geoms, self.foot_hulls)):
                sole[i, side] = d.geom_xpos[c, 2] - np.abs(d.geom_xmat[c].reshape(3, 3)[2]) @ m.geom_size[c]
                visual[i, side] = d.geom_xpos[v, 2] + np.min(hull @ d.geom_xmat[v].reshape(3, 3)[2])
        return {"direction": direction, "bend": bend, "sole": sole, "visual": visual}


def read_motion(path):
    with np.load(path) as z:
        p, quat, q = z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"]
        fps = float(z["fps"].reshape(-1)[0])
    if not all(np.isfinite(x).all() for x in (p, quat, q)) or fps <= 0:
        raise ValueError(f"Nonfinite data / invalid fps: {path}")
    if np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1)) > .001:
        raise ValueError(f"Non-unit root quaternions: {path}")
    return p, quat, q, fps


def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motions", type=Path, default=Path("data/motions/z01_lafan_v2"))
    ap.add_argument("--source", type=Path, default=Path("data/motions/g1/lafan"))
    ap.add_argument("--out", type=Path, default=Path("logs/z01_v2_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target, source = PoseAudit("z01"), PoseAudit("g1_source")
    names, m = target.robot.joint_names, target.model
    arms = [i for i, n in enumerate(names) if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    limits = np.array([m.joint(names[i]).range for i in arms])
    urdf_path = Path(__file__).resolve().parents[1] / "assets/hhtools_robots/z01/z01.urdf"
    velocity_by_name = {j.get("name"): float(j.find("limit").get("velocity"))
                        for j in ET.parse(urdf_path).findall("joint") if j.find("limit") is not None}
    velocity_limits = np.array([velocity_by_name[names[i]] for i in arms])
    clips = {}
    files = sorted(args.motions.glob("*.npz"))
    if not files:
        raise FileNotFoundError(args.motions)
    for index, file in enumerate(files):
        p, quat, q, fps = read_motion(file)
        sp, squ, sq, sfps = read_motion(args.source / file.name)
        if len(q) != len(sq) or not np.isclose(fps, sfps):
            raise ValueError(f"Unaligned source/target timelines: {file.name}")
        a, b = target.measure(p, quat, q), source.measure(sp, squ, sq)
        direction_error = np.degrees(np.arccos(np.clip(np.sum(a["direction"] * b["direction"], axis=-1), -1, 1)))
        mismatch = (b["direction"][..., 0] > .25) & (a["direction"][..., 0] < -.25)
        arm_q = q[:, arms]
        delta = np.abs(np.diff(arm_q, axis=0))
        velocity_ratio = delta * fps / velocity_limits
        near = np.minimum(arm_q - limits[:, 0], limits[:, 1] - arm_q) < np.radians(.5)
        exceeded = (arm_q < limits[:, 0] - 1e-4) | (arm_q > limits[:, 1] + 1e-4)
        lower = a["sole"].min(axis=-1)
        visual_lower = a["visual"].min(axis=-1)
        worst = np.unravel_index(delta.argmax(), delta.shape)
        row = {"frames": len(q), "fps": fps, "seconds": (len(q) - 1) / fps,
               "arm_frame_over_hard_limit_count": int(exceeded.any(axis=1).sum()),
               "arm_frame_near_limit_pct": float(near.any(axis=1).mean() * 100),
               "arm_near_limit_pct_by_joint": {names[arms[j]]: float(near[:, j].mean() * 100) for j in range(len(arms))},
               "arm_jump_over_15deg_frame_count": int((delta > np.radians(15)).any(axis=1).sum()),
               "arm_velocity_over_urdf_limit_frame_count": int((velocity_ratio > 1.001).any(axis=1).sum()),
               "max_arm_velocity_ratio_to_urdf": float(velocity_ratio.max()),
               "max_arm_frame_jump_deg": float(np.degrees(delta[worst])),
               "max_arm_frame_jump_joint": names[arms[worst[1]]], "max_arm_frame_jump_time_s": (worst[0] + 1) / fps,
               "collision_sole_min_mm": float(lower.min() * 1000),
               "visual_sole_min_mm": float(visual_lower.min() * 1000),
               "collision_sole_below_10mm_frame_count": int((lower < -.01).sum()),
               "visual_sole_below_10mm_frame_count": int((visual_lower < -.01).sum()),
               "collision_sole_worst_time_s": float(lower.argmin() / fps),
               "median_lower_collision_sole_mm": float(np.median(lower) * 1000),
               "both_collision_soles_above_10mm_pct": float((lower > .01).mean() * 100)}
        for j, side in enumerate(("left", "right")):
            candidates = np.flatnonzero(mismatch[:, j])
            row[side] = {"source_forward_target_backward_frames": int(mismatch[:, j].sum()),
                         "source_forward_target_backward_pct": float(mismatch[:, j].mean() * 100),
                         "longest_direction_mismatch_s": longest_run(mismatch[:, j]) / fps,
                         "first_direction_mismatch_s": float(candidates[0] / fps) if len(candidates) else None,
                         "median_forearm_direction_error_deg": float(np.median(direction_error[:, j])),
                         "p95_forearm_direction_error_deg": float(np.percentile(direction_error[:, j], 95)),
                         "median_source_elbow_bend_deg": float(np.median(b["bend"][:, j])),
                         "median_target_elbow_bend_deg": float(np.median(a["bend"][:, j]))}
        clips[file.stem] = row
        np.savez_compressed(args.out / (file.stem + "_diagnostics.npz"), fps=np.array(fps),
                            source_forearm=b["direction"], target_forearm=a["direction"],
                            direction_error_deg=direction_error, direction_mismatch=mismatch,
                            collision_sole_m=a["sole"], visual_sole_m=a["visual"])
        print(f"[{index + 1}/{len(files)}] {file.stem}: reverse L/R {row['left']['source_forward_target_backward_pct']:.1f}/"
              f"{row['right']['source_forward_target_backward_pct']:.1f}%, sole {row['collision_sole_min_mm']:.1f} mm", flush=True)
    frames = sum(r["frames"] for r in clips.values())
    result = {"scope": "Every frame of every npz; no physics rollout and no retarget regeneration.",
              "direction_definition": "Unit elbow-to-wrist vector in each robot's torso frame. Source x > 0.25 and target x < -0.25 is a review flag, not a universal motion constraint.",
              "jump_definition": ">15 degrees per source frame flags abrupt motion; no hardware velocity certification.",
              "velocity_definition": "Finite differences compared with the source Z01 URDF's declared velocity limits; these limits are not hardware-verified.",
              "model_sha256": hashlib.sha256(Path(target.robot.mjcf_path).read_bytes()).hexdigest(),
              "summary": {"clips": len(clips), "frames": frames,
                          "clips_with_direction_mismatch": sum(any(r[s]["source_forward_target_backward_frames"] for s in ("left", "right")) for r in clips.values()),
                          "clips_with_arm_limit_exceedance": sum(r["arm_frame_over_hard_limit_count"] > 0 for r in clips.values()),
                          "clips_with_arm_jump_over_15deg": sum(r["arm_jump_over_15deg_frame_count"] > 0 for r in clips.values()),
                          "clips_with_arm_velocity_over_urdf_limit": sum(r["arm_velocity_over_urdf_limit_frame_count"] > 0 for r in clips.values()),
                          "clips_with_collision_sole_below_10mm": sum(r["collision_sole_below_10mm_frame_count"] > 0 for r in clips.values()),
                          "clips_with_visual_sole_below_10mm": sum(r["visual_sole_below_10mm_frame_count"] > 0 for r in clips.values())},
              "clips": clips}
    (args.out / "report.json").write_text(json.dumps(result, indent=2))
    fields = ["clip", "seconds", "left_reverse_pct", "right_reverse_pct", "left_direction_error_deg", "right_direction_error_deg",
              "arm_near_limit_pct", "max_jump_deg", "max_jump_joint", "max_jump_time_s", "max_velocity_ratio_to_urdf", "collision_min_mm", "visual_min_mm", "foot_worst_time_s"]
    with (args.out / "clips.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for n, r in clips.items():
            writer.writerow([n, r["seconds"], r["left"]["source_forward_target_backward_pct"], r["right"]["source_forward_target_backward_pct"],
                             r["left"]["median_forearm_direction_error_deg"], r["right"]["median_forearm_direction_error_deg"],
                             r["arm_frame_near_limit_pct"], r["max_arm_frame_jump_deg"], r["max_arm_frame_jump_joint"], r["max_arm_frame_jump_time_s"],
                             r["max_arm_velocity_ratio_to_urdf"], r["collision_sole_min_mm"], r["visual_sole_min_mm"], r["collision_sole_worst_time_s"]])
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
