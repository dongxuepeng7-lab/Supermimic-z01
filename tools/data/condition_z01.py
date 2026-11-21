"""Constrain final Z01 trajectories, then check every frame before training.

Run with the project's Python: uv run python tools/data/condition_z01.py ...
Preserves fps, frame count, root xy/orientation and clip timing. Each violating
joint is fitted over the whole clip with angle and finite-difference speed
constraints; it is not a causal angle clip with accumulating phase lag.
Ground correction only lifts the root and does not assert dynamic feasibility.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.ndimage import maximum_filter1d
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[2]


def fit_signal(reference, lower, upper, max_step, smoothness=2.):
    """Closest smooth sequence within position and inter-frame step bounds."""
    n = len(reference)
    if n < 2 or not np.isfinite(reference).all() or max_step <= 0:
        raise ValueError("Expected a finite sequence of at least two frames and a positive speed limit")
    d1 = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format="csc")
    d2 = d1[1:] - d1[:-1]
    p = sparse.eye(n, format="csc") + smoothness * (d2.T @ d2)
    a = sparse.vstack([sparse.eye(n), d1], format="csc")
    lo, hi = np.broadcast_to(lower, (n,)), np.broadcast_to(upper, (n,))
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(p, format="csc"), q=-np.asarray(reference, dtype=float), A=a,
                 l=np.r_[lo, np.full(n - 1, -max_step)], u=np.r_[hi, np.full(n - 1, max_step)],
                 eps_abs=1e-7, eps_rel=1e-7, max_iter=30000, polishing=True, verbose=False)
    solver.warm_start(x=np.clip(reference, lo, hi))
    result = solver.solve(raise_error=False)
    if result.info.status_val != 1:
        raise RuntimeError(f"Trajectory fit did not converge: {result.info.status}")
    x = result.x
    if (np.any(x < lo - 2e-6) or np.any(x > hi + 2e-6)
            or np.max(np.abs(np.diff(x))) > max_step + 2e-6):
        raise RuntimeError("Trajectory fit failed independent constraint verification")
    return x


class Geometry:
    def __init__(self, model):
        self.m, self.d = model, mujoco.MjData(model)
        self.boxes = [g for g in range(model.ngeom) if model.geom_bodyid[g] and
                      (model.geom_contype[g] or model.geom_conaffinity[g])]
        if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in self.boxes):
            raise ValueError("Z01 checker expects box collision geometry")
        self.feet = []
        for side in ("left", "right"):
            c, v = [model.geom(side + "_ankle_roll_link_" + suffix).id for suffix in ("collision", "visual")]
            mesh = model.geom_dataid[v]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start + count].astype(float)
            self.feet.append((c, v, vertices[ConvexHull(vertices).vertices]))

    def heights(self, pose, collisions=False):
        m, d = self.m, self.d
        d.qpos[:] = pose
        if collisions:
            mujoco.mj_forward(m, d)
        else:
            mujoco.mj_kinematics(m, d)
        rot = d.geom_xmat.reshape(-1, 3, 3)
        boxes = d.geom_xpos[self.boxes, 2] - np.sum(np.abs(rot[self.boxes, 2]) * m.geom_size[self.boxes], axis=1)
        soles = [min(d.geom_xpos[c, 2] - np.abs(rot[c, 2]) @ m.geom_size[c],
                     d.geom_xpos[v, 2] + np.min(hull @ rot[v, 2])) for c, v, hull in self.feet]
        penetration = max((-float(c.dist) for c in d.contact), default=0.) if collisions else 0.
        return min(soles), float(boxes.min()), max(0., penetration)


def valid_segments(bad, fps, min_seconds=10., buffer_seconds=.5):
    buffered = maximum_filter1d(np.asarray(bad, dtype=np.uint8),
                                size=2 * int(np.ceil(buffer_seconds * fps)) + 1, mode="constant") > 0
    edges = np.diff(np.r_[False, ~buffered, False].astype(int))
    return [(int(start), int(end)) for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if end - start >= int(np.ceil(min_seconds * fps)) + 1]


def forearms(model, poses, z01=False):
    data = mujoco.MjData(model)
    torso = model.body("waist_pitch_link" if z01 else "torso_link").id
    pairs = [(model.body(s + ("_elbow_pitch_link" if z01 else "_elbow_link")).id,
              model.body(s + "_wrist_yaw_link").id) for s in ("left", "right")]
    result = np.empty((len(poses), 2, 3))
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for side, (elbow, wrist) in enumerate(pairs):
            vector = data.xpos[wrist] - data.xpos[elbow]
            result[i, side] = vector @ data.xmat[torso].reshape(3, 3) / np.linalg.norm(vector)
    return result


def condition(src, out, clean_out, velocity_margin=.9):
    if not 0 < velocity_margin <= 1:
        raise ValueError("--velocity-margin must be in (0, 1]")
    if len({p.resolve() for p in (src, out, clean_out)}) != 3:
        raise ValueError("Input, output and selected subset must be separate directories")
    files = sorted(src.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No motions in {src}")
    stale = {p.name for p in out.glob("*.npz")} - {p.name for p in files}
    if stale:
        raise ValueError(f"Output contains clips absent from input; use a fresh output directory: {sorted(stale)}")
    model = mujoco.MjModel.from_xml_path(str(REPO / "assets/z01/robot.xml"))
    names = [model.joint(j).name for j in range(1, model.njnt)]
    limits = model.jnt_range[1:]
    urdf = REPO / "assets/hhtools_robots/z01/z01.urdf"
    declared = {j.get("name"): float(j.find("limit").get("velocity")) for j in ET.parse(urdf).findall("joint")
                if j.find("limit") is not None}
    velocity = np.array([declared[n] for n in names])
    if not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ValueError("Missing/invalid URDF joint velocity limits")
    geom = Geometry(model)
    source_model = mujoco.MjModel.from_xml_path(str(REPO / "data/g1_description/g1_29dof.xml"))
    out.mkdir(parents=True, exist_ok=True)
    # A stale subset must not survive a failed/incomplete rebuild.
    clean_out.mkdir(parents=True, exist_ok=True)
    for old in clean_out.glob("*.npz"):
        if not old.is_symlink():
            raise FileExistsError(f"Refusing to replace a real file in generated subset: {old}")
    for old in clean_out.glob("*.npz"):
        old.unlink()
    rows = {}
    selections = {}
    for index, file in enumerate(files):
        with np.load(file) as z:
            fps = float(z["fps"].reshape(-1)[0])
            p, quat, original = (z[key].astype(float) for key in ("body_pos_w", "body_quat_w", "joint_pos"))
        if (fps != 50. or original.shape[1] != len(names) or len(original) < 2 or p.shape != (len(original), 1, 3)
                or quat.shape != (len(original), 1, 4) or not all(np.isfinite(a).all() for a in (p, quat, original))
                or np.max(np.abs(np.linalg.norm(quat[:, 0], axis=1) - 1)) > .001):
            raise ValueError(f"Invalid Z01 input: {file}")
        q = original.copy()
        # Numerical guard leaves room for OSQP tolerance and float32 export.
        step = velocity_margin * velocity / fps * (1 - 1e-4)
        fitted = []
        for j, name in enumerate(names):
            if (np.max(np.abs(np.diff(q[:, j]))) > step[j] or np.any(q[:, j] < limits[j, 0])
                    or np.any(q[:, j] > limits[j, 1])):
                q[:, j] = fit_signal(q[:, j], limits[j, 0] + 1e-6, limits[j, 1] - 1e-6, step[j])
                fitted.append(name)
        poses = np.concatenate((p[:, 0], quat[:, 0], q), axis=1)
        soles = np.array([geom.heights(pose)[0] for pose in poses])
        needed = np.maximum(0., .001 - soles)  # 1 mm geometric clearance
        lift = fit_signal(np.zeros(len(q)), needed + 1e-6, np.inf, .5 / fps, smoothness=100.) if needed.max() > 0 else needed
        p[:, 0, 2] += lift
        # Verify the exact exported float32 values, not only the optimizer solution.
        p, quat, q = (a.astype(np.float32) for a in (p, quat, q))
        ratio = np.abs(np.diff(q.astype(float), axis=0)) * fps / velocity
        if np.max(ratio) > velocity_margin + 1e-4 or np.any(q < limits[:, 0] - 1e-5) or np.any(q > limits[:, 1] + 1e-5):
            raise RuntimeError(f"Export constraint failure: {file.name}")
        final = np.concatenate((p[:, 0], quat[:, 0], q), axis=1)
        checks = np.array([geom.heights(pose, collisions=True) for pose in final])
        if checks[:, 0].min() < -1e-5:
            raise RuntimeError(f"Export foot-ground failure: {file.name}")
        deviation = np.abs(q - original)
        direction_bad = np.zeros(len(q), dtype=bool)
        source_path = REPO / "data/motions/g1/lafan" / file.name
        direction_check = "not checked: missing source or different timeline"
        if source_path.exists():
            with np.load(source_path) as z:
                if len(z["joint_pos"]) == len(q) and float(z["fps"].reshape(-1)[0]) == fps:
                    source_pose = np.concatenate((z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"]), axis=1)
                    source_direction, target_direction = forearms(source_model, source_pose), forearms(model, final, z01=True)
                    direction_bad = ((source_direction[:, :, 0] > .25) & (target_direction[:, :, 0] < -.25)).any(axis=1)
                    direction_check = "every frame: G1 torso-frame forearm x > .25 and Z01 x < -.25"
        bad = ((checks[:, 1] < -.03) | (checks[:, 2] > .03)
               | (deviation.max(axis=1) > np.radians(20)) | (lift > .1) | direction_bad)
        reasons = []
        if checks[:, 1].min() < -.03:
            reasons.append("body collision box below floor by >3 cm")
        if checks[:, 2].max() > .03:
            reasons.append("self penetration >3 cm")
        if deviation.max() > np.radians(20):
            reasons.append("joint trajectory correction >20 degrees")
        if lift.max() > .1:
            reasons.append("root lift >10 cm")
        if direction_bad.any():
            reasons.append("forearm forward/backward disagreement with G1")
        row = {"frames": len(q), "fps": fps, "fitted_joints": fitted,
               "max_velocity_ratio_to_urdf": float(ratio.max()), "max_joint_correction_deg": float(np.degrees(deviation.max())),
               "max_joint_correction_joint": names[np.unravel_index(deviation.argmax(), deviation.shape)[1]],
               "max_joint_correction_time_s": float(np.unravel_index(deviation.argmax(), deviation.shape)[0] / fps),
               "max_velocity_ratio_by_joint": dict(zip(names, map(float, ratio.max(axis=0)))),
               "rms_joint_correction_deg": float(np.degrees(np.sqrt(np.mean(deviation**2)))),
               "max_root_lift_mm": float(lift.max() * 1000), "min_sole_mm": float(checks[:, 0].min() * 1000),
               "min_body_collision_mm": float(checks[:, 1].min() * 1000),
               "max_self_penetration_mm": float(checks[:, 2].max() * 1000), "excluded_reasons": reasons,
               "bad_frame_count": int(bad.sum()), "selected_segments": [],
               "arm_direction_check": direction_check, "arm_direction_bad_frames": int(direction_bad.sum()),
               "input_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        rows[file.stem] = row
        destination = out / file.name
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as f:
            np.savez(f, fps=np.array([fps]), body_pos_w=p, body_quat_w=quat, joint_pos=q)
        temporary.replace(destination)
        if not bad.any():
            selections[file.stem] = {"path": str(destination.resolve()), "source_clip": file.stem,
                                     "start_frame": 0, "end_frame_exclusive": len(q), "frames": len(q)}
        else:
            # Remove half a second on either side of a bad frame. Do not throw away
            # several good minutes just because one arm swing is infeasible at 50 Hz.
            segments = out / "segments"
            segments.mkdir(exist_ok=True)
            for start, end in valid_segments(bad, fps):
                name = f"{file.stem}__seg{len(row['selected_segments']):03d}"
                path = segments / (name + ".npz")
                with path.with_suffix(".npz.tmp").open("wb") as f:
                    np.savez(f, fps=np.array([fps]), body_pos_w=p[start:end], body_quat_w=quat[start:end], joint_pos=q[start:end])
                path.with_suffix(".npz.tmp").replace(path)
                selections[name] = {"path": str(path.resolve()), "source_clip": file.stem, "start_frame": int(start),
                                    "end_frame_exclusive": int(end), "frames": int(end - start)}
                row["selected_segments"].append(name)
        print(f"[{index + 1}/{len(files)}] {file.stem}: velocity {ratio.max():.4f}x URDF, "
              f"joint change {row['max_joint_correction_deg']:.1f} deg, lift {row['max_root_lift_mm']:.1f} mm, "
              f"{'split: ' + str(len(row['selected_segments'])) + ' usable segments; ' + '; '.join(reasons) if reasons else 'selected whole'}", flush=True)
    selected = list(selections)
    report = {"source": str(src.resolve()), "velocity_margin": velocity_margin,
              "method": "whole-clip constrained joint fits; smooth upward-only root correction; every-frame geometry checks",
              "limits": "No torque, acceleration or dynamic/contact feasibility guarantee. URDF speeds are not hardware verified.",
              "model_sha256": hashlib.sha256((REPO / "assets/z01/robot.xml").read_bytes()).hexdigest(),
              "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
              "conditioner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "urdf_velocity_rad_s": dict(zip(names, map(float, velocity))),
              "calibration_sha256": hashlib.sha256((REPO / "assets/hhtools_robots/z01/r2r_calibration_g1.yaml").read_bytes()).hexdigest(),
              "segment_rule": "remove bad frames plus 0.5 s on each side; keep contiguous runs of at least 10 s",
              "clips": rows, "selected": selected, "selections": selections,
              "excluded_full_clips": [n for n in rows if rows[n]["excluded_reasons"]],
              "excluded": [n for n in rows if n not in selected and not rows[n]["selected_segments"]]}
    (out / "conditioning.json").write_text(json.dumps(report, indent=2))
    for name in selected:
        path = Path(selections[name]["path"])
        # Relative links keep a copied project self-contained.
        (clean_out / (name + ".npz")).symlink_to(os.path.relpath(path, clean_out))
    (clean_out / "selection.json").write_text(json.dumps(report, indent=2))
    print(f"{len(rows)} clips processed; {len(selected)} whole clips/segments pass every-frame screening -> {clean_out}", flush=True)
    if not selected:
        raise RuntimeError("No clips passed screening; inspect conditioning.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--clean-out", type=Path, required=True)
    ap.add_argument("--velocity-margin", type=float, default=.9)
    args = ap.parse_args()
    condition(args.src, args.out, args.clean_out, args.velocity_margin)
