"""Prepare Z01 training/retarget assets, or convert HHT's Z01 CSV output to training npz.

    uv run python tools/data/z01_data.py prepare
    uv run python tools/data/z01_data.py convert data/raw/z01_v2/csv data/motions/z01_lafan_v2_ik

The converter does not constrain speed: finalize these IK results with condition_z01.py.
"""

import argparse
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "Z01_v1.6_whole_body"
ASSETS = REPO / "assets/z01"


def prepare():
    cfg = json.loads((ASSETS / "control.json").read_text())
    tree = ET.parse(SOURCE / "mjcf/Z01_v1.6.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", "../../Z01_v1.6_whole_body/meshes")
    root.find("option").set("timestep", "0.005")
    world = root.find("worldbody")
    for geom in list(world.findall("geom")):
        world.remove(geom)  # terrain belongs to mjlab's scene, not the robot entity
    root.find("default/geom").set("contype", "1")
    root.find("default/geom").set("conaffinity", "1")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for body1, body2 in cfg.get("collision_excludes", []):
        ET.SubElement(contact, "exclude", body1=body1, body2=body2)
    old = root.find("actuator")
    root.remove(old)
    actuators = ET.SubElement(root, "actuator")
    joints = [j for j in root.findall(".//body/joint") if j.get("type") != "free"]
    if set(cfg["default_joint_pos"]) != {j.get("name") for j in joints}:
        raise ValueError("control.json must define the standing angle of every Z01 joint")
    for joint in joints:
        name = joint.get("name")
        group, = [g for g in cfg["actuators"] if re.fullmatch(g["pattern"], name)]
        joint.set("armature", str(group["armature"]))
        joint.set("damping", str(cfg["passive_damping"]))
        joint.set("frictionloss", str(cfg["frictionloss"]))
        ET.SubElement(actuators, "position", name=name, joint=name, kp=str(group["kp"]), kv=str(group["kd"]),
                      forcelimited="true", forcerange=joint.get("actuatorfrcrange"))
    standing = [cfg["default_joint_pos"][j.get("name")] for j in joints]
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframes, "key", name="standing", qpos=" ".join(map(str, [0, 0, cfg["root_height"], 1, 0, 0, 0, *standing])),
                  ctrl=" ".join(map(str, standing)))
    ET.indent(tree, space="  ")
    tree.write(ASSETS / "robot.xml", encoding="unicode")
    mujoco.MjModel.from_xml_path(str(ASSETS / "robot.xml"))

    # Keep retargeting independent from the training controller and scene. The URDF is a generated
    # copy; all original geometry and inertias are retained, and the mesh directory is staged later.
    preset = REPO / "assets/hhtools_robots/z01"
    preset.mkdir(parents=True, exist_ok=True)
    urdf = ET.parse(SOURCE / "urdf/Z01_v1.6.urdf")
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", "meshes/" + Path(mesh.get("filename")).name)
    ET.indent(urdf, space="  ")
    urdf.write(preset / "z01.urdf", encoding="unicode")
    calibration_path = preset / "r2r_calibration_g1.yaml"
    if not calibration_path.exists():
        raise FileNotFoundError(f"Restore {calibration_path}; an all-zero G1-to-Z01 arm calibration is invalid")
    print("Prepared assets/z01/robot.xml and assets/hhtools_robots/z01; existing R2R calibration preserved.")


def convert(src, out):
    from hht_csv import load_hht_csv  # select columns by Z01 joint name

    model = mujoco.MjModel.from_xml_path(str(ASSETS / "robot.xml"))
    names = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    files = sorted(src.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No retargeted CSV files found in {src}")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for file in files:
        if file.with_suffix(".csv.skip").exists():
            print("skip failed retarget:", file)
            continue
        qpos, fps = load_hht_csv(file, names)
        if fps != 50. or len(qpos) < 2 or not np.isfinite(qpos).all():
            raise ValueError(f"Invalid Z01 motion {file}: fps={fps}, frames={len(qpos)}")
        if np.any(qpos[:, 7:] < model.jnt_range[1:, 0] - .01) or np.any(qpos[:, 7:] > model.jnt_range[1:, 1] + .01):
            raise ValueError(f"Joint limits exceeded in {file}")
        destination = out / (file.stem.removeprefix("x__") + ".npz")
        np.savez(destination, fps=np.array([fps]), joint_pos=qpos[:, 7:].astype(np.float32),
                 body_pos_w=qpos[:, None, :3].astype(np.float32), body_quat_w=qpos[:, None, 3:7].astype(np.float32))
        written.append({"file": destination.name, "frames": len(qpos)})
    if not written:
        raise SystemExit("No valid Z01 clips were produced; inspect the retarget failure log.")
    (out / "index.json").write_text(json.dumps(written, indent=2))
    print(f"{len(written)} Z01 clips -> {out}")


def qa(directory, clean_out=None):
    """Sample physical reference quality separately from CSV/npz format validation.

    A successful retarget does not guarantee collision-free, dynamically feasible motion.
    The report intentionally retains difficult clips instead of silently deleting them.
    """
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "robot.xml"))
    data = mujoco.MjData(model)
    feet = [model.geom(name).id for name in ("left_ankle_roll_link_collision", "right_ankle_roll_link_collision")]
    collision_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and
                       (model.geom_contype[g] or model.geom_conaffinity[g])]
    if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in collision_geoms):
        raise ValueError("Z01 QA currently expects the source model's box collision geometry")
    report = {}
    for file in sorted(directory.glob("*.npz")):
        z = np.load(file)
        qpos = np.concatenate((z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"]), -1)
        soles, body_floor, penetration = [], [], 0.
        for frame in np.linspace(0, len(qpos) - 1, min(100, len(qpos)), dtype=int):
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            height = min(float(data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g])
                         for g in feet)
            soles.append(height)
            body_floor.append(min(float(data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g])
                                  for g in collision_geoms))
            penetration = max(penetration, max((-float(c.dist) for c in data.contact), default=0.))
        report[file.stem] = {"frames": len(qpos), "sampled_sole_min_m": min(soles),
                             "sampled_sole_median_m": float(np.median(soles)),
                             "sampled_body_floor_min_m": min(body_floor),
                             "sampled_self_penetration_max_m": penetration}
    if not report:
        raise SystemExit(f"No Z01 npz motions in {directory}")
    warnings = [name for name, row in report.items()
                if row["sampled_body_floor_min_m"] < -.03 or row["sampled_self_penetration_max_m"] > .03]
    output = {"sampling": "up to 100 evenly spaced frames per clip; diagnostics, not an exhaustive feasibility test",
              "warning_threshold_m": .03, "warning_clips": warnings, "clips": report}
    (directory / "qa.json").write_text(json.dumps(output, indent=2))
    print(f"Z01 QA: {len(report)} clips, {len(warnings)} with sampled floor/self penetration > 3 cm. "
          f"Report: {directory / 'qa.json'}")
    if clean_out is not None:
        if clean_out.resolve() == directory.resolve():
            raise ValueError("QA output subset must differ from the full motion directory")
        clean_out.mkdir(parents=True, exist_ok=True)
        selected = [name for name in report if name not in warnings]
        # Only replace our generated symlinks, never delete a user's real motion files.
        for old in clean_out.glob("*.npz"):
            if not old.is_symlink():
                raise FileExistsError(f"Refusing to replace a non-generated motion: {old}")
        for old in clean_out.glob("*.npz"):
            old.unlink()
        for name in selected:
            target = (directory / (name + ".npz")).resolve()
            (clean_out / (name + ".npz")).symlink_to(os.path.relpath(target, clean_out))
        (clean_out / "selection.json").write_text(json.dumps({"source": str(directory.resolve()),
            "sampling": output["sampling"], "threshold_m": .03, "selected": selected, "excluded": warnings}, indent=2))
        if not selected:
            raise SystemExit("No clips passed sampled QA; inspect qa.json and the retarget calibration.")
        print(f"{len(selected)} sampled-QA clips -> {clean_out} (not a guarantee of dynamic feasibility)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    p = commands.add_parser("convert")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p = commands.add_parser("qa")
    p.add_argument("directory", type=Path)
    p.add_argument("--clean-out", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "convert":
        convert(args.src, args.out)
    else:
        qa(args.directory, args.clean_out)
