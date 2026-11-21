"""Audited joint-output parameter overrides; source models are never rewritten."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path

FIELDS = (
    "joint_id",
    "param_id",
    "repo_key",
    "source_side",
    "unit",
    "nominal_value",
    "identified_value",
    "effective_A0",
    "effective_A",
    "enabled",
    "source_ref",
    "source_sha256",
    "conversion",
    "reason",
    "validation_status",
)
UNITS = {
    "armature": "kg*m^2",
    "damping": "N*m*s/rad",
    "frictionloss": "N*m",
    "load_offset": "N*m",
    "command_delay_sec": "s",
}
SUPPORTED = ("armature", "damping", "frictionloss")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def nominal_parameters(robot):
    model = robot.robot_cfg().build().compile()
    result = {}
    for name in robot.joint_names:
        j = model.joint(name)
        dof = int(model.jnt_dofadr[j.id])
        result[name] = dict(
            armature=float(model.dof_armature[dof]),
            damping=float(model.dof_damping[dof]),
            frictionloss=float(model.dof_frictionloss[dof]),
            load_offset=0.0,
            command_delay_sec=0.0,
        )
    return result


def make_manifest(robot, source, destination):
    """Conservatively select explicitly validated runtime entries, preserving every exclusion."""
    source, destination = Path(source), Path(destination)
    data = json.loads(source.read_text())
    entries = {r["joint"]: r for r in data["rows"]}
    if len(entries) != len(data["rows"]) or set(entries) - set(robot.joint_names):
        raise ValueError("Duplicate or unknown joints in identification source")
    nominal = nominal_parameters(robot)
    rows = []
    for name in robot.joint_names:
        record = entries.get(name, {})
        status = record.get("status", "missing")
        validated = "通过" in status and not any(
            s in status for s in ("未通过", "复用", "微小差异", "未改善")
        )
        for param, unit in UNITS.items():
            value = record.get("runtime", {}).get(param, "")
            enabled = validated and param in SUPPORTED and value != ""
            reason = (
                "validated runtime snapshot; original trial logs not bundled"
                if enabled
                else (
                    "no identified value"
                    if value == ""
                    else "validation unresolved: " + status
                )
            )
            if value != "" and param not in SUPPORTED:
                reason = (
                    "excluded: external load is fixture-dependent, not controller feedforward"
                    if param == "load_offset"
                    else "excluded: asynchronous recorded-event delay is not equivalent to this continuous-PD plant"
                )
            n = nominal[name][param]
            rows.append(
                dict(
                    zip(
                        FIELDS,
                        (
                            name,
                            param,
                            "dof_" + param if param in SUPPORTED else "unsupported",
                            "joint_output",
                            unit,
                            n,
                            value,
                            n,
                            value if enabled else n,
                            int(enabled),
                            str(source),
                            sha256(source),
                            "identity" if param in SUPPORTED else "not_applied",
                            reason,
                            status,
                        ),
                    )
                )
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def read_manifest(path, robot):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {(j, p) for j in robot.joint_names for p in UNITS}
    if (
        len(rows) != len(expected)
        or {(r["joint_id"], r["param_id"]) for r in rows} != expected
    ):
        raise ValueError(
            "Parameter manifest must contain each joint/parameter exactly once"
        )
    nominal = nominal_parameters(robot)
    sources = {}
    for r in rows:
        j, p = r["joint_id"], r["param_id"]
        if (
            r["enabled"] not in ("0", "1")
            or r["unit"] != UNITS[p]
            or r["source_side"] != "joint_output"
        ):
            raise ValueError(
                f"Invalid enabled flag, units or transmission side: {j}/{p}"
            )
        for key in ("nominal_value", "effective_A0", "effective_A"):
            if not math.isfinite(float(r[key])):
                raise ValueError(f"Nonfinite {j}/{p}/{key}")
        n = nominal[j][p]
        if not math.isclose(
            float(r["nominal_value"]), n, abs_tol=1e-9
        ) or not math.isclose(float(r["effective_A0"]), n, abs_tol=1e-9):
            raise ValueError(
                f"Nominal asset changed: {j}/{p}; regenerate and review manifest"
            )
        if r["enabled"] == "1":
            if (
                p not in SUPPORTED
                or r["conversion"] != "identity"
                or r["repo_key"] != "dof_" + p
            ):
                raise ValueError(f"Unsupported parameter mapping: {j}/{p}")
            if (
                not r["identified_value"]
                or not r["source_ref"]
                or not r["source_sha256"]
            ):
                raise ValueError(f"Missing identification provenance: {j}/{p}")
            value = float(r["identified_value"])
            if (
                value < 0
                or not math.isfinite(value)
                or value != float(r["effective_A"])
            ):
                raise ValueError(f"Invalid identified value: {j}/{p}")
            if sha256(r["source_ref"]) != r["source_sha256"]:
                raise ValueError(f"Identification source changed: {r['source_ref']}")
            if r["source_ref"] not in sources:
                sources[r["source_ref"]] = {
                    entry["joint"]: entry
                    for entry in json.loads(Path(r["source_ref"]).read_text())["rows"]
                }
            entry = sources[r["source_ref"]][j]
            if (
                float(entry["runtime"][p]) != value
                or entry["status"] != r["validation_status"]
            ):
                raise ValueError(f"Manifest differs from its cited source: {j}/{p}")
        elif float(r["effective_A"]) != n:
            raise ValueError(f"Disabled entry changes plant: {j}/{p}")
    return rows


def parameterized_robot(robot, path, plant):
    if plant not in ("nominal", "identified"):
        raise ValueError(plant)
    rows = read_manifest(path, robot)
    changes = {j: {} for j in robot.joint_names}
    if plant == "identified":
        for r in rows:
            if r["enabled"] == "1":
                changes[r["joint_id"]][r["param_id"]] = float(r["effective_A"])
        if not any(changes.values()):
            raise ValueError("Identified group has no enabled identified parameters")
    if plant == "nominal":
        return robot
    original_cfg = robot.robot_cfg

    def robot_cfg():
        cfg = original_cfg()
        original_spec = cfg.spec_fn

        def spec_fn():
            spec = original_spec()
            for name, values in changes.items():
                joint = spec.joint(name)
                for p, v in values.items():
                    if p == "damping":
                        joint.damping = [
                            v,
                            0.0,
                            0.0,
                        ]  # MuJoCo 3.11: polynomial damping coefficients
                    else:
                        setattr(joint, p, v)
            return spec

        # mjlab's actuator compilation overwrites armature; set it in BOTH places.
        actuators = tuple(
            dataclasses.replace(a, armature=changes[n].get("armature", a.armature))
            for n, a in zip(robot.joint_names, cfg.articulation.actuators, strict=True)
        )
        return dataclasses.replace(
            cfg,
            spec_fn=spec_fn,
            articulation=dataclasses.replace(cfg.articulation, actuators=actuators),
        )

    return dataclasses.replace(robot, robot_cfg=robot_cfg)
