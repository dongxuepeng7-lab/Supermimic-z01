"""Freeze and verify inputs before multi-machine paired experiments."""

import json
from pathlib import Path

from .parameters import sha256
from .runtime import object_hash

REPO = Path(__file__).resolve().parents[3]


def code_hash():
    return object_hash(
        {
            str(p.relative_to(REPO)): sha256(p)
            for p in sorted((REPO / "src").rglob("*.py"))
        }
    )


def asset_hash():
    paths = [
        p
        for base in (REPO / "assets/z01", REPO / "Z01_v1.6_whole_body")
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() != ".usd"
    ]
    return object_hash({str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)})


def freeze(config, params, motion_dir, responses, output):
    from .cli import preflight

    r = preflight(params, motion_dir, "C" if responses else "A0", responses)
    if not r["ready"]:
        raise ValueError(json.dumps(r, ensure_ascii=False))
    config_data = json.loads(Path(config).read_text())
    groups = ["A0", "A", "A+", "B", "C", "C-perm", "D"] if responses else ["A0", "A"]
    scenarios = [
        dict(
            scenario_id=f"{condition}:{motion}",
            condition=condition,
            motion_id=motion,
            motion_sha256=digest,
            start_frame=0,
            eval_seed=0,
            success="complete_reference_without_tracking_termination",
        )
        for condition in ("nominal", "identified")
        for motion, digest in sorted(r["motions"].items())
    ]
    protocol = dict(
        schema_version=1,
        implementation="PartyMimic Z01 local fork",
        groups=groups,
        training_seeds=[1, 2],
        training=config_data["training"],
        window_candidates=config_data.get("window_candidates", dict(short=8, long=32)),
        code_sha256=code_hash(),
        asset_sha256=asset_hash(),
        config_sha256=sha256(config),
        params_sha256=sha256(params),
        motions=r["motions"],
        motion_sha256=object_hash(r["motions"]),
        response_sha256=sha256(responses) if responses else None,
        scenarios=scenarios,
        scenarios_sha256=object_hash(scenarios),
        formal_results=False,
        heldout_status="requires independently justified actuator ranges",
        hardware_status="raw independent hardware sessions unavailable",
    )
    with open(output, "x") as f:
        json.dump(protocol, f, indent=2)
    return protocol


def verify(path, args):
    protocol = json.loads(Path(path).read_text())
    if (
        args.experiment_group not in protocol["groups"]
        or args.seed not in protocol["training_seeds"]
    ):
        raise ValueError("Group or seed outside frozen protocol")
    for key, value in protocol["training"].items():
        if getattr(args, key) != value:
            raise ValueError(f"Frozen training setting changed: {key}")
    actual = dict(
        code_sha256=code_hash(),
        asset_sha256=asset_hash(),
        params_sha256=sha256(args.parameter_manifest),
        response_sha256=sha256(args.response_manifest)
        if args.response_manifest
        else None,
    )
    for key, value in actual.items():
        # Base groups do not consume response labels, but may share a seven-group protocol.
        if (
            key == "response_sha256"
            and args.experiment_group in ("A0", "A")
            and value is None
        ):
            continue
        if protocol[key] != value:
            raise ValueError(f"Frozen input changed: {key}")
    motions = {p.name: sha256(p) for p in sorted(Path(args.motion_dir).glob("*.npz"))}
    if protocol["motions"] != motions:
        raise ValueError("Frozen motion library changed")
    return protocol
