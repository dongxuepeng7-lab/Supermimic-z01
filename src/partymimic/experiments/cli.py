"""Reproducible RA-L entry point: python tools/ral.py --help."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from . import GROUPS
from .parameters import make_manifest, parameterized_robot, read_manifest, sha256

REPO = Path(__file__).resolve().parents[3]


def common_robot(params, plant="identified"):
    from partymimic.robot import get_robot

    return parameterized_robot(get_robot("z01"), params, plant)


def load_context(checkpoint, params, motion_dir=None, device="cpu"):
    import torch
    from partymimic.dr import XiLayout
    from partymimic.kinematics import Kinematics
    from partymimic.motion import MotionLib
    from partymimic.mdp import MDP, RewardCfg
    from .models import make_world_model

    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    a = ck["args"]
    expected_hash = a.get("parameter_manifest_sha256") or sha256(
        a["parameter_manifest"]
    )
    if expected_hash != sha256(params):
        raise ValueError("Checkpoint parameter manifest differs")
    robot = common_robot(params, GROUPS[a["experiment_group"]]["plant"])
    layout = XiLayout(robot, device)
    wm = make_world_model(robot, layout, a).to(device)
    wm.load_state_dict(ck["wm"])
    wm.eval()
    if motion_dir is None:
        return ck, robot, layout, wm
    kin = Kinematics(robot, device)
    motion = MotionLib.from_dir(motion_dir, robot, kin, device)
    motion.input_hashes = {
        p.name: sha256(p) for p in sorted(Path(motion_dir).glob("*.npz"))
    }
    motion.set_interface(a["interface"], robot.tracked_bodies)
    rc, s = RewardCfg(), a["std_scale"]
    mdp = MDP(
        kin,
        motion,
        torch.tensor(robot.default_joint_pos_list(), device=device),
        layout.dim,
        robot.sensing,
        RewardCfg(
            body_pos_std=rc.body_pos_std * s,
            body_ori_std=rc.body_ori_std * s,
            root_pos_std=rc.root_pos_std * s,
            root_ori_std=rc.root_ori_std * s,
        ),
    )
    mdp.feet = [
        robot.tracked_bodies.index(n)
        for n in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    motion.set_ref_contact(mdp.feet)
    return ck, robot, layout, wm, motion, mdp


def preflight(params, motion_dir, group, response=None):
    from partymimic.robot import get_robot
    import numpy as np

    robot = get_robot("z01")
    rows = read_manifest(params, robot)
    missing, invalid, motions = [], [], {}
    for p in sorted(Path(motion_dir).glob("*.npz")):
        try:
            xyz, quat, q, fps = robot.load_motion(str(p))
            if q.ndim != 2 or q.shape[1] != robot.num_joints or q.shape[0] < 40:
                raise ValueError("expected at least 40 frames and 26 joints")
            if xyz.shape != (len(q), 3) or quat.shape != (len(q), 4) or fps != 50:
                raise ValueError("expected synchronized root poses and 50 Hz")
            if not all(np.isfinite(a).all() for a in (xyz, quat, q)):
                raise ValueError("nonfinite motion")
            motions[p.name] = sha256(p)
        except Exception as e:
            invalid.append(dict(path=str(p), reason=str(e)))
    if not motions:
        missing.append("actual Z01 motion NPZ files (symlinks alone are insufficient)")
    if GROUPS[group]["model"] != "base":
        if not response:
            missing.append("frozen paired-response manifest with fit/cal/test sessions")
        else:
            from partymimic.dr import XiLayout
            from .response import ResponseData

            r = common_robot(params)
            for split in ("fit", "cal"):
                ResponseData(response, split, r, XiLayout(r, "cpu"), sha256(params))
    if group in ("D", "C-perm"):
        missing.append(
            "use window-train: requires frozen C checkpoint, calibration and shared fit-start archive"
        )
    return dict(
        group=group,
        ready=not missing and not invalid,
        missing=missing,
        invalid=invalid,
        motion_count=len(motions),
        motions=motions,
        enabled_parameters=sum(r["enabled"] == "1" for r in rows),
        independent_hardware_test_available=False,
        implementation="PartyMimic Z01 local fork; paper baseline naming must reflect this",
    )


def training_command(a):
    cfg = json.loads(Path(a.config).read_text())
    config = cfg["training"]
    report = preflight(a.params, a.motion_dir, a.group, a.responses)
    if not report["ready"]:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    if a.group in ("D", "C-perm"):
        raise ValueError("Use window-train for frozen C branches")
    command = [
        sys.executable,
        "-m",
        "partymimic.train",
        "--experiment_group",
        a.group,
        "--parameter_manifest",
        str(Path(a.params).resolve()),
        "--motion_dir",
        str(Path(a.motion_dir).resolve()),
        "--seed",
        str(a.seed),
        "--log_dir",
        str(Path(a.out).resolve()),
    ]
    if a.responses:
        command += ["--response_manifest", str(Path(a.responses).resolve())]
    if a.protocol:
        command += ["--protocol", str(Path(a.protocol).resolve())]
    for key, value in config.items():
        if isinstance(value, bool):
            command += ["--" + ("" if value else "no-") + key]
        elif value is not None:
            command += ["--" + key, str(value)]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    s = subs.add_parser("groups")
    s = subs.add_parser("manifest")
    s.add_argument("--source", required=True)
    s.add_argument("--out", default="release/ral/params.csv")
    s = subs.add_parser("motions")
    s.add_argument("--zip", required=True)
    s.add_argument("--out")
    s.add_argument("--report")
    s = subs.add_parser("freeze")
    s.add_argument("--params", default="release/ral/params.csv")
    s.add_argument("--motion_dir", required=True)
    s.add_argument("--responses")
    s.add_argument("--config", default="configs/ral/common.json")
    s.add_argument("--out", required=True)
    for name in ("preflight", "train"):
        s = subs.add_parser(name)
        s.add_argument("--params", default="release/ral/params.csv")
        s.add_argument("--motion_dir", default="data/motions/z01_lafan_v2_clean")
        s.add_argument("--group", choices=GROUPS, required=True)
        s.add_argument("--responses")
        if name == "train":
            s.add_argument("--config", default="configs/ral/common.json")
            s.add_argument("--seed", type=int, required=True)
            s.add_argument("--out", required=True)
            s.add_argument("--dry-run", action="store_true")
            s.add_argument("--protocol")
    s = subs.add_parser("pairs")
    s.add_argument("--params", default="release/ral/params.csv")
    s.add_argument("--motion_dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--horizon", type=int, default=32)
    s.add_argument("--blocks-per-clip", type=int, default=4)
    s.add_argument("--epsilon", type=float, default=0.01)
    s = subs.add_parser("probe")
    s.add_argument("--params", default="release/ral/params.csv")
    s.add_argument("--out", required=True)
    for name in ("calibrate", "capture-starts", "window-train"):
        s = subs.add_parser(name)
        s.add_argument("--checkpoint", required=True)
        s.add_argument("--params", default="release/ral/params.csv")
        s.add_argument("--out", required=True)
        s.add_argument("--device", default="cuda:0")
        if name == "calibrate":
            s.add_argument("--responses", required=True)
            s.add_argument("--short", type=int, default=8)
            s.add_argument("--long", type=int, default=32)
            s.add_argument("--edges", default="0.5,1,2")
            s.add_argument("--tolerance", type=float, required=True)
            s.add_argument("--min-blocks", type=int, default=10)
        else:
            s.add_argument("--motion_dir", required=True)
            if name == "window-train":
                s.add_argument("--calibration", required=True)
                s.add_argument("--starts", required=True)
                s.add_argument("--group", choices=("D", "C-perm"), required=True)
                s.add_argument("--iterations", type=int, default=100)
                s.add_argument("--batch-size", type=int, default=64)
                s.add_argument("--seed", type=int, default=1)
                s.add_argument("--lr", type=float, default=1e-4)
            else:
                s.add_argument("--horizon", type=int, default=32)
                s.add_argument("--steps", type=int, default=200)
                s.add_argument(
                    "--responses",
                    help="portable replacement path for the checkpoint's identical response manifest",
                )
    a = p.parse_args()
    if a.command == "groups":
        print(json.dumps(GROUPS, indent=2))
    elif a.command == "manifest":
        from partymimic.robot import get_robot

        rows = make_manifest(get_robot("z01"), a.source, a.out)
        print(f"{len(rows)} entries; {sum(r['enabled'] for r in rows)} enabled")
    elif a.command == "motions":
        from .data import inspect_motion_zip, import_motion_zip

        info = import_motion_zip(a.zip, a.out) if a.out else inspect_motion_zip(a.zip)
        if a.report:
            Path(a.report).write_text(json.dumps(info, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in info.items()
                    if k not in ("records", "missing_link_targets")
                },
                indent=2,
            )
        )
        print("Missing targets:", len(info["missing_link_targets"]))
    elif a.command == "preflight":
        report = preflight(a.params, a.motion_dir, a.group, a.responses)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ready"] else 2
    elif a.command == "freeze":
        from .protocol import freeze

        result = freeze(a.config, a.params, a.motion_dir, a.responses, a.out)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("motions", "scenarios")},
                indent=2,
            )
        )
    elif a.command == "train":
        command = training_command(a)
        import shlex

        print(shlex.join(command), flush=True)
        if not a.dry_run:
            return subprocess.call(command, cwd=REPO)
    elif a.command == "pairs":
        from .data import build_response_pairs

        build_response_pairs(
            common_robot(a.params),
            a.motion_dir,
            a.params,
            a.out,
            a.horizon,
            a.blocks_per_clip,
            a.epsilon,
        )
    elif a.command == "probe":
        from .probe import probe

        print(json.dumps(probe(a.params, a.out), indent=2))
    elif a.command == "calibrate":
        from .response import ResponseData
        from .windows import calibrate_windows

        ck, robot, layout, wm = load_context(a.checkpoint, a.params, device=a.device)
        if ck["args"].get("experiment_group") != "C":
            raise ValueError("Calibration requires C")
        ds = ResponseData(a.responses, "cal", robot, layout, sha256(a.params), a.device)
        result = calibrate_windows(
            wm,
            ds,
            layout.normalize,
            a.checkpoint,
            a.short,
            a.long,
            [float(x) for x in a.edges.split(",")],
            a.tolerance,
            min_blocks=a.min_blocks,
        )
        with open(a.out, "x") as f:
            json.dump(result, f, indent=2)
    elif a.command == "capture-starts":
        from .starts import capture_starts

        context = load_context(a.checkpoint, a.params, a.motion_dir, a.device)
        if a.responses:
            ck = context[0]
            expected = ck["args"].get("response_manifest_sha256") or sha256(
                ck["args"]["response_manifest"]
            )
            if sha256(a.responses) != expected:
                raise ValueError("Replacement response manifest differs")
            ck["args"]["response_manifest"] = a.responses
        capture_starts(*context, a.checkpoint, a.out, a.steps, a.horizon, a.device)
    elif a.command == "window-train":
        from .windows import train_windows

        ck, robot, layout, wm, motion, mdp = load_context(
            a.checkpoint, a.params, a.motion_dir, a.device
        )
        train_windows(
            a.checkpoint,
            a.calibration,
            a.starts,
            a.out,
            a.group,
            robot,
            motion,
            mdp,
            layout,
            a.iterations,
            a.batch_size,
            a.seed,
            a.lr,
            a.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
