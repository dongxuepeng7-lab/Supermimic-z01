"""Explicit GPU integration test. Outputs are synthetic_test, never research results.

Run: uv run python tests/ral_smoke.py /tmp/unique-ral-smoke-directory
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from test_ral_experiments import response_fixture, PARAMS
from partymimic.dr import XiLayout
from partymimic.experiments.parameters import parameterized_robot
from partymimic.robot import get_robot


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    motion = root / "synthetic_motions"
    motion.mkdir()
    robot = get_robot("z01")
    frames = 100
    for name in ("fit_s0", "fit_s1"):
        q = np.repeat(np.array(robot.default_joint_pos_list())[None], frames, 0)
        pos = np.zeros((frames, 1, 3))
        pos[:, :, 2] = 0.97
        quat = np.zeros((frames, 1, 4))
        quat[:, :, 0] = 1
        np.savez(
            motion / f"{name}.npz",
            joint_pos=q,
            body_pos_w=pos,
            body_quat_w=quat,
            fps=np.array(50.0),
        )
    identified = parameterized_robot(robot, PARAMS, "identified")
    responses = response_fixture(
        root / "synthetic_responses", identified, XiLayout(identified, "cpu"), PARAMS
    )
    common = [
        "--parameter_manifest",
        str(PARAMS),
        "--motion_dir",
        str(motion),
        "--seed",
        "1",
        "--iterations",
        "2",
        "--num_envs",
        "2",
        "--buffer_steps",
        "16",
        "--warmup_steps",
        "4",
        "--horizon",
        "2",
        "--wm_window",
        "2",
        "--goal_frames",
        "2",
        "--hist",
        "2",
        "--wm_batch",
        "2",
        "--wm_batches",
        "1",
        "--pi_batch",
        "4",
        "--wm_hidden",
        "16,16",
        "--enc_hidden",
        "16,16",
        "--dec_hidden",
        "16,16",
        "--wm_mirror",
        "0",
        "--lr_decay_ret",
        "0",
        "--lr_decay_by",
        "0",
        "--v_gcons",
        "0",
        "--sil",
        "0",
        "--pi_ema",
        "0",
        "--wm_opt",
        "adam",
        "--pi_opt",
        "adam",
        "--critic_epochs",
        "1",
        "--critic_minibatches",
        "1",
        "--contact_reward",
        "0",
        "--div_k",
        "0",
        "--log_every",
        "1",
        "--ckpt_every",
        "1",
        "--amp",
        "none",
        "--eager",
        "--synthetic_test",
    ]

    def run(command, log):
        print("RUN", log, flush=True)
        with (root / log).open("w") as f:
            result = subprocess.run(
                [sys.executable, *command],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS="2"),
            )
        if result.returncode:
            print((root / log).read_text()[-9000:])
            raise RuntimeError(log)

    for group in ("A0", "A", "A+", "B", "C"):
        args = [
            "-m",
            "partymimic.train",
            *common,
            "--experiment_group",
            group,
            "--log_dir",
            str(root / group),
        ]
        if group in ("A+", "B", "C"):
            args += ["--response_manifest", str(responses)]
        run(args, f"{group}.log")
    a0 = json.loads((root / "A0/run.json").read_text())
    a = json.loads((root / "A/run.json").read_text())
    assert a0["init_sha256"] == a["init_sha256"], "A0/A initialization mismatch"
    for group in ("A0", "A", "A+", "B", "C"):
        initial = torch.load(
            root / group / "initial.pt", weights_only=False, map_location="cpu"
        )
        final = torch.load(
            root / group / "last.pt", weights_only=False, map_location="cpu"
        )
        for net in ("wm", "pi"):
            assert any(
                not torch.equal(v, final[net][k])
                for k, v in initial[net].items()
                if "weight" in k
            ), f"{group}/{net} did not update"
    ck = str(root / "C/last.pt")
    cal = str(root / "calibration.json")
    run(
        [
            "tools/ral.py",
            "calibrate",
            "--checkpoint",
            ck,
            "--params",
            str(PARAMS),
            "--responses",
            str(responses),
            "--short",
            "1",
            "--long",
            "2",
            "--edges",
            ".5",
            "--tolerance",
            "1000",
            "--min-blocks",
            "1",
            "--out",
            cal,
        ],
        "calibrate.log",
    )
    run(
        [
            "tools/ral.py",
            "capture-starts",
            "--checkpoint",
            ck,
            "--params",
            str(PARAMS),
            "--motion_dir",
            str(motion),
            "--horizon",
            "2",
            "--steps",
            "8",
            "--out",
            str(root / "starts"),
        ],
        "starts.log",
    )
    for group in ("C-perm", "D"):
        run(
            [
                "tools/ral.py",
                "window-train",
                "--checkpoint",
                ck,
                "--params",
                str(PARAMS),
                "--motion_dir",
                str(motion),
                "--calibration",
                cal,
                "--starts",
                str(root / "starts/manifest.json"),
                "--group",
                group,
                "--iterations",
                "2",
                "--batch-size",
                "4",
                "--out",
                str(root / group),
            ],
            f"{group}.log",
        )
    runs = [json.loads((root / g / "run.json").read_text()) for g in ("D", "C-perm")]
    for key in (
        "init_sha256",
        "start_schedule_sha256",
        "assigned_imagined_steps",
        "computed_imagined_steps",
    ):
        assert runs[0][key] == runs[1][key], key
    run(
        [
            "-m",
            "partymimic.evaluate",
            "--run_dir",
            str(root / "D"),
            "--ckpt",
            str(root / "D/last.pt"),
            "--motion_dir",
            str(motion),
            "--parameter_manifest",
            str(PARAMS),
            "--plant",
            "identified",
            "--out",
            str(root / "evaluation.json"),
        ],
        "evaluation.log",
    )
    result = dict(
        synthetic_test=True,
        groups=["A0", "A", "A+", "B", "C", "C-perm", "D"],
        A0_A_same_initialization=True,
        world_model_and_policy_updated=True,
        frozen_window_same_starts_and_budget=True,
        checkpoint_reload_and_evaluation=True,
    )
    (root / "smoke_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
