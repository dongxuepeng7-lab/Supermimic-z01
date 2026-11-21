"""Play a trained policy in an interactive MuJoCo window, using the training simulator.

    uv run python tools/play.py --ckpt logs/z01_lafan_v2/ckpt_012000.pt \
        --motion data/motions/z01_lafan_v2_clean/walk2_subject3__seg000.npz --loop

Space pauses/resumes, R restarts the motion. Mouse controls are MuJoCo's standard camera controls.
The viewer displays the policy's simulated robot, not a replay of the reference joint angles.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time


@contextmanager
def playback_viewer(model, data, key_callback):
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    try:
        yield viewer
    finally:
        viewer.close()
        # close() only requests exit. Let the render thread release its OpenGL context before
        # Python's GLFW atexit handler terminates GLFW (otherwise it can crash during shutdown).
        deadline = time.monotonic() + 5.0
        while viewer.m is not None and time.monotonic() < deadline:
            time.sleep(0.01)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="policy checkpoint (.pt), e.g. ckpt_012000.pt or last.pt")
    parser.add_argument("--motion", required=True, help="one reference motion (.npz or the robot's native format)")
    parser.add_argument("--robot", default=None, choices=("z01",), help="default: the checkpoint's robot (z01)")
    parser.add_argument("--loop", action="store_true", help="restart one second after the motion ends or the robot fails")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = half speed)")
    parser.add_argument("--paused", action="store_true", help="start paused; press Space to run")
    parser.add_argument("--headless", action="store_true", help="run without a window or real-time pacing")
    parser.add_argument("--steps", type=int, default=None, help="exit after this many control steps (for a smoke check)")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.headless and args.loop and args.steps is None:
        parser.error("--headless --loop requires --steps")
    if args.headless and args.paused:
        parser.error("--paused requires a viewer")
    if not args.headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("no desktop display found; run in a desktop terminal, or use --headless")
    for path in (args.ckpt, args.motion):
        if not Path(path).is_file():
            parser.error(f"file not found: {path}")
    return args


def main():
    args = parse_args()

    import mujoco
    import torch

    import partymimic.networks as networks
    from partymimic.dr import XiLayout
    from partymimic.env import TrackingSim, make_sim
    from partymimic.kinematics import Kinematics
    from partymimic.mdp import MAX_STEP_REWARD, MDP
    from partymimic.motion import CONTROL_DT, MotionLib, load_npz
    from partymimic.networks import Policy, policy_kwargs
    from partymimic.robot import get_robot

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run from an environment that can access the NVIDIA GPU.")
    torch.manual_seed(0)
    dev = "cuda:0"
    # Keep the world model and optimizer states off the GPU: playback needs only the policy.
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["args"]
    robot = get_robot(args.robot or cfg.get("robot", "z01"))
    kin = Kinematics(robot, dev)
    loader = load_npz if Path(args.motion).suffix.lower() == ".npz" else robot.load_motion
    motion = MotionLib([args.motion], kin, dev, loader)
    motion.set_interface(cfg.get("interface", "joint"), robot.tracked_bodies)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim,
              robot.sensing)
    K, H = cfg["goal_frames"], cfg["hist"]
    widths = lambda key: tuple(int(x) for x in cfg.get(key, "2048,1024,512,256").split(","))
    policy = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=H,
                    enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"),
                    **policy_kwargs(cfg, robot)).to(dev)
    policy.load_state_dict(ck["pi"])
    policy.eval()
    del ck
    networks.AMP_DTYPE = torch.bfloat16 if cfg.get("amp", "none") == "bf16" else None
    env = TrackingSim(make_sim(robot), 1, robot, motion, mdp, dev, eval_mode=True,
                      reset_noise=False, randomize=False)
    ids = torch.arange(1, device=dev)
    history, episode_steps, total_steps, reward = None, 0, 0, 0.0
    finished_at = None
    paused, restart = threading.Event(), threading.Event()
    if args.paused:
        paused.set()

    def on_key(key):
        if key == ord(" "):
            paused.clear() if paused.is_set() else paused.set()
        elif key == ord("R"):
            restart.set()

    def reset():
        nonlocal history, episode_steps, reward, finished_at
        env.done_eval.zero_()
        env.failed_eval.zero_()
        env.reset(ids, motion.clip_start.clone())
        history, episode_steps, reward, finished_at = None, 0, 0.0, None

    def step():
        nonlocal history, episode_steps, total_steps, reward, finished_at
        obs = env.observe()
        history = obs.unsqueeze(1).repeat(1, H, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
        action = policy(history, motion.goal(env.f, K))
        reward += env.step(action)["r"].item()
        episode_steps += 1
        total_steps += 1
        if bool(env.done_eval.item()):
            finished_at = time.monotonic()
            norm = reward / ((motion.clip_len[0].item() - 1) * MAX_STEP_REWARD)
            print(f"{motion.names[0]}: steps={episode_steps}, return={norm:.4f}, "
                  f"failed={bool(env.failed_eval.item())}", flush=True)

    with torch.inference_mode():
        reset()
        print(f"Playing {args.ckpt} | robot={robot.name} | motion={motion.names[0]} "
              f"({(motion.num_frames - 1) * CONTROL_DT:.1f}s)", flush=True)
        if args.headless:
            while args.steps is None or total_steps < args.steps:
                if finished_at is not None:
                    if not args.loop:
                        break
                    reset()
                step()
            print(f"Completed {total_steps} control steps.", flush=True)
            return

        # The viewer gets a CPU copy of the same compiled model/state used by MuJoCo Warp.
        model = env.sim.mj_model
        data = mujoco.MjData(model)

        def update_view():
            data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
            data.time = episode_steps * CONTROL_DT
            mujoco.mj_forward(model, data)

        update_view()
        print("Space: pause/resume | R: restart | mouse: rotate/pan/zoom | close window: exit", flush=True)
        with playback_viewer(model, data, on_key) as viewer:
            with viewer.lock():
                root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{robot.tracked_bodies[0]}")
                if root_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = root_id
                viewer.cam.lookat[:] = env.state()[0][0].cpu().numpy()
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, 135.0, -15.0
                viewer.opt.geomgroup[3] = 0  # hide collision geometry; show the robot meshes
            while viewer.is_running():
                started = time.monotonic()
                if restart.is_set():
                    restart.clear()
                    reset()
                if not paused.is_set():
                    if finished_at is not None and args.loop and started - finished_at >= 1.0:
                        reset()
                    if finished_at is None:
                        step()
                with viewer.lock():
                    update_view()
                viewer.sync()
                if args.steps is not None and total_steps >= args.steps:
                    break
                time.sleep(max(0.0, CONTROL_DT / args.speed - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
