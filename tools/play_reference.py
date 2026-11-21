"""Replay reference npz poses directly, without a policy, PD control or physics stepping.

    uv run python tools/play_reference.py --robot z01 \
        --motion data/motions/z01_lafan_v2/walk2_subject3.npz --loop

Space pauses; R restarts. --video writes an offline, constant-frame-rate preview.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import threading
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="z01", choices=("z01",))
    ap.add_argument("--motion", required=True)
    ap.add_argument("--start", type=float, default=0.)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--speed", type=float, default=1.)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--paused", action="store_true")
    ap.add_argument("--video", help="render to an mp4 instead of opening a window")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--azimuth", type=float, default=0.)
    ap.add_argument("--elevation", type=float, default=-10.)
    ap.add_argument("--distance", type=float, default=3.)
    ap.add_argument("--in-place", action="store_true", help="remove root xy translation for comparison")
    args = ap.parse_args()
    if args.speed <= 0 or args.start < 0 or (args.seconds is not None and args.seconds <= 0):
        ap.error("speed/seconds must be positive; start must be nonnegative")
    if args.fps <= 0 or min(args.width, args.height) <= 0 or args.width % 2 or args.height % 2:
        ap.error("fps must be positive; video dimensions must be positive and even")
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
    elif not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        ap.error("no display available; use --video")

    import mujoco
    import numpy as np
    from partymimic.robot import get_robot

    robot = get_robot(args.robot)
    with np.load(args.motion) as z:
        p = z["body_pos_w"][:, 0].copy()
        quat = z["body_quat_w"][:, 0].copy()
        q = z["joint_pos"].copy()
        fps = float(z["fps"].reshape(-1)[0])
    if q.shape != (len(p), robot.num_joints) or quat.shape != (len(p), 4):
        ap.error("motion dimensions do not match the selected robot")
    if len(p) < 2 or fps <= 0 or not all(np.isfinite(x).all() for x in (p, quat, q)):
        ap.error("motion must contain finite poses, at least two frames and positive fps")
    if args.in_place:
        p[:, :2] = 0.
    end = min((len(p) - 1) / fps, args.start + args.seconds) if args.seconds else (len(p) - 1) / fps
    if args.start >= end:
        ap.error("start is beyond the requested motion interval")

    # Keep the robot geometry and kinematics; add a visual-only floor if the model has none.
    spec = mujoco.MjSpec.from_file(robot.mjcf_path)
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
        spec.worldbody.add_geom(name="reference_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, .05], rgba=[.65, .68, .72, 1], contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        ap.error("reference playback requires one floating root")
    root_adr = int(model.jnt_qposadr[free[0]])
    joint_adr = np.array([model.joint(name).qposadr[0] for name in robot.joint_names])
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0

    def pose(t):
        i = min(int(round(t * fps)), len(p) - 1)
        data.qpos[root_adr:root_adr + 3] = p[i]
        data.qpos[root_adr + 3:root_adr + 7] = quat[i]
        data.qpos[joint_adr] = q[i]
        data.time = i / fps
        mujoco.mj_forward(model, data)
        cam.lookat[:] = p[i] + np.array([0., 0., .12])
        return i

    print(f"REFERENCE ONLY | {robot.name} | {Path(args.motion).name} | {args.start:.2f}..{end:.2f}s", flush=True)
    if args.video:
        from PIL import Image, ImageDraw, ImageFont

        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        font = ImageFont.load_default(size=18)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
               f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "20", args.video]
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                for t in np.arange(args.start, end, args.speed / args.fps):
                    i = pose(t)
                    renderer.update_scene(data, camera=cam, scene_option=opt)
                    im = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(im)
                    draw.rectangle((0, 0, args.width, 60), fill=(245, 245, 245))
                    draw.text((12, 7), f"{robot.name} reference | {Path(args.motion).stem}", font=font, fill=(20, 20, 20))
                    draw.text((12, 32), f"t={i / fps:.2f}s | direct joint poses, no policy", font=font, fill=(20, 20, 20))
                    proc.stdin.write(np.asarray(im).tobytes())
            finally:
                proc.stdin.close()
                code = proc.wait()
            if code:
                raise RuntimeError(f"ffmpeg exited with {code}")
        print(f"Saved {args.video}", flush=True)
        return

    from play import playback_viewer

    paused, restart = threading.Event(), threading.Event()
    if args.paused:
        paused.set()

    def on_key(key):
        if key == ord(" "):
            paused.clear() if paused.is_set() else paused.set()
        elif key == ord("R"):
            restart.set()

    t = args.start
    pose(t)
    print("Space: pause | R: restart | close window: exit", flush=True)
    with playback_viewer(model, data, on_key) as viewer:
        with viewer.lock():
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
            viewer.opt.geomgroup[3] = 0
        last = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            dt, last = now - last, now
            if restart.is_set():
                restart.clear()
                t = args.start
            elif not paused.is_set():
                t += dt * args.speed
            if t > end:
                if not args.loop:
                    break
                t = args.start + (t - args.start) % (end - args.start)
            with viewer.lock():
                pose(t)
                viewer.cam.lookat[:] = cam.lookat
            viewer.sync()
            time.sleep(1 / 120)


if __name__ == "__main__":
    main()
