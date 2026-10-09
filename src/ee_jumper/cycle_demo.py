"""Replay one learned cycle on a tether or in free 3D water."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from ee_jumper.cycles import DIRECTIONS, DynamicBench
from ee_jumper.learn_cycles import load_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--direction", choices=DIRECTIONS, default="+x")
    parser.add_argument("--mode", choices=("free", "tethered"), default="free")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--idle", action="store_true", help="Hold the home pose to check buoyancy")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gif")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    library = load_library(args.library)
    bench = DynamicBench(
        library["period_s"], mode=args.mode, max_excursion=library.get("max_excursion_rad", 0.9)
    )
    bench.reset()
    viewer = renderer = None
    frames = []
    if args.gif:
        from PIL import Image

        renderer = mujoco.Renderer(bench.model, height=360, width=480)
    elif not args.headless:
        from ee_jumper.viewer import PassiveViewer

        viewer = PassiveViewer(bench.model, bench.data)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 135, -35
    start = time.perf_counter()
    base = bench.model.body("base_link").id

    def display(step, b):
        if renderer is not None and step % 10 == 0:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = b.data.xpos[base]
            camera.distance, camera.azimuth, camera.elevation = 1.3, 135, -35
            renderer.update_scene(b.data, camera)
            frames.append(Image.fromarray(renderer.render()))
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            if step % 2 == 0:
                viewer.cam.lookat[:] = b.data.xpos[base]
                viewer.sync()
            time.sleep(max(0, start + (step + 1) * b.dt - time.perf_counter()))

    try:
        result = bench.evaluate(
            np.zeros(36) if args.idle else library["strokes"][args.direction]["parameters"],
            warmup_cycles=0,
            measurement_cycles=args.cycles,
            callback=display,
        )
        if frames:
            path = Path(args.gif)
            path.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(
            json.dumps(
                {
                    "direction": "idle" if args.idle else args.direction,
                    "mode": args.mode,
                    "displacement_m": result["displacement_m"],
                    "final_quaternion_wxyz": result["final_quaternion_wxyz"],
                    "unstable": result["unstable"],
                },
                indent=2,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
