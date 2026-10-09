"""Command line entry points; rendering and PPO dependencies remain optional."""

import argparse
import json
import time
from pathlib import Path

from ee_jumper.env import SwimEnv
from ee_jumper.gait import rollout
from ee_jumper.model import Water
from ee_jumper.train import load_gait, train_cem


def main():
    parser = argparse.ArgumentParser(description="Minimal MuJoCo Jumper swimming sandbox")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "evaluate", "train"):
        command = commands.add_parser(name)
        command.add_argument("--mode", choices=("planar", "free"), default=None)
        command.add_argument("--seconds", type=float, default=None)
        command.add_argument("--seed", type=int, default=0)
        if name == "train":
            command.add_argument("--generations", type=int, default=12)
            command.add_argument("--population", type=int, default=24)
            command.add_argument("--output", default="runs/gait.json")
        else:
            command.add_argument("--policy", help="JSON gait produced by train")
            if name == "demo":
                command.add_argument("--headless", action="store_true")
                command.add_argument("--gif", help="Save an animated GIF (install [render])")
                command.add_argument("--snapshot", help="Save a final PNG (install [render])")
    args = parser.parse_args()
    if args.command == "train":
        result = train_cem(
            args.output,
            args.generations,
            args.population,
            6.0 if args.seconds is None else args.seconds,
            args.seed,
            args.mode or "planar",
        )
        print(
            json.dumps(
                {
                    "checkpoint": args.output,
                    "baseline_x_m": result["baseline"]["x_m"],
                    "trained_x_m": result["training_result"]["x_m"],
                    "wall_seconds": result["wall_seconds"],
                },
                indent=2,
            )
        )
        return
    artifact = load_gait(args.policy) if args.policy else None
    parameters = artifact["parameters"] if artifact else None
    mode = args.mode or (artifact["mode"] if artifact else "planar")
    seconds = (
        args.seconds
        if args.seconds is not None
        else (artifact["episode_seconds"] if artifact else 6.0)
    )
    water = Water(**artifact["water"]) if artifact else Water()
    render_mode = None
    frames = []
    if args.command == "demo":
        if args.gif or args.snapshot:
            render_mode = "rgb_array"
        elif not args.headless:
            render_mode = "human"
    env = SwimEnv(mode=mode, water=water, episode_seconds=seconds, render_mode=render_mode)
    started = time.perf_counter()

    def callback(step, environment, info):
        if render_mode == "rgb_array" and args.gif and step % 2 == 0:
            from PIL import Image

            frames.append(Image.fromarray(environment.render()))
        elif render_mode == "human":
            time.sleep(max(0, started + (step + 1) * env.dt - time.perf_counter()))

    try:
        result = rollout(env, parameters, args.seed, callback)
        if args.command == "demo":
            if args.gif:
                path = Path(args.gif)
                path.parent.mkdir(parents=True, exist_ok=True)
                frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0)
            if args.snapshot:
                from PIL import Image

                path = Path(args.snapshot)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(env.render()).save(path)
        print(
            json.dumps(
                {"controller": "cem" if artifact else "idle", "mode": mode, **result}, indent=2
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
