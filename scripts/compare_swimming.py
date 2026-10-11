"""Compare two libraries from rest over the same duration in free 3D water."""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from ee_jumper.cycles import DIRECTIONS, DynamicBench, speed_quality
from ee_jumper.learn_cycles import load_library


def evaluate(job):
    name, library, seconds = job
    period = library["period_s"]
    cycles = round(seconds / period)
    if not np.isclose(cycles * period, seconds):
        raise ValueError("Comparison duration must be a whole number of cycles for both libraries")
    bench = DynamicBench(
        period,
        mode="free",
        joint_limit_margin_rad=library.get("joint_limit_margin_rad", 0.0),
        max_excursion=library.get("max_excursion_rad", 0.9),
    )
    metrics = bench.evaluate(library["strokes"][name]["parameters"], 0, cycles)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="examples/strokes-xyz-speed-v1.json")
    parser.add_argument("--after", default="examples/strokes-xyz.json")
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--output", default="runs/speed-comparison.json")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    libraries = [load_library(p) for p in (args.before, args.after)]
    jobs = [(name, lib, args.seconds) for name in DIRECTIONS for lib in libraries]
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
        measured = list(pool.map(evaluate, jobs))
    records = {}
    for i, name in enumerate(DIRECTIONS):
        before, after = measured[2 * i : 2 * i + 2]
        old_speed, new_speed = [speed_quality(m, name)["axial_speed_mps"] for m in (before, after)]
        records[name] = {
            "before": before,
            "after": after,
            "before_axial_speed_mps": old_speed,
            "after_axial_speed_mps": new_speed,
            "speed_ratio": new_speed / max(old_speed, 1e-9),
            "power_ratio": after["mean_power_w"] / max(before["mean_power_w"], 1e-9),
        }
        print(
            name,
            f"{old_speed * 100:.3f} -> {new_speed * 100:.3f} cm/s",
            f"({records[name]['speed_ratio']:.2f}x)",
            flush=True,
        )
    result = {
        "mujoco_version": mujoco.__version__,
        "seconds_from_rest": args.seconds,
        "before_period_s": libraries[0]["period_s"],
        "after_period_s": libraries[1]["period_s"],
        "frame": "initial world axes; whole-robot COM displacement / seconds",
        "conditions": "free 3D, neutral per-link COM buoyancy, still water, same initial home pose",
        "directions": records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    assert all(r["speed_ratio"] > 1 for r in records.values()), "Not every direction improved"
    assert all(not r["after"]["unstable"] for r in records.values())


if __name__ == "__main__":
    main()
