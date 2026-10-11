"""Measure integration sensitivity and closed-loop free XYZ commands."""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from ee_jumper.control import CycleController
from ee_jumper.cycles import DIRECTIONS, DynamicBench, quality
from ee_jumper.learn_cycles import load_library
from ee_jumper.learn_displacement import displacement_quality
from ee_jumper.model import step_water


def validate_one(job):
    name, library, speed = job
    reference = library["strokes"][name]
    bench = DynamicBench(
        library["period_s"],
        control_dt=0.005,
        joint_limit_margin_rad=library.get("joint_limit_margin_rad", 0.0),
        max_excursion=library.get("max_excursion_rad", 0.9),
    )
    bench.model.opt.timestep = 0.0005
    bench.substeps = 10
    fine = bench.evaluate(reference["parameters"], warmup_cycles=2, measurement_cycles=2)
    relative_error = float(
        np.linalg.norm(np.array(fine["mean_force_n"]) - reference["metrics"]["mean_force_n"])
        / np.linalg.norm(reference["metrics"]["mean_force_n"])
    )
    free_fine = None
    if library["format"] == "ee-jumper-strokes-v2":
        free_bench = DynamicBench(
            library["period_s"],
            control_dt=0.005,
            mode="free",
            joint_limit_margin_rad=library.get("joint_limit_margin_rad", 0.0),
            max_excursion=library["max_excursion_rad"],
        )
        free_bench.model.opt.timestep = 0.0005
        free_bench.substeps = 10
        free_fine = free_bench.evaluate(reference["parameters"], 0, round(6 / library["period_s"]))
    bench = DynamicBench(
        library["period_s"],
        control_dt=0.01,
        mode="free",
        joint_limit_margin_rad=library.get("joint_limit_margin_rad", 0.0),
        max_excursion=library.get("max_excursion_rad", 0.9),
    )
    bench.reset()
    model, data = bench.model, bench.data
    base = model.body("base_link").id
    controller = CycleController(model, library)
    start = data.subtree_com[base].copy()
    last_com = start.copy()
    velocity = np.zeros(3)
    selected = {}
    command = speed * DIRECTIONS[name]
    traces = []
    for step in range(1000):
        now = step * 0.01
        controller.set_command(command, now)
        result = controller.step(0.01, velocity, now)
        selected[result.active_stroke or "idle"] = (
            selected.get(result.active_stroke or "idle", 0) + 1
        )
        data.ctrl[:] = result.joint_positions
        step_water(model, data, nstep=10)
        if data.time < now + 0.01 - 1e-7 or not np.isfinite(data.qpos).all():
            raise RuntimeError(f"Simulation instability for {name}")
        mujoco.mj_forward(model, data)
        velocity = data.xmat[base].reshape(3, 3).T @ ((data.subtree_com[base] - last_com) / 0.01)
        last_com = data.subtree_com[base].copy()
        if step % 100 == 99:
            traces.append(
                {
                    "seconds": (step + 1) * 0.01,
                    "displacement_m": (last_com - start).tolist(),
                    "body_velocity_mps": velocity.tolist(),
                }
            )
    return name, {
        "fine_integration": fine,
        "force_relative_difference": relative_error,
        "fine_quality": quality(fine, name, library["minimum_force_n"]),
        "free_fine_integration": free_fine,
        "free_fine_quality": displacement_quality(free_fine, name) if free_fine else None,
        "displacement_relative_difference": (
            float(
                np.linalg.norm(
                    np.array(free_fine["displacement_m"])
                    - reference["free_validation"]["displacement_m"]
                )
                / np.linalg.norm(reference["free_validation"]["displacement_m"])
            )
            if free_fine
            else None
        ),
        "closed_loop_10s": {
            "command_body_mps": command.tolist(),
            "world_displacement_m": (last_com - start).tolist(),
            "initial_axis_progress_m": float((last_com - start) @ DIRECTIONS[name]),
            "final_quaternion_wxyz": data.qpos[3:7].tolist(),
            "selected_control_steps": selected,
            "trace": traces,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--output", default="runs/cycle-validation.json")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or not 0 < args.speed <= 0.1:
        parser.error("--speed must be in (0, .1]")
    library = load_library(args.library)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        records = dict(pool.map(validate_one, [(name, library, args.speed) for name in DIRECTIONS]))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")
    for name, record in records.items():
        print(
            name,
            "force_difference",
            record["force_relative_difference"],
            "progress_m",
            record["closed_loop_10s"]["initial_axis_progress_m"],
            flush=True,
        )
    assert all(r["force_relative_difference"] < 0.05 for r in records.values())
    if library["format"] == "ee-jumper-strokes-v2":
        assert all(r["free_fine_quality"]["valid"] for r in records.values())
        assert all(r["displacement_relative_difference"] < 0.05 for r in records.values())
    else:
        assert all(r["fine_quality"]["feasible"] for r in records.values())
    assert all(r["closed_loop_10s"]["initial_axis_progress_m"] > 0 for r in records.values())


if __name__ == "__main__":
    main()
