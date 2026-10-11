"""Learn larger strokes directly from net free-swimming COM displacement."""

import argparse
import copy
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from ee_jumper.cycles import DIRECTIONS, DynamicBench, StrokeSpace, quality
from ee_jumper.learn_cycles import _mirror_symmetry, _write, load_library
from ee_jumper.model import make_model

_BENCHES = {}


def displacement_quality(metrics, direction):
    displacement = np.asarray(metrics.get("displacement_m", [0, 0, 0]))
    axis = float(displacement @ DIRECTIONS[direction])
    lateral = float(np.linalg.norm(displacement - axis * DIRECTIONS[direction]))
    angle = metrics.get("peak_rotation_rad", np.pi)
    physical = (
        np.isfinite(displacement).all()
        and np.isfinite(
            [
                angle,
                metrics.get("penetration_m", 0),
                metrics.get("peak_actuator_nm", 0),
                metrics["work_j"],
            ]
        ).all()
        and not metrics.get("unstable", False)
        and metrics.get("penetration_m", 0) <= 0.0005
        and metrics.get("peak_actuator_nm", 0) <= 2.00001
    )
    valid = physical and axis > 0 and lateral <= 0.75 * axis and angle < 0.6
    score = axis - 0.4 * lateral - 0.03 * angle - 0.0001 * metrics["work_j"]
    score -= 1.5 * max(0, angle - 0.45)
    if not physical:
        score = -2 - min(metrics.get("penetration_m", 0) * 1000, 100)
    return {
        "axial_displacement_m": axis,
        "lateral_displacement_m": lateral,
        "valid": bool(valid),
        "score": float(score),
    }


def _measure(job):
    parameters, period, excursion, fine, mode, margin = job
    key = (period, excursion, fine, mode, margin)
    if key not in _BENCHES:
        bench = DynamicBench(
            period,
            control_dt=0.01 if fine else 0.02,
            mode=mode,
            max_excursion=excursion,
            joint_limit_margin_rad=margin,
        )
        if not fine:
            bench.model.opt.timestep = 0.002
            bench.substeps = 10
        _BENCHES[key] = bench
    bench = _BENCHES[key]
    return bench.evaluate(
        parameters,
        warmup_cycles=2 if mode == "tethered" else 0,
        measurement_cycles=2 if mode == "tethered" else round(6 / period),
    )


def train(
    output,
    warm_start,
    generations=16,
    population=20,
    workers=4,
    seed=51,
    period=1.0,
    max_excursion=1.6,
    directions="+x,-x,+y,-y,+z,-z",
):
    if generations < 1 or population < 6 or workers < 1:
        raise ValueError("Invalid search size")
    if (
        period < 0.5
        or not np.isclose(round(6 / period) * period, 6)
        or not np.isclose(round(period / 0.02) * 0.02, period)
    ):
        raise ValueError("period must divide 6 seconds and be a multiple of .02 s")
    names = directions.split(",")
    if len(set(names)) != len(names) or not set(names) <= set(DIRECTIONS):
        raise ValueError("Invalid directions")
    previous = load_library(warm_start)
    if set(names) != set(DIRECTIONS) and (
        previous["period_s"] != period or previous.get("max_excursion_rad", 0.9) != max_excursion
    ):
        raise ValueError("Partial training requires matching period and excursion")
    margin = previous.get("joint_limit_margin_rad", 0.0)
    m = make_model("free", joint_limit_margin_rad=margin)
    old = StrokeSpace(m, previous["period_s"], previous.get("max_excursion_rad", 0.9))
    new = StrokeSpace(m, period, max_excursion)
    ratio = (old.scale[new.active] / new.scale[new.active])[:, None]
    means = {
        n: (np.array(previous["strokes"][n]["parameters"]).reshape(18, 2) * ratio).ravel()
        for n in names
    }
    stds = {n: np.full(36, 0.18) for n in names}
    archives = {n: [] for n in names}
    rng = np.random.default_rng(seed)
    artifact = {
        "format": "ee-jumper-strokes-v2",
        "period_s": period,
        "max_excursion_rad": max_excursion,
        "joint_limit_margin_rad": margin,
        "frame": previous["frame"],
        "objective": "6-second free-swim net COM displacement",
        "minimum_force_n": 0.02,
        "training_objective": "displacement",
        "seed": seed,
        "generations": generations,
        "population_per_direction": population,
        "mujoco_version": mujoco.__version__,
        "physics_dt_s": 0.001,
        "control_dt_s": 0.01,
        "search_physics_dt_s": 0.002,
        "search_control_dt_s": 0.02,
        "maximum_rotation_rad": 0.6,
        "maximum_lateral_displacement_ratio": 0.75,
        "maximum_actuator_torque_nm": 2,
        "maximum_penetration_m": 0.0005,
        "minimum_force_is_diagnostic_only": True,
        "source_library": str(warm_start),
        "model_sha256": previous["model_sha256"],
        "torque_reference": "base_link origin, body frame",
        "strokes": copy.deepcopy(previous["strokes"]) if set(names) != set(DIRECTIONS) else {},
        "history": [],
    }
    start = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for generation in range(generations):
            for n in names:
                candidates = np.clip(rng.normal(means[n], stds[n], (population, 36)), -1, 1)
                candidates = np.array([_mirror_symmetry(p, n[-1] == "y") for p in candidates])
                candidates[0] = means[n] if not archives[n] else archives[n][0]["parameters"]
                candidates[1] = np.clip(candidates[0] * 1.12, -1, 1)
                measurements = list(
                    pool.map(
                        _measure,
                        [(p, period, max_excursion, False, "free", margin) for p in candidates],
                    )
                )
                combined = archives[n] + [
                    {"parameters": p.tolist(), "free_validation": r}
                    for p, r in zip(candidates, measurements)
                ]
                combined.sort(
                    key=lambda c: displacement_quality(c["free_validation"], n)["score"],
                    reverse=True,
                )
                unique = {}
                for record in combined:
                    unique.setdefault(tuple(np.round(record["parameters"], 7)), record)
                    if len(unique) >= 10:
                        break
                archives[n] = list(unique.values())
                elite = np.asarray([c["parameters"] for c in archives[n][:5]])
                means[n] = 0.3 * means[n] + 0.7 * elite.mean(axis=0)
                stds[n] = np.maximum(0.035, 0.3 * stds[n] + 0.7 * elite.std(axis=0))
            report = {
                "generation": generation + 1,
                "directions": {
                    n: displacement_quality(archives[n][0]["free_validation"], n) for n in names
                },
            }
            artifact["history"].append(report)
            artifact["search_candidates"] = archives
            _write(Path(output), artifact)
            print(json.dumps(report), flush=True)
        for n in names:
            records = archives[n]
            fine = list(
                pool.map(
                    _measure,
                    [
                        (c["parameters"], period, max_excursion, True, "free", margin)
                        for c in records
                    ],
                )
            )
            valid = [
                {
                    "parameters": c["parameters"],
                    "free_validation": f,
                    "displacement_quality": displacement_quality(f, n),
                }
                for c, f in zip(records, fine)
                if displacement_quality(f, n)["valid"]
            ]
            if not valid:
                raise RuntimeError(
                    f"No valid free-swim candidate for {n}; search candidates saved to {output}"
                )
            best = max(valid, key=lambda c: c["displacement_quality"]["score"])
            tether = _measure((best["parameters"], period, max_excursion, True, "tethered", margin))
            best["metrics"] = tether
            best["quality"] = quality(tether, n)  # diagnostic, not a gate on swimming
            targets = new.trajectory(np.linspace(0, 1, 1001), best["parameters"])[0]
            best["target_joint_peak_to_peak_rad"] = np.ptp(targets, axis=0).tolist()
            artifact["strokes"][n] = best
            print(json.dumps({"direction": n, **best["displacement_quality"]}), flush=True)
            _write(Path(output), artifact)
    artifact["all_directions_feasible"] = all(
        s["displacement_quality"]["valid"] for s in artifact["strokes"].values()
    )
    artifact["wall_seconds"] = time.perf_counter() - start
    _write(Path(output), artifact)
    return artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/strokes-displacement.json")
    p.add_argument("--warm-start", required=True)
    p.add_argument("--generations", type=int, default=16)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--period", type=float, default=1.0)
    p.add_argument("--max-excursion", type=float, default=1.6)
    p.add_argument("--directions", default="+x,-x,+y,-y,+z,-z")
    args = p.parse_args()
    result = train(**vars(args))
    print(json.dumps({"output": args.output, "seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
