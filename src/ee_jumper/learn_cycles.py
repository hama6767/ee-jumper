"""Reproducible six-direction CEM search and measured thrust library."""

import argparse
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from ee_jumper.cycles import (
    DIRECTIONS,
    N_CYCLE_PARAMETERS,
    DynamicBench,
    KinematicBench,
    quality,
    speed_quality,
)
from ee_jumper.model import ASSETS

_BENCHES = {}


def _evaluate(job):
    kind, period, parameters = job
    if (kind, period) not in _BENCHES:
        _BENCHES[kind, period] = (
            KinematicBench(period, samples=96)
            if kind == "kinematic"
            else DynamicBench(period, mode="free" if kind in ("free", "speed") else "tethered")
        )
    bench = _BENCHES[kind, period]
    if kind == "free":
        return bench.evaluate(
            parameters, warmup_cycles=0, measurement_cycles=int(np.ceil(6 / period))
        )
    if kind == "speed":
        return bench.evaluate(
            parameters,
            warmup_cycles=int(np.ceil(4 / period)),
            measurement_cycles=int(np.ceil(6 / period)),
        )
    if kind == "dynamic":
        return bench.evaluate(parameters, warmup_cycles=2, measurement_cycles=2)
    return bench.evaluate(parameters)


def _mirror_symmetry(parameters, lateral=False):
    """Reflection symmetry for X/Z; reflected time reversal for lateral thrust."""
    p = parameters.reshape(6, 3, 2).copy()
    for left, right in ((0, 1), (2, 3), (4, 5)):
        p[right] = p[left] * (np.array([-1, 1, -1]) if left == 0 else -np.ones(3))[:, None]
        if lateral:
            p[right, :, 1] *= -1
    return p.ravel()


def _write(path, artifact):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train_library(
    output,
    generations=30,
    population=48,
    workers=4,
    seed=42,
    period=2.0,
    minimum_force=0.02,
    finalists=8,
    refine_rounds=2,
    directions="+x,-x,+y,-y,+z,-z",
    warm_start=None,
    lateral_mode="time-reversed",
    objective="efficiency",
    resume_search=None,
):
    if generations < 1 or population < 4 or workers < 1 or finalists < 1 or refine_rounds < 0:
        raise ValueError("Invalid search counts")
    if not np.isfinite(minimum_force) or minimum_force <= 0:
        raise ValueError("minimum_force must be positive")
    if lateral_mode not in ("independent", "time-reversed"):
        raise ValueError("Unknown lateral symmetry mode")
    if (
        not np.isfinite(period)
        or period < 0.5
        or not np.isclose(period / 0.01, round(period / 0.01))
    ):
        raise ValueError("period must be >= 0.5 s and a multiple of 0.01 s")
    if objective not in ("efficiency", "speed"):
        raise ValueError("objective must be efficiency or speed")
    started = time.perf_counter()
    path = Path(output)
    rng = np.random.default_rng(seed)
    directions = directions.split(",")
    if (
        not directions
        or not set(directions) <= set(DIRECTIONS)
        or len(set(directions)) != len(directions)
    ):
        raise ValueError("directions must be unique comma-separated signed axes")
    if set(directions) != set(DIRECTIONS) and not warm_start:
        raise ValueError("Partial training requires a complete --warm-start library")
    means = {name: np.zeros(N_CYCLE_PARAMETERS) for name in directions}
    stds = {name: np.full(N_CYCLE_PARAMETERS, 0.45) for name in directions}
    archives = {name: [] for name in directions}
    history = []
    artifact = {
        "format": "ee-jumper-strokes-v1",
        "period_s": period,
        "frame": "body: x forward, y left, z up",
        "objective": (
            "axial_impulse / absolute_mechanical_work"
            if objective == "efficiency"
            else "free_swim_speed_with_work_and_drift_cost"
        ),
        "training_objective": objective,
        "speed_weights": {
            "lateral_speed": 0.25,
            "power_mps_per_w": 0.002,
            "rotation_mps_per_rad": 0.005,
        },
        "minimum_force_n": minimum_force,
        "maximum_lateral_ratio": 0.35,
        "maximum_torque_nm": 0.002,
        "seed": seed,
        "generations": generations,
        "population_per_direction": population,
        "refine_rounds": refine_rounds,
        "lateral_mode": lateral_mode,
        "control_dt_s": 0.01,
        "physics_dt_s": 0.001,
        "torque_reference": "base_link origin, body frame",
        "work_definition": "integral of sum(abs(actuator_force * actuator_velocity)); no electrical losses or regeneration model",
        "search_scope": "fixed period; 18 active joints; two coefficients per joint; 22 position servos at kp=3, kv=.08, torque limit=2 Nm",
        "mujoco_version": mujoco.__version__,
        "strokes": {},
        "history": history,
        "model_sha256": hashlib.sha256((ASSETS / "robot.xml").read_bytes()).hexdigest(),
    }
    previous = load_library(warm_start, require_feasible=False) if warm_start else None
    if previous:
        if previous["format"] != "ee-jumper-strokes-v1":
            raise ValueError("Use ee-jumper-learn-displacement to continue a v2 library")
        if set(directions) != set(DIRECTIONS):
            if previous["period_s"] != period or previous["minimum_force_n"] != minimum_force:
                raise ValueError("Partial warm-start period and minimum force must match")
            artifact["strokes"] = previous["strokes"].copy()
            artifact["kinematic_best"] = previous.get("kinematic_best", {}).copy()
        artifact["previous_runs"] = previous.get("previous_runs", []) + [
            {
                "seed": previous["seed"],
                "generations": previous["generations"],
                "population_per_direction": previous["population_per_direction"],
                "wall_seconds": previous.get("wall_seconds"),
                "directions": previous.get("trained_directions", list(DIRECTIONS)),
                "lateral_mode": previous.get("lateral_mode", "unspecified"),
            }
        ]
        for name in directions:
            p = np.array(previous["strokes"][name]["parameters"])
            means[name] = (
                p
                if name[-1] == "y" and lateral_mode == "independent"
                else _mirror_symmetry(p, name[-1] == "y")
            )
            stds[name][:] = 0.3
    artifact["trained_directions"] = directions
    if resume_search:
        checkpoint = json.loads(Path(resume_search).read_text())
        if (
            checkpoint.get("format") != "ee-jumper-strokes-v1"
            or checkpoint["period_s"] != period
            or checkpoint.get("training_objective", "efficiency") != objective
            or checkpoint["minimum_force_n"] != minimum_force
            or checkpoint["trained_directions"] != directions
        ):
            raise ValueError("Resume configuration must match the saved search")
        artifact = checkpoint
        artifact.setdefault("resumed_searches", []).append(
            {"source": str(resume_search), "seed": seed, "refine_rounds": refine_rounds}
        )
        archives = checkpoint.get("kinematic_candidates") or {
            n: [checkpoint["kinematic_best"][n]] for n in directions
        }
        if "rng_state" in checkpoint:
            rng.bit_generator.state = checkpoint["rng_state"]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for generation in range(0 if resume_search else generations):
            candidates = []
            for name in directions:
                samples = np.clip(
                    rng.normal(means[name], stds[name], (population, N_CYCLE_PARAMETERS)), -1, 1
                )
                if name[-1] != "y" or lateral_mode == "time-reversed":
                    samples = np.array([_mirror_symmetry(p, name[-1] == "y") for p in samples])
                if generation == 0:
                    samples[0] = means[name]
                if archives[name]:
                    samples[0] = archives[name][0]["parameters"]
                candidates.extend(samples)
            metrics = list(
                pool.map(_evaluate, [("kinematic", period, p) for p in candidates], chunksize=4)
            )
            report = {"stage": "kinematic", "generation": generation + 1, "directions": {}}
            # A single measured wrench can inform all six optimizers.
            for name in directions:
                combined = archives[name] + [
                    {"parameters": p.tolist(), "metrics": m} for p, m in zip(candidates, metrics)
                ]
                combined.sort(
                    key=lambda c: quality(c["metrics"], name, minimum_force, objective)["score"],
                    reverse=True,
                )
                unique = {}
                for c in combined:
                    key = tuple(np.round(c["parameters"], 8))
                    if key not in unique:
                        unique[key] = c
                    if len(unique) >= max(finalists, population // 4):
                        break
                archives[name] = list(unique.values())
                elite = np.array(
                    [c["parameters"] for c in archives[name][: max(4, population // 4)]]
                )
                means[name] = 0.2 * means[name] + 0.8 * elite.mean(axis=0)
                stds[name] = np.maximum(0.055, 0.2 * stds[name] + 0.8 * elite.std(axis=0))
                report["directions"][name] = quality(
                    archives[name][0]["metrics"], name, minimum_force, objective
                )
            history.append(report)
            if generation % 5 == 0 or generation == generations - 1:
                print(json.dumps(report), flush=True)
        artifact.setdefault("kinematic_best", {}).update(
            {name: archives[name][0] for name in directions}
        )
        artifact["kinematic_candidates"] = archives
        artifact["rng_state"] = rng.bit_generator.state
        _write(path, artifact)
        for name in directions:
            if resume_search and name in artifact["strokes"]:
                continue
            candidates = [c["parameters"] for c in archives[name][:finalists]]
            if previous:
                candidates.append(previous["strokes"][name]["parameters"])
            measured = list(pool.map(_evaluate, [("dynamic", period, p) for p in candidates]))
            tested = [{"parameters": p, "metrics": m} for p, m in zip(candidates, measured)]
            for iteration in range(refine_rounds):
                tested.sort(
                    key=lambda c: quality(c["metrics"], name, minimum_force, objective)["score"],
                    reverse=True,
                )
                center = np.array(tested[0]["parameters"])
                candidates = np.clip(
                    center + rng.normal(0, 0.04 / (iteration + 1), (12, N_CYCLE_PARAMETERS)), -1, 1
                )
                if name[-1] != "y" or lateral_mode == "time-reversed":
                    candidates = np.array(
                        [_mirror_symmetry(p, name[-1] == "y") for p in candidates]
                    )
                measured = list(pool.map(_evaluate, [("dynamic", period, p) for p in candidates]))
                tested.extend(
                    {"parameters": p.tolist(), "metrics": m} for p, m in zip(candidates, measured)
                )
            tested.sort(
                key=lambda c: quality(c["metrics"], name, minimum_force, objective)["score"],
                reverse=True,
            )
            best = tested[0]
            if objective == "speed":
                eligible = [
                    c for c in tested if quality(c["metrics"], name, minimum_force)["feasible"]
                ][:finalists]
                # Keep a feasible warm-start in the free-swim shortlist even
                # if stronger tethered strokes have overtaken its thrust.
                if previous:
                    baseline = previous["strokes"][name]["parameters"]
                    for candidate in tested:
                        if (
                            candidate["parameters"] == baseline
                            and quality(candidate["metrics"], name, minimum_force)["feasible"]
                            and candidate not in eligible
                        ):
                            eligible.append(candidate)
                if not eligible:
                    raise RuntimeError(f"No feasible dynamic candidates for {name}")
                free_metrics = list(
                    pool.map(_evaluate, [("speed", period, c["parameters"]) for c in eligible])
                )
                for c, m in zip(eligible, free_metrics):
                    c["speed_selection"] = m
                    c["speed_quality"] = speed_quality(m, name)
                best = max(eligible, key=lambda c: c["speed_quality"]["score"])
                if not best["speed_quality"]["valid"]:
                    raise RuntimeError(f"No stable free-swim candidates for {name}")
            artifact["strokes"][name] = {
                **best,
                "quality": quality(best["metrics"], name, minimum_force, objective),
                "dynamically_tested_candidates": len(tested),
            }
            print(
                json.dumps(
                    {"stage": "dynamic", "direction": name, **artifact["strokes"][name]["quality"]}
                ),
                flush=True,
            )
            artifact["rng_state"] = rng.bit_generator.state
            _write(path, artifact)
        free = list(
            pool.map(
                _evaluate,
                [("free", period, artifact["strokes"][name]["parameters"]) for name in directions],
            )
        )
        for name, result in zip(directions, free):
            artifact["strokes"][name]["free_validation"] = result
        artifact["wall_seconds"] = artifact.get("wall_seconds", 0) + time.perf_counter() - started
        artifact["all_directions_feasible"] = all(
            s["quality"]["feasible"] for s in artifact["strokes"].values()
        )
        _write(path, artifact)
    return artifact


def load_library(path, require_feasible=True):
    library = json.loads(Path(path).read_text())
    if library.get("format") not in ("ee-jumper-strokes-v1", "ee-jumper-strokes-v2") or set(
        library.get("strokes", {})
    ) != set(DIRECTIONS):
        raise ValueError("Expected a complete six-direction stroke library")
    if not np.isfinite(library["period_s"]) or library["period_s"] < 0.5:
        raise ValueError("Invalid library period")
    excursion = library.get("max_excursion_rad", 0.9)
    if not np.isfinite(excursion) or not 0 < excursion <= 2:
        raise ValueError("Invalid maximum joint excursion")
    for name, record in library["strokes"].items():
        p = np.asarray(record["parameters"])
        if (
            p.shape != (N_CYCLE_PARAMETERS,)
            or not np.isfinite(p).all()
            or np.max(abs(p)) > 1.000001
        ):
            raise ValueError(f"Invalid parameters for {name}")
        metrics = record["metrics"]
        force, moment = np.asarray(metrics["mean_force_n"]), np.asarray(metrics["mean_torque_nm"])
        if (
            force.shape != (3,)
            or moment.shape != (3,)
            or not np.isfinite(force).all()
            or not np.isfinite(moment).all()
            or not np.isfinite(metrics["mean_power_w"])
            or metrics["mean_power_w"] < 0
        ):
            raise ValueError(f"Invalid measured wrench or power for {name}")
        if library["format"] == "ee-jumper-strokes-v2":
            from ee_jumper.learn_displacement import displacement_quality

            free = record.get("free_validation", {})
            displacement = np.asarray(free.get("displacement_m", []))
            if (
                displacement.shape != (3,)
                or not np.isfinite(displacement).all()
                or free.get("elapsed_s", 0) <= 0
            ):
                raise ValueError(f"Invalid free-swim measurement for {name}")
            if require_feasible and not displacement_quality(free, name)["valid"]:
                raise ValueError(f"{name} does not satisfy the net displacement constraints")
        elif (
            require_feasible
            and not quality(record["metrics"], name, library["minimum_force_n"])["feasible"]
        ):
            raise ValueError(f"{name} does not satisfy the thrust/torque/energy constraints")
    return library


def main():
    parser = argparse.ArgumentParser(description="Learn periodic strokes for +/- X/Y/Z")
    parser.add_argument("--objective", choices=("efficiency", "speed"), default="efficiency")
    parser.add_argument("--output", default="runs/strokes.json")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--period", type=float, default=2.0)
    parser.add_argument("--minimum-force", type=float, default=0.02)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--directions", default="+x,-x,+y,-y,+z,-z")
    parser.add_argument("--warm-start")
    parser.add_argument(
        "--resume-search", help="Resume servo validation from saved kinematic candidates"
    )
    parser.add_argument(
        "--lateral-mode", choices=("independent", "time-reversed"), default="time-reversed"
    )
    args = parser.parse_args()
    result = train_library(**vars(args))
    print(
        json.dumps(
            {
                "output": args.output,
                "all_directions_feasible": result["all_directions_feasible"],
                "wall_seconds": result["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
