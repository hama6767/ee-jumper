"""Compare PPO checkpoints, range-only changes and the original stroke library.

Selection uses still-water 6 s trials; the selected motion is then checked on
unseen +/- 3 degree initial tilts and a 12 s rollout. No candidate is installed
unless it improves displacement by 5%, score, and passes physical constraints.
"""

import argparse
import copy
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ee_jumper.cycles import DIRECTIONS, DynamicBench, quality
from ee_jumper.learn_cycles import _write, load_library
from ee_jumper.learn_displacement import displacement_quality
from ee_jumper.refine_ppo import MotionRefinementEnv, candidate_library


def measure(job):
    name, library, seconds, tilt, fine = job
    bench = DynamicBench(
        library["period_s"],
        mode="free",
        control_dt=0.005 if fine else 0.01,
        max_excursion=library.get("max_excursion_rad", 0.9),
        joint_limit_margin_rad=library.get("joint_limit_margin_rad", 0),
    )
    if fine:
        bench.model.opt.timestep = 0.0005
        bench.substeps = 10
    angle = np.deg2rad(tilt) / 2
    q = [np.cos(angle), np.sin(angle), 0, 0]
    return bench.evaluate(
        library["strokes"][name]["parameters"],
        0,
        round(seconds / library["period_s"]),
        initial_quaternion=q,
    )


def usable(candidate, baseline, name, improvement=1.05):
    c, b = displacement_quality(candidate, name), displacement_quality(baseline, name)
    return (
        c["valid"]
        and c["axial_displacement_m"] >= improvement * b["axial_displacement_m"]
        and c["score"] > b["score"]
        and c["lateral_displacement_m"] <= max(0.04, 1.3 * b["lateral_displacement_m"])
        and candidate["peak_rotation_rad"] <= max(0.35, 1.25 * baseline["peak_rotation_rad"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="examples/strokes-xyz-cem.json")
    parser.add_argument("--directory", default="runs/ppo-refine")
    parser.add_argument("--output", default="runs/ppo-refine/selected.json")
    parser.add_argument("--report", default="runs/ppo-refine/comparison.json")
    args = parser.parse_args()
    baseline = load_library(args.baseline)
    env = MotionRefinementEnv(baseline)
    preserved = candidate_library(baseline, env, np.zeros((6, 18)))
    expanded = copy.deepcopy(preserved)
    for name in DIRECTIONS:
        expanded["strokes"][name]["parameters"] = baseline["strokes"][name]["parameters"]
    candidates = {"original": baseline, "range_only": expanded}
    for path in sorted(
        Path(args.directory).glob("candidate-*.json"), key=lambda p: int(p.stem.split("-")[1])
    ):
        candidates[path.stem] = load_library(path, require_feasible=False)
    names = list(DIRECTIONS)
    jobs = [(n, library, 6, 0, False) for n in names for library in candidates.values()]
    selected = copy.deepcopy(preserved)
    report = {
        "criterion": "at least 5% axial gain, better score, bounded lateral drift and rotation",
        "joint_limit_margin_degrees": 10,
        "directions": {},
    }
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
        measured = list(pool.map(measure, jobs))
        chosen = {}
        for i, name in enumerate(names):
            results = dict(
                zip(candidates, measured[i * len(candidates) : (i + 1) * len(candidates)])
            )
            reference = results["original"]
            eligible = [
                k for k, v in results.items() if k != "original" and usable(v, reference, name)
            ]
            chosen[name] = (
                max(eligible, key=lambda k: displacement_quality(results[k], name)["score"])
                if eligible
                else "original"
            )
            label = chosen[name]
            if label != "original":
                selected["strokes"][name]["parameters"] = candidates[label]["strokes"][name][
                    "parameters"
                ]
            report["directions"][name] = {"selected": label, "nominal": results}
            print(
                name,
                {
                    k: round(displacement_quality(v, name)["axial_displacement_m"], 4)
                    for k, v in results.items()
                },
                "selected",
                label,
                flush=True,
            )
        # Held-out conditions do not choose another checkpoint: reject to baseline on failure.
        jobs = [
            (n, lib, seconds, tilt, fine)
            for n in names
            for seconds, tilt, fine in ((6, 3, False), (6, -3, False), (12, 0, False), (6, 0, True))
            for lib in (baseline, selected)
        ]
        measurements = list(pool.map(measure, jobs))
        for i, name in enumerate(names):
            cases = []
            accepted = chosen[name] != "original"
            for j, (seconds, tilt, fine) in enumerate(
                ((6, 3, False), (6, -3, False), (12, 0, False), (6, 0, True))
            ):
                before, after = measurements[i * 8 + j * 2 : i * 8 + j * 2 + 2]
                valid = usable(after, before, name, improvement=1.0)
                cases.append(
                    {
                        "seconds": seconds,
                        "roll_deg": tilt,
                        "fine": fine,
                        "before": before,
                        "after": after,
                        "passed": valid,
                    }
                )
                accepted &= valid
            if not accepted:
                selected["strokes"][name]["parameters"] = preserved["strokes"][name]["parameters"]
                chosen[name] = "original"
            report["directions"][name]["held_out"] = cases
            report["directions"][name]["adopted"] = chosen[name]
        # All final records must describe exactly the final model and parameters.
        jobs = [(n, selected, 6, 0, False) for n in names]
        final = list(pool.map(measure, jobs))
    for name, metrics in zip(names, final):
        record = selected["strokes"][name]
        record["free_validation"] = metrics
        record["displacement_quality"] = displacement_quality(metrics, name)
        bench = DynamicBench(
            selected["period_s"],
            max_excursion=selected["max_excursion_rad"],
            joint_limit_margin_rad=selected["joint_limit_margin_rad"],
        )
        record["metrics"] = bench.evaluate(record["parameters"], 2, 2)
        record["quality"] = quality(record["metrics"], name)
        targets = bench.stroke.trajectory(np.linspace(0, 1, 1001), record["parameters"])[0]
        record["target_joint_peak_to_peak_rad"] = np.ptp(targets, axis=0).tolist()
        record["refinement_source"] = chosen[name]
    selected.pop("ppo_refinement", None)
    selected.pop("validation_pending", None)
    selected["motion_refinement"] = {
        "method": "PPO over periodic coefficients; deterministic checkpoint means",
        "baseline_sha256": hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
        "training_config": json.loads((Path(args.directory) / "config.json").read_text()),
        "adopted": chosen,
        "checkpoints": {
            label: {
                k: v
                for k, v in candidates[label]["ppo_refinement"].items()
                if k != "validation_pending"
            }
            for label in set(chosen.values())
            if label.startswith("candidate-")
        },
        "selection_report": str(Path(args.report).name),
    }
    selected["all_directions_feasible"] = all(
        r["displacement_quality"]["valid"] for r in selected["strokes"].values()
    )
    if not selected["all_directions_feasible"]:
        raise RuntimeError("Final motions failed validation; the output library was not replaced")
    _write(Path(args.output), selected)
    _write(Path(args.report), report)
    print("Final selection:", chosen, flush=True)


if __name__ == "__main__":
    main()
