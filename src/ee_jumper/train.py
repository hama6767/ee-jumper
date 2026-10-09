"""CPU-only cross-entropy search over symmetric swimming strokes."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from ee_jumper.env import SwimEnv
from ee_jumper.gait import N_PARAMETERS, rollout


def train_cem(output, generations=12, population=24, seconds=6.0, seed=0, mode="planar"):
    if generations < 1 or population < 4:
        raise ValueError("Need at least one generation and four candidates")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    env = SwimEnv(mode=mode, episode_seconds=seconds)
    started = time.perf_counter()
    mean = np.zeros(N_PARAMETERS)
    std = np.full(N_PARAMETERS, 0.6)
    best = mean.copy()
    history = []
    try:
        baseline = rollout(env, None, seed)
        best_result = rollout(env, best, seed)
        for generation in range(generations):
            candidates = np.clip(rng.normal(mean, std, (population, N_PARAMETERS)), -1, 1)
            candidates[0] = best  # preserve the best policy and a comparable score
            results = [rollout(env, candidate, seed) for candidate in candidates]
            scores = np.array([r["return"] for r in results])
            elite_ids = np.argsort(scores)[-max(2, population // 4) :]
            elites = candidates[elite_ids]
            mean = 0.2 * mean + 0.8 * elites.mean(axis=0)
            std = np.maximum(0.10, 0.2 * std + 0.8 * elites.std(axis=0))
            winner = int(elite_ids[-1])
            best, best_result = candidates[winner].copy(), results[winner]
            row = {
                "generation": generation + 1,
                "best_return": best_result["return"],
                "x_m": best_result["x_m"],
                "mean_forward_mps": best_result["mean_forward_mps"],
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        # Held-out initial-state seeds: a small check, not a robustness claim.
        evaluation = [
            {"seed": s, "policy": rollout(env, best, s), "idle": rollout(env, None, s)}
            for s in range(seed + 100, seed + 103)
        ]
        artifact = {
            "format": "ee-jumper-cem-v1",
            "mode": mode,
            "parameters": best.tolist(),
            "seed": seed,
            "episode_seconds": seconds,
            "population": population,
            "generations": generations,
            "water": asdict(env.water),
            "mujoco_version": mujoco.__version__,
            "baseline": baseline,
            "training_result": best_result,
            "evaluation": evaluation,
            "history": history,
            "wall_seconds": time.perf_counter() - started,
        }
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        return artifact
    finally:
        env.close()


def load_gait(path):
    artifact = json.loads(Path(path).read_text())
    if artifact.get("format") != "ee-jumper-cem-v1":
        raise ValueError("Unsupported gait checkpoint format")
    parameters = np.asarray(artifact["parameters"], dtype=float)
    if parameters.shape != (N_PARAMETERS,) or not np.isfinite(parameters).all():
        raise ValueError("Invalid gait parameters")
    return artifact
