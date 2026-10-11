"""PPO fine-tuning of existing periodic motions (a one-step contextual task).

An action changes 18 independent waveform coefficients, mirrored to 36. The
simulator executes four complete strokes and returns net-motion reward. This
is PPO over motion parameters, not a state-feedback joint controller. Exported
coefficients run in the normal controller without Torch or SB3.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ee_jumper.cycles import DIRECTIONS, DynamicBench, StrokeSpace
from ee_jumper.learn_cycles import _mirror_symmetry, _write, load_library
from ee_jumper.learn_displacement import displacement_quality
from ee_jumper.model import make_model


class MotionRefinementEnv(gym.Env):
    def __init__(self, library, margin_degrees=10.0, excursion=1.8, correction=0.3):
        self.library = library
        self.names = list(DIRECTIONS)
        self.margin = float(np.deg2rad(margin_degrees))
        self.excursion, self.correction = excursion, correction
        self.bench = DynamicBench(
            library["period_s"],
            control_dt=0.02,
            mode="free",
            max_excursion=excursion,
            joint_limit_margin_rad=self.margin,
        )
        self.bench.model.opt.timestep = 0.002
        self.bench.substeps = 10
        old = StrokeSpace(
            make_model("free", joint_limit_margin_rad=library.get("joint_limit_margin_rad", 0)),
            library["period_s"],
            library.get("max_excursion_rad", 0.9),
        )
        new = self.bench.stroke
        scale = np.repeat(old.scale[old.active] / new.scale[new.active], 2)
        # Zero action exactly reproduces the original target angles despite
        # the expanded joint range and amplitude envelope.
        self.anchors = np.array([library["strokes"][n]["parameters"] for n in self.names]) * scale
        self.action_space = spaces.Box(-1, 1, (18,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 1, (6,), dtype=np.float32)
        self.baseline = {}
        self.index = 0
        self.needs_reset = True

    def parameters(self, direction, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError("Expected 18 finite corrections")
        index = self.names.index(direction)
        p = self.anchors[index].reshape(6, 3, 2).copy()
        p[[0, 2, 4]] += self.correction * np.clip(action.reshape(3, 3, 2), -1, 1)
        return _mirror_symmetry(np.clip(p.ravel(), -1, 1), direction.endswith("y"))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = int(self.np_random.integers(6))
        self.needs_reset = False
        return np.eye(6, dtype=np.float32)[self.index], {}

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("reset is required after a motion trial")
        name = self.names[self.index]
        if name not in self.baseline:
            baseline = self.bench.evaluate(self.anchors[self.index], 0, 4)
            self.baseline[name] = displacement_quality(baseline, name)["score"]
        parameters = self.parameters(name, action)
        metrics = self.bench.evaluate(parameters, 0, 4)
        result = displacement_quality(metrics, name)
        reward = 100 * (result["score"] - self.baseline[name])
        self.needs_reset = True
        return (
            np.eye(6, dtype=np.float32)[self.index],
            float(reward),
            True,
            False,
            {
                "direction": name,
                "axial_m": result["axial_displacement_m"],
                "valid": result["valid"],
            },
        )


def candidate_library(source, env, actions):
    actions = np.asarray(actions)
    if actions.shape != (6, 18) or not np.isfinite(actions).all():
        raise ValueError("Expected six sets of 18 finite corrections")
    # Only the selected motions and their measurements are needed at runtime.
    result = copy.deepcopy(source)
    for key in ("search_candidates", "history"):
        result.pop(key, None)
    result["joint_limit_margin_rad"] = env.margin
    result["max_excursion_rad"] = env.excursion
    result["validation_pending"] = True
    for name, action in zip(env.names, actions):
        result["strokes"][name]["parameters"] = env.parameters(name, action).tolist()
    return result


def train(library_path, output, steps=1024, envs=4, seed=73):
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    torch.set_num_threads(1)
    source = load_library(library_path)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    env = MotionRefinementEnv(source)
    vector = SubprocVecEnv(
        [lambda: Monitor(MotionRefinementEnv(source)) for _ in range(envs)], start_method="spawn"
    )
    vector.seed(seed)
    agent = PPO(
        "MlpPolicy",
        vector,
        seed=seed,
        device="cpu",
        verbose=0,
        n_steps=16,
        batch_size=64,
        n_epochs=10,
        learning_rate=0.002,
        gamma=0.0,
        gae_lambda=1.0,
        clip_range=0.2,
        target_kl=0.04,
        policy_kwargs={"net_arch": [64, 64], "log_std_init": -0.7},
    )
    # Start from the already working motion, not a randomly biased correction.
    with torch.no_grad():
        agent.policy.action_net.weight.zero_()
        agent.policy.action_net.bias.zero_()
    records = []
    started = time.monotonic()
    _write(
        path / "config.json",
        {
            "algorithm": "PPO",
            "task": "one-step motion-parameter refinement",
            "seed": seed,
            "requested_trials": steps,
            "parallel_environments": envs,
            "ppo": {
                "network": [64, 64],
                "n_steps": 16,
                "batch_size": 64,
                "n_epochs": 10,
                "learning_rate": 0.002,
                "gamma": 0.0,
                "gae_lambda": 1.0,
                "clip_range": 0.2,
                "target_kl": 0.04,
                "log_std_init": -0.7,
            },
            "period_s": source["period_s"],
            "training_cycles": 4,
            "physics_dt_s": 0.002,
            "control_dt_s": 0.02,
            "margin_degrees": 10,
            "max_excursion_rad": 1.8,
            "coefficient_correction": env.correction,
            "sb3_version": stable_baselines3.__version__,
            "torch_version": torch.__version__,
            "source_sha256": hashlib.sha256(Path(library_path).read_bytes()).hexdigest(),
        },
    )
    try:
        while agent.num_timesteps < steps:
            count = min(256, steps - agent.num_timesteps)
            agent.learn(count, reset_num_timesteps=False)
            actions, _ = agent.predict(np.eye(6, dtype=np.float32), deterministic=True)
            candidate = candidate_library(source, env, actions)
            checkpoint = path / f"policy-{agent.num_timesteps}.zip"
            agent.save(checkpoint)
            candidate["ppo_refinement"] = {
                "algorithm": "PPO",
                "seed": seed,
                "trials": agent.num_timesteps,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "action_correction": actions.tolist(),
                "validation_pending": True,
            }
            _write(path / f"candidate-{agent.num_timesteps}.json", candidate)
            row = {
                "trials": agent.num_timesteps,
                "wall_seconds": time.monotonic() - started,
                "mean_episode_reward": float(np.mean([v["r"] for v in agent.ep_info_buffer])),
                "max_correction": float(np.abs(actions).max()),
            }
            records.append(row)
            _write(path / "training.json", records)
            print(json.dumps(row), flush=True)
    finally:
        vector.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="examples/strokes-xyz.json")
    parser.add_argument("--output", default="runs/ppo-refine")
    parser.add_argument(
        "--steps", type=int, default=1024, help="Completed four-cycle motion trials"
    )
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if args.steps < 64 or args.envs < 1:
        parser.error("Use at least 64 trials and one environment")
    train(args.library, args.output, args.steps, args.envs, args.seed)


if __name__ == "__main__":
    main()
