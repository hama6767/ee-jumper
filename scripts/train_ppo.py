"""Optional closed-loop PPO baseline. Install ee-jumper[rl] first."""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ee_jumper.env import SwimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("planar", "free"), default="planar")
    parser.add_argument("--output", default="runs/ppo")
    args = parser.parse_args()
    if args.steps < 1 or args.envs < 1:
        parser.error("--steps and --envs must be positive")
    torch.set_num_threads(1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: SwimEnv(mode=args.mode), n_envs=args.envs, seed=args.seed)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            device="cpu",
            verbose=1,
            n_steps=512,
            batch_size=128,
            policy_kwargs={"net_arch": [128, 128]},
        )
        model.learn(total_timesteps=args.steps)
        model.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
