#!/usr/bin/env python
"""CLI: jointly train the Proposer and Physics agents with MARLTrainer.

Runs entirely on numpy (no torch, no pybullet). Example:

    python scripts/train.py --episodes 200 --bin-size 8 8 8 --num-boxes 15 \\
        --randomize --save-dir checkpoints/
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo-root shim so this script runs as `python scripts/train.py` without
# installing the packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from marl_packing.agents.physics_agent import PhysicsAgent
from marl_packing.agents.proposer_agent import ProposerAgent
from marl_packing.envs.packing_env import PackingConfig, PackingEnv
from marl_packing.sim2real.domain_randomization import DomainRandomizer, RandomizationConfig
from marl_packing.training.ppo_trainer import MARLTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Proposer + Physics MARL packing agents.")
    p.add_argument("--episodes", type=int, default=200, help="number of training episodes")
    p.add_argument(
        "--bin-size", type=int, nargs=3, default=(10, 10, 10), metavar=("W", "D", "H"),
        help="bin dimensions",
    )
    p.add_argument("--num-boxes", type=int, default=20, help="boxes per episode")
    p.add_argument("--seed", type=int, default=0, help="random seed for env/agents/trainer")
    p.add_argument(
        "--save-dir", type=str, default="checkpoints",
        help="directory to write proposer.npz / physics.npz",
    )
    p.add_argument(
        "--randomize", action="store_true",
        help="wrap the env in a DomainRandomizer so training is robust to sim2real gaps",
    )
    p.add_argument("--log-every", type=int, default=10, help="print progress every N episodes")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env_cfg = PackingConfig(bin_size=tuple(args.bin_size), num_boxes=args.num_boxes, seed=args.seed)
    env = PackingEnv(env_cfg)
    if args.randomize:
        env = DomainRandomizer(env, RandomizationConfig(seed=args.seed))
        print("Domain randomization enabled: box tolerances, placement jitter, "
              "observation noise, and support-threshold shifts are perturbed per episode.")

    proposer = ProposerAgent(env, seed=args.seed)
    physics = PhysicsAgent(seed=args.seed)
    trainer_cfg = TrainerConfig(episodes=args.episodes, log_every=args.log_every, seed=args.seed)
    trainer = MARLTrainer(env, proposer, physics, trainer_cfg)

    print(
        f"Training for {args.episodes} episodes on a {tuple(args.bin_size)} bin "
        f"with {args.num_boxes} boxes/episode (seed={args.seed})..."
    )
    history = trainer.train()

    window = history[-min(10, len(history)):]
    print(
        f"Done. mean reward (last {len(window)} eps): "
        f"{np.mean([s.reward for s in window]):.3f} | "
        f"mean fill: {np.mean([s.fill_ratio for s in window]):.3f} | "
        f"mean vetoes: {np.mean([s.vetoes for s in window]):.2f} | "
        f"physics BCE: {np.mean([s.physics_loss for s in window]):.4f}"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    proposer_path = os.path.join(args.save_dir, "proposer.npz")
    physics_path = os.path.join(args.save_dir, "physics.npz")
    proposer.save(proposer_path)
    physics.save(physics_path)
    print(f"Saved proposer weights -> {proposer_path}")
    print(f"Saved physics weights  -> {physics_path}")


if __name__ == "__main__":
    main()
