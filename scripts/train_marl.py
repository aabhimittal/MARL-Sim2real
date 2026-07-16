#!/usr/bin/env python3
"""Stage 1 — Train the MARL packing system.

The proposer agent learns to propose orientations/positions; the physics
agent (PyBullet) accepts or rejects each proposal based on simulated
stability. Cooperative reward = packing density + stability.

Usage:
    python scripts/train_marl.py --episodes 200 --out checkpoints/proposer.pt
"""

import argparse
from pathlib import Path

from marl_sim2real.agents import MARLCoordinator, PhysicsAgent, ProposerAgent
from marl_sim2real.config import Config
from marl_sim2real.envs import PackingEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out", default="checkpoints/proposer.pt")
    parser.add_argument("--no-pybullet", action="store_true",
                        help="use the heuristic physics fallback")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.episodes:
        cfg.train.episodes = args.episodes

    env = PackingEnv(cfg.packing, seed=cfg.train.seed)
    proposer = ProposerAgent(env, cfg.train)
    physics = PhysicsAgent(cfg.physics, use_pybullet=not args.no_pybullet or None)
    coordinator = MARLCoordinator(env, proposer, physics)

    print(f"Training {cfg.train.episodes} episodes "
          f"(physics backend: {'pybullet' if physics.use_pybullet else 'heuristic'})")
    history = coordinator.train(cfg.train.episodes)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    proposer.save(args.out)
    physics.close()

    first, last = history[:20], history[-20:]
    print(f"\nSaved policy to {args.out}")
    print(f"density  first-20 {sum(s.density for s in first)/len(first):.3f} "
          f"-> last-20 {sum(s.density for s in last)/len(last):.3f}")
    print(f"rejects  first-20 {sum(s.rejected for s in first)/len(first):.1f} "
          f"-> last-20 {sum(s.rejected for s in last)/len(last):.1f}")


if __name__ == "__main__":
    main()
