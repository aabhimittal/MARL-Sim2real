#!/usr/bin/env python3
"""Train the MARL packing agents in simulation (with domain randomization).

Usage:
    python scripts/train.py --episodes 300 --bin 8 8 8 --out artifacts/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_sim2real.agents import PhysicsAgent, ProposerAgent
from marl_sim2real.env import BinPackingEnv
from marl_sim2real.export import export_bundle
from marl_sim2real.sim2real import DomainRandomizer
from marl_sim2real.training import MARLTrainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--bin", type=int, nargs=3, default=[8, 8, 8], metavar=("L", "W", "H"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="artifacts")
    ap.add_argument("--no-randomize", action="store_true", help="disable domain randomization")
    args = ap.parse_args()

    bin_size = tuple(args.bin)
    env = BinPackingEnv(bin_size=bin_size, seed=args.seed)
    proposer = ProposerAgent(env.observation_size, env.num_positions * 6, seed=args.seed)
    physics = PhysicsAgent(env.observation_size, seed=args.seed + 1)
    trainer = MARLTrainer(env, proposer, physics, seed=args.seed)

    randomizer = None if args.no_randomize else DomainRandomizer(seed=args.seed)
    chunk = max(1, args.episodes // 10)
    for start in range(0, args.episodes, chunk):
        if randomizer:
            params = randomizer.apply(env)
            print(f"-- physics randomized: friction={params.friction:.2f} "
                  f"support_thr={params.support_threshold:.2f} sensor_noise={params.sensor_noise:.2f}")
        trainer.train(episodes=min(chunk, args.episodes - start), log_every=chunk)

    result = trainer.evaluate(episodes=20)
    print(f"\nfinal eval: utilization={result['mean_utilization']:.1%} return={result['mean_return']:+.2f}")

    manifest = export_bundle(args.out, proposer, physics, bin_size, name="packing_policy_sim")
    print(f"exported bundle -> {manifest}")


if __name__ == "__main__":
    main()
