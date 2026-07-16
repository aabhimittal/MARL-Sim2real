#!/usr/bin/env python
"""CLI: evaluate trained agents in sim, then run the full Sim2Real cycle:
deploy against a MockRobot, measure the reality gap, fine-tune the physics
critic on the real outcomes, and redeploy to show the gap shrinking.

    python scripts/evaluate.py --load-dir checkpoints/ --episodes 20
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from marl_packing.agents.physics_agent import PhysicsAgent
from marl_packing.agents.proposer_agent import ProposerAgent
from marl_packing.envs.packing_env import PackingConfig, PackingEnv
from marl_packing.sim2real.bridge import MockRobot, Sim2RealBridge
from marl_packing.sim2real.reality_gap import RealityGapCalibrator
from marl_packing.training.ppo_trainer import MARLTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate agents in sim and across the Sim2Real bridge.")
    p.add_argument("--load-dir", type=str, default="checkpoints",
                   help="directory with proposer.npz / physics.npz (fresh agents if missing)")
    p.add_argument("--episodes", type=int, default=20, help="greedy sim evaluation episodes")
    p.add_argument("--bin-size", type=int, nargs=3, default=(10, 10, 10), metavar=("W", "D", "H"))
    p.add_argument("--num-boxes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deploy-boxes", type=int, default=40, help="boxes fed to the mock robot")
    return p.parse_args()


def deploy(proposer, physics, args, seed, calibrator=None):
    robot = MockRobot(
        PackingConfig(bin_size=tuple(args.bin_size), num_boxes=args.deploy_boxes, seed=seed),
        seed=seed,
    )
    bridge = Sim2RealBridge(proposer, physics, robot, calibrator=calibrator)
    stats = bridge.run()
    return stats, bridge.calibrator


def main() -> None:
    args = parse_args()

    env = PackingEnv(PackingConfig(bin_size=tuple(args.bin_size), num_boxes=args.num_boxes, seed=args.seed))
    proposer = ProposerAgent(env, seed=args.seed)
    physics = PhysicsAgent(seed=args.seed)

    proposer_path = os.path.join(args.load_dir, "proposer.npz")
    physics_path = os.path.join(args.load_dir, "physics.npz")
    if os.path.exists(proposer_path) and os.path.exists(physics_path):
        proposer.load(proposer_path)
        physics.load(physics_path)
        print(f"Loaded checkpoints from {args.load_dir}/")
    else:
        print(f"No checkpoints in {args.load_dir}/ — evaluating fresh (untrained) agents. "
              "Run scripts/train.py first for meaningful numbers.")

    # ---- 1. greedy evaluation in the nominal simulator --------------------
    trainer = MARLTrainer(env, proposer, physics, TrainerConfig(seed=args.seed))
    stats = [trainer.run_episode(greedy=True) for _ in range(args.episodes)]
    print(f"\n[sim] {args.episodes} greedy episodes: "
          f"mean fill {np.mean([s.fill_ratio for s in stats]):.3f} | "
          f"mean reward {np.mean([s.reward for s in stats]):.3f}")

    # ---- 2. first deployment: measure the reality gap ---------------------
    stats1, calibrator = deploy(proposer, physics, args, seed=args.seed + 1)
    report1 = calibrator.report()
    print(f"\n[real #1] placed {stats1.boxes_placed}/{stats1.boxes_attempted} "
          f"| unstable {stats1.unstable_placements} | vetoes {stats1.vetoes} "
          f"| fill {stats1.fill_ratio:.3f}")
    print(f"[gap  #1] disagreement {report1.disagreement_rate:.2%} "
          f"| critic calibration error {report1.calibration_error:.3f} "
          f"| n={report1.n_samples}")
    print(f"[gap  #1] suggested domain randomization: "
          f"dim_tolerance={report1.suggested_config.dim_tolerance:.3f}, "
          f"obs_noise_std={report1.suggested_config.obs_noise_std:.3f}")

    # ---- 3. close the gap: fine-tune the critic on real outcomes ----------
    loss = calibrator.finetune_physics_agent(physics, epochs=10)
    print(f"\n[finetune] critic BCE on real outcomes after fine-tuning: {loss:.4f}")

    # ---- 4. redeploy with the grounded critic -----------------------------
    stats2, calibrator2 = deploy(proposer, physics, args, seed=args.seed + 2,
                                 calibrator=RealityGapCalibrator())
    report2 = calibrator2.report()
    print(f"\n[real #2] placed {stats2.boxes_placed}/{stats2.boxes_attempted} "
          f"| unstable {stats2.unstable_placements} | vetoes {stats2.vetoes} "
          f"| fill {stats2.fill_ratio:.3f}")
    print(f"[gap  #2] disagreement {report2.disagreement_rate:.2%} "
          f"| critic calibration error {report2.calibration_error:.3f} "
          f"| n={report2.n_samples}")

    delta = report1.calibration_error - report2.calibration_error
    print(f"\nCalibration error change after fine-tuning: {delta:+.3f} "
          f"({'improved' if delta > 0 else 'no improvement — train longer or widen randomization'})")


if __name__ == "__main__":
    main()
