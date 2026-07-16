#!/usr/bin/env python3
"""Run the full Sim2Real pipeline end to end and export a deployable bundle.

    train (randomized sim) -> measure gap -> calibrate -> adapt -> validate -> export

Usage:
    python scripts/run_bridge.py --sim-episodes 300 --adapt-episodes 150 --out artifacts/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_sim2real.export import export_bundle
from marl_sim2real.sim2real import Sim2RealBridge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-episodes", type=int, default=300)
    ap.add_argument("--adapt-episodes", type=int, default=150)
    ap.add_argument("--bin", type=int, nargs=3, default=[8, 8, 8], metavar=("L", "W", "H"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="artifacts")
    args = ap.parse_args()

    bridge = Sim2RealBridge(bin_size=tuple(args.bin), seed=args.seed)
    result = bridge.run(sim_episodes=args.sim_episodes, adapt_episodes=args.adapt_episodes)

    print("\n=== Sim2Real summary ===")
    print(f"pre-adaptation  gap: {result.pre_gap.gap:.1%} (real util {result.pre_gap.real_utilization:.1%})")
    print(f"post-adaptation gap: {result.post_gap.gap:.1%} (real util {result.post_gap.real_utilization:.1%})")
    print(f"calibration agreement: {result.calibration.agreement:.1%}")
    print(f"accepted for deployment: {result.accepted}")

    if result.accepted:
        manifest = export_bundle(
            args.out, bridge.proposer, bridge.physics_agent, tuple(args.bin),
            bridge_result=result, name="packing_policy_real",
        )
        print(f"exported deployable bundle -> {manifest}")
    else:
        print("policy did not clear the acceptance bar; not exporting")


if __name__ == "__main__":
    main()
