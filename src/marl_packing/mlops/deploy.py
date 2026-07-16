"""CLI: evaluate a challenger model version against the current champion, print a
comparison, and (with --auto-promote) flip the registry's champion pointer if the
promotion gate passes.

This is the local stand-in for a "deploy" step -- in a system with a real inference
server, promotion here is where you'd trigger pushing the new model to serving.
"""

from __future__ import annotations

import argparse

from marl_packing.mlops.challenger import GateResult, evaluate_challenger
from marl_packing.mlops.registry import ModelRegistry
from marl_packing.utils.config import load_config
from marl_packing.utils.logging import get_logger

logger = get_logger(__name__)


def _print_report(result: GateResult) -> None:
    c = result.challenger_report
    print(
        f"Challenger {c.version_id}: sim_utilization={c.sim_utilization_pct:.2f}% "
        f"real_proxy_utilization={c.real_proxy_utilization_pct:.2f}% "
        f"sim2real_gap={c.sim2real_gap_pct:.2f}pp "
        f"stability={c.stability_success_rate_pct:.2f}%"
    )
    if result.champion_report:
        h = result.champion_report
        print(
            f"Champion   {h.version_id}: sim_utilization={h.sim_utilization_pct:.2f}% "
            f"real_proxy_utilization={h.real_proxy_utilization_pct:.2f}% "
            f"sim2real_gap={h.sim2real_gap_pct:.2f}pp "
            f"stability={h.stability_success_rate_pct:.2f}%"
        )
    print(f"Promotion gate: {'PASS' if result.promote else 'FAIL'}")
    for reason in result.reasons:
        print(f"  - {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger", required=True, help="Version id to evaluate, e.g. v3")
    parser.add_argument("--registry-dir", default="models/registry")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--config", default="configs/challenger_eval.yaml")
    parser.add_argument("--physics-config", default="configs/train_physics.yaml")
    parser.add_argument("--auto-promote", action="store_true")
    args = parser.parse_args()

    registry = ModelRegistry(args.registry_dir)
    env_config = load_config(args.env_config)
    eval_config = load_config(args.config)
    physics_config = load_config(args.physics_config)

    result = evaluate_challenger(
        args.challenger, registry, env_config, eval_config, reward_shaping=physics_config.get("reward_shaping")
    )
    _print_report(result)

    if result.promote:
        if args.auto_promote:
            registry.set_champion(args.challenger)
            logger.info("Promoted %s to champion.", args.challenger)
        else:
            logger.info("Challenger %s passes the gate but --auto-promote was not set; not promoting.", args.challenger)
    else:
        logger.info("Challenger %s did not pass the promotion gate; champion unchanged.", args.challenger)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
