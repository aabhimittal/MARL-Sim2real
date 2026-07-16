"""Champion/Challenger promotion gate.

Evaluates a challenger model version against the current champion on identical paired
benchmark episodes (same seeds, both the "sim" and "real_proxy" domain-randomization
modes -- see evaluation/benchmark.py), then decides whether the challenger clears the bar
to be promoted. See docs/MLOPS.md for the full rationale and configs/challenger_eval.yaml
for the gate thresholds.
"""

from __future__ import annotations

import dataclasses

from stable_baselines3 import PPO

from marl_packing.evaluation.benchmark import run_benchmark
from marl_packing.evaluation.metrics import sim2real_gap_pct
from marl_packing.mlops.registry import ModelRegistry, ModelVersion


@dataclasses.dataclass
class EvalReport:
    version_id: str
    sim_utilization_pct: float
    real_proxy_utilization_pct: float
    sim2real_gap_pct: float
    stability_success_rate_pct: float


@dataclasses.dataclass
class GateResult:
    promote: bool
    reasons: list[str]
    challenger_report: EvalReport
    champion_report: EvalReport | None


def _evaluate_version(
    version: ModelVersion, env_config: dict, eval_config: dict, reward_shaping: dict | None
) -> EvalReport:
    packer_model = PPO.load(version.packer_path)
    physics_model = PPO.load(version.physics_path)
    num_episodes = eval_config["num_eval_episodes"]
    seed_base = eval_config["eval_seed_base"]

    sim_metrics = run_benchmark(packer_model, physics_model, env_config, "train", num_episodes, seed_base, reward_shaping)
    real_metrics = run_benchmark(
        packer_model, physics_model, env_config, "real_proxy", num_episodes, seed_base, reward_shaping
    )

    return EvalReport(
        version_id=version.version,
        sim_utilization_pct=sim_metrics["utilization_pct"],
        real_proxy_utilization_pct=real_metrics["utilization_pct"],
        sim2real_gap_pct=sim2real_gap_pct(sim_metrics["utilization_pct"], real_metrics["utilization_pct"]),
        stability_success_rate_pct=sim_metrics["stability_success_rate_pct"],
    )


def decide_promotion(
    challenger_report: EvalReport,
    champion_report: EvalReport | None,
    eval_config: dict,
) -> GateResult:
    """Pure gate-decision logic, factored out of evaluate_challenger() so it's unit
    testable against fabricated EvalReport values without running any model or env."""
    if champion_report is None:
        promote = bool(eval_config.get("bootstrap_auto_promote", True))
        reasons = [
            "No champion registered yet -- bootstrap auto-promote."
            if promote
            else "No champion registered yet and bootstrap_auto_promote is disabled."
        ]
        return GateResult(promote=promote, reasons=reasons, challenger_report=challenger_report, champion_report=None)

    gate = eval_config["promotion_gate"]
    utilization_improvement = challenger_report.sim_utilization_pct - champion_report.sim_utilization_pct
    stability_regression = champion_report.stability_success_rate_pct - challenger_report.stability_success_rate_pct
    gap_regression = challenger_report.sim2real_gap_pct - champion_report.sim2real_gap_pct

    reasons = []
    passed = True
    if utilization_improvement < gate["min_utilization_improvement_pct"]:
        passed = False
        reasons.append(
            f"Utilization improvement {utilization_improvement:.2f}pp < required {gate['min_utilization_improvement_pct']}pp"
        )
    if stability_regression > gate["max_stability_regression_pct"]:
        passed = False
        reasons.append(
            f"Stability regression {stability_regression:.2f}pp > allowed {gate['max_stability_regression_pct']}pp"
        )
    if gap_regression > gate["max_sim2real_gap_regression_pct"]:
        passed = False
        reasons.append(
            f"Sim2Real gap regression {gap_regression:.2f}pp > allowed {gate['max_sim2real_gap_regression_pct']}pp"
        )
    if passed:
        reasons.append("All promotion gate conditions satisfied.")

    return GateResult(promote=passed, reasons=reasons, challenger_report=challenger_report, champion_report=champion_report)


def evaluate_challenger(
    challenger_id: str,
    registry: ModelRegistry,
    env_config: dict,
    eval_config: dict,
    reward_shaping: dict | None = None,
) -> GateResult:
    challenger = registry.get_version(challenger_id)
    challenger_report = _evaluate_version(challenger, env_config, eval_config, reward_shaping)
    registry.update_metrics(challenger_id, dataclasses.asdict(challenger_report))

    champion = registry.get_champion()
    if champion is None or champion.version == challenger_id:
        return decide_promotion(challenger_report, None, eval_config)

    champion_report = _evaluate_version(champion, env_config, eval_config, reward_shaping)
    return decide_promotion(challenger_report, champion_report, eval_config)
