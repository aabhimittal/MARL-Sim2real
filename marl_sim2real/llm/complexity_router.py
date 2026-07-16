"""Token-cost optimisation via complexity-based model switching.

The pipeline occasionally consults an LLM — e.g. to summarise a drift event
for operators, to explain why a packing episode failed, or to propose
recovery strategies. Those tasks vary hugely in difficulty, so routing every
call to the largest model wastes tokens (dollars) and latency.

The router scores task complexity in [0, 1] from cheap-to-compute signals and
picks the smallest model expected to handle it:

    score < low_threshold   -> small  (Haiku)   e.g. format a log line
    score < high_threshold  -> medium (Sonnet)  e.g. summarise a drift event
    otherwise               -> large  (Opus)    e.g. multi-edge failure diagnosis

Signals: prompt length, structural depth (how nested/branchy the task is),
numeric drift severity, and whether previous attempts failed (escalation).
Every decision is recorded so realised savings vs "always use the largest
model" can be reported.
"""

from __future__ import annotations

import dataclasses
import math
import os

from marl_sim2real.config import RouterConfig


@dataclasses.dataclass
class TaskSignals:
    prompt: str
    drift_sigma: float = 0.0        # max |z| involved, 0 if not drift-related
    affected_edges: int = 0         # blast radius of the event
    requires_reasoning: bool = False  # caller knows the task needs multi-step logic
    prior_failures: int = 0         # escalate after failed attempts on smaller models


@dataclasses.dataclass
class RoutingDecision:
    model: str
    tier: int                       # 0 = small, 1 = medium, 2 = large
    complexity: float
    est_input_tokens: int
    reason: str


class ComplexityRouter:
    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()
        self.decisions: list[RoutingDecision] = []

    # ---------------------------------------------------------------- score
    def complexity_score(self, signals: TaskSignals) -> float:
        # Length: saturating — 2k chars of prompt ~ moderately complex context
        length_term = 1.0 - math.exp(-len(signals.prompt) / 2000.0)
        # Drift severity: 3 sigma is routine, 8+ sigma is an anomaly worth
        # careful analysis
        drift_term = min(max(signals.drift_sigma - 3.0, 0.0) / 5.0, 1.0)
        # Blast radius: multi-edge failures need cross-edge reasoning
        radius_term = min(signals.affected_edges / 10.0, 1.0)
        reasoning_term = 1.0 if signals.requires_reasoning else 0.0
        # Escalation: each failed attempt strongly pushes the score up
        escalation = min(signals.prior_failures * 0.35, 1.0)

        score = (
            0.15 * length_term
            + 0.30 * drift_term
            + 0.20 * radius_term
            + 0.35 * reasoning_term
        )
        return float(min(score + escalation, 1.0))

    # ---------------------------------------------------------------- route
    def route(self, signals: TaskSignals) -> RoutingDecision:
        score = self.complexity_score(signals)
        cfg = self.config
        if score < cfg.low_threshold:
            tier = 0
        elif score < cfg.high_threshold:
            tier = 1
        else:
            tier = 2
        decision = RoutingDecision(
            model=cfg.models[tier],
            tier=tier,
            complexity=score,
            est_input_tokens=max(1, len(signals.prompt) // 4),
            reason=self._explain(signals, score, tier),
        )
        self.decisions.append(decision)
        return decision

    def complete(self, signals: TaskSignals, max_tokens: int = 512) -> str:
        """Route and execute. Uses the Anthropic API when ANTHROPIC_API_KEY is
        set; otherwise returns a deterministic offline stub (keeps the
        pipeline and tests runnable without network access)."""
        decision = self.route(signals)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic()
                msg = client.messages.create(
                    model=decision.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": signals.prompt}],
                )
                return msg.content[0].text
            except Exception as exc:  # network/api failure -> stub, not crash
                return f"[offline:{decision.model}] API unavailable ({exc})"
        return (
            f"[offline:{decision.model}] complexity={decision.complexity:.2f} "
            f"— would answer: {signals.prompt[:80]}..."
        )

    # -------------------------------------------------------------- report
    def savings_report(self, avg_output_tokens: int = 300) -> dict:
        """Estimated cost vs routing everything to the largest model."""
        cfg = self.config
        large_in, large_out = cfg.pricing[-1]
        routed_cost = baseline_cost = 0.0
        for d in self.decisions:
            tin, tout = d.est_input_tokens, avg_output_tokens
            price_in, price_out = cfg.pricing[d.tier]
            routed_cost += tin / 1e6 * price_in + tout / 1e6 * price_out
            baseline_cost += tin / 1e6 * large_in + tout / 1e6 * large_out
        return {
            "calls": len(self.decisions),
            "tier_counts": [sum(1 for d in self.decisions if d.tier == t) for t in range(3)],
            "routed_cost_usd": round(routed_cost, 6),
            "always_large_cost_usd": round(baseline_cost, 6),
            "savings_pct": round(100 * (1 - routed_cost / baseline_cost), 1)
            if baseline_cost else 0.0,
        }

    @staticmethod
    def _explain(signals: TaskSignals, score: float, tier: int) -> str:
        tiers = ["small", "medium", "large"]
        drivers = []
        if signals.requires_reasoning:
            drivers.append("multi-step reasoning required")
        if signals.drift_sigma > 3:
            drivers.append(f"drift at {signals.drift_sigma:.1f} sigma")
        if signals.affected_edges > 3:
            drivers.append(f"{signals.affected_edges} edges affected")
        if signals.prior_failures:
            drivers.append(f"escalated after {signals.prior_failures} failure(s)")
        driver_text = "; ".join(drivers) if drivers else "routine task"
        return f"complexity {score:.2f} -> {tiers[tier]} model ({driver_text})"
