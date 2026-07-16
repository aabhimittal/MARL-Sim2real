"""Evaluation metrics: box utilization, stability success rate, and the Sim2Real gap.

Kept as pure functions over plain lists/dicts (rather than methods on some larger
evaluator class) so they're trivial to unit test without spinning up PyBullet or SB3.
"""

from __future__ import annotations

import numpy as np


def box_utilization_pct(episode_summaries: list[dict]) -> float:
    """Mean bin volume utilization (%) across episodes."""
    if not episode_summaries:
        return 0.0
    return float(np.mean([e["utilization_pct"] for e in episode_summaries]))


def stability_success_rate_pct(episode_outcomes: list[str]) -> float:
    """% of ACCEPTED placements that were actually stable, out of all accepted placements.

    Rejected placements aren't counted here -- this measures whether the physics agent's
    accepts can be trusted, which is the outcome that actually matters for real deployment
    (an accepted-but-unstable placement is the costly failure mode; a rejected-but-stable
    placement only costs utilization, tracked separately via box_utilization_pct).
    """
    accepted = [o for o in episode_outcomes if o in ("accepted_stable", "accepted_unstable")]
    if not accepted:
        return 100.0
    stable = sum(1 for o in accepted if o == "accepted_stable")
    return 100.0 * stable / len(accepted)


def sim2real_gap_pct(sim_utilization_pct: float, real_proxy_utilization_pct: float) -> float:
    """Positive gap means the policy looks better in sim than under the real-world-proxy
    randomization range -- the classic Sim2Real overfitting signature (see docs/SIM2REAL.md)."""
    return sim_utilization_pct - real_proxy_utilization_pct
