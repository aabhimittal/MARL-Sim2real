"""Tests for the pure promotion-gate decision logic in marl_packing.mlops.challenger."""

from __future__ import annotations

from marl_packing.mlops.challenger import EvalReport, decide_promotion

GATE = {
    "min_utilization_improvement_pct": 1.0,
    "max_stability_regression_pct": 2.0,
    "max_sim2real_gap_regression_pct": 3.0,
}


def eval_config(bootstrap_auto_promote: bool = True) -> dict:
    return {
        "promotion_gate": dict(GATE),
        "bootstrap_auto_promote": bootstrap_auto_promote,
    }


def make_report(
    version_id: str = "v1",
    sim_utilization_pct: float = 70.0,
    real_proxy_utilization_pct: float = 65.0,
    sim2real_gap_pct: float = 5.0,
    stability_success_rate_pct: float = 90.0,
) -> EvalReport:
    return EvalReport(
        version_id=version_id,
        sim_utilization_pct=sim_utilization_pct,
        real_proxy_utilization_pct=real_proxy_utilization_pct,
        sim2real_gap_pct=sim2real_gap_pct,
        stability_success_rate_pct=stability_success_rate_pct,
    )


def test_bootstrap_auto_promote_enabled_promotes():
    challenger = make_report()
    result = decide_promotion(challenger, None, eval_config(bootstrap_auto_promote=True))

    assert result.promote is True
    assert result.champion_report is None
    assert result.challenger_report is challenger
    assert any("bootstrap" in r.lower() for r in result.reasons)


def test_bootstrap_auto_promote_disabled_does_not_promote():
    challenger = make_report()
    result = decide_promotion(challenger, None, eval_config(bootstrap_auto_promote=False))

    assert result.promote is False
    assert result.champion_report is None
    assert any("bootstrap_auto_promote is disabled" in r for r in result.reasons)


def test_challenger_clears_all_thresholds_promotes():
    champion = make_report(
        version_id="champion",
        sim_utilization_pct=70.0,
        real_proxy_utilization_pct=65.0,
        sim2real_gap_pct=5.0,
        stability_success_rate_pct=90.0,
    )
    challenger = make_report(
        version_id="challenger",
        sim_utilization_pct=75.0,  # +5pp improvement, clears min 1.0
        real_proxy_utilization_pct=70.0,
        sim2real_gap_pct=5.0,  # no gap regression
        stability_success_rate_pct=90.0,  # no stability regression
    )

    result = decide_promotion(challenger, champion, eval_config())

    assert result.promote is True
    assert any("satisfied" in r.lower() for r in result.reasons)


def test_challenger_fails_utilization_only_rejected():
    champion = make_report(
        version_id="champion",
        sim_utilization_pct=70.0,
        sim2real_gap_pct=5.0,
        stability_success_rate_pct=90.0,
    )
    challenger = make_report(
        version_id="challenger",
        sim_utilization_pct=70.5,  # only +0.5pp, below min 1.0pp -> fails
        sim2real_gap_pct=5.0,  # unchanged, ok
        stability_success_rate_pct=90.0,  # unchanged, ok
    )

    result = decide_promotion(challenger, champion, eval_config())

    assert result.promote is False
    assert len(result.reasons) == 1
    assert "Utilization" in result.reasons[0]


def test_challenger_fails_stability_regression_only_rejected():
    champion = make_report(
        version_id="champion",
        sim_utilization_pct=70.0,
        sim2real_gap_pct=5.0,
        stability_success_rate_pct=90.0,
    )
    challenger = make_report(
        version_id="challenger",
        sim_utilization_pct=75.0,  # clears utilization bar
        sim2real_gap_pct=5.0,  # unchanged, ok
        stability_success_rate_pct=85.0,  # regressed by 5pp, above max allowed 2.0pp -> fails
    )

    result = decide_promotion(challenger, champion, eval_config())

    assert result.promote is False
    assert len(result.reasons) == 1
    assert "Stability" in result.reasons[0]


def test_challenger_fails_sim2real_gap_regression_only_rejected():
    champion = make_report(
        version_id="champion",
        sim_utilization_pct=70.0,
        sim2real_gap_pct=5.0,
        stability_success_rate_pct=90.0,
    )
    challenger = make_report(
        version_id="challenger",
        sim_utilization_pct=75.0,  # clears utilization bar
        sim2real_gap_pct=10.0,  # widened by 5pp, above max allowed 3.0pp -> fails
        stability_success_rate_pct=90.0,  # unchanged, ok
    )

    result = decide_promotion(challenger, champion, eval_config())

    assert result.promote is False
    assert len(result.reasons) == 1
    assert "gap" in result.reasons[0].lower()


def test_challenger_fails_multiple_criteria_has_multiple_reasons():
    champion = make_report(
        version_id="champion",
        sim_utilization_pct=70.0,
        sim2real_gap_pct=5.0,
        stability_success_rate_pct=90.0,
    )
    challenger = make_report(
        version_id="challenger",
        sim_utilization_pct=70.2,  # fails utilization improvement
        sim2real_gap_pct=15.0,  # fails gap regression
        stability_success_rate_pct=80.0,  # fails stability regression
    )

    result = decide_promotion(challenger, champion, eval_config())

    assert result.promote is False
    assert len(result.reasons) == 3
    joined = " ".join(result.reasons)
    assert "Utilization" in joined
    assert "Stability" in joined
    assert "gap" in joined.lower()
