"""Pure unit tests for marl_packing.evaluation.metrics (no PyBullet/SB3 needed)."""

import pytest

from marl_packing.evaluation.metrics import (
    box_utilization_pct,
    stability_success_rate_pct,
    sim2real_gap_pct,
)


class TestBoxUtilizationPct:
    def test_empty_input_returns_zero(self):
        assert box_utilization_pct([]) == 0.0

    def test_single_episode(self):
        assert box_utilization_pct([{"utilization_pct": 42.5}]) == pytest.approx(42.5)

    def test_averages_across_episodes(self):
        summaries = [
            {"utilization_pct": 10.0},
            {"utilization_pct": 20.0},
            {"utilization_pct": 30.0},
        ]
        assert box_utilization_pct(summaries) == pytest.approx(20.0)

    def test_returns_python_float(self):
        result = box_utilization_pct([{"utilization_pct": 55.0}])
        assert type(result) is float

    def test_ignores_extra_keys(self):
        summaries = [
            {"utilization_pct": 50.0, "n_placed": 7, "episode": 0},
            {"utilization_pct": 70.0, "n_placed": 9, "episode": 1},
        ]
        assert box_utilization_pct(summaries) == pytest.approx(60.0)


class TestStabilitySuccessRatePct:
    def test_empty_input_returns_100(self):
        assert stability_success_rate_pct([]) == 100.0

    def test_only_rejected_outcomes_returns_100(self):
        outcomes = ["rejected_low_confidence", "rejected_overhang", "rejected_collision"]
        assert stability_success_rate_pct(outcomes) == 100.0

    def test_all_accepted_stable(self):
        assert stability_success_rate_pct(["accepted_stable"] * 4) == 100.0

    def test_all_accepted_unstable(self):
        assert stability_success_rate_pct(["accepted_unstable"] * 3) == 0.0

    def test_mixed_list_ignores_rejected(self):
        # 3 accepted (2 stable, 1 unstable); the rejected_* entries must not
        # enter the denominator: rate = 2/3, not 2/5.
        outcomes = [
            "accepted_stable",
            "rejected_low_confidence",
            "accepted_unstable",
            "rejected_overhang",
            "accepted_stable",
        ]
        assert stability_success_rate_pct(outcomes) == pytest.approx(100.0 * 2 / 3)

    def test_half_and_half(self):
        outcomes = ["accepted_stable", "accepted_unstable"]
        assert stability_success_rate_pct(outcomes) == pytest.approx(50.0)


class TestSim2RealGapPct:
    def test_positive_gap_means_sim_better_than_real(self):
        # Sim looks better than the real-world proxy -> positive gap.
        assert sim2real_gap_pct(80.0, 65.0) == pytest.approx(15.0)
        assert sim2real_gap_pct(80.0, 65.0) > 0

    def test_negative_gap_when_real_proxy_better(self):
        assert sim2real_gap_pct(60.0, 70.0) == pytest.approx(-10.0)
        assert sim2real_gap_pct(60.0, 70.0) < 0

    def test_zero_gap_when_equal(self):
        assert sim2real_gap_pct(55.5, 55.5) == pytest.approx(0.0)
