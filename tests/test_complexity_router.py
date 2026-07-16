from marl_sim2real.llm import ComplexityRouter, TaskSignals


def test_simple_task_routes_small():
    router = ComplexityRouter()
    decision = router.route(TaskSignals(prompt="Format this log line."))
    assert decision.tier == 0
    assert "haiku" in decision.model


def test_hard_task_routes_large():
    router = ComplexityRouter()
    decision = router.route(TaskSignals(
        prompt="Diagnose cascading multi-edge failure " * 50,
        drift_sigma=9.0,
        affected_edges=12,
        requires_reasoning=True,
    ))
    assert decision.tier == 2
    assert "opus" in decision.model


def test_escalation_after_failures():
    router = ComplexityRouter()
    base = TaskSignals(prompt="Summarise the drift event.")
    first = router.route(base)
    retried = router.route(TaskSignals(prompt=base.prompt, prior_failures=2))
    assert retried.tier > first.tier


def test_monotonic_in_drift_severity():
    router = ComplexityRouter()
    low = router.complexity_score(TaskSignals(prompt="x", drift_sigma=3.0))
    high = router.complexity_score(TaskSignals(prompt="x", drift_sigma=8.0))
    assert high > low


def test_savings_report_counts_and_saves():
    router = ComplexityRouter()
    router.route(TaskSignals(prompt="tiny"))
    router.route(TaskSignals(prompt="big " * 600, requires_reasoning=True,
                             drift_sigma=9.0, affected_edges=10))
    report = router.savings_report()
    assert report["calls"] == 2
    assert sum(report["tier_counts"]) == 2
    assert report["routed_cost_usd"] < report["always_large_cost_usd"]
    assert report["savings_pct"] > 0


def test_offline_completion_stub(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    router = ComplexityRouter()
    out = router.complete(TaskSignals(prompt="Say hello."))
    assert out.startswith("[offline:")
