"""Industrial fleet-reliability edge cases: stuck sensors, impossible
readings, systemic-vs-local drift, degenerate baselines, SLA ETA bounds."""

import numpy as np
import pytest

from marl_sim2real.config import DriftConfig, GNNConfig
from marl_sim2real.gnn import DynamicGNN, PathOptimizer, WarehouseGraph
from marl_sim2real.sim2real import (
    DriftDetector,
    DriftScope,
    DriftScopeClassifier,
    EdgeHealth,
    SensorHealthMonitor,
    SimBaseline,
)
from marl_sim2real.sim2real.ideal_data_generator import (
    IdealDataGenerator,
    RealWorldSimulator,
)

E = 6  # edges in the toy monitors below


# ------------------------------------------------------------- sensor health
def test_stuck_sensor_flagged_and_masked():
    mon = SensorHealthMonitor(num_edges=E, stuck_ticks=3)
    frozen = np.full(E, 2.5)
    for _ in range(2):
        report = mon.sanitize(frozen)  # streak below threshold
        assert report.all_ok
    report = mon.sanitize(frozen)  # 3rd identical tick -> STUCK
    assert not report.all_ok
    assert all(s is EdgeHealth.STUCK for s in report.states)
    assert np.isnan(report.clean).all()  # masked for downstream dropout handling


def test_noisy_healthy_sensor_not_flagged():
    rng = np.random.default_rng(0)
    mon = SensorHealthMonitor(num_edges=E, stuck_ticks=3)
    for _ in range(20):
        report = mon.sanitize(2.0 + rng.normal(0, 0.05, E))
    assert report.all_ok


def test_invalid_readings_masked():
    mon = SensorHealthMonitor(num_edges=E)
    obs = np.full(E, 2.0)
    obs[1] = -0.4      # clock skew
    obs[2] = 0.0       # underflow
    obs[3] = 1e9       # unit bug (ns instead of s)
    report = mon.sanitize(obs)
    assert set(report.faulty_edges.tolist()) == {1, 2, 3}
    assert report.states[1] is EdgeHealth.INVALID
    assert np.isnan(report.clean[[1, 2, 3]]).all()
    assert report.clean[0] == 2.0  # healthy edges untouched


def test_stuck_sensor_recovers():
    mon = SensorHealthMonitor(num_edges=E, stuck_ticks=3)
    frozen = np.full(E, 2.5)
    for _ in range(4):
        mon.sanitize(frozen)
    assert mon.states[0] is EdgeHealth.STUCK
    report = mon.sanitize(np.linspace(2.0, 3.0, E))  # sane varying readings
    assert report.all_ok


# --------------------------------------------------------- drift scope logic
def _drifted_detector(drift_edges, factor):
    graph = WarehouseGraph(num_nodes=8, seed=0)
    baseline, _, _ = IdealDataGenerator(graph, use_pybullet=False, seed=0).generate(
        runs_per_edge=15
    )
    detector = DriftDetector(
        baseline, DriftConfig(k_sigma=3.0, window_size=8, min_samples=4)
    )
    real = RealWorldSimulator(baseline, seed=1)
    real.inject_drift(np.asarray(drift_edges), factor=factor)
    event = None
    for _ in range(13):
        event = detector.update(real.observe()) or event
    assert event is not None
    return detector, event, graph


def test_local_drift_classified_local():
    detector, event, _ = _drifted_detector([0, 3], factor=1.8)
    verdict = DriftScopeClassifier().classify(event, detector)
    assert verdict.scope is DriftScope.LOCAL
    assert verdict.recalibrate


def test_systemic_drift_blocks_recalibration():
    """Firmware regression: every edge slows 40% — map must NOT be recalibrated."""
    graph = WarehouseGraph(num_nodes=8, seed=0)
    detector, event, _ = _drifted_detector(np.arange(graph.num_edges), factor=1.4)
    verdict = DriftScopeClassifier().classify(event, detector)
    assert verdict.scope is DriftScope.SYSTEMIC
    assert not verdict.recalibrate


# ------------------------------------------------------ degenerate baselines
def test_zero_variance_baseline_no_inf():
    baseline = SimBaseline(mean=np.full(4, 2.0), std=np.zeros(4), samples=1)
    detector = DriftDetector(baseline, DriftConfig(window_size=4, min_samples=2))
    event = None
    for _ in range(4):
        event = detector.update(np.full(4, 2.5)) or event
    assert event is not None  # any deviation from a zero-sigma baseline drifts
    assert np.isfinite(event.z_scores).all()


# ------------------------------------------------------------------ SLA ETAs
def _optimizer():
    graph = WarehouseGraph(num_nodes=8, seed=0)
    gnn = DynamicGNN(graph, GNNConfig(hidden_dim=32, num_layers=2))
    baseline, _, _ = IdealDataGenerator(graph, use_pybullet=False, seed=0).generate(
        runs_per_edge=10
    )
    return graph, PathOptimizer(graph, gnn), baseline


def test_eta_bound_exceeds_mean_and_grows_with_k():
    graph, opt, baseline = _optimizer()
    nf, ef = graph.sample_features()
    path, _ = opt.shortest_path(0, graph.num_nodes - 1, nf, ef)
    mean_eta, bound2 = opt.eta_bound(path, baseline, k=2.0)
    _, bound3 = opt.eta_bound(path, baseline, k=3.0)
    assert bound2 > mean_eta > 0
    assert bound3 > bound2


def test_eta_bound_trivial_and_invalid_paths():
    graph, opt, baseline = _optimizer()
    assert opt.eta_bound([4], baseline) == (0.0, 0.0)  # start == goal
    with pytest.raises(ValueError):
        opt.eta_bound([0, 0], baseline)  # self-loop is not a graph edge
