import numpy as np

from marl_sim2real.config import DriftConfig, GNNConfig
from marl_sim2real.gnn import DynamicGNN, WarehouseGraph
from marl_sim2real.sim2real import DriftDetector, IdealDataGenerator, SelfCorrection
from marl_sim2real.sim2real.ideal_data_generator import RealWorldSimulator, SimBaseline


def make_baseline(graph):
    generator = IdealDataGenerator(graph, use_pybullet=False, seed=0)
    baseline, snapshots, samples = generator.generate(runs_per_edge=15)
    return baseline, snapshots, samples


def test_no_false_positive_without_drift():
    graph = WarehouseGraph(num_nodes=8, seed=0)
    baseline, _, _ = make_baseline(graph)
    detector = DriftDetector(baseline, DriftConfig(k_sigma=3.0, window_size=10, min_samples=5))
    real = RealWorldSimulator(baseline, seed=1)
    for _ in range(30):
        assert detector.update(real.observe()) is None


def test_drift_detected_after_injection():
    graph = WarehouseGraph(num_nodes=8, seed=0)
    baseline, _, _ = make_baseline(graph)
    cfg = DriftConfig(k_sigma=3.0, window_size=10, min_samples=5)
    detector = DriftDetector(baseline, cfg)
    real = RealWorldSimulator(baseline, seed=1)
    drifted_edges = np.array([0, 3])
    real.inject_drift(drifted_edges, factor=1.8)

    event = None
    for _ in range(cfg.window_size + 5):
        event = detector.update(real.observe())
        if event is not None:
            break
    assert event is not None
    assert set(drifted_edges).issubset(set(event.edge_ids.tolist()))
    assert (np.abs(event.z_scores) > cfg.k_sigma).all()


def test_self_correction_reduces_error_and_resets():
    graph = WarehouseGraph(num_nodes=8, seed=0)
    baseline, snapshots, samples = make_baseline(graph)
    gnn = DynamicGNN(graph, GNNConfig(hidden_dim=32, num_layers=2))
    gnn.fit(snapshots, samples, epochs=25)

    cfg = DriftConfig(k_sigma=3.0, window_size=10, min_samples=5, recalib_steps=60)
    detector = DriftDetector(baseline, cfg)
    corrector = SelfCorrection(gnn, detector, cfg)
    real = RealWorldSimulator(baseline, seed=1)
    real.inject_drift(np.array([1, 4]), factor=1.8)

    recent = []
    event = None
    for _ in range(cfg.window_size + 5):
        obs = real.observe()
        recent.append((graph.sample_features(), obs))
        event = detector.update(obs)
        if event is not None:
            break
    assert event is not None

    record = corrector.handle(
        event,
        snapshots=[s for s, _ in recent],
        real_targets=[o for _, o in recent],
    )
    assert record.post_error < record.pre_error
    # windows for corrected edges were cleared
    for edge_id in event.edge_ids:
        assert len(detector.windows[int(edge_id)]) == 0


def test_baseline_roundtrip(tmp_path):
    baseline = SimBaseline(mean=np.array([1.0, 2.0]), std=np.array([0.1, 0.2]), samples=5)
    path = tmp_path / "baseline.json"
    baseline.save(path)
    loaded = SimBaseline.load(path)
    assert np.allclose(loaded.mean, baseline.mean)
    assert np.allclose(loaded.std, baseline.std)
    assert loaded.samples == 5
