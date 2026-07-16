import numpy as np

from marl_sim2real.config import GNNConfig
from marl_sim2real.gnn import DynamicGNN, PathOptimizer, WarehouseGraph


def small_setup():
    graph = WarehouseGraph(num_nodes=8, seed=0)
    gnn = DynamicGNN(graph, GNNConfig(hidden_dim=32, num_layers=2))
    return graph, gnn


def test_latency_predictions_positive_and_shaped():
    graph, gnn = small_setup()
    nf, ef = graph.sample_features()
    latencies = gnn.predict_latencies(nf, ef)
    assert latencies.shape == (graph.num_edges,)
    assert (latencies > -np.log(2.0)).all()  # softplus head + bounded bias


def test_training_reduces_loss():
    graph, gnn = small_setup()
    rng = np.random.default_rng(0)
    snapshots, targets = [], []
    for _ in range(6):
        nf, ef = graph.sample_features()
        snapshots.append((nf, ef))
        targets.append((2.0 + 3.0 * ef[:, 0] + rng.normal(0, 0.05, graph.num_edges)))
    losses = gnn.fit(snapshots, targets, epochs=30)
    assert losses[-1] < losses[0]


def test_dynamic_predictions_change_with_features():
    graph, gnn = small_setup()
    nf1, ef1 = graph.sample_features(congestion_level=0.1)
    nf2, ef2 = graph.sample_features(congestion_level=0.9)
    out1 = gnn.predict_latencies(nf1, ef1)
    out2 = gnn.predict_latencies(nf2, ef2)
    assert not np.allclose(out1, out2)


def test_path_optimizer_finds_valid_path():
    graph, gnn = small_setup()
    optimizer = PathOptimizer(graph, gnn)
    nf, ef = graph.sample_features()
    path, cost = optimizer.shortest_path(0, graph.num_nodes - 1, nf, ef)
    assert path[0] == 0 and path[-1] == graph.num_nodes - 1
    assert cost > 0
    # Path must follow actual graph edges
    edge_set = set(map(tuple, graph.edge_index.T.tolist()))
    for u, v in zip(path, path[1:]):
        assert (u, v) in edge_set
