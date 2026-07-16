"""Shortest-path planning over GNN-predicted edge latencies.

Every planning call re-queries the DynamicGNN with the latest graph snapshot,
so routes adapt in real time as congestion changes.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from marl_sim2real.gnn.dynamic_gnn import DynamicGNN, WarehouseGraph


class PathOptimizer:
    def __init__(self, graph: WarehouseGraph, gnn: DynamicGNN):
        self.graph = graph
        self.gnn = gnn

    def build_weighted_graph(self, node_feats: np.ndarray, edge_feats: np.ndarray) -> nx.DiGraph:
        latencies = self.gnn.predict_latencies(node_feats, edge_feats)
        g = nx.DiGraph()
        g.add_nodes_from(range(self.graph.num_nodes))
        src, dst = self.graph.edge_index
        for edge_id, (u, v) in enumerate(zip(src, dst)):
            g.add_edge(int(u), int(v), weight=float(max(latencies[edge_id], 1e-6)),
                       edge_id=edge_id)
        return g

    def shortest_path(
        self, start: int, goal: int, node_feats: np.ndarray, edge_feats: np.ndarray
    ) -> tuple:
        """Returns (path_nodes, expected_latency_seconds)."""
        g = self.build_weighted_graph(node_feats, edge_feats)
        path = nx.dijkstra_path(g, start, goal, weight="weight")
        cost = nx.path_weight(g, path, weight="weight")
        return path, float(cost)
