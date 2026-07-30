"""Shortest-path planning over GNN-predicted edge latencies.

Every planning call re-queries the DynamicGNN with the latest graph snapshot,
so routes adapt in real time as congestion changes.

Industrial additions:
- **Edge blocking** — a stalled forklift, a spill, or maintenance closes an
  aisle; ``block_edge`` removes it from planning until ``unblock_edge``.
- **Explicit unreachability** — when blocking disconnects the goal, planners
  must fail loudly (``NoRouteError``), never fall back to a stale route.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from marl_sim2real.gnn.dynamic_gnn import DynamicGNN, WarehouseGraph


class NoRouteError(RuntimeError):
    """Raised when every route between start and goal is blocked."""


class PathOptimizer:
    def __init__(self, graph: WarehouseGraph, gnn: DynamicGNN):
        self.graph = graph
        self.gnn = gnn
        self.blocked: set[int] = set()

    # --------------------------------------------------------------- blocking
    def block_edge(self, edge_id: int) -> None:
        if not 0 <= edge_id < self.graph.num_edges:
            raise ValueError(f"edge_id {edge_id} out of range")
        self.blocked.add(edge_id)

    def unblock_edge(self, edge_id: int) -> None:
        self.blocked.discard(edge_id)

    def block_between(self, u: int, v: int) -> list[int]:
        """Block every edge connecting nodes u and v (both directions).
        Returns the blocked edge ids — an aisle closure in one call."""
        src, dst = self.graph.edge_index
        ids = [
            int(e) for e in range(self.graph.num_edges)
            if {int(src[e]), int(dst[e])} == {u, v}
        ]
        for e in ids:
            self.blocked.add(e)
        return ids

    # --------------------------------------------------------------- planning
    def build_weighted_graph(self, node_feats: np.ndarray, edge_feats: np.ndarray) -> nx.DiGraph:
        latencies = self.gnn.predict_latencies(node_feats, edge_feats)
        g = nx.DiGraph()
        g.add_nodes_from(range(self.graph.num_nodes))
        src, dst = self.graph.edge_index
        for edge_id, (u, v) in enumerate(zip(src, dst)):
            if edge_id in self.blocked:
                continue
            g.add_edge(int(u), int(v), weight=float(max(latencies[edge_id], 1e-6)),
                       edge_id=edge_id)
        return g

    def shortest_path(
        self, start: int, goal: int, node_feats: np.ndarray, edge_feats: np.ndarray
    ) -> tuple:
        """Returns (path_nodes, expected_latency_seconds).

        Raises :class:`NoRouteError` if blocking has disconnected the goal.
        """
        g = self.build_weighted_graph(node_feats, edge_feats)
        try:
            path = nx.dijkstra_path(g, start, goal, weight="weight")
        except nx.NetworkXNoPath as exc:
            raise NoRouteError(
                f"no route {start} -> {goal}: {len(self.blocked)} edge(s) blocked"
            ) from exc
        cost = nx.path_weight(g, path, weight="weight")
        return path, float(cost)

    # ------------------------------------------------------------------- SLA
    def eta_bound(self, path: list, baseline, k: float = 2.0) -> tuple:
        """Probabilistic ETA promise for a route: (mean_eta, upper_bound).

        Industrial dispatch needs a *deliverable* time window, not a point
        estimate. Using the sim baseline's per-edge latency variance and
        treating edges as independent: bound = sum(mu_e) + k * sqrt(sum(var_e))
        over the first matching edge of each hop. k=2 covers ~97.7% of runs
        under the normal approximation.
        """
        if len(path) < 2:
            return 0.0, 0.0
        src, dst = self.graph.edge_index
        mean_eta = var = 0.0
        for u, v in zip(path, path[1:]):
            hits = np.flatnonzero((src == u) & (dst == v))
            if hits.size == 0:
                raise ValueError(f"path hop {u}->{v} is not a graph edge")
            e = int(hits[0])
            mean_eta += float(baseline.mean[e])
            var += float(baseline.std[e]) ** 2
        return mean_eta, mean_eta + k * float(np.sqrt(var))
