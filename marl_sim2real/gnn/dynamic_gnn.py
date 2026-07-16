"""Dynamic Graph Neural Network for real-time edge-latency prediction.

The warehouse is a graph of waypoints. Node features (queue length, congestion,
battery of nearby robots, ...) and edge features (distance, load, surface type)
change every tick — the GNN re-embeds the graph each call, which is what makes
it "dynamic": same weights, time-varying inputs.

Implemented with plain torch message passing (no torch_geometric dependency).
Each edge also carries a learnable bias used by the sim2real self-correction
module for fast recalibration without touching the shared GNN weights.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from marl_sim2real.config import GNNConfig


class WarehouseGraph:
    """A random-geometric warehouse waypoint graph with dynamic features."""

    def __init__(self, num_nodes: int = 20, radius: float = 0.35, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.positions = self.rng.random((num_nodes, 2))
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < radius:
                    edges.append((i, j))
                    edges.append((j, i))
        # Ensure connectivity with a chain fallback
        for i in range(num_nodes - 1):
            if (i, i + 1) not in edges:
                edges.append((i, i + 1))
                edges.append((i + 1, i))
        self.edge_index = np.array(edges, dtype=np.int64).T  # (2, E)
        self.num_edges = self.edge_index.shape[1]

    def distances(self) -> np.ndarray:
        src, dst = self.edge_index
        return np.linalg.norm(self.positions[src] - self.positions[dst], axis=1)

    def sample_features(self, congestion_level: float = 0.3) -> tuple:
        """Sample a snapshot of dynamic node/edge features (one tick)."""
        n, e = self.num_nodes, self.num_edges
        node_feats = np.column_stack([
            self.positions,                                   # static layout (2)
            self.rng.random(n) * congestion_level,            # queue length
            self.rng.random(n),                               # robot density
        ]).astype(np.float32)
        edge_feats = np.column_stack([
            self.distances(),                                 # distance
            self.rng.random(e) * congestion_level,            # current load
            self.rng.integers(0, 2, e).astype(np.float32),    # surface type
        ]).astype(np.float32)
        return node_feats, edge_feats


class GNNLayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim), nn.ReLU(), nn.Linear(dim, dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim),
        )

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        messages = self.message_mlp(torch.cat([h[src], h[dst], edge_attr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        return h + self.update_mlp(torch.cat([h, agg], dim=-1))


class DynamicGNN(nn.Module):
    """Predicts per-edge traversal latency (seconds) from a graph snapshot."""

    def __init__(self, graph: WarehouseGraph, config: GNNConfig | None = None):
        super().__init__()
        self.config = config or GNNConfig()
        cfg = self.config
        self.graph = graph
        self.node_encoder = nn.Linear(cfg.node_feat_dim, cfg.hidden_dim)
        self.edge_encoder = nn.Linear(cfg.edge_feat_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList(
            GNNLayer(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.num_layers)
        )
        self.latency_head = nn.Sequential(
            nn.Linear(3 * cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1), nn.Softplus(),   # latency > 0
        )
        # Per-edge calibration bias: the sim2real self-correction knob.
        self.edge_bias = nn.Parameter(torch.zeros(graph.num_edges))

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        edge_index = torch.from_numpy(self.graph.edge_index)
        h = self.node_encoder(node_feats)
        e = self.edge_encoder(edge_feats)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        src, dst = edge_index
        edge_repr = torch.cat([h[src], h[dst], e], dim=-1)
        latency = self.latency_head(edge_repr).squeeze(-1)
        return latency + torch.nn.functional.softplus(self.edge_bias) - np.log(2.0)

    def predict_latencies(self, node_feats: np.ndarray, edge_feats: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            out = self(torch.from_numpy(node_feats), torch.from_numpy(edge_feats))
        return out.numpy()

    def fit(
        self,
        snapshots: list,
        targets: list,
        epochs: int = 100,
        lr: float | None = None,
        only_bias: bool = False,
        edge_mask: np.ndarray | None = None,
    ) -> list:
        """Train on (node_feats, edge_feats) -> per-edge latency targets.

        only_bias=True freezes the shared network and adjusts just the
        per-edge calibration biases (used by self-correction).
        edge_mask restricts the loss to a subset of edges (e.g. drifted ones).
        """
        params = [self.edge_bias] if only_bias else list(self.parameters())
        optim = torch.optim.Adam(params, lr=lr or self.config.lr)
        loss_fn = nn.MSELoss()
        mask_t = None
        if edge_mask is not None:
            mask_t = torch.from_numpy(np.asarray(edge_mask, dtype=np.int64))
        losses = []
        self.train()
        for _ in range(epochs):
            total = 0.0
            for (nf, ef), y in zip(snapshots, targets):
                pred = self(torch.from_numpy(nf), torch.from_numpy(ef))
                target = torch.from_numpy(y.astype(np.float32))
                if mask_t is not None:
                    loss = loss_fn(pred[mask_t], target[mask_t])
                else:
                    loss = loss_fn(pred, target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total += float(loss.item())
            losses.append(total / max(len(snapshots), 1))
        return losses
