#!/usr/bin/env python3
"""Stage 2 — Generate "ideal" traffic data with PyBullet and pre-train the GNN.

Simulates robot traversals over every warehouse edge under sampled congestion
snapshots, producing:
  * a per-edge latency baseline (mean/std) for drift detection
  * (snapshot, latency) pairs used to train the DynamicGNN

Usage:
    python scripts/generate_sim_data.py --runs 30 --out data/
"""

import argparse
from pathlib import Path

import torch

from marl_sim2real.config import Config
from marl_sim2real.gnn import DynamicGNN, WarehouseGraph
from marl_sim2real.sim2real import IdealDataGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", default="data")
    parser.add_argument("--no-pybullet", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    graph = WarehouseGraph(num_nodes=args.nodes, seed=0)
    print(f"Warehouse graph: {graph.num_nodes} nodes, {graph.num_edges} directed edges")

    generator = IdealDataGenerator(graph, use_pybullet=not args.no_pybullet or None)
    baseline, snapshots, samples = generator.generate(runs_per_edge=args.runs)
    generator.close()
    baseline.save(out / "sim_baseline.json")
    print(f"Baseline saved: mean latency {baseline.mean.mean():.2f}s "
          f"(std {baseline.std.mean():.3f}s) over {args.runs} runs/edge")

    gnn = DynamicGNN(graph, cfg.gnn)
    losses = gnn.fit(snapshots, samples, epochs=args.epochs)
    torch.save(gnn.state_dict(), out / "gnn.pt")
    print(f"GNN trained: loss {losses[0]:.4f} -> {losses[-1]:.4f}; saved to {out/'gnn.pt'}")


if __name__ == "__main__":
    main()
