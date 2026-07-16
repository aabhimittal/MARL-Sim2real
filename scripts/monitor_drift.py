#!/usr/bin/env python3
"""Stage 3 — Live sim2real drift monitoring with self-correction.

Streams (simulated) real-world latency observations into the DriftDetector.
When any edge deviates from the PyBullet baseline by more than k sigma, the
SelfCorrection module recalibrates the GNN's per-edge biases and updates the
baseline. The ComplexityRouter picks an appropriately sized LLM to narrate
each event for operators, and reports token-cost savings at the end.

Usage:
    python scripts/monitor_drift.py --ticks 120 --drift-at 40 --k 3.0
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from marl_sim2real.config import Config
from marl_sim2real.gnn import DynamicGNN, PathOptimizer, WarehouseGraph
from marl_sim2real.llm import ComplexityRouter, TaskSignals
from marl_sim2real.sim2real import DriftDetector, SelfCorrection, SimBaseline
from marl_sim2real.sim2real.ideal_data_generator import RealWorldSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="data")
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--drift-at", type=int, default=40,
                        help="tick at which real-world drift is injected")
    parser.add_argument("--drift-factor", type=float, default=1.6)
    parser.add_argument("--k", type=float, default=None, help="k-sigma threshold")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.k is not None:
        cfg.drift.k_sigma = args.k
    data = Path(args.data)

    graph = WarehouseGraph(num_nodes=20, seed=0)
    baseline = SimBaseline.load(data / "sim_baseline.json")
    gnn = DynamicGNN(graph, cfg.gnn)
    gnn.load_state_dict(torch.load(data / "gnn.pt", weights_only=True))

    detector = DriftDetector(baseline, cfg.drift)
    corrector = SelfCorrection(gnn, detector, cfg.drift, log_path=data / "corrections.jsonl")
    optimizer = PathOptimizer(graph, gnn)
    router = ComplexityRouter(cfg.router)
    real_world = RealWorldSimulator(baseline, seed=1)

    rng = np.random.default_rng(2)
    drift_edges = rng.choice(graph.num_edges, size=4, replace=False)
    recent = deque(maxlen=cfg.drift.window_size)

    print(f"Monitoring {graph.num_edges} edges, k={cfg.drift.k_sigma} sigma, "
          f"window={cfg.drift.window_size}\n")

    for tick in range(1, args.ticks + 1):
        if tick == args.drift_at:
            real_world.inject_drift(drift_edges, factor=args.drift_factor)
            print(f"tick {tick}: [injected] real-world drift x{args.drift_factor} "
                  f"on edges {sorted(int(e) for e in drift_edges)}")

        observation = real_world.observe()
        snapshot = graph.sample_features()
        recent.append((snapshot, observation))

        event = detector.update(observation)
        if event is not None:
            record = corrector.handle(
                event,
                snapshots=[s for s, _ in recent],
                real_targets=[o for _, o in recent],
            )
            signals = TaskSignals(
                prompt=(f"Explain this warehouse robot drift event to an operator: "
                        f"{event.summary()}. Recalibration reduced edge error from "
                        f"{record.pre_error:.3f}s to {record.post_error:.3f}s."),
                drift_sigma=float(np.max(np.abs(event.z_scores))),
                affected_edges=len(event.edge_ids),
                requires_reasoning=(
                    len(event.edge_ids) > 3 or float(np.max(np.abs(event.z_scores))) > 8.0
                ),
            )
            print("  LLM:", router.complete(signals)[:120])

        # Routine tick summary every 20 ticks -> cheap model
        if tick % 20 == 0:
            signals = TaskSignals(prompt=f"One-line status: tick {tick}, all edges nominal.")
            router.route(signals)

    # Demonstrate that planning reflects the corrected weights
    nf, ef = graph.sample_features()
    path, eta = optimizer.shortest_path(0, graph.num_nodes - 1, nf, ef)
    print(f"\nPlanned route 0 -> {graph.num_nodes - 1}: {path} (ETA {eta:.2f}s)")
    print(f"Corrections applied: {len(corrector.history)}")
    print("Token savings:", router.savings_report())


if __name__ == "__main__":
    main()
