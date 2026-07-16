#!/usr/bin/env python3
"""End-to-end demo: MARL packing -> sim data -> GNN -> drift -> self-correction.

Runs a compact version of every stage in one process. For real training use
the stage scripts (train_marl.py, generate_sim_data.py, monitor_drift.py).

Usage:
    python scripts/run_pipeline.py
"""

from collections import deque

import numpy as np

from marl_sim2real.agents import MARLCoordinator, PhysicsAgent, ProposerAgent
from marl_sim2real.config import Config
from marl_sim2real.envs import PackingEnv
from marl_sim2real.gnn import DynamicGNN, PathOptimizer, WarehouseGraph
from marl_sim2real.llm import ComplexityRouter, TaskSignals
from marl_sim2real.sim2real import DriftDetector, IdealDataGenerator, SelfCorrection
from marl_sim2real.sim2real.ideal_data_generator import RealWorldSimulator


def main() -> None:
    cfg = Config()
    router = ComplexityRouter(cfg.router)

    print("=" * 70)
    print("STAGE 1 - MARL packing (proposer + PyBullet physics agent)")
    print("=" * 70)
    cfg.packing.bin_size = (6, 6, 8)
    cfg.packing.max_items = 6
    cfg.physics.sim_steps = 60
    env = PackingEnv(cfg.packing, seed=cfg.train.seed)
    proposer = ProposerAgent(env, cfg.train)
    physics = PhysicsAgent(cfg.physics)
    coordinator = MARLCoordinator(env, proposer, physics)
    coordinator.train(episodes=15, log_every=5)
    physics.close()

    print()
    print("=" * 70)
    print("STAGE 2 - PyBullet ideal traffic data + GNN pre-training")
    print("=" * 70)
    graph = WarehouseGraph(num_nodes=12, seed=0)
    generator = IdealDataGenerator(graph)
    baseline, snapshots, samples = generator.generate(runs_per_edge=10)
    generator.close()
    gnn = DynamicGNN(graph, cfg.gnn)
    losses = gnn.fit(snapshots, samples, epochs=40)
    print(f"graph: {graph.num_nodes} nodes / {graph.num_edges} edges | "
          f"GNN loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    print()
    print("=" * 70)
    print("STAGE 3 - drift detection + self-correction + routed LLM narration")
    print("=" * 70)
    detector = DriftDetector(baseline, cfg.drift)
    corrector = SelfCorrection(gnn, detector, cfg.drift)
    optimizer = PathOptimizer(graph, gnn)
    real_world = RealWorldSimulator(baseline, seed=1)
    drift_edges = np.random.default_rng(2).choice(graph.num_edges, 3, replace=False)
    recent = deque(maxlen=cfg.drift.window_size)

    for tick in range(1, 61):
        if tick == 20:
            real_world.inject_drift(drift_edges, factor=1.7)
            print(f"tick {tick}: [injected] drift x1.7 on edges "
                  f"{sorted(int(e) for e in drift_edges)}")
        observation = real_world.observe()
        recent.append((graph.sample_features(), observation))
        event = detector.update(observation)
        if event is not None:
            corrector.handle(event, [s for s, _ in recent], [o for _, o in recent])
            signals = TaskSignals(
                prompt=f"Explain to an operator: {event.summary()}",
                drift_sigma=float(np.max(np.abs(event.z_scores))),
                affected_edges=len(event.edge_ids),
                requires_reasoning=(
                    len(event.edge_ids) > 3 or float(np.max(np.abs(event.z_scores))) > 8.0
                ),
            )
            print("  LLM:", router.complete(signals)[:110])

    nf, ef = graph.sample_features()
    path, eta = optimizer.shortest_path(0, graph.num_nodes - 1, nf, ef)
    print(f"\nroute 0 -> {graph.num_nodes - 1} on corrected GNN: {path} (ETA {eta:.2f}s)")
    print(f"corrections: {len(corrector.history)} | token savings: {router.savings_report()}")


if __name__ == "__main__":
    main()
