"""Generate "ideal" robot traffic data with PyBullet.

For every edge of the warehouse graph we simulate a differential-drive-like
robot (a box with velocity control) traversing the edge in PyBullet and record
the traversal latency. Repeating this under sampled congestion snapshots gives
a per-edge distribution: the sim baseline (mean, std) that real-world latency
is compared against for drift detection.

A fast kinematic fallback is used when PyBullet is unavailable so tests and
CI stay green.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from marl_sim2real.gnn.dynamic_gnn import WarehouseGraph

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYBULLET_AVAILABLE = False

ROBOT_SPEED = 0.5          # m/s nominal cruise speed
WORLD_SCALE = 10.0         # graph unit square -> 10m x 10m warehouse


@dataclasses.dataclass
class SimBaseline:
    """Per-edge latency statistics from the ideal simulation."""
    mean: np.ndarray   # (E,)
    std: np.ndarray    # (E,)
    samples: int

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "samples": self.samples,
        }))

    @classmethod
    def load(cls, path: str | Path) -> SimBaseline:
        data = json.loads(Path(path).read_text())
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
            samples=int(data["samples"]),
        )


class IdealDataGenerator:
    def __init__(self, graph: WarehouseGraph, use_pybullet: bool | None = None, seed: int = 0):
        self.graph = graph
        self.rng = np.random.default_rng(seed)
        self.use_pybullet = PYBULLET_AVAILABLE if use_pybullet is None else use_pybullet
        self._client = None

    # ------------------------------------------------------------------ API
    def generate(self, runs_per_edge: int = 20, congestion_level: float = 0.3) -> tuple:
        """Returns (baseline, snapshots, latency_samples).

        snapshots: list of (node_feats, edge_feats) — GNN training inputs.
        latency_samples: list of (E,) arrays — GNN training targets.
        """
        snapshots, samples = [], []
        for _ in range(runs_per_edge):
            node_feats, edge_feats = self.graph.sample_features(congestion_level)
            latencies = self._traverse_all_edges(edge_feats)
            snapshots.append((node_feats, edge_feats))
            samples.append(latencies)
        stacked = np.stack(samples)  # (runs, E)
        baseline = SimBaseline(
            mean=stacked.mean(axis=0),
            std=np.maximum(stacked.std(axis=0), 1e-3),
            samples=runs_per_edge,
        )
        return baseline, snapshots, samples

    def close(self) -> None:
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None

    # ------------------------------------------------------------- internals
    def _traverse_all_edges(self, edge_feats: np.ndarray) -> np.ndarray:
        distances = self.graph.distances() * WORLD_SCALE
        loads = edge_feats[:, 1]
        if self.use_pybullet:
            base_latency = self._pybullet_edge_times(distances)
        else:
            base_latency = distances / ROBOT_SPEED
        # Congestion slows traversal; small process noise models sensor jitter
        noise = self.rng.normal(1.0, 0.02, size=base_latency.shape)
        return base_latency * (1.0 + 0.8 * loads) * noise

    def _pybullet_edge_times(self, distances: np.ndarray) -> np.ndarray:
        """Simulate one representative straight-line run, then scale by distance.

        Simulating a friction-affected velocity-controlled robot over a unit
        run captures dynamics (acceleration ramp, friction losses) that a pure
        kinematic model misses; per-edge times scale with distance.
        """
        if self._client is None:
            self._client = p.connect(p.DIRECT)
        client = self._client
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=client)
        floor = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        p.createMultiBody(0, floor, physicsClientId=client)

        half = (0.15, 0.1, 0.05)
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
        robot = p.createMultiBody(1.0, shape, basePosition=(0, 0, half[2]),
                                  physicsClientId=client)
        p.changeDynamics(robot, -1, lateralFriction=0.05, physicsClientId=client)

        target_x, steps, max_steps = 1.0, 0, 240 * 30
        while steps < max_steps:
            pos, _ = p.getBasePositionAndOrientation(robot, physicsClientId=client)
            if pos[0] >= target_x:
                break
            p.resetBaseVelocity(robot, linearVelocity=(ROBOT_SPEED, 0, 0),
                                physicsClientId=client)
            p.stepSimulation(physicsClientId=client)
            steps += 1
        seconds_per_metre = (steps / 240.0) / target_x
        return distances * seconds_per_metre


class RealWorldSimulator:
    """Stand-in for the physical robot fleet: replays sim latencies plus
    real-world effects (calibration offset, wear, slippage). Drift can be
    injected to exercise the detection/correction loop end to end.
    """

    def __init__(self, baseline: SimBaseline, seed: int = 1):
        # Copy: the detector/corrector mutate the shared baseline during
        # recalibration, but physical reality must not shift with it.
        self.true_mean = baseline.mean.copy()
        self.rng = np.random.default_rng(seed)
        self.drift_factor = np.ones_like(self.true_mean)

    def inject_drift(self, edge_ids: np.ndarray | list, factor: float = 1.5) -> None:
        """Degrade specific edges (e.g. a wheel losing traction on that aisle)."""
        self.drift_factor[np.asarray(edge_ids)] = factor

    def observe(self) -> np.ndarray:
        """One tick of real-world per-edge latency measurements."""
        noise = self.rng.normal(1.0, 0.03, size=self.true_mean.shape)
        return self.true_mean * self.drift_factor * noise
