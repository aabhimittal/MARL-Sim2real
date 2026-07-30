"""k-sigma sim2real drift detection.

For each graph edge we keep a rolling window of real-world latency
observations. Drift triggers on an edge when the window mean deviates from
the simulation baseline mean by more than k standard deviations:

    |mean(real_window) - sim_mean| > k * sim_std

Windowed means (rather than single samples) make the detector robust to
one-off sensor spikes while still reacting within `window_size` ticks.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np

from marl_sim2real.config import DriftConfig
from marl_sim2real.sim2real.ideal_data_generator import SimBaseline


@dataclasses.dataclass
class DriftEvent:
    tick: int
    edge_ids: np.ndarray        # edges that breached the k-sigma bound
    z_scores: np.ndarray        # z-score per drifted edge
    observed_mean: np.ndarray   # rolling real-world mean per drifted edge
    expected_mean: np.ndarray   # sim baseline mean per drifted edge

    def summary(self) -> str:
        worst = int(np.argmax(np.abs(self.z_scores)))
        return (
            f"tick {self.tick}: {len(self.edge_ids)} edge(s) drifted; "
            f"worst edge {int(self.edge_ids[worst])} at "
            f"{self.z_scores[worst]:+.1f} sigma "
            f"(real {self.observed_mean[worst]:.2f}s vs sim {self.expected_mean[worst]:.2f}s)"
        )


class DriftDetector:
    def __init__(self, baseline: SimBaseline, config: DriftConfig | None = None):
        self.baseline = baseline
        self.config = config or DriftConfig()
        num_edges = baseline.mean.shape[0]
        self.windows = [deque(maxlen=self.config.window_size) for _ in range(num_edges)]
        self.tick = 0

    def update(self, real_latencies: np.ndarray) -> DriftEvent | None:
        """Feed one tick of real-world per-edge latencies; returns a DriftEvent
        if any edge breaches the k-sigma bound, else None."""
        self.tick += 1
        for edge_id, value in enumerate(real_latencies):
            self.windows[edge_id].append(float(value))

        counts = np.array([len(w) for w in self.windows])
        if counts.min() < self.config.min_samples:
            return None

        window_means = np.array([np.mean(w) for w in self.windows])
        # Floor sigma: a hand-edited or degenerate baseline (std=0) must not
        # produce inf/NaN z-scores.
        z = (window_means - self.baseline.mean) / np.maximum(self.baseline.std, 1e-9)
        drifted = np.flatnonzero(np.abs(z) > self.config.k_sigma)
        if drifted.size == 0:
            return None
        return DriftEvent(
            tick=self.tick,
            edge_ids=drifted,
            z_scores=z[drifted],
            observed_mean=window_means[drifted],
            expected_mean=self.baseline.mean[drifted],
        )

    def current_window_means(self) -> np.ndarray:
        return np.array([np.mean(w) if w else np.nan for w in self.windows])

    def reset_edges(self, edge_ids: np.ndarray) -> None:
        """Clear windows after recalibration so corrected edges start fresh."""
        for edge_id in edge_ids:
            self.windows[int(edge_id)].clear()
