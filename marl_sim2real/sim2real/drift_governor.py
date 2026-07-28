"""Hysteresis governor around the k-sigma drift detector.

Two failure modes show up when a raw drift detector meets a real fleet:

1. **Sensor dropout** — telemetry gaps arrive as NaN latencies.  Fed straight
   into the rolling windows they poison the window means (`mean([.., nan])`
   is NaN), which silently disables detection for those edges.  The governor
   filters non-finite samples per edge and reports the dropout rate instead.
2. **Correction thrash** — a marginal edge oscillating around the k-sigma
   bound fires a recalibration every tick, and each recalibration costs real
   compute (and, on a physical cell, downtime).  The governor debounces
   (``confirm_ticks`` consecutive breaches required) and enforces a cooldown
   after every confirmed event.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from marl_sim2real.sim2real.drift_detector import DriftDetector, DriftEvent


@dataclasses.dataclass
class GovernorReport:
    tick: int
    event: DriftEvent | None       # confirmed event (post-debounce), if any
    suppressed: bool               # a raw event was seen but held back
    sensor_dropout_rate: float     # fraction of this tick's readings that were non-finite


class DriftGovernor:
    def __init__(self, detector: DriftDetector, confirm_ticks: int = 3,
                 cooldown_ticks: int = 20):
        self.detector = detector
        self.confirm_ticks = confirm_ticks
        self.cooldown_ticks = cooldown_ticks
        self._breach_streak = 0
        self._cooldown_until = 0
        self._tick = 0
        self.confirmed_events: list[DriftEvent] = []

    def update(self, real_latencies: np.ndarray) -> GovernorReport:
        self._tick += 1
        readings = np.asarray(real_latencies, dtype=float)
        finite = np.isfinite(readings)
        dropout_rate = float(1.0 - finite.mean()) if readings.size else 0.0

        # Feed only finite samples: NaNs must not poison the rolling windows.
        # Per-edge feed keeps window alignment when only some sensors drop out.
        for edge_id, value in enumerate(readings):
            if finite[edge_id]:
                self.detector.windows[edge_id].append(float(value))
        self.detector.tick += 1
        raw = self._detect()

        if raw is None:
            self._breach_streak = 0
            return GovernorReport(self._tick, None, False, dropout_rate)

        if self._tick < self._cooldown_until:
            return GovernorReport(self._tick, None, True, dropout_rate)

        self._breach_streak += 1
        if self._breach_streak < self.confirm_ticks:
            return GovernorReport(self._tick, None, True, dropout_rate)

        # confirmed: emit, start cooldown, reset debounce
        self._breach_streak = 0
        self._cooldown_until = self._tick + self.cooldown_ticks
        self.confirmed_events.append(raw)
        return GovernorReport(self._tick, raw, False, dropout_rate)

    # ------------------------------------------------------------------ internals
    def _detect(self) -> DriftEvent | None:
        """Re-run the detector's decision logic on current windows (samples
        were already appended by ``update``)."""
        d = self.detector
        counts = np.array([len(w) for w in d.windows])
        # Unlike the raw detector, a dead sensor (window never filling) must
        # not disable detection fleet-wide: edges are evaluated individually
        # once THEY have enough samples.
        ready = counts >= d.config.min_samples
        if not ready.any():
            return None
        # Median, not mean: a single 50 s forklift stall in an 8-tick window
        # shifts the mean for 8 ticks (outlasting any debounce) but never
        # moves the median. Persistent drift moves both.
        window_means = np.array([np.median(w) if ok else np.nan
                                 for w, ok in zip(d.windows, ready)])
        with np.errstate(invalid="ignore"):
            z = (window_means - d.baseline.mean) / d.baseline.std
        z = np.where(ready, z, 0.0)
        drifted = np.flatnonzero(np.abs(z) > d.config.k_sigma)
        if drifted.size == 0:
            return None
        return DriftEvent(
            tick=d.tick,
            edge_ids=drifted,
            z_scores=z[drifted],
            observed_mean=window_means[drifted],
            expected_mean=d.baseline.mean[drifted],
        )
