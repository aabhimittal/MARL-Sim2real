"""Per-edge sensor health monitoring for the real-world telemetry stream.

Two fleet failure modes the NaN-dropout path (DriftGovernor) cannot catch:

- **Stuck sensor** — a dead encoder or frozen telemetry relay repeats the
  exact same latency every tick. The value is finite and plausible, so it
  sails through dropout filtering and *suppresses* real drift on that edge.
- **Invalid reading** — clock skew or integer underflow produces zero or
  negative latencies, and unit bugs produce absurdly large ones. Feeding
  them to the detector fires false k-sigma alarms.

``SensorHealthMonitor.sanitize()`` screens each tick: invalid values and
stuck streaks are replaced with NaN (so downstream dropout handling applies)
and the edge's health state is tracked (OK / STUCK / INVALID) with automatic
recovery once sane, varying readings resume.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import numpy as np


class EdgeHealth(Enum):
    OK = "ok"
    STUCK = "stuck"
    INVALID = "invalid"


@dataclasses.dataclass
class HealthReport:
    clean: np.ndarray            # observations with faulty edges set to NaN
    states: list                  # EdgeHealth per edge
    faulty_edges: np.ndarray      # indices currently not OK

    @property
    def all_ok(self) -> bool:
        return self.faulty_edges.size == 0


class SensorHealthMonitor:
    def __init__(
        self,
        num_edges: int,
        stuck_ticks: int = 5,
        min_latency: float = 1e-3,
        max_latency: float = 3600.0,
        stuck_tolerance: float = 0.0,
    ):
        """stuck_ticks: identical readings in a row before an edge is STUCK.
        stuck_tolerance: |delta| below this counts as "identical" (encoder
        quantisation can make a frozen value jitter by one LSB)."""
        self.stuck_ticks = stuck_ticks
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.stuck_tolerance = stuck_tolerance
        self._last = np.full(num_edges, np.nan)
        self._repeat_count = np.zeros(num_edges, dtype=np.int64)
        self.states = [EdgeHealth.OK] * num_edges

    def sanitize(self, observation: np.ndarray) -> HealthReport:
        obs = np.asarray(observation, dtype=np.float64)
        clean = obs.copy()

        finite = np.isfinite(obs)
        invalid = finite & ((obs < self.min_latency) | (obs > self.max_latency))

        # Stuck streak accounting (only meaningful for finite, valid readings)
        same = finite & ~invalid & (np.abs(obs - self._last) <= self.stuck_tolerance)
        self._repeat_count = np.where(same, self._repeat_count + 1, 0)
        self._last = np.where(finite, obs, self._last)
        stuck = self._repeat_count >= self.stuck_ticks - 1

        for e in range(obs.shape[0]):
            if invalid[e]:
                self.states[e] = EdgeHealth.INVALID
            elif stuck[e]:
                self.states[e] = EdgeHealth.STUCK
            elif finite[e]:
                self.states[e] = EdgeHealth.OK  # sane varying reading -> recover

        mask = invalid | stuck
        clean[mask] = np.nan
        return HealthReport(
            clean=clean,
            states=list(self.states),
            faulty_edges=np.flatnonzero(mask),
        )
