"""Classify a drift event as LOCAL (the world changed) or SYSTEMIC (the
robot changed).

A spill in one aisle slows two edges: recalibrating those edge weights is
correct. A firmware regression, dragging brake, or clock-rate fault slows
*every* traversal: recalibrating the map to match would corrupt a correct
world model with a robot defect — the right action is to flag the vehicle
for maintenance and leave the map alone.

Heuristic: an event is SYSTEMIC when a large fraction of all edges breach
the k-sigma bound at once, or when the fleet-wide median |z| is itself
elevated (broad shift too uniform to be geography).
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import numpy as np

from marl_sim2real.sim2real.drift_detector import DriftDetector, DriftEvent


class DriftScope(Enum):
    LOCAL = "local"          # recalibrate the drifted edges
    SYSTEMIC = "systemic"    # do NOT recalibrate; flag vehicle/fleet fault


@dataclasses.dataclass
class ScopeVerdict:
    scope: DriftScope
    drifted_fraction: float   # drifted edges / all edges
    median_abs_z: float       # fleet-wide median |z| (all edges, not just drifted)
    reason: str

    @property
    def recalibrate(self) -> bool:
        return self.scope is DriftScope.LOCAL


class DriftScopeClassifier:
    def __init__(self, systemic_fraction: float = 0.5, systemic_median_z: float = 2.0):
        self.systemic_fraction = systemic_fraction
        self.systemic_median_z = systemic_median_z

    def classify(self, event: DriftEvent, detector: DriftDetector) -> ScopeVerdict:
        num_edges = detector.baseline.mean.shape[0]
        fraction = len(event.edge_ids) / num_edges

        window_means = detector.current_window_means()
        std = np.maximum(detector.baseline.std, 1e-9)
        z_all = (window_means - detector.baseline.mean) / std
        median_abs_z = float(np.nanmedian(np.abs(z_all)))

        if fraction >= self.systemic_fraction:
            scope, reason = DriftScope.SYSTEMIC, (
                f"{fraction:.0%} of edges drifted at once — vehicle/fleet fault suspected"
            )
        elif median_abs_z >= self.systemic_median_z:
            scope, reason = DriftScope.SYSTEMIC, (
                f"fleet-wide median |z|={median_abs_z:.1f} — uniform shift, not geography"
            )
        else:
            scope, reason = DriftScope.LOCAL, (
                f"{len(event.edge_ids)} edge(s) drifted locally (median |z|={median_abs_z:.1f})"
            )
        return ScopeVerdict(scope, fraction, median_abs_z, reason)
