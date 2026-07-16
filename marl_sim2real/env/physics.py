"""Lightweight rigid-stability engine for the 3D bin-packing environment.

This is not a full dynamics simulator: for packing, the questions that matter
are (a) is the box supported, (b) is its center of mass over the support
polygon, and (c) does it survive small perturbations.  All three are cheap to
evaluate on a heightmap, which is what lets us run thousands of MARL episodes
per minute and later randomize the physics parameters for Sim2Real.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhysicsParams:
    """Parameters the Sim2Real layer randomizes / calibrates.

    ``support_threshold``  minimum supported-area fraction to count as stable.
    ``com_margin``         how far (in cells) the COM may sit outside the
                           supported region before the box topples.
    ``friction``           scales lateral perturbation resistance.
    ``mass_noise``         std of multiplicative noise applied to item mass.
    ``sensor_noise``       std of additive noise on heightmap observations.
    """

    support_threshold: float = 0.55
    com_margin: float = 0.5
    friction: float = 0.6
    mass_noise: float = 0.0
    sensor_noise: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.support_threshold, self.com_margin, self.friction,
             self.mass_noise, self.sensor_noise],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "PhysicsParams":
        return cls(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]))


@dataclass
class StabilityReport:
    stable: bool
    support_ratio: float
    com_inside: bool
    topple_score: float  # 0 = rock solid, 1 = certain topple
    details: dict = field(default_factory=dict)


class StabilityEngine:
    """Evaluates whether a box placement is stable on the current heightmap."""

    def __init__(self, params: PhysicsParams | None = None, rng: np.random.Generator | None = None):
        self.params = params or PhysicsParams()
        self.rng = rng or np.random.default_rng(0)

    def rest_height(self, heightmap: np.ndarray, x: int, y: int, l: int, w: int) -> int:
        """Height at which the box footprint comes to rest (drop from above)."""
        footprint = heightmap[x:x + l, y:y + w]
        return int(footprint.max()) if footprint.size else 0

    def evaluate(
        self,
        heightmap: np.ndarray,
        x: int,
        y: int,
        dims: tuple[int, int, int],
        mass: float = 1.0,
    ) -> StabilityReport:
        l, w, h = dims
        p = self.params
        footprint = heightmap[x:x + l, y:y + w]
        z = int(footprint.max()) if footprint.size else 0

        # (a) support ratio: fraction of footprint cells touching at rest height
        supported = footprint == z
        support_ratio = float(supported.mean()) if supported.size else 0.0

        # (b) center of mass vs. supported region centroid
        if supported.any():
            sup_idx = np.argwhere(supported).astype(np.float64)
            sup_min = sup_idx.min(axis=0)
            sup_max = sup_idx.max(axis=0)
            com = np.array([(l - 1) / 2.0, (w - 1) / 2.0])
            margin = p.com_margin
            com_inside = bool(
                (com >= sup_min - margin).all() and (com <= sup_max + margin).all()
            )
        else:
            com_inside = False

        # (c) perturbation / topple score: taller + heavier + less friction = worse
        effective_mass = mass * float(np.exp(self.rng.normal(0.0, p.mass_noise))) if p.mass_noise > 0 else mass
        aspect = h / max(1.0, min(l, w))
        topple_score = float(
            np.clip((aspect * 0.15 + (1.0 - support_ratio) * 0.6) * (1.2 - p.friction) * np.sqrt(effective_mass), 0.0, 1.0)
        )

        stable = support_ratio >= p.support_threshold and com_inside and topple_score < 0.5
        return StabilityReport(
            stable=stable,
            support_ratio=support_ratio,
            com_inside=com_inside,
            topple_score=topple_score,
            details={"rest_z": z, "effective_mass": effective_mass},
        )

    def observe(self, heightmap: np.ndarray) -> np.ndarray:
        """Heightmap as seen through (possibly noisy) sensors — Sim2Real hook."""
        if self.params.sensor_noise > 0:
            noise = self.rng.normal(0.0, self.params.sensor_noise, size=heightmap.shape)
            return np.clip(heightmap + noise, 0, None)
        return heightmap.astype(np.float64)
