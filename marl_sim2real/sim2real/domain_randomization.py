"""Domain randomization — the first half of the Sim2Real bridge.

Instead of training against one set of physics parameters (which the policy
would overfit), each episode samples parameters from configurable ranges.
A policy that packs well across the whole distribution transfers to any real
world whose physics lie inside (or near) that distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env.physics import PhysicsParams


@dataclass
class RandomizationRanges:
    support_threshold: tuple[float, float] = (0.45, 0.7)
    com_margin: tuple[float, float] = (0.2, 0.8)
    friction: tuple[float, float] = (0.35, 0.85)
    mass_noise: tuple[float, float] = (0.0, 0.15)
    sensor_noise: tuple[float, float] = (0.0, 0.2)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "support_threshold": self.support_threshold,
            "com_margin": self.com_margin,
            "friction": self.friction,
            "mass_noise": self.mass_noise,
            "sensor_noise": self.sensor_noise,
        }


@dataclass
class DomainRandomizer:
    ranges: RandomizationRanges = field(default_factory=RandomizationRanges)
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def sample(self) -> PhysicsParams:
        r = self.ranges
        return PhysicsParams(
            support_threshold=float(self.rng.uniform(*r.support_threshold)),
            com_margin=float(self.rng.uniform(*r.com_margin)),
            friction=float(self.rng.uniform(*r.friction)),
            mass_noise=float(self.rng.uniform(*r.mass_noise)),
            sensor_noise=float(self.rng.uniform(*r.sensor_noise)),
        )

    def apply(self, env) -> PhysicsParams:
        """Sample new physics and install them into an env's stability engine."""
        params = self.sample()
        env.engine.params = params
        return params
