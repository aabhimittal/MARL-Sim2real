"""Stand-in for the real deployment target.

In production this module is replaced by a driver that talks to the physical
cell (robot arm + depth camera + force sensors).  For development and CI it
wraps the simulator with a *hidden* set of physics parameters plus sensor
noise — the "reality" our sim doesn't exactly know.  Everything downstream
(log collection, calibration, gap measurement) only uses the public interface,
so swapping in real hardware changes no other code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.bin_packing_env import BinPackingEnv
from ..env.physics import PhysicsParams


# The "true" physics of the real cell. Hidden from the training code on purpose.
DEFAULT_REAL_PARAMS = PhysicsParams(
    support_threshold=0.62,
    com_margin=0.35,
    friction=0.48,
    mass_noise=0.08,
    sensor_noise=0.12,
)


@dataclass
class PlacementLog:
    """One observed real-world placement attempt (what a camera + F/T sensor gives us)."""

    heightmap: np.ndarray
    x: int
    y: int
    dims: tuple[int, int, int]
    mass: float
    stable: bool


class RealWorldCell:
    """Public interface: ``attempt_placement`` and ``collect_logs``."""

    def __init__(self, bin_size=(8, 8, 8), params: PhysicsParams | None = None, seed: int = 42):
        self.env = BinPackingEnv(bin_size=bin_size, physics=params or DEFAULT_REAL_PARAMS, seed=seed)
        self.rng = np.random.default_rng(seed)

    def collect_logs(self, n_attempts: int = 200) -> list[PlacementLog]:
        """Run random placement attempts and record ground-truth outcomes.

        This mimics an operator running scripted trials on the physical cell to
        gather calibration data.
        """
        logs: list[PlacementLog] = []
        env = self.env
        env.reset()
        while len(logs) < n_attempts:
            mask = env.valid_actions_mask()
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                env.reset()
                continue
            flat = int(self.rng.choice(valid))
            o = flat % 6
            rest = flat // 6
            y = rest % env.W
            x = rest // env.W
            report = env.check_placement(x, y, o)
            dims = env.current_item.oriented(o)
            logs.append(
                PlacementLog(
                    heightmap=env.heightmap.copy(),
                    x=x, y=y, dims=dims,
                    mass=env.current_item.mass,
                    stable=bool(report is not None and report.stable),
                )
            )
            step = env.step((x, y, o))
            if step.done:
                env.reset()
        return logs
