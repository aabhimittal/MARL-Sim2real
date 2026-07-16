"""3D bin-packing environment driven by a heightmap.

The bin is an ``L x W`` grid with maximum height ``H``.  Each step the agent
must place the current item (an axis-aligned box) by choosing a grid position
and one of six axis-aligned orientations.  Placement is validated by the
:class:`~marl_sim2real.env.physics.StabilityEngine`, which is the "physics
ground truth" the Physics Agent learns to imitate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import PhysicsParams, StabilityEngine, StabilityReport

# All six axis-aligned orientations of a box (permutations of its dims).
ORIENTATIONS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


@dataclass
class Item:
    dims: tuple[int, int, int]
    mass: float = 1.0

    def oriented(self, orientation: int) -> tuple[int, int, int]:
        p = ORIENTATIONS[orientation]
        return (self.dims[p[0]], self.dims[p[1]], self.dims[p[2]])


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class BinPackingEnv:
    """Gym-like interface: ``reset() -> obs``, ``step(action) -> StepResult``.

    Action: ``(x, y, orientation)`` with ``orientation in [0, 6)``.
    Observation: flattened noisy heightmap (normalized) + current item dims
    (normalized) — shape ``(L*W + 3,)``.
    """

    def __init__(
        self,
        bin_size: tuple[int, int, int] = (8, 8, 8),
        max_items: int = 10,
        physics: PhysicsParams | None = None,
        seed: int = 0,
        item_dim_range: tuple[int, int] = (1, 4),
    ):
        self.L, self.W, self.H = bin_size
        self.max_items = max_items
        self.item_dim_range = item_dim_range
        self.rng = np.random.default_rng(seed)
        self.engine = StabilityEngine(physics or PhysicsParams(), rng=self.rng)
        self.heightmap = np.zeros((self.L, self.W), dtype=np.int64)
        self.items_placed = 0
        self.current_item: Item | None = None
        self.placed_volume = 0

    # ------------------------------------------------------------------ api
    @property
    def observation_size(self) -> int:
        return self.L * self.W + 3

    @property
    def num_positions(self) -> int:
        return self.L * self.W

    def reset(self) -> np.ndarray:
        self.heightmap[:] = 0
        self.items_placed = 0
        self.placed_volume = 0
        self.current_item = self._sample_item()
        return self._observation()

    def step(self, action: tuple[int, int, int]) -> StepResult:
        assert self.current_item is not None, "call reset() first"
        x, y, orientation = action
        dims = self.current_item.oriented(orientation)
        report = self.check_placement(x, y, orientation)

        info: dict = {"stability": report, "action": action, "dims": dims}
        if report is None:  # out of bounds / over height
            reward = -1.0
            info["placed"] = False
        elif not report.stable:
            reward = -0.5 - 0.5 * report.topple_score
            info["placed"] = False
        else:
            l, w, h = dims
            z = report.details["rest_z"]
            self.heightmap[x:x + l, y:y + w] = z + h
            vol = l * w * h
            self.placed_volume += vol
            # dense reward: normalized volume + compactness (low rest height)
            reward = vol / (self.L * self.W * self.H) * 10.0 + (1.0 - z / self.H) * 0.2
            info["placed"] = True

        self.items_placed += 1
        done = self.items_placed >= self.max_items
        if not done:
            self.current_item = self._sample_item()
        info["utilization"] = self.utilization()
        return StepResult(self._observation(), float(reward), done, info)

    def check_placement(self, x: int, y: int, orientation: int) -> StabilityReport | None:
        """Physics ground truth for a candidate action (None = out of bounds)."""
        assert self.current_item is not None
        l, w, h = self.current_item.oriented(orientation)
        if x < 0 or y < 0 or x + l > self.L or y + w > self.W:
            return None
        z = self.engine.rest_height(self.heightmap, x, y, l, w)
        if z + h > self.H:
            return None
        return self.engine.evaluate(self.heightmap, x, y, (l, w, h), self.current_item.mass)

    def utilization(self) -> float:
        return self.placed_volume / float(self.L * self.W * self.H)

    def valid_actions_mask(self) -> np.ndarray:
        """Boolean mask over the (L*W*6) flat action space (bounds-only check)."""
        assert self.current_item is not None
        mask = np.zeros((self.L, self.W, 6), dtype=bool)
        for o in range(6):
            l, w, h = self.current_item.oriented(o)
            if l <= self.L and w <= self.W and h <= self.H:
                mask[: self.L - l + 1, : self.W - w + 1, o] = True
        return mask.reshape(-1)

    # ------------------------------------------------------------- internals
    def _sample_item(self) -> Item:
        lo, hi = self.item_dim_range
        dims = tuple(int(d) for d in self.rng.integers(lo, hi + 1, size=3))
        mass = float(self.rng.uniform(0.5, 2.0))
        return Item(dims=dims, mass=mass)

    def _observation(self) -> np.ndarray:
        hm = self.engine.observe(self.heightmap).reshape(-1) / float(self.H)
        assert self.current_item is not None
        item = np.array(self.current_item.dims, dtype=np.float64) / float(max(self.L, self.W, self.H))
        return np.concatenate([hm, item]).astype(np.float64)
