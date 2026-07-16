"""3D bin-packing environment with a heightmap state representation.

The bin is a W x D grid; each cell stores the current stack height. An item is
an axis-aligned box (w, d, h in cells). An action is (orientation, x, y):
orientation selects one of the 6 axis permutations of the box, and (x, y) is
the grid position of the box's minimum corner. The item rests on top of the
maximum height under its footprint (gravity placement).
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from marl_sim2real.config import PackingConfig

# All 6 unique axis permutations of a box (w, d, h)
ORIENTATIONS = list(itertools.permutations(range(3)))


@dataclasses.dataclass(frozen=True)
class Item:
    dims: tuple  # (w, d, h) in grid cells

    def oriented(self, orientation: int) -> tuple:
        perm = ORIENTATIONS[orientation]
        return tuple(self.dims[axis] for axis in perm)


@dataclasses.dataclass(frozen=True)
class Placement:
    item: Item
    orientation: int
    x: int
    y: int
    z: int  # resting height in cells, computed by the env

    def oriented_dims(self) -> tuple:
        return self.item.oriented(self.orientation)


class PackingEnv:
    """Gym-style environment shared by the proposer and physics agents."""

    def __init__(self, config: PackingConfig | None = None, seed: int | None = None):
        self.config = config or PackingConfig()
        self.rng = np.random.default_rng(seed)
        self.W, self.D, self.H = self.config.bin_size
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> np.ndarray:
        self.heightmap = np.zeros((self.W, self.D), dtype=np.int32)
        self.placements: list[Placement] = []
        self.items = [self._sample_item() for _ in range(self.config.max_items)]
        self.item_idx = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        """State = normalized heightmap flattened + current item dims."""
        hm = self.heightmap.astype(np.float32).ravel() / self.H
        item = self.current_item()
        dims = np.asarray(item.dims if item else (0, 0, 0), dtype=np.float32)
        dims = dims / max(self.W, self.D, self.H)
        return np.concatenate([hm, dims])

    def current_item(self) -> Item | None:
        if self.item_idx >= len(self.items):
            return None
        return self.items[self.item_idx]

    def action_space_size(self) -> int:
        return len(ORIENTATIONS) * self.W * self.D

    def decode_action(self, action: int) -> tuple:
        orientation, rest = divmod(action, self.W * self.D)
        x, y = divmod(rest, self.D)
        return orientation, x, y

    def try_place(self, action: int) -> Placement | None:
        """Compute the resting placement for an action, or None if infeasible."""
        item = self.current_item()
        if item is None:
            return None
        orientation, x, y = self.decode_action(action)
        w, d, h = item.oriented(orientation)
        if x + w > self.W or y + d > self.D:
            return None
        z = int(self.heightmap[x : x + w, y : y + d].max())
        if z + h > self.H:
            return None
        return Placement(item=item, orientation=orientation, x=x, y=y, z=z)

    def commit(self, placement: Placement) -> None:
        w, d, h = placement.oriented_dims()
        x, y = placement.x, placement.y
        self.heightmap[x : x + w, y : y + d] = placement.z + h
        self.placements.append(placement)
        self.item_idx += 1

    def skip_item(self) -> None:
        """Advance past the current item without placing it (rejected/unstable)."""
        self.item_idx += 1

    def done(self) -> bool:
        return self.item_idx >= len(self.items)

    # -------------------------------------------------------------- metrics
    def support_ratio(self, placement: Placement) -> float:
        """Fraction of the footprint resting directly on the surface below."""
        w, d, _ = placement.oriented_dims()
        region = self.heightmap[placement.x : placement.x + w, placement.y : placement.y + d]
        return float(np.mean(region == placement.z))

    def packing_density(self) -> float:
        placed = sum(np.prod(p.oriented_dims()) for p in self.placements)
        return float(placed) / float(self.W * self.D * self.H)

    def placement_to_world(self, placement: Placement) -> tuple:
        """Convert a grid placement to world-frame (position, half_extents) in metres."""
        cell = self.config.cell_size
        w, d, h = placement.oriented_dims()
        half = (w * cell / 2, d * cell / 2, h * cell / 2)
        pos = (
            (placement.x + w / 2) * cell,
            (placement.y + d / 2) * cell,
            (placement.z + h / 2) * cell,
        )
        return pos, half

    # -------------------------------------------------------------- private
    def _sample_item(self) -> Item:
        lo, hi = self.config.min_item_dim, self.config.max_item_dim
        return Item(dims=tuple(int(v) for v in self.rng.integers(lo, hi + 1, size=3)))
