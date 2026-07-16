"""3D bin packing environment for two cooperating agents.

The **Proposer Agent** picks WHERE and HOW to place the next box: an action is
a (x, y, orientation) triple; the box gravity-drops onto the pile. The
**Physics Agent** then judges the placement's stability and can veto it
(the box is returned to the queue and the proposer is penalized).

State exposed to agents:
  - heightmap: (W, D) top-down height field of the bin
  - next box dims + a lookahead window of upcoming boxes

The environment itself is deterministic given the box sequence; physics
uncertainty lives in the Physics Agent's simulation backend, and reality
uncertainty is injected by the sim2real domain-randomization wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from marl_packing.utils.geometry import (
    NUM_ORIENTATIONS,
    box_volume,
    drop_height,
    oriented_dims,
    support_ratio,
)


@dataclass
class PackingConfig:
    bin_size: Tuple[int, int, int] = (10, 10, 10)  # (W, D, H)
    num_boxes: int = 20
    min_box: int = 1
    max_box: int = 4
    lookahead: int = 3  # upcoming boxes visible to the proposer
    reward_volume_scale: float = 10.0  # reward per unit of packed-volume fraction
    penalty_invalid: float = -1.0  # out-of-bounds / over-height placement
    penalty_veto: float = -0.5  # physics agent rejected the placement
    penalty_unstable: float = -2.0  # placement collapsed post-hoc
    seed: Optional[int] = None


@dataclass
class StepResult:
    obs: Dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict = field(default_factory=dict)


class PackingEnv:
    """Sequential box-placement environment on an integer voxel grid."""

    def __init__(self, config: Optional[PackingConfig] = None):
        self.cfg = config or PackingConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.W, self.D, self.H = self.cfg.bin_size
        self.reset()

    # ------------------------------------------------------------------ API

    @property
    def action_space_size(self) -> int:
        return self.W * self.D * NUM_ORIENTATIONS

    def decode_action(self, action: int) -> Tuple[int, int, int]:
        """Flat action index -> (x, y, orientation)."""
        orientation = action % NUM_ORIENTATIONS
        cell = action // NUM_ORIENTATIONS
        return cell // self.D, cell % self.D, orientation

    def encode_action(self, x: int, y: int, orientation: int) -> int:
        return (x * self.D + y) * NUM_ORIENTATIONS + orientation

    def reset(self, box_sequence: Optional[List[Tuple[int, int, int]]] = None) -> Dict:
        self.heightmap = np.zeros((self.W, self.D), dtype=np.int32)
        if box_sequence is not None:
            self.boxes = [tuple(int(v) for v in b) for b in box_sequence]
        else:
            self.boxes = [
                tuple(self._rng.integers(self.cfg.min_box, self.cfg.max_box + 1, size=3))
                for _ in range(self.cfg.num_boxes)
            ]
        self.cursor = 0
        self.placements: List[Dict] = []
        self.packed_volume = 0
        self.veto_count = 0
        return self.observe()

    def observe(self) -> Dict[str, np.ndarray]:
        window = np.zeros((self.cfg.lookahead, 3), dtype=np.int32)
        for i in range(self.cfg.lookahead):
            j = self.cursor + i
            if j < len(self.boxes):
                window[i] = self.boxes[j]
        return {
            "heightmap": self.heightmap.copy(),
            "next_boxes": window,
            "remaining": np.array([len(self.boxes) - self.cursor], dtype=np.int32),
        }

    def current_box(self) -> Optional[Tuple[int, int, int]]:
        return self.boxes[self.cursor] if self.cursor < len(self.boxes) else None

    def valid_action_mask(self) -> np.ndarray:
        """Boolean mask over the flat action space: placements that fit the bin."""
        mask = np.zeros(self.action_space_size, dtype=bool)
        box = self.current_box()
        if box is None:
            return mask
        for orientation in range(NUM_ORIENTATIONS):
            dx, dy, dz = oriented_dims(box, orientation)
            if dx > self.W or dy > self.D:
                continue
            # Vectorized over all anchor cells for this orientation.
            for x in range(self.W - dx + 1):
                for y in range(self.D - dy + 1):
                    z = drop_height(self.heightmap, x, y, dx, dy)
                    if z + dz <= self.H:
                        mask[self.encode_action(x, y, orientation)] = True
        return mask

    def preview(self, action: int) -> Dict:
        """Geometry of a placement WITHOUT committing it (for the Physics Agent)."""
        box = self.current_box()
        if box is None:
            raise RuntimeError("no box left to place")
        x, y, orientation = self.decode_action(action)
        dx, dy, dz = oriented_dims(box, orientation)
        in_bounds = x + dx <= self.W and y + dy <= self.D
        z = drop_height(self.heightmap, x, y, dx, dy) if in_bounds else self.H
        return {
            "x": x, "y": y, "z": z,
            "dims": (dx, dy, dz),
            "in_bounds": in_bounds and z + dz <= self.H,
            "support": support_ratio(self.heightmap, x, y, dx, dy, z) if in_bounds else 0.0,
            "heightmap": self.heightmap,
        }

    def step(self, action: int, vetoed: bool = False, stable: bool = True) -> StepResult:
        """Commit (or veto) the proposer's placement.

        The training loop queries the Physics Agent between `preview` and
        `step`, passing its verdict via `vetoed` / `stable`.
        """
        box = self.current_box()
        if box is None:
            return StepResult(self.observe(), 0.0, True, False, {"reason": "done"})

        p = self.preview(action)
        info: Dict = {"placement": p}

        if not p["in_bounds"]:
            self.cursor += 1  # box is discarded: nowhere legal was proposed
            reward = self.cfg.penalty_invalid
            info["reason"] = "invalid"
        elif vetoed:
            self.cursor += 1  # vetoed box goes back to the depot, not the bin
            self.veto_count += 1
            reward = self.cfg.penalty_veto
            info["reason"] = "vetoed"
        else:
            dx, dy, dz = p["dims"]
            self.heightmap[p["x"] : p["x"] + dx, p["y"] : p["y"] + dy] = p["z"] + dz
            vol = box_volume(p["dims"])
            self.packed_volume += vol
            self.placements.append(p)
            self.cursor += 1
            reward = self.cfg.reward_volume_scale * vol / (self.W * self.D * self.H)
            if not stable:
                reward += self.cfg.penalty_unstable
                info["reason"] = "unstable"
            else:
                info["reason"] = "placed"

        terminated = self.cursor >= len(self.boxes)
        if terminated:
            info["fill_ratio"] = self.fill_ratio()
        return StepResult(self.observe(), reward, terminated, False, info)

    def fill_ratio(self) -> float:
        return self.packed_volume / float(self.W * self.D * self.H)
