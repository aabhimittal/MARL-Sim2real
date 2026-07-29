"""Multi-stop delivery (LIFO unload-order) constraints.

A mixed-load truck is unloaded stop by stop: everything bound for stop 1
leaves first. Any item stacked *above* a stop-1 item therefore has to be
dug out and restacked at the dock unless it also leaves at stop 1 — in
industrial terms, a later-stop item must never sit vertically over an
earlier-stop item ("burial"). The check is purely geometric: a candidate
blocks every committed placement whose footprint it overlaps at a lower
height, regardless of whether it rests on it directly.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from marl_sim2real.config import PackingConfig
from marl_sim2real.envs.constraints import ConstrainedPackingEnv, _footprint_overlap
from marl_sim2real.envs.packing_env import Placement


@dataclasses.dataclass
class OrderReport:
    ok: bool
    violations: list  # (buried_index, buried_stop, candidate_stop)

    def worst(self) -> tuple | None:
        if not self.violations:
            return None
        # Largest stop gap = most restacking work at the dock.
        return max(self.violations, key=lambda v: v[2] - v[1])


def check_unload_order(
    placements: list[Placement],
    stops: list[int],
    candidate: Placement,
    candidate_stop: int,
) -> OrderReport:
    """Would ``candidate`` bury any item that unloads at an earlier stop?"""
    violations = []
    for i, p in enumerate(placements):
        _, _, h = p.oriented_dims()
        if (
            p.z + h <= candidate.z
            and _footprint_overlap(p, candidate) > 0
            and stops[i] < candidate_stop
        ):
            violations.append((i, stops[i], candidate_stop))
    return OrderReport(ok=not violations, violations=violations)


class DeliveryPackingEnv(ConstrainedPackingEnv):
    """ConstrainedPackingEnv whose items carry a delivery stop.

    Adds ``check_unload_order(placement)`` — the hook the MARL coordinator
    consults before committing, mirroring ``check_crush``. The current item's
    normalized stop is appended to the observation so the proposer can learn
    to keep late-stop freight low and early-stop freight on top.
    """

    def __init__(self, config: PackingConfig | None = None, seed: int | None = None,
                 num_stops: int = 3, **constraint_kwargs):
        self.num_stops = num_stops
        self.item_stops: list[int] = []
        self.placement_stops: list[int] = []
        super().__init__(config=config, seed=seed, **constraint_kwargs)

    def reset(self):
        super().reset()
        self.item_stops = [
            int(self.rng.integers(1, self.num_stops + 1)) for _ in self.items
        ]
        self.placement_stops = []
        return self.observe()

    def observe(self):
        base = super().observe()
        stop = self.current_stop()
        frac = np.asarray([0.0 if stop is None else stop / self.num_stops], dtype=np.float32)
        return np.concatenate([base, frac])

    def current_stop(self) -> int | None:
        if self.item_idx >= len(self.item_stops):
            return None
        return self.item_stops[self.item_idx]

    def check_unload_order(self, placement: Placement) -> OrderReport:
        stop = self.current_stop()
        if stop is None:
            return OrderReport(ok=True, violations=[])
        return check_unload_order(self.placements, self.placement_stops, placement, stop)

    def commit(self, placement: Placement) -> None:
        stop = self.current_stop()
        super().commit(placement)
        self.placement_stops.append(stop if stop is not None else self.num_stops)
