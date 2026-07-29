"""Industrial cargo constraints: load-bearing limits and fragility.

Real packing cells crush cargo long before they topple it: a stable stack can
still put 40 kg of bearings on a carton rated for 5.  This module tracks how
much mass rests on every committed placement and vetoes candidates that would
exceed any supporter's load rating.

Mass is distributed to direct supporters proportionally to the overlapping
footprint area — the standard first-order approximation used by packaging
engineers (full FEM is neither needed nor tractable per proposal).
"""

from __future__ import annotations

import dataclasses
import math

from marl_sim2real.config import PackingConfig
from marl_sim2real.envs.packing_env import PackingEnv, Placement


@dataclasses.dataclass(frozen=True)
class CargoSpec:
    """Physical properties of one item beyond its geometry."""

    mass: float = 1.0
    max_load: float = math.inf   # mass this item tolerates on top of it

    @property
    def fragile(self) -> bool:
        return math.isfinite(self.max_load)


@dataclasses.dataclass
class CrushReport:
    ok: bool
    violations: list  # (supporter_index, current_load, added_load, max_load)

    def worst(self) -> tuple | None:
        if not self.violations:
            return None
        return max(self.violations, key=lambda v: (v[1] + v[2]) - v[3])


def _footprint_overlap(a: Placement, b: Placement) -> int:
    """Overlapping XY area (in cells) between two placements' footprints."""
    aw, ad, _ = a.oriented_dims()
    bw, bd, _ = b.oriented_dims()
    ox = min(a.x + aw, b.x + bw) - max(a.x, b.x)
    oy = min(a.y + ad, b.y + bd) - max(a.y, b.y)
    return max(0, ox) * max(0, oy)


class LoadTracker:
    """Tracks per-placement supported load for a sequence of committed items."""

    def __init__(self) -> None:
        self.specs: list[CargoSpec] = []
        self.load_on: list[float] = []   # mass currently resting on placement i
        self._placements: list[Placement] = []

    # ------------------------------------------------------------------ query
    def supporters(self, placements: list[Placement], candidate: Placement) -> list[tuple[int, float]]:
        """(index, share) of committed placements directly under the candidate.

        A supporter's top face must be exactly at the candidate's rest height
        and overlap its footprint; shares are proportional to overlap area.
        """
        found: list[tuple[int, int]] = []
        for i, p in enumerate(placements):
            _, _, h = p.oriented_dims()
            if p.z + h != candidate.z:
                continue
            overlap = _footprint_overlap(p, candidate)
            if overlap > 0:
                found.append((i, overlap))
        total = sum(o for _, o in found)
        if total == 0:
            return []
        return [(i, o / total) for i, o in found]

    def check(self, placements: list[Placement], candidate: Placement, spec: CargoSpec) -> CrushReport:
        """Would committing ``candidate`` crush any supporter?"""
        violations = []
        for idx, share in self.supporters(placements, candidate):
            added = spec.mass * share
            if self.load_on[idx] + added > self.specs[idx].max_load:
                violations.append((idx, self.load_on[idx], added, self.specs[idx].max_load))
        return CrushReport(ok=not violations, violations=violations)

    # ----------------------------------------------------------------- commit
    def commit(self, placements: list[Placement], candidate: Placement, spec: CargoSpec) -> None:
        """Record the placement and propagate its mass onto supporters.

        Note: ``placements`` is the committed list *before* the candidate is
        appended (matching ``PackingEnv.commit`` call order).
        """
        for idx, share in self.supporters(placements, candidate):
            self.load_on[idx] += spec.mass * share
        self.specs.append(spec)
        self.load_on.append(0.0)
        self._placements.append(candidate)

    def reset(self) -> None:
        self.specs.clear()
        self.load_on.clear()
        self._placements.clear()


class ConstrainedPackingEnv(PackingEnv):
    """PackingEnv that assigns each item a CargoSpec and vetoes crushing
    placements.  ``check_crush(placement)`` is the hook the MARL coordinator
    consults before committing; heavier items also make fragility common
    enough for the proposer to encounter it during training.
    """

    def __init__(self, config: PackingConfig | None = None, seed: int | None = None,
                 fragile_fraction: float = 0.3,
                 mass_range: tuple[float, float] = (0.5, 3.0),
                 fragile_max_load: float = 2.0):
        self.fragile_fraction = fragile_fraction
        self.mass_range = mass_range
        self.fragile_max_load = fragile_max_load
        self.tracker = LoadTracker()
        self.item_specs: list[CargoSpec] = []
        super().__init__(config=config, seed=seed)

    def reset(self):
        obs = super().reset()
        self.tracker.reset()
        self.item_specs = [self._sample_spec() for _ in self.items]
        return obs

    def current_spec(self) -> CargoSpec | None:
        if self.item_idx >= len(self.item_specs):
            return None
        return self.item_specs[self.item_idx]

    def check_crush(self, placement: Placement) -> CrushReport:
        spec = self.current_spec() or CargoSpec()
        return self.tracker.check(self.placements, placement, spec)

    def commit(self, placement: Placement) -> None:
        spec = self.current_spec() or CargoSpec()
        self.tracker.commit(self.placements, placement, spec)
        super().commit(placement)

    def _sample_spec(self) -> CargoSpec:
        mass = float(self.rng.uniform(*self.mass_range))
        if self.rng.random() < self.fragile_fraction:
            return CargoSpec(mass=mass, max_load=self.fragile_max_load)
        return CargoSpec(mass=mass)
