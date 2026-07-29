"""Multi-bin (pallet rollover) packing.

A real packing cell never stops at one pallet: when the current pallet can't
take the next carton, it is closed, wrapped, and a fresh one is opened. The
cost driver is the *pallet count* — every extra bin is freight and floor
space — so the KPIs here are bins used, per-bin density, and diverted items
(cartons that fit no bin at all and leave on the manual-handling line).

``MultiBinPackingEnv`` extends ``DeliveryPackingEnv`` (and therefore also
carries load-bearing and unload-order constraints). The MARL coordinator
drives rollover through one duck-typed hook: when no feasible placement
exists, it calls ``roll_bin_if_useful()`` and retries the same item in the
fresh bin; only if that returns False is the item truly diverted.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from marl_sim2real.config import PackingConfig
from marl_sim2real.envs.delivery import DeliveryPackingEnv


@dataclasses.dataclass(frozen=True)
class BinRecord:
    density: float
    items_placed: int


class MultiBinPackingEnv(DeliveryPackingEnv):
    def __init__(self, config: PackingConfig | None = None, seed: int | None = None,
                 max_bins: int = 6, **delivery_kwargs):
        self.max_bins = max_bins
        self.closed_bins: list[BinRecord] = []
        self.diverted = 0
        self._total_placed_volume = 0.0
        super().__init__(config=config, seed=seed, **delivery_kwargs)

    # ------------------------------------------------------------ lifecycle
    def reset(self):
        obs = super().reset()
        self.closed_bins = []
        self.diverted = 0
        self._total_placed_volume = 0.0
        return obs

    def commit(self, placement) -> None:
        super().commit(placement)
        self._total_placed_volume += float(np.prod(placement.oriented_dims()))

    def roll_bin_if_useful(self) -> bool:
        """Close the current bin and open a fresh one, if that can help.

        Useless (returns False) when the current bin is still empty — a fresh
        bin would be identical — or when the pallet budget is exhausted.
        """
        if not self.placements or self.bins_used >= self.max_bins:
            return False
        self.closed_bins.append(
            BinRecord(density=self.packing_density(), items_placed=len(self.placements))
        )
        self.heightmap = np.zeros((self.W, self.D), dtype=np.int32)
        self.placements = []
        self.placement_stops = []
        self.tracker.reset()
        return True

    def divert_item(self) -> None:
        """Send the current item to the manual-handling line.

        Called by the coordinator only when the item fits nowhere even after a
        rollover attempt — distinct from ``skip_item``, which also fires for
        stability/crush rejections and is not a diversion.
        """
        self.diverted += 1
        self.skip_item()

    # -------------------------------------------------------------- metrics
    @property
    def bins_used(self) -> int:
        return len(self.closed_bins) + 1  # closed bins + the open one

    def bin_summary(self) -> list[BinRecord]:
        """All bins including the currently open one."""
        return self.closed_bins + [
            BinRecord(density=self.packing_density(), items_placed=len(self.placements))
        ]

    def overall_density(self) -> float:
        """Placed volume over the volume of every bin opened — the number a
        line manager actually optimizes (pallets are paid for whole).
        """
        bin_volume = float(self.W * self.D * self.H)
        return self._total_placed_volume / (bin_volume * self.bins_used)
