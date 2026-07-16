"""The two-agent packing environment.

`packer` proposes where and how to place the next box (position + one of 6
axis-aligned rotations). `physics` then observes that specific proposal and
decides accept/reject. On accept, a PyBullet drop simulation determines
whether the stack actually stays stable; on reject, the same simulation is
run privately (never committed) purely to score whether the reject was
correct, so the physics agent gets a clean supervised-ish signal.

This dependency -- physics must see the packer's actual proposal before it
can act -- is exactly what PettingZoo's turn-based `AECEnv` API is for, so
that's what this class implements (as opposed to `ParallelEnv`, which
assumes agents act simultaneously without seeing each other's this-step
action).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

from marl_packing.envs.domain_randomization import get_randomization_profile
from marl_packing.envs.pybullet_sim import DropSimulator, PlacedBox

# The 6 axis-aligned orientations a box can rest in, expressed as an index
# permutation applied to its sampled (w, d, h) dims.
_ROTATIONS = [
    (0, 1, 2),
    (1, 0, 2),
    (0, 2, 1),
    (2, 0, 1),
    (1, 2, 0),
    (2, 1, 0),
]

# Scales the raw [0, ~1] utilization-fraction reward up to a more useful PPO signal magnitude.
_UTILIZATION_REWARD_SCALE = 10.0
_DISCARD_PENALTY_SCALE = 5.0
_INVALID_PLACEMENT_PENALTY_SCALE = 10.0

DEFAULT_REWARD_SHAPING = {
    "false_accept_penalty": -1.0,
    "false_reject_penalty": -0.3,
    "correct_decision_reward": 0.2,
}


class PackingEnv(AECEnv):
    metadata = {"render_modes": [], "name": "packing_v0", "is_parallelizable": False}

    def __init__(self, config: dict, randomization_mode: str = "train", reward_shaping: dict | None = None):
        super().__init__()
        self.config = config
        self.randomization_mode = randomization_mode
        self.reward_shaping = {**DEFAULT_REWARD_SHAPING, **(reward_shaping or {})}

        self.possible_agents = ["packer", "physics"]
        self.agents = self.possible_agents[:]

        self.bin_dims = np.array(config["bin_dims"], dtype=np.float32)
        self.bin_volume = float(np.prod(self.bin_dims))
        self.grid_res = float(config["grid_resolution"])
        self.grid_shape = (
            max(round(self.bin_dims[0] / self.grid_res), 1),
            max(round(self.bin_dims[1] / self.grid_res), 1),
        )
        self.num_items_per_episode = int(config["num_items_per_episode"])
        item_range = config["item_size_range"]
        self._item_min = np.array([item_range["x"][0], item_range["y"][0], item_range["z"][0]], dtype=np.float32)
        self._item_max = np.array([item_range["x"][1], item_range["y"][1], item_range["z"][1]], dtype=np.float32)
        max_dim = float(self._item_max.max())
        bin_max = float(self.bin_dims.max())

        self._observation_spaces = {
            "packer": spaces.Dict(
                {
                    "heightmap": spaces.Box(low=0.0, high=self.bin_dims[2], shape=self.grid_shape, dtype=np.float32),
                    "next_item": spaces.Box(low=0.0, high=max_dim, shape=(3,), dtype=np.float32),
                }
            ),
            "physics": spaces.Dict(
                {
                    "heightmap": spaces.Box(low=0.0, high=self.bin_dims[2], shape=self.grid_shape, dtype=np.float32),
                    "proposed_item": spaces.Box(
                        low=0.0, high=max(bin_max, max_dim), shape=(6,), dtype=np.float32
                    ),
                }
            ),
        }
        self._action_spaces = {
            "packer": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            "physics": spaces.Discrete(2),
        }

        self._sim = DropSimulator(config)
        self._rng = np.random.default_rng(config.get("seed"))
        self.render_mode = None

        # Episode-local state, populated by reset().
        self._heightmap = None
        self._placed_boxes: list[PlacedBox] = []
        self._item_queue: list[np.ndarray] = []
        self._item_idx = 0
        self._pending_placement: _PendingPlacement | None = None
        self._utilized_volume = 0.0
        self._items_placed = 0

    # -- PettingZoo required interface -------------------------------------------------

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def observe(self, agent: str):
        heightmap = self._heightmap.copy()
        if agent == "packer":
            next_item = self._item_queue[self._item_idx] if self._item_idx < len(self._item_queue) else np.zeros(3, dtype=np.float32)
            return {"heightmap": heightmap, "next_item": next_item.astype(np.float32)}
        else:
            if self._pending_placement is None:
                proposed = np.zeros(6, dtype=np.float32)
            else:
                pp = self._pending_placement
                proposed = np.array([*pp.center, *pp.dims], dtype=np.float32)
            return {"heightmap": heightmap, "proposed_item": proposed}

    def close(self) -> None:
        self._sim.close()

    def render(self):
        return None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._heightmap = np.zeros(self.grid_shape, dtype=np.float32)
        self._placed_boxes = []
        self._item_idx = 0
        self._pending_placement = None
        self._utilized_volume = 0.0
        self._items_placed = 0
        self._item_queue = [
            self._rng.uniform(self._item_min, self._item_max).astype(np.float32)
            for _ in range(self.num_items_per_episode)
        ]

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.rewards = {a: 0.0 for a in self.agents}

        if agent == "packer":
            self._apply_packer_action(np.asarray(action, dtype=np.float32))
        elif agent == "physics":
            self._apply_physics_action(int(action))
        else:  # pragma: no cover - defensive, possible_agents is fixed
            raise ValueError(f"Unknown agent {agent!r}")

        if self._agent_selector.is_last():
            self._advance_item()
        self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()

    # -- Internals -----------------------------------------------------------------------

    def _apply_packer_action(self, action: np.ndarray) -> None:
        x_frac, y_frac, rot_frac = np.clip(action, 0.0, 1.0)
        item_dims = self._item_queue[self._item_idx]
        rot_id = min(int(rot_frac * len(_ROTATIONS)), len(_ROTATIONS) - 1)
        perm = _ROTATIONS[rot_id]
        w, d, h = item_dims[perm[0]], item_dims[perm[1]], item_dims[perm[2]]

        max_x = max(float(self.bin_dims[0]) - w, 0.0)
        max_y = max(float(self.bin_dims[1]) - d, 0.0)
        x0 = float(x_frac) * max_x
        y0 = float(y_frac) * max_y

        ix0, ix1, iy0, iy1 = self._footprint_cells(x0, y0, w, d)
        base_z = float(self._heightmap[ix0:ix1, iy0:iy1].max()) if ix1 > ix0 and iy1 > iy0 else 0.0
        top_z = base_z + h
        fits_vertically = top_z <= float(self.bin_dims[2]) + 1e-6

        self._pending_placement = _PendingPlacement(
            center=np.array([x0 + w / 2, y0 + d / 2, base_z + h / 2], dtype=np.float32),
            dims=np.array([w, d, h], dtype=np.float32),
            footprint_cells=(ix0, ix1, iy0, iy1),
            base_z=base_z,
            valid=fits_vertically,
        )

    def _apply_physics_action(self, accept: int) -> None:
        pp = self._pending_placement
        item_frac_volume = float(np.prod(pp.dims)) / self.bin_volume

        if not pp.valid:
            packer_r = -_INVALID_PLACEMENT_PENALTY_SCALE * item_frac_volume
            physics_r = 0.0
            outcome = "invalid_height"
        elif accept:
            candidate = PlacedBox(position=pp.center, dims=pp.dims)
            result = self._sim.simulate_drop(
                self._placed_boxes, candidate, self._rng, get_randomization_profile(self.config, self.randomization_mode)
            )
            if result.stable:
                self._commit_placement(pp)
                packer_r = _UTILIZATION_REWARD_SCALE * item_frac_volume
                physics_r = self.reward_shaping["correct_decision_reward"]
                outcome = "accepted_stable"
            else:
                packer_r = -_DISCARD_PENALTY_SCALE * item_frac_volume
                physics_r = self.reward_shaping["false_accept_penalty"]
                outcome = "accepted_unstable"
        else:
            candidate = PlacedBox(position=pp.center, dims=pp.dims)
            ghost = self._sim.simulate_drop(
                self._placed_boxes, candidate, self._rng, get_randomization_profile(self.config, self.randomization_mode)
            )
            packer_r = -_DISCARD_PENALTY_SCALE * item_frac_volume
            if ghost.stable:
                physics_r = self.reward_shaping["false_reject_penalty"]
                outcome = "rejected_would_have_been_stable"
            else:
                physics_r = self.reward_shaping["correct_decision_reward"]
                outcome = "rejected_correctly"

        self.rewards["packer"] = packer_r
        self.rewards["physics"] = physics_r
        self.infos["packer"] = {"outcome": outcome, "item_frac_volume": item_frac_volume}
        self.infos["physics"] = {"outcome": outcome, "item_frac_volume": item_frac_volume}

    def _commit_placement(self, pp: "_PendingPlacement") -> None:
        ix0, ix1, iy0, iy1 = pp.footprint_cells
        top_z = pp.base_z + pp.dims[2]
        self._heightmap[ix0:ix1, iy0:iy1] = top_z
        self._placed_boxes.append(PlacedBox(position=pp.center.copy(), dims=pp.dims.copy()))
        self._utilized_volume += float(np.prod(pp.dims))
        self._items_placed += 1

    def _footprint_cells(self, x0: float, y0: float, w: float, d: float) -> tuple[int, int, int, int]:
        gx, gy = self.grid_shape
        ix0 = int(np.clip(np.floor(x0 / self.grid_res), 0, gx - 1))
        ix1 = int(np.clip(np.ceil((x0 + w) / self.grid_res), ix0 + 1, gx))
        iy0 = int(np.clip(np.floor(y0 / self.grid_res), 0, gy - 1))
        iy1 = int(np.clip(np.ceil((y0 + d) / self.grid_res), iy0 + 1, gy))
        return ix0, ix1, iy0, iy1

    def _advance_item(self) -> None:
        self._item_idx += 1
        self._pending_placement = None
        episode_done = self._item_idx >= self.num_items_per_episode
        if episode_done:
            summary = {
                "utilization_pct": 100.0 * self._utilized_volume / self.bin_volume,
                "items_placed": self._items_placed,
                "items_total": self.num_items_per_episode,
            }
            for a in self.agents:
                self.terminations[a] = True
                self.infos[a] = {**self.infos.get(a, {}), "episode_summary": summary}


@dataclasses.dataclass
class _PendingPlacement:
    center: np.ndarray  # (3,) box center position if committed
    dims: np.ndarray  # (3,) rotated (w, d, h)
    footprint_cells: tuple[int, int, int, int]  # (ix0, ix1, iy0, iy1) into the heightmap
    base_z: float  # height of the stack under the footprint before this box
    valid: bool  # False if the box would exceed the bin's height
