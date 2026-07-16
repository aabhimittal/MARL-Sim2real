"""Domain randomization: train against perturbed physics so the policy is
robust to the sim-vs-reality mismatch instead of overfitting the simulator.

Randomized per episode (redrawn on every `env.reset` via the wrapper):
  - box dimension tolerance  (manufacturing / measurement error)
  - placement jitter          (gripper positioning error)
  - support threshold shift   (friction & surface-compliance variation)
  - observation noise         (depth-camera error on the heightmap)

The wrapper composes around `PackingEnv` without modifying it, and its
parameter ranges are exactly what the RealityGapCalibrator tunes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from marl_packing.envs.packing_env import PackingEnv, StepResult


@dataclass
class RandomizationConfig:
    dim_tolerance: float = 0.15      # P(a box edge is measured ±1 voxel off)
    placement_jitter: float = 0.1    # P(the commit lands 1 cell off target)
    obs_noise_std: float = 0.2       # gaussian noise on heightmap readings
    support_shift_std: float = 0.05  # perturbation of stability thresholds
    seed: Optional[int] = None

    def scaled(self, factor: float) -> "RandomizationConfig":
        """A copy with all magnitudes scaled — used for curriculum ramps."""
        return replace(
            self,
            dim_tolerance=self.dim_tolerance * factor,
            placement_jitter=self.placement_jitter * factor,
            obs_noise_std=self.obs_noise_std * factor,
            support_shift_std=self.support_shift_std * factor,
        )


class DomainRandomizer:
    """Wraps a PackingEnv, perturbing dynamics and observations per episode."""

    def __init__(self, env: PackingEnv, config: Optional[RandomizationConfig] = None):
        self.env = env
        self.rand_cfg = config or RandomizationConfig()
        self._rng = np.random.default_rng(self.rand_cfg.seed)
        self._episode_support_shift = 0.0

    # Delegate everything the agents use straight through.
    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, box_sequence=None) -> Dict:
        obs = self.env.reset(box_sequence)
        self._episode_support_shift = self._rng.normal(0.0, self.rand_cfg.support_shift_std)
        self._perturb_boxes()
        return self._noisy_obs(self.env.observe())

    def step(self, action: int, vetoed: bool = False, stable: bool = True) -> StepResult:
        action = self._jitter_action(action)
        result = self.env.step(action, vetoed=vetoed, stable=stable)
        return StepResult(
            self._noisy_obs(result.obs),
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )

    @property
    def support_shift(self) -> float:
        """Per-episode stability-threshold offset for the Physics Agent."""
        return self._episode_support_shift

    # ------------------------------------------------------------ internals

    def _perturb_boxes(self) -> None:
        boxes = []
        for dims in self.env.boxes:
            dims = list(dims)
            for k in range(3):
                if self._rng.random() < self.rand_cfg.dim_tolerance:
                    dims[k] = int(np.clip(dims[k] + self._rng.choice([-1, 1]), 1, None))
            boxes.append(tuple(dims))
        self.env.boxes = boxes

    def _jitter_action(self, action: int) -> int:
        if self._rng.random() >= self.rand_cfg.placement_jitter:
            return action
        x, y, o = self.env.decode_action(action)
        x = int(np.clip(x + self._rng.choice([-1, 1]), 0, self.env.W - 1))
        y = int(np.clip(y + self._rng.choice([-1, 1]), 0, self.env.D - 1))
        return self.env.encode_action(x, y, o)

    def _noisy_obs(self, obs: Dict) -> Dict:
        noise = self._rng.normal(0.0, self.rand_cfg.obs_noise_std, obs["heightmap"].shape)
        noisy = np.clip(np.round(obs["heightmap"] + noise), 0, self.env.H)
        return {**obs, "heightmap": noisy.astype(obs["heightmap"].dtype)}
