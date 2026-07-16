"""Wraps the shared two-agent `PackingEnv` (a PettingZoo AECEnv) into single-agent
Gymnasium environments, so each agent can be trained with an off-the-shelf single-agent
algorithm (SB3 PPO).

The opponent's action is produced by a frozen policy supplied at construction time. That,
plus alternating which wrapper trains between rounds (see training/train.py), is what makes
this "alternating independent-learner" MARL: train the packer with physics frozen, then
swap, repeat -- a simple, well-established scheme that avoids depending on a heavier
native multi-agent trainer.
"""

from __future__ import annotations

from typing import Protocol

import gymnasium as gym
import numpy as np

from marl_packing.envs.packing_env import PackingEnv


class Policy(Protocol):
    def predict(self, observation, deterministic: bool = False): ...


class RandomOpponent:
    """Stand-in opponent used before the other agent has been trained yet (round 0)."""

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def predict(self, observation, deterministic: bool = False):
        return self.action_space.sample(), None


class PackerSingleAgentEnv(gym.Env):
    """Single-agent view of `packer`'s turn. Physics's turn is played out internally by
    `physics_policy` immediately after each packer action, so from SB3's perspective this
    is an ordinary Gymnasium env with a one-item-per-step transition."""

    def __init__(self, base_env: PackingEnv, physics_policy: Policy | None = None, seed: int | None = None):
        super().__init__()
        self.base_env = base_env
        self.physics_policy = physics_policy or RandomOpponent(base_env.action_space("physics"))
        self.observation_space = base_env.observation_space("packer")
        self.action_space = base_env.action_space("packer")
        self._seed = seed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base_env.reset(seed=seed if seed is not None else self._seed)
        assert self.base_env.agent_selection == "packer"
        return self.base_env.observe("packer"), {}

    def step(self, action):
        assert self.base_env.agent_selection == "packer"
        self.base_env.step(np.asarray(action, dtype=np.float32))

        reward = 0.0
        info: dict = {}
        terminated = self.base_env.terminations["packer"]
        truncated = self.base_env.truncations["packer"]

        if not (terminated or truncated):
            assert self.base_env.agent_selection == "physics"
            physics_obs = self.base_env.observe("physics")
            physics_action, _ = self.physics_policy.predict(physics_obs, deterministic=False)
            self.base_env.step(int(physics_action))
            reward = float(self.base_env.rewards.get("packer", 0.0))
            terminated = self.base_env.terminations["packer"]
            truncated = self.base_env.truncations["packer"]
            if terminated or truncated:
                info = dict(self.base_env.infos.get("packer", {}))

        obs = self.base_env.observe("packer")
        return obs, reward, terminated, truncated, info

    def close(self):
        pass  # base_env owns the PyBullet client's lifecycle; closed by the caller.


class PhysicsSingleAgentEnv(gym.Env):
    """Single-agent view of `physics`'s turn. The packer's turn is played out internally by
    `packer_policy` (both at reset, to produce the first proposal, and after each physics
    decision, to produce the next one)."""

    def __init__(self, base_env: PackingEnv, packer_policy: Policy | None = None, seed: int | None = None):
        super().__init__()
        self.base_env = base_env
        self.packer_policy = packer_policy or RandomOpponent(base_env.action_space("packer"))
        self.observation_space = base_env.observation_space("physics")
        self.action_space = base_env.action_space("physics")
        self._seed = seed

    def _drive_packer_turn(self) -> None:
        assert self.base_env.agent_selection == "packer"
        packer_obs = self.base_env.observe("packer")
        packer_action, _ = self.packer_policy.predict(packer_obs, deterministic=False)
        self.base_env.step(np.asarray(packer_action, dtype=np.float32))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base_env.reset(seed=seed if seed is not None else self._seed)
        self._drive_packer_turn()
        assert self.base_env.agent_selection == "physics"
        return self.base_env.observe("physics"), {}

    def step(self, action):
        assert self.base_env.agent_selection == "physics"
        self.base_env.step(int(action))

        reward = float(self.base_env.rewards.get("physics", 0.0))
        terminated = self.base_env.terminations["physics"]
        truncated = self.base_env.truncations["physics"]
        info = dict(self.base_env.infos.get("physics", {})) if (terminated or truncated) else {}

        if not (terminated or truncated):
            self._drive_packer_turn()

        obs = self.base_env.observe("physics")
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
