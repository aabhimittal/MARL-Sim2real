"""Proposer Agent: a masked-softmax policy over (x, y, orientation) placements.

Observation encoding: flattened heightmap (normalized by bin height) + the
lookahead window of upcoming box dimensions (normalized by max box edge).
Policy and value share the observation but use separate MLPs, which keeps
the manual-backprop bookkeeping trivial.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from marl_packing.agents.networks import MLP, softmax
from marl_packing.envs.packing_env import PackingEnv


class ProposerAgent:
    def __init__(
        self,
        env: PackingEnv,
        hidden: int = 128,
        lr: float = 3e-3,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.obs_dim = env.W * env.D + env.cfg.lookahead * 3 + 1
        self.n_actions = env.action_space_size
        self.policy = MLP([self.obs_dim, hidden, hidden, self.n_actions], seed=seed, lr=lr)
        self.value = MLP([self.obs_dim, hidden, 1], seed=None if seed is None else seed + 1, lr=lr)
        self._rng = np.random.default_rng(seed)

    def encode_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        hm = obs["heightmap"].astype(np.float64).ravel() / max(self.env.H, 1)
        boxes = obs["next_boxes"].astype(np.float64).ravel() / max(self.env.cfg.max_box, 1)
        remaining = obs["remaining"].astype(np.float64) / max(self.env.cfg.num_boxes, 1)
        return np.concatenate([hm, boxes, remaining])

    def act(
        self,
        obs: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
        greedy: bool = False,
    ) -> Tuple[int, float, float]:
        """Sample an action. Returns (action, log_prob, value_estimate)."""
        x = self.encode_obs(obs)
        logits = self.policy.forward(x)[0]
        probs = softmax(logits, mask)
        if greedy:
            action = int(np.argmax(probs))
        else:
            action = int(self._rng.choice(self.n_actions, p=probs))
        v = float(self.value.forward(x)[0, 0])
        return action, float(np.log(probs[action] + 1e-12)), v

    def evaluate(
        self, obs_batch: np.ndarray, actions: np.ndarray, masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch (log_probs, values, probs) for a PPO update pass."""
        logits = self.policy.forward(obs_batch)
        probs = softmax(logits, masks)
        idx = np.arange(len(actions))
        log_probs = np.log(probs[idx, actions] + 1e-12)
        values = self.value.forward(obs_batch)[:, 0]
        return log_probs, values, probs

    # -- gradient plumbing used by the trainer ------------------------------

    def value_step(self, values: np.ndarray, returns: np.ndarray) -> None:
        self.value.backward((values - returns)[:, None])

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        state = {f"policy_{k}": v for k, v in self.policy.state_dict().items()}
        state |= {f"value_{k}": v for k, v in self.value.state_dict().items()}
        np.savez(path, **state)

    def load(self, path: str) -> None:
        data = dict(np.load(path))
        self.policy.load_state_dict(
            {k[len("policy_"):]: v for k, v in data.items() if k.startswith("policy_")}
        )
        self.value.load_state_dict(
            {k[len("value_"):]: v for k, v in data.items() if k.startswith("value_")}
        )
