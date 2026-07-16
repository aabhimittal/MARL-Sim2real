"""Proposer agent: a policy-gradient (REINFORCE with baseline) network that
proposes packing actions (orientation, x, y) given the bin heightmap and the
current item's dimensions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl_sim2real.config import TrainConfig
from marl_sim2real.envs.packing_env import PackingEnv


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)


class ProposerAgent:
    def __init__(self, env: PackingEnv, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        torch.manual_seed(self.config.seed)
        obs_dim = env.observe().shape[0]
        self.action_dim = env.action_space_size()
        self.net = PolicyNet(obs_dim, self.action_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        self._episode: list[dict] = []

    # ------------------------------------------------------------- acting
    def act(self, obs: np.ndarray, feasible_mask: np.ndarray) -> int:
        """Sample an action, masking geometrically infeasible placements."""
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        logits, value = self.net(obs_t)
        logits = logits.squeeze(0)
        mask = torch.from_numpy(feasible_mask.astype(np.bool_))
        logits = logits.masked_fill(~mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        self._episode.append({
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy(),
            "value": value.squeeze(0),
            "reward": 0.0,
        })
        return int(action.item())

    def record_reward(self, reward: float) -> None:
        if self._episode:
            self._episode[-1]["reward"] += float(reward)

    # ------------------------------------------------------------ learning
    def finish_episode(self) -> float:
        """REINFORCE with a learned value baseline. Returns policy loss value."""
        if not self._episode:
            return 0.0
        gamma = self.config.gamma
        returns, running = [], 0.0
        for step in reversed(self._episode):
            running = step["reward"] + gamma * running
            returns.append(running)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)

        log_probs = torch.stack([s["log_prob"] for s in self._episode])
        values = torch.stack([s["value"] for s in self._episode])
        entropies = torch.stack([s["entropy"] for s in self._episode])

        advantage = returns_t - values.detach()
        policy_loss = -(log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + 0.5 * value_loss - self.config.entropy_coef * entropies.mean()

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optim.step()
        self._episode = []
        return float(loss.item())

    # -------------------------------------------------------------- helpers
    @staticmethod
    def feasible_mask(env: PackingEnv) -> np.ndarray:
        mask = np.zeros(env.action_space_size(), dtype=bool)
        for action in range(env.action_space_size()):
            if env.try_place(action) is not None:
                mask[action] = True
        return mask

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, weights_only=True))
