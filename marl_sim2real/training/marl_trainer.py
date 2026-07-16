"""Co-training loop for the Proposer and Physics agents.

Per step:
  1. Proposer emits top-K candidate placements.
  2. Physics Agent scores each candidate; the best non-vetoed one is executed.
  3. The environment's ground-truth stability engine labels every candidate,
     giving the Physics Agent a supervised batch (experience replay).
  4. Episode returns update the Proposer via REINFORCE.

The veto loop means the Proposer is rewarded for proposals the Physics Agent
accepts AND that survive real physics — the two agents co-adapt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agents.physics_agent import PhysicsAgent
from ..agents.proposer import ProposerAgent, flat_to_action
from ..env.bin_packing_env import BinPackingEnv


@dataclass
class TrainStats:
    episode_returns: list[float] = field(default_factory=list)
    utilizations: list[float] = field(default_factory=list)
    physics_losses: list[float] = field(default_factory=list)
    physics_accuracy: float = 0.0

    def summary(self, last: int = 50) -> dict:
        r = self.episode_returns[-last:]
        u = self.utilizations[-last:]
        return {
            "episodes": len(self.episode_returns),
            "mean_return": float(np.mean(r)) if r else 0.0,
            "mean_utilization": float(np.mean(u)) if u else 0.0,
            "physics_accuracy": self.physics_accuracy,
        }


class ReplayBuffer:
    def __init__(self, capacity: int = 5000, seed: int = 0):
        self.capacity = capacity
        self.features: list[np.ndarray] = []
        self.labels: list[float] = []
        self.rng = np.random.default_rng(seed)

    def add(self, feat: np.ndarray, label: float) -> None:
        if len(self.features) >= self.capacity:
            idx = self.rng.integers(0, self.capacity)
            self.features[idx] = feat
            self.labels[idx] = label
        else:
            self.features.append(feat)
            self.labels.append(label)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        n = min(batch_size, len(self.features))
        idx = self.rng.choice(len(self.features), size=n, replace=False)
        return (
            np.stack([self.features[i] for i in idx]),
            np.array([self.labels[i] for i in idx]),
        )

    def __len__(self) -> int:
        return len(self.features)


class MARLTrainer:
    def __init__(self, env: BinPackingEnv, proposer: ProposerAgent, physics: PhysicsAgent, top_k: int = 5, seed: int = 0):
        self.env = env
        self.proposer = proposer
        self.physics = physics
        self.top_k = top_k
        self.buffer = ReplayBuffer(seed=seed)
        self.scale = float(max(env.L, env.W, env.H))
        self.stats = TrainStats()

    def run_episode(self, train: bool = True, use_veto: bool = True) -> tuple[float, float]:
        env = self.env
        obs = env.reset()
        trajectory: list[tuple[np.ndarray, int, float]] = []
        done = False
        while not done:
            mask = env.valid_actions_mask()
            candidates = self.proposer.propose(obs, mask=mask, top_k=self.top_k, sample=train)
            if not candidates:
                break

            # Label every candidate with ground-truth physics -> replay buffer
            feats, labels = [], []
            for flat in candidates:
                x, y, o = flat_to_action(flat, env.L, env.W)
                report = env.check_placement(x, y, o)
                dims = env.current_item.oriented(o)
                feat = self.physics.encode(obs, x, y, dims, self.scale)
                label = 1.0 if (report is not None and report.stable) else 0.0
                feats.append(feat)
                labels.append(label)
                if train:
                    self.buffer.add(feat, label)

            # Physics Agent picks the most-stable candidate it doesn't veto
            if use_veto:
                ranked = self.physics.rank(feats)
                chosen_idx = ranked[0][0]  # falls back to best-scoring even if all vetoed
            else:
                chosen_idx = 0
            flat = candidates[chosen_idx]

            step = env.step(flat_to_action(flat, env.L, env.W))
            trajectory.append((obs, flat, step.reward))
            obs, done = step.observation, step.done

        episode_return = 0.0
        if train and trajectory:
            episode_return = self.proposer.update(trajectory)
            if len(self.buffer) >= 64:
                for _ in range(4):  # several replay batches per episode
                    f, y = self.buffer.sample(64)
                    loss = self.physics.train_batch(f, y)
                self.stats.physics_losses.append(loss)
        else:
            episode_return = float(sum(r for _, _, r in trajectory))
        return episode_return, env.utilization()

    def train(self, episodes: int = 300, log_every: int = 50, verbose: bool = True) -> TrainStats:
        self.stats = TrainStats()
        for ep in range(episodes):
            ret, util = self.run_episode(train=True)
            self.stats.episode_returns.append(ret)
            self.stats.utilizations.append(util)
            if verbose and (ep + 1) % log_every == 0:
                s = self.stats.summary()
                print(
                    f"episode {ep + 1:5d}  return={s['mean_return']:+.2f}  "
                    f"utilization={s['mean_utilization']:.1%}"
                )
        if len(self.buffer) >= 64:
            f, y = self.buffer.sample(min(512, len(self.buffer)))
            self.stats.physics_accuracy = self.physics.accuracy(f, y)
        return self.stats

    def evaluate(self, episodes: int = 20) -> dict:
        utils, rets = [], []
        for _ in range(episodes):
            ret, util = self.run_episode(train=False)
            rets.append(ret)
            utils.append(util)
        return {
            "mean_return": float(np.mean(rets)),
            "mean_utilization": float(np.mean(utils)),
            "episodes": episodes,
        }
