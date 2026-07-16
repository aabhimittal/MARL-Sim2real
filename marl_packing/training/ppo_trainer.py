"""Joint training loop for the Proposer and Physics agents.

Per placement:
    1. Proposer samples (x, y, orientation) under the valid-action mask.
    2. Physics Agent previews the placement and issues a verdict
       (veto / stable / confidence).
    3. Environment commits or rejects; the proposer's reward reflects both
       packed volume and the physics outcome.
    4. Physics Agent's critic is trained online against the settle-test
       ground truth (its own simulation backend), so it keeps calibrating.

The proposer is updated with PPO-lite: clipped surrogate objective, GAE
advantages, minibatch epochs — implemented on the numpy MLPs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from marl_packing.agents.physics_agent import PhysicsAgent
from marl_packing.agents.proposer_agent import ProposerAgent
from marl_packing.envs.packing_env import PackingEnv


@dataclass
class TrainerConfig:
    episodes: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    log_every: int = 10
    seed: Optional[int] = None


@dataclass
class EpisodeStats:
    episode: int
    reward: float
    fill_ratio: float
    vetoes: int
    physics_loss: float


class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.masks: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs, action, mask, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.masks.append(mask)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, gamma: float, lam: float):
        """Standard GAE over a sequence of (possibly multiple) episodes."""
        T = len(self.rewards)
        adv = np.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            next_value = 0.0 if (t == T - 1 or self.dones[t]) else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last = delta + gamma * lam * (0.0 if self.dones[t] else last)
            adv[t] = last
        returns = adv + np.array(self.values)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def as_arrays(self):
        return (
            np.stack(self.obs),
            np.array(self.actions),
            np.stack(self.masks),
            np.array(self.log_probs),
        )


class MARLTrainer:
    def __init__(
        self,
        env: PackingEnv,
        proposer: ProposerAgent,
        physics: PhysicsAgent,
        config: Optional[TrainerConfig] = None,
    ):
        self.env = env
        self.proposer = proposer
        self.physics = physics
        self.cfg = config or TrainerConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.history: List[EpisodeStats] = []

    # ------------------------------------------------------------- rollouts

    def run_episode(self, buffer: Optional[RolloutBuffer] = None, greedy: bool = False) -> EpisodeStats:
        obs = self.env.reset()
        total_reward, physics_losses = 0.0, []
        done = False
        while not done:
            mask = self.env.valid_action_mask()
            if not mask.any():
                break  # nothing fits anywhere: episode over
            action, log_prob, value = self.proposer.act(obs, mask, greedy=greedy)

            placement = self.env.preview(action)
            verdict = self.physics.judge(placement)
            ground_truth = self.physics.simulate(placement)
            physics_losses.append(self.physics.learn(verdict.features, ground_truth))

            result = self.env.step(action, vetoed=verdict.veto, stable=ground_truth)
            total_reward += result.reward
            done = result.terminated

            if buffer is not None:
                buffer.add(
                    self.proposer.encode_obs(obs), action, mask,
                    log_prob, value, result.reward, done,
                )
            obs = result.obs

        return EpisodeStats(
            episode=len(self.history),
            reward=total_reward,
            fill_ratio=self.env.fill_ratio(),
            vetoes=self.env.veto_count,
            physics_loss=float(np.mean(physics_losses)) if physics_losses else 0.0,
        )

    # --------------------------------------------------------------- update

    def ppo_update(self, buffer: RolloutBuffer) -> None:
        obs, actions, masks, old_log_probs = buffer.as_arrays()
        adv, returns = buffer.compute_gae(self.cfg.gamma, self.cfg.gae_lambda)
        n = len(actions)
        for _ in range(self.cfg.update_epochs):
            order = self._rng.permutation(n)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = order[start : start + self.cfg.minibatch_size]
                log_probs, values, probs = self.proposer.evaluate(
                    obs[mb], actions[mb], masks[mb]
                )
                ratio = np.exp(log_probs - old_log_probs[mb])
                # Clipped-surrogate coefficient: zero gradient where clipped.
                unclipped_better = np.minimum(
                    ratio * adv[mb],
                    np.clip(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv[mb],
                ) == ratio * adv[mb]
                coeff = np.where(unclipped_better, ratio * adv[mb], 0.0)
                # Policy-gradient term: d(-coeff * log pi(a))/d(logits).
                grad = (probs - np.eye(probs.shape[1])[actions[mb]]) * coeff[:, None]
                # Entropy bonus (exact): d(-H)/d(logit_j) = p_j (log p_j + H).
                log_p = np.log(probs + 1e-12)
                H = -(probs * log_p).sum(axis=1, keepdims=True)
                grad += self.cfg.entropy_coef * probs * (log_p + H)
                self.proposer.policy.backward(grad)
                self.proposer.value_step(values, returns[mb])

    # ---------------------------------------------------------------- train

    def train(self, callback=None) -> List[EpisodeStats]:
        for ep in range(self.cfg.episodes):
            buffer = RolloutBuffer()
            stats = self.run_episode(buffer)
            if buffer.rewards:
                self.ppo_update(buffer)
            self.history.append(stats)
            if callback:
                callback(stats)
            elif (ep + 1) % self.cfg.log_every == 0:
                recent = self.history[-self.cfg.log_every:]
                print(
                    f"ep {ep + 1:4d} | reward {np.mean([s.reward for s in recent]):7.3f} "
                    f"| fill {np.mean([s.fill_ratio for s in recent]):.3f} "
                    f"| vetoes {np.mean([s.vetoes for s in recent]):.1f} "
                    f"| physics BCE {np.mean([s.physics_loss for s in recent]):.4f}"
                )
        return self.history
