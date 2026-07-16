"""Proposer Agent — proposes packing positions and orientations.

A softmax policy over the flat action space ``L*W*6`` (position x orientation),
trained with REINFORCE + a moving-average baseline.  At decision time it emits
its top-K candidate actions so the Physics Agent can veto unstable ones —
that negotiation is the multi-agent part of the system.
"""

from __future__ import annotations

import numpy as np

from .networks import MLP, softmax


class ProposerAgent:
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 128, seed: int = 0, lr: float = 3e-3):
        self.num_actions = num_actions
        self.net = MLP([obs_size, hidden, hidden, num_actions], seed=seed)
        self.lr = lr
        self.baseline = 0.0
        self.baseline_beta = 0.95
        self.rng = np.random.default_rng(seed)
        self.entropy_coef = 0.01

    # ---------------------------------------------------------------- acting
    def action_probs(self, obs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        logits = self.net.forward(obs)[0]
        if mask is not None:
            logits = np.where(mask, logits, -1e9)
        return softmax(logits[None, :])[0]

    def propose(
        self,
        obs: np.ndarray,
        mask: np.ndarray | None = None,
        top_k: int = 5,
        sample: bool = True,
    ) -> list[int]:
        """Return top-K candidate flat actions, best first."""
        probs = self.action_probs(obs, mask)
        if sample:
            k = min(top_k, int((probs > 0).sum()))
            if k == 0:
                return []
            return list(self.rng.choice(self.num_actions, size=k, replace=False, p=probs))
        return list(np.argsort(-probs)[:top_k])

    # -------------------------------------------------------------- learning
    def update(self, trajectory: list[tuple[np.ndarray, int, float]], gamma: float = 0.99) -> float:
        """REINFORCE with baseline over one episode.

        ``trajectory`` is a list of (obs, chosen_flat_action, reward).
        Returns the episode return (for logging).
        """
        if not trajectory:
            return 0.0
        rewards = np.array([r for _, _, r in trajectory], dtype=np.float64)
        returns = np.zeros_like(rewards)
        acc = 0.0
        for t in reversed(range(len(rewards))):
            acc = rewards[t] + gamma * acc
            returns[t] = acc

        episode_return = float(returns[0])
        self.baseline = self.baseline_beta * self.baseline + (1 - self.baseline_beta) * episode_return

        for (obs, action, _), g in zip(trajectory, returns):
            advantage = g - self.baseline
            logits = self.net.forward(obs)
            probs = softmax(logits)[0]
            # d(-logpi(a) * A)/dlogits = (probs - onehot(a)) * A ; entropy bonus
            grad = probs.copy()
            grad[action] -= 1.0
            grad *= advantage
            grad += self.entropy_coef * probs * (np.log(probs + 1e-9) + 1.0)  # -H gradient
            self.net.backward(grad[None, :] / len(trajectory), lr=self.lr)
        return episode_return

    # ------------------------------------------------------------- serialize
    def get_params(self) -> dict[str, np.ndarray]:
        return self.net.get_params()

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        self.net.set_params(params)


def flat_to_action(flat: int, L: int, W: int) -> tuple[int, int, int]:
    """Flat index over (L, W, 6) -> (x, y, orientation)."""
    o = flat % 6
    rest = flat // 6
    y = rest % W
    x = rest // W
    return x, y, o


def action_to_flat(x: int, y: int, o: int, L: int, W: int) -> int:
    return (x * W + y) * 6 + o
