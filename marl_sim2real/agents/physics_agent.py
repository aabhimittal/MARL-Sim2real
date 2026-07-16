"""Physics Agent — learns to predict placement stability and veto bad proposals.

During simulation training it is supervised by the ground-truth
:class:`StabilityEngine`.  After Sim2Real calibration it is fine-tuned on
(scarce, noisy) real-world stability labels.  On the edge device — where no
physics engine exists — this learned model IS the physics check.
"""

from __future__ import annotations

import numpy as np

from .networks import MLP, sigmoid


class PhysicsAgent:
    def __init__(self, obs_size: int, hidden: int = 64, seed: int = 1, lr: float = 5e-3):
        # input: env observation + candidate (x, y, l, w, h) normalized
        self.input_size = obs_size + 5
        self.net = MLP([self.input_size, hidden, hidden, 1], seed=seed)
        self.lr = lr
        self.threshold = 0.5

    @staticmethod
    def encode(obs: np.ndarray, x: int, y: int, dims: tuple[int, int, int], scale: float) -> np.ndarray:
        cand = np.array([x, y, dims[0], dims[1], dims[2]], dtype=np.float64) / scale
        return np.concatenate([obs, cand])

    def stability_prob(self, features: np.ndarray) -> float:
        return float(sigmoid(self.net.forward(features))[0, 0])

    def veto(self, features: np.ndarray) -> bool:
        """True = reject the proposal as unstable."""
        return self.stability_prob(features) < self.threshold

    def rank(self, candidates: list[np.ndarray]) -> list[tuple[int, float]]:
        """Return (candidate_index, stability_prob) sorted most-stable first."""
        scored = [(i, self.stability_prob(f)) for i, f in enumerate(candidates)]
        return sorted(scored, key=lambda t: -t[1])

    def train_batch(self, features: np.ndarray, labels: np.ndarray) -> float:
        """One SGD step of binary cross-entropy. Returns loss."""
        logits = self.net.forward(features)
        probs = sigmoid(logits)
        y = labels.reshape(-1, 1).astype(np.float64)
        eps = 1e-9
        loss = float(-(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)).mean())
        self.net.backward((probs - y) / len(y), lr=self.lr)
        return loss

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        probs = sigmoid(self.net.forward(features)).reshape(-1)
        return float(((probs >= self.threshold) == labels.astype(bool)).mean())

    # ------------------------------------------------------------- serialize
    def get_params(self) -> dict[str, np.ndarray]:
        return self.net.get_params()

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        self.net.set_params(params)
