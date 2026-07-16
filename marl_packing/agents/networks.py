"""Minimal numpy neural nets: an MLP with manual backprop.

Kept dependency-free on purpose so the whole project trains in CI without
torch. The API mirrors what you'd need to swap in a torch module later:
`forward`, `backward`, `step`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class MLP:
    """Fully-connected net with tanh hidden activations and a linear head."""

    def __init__(self, sizes: List[int], seed: Optional[int] = None, lr: float = 3e-3):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W = [
            rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            for fan_in, fan_out in zip(sizes[:-1], sizes[1:])
        ]
        self.b = [np.zeros(fan_out) for fan_out in sizes[1:]]
        self._cache: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float64)
        self._cache = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            self._cache.append(x)
        return x

    def backward(self, grad_out: np.ndarray) -> None:
        """Accumulate gradients for the last `forward` batch and apply SGD."""
        grad = np.atleast_2d(grad_out)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            a_in = self._cache[i]
            gW[i] = a_in.T @ grad / len(grad)
            gb[i] = grad.mean(axis=0)
            if i > 0:
                grad = grad @ self.W[i].T
                grad = grad * (1.0 - self._cache[i] ** 2)  # tanh'
        for i in range(len(self.W)):
            self.W[i] -= self.lr * np.clip(gW[i], -1.0, 1.0)
            self.b[i] -= self.lr * np.clip(gb[i], -1.0, 1.0)

    # -- persistence -------------------------------------------------------

    def state_dict(self) -> dict:
        return {f"W{i}": W for i, W in enumerate(self.W)} | {
            f"b{i}": b for i, b in enumerate(self.b)
        }

    def load_state_dict(self, state: dict) -> None:
        self.W = [np.array(state[f"W{i}"]) for i in range(len(self.W))]
        self.b = [np.array(state[f"b{i}"]) for i in range(len(self.b))]


def softmax(logits: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Numerically stable masked softmax over the last axis."""
    z = np.array(logits, dtype=np.float64)
    if mask is not None:
        z = np.where(mask, z, -1e9)
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
