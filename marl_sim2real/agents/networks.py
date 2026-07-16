"""Minimal numpy MLP with manual backprop.

Deliberately dependency-free (no torch) so the whole MARL stack trains inside
CI and on edge devices.  Two heads are built from this: a softmax policy for
the Proposer Agent and a sigmoid classifier for the Physics Agent.
"""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0):
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0, np.sqrt(2.0 / sizes[i]), size=(sizes[i], sizes[i + 1]))
            for i in range(len(sizes) - 1)
        ]
        self.biases = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self._cache: list[np.ndarray] = []

    # ---------------------------------------------------------------- forward
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, in) -> logits (batch, out). Caches activations for backward."""
        x = np.atleast_2d(x)
        self._cache = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.tanh(x)
            self._cache.append(x)
        return x

    # --------------------------------------------------------------- backward
    def backward(self, dlogits: np.ndarray, lr: float, clip: float = 5.0) -> None:
        """SGD step given gradient of loss w.r.t. output logits."""
        grad = np.atleast_2d(dlogits)
        for i in reversed(range(len(self.weights))):
            a_prev = self._cache[i]
            dw = a_prev.T @ grad
            db = grad.sum(axis=0)
            dw = np.clip(dw, -clip, clip)
            db = np.clip(db, -clip, clip)
            if i > 0:
                grad = (grad @ self.weights[i].T) * (1.0 - self._cache[i] ** 2)
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    # -------------------------------------------------------------- serialize
    def get_params(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {"sizes": np.array(self.sizes)}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            out[f"w{i}"] = w
            out[f"b{i}"] = b
        return out

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        n = len(self.weights)
        self.weights = [np.array(params[f"w{i}"]) for i in range(n)]
        self.biases = [np.array(params[f"b{i}"]) for i in range(n)]

    @classmethod
    def from_params(cls, params: dict[str, np.ndarray]) -> "MLP":
        net = cls([int(s) for s in params["sizes"]])
        net.set_params(params)
        return net


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
