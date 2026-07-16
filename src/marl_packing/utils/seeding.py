"""Deterministic seeding helpers shared across env, training, and evaluation."""

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed python/numpy global RNGs and return a dedicated numpy Generator.

    A dedicated Generator is returned (rather than relying solely on global
    state) so callers can pass it explicitly into environments that need
    reproducible-but-independent randomness streams (e.g. domain randomization).
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
