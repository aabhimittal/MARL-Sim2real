"""Tests for evaluation/calibration.py's CEM-based Sim2Real parameter calibration.

Kept fast (few logs, few CEM iterations) since each evaluation runs a real PyBullet drop
simulation -- these are integration tests, not pure-function tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from marl_packing.envs.pybullet_sim import DropSimulator, PlacedBox
from marl_packing.evaluation.calibration import CalibratedParams, PlacementLog, calibrate
from marl_packing.utils.config import load_config


@pytest.fixture
def env_config():
    return load_config("configs/env.yaml")


def _synthetic_logs(env_config, n=6, seed=0):
    rng = np.random.default_rng(seed)
    profile = {
        "friction_range": [0.5, 0.5],
        "mass_scale_range": [1.0, 1.0],
        "restitution_range": [0.05, 0.05],
        "pose_noise_m": 0.0,
    }
    sim = DropSimulator(env_config)
    logs = []
    try:
        for i in range(n):
            dims = rng.uniform(0.1, 0.2, size=3).astype(np.float32)
            pos = np.array([0.5, 0.5, dims[2] / 2], dtype=np.float32)
            candidate = PlacedBox(position=pos, dims=dims)
            result = sim.simulate_drop([], candidate, rng, profile)
            logs.append(PlacementLog(existing_boxes=[], candidate=candidate, stable=result.stable))
    finally:
        sim.close()
    return logs


def test_calibrate_raises_on_empty_logs(env_config):
    with pytest.raises(ValueError):
        calibrate([], env_config)


def test_calibrate_returns_params_within_search_bounds(env_config):
    logs = _synthetic_logs(env_config)
    result = calibrate(logs, env_config, iterations=2, population=8, seed=1)

    assert 0.05 <= result.params.friction <= 1.2
    assert 0.5 <= result.params.mass_scale <= 1.5
    assert 0.0 <= result.params.restitution <= 0.5


def test_calibrate_agreement_is_a_valid_fraction(env_config):
    logs = _synthetic_logs(env_config)
    result = calibrate(logs, env_config, iterations=2, population=8, seed=1)

    assert 0.0 <= result.agreement <= 1.0
    assert len(result.history) == 2
    # History tracks the best-so-far score, so it is monotonically non-decreasing.
    assert all(b >= a for a, b in zip(result.history, result.history[1:]))


def test_calibrated_params_as_randomization_profile_is_degenerate():
    params = CalibratedParams(friction=0.6, mass_scale=1.1, restitution=0.2)
    profile = params.as_randomization_profile(pose_noise_m=0.005)

    assert profile["friction_range"] == [0.6, 0.6]
    assert profile["mass_scale_range"] == [1.1, 1.1]
    assert profile["restitution_range"] == [0.2, 0.2]
    assert profile["pose_noise_m"] == 0.005
