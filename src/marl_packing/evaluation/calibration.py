"""Sim2Real calibration via system identification: fit the PyBullet domain-randomization
point (friction, mass_scale, restitution) that best reproduces real-world placement
outcomes, using the Cross-Entropy Method (CEM).

`configs/env.yaml`'s `real_proxy` profile is currently a hand-picked, wider-than-training
randomization range standing in for real-world variation (see docs/SIM2REAL.md) -- an
honest placeholder, not a claim of real validation. This module is the concrete mechanism
for replacing that guess once actual field data exists: collect `PlacementLog`s from a
real cell (or a `real_world_reference.csv`-style operator trial log), then call
`calibrate()` to search for the physics point that makes `DropSimulator` agree with what
was actually observed. CEM is derivative-free and robust to this objective's noisy,
discrete (agreement-fraction) nature -- it converges in a few dozen simulator evaluations
rather than needing gradients through the physics engine.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from marl_packing.envs.pybullet_sim import DropSimulator, PlacedBox

# Search bounds for (friction, mass_scale, restitution), covering a generous superset of
# both configs/env.yaml's "train" and "real_proxy" ranges.
_BOUNDS = np.array([[0.05, 1.2], [0.5, 1.5], [0.0, 0.5]])


@dataclasses.dataclass
class PlacementLog:
    """One observed real-world placement attempt: the stack state it was dropped onto,
    the candidate box, and whether it was actually stable."""

    existing_boxes: list[PlacedBox]
    candidate: PlacedBox
    stable: bool


@dataclasses.dataclass
class CalibratedParams:
    friction: float
    mass_scale: float
    restitution: float

    def as_randomization_profile(self, pose_noise_m: float = 0.0) -> dict:
        """A degenerate (zero-width) domain-randomization profile pinned to this
        calibrated point -- drop this straight into `DropSimulator.simulate_drop` or use it
        to re-center `configs/env.yaml`'s `real_proxy` range."""
        return {
            "friction_range": [self.friction, self.friction],
            "mass_scale_range": [self.mass_scale, self.mass_scale],
            "restitution_range": [self.restitution, self.restitution],
            "pose_noise_m": pose_noise_m,
        }


@dataclasses.dataclass
class CalibrationResult:
    params: CalibratedParams
    agreement: float  # fraction of logs the calibrated params reproduce
    history: list[float]  # best agreement per CEM iteration, for convergence diagnostics


def _agreement(sim: DropSimulator, params: CalibratedParams, logs: list[PlacementLog], seed: int) -> float:
    rng = np.random.default_rng(seed)
    profile = params.as_randomization_profile()
    correct = 0
    for log in logs:
        result = sim.simulate_drop(log.existing_boxes, log.candidate, rng, profile)
        correct += int(result.stable == log.stable)
    return correct / max(1, len(logs))


def calibrate(
    logs: list[PlacementLog],
    sim_config: dict,
    iterations: int = 8,
    population: int = 24,
    elite_frac: float = 0.25,
    seed: int = 0,
) -> CalibrationResult:
    if not logs:
        raise ValueError("calibrate() requires at least one PlacementLog")

    rng = np.random.default_rng(seed)
    lo, hi = _BOUNDS[:, 0], _BOUNDS[:, 1]
    mean = (lo + hi) / 2.0
    std = (hi - lo) / 2.0
    n_elite = max(2, int(population * elite_frac))
    history: list[float] = []
    best_vec, best_score = mean.copy(), -1.0

    sim = DropSimulator(sim_config)
    try:
        for _ in range(iterations):
            samples = np.clip(rng.normal(mean, std, size=(population, 3)), lo, hi)
            scores = np.array(
                [_agreement(sim, CalibratedParams(*sample), logs, seed) for sample in samples]
            )
            elite = samples[np.argsort(-scores)[:n_elite]]
            mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
            if scores.max() > best_score:
                best_score = float(scores.max())
                best_vec = samples[int(np.argmax(scores))]
            history.append(best_score)
    finally:
        sim.close()

    return CalibrationResult(params=CalibratedParams(*best_vec), agreement=best_score, history=history)
