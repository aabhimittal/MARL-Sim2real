"""System identification: fit simulator physics to real-world placement logs.

Cross-entropy-method (CEM) search over :class:`PhysicsParams`, scoring each
candidate by how well a simulator with those parameters reproduces observed
real stability outcomes.  CEM is derivative-free, robust to the discrete /
noisy objective, and converges in a few dozen simulator evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.physics import PhysicsParams, StabilityEngine
from .domain_randomization import RandomizationRanges
from .real_world import PlacementLog


@dataclass
class CalibrationResult:
    params: PhysicsParams
    agreement: float          # fraction of logs the calibrated sim reproduces
    history: list[float]      # best agreement per CEM iteration


def _agreement(params: PhysicsParams, logs: list[PlacementLog], seed: int = 0) -> float:
    engine = StabilityEngine(params, rng=np.random.default_rng(seed))
    correct = 0
    for log in logs:
        report = engine.evaluate(log.heightmap, log.x, log.y, log.dims, log.mass)
        correct += int(report.stable == log.stable)
    return correct / max(1, len(logs))


def calibrate(
    logs: list[PlacementLog],
    ranges: RandomizationRanges | None = None,
    iterations: int = 8,
    population: int = 24,
    elite_frac: float = 0.25,
    seed: int = 0,
) -> CalibrationResult:
    ranges = ranges or RandomizationRanges()
    rng = np.random.default_rng(seed)
    bounds = np.array(list(ranges.as_dict().values()), dtype=np.float64)  # (5, 2)
    lo, hi = bounds[:, 0], bounds[:, 1]
    mean = (lo + hi) / 2.0
    std = (hi - lo) / 2.0
    n_elite = max(2, int(population * elite_frac))
    history: list[float] = []
    best_vec, best_score = mean.copy(), -1.0

    for _ in range(iterations):
        samples = rng.normal(mean, std, size=(population, 5))
        samples = np.clip(samples, lo, hi)
        scores = np.array([_agreement(PhysicsParams.from_vector(v), logs, seed) for v in samples])
        elite = samples[np.argsort(-scores)[:n_elite]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
        if scores.max() > best_score:
            best_score = float(scores.max())
            best_vec = samples[int(np.argmax(scores))]
        history.append(best_score)

    return CalibrationResult(
        params=PhysicsParams.from_vector(best_vec),
        agreement=best_score,
        history=history,
    )
