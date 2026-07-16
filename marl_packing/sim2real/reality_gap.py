"""Reality-gap measurement and calibration (lightweight system identification).

Given paired outcomes — the simulator's stability prediction vs. what the real
robot observed for the same placements — this module:

  1. quantifies the gap (disagreement rate, calibration error, per-feature
     drift of the false predictions), and
  2. proposes updated `RandomizationConfig` ranges so the randomized simulator
     brackets reality ("adaptive domain randomization"), and
  3. fine-tunes the Physics Agent's critic on the real outcomes.

Real deployments feed `record` from robot telemetry; tests and demos feed it
from a second, perturbed simulator standing in for reality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from marl_packing.agents.physics_agent import PhysicsAgent
from marl_packing.sim2real.domain_randomization import RandomizationConfig


@dataclass
class GapReport:
    n_samples: int
    disagreement_rate: float      # sim verdict != real outcome
    sim_stable_rate: float
    real_stable_rate: float
    calibration_error: float      # |critic confidence - empirical stability|
    suggested_config: RandomizationConfig


@dataclass
class PairedOutcome:
    features: np.ndarray          # PhysicsAgent.features at decision time
    sim_stable: bool              # what the simulator predicted
    real_stable: bool             # what actually happened on hardware
    confidence: float             # critic confidence at decision time


class RealityGapCalibrator:
    def __init__(self, base_config: Optional[RandomizationConfig] = None):
        self.base = base_config or RandomizationConfig()
        self.samples: List[PairedOutcome] = []

    def record(
        self,
        features: np.ndarray,
        sim_stable: bool,
        real_stable: bool,
        confidence: float = 0.5,
    ) -> None:
        self.samples.append(
            PairedOutcome(np.asarray(features, dtype=np.float64), sim_stable, real_stable, confidence)
        )

    # ------------------------------------------------------------- analysis

    def report(self) -> GapReport:
        if not self.samples:
            return GapReport(0, 0.0, 0.0, 0.0, 0.0, self.base)
        sim = np.array([s.sim_stable for s in self.samples])
        real = np.array([s.real_stable for s in self.samples])
        conf = np.array([s.confidence for s in self.samples])
        disagreement = float((sim != real).mean())
        calibration = float(np.abs(conf - real.astype(float)).mean())
        return GapReport(
            n_samples=len(self.samples),
            disagreement_rate=disagreement,
            sim_stable_rate=float(sim.mean()),
            real_stable_rate=float(real.mean()),
            calibration_error=calibration,
            suggested_config=self._suggest_config(disagreement),
        )

    def _suggest_config(self, disagreement: float) -> RandomizationConfig:
        """Widen randomization in proportion to the observed gap.

        A 0% gap keeps the base ranges; a large gap scales them up (capped at
        3x) so training covers the regime reality actually lives in.
        """
        factor = float(np.clip(1.0 + 4.0 * disagreement, 1.0, 3.0))
        return self.base.scaled(factor)

    # ------------------------------------------------------------ transfer

    def finetune_physics_agent(self, agent: PhysicsAgent, epochs: int = 5) -> float:
        """Fine-tune the stability critic on REAL outcomes (the Sim2Real step
        for the Physics Agent). Returns the final mean BCE loss.
        """
        if not self.samples:
            return 0.0
        last = 0.0
        for _ in range(epochs):
            losses = [agent.learn(s.features, s.real_stable) for s in self.samples]
            last = float(np.mean(losses))
        return last
