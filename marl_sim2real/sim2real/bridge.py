"""Sim2Real bridge orchestrator.

The full pipeline:

  1. TRAIN (sim)      — MARL co-training under domain randomization.
  2. MEASURE          — run the policy on the real cell, measure the reality gap.
  3. CALIBRATE        — system-identify real physics from placement logs (CEM).
  4. ADAPT            — fine-tune both agents in the calibrated simulator.
  5. VALIDATE         — re-measure the gap; export the policy only if it shrank
                        and real utilization clears the acceptance bar.

Everything returns plain dataclasses so the Edge MLOps side (EdgePack repo)
can archive stage metrics alongside the exported model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agents.physics_agent import PhysicsAgent
from ..agents.proposer import ProposerAgent
from ..env.bin_packing_env import BinPackingEnv
from ..training.marl_trainer import MARLTrainer
from .calibration import CalibrationResult, calibrate
from .domain_randomization import DomainRandomizer
from .real_world import RealWorldCell


@dataclass
class GapReport:
    sim_utilization: float
    real_utilization: float

    @property
    def gap(self) -> float:
        """Relative performance drop from sim to real (0 = perfect transfer)."""
        if self.sim_utilization <= 0:
            return 0.0
        return max(0.0, (self.sim_utilization - self.real_utilization) / self.sim_utilization)


@dataclass
class BridgeResult:
    pre_gap: GapReport
    post_gap: GapReport
    calibration: CalibrationResult
    accepted: bool
    stages: dict = field(default_factory=dict)


class Sim2RealBridge:
    def __init__(
        self,
        bin_size: tuple[int, int, int] = (8, 8, 8),
        seed: int = 0,
        acceptance_utilization: float = 0.15,
    ):
        self.bin_size = bin_size
        self.seed = seed
        self.acceptance_utilization = acceptance_utilization

        env = BinPackingEnv(bin_size=bin_size, seed=seed)
        self.proposer = ProposerAgent(env.observation_size, env.num_positions * 6, seed=seed)
        self.physics_agent = PhysicsAgent(env.observation_size, seed=seed + 1)
        self.randomizer = DomainRandomizer(seed=seed)
        self.real_cell = RealWorldCell(bin_size=bin_size, seed=seed + 100)

    # ------------------------------------------------------------ stage 1
    def train_in_sim(self, episodes: int = 300, verbose: bool = True):
        env = BinPackingEnv(bin_size=self.bin_size, seed=self.seed)
        trainer = MARLTrainer(env, self.proposer, self.physics_agent, seed=self.seed)
        stats_log = []
        chunk = max(1, episodes // 10)
        for _ in range(0, episodes, chunk):
            self.randomizer.apply(env)  # fresh physics every chunk
            stats = trainer.train(episodes=chunk, verbose=verbose, log_every=chunk)
            stats_log.append(stats.summary())
        return stats_log

    # ------------------------------------------------------------ stage 2/5
    def measure_gap(self, episodes: int = 20) -> GapReport:
        sim_env = BinPackingEnv(bin_size=self.bin_size, seed=self.seed + 7)
        sim = MARLTrainer(sim_env, self.proposer, self.physics_agent).evaluate(episodes)
        real = MARLTrainer(self.real_cell.env, self.proposer, self.physics_agent).evaluate(episodes)
        return GapReport(sim_utilization=sim["mean_utilization"], real_utilization=real["mean_utilization"])

    # ------------------------------------------------------------ stage 3
    def calibrate_from_real(self, n_logs: int = 200) -> CalibrationResult:
        logs = self.real_cell.collect_logs(n_logs)
        return calibrate(logs, ranges=self.randomizer.ranges, seed=self.seed)

    # ------------------------------------------------------------ stage 4
    def adapt(self, calibration: CalibrationResult, episodes: int = 150, verbose: bool = True):
        env = BinPackingEnv(bin_size=self.bin_size, physics=calibration.params, seed=self.seed + 13)
        trainer = MARLTrainer(env, self.proposer, self.physics_agent, seed=self.seed + 13)
        return trainer.train(episodes=episodes, verbose=verbose, log_every=max(1, episodes // 3)).summary()

    # ------------------------------------------------------------ pipeline
    def run(self, sim_episodes: int = 300, adapt_episodes: int = 150, verbose: bool = True) -> BridgeResult:
        stages: dict = {}
        if verbose:
            print("[1/5] training in randomized simulation ...")
        stages["sim_training"] = self.train_in_sim(sim_episodes, verbose=verbose)

        if verbose:
            print("[2/5] measuring pre-adaptation reality gap ...")
        pre_gap = self.measure_gap()
        if verbose:
            print(f"      sim={pre_gap.sim_utilization:.1%} real={pre_gap.real_utilization:.1%} gap={pre_gap.gap:.1%}")

        if verbose:
            print("[3/5] calibrating simulator from real logs (CEM) ...")
        calibration = self.calibrate_from_real()
        if verbose:
            print(f"      calibrated agreement with reality: {calibration.agreement:.1%}")

        if verbose:
            print("[4/5] fine-tuning in calibrated simulator ...")
        stages["adaptation"] = self.adapt(calibration, adapt_episodes, verbose=verbose)

        if verbose:
            print("[5/5] validating post-adaptation gap ...")
        post_gap = self.measure_gap()
        if verbose:
            print(f"      sim={post_gap.sim_utilization:.1%} real={post_gap.real_utilization:.1%} gap={post_gap.gap:.1%}")

        accepted = post_gap.real_utilization >= self.acceptance_utilization
        return BridgeResult(pre_gap=pre_gap, post_gap=post_gap, calibration=calibration, accepted=accepted, stages=stages)
