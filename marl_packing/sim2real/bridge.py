"""Deployment bridge: runs a trained sim policy against real hardware.

`RobotInterface` is the seam between this repo and a physical cell — implement
it against your robot's SDK (ROS action server, vendor REST API, ...).
`MockRobot` implements the same interface on top of a perturbed simulator so
the full deployment loop is testable without hardware.

Per box, the bridge:
  1. reads the REAL heightmap from the robot's depth camera,
  2. syncs the sim environment to that observation (closing the loop each
     placement, so sim drift never accumulates),
  3. asks the proposer for a greedy action and the Physics Agent for a
     verdict, retrying with the next-best action on a veto (safety filter),
  4. executes the placement and records the real outcome into the
     RealityGapCalibrator for continual calibration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from marl_packing.agents.networks import softmax
from marl_packing.agents.physics_agent import PhysicsAgent
from marl_packing.agents.proposer_agent import ProposerAgent
from marl_packing.envs.packing_env import PackingConfig, PackingEnv
from marl_packing.sim2real.reality_gap import RealityGapCalibrator


class RobotInterface(ABC):
    """What the bridge needs from a physical packing cell."""

    @abstractmethod
    def read_heightmap(self) -> np.ndarray:
        """Current bin height field from the depth camera (W, D) ints."""

    @abstractmethod
    def next_box(self) -> Optional[Tuple[int, int, int]]:
        """Dims of the box at the pick point, or None when the feed is empty."""

    @abstractmethod
    def execute_placement(self, x: int, y: int, orientation: int) -> Dict:
        """Pick, rotate, place. Returns {'success': bool, 'stable': bool}."""

    def skip_box(self) -> None:
        """Divert the current box off the line (e.g. to a manual station).

        Called when the Physics Agent vetoes every candidate placement.
        Default is a no-op for cells without a diverter.
        """


class MockRobot(RobotInterface):
    """A 'real world' stand-in: a private env with perturbed physics.

    Its stability outcomes deviate from the nominal simulator (stricter
    support threshold + sensor noise), which is exactly the kind of gap the
    calibrator is meant to detect.
    """

    def __init__(
        self,
        config: Optional[PackingConfig] = None,
        support_threshold: float = 0.5,
        sensor_noise: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.env = PackingEnv(config or PackingConfig())
        self.support_threshold = support_threshold
        self.sensor_noise = sensor_noise
        self._rng = np.random.default_rng(seed)

    def read_heightmap(self) -> np.ndarray:
        hm = self.env.heightmap.astype(np.float64)
        hm += self._rng.normal(0.0, self.sensor_noise, hm.shape)
        return np.clip(np.round(hm), 0, self.env.H).astype(np.int32)

    def next_box(self) -> Optional[Tuple[int, int, int]]:
        return self.env.current_box()

    def execute_placement(self, x: int, y: int, orientation: int) -> Dict:
        action = self.env.encode_action(x, y, orientation)
        p = self.env.preview(action)
        if not p["in_bounds"]:
            self.env.step(action)  # consumes the box as an invalid attempt
            return {"success": False, "stable": False}
        # "Real" physics: harsher support requirement than the nominal sim.
        stable = p["z"] == 0 or p["support"] >= self.support_threshold
        self.env.step(action, stable=stable)
        return {"success": True, "stable": bool(stable)}

    def skip_box(self) -> None:
        self.env.cursor += 1


@dataclass
class DeploymentStats:
    boxes_attempted: int = 0
    boxes_placed: int = 0
    unstable_placements: int = 0
    vetoes: int = 0
    fill_ratio: float = 0.0


class Sim2RealBridge:
    def __init__(
        self,
        proposer: ProposerAgent,
        physics: PhysicsAgent,
        robot: RobotInterface,
        calibrator: Optional[RealityGapCalibrator] = None,
        max_retries: int = 3,
    ):
        self.proposer = proposer
        self.physics = physics
        self.robot = robot
        self.calibrator = calibrator or RealityGapCalibrator()
        self.max_retries = max_retries
        # A digital-twin env used purely for masks/previews, resynced to the
        # robot's observations before every decision.
        self.twin = proposer.env

    # ------------------------------------------------------------- one box

    def sync_twin(self) -> None:
        self.twin.heightmap = self.robot.read_heightmap().astype(np.int32)

    def place_next_box(self, stats: DeploymentStats) -> bool:
        """Plan and execute one placement. Returns False when the feed is empty."""
        box = self.robot.next_box()
        if box is None:
            return False
        self.sync_twin()
        self.twin.boxes = [box]
        self.twin.cursor = 0
        stats.boxes_attempted += 1

        obs = self.twin.observe()
        mask = self.twin.valid_action_mask()
        if not mask.any():
            return False  # bin is effectively full for this box

        # Greedy action ranking; walk down it past any Physics-Agent vetoes.
        x_enc = self.proposer.encode_obs(obs)
        probs = softmax(self.proposer.policy.forward(x_enc)[0], mask)
        ranking = np.argsort(-probs)

        for action in ranking[: self.max_retries]:
            if not mask[action]:
                break
            placement = self.twin.preview(int(action))
            verdict = self.physics.judge(placement)
            if verdict.veto:
                stats.vetoes += 1
                continue
            x, y, orientation = self.twin.decode_action(int(action))
            outcome = self.robot.execute_placement(x, y, orientation)
            self.calibrator.record(
                verdict.features,
                sim_stable=verdict.stable,
                real_stable=outcome["stable"],
                confidence=verdict.confidence,
            )
            if outcome["success"]:
                stats.boxes_placed += 1
                if not outcome["stable"]:
                    stats.unstable_placements += 1
            return True
        self.robot.skip_box()  # every candidate vetoed: divert, keep the line moving
        return True

    # ------------------------------------------------------------- episode

    def run(self, max_boxes: int = 1000) -> DeploymentStats:
        stats = DeploymentStats()
        for _ in range(max_boxes):
            if not self.place_next_box(stats):
                break
        if hasattr(self.robot, "env"):
            stats.fill_ratio = self.robot.env.fill_ratio()
        return stats
