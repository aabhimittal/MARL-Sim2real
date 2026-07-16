"""Physics agent: validates proposed placements by dropping the item in a
PyBullet simulation and measuring how far it settles from the proposed pose.

Runs headless (DIRECT mode). If PyBullet is unavailable, falls back to an
analytic support-ratio heuristic so the rest of the pipeline still works.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from marl_sim2real.config import PhysicsConfig
from marl_sim2real.envs.packing_env import PackingEnv, Placement

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYBULLET_AVAILABLE = False


@dataclasses.dataclass
class StabilityReport:
    stable: bool
    displacement: float      # metres the item drifted while settling
    tilt_deg: float          # final tilt from upright
    score: float             # 1.0 = perfectly stable, 0.0 = fell over
    used_pybullet: bool


class PhysicsAgent:
    def __init__(self, config: PhysicsConfig | None = None, use_pybullet: bool | None = None):
        self.config = config or PhysicsConfig()
        self.use_pybullet = PYBULLET_AVAILABLE if use_pybullet is None else use_pybullet
        self._client = None

    # ------------------------------------------------------------------ API
    def evaluate(self, env: PackingEnv, placement: Placement) -> StabilityReport:
        if self.use_pybullet:
            return self._evaluate_pybullet(env, placement)
        return self._evaluate_heuristic(env, placement)

    def close(self) -> None:
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None

    # ------------------------------------------------------------- pybullet
    def _connect(self) -> int:
        if self._client is None:
            self._client = p.connect(p.DIRECT)
        return self._client

    def _evaluate_pybullet(self, env: PackingEnv, placement: Placement) -> StabilityReport:
        cfg = self.config
        client = self._connect()
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0, 0, cfg.gravity, physicsClientId=client)
        p.setTimeStep(cfg.timestep, physicsClientId=client)

        # Floor
        floor = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        p.createMultiBody(0, floor, physicsClientId=client)

        # Already-placed items are static obstacles (they are assumed settled)
        for prev in env.placements:
            pos, half = env.placement_to_world(prev)
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
            p.createMultiBody(0, shape, basePosition=pos, physicsClientId=client)

        # Candidate item: dynamic body released just above its proposed pose
        pos, half = env.placement_to_world(placement)
        drop_pos = (pos[0], pos[1], pos[2] + 0.005)
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
        body = p.createMultiBody(
            baseMass=1.0, baseCollisionShapeIndex=shape,
            basePosition=drop_pos, physicsClientId=client,
        )
        p.changeDynamics(body, -1, lateralFriction=0.6, physicsClientId=client)

        for _ in range(cfg.sim_steps):
            p.stepSimulation(physicsClientId=client)

        final_pos, final_orn = p.getBasePositionAndOrientation(body, physicsClientId=client)
        displacement = math.dist(final_pos[:2], pos[:2])
        tilt_deg = self._tilt_from_quaternion(final_orn)

        stable = displacement <= cfg.max_displacement and tilt_deg <= cfg.max_tilt_deg
        score = self._score(displacement, tilt_deg)
        return StabilityReport(stable, displacement, tilt_deg, score, used_pybullet=True)

    @staticmethod
    def _tilt_from_quaternion(quat) -> float:
        """Angle in degrees between the body's local +z and world +z."""
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        cos_tilt = float(np.clip(rot[2, 2], -1.0, 1.0))
        return math.degrees(math.acos(cos_tilt))

    # ------------------------------------------------------------- fallback
    def _evaluate_heuristic(self, env: PackingEnv, placement: Placement) -> StabilityReport:
        support = env.support_ratio(placement)
        w, d, h = placement.oriented_dims()
        aspect = h / max(min(w, d), 1)
        stable = support >= 0.6 and aspect <= 3.0
        # Map support/aspect into pseudo displacement/tilt for a consistent report
        displacement = (1.0 - support) * self.config.max_displacement * 2
        tilt_deg = max(0.0, (aspect - 1.0)) * 4.0
        score = self._score(displacement, tilt_deg)
        return StabilityReport(stable, displacement, tilt_deg, score, used_pybullet=False)

    def _score(self, displacement: float, tilt_deg: float) -> float:
        cfg = self.config
        disp_term = max(0.0, 1.0 - displacement / (2 * cfg.max_displacement))
        tilt_term = max(0.0, 1.0 - tilt_deg / (2 * cfg.max_tilt_deg))
        return float(disp_term * tilt_term)
