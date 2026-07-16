"""PyBullet-backed drop simulation and stability measurement.

Builds a headless (DIRECT-mode) PyBullet world containing the boxes already
committed to the bin, drops a candidate box at its target (x, y) footprint,
and steps physics until every body settles (or a timeout is hit). A drop is
judged "stable" iff the world settles within the timeout AND no box (the new
one, or any existing one it disturbs) tilts or shifts past the configured
thresholds.

One DropSimulator owns one PyBullet DIRECT client, reused across an episode's
many drop checks (rebuilding the scene from scratch each call is far cheaper
than paying PyBullet's connect/disconnect overhead per drop).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pybullet as p

# Rough cardboard/mixed-goods density used to convert a box's volume into a
# plausible mass for the physics sim. Not calibrated against real materials --
# only the box's own weight distribution vs. friction/restitution matters for
# whether a stack topples, not the absolute mass scale.
_DENSITY_KG_PER_M3 = 300.0

# Drop the candidate from slightly above its resting height so contact/impact
# dynamics are exercised rather than spawning it already touching the stack.
_DROP_HEIGHT_BONUS_M = 0.02


@dataclasses.dataclass
class PlacedBox:
    position: np.ndarray  # (3,) center position (x, y, z) in the bin's local frame, meters
    dims: np.ndarray  # (3,) full extents (w, d, h), meters


@dataclasses.dataclass
class StabilityResult:
    stable: bool
    settled: bool
    max_tilt_deg: float
    max_displacement_m: float
    final_position: np.ndarray


class DropSimulator:
    def __init__(self, config: dict):
        self.config = config
        self._client = p.connect(p.DIRECT)
        self._sim_hz = config["sim_hz"]

    def close(self) -> None:
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _reset_world(self) -> None:
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, self.config["gravity"], physicsClientId=self._client)
        p.setTimeStep(1.0 / self._sim_hz, physicsClientId=self._client)
        floor_half_extents = [5.0, 5.0, 0.05]
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=floor_half_extents, physicsClientId=self._client
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_shape,
            basePosition=[0, 0, -0.05],
            physicsClientId=self._client,
        )

    def _spawn_box(self, box: PlacedBox, mass: float, friction: float, restitution: float) -> int:
        half_extents = (box.dims / 2.0).tolist()
        col_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self._client
        )
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            basePosition=box.position.tolist(),
            physicsClientId=self._client,
        )
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=friction,
            restitution=restitution,
            physicsClientId=self._client,
        )
        return body_id

    def simulate_drop(
        self,
        existing_boxes: list[PlacedBox],
        candidate: PlacedBox,
        rng: np.random.Generator,
        randomization: dict,
    ) -> StabilityResult:
        """Rebuild the world from `existing_boxes` (settled ground-truth positions) and drop
        `candidate` at its target footprint. Read-only w.r.t. `existing_boxes`/`candidate` --
        callers decide whether to commit the outcome."""
        self._reset_world()
        friction = float(rng.uniform(*randomization["friction_range"]))
        restitution = float(rng.uniform(*randomization["restitution_range"]))
        mass_scale = float(rng.uniform(*randomization["mass_scale_range"]))
        pose_noise = randomization["pose_noise_m"]

        existing_body_ids = []
        existing_pre_positions = []
        for box in existing_boxes:
            noisy_pos = box.position + (rng.normal(0, pose_noise, size=3) if pose_noise > 0 else 0.0)
            mass = float(np.prod(box.dims) * _DENSITY_KG_PER_M3 * mass_scale)
            body_id = self._spawn_box(
                PlacedBox(position=noisy_pos, dims=box.dims), mass=mass, friction=friction, restitution=restitution
            )
            existing_body_ids.append(body_id)
            existing_pre_positions.append(noisy_pos)

        drop_pos = candidate.position.copy()
        drop_pos[2] += _DROP_HEIGHT_BONUS_M
        candidate_mass = float(np.prod(candidate.dims) * _DENSITY_KG_PER_M3 * mass_scale)
        candidate_body_id = self._spawn_box(
            PlacedBox(position=drop_pos, dims=candidate.dims), mass=candidate_mass, friction=friction, restitution=restitution
        )

        settled = self._step_until_settled([*existing_body_ids, candidate_body_id])

        max_tilt = 0.0
        max_disp = 0.0
        for body_id, pre_pos in zip(existing_body_ids, existing_pre_positions):
            pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self._client)
            max_tilt = max(max_tilt, _orientation_tilt_deg(orn))
            max_disp = max(max_disp, float(np.linalg.norm(np.array(pos) - pre_pos)))

        cand_pos, cand_orn = p.getBasePositionAndOrientation(candidate_body_id, physicsClientId=self._client)
        max_tilt = max(max_tilt, _orientation_tilt_deg(cand_orn))

        stable = (
            settled
            and max_tilt <= self.config["tilt_fail_deg"]
            and max_disp <= self.config["displacement_fail_m"]
        )
        return StabilityResult(
            stable=stable,
            settled=settled,
            max_tilt_deg=max_tilt,
            max_displacement_m=max_disp,
            final_position=np.array(cand_pos),
        )

    def _step_until_settled(self, body_ids: list[int]) -> bool:
        lin_thresh = self.config["settle_lin_vel_thresh"]
        ang_thresh = self.config["settle_ang_vel_thresh"]
        hold_needed = self.config["settle_hold_steps"]
        timeout = self.config["settle_timeout_steps"]

        hold_count = 0
        for _ in range(timeout):
            p.stepSimulation(physicsClientId=self._client)
            all_slow = True
            for body_id in body_ids:
                lin_vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=self._client)
                if np.linalg.norm(lin_vel) > lin_thresh or np.linalg.norm(ang_vel) > ang_thresh:
                    all_slow = False
                    break
            hold_count = hold_count + 1 if all_slow else 0
            if hold_count >= hold_needed:
                return True
        return False


def _orientation_tilt_deg(orn_quat) -> float:
    """Angle (degrees) between the box's local +Z axis and world +Z -- i.e. how far it has
    tipped from upright."""
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn_quat)).reshape(3, 3)
    local_z_world = rot_matrix @ np.array([0.0, 0.0, 1.0])
    cos_angle = float(np.clip(local_z_world[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))
