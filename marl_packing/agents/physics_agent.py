"""Physics Agent: judges whether a proposed placement is statically stable.

Three cooperating layers, cheapest first:

1. **Analytic checks** (always on): support ratio, center-of-mass over the
   support region, height-to-base slenderness. Fast necessary conditions.
2. **Learned critic** (always on): a small MLP trained online on simulation
   outcomes; it generalizes beyond the hand-written rules and is the piece
   that transfers through the Sim2Real bridge (it can be fine-tuned on real
   robot outcomes).
3. **PyBullet backend** (optional): full rigid-body settle test. Used when
   pybullet is installed and `use_pybullet=True`; otherwise a deterministic
   analytic settle model stands in.

The verdict feeds back into the environment as a veto and into the
proposer's reward, making training an adversarial-cooperative MARL loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from marl_packing.agents.networks import MLP
from marl_packing.utils.geometry import support_polygon_contains_com


@dataclass
class StabilityVerdict:
    stable: bool
    veto: bool
    confidence: float  # critic probability of stability, [0, 1]
    features: np.ndarray
    reason: str


class PhysicsAgent:
    FEATURE_DIM = 7

    def __init__(
        self,
        veto_threshold: float = 0.25,
        use_pybullet: bool = False,
        seed: Optional[int] = None,
        lr: float = 1e-2,
    ):
        self.veto_threshold = veto_threshold
        self.critic = MLP([self.FEATURE_DIM, 32, 32, 1], seed=seed, lr=lr)
        self._bullet = None
        if use_pybullet:
            try:
                import pybullet  # noqa: F401

                self._bullet = pybullet
            except ImportError:
                pass  # graceful fallback to the analytic settle model

    # ---------------------------------------------------------------- judge

    def features(self, placement: Dict) -> np.ndarray:
        """Physics-relevant scalar features of a previewed placement."""
        dx, dy, dz = placement["dims"]
        z, support = placement["z"], placement["support"]
        hm = placement["heightmap"]
        com_ok = support_polygon_contains_com(
            hm, placement["x"], placement["y"], dx, dy, z
        )
        slenderness = dz / max(min(dx, dy), 1)
        return np.array(
            [
                support,                       # fraction of footprint supported
                float(com_ok),                 # COM over support region
                slenderness / 4.0,             # tall-and-thin risk, normalized
                z / max(hm.shape[0], 1),       # stack height at placement
                dx * dy / float(hm.size),      # footprint fraction of bin
                float(z == 0),                 # resting on the floor
                1.0,                           # bias
            ],
            dtype=np.float64,
        )

    def analytic_stable(self, placement: Dict) -> bool:
        """Hand-written necessary conditions for static stability."""
        dx, dy, dz = placement["dims"]
        if placement["z"] == 0:
            return True
        if placement["support"] < 0.3:
            return False
        if not support_polygon_contains_com(
            placement["heightmap"], placement["x"], placement["y"], dx, dy, placement["z"]
        ):
            return False
        if dz / max(min(dx, dy), 1) > 3.0 and placement["support"] < 0.8:
            return False
        return True

    def simulate(self, placement: Dict) -> bool:
        """Ground-truth settle test: PyBullet if available, analytic otherwise."""
        if self._bullet is not None:
            return self._pybullet_settle(placement)
        return self.analytic_stable(placement)

    def judge(self, placement: Dict) -> StabilityVerdict:
        feats = self.features(placement)
        logit = float(self.critic.forward(feats)[0, 0])
        confidence = 1.0 / (1.0 + np.exp(-logit))
        analytic = self.analytic_stable(placement)
        # Veto when both the rules and the learned critic distrust it.
        veto = (not analytic) and confidence < self.veto_threshold
        stable = analytic
        reason = "stable" if stable else ("vetoed" if veto else "risky")
        return StabilityVerdict(stable, veto, confidence, feats, reason)

    # ---------------------------------------------------------------- learn

    def learn(self, feats: np.ndarray, was_stable: bool) -> float:
        """Online logistic-regression update of the critic on an outcome.

        Returns the binary cross-entropy loss before the update.
        """
        logit = self.critic.forward(feats)  # shape (1, 1)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = float(was_stable)
        loss = float(-(y * np.log(p[0, 0] + 1e-9) + (1 - y) * np.log(1 - p[0, 0] + 1e-9)))
        self.critic.backward(p - y)  # d(BCE)/d(logit) = p - y
        return loss

    # ------------------------------------------------------------- backends

    def _pybullet_settle(self, placement: Dict) -> bool:
        """Drop the box in PyBullet DIRECT mode and check displacement."""
        p = self._bullet
        cid = p.connect(p.DIRECT)
        try:
            p.setGravity(0, 0, -9.81, physicsClientId=cid)
            p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
            hm = placement["heightmap"]
            # Rebuild the existing pile as static boxes from the heightmap.
            for i in range(hm.shape[0]):
                for j in range(hm.shape[1]):
                    h = int(hm[i, j])
                    if h > 0:
                        col = p.createCollisionShape(
                            p.GEOM_BOX, halfExtents=[0.5, 0.5, h / 2.0],
                            physicsClientId=cid,
                        )
                        p.createMultiBody(
                            0, col, basePosition=[i + 0.5, j + 0.5, h / 2.0],
                            physicsClientId=cid,
                        )
            dx, dy, dz = placement["dims"]
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[dx / 2.0, dy / 2.0, dz / 2.0],
                physicsClientId=cid,
            )
            start = [
                placement["x"] + dx / 2.0,
                placement["y"] + dy / 2.0,
                placement["z"] + dz / 2.0 + 0.05,
            ]
            body = p.createMultiBody(1.0, col, basePosition=start, physicsClientId=cid)
            for _ in range(240):
                p.stepSimulation(physicsClientId=cid)
            pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=cid)
            drift = np.linalg.norm(np.array(pos[:2]) - np.array(start[:2]))
            tilt = 2 * np.arccos(min(abs(orn[3]), 1.0))
            return bool(drift < 0.3 and tilt < 0.2)
        finally:
            p.disconnect(cid)

    # ---------------------------------------------------------- persistence

    def save(self, path: str) -> None:
        np.savez(path, **self.critic.state_dict())

    def load(self, path: str) -> None:
        self.critic.load_state_dict(dict(np.load(path)))
