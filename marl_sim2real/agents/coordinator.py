"""MARL coordinator: runs the cooperative proposer/physics loop.

Protocol per item:
  1. Proposer samples a placement action from its policy.
  2. Physics agent drops the item in simulation and reports stability.
  3. If stable, the placement is committed and both the density gain and the
     stability score become reward. If unstable, the item is rejected and the
     proposer receives a penalty — the two agents share one objective, making
     this a cooperative MARL setup.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from marl_sim2real.agents.physics_agent import PhysicsAgent
from marl_sim2real.agents.proposer_agent import ProposerAgent
from marl_sim2real.envs.packing_env import PackingEnv


@dataclasses.dataclass
class EpisodeStats:
    density: float
    placed: int
    rejected: int
    mean_stability: float
    loss: float
    diverted: int = 0  # items that fit no bin at all (manual-handling line)


class MARLCoordinator:
    STABILITY_WEIGHT = 0.5
    DENSITY_WEIGHT = 10.0
    REJECT_PENALTY = -0.5
    CRUSH_PENALTY = -0.75  # crushing cargo is worse than a rejected proposal
    LIFO_PENALTY = -0.6    # burying an earlier-stop item means restacking at the dock

    def __init__(self, env: PackingEnv, proposer: ProposerAgent, physics: PhysicsAgent):
        self.env = env
        self.proposer = proposer
        self.physics = physics

    def run_episode(self, train: bool = True) -> EpisodeStats:
        obs = self.env.reset()
        placed = rejected = diverted = 0
        stability_scores: list[float] = []

        while not self.env.done():
            mask = ProposerAgent.feasible_mask(self.env)
            if not mask.any():
                # Multi-bin cells can open a fresh pallet and retry the item.
                roll = getattr(self.env, "roll_bin_if_useful", None)
                if roll is not None and roll():
                    obs = self.env.observe()
                    continue
                divert = getattr(self.env, "divert_item", self.env.skip_item)
                divert()
                diverted += 1
                obs = self.env.observe()
                continue

            action = self.proposer.act(obs, mask)
            placement = self.env.try_place(action)
            assert placement is not None  # masked sampling guarantees feasibility

            report = self.physics.evaluate(self.env, placement)
            # Load-bearing veto (ConstrainedPackingEnv): stable but crushing
            # placements are rejected with a stiffer penalty.
            check_crush = getattr(self.env, "check_crush", None)
            if report.stable and check_crush is not None and not check_crush(placement).ok:
                self.proposer.record_reward(self.CRUSH_PENALTY)
                self.env.skip_item()
                rejected += 1
                obs = self.env.observe()
                continue
            # Unload-order veto (DeliveryPackingEnv): stable but burying an
            # item bound for an earlier delivery stop.
            check_order = getattr(self.env, "check_unload_order", None)
            if report.stable and check_order is not None and not check_order(placement).ok:
                self.proposer.record_reward(self.LIFO_PENALTY)
                self.env.skip_item()
                rejected += 1
                obs = self.env.observe()
                continue
            if report.stable:
                volume_gain = float(np.prod(placement.oriented_dims()))
                volume_gain /= float(np.prod(self.env.config.bin_size))
                reward = (
                    self.DENSITY_WEIGHT * volume_gain
                    + self.STABILITY_WEIGHT * report.score
                )
                self.env.commit(placement)
                placed += 1
                stability_scores.append(report.score)
            else:
                reward = self.REJECT_PENALTY
                self.env.skip_item()
                rejected += 1

            self.proposer.record_reward(reward)
            obs = self.env.observe()

        loss = self.proposer.finish_episode() if train else 0.0
        if not train:
            self.proposer._episode = []
        overall = getattr(self.env, "overall_density", self.env.packing_density)
        return EpisodeStats(
            density=float(overall()),
            placed=placed,
            rejected=rejected,
            mean_stability=float(np.mean(stability_scores)) if stability_scores else 0.0,
            loss=loss,
            diverted=diverted,
        )

    def train(self, episodes: int, log_every: int = 10) -> list[EpisodeStats]:
        history = []
        for ep in range(episodes):
            stats = self.run_episode(train=True)
            history.append(stats)
            if log_every and (ep + 1) % log_every == 0:
                recent = history[-log_every:]
                print(
                    f"episode {ep + 1:4d} | "
                    f"density {np.mean([s.density for s in recent]):.3f} | "
                    f"placed {np.mean([s.placed for s in recent]):.1f} | "
                    f"rejected {np.mean([s.rejected for s in recent]):.1f} | "
                    f"stability {np.mean([s.mean_stability for s in recent]):.3f}"
                )
        return history
