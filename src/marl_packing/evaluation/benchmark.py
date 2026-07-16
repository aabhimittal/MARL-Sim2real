"""Runs a trained (packer, physics) model pair through N deterministic benchmark episodes
under a given domain-randomization mode, and aggregates the metrics in
evaluation/metrics.py.

`randomization_mode="train"` benchmarks under the same distribution the models trained on
("sim" performance); `randomization_mode="real_proxy"` benchmarks under the wider range
standing in for real-world variation (see envs/domain_randomization.py and
docs/SIM2REAL.md). Comparing the two is how mlops/challenger.py computes the Sim2Real gap.
"""

from __future__ import annotations

import numpy as np

from marl_packing.envs.packing_env import PackingEnv
from marl_packing.evaluation.metrics import box_utilization_pct, stability_success_rate_pct


def run_benchmark(
    packer_model,
    physics_model,
    env_config: dict,
    randomization_mode: str,
    num_episodes: int,
    seed_base: int,
    reward_shaping: dict | None = None,
) -> dict:
    env = PackingEnv(env_config, randomization_mode=randomization_mode, reward_shaping=reward_shaping)
    episode_summaries: list[dict] = []
    all_outcomes: list[str] = []

    try:
        for i in range(num_episodes):
            env.reset(seed=seed_base + i)
            episode_summary = None

            while env.agents:
                agent = env.agent_selection
                if env.terminations[agent] or env.truncations[agent]:
                    env.step(None)
                    continue

                obs = env.observe(agent)
                if agent == "packer":
                    action, _ = packer_model.predict(obs, deterministic=True)
                    env.step(np.asarray(action, dtype=np.float32))
                else:
                    action, _ = physics_model.predict(obs, deterministic=True)
                    env.step(int(action))
                    outcome = env.infos["physics"].get("outcome")
                    if outcome:
                        all_outcomes.append(outcome)
                    # Capture the episode summary the instant termination fires -- PettingZoo
                    # clears an agent's `infos` once it's been drained as a "dead" agent on a
                    # later step(), so it must be read here, not after the while loop exits.
                    if env.terminations["physics"]:
                        episode_summary = env.infos["packer"].get("episode_summary")

            episode_summaries.append(
                episode_summary
                or {"utilization_pct": 0.0, "items_placed": 0, "items_total": env.num_items_per_episode}
            )
    finally:
        env.close()

    return {
        "utilization_pct": box_utilization_pct(episode_summaries),
        "stability_success_rate_pct": stability_success_rate_pct(all_outcomes),
        "num_episodes": num_episodes,
        "randomization_mode": randomization_mode,
    }
