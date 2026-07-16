"""Tests for PackingEnv (PettingZoo AECEnv) against the real configs/env.yaml config."""

from __future__ import annotations

import pytest
from pettingzoo.test import api_test

from marl_packing.envs.packing_env import PackingEnv
from marl_packing.utils.config import load_config

_VALID_OUTCOMES = {
    "invalid_height",
    "accepted_stable",
    "accepted_unstable",
    "rejected_would_have_been_stable",
    "rejected_correctly",
}


@pytest.fixture
def env():
    config = load_config("configs/env.yaml")
    e = PackingEnv(config, randomization_mode="train")
    try:
        yield e
    finally:
        e.close()


def test_pettingzoo_api_compliance(env):
    # PettingZoo's own compliance checker -- should pass with only warnings, no exceptions.
    api_test(env, num_cycles=50, verbose_progress=False)


def test_reset_initial_state(env):
    env.reset(seed=1)
    assert env.agent_selection == "packer"
    assert env.agents == ["packer", "physics"]


def test_full_episode_random_actions(env):
    config = load_config("configs/env.yaml")
    num_items = int(config["num_items_per_episode"])
    max_steps = 2 * num_items + 10  # each item takes 2 agent-steps (packer, physics) + buffer

    env.reset(seed=1)

    episode_summary = None
    physics_outcomes = []
    steps = 0

    while env.agents:
        agent = env.agent_selection

        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue

        obs = env.observe(agent)
        assert obs is not None

        action = env.action_space(agent).sample()
        env.step(action)
        steps += 1

        if agent == "physics":
            outcome = env.infos["physics"].get("outcome")
            if outcome is not None:
                physics_outcomes.append(outcome)

            # Capture the episode summary right when physics first terminates, since
            # infos get cleared on later dead-agent steps (a known AEC quirk).
            if env.terminations["physics"] and episode_summary is None:
                episode_summary = env.infos["packer"].get("episode_summary")

        assert steps <= max_steps, f"episode did not terminate within {max_steps} steps"

    assert episode_summary is not None, "expected an episode_summary to appear in packer infos"
    assert set(episode_summary.keys()) >= {"utilization_pct", "items_placed", "items_total"}
    assert episode_summary["items_total"] == num_items

    assert physics_outcomes, "expected at least one physics outcome to be recorded"
    for outcome in physics_outcomes:
        assert outcome in _VALID_OUTCOMES
