"""Tests for marl_packing.training.ppo_trainer.MARLTrainer."""

from __future__ import annotations

import numpy as np

from marl_packing.agents.physics_agent import PhysicsAgent
from marl_packing.agents.proposer_agent import ProposerAgent
from marl_packing.envs.packing_env import PackingConfig, PackingEnv
from marl_packing.training.ppo_trainer import EpisodeStats, MARLTrainer, RolloutBuffer, TrainerConfig


def build(episodes=3, seed=0):
    env_cfg = PackingConfig(bin_size=(5, 5, 5), num_boxes=4, min_box=1, max_box=2, lookahead=2, seed=seed)
    env = PackingEnv(env_cfg)
    proposer = ProposerAgent(env, hidden=16, seed=seed)
    physics = PhysicsAgent(seed=seed, lr=0.05)
    trainer_cfg = TrainerConfig(
        episodes=episodes, update_epochs=1, minibatch_size=8, log_every=10_000, seed=seed
    )
    trainer = MARLTrainer(env, proposer, physics, trainer_cfg)
    return trainer


def test_run_episode_returns_stats_and_fills_buffer():
    trainer = build(episodes=1)
    buffer = RolloutBuffer()
    stats = trainer.run_episode(buffer)
    assert isinstance(stats, EpisodeStats)
    assert 0.0 <= stats.fill_ratio <= 1.0
    assert stats.vetoes >= 0
    assert np.isfinite(stats.physics_loss)
    assert len(buffer.rewards) > 0
    assert len(buffer.obs) == len(buffer.actions) == len(buffer.rewards)


def test_run_episode_greedy_is_deterministic_in_action_choice():
    trainer1 = build(episodes=1, seed=7)
    trainer2 = build(episodes=1, seed=7)
    s1 = trainer1.run_episode(greedy=True)
    s2 = trainer2.run_episode(greedy=True)
    # Same seeds & config -> same box sequence & same greedy policy -> same fill.
    assert s1.fill_ratio == s2.fill_ratio


def test_train_runs_without_error_and_history_has_right_length():
    trainer = build(episodes=4, seed=1)
    history = trainer.train()
    assert len(history) == 4
    assert trainer.history is history
    for i, stats in enumerate(history):
        assert stats.episode == i
        assert np.isfinite(stats.reward)
        assert 0.0 <= stats.fill_ratio <= 1.0


def test_train_calls_callback_once_per_episode():
    trainer = build(episodes=3, seed=2)
    seen = []
    trainer.train(callback=lambda s: seen.append(s.episode))
    assert seen == [0, 1, 2]


def test_ppo_update_does_not_crash_on_small_batch():
    trainer = build(episodes=1, seed=3)
    buffer = RolloutBuffer()
    trainer.run_episode(buffer)
    # Should run without raising, and should change the policy's weights.
    w_before = [w.copy() for w in trainer.proposer.policy.W]
    trainer.ppo_update(buffer)
    w_after = trainer.proposer.policy.W
    changed = any(not np.allclose(a, b) for a, b in zip(w_before, w_after))
    assert changed
