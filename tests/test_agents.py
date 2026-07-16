"""Tests for marl_packing.agents (ProposerAgent, PhysicsAgent, networks)."""

from __future__ import annotations

import numpy as np

from marl_packing.agents.networks import MLP, softmax
from marl_packing.agents.physics_agent import PhysicsAgent
from marl_packing.agents.proposer_agent import ProposerAgent
from marl_packing.envs.packing_env import PackingConfig, PackingEnv


def make_env(bin_size=(5, 5, 5), num_boxes=4, lookahead=2, seed=0):
    cfg = PackingConfig(bin_size=bin_size, num_boxes=num_boxes, lookahead=lookahead, seed=seed)
    return PackingEnv(cfg)


def make_placement(x, y, dims, z, support, bin_shape=(5, 5)):
    return {
        "x": x, "y": y, "dims": dims, "z": z, "support": support,
        "heightmap": np.zeros(bin_shape, dtype=np.int32),
        "in_bounds": True,
    }


# --------------------------------------------------------------------- networks


def test_softmax_masked_zeroes_out_masked_entries():
    logits = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False, True])
    probs = softmax(logits, mask)
    assert probs[1] == 0.0
    assert abs(probs.sum() - 1.0) < 1e-9


def test_mlp_forward_shape():
    mlp = MLP([4, 8, 2], seed=0)
    out = mlp.forward(np.zeros(4))
    assert out.shape == (1, 2)


def test_mlp_state_dict_roundtrip():
    mlp = MLP([3, 5, 1], seed=0)
    x = np.array([0.1, -0.2, 0.3])
    out_before = mlp.forward(x).copy()
    state = mlp.state_dict()
    mlp2 = MLP([3, 5, 1], seed=99)  # different init
    mlp2.load_state_dict(state)
    out_after = mlp2.forward(x)
    assert np.allclose(out_before, out_after)


# ----------------------------------------------------------------- PhysicsAgent


def test_floor_placement_is_analytically_stable():
    agent = PhysicsAgent(seed=0)
    placement = make_placement(0, 0, (2, 2, 2), z=0, support=0.0)
    assert agent.analytic_stable(placement) is True


def test_low_support_overhang_fails_analytic_stability():
    agent = PhysicsAgent(seed=0)
    placement = make_placement(1, 1, (2, 2, 2), z=2, support=0.1)
    assert agent.analytic_stable(placement) is False


def test_tall_slender_low_support_fails_analytic_stability():
    agent = PhysicsAgent(seed=0)
    # dz / min(dx, dy) = 4 / 1 = 4 > 3, and support 0.5 < 0.8: unstable.
    placement = make_placement(0, 0, (1, 1, 4), z=1, support=0.5, bin_shape=(5, 5))
    placement["heightmap"][0, 0] = 1  # so support_polygon_contains_com can be True
    assert agent.analytic_stable(placement) is False


def test_full_support_at_height_is_stable():
    agent = PhysicsAgent(seed=0)
    hm = np.full((5, 5), 2, dtype=np.int32)
    placement = {
        "x": 0, "y": 0, "dims": (2, 2, 2), "z": 2, "support": 1.0,
        "heightmap": hm, "in_bounds": True,
    }
    assert agent.analytic_stable(placement) is True


def test_features_has_expected_dim_and_bias_term():
    agent = PhysicsAgent(seed=0)
    placement = make_placement(0, 0, (2, 2, 2), z=0, support=1.0)
    feats = agent.features(placement)
    assert feats.shape == (PhysicsAgent.FEATURE_DIM,)
    assert feats[-1] == 1.0  # bias


def test_learn_reduces_bce_on_repeated_identical_samples():
    agent = PhysicsAgent(seed=0, lr=0.1)
    placement = make_placement(0, 0, (2, 2, 2), z=1, support=0.9)
    feats = agent.features(placement)
    losses = [agent.learn(feats, True) for _ in range(50)]
    assert losses[-1] < losses[0]
    # Loss should trend downward, not just be noisy: compare early vs late average.
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_physics_agent_save_load_roundtrip(tmp_path):
    agent = PhysicsAgent(seed=1, lr=0.1)
    placement = make_placement(0, 0, (2, 2, 2), z=1, support=0.9)
    feats = agent.features(placement)
    for _ in range(5):
        agent.learn(feats, True)
    before = agent.critic.forward(feats).copy()

    path = str(tmp_path / "physics.npz")
    agent.save(path)

    agent2 = PhysicsAgent(seed=2, lr=0.1)  # different init
    agent2.load(path)
    after = agent2.critic.forward(feats)
    assert np.allclose(before, after)


def test_judge_returns_verdict_with_expected_fields():
    agent = PhysicsAgent(seed=0)
    placement = make_placement(0, 0, (2, 2, 2), z=0, support=0.0)
    verdict = agent.judge(placement)
    assert isinstance(verdict.stable, bool)
    assert isinstance(verdict.veto, bool)
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.reason in ("stable", "vetoed", "risky")


# ---------------------------------------------------------------- ProposerAgent


def test_proposer_act_always_respects_mask():
    env = make_env()
    agent = ProposerAgent(env, hidden=16, seed=0)
    obs = env.reset(box_sequence=[(2, 2, 2)] * 4)
    mask = env.valid_action_mask()
    assert mask.any()
    for _ in range(200):
        action, log_prob, value = agent.act(obs, mask, greedy=False)
        assert mask[action]
        assert np.isfinite(log_prob)
        assert np.isfinite(value)


def test_proposer_act_greedy_is_deterministic():
    env = make_env()
    agent = ProposerAgent(env, hidden=16, seed=0)
    obs = env.reset(box_sequence=[(2, 2, 2)] * 4)
    mask = env.valid_action_mask()
    a1, lp1, v1 = agent.act(obs, mask, greedy=True)
    a2, lp2, v2 = agent.act(obs, mask, greedy=True)
    assert a1 == a2
    assert lp1 == lp2
    assert v1 == v2


def test_proposer_save_load_roundtrip_preserves_outputs(tmp_path):
    env = make_env()
    agent = ProposerAgent(env, hidden=16, seed=0)
    obs = env.reset(box_sequence=[(2, 2, 2)] * 4)
    mask = env.valid_action_mask()
    action_before, logp_before, value_before = agent.act(obs, mask, greedy=True)

    path = str(tmp_path / "proposer.npz")
    agent.save(path)

    agent2 = ProposerAgent(env, hidden=16, seed=123)  # different init
    agent2.load(path)
    action_after, logp_after, value_after = agent2.act(obs, mask, greedy=True)

    assert action_before == action_after
    assert abs(logp_before - logp_after) < 1e-9
    assert abs(value_before - value_after) < 1e-9


def test_proposer_evaluate_matches_act_log_prob_shape():
    env = make_env()
    agent = ProposerAgent(env, hidden=16, seed=0)
    obs = env.reset(box_sequence=[(2, 2, 2)] * 4)
    mask = env.valid_action_mask()
    action, log_prob, value = agent.act(obs, mask, greedy=True)

    obs_batch = agent.encode_obs(obs)[None, :]
    log_probs, values, probs = agent.evaluate(
        obs_batch, np.array([action]), mask[None, :]
    )
    assert log_probs.shape == (1,)
    assert values.shape == (1,)
    assert abs(log_probs[0] - log_prob) < 1e-9
