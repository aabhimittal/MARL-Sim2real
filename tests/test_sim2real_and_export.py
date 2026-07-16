import numpy as np

from marl_sim2real.agents import PhysicsAgent, ProposerAgent
from marl_sim2real.env import BinPackingEnv, PhysicsParams
from marl_sim2real.export import export_bundle, load_bundle
from marl_sim2real.sim2real import (
    DomainRandomizer,
    RandomizationRanges,
    RealWorldCell,
    Sim2RealBridge,
    calibrate,
)


def test_domain_randomizer_samples_within_ranges():
    dr = DomainRandomizer(seed=0)
    for _ in range(20):
        p = dr.sample()
        r = dr.ranges
        assert r.friction[0] <= p.friction <= r.friction[1]
        assert r.support_threshold[0] <= p.support_threshold <= r.support_threshold[1]


def test_randomizer_apply_installs_params():
    env = BinPackingEnv(seed=0)
    dr = DomainRandomizer(seed=1)
    params = dr.apply(env)
    assert env.engine.params is params


def test_calibration_recovers_real_physics_behaviour():
    hidden = PhysicsParams(support_threshold=0.6, com_margin=0.4, friction=0.5,
                           mass_noise=0.0, sensor_noise=0.0)
    cell = RealWorldCell(bin_size=(6, 6, 6), params=hidden, seed=7)
    logs = cell.collect_logs(n_attempts=120)
    result = calibrate(logs, ranges=RandomizationRanges(), iterations=5, population=16, seed=0)
    # calibrated sim should reproduce most real outcomes
    assert result.agreement > 0.7
    assert len(result.history) == 5


def test_bridge_end_to_end_smoke():
    bridge = Sim2RealBridge(bin_size=(5, 5, 5), seed=0, acceptance_utilization=0.01)
    result = bridge.run(sim_episodes=30, adapt_episodes=20, verbose=False)
    assert 0.0 <= result.post_gap.gap <= 1.0
    assert result.calibration.agreement > 0.5
    assert "sim_training" in result.stages and "adaptation" in result.stages


def test_export_and_load_roundtrip(tmp_path):
    env = BinPackingEnv(bin_size=(5, 5, 5), seed=0)
    proposer = ProposerAgent(env.observation_size, env.num_positions * 6, hidden=16, seed=0)
    physics = PhysicsAgent(env.observation_size, hidden=8, seed=1)

    manifest_path = export_bundle(tmp_path, proposer, physics, (5, 5, 5))
    p2, f2, manifest = load_bundle(manifest_path)

    obs = env.reset()
    np.testing.assert_allclose(proposer.action_probs(obs), p2.action_probs(obs))
    assert manifest["bin_size"] == [5, 5, 5]
    assert manifest["weights_sha256"]


def test_export_detects_tampering(tmp_path):
    import pytest

    env = BinPackingEnv(bin_size=(5, 5, 5), seed=0)
    proposer = ProposerAgent(env.observation_size, env.num_positions * 6, hidden=16, seed=0)
    physics = PhysicsAgent(env.observation_size, hidden=8, seed=1)
    manifest_path = export_bundle(tmp_path, proposer, physics, (5, 5, 5))

    weights = tmp_path / "packing_policy.npz"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(manifest_path)
