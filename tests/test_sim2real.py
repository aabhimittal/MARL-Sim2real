"""Tests for marl_packing.sim2real (DomainRandomizer, Sim2RealBridge, MockRobot,
RealityGapCalibrator)."""

from __future__ import annotations

import numpy as np

from marl_packing.agents.physics_agent import PhysicsAgent
from marl_packing.agents.proposer_agent import ProposerAgent
from marl_packing.envs.packing_env import PackingConfig, PackingEnv
from marl_packing.sim2real.bridge import MockRobot, Sim2RealBridge
from marl_packing.sim2real.domain_randomization import DomainRandomizer, RandomizationConfig
from marl_packing.sim2real.reality_gap import RealityGapCalibrator


BIN = (5, 5, 5)


# ------------------------------------------------------------- RandomizationConfig


def test_scaled_scales_all_magnitude_fields():
    cfg = RandomizationConfig(
        dim_tolerance=0.1, placement_jitter=0.2, obs_noise_std=0.3, support_shift_std=0.4, seed=5
    )
    scaled = cfg.scaled(2.0)
    assert scaled.dim_tolerance == 0.2
    assert scaled.placement_jitter == 0.4
    assert scaled.obs_noise_std == 0.6
    assert scaled.support_shift_std == 0.8
    # Original is untouched (dataclasses.replace makes a copy).
    assert cfg.dim_tolerance == 0.1
    assert scaled.seed == 5


# ------------------------------------------------------------------ DomainRandomizer


def test_domain_randomizer_reset_and_step_preserve_shapes_and_bounds():
    env = PackingEnv(PackingConfig(bin_size=BIN, num_boxes=6, min_box=1, max_box=2, seed=0))
    randomizer = DomainRandomizer(env, RandomizationConfig(seed=1))
    obs = randomizer.reset()
    assert obs["heightmap"].shape == (env.W, env.D)
    assert isinstance(randomizer.support_shift, float)

    steps = 0
    while env.current_box() is not None and steps < 20:
        mask = randomizer.valid_action_mask()
        if not mask.any():
            break
        action = int(np.argmax(mask))
        result = randomizer.step(action)
        assert result.obs["heightmap"].shape == (env.W, env.D)
        # Noisy observation stays within the physical height bounds.
        assert np.all(result.obs["heightmap"] >= 0)
        assert np.all(result.obs["heightmap"] <= env.H)
        # The underlying (non-noisy) heightmap never exceeds the bin's walls.
        assert np.all(env.heightmap >= 0)
        assert np.all(env.heightmap <= env.H)
        steps += 1
    assert steps > 0


def test_domain_randomizer_delegates_unknown_attrs_to_env():
    env = PackingEnv(PackingConfig(bin_size=BIN, num_boxes=3, seed=0))
    randomizer = DomainRandomizer(env)
    assert randomizer.W == env.W
    assert randomizer.action_space_size == env.action_space_size


def test_domain_randomizer_perturbed_boxes_stay_at_least_one():
    env = PackingEnv(PackingConfig(bin_size=BIN, num_boxes=10, min_box=2, max_box=3, seed=0))
    randomizer = DomainRandomizer(env, RandomizationConfig(dim_tolerance=1.0, seed=2))
    randomizer.reset()
    for dims in env.boxes:
        assert all(d >= 1 for d in dims)


# ------------------------------------------------------------------------ MockRobot


def test_mock_robot_basic_interface():
    robot = MockRobot(PackingConfig(bin_size=BIN, num_boxes=3, min_box=1, max_box=2, seed=0), seed=1)
    hm = robot.read_heightmap()
    assert hm.shape == (5, 5)
    box = robot.next_box()
    assert box is not None and len(box) == 3
    outcome = robot.execute_placement(0, 0, 0)
    assert "success" in outcome and "stable" in outcome


# -------------------------------------------------------------------- Sim2RealBridge


def _build_bridge(seed=0, support_threshold=0.5):
    env_cfg = PackingConfig(bin_size=BIN, num_boxes=6, min_box=1, max_box=2, seed=seed)
    proposer_env = PackingEnv(env_cfg)
    proposer = ProposerAgent(proposer_env, hidden=16, seed=seed)
    physics = PhysicsAgent(seed=seed, lr=0.05)
    robot = MockRobot(
        PackingConfig(bin_size=BIN, num_boxes=6, min_box=1, max_box=2, seed=seed + 1),
        support_threshold=support_threshold,
        sensor_noise=0.0,
        seed=seed + 2,
    )
    calibrator = RealityGapCalibrator()
    bridge = Sim2RealBridge(proposer, physics, robot, calibrator, max_retries=3)
    return bridge, calibrator, physics


def test_bridge_full_run_produces_sane_deployment_stats():
    bridge, calibrator, _ = _build_bridge()
    stats = bridge.run(max_boxes=6)
    assert stats.boxes_attempted >= stats.boxes_placed
    assert stats.boxes_placed >= 0
    assert stats.unstable_placements <= stats.boxes_placed
    assert 0.0 <= stats.fill_ratio <= 1.0


def test_bridge_run_collects_calibrator_samples():
    bridge, calibrator, _ = _build_bridge()
    bridge.run(max_boxes=6)
    assert len(calibrator.samples) > 0
    for sample in calibrator.samples:
        assert sample.features.shape == (PhysicsAgent.FEATURE_DIM,)
        assert isinstance(sample.sim_stable, (bool, np.bool_))
        assert isinstance(sample.real_stable, (bool, np.bool_))


def test_calibrator_report_fields_are_sane():
    bridge, calibrator, _ = _build_bridge()
    bridge.run(max_boxes=6)
    report = calibrator.report()
    assert report.n_samples == len(calibrator.samples)
    assert 0.0 <= report.disagreement_rate <= 1.0
    assert 0.0 <= report.sim_stable_rate <= 1.0
    assert 0.0 <= report.real_stable_rate <= 1.0
    assert 0.0 <= report.calibration_error <= 1.0
    assert isinstance(report.suggested_config, RandomizationConfig)


def test_calibrator_report_empty_is_well_defined():
    calibrator = RealityGapCalibrator()
    report = calibrator.report()
    assert report.n_samples == 0
    assert report.disagreement_rate == 0.0
    assert report.suggested_config is calibrator.base


def test_finetune_physics_agent_returns_finite_loss():
    bridge, calibrator, physics = _build_bridge()
    bridge.run(max_boxes=6)
    loss = calibrator.finetune_physics_agent(physics, epochs=3)
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_finetune_physics_agent_no_samples_returns_zero():
    calibrator = RealityGapCalibrator()
    physics = PhysicsAgent(seed=0)
    assert calibrator.finetune_physics_agent(physics, epochs=3) == 0.0


def test_suggested_config_widens_with_disagreement():
    calibrator = RealityGapCalibrator(RandomizationConfig(dim_tolerance=0.1))
    # Perfect agreement -> factor 1.0 -> config unchanged in magnitude.
    for _ in range(5):
        calibrator.record(np.zeros(PhysicsAgent.FEATURE_DIM), sim_stable=True, real_stable=True, confidence=0.9)
    report = calibrator.report()
    assert report.suggested_config.dim_tolerance == calibrator.base.dim_tolerance

    calibrator2 = RealityGapCalibrator(RandomizationConfig(dim_tolerance=0.1))
    for _ in range(5):
        calibrator2.record(np.zeros(PhysicsAgent.FEATURE_DIM), sim_stable=True, real_stable=False, confidence=0.9)
    report2 = calibrator2.report()
    assert report2.suggested_config.dim_tolerance > calibrator2.base.dim_tolerance
