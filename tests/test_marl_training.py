from marl_sim2real.agents import MARLCoordinator, PhysicsAgent, ProposerAgent
from marl_sim2real.config import PackingConfig, TrainConfig
from marl_sim2real.envs import PackingEnv


def test_episode_runs_and_learns_shape():
    cfg = PackingConfig(bin_size=(5, 5, 6), max_items=3)
    env = PackingEnv(cfg, seed=1)
    proposer = ProposerAgent(env, TrainConfig(seed=1))
    physics = PhysicsAgent(use_pybullet=False)  # fast heuristic for CI
    coordinator = MARLCoordinator(env, proposer, physics)

    stats = coordinator.run_episode(train=True)
    assert stats.placed + stats.rejected <= cfg.max_items
    assert 0.0 <= stats.density <= 1.0


def test_training_loop_multiple_episodes():
    cfg = PackingConfig(bin_size=(5, 5, 6), max_items=3)
    env = PackingEnv(cfg, seed=2)
    proposer = ProposerAgent(env, TrainConfig(seed=2))
    physics = PhysicsAgent(use_pybullet=False)
    coordinator = MARLCoordinator(env, proposer, physics)

    history = coordinator.train(episodes=5, log_every=0)
    assert len(history) == 5
    # The policy buffer must be flushed between episodes
    assert proposer._episode == []


def test_feasible_mask_only_valid_actions():
    cfg = PackingConfig(bin_size=(5, 5, 6), max_items=3)
    env = PackingEnv(cfg, seed=3)
    mask = ProposerAgent.feasible_mask(env)
    assert mask.any()
    for action in range(env.action_space_size()):
        assert (env.try_place(action) is not None) == bool(mask[action])
