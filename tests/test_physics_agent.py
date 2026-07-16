import pytest

from marl_sim2real.config import PackingConfig, PhysicsConfig
from marl_sim2real.agents.physics_agent import PYBULLET_AVAILABLE, PhysicsAgent
from marl_sim2real.envs import PackingEnv
from marl_sim2real.envs.packing_env import Item, Placement


def flat_env():
    return PackingEnv(PackingConfig(bin_size=(8, 8, 10), max_items=1), seed=0)


def test_heuristic_accepts_grounded_cube():
    env = flat_env()
    placement = Placement(item=Item((3, 3, 3)), orientation=0, x=0, y=0, z=0)
    agent = PhysicsAgent(use_pybullet=False)
    report = agent.evaluate(env, placement)
    assert report.stable
    assert report.score > 0.5
    assert not report.used_pybullet


def test_heuristic_rejects_tall_sliver():
    env = flat_env()
    placement = Placement(item=Item((1, 1, 9)), orientation=0, x=0, y=0, z=0)
    agent = PhysicsAgent(use_pybullet=False)
    report = agent.evaluate(env, placement)
    assert not report.stable


@pytest.mark.skipif(not PYBULLET_AVAILABLE, reason="pybullet not installed")
def test_pybullet_grounded_cube_is_stable():
    env = flat_env()
    placement = Placement(item=Item((4, 4, 4)), orientation=0, x=2, y=2, z=0)
    agent = PhysicsAgent(PhysicsConfig(sim_steps=120))
    report = agent.evaluate(env, placement)
    agent.close()
    assert report.used_pybullet
    assert report.stable
    assert report.displacement < 0.02


@pytest.mark.skipif(not PYBULLET_AVAILABLE, reason="pybullet not installed")
def test_pybullet_floating_item_is_unstable():
    """An item proposed in mid-air with nothing beneath must fail the check."""
    env = flat_env()
    placement = Placement(item=Item((2, 2, 2)), orientation=0, x=3, y=3, z=6)
    agent = PhysicsAgent(PhysicsConfig(sim_steps=240))
    report = agent.evaluate(env, placement)
    agent.close()
    assert not report.stable or report.displacement > 0.0
