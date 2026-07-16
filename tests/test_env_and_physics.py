import numpy as np
import pytest

from marl_sim2real.env import BinPackingEnv, PhysicsParams, StabilityEngine
from marl_sim2real.env.bin_packing_env import Item


def test_orientations_cover_all_permutations():
    item = Item(dims=(1, 2, 3))
    seen = {item.oriented(o) for o in range(6)}
    assert len(seen) == 6


def test_flat_placement_is_stable():
    engine = StabilityEngine(PhysicsParams())
    hm = np.zeros((8, 8), dtype=np.int64)
    report = engine.evaluate(hm, 0, 0, (2, 2, 2), mass=1.0)
    assert report.stable
    assert report.support_ratio == 1.0
    assert report.details["rest_z"] == 0


def test_overhang_placement_is_unstable():
    engine = StabilityEngine(PhysicsParams())
    hm = np.zeros((8, 8), dtype=np.int64)
    hm[0:1, 0:1] = 3  # a single tall pillar
    # 3x3 box resting on the pillar: only 1/9 of base supported
    report = engine.evaluate(hm, 0, 0, (3, 3, 2), mass=1.0)
    assert not report.stable
    assert report.support_ratio == pytest.approx(1 / 9)


def test_env_step_places_item_and_updates_heightmap():
    env = BinPackingEnv(bin_size=(6, 6, 6), seed=3)
    env.reset()
    env.current_item = Item(dims=(2, 2, 2))
    result = env.step((0, 0, 0))
    assert result.info["placed"]
    assert env.heightmap[0:2, 0:2].min() == 2
    assert result.reward > 0
    assert 0 < result.info["utilization"] <= 1


def test_env_rejects_out_of_bounds():
    env = BinPackingEnv(bin_size=(6, 6, 6), seed=3)
    env.reset()
    env.current_item = Item(dims=(4, 4, 4))
    result = env.step((4, 4, 0))  # 4+4 > 6
    assert not result.info["placed"]
    assert result.reward < 0


def test_valid_actions_mask_matches_flat_indexing():
    from marl_sim2real.agents import flat_to_action

    env = BinPackingEnv(bin_size=(5, 5, 5), seed=1)
    env.reset()
    env.current_item = Item(dims=(2, 3, 1))
    mask = env.valid_actions_mask()
    for flat in np.flatnonzero(mask):
        x, y, o = flat_to_action(int(flat), env.L, env.W)
        l, w, h = env.current_item.oriented(o)
        assert x + l <= env.L and y + w <= env.W


def test_sensor_noise_changes_observation():
    noisy = BinPackingEnv(bin_size=(6, 6, 6), physics=PhysicsParams(sensor_noise=0.3), seed=5)
    clean = BinPackingEnv(bin_size=(6, 6, 6), physics=PhysicsParams(sensor_noise=0.0), seed=5)
    o1, o2 = noisy.reset(), clean.reset()
    assert not np.allclose(o1[:36], o2[:36])
