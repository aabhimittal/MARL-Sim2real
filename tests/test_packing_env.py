import numpy as np

from marl_sim2real.config import PackingConfig
from marl_sim2real.envs import PackingEnv


def make_env():
    cfg = PackingConfig(bin_size=(6, 6, 8), max_items=4)
    return PackingEnv(cfg, seed=0)


def test_observation_shape_and_range():
    env = make_env()
    obs = env.observe()
    assert obs.shape == (6 * 6 + 3,)
    assert (obs >= 0).all() and (obs <= 1).all()


def test_gravity_placement_stacks():
    env = make_env()
    # Place first item at origin
    placement = env.try_place(0)
    assert placement is not None and placement.z == 0
    env.commit(placement)
    # A second item at the same (x, y) must rest on top of the first
    placement2 = env.try_place(0)
    if placement2 is not None:
        w, d, h = env.placements[0].oriented_dims()
        assert placement2.z == h


def test_out_of_bounds_rejected():
    env = make_env()
    # action at max x,y with a >1-cell item must be infeasible
    action = 0 * (6 * 6) + 5 * 6 + 5  # orientation 0, x=5, y=5
    item = env.current_item()
    if min(item.dims) > 1:
        assert env.try_place(action) is None


def test_density_increases_with_commits():
    env = make_env()
    assert env.packing_density() == 0.0
    placement = env.try_place(0)
    env.commit(placement)
    assert env.packing_density() > 0.0


def test_support_ratio_full_on_floor():
    env = make_env()
    placement = env.try_place(0)
    assert env.support_ratio(placement) == 1.0


def test_world_conversion_scale():
    env = make_env()
    placement = env.try_place(0)
    pos, half = env.placement_to_world(placement)
    w, d, h = placement.oriented_dims()
    assert np.isclose(half[0], w * env.config.cell_size / 2)
    assert np.isclose(pos[2], h * env.config.cell_size / 2)  # rests on floor
