"""Tests for marl_packing.envs.packing_env.PackingEnv."""

from __future__ import annotations

import numpy as np
import pytest

from marl_packing.envs.packing_env import PackingConfig, PackingEnv
from marl_packing.utils.geometry import NUM_ORIENTATIONS, oriented_dims


def make_env(bin_size=(5, 5, 5), num_boxes=3, lookahead=2, seed=0, **kwargs):
    cfg = PackingConfig(bin_size=bin_size, num_boxes=num_boxes, lookahead=lookahead, seed=seed, **kwargs)
    return PackingEnv(cfg)


# --------------------------------------------------------------- encode/decode


def test_encode_decode_roundtrip_all_cells_and_orientations():
    env = make_env(bin_size=(5, 5, 5))
    for x in range(env.W):
        for y in range(env.D):
            for o in range(NUM_ORIENTATIONS):
                action = env.encode_action(x, y, o)
                assert env.decode_action(action) == (x, y, o)


def test_action_space_size():
    env = make_env(bin_size=(5, 5, 5))
    assert env.action_space_size == 5 * 5 * NUM_ORIENTATIONS


# ---------------------------------------------------------------- reset/observe


def test_reset_with_fixed_box_sequence():
    env = make_env(bin_size=(5, 5, 5), num_boxes=99)
    seq = [(2, 2, 2), (1, 1, 1)]
    obs = env.reset(box_sequence=seq)
    assert env.boxes == seq
    assert env.cursor == 0
    assert np.all(obs["heightmap"] == 0)
    assert obs["next_boxes"].shape == (env.cfg.lookahead, 3)
    assert tuple(obs["next_boxes"][0]) == seq[0]
    assert obs["remaining"][0] == len(seq)


def test_observe_lookahead_beyond_queue_is_zero_padded():
    env = make_env(bin_size=(5, 5, 5), lookahead=3)
    seq = [(1, 1, 1)]
    obs = env.reset(box_sequence=seq)
    assert tuple(obs["next_boxes"][1]) == (0, 0, 0)
    assert tuple(obs["next_boxes"][2]) == (0, 0, 0)


def test_current_box_none_after_queue_exhausted():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(1, 1, 1)])
    assert env.current_box() == (1, 1, 1)
    env.step(env.encode_action(0, 0, 0))
    assert env.current_box() is None


# --------------------------------------------------------------- valid_action_mask


def test_valid_action_mask_marks_fitting_placement_valid_and_overheight_invalid():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(2, 2, 2)])
    # Rig the heightmap: column (0,0)-(1,1) is already at height 4, so a
    # (2, 2, 2) box dropped there would reach z=6 > H=5 -- invalid. Column
    # (2,0)-(3,1) is empty, so the same box fits there -- valid.
    env.heightmap[0:2, 0:2] = 4
    mask = env.valid_action_mask()

    valid_action = env.encode_action(2, 0, 0)
    invalid_action = env.encode_action(0, 0, 0)
    assert mask[valid_action] == True  # noqa: E712
    assert mask[invalid_action] == False  # noqa: E712


def test_valid_action_mask_empty_when_no_box_left():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(1, 1, 1)])
    env.step(env.encode_action(0, 0, 0))
    mask = env.valid_action_mask()
    assert not mask.any()


def test_valid_action_mask_respects_bin_bounds():
    box = (2, 2, 1)
    env = make_env(bin_size=(3, 3, 5))
    env.reset(box_sequence=[box])
    mask = env.valid_action_mask()
    # The bin is empty, so a placement is valid iff its oriented footprint
    # fits within the (W, D) bounds at that anchor (height never binds here).
    for x in range(env.W):
        for y in range(env.D):
            for o in range(NUM_ORIENTATIONS):
                dx, dy, _ = oriented_dims(box, o)
                a = env.encode_action(x, y, o)
                fits = x + dx <= env.W and y + dy <= env.D
                assert mask[a] == fits


# --------------------------------------------------------------------- preview


def test_preview_reports_drop_height_and_support():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(2, 2, 1), (2, 2, 1)])
    env.step(env.encode_action(0, 0, 0))  # first box lands at z=0..1
    p = env.preview(env.encode_action(0, 0, 0))
    assert p["z"] == 1
    assert p["in_bounds"] is True
    assert p["support"] == 1.0


def test_preview_out_of_bounds():
    env = make_env(bin_size=(3, 3, 5))
    env.reset(box_sequence=[(2, 2, 1)])
    p = env.preview(env.encode_action(2, 2, 0))
    assert p["in_bounds"] is False


# ------------------------------------------------------------------------ step


def test_step_placed_reward_is_positive_and_updates_heightmap():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(2, 2, 2)])
    result = env.step(env.encode_action(0, 0, 0))
    assert result.reward > 0
    assert result.info["reason"] == "placed"
    assert env.heightmap[0, 0] == 2
    assert result.terminated is True


def test_step_invalid_reward_is_penalty_invalid():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(2, 2, 2)])
    # x=4 -> x+dx=6 > W=5: out of bounds.
    result = env.step(env.encode_action(4, 4, 0))
    assert result.reward == env.cfg.penalty_invalid
    assert result.info["reason"] == "invalid"
    assert env.packed_volume == 0
    assert env.cursor == 1


def test_step_vetoed_reward_is_penalty_veto():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(2, 2, 2)])
    result = env.step(env.encode_action(0, 0, 0), vetoed=True)
    assert result.reward == env.cfg.penalty_veto
    assert result.info["reason"] == "vetoed"
    assert env.veto_count == 1
    assert env.packed_volume == 0  # box did not enter the bin
    assert np.all(env.heightmap == 0)


def test_step_unstable_reward_is_lower_than_stable_and_still_placed():
    env_stable = make_env(bin_size=(5, 5, 5))
    env_stable.reset(box_sequence=[(2, 2, 2)])
    r_stable = env_stable.step(env_stable.encode_action(0, 0, 0), stable=True)

    env_unstable = make_env(bin_size=(5, 5, 5))
    env_unstable.reset(box_sequence=[(2, 2, 2)])
    r_unstable = env_unstable.step(env_unstable.encode_action(0, 0, 0), stable=False)

    assert r_unstable.info["reason"] == "unstable"
    assert r_unstable.reward < r_stable.reward
    # The box is still committed to the bin even though marked unstable.
    assert env_unstable.packed_volume == env_stable.packed_volume


def test_step_after_queue_exhausted_is_a_noop_done():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(1, 1, 1)])
    env.step(env.encode_action(0, 0, 0))
    result = env.step(env.encode_action(0, 0, 0))
    assert result.reward == 0.0
    assert result.terminated is True
    assert result.info["reason"] == "done"


# --------------------------------------------------------------------- fill_ratio


def test_fill_ratio_matches_packed_volume_fraction():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(2, 2, 2), (1, 1, 1)])
    env.step(env.encode_action(0, 0, 0))
    env.step(env.encode_action(2, 2, 0))
    expected = (8 + 1) / float(5 * 5 * 5)
    assert env.fill_ratio() == pytest.approx(expected)


def test_fill_ratio_zero_when_nothing_placed():
    env = make_env(bin_size=(5, 5, 5))
    env.reset(box_sequence=[(2, 2, 2)])
    assert env.fill_ratio() == 0.0


# --------------------------------------------------------------------- determinism


def test_determinism_with_fixed_box_sequence_and_action_sequence():
    seq = [(2, 2, 2), (1, 3, 1), (2, 1, 3)]

    def run():
        env = make_env(bin_size=(5, 5, 5))
        env.reset(box_sequence=seq)
        heightmaps = []
        while env.current_box() is not None:
            mask = env.valid_action_mask()
            action = int(np.argmax(mask))  # deterministic pick
            env.step(action)
            heightmaps.append(env.heightmap.copy())
        return heightmaps, env.fill_ratio()

    hm1, fr1 = run()
    hm2, fr2 = run()
    assert fr1 == fr2
    for a, b in zip(hm1, hm2):
        assert np.array_equal(a, b)
