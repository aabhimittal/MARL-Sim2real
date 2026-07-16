"""Tests for marl_packing.utils.geometry."""

from __future__ import annotations

import numpy as np

from marl_packing.utils.geometry import (
    NUM_ORIENTATIONS,
    box_volume,
    drop_height,
    oriented_dims,
    support_polygon_contains_com,
    support_ratio,
    unique_orientations,
)


def test_num_orientations_is_six():
    assert NUM_ORIENTATIONS == 6


def test_oriented_dims_is_a_permutation_of_input():
    dims = (2, 3, 4)
    for o in range(NUM_ORIENTATIONS):
        out = oriented_dims(dims, o)
        assert sorted(out) == sorted(dims)


def test_oriented_dims_identity_orientation():
    dims = (2, 3, 4)
    # Orientation 0 is the identity permutation (0, 1, 2).
    assert oriented_dims(dims, 0) == dims


def test_oriented_dims_wraps_modulo_num_orientations():
    dims = (2, 3, 4)
    assert oriented_dims(dims, 0) == oriented_dims(dims, NUM_ORIENTATIONS)


def test_unique_orientations_cube_has_one():
    assert unique_orientations((3, 3, 3)) == [0]


def test_unique_orientations_all_distinct_dims_has_six():
    orients = unique_orientations((2, 3, 4))
    assert len(orients) == 6
    footprints = {oriented_dims((2, 3, 4), o) for o in orients}
    assert len(footprints) == 6


def test_unique_orientations_two_equal_dims_has_three():
    # (2, 2, 4): swapping the two equal axes gives a duplicate footprint.
    orients = unique_orientations((2, 2, 4))
    assert len(orients) == 3


def test_drop_height_on_empty_heightmap_is_zero():
    hm = np.zeros((5, 5), dtype=np.int32)
    assert drop_height(hm, 0, 0, 2, 2) == 0


def test_drop_height_rests_on_tallest_column():
    hm = np.zeros((5, 5), dtype=np.int32)
    hm[1, 1] = 3
    assert drop_height(hm, 0, 0, 2, 2) == 3
    # Away from the tall column, still zero.
    assert drop_height(hm, 3, 3, 2, 2) == 0


def test_support_ratio_full_support():
    hm = np.full((5, 5), 2, dtype=np.int32)
    assert support_ratio(hm, 0, 0, 2, 2, 2) == 1.0


def test_support_ratio_partial_support():
    hm = np.zeros((4, 4), dtype=np.int32)
    hm[0:2, 0:2] = 1  # a 2x2 pillar of height 1 in the corner
    # A 4x4 footprint (whole grid) resting at z=1 is only supported by the
    # 2x2 pillar: 4 of 16 cells.
    ratio = support_ratio(hm, 0, 0, 4, 4, 1)
    assert abs(ratio - 4 / 16) < 1e-9


def test_support_ratio_no_support_at_wrong_z():
    hm = np.zeros((3, 3), dtype=np.int32)
    assert support_ratio(hm, 0, 0, 2, 2, 5) == 0.0


def test_support_polygon_contains_com_floor_is_always_true():
    hm = np.zeros((4, 4), dtype=np.int32)
    assert support_polygon_contains_com(hm, 0, 0, 2, 2, 0) is True


def test_support_polygon_contains_com_full_support_true():
    hm = np.full((4, 4), 2, dtype=np.int32)
    assert support_polygon_contains_com(hm, 0, 0, 2, 2, 2) is True


def test_support_polygon_contains_com_overhang_false():
    # Footprint of 4x4 at z=1, but only a single corner cell is at height 1:
    # the COM (at index 1.5, 1.5) is far outside that corner -> unstable.
    hm = np.zeros((4, 4), dtype=np.int32)
    hm[0, 0] = 1
    assert support_polygon_contains_com(hm, 0, 0, 4, 4, 1) is False


def test_box_volume():
    assert box_volume((2, 3, 4)) == 24
    assert box_volume((1, 1, 1)) == 1
