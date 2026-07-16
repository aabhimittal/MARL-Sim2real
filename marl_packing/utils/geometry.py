"""Geometry helpers for axis-aligned 3D box packing.

All boxes are axis-aligned cuboids on an integer voxel grid. An orientation is
one of the 6 permutations of a box's (length, width, height) dimensions.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np

Dims = Tuple[int, int, int]

# The 6 axis-aligned rotations of a cuboid, as permutations of (l, w, h).
ORIENTATIONS: List[Tuple[int, int, int]] = list(itertools.permutations((0, 1, 2)))
NUM_ORIENTATIONS = len(ORIENTATIONS)


def oriented_dims(dims: Dims, orientation: int) -> Dims:
    """Return box dimensions after applying one of the 6 axis-aligned rotations."""
    perm = ORIENTATIONS[orientation % NUM_ORIENTATIONS]
    return (dims[perm[0]], dims[perm[1]], dims[perm[2]])


def unique_orientations(dims: Dims) -> List[int]:
    """Orientation indices that produce distinct footprints (cubes have 1, not 6)."""
    seen, result = set(), []
    for i in range(NUM_ORIENTATIONS):
        d = oriented_dims(dims, i)
        if d not in seen:
            seen.add(d)
            result.append(i)
    return result


def drop_height(heightmap: np.ndarray, x: int, y: int, dx: int, dy: int) -> int:
    """Z at which a (dx, dy) footprint rests when dropped at (x, y).

    The box settles on the tallest column beneath its footprint (gravity drop,
    no tilting -- tilt risk is the Physics Agent's job to judge).
    """
    region = heightmap[x : x + dx, y : y + dy]
    return int(region.max()) if region.size else 0


def support_ratio(heightmap: np.ndarray, x: int, y: int, dx: int, dy: int, z: int) -> float:
    """Fraction of the footprint actually resting on something at height z."""
    region = heightmap[x : x + dx, y : y + dy]
    if region.size == 0:
        return 0.0
    return float((region == z).sum()) / region.size


def support_polygon_contains_com(
    heightmap: np.ndarray, x: int, y: int, dx: int, dy: int, z: int
) -> bool:
    """True if the box's center of mass sits over the axis-aligned bounding box
    of its supported cells -- a fast necessary condition for static stability.
    """
    region = heightmap[x : x + dx, y : y + dy]
    xs, ys = np.nonzero(region == z)
    if xs.size == 0:
        return z == 0  # resting on the floor
    com_x, com_y = (dx - 1) / 2.0, (dy - 1) / 2.0
    return bool(xs.min() <= com_x <= xs.max() and ys.min() <= com_y <= ys.max())


def box_volume(dims: Dims) -> int:
    return int(dims[0]) * int(dims[1]) * int(dims[2])
