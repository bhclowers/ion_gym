"""
voxel_views coarsens the 3-D solver-voxel view until its faces fit the
budget, names the pooling stride in the legend, and stops coarsening at
_MAX_VOXEL_POOL_STRIDE + 1. (The stride search was a `while True` loop
with that cap inside; it is now a bounded for-loop. This passes before
and after that change.)
"""
import re

import numpy as np

from ion_gym.viz.viz_core import _MAX_VOXEL_POOL_STRIDE, voxel_views


def _stride(fig3):
    m = re.search(r"\[(\d+)x pooled\]", fig3.data[0].name)
    return int(m.group(1)) if m else 1


def test_small_grids_are_drawn_unpooled():
    lab = np.zeros((10, 10, 10), np.int16)
    lab[2:5, 2:5, 2:5] = 1
    assert _stride(voxel_views(lab, 0.5)["3d"]) == 1


def test_a_tight_budget_pools_and_says_so():
    lab = np.zeros((40, 40, 40), np.int16)
    lab[5:35, 5:35, 5:35] = 1
    s = _stride(voxel_views(lab, 0.5, max_voxels=2000)["3d"])
    assert 1 < s <= _MAX_VOXEL_POOL_STRIDE + 1


def test_pooling_stops_at_the_cap():
    lab = np.ones((60, 60, 60), np.int16)
    assert _stride(voxel_views(lab, 0.5, max_voxels=1)["3d"]) == \
        _MAX_VOXEL_POOL_STRIDE + 1
