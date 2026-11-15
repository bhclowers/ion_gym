"""
save_trajectories accepts every form its docstring names.

(index, traj, summary) tuples were all skipped: a tuple has an .index
method, so the IonResult attribute test matched first, found no .traj and
dropped the ion. A generator of results was spent by the save loop, so the
skipped count taken from it afterwards came out negative.
"""
import numpy as np

from ion_gym.io import field_io
from ion_gym.io.sim_spec import (ElectrodeSpec, GeometrySpec, ShapeSpec,
                                 SimSpec)


def _spec():
    rect = ShapeSpec(type="rect",
                     params=dict(x_mm=2.0, y_mm=10.0, width_mm=16.0,
                                 height_mm=1.0))
    return SimSpec(name="traj_forms", geometry=GeometrySpec(
        width_mm=20.0, height_mm=12.0, mm_per_gu=0.5,
        electrodes=[ElectrodeSpec(name="top", dc=100.0, shapes=[rect])]))


COLS = ["t", "x", "y"]               # the toy trajectories' 3 columns


class _Result:                              # minimal IonResult shape
    def __init__(self, i, t, s):
        self.index, self.traj, self.summary = i, t, s


def test_tuples_with_a_summary_are_saved(tmp_path):
    a, b = np.arange(12.0).reshape(4, 3), np.ones((2, 3))
    p = field_io.save_trajectories(
        [(0, a, {"kind": 0}), (3, b, {"kind": 1})], _spec(),
        path=str(tmp_path / "t.traj.npz"), columns=COLS)
    trajs, meta, _ = field_io.load_trajectories(p)
    assert sorted(trajs) == [0, 3]
    assert np.array_equal(trajs[0], a) and np.array_equal(trajs[3], b)
    assert [s["kind"] for s in meta["summaries"]] == [0, 1]
    assert meta["n_skipped"] == 0


def test_index_trajectory_pairs_are_saved(tmp_path):
    p = field_io.save_trajectories(
        [[5, np.zeros((3, 3))], (6, None)], _spec(),
        path=str(tmp_path / "t.traj.npz"), columns=COLS)
    trajs, meta, _ = field_io.load_trajectories(p)
    assert sorted(trajs) == [5]
    assert meta["n_skipped"] == 1


def test_a_generator_counts_its_skipped_ions(tmp_path):
    res = (_Result(i, None if i == 1 else np.ones((2, 3)), {"kind": 0})
           for i in range(3))
    p = field_io.save_trajectories(res, _spec(),
                                   path=str(tmp_path / "t.traj.npz"), columns=COLS)
    trajs, meta, _ = field_io.load_trajectories(p)
    assert sorted(trajs) == [0, 2]
    assert meta["n_skipped"] == 1
