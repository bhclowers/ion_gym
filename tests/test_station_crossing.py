"""
A sample exactly on a station plane is one crossing, not two.

station_hits found crossings where np.sign(q) changed, and a sample ON
the plane has sign 0, so -, 0, + changed sign twice and was reported as
two hits at the same instant. The test is now half-open.
"""
import numpy as np

from ion_gym.io.sim_spec import StationSpec
from ion_gym.physics.stations import station_hits

COLS = ["t", "x", "y", "z", "vx", "vy", "vz"]
PLANE = StationSpec(name="det", kind="detect", on_hit="pass", axis="x",
                    pos_mm=0.0)


def _track(xs, t0=0.1):
    n = len(xs)
    tr = np.zeros((n, len(COLS)))
    tr[:, 0] = t0 + 0.1 * np.arange(n)
    tr[:, 1] = xs
    tr[:, 4] = 1.0
    return tr


def test_a_sample_on_the_plane_is_one_crossing():
    hits = station_hits(_track([-1.0, -0.5, 0.0, 0.5]), COLS, PLANE)
    assert len(hits) == 1
    assert abs(hits[0]["t_us"] - 0.3) < 1e-12 and hits[0]["x"] == 0.0


def test_ordinary_crossings_both_ways_still_count():
    hits = station_hits(_track([-1.0, 1.0, -1.0, 1.0]), COLS, PLANE)
    assert [h["k"] for h in hits] == [0, 1, 2]


def test_landing_on_the_plane_from_above_is_one_crossing():
    hits = station_hits(_track([1.0, 0.0, -1.0]), COLS, PLANE)
    assert len(hits) == 1


def test_a_birth_on_the_plane_is_not_a_crossing():
    hits = station_hits(_track([0.0, 0.5, 1.0], t0=0.5), COLS, PLANE)
    assert hits == []
