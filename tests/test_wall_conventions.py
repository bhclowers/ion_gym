"""
Where each route puts a metal surface (review C-06), as documented in
docs/SETUP_GUIDE.md, "Where the metal surface sits".

* planar and r-z (raster2d._metal_nn): an ion is in metal when its
  NEAREST node is metal, so the surface lies half a cell outside the
  outermost metal nodes and a one-node plate still stops ions;
* 3-D (tracer3d._in_metal_3d): an ion is in metal only inside a cell
  whose eight corners are all metal, so the surface lies on the
  outermost metal nodes and a one-node plate stops nothing.

A slit whose faces sit on nodes w cells apart is therefore open w - 1
cells on planar and r-z and w cells on 3-D. Positions are in grid units.
"""
import numpy as np
import pytest

from ion_gym.physics.raster2d import _metal_nn
from ion_gym.physics.tracer3d import _in_metal_3d

NX, NY, NZ = 6, 40, 4
STEP = 1e-3


def _hits_2d(ele):
    ys = np.arange(0.0, NY - 1, STEP)
    return ys[[bool(_metal_nn(ele, 2.0, y, NX, NY)) for y in ys]]


def _hits_3d(ele):
    ys = np.arange(0.0, NY - 1, STEP)
    return ys[[bool(_in_metal_3d(ele, 2.0, y, 1.5, NX, NY, NZ))
               for y in ys]]


def _plate(rows):
    e2 = np.zeros((NX, NY))
    e3 = np.zeros((NX, NY, NZ), dtype=np.bool_)
    for j in rows:
        e2[:, j] = 1.0
        e3[:, j, :] = True
    return e2, e3


def test_a_thick_plate_face():
    e2, e3 = _plate(range(20, 30))
    assert _hits_2d(e2).min() == pytest.approx(19.5, abs=STEP)
    assert _hits_3d(e3).min() == pytest.approx(20.0, abs=STEP)


def test_a_one_node_plate():
    e2, e3 = _plate([20])
    h2 = _hits_2d(e2)
    assert h2.min() == pytest.approx(19.5, abs=STEP)
    assert h2.max() == pytest.approx(20.5, abs=2 * STEP)   # [19.5, 20.5)
    assert _hits_3d(e3).size == 0


@pytest.mark.parametrize("w", [4, 8])
def test_a_slit_is_one_cell_narrower_on_planar(w):
    lo, hi = 10, 10 + w          # faces on nodes lo and hi
    e2, e3 = _plate(list(range(2, lo + 1)) + list(range(hi, 36)))
    ys = np.arange(lo, hi + STEP / 2, STEP)
    open2 = ys[[not bool(_metal_nn(e2, 2.0, y, NX, NY)) for y in ys]]
    open3 = ys[[not bool(_in_metal_3d(e3, 2.0, y, 1.5, NX, NY, NZ))
                for y in ys]]
    assert open2.max() - open2.min() == pytest.approx(w - 1, abs=2 * STEP)
    assert open3.max() - open3.min() == pytest.approx(w, abs=2 * STEP)
