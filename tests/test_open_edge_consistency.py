"""
A full-depth extrusion solves to the planar field: open edges mean the
same thing on every route.

The 2-D solver put the zero-gradient plane of an open edge half a node
OUTSIDE the domain (ghost = edge value, first order); the 3-D solvers put
it ON the edge node (reflection, second order). The same geometry then
solved about 1 V apart near the edges at 0.25 mm. 2-D now uses the
reflection too (it is the second-order rule, and 2-D fold planes already
used it). Checked here: a deck and its full-depth extrusion agree to the
3-D solver's tolerance on every z-slice; and on an exact solution the 2-D
open edge converges at second order.
"""
import copy
import warnings

import numpy as np
import pytest

from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.sim_build import build_run
from ion_gym.physics.solver2d import solve_laplace

DEPTH = 4.0
SPAN_V = 100.0


def _deck(depth):
    def plate(name, x, dc):
        shape = {"type": "rect", "x_mm": x, "y_mm": 2.0, "width_mm": 1.0,
                 "height_mm": 6.0}
        if depth:
            shape["extrude"] = {"axis": "z", "lo_mm": 0.0, "hi_mm": depth}
        return {"name": name, "dc": dc, "shapes": [shape]}
    return {
        "name": "two plates",
        "geometry": {"width_mm": 20.0, "height_mm": 10.0, "depth_mm": depth,
                     "mm_per_gu": 0.25,
                     "symmetry": {"coords": "xyz", "planes": {
                         "x": "none", "y": "none", "z": "none"}},
                     "electrodes": [plate("left", 4.0, SPAN_V),
                                    plate("right", 15.0, 0.0)]},
        "source": {"n_ions": 1, "distribution": "point", "x0_mm": 10.0,
                   "y0_mm": 5.0, "z0_mm": depth / 2, "direction": [1, 0, 0],
                   "ke_lo": 1.0, "ke_hi": 1.0, "mz_list": [100.0],
                   "charge": 1, "tob_span_us": 0.0, "seed": 1},
        "integration": {"dt_ns": 1.0, "t_max_us": 1.0, "rec_every": 10,
                        "max_records": 1000}}


def _static_field(depth):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_run(SimSpec.from_dict(copy.deepcopy(_deck(depth))))[0]
    return np.asarray(model.A, float)


def test_a_full_depth_extrusion_equals_the_planar_solve():
    flat = _static_field(0.0)
    deep = _static_field(DEPTH)
    assert deep.shape[:2] == flat.shape
    worst = max(np.abs(deep[:, :, k] - flat).max()
                for k in range(deep.shape[2]))
    # measured 6.5e-6 V (float32 bases); the old 2-D edge rule left
    # 0.53 V here, on this 100 V span
    assert worst < 1e-3, worst


def _exact_case(n):
    """Unit square: phi = sin(pi y) on x = 0, 0 on y = 0 and 1, an open
    edge at x = 1 (the last node). Exact: sin(pi y) cosh(pi(1-x))/cosh(pi)."""
    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)
    fixed = np.zeros((n + 1, n + 1), bool)
    val = np.zeros((n + 1, n + 1))
    fixed[0, :] = True
    val[0, :] = np.sin(np.pi * y)
    fixed[:, 0] = fixed[:, -1] = True
    phi = solve_laplace(fixed, val, 1.0 / n, symmetry="planar",
                        neumann_edges=("z1",))
    X, Y = np.meshgrid(x, y, indexing="ij")
    exact = np.sin(np.pi * Y) * np.cosh(np.pi * (1 - X)) / np.cosh(np.pi)
    return np.abs(phi - exact).max()


def test_the_2d_open_edge_converges_at_second_order():
    e32, e64 = _exact_case(32), _exact_case(64)
    assert np.log2(e32 / e64) == pytest.approx(2.0, abs=0.1)
