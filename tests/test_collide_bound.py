"""
The hard-sphere partner draw (collision3d._collide) is bounded.

Its rejection loop was `while True`: a NaN ion velocity or an infinite
thermal speed never passes the acceptance test, so the nogil kernel spun
forever (review F11, called directly). It now gives up after
MAX_PARTNER_TRIES draws and raises by name. For finite inputs the draw
sequence is unchanged: a seeded run is pinned to the values the
unbounded loop produced.
"""
import subprocess
import sys

import numpy as np
import pytest
from numba import njit

from ion_gym.physics.collision3d import _collide

# run(7, 2000) with the unbounded loop: sum, sum of |v|, last velocity
PINNED = (-21.505474884942984, 453.56068870741814,
          (-0.04958323625398979, 0.026905754191870357, 0.007792414475627368))


@njit
def _run(seed, n):
    np.random.seed(seed)
    out = np.empty((n, 3))
    vx, vy, vz = 0.3, -0.1, 0.05
    for i in range(n):
        vx, vy, vz = _collide(vx, vy, vz, 0.0, 0.0, 0.0, 100.0, 28.0,
                              0.17, np.sqrt(vx * vx + vy * vy + vz * vz))
        out[i, 0], out[i, 1], out[i, 2] = vx, vy, vz
    return out


def test_finite_inputs_draw_exactly_as_before():
    a = _run(7, 2000)
    assert float(a.sum()) == PINNED[0]
    assert float(np.abs(a).sum()) == PINNED[1]
    assert tuple(float(v) for v in a[-1]) == PINNED[2]


@pytest.mark.parametrize("args", [
    "float('nan'), 0.1, 0.0, 0.17, 0.2",
    "0.1, 0.0, 0.0, float('inf'), 0.1",
], ids=["nan_velocity", "inf_thermal_speed"])
def test_a_non_finite_input_is_refused_not_looped(args):
    code = ("from ion_gym.physics.collision3d import _collide\n"
            "vx, vy, vz, sig, sp = " + args + "\n"
            "try:\n"
            "    _collide(vx, vy, vz, 0.0, 0.0, 0.0, 100.0, 28.0, sig, sp)\n"
            "except ValueError as e:\n"
            "    assert 'no gas partner accepted' in str(e), e\n"
            "    print('refused')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "refused"
