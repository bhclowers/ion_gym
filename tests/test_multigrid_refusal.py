"""test_multigrid_refusal.py — the V-cycle's never-silent contract.

Two DELIBERATE-FAILURE tests:
a convergence contract is only proven by the paths that refuse. Success
paths are covered elsewhere; these require the exception.

* iteration cap: an impossibly tight tolerance with max_cycles=1 must
  raise SolveNotConverged naming the exhausted cap.
* stagnation: a tolerance below the float32 transfer-operator precision
  floor must raise SolveNotConverged naming the stall — and the same
  call with accept_stagnation=True must return, because the opt-in is
  the one sanctioned way to take a stalled exploratory iterate.
"""
import numpy as np
import pytest

from ion_gym.physics.multigrid3d import solve3d_vmg
from ion_gym.physics.solver3d import SolveNotConverged


def _plates(n=20):
    fixed = np.zeros((n, n, n), bool)
    val = np.zeros((n, n, n))
    fixed[0], fixed[-1] = True, True
    val[0] = 100.0
    return fixed, val


def test_iteration_cap_raises_not_returns():
    fixed, val = _plates()
    with pytest.raises(SolveNotConverged, match="exhausted max_cycles"):
        solve3d_vmg(fixed, val, tol=1e-30, max_cycles=1)


def test_stagnation_raises_and_opt_in_accepts():
    fixed, val = _plates()
    # tol far below the float32 transfer floor: the iterate must stall
    # before reaching it, and the stall must REFUSE, not return.
    with pytest.raises(SolveNotConverged, match="STAGNATED"):
        solve3d_vmg(fixed, val, tol=1e-14, max_cycles=200)
    # the explicit opt-in is the sanctioned escape hatch: same problem,
    # stated acceptance, and the achieved delta comes back for quoting.
    phi, sweeps, d = solve3d_vmg(fixed, val, tol=1e-14, max_cycles=200,
                                 accept_stagnation=True)
    assert np.isfinite(d) and d > 1e-14
    assert phi.shape == fixed.shape
