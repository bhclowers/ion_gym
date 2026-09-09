"""
test_multigrid — gates for the V-cycle multigrid accelerator (multigrid3d).

The contract under test: solve3d_vmg converges to the SAME field as the
validated solve3d (its fine smoother and its convergence criterion ARE the
validated `_sweep`), in fewer fine sweeps. Gates:

  1  same fixed point as solve3d on a mixed Dirichlet scene (plate + walls)
  2  THIN-metal scene (2-node plate): the case that diverged before the
     block-ANY coarse masks — must converge and match
  3  mirror plane: z-mirrored scene matches plain solve3d with same mirror
  4  acceleration: fewer fine sweeps than SOR on a domain large enough for
     the hierarchy to engage

Run: python3 -m pytest test_multigrid.py -q
"""
import numpy as np

from ion_gym.physics.solver3d import solve3d
from ion_gym.physics.multigrid3d import solve3d_vmg


def _scene_plate(n=65, thin=False):
    fixed = np.zeros((n, n, 33), bool)
    val = np.zeros(fixed.shape)
    w = 1 if thin else 3
    fixed[n//4:3*n//4, n//2:n//2+w, 8:25] = True
    val[fixed] = 1e4
    fixed[:, 0, :] = True          # grounded wall
    fixed[:, -1, :] = True
    return fixed, val


def test_gate1_same_fixed_point():
    fixed, val = _scene_plate()
    p_sor, sw_sor, _ = solve3d(fixed, val, tol=1e-6)
    p_vmg, sw_vmg, d = solve3d_vmg(fixed, val, tol=1e-6)
    assert d < 1e-6
    # both are fixed points of the same validated sweep at the same tol;
    # they agree to the convergence floor
    assert np.abs(p_vmg - p_sor).max() < 5e-4, \
        f"fields differ by {np.abs(p_vmg-p_sor).max():.2e} V"


def test_gate2_thin_metal():
    """2-node-thick plate: vanishes under injected coarse masks; block-ANY
    keeps it. This exact failure mode drove the original divergence."""
    fixed, val = _scene_plate(thin=True)
    p_sor, _, _ = solve3d(fixed, val, tol=1e-6)
    p_vmg, _, d = solve3d_vmg(fixed, val, tol=1e-6, max_cycles=60)
    assert d < 1e-6, f"vmg failed to converge (d={d:.2e})"
    assert np.abs(p_vmg - p_sor).max() < 5e-4


def test_gate3_mirror():
    n = 65
    fixed = np.zeros((n, n, 17), bool)
    val = np.zeros(fixed.shape)
    fixed[n//4:3*n//4, n//4:3*n//4, 12:15] = True   # board above z-mirror
    val[fixed] = 1e4
    fixed[0, :, :] = True                            # grounded end wall
    p_sor, _, _ = solve3d(fixed, val, mirror=(False, False, True), tol=1e-6)
    p_vmg, _, d = solve3d_vmg(fixed, val, mirror=(False, False, True),
                              tol=1e-6)
    assert d < 1e-6
    assert np.abs(p_vmg - p_sor).max() < 5e-4


def test_gate4_acceleration():
    """On a domain big enough for the hierarchy, vmg must need far fewer
    validated fine sweeps than SOR (the entire point)."""
    n = 129
    fixed = np.zeros((n, 65, 33), bool)
    val = np.zeros(fixed.shape)
    fixed[10:119, 30:33, 8:25] = True
    val[fixed] = 1e4
    fixed[:, 0, :] = True
    p_sor, sw_sor, _ = solve3d(fixed, val, tol=1e-6)
    p_vmg, sw_vmg, d = solve3d_vmg(fixed, val, tol=1e-6)
    assert d < 1e-6
    assert np.abs(p_vmg - p_sor).max() < 5e-4
    assert sw_vmg < 0.5 * sw_sor, \
        f"no acceleration: vmg {sw_vmg} vs sor {sw_sor} sweeps"


if __name__ == "__main__":
    for fn in [test_gate1_same_fixed_point, test_gate2_thin_metal,
               test_gate3_mirror, test_gate4_acceleration]:
        fn(); print("PASS", fn.__name__)
