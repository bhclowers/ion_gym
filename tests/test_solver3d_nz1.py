"""
test_solver3d_nz1.py — lock in the nz=1 (thin-slab) solver correctness.

solver3d's SOR sweep once read an out-of-range z-neighbour when nz==1
(phi[i,j,1] with only index 0 present), which broke x/y symmetry of the
WHOLE solve: an electrode on the +x axis and the same electrode on the +y
axis gave different axis potentials. Every planar (2-D) basis solve routed
through solver3d at nz=1 was silently corrupted by this (the planar einzel
read -94 V instead of the true -71.4 V). The fix makes a size-1 axis's
ghost equal the node itself, reducing the stencil to the exact lower-D
Laplace.

Gates:
  Z-1 SYMMETRY: at nz=1, a +x electrode and a +y electrode (mirror images)
      give equal on-axis potential.
  Z-2 DIM-INVARIANCE: the nz=1 solution equals the nz=3 / nz=5 solution
      for a z-uniform geometry (a size-1 z-axis IS the 2-D problem).
  Z-3 2-D LAPLACE: an interior node with a size-1 z-axis satisfies the
      pure 2-D 5-point mean (no spurious z self-coupling).
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ion_gym.physics.solver3d import solve3d

N, C = 51, 25


def _axis_potential(ex, ey, nz):
    metal = np.zeros((N, N, nz), bool)
    metal[0, :, :] = metal[-1, :, :] = True
    metal[:, 0, :] = metal[:, -1, :] = True
    metal[ex - 2:ex + 3, ey - 2:ey + 3, :] = True
    val = np.zeros((N, N, nz))
    val[ex - 2:ex + 3, ey - 2:ey + 3, :] = 1e4
    phi, _, _ = solve3d(metal, val, tol=1e-8, max_sweeps=20000, omega=1.0)
    return phi[C, C, nz // 2], phi


def main():
    ok = True

    # Z-1 symmetry at nz=1
    vx, _ = _axis_potential(40, 25, 1)     # electrode on +x axis
    vy, _ = _axis_potential(25, 40, 1)     # same, on +y axis
    g1 = abs(vx - vy) < 1e-4
    ok &= g1
    print(f"Z-1 nz=1 x/y symmetry: +x axis {vx:.4f} vs +y axis {vy:.4f} "
          f"-> {'PASS' if g1 else 'FAIL'}")

    # Z-2 dimension invariance: nz=1 == nz=3 == nz=5
    v1, _ = _axis_potential(40, 25, 1)
    v3, _ = _axis_potential(40, 25, 3)
    v5, _ = _axis_potential(40, 25, 5)
    g2 = abs(v1 - v3) < 1e-3 and abs(v1 - v5) < 1e-3
    ok &= g2
    print(f"Z-2 dim-invariance: nz=1 {v1:.4f}, nz=3 {v3:.4f}, nz=5 "
          f"{v5:.4f} -> {'PASS' if g2 else 'FAIL'}")

    # Z-3 pure 2-D Laplace at an interior vacuum node (nz=1): the node
    # equals the mean of its four in-plane neighbours (no z bias).
    _, phi = _axis_potential(40, 25, 1)
    i, j = 15, 30                          # an interior vacuum node
    mean4 = 0.25 * (phi[i - 1, j, 0] + phi[i + 1, j, 0]
                    + phi[i, j - 1, 0] + phi[i, j + 1, 0])
    g3 = abs(phi[i, j, 0] - mean4) < 1e-3
    ok &= g3
    print(f"Z-3 2-D Laplace: node {phi[i, j, 0]:.5f} vs 4-neighbour mean "
          f"{mean4:.5f} -> {'PASS' if g3 else 'FAIL'}")

    print("\nSOLVER3D nz=1 GATES:", "ALL PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
