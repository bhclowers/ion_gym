"""
test_solver3d_analytic.py — S-1 gates for the native 3-D Laplace solver.

  S1-a EXACTNESS: phi = x^2 + y^2 - 2 z^2 is in the null space of the
       discrete 7-point Laplacian, so with a Dirichlet shell the solver
       must reproduce it to solver tolerance on every interior node.
  S1-b ORDER: smooth harmonic sin(ax) sin(by) sinh(cz), c^2 = a^2 + b^2;
       interior error must scale ~h^2 (order > 1.8 between two grids).
  FOLD CERTIFICATION: a problem
       symmetric about x=0 and y=0 solved FULL vs FOLDED with mirror
       ghosts must agree to solver tolerance on the shared quadrant.
       This is the per-problem proof required before any
       sweep relies on a symmetry plane.
  S1-d SHORTLEY-WELLER: coaxial cylinder capacitor with a NON-lattice
       inner radius (r = 6.3 gu). Exact 2-D solution ln(r)/ln ratio;
       outer Dirichlet ring from the exact values. The fractional-leg
       solve must beat the node-Dirichlet solve by >= 10x in max interior
       error (the node solve misplaces the inner surface by up to half a
       cell; S-W restores it).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ion_gym.physics.solver3d import solve3d, fractions_from_scene
from ion_gym.physics.scene3d import GeomScene, GridSpec, Electrode, Shape, Cylinder


def shell(nx, ny, nz):
    f = np.zeros((nx, ny, nz), bool)
    f[0], f[-1] = True, True
    f[:, 0], f[:, -1] = True, True
    f[:, :, 0], f[:, :, -1] = True, True
    return f


def test_exact():
    n = 33
    i, j, k = np.meshgrid(*[np.arange(n, dtype=float)] * 3, indexing="ij")
    exact = i * i + j * j - 2 * k * k
    fixed = shell(n, n, n)
    val = np.where(fixed, exact, 0.0)
    phi, sw, d = solve3d(fixed, val, tol=1e-9)
    err = np.abs(phi - exact)[~fixed].max()
    assert err < 1e-6, err
    print(f"S1-a exactness: max err {err:.2e} ({sw} sweeps)      PASS")


def test_order():
    errs = []
    for n in (17, 33):
        L = 1.0
        h = L / (n - 1)
        a = b = np.pi / L
        c = np.sqrt(a * a + b * b)
        x = np.arange(n) * h
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        exact = np.sin(a * X) * np.sin(b * Y) * np.sinh(c * Z)
        fixed = shell(n, n, n)
        val = np.where(fixed, exact, 0.0)
        phi, _, _ = solve3d(fixed, val, tol=1e-10)
        errs.append(np.abs(phi - exact)[~fixed].max())
    order = np.log2(errs[0] / errs[1])
    assert order > 1.8, (errs, order)
    print(f"S1-b order: errs {errs[0]:.2e} -> {errs[1]:.2e}, "
          f"order {order:.2f}      PASS")


def _coax_scene(cx, cy, r, n, nz):
    g = GridSpec(n, n, nz, mm_per_gu=1.0)
    return GeomScene(grid=g, units="gu", name="coax", electrodes=[
        Electrode(1, "inner", [Shape([Cylinder(cx, cy, float(nz + 2),
                                               r, float(nz + 4))])])])


def test_fold():
    n, nz, r = 41, 9, 6.3
    # FULL domain 81x81 with axis at (40,40)
    N = 2 * n - 1
    ii, jj = np.meshgrid(np.arange(N, dtype=float),
                         np.arange(N, dtype=float), indexing="ij")
    rr = np.hypot(ii - 40, jj - 40)
    inner_f = (rr <= r)[:, :, None] & np.ones((1, 1, nz), bool)
    ring_f = (rr >= 32)[:, :, None] & np.ones((1, 1, nz), bool)
    exact2d = np.log(np.maximum(rr, r) / 32.0) / np.log(r / 32.0) * 100.0
    val_f = np.where(inner_f, 100.0, 0.0) + np.where(
        ring_f, exact2d[:, :, None], 0.0)
    fixed_f = inner_f | ring_f
    phi_f, _, _ = solve3d(fixed_f, val_f, tol=1e-8)
    # FOLDED quadrant 41x41 with mirror x,y at index 0 (axis at (0,0))
    ii, jj = np.meshgrid(np.arange(n, dtype=float),
                         np.arange(n, dtype=float), indexing="ij")
    rr = np.hypot(ii, jj)
    inner_q = (rr <= r)[:, :, None] & np.ones((1, 1, nz), bool)
    ring_q = (rr >= 32)[:, :, None] & np.ones((1, 1, nz), bool)
    exact2d = np.log(np.maximum(rr, r) / 32.0) / np.log(r / 32.0) * 100.0
    val_q = np.where(inner_q, 100.0, 0.0) + np.where(
        ring_q, exact2d[:, :, None], 0.0)
    phi_q, _, _ = solve3d(inner_q | ring_q, val_q,
                          mirror=(True, True, False), tol=1e-8)
    d = np.abs(phi_q - phi_f[40:, 40:, :]).max()
    assert d < 1e-5, d
    print(f"S1-c fold certification: full-vs-folded max |d| {d:.2e}   PASS")


def test_shortley_weller():
    n, nz, r = 81, 9, 6.3
    cx = cy = 40.0
    ii, jj = np.meshgrid(np.arange(n, dtype=float),
                         np.arange(n, dtype=float), indexing="ij")
    rr = np.hypot(ii - cx, jj - cy)
    inner = (rr <= r)[:, :, None] & np.ones((1, 1, nz), bool)
    ring = (rr >= 32)[:, :, None] & np.ones((1, 1, nz), bool)
    exact = (np.log(np.maximum(rr, r) / 32.0) / np.log(r / 32.0)
             * 100.0)[:, :, None] * np.ones((1, 1, nz))
    fixed = inner | ring
    val = np.where(inner, 100.0, 0.0) + np.where(ring, exact, 0.0)
    vac = ~fixed & (rr[:, :, None] > r) & (rr[:, :, None] < 32)

    phi0, _, _ = solve3d(fixed, val, tol=1e-8)
    e0 = np.abs(phi0 - exact)[vac].max()

    sc = _coax_scene(cx, cy, r, n, nz)
    th = fractions_from_scene(sc, fixed & inner)   # legs only at the metal
    phi1, _, _ = solve3d(fixed, val, theta=th, tol=1e-8, stencil="sw")
    e1 = np.abs(phi1 - exact)[vac].max()
    assert e1 * 10 <= e0, (e0, e1)
    print(f"S1-d Shortley-Weller: node-Dirichlet max err {e0:.4f} V -> "
          f"S-W {e1:.4f} V ({e0/e1:.0f}x better)      PASS")
    # external_reference linear-ghost stencil: lower order but must still beat node
    phi2, _, _ = solve3d(fixed, val, theta=th, tol=1e-8, stencil="ghost_linear")
    e2 = np.abs(phi2 - exact)[vac].max()
    assert e2 * 3 <= e0, (e0, e2)
    print(f"S1-e external_reference linear-ghost stencil: max err {e2:.4f} V "
          f"({e0/e2:.0f}x better than node-Dirichlet)      PASS")


if __name__ == "__main__":
    test_exact()
    test_order()
    test_fold()
    test_shortley_weller()
    print("\nSOLVER3D ANALYTIC GATES: ALL PASS")
