"""
Analytic gates for solver2d: the closed-form anchors every cross-check
presupposes. If these fail, no comparison downstream means anything.

1. Coaxial capacitor  (cylindrical): phi(r) = V ln(r/a)/ln(b/a), z-invariant.
   Exercises the (1/r) d/dr term directly.
2. Parallel plates    (planar): linear phi — regression vs the old solver's
   physics (trivial but catches sign/assembly bugs).
3. Axis stencil check (cylindrical): potential of two biased apertures must be
   smooth and even in r across the axis -> finite on-axis curvature, no kink.
4. Grid convergence   : error vs h should fall ~h^2 for the coaxial case.
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import numpy as np
from ion_gym.physics.solver2d import solve_laplace


def coaxial(h, a=2.0, b=8.0, V=100.0):
    Lz = 6.0
    nz = int(round(Lz / h)) + 1
    nr = int(round(b / h)) + 1
    R = np.linspace(0, b, nr)
    fixed = np.zeros((nz, nr), bool)
    val = np.zeros((nz, nr))
    ia = int(round(a / h))
    fixed[:, ia] = True; val[:, ia] = 0.0          # inner conductor r=a
    fixed[:, -1] = True; val[:, -1] = V            # outer conductor r=b
    # inside r<a irrelevant; pin it to 0 so the system is well-posed
    fixed[:, :ia] = True; val[:, :ia] = 0.0
    phi = solve_laplace(fixed, val, h, symmetry="cylindrical",
                        neumann_edges=("z0", "z1"))
    r = R[ia:]
    exact = V * np.log(np.maximum(r, a) / a) / np.log(b / a)
    err = np.abs(phi[nz // 2, ia:] - exact)
    return np.max(err), V


def planar_plates(h=0.1):
    Lz, Lu, V = 10.0, 4.0, 50.0
    nz = int(round(Lz / h)) + 1
    nu = int(round(Lu / h)) + 1
    fixed = np.zeros((nz, nu), bool); val = np.zeros((nz, nu))
    fixed[0, :] = True;  val[0, :] = 0.0
    fixed[-1, :] = True; val[-1, :] = V
    phi = solve_laplace(fixed, val, h, symmetry="planar",
                        neumann_edges=("u0", "u1"))
    Z = np.linspace(0, Lz, nz)
    exact = V * Z / Lz
    return np.max(np.abs(phi[:, nu // 2] - exact)), V


def axis_smoothness(h=0.05):
    """Paraxial gate: for axisymmetric fields, phi(z,r) - phi(z,0) must equal
    -phi''(z,0) r^2/4 near the axis. Tests the axis stencil where it matters
    (this curvature IS the lens strength)."""
    from ion_gym.physics.solver2d import cylinder_einzel
    Z, R, fixed, val, bands = cylinder_einzel([0.0, 100.0], bore_r=3.0, thick=1.0,
                                              egap=4.0, lead=6.0, r_max=9.0, h=h)
    phi = solve_laplace(fixed, val, h, symmetry="cylindrical", neumann_edges=("u1",))
    d2 = np.gradient(np.gradient(phi[:, 0], Z), Z)
    worst = 0.0
    for zt in np.linspace(bands[0][1] + 0.5, bands[1][0] - 0.5, 5):
        iz = np.argmin(np.abs(Z - zt))
        for jr in (2, 4, 6):
            r = R[jr]
            num = phi[iz, jr] - phi[iz, 0]
            par = -d2[iz] * r * r / 4.0
            if abs(par) > 1e-6:
                worst = max(worst, abs(num - par) / abs(par))
    return worst, phi


if __name__ == "__main__":
    print("== 1. coaxial capacitor (cylindrical exact solution) ==")
    for h in (0.2, 0.1, 0.05):
        e, V = coaxial(h)
        print(f"   h={h:5.2f} mm  max|err| = {e:.4e} V  ({e/V*100:.4f}% of {V:.0f} V)")

    print("== 2. parallel plates (planar regression) ==")
    e, V = planar_plates()
    print(f"   max|err| = {e:.3e} V on {V:.0f} V (should be ~machine/discretisation zero)")

    print("== 3. axis stencil smoothness (no kink at r=0) ==")
    q, _ = axis_smoothness()
    print(f"   paraxial-consistency worst rel. dev = {q*100:.2f}%  (want ~grid-level)")

    print("== 4. convergence order (coaxial) ==")
    hs = np.array([0.4, 0.2, 0.1, 0.05])
    es = np.array([coaxial(h)[0] for h in hs])
    order = np.polyfit(np.log(hs), np.log(es), 1)[0]
    print(f"   errors: {', '.join(f'{e:.3e}' for e in es)}")
    print(f"   observed order ~ h^{order:.2f}  (expect ~2)")
