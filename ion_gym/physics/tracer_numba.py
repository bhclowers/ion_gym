"""
ion_gym.tracer_numba
---------------------
The 2-D field build and the jitted bilinear sampler that ionbench and
tracer_tdep fly through. This module holds no integrator.

  1. MIRROR-EXTENSION. build_field pre-extends a cylindrical field to
     negative r, so E_r is odd, E_r(axis)=0, and the sampler needs NO
     special axis/sign logic -- signed r just indexes the extended grid.
  2. BILINEAR sampling with RegularGridInterpolator's out-of-bounds rule
     (bounds_error=False, fill_value=0.0): any point outside
     [Z0,Z1]x[U0,U1] returns 0.
"""

import numpy as np
from numba import njit


def build_field(Z, U, phi, symmetry="cylindrical"):
    """Return (Z, Ue, Ez, Eu) with Ez/Eu in V/m on the (mirror-extended) grid.
    Mirrors Field2D.__init__ exactly so the jitted sampler matches the NumPy
    interpolator node-for-node."""
    if symmetry == "cylindrical":
        assert abs(U[0]) < 1e-12, "cylindrical grid must start at the axis"
        Ue = np.concatenate([-U[:0:-1], U])
        pe = np.concatenate([phi[:, :0:-1], phi], axis=1)
    else:
        Ue, pe = U.copy(), phi.copy()
    dpz, dpu = np.gradient(pe, Z, Ue)          # V/mm
    Ez = np.ascontiguousarray(-dpz * 1e3)      # V/m
    Eu = np.ascontiguousarray(-dpu * 1e3)
    return (np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(Ue, dtype=np.float64), Ez, Eu)


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _sample(Zg, Ug, F, z_mm, u_mm):
    """Bilinear sample of F on the (possibly non-uniform-safe but here uniform)
    grid Zg x Ug. Matches RegularGridInterpolator with bounds_error=False,
    fill_value=0.0: any point outside the grid box returns 0.0.
    Grids are uniform -> O(1) index via spacing."""
    nz = Zg.shape[0]
    nu = Ug.shape[0]
    z0 = Zg[0]; u0 = Ug[0]
    dz = Zg[1] - Zg[0]
    du = Ug[1] - Ug[0]
    if z_mm < z0 or z_mm > Zg[nz - 1] or u_mm < u0 or u_mm > Ug[nu - 1]:
        return 0.0
    fz = (z_mm - z0) / dz
    fu = (u_mm - u0) / du
    iz = int(fz)
    iu = int(fu)
    if iz >= nz - 1:
        iz = nz - 2
    if iu >= nu - 1:
        iu = nu - 2
    tz = fz - iz
    tu = fu - iu
    f00 = F[iz, iu]
    f10 = F[iz + 1, iu]
    f01 = F[iz, iu + 1]
    f11 = F[iz + 1, iu + 1]
    return ((f00 * (1.0 - tz) + f10 * tz) * (1.0 - tu)
            + (f01 * (1.0 - tz) + f11 * tz) * tu)
