"""
ion_gym.tracer
--------------
Meridional-plane ion tracer for axisymmetric fields (and planar 2-D fields).

For the external cross-check we restrict to ions launched with zero azimuthal
velocity: in an axisymmetric field such ions stay in one meridional plane, so
2-D (z, r) tracing is EXACT, and comparison against an independent 3-D flight of the
revolved array is apples-to-apples.

Axis crossings: r is carried as a signed coordinate; the field is sampled at
|r| with E_r sign-flipped for r < 0 (E_r is odd in r by symmetry).

Records, per ion: trajectory, TOF and (r, vr, vz) at a set of z "record
planes" (linear interpolation across the straddling step) — these are the
quantities differenced against externally recorded output.

This is the module Numba lands in first (build scope item 1). Kept as plain
NumPy until the validation gate passes.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

E_CHG = 1.602176634e-19   # C
AMU   = 1.66053907e-27    # kg


class Field2D:
    def __init__(self, Z, U, phi, symmetry="cylindrical"):
        self.Z, self.symmetry = Z, symmetry
        if symmetry == "cylindrical":
            # mirror-extend phi to negative r: phi even in r -> E_r exactly odd,
            # E_r(axis) exactly 0, and central differences everywhere near axis.
            assert abs(U[0]) < 1e-12, "cylindrical grid must start at the axis"
            Ue = np.concatenate([-U[:0:-1], U])
            pe = np.concatenate([phi[:, :0:-1], phi], axis=1)
        else:
            Ue, pe = U, phi
        self.U = Ue
        dpz, dpu = np.gradient(pe, Z, Ue)          # V/mm
        self._Ez = RegularGridInterpolator((Z, Ue), -dpz * 1e3,
                                           bounds_error=False, fill_value=0.0)
        self._Eu = RegularGridInterpolator((Z, Ue), -dpu * 1e3,
                                           bounds_error=False, fill_value=0.0)

    def E(self, z_mm, u_mm):
        """E field (V/m) at (z, u) in mm; signed r handled by the mirrored grid."""
        pt = np.array([[z_mm, u_mm]])
        return float(self._Ez(pt)[0]), float(self._Eu(pt)[0])


def fly(field, mz_Da, K_eV, z0_mm, u0_mm, ang0_mrad=0.0,
        record_planes_mm=(), dt_frac=0.02, h_mm=0.1, max_steps=2_000_000,
        metal=None):
    """Fly one ion. K_eV = total kinetic energy per charge at launch; ang0
    tilts the velocity toward +u. Returns dict with trajectory, exit state,
    and per-plane records [(z_plane, t_ns, u_mm, vz, vu), ...].

    metal: optional callable metal(z_mm, u_mm) -> bool. When the ion steps
    into metal, the step is linearly interpolated back to the |u| surface it
    crossed and the ion is marked impact='electrode'.

    (PRE-EXISTING DEFECT: these were TWO
    adjacent docstrings. Only the first binds, so `help(fly)` showed the
    metal note and hid the function's actual contract. Merged.)"""
    m = mz_Da * AMU
    v0 = np.sqrt(2.0 * K_eV * E_CHG / m)
    a0 = ang0_mrad * 1e-3
    vz, vu = v0 * np.cos(a0), v0 * np.sin(a0)
    z, u = z0_mm * 1e-3, u0_mm * 1e-3
    t = 0.0
    dt = dt_frac * (h_mm * 1e-3) / v0
    Lz = field.Z[-1] * 1e-3
    qm = E_CHG / m

    planes = sorted(record_planes_mm)
    next_plane = 0
    impact = None
    records = []
    zs, us = [z * 1e3], [u * 1e3]

    def acc(z_m, u_m):
        Ez, Eu = field.E(z_m * 1e3, u_m * 1e3)
        return qm * Ez, qm * Eu

    steps = 0
    while z < Lz and steps < max_steps:
        z_prev, u_prev, t_prev = z, u, t
        vz_prev, vu_prev = vz, vu
        # RK4
        az1, au1 = acc(z, u)
        k1 = (vz, vu, az1, au1)
        az2, au2 = acc(z + 0.5*dt*k1[0], u + 0.5*dt*k1[1])
        k2 = (vz + 0.5*dt*k1[2], vu + 0.5*dt*k1[3], az2, au2)
        az3, au3 = acc(z + 0.5*dt*k2[0], u + 0.5*dt*k2[1])
        k3 = (vz + 0.5*dt*k2[2], vu + 0.5*dt*k2[3], az3, au3)
        az4, au4 = acc(z + dt*k3[0], u + dt*k3[1])
        k4 = (vz + dt*k3[2], vu + dt*k3[3], az4, au4)
        z  += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        u  += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        vz += dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        vu += dt/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        t += dt
        # plane crossings (linear within the step)
        while next_plane < len(planes) and z_prev * 1e3 < planes[next_plane] <= z * 1e3:
            zp = planes[next_plane] * 1e-3
            f = (zp - z_prev) / (z - z_prev)
            records.append(dict(z_mm=planes[next_plane],
                                t_ns=(t_prev + f * dt) * 1e9,
                                u_mm=(u_prev + f * (u - u_prev)) * 1e3,
                                vz=vz_prev + f * (vz - vz_prev),
                                vu=vu_prev + f * (vu - vu_prev)))
            next_plane += 1
        if metal is not None and metal(z * 1e3, u * 1e3):
            # interpolate back to the radial surface crossed during this step
            ua, ub = abs(u_prev) * 1e3, abs(u) * 1e3
            f = 1.0
            for surf in (metal.surfaces if hasattr(metal, "surfaces") else ()):
                if ua < surf <= ub:
                    f = (surf - ua) / (ub - ua); break
            t = t_prev + f * dt
            z = z_prev + f * (z - z_prev)
            u = u_prev + f * (u - u_prev)
            vz = vz_prev + f * (vz - vz_prev)
            vu = vu_prev + f * (vu - vu_prev)
            zs.append(z * 1e3); us.append(u * 1e3)
            impact = "electrode"
            break
        zs.append(z * 1e3); us.append(u * 1e3)
        steps += 1
        if z < -1e-6:          # reflected out the entrance
            break

    # interpolate the exit state back onto z = Lz (last step overshoots)
    if impact is None and z > Lz and z != z_prev:
        f = (Lz - z_prev) / (z - z_prev)
        t = t_prev + f * dt
        u = u_prev + f * (u - u_prev)
        vz = vz_prev + f * (vz - vz_prev)
        vu = vu_prev + f * (vu - vu_prev)
        z = Lz
        zs[-1] = Lz * 1e3; us[-1] = u * 1e3
    KE = 0.5 * m * (vz*vz + vu*vu) / E_CHG
    return dict(z=np.array(zs), u=np.array(us), t_ns=t * 1e9,
                KE_eV=KE, ang_mrad=1e3 * np.arctan2(vu, vz),
                u_exit_mm=u * 1e3, impact=impact,
                records=records, steps=steps)
