"""
ion_gym.tracer_tdep
-------------------
Time-dependent tracer: fast-adjust superposition + voltages(t).

A time-domain flight evaluates, at every step, a field that is a
weighted sum of per-electrode unit-voltage basis grids:

    phi(x, t) = phi_base(x) + sum_i  V_i(t)/V_unit * phi_i(x)          (V_unit=1e4)

We port that. The basis fields are pre-differentiated to V/m ONCE (each
mirror-extended exactly like FieldNumba), and the kernel forms the E field per
RK4 substep as E_base + sum_i (V_i(t)/1e4) E_i. For the buncher there is one
driven electrode and a single step waveform, but the machinery is general.

Two node-centred conventions preserved:
  * SWITCH TIME lands on a step boundary. A stepped tstep_adjust clamps
    the step so it ends exactly at switch_time; we do the same — when a step
    would cross a waveform breakpoint, it is shortened to land on it, so the
    discontinuity is never integrated across.
  * PER-ION CLOCK. The waveform is a function of the ion's own
    time_of_flight (with a possible time-of-birth offset), matching
    the per-ion time-of-flight clock convention.

fastmath stays OFF (bit-reproducibility with the DC path).
"""

# PROVENANCE
#   origin    time-domain fast-adjust superposition
#             (phi = base + sum_i V_i(t)/1e4 * phi_i, V_UNIT = 1e4 V is the
#             the unit-voltage basis convention); a stepped tstep adjust (a step
#             lands exactly on a waveform breakpoint) and ion_time_of_flight
#             (per-ion clock)
#   derived   RK4 body and bilinear sampling from tracer_numba (build_field,
#             _sample); fastmath OFF for bit-reproducibility with the DC path
#   verified  unknown — the legacy validate_buncher golden is data-blocked
#             (the original buncher basis set was never supplied); native
#             reconstruction is a known open item

import numpy as np
from numba import njit

from ion_gym.physics.tracer_numba import build_field, _sample

E_CHG = 1.602176634e-19
AMU = 1.66053907e-27
V_UNIT = 1.0e4          # basis grids are 10 kV unit solutions


class TDepField:
    """Holds base + basis V/m arrays on a shared mirror-extended grid.
    voltages(t_us) -> 1D array of the N basis electrode voltages at time t."""
    def __init__(self, Z, U, phi_base, phi_basis, symmetry="cylindrical"):
        Zg, Ug, Ez0, Eu0 = build_field(Z, U, phi_base, symmetry)
        self.Z, self.U = Zg, Ug
        self.Ez0, self.Eu0 = Ez0, Eu0
        Ezs, Eus, phis = [], [], []
        for pb in phi_basis:
            _, _, ez, eu = build_field(Z, U, pb, symmetry)
            Ezs.append(ez); Eus.append(eu); phis.append(pb)
        self.Ez_basis = np.ascontiguousarray(np.stack(Ezs))   # (N, nz, nu)
        self.Eu_basis = np.ascontiguousarray(np.stack(Eus))
        self.phi_base = phi_base
        self.phi_basis = phis
        self.symmetry = symmetry

    def phi_at(self, voltages):
        """Reconstruct the scalar potential on the NATIVE grid for plotting."""
        p = self.phi_base.copy()
        for V, pb in zip(voltages, self.phi_basis):
            p = p + (V / V_UNIT) * pb
        return p


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _sample_tdep(Zg, Ug, Ez0, Eu0, Ez_b, Eu_b, volts, z_mm, u_mm):
    """E field (V/m) = base + sum_i volts[i]/V_UNIT * basis_i, bilinear."""
    ez = _sample(Zg, Ug, Ez0, z_mm, u_mm)
    eu = _sample(Zg, Ug, Eu0, z_mm, u_mm)
    n = volts.shape[0]
    for i in range(n):
        w = volts[i] / 1.0e4
        if w != 0.0:
            ez += w * _sample(Zg, Ug, Ez_b[i], z_mm, u_mm)
            eu += w * _sample(Zg, Ug, Eu_b[i], z_mm, u_mm)
    return ez, eu


@njit(cache=True, fastmath=False, nogil=True)
def _fly_tdep_core(Zg, Ug, Ez0, Eu0, Ez_b, Eu_b, qm, vz0, vu0, z0_m, u0_m,
                   dt0, Lz_m, max_steps, tob_us, breakpts_us, wave_levels,
                   z_hist, u_hist):
    """wave_levels: (n_break+1, N) voltage of each electrode on each segment;
    breakpts_us: sorted interior breakpoints (switch times), in the ION clock.
    The active segment index advances as tof crosses each breakpoint; a step is
    shortened so it lands exactly on the next breakpoint."""
    vz = vz0; vu = vu0
    z = z0_m; u = u0_m
    tof = 0.0                       # ion clock (us handled via seconds here)
    z_hist[0] = z * 1e3
    u_hist[0] = u * 1e3
    n = 1
    n_break = breakpts_us.shape[0]
    seg = 0
    steps = 0
    z_prev = z; u_prev = u; vz_prev = vz; vu_prev = vu; tof_prev = 0.0

    while z < Lz_m and steps < max_steps:
        z_prev = z; u_prev = u; tof_prev = tof
        vz_prev = vz; vu_prev = vu

        dt = dt0
        # clamp the step so it ends exactly on the next breakpoint (tstep_adjust)
        tof_us = tof * 1e6
        if seg < n_break:
            nb = breakpts_us[seg]
            if tof_us < nb and tof_us + dt * 1e6 > nb:
                dt = (nb - tof_us) * 1e-6

        volts = wave_levels[seg]

        az1 = qm * _sample_tdep(Zg, Ug, Ez0, Eu0, Ez_b, Eu_b, volts, z*1e3, u*1e3)[0]
        au1 = qm * _sample_tdep(Zg, Ug, Ez0, Eu0, Ez_b, Eu_b, volts, z*1e3, u*1e3)[1]
        k1z = vz; k1u = vu
        z2 = z + 0.5*dt*k1z; u2 = u + 0.5*dt*k1u
        e2 = _sample_tdep(Zg, Ug, Ez0, Eu0, Ez_b, Eu_b, volts, z2*1e3, u2*1e3)
        az2 = qm*e2[0]; au2 = qm*e2[1]
        k2z = vz + 0.5*dt*az1; k2u = vu + 0.5*dt*au1
        z3 = z + 0.5*dt*k2z; u3 = u + 0.5*dt*k2u
        e3 = _sample_tdep(Zg, Ug, Ez0, Eu0, Ez_b, Eu_b, volts, z3*1e3, u3*1e3)
        az3 = qm*e3[0]; au3 = qm*e3[1]
        k3z = vz + 0.5*dt*az2; k3u = vu + 0.5*dt*au2
        z4 = z + dt*k3z; u4 = u + dt*k3u
        e4 = _sample_tdep(Zg, Ug, Ez0, Eu0, Ez_b, Eu_b, volts, z4*1e3, u4*1e3)
        az4 = qm*e4[0]; au4 = qm*e4[1]
        k4z = vz + dt*az3; k4u = vu + dt*au3

        z += dt/6.0*(k1z + 2*k2z + 2*k3z + k4z)
        u += dt/6.0*(k1u + 2*k2u + 2*k3u + k4u)
        vz += dt/6.0*(az1 + 2*az2 + 2*az3 + az4)
        vu += dt/6.0*(au1 + 2*au2 + 2*au3 + au4)
        tof += dt

        # advance segment if we've reached/passed the breakpoint
        if seg < n_break and tof * 1e6 >= breakpts_us[seg] - 1e-15:
            seg += 1

        z_hist[n] = z * 1e3
        u_hist[n] = u * 1e3
        n += 1
        steps += 1
        if z < -1e-6:
            break

    # exit clamp to z = Lz (field-free coast handled by caller if beyond the field array)
    if z > Lz_m and z != z_prev:
        f = (Lz_m - z_prev) / (z - z_prev)
        tof = tof_prev + f * (tof - tof_prev)
        u = u_prev + f * (u - u_prev)
        vz = vz_prev + f * (vz - vz_prev)
        vu = vu_prev + f * (vu - vu_prev)
        z = Lz_m
        z_hist[n-1] = Lz_m * 1e3
        u_hist[n-1] = u * 1e3

    return n, z, u, vz, vu, tof, steps


def fly_tdep(field, mz_Da, K_eV, z0_mm, u0_mm, breakpoints_us, wave_levels,
             tob_us=0.0, ang0_mrad=0.0, dt_frac=0.02, h_mm=1.0,
             max_steps=2_000_000):
    """Fly one ion through a time-dependent field.

    breakpoints_us : sorted interior switch times in the ION's own clock.
    wave_levels    : (len(breakpoints)+1, N) electrode voltages per segment.
    tob_us         : time of birth; only matters if the waveform were on a
                     global clock. Here the per-ion time-of-flight clock is used, so the
                     switch is at breakpoints_us regardless of tob — the birth
                     stagger changes WHERE the ion is when the switch hits,
                     which is the physics under test. tob is threaded through
                     for the global-clock case but unused for per-ion clock.
    """
    m = mz_Da * AMU
    v0 = np.sqrt(2.0 * K_eV * E_CHG / m)
    a0 = ang0_mrad * 1e-3
    vz0, vu0 = v0 * np.cos(a0), v0 * np.sin(a0)
    dt0 = dt_frac * (h_mm * 1e-3) / v0
    Lz_m = field.Z[-1] * 1e-3
    qm = E_CHG / m

    bp = np.ascontiguousarray(np.array(breakpoints_us, np.float64))
    wl = np.ascontiguousarray(np.array(wave_levels, np.float64))
    z_hist = np.empty(max_steps + 2, np.float64)
    u_hist = np.empty(max_steps + 2, np.float64)

    (n, z, u, vz, vu, tof, steps) = _fly_tdep_core(
        field.Z, field.U, field.Ez0, field.Eu0, field.Ez_basis, field.Eu_basis,
        qm, vz0, vu0, z0_mm*1e-3, u0_mm*1e-3, dt0, Lz_m, max_steps,
        tob_us, bp, wl, z_hist, u_hist)

    KE = 0.5 * m * (vz*vz + vu*vu) / E_CHG
    return dict(z=z_hist[:n].copy(), u=u_hist[:n].copy(),
                tof_us=tof * 1e6, KE_eV=KE, ang_mrad=1e3*np.arctan2(vu, vz),
                vz_mm_us=vz*1e-3, vu_mm_us=vu*1e-3,
                z_exit_mm=z*1e3, u_exit_mm=u*1e3, steps=steps)
