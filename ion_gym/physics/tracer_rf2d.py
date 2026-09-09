"""
ion_gym.tracer_rf2d
-------------------
2-D TRANSVERSE ion tracer for RF multipole guides (octupole, quad, ...).

Why a new tracer and not tracer_tdep: a multipole guide is a genuine (x, y)
transverse problem — the ion moves independently in x AND y under a rotating/
oscillating field, which the meridional (z, u) tracer cannot represent. But
the field is z-invariant and the axial velocity is unchanged by the transverse
RF (verified on a reference octupole: vz constant to 0 digits, z = vz*t
exactly). So the full 3-D flight collapses to 2-D transverse dynamics with a
trivial z = z0 + vz*t bookkeeping, and time is the independent variable.

Field model (fast-adjust superposition):
    V01(t) = +[sin(w t + theta) * rfvolts + dcvolts]
    V02(t) = -[...]
    E(x, y, t) = V01(t)/1e4 * E_pa1(x,y) + V02(t)/1e4 * E_pa2(x,y)
with w = 2 pi f. pa1/pa2 are the two 10 kV unit-basis arrays (odd/even rods).

Integrator: RK4 in (x, y), field re-evaluated (including the time-varying
waveform) at every substep — this is where RF phase fidelity over many cycles
is won or lost. dt is capped as a fraction of the RF PERIOD (not grid transit),
matching the physics that sets the stiffness. fastmath OFF.

The kernel reuses _sample from tracer_numba (bilinear on the mirror-... no:
planar here, no mirror-extension — the (x,y) plane is real on both sides).
"""

import numpy as np
from numba import njit

from ion_gym.physics.tracer_numba import _sample

E_CHG = 1.602176634e-19
AMU = 1.66053907e-27


def build_planar_field(X, Y, phi):
    """Planar (x,y) field, V/m. No mirror-extension: both transverse axes are
    physical. Returns contiguous X, Y, Ex, Ey."""
    dpx, dpy = np.gradient(phi, X, Y)          # V/mm
    return (np.ascontiguousarray(X, np.float64),
            np.ascontiguousarray(Y, np.float64),
            np.ascontiguousarray(-dpx * 1e3),  # V/m
            np.ascontiguousarray(-dpy * 1e3))


class RFMultipoleField:
    """Holds the per-electrode basis E-fields for an RF multipole.
    basis_phi: list of the N unit (10 kV) node-centred grids on the (X,Y) grid."""
    def __init__(self, X, Y, basis_phi):
        self.X = np.ascontiguousarray(X, np.float64)
        self.Y = np.ascontiguousarray(Y, np.float64)
        Exs, Eys = [], []
        for p in basis_phi:
            _, _, ex, ey = build_planar_field(X, Y, p)
            Exs.append(ex); Eys.append(ey)
        self.Ex_basis = np.ascontiguousarray(np.stack(Exs))
        self.Ey_basis = np.ascontiguousarray(np.stack(Eys))
        self.basis_phi = basis_phi


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _E_at(Xg, Yg, Ex_b, Ey_b, weights, x_mm, y_mm):
    ex = 0.0; ey = 0.0
    for i in range(weights.shape[0]):
        w = weights[i]
        if w != 0.0:
            ex += w * _sample(Xg, Yg, Ex_b[i], x_mm, y_mm)
            ey += w * _sample(Xg, Yg, Ey_b[i], x_mm, y_mm)
    return ex, ey


@njit(cache=True, fastmath=False, nogil=True)
def _fly_rf_core(Xg, Yg, Ex_b, Ey_b, qm, x0_m, y0_m, vx0, vy0,
                 omega_us, theta, rfvolts, dcvolts, rod_sign,
                 dt_us, tof_max_us, r_escape_mm, max_steps,
                 x_hist, y_hist, t_hist):
    """weights[i] = rod_sign[i] * (sin(w t + theta) rf + dc) / 1e4.
    omega_us in rad/us; dt_us in us; positions in metres internally."""
    x = x0_m; y = y0_m
    vx = vx0; vy = vy0            # m/s
    tof = 0.0                     # us
    n = 0
    x_hist[0] = x * 1e3; y_hist[0] = y * 1e3; t_hist[0] = 0.0
    n = 1
    nrod = rod_sign.shape[0]
    weights = np.empty(nrod, np.float64)
    dt = dt_us * 1e-6             # s
    steps = 0

    def fill_weights(t_us):
        v = np.sin(omega_us * t_us + theta) * rfvolts + dcvolts
        for i in range(nrod):
            weights[i] = rod_sign[i] * v / 1.0e4
        return weights

    while tof < tof_max_us and steps < max_steps:
        # RK4 with time-varying field; weights evaluated at t, t+dt/2, t+dt
        w1 = fill_weights(tof).copy()
        ax1, ay1 = _E_at(Xg, Yg, Ex_b, Ey_b, w1, x*1e3, y*1e3)
        ax1 *= qm; ay1 *= qm
        k1x, k1y = vx, vy

        w2 = fill_weights(tof + 0.5*dt*1e6).copy()
        x2 = x + 0.5*dt*k1x; y2 = y + 0.5*dt*k1y
        ax2, ay2 = _E_at(Xg, Yg, Ex_b, Ey_b, w2, x2*1e3, y2*1e3)
        ax2 *= qm; ay2 *= qm
        k2x, k2y = vx + 0.5*dt*ax1, vy + 0.5*dt*ay1

        x3 = x + 0.5*dt*k2x; y3 = y + 0.5*dt*k2y
        ax3, ay3 = _E_at(Xg, Yg, Ex_b, Ey_b, w2, x3*1e3, y3*1e3)
        ax3 *= qm; ay3 *= qm
        k3x, k3y = vx + 0.5*dt*ax2, vy + 0.5*dt*ay2

        w4 = fill_weights(tof + dt*1e6).copy()
        x4 = x + dt*k3x; y4 = y + dt*k3y
        ax4, ay4 = _E_at(Xg, Yg, Ex_b, Ey_b, w4, x4*1e3, y4*1e3)
        ax4 *= qm; ay4 *= qm
        k4x, k4y = vx + dt*ax3, vy + dt*ay3

        x += dt/6.0*(k1x + 2*k2x + 2*k3x + k4x)
        y += dt/6.0*(k1y + 2*k2y + 2*k3y + k4y)
        vx += dt/6.0*(ax1 + 2*ax2 + 2*ax3 + ax4)
        vy += dt/6.0*(ay1 + 2*ay2 + 2*ay3 + ay4)
        tof += dt_us

        x_hist[n] = x*1e3; y_hist[n] = y*1e3; t_hist[n] = tof
        n += 1
        steps += 1
        if (x*1e3)**2 + (y*1e3)**2 > r_escape_mm*r_escape_mm:
            break

    return n, x, y, vx, vy, tof, steps


def fly_rf(field, mz_Da, x0_mm, y0_mm, vx0_mm_us, vy0_mm_us,
           freq_hz, rfvolts, dcvolts, rod_sign, phase_deg=0.0,
           tof_max_us=None, vz_mm_us=None, z_len_mm=None,
           steps_per_period=200, r_escape_mm=None, max_steps=4_000_000):
    """Fly one ion transversely. Either give tof_max_us, or (vz_mm_us, z_len_mm)
    to fly the length of the guide. rod_sign: per-basis +/-1 (odd/even rods).
    """
    m = mz_Da * AMU
    qm = E_CHG / m
    if tof_max_us is None:
        tof_max_us = z_len_mm / vz_mm_us
    period_us = 1.0 / (freq_hz * 1e-6)
    dt_us = period_us / steps_per_period
    omega_us = freq_hz * 1e-6 * 2 * np.pi          # rad/us
    theta = phase_deg * np.pi / 180.0
    if r_escape_mm is None:
        r_escape_mm = max(field.X[-1], field.Y[-1])

    N = int(tof_max_us / dt_us) + 4
    x_hist = np.empty(N, np.float64)
    y_hist = np.empty(N, np.float64)
    t_hist = np.empty(N, np.float64)
    rs = np.ascontiguousarray(np.array(rod_sign, np.float64))

    n, x, y, vx, vy, tof, steps = _fly_rf_core(
        field.X, field.Y, field.Ex_basis, field.Ey_basis, qm,
        x0_mm*1e-3, y0_mm*1e-3, vx0_mm_us*1e3, vy0_mm_us*1e3,
        omega_us, theta, rfvolts, dcvolts, rs,
        dt_us, tof_max_us, r_escape_mm, max_steps, x_hist, y_hist, t_hist)

    return dict(x=x_hist[:n].copy(), y=y_hist[:n].copy(), t_us=t_hist[:n].copy(),
                x_mm=x*1e3, y_mm=y*1e3, vx_mm_us=vx*1e-3, vy_mm_us=vy*1e-3,
                tof_us=tof, steps=steps,
                escaped=(x*1e3)**2 + (y*1e3)**2 > r_escape_mm*r_escape_mm)
