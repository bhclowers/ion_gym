"""
ion_gym.tracer_gas
------------------
RF-field ion tracer WITH the validated HS collision kernel — the cooler
workhorse. The integration body (RK4, field = E_A + s(t) E_B on the
absolute clock, electrode-aware trilinear gradients, 8-corner impact with
bisection) is copied from tracer3d._fly3d verbatim so the ONLY new
physics is the per-step collision test (the HS model applies collisions in
other_actions after each step; we do the same, post-RK4-step).

The buffer gas is a parameter everywhere (mass via collision3d.gas_mass:
preset name or amu; sigma always user-supplied per ion-gas pair; optional
bulk gas-flow velocity — the funnel hook).

Statistical use note: observables from this tracer are ensemble
distributions; dt must satisfy BOTH the RF resolution (~1 ns at 1.1 MHz,
as in all banked flies) and steps_per_MFP >= 20 at the fastest ion speed.
At cooler pressures (a few Pa) the RF condition dominates by ~100x.
"""

# PROVENANCE
#   origin    the HS hard-sphere collision model,
#             companion reference Appelhans & Dahl, Int. J. Mass Spectrom.
#             (SDS/collisional modelling lineage per the v146 handoff);
#             Maxwell-Boltzmann buffer-gas statistics
#   derived   integration body copied VERBATIM from tracer3d._fly3d (the only
#             new physics is the post-step collision test, as HS applies
#             collisions in other_actions after each step); collision kernel
#             from collision3d (_mfp_mm, _collide — C1-gated)
#   verified  by the collision gates (quad cooler retuned to
#             q=0.15/0.3): equilibrium sizes within 2-8% of kT/(m w_sec^2)
#             for He AND N2, cooling taus match the ODE prediction to ~1%,
#             RF-heating ordering N2 > He > 0 — ALL PASS. Plus the one-time
#             retired-c3 IMS cross-check vs reference HS at ~150 Td (100 vs
#             100 ions, distributional): ALL PASS, K_eff 274 cm2/V/s
#             reported.

import math

import numpy as np
from numba import njit

from ion_gym.physics.tracer3d import _tri, _in_metal_3d
from ion_gym.physics.collision3d import _mfp_mm, _collide, KB, KG_AMU, E_CHG, gas_mass

AMU = KG_AMU


@njit(cache=True, nogil=True)
def _fly_rf_gas(qm, m_ion_amu, x0, y0, z0, vx0, vy0, vz0, tob_us, dt_s,
                t_max_s, EAx, EAy, EAz, EBx, EBy, EBz, ele, h_mm,
                rf_V, dc_V, om_rad_us,
                T_k, P_pa, sigma_m2, m_gas_amu, gfx, gfy, gfz,
                rec, record_every, seed):
    """rec: (n_rec, 7) preallocated [t_us, x, y, z, vx, vy, vz(mm/us)].
    Returns (nrec, kind, n_collisions); kind 0 impact, 1 out of box,
    2 time out."""
    np.random.seed(seed)
    nx, ny, nz = EAx.shape
    inv_h = 1.0 / h_mm
    c_star = math.sqrt(2.0 * KB * T_k / (m_gas_amu * AMU)) / 1000.0
    c_bar = math.sqrt(8.0 * KB * T_k / (math.pi * m_gas_amu * AMU)) / 1000.0
    sig1d = math.sqrt(KB * T_k / (m_gas_amu * AMU)) / 1000.0
    x = x0; y = y0; z = z0
    vx = vx0; vy = vy0; vz = vz0          # m/s (tracer convention)
    t = 0.0
    c = 1e3
    nrec = 0
    rec[0, 0] = tob_us
    rec[0, 1] = x; rec[0, 2] = y; rec[0, 3] = z
    rec[0, 4] = vx * 1e-3; rec[0, 5] = vy * 1e-3; rec[0, 6] = vz * 1e-3
    nrec = 1
    step = 0
    kind = 2
    ncol = 0
    while t < t_max_s:
        t_us = tob_us + t * 1e6
        h_us = dt_s * 1e6
        s1 = np.sin(om_rad_us * t_us) * rf_V + dc_V
        s2 = np.sin(om_rad_us * (t_us + 0.5 * h_us)) * rf_V + dc_V
        s4 = np.sin(om_rad_us * (t_us + h_us)) * rf_V + dc_V

        gx = x * inv_h; gy = y * inv_h; gz = z * inv_h
        ax1 = qm * 1e3 * (_tri(EAx, gx, gy, gz, nx, ny, nz) + s1 * _tri(EBx, gx, gy, gz, nx, ny, nz))
        ay1 = qm * 1e3 * (_tri(EAy, gx, gy, gz, nx, ny, nz) + s1 * _tri(EBy, gx, gy, gz, nx, ny, nz))
        az1 = qm * 1e3 * (_tri(EAz, gx, gy, gz, nx, ny, nz) + s1 * _tri(EBz, gx, gy, gz, nx, ny, nz))
        k1x = vx; k1y = vy; k1z = vz
        px = x + 0.5 * dt_s * k1x * c; py = y + 0.5 * dt_s * k1y * c; pz = z + 0.5 * dt_s * k1z * c
        gx = px * inv_h; gy = py * inv_h; gz = pz * inv_h
        ax2 = qm * 1e3 * (_tri(EAx, gx, gy, gz, nx, ny, nz) + s2 * _tri(EBx, gx, gy, gz, nx, ny, nz))
        ay2 = qm * 1e3 * (_tri(EAy, gx, gy, gz, nx, ny, nz) + s2 * _tri(EBy, gx, gy, gz, nx, ny, nz))
        az2 = qm * 1e3 * (_tri(EAz, gx, gy, gz, nx, ny, nz) + s2 * _tri(EBz, gx, gy, gz, nx, ny, nz))
        k2x = vx + 0.5 * dt_s * ax1; k2y = vy + 0.5 * dt_s * ay1; k2z = vz + 0.5 * dt_s * az1
        px = x + 0.5 * dt_s * k2x * c; py = y + 0.5 * dt_s * k2y * c; pz = z + 0.5 * dt_s * k2z * c
        gx = px * inv_h; gy = py * inv_h; gz = pz * inv_h
        ax3 = qm * 1e3 * (_tri(EAx, gx, gy, gz, nx, ny, nz) + s2 * _tri(EBx, gx, gy, gz, nx, ny, nz))
        ay3 = qm * 1e3 * (_tri(EAy, gx, gy, gz, nx, ny, nz) + s2 * _tri(EBy, gx, gy, gz, nx, ny, nz))
        az3 = qm * 1e3 * (_tri(EAz, gx, gy, gz, nx, ny, nz) + s2 * _tri(EBz, gx, gy, gz, nx, ny, nz))
        k3x = vx + 0.5 * dt_s * ax2; k3y = vy + 0.5 * dt_s * ay2; k3z = vz + 0.5 * dt_s * az2
        px = x + dt_s * k3x * c; py = y + dt_s * k3y * c; pz = z + dt_s * k3z * c
        gx = px * inv_h; gy = py * inv_h; gz = pz * inv_h
        ax4 = qm * 1e3 * (_tri(EAx, gx, gy, gz, nx, ny, nz) + s4 * _tri(EBx, gx, gy, gz, nx, ny, nz))
        ay4 = qm * 1e3 * (_tri(EAy, gx, gy, gz, nx, ny, nz) + s4 * _tri(EBy, gx, gy, gz, nx, ny, nz))
        az4 = qm * 1e3 * (_tri(EAz, gx, gy, gz, nx, ny, nz) + s4 * _tri(EBz, gx, gy, gz, nx, ny, nz))
        k4x = vx + dt_s * ax3; k4y = vy + dt_s * ay3; k4z = vz + dt_s * az3

        x = x + dt_s / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x) * c
        y = y + dt_s / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y) * c
        z = z + dt_s / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z) * c
        vx = vx + dt_s / 6.0 * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
        vy = vy + dt_s / 6.0 * (ay1 + 2 * ay2 + 2 * ay3 + ay4)
        vz = vz + dt_s / 6.0 * (az1 + 2 * az2 + 2 * az3 + az4)
        t += dt_s
        step += 1

        # ---- HS collision test (post-step, gas-flow frame, mm/us)
        wx = vx * 1e-3 - gfx
        wy = vy * 1e-3 - gfy
        wz = vz * 1e-3 - gfz
        sp = math.sqrt(wx * wx + wy * wy + wz * wz)
        if sp < 1e-7:
            sp = 1e-7
        lam = _mfp_mm(sp, T_k, P_pa, sigma_m2, c_star, c_bar)
        if np.random.random() < 1.0 - math.exp(-sp * (dt_s * 1e6) / lam):
            wx, wy, wz = _collide(wx, wy, wz, 0.0, 0.0, 0.0,
                                  m_ion_amu, m_gas_amu, sig1d, sp)
            vx = (wx + gfx) * 1e3
            vy = (wy + gfy) * 1e3
            vz = (wz + gfz) * 1e3
            ncol += 1

        if _in_metal_3d(ele, x * inv_h, y * inv_h, z * inv_h, nx, ny, nz):
            kind = 0
            break
        if (x < 0 or y < 0 or z < 0 or x * inv_h > nx - 1
                or y * inv_h > ny - 1 or z * inv_h > nz - 1):
            kind = 1
            break
        if step % record_every == 0 and nrec < rec.shape[0]:
            rec[nrec, 0] = tob_us + t * 1e6
            rec[nrec, 1] = x; rec[nrec, 2] = y; rec[nrec, 3] = z
            rec[nrec, 4] = vx * 1e-3; rec[nrec, 5] = vy * 1e-3
            rec[nrec, 6] = vz * 1e-3
            nrec += 1
    return nrec, kind, ncol


def fly_rf_gas(fields, mz_Da, r0_mm, v0_mm_us, tob_us, gas_params,
               dt_ns=1.0, t_max_us=1000.0, record_every=500,
               max_records=100000, seed=1, ion_label=""):
    """One ion in RF fields + gas. fields: the tracer3d dict (EAx..ele,
    h_mm, rf_V, dc_V, om_rad_us). gas_params: dict(T_k, P_pa, sigma_m2,
    gas=<name or amu>, flow_mm_us=(0,0,0)). `ion_label`: names the ion
    in the birth-in-metal refusal (parity with fly3d)."""
    from ion_gym.physics.tracer3d import refuse_birth_in_metal_3d
    refuse_birth_in_metal_3d(
        fields["ele"], fields["h_mm"],
        float(r0_mm[0]), float(r0_mm[1]), float(r0_mm[2]),
        ion_label=ion_label,
        world_off_mm=fields.get("world_off_mm", (0.0, 0.0, 0.0)))
    m = mz_Da * AMU
    qm = E_CHG / m
    mg = gas_mass(gas_params.get("gas", "He"))
    fl = gas_params.get("flow_mm_us", (0.0, 0.0, 0.0))
    rec = np.empty((max_records, 7))
    nrec, kind, ncol = _fly_rf_gas(
        qm, mz_Da, float(r0_mm[0]), float(r0_mm[1]), float(r0_mm[2]),
        v0_mm_us[0] * 1e3, v0_mm_us[1] * 1e3, v0_mm_us[2] * 1e3,
        float(tob_us), dt_ns * 1e-9, t_max_us * 1e-6,
        fields["EAx"], fields["EAy"], fields["EAz"],
        fields["EBx"], fields["EBy"], fields["EBz"],
        fields["ele"], fields["h_mm"], fields["rf_V"], fields["dc_V"],
        fields["om_rad_us"], gas_params["T_k"], gas_params["P_pa"],
        gas_params["sigma_m2"], mg, fl[0], fl[1], fl[2],
        rec, record_every, seed)
    return dict(rec=rec[:nrec].copy(), kind=kind, n_collisions=ncol)
