"""
ion_gym.collision3d
-------------------
Hard-sphere (HS) ion-neutral collision kernel. The conventions below
are stated on their own kinetic-theory authority and each is verified by
this repo's own gates (thermalization, transport, and the collision
checks); the model
stands or falls on these equations and their measured validation, not on
whose implementation was read first:

  * MFP: lambda = kT * (v_ion / c_bar_rel) / (P * sigma), with the EXACT
    Maxwell mean relative speed
      c_bar_rel = c_bar_gas * [ (s + 1/(2s)) (sqrt(pi)/2) erf(s)
                                + exp(-s^2)/2 ],   s = v_ion / c_star,
    c_star = sqrt(2kT/M), c_bar_gas = sqrt(8kT/(pi M)) — not the
    sqrt(v^2 + c_bar^2) shortcut (that source lists it as the approximate
    alternative). Speeds are taken relative to the mean gas-flow frame.
    (One documented deviation: the source lazily recomputes the MFP only on
    5% ion-speed changes as a performance optimization; we compute per
    step — the limit that source approximates. Revisit if the C-3 external
    cross-check ever gates at the level where this matters.)
  * Collision probability per step: P = 1 - exp(-v dt / lambda).
  * Colliding-partner velocity: REJECTION-BIASED Maxwell — draw 3-D
    Gaussian (sigma_1D = sqrt(kT/M)), accept with probability
    |v_gas - v_ion| / (v_ion + 3 sqrt(3) sigma_1D) — implements
    p(v) ~ |v_rel| f_Maxwell(v), the relative-speed flux bias whose
    omission is the classic under-heating bug.
  * Scattering: hard-sphere impact-parameter construction —
    impact_angle alpha = asin(sqrt(U)) from the relative-velocity axis,
    azimuth uniform; ONLY the line-of-centers component attenuates by
    (m - M)/(m + M). The source's polar/rotation sequence reduces exactly to
      v_ion' = v_ion - (2M/(m+M)) (v_rel . r_hat) r_hat,
    with r_hat at angle alpha from v_rel_hat — implemented in that
    closed vector form (verified equivalent; equals isotropic-in-CM for
    hard spheres, as it must).

Units: mm, us, amu, V/mm, K, Pa, m^2 (sigma) — matching that source.
RNG: numpy Generator-compatible via numba's np.random; ensembles run
serially with per-ion seeds for exact reproducibility.
"""

import math

import numpy as np
from numba import njit

KB = 1.3806505e-23        # J/K       (source's value)
KG_AMU = 1.6605402e-27    # kg/amu    (source's value)
E_CHG = 1.602176634e-19   # C

# Buffer-gas presets (amu). The gas is a USER PARAMETER everywhere — He
# is a default, never an assumption. Collision cross-sections are
# ion-gas-PAIR properties and therefore always supplied by the user
# (sigma_m2), not looked up here.
GASES = {"He": 4.002602, "H2": 2.01588, "N2": 28.0134, "air": 28.9647,
         "Ar": 39.948, "CO2": 44.0095, "Kr": 83.798, "Xe": 131.293}


def gas_mass(gas):
    """Resolve a gas spec: preset name (str) or explicit mass in amu."""
    if isinstance(gas, str):
        if gas not in GASES:
            raise KeyError(f"unknown gas '{gas}'; presets: "
                           f"{sorted(GASES)} — or pass a mass in amu")
        return GASES[gas]
    return float(gas)


def gas_kernel_scalars(collisions):
    """Resolve a collision spec (or None) into the scalars every 2-D
    tracer kernel takes: (enabled, T_k, P_pa, sigma_m2, c_star, c_bar,
    sig1d, m_gas).

    ONE derivation, used by every route. Gas is a property of a STAGE,
    not of a geometry class: a drift cell may be r-z and an ion guide
    planar, and either can be gas-filled. Any route-specific copy of
    this arithmetic is a second authority that will drift from this one,
    and the failure is silent -- a stage flown at the wrong pressure
    still completes and still reports a plausible number.

    `None` (or a disabled spec) means vacuum, expressed as P = 0 exactly
    as the kernels spell it, so a vacuum stage needs no special branch.
    The thermal scalars are still returned at a sane reference
    temperature rather than as zeros, because the kernels divide by them.
    """
    enabled = bool(getattr(collisions, "enabled", False)) \
        if collisions is not None else False
    T = float(getattr(collisions, "T_k", 298.0)) if enabled else 298.0
    mg = gas_mass(getattr(collisions, "gas", "He")) if enabled else 4.0
    return dict(
        enabled=enabled,
        T_k=T,
        P_pa=float(getattr(collisions, "P_pa", 0.0)) if enabled else 0.0,
        sigma_m2=float(getattr(collisions, "sigma_m2", 0.0))
        if enabled else 0.0,
        c_star=math.sqrt(2 * KB * T / (mg * KG_AMU)) / 1000.0,
        c_bar=math.sqrt(8 * KB * T / (math.pi * mg * KG_AMU)) / 1000.0,
        sig1d=math.sqrt(KB * T / (mg * KG_AMU)) / 1000.0,
        m_gas=mg,
    )


@njit(cache=True, nogil=True)
def _c_bar_rel(speed_mm_us, c_star, c_bar):
    """Exact Maxwell mean relative speed (mm/us)."""
    s = speed_mm_us / c_star
    if s < 1e-12:
        return c_bar
    return c_bar * ((s + 1.0 / (2.0 * s)) * 0.5 * math.sqrt(math.pi)
                    * math.erf(s) + 0.5 * math.exp(-s * s))


@njit(cache=True, nogil=True)
def _mfp_mm(speed_mm_us, T_k, P_pa, sigma_m2, c_star, c_bar):
    cbr = _c_bar_rel(speed_mm_us, c_star, c_bar)
    return 1000.0 * KB * T_k * (speed_mm_us / cbr) / (P_pa * sigma_m2)


@njit(cache=True, nogil=True)
def _collide(vx, vy, vz, ugx, ugy, ugz, m_ion, m_gas, sig1d, speed_rel):
    """One elastic hard-sphere collision. (vx,vy,vz) ion velocity in the
    gas-flow frame; returns the post-collision ion velocity (same frame).
    (ugx..) unused slot kept for signature stability."""
    # rejection-biased Maxwell partner
    scale = speed_rel + sig1d * 5.196152422706632   # 3*sqrt(3)
    gx = gy = gz = 0.0
    while True:
        gx = np.random.normal() * sig1d
        gy = np.random.normal() * sig1d
        gz = np.random.normal() * sig1d
        rx, ry, rz = vx - gx, vy - gy, vz - gz
        rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
        if np.random.random() < rlen / scale:
            break
    if rlen < 1e-12:
        return vx, vy, vz
    # r_hat at impact angle from the relative-velocity axis
    alpha = math.asin(math.sqrt(0.999999999 * np.random.random()))
    theta = 2.0 * math.pi * np.random.random()
    ux, uy, uz = rx / rlen, ry / rlen, rz / rlen
    # perpendicular basis
    if abs(ux) < 0.9:
        px, py, pz = 0.0, -uz, uy
    else:
        px, py, pz = -uz, 0.0, ux
    pl = math.sqrt(px * px + py * py + pz * pz)
    px, py, pz = px / pl, py / pl, pz / pl
    qx = uy * pz - uz * py
    qy = uz * px - ux * pz
    qz = ux * py - uy * px
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    rhx = ca * ux + sa * (ct * px + st * qx)
    rhy = ca * uy + sa * (ct * py + st * qy)
    rhz = ca * uz + sa * (ct * pz + st * qz)
    # attenuate line-of-centers component: v' = v - (2M/(m+M))(v_rel.r)r
    f = 2.0 * m_gas / (m_ion + m_gas) * (rx * rhx + ry * rhy + rz * rhz)
    return vx - f * rhx, vy - f * rhy, vz - f * rhz


@njit(cache=True, nogil=True)
def fly_gas_uniform(m_ion, q_e, r0, v0, E_vmm, T_k, P_pa, sigma_m2,
                    m_gas, gas_flow, dt_us, n_steps, record_every, seed):
    """One ion in uniform field + HS gas. Records every record_every
    steps: t, x, y, z, vx, vy, vz. Returns (rec, n_collisions)."""
    np.random.seed(seed)
    c_star = math.sqrt(2.0 * KB * T_k / (m_gas * KG_AMU)) / 1000.0
    c_bar = math.sqrt(8.0 * KB * T_k / (math.pi * m_gas * KG_AMU)) / 1000.0
    sig1d = math.sqrt(KB * T_k / (m_gas * KG_AMU)) / 1000.0
    # acceleration mm/us^2 = q E / m : (C * V/mm) / kg -> m/s^2 = mm/us^2*1e-9? 
    # a[mm/us^2] = q_e*E_CHG * E[V/mm]*1e3[V/m per V/mm... careful]
    # E V/mm = 1e3 V/m; F = qE -> a m/s^2 = q*1e3*E/m_kg; mm/us^2 = 1e-3 m/us^2
    # = 1e-3 * 1e-12 m/s^2 ... 1 mm/us^2 = 1e9 m/s^2
    acc = q_e * E_CHG * 1e3 / (m_ion * KG_AMU) * 1e-9   # (V/mm)->mm/us^2
    ax, ay, az = acc * E_vmm[0], acc * E_vmm[1], acc * E_vmm[2]
    x, y, z = r0[0], r0[1], r0[2]
    vx, vy, vz = v0[0], v0[1], v0[2]
    n_rec = n_steps // record_every + 1
    rec = np.empty((n_rec, 7))
    ncol = 0
    ir = 0
    for step in range(n_steps):
        if step % record_every == 0:
            rec[ir, 0] = step * dt_us
            rec[ir, 1], rec[ir, 2], rec[ir, 3] = x, y, z
            rec[ir, 4], rec[ir, 5], rec[ir, 6] = vx, vy, vz
            ir += 1
        # velocity Verlet (uniform field: exact)
        vx += 0.5 * ax * dt_us
        vy += 0.5 * ay * dt_us
        vz += 0.5 * az * dt_us
        x += vx * dt_us
        y += vy * dt_us
        z += vz * dt_us
        vx += 0.5 * ax * dt_us
        vy += 0.5 * ay * dt_us
        vz += 0.5 * az * dt_us
        # collision test in the gas-flow frame
        wx, wy, wz = vx - gas_flow[0], vy - gas_flow[1], vz - gas_flow[2]
        sp = math.sqrt(wx * wx + wy * wy + wz * wz)
        if sp < 1e-7:
            sp = 1e-7
        lam = _mfp_mm(sp, T_k, P_pa, sigma_m2, c_star, c_bar)
        if np.random.random() < 1.0 - math.exp(-sp * dt_us / lam):
            wx, wy, wz = _collide(wx, wy, wz, 0.0, 0.0, 0.0,
                                  m_ion, m_gas, sig1d, sp)
            vx = wx + gas_flow[0]
            vy = wy + gas_flow[1]
            vz = wz + gas_flow[2]
            ncol += 1
    rec[ir, 0] = n_steps * dt_us
    rec[ir, 1], rec[ir, 2], rec[ir, 3] = x, y, z
    rec[ir, 4], rec[ir, 5], rec[ir, 6] = vx, vy, vz
    return rec[:ir + 1], ncol


def ensemble(n_ions, m_ion, q_e, r0, v0, E_vmm, T_k, P_pa, sigma_m2,
             gas="He", gas_flow=(0.0, 0.0, 0.0), dt_us=5e-4,
             t_us=100.0, record_every=200, seed0=1234):
    m_gas = gas_mass(gas)
    """Serial deterministic ensemble; returns (recs list, collisions).
    gas: preset name ('He','N2','Ar',...) or explicit mass in amu."""
    n_steps = int(t_us / dt_us)
    recs, cols = [], []
    E = np.asarray(E_vmm, float)
    gf = np.asarray(gas_flow, float)
    for i in range(n_ions):
        r, nc = fly_gas_uniform(m_ion, q_e, np.asarray(r0, float),
                                np.asarray(v0, float), E, T_k, P_pa,
                                sigma_m2, m_gas, gf, dt_us, n_steps,
                                record_every, seed0 + i)
        recs.append(r)
        cols.append(nc)
    return recs, np.array(cols)
