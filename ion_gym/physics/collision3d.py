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

# Constants come from the one authority (PI ruling 2026-10-13, R1: CODATA
# 2018 everywhere). This module previously carried the source's kB/amu
# (1.3806505e-23, 1.6605402e-27); eight physics modules import these names.
from ion_gym.physics.constants import K_B as KB, KG_AMU, E_CHG  # noqa: E402,F401
from ion_gym.io.sim_spec import CollisionSpec   # the one gas default (R3)

# Buffer-gas presets (amu). The gas is a USER PARAMETER everywhere; the
# one default is CollisionSpec's (see resolve_collisions). Collision
# cross-sections are ion-gas-PAIR properties and therefore always
# supplied by the user (sigma_m2), not looked up here. "air" is the
# standard 28.9647 for a deck that names air; sds.M_AIR is a different
# number with a different job (the SDS tables' reference gas).
GASES = {"He": 4.002602, "H2": 2.01588, "N2": 28.0134, "air": 28.9647,
         "Ar": 39.948, "CO2": 44.0095, "Kr": 83.798, "Xe": 131.293}

# What a VACUUM stage hands the kernels. NOT a gas default and never read
# as physics: P_pa = 0 means no collision is ever drawn. The thermal
# scalars stay finite (some kernels divide by them) and keep their pre-A4
# values so every vacuum flight is bit-identical to before.
_VACUUM_REF_T_K = 298.0
_VACUUM_REF_M_GAS = 4.0

# dt guidance for HS: collision occurrence is one draw per step,
# P = 1 - exp(-x) with x = v dt / lambda = dt / tau, and at most ONE
# collision is applied per step. The kept fraction of the true collision
# rate is therefore (1 - exp(-x)) / x: 0.975 at x = 1/20, 0.952 at 1/10,
# 0.87 at 0.29. Lost collisions make the ion too mobile.
# 2026-10-17: raised from 10 to 20 to match SIMION collision_hs1.lua,
# whose tstep_adjust caps every step at MFP / speed / _steps_per_MFP with
# _steps_per_MFP = 20 by default (about a 2.5 % rate undercount). ion_gym
# flies a FIXED dt, so the rule is checked at build time by hs_step_note
# and reported as a loud advisory, never a refusal (PI ruling 2026-10-17).
HS_DT_PER_COLLISION_TIME = 20.0


def hs_step_table(spec, dt_ns):
    """Per-m/z hard-sphere step check, the numbers behind hs_step_note.

    Returns None when the check does not apply (vacuum, SDS, no dt, or a
    cross-section model validate() refuses), else a list of dicts, one per
    declared m/z: mz, sigma_m2 (from ccs.ion_sigma_m2, so a per-ion CCS
    model is honoured), tau_ns (mean collision time of an ion AT REST,
    1 / (n sigma c_bar_gas)), x = dt / tau, and kept = (1 - exp(-x)) / x,
    the fraction of the true collision rate a one-collision-per-step
    kernel applies. A moving ion collides more often, so tau here is an
    UPPER bound and x a lower bound on the real coarseness."""
    from ion_gym.physics.ccs import ion_sigma_m2, validate_sigma_model
    col = resolve_collisions(getattr(spec, "collisions", None))
    if col is None:
        return None
    if str(col.model).lower() != "hs":
        # SDS has no discrete collisions, so a per-step collision
        # probability does not exist to check; nothing is the outcome
        return None
    if dt_ns is None:
        # no step resolved yet: the dt advisories already report an
        # unpinned dt, and there is nothing to compare against
        return None
    if validate_sigma_model(col, spec.source.charge):
        # validate() refuses this deck by name; there is no valid cross
        # section to evaluate, so the check has nothing to add
        return None
    T = float(col.T_k)
    n_gas = float(col.P_pa) / (KB * T)
    m_gas = gas_mass(col.gas) * KG_AMU
    c_bar_mps = math.sqrt(8.0 * KB * T / (math.pi * m_gas))
    dt_s = float(dt_ns) * 1.0e-9
    rows = []
    for mz in spec.source.mz_list:
        sigma = ion_sigma_m2(col, mz, spec.source.charge)
        rate = n_gas * sigma * c_bar_mps
        x = dt_s * rate
        rows.append(dict(mz=float(mz), sigma_m2=sigma,
                         tau_ns=1.0e9 / rate, x=x,
                         kept=(1.0 - math.exp(-x)) / x))
    return rows


def hs_step_note(spec, dt_ns):
    """Loud advisory when a hard-sphere step is coarse against the mean
    collision time (any declared m/z above dt/tau = 1/HS_DT_PER_COLLISION_
    TIME), or None when it is not. Numbers from hs_step_table, evaluated
    for an ion at rest: a LOWER bound on the problem, so the note's
    absence does not certify a field-driven ion. Warns, never refuses
    (PI ruling 2026-10-17)."""
    rows = hs_step_table(spec, dt_ns)
    if rows is None:
        # the check does not apply; hs_step_table states why per case
        return None
    limit = 1.0 / HS_DT_PER_COLLISION_TIME
    worst = max(r["x"] for r in rows)
    if worst <= limit:
        return None
    col = resolve_collisions(spec.collisions)
    dt_ok = float(dt_ns) * limit / worst
    parts = [f"m/z {r['mz']:g}: tau {r['tau_ns']:.2f} ns, dt/tau "
             f"{r['x']:.3f}, rate kept {r['kept']:.3f}" for r in rows]
    return (f"WARNING collisions: hard-sphere step dt = {float(dt_ns):g} ns "
            f"is coarse against the mean collision time at "
            f"{float(col.P_torr):g} Torr {col.gas}, {float(col.T_k):g} K. "
            f"At most one collision is applied per step, so the collision "
            f"rate is undercounted and ions come out too mobile. For an ion "
            f"at rest (a LOWER bound; moving ions collide more often): "
            + "; ".join(parts)
            + f". The rule dt <= tau / {HS_DT_PER_COLLISION_TIME:g} (SIMION "
            f"collision_hs1 default) asks for dt <= {dt_ok:.3g} ns here. "
            f"Fly at that step, or demonstrate a dt plateau for the "
            f"observable being reported.")


def gas_mass(gas):
    """Resolve a gas spec: preset name (str) or explicit mass in amu."""
    if isinstance(gas, str):
        if gas not in GASES:
            raise KeyError(f"unknown gas '{gas}'; presets: "
                           f"{sorted(GASES)} — or pass a mass in amu")
        return GASES[gas]
    return float(gas)


def resolve_collisions(collisions):
    """THE gas authority (L-531; PI ruling 2026-10-14, R3): a collisions
    argument in any accepted form -> a CollisionSpec when gas is ON, or
    None for vacuum.

    - None, or enabled=False: vacuum. No gas, temperature or pressure is
      resolved, so nothing is defaulted.
    - a dict (the tw2d / tool form): loaded through CollisionSpec.from_dict,
      so a missing field takes the SPEC's value (N2, 273 K, its P_torr and
      sigma_m2) -- the one default -- and P_torr is honored, not ignored.
    - a CollisionSpec: returned as is.

    Routes call this (or gas_kernel_scalars) instead of keeping their own
    getattr fallbacks; a second fallback is a second authority."""
    if collisions is None:
        return None
    if isinstance(collisions, dict):
        collisions = CollisionSpec.from_dict(collisions)
    if not isinstance(collisions, CollisionSpec):
        raise TypeError(
            f"collisions must be None, a dict or a CollisionSpec, got "
            f"{type(collisions).__name__}")
    return collisions if collisions.enabled else None


def gas_kernel_scalars(collisions):
    """Resolve a collision spec (or None) into the scalars every 2-D
    tracer kernel takes: (enabled, T_k, P_pa, c_star, c_bar, sig1d,
    m_gas).

    The hard-sphere CROSS SECTION is deliberately NOT here (2026-10-17):
    it is a per-ION quantity, and physics.ccs.ion_sigma_m2(collisions,
    mz, charge) is its one authority. Leaving a run-wide sigma_m2 in this
    dict would let a route read it and silently fly every ion with one
    size under sigma_model='ccs_fit'.

    ONE derivation, used by every route. Gas is a property of a STAGE,
    not of a geometry class: a drift cell may be r-z and an ion guide
    planar, and either can be gas-filled. Any route-specific copy of
    this arithmetic is a second authority that will drift from this one,
    and the failure is silent -- a stage flown at the wrong pressure
    still completes and still reports a plausible number.

    `None` (or a disabled spec) means vacuum, expressed as P = 0 exactly
    as the kernels spell it, so a vacuum stage needs no special branch.
    The thermal scalars are the inert _VACUUM_REF_* placeholders rather
    than zeros, because the kernels divide by them.
    """
    col = resolve_collisions(collisions)
    if col is None:
        T, mg = _VACUUM_REF_T_K, _VACUUM_REF_M_GAS
        P_pa = 0.0
    else:
        T, mg = float(col.T_k), gas_mass(col.gas)
        P_pa = float(col.P_pa)
    return dict(
        enabled=col is not None,
        T_k=T,
        P_pa=P_pa,
        c_star=math.sqrt(2 * KB * T / (mg * KG_AMU)) / 1000.0,
        c_bar=math.sqrt(8 * KB * T / (math.pi * mg * KG_AMU)) / 1000.0,
        sig1d=math.sqrt(KB * T / (mg * KG_AMU)) / 1000.0,
        m_gas=mg,
    )


def gas_flow(collisions):
    """THE declared bulk gas velocity (fx, fy, fz) in mm/us, from
    collisions.flow_mm_us through resolve_collisions -- (0, 0, 0) for
    vacuum, where there is no gas to move. A malformed declaration (not
    three numbers) REFUSES by name. Every route and staged region reads
    the flow here (L-563: it was read only by the SDS paths, so every
    hard-sphere route flew a declared flow as still gas)."""
    col = resolve_collisions(collisions)
    if col is None:
        return (0.0, 0.0, 0.0)
    flow = tuple(float(v) for v in col.flow_mm_us)
    if len(flow) != 3:
        raise ValueError(f"collisions.flow_mm_us must be 3 values (mm/us), "
                         f"got {list(col.flow_mm_us)}")
    return flow


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
def hs_collision_step(vx, vy, vz, gfx, gfy, gfz, dt, dt_to_us, T_k, P_pa,
                      sigma_m2, c_star, c_bar, m_ion, m_gas, sig1d):
    """ONE hard-sphere collision opportunity over a step, in the GAS-FLOW
    frame: the collision test and the scatter use the ion's velocity
    relative to the moving gas (gfx, gfy, gfz; mm/us), and the result is
    shifted back to the lab frame. Exactly fly_gas_uniform's step below,
    shared by every kernel so the physics has one copy (L-563).

    Velocities in mm/us. The step is dt in the CALLER's unit with
    dt_to_us its factor to microseconds (1.0 for the mm/us kernels, 1e6
    for those stepping in seconds), applied as (sp * dt) * dt_to_us --
    the order each kernel already evaluated. Returns (vx, vy, vz,
    collided). With zero flow every operation reduces to the still-gas
    step the kernels carried (v - 0.0 is v), in the same random-draw
    order, so zero-flow flights are bit-identical to before."""
    wx = vx - gfx
    wy = vy - gfy
    wz = vz - gfz
    sp = math.sqrt(wx * wx + wy * wy + wz * wz)
    if sp < 1e-7:
        sp = 1e-7
    lam = _mfp_mm(sp, T_k, P_pa, sigma_m2, c_star, c_bar)
    if np.random.random() < 1.0 - math.exp(-sp * dt * dt_to_us / lam):
        wx, wy, wz = _collide(wx, wy, wz, 0.0, 0.0, 0.0,
                              m_ion, m_gas, sig1d, sp)
        return wx + gfx, wy + gfy, wz + gfz, True
    return vx, vy, vz, False


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
