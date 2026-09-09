"""
ion_gym.tracer_tw2d — 2-D multi-channel (travelling-wave) tracer.

Port of tracer3d._fly3d to a 2-D field plane: E(x,y,t) = E_A + sum_k
w_k(t) * E_k, one channel per DRIVE GROUP (electrodes sharing a group sum
their unit bases into that channel's field). Waveform evaluation REUSES
tracer3d._wave_eval (K_SIN/K_COS/K_SQUARE/K_TAB_*) — the phase/frequency
conventions are pinned in test_tw2d.py and never re-derived here.

Why this exists (measured, v83): per-ion fly 0.01 s in 2-D vs 9.2 s through
the 3-D channel kernel (662x) — roll-over sweeps and optimizer loops need
the 2-D kernel. A stepped multi-phase SQUARE wave cannot fold into one
quadrature pair, so each group is its own channel here.

Conventions (from the contract):
  * RFGroupSpec.frequency_hz is the WAVEFORM frequency f_waveform;
    v_wave = N_PHASE * PITCH_MM * f_waveform.
  * phase_deg steps of +360/n with cyclic electrode assignment march the
    wave toward +x.
  * 'square' = sign(sin(om t + ph)); amplitude_v applied to UNIT (/V) bases
    at fly time; birth tob shifts the phase an ion is born into.
Kinematics mirror _fly3d exactly: RK4, positions in LOCAL mm, velocities in
m/s internally (recorded in mm/us), HS collisions, all-corners impact rule
with bisection back to the boundary. z is carried inertially (E_z = 0) so a
z-invariant 3-D flight and this 2-D flight are the same dynamical system —
that is Gate B.
"""
import math
import numpy as np
from numba import njit

from ion_gym.physics.tracer3d import (_wave_eval, _mfp_mm, _collide,
                      K_SIN, K_COS, K_SQUARE, E_CHG, AMU)
from ion_gym.physics.collision3d import KB, KG_AMU, gas_mass


# ------------------------------------------------------------- interpolation
@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _bilin(F, x, y, nx, ny):
    """Bilinear sample at grid-unit (x, y); clamps to the edge cell —
    the 2-D restriction of tracer3d._tri."""
    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    if x > nx - 1:
        x = nx - 1.0
    if y > ny - 1:
        y = ny - 1.0
    i = int(x); j = int(y)
    if i > nx - 2:
        i = nx - 2
    if j > ny - 2:
        j = ny - 2
    tx = x - i; ty = y - j
    c0 = F[i, j] * (1 - tx) + F[i + 1, j] * tx
    c1 = F[i, j + 1] * (1 - tx) + F[i + 1, j + 1] * tx
    return c0 * (1 - ty) + c1 * ty


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _in_metal_2d(ele, x, y, nx, ny):
    """ALL-FOUR-corners cell rule (2-D restriction of _in_metal_3d)."""
    if x < 0.0 or y < 0.0 or x > nx - 1 or y > ny - 1:
        return False
    i = int(x); j = int(y)
    if i > nx - 2:
        i = nx - 2
    if j > ny - 2:
        j = ny - 2
    return (ele[i, j] and ele[i + 1, j] and ele[i, j + 1]
            and ele[i + 1, j + 1])


# ------------------------------------------------------------- njit core
@njit(cache=True, fastmath=False, nogil=True)
def _fly_tw2d(qm, x0, y0, z0, vx0, vy0, vz0, tob_us, dt_s, t_max_s,
              EAx, EAy, ExK, EyK,
              ch_kind, ch_om, ch_ph, ch_amp, ch_off, ch_duty, tab_t, tab_v, tab_off,
              ele, h_mm, xs, ys, zs, ts, vxs, vys, vzs, record_every,
              collide_on, T_k, P_pa, sigma_m2, c_star, c_bar, sig1d, m_gas,
              seed):
    """2-D mirror of tracer3d._fly3d. E_z = 0 (z inertial). Returns
    (nrec, kind) with kind 0 = metal impact, 1 = left box, 2 = time out."""
    nx, ny = EAx.shape
    inv_h = 1.0 / h_mm
    K = ch_kind.shape[0]
    m_ion_amu = (E_CHG / qm) / AMU
    ncol = 0
    if collide_on:
        np.random.seed(seed)
    x = x0; y = y0; z = z0
    vx = vx0; vy = vy0; vz = vz0            # m/s
    t = 0.0
    c = 1e3                                 # m/s -> mm/us
    xs[0] = x; ys[0] = y; zs[0] = z; ts[0] = tob_us
    vxs[0] = vx * 1e-3; vys[0] = vy * 1e-3; vzs[0] = vz * 1e-3
    nrec = 1
    step = 0
    kind = 2

    while t < t_max_s:
        xo = x; yo = y; zo = z
        vxo = vx; vyo = vy; to = t
        t_us = tob_us + t * 1e6
        h_us = dt_s * 1e6

        # --- RK4 (inline _acc: field accel at local mm point, LAB us) ---
        gx = x * inv_h; gy = y * inv_h
        ex = _bilin(EAx, gx, gy, nx, ny); ey = _bilin(EAy, gx, gy, nx, ny)
        for kk in range(K):
            w = _wave_eval(ch_kind[kk], ch_om[kk], ch_ph[kk], ch_amp[kk],
                           ch_off[kk], ch_duty[kk], tab_t, tab_v,
                           tab_off[kk], tab_off[kk + 1], t_us)
            ex += w * _bilin(ExK[kk], gx, gy, nx, ny)
            ey += w * _bilin(EyK[kk], gx, gy, nx, ny)
        ax1 = qm * 1e3 * ex; ay1 = qm * 1e3 * ey
        k1x = vx; k1y = vy

        px = x + 0.5 * dt_s * k1x * c; py = y + 0.5 * dt_s * k1y * c
        gx = px * inv_h; gy = py * inv_h
        ex = _bilin(EAx, gx, gy, nx, ny); ey = _bilin(EAy, gx, gy, nx, ny)
        for kk in range(K):
            w = _wave_eval(ch_kind[kk], ch_om[kk], ch_ph[kk], ch_amp[kk],
                           ch_off[kk], ch_duty[kk], tab_t, tab_v,
                           tab_off[kk], tab_off[kk + 1], t_us + 0.5 * h_us)
            ex += w * _bilin(ExK[kk], gx, gy, nx, ny)
            ey += w * _bilin(EyK[kk], gx, gy, nx, ny)
        ax2 = qm * 1e3 * ex; ay2 = qm * 1e3 * ey
        k2x = vx + 0.5 * dt_s * ax1; k2y = vy + 0.5 * dt_s * ay1

        px = x + 0.5 * dt_s * k2x * c; py = y + 0.5 * dt_s * k2y * c
        gx = px * inv_h; gy = py * inv_h
        ex = _bilin(EAx, gx, gy, nx, ny); ey = _bilin(EAy, gx, gy, nx, ny)
        for kk in range(K):
            w = _wave_eval(ch_kind[kk], ch_om[kk], ch_ph[kk], ch_amp[kk],
                           ch_off[kk], ch_duty[kk], tab_t, tab_v,
                           tab_off[kk], tab_off[kk + 1], t_us + 0.5 * h_us)
            ex += w * _bilin(ExK[kk], gx, gy, nx, ny)
            ey += w * _bilin(EyK[kk], gx, gy, nx, ny)
        ax3 = qm * 1e3 * ex; ay3 = qm * 1e3 * ey
        k3x = vx + 0.5 * dt_s * ax2; k3y = vy + 0.5 * dt_s * ay2

        px = x + dt_s * k3x * c; py = y + dt_s * k3y * c
        gx = px * inv_h; gy = py * inv_h
        ex = _bilin(EAx, gx, gy, nx, ny); ey = _bilin(EAy, gx, gy, nx, ny)
        for kk in range(K):
            w = _wave_eval(ch_kind[kk], ch_om[kk], ch_ph[kk], ch_amp[kk],
                           ch_off[kk], ch_duty[kk], tab_t, tab_v,
                           tab_off[kk], tab_off[kk + 1], t_us + h_us)
            ex += w * _bilin(ExK[kk], gx, gy, nx, ny)
            ey += w * _bilin(EyK[kk], gx, gy, nx, ny)
        ax4 = qm * 1e3 * ex; ay4 = qm * 1e3 * ey
        k4x = vx + dt_s * ax3; k4y = vy + dt_s * ay3

        x = x + dt_s / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x) * c
        y = y + dt_s / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y) * c
        z = z + dt_s * vz * c                 # inertial z (E_z = 0)
        vx = vx + dt_s / 6.0 * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
        vy = vy + dt_s / 6.0 * (ay1 + 2 * ay2 + 2 * ay3 + ay4)
        t += dt_s
        step += 1

        if collide_on:
            spx = vx * 1e-3; spy = vy * 1e-3; spz = vz * 1e-3
            sp = math.sqrt(spx * spx + spy * spy + spz * spz)
            if sp < 1e-7:
                sp = 1e-7
            lam = _mfp_mm(sp, T_k, P_pa, sigma_m2, c_star, c_bar)
            if np.random.random() < 1.0 - math.exp(-sp * dt_s * 1e6 / lam):
                nvx, nvy, nvz = _collide(spx, spy, spz, 0.0, 0.0, 0.0,
                                         m_ion_amu, m_gas, sig1d, sp)
                vx = nvx * 1e3; vy = nvy * 1e3; vz = nvz * 1e3
                ncol += 1

        if _in_metal_2d(ele, x * inv_h, y * inv_h, nx, ny):
            f0 = 0.0; f1 = 1.0
            for _ in range(40):
                fm = 0.5 * (f0 + f1)
                if _in_metal_2d(ele, (xo + fm * (x - xo)) * inv_h,
                                (yo + fm * (y - yo)) * inv_h, nx, ny):
                    f1 = fm
                else:
                    f0 = fm
            f = f1
            x = xo + f * (x - xo); y = yo + f * (y - yo)
            z = zo + f * (z - zo)
            vx = vxo + f * (vx - vxo); vy = vyo + f * (vy - vyo)
            t = to + f * dt_s
            kind = 0
            if nrec < xs.shape[0]:
                xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
                vxs[nrec] = vx * 1e-3; vys[nrec] = vy * 1e-3
                vzs[nrec] = vz * 1e-3
                ts[nrec] = tob_us + t * 1e6; nrec += 1
            break

        if (x < 0.0 or y < 0.0 or x > (nx - 1) * h_mm
                or y > (ny - 1) * h_mm):
            kind = 1
            if nrec < xs.shape[0]:
                xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
                vxs[nrec] = vx * 1e-3; vys[nrec] = vy * 1e-3
                vzs[nrec] = vz * 1e-3
                ts[nrec] = tob_us + t * 1e6; nrec += 1
            break

        if step % record_every == 0 and nrec < xs.shape[0]:
            xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
            vxs[nrec] = vx * 1e-3; vys[nrec] = vy * 1e-3
            vzs[nrec] = vz * 1e-3
            ts[nrec] = tob_us + t * 1e6; nrec += 1

    return nrec, kind, ncol


# ------------------------------------------------------------- field pack
def build_tw2d_fields(bases, groups, assign, dc, h_mm, ele=None):
    """{idx: (nx,ny) unit basis /V} + RFGroupSpec drive -> fields dict.

    One channel PER GROUP: electrodes assigned to a group sum their unit
    bases into that channel; the channel waveform is the group's (kind from
    waveform, om = 2*pi*f in rad/us, phase in rad, amp = amplitude_v).
    E arrays are -grad(phi) by central differences (V/mm) — the same
    operation both packs must share for Gate B equivalence."""
    idx0 = next(iter(bases))
    nx, ny = bases[idx0].shape
    A = np.zeros((nx, ny))
    for i, b in bases.items():
        A = A + float(dc.get(i, 0.0)) * b
    by_group = {}
    for i, gname in assign.items():
        if gname is None or i not in bases:
            continue
        by_group.setdefault(gname, np.zeros((nx, ny)))
        by_group[gname] += bases[i]
    gmap = {g.name: g for g in groups}
    names = [n for n in by_group if n in gmap]
    K = len(names)
    ExK = np.zeros((K, nx, ny)); EyK = np.zeros((K, nx, ny))
    ch_kind = np.zeros(K, np.int64); ch_om = np.zeros(K)
    ch_duty = np.full(K, 0.5)
    ch_ph = np.zeros(K); ch_amp = np.zeros(K); ch_off = np.zeros(K)

    def _grad(phi):
        gx, gy = np.gradient(-phi, h_mm)
        return gx, gy
    EAx, EAy = _grad(A)
    for k, n in enumerate(names):
        g = gmap[n]
        ex, ey = _grad(by_group[n])
        ExK[k] = ex; EyK[k] = ey
        wf = getattr(g, "waveform", "sin")
        ch_kind[k] = K_SQUARE if wf == "square" else (
            K_COS if wf == "cos" else K_SIN)
        ch_om[k] = 2 * math.pi * g.frequency_hz * 1e-6      # rad/us
        ch_ph[k] = math.radians(g.phase_deg)
        ch_amp[k] = g.amplitude_v
        ch_duty[k] = float(getattr(g, "duty", 0.5))
    if ele is None:
        ele = np.zeros((nx, ny), bool)
    return dict(EAx=EAx, EAy=EAy, ExK=ExK, EyK=EyK,
                ch_kind=ch_kind, ch_om=ch_om, ch_ph=ch_ph, ch_amp=ch_amp,
                ch_off=ch_off, ch_duty=ch_duty,
                tab_t=np.zeros(1), tab_v=np.zeros(1),
                tab_off=np.zeros(K + 1, np.int64),
                ele=np.ascontiguousarray(ele), h_mm=float(h_mm))


# ------------------------------------------------------------- wrapper
def fly_tw2d(fields, mz_Da, r0_mm, v0_mm_us, tob_us=0.0, dt_ns=1.0,
             t_max_us=50.0, record_every=10, max_records=400000,
             collisions=None, seed=1, ion_label=""):
    """fly3d-compatible flight in the 2-D field plane. Returns the same
    dict keys: x, y, z, vx, vy, vz (mm/us), t_us, tof_us, kind, n_col.
    `ion_label`: names the ion in the birth-in-metal refusal (parity
    with fly3d)."""
    # birth-in-metal refusal, THIS route's impact predicate (the 4-corner
    # cell rule) so the refusal boundary equals the impact boundary.
    # build_tw2d_fields normalizes a missing mask to all-False, in which
    # case nothing can impact and this naturally never fires.
    _ele = np.ascontiguousarray(fields["ele"])
    _h = float(fields["h_mm"])
    if _in_metal_2d(_ele, float(r0_mm[0]) / _h, float(r0_mm[1]) / _h,
                    _ele.shape[0], _ele.shape[1]):
        who = f"{ion_label} " if ion_label else ""
        raise ValueError(
            f"{who}birth at (x={float(r0_mm[0]):.3f}, "
            f"y={float(r0_mm[1]):.3f}) mm (local frame) lies inside "
            f"the effective metal (all-four-corner cell rule, node "
            f"pitch {_h} mm): move the source or shrink its extent")
    qm = E_CHG / (mz_Da * AMU)
    on = collisions is not None and (
        collisions.get("enabled", True) if isinstance(collisions, dict)
        else getattr(collisions, "enabled", True))
    if on:
        g = (collisions.get("gas", "He") if isinstance(collisions, dict)
             else getattr(collisions, "gas", "He"))
        T_k = float(collisions.get("T_k", 298.0)
                    if isinstance(collisions, dict)
                    else getattr(collisions, "T_k", 298.0))
        P_pa = float(collisions.get("P_pa", 0.0)
                     if isinstance(collisions, dict)
                     else getattr(collisions, "P_pa", 0.0))
        sigma = float(collisions.get("sigma_m2", 2.27e-18)
                      if isinstance(collisions, dict)
                      else getattr(collisions, "sigma_m2", 2.27e-18))
        mg = gas_mass(g)
        c_star = math.sqrt(2 * KB * T_k / (mg * KG_AMU)) / 1000.0
        c_bar = math.sqrt(8 * KB * T_k / (math.pi * mg * KG_AMU)) / 1000.0
        sig1d = math.sqrt(KB * T_k / (mg * KG_AMU)) / 1000.0
    else:
        T_k = 298.0; P_pa = 0.0; sigma = 2.27e-18
        mg = 4.0; c_star = c_bar = sig1d = 1.0
    nmax = int(min(max_records,
                   t_max_us * 1e3 / dt_ns / max(record_every, 1) + 8))
    xs = np.empty(nmax); ys = np.empty(nmax); zs = np.empty(nmax)
    ts = np.empty(nmax)
    vxs = np.empty(nmax); vys = np.empty(nmax); vzs = np.empty(nmax)
    nrec, kind, ncol = _fly_tw2d(
        qm, float(r0_mm[0]), float(r0_mm[1]),
        float(r0_mm[2]) if len(r0_mm) > 2 else 0.0,
        float(v0_mm_us[0]) * 1e3, float(v0_mm_us[1]) * 1e3,
        (float(v0_mm_us[2]) if len(v0_mm_us) > 2 else 0.0) * 1e3,
        float(tob_us), dt_ns * 1e-9, t_max_us * 1e-6,
        fields["EAx"], fields["EAy"], fields["ExK"], fields["EyK"],
        fields["ch_kind"], fields["ch_om"], fields["ch_ph"],
        fields["ch_amp"], fields["ch_off"],
        fields.get("ch_duty", np.full(fields["ch_kind"].shape[0], 0.5)),
        fields["tab_t"],
        fields["tab_v"], fields["tab_off"], fields["ele"],
        fields["h_mm"], xs, ys, zs, ts, vxs, vys, vzs,
        int(record_every), on, T_k, P_pa, sigma, c_star, c_bar, sig1d,
        mg, int(seed))
    return dict(x=xs[:nrec], y=ys[:nrec], z=zs[:nrec], t_us=ts[:nrec],
                vx=vxs[:nrec], vy=vys[:nrec], vz=vzs[:nrec],
                tof_us=float(ts[nrec - 1]), kind=int(kind), n_col=int(ncol))
