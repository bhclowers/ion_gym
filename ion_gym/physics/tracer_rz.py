"""
ion_gym.tracer_rz
-----------------
THE r-z tracer — the collision-validated recording kernel every native
cylindrical flight uses: round einzel, IMS drift tube, funnel, the
reflectron TOF assembly. Relocated VERBATIM from a retired
device-named module: a device-named module hosting the universal r-z
tracer was a naming lie; the kernel itself was never funnel-specific.

Provenance carried with the move:
- Body identical to the validated collision kernel plus recording
  (that validation remains the physics gate).
- The impact fix lives here: nearest-node metal predicate
  (_metal_nn) + 40-step bisection backtrack to the impact surface, with
  r recomputed per candidate (the radial coordinate is nonlinear in
  the interpolant).
- Coordinates: x is the axis (mm); the field is 2-D cylindrical on the
  half-plane, so a 3-D ion at (x,y,z) samples radius r = hypot(y,z).
- Recorded E-field is the RAW field (V/mm, the quantity external tools
  reports); the tracer's internal 1e-9*acc scaling is undone at write.

Consumer: build_rz (every native r-z flight). The interim funnel
build half that also flew this kernel is retired.
"""

import math

import numpy as np
from numba import njit

from ion_gym.physics.collision3d import _mfp_mm, _collide
from ion_gym.physics.interp import _bilin
from ion_gym.physics.raster2d import (_metal_nn)
from ion_gym.physics.build_planar import _wave_eval   # THE 2-D waveform


@njit(cache=True, nogil=True)
def _write_row(rec, k, t, x, y, z, vx, vy, vz, ez, er, ncol, m_ion, path,
               eat, ket,
               cs, cke, cke_x, cke_y, cke_z, cef, cea, cer, crad, cncol, cpath, ceat, cket):
    """Write base kinematics + selected optional channels into rec[k].
    Column layout: [t,x,y,z,vx,vy,vz] then the enabled optional channels
    in OPTIONAL_CHANNELS order (speed, ke_ev, e_field, e_axial, e_radial,
    radius, n_col, path_mm, e_axial_tint, ke_tint). Flags cs..cket select
    them; disabled ones are simply not advanced past (caller sized rec to
    the enabled count). `path` (mm), `eat` (V/mm us) and `ket` (eV us)
    are cumulative kernel accumulators: arc length, integral of the RAW
    axial field over time, integral of kinetic energy over time."""
    rec[k, 0] = t
    rec[k, 1] = x
    rec[k, 2] = y
    rec[k, 3] = z
    rec[k, 4] = vx
    rec[k, 5] = vy
    rec[k, 6] = vz
    c = 7
    sp = math.sqrt(vx * vx + vy * vy + vz * vz)
    r = math.sqrt(y * y + z * z)
    if cs:
        rec[k, c] = sp
        c += 1
    if cke:
        # KE in eV: 0.5 m v^2, v in mm/us = 1e3 m/s; m in kg = m_ion*AMU
        rec[k, c] = 0.5 * m_ion * 1.6605402e-27 * (sp * 1e3) ** 2 \
            / 1.602176634e-19
        c += 1
    # Cartesian component KEs: the r-z SOLVE is a half-plane
    # but the kernel state is full 3-D Cartesian (vx, vy, vz are BASE
    # channels), so per-axis KE is exactly defined and matches the
    # planar/3-D routes' definition; ke_x + ke_y + ke_z == ke_ev by
    # construction. x is the solver axial coordinate on this route.
    _kes = 0.5 * m_ion * 1.6605402e-27 * 1e6 / 1.602176634e-19
    if cke_x:
        rec[k, c] = _kes * vx * vx
        c += 1
    if cke_y:
        rec[k, c] = _kes * vy * vy
        c += 1
    if cke_z:
        rec[k, c] = _kes * vz * vz
        c += 1
    if cef:
        rec[k, c] = math.sqrt(ez * ez + er * er)
        c += 1
    if cea:
        rec[k, c] = ez
        c += 1
    if cer:
        rec[k, c] = er
        c += 1
    if crad:
        rec[k, c] = r
        c += 1
    if cncol:
        rec[k, c] = ncol
        c += 1
    if cpath:
        rec[k, c] = path
        c += 1
    if ceat:
        rec[k, c] = eat
        c += 1
    if cket:
        rec[k, c] = ket
        c += 1


@njit(cache=True, nogil=True)
def _fly_rec_full(x, y, z, vx, vy, vz, tob, m_ion, EzA, EuA, EzK, EuK,
                  ch_kind, ch_om, ch_ph, ch_duty, tab_t, tab_v, tab_off,
                  EzG, EuG, tau_gate,
                  ele, u0, mm, acc, dt, t_max_us,
                  T_k, P_pa, sigma, c_star, c_bar, sig1d, m_gas,
                  rec, rec_every, seed,
                  cs, cke, cke_x, cke_y, cke_z, cef, cea, cer, crad, cncol, cpath, ceat, cket,
                  bnd_on, bnd_val, pl_col, pl_val, pl_sgn, pl_w, pl_kind):
    """Records base kinematics + enabled optional channels every
    rec_every steps. E-field stored is in V/mm per the channel contract
    (basis arrays are V/m; the write scale converts). Returns (nrec, kind, ncol).

    DRIVE CHANNELS (L-455): the field is
        E(x, r, t) = E_A + sum_k w_k(t) * E_k + step(t - tau_gate) * E_G
    with per-group basis fields EzK/EuK (amplitude baked in) and w_k the
    UNIT waveform build_planar._wave_eval evaluates — sin, cos,
    square(duty), tables — on the LAB clock. This supersedes the single
    rf_V * sin(om t) * B fold that flew every sin group at one frequency
    and the largest amplitude, and squares as sines."""
    np.random.seed(seed)
    nx, nu = EzA.shape
    K = ch_kind.shape[0]

    def _efield(px, pr, tt):
        gx = px / mm
        gu = (pr - u0) / mm
        ez = _bilin(EzA, gx, gu, nx, nu)
        er = _bilin(EuA, gx, gu, nx, nu)
        for k in range(K):
            w = _wave_eval(ch_kind[k], ch_om[k], ch_ph[k], ch_duty[k],
                           tab_t, tab_v, tab_off[k], tab_off[k + 1], tt)
            ez = ez + w * _bilin(EzK[k], gx, gu, nx, nu)
            er = er + w * _bilin(EuK[k], gx, gu, nx, nu)
        if tau_gate >= 0.0 and tt >= tau_gate:
            ez = ez + _bilin(EzG, gx, gu, nx, nu)
            er = er + _bilin(EuG, gx, gu, nx, nu)
        return ez * 1e-9 * acc, er * 1e-9 * acc
    t = 0.0
    ncol = 0
    step = 0
    # cumulative arc length (mm): sum of per-step |dr|. Within a leapfrog
    # step the position update is a straight segment, so this is exact
    # to the integrator; collisions change velocity, never position.
    path = 0.0
    # time integrals of the axial field (V/mm us; fscale converts) and kinetic
    # energy (eV us): midpoint rule over each dt with the step's field and
    # the post-step velocity; exact per-step, no dependence on rec_every.
    eat = 0.0
    ket = 0.0
    ke_scale = 0.5 * m_ion * 1.6605402e-27 * 1e6 / 1.602176634e-19
    # field at the birth point for row 0 (lab clock = tob)
    r0 = math.sqrt(y * y + z * z)
    ez0, er0 = _efield(x, r0, tob)
    # RECORDED FIELD UNITS: V/mm, per the channel contract (sim_spec
    # OPTIONAL_CHANNELS). The basis arrays EzA/EuA/... are stored in V/m
    # (ionbench.build_field_aware scales the mm-grid gradient by 1e3 so
    # the acceleration path's `E * 1e-9 * acc` is dimensionally exact).
    # fscale therefore does TWO things at once: it un-scales the
    # acceleration factor (1 / (1e-9 * acc)) to recover the raw stored
    # field, and converts that V/m value to the declared V/mm (* 1e-3).
    # Root cause of a real 1000x defect: an earlier note here claiming
    # "changing THIS scale does not move the recorded number" was a
    # confounded experiment — this IS the only write path; the 1000x came
    # from the V/m basis storage meeting the V/mm channel label.
    fscale = 1e-3 / (1e-9 * acc) if acc != 0 else 0.0
    _write_row(rec, 0, tob, x, y, z, vx, vy, vz,
               ez0 * fscale, er0 * fscale, ncol, m_ion, path, eat, ket,
               cs, cke, cke_x, cke_y, cke_z, cef, cea, cer, crad, cncol, cpath, ceat, cket)
    nrec = 1
    kind = 2
    face = -1                     # terminating bound face, -1 = none
    ez = ez0
    er = er0
    while t < t_max_us:
        # old state for the impact backtrack (bisection to the boundary)
        xo = x
        yo = y
        zo = z
        vxo = vx
        vyo = vy
        vzo = vz
        to = t
        r = math.sqrt(y * y + z * z)
        ez, er = _efield(x, r, tob + t)
        rr = r if r > 1e-9 else 1e-9
        vx += 0.5 * ez * dt
        vy += 0.5 * er * y / rr * dt
        vz += 0.5 * er * z / rr * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt
        path += math.sqrt((x - xo) * (x - xo) + (y - yo) * (y - yo)
                          + (z - zo) * (z - zo))
        r = math.sqrt(y * y + z * z)
        ez, er = _efield(x, r, tob + t + dt)
        rr = r if r > 1e-9 else 1e-9
        vx += 0.5 * ez * dt
        vy += 0.5 * er * y / rr * dt
        vz += 0.5 * er * z / rr * dt
        t += dt
        step += 1
        sp = math.sqrt(vx * vx + vy * vy + vz * vz)
        # accumulate BEFORE the collision test: the KE integral is of the
        # free-flight state; the collision then resets the velocity for
        # the next step (positions are continuous through a collision)
        eat += ez * fscale * dt
        ket += ke_scale * sp * sp * dt
        if P_pa > 0.0:
            if sp < 1e-7:
                sp = 1e-7
            lam = _mfp_mm(sp, T_k, P_pa, sigma, c_star, c_bar)
            if np.random.random() < 1.0 - math.exp(-sp * dt / lam):
                vx, vy, vz = _collide(vx, vy, vz, 0.0, 0.0, 0.0,
                                      m_ion, m_gas, sig1d, sp)
                ncol += 1
        if step % rec_every == 0 and nrec < rec.shape[0] - 1:
            _write_row(rec, nrec, tob + t, x, y, z, vx, vy, vz,
                       ez * fscale, er * fscale, ncol, m_ion,
                       path, eat, ket,
                       cs, cke, cke_x, cke_y, cke_z, cef, cea, cer, crad, cncol, cpath, ceat, cket)
            nrec += 1
        r = math.sqrt(y * y + z * z)
        if _metal_nn(ele, x / mm, (r - u0) / mm, nx, nu):
            # bisect back to the impact surface (nearest-node
            # shell; r recomputed per candidate since the
            # radial coordinate is nonlinear in the interpolant).
            f0 = 0.0
            f1 = 1.0
            for _bs in range(40):
                fm = 0.5 * (f0 + f1)
                ym = yo + fm * (y - yo)
                zm = zo + fm * (z - zo)
                rm = math.sqrt(ym * ym + zm * zm)
                if _metal_nn(ele, (xo + fm * (x - xo)) / mm,
                             (rm - u0) / mm, nx, nu):
                    f1 = fm
                else:
                    f0 = fm
            # the last segment was only flown to fraction f1: remove the
            # (1 - f1) part of it from the arc length before moving x,y,z
            path -= (1.0 - f1) * math.sqrt(
                (x - xo) * (x - xo) + (y - yo) * (y - yo)
                + (z - zo) * (z - zo))
            x = xo + f1 * (x - xo)
            y = yo + f1 * (y - yo)
            z = zo + f1 * (z - zo)
            vx = vxo + f1 * (vx - vxo)
            vy = vyo + f1 * (vy - vyo)
            vz = vzo + f1 * (vz - vzo)
            t = to + f1 * dt
            kind = 0
            break
        if x < 0 or r / mm > (nu - 2):
            kind = 1
            break
        # STATION PLANES (fate 5 impact_plane / 6 detect). Identical
        # crossing math, window sense and step ordering to tracer3d and
        # the planar kernel — metal, then box exit, then stations, then
        # declared bounds — from the ONE shared plane builder
        # (physics.stations.station_planes). This kernel carries a full
        # 3-D transverse state (y, z with r = sqrt(y^2+z^2) supplying the
        # cylindrical field lookup only), so a rectangular y/z window
        # needs no r-z-specific reinterpretation: it is the same window
        # the other routes evaluate.
        if pl_col.shape[0] > 0:
            hit_pl = False
            for ip in range(pl_col.shape[0]):
                pcol = pl_col[ip]
                if pcol == 0:
                    cn = x
                    co = xo
                elif pcol == 1:
                    cn = y
                    co = yo
                else:
                    cn = z
                    co = zo
                sgn = pl_sgn[ip]
                val = pl_val[ip]
                if sgn == 0.0:
                    crossed = (cn - val) * (co - val) <= 0.0 and cn != co
                else:
                    crossed = ((cn - val) * sgn >= 0.0
                               and (co - val) * sgn < 0.0)
                if crossed:
                    den = cn - co
                    if den == 0.0:
                        f = 0.0
                    else:
                        f = (val - co) / den
                    if f < 0.0:
                        f = 0.0
                    if f > 1.0:
                        f = 1.0
                    # candidate crossing point FIRST: a pass-window hit
                    # must leave the step untouched
                    xc = xo + f * (x - xo)
                    yc = yo + f * (y - yo)
                    zc = zo + f * (z - zo)
                    if pcol == 0:
                        w1 = yc
                        w2 = zc
                    elif pcol == 1:
                        w1 = xc
                        w2 = zc
                    else:
                        w1 = xc
                        w2 = yc
                    _ins = (pl_w[ip, 0] <= w1 <= pl_w[ip, 1]
                            and pl_w[ip, 2] <= w2 <= pl_w[ip, 3])
                    if pl_kind[ip] == 6:
                        if not _ins:
                            continue    # detector patch: outside passes
                    elif _ins:
                        continue        # plate: inside the aperture passes
                    # the step was only flown to fraction f: remove the
                    # unflown remainder from the arc length, exactly as
                    # the metal-impact backtrack above does
                    path -= (1.0 - f) * math.sqrt(
                        (x - xo) * (x - xo) + (y - yo) * (y - yo)
                        + (z - zo) * (z - zo))
                    x = xc
                    y = yc
                    z = zc
                    vx = vxo + f * (vx - vxo)
                    vy = vyo + f * (vy - vyo)
                    vz = vzo + f * (vz - vzo)
                    t = to + f * dt
                    kind = pl_kind[ip]
                    hit_pl = True
                    break
            if hit_pl:
                break
        # optional bounding/impact planes (fate 3). In r-z, x is axis, the
        # transverse coordinate is (y,z); check y and z against the
        # y-bounds (radial aperture) symmetrically.
        # Decomposed so the kernel reports WHICH face terminated
        # the flight -- fly_staged's seam-vs-bounds discrimination reads
        # it. Same test order the compound form evaluated.
        if bnd_on[0] and x < bnd_val[0]:
            kind = 3
            face = 0
            break
        if bnd_on[1] and x > bnd_val[1]:
            kind = 3
            face = 1
            break
        if bnd_on[2] and y < bnd_val[2]:
            kind = 3
            face = 2
            break
        if bnd_on[3] and y > bnd_val[3]:
            kind = 3
            face = 3
            break
        if bnd_on[4] and z < bnd_val[4]:
            kind = 3
            face = 4
            break
        if bnd_on[5] and z > bnd_val[5]:
            kind = 3
            face = 5
            break
        # AXIAL HIGH-SIDE GRID TERMINATION. This face was the one
        # open edge of the box: x < 0 and radial escape terminated
        # (kind 1), but x past the grid end kept integrating on what
        # _bilin's unclamped fraction made a linear extrapolation of the
        # last two field columns -- fabricated field, growing with
        # distance. Interpolation is defined through node nx-1 exactly,
        # so the flight ends beyond it like every other face. Checked
        # AFTER the bounds so a declared bound AT the grid edge (the
        # shipped einzel's x_max) keeps winning on the shared step, and
        # legitimate downstream flight is the drift extension's job --
        # exact algebra, not extrapolated integration.
        if x / mm > nx - 1:
            kind = 1
            break
    _write_row(rec, nrec, tob + t, x, y, z, vx, vy, vz,
               ez * fscale, er * fscale, ncol, m_ion,
               path, eat, ket,
               cs, cke, cke_x, cke_y, cke_z, cef, cea, cer, crad, cncol, cpath, ceat, cket)
    nrec += 1
    return nrec, kind, ncol, face
