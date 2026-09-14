"""
ion_gym.tracer3d
-----------------
Full-3D ion tracer for node-centred grids, validated on the
quad_monolithic article (see internal validation record).

Field model (v63, MULTI-DRIVE CHANNEL KERNEL):

    E(x, t) = E_A(x) + sum_k  w_k(t) * E_k(x)                    (V/mm)

with each channel k a solved fast-adjust basis E_k and a scalar waveform
w_k(t) on the ABSOLUTE / LAB clock (t includes the ion's birth time tob --
the node-centred convention pinned on the buncher and re-confirmed on the quad).
Because the field is linear in electrode voltages and each E_k is a Laplace
basis, this superposition is EXACT, not a pseudopotential approximation; the
tracer never touches pseudopotential (that stays a visualization concept).

Waveform kinds (ch_kind): 0 sin, 1 cos, 2 sign(sin) [square], 3 table-hold,
4 table-linear. Each channel carries an AFFINE envelope w_k = amp*base + off:
the legacy single-sinusoid model s(t)=sin(w t)*rf_V+dc_V is one
sin channel with amp=rf_V, off=dc_V over the single B basis, reproducing the
v61 arithmetic BIT-FOR-BIT (see test_multidrive3d Gate 1). Tables carry volts
directly (amp=1, off=0) and bake to breakpoint arrays before the fly.

Conventions carried from the validated 2-D work:
  * near-electrode field: one-sided differences at metal-adjacent nodes
    (build_field_aware_3d); naive central differences halve the surface field.
  * impact rule: a point is in metal iff ALL EIGHT corner nodes of its
    containing voxel are electrode nodes.
  * fastmath=False everywhere; the njit trilinear sampler is bit-checked
    against scipy RegularGridInterpolator.
  * waveforms are functions of (x, t) only -- NO ion-state feedback.
"""

import math

import numpy as np
from numba import njit

from ion_gym.physics.collision3d import (KB, KG_AMU, gas_mass, _mfp_mm, _collide)
import ion_gym.physics.sds as _sds  # module ref without the package-root edge
from ion_gym.physics.sds import (_diff_dist_steps, _sphere_rand, ion_params,
                 load_diffusion_statistics, load_massdata,
                 N_DIST_COLLISIONS as _NDC)

E_CHG = 1.602176634e-19
AMU = 1.66053906660e-27

# waveform-kind codes (shared with build_planar semantics)
K_SIN, K_COS, K_SQUARE, K_TAB_HOLD, K_TAB_LIN = 0, 1, 2, 3, 4


# ------------------------------------------------------------- field builder
@njit(cache=True, fastmath=False, nogil=True)
def _fa3d_correct(dx, dy, dz, phi, ele, h_mm):
    """Apply the electrode-aware one-sided-difference correction IN PLACE to
    the central-difference gradient (dx,dy,dz). At each METAL node with
    vacuum on exactly one side along an axis, that axis derivative is
    replaced by the one-sided difference INTO the vacuum; vacuum nodes are
    never touched. This is the exact rule the pure-Python loop implemented —
    only the loop is moved into njit, so the numbers are bit-identical while
    the pass drops from a Python walk over every metal node (minutes on an
    8M-node scene: build_field_aware_3d ran 11x per SLIM build and was the
    'resolves on every fly' lag) to a compiled sweep (seconds)."""
    nx, ny, nz = phi.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not ele[i, j, k]:
                    continue
                # x
                vp = (i + 1 < nx) and (not ele[i + 1, j, k])
                vm = (i - 1 >= 0) and (not ele[i - 1, j, k])
                if vp and not vm:
                    dx[i, j, k] = (phi[i + 1, j, k] - phi[i, j, k]) / h_mm
                elif vm and not vp:
                    dx[i, j, k] = (phi[i, j, k] - phi[i - 1, j, k]) / h_mm
                # y
                vp = (j + 1 < ny) and (not ele[i, j + 1, k])
                vm = (j - 1 >= 0) and (not ele[i, j - 1, k])
                if vp and not vm:
                    dy[i, j, k] = (phi[i, j + 1, k] - phi[i, j, k]) / h_mm
                elif vm and not vp:
                    dy[i, j, k] = (phi[i, j, k] - phi[i, j - 1, k]) / h_mm
                # z
                vp = (k + 1 < nz) and (not ele[i, j, k + 1])
                vm = (k - 1 >= 0) and (not ele[i, j, k - 1])
                if vp and not vm:
                    dz[i, j, k] = (phi[i, j, k + 1] - phi[i, j, k]) / h_mm
                elif vm and not vp:
                    dz[i, j, k] = (phi[i, j, k] - phi[i, j, k - 1]) / h_mm


def build_field_aware_3d(phi, ele, h_mm, electrode_aware=True):
    """E = -grad(phi) in V/mm on the node grid. With electrode_aware=True
    (default) applies the validated 3-D rule (ionbench.build_field_aware, tof
    Gate 1): central differences everywhere, then at METAL nodes with vacuum
    on exactly one side along an axis, replace that axis derivative with the
    one-sided difference into the vacuum — ESSENTIAL where ions ride close to
    electrode surfaces (an ion born at the surface is mis-accelerated ~1% of
    gap energy otherwise). With electrode_aware=False it is a plain central-
    difference gradient everywhere (what the retired slim3d builder used):
    correct where ions stay off the surfaces (SLIM central-gap transport) and
    cheaper — it skips the per-metal-node pass entirely. Returns (Ex,Ey,Ez)
    float64, V/mm.

    The metal-node correction runs in an njit kernel (_fa3d_correct); the
    earlier pure-Python loop was the compose cost that made every SLIM fly
    re-lag. np.gradient stays in numpy (vectorized, unsupported in njit)."""
    d = np.gradient(phi, h_mm)            # [d/dx, d/dy, d/dz] V/mm
    dx = np.ascontiguousarray(d[0])
    dy = np.ascontiguousarray(d[1])
    dz = np.ascontiguousarray(d[2])
    if electrode_aware:
        _fa3d_correct(dx, dy, dz, np.ascontiguousarray(phi),
                      np.ascontiguousarray(ele), float(h_mm))
    return -dx, -dy, -dz


# ------------------------------------------------------------- njit sampling
@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _tri(F, x, y, z, nx, ny, nz):
    """Trilinear sample of F at grid-unit coordinates (x,y,z); clamps to the
    box edge cell (matches RegularGridInterpolator on in-box points)."""
    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    if z < 0.0:
        z = 0.0
    if x > nx - 1:
        x = nx - 1.0
    if y > ny - 1:
        y = ny - 1.0
    if z > nz - 1:
        z = nz - 1.0
    i = int(x); j = int(y); k = int(z)
    if i > nx - 2:
        i = nx - 2
    if j > ny - 2:
        j = ny - 2
    if k > nz - 2:
        k = nz - 2
    tx = x - i; ty = y - j; tz = z - k
    c00 = F[i, j, k] * (1 - tx) + F[i + 1, j, k] * tx
    c10 = F[i, j + 1, k] * (1 - tx) + F[i + 1, j + 1, k] * tx
    c01 = F[i, j, k + 1] * (1 - tx) + F[i + 1, j, k + 1] * tx
    c11 = F[i, j + 1, k + 1] * (1 - tx) + F[i + 1, j + 1, k + 1] * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _in_metal_3d(ele, x, y, z, nx, ny, nz):
    """ALL-EIGHT-corners voxel rule at grid-unit coords. Outside box: False."""
    if x < 0.0 or y < 0.0 or z < 0.0 or x > nx - 1 or y > ny - 1 or z > nz - 1:
        return False
    i = int(x); j = int(y); k = int(z)
    if i > nx - 2:
        i = nx - 2
    if j > ny - 2:
        j = ny - 2
    if k > nz - 2:
        k = nz - 2
    return (ele[i, j, k] and ele[i + 1, j, k] and ele[i, j + 1, k]
            and ele[i + 1, j + 1, k] and ele[i, j, k + 1]
            and ele[i + 1, j, k + 1] and ele[i, j + 1, k + 1]
            and ele[i + 1, j + 1, k + 1])


def refuse_birth_in_metal_3d(ele, h_mm, x_mm, y_mm, z_mm, ion_label="",
                             world_off_mm=(0.0, 0.0, 0.0)):
    """3-D birth-refusal parity with the 2-D routes:
    an ion born inside metal used to record a silent impact at step ~1 —
    indistinguishable from physics, masquerading as 0% transmission and
    hiding source/geometry authoring errors (the r-z launch bug's
    disguise). Refuse loudly at birth instead.

    Uses _in_metal_3d — the EXACT predicate the kernels impact on
    (all-eight-corners voxel rule) — so the refusal boundary equals the
    impact boundary; a different test here would refuse ions that fly or
    admit ions that die. The 3-D `ele` is a bool occupancy array (no
    labels), so unlike the 2-D refusal the offending electrode cannot
    be named — the position and the rule are the diagnostic."""
    ele = np.ascontiguousarray(ele)
    nx, ny, nz = ele.shape
    inv_h = 1.0 / float(h_mm)
    if _in_metal_3d(ele, x_mm * inv_h, y_mm * inv_h, z_mm * inv_h,
                    nx, ny, nz):
        who = f"{ion_label} " if ion_label else ""
        # Report in the DECK frame the author writes in (a
        # signed-origin deck once got stored-frame numbers it could not
        # recognize). world_off_mm composes mirror + declared origin;
        # zeros on legacy decks, where both readings coincide.
        ox, oy, oz = (float(v) for v in world_off_mm)
        wx, wy, wz = x_mm + ox, y_mm + oy, z_mm + oz
        stored = ("" if not (ox or oy or oz) else
                  f" [stored frame ({x_mm:.3f}, {y_mm:.3f}, {z_mm:.3f})]")
        raise ValueError(
            f"{who}birth at (x={wx:.3f}, y={wy:.3f}, z={wz:.3f}) mm "
            f"(deck frame){stored} lies inside the effective metal "
            f"(all-eight-corner voxel rule, node pitch {h_mm} mm): move "
            f"the source or shrink its extent")


# ------------------------------------------------------------- waveform eval
@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _wave_eval(kind, om, ph, amp, off, duty, tab_t, tab_v, o0, o1, t):
    """Affine waveform w(t) = amp*base(t) + off of one channel at LAB time t
    (us). base: 0 sin, 1 cos, 2 square, 3 table-hold, 4 table-linear.
    `duty` (squares only): fraction of the period at the HIGH level,
    starting at phase 0. duty == 0.5 evaluates as sign(sin) EXACTLY (the
    frozen-anchored historical formula); any other duty uses the phase-
    fraction test high iff frac((om*t+ph)/2pi) < duty. So 11110000 is
    duty 0.5, 11000000 is duty 0.25, at the same stepped group phases.
    Tables clamp to end values; interior lookup is binary search on the
    channel's [o0, o1) slice of the shared breakpoint arrays. amp/off carry
    the drive amplitude and DC offset; table values already carry
    volts so callers pass amp=1, off=0 for tables."""
    if kind == K_SIN:
        return amp * math.sin(om * t + ph) + off
    if kind == K_COS:
        return amp * math.cos(om * t + ph) + off
    if kind == K_SQUARE:
        if duty == 0.5:
            base = 1.0 if math.sin(om * t + ph) >= 0.0 else -1.0
        else:
            frac = ((om * t + ph) / (2.0 * math.pi)) % 1.0
            base = 1.0 if frac < duty else -1.0
        return amp * base + off
    n = o1 - o0
    if n == 0:
        return off
    if t <= tab_t[o0]:
        return amp * tab_v[o0] + off
    if t >= tab_t[o1 - 1]:
        return amp * tab_v[o1 - 1] + off
    lo = o0
    hi = o1 - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tab_t[mid] <= t:
            lo = mid
        else:
            hi = mid
    if kind == K_TAB_HOLD:
        return amp * tab_v[lo] + off
    f = (t - tab_t[lo]) / (tab_t[hi] - tab_t[lo])
    return amp * (tab_v[lo] + f * (tab_v[hi] - tab_v[lo])) + off


# ------------------------------------------------------------- 3-D integrator
@njit(cache=True, fastmath=False, nogil=True)
def _fly3d(qm, m_ion_amu, x0, y0, z0, vx0, vy0, vz0, tob_us, dt_s, t_max_s,
          EAx, EAy, EAz, ExK, EyK, EzK,
          ch_kind, ch_om, ch_ph, ch_amp, ch_off, ch_duty, tab_t, tab_v, tab_off,
          ele, h_mm, xs, ys, zs, ts, vxs, vys, vzs, exs, eys, ezs,
          record_every,
          collide_on, T_k, P_pa, sigma_m2, c_star, c_bar, sig1d, m_gas, seed,
          pl_col, pl_val, pl_sgn, pl_w, pl_kind,
          tp_col, tp_accept, tp_emit, tp_sgn, tp_maxp, wrs):
    """RK4 in LOCAL mm; field E(x,t) = E_A + sum_k w_k(t)*E_k (V/mm), each
    w_k on the absolute clock in us. Returns (nrec, x,y,z, vx,vy,vz, t_s, kind)
    with kind: 0 = impact on metal, 1 = left array box, 2 = time out,
    3 = crossed an enabled BOUNDING PLANE, 4 = transporter max_passes,
    5 = station impact plane (crossed an impact plane OUTSIDE its
    pass window; sgn 0 on a plane means BOTH crossing directions).
    pl_w is (n,4): [a_lo,a_hi,b_lo,b_hi] over the two transverse axes
    in ascending order; a crossing whose interpolated point lies
    inside that window PASSES. Bounding planes carry an impossible
    window (+inf,-inf), so every crossing kills: bit-identical to the
    pre-window behavior. pl_kind is the fate each plane assigns.

    TRANSPORTER (device-agnostic periodic plane pair):
    tp_col -1 disables; else axis 0/1/2, accept/emit planes in LOCAL mm.
    On crossing tp_accept in direction tp_sgn within a step, the ion AND
    the step's start point (so impact/plane bisection stays in one frame)
    are rigidly shifted by (tp_emit - tp_accept); velocity, clock, and
    every drive phase untouched. Exactness is the W1 field-equivalence
    gate's claim, not the kernel's. wrs stores the cumulative pass count
    at every recorded row; unwrapped = stored + passes*(tp_accept-tp_emit).

    Bounding planes are detected HERE, per step, and the state is interpolated
    to the crossing within the step. They used to be found post-hoc by scanning
    the RECORDED trajectory, which is decimated by record_every -- so the
    reported exit state was the first recorded sample PAST the plane, up to
    record_every*dt of overshoot (~2 mm at rec_every=200, dt=8 ns, 1.3 mm/us).
    That is larger than the transverse spreads the exit statistics measure, so
    the exit spot was dominated by sampling, not by the ion optics. Nor can it
    be fixed by interpolating the recorded points: at 2.4 MHz they are ~4 RF
    cycles apart and the transverse motion between them is not smooth.

    Bit-exactness note: for a single sin channel (amp=rf_V, off=dc_V) the
    per-substep scalar amp*sin(om*t+ph)+off equals the v61 fused scalar
    sin(om*t)*rf_V+dc_V (IEEE mul is commutative, +0.0 phase is identity), and
    the field is assembled as A + w*B in the same association -> array_equal
    with tracer3d_v61_frozen (test_multidrive3d Gate 1)."""
    nx, ny, nz = EAx.shape
    inv_h = 1.0 / h_mm
    K = ch_kind.shape[0]
    # m_ion_amu (true ion mass, Da) is a PARAMETER: it was formerly
    # back-derived as (E_CHG/qm)/AMU, which equals mass/charge and is
    # only correct for charge=1 — at z=2 the collision partner
    # kinematics ran at HALF the ion mass (issues.md 2026-09-09).
    ncol = 0
    if collide_on:
        np.random.seed(seed)
    x = x0; y = y0; z = z0
    vx = vx0; vy = vy0; vz = vz0          # m/s
    t = 0.0                               # elapsed s
    c = 1e3                               # m/s -> mm/us
    xs[0] = x; ys[0] = y; zs[0] = z; ts[0] = tob_us
    vxs[0] = vx * 1e-3; vys[0] = vy * 1e-3; vzs[0] = vz * 1e-3  # mm/us
    nrec = 1
    step = 0
    kind = 2
    tp_d = tp_emit - tp_accept
    tp_pass = 0
    wrs[0] = 0

    # field accel (already * qm*1e3) at local mm point (px,py,pz), LAB us tt
    def _acc(px, py, pz, tt):
        gx = px * inv_h; gy = py * inv_h; gz = pz * inv_h
        ex = _tri(EAx, gx, gy, gz, nx, ny, nz)
        ey = _tri(EAy, gx, gy, gz, nx, ny, nz)
        ez = _tri(EAz, gx, gy, gz, nx, ny, nz)
        for kk in range(K):
            w = _wave_eval(ch_kind[kk], ch_om[kk], ch_ph[kk], ch_amp[kk],
                           ch_off[kk], ch_duty[kk], tab_t, tab_v,
                           tab_off[kk], tab_off[kk + 1], tt)
            ex = ex + w * _tri(ExK[kk], gx, gy, gz, nx, ny, nz)
            ey = ey + w * _tri(EyK[kk], gx, gy, gz, nx, ny, nz)
            ez = ez + w * _tri(EzK[kk], gx, gy, gz, nx, ny, nz)
        return qm * 1e3 * ex, qm * 1e3 * ey, qm * 1e3 * ez

    # RAW field E(x,t) in V/mm at a local mm point, LAB us tt — the same
    # sum _acc uses, but WITHOUT the qm*1e3 accel scaling, for RECORDING
    # the field the ion sees at each stored step (Analysis: e_field etc.).
    # Separate from _acc so the acceleration path is byte-unchanged and the
    # frozen multidrive3d anchor holds.
    def _efield(px, py, pz, tt):
        gx = px * inv_h; gy = py * inv_h; gz = pz * inv_h
        ex = _tri(EAx, gx, gy, gz, nx, ny, nz)
        ey = _tri(EAy, gx, gy, gz, nx, ny, nz)
        ez = _tri(EAz, gx, gy, gz, nx, ny, nz)
        for kk in range(K):
            w = _wave_eval(ch_kind[kk], ch_om[kk], ch_ph[kk], ch_amp[kk],
                           ch_off[kk], ch_duty[kk], tab_t, tab_v,
                           tab_off[kk], tab_off[kk + 1], tt)
            ex = ex + w * _tri(ExK[kk], gx, gy, gz, nx, ny, nz)
            ey = ey + w * _tri(EyK[kk], gx, gy, gz, nx, ny, nz)
            ez = ez + w * _tri(EzK[kk], gx, gy, gz, nx, ny, nz)
        return ex, ey, ez

    # birth-point field (step 0 position/velocity recorded above; the field
    # sampler is only defined here, so fill exs/eys/ezs[0] now)
    _e0x, _e0y, _e0z = _efield(x0, y0, z0, tob_us)
    exs[0] = _e0x; eys[0] = _e0y; ezs[0] = _e0z

    while t < t_max_s:
        xo = x; yo = y; zo = z; vxo = vx; vyo = vy; vzo = vz; to = t

        t_us = tob_us + t * 1e6
        h_us = dt_s * 1e6

        ax1, ay1, az1 = _acc(x, y, z, t_us)
        k1x = vx; k1y = vy; k1z = vz

        px = x + 0.5 * dt_s * k1x * c; py = y + 0.5 * dt_s * k1y * c; pz = z + 0.5 * dt_s * k1z * c
        ax2, ay2, az2 = _acc(px, py, pz, t_us + 0.5 * h_us)
        k2x = vx + 0.5 * dt_s * ax1; k2y = vy + 0.5 * dt_s * ay1; k2z = vz + 0.5 * dt_s * az1

        px = x + 0.5 * dt_s * k2x * c; py = y + 0.5 * dt_s * k2y * c; pz = z + 0.5 * dt_s * k2z * c
        ax3, ay3, az3 = _acc(px, py, pz, t_us + 0.5 * h_us)
        k3x = vx + 0.5 * dt_s * ax2; k3y = vy + 0.5 * dt_s * ay2; k3z = vz + 0.5 * dt_s * az2

        px = x + dt_s * k3x * c; py = y + dt_s * k3y * c; pz = z + dt_s * k3z * c
        ax4, ay4, az4 = _acc(px, py, pz, t_us + h_us)
        k4x = vx + dt_s * ax3; k4y = vy + dt_s * ay3; k4z = vz + dt_s * az3

        x = x + dt_s / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x) * c
        y = y + dt_s / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y) * c
        z = z + dt_s / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z) * c
        vx = vx + dt_s / 6.0 * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
        vy = vy + dt_s / 6.0 * (ay1 + 2 * ay2 + 2 * ay3 + ay4)
        vz = vz + dt_s / 6.0 * (az1 + 2 * az2 + 2 * az3 + az4)
        t += dt_s
        step += 1

        # HS buffer-gas collision (full 3-D; same model as the 2-D kernel,
        # velocities carried in m/s here, converted to mm/us for collision3d).
        if collide_on:
            spx = vx * 1e-3; spy = vy * 1e-3; spz = vz * 1e-3   # mm/us
            sp = math.sqrt(spx * spx + spy * spy + spz * spz)
            if sp < 1e-7:
                sp = 1e-7
            lam = _mfp_mm(sp, T_k, P_pa, sigma_m2, c_star, c_bar)
            if np.random.random() < 1.0 - math.exp(-sp * dt_s * 1e6 / lam):
                nvx, nvy, nvz = _collide(spx, spy, spz, 0.0, 0.0, 0.0,
                                         m_ion_amu, m_gas, sig1d, sp)
                vx = nvx * 1e3; vy = nvy * 1e3; vz = nvz * 1e3
                ncol += 1

        # ---- transporter plane (interior; checked before any terminal test)
        if tp_col >= 0:
            if tp_col == 0:
                cn = x; co = xo
            elif tp_col == 1:
                cn = y; co = yo
            else:
                cn = z; co = zo
            if (cn - tp_accept) * tp_sgn >= 0.0 and (co - tp_accept) * tp_sgn < 0.0:
                tp_pass += 1
                if tp_pass >= tp_maxp:
                    kind = 4
                    if nrec < xs.shape[0]:
                        xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
                        vxs[nrec] = vx * 1e-3; vys[nrec] = vy * 1e-3
                        vzs[nrec] = vz * 1e-3
                        _rec_us = tob_us + t * 1e6
                        exs[nrec], eys[nrec], ezs[nrec] = _efield(x, y, z, _rec_us)
                        ts[nrec] = _rec_us; wrs[nrec] = tp_pass; nrec += 1
                    break
                if tp_col == 0:
                    x += tp_d; xo += tp_d
                elif tp_col == 1:
                    y += tp_d; yo += tp_d
                else:
                    z += tp_d; zo += tp_d

        # impact: all-eight-corners voxel rule, bisect back to the boundary
        if _in_metal_3d(ele, x * inv_h, y * inv_h, z * inv_h, nx, ny, nz):
            f0 = 0.0; f1 = 1.0
            for _ in range(40):
                fm = 0.5 * (f0 + f1)
                if _in_metal_3d(ele, (xo + fm * (x - xo)) * inv_h,
                                (yo + fm * (y - yo)) * inv_h,
                                (zo + fm * (z - zo)) * inv_h, nx, ny, nz):
                    f1 = fm
                else:
                    f0 = fm
            f = f1
            x = xo + f * (x - xo); y = yo + f * (y - yo); z = zo + f * (z - zo)
            vx = vxo + f * (vx - vxo); vy = vyo + f * (vy - vyo)
            vz = vzo + f * (vz - vzo)
            t = to + f * dt_s
            kind = 0
            if nrec < xs.shape[0]:
                xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
                vxs[nrec] = vx * 1e-3; vys[nrec] = vy * 1e-3
                vzs[nrec] = vz * 1e-3
                _rec_us = tob_us + t * 1e6
                exs[nrec], eys[nrec], ezs[nrec] = _efield(x, y, z, _rec_us)
                ts[nrec] = _rec_us; wrs[nrec] = tp_pass; nrec += 1
            break

        # bounding plane crossed? interpolate to the crossing WITHIN the step
        if pl_col.shape[0] > 0:
            hit_pl = False
            for ip in range(pl_col.shape[0]):
                col = pl_col[ip]
                if col == 0:
                    cn = x; co = xo
                elif col == 1:
                    cn = y; co = yo
                else:
                    cn = z; co = zo
                sgn = pl_sgn[ip]
                val = pl_val[ip]
                if sgn == 0.0:
                    crossed = ((cn - val) * (co - val) <= 0.0
                               and cn != co)
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
                    if col == 0:
                        t1 = yc; t2 = zc
                    elif col == 1:
                        t1 = xc; t2 = zc
                    else:
                        t1 = xc; t2 = yc
                    _ins = (pl_w[ip, 0] <= t1 <= pl_w[ip, 1]
                            and pl_w[ip, 2] <= t2 <= pl_w[ip, 3])
                    if pl_kind[ip] == 6:
                        if not _ins:
                            continue    # detector patch: outside
                    elif _ins:          #   the patch passes
                        continue        # plate: inside the aperture
                                        #   window passes
                    x = xc; y = yc; z = zc
                    vx = vxo + f * (vx - vxo)
                    vy = vyo + f * (vy - vyo)
                    vz = vzo + f * (vz - vzo)
                    t = to + f * dt_s
                    kind = pl_kind[ip]
                    hit_pl = True
                    break
            if hit_pl:
                if nrec < xs.shape[0]:
                    xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
                    vxs[nrec] = vx * 1e-3; vys[nrec] = vy * 1e-3
                    vzs[nrec] = vz * 1e-3
                    _rec_us = tob_us + t * 1e6
                    exs[nrec], eys[nrec], ezs[nrec] = _efield(x, y, z, _rec_us)
                    ts[nrec] = _rec_us; nrec += 1
                break

        # left the array box
        if (x < 0.0 or y < 0.0 or z < 0.0 or x > (nx - 1) * h_mm
                or y > (ny - 1) * h_mm or z > (nz - 1) * h_mm):
            kind = 1
            if nrec < xs.shape[0]:
                xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
                vxs[nrec] = vx * 1e-3; vys[nrec] = vy * 1e-3
                vzs[nrec] = vz * 1e-3
                _rec_us = tob_us + t * 1e6
                exs[nrec], eys[nrec], ezs[nrec] = _efield(x, y, z, _rec_us)
                ts[nrec] = _rec_us; wrs[nrec] = tp_pass; nrec += 1
            break

        if step % record_every == 0 and nrec < xs.shape[0]:
            xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
            vxs[nrec] = vx * 1e-3; vys[nrec] = vy * 1e-3
            vzs[nrec] = vz * 1e-3
            wrs[nrec] = tp_pass
            _rec_us = tob_us + t * 1e6
            exs[nrec], eys[nrec], ezs[nrec] = _efield(x, y, z, _rec_us)
            ts[nrec] = _rec_us; nrec += 1

    return nrec, x, y, z, vx, vy, vz, t, kind, ncol


# ------------------------------------------------------------- channel packer
def _empty_channels(shape):
    """Zero-channel field pack (static-only device)."""
    z0 = np.zeros((0,) + shape)
    return dict(ExK=z0, EyK=z0, EzK=z0,
                ch_kind=np.zeros(0, np.int64), ch_om=np.zeros(0),
                ch_ph=np.zeros(0), ch_amp=np.zeros(0), ch_off=np.zeros(0),
                ch_duty=np.zeros(0),
                tab_t=np.zeros(0), tab_v=np.zeros(0),
                tab_off=np.zeros(1, np.int64))


def single_sin_channels(EBx, EBy, EBz, rf_V, dc_V, om_rad_us, ph_rad=0.0):
    """Pack the legacy single-sinusoid model s(t)=sin(w t)*rf_V+dc_V as ONE
    affine sin channel over the B basis. Reproduces v61 arithmetic
    bit-for-bit in _fly3d."""
    return dict(
        ExK=np.ascontiguousarray(EBx[None]),
        EyK=np.ascontiguousarray(EBy[None]),
        EzK=np.ascontiguousarray(EBz[None]),
        ch_kind=np.array([K_SIN], np.int64),
        ch_om=np.array([om_rad_us], np.float64),
        ch_ph=np.array([ph_rad], np.float64),
        ch_amp=np.array([rf_V], np.float64),
        ch_off=np.array([dc_V], np.float64),
        ch_duty=np.array([0.5], np.float64),
        tab_t=np.zeros(0), tab_v=np.zeros(0),
        tab_off=np.zeros(2, np.int64))


def _resolve_channels(fields):
    """Return a channel pack from a fields dict, accepting either the new
    channel arrays (ExK...tab_off present) or the legacy single-sin scalars
    (EBx/EBy/EBz + rf_V/dc_V/om_rad_us). Legacy path is expanded to one sin
    channel so validate_quad3d.py runs unmodified through the new kernel."""
    if "ExK" in fields:
        # PACK CONTRACT: ch_duty is OPTIONAL-WITH-DEFAULT 0.5 (declared
        # here, the one normalization seam) so every legacy constructor
        # (slim3d, gate harnesses, saved packs) stays valid unchanged.
        if "ch_duty" not in fields:
            fields = dict(fields)
            fields["ch_duty"] = np.full(fields["ch_kind"].shape[0], 0.5)
        return fields
    if "EBx" in fields:
        return single_sin_channels(fields["EBx"], fields["EBy"], fields["EBz"],
                                   fields["rf_V"], fields["dc_V"],
                                   fields["om_rad_us"],
                                   fields.get("ph_rad", 0.0))
    return _empty_channels(fields["EAx"].shape)


# ------------------------------------------------------------- public fly
def fly3d(fields, mz_Da, r0_mm, v0_mm_us, tob_us, dt_ns=1.0, t_max_us=50.0,
          record_every=10, max_records=100000, collisions=None, seed=1,
          planes=None, ion_label="", transporter=None, charge=1):
    """Fly one ion in LOCAL frame. `fields` carries EAx/EAy/EAz (V/mm),
    ele (bool), h_mm, and EITHER new channel arrays (ExK,EyK,EzK stacked
    (K,nx,ny,nz); ch_kind,ch_om,ch_ph,ch_amp,ch_off; tab_t,tab_v,tab_off) OR
    the legacy single-sin scalars (EBx,EBy,EBz,rf_V,dc_V,om_rad_us).

    `collisions`: None/vacuum, or a dict/namespace with .gas (name or amu),
    .T_k, .P_pa, .sigma_m2 -> HS buffer-gas scattering (same model as the
    2-D kernel), reproducible per-ion via `seed`.

    `ion_label` (e.g. "ion 7"): names the ion in the birth-in-metal
    refusal; empty -> position-only diagnostic."""
    ch = _resolve_channels(fields)
    refuse_birth_in_metal_3d(
        fields["ele"], fields["h_mm"],
        float(r0_mm[0]), float(r0_mm[1]), float(r0_mm[2]),
        ion_label=ion_label,
        world_off_mm=fields.get("world_off_mm", (0.0, 0.0, 0.0)))
    # CHARGE STATE. This kernel once hard-coded q = +1e, so a
    # spec declaring source.charge = 2 flew a SINGLY charged ion at the
    # declared mass -- the acceleration was wrong by the charge factor while
    # every label said otherwise. mz_list carries the ion MASS in Da (same
    # convention as the r-z and SDS routes), so q/m = charge*e/(mass*amu).
    if int(charge) == 0:
        raise ValueError(
            "charge = 0: an uncharged ion has no electric acceleration. "
            "State the charge state (source.charge) rather than flying a "
            "neutral through an electrostatic field.")
    m = mz_Da * AMU
    qm = int(charge) * E_CHG / m

    def _get(o, k, d=None):
        return getattr(o, k, d) if not isinstance(o, dict) else o.get(k, d)
    on = collisions is not None and _get(collisions, "enabled", True)
    if on:
        gas = _get(collisions, "gas", "N2")
        T_k = float(_get(collisions, "T_k", 273.0))
        # pressure authored in Torr (preferred); fall back to Pa if given
        _TORR_PA = 133.32236842105263
        p_torr = _get(collisions, "P_torr", None)
        p_pa = _get(collisions, "P_pa", None)
        if p_torr is not None:
            P_pa = float(p_torr) * _TORR_PA
        elif p_pa is not None:
            P_pa = float(p_pa)
        else:
            P_pa = 1.0 * _TORR_PA
        sigma_m2 = float(_get(collisions, "sigma_m2", 2.27e-18))
        mg = gas_mass(gas)
        c_star = math.sqrt(2 * KB * T_k / (mg * KG_AMU)) / 1000.0
        c_bar = math.sqrt(8 * KB * T_k / (math.pi * mg * KG_AMU)) / 1000.0
        sig1d = math.sqrt(KB * T_k / (mg * KG_AMU)) / 1000.0
    else:
        T_k = P_pa = sigma_m2 = 0.0
        c_star = c_bar = sig1d = 1.0
        mg = 4.0

    xs = np.empty(max_records); ys = np.empty(max_records)
    zs = np.empty(max_records); ts = np.empty(max_records)
    vxs = np.empty(max_records); vys = np.empty(max_records)
    vzs = np.empty(max_records)
    # per-recorded-step field E(x,t) in V/mm (Analysis: e_field, e_axial,
    # e_x/y/z). Sampled in the kernel at each stored point via _efield.
    exs = np.empty(max_records); eys = np.empty(max_records)
    ezs = np.empty(max_records)
    # BOUNDING PLANES -> the kernel, so the crossing is found per step and the
    # exit state interpolated WITHIN the step. `planes` is a list of
    # (axis, value_mm, sign) with axis in 'xyz' and sign +1 (crossing upward
    # through the plane) or -1 (downward). Empty -> no plane termination.
    # entries are (axis, value_mm, sign) for whole-plane kills (bounds;
    # impossible pass window, fate 3) or (axis, value_mm, sign, window4,
    # kind) for windowed planes (stations): window4 = [a_lo,a_hi,b_lo,
    # b_hi] over the two transverse axes ascending; sign 0 = both
    # crossing directions. Window SENSE depends on kind: kind 5
    # (impact plane) is a PLATE — inside passes, outside splats;
    # kind 6 (detect) is a DETECTOR PATCH — inside ABSORBS (fate 6,
    # the detection event), outside passes (ruled 2026-09-12:
    # record = pass+log, detect = splat+log).
    pl_col, pl_val, pl_sgn, pl_w, pl_kind = pack_planes(planes, "fly3d")
    # TRANSPORTER contract: dict/namespace with axis ('x'|'y'|
    # 'z'), accept_mm, emit_mm (LOCAL frame -- caller converts), direction
    # (+1/-1), max_passes. None -> disabled.
    if transporter is not None:
        _AXT = {"x": 0, "y": 1, "z": 2}
        tp_col = int(_AXT[str(_get(transporter, "axis"))])
        tp_accept = float(_get(transporter, "accept_mm"))
        tp_emit = float(_get(transporter, "emit_mm"))
        tp_sgn = float(_get(transporter, "direction", -1))
        tp_maxp = int(_get(transporter, "max_passes", 10 ** 9))
        if tp_accept == tp_emit:
            raise ValueError("transporter: accept_mm == emit_mm -- a "
                             "zero-length transporter is a no-op in disguise")
    else:
        tp_col, tp_accept, tp_emit, tp_sgn, tp_maxp = -1, 0.0, 0.0, -1.0, 1
    wrs = np.zeros(max_records, np.int64)

    nrec, x, y, z, vx, vy, vz, t, kind, ncol = _fly3d(
        qm, float(mz_Da), float(r0_mm[0]), float(r0_mm[1]), float(r0_mm[2]),
        v0_mm_us[0] * 1e3, v0_mm_us[1] * 1e3, v0_mm_us[2] * 1e3,
        float(tob_us), dt_ns * 1e-9, t_max_us * 1e-6,
        fields["EAx"], fields["EAy"], fields["EAz"],
        np.ascontiguousarray(ch["ExK"]), np.ascontiguousarray(ch["EyK"]),
        np.ascontiguousarray(ch["EzK"]),
        ch["ch_kind"], ch["ch_om"], ch["ch_ph"], ch["ch_amp"], ch["ch_off"],
        ch["ch_duty"],
        ch["tab_t"], ch["tab_v"], ch["tab_off"],
        fields["ele"], fields["h_mm"],
        xs, ys, zs, ts, vxs, vys, vzs, exs, eys, ezs, record_every,
        on, T_k, P_pa, sigma_m2, c_star, c_bar, sig1d, mg, int(seed),
        pl_col, pl_val, pl_sgn, pl_w, pl_kind,
        tp_col, tp_accept, tp_emit, tp_sgn, tp_maxp, wrs)
    KE = 0.5 * m * (vx * vx + vy * vy + vz * vz) / E_CHG
    # No silent truncation: a full record buffer once
    # silently truncated the trace while the flight CONTINUED to its fate,
    # so trajectories appeared to "just stop" mid-gap and fates described
    # off-camera events. Truncation must announce itself and the summary
    # must carry the TRUE final state.
    t_end_us = tob_us + t * 1e6
    truncated = (nrec >= xs.shape[0]
                 and nrec > 0 and ts[nrec - 1] < t_end_us - 1e-9)
    if truncated:
        print(f"[fly3d] {ion_label}: RECORD BUFFER FULL at "
              f"t={ts[nrec - 1]:.1f} us -- flight CONTINUED unrecorded to "
              f"t={t_end_us:.1f} us (fate '{_KIND_NAME.get(kind, kind)}' at "
              f"x={x:.2f}, y={y:.2f}, z={z:.2f} mm field-frame). The "
              f"displayed trace ends at the buffer, NOT at the fate. "
              f"Raise record_every or max_records to cover the flight.")
    return dict(x=xs[:nrec].copy(), y=ys[:nrec].copy(), z=zs[:nrec].copy(),
                vx=vxs[:nrec].copy(), vy=vys[:nrec].copy(),
                vz=vzs[:nrec].copy(),
                ex=exs[:nrec].copy(), ey=eys[:nrec].copy(),
                ez=ezs[:nrec].copy(),
                wrap_passes=wrs[:nrec].copy(),
                t_us=ts[:nrec].copy(), tof_us=t_end_us, KE_eV=KE,
                v_mm_us=np.array([vx, vy, vz]) * 1e-3, kind=kind, ncol=ncol,
                truncated=truncated,
                final_xyz_mm=(float(x), float(y), float(z)),
                t_end_us=float(t_end_us))


from ion_gym.physics.ion_envelope import FATE_NAME as _KIND_NAME


# ============================================================ SDS integrator
@njit(cache=True)
def _sds_field(EAx, EAy, EAz, ExK, EyK, EzK, ch_kind, ch_om, ch_ph,
               ch_amp, ch_off, ch_duty, tab_t, tab_v, tab_off,
               gx, gy, gz, tt, nx, ny, nz, K):
    """Total E (V/mm) at grid coords (gx,gy,gz), time tt — the ONE
    field authority for the SDS kernel: the motion loop and the
    record buffer both call this, so a recorded field is exactly the
    field the ion felt (2026-09-12 extension: SDS records ex/ey/ez
    like the base kernel, so field channels are available under
    model='sds' instead of refusing)."""
    ex = _tri(EAx, gx, gy, gz, nx, ny, nz)
    ey = _tri(EAy, gx, gy, gz, nx, ny, nz)
    ez = _tri(EAz, gx, gy, gz, nx, ny, nz)
    for kk in range(K):
        w = _wave_eval(ch_kind[kk], ch_om[kk], ch_ph[kk], ch_amp[kk],
                       ch_off[kk], ch_duty[kk], tab_t, tab_v,
                       tab_off[kk], tab_off[kk + 1], tt)
        ex = ex + w * _tri(ExK[kk], gx, gy, gz, nx, ny, nz)
        ey = ey + w * _tri(EyK[kk], gx, gy, gz, nx, ny, nz)
        ez = ez + w * _tri(EzK[kk], gx, gy, gz, nx, ny, nz)
    return ex, ey, ez
_N_DIST_COLLISIONS = float(_NDC)


def pack_planes(planes, who="tracer"):
    """Plane tuples -> the five kernel arrays (col, val, sgn, window4, kind).

    ONE packer for every kernel that takes planes (fly3d and fly3d_sds
    here; the planar and r-z routes build the same arrays from the same
    shared plane builder). A 3-tuple is a bounding plane (fate 3, an
    impossible window so nothing is ever "inside"); a 5-tuple is a
    windowed station carrying its own fate. Any other width is refused
    by name rather than packed into a silently wrong array.
    """
    _AX = {"x": 0, "y": 1, "z": 2}
    _INF = float("inf")
    if not planes:
        return (np.empty(0, np.int64), np.empty(0, np.float64),
                np.empty(0, np.float64), np.empty((0, 4), np.float64),
                np.empty(0, np.int64))
    pl_col = np.array([_AX[str(p[0])] for p in planes], np.int64)
    pl_val = np.array([float(p[1]) for p in planes], np.float64)
    pl_sgn = np.array([float(p[2]) for p in planes], np.float64)
    pl_w = np.empty((len(planes), 4), np.float64)
    pl_kind = np.empty(len(planes), np.int64)
    for _i, p in enumerate(planes):
        if len(p) == 3:
            pl_w[_i] = (_INF, -_INF, _INF, -_INF)
            pl_kind[_i] = 3
        elif len(p) == 5:
            pl_w[_i] = [float(v) for v in p[3]]
            pl_kind[_i] = int(p[4])
        else:
            raise ValueError(
                f"{who}: plane entry {_i} has {len(p)} fields — "
                f"3 (bounds) or 5 (windowed station) only")
    return pl_col, pl_val, pl_sgn, pl_w, pl_kind


@njit(cache=True, fastmath=False, nogil=True)
def _fly3d_sds(qm, x0, y0, z0, vx0, vy0, vz0, tob_us, dt_us, t_max_us,
               EAx, EAy, EAz, ExK, EyK, EzK,
               ch_kind, ch_om, ch_ph, ch_amp, ch_off, ch_duty, tab_t, tab_v, tab_off,
               ele, h_mm, damping, mfp_mm, V_mm_us, log_mr,
               stats, vgx, vgy, vgz, diffusion_on, seed,
               xs, ys, zs, ts, vxs, vys, vzs, exs, eys, ezs,
               record_every, pl_col, pl_val, pl_sgn, pl_w, pl_kind):
    """SDS dynamics in mm/us: field acceleration damped toward the mobility
    drift (Stokes, apply_stokes_damping) + ICDF random-walk diffusion
    (apply_diffusion). Faithful to the published SDS formulation. Gas P,T,velocity are
    uniform here, so damping/mfp/V are per-ion scalars. Returns
    (nrec, x,y,z, vx,vy,vz, t_us, kind)."""
    np.random.seed(seed)
    nx, ny, nz = EAx.shape
    inv_h = 1.0 / h_mm
    K = ch_kind.shape[0]
    x = x0; y = y0; z = z0
    vx = vx0; vy = vy0; vz = vz0          # mm/us
    t = 0.0
    xs[0] = x; ys[0] = y; zs[0] = z; ts[0] = tob_us
    vxs[0] = vx; vys[0] = vy; vzs[0] = vz
    exs[0], eys[0], ezs[0] = _sds_field(
        EAx, EAy, EAz, ExK, EyK, EzK, ch_kind, ch_om, ch_ph, ch_amp,
        ch_off, ch_duty, tab_t, tab_v, tab_off,
        x * inv_h, y * inv_h, z * inv_h, tob_us, nx, ny, nz, K)
    nrec = 1; step = 0; kind = 2

    while t < t_max_us:
        tt = tob_us + t
        gx = x * inv_h; gy = y * inv_h; gz = z * inv_h
        ex, ey, ez = _sds_field(
            EAx, EAy, EAz, ExK, EyK, EzK, ch_kind, ch_om, ch_ph,
            ch_amp, ch_off, ch_duty, tab_t, tab_v, tab_off,
            gx, gy, gz, tt, nx, ny, nz, K)
        # field acceleration in mm/us^2  (a[m/s2]*1e-9; qm*E[V/mm]*1e-6)
        afx = qm * 1e-6 * ex; afy = qm * 1e-6 * ey; afz = qm * 1e-6 * ez

        # Stokes' law viscous mobility (apply_stokes_damping)
        if damping > 0.0:
            tterm = damping * dt_us
            factor = (1.0 - math.exp(-tterm)) / tterm
            aex = factor * (afx - (vx - vgx) * damping)
            aey = factor * (afy - (vy - vgy) * damping)
            aez = factor * (afz - (vz - vgz) * damping)
        else:
            aex = afx; aey = afy; aez = afz

        vx = vx + aex * dt_us; vy = vy + aey * dt_us; vz = vz + aez * dt_us
        # pre-step state for plane crossing: the crossing is tested over
        # the WHOLE step, advection PLUS the diffusion jump below, so a
        # diffusing ion cannot hop across a detector between samples.
        xo = x; yo = y; zo = z
        vxo = vx; vyo = vy; vzo = vz
        to = t
        x = x + vx * dt_us; y = y + vy * dt_us; z = z + vz * dt_us

        # random-walk diffusion (apply_diffusion)
        if diffusion_on and mfp_mm > 0.0:
            dist_steps = _diff_dist_steps(stats, log_mr)
            ncoll = V_mm_us / mfp_mm * dt_us
            r = math.sqrt(ncoll / _N_DIST_COLLISIONS) * dist_steps * mfp_mm
            jx, jy, jz = _sphere_rand(r)
            x = x + jx; y = y + jy; z = z + jz

        t += dt_us; step += 1

        if _in_metal_3d(ele, x * inv_h, y * inv_h, z * inv_h, nx, ny, nz):
            kind = 0
            if nrec < xs.shape[0]:
                xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
                vxs[nrec] = vx; vys[nrec] = vy; vzs[nrec] = vz
                exs[nrec], eys[nrec], ezs[nrec] = _sds_field(
                    EAx, EAy, EAz, ExK, EyK, EzK, ch_kind, ch_om,
                    ch_ph, ch_amp, ch_off, ch_duty, tab_t, tab_v,
                    tab_off, x * inv_h, y * inv_h, z * inv_h,
                    tob_us + t, nx, ny, nz, K)
                ts[nrec] = tob_us + t; nrec += 1
            break
        if (x < 0.0 or y < 0.0 or z < 0.0 or x > (nx - 1) * h_mm
                or y > (ny - 1) * h_mm or z > (nz - 1) * h_mm):
            kind = 1
            if nrec < xs.shape[0]:
                xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
                vxs[nrec] = vx; vys[nrec] = vy; vzs[nrec] = vz
                exs[nrec], eys[nrec], ezs[nrec] = _sds_field(
                    EAx, EAy, EAz, ExK, EyK, EzK, ch_kind, ch_om,
                    ch_ph, ch_amp, ch_off, ch_duty, tab_t, tab_v,
                    tab_off, x * inv_h, y * inv_h, z * inv_h,
                    tob_us + t, nx, ny, nz, K)
                ts[nrec] = tob_us + t; nrec += 1
            break
        # STATION AND BOUND PLANES. The SDS kernel had NO plane handling
        # at all (fates 0 metal, 1 box exit, 2 timeout), so under
        # collisions.model="sds" a declared detector, impact plane OR
        # bounding plane was silently inert while the same deck honoured
        # them under "hs" — the model choice quietly changed which
        # declarations the run obeyed. Same crossing math, window sense
        # and ordering as fly3d/_fly_planar/tracer_rz, from the one
        # shared plane builder.
        if pl_col.shape[0] > 0:
            hit_pl = False
            for ip in range(pl_col.shape[0]):
                pcol = pl_col[ip]
                if pcol == 0:
                    cn = x; co = xo
                elif pcol == 1:
                    cn = y; co = yo
                else:
                    cn = z; co = zo
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
                    xc = xo + f * (x - xo)
                    yc = yo + f * (y - yo)
                    zc = zo + f * (z - zo)
                    if pcol == 0:
                        w1 = yc; w2 = zc
                    elif pcol == 1:
                        w1 = xc; w2 = zc
                    else:
                        w1 = xc; w2 = yc
                    _ins = (pl_w[ip, 0] <= w1 <= pl_w[ip, 1]
                            and pl_w[ip, 2] <= w2 <= pl_w[ip, 3])
                    if pl_kind[ip] == 6:
                        if not _ins:
                            continue    # detector patch: outside passes
                    elif pl_kind[ip] == 5 and _ins:
                        continue        # plate: inside the aperture passes
                    x = xc; y = yc; z = zc
                    vx = vxo + f * (vx - vxo)
                    vy = vyo + f * (vy - vyo)
                    vz = vzo + f * (vz - vzo)
                    t = to + f * (t - to)
                    kind = pl_kind[ip]
                    hit_pl = True
                    break
            if hit_pl:
                if nrec < xs.shape[0]:
                    xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
                    vxs[nrec] = vx; vys[nrec] = vy; vzs[nrec] = vz
                    exs[nrec], eys[nrec], ezs[nrec] = _sds_field(
                        EAx, EAy, EAz, ExK, EyK, EzK, ch_kind, ch_om,
                        ch_ph, ch_amp, ch_off, ch_duty, tab_t, tab_v,
                        tab_off, x * inv_h, y * inv_h, z * inv_h,
                        tob_us + t, nx, ny, nz, K)
                    ts[nrec] = tob_us + t; nrec += 1
                break
        if step % record_every == 0 and nrec < xs.shape[0]:
            xs[nrec] = x; ys[nrec] = y; zs[nrec] = z
            vxs[nrec] = vx; vys[nrec] = vy; vzs[nrec] = vz
            exs[nrec], eys[nrec], ezs[nrec] = _sds_field(
                EAx, EAy, EAz, ExK, EyK, EzK, ch_kind, ch_om, ch_ph,
                ch_amp, ch_off, ch_duty, tab_t, tab_v, tab_off,
                x * inv_h, y * inv_h, z * inv_h, tob_us + t,
                nx, ny, nz, K)
            ts[nrec] = tob_us + t; nrec += 1

    return nrec, x, y, z, vx, vy, vz, t, kind


# module-level SDS data (loaded once)
_SDS_STATS = None
_SDS_MASSDATA = None


def _sds_data():
    global _SDS_STATS, _SDS_MASSDATA
    if _SDS_STATS is None:
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        _SDS_STATS = load_diffusion_statistics(os.path.join(here, _sds.JUMP_ICDF_FILE))
        _SDS_MASSDATA = load_massdata(os.path.join(here, _sds.MOBILITY_FILE))
    return _SDS_STATS, _SDS_MASSDATA


def fly3d_sds(fields, mz_Da, charge, r0_mm, v0_mm_us, tob_us, collisions,
              dt_ns=5.0, t_max_us=50.0, record_every=50, max_records=200000,
              seed=1, diffusion=True, ion_label="", planes=None):
    """Fly one ion under the SDS collision model. `collisions` carries gas,
    gas_diam_nm, T_k, P_torr; per-ion mobility/MFP/diffusion come from the
    the reference mass table + estimators. Returns the same dict shape as fly3d
    (velocities in mm/us). `ion_label`: see fly3d."""
    refuse_birth_in_metal_3d(
        fields["ele"], fields["h_mm"],
        float(r0_mm[0]), float(r0_mm[1]), float(r0_mm[2]),
        ion_label=ion_label,
        world_off_mm=fields.get("world_off_mm", (0.0, 0.0, 0.0)))
    def _get(o, k, d):
        return getattr(o, k, d) if not isinstance(o, dict) else o.get(k, d)
    gas = _get(collisions, "gas", "N2")
    mgas = _get(collisions, "gas_mass_amu", None)
    if mgas is None:
        mgas = 28.94515 if gas in ("air", "N2") else gas_mass(gas)
    gdiam = float(_get(collisions, "gas_diam_nm", 0.366))
    T_k = float(_get(collisions, "T_k", 298.15))
    p_torr = _get(collisions, "P_torr", None)
    if p_torr is None:
        p_torr = float(_get(collisions, "P_pa", 133.322)) / 133.322
    stats, massdata = _sds_data()
    P = ion_params(mz_Da, charge, mgas, gdiam, T_k, float(p_torr), massdata)

    ch = _resolve_channels(fields)
    m = mz_Da * AMU
    qm = E_CHG / m
    xs = np.empty(max_records); ys = np.empty(max_records)
    zs = np.empty(max_records); ts = np.empty(max_records)
    vxs = np.empty(max_records); vys = np.empty(max_records)
    vzs = np.empty(max_records)
    exs = np.empty(max_records); eys = np.empty(max_records)
    ezs = np.empty(max_records)
    vg = [v * 1e-3 for v in _get(collisions, "flow_m_s", (0.0, 0.0, 0.0))]
    nrec, x, y, z, vx, vy, vz, t, kind = _fly3d_sds(
        qm, float(r0_mm[0]), float(r0_mm[1]), float(r0_mm[2]),
        float(v0_mm_us[0]), float(v0_mm_us[1]), float(v0_mm_us[2]),
        float(tob_us), dt_ns * 1e-3, float(t_max_us),
        fields["EAx"], fields["EAy"], fields["EAz"],
        np.ascontiguousarray(ch["ExK"]), np.ascontiguousarray(ch["EyK"]),
        np.ascontiguousarray(ch["EzK"]),
        ch["ch_kind"], ch["ch_om"], ch["ch_ph"], ch["ch_amp"], ch["ch_off"],
        ch["ch_duty"],
        ch["tab_t"], ch["tab_v"], ch["tab_off"],
        fields["ele"], fields["h_mm"],
        P["damping"], P["mfp_mm"], P["V_mm_us"], P["log_mr_ratio"],
        stats, vg[0], vg[1], vg[2], bool(diffusion), int(seed),
        xs, ys, zs, ts, vxs, vys, vzs, exs, eys, ezs, record_every,
        *pack_planes(planes, "fly3d_sds"))
    KE = 0.5 * m * ((vx*1e3)**2 + (vy*1e3)**2 + (vz*1e3)**2) / E_CHG
    # FLIGHT-OUTPUT CONTRACT. This wrapper once returned a partial
    # dict, which only worked because the (now retired) import route had its
    # own bespoke assembly. The native 3-D route consumes the full contract,
    # so the terminal state is reported here from the kernel's own exact
    # final values. n_col is DELIBERATELY ABSENT, not zero: SDS is a
    # continuum mobility+diffusion model (Appelhans & Dahl 2005) with no
    # discrete collision events to count, and reporting 0 would read as a
    # collisionless flight. Consumers that need it must refuse, not default.
    _t_end = float(tob_us + t)
    return dict(x=xs[:nrec].copy(), y=ys[:nrec].copy(), z=zs[:nrec].copy(),
                vx=vxs[:nrec].copy(), vy=vys[:nrec].copy(), vz=vzs[:nrec].copy(),
                t_us=ts[:nrec].copy(), tof_us=tob_us + t, KE_eV=KE,
                ex=exs[:nrec].copy(), ey=eys[:nrec].copy(),
                ez=ezs[:nrec].copy(),
                v_mm_us=np.array([vx, vy, vz]), kind=kind, sds=P,
                final_xyz_mm=(float(x), float(y), float(z)),
                t_end_us=_t_end, truncated=bool(nrec >= len(xs)))
