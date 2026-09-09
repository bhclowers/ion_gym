"""
ion_gym.tracer3d_v61_frozen (FROZEN for Gate 1)
-----------------
Full-3D ion tracer for node-centred grids, validated on the
quad_monolithic article (see internal validation record).

Field model: phi(x, t) = A(x) + s(t) * B(x) on a regular 3-D grid (unfolded
to the full physical volume; the mirror-unfold convention), with s(t) an arbitrary
scalar waveform on the ABSOLUTE clock (t includes the ion's birth time — the
node-centred convention pinned on the buncher and re-confirmed on the quad).

Conventions carried from the validated 2-D work:
  * near-electrode field: one-sided differences at metal-adjacent nodes
    (build_field_aware_3d); naive central differences halve the surface field
    (tof lesson, re-observed at the quad detector face at exactly -50%).
  * impact rule: a point is in metal iff ALL EIGHT corner nodes of its
    containing voxel are electrode nodes (3-D generalization of the
    all-four-corners cell rule pinned on tof + einzel); thin ideal grids
    remain transparent, solid plates impact at the first node plane.
  * fastmath=False everywhere; the njit trilinear sampler is bit-checked
    against scipy RegularGridInterpolator.
"""

import numpy as np
from numba import njit

E_CHG = 1.602176634e-19
AMU = 1.66053906660e-27


# ------------------------------------------------------------- field builder
def build_field_aware_3d(phi, ele, h_mm):
    """E = -grad(phi) in V/mm on the node grid, electrode-aware. Faithful 3-D
    port of the validated 2-D rule (ionbench.build_field_aware, tof Gate 1):
    plain central differences everywhere (np.gradient, one-sided at array
    edges), then at METAL nodes with vacuum on exactly one side along an axis,
    replace that axis derivative with the one-sided difference into the
    vacuum. Vacuum nodes are never overridden -- central differencing there is
    the correct second-order scheme (over-applying the override to
    metal-adjacent vacuum nodes degrades the match by ~1000x; measured on the
    quad article). Returns (Ex, Ey, Ez) float64, V/mm."""
    d = np.gradient(phi, h_mm)            # [d/dx, d/dy, d/dz] V/mm
    nx, ny, nz = phi.shape
    mi, mj, mk = np.where(ele)
    for i, j, k in zip(mi, mj, mk):
        for ax, dp in enumerate(d):
            if ax == 0:
                vp = i + 1 < nx and not ele[i + 1, j, k]
                vm = i - 1 >= 0 and not ele[i - 1, j, k]
                if vp and not vm:
                    dp[i, j, k] = (phi[i + 1, j, k] - phi[i, j, k]) / h_mm
                elif vm and not vp:
                    dp[i, j, k] = (phi[i, j, k] - phi[i - 1, j, k]) / h_mm
            elif ax == 1:
                vp = j + 1 < ny and not ele[i, j + 1, k]
                vm = j - 1 >= 0 and not ele[i, j - 1, k]
                if vp and not vm:
                    dp[i, j, k] = (phi[i, j + 1, k] - phi[i, j, k]) / h_mm
                elif vm and not vp:
                    dp[i, j, k] = (phi[i, j, k] - phi[i, j - 1, k]) / h_mm
            else:
                vp = k + 1 < nz and not ele[i, j, k + 1]
                vm = k - 1 >= 0 and not ele[i, j, k - 1]
                if vp and not vm:
                    dp[i, j, k] = (phi[i, j, k + 1] - phi[i, j, k]) / h_mm
                elif vm and not vp:
                    dp[i, j, k] = (phi[i, j, k] - phi[i, j, k - 1]) / h_mm
    return -d[0], -d[1], -d[2]


# ------------------------------------------------------------- njit sampling
@njit(cache=True, fastmath=False, inline="always")
def _tri(F, x, y, z, nx, ny, nz):
    """Trilinear sample of F at grid-unit coordinates (x,y,z); clamps to the
    box edge cell (matches RegularGridInterpolator on in-box points; callers
    guard the domain)."""
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


@njit(cache=True, fastmath=False, inline="always")
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


@njit(cache=True, fastmath=False)
def _fly3d(qm, x0, y0, z0, vx0, vy0, vz0, tob_us, dt_s, t_max_s,
           EAx, EAy, EAz, EBx, EBy, EBz, ele, h_mm,
           rf_V, dc_V, om_rad_us, xs, ys, zs, ts,
           vxs, vys, vzs, record_every):
    """RK4 in LOCAL mm; field E(x,t) = E_A + s(t)*E_B (V/mm), s on the
    absolute clock in us. Returns (nrec, x, y, z, vx, vy, vz, t_s, kind)
    with kind: 0 = impact on metal, 1 = left array box, 2 = time out."""
    nx, ny, nz = EAx.shape
    inv_h = 1.0 / h_mm
    x = x0; y = y0; z = z0
    vx = vx0; vy = vy0; vz = vz0          # m/s
    t = 0.0                               # elapsed s
    c = 1e3                               # m/s -> mm/us with dt in s: mm = v*dt*1e3
    nrec = 0
    xs[0] = x; ys[0] = y; zs[0] = z; ts[0] = tob_us
    vxs[0] = vx * 1e-3; vys[0] = vy * 1e-3; vzs[0] = vz * 1e-3  # mm/us
    nrec = 1
    step = 0
    kind = 2

    while t < t_max_s:
        xo = x; yo = y; zo = z; vxo = vx; vyo = vy; vzo = vz; to = t

        # s(t) at RK sub-times (absolute us)
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
                ts[nrec] = tob_us + t * 1e6; nrec += 1
            break

        # left the array box
        if (x < 0.0 or y < 0.0 or z < 0.0 or x > (nx - 1) * h_mm
                or y > (ny - 1) * h_mm or z > (nz - 1) * h_mm):
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

    return nrec, x, y, z, vx, vy, vz, t, kind


def fly3d(fields, mz_Da, r0_mm, v0_mm_us, tob_us, dt_ns=1.0, t_max_us=50.0,
          record_every=10, max_records=100000):
    """Fly one ion in LOCAL frame. fields: dict with EAx..EBz (V/mm on the
    unfolded grid), ele (bool), h_mm, rf_V, dc_V, om_rad_us."""
    m = mz_Da * AMU
    qm = E_CHG / m
    xs = np.empty(max_records); ys = np.empty(max_records)
    zs = np.empty(max_records); ts = np.empty(max_records)
    vxs = np.empty(max_records); vys = np.empty(max_records)
    vzs = np.empty(max_records)
    nrec, x, y, z, vx, vy, vz, t, kind = _fly3d(
        qm, float(r0_mm[0]), float(r0_mm[1]), float(r0_mm[2]),
        v0_mm_us[0] * 1e3, v0_mm_us[1] * 1e3, v0_mm_us[2] * 1e3,
        float(tob_us), dt_ns * 1e-9, t_max_us * 1e-6,
        fields["EAx"], fields["EAy"], fields["EAz"],
        fields["EBx"], fields["EBy"], fields["EBz"],
        fields["ele"], fields["h_mm"], fields["rf_V"], fields["dc_V"],
        fields["om_rad_us"], xs, ys, zs, ts, vxs, vys, vzs, record_every)
    KE = 0.5 * m * (vx * vx + vy * vy + vz * vz) / E_CHG
    return dict(x=xs[:nrec].copy(), y=ys[:nrec].copy(), z=zs[:nrec].copy(),
                vx=vxs[:nrec].copy(), vy=vys[:nrec].copy(),
                vz=vzs[:nrec].copy(),
                t_us=ts[:nrec].copy(), tof_us=tob_us + t * 1e6, KE_eV=KE,
                v_mm_us=np.array([vx, vy, vz]) * 1e-3, kind=kind)
