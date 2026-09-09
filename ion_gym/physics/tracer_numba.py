"""
ion_gym.tracer_numba
---------------------
Numba port of tracer.fly's hot path: RK4 integration + field sampling.

Design constraints (from the validation handoff — these are the five places a
transcription error would hide, each preserved verbatim from tracer.py):

  1. MIRROR-EXTENSION. The cylindrical field is pre-extended to negative r in
     build_field() exactly as Field2D does, so E_r is odd, E_r(axis)=0, and
     the sampler needs NO special axis/sign logic — signed r just indexes the
     extended grid. (sharp edge 1 + 2 collapsed into the grid, as in NumPy.)
  2. BILINEAR sampling replaces RegularGridInterpolator (not Numba-able).
     Same math RGI uses: clamp-to-edge out-of-bounds -> fill_value 0.0 is
     instead emulated by returning 0 outside [Z0,Z1]x[U0,U1] to match RGI's
     bounds_error=False, fill_value=0.0.
  3. PLANE-CROSSING interpolation: linear-in-step, identical fractional f.
  4. ELECTRODE-IMPACT: interpolate back to the |u| surface crossed in-step.
  5. EXIT CLAMP: interpolate to z=Lz; positions written in METRES*1e3 = mm
     (the units bug fixed in round 1 — zs[-1] = Lz*1e3, not Lz).

The jitted kernel returns primitive arrays only; the Python wrapper repackages
into the same dict tracer.fly emits, so callers are unchanged.
"""

import numpy as np
from numba import njit

E_CHG = 1.602176634e-19
AMU = 1.66053907e-27


def build_field(Z, U, phi, symmetry="cylindrical"):
    """Return (Z, Ue, Ez, Eu) with Ez/Eu in V/m on the (mirror-extended) grid.
    Mirrors Field2D.__init__ exactly so the jitted sampler matches the NumPy
    interpolator node-for-node."""
    if symmetry == "cylindrical":
        assert abs(U[0]) < 1e-12, "cylindrical grid must start at the axis"
        Ue = np.concatenate([-U[:0:-1], U])
        pe = np.concatenate([phi[:, :0:-1], phi], axis=1)
    else:
        Ue, pe = U.copy(), phi.copy()
    dpz, dpu = np.gradient(pe, Z, Ue)          # V/mm
    Ez = np.ascontiguousarray(-dpz * 1e3)      # V/m
    Eu = np.ascontiguousarray(-dpu * 1e3)
    return (np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(Ue, dtype=np.float64), Ez, Eu)


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _sample(Zg, Ug, F, z_mm, u_mm):
    """Bilinear sample of F on the (possibly non-uniform-safe but here uniform)
    grid Zg x Ug. Matches RegularGridInterpolator with bounds_error=False,
    fill_value=0.0: any point outside the grid box returns 0.0.
    Grids are uniform -> O(1) index via spacing."""
    nz = Zg.shape[0]
    nu = Ug.shape[0]
    z0 = Zg[0]; u0 = Ug[0]
    dz = Zg[1] - Zg[0]
    du = Ug[1] - Ug[0]
    if z_mm < z0 or z_mm > Zg[nz - 1] or u_mm < u0 or u_mm > Ug[nu - 1]:
        return 0.0
    fz = (z_mm - z0) / dz
    fu = (u_mm - u0) / du
    iz = int(fz)
    iu = int(fu)
    if iz >= nz - 1:
        iz = nz - 2
    if iu >= nu - 1:
        iu = nu - 2
    tz = fz - iz
    tu = fu - iu
    f00 = F[iz, iu]
    f10 = F[iz + 1, iu]
    f01 = F[iz, iu + 1]
    f11 = F[iz + 1, iu + 1]
    return ((f00 * (1.0 - tz) + f10 * tz) * (1.0 - tu)
            + (f01 * (1.0 - tz) + f11 * tz) * tu)


@njit(cache=True, fastmath=False, nogil=True)
def _fly_core(Zg, Ug, Ez, Eu, qm, vz0, vu0, z0_m, u0_m, dt, Lz_m,
              max_steps, planes_m, surfaces_m, has_metal,
              band_lo, band_hi, z_hist, u_hist):
    """Core integrator. Writes trajectory into z_hist/u_hist (preallocated,
    length max_steps+1). Returns a flat result tuple; plane records go into
    rec_out (packed rows: z_mm, t_ns, u_mm, vz, vu)."""
    vz = vz0; vu = vu0
    z = z0_m; u = u0_m
    t = 0.0
    z_hist[0] = z * 1e3
    u_hist[0] = u * 1e3
    n = 1

    n_planes = planes_m.shape[0]
    next_plane = 0
    rec = np.empty((n_planes, 5), np.float64)
    n_rec = 0

    impact_code = 0            # 0 none, 1 electrode
    steps = 0
    z_prev = z; u_prev = u; t_prev = t; vz_prev = vz; vu_prev = vu

    while z < Lz_m and steps < max_steps:
        z_prev = z; u_prev = u; t_prev = t
        vz_prev = vz; vu_prev = vu

        # RK4 (accel = qm * E; E sampled in mm, returned V/m)
        az1 = qm * _sample(Zg, Ug, Ez, z * 1e3, u * 1e3)
        au1 = qm * _sample(Zg, Ug, Eu, z * 1e3, u * 1e3)
        k1z = vz; k1u = vu

        z2 = z + 0.5 * dt * k1z; u2 = u + 0.5 * dt * k1u
        az2 = qm * _sample(Zg, Ug, Ez, z2 * 1e3, u2 * 1e3)
        au2 = qm * _sample(Zg, Ug, Eu, z2 * 1e3, u2 * 1e3)
        k2z = vz + 0.5 * dt * az1; k2u = vu + 0.5 * dt * au1

        z3 = z + 0.5 * dt * k2z; u3 = u + 0.5 * dt * k2u
        az3 = qm * _sample(Zg, Ug, Ez, z3 * 1e3, u3 * 1e3)
        au3 = qm * _sample(Zg, Ug, Eu, z3 * 1e3, u3 * 1e3)
        k3z = vz + 0.5 * dt * az2; k3u = vu + 0.5 * dt * au2

        z4 = z + dt * k3z; u4 = u + dt * k3u
        az4 = qm * _sample(Zg, Ug, Ez, z4 * 1e3, u4 * 1e3)
        au4 = qm * _sample(Zg, Ug, Eu, z4 * 1e3, u4 * 1e3)
        k4z = vz + dt * az3; k4u = vu + dt * au3

        z += dt / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z)
        u += dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        vz += dt / 6.0 * (az1 + 2 * az2 + 2 * az3 + az4)
        vu += dt / 6.0 * (au1 + 2 * au2 + 2 * au3 + au4)
        t += dt

        # plane crossings (linear within the step) — same predicate as NumPy
        while (next_plane < n_planes
               and z_prev * 1e3 < planes_m[next_plane] * 1e3 <= z * 1e3):
            zp = planes_m[next_plane]
            f = (zp - z_prev) / (z - z_prev)
            rec[n_rec, 0] = planes_m[next_plane] * 1e3
            rec[n_rec, 1] = (t_prev + f * dt) * 1e9
            rec[n_rec, 2] = (u_prev + f * (u - u_prev)) * 1e3
            rec[n_rec, 3] = vz_prev + f * (vz - vz_prev)
            rec[n_rec, 4] = vu_prev + f * (vu - vu_prev)
            n_rec += 1
            next_plane += 1

        # electrode impact: metal(z,u) = |u|>=surf inside any band
        if has_metal:
            z_now = z * 1e3
            u_now = u * 1e3
            in_band = False
            for bi in range(band_lo.shape[0]):
                if band_lo[bi] <= z_now <= band_hi[bi]:
                    in_band = True
                    break
            if in_band and abs(u_now) >= surfaces_m[0]:
                ua = abs(u_prev) * 1e3
                ub = abs(u) * 1e3
                f = 1.0
                for si in range(surfaces_m.shape[0]):
                    surf = surfaces_m[si]
                    if ua < surf <= ub:
                        f = (surf - ua) / (ub - ua)
                        break
                t = t_prev + f * dt
                z = z_prev + f * (z - z_prev)
                u = u_prev + f * (u - u_prev)
                vz = vz_prev + f * (vz - vz_prev)
                vu = vu_prev + f * (vu - vu_prev)
                z_hist[n] = z * 1e3
                u_hist[n] = u * 1e3
                n += 1
                impact_code = 1
                break

        z_hist[n] = z * 1e3
        u_hist[n] = u * 1e3
        n += 1
        steps += 1
        if z < -1e-6:          # reflected out the entrance
            break

    # exit clamp (only if not impactted), positions in mm
    if impact_code == 0 and z > Lz_m and z != z_prev:
        f = (Lz_m - z_prev) / (z - z_prev)
        t = t_prev + f * dt
        u = u_prev + f * (u - u_prev)
        vz = vz_prev + f * (vz - vz_prev)
        vu = vu_prev + f * (vu - vu_prev)
        z = Lz_m
        z_hist[n - 1] = Lz_m * 1e3
        u_hist[n - 1] = u * 1e3

    return n, n_rec, rec, z, u, vz, vu, t, impact_code, steps


def fly(field, mz_Da, K_eV, z0_mm, u0_mm, ang0_mrad=0.0,
        record_planes_mm=(), dt_frac=0.02, h_mm=0.1, max_steps=2_000_000,
        metal=None):
    """Numba-backed drop-in for tracer.fly. `field` is a FieldNumba (below)."""
    m = mz_Da * AMU
    v0 = np.sqrt(2.0 * K_eV * E_CHG / m)
    a0 = ang0_mrad * 1e-3
    vz0, vu0 = v0 * np.cos(a0), v0 * np.sin(a0)
    dt = dt_frac * (h_mm * 1e-3) / v0
    Lz_m = field.Z[-1] * 1e-3
    qm = E_CHG / m

    planes_m = np.array(sorted(record_planes_mm), np.float64) * 1e-3 \
        if len(record_planes_mm) else np.empty(0, np.float64)

    if metal is not None:
        surfaces_m = np.array(getattr(metal, "surfaces", (18.0,)), np.float64)
        bands = np.array(getattr(metal, "bands",
                                 ((0.0, 28.0), (30.0, 56.0), (58.0, 90.0))),
                         np.float64)
        band_lo = np.ascontiguousarray(bands[:, 0])
        band_hi = np.ascontiguousarray(bands[:, 1])
        has_metal = True
    else:
        surfaces_m = np.empty(1, np.float64)
        band_lo = np.empty(0, np.float64)
        band_hi = np.empty(0, np.float64)
        has_metal = False

    z_hist = np.empty(max_steps + 2, np.float64)
    u_hist = np.empty(max_steps + 2, np.float64)

    (n, n_rec, rec, z, u, vz, vu, t, impact_code, steps) = _fly_core(
        field.Z, field.U, field.Ez, field.Eu, qm, vz0, vu0,
        z0_mm * 1e-3, u0_mm * 1e-3, dt, Lz_m, max_steps,
        planes_m, surfaces_m, has_metal, band_lo, band_hi, z_hist, u_hist)

    KE = 0.5 * m * (vz * vz + vu * vu) / E_CHG
    records = [dict(z_mm=rec[i, 0], t_ns=rec[i, 1], u_mm=rec[i, 2],
                    vz=rec[i, 3], vu=rec[i, 4]) for i in range(n_rec)]
    return dict(z=z_hist[:n].copy(), u=u_hist[:n].copy(), t_ns=t * 1e9,
                KE_eV=KE, ang_mrad=1e3 * np.arctan2(vu, vz),
                u_exit_mm=u * 1e3,
                impact="electrode" if impact_code == 1 else None,
                records=records, steps=steps)


class FieldNumba:
    """Field wrapper holding the mirror-extended V/m arrays the jitted core
    reads. Interface-compatible with tracer.Field2D where callers need .Z."""
    def __init__(self, Z, U, phi, symmetry="cylindrical"):
        self.Z, self.U, self.Ez, self.Eu = build_field(Z, U, phi, symmetry)
        self.symmetry = symmetry
        self.phi = phi                     # kept for plotting (native grid)

    def E(self, z_mm, u_mm):
        return (_sample(self.Z, self.U, self.Ez, z_mm, u_mm),
                _sample(self.Z, self.U, self.Eu, z_mm, u_mm))

    # ---- cache-backed construction (build scope item 2) -------------------
    @classmethod
    def _from_arrays(cls, Z, U, Ez, Eu, symmetry, phi=None):
        obj = cls.__new__(cls)
        obj.Z, obj.U, obj.Ez, obj.Eu = Z, U, Ez, Eu
        obj.symmetry = symmetry
        obj.phi = phi
        return obj

    @classmethod
    def cached(cls, spec, solve_fn, symmetry="cylindrical", root=None,
               store_phi=True):
        """Return a FieldNumba for `spec`, solving only on a cache miss.

        spec      : JSON-able dict fully describing the solve (the cache key).
        solve_fn  : zero-arg callable -> (Z_native, U_native, phi_native),
                    invoked ONLY on miss.
        On a hit, the pre-derived Ez/Eu (V/m, mirror-extended) are loaded
        zero-copy as read-only memmaps and the jitted kernel reads them with
        no gradient recompute and no copy. phi (native grid) is memmapped too
        for plotting.
        """
        from ion_gym.io import fa_cache
        kw = {} if root is None else dict(root=root)
        arrs, meta = fa_cache.load(spec, mmap=True, **kw)
        if arrs is not None:
            return cls._from_arrays(
                np.asarray(arrs["Z"]), np.asarray(arrs["Ue"]),
                arrs["Ez"], arrs["Eu"], meta["spec"].get("symmetry", symmetry),
                phi=arrs.get("phi"))
        # miss: solve on the native grid, derive extended V/m fields, store
        Zn, Un, phi = solve_fn()
        Z, Ue, Ez, Eu = build_field(Zn, Un, phi, symmetry)
        payload = dict(Z=Z, Ue=Ue, Ez=Ez, Eu=Eu)
        if store_phi:
            payload["phi"] = np.ascontiguousarray(phi)
            payload["U_native"] = np.ascontiguousarray(Un)
        full_spec = dict(spec)
        full_spec.setdefault("symmetry", symmetry)
        fa_cache.store(full_spec, payload, **kw)
        return cls._from_arrays(Z, Ue, Ez, Eu, symmetry,
                                phi=phi if store_phi else None)
