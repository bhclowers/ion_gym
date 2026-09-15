"""
ion_gym.flyer_plane
-------------------
Canonical width-plane (s, u) flyer WITH electrode-hit detection -- Task E1.

THE HIT CONVENTION (enforced by the hit-convention gates)
=====================================================================
1. GEOMETRY AUTHORITY. The hit test derives from the SAME analytic source
   that builds the solve mask: ``plate_positions()`` of the stack module
   (exact axis-aligned rectangles in the width plane), plus the drift rails
   where the stack defines them. No parallel hardcoded geometry -- this is
   the trajectory-side instance of the display==solve-input doctrine.
   Where a future scene carries sub-voxel fractional boundaries (a surface-fraction sidecar),
   the analytic scene stays the authority; a bare field array with no analytic
   source must hit-test against its own voxel mask and SAY SO.

2. SEGMENT TEST, NOT ENDPOINT TEST. Each velocity-Verlet step's straight
   chord (s,u) -> (s_n,u_n) is intersected against every electrode
   rectangle (vectorised slab method). A ray cannot tunnel through metal
   thinner than a step, and grazing corner clips within one step are
   caught. Correctness must not depend on dt being "small enough".

3. CLOSED SOLIDS. Electrode rectangles are closed sets: touching the
   surface (tangent graze, exact rim contact) IS a hit. This matches the
   the electrode-surface convention used by the CSG importer.

4. FIRST EVENT WINS. Within a step, the earliest chord parameter decides:
   electrode hit vs exit-plane crossing vs domain departure. Exit-crossing
   arithmetic is kept bit-identical to the frozen Gate-C ``fly`` so that
   detection is purely observational for surviving rays.

5. NOTHING SILENT. Every ray ends in exactly one reported state:
   EXIT (crossed s_end), HIT (electrode label + position + time),
   LOST (left the solve domain other than through s_end), or STUCK
   (t_max exhausted). Leaving the grid no longer coasts on edge-clamped
   interpolation; hitting metal no longer coasts through field-free
   conductor interiors. Downstream metrics must consume the status --
   budget/physics gates assert all-EXIT; acceptance scans report loss
   fractions.

The integrator (velocity-Verlet, bilinear field interpolation, linear
in-step event interpolation) is copied verbatim from the frozen
reference-tracer equivalence fixture; the gate holds surviving-ray
observables to BIT-IDENTICAL, not merely tolerant.
"""
import numpy as np

ACC = 96.485          # mm/us^2 per (V/mm / Da): a = ACC * q * E / m
DT = 1e-3             # us  (must match the frozen Gate-C flyer)

EXIT, HIT, LOST, STUCK, PLANE = 0, 1, 2, 3, 4
STATUS_NAMES = {EXIT: "EXIT", HIT: "HIT", LOST: "LOST", STUCK: "STUCK",
                PLANE: "PLANE"}


# ---------------------------------------------------------------------------
# geometry: analytic rectangles from the stack's own plate table
# ---------------------------------------------------------------------------
class PlaneGeometry:
    """Closed axis-aligned electrode rectangles in the (s, u) width plane,
    plus the solve-domain box. Built from the stack module's
    ``plate_positions()`` -- the same analytic source ``build_plane``
    rasterises -- so the flyer and the solver share one geometry truth."""

    def __init__(self, rects, labels, s_lo, s_hi, u_half):
        # rects: (N, 4) float array of [s0, s1, u0, u1], closed
        self.rects = np.asarray(rects, float).reshape(-1, 4)
        self.labels = list(labels)
        self.s_lo, self.s_hi, self.u_half = s_lo, s_hi, u_half

    @classmethod
    def from_stack(cls, stack, plane="width", s_lo=-3.0, s_hi=None,
                   u_half=None, drift_rails=None):
        """Build from a stack module exposing plate_positions().
        Mirrors build_plane's construction exactly:
          solid plate (L is None)  -> full-height rectangle
          slotted plate            -> two rim rectangles |u| >= half-gap
        drift_rails: (s_start, |u|) analyzer boards past the stack, or
        None to auto-detect compact_stack's convention (rails at u_half
        from the last plate onward when the module defines U_HALF)."""
        plates = stack.plate_positions()
        stack_end = plates[-1][3]
        if u_half is None:
            u_half = getattr(stack, "U_HALF", 20.0)
        if s_hi is None:
            s_hi = stack_end + 12.0
        if drift_rails is None and hasattr(stack, "U_HALF"):
            drift_rails = (plates[-1][2], u_half)

        rects, labels = [], []
        big = u_half + 10.0                    # rims extend past the domain
        for n, d, sf, sb, L, W in plates:
            if L is None:
                rects.append([sf, sb, -big, big])
                labels.append(d)
            else:
                half = (W if plane == "width" else L) / 2.0
                rects.append([sf, sb, half, big])
                labels.append(d)
                rects.append([sf, sb, -big, -half])
                labels.append(d)
        if drift_rails is not None:
            s_r, u_r = drift_rails
            rects.append([s_r, s_hi + 10.0, u_r, big])
            labels.append("F")
            rects.append([s_r, s_hi + 10.0, -big, -u_r])
            labels.append("F")
        return cls(rects, labels, s_lo, s_hi, u_half)

    # -- vectorised first-contact test -------------------------------------
    def first_hit(self, s, u, s_n, u_n):
        """First intersection of chords (s,u)->(s_n,u_n) with any closed
        rectangle. Returns (f, idx): chord parameter in [0,1] (inf when no
        hit) and rectangle index (-1 when no hit). Slab method; boundary
        contact counts (closed solids)."""
        s = np.atleast_1d(np.asarray(s, float))
        u = np.atleast_1d(np.asarray(u, float))
        s_n = np.atleast_1d(np.asarray(s_n, float))
        u_n = np.atleast_1d(np.asarray(u_n, float))
        m, r = s.size, self.rects.shape[0]
        if r == 0:
            # pure-drift geometry (no electrodes): nothing to hit
            return np.full(m, np.inf), np.full(m, -1)
        ds = (s_n - s)[:, None]
        du = (u_n - u)[:, None]
        S0, S1 = self.rects[:, 0][None, :], self.rects[:, 1][None, :]
        U0, U1 = self.rects[:, 2][None, :], self.rects[:, 3][None, :]
        sB = s[:, None]
        uB = u[:, None]

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (S0 - sB) / ds
            t2 = (S1 - sB) / ds
            slo = np.minimum(t1, t2)
            shi = np.maximum(t1, t2)
            # chord parallel to a slab axis: inside -> (-inf, +inf), else miss
            par = ds == 0
            inside = (sB >= S0) & (sB <= S1)
            slo = np.where(par, np.where(inside, -np.inf, np.inf), slo)
            shi = np.where(par, np.where(inside, np.inf, -np.inf), shi)

            t1 = (U0 - uB) / du
            t2 = (U1 - uB) / du
            ulo = np.minimum(t1, t2)
            uhi = np.maximum(t1, t2)
            par = du == 0
            inside = (uB >= U0) & (uB <= U1)
            ulo = np.where(par, np.where(inside, -np.inf, np.inf), ulo)
            uhi = np.where(par, np.where(inside, np.inf, -np.inf), uhi)

        lo = np.maximum(np.maximum(slo, ulo), 0.0)
        hi = np.minimum(np.minimum(shi, uhi), 1.0)
        ok = lo <= hi                          # closed: touching counts
        f = np.where(ok, lo, np.inf)           # (m, r)
        idx = np.argmin(f, axis=1)
        fmin = f[np.arange(m), idx]
        idx = np.where(np.isfinite(fmin), idx, -1)
        return fmin, idx

    def domain_exit(self, s, u, s_n, u_n):
        """Chord parameter where a step leaves the solve domain box
        (excluding the s_end exit, handled separately). inf when it stays
        inside."""
        s = np.atleast_1d(np.asarray(s, float))
        u = np.atleast_1d(np.asarray(u, float))
        s_n = np.atleast_1d(np.asarray(s_n, float))
        u_n = np.atleast_1d(np.asarray(u_n, float))
        f = np.full(s.shape, np.inf)
        for a, a_n, lo, hi in ((s, s_n, self.s_lo, self.s_hi),
                               (u, u_n, -self.u_half, self.u_half)):
            d = a_n - a
            with np.errstate(divide="ignore", invalid="ignore"):
                f_lo = np.where((d < 0) & (a_n < lo), (lo - a) / d, np.inf)
                f_hi = np.where((d > 0) & (a_n > hi), (hi - a) / d, np.inf)
            f = np.minimum(f, np.minimum(f_lo, f_hi))
        return f


# ---------------------------------------------------------------------------
# checked flyer
# ---------------------------------------------------------------------------
def fly_checked(field, geom, s0, u0, vs0=None, vu0=None, mz=1000.0,
                s_end=None, dt=DT, t_max=8.0, planes=None):
    """Velocity-Verlet identical to the frozen Gate-C ``fly`` (same
    arithmetic, same exit interpolation), plus per-step first-event
    detection per the E1 convention. Never raises on hits/losses.

    planes: optional list of plane_gate.Plane (TASK P0). terminate planes
    join first-event-wins ranked after the frozen s_end exit and before
    electrode hits at equal chord fraction (fate PLANE, label = plane
    name; finite u window = aperture, outside rays pass). record/port
    planes are transparent: qualifying crossings are appended to the
    returned res['crossings'][name] Recording, capped at each ray's
    terminal chord fraction on its final step. planes=None is the
    original frozen path, bit-identical (gate P0a).

    Returns dict with per-ray arrays:
      status  : EXIT / HIT / LOST / STUCK / PLANE
      t, s, u : event time and position (exit plane, hit point, or domain
                crossing; last state if STUCK)
      vs, vu  : velocity at the event (linear in-step interpolation,
                matching the frozen exit convention)
      label   : electrode designation for HIT rays, plane name for PLANE
                rays, '' otherwise
      e_drift : energy-conservation diagnostic for EXIT rays (nan otherwise)
      crossings : dict name -> plane_gate.Recording (record/port planes)
    """
    if s_end is None:
        raise ValueError("s_end must be given explicitly")
    if planes is not None:
        from ion_gym.physics.plane_gate import Recording, crossing_fraction
        term_planes = [p for p in planes if p.role == "terminate"]
        rec_planes = [p for p in planes if p.role in ("record", "port")]
        names = [p.name for p in planes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate plane names: {names}")
        recs = {p.name: Recording(p) for p in rec_planes}
    else:
        term_planes, rec_planes, recs = [], [], {}
    s = np.array(s0, float)
    u = np.array(u0, float)
    vs = np.zeros_like(s) if vs0 is None else np.array(vs0, float)
    vu = np.zeros_like(s) if vu0 is None else np.array(vu0, float)
    t = np.zeros_like(s)
    n = s.size
    done = np.zeros(n, bool)
    status = np.full(n, STUCK, np.int8)
    te = np.zeros(n)
    se = np.array(s, float)
    ue = np.array(u, float)
    vse = np.zeros(n)
    vue = np.zeros(n)
    label = np.array([""] * n, dtype="U16")

    k = ACC / mz
    Es, Eu = field.E(s, u)
    E0 = mz * (vs**2 + vu**2) / (2 * ACC) + field.pot(s, u)

    def record(mask, f, st, s_n, u_n, vs_n, vu_n):
        te[mask] = t[mask] + f[mask] * dt
        se[mask] = s[mask] + f[mask] * (s_n[mask] - s[mask])
        ue[mask] = u[mask] + f[mask] * (u_n[mask] - u[mask])
        vse[mask] = vs[mask] + f[mask] * (vs_n[mask] - vs[mask])
        vue[mask] = vu[mask] + f[mask] * (vu_n[mask] - vu[mask])
        status[mask] = st

    nstep = int(t_max / dt)
    for _ in range(nstep):
        vs_h = vs + 0.5 * dt * k * Es
        vu_h = vu + 0.5 * dt * k * Eu
        s_n = s + dt * vs_h
        u_n = u + dt * vu_h
        Es_n, Eu_n = field.E(s_n, u_n)
        vs_n = vs_h + 0.5 * dt * k * Es_n
        vu_n = vu_h + 0.5 * dt * k * Eu_n

        live = ~done
        if live.any():
            # candidate events on this step's chord
            f_hit, idx = geom.first_hit(s, u, s_n, u_n)
            f_dom = geom.domain_exit(s, u, s_n, u_n)
            with np.errstate(divide="ignore", invalid="ignore"):
                f_exit = np.where(s_n >= s_end,
                                  (s_end - s) / (s_n - s), np.inf)

            if planes is None:
                # frozen path -- ORIGINAL expressions, untouched (gate P0a)
                f_evt = np.minimum(np.minimum(f_hit, f_dom), f_exit)
                has = live & np.isfinite(f_evt)

                m_exit = has & (f_exit <= f_hit) & (f_exit <= f_dom)
                m_hit = has & ~m_exit & (f_hit <= f_dom)
                m_lost = has & ~m_exit & ~m_hit
                m_plane = np.zeros(n, bool)
                f_pl = None
            else:
                if term_planes:
                    Ft = crossing_fraction(term_planes, s, u, s_n, u_n,
                                           live)
                    ipl = np.argmin(Ft, axis=0)
                    f_pl = Ft[ipl, np.arange(n)]
                else:
                    f_pl = np.full(n, np.inf)
                    ipl = np.zeros(n, int)
                f_evt = np.minimum(np.minimum(f_hit, f_dom),
                                   np.minimum(f_exit, f_pl))
                has = live & np.isfinite(f_evt)

                # frozen exit keeps precedence; terminate planes outrank
                # metal at equal fraction (detector face on a plate is
                # the detector)
                m_exit = has & (f_exit <= f_hit) & (f_exit <= f_dom) \
                    & (f_exit <= f_pl)
                m_plane = has & ~m_exit & (f_pl <= f_hit) & (f_pl <= f_dom)
                m_hit = has & ~m_exit & ~m_plane & (f_hit <= f_dom)
                m_lost = has & ~m_exit & ~m_plane & ~m_hit

            if m_exit.any():
                # bit-identical to frozen fly: interpolate at f_exit
                record(m_exit, f_exit, EXIT, s_n, u_n, vs_n, vu_n)
                se[m_exit] = s_end
            if m_plane.any():
                record(m_plane, f_pl, PLANE, s_n, u_n, vs_n, vu_n)
                for i in np.where(m_plane)[0]:
                    se[i] = term_planes[ipl[i]].s0
                    label[i] = term_planes[ipl[i]].name
            if m_hit.any():
                record(m_hit, f_hit, HIT, s_n, u_n, vs_n, vu_n)
                for i in np.where(m_hit)[0]:
                    label[i] = geom.labels[idx[i]]
            if m_lost.any():
                record(m_lost, f_dom, LOST, s_n, u_n, vs_n, vu_n)
            done |= has

            if rec_planes:
                # transparent crossings, capped at each ray's terminal
                # fraction on this step (a crossing after death is not a
                # crossing; simultaneous with death counts)
                f_cap = np.full(n, np.inf)
                f_cap[m_exit] = f_exit[m_exit]
                f_cap[m_plane] = f_pl[m_plane]
                f_cap[m_hit] = f_hit[m_hit]
                f_cap[m_lost] = f_dom[m_lost]
                Fr = crossing_fraction(rec_planes, s, u, s_n, u_n, live)
                for ip, p in enumerate(rec_planes):
                    f = Fr[ip]
                    m = np.isfinite(f) & (f <= f_cap)
                    if m.any():
                        recs[p.name]._append(
                            np.where(m)[0],
                            t[m] + f[m] * dt,
                            u[m] + f[m] * (u_n[m] - u[m]),
                            vs[m] + f[m] * (vs_n[m] - vs[m]),
                            vu[m] + f[m] * (vu_n[m] - vu[m]))

        s, u, vs, vu = s_n, u_n, vs_n, vu_n
        Es, Eu = Es_n, Eu_n
        t = t + dt
        if done.all():
            break

    stuck = ~done
    if stuck.any():
        te[stuck] = t[stuck]
        se[stuck] = s[stuck]
        ue[stuck] = u[stuck]
        vse[stuck] = vs[stuck]
        vue[stuck] = vu[stuck]

    e_drift = np.full(n, np.nan)
    ex = status == EXIT
    if ex.any():
        E1 = mz * (vse[ex]**2 + vue[ex]**2) / (2 * ACC) + field.pot(
            np.full(ex.sum(), s_end), ue[ex])
        e_drift[ex] = np.abs(E1 - E0[ex]) / np.abs(E0).max()

    return dict(status=status, t=te, s=se, u=ue, vs=vse, vu=vue,
                label=label, e_drift=e_drift, crossings=recs)


def loss_report(res):
    """One-line human summary of a fly_checked result."""
    st = res["status"]
    n = st.size
    parts = [f"{(st == c).sum()}/{n} {STATUS_NAMES[c]}"
             for c in (EXIT, HIT, LOST, STUCK) if (st == c).any()]
    hits = st == HIT
    if hits.any():
        where = ", ".join(
            f"{res['label'][i]}@(s={res['s'][i]:.2f},u={res['u'][i]:+.2f})"
            for i in np.where(hits)[0][:6])
        parts.append("hits: " + where + (" ..." if hits.sum() > 6 else ""))
    return "; ".join(parts)


# ===================================================================
# The frozen reference flyer family — MOVED HERE from the validation
# harnesses (a function lives in the module that owns the data it
# operates on). This module already
# owned ACC/DT "(must match the frozen reference flyer)"; now it owns the
# flyer itself. Gates depend on core; never the reverse.
#
# Behavior-identical to the gate originals except that silently-captured
# module globals became EXPLICIT parameters: Field takes h; fly and
# space_focus/spread_at take mz/s_end. Every call site passes the value
# the old module-global supplied (h=0.05, mz=1000, per-stack s_end).
# ===================================================================

SIG300 = 0.049935     # mm/us, 1-D rms thermal speed at 300 K, m/z 1000
                      # (K10 single source: was re-typed in two gates)


class Field:
    """Bilinear-interpolated planar (s, u) field from node arrays.
    h is the grid spacing in mm — EXPLICIT (the gate original silently
    captured its module's H)."""

    def __init__(self, V, Es, Eu, ss, uu, h):
        self.V, self.Es, self.Eu = V, Es, Eu
        self.s0, self.u0, self.h = ss[0], uu[0], h
        self.ns, self.nu = V.shape

    def _interp(self, A, s, u):
        x = np.clip((s - self.s0) / self.h, 0, self.ns - 1.001)
        y = np.clip((u - self.u0) / self.h, 0, self.nu - 1.001)
        i = x.astype(int); j = y.astype(int)
        fx = x - i; fy = y - j
        return (A[i, j] * (1 - fx) * (1 - fy) + A[i + 1, j] * fx * (1 - fy)
                + A[i, j + 1] * (1 - fx) * fy + A[i + 1, j + 1] * fx * fy)

    def E(self, s, u):
        return self._interp(self.Es, s, u), self._interp(self.Eu, s, u)

    def pot(self, s, u):
        return self._interp(self.V, s, u)


class NullField:
    """E = 0 everywhere, V = 0 -- ballistic test field."""
    def E(self, s, u):
        z = np.zeros_like(np.asarray(s, float))
        return z, z.copy()

    def pot(self, s, u):
        return np.zeros_like(np.asarray(s, float))


class UniformField:
    """Constant E_s [V/mm], E_u = 0."""
    def __init__(self, Es):
        self.Es = Es

    def E(self, s, u):
        s = np.asarray(s, float)
        return np.full_like(s, self.Es), np.zeros_like(s)

    def pot(self, s, u):
        return -self.Es * np.asarray(s, float)


def combine(units, volts):
    V = sum(volts[n] * units[n][0] for n in volts)
    Es = sum(volts[n] * units[n][1] for n in volts)
    Eu = sum(volts[n] * units[n][2] for n in volts)
    return V, Es, Eu


def fly(field, s0, u0, vs0=None, vu0=None, *, mz, s_end, dt=DT, t_max=8.0):
    """Vectorised velocity-Verlet to the field-free exit plane. Returns
    (t_exit, vs, vu, u_at_exit, e_drift) with exact crossing interpolation.
    The frozen Gate-C reference flyer. mz and s_end are REQUIRED (the gate
    original defaulted them from its instrument's module globals)."""
    s = np.array(s0, float); u = np.array(u0, float)
    vs = np.zeros_like(s) if vs0 is None else np.array(vs0, float)
    vu = np.zeros_like(s) if vu0 is None else np.array(vu0, float)
    t = np.zeros_like(s)
    done = np.zeros(s.shape, bool)
    te = np.zeros_like(s); vse = np.zeros_like(s)
    vue = np.zeros_like(s); ue = np.zeros_like(s)
    k = ACC / mz
    Es, Eu = field.E(s, u)
    E0 = mz * (vs**2 + vu**2) / (2 * ACC) + field.pot(s, u)
    nstep = int(t_max / dt)
    for _ in range(nstep):
        vs_h = vs + 0.5 * dt * k * Es
        vu_h = vu + 0.5 * dt * k * Eu
        s_n = s + dt * vs_h
        u_n = u + dt * vu_h
        Es_n, Eu_n = field.E(s_n, u_n)
        vs_n = vs_h + 0.5 * dt * k * Es_n
        vu_n = vu_h + 0.5 * dt * k * Eu_n
        cross = (~done) & (s_n >= s_end)
        if cross.any():
            f = (s_end - s[cross]) / (s_n[cross] - s[cross])
            te[cross] = t[cross] + f * dt
            vse[cross] = vs[cross] + f * (vs_n[cross] - vs[cross])
            vue[cross] = vu[cross] + f * (vu_n[cross] - vu[cross])
            ue[cross] = u[cross] + f * (u_n[cross] - u[cross])
            done |= cross
        s, u, vs, vu = s_n, u_n, vs_n, vu_n
        Es, Eu = Es_n, Eu_n
        t = t + dt
        if done.all():
            break
    if not done.all():
        raise RuntimeError("ions did not exit; check state/geometry")
    E1 = mz * (vse**2 + vue**2) / (2 * ACC) + field.pot(
        np.full_like(te, s_end), ue)
    return te, vse, vue, ue, np.abs(E1 - E0) / np.abs(E0).max()


def space_focus(te, vse, s_end):
    """T_i(L) = te_i + (L - s_end)/vse_i is linear in L; the L* minimising
    var(T) solves a scalar quadratic -> closed form."""
    a = 1.0 / vse
    b = te - s_end / vse
    va = a - a.mean(); vb = b - b.mean()
    L = -np.dot(va, vb) / np.dot(va, va)
    T = a * L + b
    return L, T.max() - T.min()


def spread_at(te, vse, L, s_end):
    T = te + (L - s_end) / vse
    return T.max() - T.min()
