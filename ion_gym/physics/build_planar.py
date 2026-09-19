"""
ion_gym.build_planar
--------------------
The first FULLY INDEPENDENT builder: a SimSpec with planar symmetry and
inline shapes -> rasterized electrode masks -> natively-solved fast-adjust
bases (solver3d, no node-centred grid anywhere) -> a field model + fly_fn.

This is the einzel path, and the proof that "spec -> solve -> fly" needs
no external tool to RUN (an external solve stays available only for optional
cross-checks). Geometry is a 2-D cartesian slice in the x-y plane; the
solve is a single-slab 3-D solve (nz=1) so it reuses the validated
solver3d kernel and its pinned conventions directly.

Shape semantics match ion_playground exactly (rect top-left + w/h in mm;
ellipse cx/cy/rx/ry; cutout subtracts) so an ion_playground einzel scene
and this builder rasterize identically.
"""

import math

import numpy as np
from numba import njit

from ion_gym.io.sim_spec import SimSpec
# 2-D raster substrate lives in raster2d — import
# from the owner (no re-exports).
from ion_gym.physics.raster2d import (
    plane_grid_views,
    electrode_mask,
    el_masks_from_labels,
    planar_fold_axes,
    _fold_crop,
    _fold_reflect,
    anchored_grid,
    _metal_nn,
    refuse_birth_in_metal)
from ion_gym.io.sim_spec import BASE_CHANNELS
from ion_gym.physics.sim_build import generate_births
from ion_gym.physics.symmetry import (verify_symmetry, verify_symmetry_shapes, reduction_summary)
from ion_gym.physics.collision3d import (E_CHG, KG_AMU, KB, gas_mass, _mfp_mm, _collide)
from ion_gym.physics import sds as _sds_mod
from ion_gym.physics.sds import (_diff_dist_steps, ion_params as _sds_ion_params,
                 load_diffusion_statistics as _sds_load_stats,
                 load_massdata as _sds_load_mass,
                 N_DIST_COLLISIONS as _SDS_NDC)




# ------------------------------------------------------- drive waveforms
# Channel kinds shared by the kernel and the python-side evaluator:
#   0 sin(om t + ph)   1 cos(om t + ph)   2 sign(sin(om t + ph))
#   3 table, zero-order hold              4 table, linear interp
# Tables clamp to their end values outside the breakpoint range. ALL
# waveforms run on the LAB clock (tob + flight time) — fields are
# lab-frame objects; a birth stagger changes the phase an ion is born
# into, which is the physics.
K_SIN, K_COS, K_SQUARE, K_TAB_HOLD, K_TAB_LIN = 0, 1, 2, 3, 4


def _wave_py(g, t_us):
    """Python-side unit waveform of an RFGroupSpec at lab time t_us —
    the exact mirror of the kernel's _wave_eval, for PE/plot use."""
    om = g.frequency_hz * 1e-6 * 2 * math.pi
    ph = math.radians(g.phase_deg)
    if g.waveform == "sin":
        return math.sin(om * t_us + ph)
    if g.waveform == "square":
        d = float(getattr(g, "duty", 0.5))
        if d == 0.5:
            return 1.0 if math.sin(om * t_us + ph) >= 0.0 else -1.0
        return 1.0 if ((om * t_us + ph) / (2.0 * math.pi)) % 1.0 < d else -1.0
    tt = np.asarray(g.table_t_us, float)
    vv = np.asarray(g.table_v, float)
    if t_us <= tt[0]:
        return float(vv[0])
    if t_us >= tt[-1]:
        return float(vv[-1])
    i = int(np.searchsorted(tt, t_us, side="right") - 1)
    if g.interp == "hold":
        return float(vv[i])
    f = (t_us - tt[i]) / (tt[i + 1] - tt[i])
    return float(vv[i] + f * (vv[i + 1] - vv[i]))


# ------------------------------------------------------------- field model
# The excitation the bases are SOLVED at (solver3d / multigrid3d v_basis).
# It lived as a default kwarg in two solvers and as a bare `/ 1e4` here: three
# copies of one number, so moving the solver default would have silently
# rescaled every composed field with nothing going red.  ONE definition; it is
# passed to the solver explicitly and divided out explicitly.
V_BASIS = 1e4


class PlanarModel:

    """Planar field model: A (static) + N drive CHANNELS, all on the x-y
    grid, plus the electrode mask. Mirrors the shared r-z model plotting surface
    API so the viewer is shared.

    DRIVE CHANNELS (the reference-scripting equivalent): the field is linear
    in electrode voltages, so any state-independent drive program is
        E(x, t) = E_A(x) + sum_k w_k(t) * E_k(x)
    with precomposed basis fields E_k and scalar waveforms w_k. Sin groups
    sharing a frequency collapse to ONE quadrature pair per frequency
    (sum_g sin(wt+phi_g) B_g = sin(wt) Bs + cos(wt) Bc), so an 8-phase TW
    costs two channels; square/table groups get one channel each. Multi-
    frequency drives (SLIM: ~40 kHz TW + ~MHz RF rungs) are just more
    channels — the v61 single-frequency refusal is lifted.

    The SIMULATION always integrates these real time-varying fields.
    The pseudopotential exists only in pe_surface (visualization)."""
    # Which principal planes this model HAS.  Declared, never sniffed.
    # A planar solve has NO z: xz/yz would be a fabricated third dimension.
    PLANES = ('xy',)

    def __init__(self, A, Bk, ele, Ex, Ey, mm_per_gu, spec, el_bands=None,
                 drives=None, bases=None, anchor_mm=(0.0, 0.0)):
        # Sub-cell grid anchor: node i sits at i*h + anchor. (0,0)
        # for every undeclaring spec. THE frame authority — extent(),
        # bands, and the fly boundary all map through it, so displayed
        # coordinates equal the user's spec frame exactly.
        self.anchor_mm = (float(anchor_mm[0]), float(anchor_mm[1]))
        self.A = A
        # Named by the deck's own electrode names, so a shape
        # in a figure can be matched to an entry in the spec.
        self.el_masks = el_masks_from_labels(
            ele, getattr(spec.geometry, 'electrodes', None))
        # PER-ELECTRODE BASES, kept (a reference; they are already in memory
        # and in the cache).  The field is linear in electrode voltages, so
        # basis_i / 1e4 IS the dimensionless response phi/V_i with every other
        # electrode grounded -- which is what a coupling coefficient (an axial
        # penetration alpha, a pickup fraction, a divider ratio) is DEFINED as.
        #
        # Kept because the alternative is what the Q3 report actually did: it
        # built a SECOND SimSpec with the probe electrode at 1 V and everything
        # else at 0, and solved it again, to read a number the solver had
        # already computed and thrown away.  A viewer that has to re-solve the
        # model to draw it is not showing you the model.  Keys are 1-based,
        # matching the int16 labels in `ele`.
        self.Bk = Bk                 # sin groups as (B, freq_hz, phase_deg)
        self.ele = ele
        self.ExA, self.EyA = Ex, Ey
        self.mm_per_gu = mm_per_gu
        self.spec = spec
        # per-electrode transverse extents (x_lo,x_hi,y_lo,y_hi) in mm,
        # for drawing rod bands in the transport (xz/yz) views. The 2-D
        # solve is a cross-section; electrodes are translationally
        # invariant along the transport axis, so each projects to a band.
        self.el_bands = el_bands or []
        self.bases = bases if bases is not None else {}

        # drives: list of (B_phi, RFGroupSpec). Back-compat: callers that
        # only pass Bk (sin tuples) get synthesized sin drives.
        if drives is None:
            from ion_gym.io.sim_spec import RFGroupSpec
            drives = [(B0, RFGroupSpec(name=f"_bk{i}", frequency_hz=f0,
                                       amplitude_v=1.0, phase_deg=ph0))
                      for i, (B0, f0, ph0) in enumerate(Bk)]
        self.drives = drives

        # ---- compose channels ------------------------------------------
        # sin groups -> per-frequency quadrature pairs, in first-appearance
        # order (so the single-frequency case reproduces the v61 kernel's
        # A + sin*Bs + cos*Bc arithmetic bit-for-bit). square/table groups
        # append one channel each, in spec order.
        sin_by_f = {}
        others = []
        for B0, g in drives:
            g.validate()
            if g.waveform == "sin":
                key = round(g.frequency_hz, 6)
                ent = sin_by_f.setdefault(
                    key, [np.zeros_like(A), np.zeros_like(A)])
                ent[0] += math.cos(math.radians(g.phase_deg)) * B0
                ent[1] += math.sin(math.radians(g.phase_deg)) * B0
            else:
                others.append((B0, g))

        chan_phi, kinds, oms, phs, duties, tabs = [], [], [], [], [], []
        for f0, (Bs, Bc) in sin_by_f.items():
            om = f0 * 1e-6 * 2 * math.pi                 # rad/us
            chan_phi += [Bs, Bc]
            kinds += [K_SIN, K_COS]
            oms += [om, om]
            phs += [0.0, 0.0]                            # folded into Bs/Bc
            duties += [0.5, 0.5]                         # unused by sin/cos
            tabs += [([], []), ([], [])]
        for B0, g in others:
            om = g.frequency_hz * 1e-6 * 2 * math.pi
            chan_phi.append(B0)
            oms.append(om)
            phs.append(math.radians(g.phase_deg))
            if g.waveform == "square":
                kinds.append(K_SQUARE)
                duties.append(float(g.duty))
                tabs.append(([], []))
            else:
                kinds.append(K_TAB_HOLD if g.interp == "hold" else K_TAB_LIN)
                duties.append(0.5)                       # unused by tables
                tabs.append((list(g.table_t_us), list(g.table_v)))

        self.chan_phi = chan_phi
        Kn = len(chan_phi)
        if Kn:
            ExK, EyK = [], []
            for P in chan_phi:
                ex, ey = _grad2d(np.ascontiguousarray(P), mm_per_gu)
                ExK.append(ex)
                EyK.append(ey)
            self.ExK = np.ascontiguousarray(np.stack(ExK))
            self.EyK = np.ascontiguousarray(np.stack(EyK))
        else:
            self.ExK = np.zeros((0,) + A.shape)
            self.EyK = np.zeros((0,) + A.shape)
        self.ch_kind = np.array(kinds, np.int64)
        self.ch_om = np.array(oms, np.float64)
        self.ch_ph = np.array(phs, np.float64)
        self.ch_duty = np.array(duties, np.float64)
        off = [0]
        tt, tv = [], []
        for (a, b) in tabs:
            tt += list(a)
            tv += list(b)
            off.append(len(tt))
        self.tab_t = np.array(tt, np.float64)
        self.tab_v = np.array(tv, np.float64)
        self.tab_off = np.array(off, np.int64)

        # ---- legacy attributes (pe_view, older tests): the FIRST sin
        # frequency's quadrature pair, zeros if none. ------------------
        if sin_by_f:
            f1, (Bs1, Bc1) = next(iter(sin_by_f.items()))
            self.ExB, self.EyB = self.ExK[0], self.EyK[0]
            self.ExBc, self.EyBc = self.ExK[1], self.EyK[1]
            self.rf_om = f1 * 1e-6 * 2 * math.pi
        else:
            z = np.zeros_like(A)
            self.ExB, self.EyB = z, z
            self.ExBc, self.EyBc = z, z
            self.rf_om = 0.0
        self.rf_phase0 = 0.0

    def phi_at(self, t_us):
        """Instantaneous potential on the grid at LAB time t_us:
        A + sum_g w_g(t) B_g over every drive group. This is what the
        tracer integrates (the real field), and the honest picture of a
        slow travelling wave."""
        P = self.A.copy()
        for B0, g in self.drives:
            P = P + _wave_py(g, t_us) * B0
        return P

    def extent(self):
        nx, ny = self.A.shape
        ax, ay = self.anchor_mm
        return (np.arange(nx) * self.mm_per_gu + ax,
                np.arange(ny) * self.mm_per_gu + ay)

    def pe_surface(self, mz=None, charge=1, t_us=0.0, plane="xy"):
        """Effective potential-energy landscape in eV — VISUALIZATION
        ONLY (the tracer always integrates the real time-varying field).

        Per drive group, by its resolved pe_mode:
          'pseudo'  (fast confinement RF): Dehmelt envelope
                        V = q |E0|^2 / (4 m Omega^2)
                    summed per frequency (sin groups quadrature-compose
                    per frequency; valid for well-separated frequencies
                    and the adiabatic regime, Mathieu q <~ 0.4). Square
                    drives (digital trap) carry the harmonic-sum factor
                    pi^2/6 (sum over odd n of (4/n pi)^2 / n^2).
          'instant' (slow TW the ions SURF, not average): the real
                    instantaneous potential w(t_us) * B at view time
                    t_us. This is the physical picture for a ~40 kHz
                    travelling wave — the adiabatic average does not
                    exist for it.
        Defaults: sin -> pseudo, square/table -> instant (overridable per
        group via RFGroupSpec.pe_mode). Returns (x, y, PE_eV, ele)."""
        if plane not in self.PLANES:
            raise ValueError(
                f"{type(self).__name__} has no {plane!r} plane; it has "
                f"{self.PLANES}. (Declared capability -- not a signature sniff.)")
        x, y = self.extent()
        if mz is None:
            mz = self.spec.source.mz_list[0]
        m_kg = mz * 1.6605402e-27
        q_c = charge * 1.602176634e-19
        pe = charge * self.A.copy()               # DC part: z*phi in eV
        h = self.mm_per_gu

        pseudo_sin = {}          # freq -> [Bs, Bc]
        pseudo_sq = {}           # freq -> {phase folding}
        for B0, g in self.drives:
            mode = g.resolved_pe_mode()
            if mode == "instant":
                pe = pe + charge * _wave_py(g, t_us) * B0
                continue
            if g.waveform == "sin":
                ent = pseudo_sin.setdefault(
                    round(g.frequency_hz, 6),
                    [np.zeros_like(self.A), np.zeros_like(self.A)])
                ent[0] += math.cos(math.radians(g.phase_deg)) * B0
                ent[1] += math.sin(math.radians(g.phase_deg)) * B0
            elif g.waveform == "square":
                # square groups only fold when mutually in/anti-phase
                # (0/180 deg apart) — the digital-trap / RF-rung case.
                fk = round(g.frequency_hz, 6)
                ent = pseudo_sq.setdefault(fk, {})
                ph = g.phase_deg % 360.0
                base = min(ent.keys(), default=ph)
                d = (ph - base) % 360.0
                if min(d, 360.0 - d) > 1e-6 and abs(d - 180.0) > 1e-6:
                    raise ValueError(
                        "pe_surface: square-wave pseudo groups at "
                        f"{fk:g} Hz have phases neither in- nor anti-"
                        "phase; the envelope is ill-defined — set "
                        "pe_mode='instant' on those groups.")
                sign = 1.0 if min(d, 360.0 - d) <= 1e-6 else -1.0
                ent.setdefault(base, np.zeros_like(self.A))
                ent[base] += sign * B0
            else:
                raise ValueError(
                    f"pe_surface: table group {g.name!r} has no adiabatic "
                    "envelope — use pe_mode='instant'.")

        def _dehmelt(Bs, Bc, f_hz, factor=1.0):
            exs, eys = _grad2d(np.ascontiguousarray(Bs), h)
            e0sq = exs ** 2 + eys ** 2
            if Bc is not None:
                exc, eyc = _grad2d(np.ascontiguousarray(Bc), h)
                e0sq = e0sq + exc ** 2 + eyc ** 2
            om = f_hz * 2 * math.pi                      # rad/s
            return factor * q_c * e0sq / (4.0 * m_kg * om ** 2)   # volts

        for f0, (Bs, Bc) in pseudo_sin.items():
            pe = pe + charge * _dehmelt(Bs, Bc, f0)
        for f0, ent in pseudo_sq.items():
            for _base, Beff in ent.items():
                pe = pe + charge * _dehmelt(Beff, None, f0,
                                            factor=math.pi ** 2 / 6.0)
        return x, y, pe, self.ele

    def potential_image(self, rf_phase=None):
        # UNIFORM CONTRACT (all three models accept `rf_phase`).  This model's
        # picture is ALREADY the static field plus every drive group at its
        # peak, so a specific snapshot phase is not representable here -- and
        # silently ignoring it would show a picture that is NOT what was asked
        # for.  Refuse with the pointer instead.  (The old world had two
        # contracts under one name, bridged by an `except TypeError` sniff in
        # sim_app -- the same defect D1 killed three times in pe_view.)
        if rf_phase is not None:
            raise ValueError(
                "PlanarModel.potential_image() already folds every drive at "
                "its PEAK; it cannot render a specific rf_phase snapshot. "
                "For the time-resolved field use phi_at(t_us).")
        x, y = self.extent()
        # show the static field PLUS the drives at the REFERENCE PEAK
        # instant (w t = pi/2): a sin drive contributes
        # A*sin(pi/2 + phi) = A*cos(phi) -- the same cos(phase) fold the
        # kernel's Bs channel and pe_surface's quadrature composition
        # already use, and the composition certified for the 3-D route
        # (E_peak amps = ch_amp*cos(ch_ph)). Summing every group
        # at +1 instead was measured on the analytic
        # quadrupole (RFA 0 deg / RFB 180 deg): the anti-phase pair's
        # quadrupole term cancelled and the "potential" was the constant
        # common mode (241.335 V everywhere, max-min ~ 1e-11), so the
        # contour overlay drew nothing. cos(0) = 1 keeps every
        # single-group phase-0 deck byte-identical. Non-sin drives
        # (square/table) keep their prior full-basis convention: their
        # kinds have no shared "peak instant" with the sin reference,
        # and changing their picture is not this defect.
        # For a DC device (no drives) this is just A, unchanged.
        # For the time-resolved picture use phi_at(t_us).
        P = self.A
        for B0, g in self.drives:
            w = (math.cos(math.radians(g.phase_deg))
                 if g.waveform == "sin" else 1.0)
            P = P + w * B0
        return x, y, P, self.ele

    def efield_magnitude(self):
        # |E| of the static + peak-instant drive field (matches
        # potential_image): the FOLDED channels weighted by kind --
        # K_SIN (the Bs quadrature term, sin(pi/2) = 1) at +1, K_COS
        # (the Bc term, cos(pi/2) = 0) at 0, square/table channels at
        # +1 (their prior convention, see potential_image). Summing ALL
        # channels at +1 added the Bc quadrature term into a snapshot
        # it is zero in, and composed anti-phase sin pairs as a common
        # mode (the same quadrupole measurement as potential_image).
        ex = self.ExA.copy()
        ey = self.EyA.copy()
        for k in range(self.ExK.shape[0]):
            w = (1.0 if self.ch_kind[k] != K_COS else 0.0)
            ex = ex + w * self.ExK[k]
            ey = ey + w * self.EyK[k]
        return np.hypot(ex, ey)


def assemble_drive_groups(spec, bases, v_basis=V_BASIS):
    """A (static) + per-GROUP drive potentials from per-electrode bases.

    THE one 2-D assembly (planar and stl2d call it), so a drive feature
    cannot be honoured on one route and dropped on the other:
      * every electrode contributes el.dc * basis to A;
      * an electrode in SEVERAL groups contributes its basis to each
        (membership order preserved; the old first-group-only read is
        the L-455 defect this replaces);
      * a group's offset_v is a STATIC shift of its members
        (V(t) = amplitude_v * w(t) + offset_v, sim_spec contract), so
        offset * basis folds into A exactly, for every waveform kind
        and even when amplitude_v is 0;
      * a group with amplitude_v == 0 gets no drive channel (its offset
        is already in A);
      * frequency 0 is NOT skipped: sin/square at 0 Hz are constants the
        kernel evaluates as such (the old silent skip dropped them).
    Returns (A, drives) with drives = [(B_phi, RFGroupSpec)] in
    first-appearance order; amplitude is baked into B_phi (the kernel's
    w(t) is the UNIT waveform).
    """
    g = spec.geometry
    shape = next(iter(bases.values())).shape
    A = np.zeros(shape)
    gm = {gr.name: gr for gr in (g.rf_groups or [])}
    group_B, group_obj, order = {}, {}, []
    for idx, el in enumerate(g.electrodes, start=1):
        # positional basis keying, exactly as both 2-D builds always did
        # (el.basis remapping is a 3-D/shared-basis concept; changing the
        # 2-D keying here would silently re-map planar decks)
        fa = bases[idx] / v_basis
        A = A + el.dc * fa
        for gname in el.group_names():
            if gname not in gm:
                raise ValueError(
                    f"electrode {el.name!r} names drive group {gname!r} "
                    f"not in geometry.rf_groups ({sorted(gm)})")
            drv = gm[gname]
            drv.validate()
            off = float(getattr(drv, "offset_v", 0.0))
            if off:
                A = A + off * fa
            if drv.amplitude_v == 0.0:
                continue
            if drv.name not in group_B:
                group_B[drv.name] = np.zeros(shape)
                group_obj[drv.name] = drv
                order.append(drv.name)
            group_B[drv.name] += drv.amplitude_v * fa
    return A, [(group_B[n], group_obj[n]) for n in order]


@njit(cache=True, nogil=True)
def _grad2d(phi, h):
    nx, ny = phi.shape
    Ex = np.zeros((nx, ny))
    Ey = np.zeros((nx, ny))
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            Ex[i, j] = -(phi[i + 1, j] - phi[i - 1, j]) / (2 * h) * 1e3
            Ey[i, j] = -(phi[i, j + 1] - phi[i, j - 1]) / (2 * h) * 1e3
    return Ex, Ey       # V/m (h in mm -> 1e3)


_PLANAR_BASIS_CACHE = {}


def clear_memory_cache():
    """Drop the in-process basis cache (_PLANAR_BASIS_CACHE). The disk
    cache is separate (fa_cache.clear_all). Returns the number
    of entries dropped — a visible, reported clear, not a
    silent wipe."""
    n = len(_PLANAR_BASIS_CACHE)
    _PLANAR_BASIS_CACHE.clear()
    return n


def _freeze(bases, ele):
    """Make cached arrays read-only before they are shared: the
    in-memory cache serves bases/ele BY
    REFERENCE, so every build of the same geometry aliases the same
    arrays — an in-place mutation by any caller silently corrupted every
    subsequent build in-process (proven by G3's mutation harness; the
    basis_cache .astype(bool) hazard class). Zero legitimate writers
    exist (grep-verified: every .ele[/.bases[ site reads), so freezing
    turns silent cross-build corruption into an immediate loud
    ValueError at the mutation site."""
    for b in bases.values():
        b.flags.writeable = False
    ele.flags.writeable = False
    return bases, ele


def _geometry_key(spec):
    """Hash EVERYTHING THE SOLVER SEES (geometry + grid flags + symmetry),
    and NOTHING ELSE -- so a voltage change reuses cached bases (fast-adjust:
    re-weight, no re-solve) while any change that alters the solve or the HIT
    MASK gets its own entry.

    ONE SOURCE OF TRUTH: this delegates to basis_cache.geometry_key_dict, the
    same canonicalisation the on-disk L2 cache keys on.  It used to be a
    separate hand-rolled repr() of name+shapes, which silently omitted
    is_grid, symmetry, depth_mm, stl and basis -- so two specs differing ONLY
    in (say) is_grid collided in the L1 dict and the second one flew the
    FIRST one's hit mask.  Two caches with two different notions of "the same
    geometry" is one cache too many.
    """
    # ImportError ONLY.  Catching Exception here meant that if basis_cache.key
    # RAISED, we silently computed a DIFFERENT key -- which is the exact defect
    # the docstring above warns about: two notions of "the same geometry" is one
    # cache too many.  A key that throws is a correctness bug, not a cache miss.
    try:
        from ion_gym.io import basis_cache
    except ImportError:
        pass                                # optional; fall through to local key
    else:
        return basis_cache.key(spec)
    g = spec.geometry                       # fallback: never lose is_grid
    parts = [g.width_mm, g.height_mm, g.mm_per_gu,
             getattr(g, "depth_mm", 0.0), repr(getattr(g, "symmetry", None))]
    for el in g.electrodes:
        parts += [el.name, bool(getattr(el, "is_grid", False)),
                  getattr(el, "stl", None), getattr(el, "basis", None)]
        for s in el.shapes:
            parts.append(s.type)
            parts.extend(sorted(s.params.items()))
            for c in s.children:
                parts.append(c.type)
                parts.extend(sorted(c.params.items()))
    return repr(parts)


def planar_is_cached(spec):
    """True if the planar bases for this geometry are already solved and
    cached (so a rebuild only re-weights — fast). Used by the UI to decide
    whether to show a solving spinner. Checks BOTH tiers: the in-process
    L1 dict and the on-disk L2 basis cache (a warm disk cache means a new
    process rebuilds fast too, so the spinner would be a lie)."""
    if (_geometry_key(spec), "float64") in _PLANAR_BASIS_CACHE:
        return True
    # ImportError ONLY.  `except Exception: return False` here answered "will
    # this spec need a solve?" with a confident NO whenever anything at all went
    # wrong -- a broken key, an unreadable cache root -- and the UI believed it.
    # A question we cannot answer must not be answered with a guess.
    try:
        from ion_gym.io import basis_cache
        from ion_gym.io import fa_cache
        import os
    except ImportError:
        return True                         # cannot know it is cached -> solve
    return os.path.exists(os.path.join(
        fa_cache.DEFAULT_ROOT, basis_cache.key(spec), "meta.json"))


def _assert_planar_fold_symmetry(bases, fold, h, spec, *, served_from):
    """THE A7 SYMMETRY GATE for the planar route — one body, called from
    BOTH the solve path and the basis-cache-hit path.

    Proves, never assumes, that the arrays ABOUT TO BE USED are
    bit-symmetric about each fold plane and that the E component normal
    to the plane is identically zero on the plane row. Both hold exactly
    when the fold machinery is right (reflection is a permutation; the
    central difference of a symmetric array at its own plane is
    (a-a)/2h).

    WHY THIS RUNS ON CACHE HITS TOO.
    It used to live inside the solve branch, on the reasoning that
    "cached entries re-serve arrays this gate approved when they were
    solved". That reasoning holds only if the cache key is complete, and
    a stale hit from an incomplete key is a defect class this codebase
    has hit before. Measured consequence of the old placement: on a
    warm cache every planar mirrored deck built with its fold UNCHECKED,
    and even on a COLD run a deck whose geometry a sibling variant had
    already banked skipped the gate. Checking
    the SERVED arrays makes the guarantee independent of cache-key
    completeness and makes the planar route match the 3-D route, whose
    gate already runs after channel composition on every build.

    `served_from` names the path in the PASS line and in the refusal
    context, so the operator can tell which guarantee they just got.
    Scoped to FOLDED axes only: a declared mirror that did NOT fold
    solves the full domain and its raster asymmetry is real geometry,
    not a machinery defect.
    """
    if not fold:
        return
    from ion_gym.physics.symmetry import assert_mirror_field_symmetry
    for _ax, _p in sorted(fold.items()):
        _c = "xy"[_ax]
        for _i in sorted(bases):
            if bases[_i].ndim != 2:
                raise ValueError(
                    f"planar A7 symmetry gate ({served_from}): basis "
                    f"{_i} has rank {bases[_i].ndim}, expected a 2-D "
                    f"(nx, ny) array. The gate slices the plane row with "
                    f"np.take on axis {_ax} and cannot state what it "
                    f"proved about an array of another rank.")
        # normal E on the plane row, by the SAME central difference
        # _grad2d applies there (row-sliced: the full-array _grad2d is a
        # Python double loop and the gate needs one row per basis, not
        # nx*ny nodes)
        _normal = []
        for _i in sorted(bases):
            _b = bases[_i]
            _hi = np.take(_b, _p + 1, axis=_ax)
            _lo = np.take(_b, _p - 1, axis=_ax)
            _normal.append((f"E{_c}(basis_{_i}) plane row",
                            -(_hi - _lo) / (2 * h) * 1e3))
        assert_mirror_field_symmetry(
            axis_index=_ax, plane_node=_p,
            potentials=[(f"basis_{_i}", bases[_i]) for _i in sorted(bases)],
            normal_E=_normal,
            context=(f"planar folded build, bases {served_from} "
                     f"({getattr(spec, 'name', None) or 'unnamed spec'})"))
        print(f"planar: A7 symmetry gate PASS ({served_from}): {_c}-fold "
              f"plane on node {_p} — bases bit-symmetric, E_{_c} "
              f"identically zero on the plane row")


def build_planar_model(spec: SimSpec, tol=1e-4, verbose=False,
                       solve_dtype=np.float64):
    """Rasterize + native-solve a planar SimSpec. Returns a PlanarModel.
    No external solver anywhere: masks come from the spec's shapes, bases from
    solver3d. Bases are CACHED by geometry, so re-running with different
    voltages re-weights instantly (the fast-adjust invariant)."""
    g = spec.geometry
    h = g.mm_per_gu
    # NODE-CENTRED GRID: W/h + 1 nodes span [0, W]
    # INCLUSIVE — exactly a node-centred grid. Without the +1, node-centred
    # sampling leaves the domain one cell short of the declared width
    # and the array's mirror plane misses the geometry's (the symmetry
    # checker caught this immediately, 6.5% mismatch on the einzel).
    # Un-anchored: the historical node count (round), byte-identical for
    # every pre-existing spec. Anchored axes RESIZE: a shifted
    # lattice with the old count leaves up to h/2 of the declared extent
    # uncovered AND breaks node symmetry about the plane (measured on the
    # SLIM slice: last node 5.125 < domain 5.15, plane node 52 vs
    # opposite count 51). ceil((extent - d)/h) covers the far edge; for
    # d=0 it equals the historical count on exact-multiple extents.
    xs_g, ys_g, anchor = anchored_grid(spec)
    nx, ny = len(xs_g), len(ys_g)

    # dtype is result-affecting (float32 deltas ~1e-5 rel), so it is
    # part of the IN-MEMORY key; non-float64 solves are EPHEMERAL —
    # never banked persistently (float64-only cache: certified entries
    # can never be served to or from a reduced-precision run).
    _bank64 = np.dtype(solve_dtype) == np.float64
    ckey = (_geometry_key(spec), np.dtype(solve_dtype).name)
    cached = _PLANAR_BASIS_CACHE.get(ckey)
    if cached is None:
        # L2: disk basis cache (basis_cache), keyed on GEOMETRY ONLY, so a
        # voltage change re-weights and a fresh PROCESS does not re-solve.
        # Fail-open: a cache problem must never block a solve (it is an
        # accelerator, not an authority) -- but a CORRUPT entry raises out
        # of fa_cache.load and is not silently flown.
        try:
            from ion_gym.io import basis_cache
            dbases, dele = (basis_cache.load(spec) if _bank64
                            else (None, None))
            if dbases is not None:
                cached = (dbases, dele)
                _PLANAR_BASIS_CACHE[ckey] = _freeze(*cached)
                # Reported unconditionally — a 2-minute
                # solve avoided silently is progress hidden from the
                # operator.
                print(f"planar bases: disk cache hit "
                      f"({basis_cache.key(spec)})")
        except ImportError as e:
            # The cache being unavailable is a REPORTED condition,
            # not a silent one — a cold solve where you expected warm is
            # a diagnosis someone will need.
            print(f"planar bases: basis_cache unavailable ({e}) — "
                  f"solving cold")
    if cached is not None:
        bases, ele = cached
        # A7 GATE ON THE CACHE-HIT PATH — the guarantee is uniform
        # across routes. The fold is
        # RE-DERIVED FROM THE SPEC through planar_fold_axes — the same
        # authority the solve path uses — and deliberately NOT read from
        # the cache entry: a fold description carried in the entry would
        # be trusted exactly as far as the key that served it, which is
        # the thing this gate exists to stop trusting.
        #
        # Cost is paid ONLY by decks that declare a mirror plane: the
        # rasterization below is skipped entirely otherwise, so warm
        # builds of non-mirrored decks are unchanged.
        _sym_hit = spec.geometry.symmetry.normalized()
        if any(_sym_hit.kind(a) == "mirror" for a in ("x", "y")):
            _Xh, _Yh = plane_grid_views(xs_g, ys_g,
                                        what="planar A7 gate (cache hit)")
            _m2h = {}
            for _idx, _el in enumerate(g.electrodes, start=1):
                if not _el.shapes:
                    raise ValueError(
                        f"planar A7 gate (cache hit): electrode "
                        f"{_el.name!r} has no inline shapes, so the fold "
                        f"cannot be re-derived and the served bases "
                        f"cannot be checked. Refusing to fly unverified "
                        f"folded bases.")
                _m2h[_idx] = electrode_mask(_el, _Xh, _Yh)
            _fold_hit = planar_fold_axes(
                _sym_hit, _m2h, xs_g, ys_g,
                {"x": g.width_mm, "y": g.height_mm}, report=False)
            _assert_planar_fold_symmetry(bases, _fold_hit, h, spec,
                                         served_from="cache hit")
    else:
        # KERNEL FRAME (display and geometry must match
        # computation): the fly kernel samples node i at i*h (gx=px/mm,
        # node-centred convention) and the validated CSG rasterizer —
        # already rasterizes at i*h. Sampling at (i+0.5)h here put every
        # NATIVE electrode's effective metal at spec MINUS h/2 in both
        # axes (proven per-edge on the einzel fixture). Cache key is
        # salted ("raster": 2) so no old-frame mask can be served.
        xs, ys = xs_g, ys_g
        if verbose and any(anchor):
            print(f"anchored grid: node lattice shifted by "
                  f"(dx={anchor[0]:+g}, dy={anchor[1]:+g}) mm so the "
                  f"declared plane(s) sit exactly on nodes")
        X, Y = plane_grid_views(xs, ys, what="planar rasterizer")
        # solver3d is now correct at nz=1 (a size-1 z-axis reduces to the
        # exact 2-D Laplace; see test_solver3d_nz1.py), so a single z-plane
        # slab is right and fastest.
        masks = {}
        for idx, el in enumerate(g.electrodes, start=1):
            if not el.shapes:
                raise ValueError(
                    f"planar builder needs inline shapes; electrode "
                    f"{el.name!r} has none (STL planar path not wired)")
            m2 = electrode_mask(el, X, Y)
            # THIN-ELECTRODE REFUSAL on an anchored grid (the
            # surface-enhancement caveat): the shifted lattice can step
            # clean over an electrode thinner than one grid unit, and a
            # silently-empty mask flies a geometry with that electrode
            # MISSING. Scoped to anchored axes only — un-anchored
            # behaviour is deliberately unchanged (a global empty-mask
            # policy is a separate decision, not smuggled in here).
            if any(anchor) and not m2.any():
                raise ValueError(
                    f"electrode {el.name!r} rasterizes to ZERO nodes on "
                    f"the anchored grid (anchor dx={anchor[0]:+g}, "
                    f"dy={anchor[1]:+g}, pitch {h} mm) — it is thinner "
                    f"than one grid unit and no node lands inside it. "
                    f"Refine the pitch or thicken the electrode.")
            masks[idx] = m2[:, :, None]
        # DECLARE-AND-VERIFY symmetry: prove each declared mirror/
        # translational plane against the actual masks BEFORE using it to
        # reduce the solve. A verified mirror plane becomes a mirror
        # boundary in solver3d (exact + halves the work); an unverified
        # declaration REFUSES loudly rather than folding a false symmetry
        # into a clean-looking wrong field.
        sym = spec.geometry.symmetry.normalized()
        # DECLARE-AND-VERIFY, two tiers. The shape tier proves each
        # declared MIRROR against the CONTINUOUS shapes, exactly — the
        # mask-level check measures the raster, where closed-edge
        # float-noise puts a one-node skin on
        # symmetric rect pairs (4.9% on a mirror's boards at h=0.08),
        # so it refused true symmetries. A failed shape-tier check is a
        # FALSE declaration and refuses loudly, as before.
        ok, report = verify_symmetry_shapes(
            sym, g.electrodes, {"x": g.width_mm, "y": g.height_mm})
        if not ok:
            bad = [d for a, k, o, d, sc in report if not o]
            raise ValueError(
                "declared symmetry not satisfied by the geometry — "
                "refusing to fold a false plane: " + "; ".join(bad)
                + ". Set that axis to 'none' or fix the layout.")
        # Translational declarations have no finite-shape witness; they
        # keep the mask check with its original refuse-on-failure
        # semantics (unchanged behaviour).
        mask2d = {i: m[:, :, 0] for i, m in masks.items()}
        mok_all, mreport = verify_symmetry(sym, mask2d)
        tbad = [f"{a}:{k} (mismatch {f:.1%}, {sc})"
                for a, k, o, f, sc in mreport
                if not o and k == "translational"]
        if tbad:
            raise ValueError(
                "declared symmetry not satisfied by the geometry — "
                "refusing to fold a false plane: " + "; ".join(tbad)
                + ". Set that axis to 'none' or fix the layout.")
        if verbose and report:
            print("symmetry verified:", reduction_summary(sym))
        # FOLD PRECONDITION (Slice 1, field-invariant by construction): a
        # verified mirror is EXPLOITED only under exactly the pre-existing
        # conditions — the DISCRETE masks pass the legacy mask check on
        # that axis AND no plane_mm is declared for it (an explicitly
        # located plane is not yet foldable on the anchored grid).
        # Anything else solves the full domain — the always-correct path,
        # same field, more time — and says so (a skipped optimisation
        # is a reported decision, never a silent branch).
        # THE REAL FOLD (the solver's own mirror flag was shown to be
        # a no-op, so this is the FIRST fold this path has
        # ever had). Authority: planar_fold_axes — shared with sizing so
        # the estimate reports the solve that actually runs. Per axis it
        # requires: declared mirror (shape tier verified above), the
        # plane ON a node of the (anchored) lattice, the domain node-
        # symmetric about that node, and every electrode's discrete mask
        # exactly mirror-equal. Anything short of all four solves the
        # full domain and says why.
        fold = planar_fold_axes(sym, mask2d, xs, ys,
                                {"x": g.width_mm, "y": g.height_mm},
                                report=True)
                                            # the fold happens by cropping
        # Multigrid, not plain SOR: planar geometries have OPEN boundaries
        # that make SOR crawl (the einzel took ~16.5k sweeps / 6.6 s per
        # basis; multigrid needs ~114 fine sweeps / ~1 s and converges to
        # the same field via the validated fine smoother). Falls back to SOR
        # if multigrid is unavailable.
        # ImportError ONLY.  This used to be `except Exception`, which meant
        # that if multigrid THREW -- on one geometry, for one bug -- the field
        # was silently recomputed by a DIFFERENT SOLVER and nobody was told.
        # A certified number would then depend on whether an exception happened
        # to fire.  "Falls back if multigrid is UNAVAILABLE" is an ImportError;
        # anything else is a solver bug and must surface.
        # ENGINE: planar bases are
        # solved DIRECTLY — one sparse LU factorization per geometry
        # (laplace_factor, the ghost stencil), then one triangular
        # back-substitution per electrode basis. Replaces per-basis
        # multigrid (~10^2 sweeps to tol each): exact instead of
        # tol-approximate, and k bases cost one factorization. Verified
        # against the mg engine at the mg tolerance on the anchors.
        # dtype: float32 is the declared screen-map option (certified
        # paths stay float64).
        from ion_gym.physics.solver2d import laplace_factor
        _factor = {}
        # FOLD-PLANE BC (root fix of a 38 V y-fold
        # defect): a fold plane needs the EVEN-REFLECTION ghost
        # (phi_ghost = phi_inner, solver2d 'mirror'), NOT the open-edge
        # ghost_linear (phi_ghost = phi_edge) — the earlier claim that
        # the solver's edge ghost "is the mirror boundary" was false,
        # and on a plane with transverse field curvature ghost_linear
        # halves the second difference there (O(h^2 * phi'') plane
        # error; invisible on curvature-flat planes, which is how the
        # flat-plane fixture certified it). _fold_crop keeps
        # [plane_node:], so folded axis 0 is the solver's z0 edge and
        # folded axis 1 its u0 edge; outer edges keep ghost_linear so
        # the folded problem matches the full-domain treatment exactly.
        _fold_ghost = dict.fromkeys(("z0", "z1", "u0", "u1"),
                                    "ghost_linear")
        if 0 in fold:
            _fold_ghost["z0"] = "mirror"
        if 1 in fold:
            _fold_ghost["u0"] = "mirror"

        def _solve(mm, only=None):
            metal = np.zeros(next(iter(mm.values())).shape[:2], bool)
            for m3 in mm.values():
                metal |= m3[:, :, 0]
            fkey = (metal.shape, int(metal.sum()))
            if fkey not in _factor:
                _factor.clear()          # one geometry per model; a new
                                         # mask set means the fold state
                                         # changed — refactor, loudly
                if verbose:
                    print("planar: LU factorization "
                          f"({metal.shape[0]}x{metal.shape[1]}, "
                          f"dtype {np.dtype(solve_dtype).name})")
                _factor[fkey] = laplace_factor(
                    metal, h=1.0, symmetry="planar",
                    edge_ghost=_fold_ghost, dtype=solve_dtype)
            solve_rhs = _factor[fkey]
            out = {}
            for i in sorted(mm):
                if only is not None and i not in only:
                    continue
                val = np.where(mm[i][:, :, 0], float(V_BASIS), 0.0)
                out[i] = solve_rhs(val)[:, :, None]
            return out
        if fold:
            # THE FOLD: solve the cropped half (plane node at index 0,
            # where the universal even-reflection ghost is the exact
            # mirror BC), reflect bases back to full shape. Everything
            # downstream — cache banking, model, display — sees full
            # arrays; only the WORK halves per folded axis.
            _solve_inner = _solve

            def _solve(mm, only=None):
                cm = {i: _fold_crop(m, fold) for i, m in mm.items()}
                return {i: _fold_reflect(ph, fold)
                        for i, ph in _solve_inner(cm, only=only).items()}
            if verbose:
                print("symmetry: folding solve on " +
                      ", ".join(f"{'xy'[a]} (plane node {p})"
                                for a, p in sorted(fold.items()))
                      + " — work halved per axis; bases reflected back")
        # BANK AS YOU GO: the bankable UNIT is the BASIS. A model
        # whose full basis set outlives the sandbox call limit must resume,
        # so each completed basis is stored IMMEDIATELY under a partial key
        # (geometry key + which basis — can never collide with, or be
        # served as, the complete entry). Partials are re-read on the next
        # attempt and only the missing bases are solved; the complete entry
        # is stored once whole. Leftover partials are orphaned npz (a few
        # MB) — harmless, overwritten by identical geometry, never loaded
        # as complete entries.
        bases3 = {}
        _cache_ok = True
        try:
            from ion_gym.io import basis_cache
            from ion_gym.io.fa_cache import load as _pl, store as _ps
            from ion_gym.io.basis_cache import geometry_key_dict as _gkd
        except ImportError:
            _cache_ok = False               # no cache: solve everything
        if not _cache_ok or not _bank64:
            # ephemeral dtype (float32) or cache unavailable: straight
            # solve, no partial banking — a declared, reported path.
            bases3 = _solve(masks)
        else:
            for idx in sorted(masks):
                pkey = dict(_gkd(spec), __partial_basis__=int(idx))
                arrs, _meta = _pl(pkey)
                if arrs is not None:
                    bases3[idx] = np.ascontiguousarray(arrs["phi"], float)
                    print(f"  basis {idx}: partial disk cache hit")
            missing = [i for i in sorted(masks) if i not in bases3]
            for idx in missing:
                import time as _time
                _t0 = _time.time()
                bases3.update(_solve(masks, only={idx}))
                try:
                    _ps(dict(_gkd(spec), __partial_basis__=int(idx)),
                        {"phi": bases3[idx]})
                    print(f"  [{_time.strftime('%H:%M:%S')}] basis {idx}: "
                          f"banked (partial, solve "
                          f"{_time.time()-_t0:.2f}s)")
                except (OSError, ValueError) as e:
                    # Narrowed (was `except Exception`): the
                    # expected bank failures are disk (OSError) and
                    # serialization (ValueError) -- the same pair the main
                    # store handler below names. Anything else is a bug and
                    # must surface.
                    print(f"  basis {idx}: PARTIAL BANK FAILED ({e!r}); "
                          f"solve unaffected, will re-solve next process")
        bases = {i: b[:, :, 0] for i, b in bases3.items()}
        # A7 SYMMETRY GATE — body in _assert_planar_fold_symmetry above,
        # which is also called on the cache-hit path so both routes and
        # both paths carry the SAME guarantee.
        _assert_planar_fold_symmetry(bases, fold, h, spec,
                                     served_from="solved")
        # HIT MASK vs DIRICHLET MASK -- they are NOT the same thing.
        # Every electrode is a Dirichlet boundary for the SOLVE (that is what
        # `masks` is for, above).  Only a SOLID electrode is something an ion
        # can hit.  An electrode declared is_grid=True is a boundary CONDITION
        # the ion flies through: a mesh, a gridless mirror's entrance plane, a
        # detector face held at potential, or the drift-copper column that
        # pins a region's gauge to zero.  `is_grid` has been in ElectrodeSpec
        # all along and was silently ignored here, which made every such plane
        # a wall -- ions died on a boundary condition.  Root cause fixed:
        # is_grid electrodes are Dirichlet in the solve and TRANSPARENT to the
        # kernel.  (No existing spec sets is_grid, so no cached mask and no
        # certified fly changes -- gated in test_is_grid.py.)
        # `ele` must carry INT16 LABELS, not a boolean OR-fold.
        #
        # It was doing two jobs at once: the kernel's wall mask ("is this node
        # metal?") and the display's label array ("WHICH electrode is this?").
        # As a bool, `ele == 1` is true for EVERY conductor (True == 1) and
        # `ele == 3` is true for none -- so the PE view, the electrode legend
        # and every `ele == i` colour lookup drape all electrodes at electrode
        # #1's voltage. The solve is unaffected (bases come from `masks`), but
        # DISPLAY is wrong, which is precisely what "display must equal solver
        # input" exists to forbid.
        #
        # is_grid electrodes stay OUT (label 0), exactly as before: they are
        # Dirichlet in the solve and TRANSPARENT to the kernel, and relabelling
        # them would resurrect the is_grid-as-wall bug. The kernels test
        # nonzero-ness (`.astype(uint8)` + truthiness; `.astype(float64)` +
        # `> 0.5`), so int16 changes nothing for them.
        ele = np.zeros((nx, ny), np.int16)
        for idx, el in enumerate(g.electrodes, start=1):
            if getattr(el, "is_grid", False):
                continue
            ele[masks[idx][:, :, 0]] = idx
        _PLANAR_BASIS_CACHE[ckey] = _freeze(bases, ele)
        # BANK AS YOU GO: run tools BANK their work as it
        # completes. basis_cache.store had long existed with ZERO
        # callers — the disk cache was read-only from the solve path, so a
        # killed process lost every solve it had finished. Same
        # optional-import policy as the load side above.
        try:
            from ion_gym.io import basis_cache
        except ImportError:
            pass                        # optional; L1 still holds the bases
        else:
            try:
                if _bank64:
                    basis_cache.store(spec, bases, ele)
                import time as _time
                if _bank64:
                    print(f"[{_time.strftime('%H:%M:%S')}] planar "
                          f"bases: stored to disk cache "
                          f"({basis_cache.key(spec)})")
            except (OSError, ValueError) as e:
                # LOUD, not fatal. The cache is an accelerator, not an
                # authority — but a silent write failure is a swallow.
                # Narrowed (was `except Exception`): disk +
                # serialization failures are the expected kinds; anything
                # else surfaces.
                # DEDUPED: a near-identical older store block sat
                # directly below this one — the bases were stored TWICE per
                # solve and the two copies had drifted (two owners of one
                # job). This block is the ONE store.
                print(f"planar bases: DISK CACHE STORE FAILED ({e!r}); "
                      f"solve unaffected but NOT banked — this run's work "
                      f"will be repeated next process")

    # assemble A (DC + drive offsets) + per-DRIVE-GROUP bases through
    # the ONE shared 2-D assembly (multi-membership, offset_v, 0 Hz all
    # honoured there — see assemble_drive_groups). A voltage or drive
    # change re-weights cached bases — never re-solves.
    A, drives = assemble_drive_groups(spec, bases)
    # back-compat Bk: the sin drives as (B, freq_hz, phase_deg) tuples
    Bk = [(B, gg.frequency_hz, gg.phase_deg) for B, gg in drives
          if gg.waveform == "sin"]

    Ex, Ey = _grad2d(np.ascontiguousarray(A), h)
    bands = _electrode_bands(bases, len(g.electrodes), h, anchor=anchor)
    return PlanarModel(A, Bk, ele, Ex, Ey, h, spec, el_bands=bands,
                       drives=drives, bases=bases, anchor_mm=anchor)


def _electrode_bands(bases, n_el, h, anchor=(0.0, 0.0)):
    """Per-electrode transverse extents (x_lo,x_hi,y_lo,y_hi) in mm from
    the solved bases (electrode i sits where basis_i ~ 1e4 V). None for an
    empty electrode. Used to draw rod bands in the transport (xz/yz)
    views. Node i sits at i*h + anchor — bands are user-frame mm."""
    ax, ay = anchor
    bands = []
    for idx in range(1, n_el + 1):
        b = bases.get(idx)
        if b is None:
            bands.append(None)
            continue
        m = b > 0.99e4
        if not m.any():
            bands.append(None)
            continue
        xi, yi = np.where(m)
        bands.append((xi.min() * h + ax, xi.max() * h + ax,
                      yi.min() * h + ay, yi.max() * h + ay))
    return bands


# --------------------------------------------------------------- tracer
@njit(cache=True, nogil=True)
def _bilin(F, gx, gy, nx, ny):
    i = int(gx)
    j = int(gy)
    if i < 0: i = 0
    if j < 0: j = 0
    if i > nx - 2: i = nx - 2
    if j > ny - 2: j = ny - 2
    # Fraction CLAMPED (index clamp alone = silent linear
    # extrapolation of the edge cell for any out-of-grid sample).
    fx = gx - i
    fy = gy - j
    if fx < 0.0: fx = 0.0
    if fx > 1.0: fx = 1.0
    if fy < 0.0: fy = 0.0
    if fy > 1.0: fy = 1.0
    return (F[i, j] * (1 - fx) * (1 - fy) + F[i + 1, j] * fx * (1 - fy)
            + F[i, j + 1] * (1 - fx) * fy + F[i + 1, j + 1] * fx * fy)


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _wave_eval(kind, om, ph, duty, tab_t, tab_v, o0, o1, t):
    """Unit waveform w(t) of one drive channel at LAB time t (us).
    kinds: 0 sin, 1 cos, 2 square(duty), 3 table-hold, 4 table-linear.
    `duty` (squares only): fraction of the period HIGH from phase 0.
    duty == 0.5 evaluates as sign(sin) EXACTLY (the frozen historical
    formula, same anchoring as tracer3d._wave_eval); any other duty is
    the phase-fraction test. Tables clamp to end values; interior lookup
    is binary search on the channel's [o0, o1) slice of the shared
    breakpoint arrays."""
    if kind == 0:
        return math.sin(om * t + ph)
    if kind == 1:
        return math.cos(om * t + ph)
    if kind == 2:
        if duty == 0.5:
            return 1.0 if math.sin(om * t + ph) >= 0.0 else -1.0
        frac = ((om * t + ph) / (2.0 * math.pi)) % 1.0
        return 1.0 if frac < duty else -1.0
    n = o1 - o0
    if n == 0:
        return 0.0
    if t <= tab_t[o0]:
        return tab_v[o0]
    if t >= tab_t[o1 - 1]:
        return tab_v[o1 - 1]
    lo = o0
    hi = o1 - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tab_t[mid] <= t:
            lo = mid
        else:
            hi = mid
    if kind == 3:
        return tab_v[lo]
    f = (t - tab_t[lo]) / (tab_t[hi] - tab_t[lo])
    return tab_v[lo] + f * (tab_v[hi] - tab_v[lo])


@njit(cache=True, nogil=True)
def _fly_planar(x, y, vx, vy, tob, m_ion, ExA, EyA, ExK, EyK,
                ch_kind, ch_om, ch_ph, ch_duty, tab_t, tab_v, tab_off,
                ele, mm, acc, dt, t_max_us, collide_on, T_k, P_pa,
                sigma, c_star, c_bar, sig1d, m_gas, rec, rec_every,
                nch_flags, bnd_on, bnd_val, seed, z0, vz,
                sds_on, sds_damping, sds_mfp, sds_V, sds_logmr, sds_stats,
                pl_col, pl_val, pl_sgn, pl_w, pl_kind):
    """Planar Verlet + optional HS + N DRIVE CHANNELS, with a FIELD-FREE
    axial drift along z. The transverse field is
        E(x,y,t) = E_A + sum_k w_k(t) E_k
    with w_k evaluated by _wave_eval on the LAB clock (tob + t) — sin/cos
    quadrature pairs (one per frequency), square waves, and breakpoint
    tables all through the same channel list; this is the
    scripting equivalent for state-independent drive programs. z carries
    no force, so z = z0 + vz*t exactly (ideal-guide limit). Set vz=0 for
    a pure 2-D slice. Records [t,x,y,z,vx,vy,vz] + channels. Returns
    (nrec, kind, ncol)."""
    np.random.seed(seed)
    nx, ny = ExA.shape
    t = 0.0
    ncol = 0
    z = z0
    cs, cke, cke_x, cke_y, cke_z, cef, cea, cer, cex, cey, cez, crad, cncol \
        = nch_flags
    # axis for the 'radius' channel = domain centre (transverse axis)
    _AXX = 0.5 * (nx - 1) * mm
    _AXY = 0.5 * (ny - 1) * mm

    K = ch_kind.shape[0]

    def _efield(px, py, tt):
        gx = px / mm
        gy = py / mm
        ex = _bilin(ExA, gx, gy, nx, ny)
        ey = _bilin(EyA, gx, gy, nx, ny)
        for k in range(K):
            w = _wave_eval(ch_kind[k], ch_om[k], ch_ph[k], ch_duty[k],
                           tab_t, tab_v, tab_off[k], tab_off[k + 1], tt)
            ex = ex + w * _bilin(ExK[k], gx, gy, nx, ny)
            ey = ey + w * _bilin(EyK[k], gx, gy, nx, ny)
        return ex, ey

    # row writer inline
    def _row(k, tt, xx, yy, zz, vxx, vyy, exv, eyv, nc):
        rec[k, 0] = tt
        rec[k, 1] = xx
        rec[k, 2] = yy
        rec[k, 3] = zz
        rec[k, 4] = vxx
        rec[k, 5] = vyy
        rec[k, 6] = vz
        c = 7
        sp = math.sqrt(vxx * vxx + vyy * vyy + vz * vz)
        if cs:
            rec[k, c] = sp
            c += 1
        if cke:
            rec[k, c] = 0.5 * m_ion * 1.6605402e-27 * (sp * 1e3) ** 2 \
                / 1.602176634e-19; c += 1
        # per-axis KE (eV). Same 1/2 m v_i^2 / e as the total, projected
        # onto each component: vxx,vyy are the current velocities, vz is the
        # (ballistic, field-free) z-velocity in this 2-D solve. By
        # construction ke_x+ke_y+ke_z == ke_ev, which the channel test
        # asserts.
        if cke_x:
            rec[k, c] = 0.5 * m_ion * 1.6605402e-27 * (vxx * 1e3) ** 2 \
                / 1.602176634e-19; c += 1
        if cke_y:
            rec[k, c] = 0.5 * m_ion * 1.6605402e-27 * (vyy * 1e3) ** 2 \
                / 1.602176634e-19; c += 1
        if cke_z:
            rec[k, c] = 0.5 * m_ion * 1.6605402e-27 * (vz * 1e3) ** 2 \
                / 1.602176634e-19; c += 1
        if cef:
            rec[k, c] = math.sqrt(exv * exv + eyv * eyv)
            c += 1
        if cea:
            rec[k, c] = exv
            c += 1
        if cer:
            rec[k, c] = eyv
            c += 1
        if cex:
            rec[k, c] = exv
            c += 1
        if cey:
            rec[k, c] = eyv
            c += 1
        if cez:
            rec[k, c] = 0.0
            c += 1
        if crad:
            rec[k, c] = math.sqrt((xx - _AXX) ** 2 + (yy - _AXY) ** 2)
            c += 1
        if cncol:
            rec[k, c] = nc
            c += 1

    ex, ey = _efield(x, y, tob)
    # record V/mm per the channel contract; _efield returns the V/m
    # basis (see _grad2d's 1e3), hence the 1e-3 at every record site.
    _row(0, tob, x, y, z, vx, vy, ex * 1e-3, ey * 1e-3, ncol)
    nrec = 1
    kind = 2
    face = -1                     # terminating bound face, -1 = none
    while t < t_max_us:
        # old state for the impact backtrack (bisection to the boundary)
        xo = x
        yo = y
        zo = z                    # station planes interpolate on z too
        vxo = vx
        vyo = vy
        to = t
        ex, ey = _efield(x, y, tob + t)
        ex *= 1e-9 * acc
        ey *= 1e-9 * acc
        if sds_on:
            tterm = sds_damping * dt
            factor = (1.0 - math.exp(-tterm)) / tterm
            ex = factor * (ex - vx * sds_damping)
            ey = factor * (ey - vy * sds_damping)
        vx += 0.5 * ex * dt
        vy += 0.5 * ey * dt
        x += vx * dt
        y += vy * dt
        ex, ey = _efield(x, y, tob + t + dt)
        ex *= 1e-9 * acc
        ey *= 1e-9 * acc
        if sds_on:
            tterm = sds_damping * dt
            factor = (1.0 - math.exp(-tterm)) / tterm
            ex = factor * (ex - vx * sds_damping)
            ey = factor * (ey - vy * sds_damping)
        vx += 0.5 * ex * dt
        vy += 0.5 * ey * dt
        z += vz * dt                       # field-free axial drift
        t += dt
        if sds_on:
            # SDS random-walk diffusion (in-plane component of the isotropic
            # 3-D jump: magnitude scaled by sqrt(2/3) so the 2-D diffusion
            # coefficient stays correct).
            dist_steps = _diff_dist_steps(sds_stats, sds_logmr)
            ncoll = sds_V / sds_mfp * dt
            r = math.sqrt(ncoll / _SDS_NDC) * dist_steps * sds_mfp
            th = 2.0 * math.pi * np.random.random()
            rr = r * 0.816496580927726
            x += rr * math.cos(th)
            y += rr * math.sin(th)
            ncol += 1
        elif collide_on:
            sp = math.sqrt(vx * vx + vy * vy)
            if sp < 1e-7:
                sp = 1e-7
            lam = _mfp_mm(sp, T_k, P_pa, sigma, c_star, c_bar)
            if np.random.random() < 1.0 - math.exp(-sp * dt / lam):
                vx, vy, _z = _collide(vx, vy, 0.0, 0.0, 0.0, 0.0,
                                      m_ion, m_gas, sig1d, sp)
                ncol += 1
        if _metal_nn(ele, x / mm, y / mm, nx, ny):
            # bisect back to the impact surface (same 40-step refinement
            # as tracer3d) so the recorded end state sits ON the shell,
            # not up to one full step inside it.
            f0 = 0.0
            f1 = 1.0
            for _bs in range(40):
                fm = 0.5 * (f0 + f1)
                if _metal_nn(ele, (xo + fm * (x - xo)) / mm,
                             (yo + fm * (y - yo)) / mm, nx, ny):
                    f1 = fm
                else:
                    f0 = fm
            x = xo + f1 * (x - xo)
            y = yo + f1 * (y - yo)
            vx = vxo + f1 * (vx - vxo)
            vy = vyo + f1 * (vy - vyo)
            t = to + f1 * dt
            kind = 0
            break
        if x < 0 or y < 0 or x / mm > nx - 1 or y / mm > ny - 1:
            kind = 1
            break
        # STATION PLANES (fate 5 impact_plane / 6 detect). Same crossing
        # math, window sense and step ordering as tracer3d's block —
        # metal impact first, then box exit, then stations, then the
        # declared bounds — so the two routes cannot diverge on the
        # contract. The plane list itself comes from the ONE shared
        # builder (physics.stations.station_planes), already converted
        # into this kernel's anchored frame by the caller. sgn is 0 for
        # a station: it terminates on a crossing in either direction.
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
                    x = xc
                    y = yc
                    z = zc
                    vx = vxo + f * (vx - vxo)
                    vy = vyo + f * (vy - vyo)
                    t = to + f * dt
                    kind = pl_kind[ip]
                    hit_pl = True
                    break
            if hit_pl:
                break
        # optional bounding/impact planes (fate 3). bnd_on: 6 flags
        # [x_min,x_max,y_min,y_max,z_min,z_max]; bnd_val: 6 values. z now
        # carries the real axial drift, so z bounds are meaningful (e.g.
        # the transport-axis exit plane at the end of a quad).
        # The check is decomposed so the kernel can REPORT WHICH
        # face terminated the flight. A bare kind = 3 cannot distinguish
        # "reached the declared seam" from "left through a declared
        # bound", and fly_staged hand-off correctness depends on exactly
        # that distinction. face is the first armed face found crossed,
        # in the same test order the compound form evaluated.
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
        step_rec = int(t / dt) % rec_every == 0
        if step_rec and nrec < rec.shape[0] - 1:
            _row(nrec, tob + t, x, y, z, vx, vy, ex / (1e-6 * acc),
                 ey / (1e-6 * acc), ncol)
            nrec += 1
    _row(nrec, tob + t, x, y, z, vx, vy, ex / (1e-6 * acc + 1e-30),
         ey / (1e-6 * acc + 1e-30), ncol)
    nrec += 1
    return nrec, kind, ncol, face


def make_planar_fly_fn(model: PlanarModel, births, spec: SimSpec):
    # PLANAR-SUPPORTED channels, in the KERNEL'S unpack order (the 13
    # flags _fly_planar destructures). REGRESSION GUARD: this list
    # once was `list(OPTIONAL_CHANNELS.keys())`, so every channel added to the
    # global registry for OTHER routes (path_mm / e_axial_tint / ke_tint
    # from the r-z accumulators, wrap_passes from the 3-D transporter)
    # silently grew the flags tuple 13 -> 17 and broke EVERY planar
    # flight with a numba unpack error. The route owns its list; anything
    # requested beyond it refuses BY NAME (same contract the r-z route
    # got this session).
    order = ["speed", "ke_ev", "ke_x", "ke_y", "ke_z", "e_field",
             "e_axial", "e_radial", "e_x", "e_y", "e_z", "radius",
             "n_col"]
    _unsupported = [c for c in spec.integration.record_channels
                    if c not in order]
    if _unsupported:
        raise ValueError(
            f"planar route does not support record channel(s) "
            f"{_unsupported}; supported: {order}. (path_mm/e_axial_tint/"
            f"ke_tint are r-z kernel accumulators; wrap_passes is the 3-D "
            f"transporter's.)")
    if getattr(spec, "transporter", None) is not None:
        raise ValueError("the planar route does not implement the periodic "
                         "transporter (CHARTER_transporter); remove the "
                         "'transporter' block or fly a 3-D shapes route")

    chans = spec.integration.record_channels
    flags = np.array([c in chans for c in order], np.bool_)
    col_names = BASE_CHANNELS + [c for c in order if c in chans]
    ncol_total = len(col_names)
    # DESYNC GUARD: col_names is built here from OPTIONAL_CHANNELS order and
    # the _row writer walks the SAME order; both must agree with the spec's
    # promise. A mismatch means a channel was added to one place and not the
    # other (the class of bug that shipped field components under speed's
    # name). Fail loud at build time rather than write mislabelled columns.
    assert col_names == spec.column_names(), (
        f"planar column desync: {col_names} != {spec.column_names()}")
    # (m_ion was read here only to feed a DEAD expression -- the value was
    # computed and discarded; the live acceleration is per-ion acc_i below.)
    # CHARGE STATE: the planar route once hard-coded q = +1e
    # in its acceleration scale, so a spec declaring source.charge = 2 flew
    # a singly charged ion at the declared mass. Same fix and same refusal
    # as the r-z route (build_rz): q/m = charge*e/(mass*amu).
    q_e = int(spec.source.charge)
    if q_e == 0:
        raise ValueError("source.charge = 0: an uncharged ion has no "
                         "electric acceleration; state the charge state.")
    col = spec.collisions
    mg = gas_mass(col.gas) if col.enabled else 4.0
    c_star = math.sqrt(2 * KB * col.T_k / (mg * KG_AMU)) / 1000.0 \
        if col.enabled else 1.0
    c_bar = math.sqrt(8 * KB * col.T_k / (math.pi * mg * KG_AMU)) / 1000.0 \
        if col.enabled else 1.0
    sig1d = math.sqrt(KB * col.T_k / (mg * KG_AMU)) / 1000.0 \
        if col.enabled else 1.0
    # SDS (diffusion) collision model: load statistics + per-mass parameters.
    sds_on = bool(col.enabled and getattr(col, "model", "hs") == "sds")
    if sds_on:
        import os
        _sdir = os.path.dirname(os.path.abspath(__file__))
        _sds_stats = _sds_load_stats(os.path.join(_sdir, _sds_mod.JUMP_ICDF_FILE))
        _sds_massdata = _sds_load_mass(os.path.join(_sdir, _sds_mod.MOBILITY_FILE))
        _sds_mgas = 28.94515 if col.gas in ("air", "N2") else gas_mass(col.gas)
    else:
        _sds_stats = np.zeros((5, 1002))
        _sds_massdata = []
        _sds_mgas = 28.94515
    dt = spec.integration.dt_ns * 1e-3
    # Record buffer must hold every recorded step, or the trajectory gets
    # truncated mid-flight and the viewer draws a straight line from the
    # last recorded point to the final impact. Records = steps/rec_every,
    # plus the birth and final rows; cap to bound memory on very long runs.
    rec_every = max(1, spec.integration.rec_every)
    n_steps = int(spec.integration.t_max_us / dt) + 2
    # keep total records under a target so the plot stays smooth AND the
    # buffer always fits (no mid-flight truncation -> no straight-line
    # artifact). For very long runs this thins the recording automatically.
    _TARGET_RECORDS = 40_000
    if n_steps // rec_every > _TARGET_RECORDS:
        rec_every = -(-n_steps // _TARGET_RECORDS)   # ceil division
    max_records = n_steps // rec_every + 8
    nch = tuple(bool(f) for f in flags)
    bflags, bvals = spec.bounds.as_tuple()
    bnd_on = np.array(bflags, np.bool_)
    bnd_val = np.array(bvals, np.float64)
    # ANCHORED FRAME BOUNDARY: the kernel's lattice puts node i at
    # i*h; the model's node i sits at i*h + anchor. Everything the USER
    # states (births, positional bounds) maps INTO the kernel frame here,
    # and everything the kernel records maps back OUT below — records and
    # displayed axes stay in the user's spec frame exactly (display ==
    # computation). anchor == (0,0) for every undeclaring spec, making
    # every mapping the identity.
    _anx, _any = model.anchor_mm
    births_k = births
    if _anx or _any:
        births_k = births.copy()
        births_k[:, 0] -= _anx
        births_k[:, 1] -= _any
        bnd_val = bnd_val.copy()
        bnd_val[0] -= _anx
        bnd_val[1] -= _anx      # x_min, x_max
        bnd_val[2] -= _any
        bnd_val[3] -= _any      # y_min, y_max

    # STATION PLANES for the kernel, from the ONE shared builder. The
    # kernel flies the ANCHORED frame (node i at i*h), so stations —
    # authored in the user frame like bounds and births above — are
    # converted with the same anchor offset; z carries no anchor (the
    # planar route's z is the field-free drift coordinate, recorded
    # as-is), hence the 0.0. This also serves the stl2d route, which
    # flies through this same fly_fn.
    from ion_gym.physics.stations import station_planes as _st_planes
    _AXCOL = {"x": 0, "y": 1, "z": 2}
    _pl = _st_planes(spec, (_anx, _any, 0.0))
    pl_col = np.array([_AXCOL[p[0]] for p in _pl], np.int64)
    pl_val = np.array([p[1] for p in _pl], np.float64)
    pl_sgn = np.array([p[2] for p in _pl], np.float64)
    pl_w = (np.array([list(p[3]) for p in _pl], np.float64)
            if _pl else np.empty((0, 4), np.float64))
    pl_kind = np.array([p[4] for p in _pl], np.int64)

    # DRIFT EXTENSION: residual |E| on each boundary face, ONCE
    # per model. Static A exactly; each drive channel at its worst-case
    # gain -- |w| <= 1 for sin/cos/square waveforms, max|table| for
    # table channels (a documented conservative bound, never a guess).
    from ion_gym.physics.drift_extension import (edge_field_max,
                                                 extend_ballistic
                                                 as _extend)
    _tabmax = float(np.max(np.abs(model.tab_v))) if model.tab_v.size \
        else 1.0
    _chw = [(_tabmax if int(k) in (3, 4) else 1.0)
            for k in model.ch_kind]
    _ex_c = [(model.ExA, 1.0)] + [(model.ExK[i2], _chw[i2])
                                  for i2 in range(model.ExK.shape[0])]
    _ey_c = [(model.EyA, 1.0)] + [(model.EyK[i2], _chw[i2])
                                  for i2 in range(model.EyK.shape[0])]
    # planar basis arrays are V/m (_grad2d); extend_ballistic's contract
    # is V/mm — convert here (see the
    # matching note in build_rz).
    _edge_e = {"x_lo": 1e-3 * edge_field_max([_ex_c, _ey_c], 0, "lo"),
               "x_hi": 1e-3 * edge_field_max([_ex_c, _ey_c], 0, "hi"),
               "y_lo": 1e-3 * edge_field_max([_ex_c, _ey_c], 1, "lo"),
               "y_hi": 1e-3 * edge_field_max([_ex_c, _ey_c], 1, "hi")}
    _AXX = 0.5 * (model.ele.shape[0] - 1) * model.mm_per_gu
    _AXY = 0.5 * (model.ele.shape[1] - 1) * model.mm_per_gu

    def fly_fn(i):
        b = births_k[i]
        refuse_birth_in_metal(
            model.ele, b[0] / model.mm_per_gu, b[1] / model.mm_per_gu,
            model.mm_per_gu, spec, i,
            f"({births[i, 0]:.3f}, {births[i, 1]:.3f}) mm")
        # per-ion mass through mz_of, THE single authority:
        # generate_births assigns m/z in CONTIGUOUS BLOCKS
        # (i // n_ions); an older i % len CYCLING here flew
        # most ions of a multi-m/z run with a mass that did not match
        # their birth kinematics. The 3-D route already used mz_of.
        from ion_gym.physics.ion_envelope import per_ion
        env = per_ion(spec, i)          # Tier 3: one mass+seed derivation
        mz_i = env.mz
        m_i = mz_i
        acc_i = q_e * E_CHG / (m_i * KG_AMU)          # charge*e/(mass*amu)
        if sds_on:
            _P = _sds_ion_params(mz_i, float(q_e), _sds_mgas, col.gas_diam_nm,
                                 col.T_k, col.P_torr, _sds_massdata)
            sds_damping = _P["damping"]
            sds_mfp = _P["mfp_mm"]
            sds_V = _P["V_mm_us"]
            sds_logmr = _P["log_mr_ratio"]
        else:
            sds_damping = sds_mfp = sds_V = sds_logmr = 0.0
        rec = np.empty((max_records, ncol_total))
        n, kind, ncol, _bface = _fly_planar(
            b[0], b[1], b[3], b[4], b[6], m_i,
            model.ExA, model.EyA, model.ExK, model.EyK,
            model.ch_kind, model.ch_om, model.ch_ph, model.ch_duty,
            model.tab_t, model.tab_v, model.tab_off,
            model.ele.astype(np.float64),
            model.mm_per_gu, acc_i, dt, spec.integration.t_max_us,
            col.enabled, col.T_k, col.P_pa, col.sigma_m2, c_star, c_bar,
            sig1d, mg, rec, rec_every, nch,
            bnd_on, bnd_val, env.seed, b[2], b[5],
            sds_on, sds_damping, sds_mfp, sds_V, sds_logmr, _sds_stats,
            pl_col, pl_val, pl_sgn, pl_w, pl_kind)
        traj = rec[:n].copy()
        from ion_gym.io.records import TrajRecord
        rec_view = TrajRecord(traj, col_names)
        if _anx or _any:
            rec_view["x"] += _anx                    # x back to user frame
            rec_view["y"] += _any                    # y back to user frame
        # Honor declared downstream bounds/stations by exact
        # ballistic algebra when the flight left the solved domain
        # (kind 1). Runs in the USER frame (bounds/stations are stated
        # there); face classification mirrors the kernel's termination
        # test in the kernel frame. The planar radius convention
        # (kernel-frame grid centre) is supplied HERE.
        if kind == 1:
            _xk = float(traj[-1, 1]) - _anx
            _yk = float(traj[-1, 2]) - _any
            if _xk < 0.0:
                _fk, _fn = "x_lo", "x low"
            elif _xk / model.mm_per_gu > model.ele.shape[0] - 1:
                _fk, _fn = "x_hi", "x high"
            elif _yk < 0.0:
                _fk, _fn = "y_lo", "y low"
            else:
                _fk, _fn = "y_hi", "y high"
            # face map, planar convention: bounds axes are the
            # trajectory's own x (0) and y (1).
            _fa, _fs = {"x_lo": (0, "lo"), "x_hi": (0, "hi"),
                        "y_lo": (1, "lo"), "y_hi": (1, "hi")}[_fk]
            traj, kind, _ext = _extend(
                traj, col_names, kind, spec,
                edge_e_vpermm=_edge_e[_fk], face_axis=_fa, face_side=_fs,
                edge_face=_fn,
                derived={"radius": lambda prev, row, idx, dt: math.sqrt(
                    (row[idx["x"]] - _anx - _AXX) ** 2
                    + (row[idx["y"]] - _any - _AXY) ** 2)},
                label=f"ion (seed {env.seed})")
            rec_view = TrajRecord(traj, col_names)
        end = rec_view.row(-1)
        from ion_gym.physics.ion_envelope import make_summary
        summary = make_summary(
            kind=kind, tof=float(end["t"]), mz=env.mz,
            x_end=float(end["x"]), y_end=float(end["y"]),
            z_end=float(end["z"]),
            r_end=float(math.hypot(
                end["x"] - births[:, 0].mean(),
                end["y"] - births[:, 1].mean())), n_col=ncol)
        return traj, summary
    return fly_fn, col_names


def unsupported_drive_features(spec):
    """Declared drive features the PLANAR kernel does NOT apply: NONE,
    as of 2026-09-16 (L-455). assemble_drive_groups honours multi-group
    membership and folds offset_v into the static field; the kernel
    evaluates sin, cos, square (with duty) and table (hold and linear)
    waveforms per channel, and 0 Hz drives are constants, not skips.
    Kept as the route's contract statement for the field export door."""
    return []




def build_planar_run(spec: SimSpec, verbose=False, solve_dtype=None):
    """SimSpec (planar) -> (model, fly_fn, col_names, births). Fully
    independent — no external solver. solve_dtype: None -> float64; float32 is
    the declared screen-map option (bank rows record it)."""
    model = build_planar_model(
        spec, verbose=verbose,
        solve_dtype=np.float64 if solve_dtype is None else solve_dtype)
    births = generate_births(spec)
    fly_fn, cols = make_planar_fly_fn(model, births, spec)
    # STAGED-ASSEMBLY PACK. A staged assembly needs to fly an
    # ARBITRARY state through an already-solved region, which `fly_fn`
    # cannot do — it closes over this spec's own births by index.
    #
    # The pack is ROUTE-TAGGED rather than converted. A planar solve is
    # not a degenerate 3-D solve and must not be extruded into one: the
    # geometry is z-invariant by construction, so a 3-D pack spanning the
    # drift would replicate an identical plane thousands of times to
    # express a symmetry the planar kernel already has exactly. The
    # consumer dispatches on `route` instead.
    model.fly_fields = dict(
        route="planar",
        ExA=model.ExA, EyA=model.EyA, ExK=model.ExK, EyK=model.EyK,
        ch_kind=model.ch_kind, ch_om=model.ch_om, ch_ph=model.ch_ph,
        ch_duty=model.ch_duty,
        tab_t=model.tab_t, tab_v=model.tab_v, tab_off=model.tab_off,
        ele=model.ele, h_mm=model.mm_per_gu,
        anchor_mm=tuple(float(v) for v in model.anchor_mm),
    )
    return model, fly_fn, cols, births
