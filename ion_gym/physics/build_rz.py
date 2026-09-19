"""
ion_gym.build_rz
----------------
The cylindrical (r-z) native builder: a SimSpec with coords='rz' ->
rasterized electrode masks in the z-r half-plane -> bases solved by the
validated direct solver (solver2d.solve_laplace, exact sparse solve with
the correct (1/r) phi_r term and on-axis handling) -> field model +
fly_fn. No external solver anywhere in the run path.

Covers round-bore einzel, IMS drift tubes, and the axial-transport field
of rod systems — everything whose field is genuinely 2-D in (z, r). The
same voltage re-weighting and channel-recording as the planar/funnel
paths; the funnel's validated r-z tracer body is reused.

Geometry convention: in an rz spec, the geometry x-axis is the cylinder
AXIS (z), and the y-axis is RADIUS (r >= 0). Shapes are given in that
(z, r) plane in mm; an electrode that is a ring/tube is a rect in (z, r).
The on-axis mirror (r=0) is handled by the solver, not declared.
"""

import math

import numpy as np

from ion_gym.io.sim_spec import SimSpec, OPTIONAL_CHANNELS, BASE_CHANNELS
from ion_gym.physics.sim_build import generate_births
from ion_gym.physics.solver2d import solve_laplace
from ion_gym.physics.ionbench import build_field_aware
from ion_gym.physics.build_planar import (electrode_mask,        # shape rasterizer (shared)
                                          K_SIN, K_SQUARE)       # channel kinds (shared)
from ion_gym.physics.symmetry import (verify_symmetry, verify_symmetry_shapes, reduction_summary)
from ion_gym.physics.collision3d import (E_CHG, KG_AMU, KB, gas_mass)
from ion_gym.physics.tracer_rz import _fly_rec_full   # THE r-z tracer


_RZ_BASIS_CACHE = {}


def clear_memory_cache():
    """Drop the in-process basis cache (_RZ_BASIS_CACHE). The disk
    cache is separate (fa_cache.clear_all). Returns the number
    of entries dropped — a visible, reported clear, not a
    silent wipe."""
    n = len(_RZ_BASIS_CACHE)
    _RZ_BASIS_CACHE.clear()
    return n


def rz_is_cached(spec):
    """True if the r-z bases for this geometry are already solved and
    cached — in process OR on disk (a rebuild only re-weights — fast).
    Used by the UI to decide whether to show a solving spinner.  The disk
    probe is the same cheap meta.json existence check build_planar uses:
    no arrays are touched."""
    import os
    from ion_gym.io import fa_cache
    ckey = _geometry_key(spec)
    if ckey in _RZ_BASIS_CACHE:
        return True
    return os.path.exists(os.path.join(
        os.path.expanduser(fa_cache.DEFAULT_ROOT), ckey, "meta.json"))


def _geometry_key(spec):
    """ONE SOURCE OF TRUTH: delegate to
    basis_cache.key — the same canonical, voltage-free key the disk cache
    and field_io use — so the in-memory and on-disk layers can never
    disagree about identity.  The retired local key already included
    is_grid; the canonical key is a superset: full
    electrode dicts minus voltage/name fields (is_grid included), grid
    extents, mm_per_gu, symmetry (whose `coords` distinguishes rz from
    planar), builder, and CACHE_FORMAT so a stored-format change
    invalidates what it invalidates."""
    from ion_gym.io import basis_cache
    return basis_cache.key(spec)


class RZModel:

    """Cylindrical field model on the z-r half-plane; defines the r-z
    model interface the viewer and driver share. Fields carry the r-z
    tracer's expected keys."""
    # Which principal planes this model HAS.  Declared, never sniffed.
    # An r-z model's plane is r-z.  It is NOT xy -- saying so was the bug.
    PLANES = ('rz',)
    # What potential_image() returns with NO argument: the DC potential.
    # The drive peak is rf_phase=pi/2 (each channel's own waveform at that
    # phase). Declared, never sniffed -- viz_core.peak_potential_image.
    POTENTIAL_IMAGE_DEFAULT = "dc"

    def __init__(self, A, ele, EzA, EuA, EzK, EuK, ch_kind, ch_om,
                 ch_ph, ch_duty, drives, u0, mm,
                 T_k, P_pa, sigma_m2, m_gas, spec, *,
                 EzG=None, EuG=None, tau_gate=-1.0):
        self.A = A
        from ion_gym.physics.raster2d import (el_masks_from_labels)
        # Named by the deck's own electrode names.
        self.el_masks = el_masks_from_labels(
            ele, getattr(spec.geometry, 'electrodes', None))
        self.ele = ele
        self.EzA, self.EuA = EzA, EuA
        # DRIVE CHANNELS (L-455; supersedes the single rf_V*sin*B fold):
        # EzK/EuK are (K, nz, 2nr-1) per-GROUP aware fields with the
        # group amplitude baked in; the kernel weights each with the
        # UNIT waveform (build_planar._wave_eval). `drives` keeps the
        # (folded potential, RFGroupSpec) pairs for display and export.
        self.EzK, self.EuK = EzK, EuK
        self.ch_kind = np.asarray(ch_kind, np.int64)
        self.ch_om = np.asarray(ch_om, np.float64)
        self.ch_ph = np.asarray(ch_ph, np.float64)
        self.ch_duty = np.asarray(ch_duty, np.float64)
        self.drives = drives
        self.chan_phi = [B0 for B0, _g in drives]
        self.u0 = u0
        self.mm_per_gu = mm
        # gate channel: step(t - tau_gate) * G; zeros/-1 = no gate
        self.EzG = EzG if EzG is not None else np.zeros_like(EzA)
        self.EuG = EuG if EuG is not None else np.zeros_like(EuA)
        self.tau_gate = float(tau_gate)
        self.T_k, self.P_pa = T_k, P_pa
        self.sigma_m2, self.m_gas = sigma_m2, m_gas
        self.spec = spec

    @property
    def Bk(self):
        """Sin drives as (B_phi, freq_hz, phase_deg) — the planar-model
        surface pe_view keys RF presence on."""
        return [(B0, g.frequency_hz, g.phase_deg) for B0, g in self.drives
                if g.waveform == "sin"]

    def _w_at_phase(self, k, phase_rad):
        """Unit waveform of channel k with its own clock at `phase_rad`
        (mirrors the kernel: sin, or square with duty)."""
        a = phase_rad + float(self.ch_ph[k])
        if int(self.ch_kind[k]) == K_SQUARE:
            d = float(self.ch_duty[k])
            if d == 0.5:
                return 1.0 if math.sin(a) >= 0.0 else -1.0
            return 1.0 if (a / (2.0 * math.pi)) % 1.0 < d else -1.0
        return math.sin(a)

    # ---- plotting surfaces (mirrored to +-r), the shared r-z model API
    def extent(self):
        nz, nr = self.A.shape
        return np.arange(nz) * self.mm_per_gu, np.arange(nr) * self.mm_per_gu

    def has_drive(self):
        """True if this model carries any time-dependent drive. Declared
        by the model from its OWN data (`drives`, always set in __init__),
        read by pe_view.model_has_rf -- never sniffed (2026-09-18)."""
        return bool(self.drives)

    def potential_image(self, rf_phase=None):
        """DC potential (rf_phase=None), or the drive snapshot with every
        channel's own waveform evaluated at `rf_phase` on its clock (the
        single-sin deck reproduces the old sin(phase)*rf_V*B exactly)."""
        z, r = self.extent()
        phi = self.A.copy()
        if rf_phase is not None:
            for k, B0 in enumerate(self.chan_phi):
                phi = phi + self._w_at_phase(k, float(rf_phase)) * B0
        r_full = np.concatenate([-r[::-1], r[1:]])
        img = np.concatenate([phi[:, ::-1], phi[:, 1:]], axis=1)
        em = np.concatenate([self.ele[:, ::-1], self.ele[:, 1:]], axis=1)
        return z, r_full, img, em

    def pe_surface(self, mz=None, charge=1, plane="rz", t_us=0.0):
        """Effective (adiabatic) potential-energy landscape in eV: DC
        potential energy + the RF Dehmelt pseudopotential
        V_pseudo = q|E0|^2/(4 m Omega^2), per drive group by its
        resolved pe_mode exactly as the planar model: sin -> pseudo
        (quadrature-composed per frequency), square -> pseudo carries the
        digital-trap harmonic factor pi^2/6, 'instant' drives add their
        real potential at t_us. Mass-dependent (heavier -> shallower
        well); valid in the adiabatic regime (Mathieu q ≲ ~0.4).
        Returns (z, r_full, PE_eV, em) mirrored to +-r like
        potential_image."""
        if plane not in self.PLANES:
            raise ValueError(
                f"{type(self).__name__} has no {plane!r} plane; it has "
                f"{self.PLANES}. (Declared capability -- not a signature sniff.)")
        z, r = self.extent()
        if mz is None:
            mz = self.spec.source.mz_list[0]
        # A is on the folded (r>=0) grid; the aware fields EzB/EuB are on
        # the UNFOLDED (+-r) grid (nz, 2nr-1) and already in V/m. Mirror A
        # and add the pseudopotential on the unfolded grid directly.
        r_full = np.concatenate([-r[::-1], r[1:]])
        img = charge * np.concatenate([self.A[:, ::-1], self.A[:, 1:]],
                                      axis=1)
        m_kg = mz * 1.6605402e-27
        q_c = charge * 1.602176634e-19
        # sin channels sharing a frequency compose in QUADRATURE:
        # E(t) = sin(wt) Es + cos(wt) Ec with Es = sum cos(ph) E_k,
        # Ec = sum sin(ph) E_k, so the Dehmelt |E0|^2 = |Es|^2 + |Ec|^2
        # (the 0/180 funnel pair reduces to the old rf_V*(B_A - B_B)).
        quad = {}
        for k, (B0, g) in enumerate(self.drives):
            if int(self.ch_kind[k]) != K_SIN or self.ch_om[k] <= 0.0:
                continue
            if g.resolved_pe_mode() != "pseudo":
                continue
            ent = quad.setdefault(round(float(self.ch_om[k]), 12),
                                  [np.zeros_like(self.EzA),
                                   np.zeros_like(self.EzA),
                                   np.zeros_like(self.EzA),
                                   np.zeros_like(self.EzA)])
            c, sph = math.cos(self.ch_ph[k]), math.sin(self.ch_ph[k])
            ent[0] += c * self.EzK[k]
            ent[1] += c * self.EuK[k]
            ent[2] += sph * self.EzK[k]
            ent[3] += sph * self.EuK[k]
        for om_us, (esz, esu, ecz, ecu) in quad.items():
            om = om_us * 1e6                                      # rad/s
            e0_sq = esz ** 2 + esu ** 2 + ecz ** 2 + ecu ** 2     # (V/m)^2
            img = img + charge * q_c * e0_sq / (4.0 * m_kg * om ** 2)
        for k, (B0, g) in enumerate(self.drives):
            mode = g.resolved_pe_mode()
            if mode == "instant":
                # the slow-drive picture: the real potential at t_us
                w = self._w_at_phase(k, float(self.ch_om[k]) * t_us)
                img = img + charge * w * np.concatenate(
                    [B0[:, ::-1], B0[:, 1:]], axis=1)
            elif int(self.ch_kind[k]) == K_SQUARE and self.ch_om[k] > 0.0:
                # digital (square) pseudo: harmonic sum factor pi^2/6,
                # per channel (cross-channel square interference at one
                # frequency is not composed — same scope as planar)
                om = float(self.ch_om[k]) * 1e6
                e0_sq = self.EzK[k] ** 2 + self.EuK[k] ** 2
                img = img + charge * q_c * e0_sq * (math.pi ** 2 / 6.0) \
                    / (4.0 * m_kg * om ** 2)
        em = np.concatenate([self.ele[:, ::-1], self.ele[:, 1:]], axis=1)
        return z, r_full, img, em

    def efield_magnitude(self):
        """Drive |E| at the sin-reference peak (V/m), matching the planar
        convention: sin channels at cos(phase) — the exact quadrature
        snapshot — squares at +1. The single-sin deck reproduces the old
        rf_V * |grad B| bit-for-bit."""
        ez = np.zeros_like(self.EzA)
        eu = np.zeros_like(self.EuA)
        for k in range(len(self.chan_phi)):
            w = (math.cos(self.ch_ph[k])
                 if int(self.ch_kind[k]) == K_SIN else 1.0)
            ez = ez + w * self.EzK[k]
            eu = eu + w * self.EuK[k]
        return np.hypot(ez, eu)


def build_rz_model(spec: SimSpec, verbose=False, masks_override=None):
    """Rasterize + natively solve a cylindrical SimSpec. Bases cached by
    geometry (voltage change re-weights, no re-solve).

    masks_override: {index: (nz,nr) bool} — inject pre-voxelized electrode
    masks (imported geometry) instead of rasterizing from inline shapes. Skips
    the shape requirement and the shape-based symmetry verification (the
    masks are the ground truth)."""
    g = spec.geometry
    h = g.mm_per_gu
    # node-centred grid (see build_planar): n+1 nodes span
    # the declared extent INCLUSIVE — and the r axis finally gets a TRUE
    # r = 0 node (cell-centred sampling put the innermost node at h/2,
    # so the cylindrical axis itself was never on the grid).
    # THE counting function: round+1 == cells+1 for every
    # conforming deck (byte-identical grid); a non-conforming extent is
    # refused with the two nearest conforming extents instead of the
    # remainder being silently rounded away.
    from ion_gym.io.lattice import gu_nodes
    nz = gu_nodes(g.width_mm, h, axis="z",
                  what="width_mm axial extent")     # axis extent (z)
    nr = gu_nodes(g.height_mm, h, axis="r",
                  what="height_mm radial extent")   # radial extent (r)

    ckey = _geometry_key(spec)
    cached = None if masks_override is not None else _RZ_BASIS_CACHE.get(ckey)
    if cached is None and masks_override is None:
        # DISK TIER: the r-z builder was
        # the one memory-only builder, so its fields could not be saved or
        # survive a process (field_io's "cache-backed builders only"
        # refusal).  Same basis_cache pack/load as planar — a miss returns
        # (None, None), a bool-`ele` legacy entry is refused loudly inside
        # basis_cache.load, and a hit warms the in-process dict.
        from ion_gym.io import basis_cache
        dbases, dele = basis_cache.load(spec)
        if dbases is not None:
            cached = (dbases, dele)
            _RZ_BASIS_CACHE[ckey] = cached
            if verbose:
                print(f"rz bases: disk cache hit ({ckey})")
    if cached is not None:
        bases, ele = cached
    else:
        if masks_override is not None:
            masks = {i: np.asarray(m, bool) for i, m in
                     masks_override.items()}
        else:
            # rasterize each electrode in the (z, r) plane. build_planar's
            # electrode_mask works on any (X, Y) meshgrid; here X=z, Y=r.
            # kernel frame — see build_planar's rasterizer note:
            # node i acts at i*h; sampling there makes the
            # effective metal equal the spec.
            zs = np.arange(nz) * h
            rs = np.arange(nr) * h
            from ion_gym.physics.raster2d import (plane_grid_views)
            Z, R = plane_grid_views(zs, rs, what="rz rasterizer")
            masks = {}
            for idx, el in enumerate(g.electrodes, start=1):
                if not el.shapes:
                    raise ValueError(
                        f"rz builder needs inline shapes; electrode "
                        f"{el.name!r} has none (STL rz path not wired)")
                masks[idx] = electrode_mask(el, Z, R)

            # declared symmetry: rz forbids y(=r) mirror; remaining
            # declarations are verified here before use. (The r=0 axis is
            # intrinsic to the cylindrical solver, not declared.)
            # Two tiers:
            # MIRROR planes are proven EXACTLY on the continuous shapes
            # (the mask check measures the raster's half-open
            # skin, not the geometry); TRANSLATIONAL planes keep the mask
            # check (no finite-shape witness exists for invariance).
            # Refusal semantics unchanged for both.
            sym = spec.geometry.symmetry.normalized()
            ok, report = verify_symmetry_shapes(
                sym, g.electrodes,
                {"x": g.width_mm, "y": g.height_mm})
            if not ok:
                bad = [d for a, k, o, d, sc in report if not o]
                raise ValueError("declared symmetry not satisfied — refusing "
                                 "to fold a false plane: " + "; ".join(bad))
            mok, mreport = verify_symmetry(sym, masks)
            tbad = [f"{a}:{k} ({f:.1%}, {sc})"
                    for a, k, o, f, sc in mreport
                    if not o and k == "translational"]
            if tbad:
                raise ValueError("declared symmetry not satisfied — refusing "
                                 "to fold a false plane: " + "; ".join(tbad))
            if verbose and (report or mreport):
                print("symmetry verified:", reduction_summary(sym))

        metal = np.zeros((nz, nr), bool)
        for m in masks.values():
            metal |= m
        # solve one basis per electrode: that electrode at 1e4 V, all
        # other metal at 0 — the fast-adjust bases. Direct sparse solve
        # (exact; no omega/convergence concerns).
        bases = {}
        for idx, m in masks.items():
            fixed = metal
            val = np.zeros((nz, nr))
            val[m] = 1e4
            phi = solve_laplace(fixed, val, h, symmetry="cylindrical",
                                edge_ghost="ghost_linear")
            bases[idx] = phi
            if verbose:
                print(f"basis {idx} ({g.electrodes[idx-1].name}): solved")
        # ele must carry INT16 LABELS, not a boolean mask.
        # A bool `ele` makes `ele == 1` true for EVERY conductor (True == 1)
        # and `ele == 3` true for none, so the PE view and the electrode
        # legend drape every electrode at electrode #1's voltage. This is the
        # identical defect that was root-caused in build_planar; build_rz was
        # never fixed. It does not corrupt the solve (the bases are built from
        # the per-electrode masks, not from `ele`) -- it corrupts DISPLAY,
        # which is exactly the class of bug that "display must equal solver
        # input" exists to forbid.
        # Bool consumers are unaffected: they test `ele > 0.5`.
        ele = np.zeros((nz, nr), np.int16)
        # is_grid TRANSPARENCY: an electrode declared
        # is_grid=True is a Dirichlet boundary for the SOLVE (it is in
        # `masks` above) but the ion flies THROUGH it — a mesh, a
        # reflectron entrance plane, a TOF source exit grid, a detector
        # face. It must stay OUT of the collide-mask `ele` (label 0) or
        # the ion dies on a boundary condition. This is the SAME fix
        # build_planar carries; build_rz never got it, so grids were walls
        # (found flying the reflectron TOF: 800 eV ions impactted on the
        # 0 V entrance grid instead of entering the mirror).
        for idx, m in masks.items():
            if getattr(g.electrodes[idx - 1], "is_grid", False):
                continue
            ele[m] = idx
        if masks_override is None:
            _RZ_BASIS_CACHE[ckey] = (bases, ele)
            # DISK PUBLISH: atomic
            # basis_cache.store — after this, save_field works for r-z
            # and the next process's build is a disk hit.  A failed
            # publish must not kill a successful solve, but it must be
            # SAID: a silent miss here re-solves forever and
            # save_field "mysteriously" refuses.
            try:
                from ion_gym.io import basis_cache
                basis_cache.store(spec, bases, ele)
            except OSError as e:
                print(f"rz bases: disk cache publish FAILED ({e}) — the "
                      f"solve is good but will not persist; save-field "
                      f"and cross-session reuse are unavailable until "
                      f"the cache directory is writable.")

    # assemble A (DC + drive offsets) + per-GROUP drive channels. DC and
    # drives are independent: every electrode's dc always applies; drive
    # membership is via EVERY named group (the first-group-only read and
    # the single signed-B fold were the L-455 defect this replaces). A
    # group's offset_v is a STATIC shift of its members (V(t) =
    # amplitude_v * w(t) + offset_v), so it folds into A exactly, for
    # every waveform kind and even when amplitude_v is 0.
    A = np.zeros((nz, nr))
    gmap = {gr.name: gr for gr in (g.rf_groups or [])}
    chan_B, chan_obj, chan_order = {}, {}, []
    # GATE channel (for e.g. CDMS ELIT gating): a group with a 2-point
    # hold table [t0, tau] -> [v0, v1] is a STEP; its members contribute
    # amp*(v1 - v0) to the gate basis G, applied from tau on. Exactly
    # parallel to the sin*B channel; ONE tau per model (a second distinct
    # tau refuses with a diagnostic -- single-gate tracer, same doctrine
    # as the single-B fold). dc holds the PRE-gate state; dc + amp*(v1-v0)
    # is the post-gate state. Longer tables refuse: not a step.
    G = np.zeros((nz, nr))
    tau_gate = -1.0
    for idx, el in enumerate(g.electrodes, start=1):
        fa = bases[idx] / 1e4
        A = A + el.dc * fa
        for gr in (g.rf_groups or []):
            if gr.waveform != "table" or gr.name not in (el.rf_groups or []):
                continue
                # (membership for the gate is the RAW rf_groups list, as
            # before; the drive channels below use group_names())
            tt, tv = list(gr.table_t_us or []), list(gr.table_v or [])
            if len(tt) != 2 or len(tv) != 2 or gr.interp != "hold":
                raise ValueError(
                    f"r-z gate: group {gr.name!r} on electrode "
                    f"{el.name!r} has a {len(tt)}-point/'{gr.interp}' "
                    "table; the r-z tracer supports exactly a 2-point "
                    "HOLD step (a gate). Author it as [t0, tau] -> "
                    "[v0, v1], interp 'hold'.")
            if tau_gate >= 0.0 and abs(tt[1] - tau_gate) > 1e-9:
                raise ValueError(
                    f"r-z gate: two distinct gate times ({tau_gate} and "
                    f"{tt[1]} us). The single-gate tracer switches ONCE; "
                    "put all gated electrodes on tables sharing one tau.")
            tau_gate = tt[1]
            G = G + gr.amplitude_v * (tv[1] - tv[0]) * fa
        for gname in el.group_names():
            if gname not in gmap:
                raise ValueError(
                    f"electrode {el.name!r} names drive group {gname!r} "
                    f"not in geometry.rf_groups ({sorted(gmap)})")
            drv = gmap[gname]
            if drv.waveform == "table":
                continue          # table groups are the gate, handled above
            drv.validate()
            off = float(getattr(drv, "offset_v", 0.0))
            if off:
                A = A + off * fa
            if drv.amplitude_v == 0.0:
                continue
            if drv.name not in chan_B:
                chan_B[drv.name] = np.zeros((nz, nr))
                chan_obj[drv.name] = drv
                chan_order.append(drv.name)
            # amplitude BAKED into the channel potential; the kernel's
            # w(t) is the UNIT waveform (frequency 0 is NOT skipped —
            # sin/square at 0 Hz are constants the kernel evaluates)
            chan_B[drv.name] += drv.amplitude_v * fa
    drives = [(chan_B[n], chan_obj[n]) for n in chan_order]

    Z_mm = np.arange(nz) * h
    U_mm = np.arange(nr) * h
    _, Ue, EzA, EuA = build_field_aware(Z_mm, U_mm, A, ele,
                                        symmetry="cylindrical")
    EzK_l, EuK_l = [], []
    for B0, _drv in drives:
        _, _, Ez_k, Eu_k = build_field_aware(Z_mm, U_mm, B0, ele,
                                             symmetry="cylindrical")
        EzK_l.append(Ez_k)
        EuK_l.append(Eu_k)
    EzK = (np.ascontiguousarray(np.stack(EzK_l)) if EzK_l
           else np.zeros((0,) + EzA.shape))
    EuK = (np.ascontiguousarray(np.stack(EuK_l)) if EuK_l
           else np.zeros((0,) + EuA.shape))
    ch_kind = np.array([K_SQUARE if d.waveform == "square" else K_SIN
                        for _B, d in drives], np.int64)
    ch_om = np.array([d.frequency_hz * 1e-6 * 2 * math.pi
                      for _B, d in drives], np.float64)
    ch_ph = np.array([math.radians(d.phase_deg) for _B, d in drives],
                     np.float64)
    ch_duty = np.array([float(d.duty) for _B, d in drives], np.float64)
    _, _, EzG, EuG = build_field_aware(Z_mm, U_mm, G, ele,
                                       symmetry="cylindrical")
    col = spec.collisions
    return RZModel(A, ele, EzA, EuA, EzK, EuK, ch_kind, ch_om, ch_ph,
                   ch_duty, drives, Ue[0], h,
                   col.T_k, col.P_pa, col.sigma_m2,
                   gas_mass(col.gas) if col.enabled else 4.0, spec,
                   EzG=EzG, EuG=EuG, tau_gate=tau_gate)


def unsupported_drive_features(spec):
    """Declared drive features the r-z kernel does NOT apply: NONE, as of
    2026-09-16 (L-455). The kernel flies per-group channels — every sin
    and square group at its own amplitude, frequency, phase and duty,
    with offset_v folded into the static field and multi-group
    membership honoured — plus the 2-point hold-table GATE. General
    table waveforms REFUSE at build with a diagnostic (they never fly
    wrong), so nothing builds-but-flies-differently. Kept as the route's
    contract statement for the field export door."""
    return []



# The r-z drive channels are sin/square only (general tables refuse at
# build; the 2-point hold table IS the gate channel), so the kernel's
# shared breakpoint arrays are empty — one definition, both call sites.
_TAB_T0 = np.zeros(0, np.float64)
_TAB_V0 = np.zeros(0, np.float64)


def _tab_off0(model):
    return np.zeros(model.EzK.shape[0] + 1, np.int64)


def make_rz_fly_fn(model: RZModel, births, spec: SimSpec):
    """(fly_fn, col_names) reusing the funnel's validated r-z tracer.
    ele mask is mirror-extended (funnel convention)."""
    # The r-z kernel produces exactly these optional channels, in
    # OPTIONAL_CHANNELS order. A requested channel outside this set
    # (Cartesian e_x/e_y/e_z, ke_x/y/z) used to be silently allocated as
    # an UNWRITTEN column of np.empty garbage: col_names listed it, the
    # kernel never wrote it. Refuse with the name instead.
    rz_channels = ("speed", "ke_ev", "ke_x", "ke_y", "ke_z", "e_field",
                   "e_axial", "e_radial", "radius", "n_col", "path_mm",
                   "e_axial_tint", "ke_tint")
    if getattr(spec, "transporter", None) is not None:
        raise ValueError("the r-z route does not implement the periodic "
                         "transporter (CHARTER_transporter); remove the "
                         "'transporter' block or fly a 3-D shapes route")

    unsupported = [c for c in spec.integration.record_channels
                   if c not in rz_channels]
    if unsupported:
        raise ValueError(
            f"r-z tracer cannot record channel(s) {unsupported}; the r-z "
            f"kernel produces {list(rz_channels)}. (The kernel state is "
            f"full 3-D Cartesian, so ke_x/ke_y/ke_z ARE available "
            f"[2026-08-19]; the FIELD, however, is solved as (axial, "
            f"radial) on the half-plane, so e_x/e_y/e_z are not "
            f"produced -- record e_axial/e_radial instead.)")
    chans = [c for c in OPTIONAL_CHANNELS
             if c in spec.integration.record_channels]
    col_names = BASE_CHANNELS + chans
    # default mass (mz_list[0]); the actual per-ion mass/accel are chosen
    # inside fly_fn so a single run can mix species.
    spec.source.mz_list[0]
    col = spec.collisions
    mg = model.m_gas
    T = col.T_k if col.enabled else 298.0
    c_star = math.sqrt(2 * KB * T / (mg * KG_AMU)) / 1000.0
    c_bar = math.sqrt(8 * KB * T / (math.pi * mg * KG_AMU)) / 1000.0
    sig1d = math.sqrt(KB * T / (mg * KG_AMU)) / 1000.0
    P = col.P_pa if col.enabled else 0.0     # P=0 -> collisionless
    ee = np.concatenate([model.ele[:, :0:-1], model.ele],
                        axis=1).astype(np.float64)
    # r-z tracer flag order == rz_channels order (13 flags; the tuple is
    # built from THIS local registry, never the global one -- the planar
    # route's global-registry flag bug is the cautionary tale).
    f13 = tuple(c in chans for c in rz_channels)
    # ion charge state (SourceSpec.charge, signed integer): the kernel's
    # acceleration scale is q/m, and mz_list carries the ion MASS in Da
    # (node-centred convention, same as the SDS route: fly3d_sds(mz, charge)).
    # Previously the r-z route silently flew every ion at |z| = 1.
    q_e = int(spec.source.charge)
    if q_e == 0:
        raise ValueError("source.charge = 0: an uncharged ion has no "
                         "electric acceleration; state the charge state.")

    dt = spec.integration.dt_ns * 1e-3
    _bf, _bv = spec.bounds.as_tuple()
    bnd_on = np.array(_bf, np.bool_)
    bnd_val = np.array(_bv, np.float64)

    # STATION PLANES for the kernel, from the ONE shared builder. This
    # route flies the frame the deck is authored in (no anchor offset —
    # bounds go in raw, just above), so the offset is zeros. The kernel
    # carries the full (y, z) transverse state, so windows are the same
    # rectangles every other route evaluates.
    from ion_gym.physics.stations import station_planes as _st_planes
    _AXCOL = {"x": 0, "y": 1, "z": 2}
    _pl = _st_planes(spec, (0.0, 0.0, 0.0))
    pl_col = np.array([_AXCOL[p[0]] for p in _pl], np.int64)
    pl_val = np.array([p[1] for p in _pl], np.float64)
    pl_sgn = np.array([p[2] for p in _pl], np.float64)
    pl_w = (np.array([list(p[3]) for p in _pl], np.float64)
            if _pl else np.empty((0, 4), np.float64))
    pl_kind = np.array([p[4] for p in _pl], np.int64)

    # DRIFT EXTENSION: residual |E| on each OPEN boundary face,
    # measured ONCE per model (the guard input). Component magnitudes
    # are worst-case bounds: static A exactly, every drive channel at
    # its unit-waveform peak |w| = 1 (amplitude is baked into the
    # channel), gate basis G at its full gain of 1.
    from ion_gym.physics.drift_extension import (edge_field_max,
                                                 extend_ballistic)
    _ez_c = [(model.EzA, 1.0)] \
        + [(model.EzK[_k], 1.0) for _k in range(model.EzK.shape[0])] \
        + [(model.EzG, 1.0)]
    _eu_c = [(model.EuA, 1.0)] \
        + [(model.EuK[_k], 1.0) for _k in range(model.EuK.shape[0])] \
        + [(model.EuG, 1.0)]
    # edge_field_max is unit-passthrough and its consumer contract is
    # V/mm (extend_ballistic's edge_e_vpermm vs EDGE_FIELD_MAX_V_PER_MM);
    # the r-z basis arrays are V/m, so convert here (unconverted, the
    # guard saw 1000x and over-refused).
    _edge_e = {"x_lo": 1e-3 * edge_field_max([_ez_c, _eu_c], 0, "lo"),
               "x_hi": 1e-3 * edge_field_max([_ez_c, _eu_c], 0, "hi"),
               "r_hi": 1e-3 * edge_field_max([_ez_c, _eu_c], 1, "hi")}
    _nx_gu = model.EzA.shape[0]

    def fly_fn(i):
        b = births[i]
        from ion_gym.physics.raster2d import (refuse_birth_in_metal)
        _r = math.hypot(b[1], b[2])
        refuse_birth_in_metal(
            model.ele, b[0] / model.mm_per_gu, _r / model.mm_per_gu,
            model.mm_per_gu, spec, i,
            f"(x={b[0]:.3f}, r={_r:.3f}) mm")
        # per-ion mass + seed from THE envelope:
        # one derivation for every route — mz via mz_of (contiguous
        # blocks, matching births) and
        # the reproducibility seed. Both the acceleration scaling and
        # the recorded KE depend on mass.
        from ion_gym.physics.ion_envelope import per_ion
        env = per_ion(spec, i)
        m_i = env.mz
        acc_i = q_e * E_CHG / (m_i * KG_AMU)
        # Size the record buffer to the ACTUAL number of records the run
        # will produce (t_max / dt / rec_every) + margin, capped for memory.
        # A fixed buffer truncated the trajectory when it filled, and the
        # final endpoint written afterward drew a straight line to the end —
        # which LOOKED like ions streaming through without colliding
        # (physics was fine; only the recording was clipped). +8 covers the
        # row-0 write, the final-row write, and rounding.
        n_rows = int(spec.integration.t_max_us / dt
                     / max(spec.integration.rec_every, 1)) + 8
        n_rows = min(max(n_rows, 16), 4_000_000)
        rec = np.empty((n_rows, len(col_names)))
        n, kind, ncol, _bface = _fly_rec_full(
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], m_i,
            model.EzA, model.EuA, model.EzK, model.EuK,
            model.ch_kind, model.ch_om, model.ch_ph, model.ch_duty,
            _TAB_T0, _TAB_V0, _tab_off0(model),
            model.EzG, model.EuG, model.tau_gate, ee, model.u0,
            model.mm_per_gu, acc_i, dt,
            spec.integration.t_max_us, T, P, model.sigma_m2, c_star,
            c_bar, sig1d, mg, rec, spec.integration.rec_every,
            env.seed, *f13, bnd_on, bnd_val,
            pl_col, pl_val, pl_sgn, pl_w, pl_kind)
        traj = rec[:n].copy()
        # Honor declared downstream bounds/stations by exact
        # ballistic algebra when the flight left the solved domain.
        # Escape-face classification mirrors the kernel's own
        # termination tests; the r-z radius convention (sqrt(y^2+z^2))
        # is supplied HERE, where the route owns it.
        if kind == 1:
            _xe = float(traj[-1, 1])
            if _xe < 0.0:
                _fk, _fn = "x_lo", "x low (axial entrance)"
            elif _xe / model.mm_per_gu > _nx_gu - 1:
                _fk, _fn = "x_hi", "x high (axial exit)"
            else:
                _fk, _fn = "r_hi", "radial edge"
            # face map, r-z convention (BoundsSpec: x is the axis, y the
            # radius): x_lo/x_hi -> axis 0; the radial edge -> axis 1 hi.
            _fa, _fs = {"x_lo": (0, "lo"), "x_hi": (0, "hi"),
                        "r_hi": (1, "hi")}[_fk]
            traj, kind, _ext = extend_ballistic(
                traj, col_names, kind, spec,
                edge_e_vpermm=_edge_e[_fk], face_axis=_fa, face_side=_fs,
                edge_face=_fn,
                derived={"radius": lambda prev, row, idx, dt:
                         math.hypot(row[idx["y"]], row[idx["z"]])},
                label=f"ion {i}")
        from ion_gym.io.records import TrajRecord
        end = TrajRecord(traj, col_names).row(-1)
        from ion_gym.physics.ion_envelope import make_summary
        summary = make_summary(
            kind=kind, tof=float(end["t"]), mz=env.mz,
            x_end=float(end["x"]), y_end=float(end["y"]),
            z_end=float(end["z"]),
            r_end=float(math.hypot(end["y"], end["z"])), n_col=ncol)
        return traj, summary
    return fly_fn, col_names


def rz_fly_fields(model: "RZModel", spec: SimSpec):
    """The staged-assembly pack for an r-z region.

    A staged assembly must fly an ARBITRARY state through an
    already-solved region, which `fly_fn` cannot do -- it closes over
    this spec's own births by index. This gathers exactly what
    `_fly_rec_full` needs, so a gas cell or ion guide can be a STAGE of
    an instrument rather than a separate study.

    Route-tagged, not converted. An r-z solve is a half-plane (axial,
    radial) field with full 3-D Cartesian ion state; it is not a
    degenerate 3-D grid and must not be revolved into one to join an
    assembly. The consumer dispatches on `route`.

    GAS IS NOT IN THIS PACK. Gas belongs to the STAGE, not to a
    geometry class -- a drift cell may be r-z and an ion guide planar,
    and either can be gas-filled. `Region.gas` already carries it for
    every route, resolved from the stage's own `spec.collisions` with an
    explicit vacuum override. Duplicating it here would be a second
    authority that drifts from the first, and the drift would be silent:
    a stage flown at the wrong pressure still completes and still
    reports a plausible number.
    """
    return dict(
        route="rz",
        EzA=model.EzA, EuA=model.EuA, EzK=model.EzK, EuK=model.EuK,
        ch_kind=model.ch_kind, ch_om=model.ch_om, ch_ph=model.ch_ph,
        ch_duty=model.ch_duty,
        tab_t=_TAB_T0, tab_v=_TAB_V0, tab_off=_tab_off0(model),
        EzG=model.EzG, EuG=model.EuG, tau_gate=model.tau_gate,
        ele=np.concatenate([model.ele[:, :0:-1], model.ele],
                           axis=1).astype(np.float64),
        u0=model.u0, h_mm=model.mm_per_gu,
        charge=int(spec.source.charge),
    )


def build_rz_run(spec: SimSpec, verbose=False):
    """SimSpec (rz) -> (model, fly_fn, col_names, births). Independent."""
    model = build_rz_model(spec, verbose=verbose)
    births = generate_births(spec)
    fly_fn, cols = make_rz_fly_fn(model, births, spec)
    model.fly_fields = rz_fly_fields(model, spec)
    return model, fly_fn, cols, births
