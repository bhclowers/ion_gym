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
from ion_gym.physics.build_planar import electrode_mask          # shape rasterizer (shared)
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

    def __init__(self, A, B, ele, EzA, EuA, EzB, EuB, u0, mm, rf_V, om,
                 T_k, P_pa, sigma_m2, m_gas, spec, *,
                 EzG=None, EuG=None, tau_gate=-1.0):
        self.A = A
        from ion_gym.physics.raster2d import (el_masks_from_labels)
        # Named by the deck's own electrode names.
        self.el_masks = el_masks_from_labels(
            ele, getattr(spec.geometry, 'electrodes', None))
        self.B = B
        self.ele = ele
        self.EzA, self.EuA = EzA, EuA
        self.EzB, self.EuB = EzB, EuB
        self.u0 = u0
        self.mm_per_gu = mm
        self.rf_V = rf_V
        self.om_rad_us = om
        # gate channel: step(t - tau_gate) * G; zeros/-1 = no gate
        self.EzG = EzG if EzG is not None else np.zeros_like(EzA)
        self.EuG = EuG if EuG is not None else np.zeros_like(EuA)
        self.tau_gate = float(tau_gate)
        self.T_k, self.P_pa = T_k, P_pa
        self.sigma_m2, self.m_gas = sigma_m2, m_gas
        self.spec = spec

    # ---- plotting surfaces (mirrored to +-r), the shared r-z model API
    def extent(self):
        nz, nr = self.A.shape
        return np.arange(nz) * self.mm_per_gu, np.arange(nr) * self.mm_per_gu

    def potential_image(self, rf_phase=None):
        z, r = self.extent()
        phi = self.A.copy()
        if rf_phase is not None:
            phi = phi + math.sin(rf_phase) * self.rf_V * self.B
        r_full = np.concatenate([-r[::-1], r[1:]])
        img = np.concatenate([phi[:, ::-1], phi[:, 1:]], axis=1)
        em = np.concatenate([self.ele[:, ::-1], self.ele[:, 1:]], axis=1)
        return z, r_full, img, em

    def pe_surface(self, mz=None, charge=1, plane="rz"):
        """Effective (adiabatic) potential-energy landscape in eV: DC
        potential energy + the RF Dehmelt pseudopotential
        V_pseudo = q|E0|^2/(4 m Omega^2). The r-z path keeps B as a UNIT
        basis with the amplitude applied at fly time, so the cycle-peak RF
        field is |E0| = rf_V * |grad B|. Mass-dependent (heavier ->
        shallower well); valid in the adiabatic regime (Mathieu q ≲ ~0.4).
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
        if self.rf_V != 0.0 and self.om_rad_us > 0.0:
            e0 = self.rf_V * np.hypot(self.EzB, self.EuB)         # V/m
            om = self.om_rad_us * 1e6                             # rad/s
            m_kg = mz * 1.6605402e-27
            v_pseudo = (charge * 1.602176634e-19) * e0 ** 2 \
                / (4.0 * m_kg * om ** 2)
            img = img + charge * v_pseudo
        em = np.concatenate([self.ele[:, ::-1], self.ele[:, 1:]], axis=1)
        return z, r_full, img, em

    def efield_magnitude(self):
        return self.rf_V * np.hypot(self.EzB, self.EuB)


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

    # assemble A (DC) + RF basis from resolved GROUPS. DC and RF are
    # independent: every electrode's dc always applies; RF membership is
    # via its resolved group (named rf_group, or legacy fields).
    A = np.zeros((nz, nr))
    # collect per-group bases keyed by (freq, phase)
    gbases = {}
    rf_V = 0.0
    om = 0.0
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
        amp, freq, phase = g.electrode_rf(el)
        if amp != 0.0 and freq != 0.0 and getattr(
                next((x for x in (g.rf_groups or [])
                      if x.name in (el.rf_groups or [])), None),
                "waveform", "sin") == "table":
            continue          # table groups are the gate, not sin RF
        if amp != 0.0 and freq != 0.0:
            key = (round(freq, 3), round(phase, 3))
            gbases.setdefault(key, [np.zeros((nz, nr)), amp])
            gbases[key][0] += fa
            rf_V = max(rf_V, amp)
            om = freq * 1e-6 * 2 * math.pi
    # The r-z tracer flies ONE RF basis B with a two-phase sign (0/180).
    # For the common funnel/quad case (two groups 180 apart) this is
    # exact: fold the two phase groups into +/- B. For >2 distinct phases
    # (a travelling wave) the single-B tracer is insufficient — that needs
    # the multi-B tracer extension (flagged below); we assemble the
    # dominant two-phase B and warn.
    B = np.zeros((nz, nr))
    phases = sorted({k[1] for k in gbases})
    for (freq, phase), (basis, amp) in gbases.items():
        sign = 1.0 if (phase % 360.0) < 90.0 or (phase % 360.0) >= 270.0 \
            else -1.0
        B = B + sign * basis
    if len(phases) > 2:
        import warnings
        warnings.warn(
            "r-z build: >2 RF phases (travelling wave) folded into a "
            "two-phase B — the single-B tracer approximates it. Multi-B "
            "travelling-wave tracer is the next extension.")

    Z_mm = np.arange(nz) * h
    U_mm = np.arange(nr) * h
    _, Ue, EzA, EuA = build_field_aware(Z_mm, U_mm, A, ele,
                                        symmetry="cylindrical")
    _, _, EzB, EuB = build_field_aware(Z_mm, U_mm, B, ele,
                                       symmetry="cylindrical")
    _, _, EzG, EuG = build_field_aware(Z_mm, U_mm, G, ele,
                                       symmetry="cylindrical")
    col = spec.collisions
    return RZModel(A, B, ele, EzA, EuA, EzB, EuB, Ue[0], h, rf_V, om,
                   col.T_k, col.P_pa, col.sigma_m2,
                   gas_mass(col.gas) if col.enabled else 4.0, spec,
                   EzG=EzG, EuG=EuG, tau_gate=tau_gate)


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

    # DRIFT EXTENSION: residual |E| on each OPEN boundary face,
    # measured ONCE per model (the guard input). Component magnitudes
    # are worst-case bounds: static A exactly, RF basis B at |rf_V|,
    # gate basis G at its full gain of 1.
    from ion_gym.physics.drift_extension import (edge_field_max,
                                                 extend_ballistic)
    _ez_c = [(model.EzA, 1.0), (model.EzB, abs(model.rf_V)),
             (model.EzG, 1.0)]
    _eu_c = [(model.EuA, 1.0), (model.EuB, abs(model.rf_V)),
             (model.EuG, 1.0)]
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
            model.EzA, model.EuA, model.EzB, model.EuB,
            model.EzG, model.EuG, model.tau_gate, ee, model.u0,
            model.mm_per_gu, acc_i, model.rf_V, model.om_rad_us, dt,
            spec.integration.t_max_us, T, P, model.sigma_m2, c_star,
            c_bar, sig1d, mg, rec, spec.integration.rec_every,
            env.seed, *f13, bnd_on, bnd_val)
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
        EzA=model.EzA, EuA=model.EuA, EzB=model.EzB, EuB=model.EuB,
        EzG=model.EzG, EuG=model.EuG, tau_gate=model.tau_gate,
        ele=np.concatenate([model.ele[:, :0:-1], model.ele],
                           axis=1).astype(np.float64),
        u0=model.u0, h_mm=model.mm_per_gu,
        rf_V=model.rf_V, om_rad_us=model.om_rad_us,
        charge=int(spec.source.charge),
    )


def build_rz_run(spec: SimSpec, verbose=False):
    """SimSpec (rz) -> (model, fly_fn, col_names, births). Independent."""
    model = build_rz_model(spec, verbose=verbose)
    births = generate_births(spec)
    fly_fn, cols = make_rz_fly_fn(model, births, spec)
    model.fly_fields = rz_fly_fields(model, spec)
    return model, fly_fn, cols, births
