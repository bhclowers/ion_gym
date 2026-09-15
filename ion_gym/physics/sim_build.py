"""
ion_gym.sim_build
-----------------
Turns a SimSpec (sim_spec.py) into the things a run needs:
  * generate_births(spec)          -> (N,7) birth table, ALL distributions
  * build_voltage_field(spec)      -> A, B basis fields from the voltage
                                      assignment (fast-adjust: re-weighting
                                      voltages never re-solves)
  * build_run(spec)                -> (model, fly_fn, col_names, births)

SCOPE (honest, staged — the schema is general; the builder grows to meet
it rung by rung, each validated):
  * SOURCE generation: GENERAL now — point/disc/line/grid/file, thermal or
    KE, m/z mix, tob spread. Geometry-agnostic.
  * FIELD assembly + fly: the CYLINDRICAL path (funnel/IMS) is wired and
    validated (reused the F-1 reference kernel via a voltage re-weighting;
    that kernel has since been removed along with its gates).
    PLANAR-2D and full-3D builders are stubbed with clear NotImplemented
    messages pointing at the validated pieces they'll compose (the 2-D
    aware-field tracer, the native solver3d + mesh legs from v18). This
    keeps the app honest: it runs what's proven and refuses what isn't,
    rather than silently producing an unvalidated field.

The voltage-adjustment invariant the app relies on: bases are solved once
per geometry; changing dc/rf per electrode only changes how A and the B_k
are summed. So "adjust voltages and redraw fields" is a re-weight, not a
re-solve — instant, and exact (the fast-adjust property banked since
Rung 1).
"""

import math

import numpy as np

from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.sizing import record_volume
from ion_gym.physics.collision3d import E_CHG, KG_AMU, KB

# Advisory texts already printed this process (see the advisory loop
# in build_run): each unique text prints exactly once per session.
_ADVISED: set = set()

# Preview cost gate: the transport-view geometry
# preview voxelizes a 3-D builder's electrodes BEFORE any solve. Above
# this node count the rasterization is too slow to run on a UI refresh,
# so preview_masks3d returns no preview instead (the solve still runs on
# commit). ~30 M nodes rasterizes in a couple of seconds; a 274 M-node
# import (the case that produced the old 48 h solve estimate) is well
# above it and is correctly skipped rather than hanging the UI.
PREVIEW_VOXEL_BUDGET = 30_000_000

# Gaussian FWHM = FWHM_PER_SIGMA * sigma.  Named because every beam spread
# in this project is quoted as FWHM while every generator wants sigma, and
# a bare 2.3548 in a function body is the kind of literal that drifts.
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


# ---------------------------------------------------------------- source
def generate_births(spec: SimSpec):
    """(N,7) = [x,y,z,vx,vy,vz,tob] for any SourceSpec distribution.
    Velocities in mm/us; positions mm; tob us."""
    s = spec.source
    from ion_gym.physics.ion_envelope import resolve_run_seed
    rng = np.random.default_rng(resolve_run_seed(spec))

    if s.distribution == "file":
        import os
        bf = s.births_file
        if bf and not os.path.isabs(bf) and not os.path.exists(bf):
            base = getattr(spec, "_loaded_from", None)
            if base:
                cand = os.path.join(os.path.dirname(base), bf)
                if os.path.exists(cand):
                    bf = cand
        s = type(s).from_dict({**s.to_dict(), "births_file": bf})
        import csv
        REQUIRED = ("x", "y", "z", "vx", "vy", "vz")   # tob defaults to 0
        rows = []
        with open(s.births_file) as fh:
            rd = csv.DictReader(fh)
            have = set(rd.fieldnames or ())
            missing = [c for c in REQUIRED if c not in have]
            if missing:
                raise ValueError(
                    f"births_file {s.births_file!r}: missing column(s) "
                    f"{missing}; header has {sorted(have)}. These were "
                    f"previously defaulted to 0.0, which for vz means a "
                    f"packet with NO drift velocity -- it would fly, miss "
                    f"the detector window and read as a transmission loss "
                    f"rather than a malformed file.")
            for r in rd:
                rows.append([float(r.get(k, 0.0)) for k in
                             (*REQUIRED, "tob")])
        out = np.array(rows)
        # mz_of() assigns mass by contiguous block: ion i -> mz_list[i//n_ions].
        # If the row count is not n_ions * len(mz_list) that mapping silently
        # mislabels masses, so refuse rather than fly the wrong m/z.
        want = s.n_ions * max(1, len(s.mz_list))
        if len(out) != want:
            raise ValueError(
                f"births_file {s.births_file!r} has {len(out)} rows but the "
                f"spec declares n_ions={s.n_ions} x {max(1, len(s.mz_list))} "
                f"m/z = {want}. Masses are assigned by contiguous block "
                f"(ion i -> mz_list[i//n_ions]), so a mismatch mislabels "
                f"every ion past the first block. Set n_ions to match the "
                f"file, or regenerate the file.")
        return out

    n_per = s.n_ions
    n_mz = max(1, len(s.mz_list))
    n = n_per * n_mz            # n_ions is PER m/z -> total across all masses
    out = np.zeros((n, 7))
    ax = {"x": 0, "y": 1, "z": 2}[s.axis]
    # float dtype is ESSENTIAL: integer centres would make pos an int64
    # array and silently TRUNCATE every random offset, quantizing any
    # distribution born at whole-mm centres.
    p0 = np.array([s.x0_mm, s.y0_mm, s.z0_mm], dtype=float)

    for i in range(n):
        pos = p0.copy()
        if s.distribution == "point":
            pass
        elif s.distribution in ("disc", "grid"):
            if s.distribution == "disc":
                rr = s.r_mm * math.sqrt(rng.uniform())
                th = rng.uniform(0, 2 * math.pi)
            else:
                g = int(math.ceil(math.sqrt(n_per)))
                ib = i % n_per            # index within this mass block
                gi, gj = ib % g, ib // g
                u = (gi / max(g - 1, 1) - 0.5) * 2 * s.r_mm
                v = (gj / max(g - 1, 1) - 0.5) * 2 * s.r_mm
                rr = math.hypot(u, v)
                th = math.atan2(v, u)
            # place the disc in the plane normal to `axis`
            a, b = [k for k in range(3) if k != ax]
            pos[a] += rr * math.cos(th)
            pos[b] += rr * math.sin(th)
        elif s.distribution == "line":
            ib = i % n_per
            pos[ax] += s.len_mm * (ib / max(n_per - 1, 1))
        elif s.distribution == "box":
            b = (list(s.box_mm) + [0.0, 0.0, 0.0])[:3]
            for k in range(3):
                pos[k] += b[k] * (rng.random() - 0.5)
        elif s.distribution == "gaussian":
            # Truncated Gaussian. Rejection sampling is EXACT
            # for the truncated normal (accepted draws are the
            # conditional distribution), and with trunc >= 1 sigma
            # enforced by gaussian_layout() acceptance is >= 68% per
            # axis, so the loop is short. The rng stream stays the run
            # seed's, so births are reproducible per seed like every
            # other distribution.
            sig_g, trc_g = s.gaussian_layout()
            for k in range(3):
                if sig_g[k] <= 0.0:
                    continue
                while True:
                    dpos = rng.normal() * sig_g[k]
                    if abs(dpos) <= trc_g[k]:
                        break
                pos[k] += dpos
        out[i, 0:3] = pos

        # kinematics — contiguous mass blocks (ion i -> mz_list[i // n_per])
        mz = s.mz_list[(i // n_per) % n_mz]
        m_kg = mz * KG_AMU
        # Velocity rule: a directed beam (ke_lo/ke_hi + direction) and a
        # thermal bath (temperature_k) SUPERPOSE when both are declared.
        # Physically that is a beam AT a temperature -- the configuration
        # every real oa-TOF source is, and the one that carries the
        # turn-around term. Historically temperature_k > 0 silently
        # REPLACED the beam; no known spec declared both, so superposing
        # defines new behavior without changing any existing config.
        v = np.zeros(3)
        if s.ke_hi > 0 or s.ke_lo > 0:
            ke = rng.uniform(s.ke_lo, s.ke_hi)
            sp = math.sqrt(2 * ke * E_CHG / m_kg) / 1000.0
            d = np.array(s.direction, float)
            _n = float(np.linalg.norm(d))
            if _n == 0.0:
                # Defence in depth behind SimSpec.validate(): the old
                # `or 1.0` guard turned this contradiction (energy
                # declared, no direction to put it in) into a zero
                # directed velocity, so the deck's ke_lo..ke_hi never
                # reached the solver and nothing said so. Refuse with
                # the same fix the validator names.
                raise ValueError(
                    f"source.direction is the zero vector while "
                    f"ke_lo..ke_hi = {s.ke_lo}..{s.ke_hi} eV declares a "
                    f"directed beam — the energy has nowhere to point. "
                    f"Give the beam a direction (e.g. [1.0, 0.0, 0.0]) "
                    f"or declare ke_lo = ke_hi = 0 for births at rest.")
            d = d / _n
            v = v + sp * d
        if s.temperature_k > 0:
            sig = math.sqrt(KB * s.temperature_k / m_kg) / 1000.0
            v = v + rng.normal(size=3) * sig
        dvf = list(s.dv_fwhm_ms or [])
        if any(x for x in dvf):
            if len(dvf) != 3:
                raise ValueError(
                    f"source: dv_fwhm_ms must be a 3-vector of per-axis "
                    f"velocity FWHM in m/s (x, y, z); got {s.dv_fwhm_ms!r} "
                    f"with {len(dvf)} entries. Silently padding would "
                    f"assign a spread to the wrong axis, and in a planar "
                    f"MRT the three axes are different physics.")
            if any(x < 0 for x in dvf):
                raise ValueError(
                    f"source: dv_fwhm_ms entries are FWHM widths and cannot "
                    f"be negative; got {s.dv_fwhm_ms!r}.")
            # m/s -> mm/us is a factor 1e-3; FWHM -> sigma divides by 2.3548
            sig3 = np.array(dvf, float) * 1e-3 / FWHM_PER_SIGMA
            v = v + rng.normal(size=3) * sig3
        vd = list(getattr(s, "v_drift_mm_us", None) or [])
        if any(x for x in vd):
            if len(vd) != 3:
                raise ValueError(
                    f"source: v_drift_mm_us must be a 3-vector of per-axis "
                    f"drift velocity in mm/us; got {vd!r} with {len(vd)} "
                    f"entries. Padding would put the drift on the wrong "
                    f"axis, and in a planar MRT the fold direction is one "
                    f"specific axis.")
            v = v + np.array(vd, float)
        out[i, 3:6] = v
        out[i, 6] = rng.uniform(0, s.tob_span_us)
    return out


def mz_of(spec, i):
    n_per = spec.source.n_ions
    n_mz = max(1, len(spec.source.mz_list))
    return spec.source.mz_list[(i // n_per) % n_mz]


# --------------------------------------------------------- field assembly
def build_voltage_field(spec: SimSpec, bases):
    """Assemble A (static DC) and the list of RF basis fields B_k from the
    DRIVE GROUPS, given per-electrode basis potentials `bases`. Each
    RFGroupSpec is one B_k (sum of its members' bases) carrying the
    group's (frequency, phase). A re-weight of pre-solved bases (no
    solve), so the app can sweep voltages live.

    Drive model: frequency/phase live on the GROUP.
    A rod pair is two groups 180 deg apart, each a distinct B_k."""
    els = spec.geometry.electrodes
    shape = next(iter(bases.values())).shape
    A = np.zeros(shape)
    groups = {gr.name: gr for gr in spec.geometry.rf_groups}
    accum = {name: np.zeros(shape) for name in groups}
    for i, e in enumerate(els, start=1):
        idx = e.basis if e.basis is not None else i
        fa = bases[idx]
        A = A + e.dc * fa
        for gname in e.group_names():
            if gname not in groups:
                raise ValueError(
                    f"electrode {e.name!r} names drive group {gname!r} "
                    f"not in geometry.rf_groups ({sorted(groups)})")
            accum[gname] = accum[gname] + gr_amp(groups[gname]) * fa
    Bk = [(accum[name], groups[name].frequency_hz, groups[name].phase_deg)
          for name in groups if groups[name].amplitude_v]
    return A, Bk


def gr_amp(gr):
    return float(gr.amplitude_v)


# ------------------------------------------------------------- run build
# --------------------------------------------------------------- routing
from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class BuildRoute:
    """WHAT build_run(spec) will do, stated as a declared property.

    This exists because six call sites in sim_app were answering "is this
    3-D?" from `depth_mm > 0` -- a correlate that was wrong twice over (the
    STL quadrupole: 2-D field on a z-transporting body; the 3-D SLIM:
    depth_mm==0 while the FIELD is 3-D, patched with a device-named builder
    special cases).  Dispatch is on a DECLARED property, and there is
    exactly one authority on what a spec builds -- the builder's own routing.
    `build_route()` IS that routing, factored so it can be ASKED, and
    `build_run()` dispatches on it, so the answer the app displays and the
    branch the builder takes CANNOT drift apart.

    field_dims answers the FIELD's question only: is the solved potential a
    function of z?  It says NOTHING about the body -- a straight multipole
    is a 3-D instrument on a 2-D field, and that is the normal case, not an
    edge case.  The BODY's extent is a property of the bodies (viz_core
    Scene.box()) and is deliberately not answerable here.
    """
    builder: str        # one of KNOWN_BUILDERS
                        # | 'stl2d' | 'planar' | 'unwired'
    field_dims: int     # 2 or 3: is the solved potential a function of z?
    coords: str         # 'rz' | 'xyz' (declared, passed through)


# Every builder this router implements. The list IS the contract: a retired
# builder simply leaves it and is refused by name at classification time,
# rather than earning a permanent named branch.
KNOWN_BUILDERS = ("stl3d", "scene3d", "shapes3d", "stl2d", "rz", "planar")


def build_route(spec: SimSpec) -> BuildRoute:
    """Classify a spec exactly as build_run() will dispatch it."""
    b = getattr(spec, "builder", "")
    g = spec.geometry
    coords = g.symmetry.coords
    # Generic unknown-builder refusal. The message may quote the name it
    # was handed, but the router deliberately carries no branch that knows
    # any particular retired builder: naming them here would leave a
    # permanent if-statement per retirement. Whatever arrives unrecognised
    # is quoted back against the current KNOWN_BUILDERS.
    if b and b not in KNOWN_BUILDERS:
        raise ValueError(
            f"spec declares builder={b!r}, which this package does not "
            f"provide. Known builders: {list(KNOWN_BUILDERS)}. Geometry is "
            f"authored natively -- see notebooks/08_shapes_io.ipynb for the "
            f"shapes route (boxes, cylinders, polygons, extrusion). Refusing "
            f"to reroute: that would solve a different geometry than the "
            f"spec declares.")
    has_stl = any(e.stl for e in g.electrodes)
    if coords == "xyz" and g.depth_mm > 0.0 and has_stl:
        return BuildRoute("stl3d", 3, coords)
    if coords == "rz":
        return BuildRoute("rz", 2, coords)
    if coords == "xyz" and g.depth_mm == 0.0:
        return BuildRoute("stl2d" if has_stl else "planar", 2, coords)
    if b == "scene3d":
        return BuildRoute("scene3d", 3, coords)
    if has_stl:
        return BuildRoute("stl3d", 3, coords)
    # NATIVE INLINE SHAPES IN 3-D: xyz, depth>0, inline
    # shapes, no STL — the extruded-shape route (build_shapes3d). This
    # combination previously fell through to "unwired" (a
    # NotImplementedError in build_run), so no working configuration
    # changes route. An explicit builder="shapes3d" also lands here.
    if (coords == "xyz" and g.depth_mm > 0.0
            and any(e.shapes for e in g.electrodes)
            and b in ("", "shapes3d")):
        return BuildRoute("shapes3d", 3, coords)
    return BuildRoute("unwired", 3, coords)


def build_needs_solve(spec: SimSpec):
    """True if building this spec will actually SOLVE (cache miss).

    The name is literal: this verifies nothing and builds nothing. It
    PREDICTS whether the expensive Laplace solve is about to run, so a
    caller can warn the user or quote the cost before committing to it.
    """
    # Dispatches on the SAME BuildRoute as build_run (one classification;
    # this function used to duplicate build_run's conditionals inline, which
    # is exactly how two answers to "what will this spec build?" drift apart).
    r = build_route(spec)
    if r.builder in ("stl3d", "scene3d", "shapes3d"):
        # full 3-D STL, native scene3d, and native extruded shapes all
        # solve through build_stl3d's
        # multigrid bases and share fa_cache. Two costs matter here:
        #   (a) the BASES solve (slow, once per geometry) — cached or not;
        #   (b) compose_drive_channels, which runs a full-grid gradient
        #       ONCE PER DRIVE GROUP (+1 for the static field) on EVERY
        #       build, cached bases or not. For a multi-channel drive on a
        #       big grid (e.g. SLIM: ~11 passes over 15M nodes) that is
        #       seconds-to-minutes of work.
        # The UI uses this to decide inline (fast, no spinner) vs off-thread
        # (spinner + live status). Treating "bases cached" as "fast" sent a
        # cached SLIM down the INLINE path, freezing the UI for the whole
        # channel compose with no spinner and no status to show for it.
        # So: slow if the bases miss OR the drive
        # composition will be heavy (more than one group on a large grid).
        try:
            from ion_gym.io import fa_cache
            from ion_gym.physics.build_stl3d import _bases_cache_key
            if r.builder == "scene3d":
                from ion_gym.physics.build_scene3d import _geom_bytes
                key = _bases_cache_key(spec, geom_bytes=_geom_bytes,
                                       tag="scene3d-v1")
            elif r.builder == "shapes3d":
                from ion_gym.physics.build_shapes3d import _geom_bytes
                key = _bases_cache_key(spec, geom_bytes=_geom_bytes,
                                       tag="shapes3d-v1")
            else:
                key = _bases_cache_key(spec)
        except ImportError:
            return True               # cannot know it is cached -> solve
        arrs, _ = fa_cache.load(key)
        if arrs is None:
            return True               # bases miss -> definitely slow
        # bases cached: still off-thread if the channel compose is heavy.
        try:
            from ion_gym.physics.build_stl3d import _grid_from_spec
            nx, ny, nz, _ = _grid_from_spec(spec)
            n_groups = len(getattr(spec.geometry, "rf_groups", []) or [])
            nodes = int(nx) * int(ny) * int(nz)
            # one gradient pass ~ tens of ms per 1e6 nodes; > ~2 gradient
            # passes over a multi-million-node grid is not "instant".
            heavy = n_groups >= 1 and nodes * (n_groups + 1) > 5_000_000
            return heavy
        except (TypeError, ValueError, AttributeError) as e:
            # sizing inputs incomplete -> the safe (off-thread) answer.
            # Narrowed from a blanket Exception, and it REPORTS — a
            # silent conservative answer hid sizing bugs; scheduling
            # may substitute, but never silently.
            print(f"[sim_build] solve-size estimate unavailable "
                  f"({type(e).__name__}: {e}) — scheduling off-thread")
            return True
    if r.builder == "stl2d":
        from ion_gym.physics.build_stl import stl_will_solve
        return stl_will_solve(spec)
    if r.builder == "planar":
        from ion_gym.physics.build_planar import planar_is_cached
        return not planar_is_cached(spec)
    if r.builder == "rz":
        g = spec.geometry
        has_shapes = any(e.shapes for e in g.electrodes)
        has_stl = any(e.stl for e in g.electrodes)
        if has_shapes and not has_stl:
            # native r-z solve — slow only on a cache miss
            from ion_gym.physics.build_rz import rz_is_cached
            return not rz_is_cached(spec)
        # shipped-funnel / STL path reused PRE-SOLVED external bases —
        # always a fast load/re-weight, never a fresh Laplace solve.
        return False
    # anything else (scene3d / unwired): conservatively show the spinner
    return True


def sizing_for(spec: SimSpec, *, pitch=None, min_feature_mm=None,
               field_method=None, channel_dtype=None):
    """Solve-cost estimate for the solve build_run(spec) will ACTUALLY run.

    Dispatches on the SAME BuildRoute as build_run / build_needs_solve (one
    classification; a second copy of the conditionals is how two answers to
    "what will this spec build?" drift apart). The generic spec-domain
    estimator (sizing.propose_sizing_simspec) is correct wherever the solve
    domain IS the spec's own domain -- every native / STL / scene route.
    (One retired device-named builder was the exception; it has been
    removed.)"""
    build_route(spec)
    from ion_gym.physics.sizing import propose_sizing_simspec
    return propose_sizing_simspec(spec, pitch=pitch,
                                  min_feature_mm=min_feature_mm,
                                  field_method=field_method,
                                  channel_dtype=channel_dtype)


def preview_masks3d(spec: SimSpec):
    """Pre-solve, NAMED per-electrode 3-D geometry for the transport-view
    preview — dispatched on the SAME BuildRoute as build_run, so the UI
    never names a builder.  Returns (masks, h_mm) or None when the route has
    no cheap pre-solve 3-D geometry source.  Returns
    (masks, h_mm, origin_mm, mirror) — mirror is the fold declaration
    for half-stored masks.  Adding a route here is how a new 3-D builder
    gets a real transport preview."""
    r = build_route(spec)
    if r.builder in ("stl3d", "scene3d", "shapes3d"):
        # COST GATE: a preview must be cheap. Ask the
        # (multigrid-corrected) sizing authority for the voxel count FIRST
        # — that is pure arithmetic, no voxelization — and skip the
        # preview above PREVIEW_VOXEL_BUDGET rather than rasterize a
        # hundreds-of-millions-of-nodes import for a mere outline
        # (voxelizing at that scale would hang the UI).
        # Over budget returns None: no preview, exactly as before, no
        # regression; the solve itself still runs when the user commits.
        prop = sizing_for(spec)
        if prop.n_voxels > PREVIEW_VOXEL_BUDGET:
            return None
        h = float(spec.geometry.mm_per_gu)
        if r.builder == "stl3d":
            from ion_gym.physics.build_stl3d import stl_masks_3d
            masks = stl_masks_3d(spec)
        elif r.builder == "shapes3d":
            from ion_gym.physics.build_shapes3d import shapes_masks_3d
            masks = shapes_masks_3d(spec)
        else:
            from ion_gym.physics.build_scene3d import scene_masks_3d
            masks = scene_masks_3d(spec)
        # (nx,ny,nz) is already the spec (x,y,z) frame — no transpose. The
        # masks are the STORED (folded) half on declared mirror axes, so
        # declare the fold to the renderer (transient unfold, same
        # 'reflect_lo' the build uses) and place the FULL array's node 0
        # in the CANONICAL frame: mirrored axes read [-H,+H] with the
        # mirror plane at 0 — so the pre-solve
        # preview and the solved view agree, instead of the preview
        # showing half the instrument in the [0,H] frame and snapping on
        # solve (the same pre/post disagreement the r-z preview fix
        # addressed).
        from ion_gym.physics.build_stl3d import _declared_mirror_axes
        _mx = _declared_mirror_axes(spec)
        mirror = {a: "reflect_lo"
                  for a, on in zip("xyz", _mx) if on} or None
        # DECLARED FRAME: the origin was seeded at zero and only moved
        # for MIRRORED axes, so a deck whose domain is declared away
        # from zero -- any signed-frame deck -- previewed a SECOND copy
        # of the ladder shifted by the whole origin, visible as
        # electrodes appearing to extend to infinity in the View tab.
        # geometry.origin_mm is the deck's own statement of where its
        # domain starts; seed from it, then let the mirror rule below
        # override the folded axes (which are half-domains and genuinely
        # start at -(n-1)h). origin_mm is [0, 0] on every legacy deck.
        _dorg = list(getattr(spec.geometry, "origin_mm", None) or ())
        origin = [float(_dorg[i]) if i < len(_dorg) else 0.0
                  for i in range(3)]
        if any(_mx):
            n0 = next(iter(masks.values())).shape
            for i, on in enumerate(_mx):
                if on:
                    origin[i] = -(n0[i] - 1) * h
        return (masks, h, tuple(origin), mirror)
    return None


def has_3d_transport(spec: SimSpec):
    """Does this spec's route display a 2-D cross-section of a genuinely
    3-D transport device?  Lives HERE, beside build_run's own dispatch —
    the one classification authority — so the UI asks a question instead
    of naming a builder. stl3d/scene3d/shapes3d are all 3-D builders whose
    geometry the transport view previews; the
    preview itself is cost-gated in preview_masks3d, so a heavy STL simply
    returns no preview rather than being excluded here by name."""
    return build_route(spec).builder in ("stl3d", "scene3d",
                                         "shapes3d")


def build_run(spec: SimSpec, verbose=False,
              solve_dtype=None, record_budget_gb=None):
    """(model, fly_fn, col_names, births). Dispatches on geometry
    symmetry; only validated paths are wired. verbose=True prints stage
    timings from the STL builder (voxelize / solve / cache hit).

    record_budget_gb : memory the TRAJECTORY RECORD may use, in GB.
        None (default) derives the ceiling from system RAM, the same way
        the app does. The guard lives here, not only in the UI, because
        the UI is not the only caller: a notebook or a script can ask
        for a record larger than the machine and take it down exactly as
        a click could -- the failure that motivated this was 1000 ions x
        400k samples x 12 channels, and nothing about that arithmetic
        needs a browser. Large records WARN; oversized ones RAISE before
        anything is built. To take a deliberately huge record, DECLARE
        it -- record_budget_gb=200 -- so the intent is in the call
        rather than hidden in a flag.
    """
    # DC ladders are DERIVED voltages: resolve them before anything reads
    # electrode.dc. Doing it here (one door, all builders) means a builder
    # cannot forget, and a group member's dc can never be stale.
    spec.resolve_dc_groups()
    # ONE DOOR for every route: a recording that cannot cover the
    # declared flight is refused here, before anything is built, rather
    # than truncating mid-flight on whichever builder happens to run.
    from ion_gym.io.sim_spec import check_recording_capacity
    _rec_err = check_recording_capacity(spec.integration)
    if _rec_err:
        raise ValueError(_rec_err)
    errs = spec.validate()
    if errs:
        raise ValueError("invalid spec: " + "; ".join(errs))
    # advisories are WARNINGS, not refusals: the
    # configuration is legal but ill-advised; state the reason and
    # proceed. Printed unconditionally the FIRST time — an advisory
    # nobody sees is a silent guard — but ONCE per unique text per
    # process (a sweep calls build_run per point, and the
    # identical advisory would print dozens of times; a verbatim repeat
    # carries no information). The suppression is declared on the first
    # print, so nothing is silently withheld; a different pitch, deck,
    # or operating point changes the text and prints anew.
    for _adv in spec.advisories():
        if _adv in _ADVISED:
            continue    # stated verbatim earlier this process, with the
                        # once-per-process policy declared on that print
        _ADVISED.add(_adv)
        print(f"[ADVISORY] {_adv} [printed once per session]")
    # RECORD-VOLUME GUARD, before a single basis is solved. Same estimate
    # and same ceilings the app quotes (physics.sizing.record_volume), so
    # a script and a click refuse at the same place for the same reason.
    _rv = record_volume(spec, budget_gb=record_budget_gb)
    if _rv["tier"] == "refuse":
        raise MemoryError(
            "trajectory record too large: "
            + "; ".join(x.replace("**", "") for x in _rv["lines"])
            + ". Nothing was built. " + " ".join(_rv["levers"][:3])
            + " rec_every is STORAGE ONLY -- tof, fate, impact position "
            "and collision count are identical at any rec_every. To take "
            "this record deliberately, pass "
            "build_run(..., record_budget_gb=<GB>).")
    if _rv["tier"] == "warn":
        _warn = ("[RECORD] " + "; ".join(x.replace("**", "")
                                         for x in _rv["lines"]))
        if _warn not in _ADVISED:
            _ADVISED.add(_warn)
            print(_warn + " [printed once per session]")
    # ONE classification, all consumers: build_run dispatches on the SAME
    # BuildRoute the app is allowed to ask about (build_route above), so the
    # dimensionality the UI reports and the builder actually taken cannot
    # drift apart.  This is dispatch on declared geometry, not a
    # correlate -- coords, depth_mm (the SOLVE domain's z extent), e.stl and
    # spec.builder are all declared fields.
    r = build_route(spec)
    if r.builder == "stl3d":
        # full 3-D STL scene: real 3-D solve + 3-D fly (voxelize, Rung-2)
        from ion_gym.physics.build_stl3d import build_stl3d_run
        return build_stl3d_run(spec, verbose=verbose)
    if r.builder == "rz":
        # 2-D cylindrical (funnel/IMS/round einzel). The funnel example
        # still reuses the shipped field array; a general r-z geometry solves its
        # own bases (the r-z builder rung).
        return _build_cylindrical(spec)
    if r.builder == "stl2d":
        from ion_gym.physics.build_stl import build_stl_run
        # ground_border=None lets build_stl_run auto-decide from the
        # rasterized masks (open geometries like a quadrupole get a
        # grounded frame; walled lenses don't need one).
        return build_stl_run(spec, verbose=verbose, ground_border=None)
    if r.builder == "planar":
        # 2-D Cartesian slice (planar einzel, wedge pusher, PCB elements).
        from ion_gym.physics.build_planar import build_planar_run
        return build_planar_run(spec, solve_dtype=solve_dtype)
    if r.builder == "scene3d":
        # analytic CSG, rasterize3d (NO tessellation)
        from ion_gym.physics.build_scene3d import build_scene3d_run
        return build_scene3d_run(spec, verbose=verbose)
    if r.builder == "shapes3d":
        # native extruded inline shapes: no imported geometry, no STL —
        # cross-sections + extrude descriptors rasterized straight to
        # voxel masks, same solver/cache/flyer as stl3d/scene3d.
        from ion_gym.physics.build_shapes3d import build_shapes3d_run
        return build_shapes3d_run(spec, verbose=verbose)
    raise NotImplementedError(
        f"no builder is wired for this spec (route {r.builder!r}, coords "
        f"{r.coords!r}, depth_mm={spec.geometry.depth_mm:g}, "
        f"electrodes with shapes="
        f"{sum(1 for e in spec.geometry.electrodes if e.shapes)}, with "
        f"stl={sum(1 for e in spec.geometry.electrodes if e.stl)}).")


def _build_cylindrical(spec: SimSpec):
    """Cylindrical (r-z) dispatch. Electrodes with INLINE SHAPES are
    solved natively by the r-z builder (build_rz) — no external solver.

    RETIRED: the STL/shipped-funnel branch that reused pre-solved
    external bases, removed after the native route was shown equivalent
    on the shipped funnel. That funnel is now a declarative JSON on this native
    route. r-z electrodes referencing STL files REFUSE below — the
    STL->r-z rasterizer rung never landed, and silently routing them
    anywhere else would fly the wrong geometry."""
    has_shapes = any(e.shapes for e in spec.geometry.electrodes)
    has_stl = any(e.stl for e in spec.geometry.electrodes)
    if has_shapes and not has_stl:
        from ion_gym.physics.build_rz import build_rz_run
        return build_rz_run(spec)
    stl_names = [e.name for e in spec.geometry.electrodes if e.stl]
    raise ValueError(
        f"r-z build: electrodes reference STL files ({stl_names}) but the "
        f"STL->r-z rasterizer is not implemented, and the retired "
        f"pre-solved-bases route (ledger #2, 2026-08-06) no longer exists. "
        f"Author the electrodes as inline shapes in the (z, r) plane — see "
        f"examples/ion_funnel_rz.json.")
