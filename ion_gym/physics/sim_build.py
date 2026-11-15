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
    KE, mass mix, tob spread. Geometry-agnostic.
  * FIELD assembly + fly: the CYLINDRICAL path (funnel/IMS) is wired and
    validated (reused the F-1 reference kernel via a voltage re-weighting;
    that kernel has since been removed along with its gates).
    PLANAR-2D and full-3D builders are stubbed with clear NotImplemented
    messages pointing at the validated pieces they'll compose (the 2-D
    aware-field tracer, the native solver3d + mesh legs). This
    keeps the app honest: it runs what's proven and refuses what isn't,
    rather than silently producing an unvalidated field.

The voltage-adjustment invariant the app relies on: bases are solved once
per geometry; changing dc/rf per electrode only changes how A and the B_k
are summed. So "adjust voltages and redraw fields" is a re-weight, not a
re-solve — instant, and exact (the fast-adjust property banked since
Rung 1).
"""

# The public names (docs/PUBLIC_API.md); everything else is internal.
__all__ = ["build_run", "preflight", "build_route", "BuildRoute",
           "build_needs_solve", "generate_births", "mz_of", "sizing_for"]

import math

import numpy as np

from ion_gym.advisories import advise
from ion_gym.io.sim_spec import SimSpec, set_pitch
from ion_gym.physics.sizing import record_volume
from ion_gym.physics.collision3d import E_CHG, KG_AMU, KB

# Preview cost gate: the transport-view geometry
# preview voxelizes a 3-D builder's electrodes BEFORE any solve. Above
# this node count the rasterization is too slow to run on a UI refresh,
# so preview_masks3d returns no preview instead (the solve still runs on
# commit). ~30 M nodes rasterizes in a couple of seconds; a 274 M-node
# import (the case that produced the old 48 h solve estimate) is well
# above it and is correctly skipped rather than hanging the UI.
PREVIEW_VOXEL_BUDGET = 30_000_000

# Preview WORK budget for the shape rasterizer, in raster2d work units
# (grid nodes x whole-grid passes per shape; a polygon costs two passes
# per edge). ~10 s of preview is acceptable. Measured
# on the build container: the Orbitrap at 0.02 mm/gu is 1.4e9 units and
# took 10.4 s (~7.5e-9 s per unit), so 1e9 units is ~7.5 s there; other
# machines differ by their numpy speed. Node count alone missed the
# worst case: the Orbitrap has two
# electrodes but 965 polygon vertices, and at 0.01 mm/gu its 2.9M-node
# preview took 62 s. Over budget, the preview is DRAWN at a coarser pitch
# and says so (preview_pitch); the solve pitch is untouched.
PREVIEW_WORK_BUDGET = 1.0e9
# preview_pitch search: the first guess scales the pitch by the budget
# ratio; if set_pitch's lattice re-derivation leaves it still over, the
# pitch grows by this factor, at most this many times, then refuses.
PREVIEW_PITCH_STEP = 1.1
PREVIEW_PITCH_TRIES = 40

# Gaussian FWHM = FWHM_PER_SIGMA * sigma.  Named because every beam spread
# in this project is quoted as FWHM while every generator wants sigma, and
# a bare 2.3548 in a function body is the kind of literal that drifts.
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

# Bound on the truncated-Gaussian birth draw. gaussian_layout() keeps the
# truncation at >= 1 sigma (>= 68% acceptance per draw), so this many
# misses in a row is a broken layout, refused rather than looped on.
GAUSSIAN_BIRTH_TRIES = 1000


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
        # A relative births_file belongs to the DECK it was loaded from,
        # as STL paths do (stl_resolve). It used to be looked up in the
        # working folder FIRST, so a same-named file there silently
        # replaced the deck's ions (review R-07). A spec built in code (no
        # deck) still reads relative to the working folder.
        base = getattr(spec, "_loaded_from", None)
        if bf and not os.path.isabs(bf) and base:
            cand = os.path.join(os.path.dirname(os.path.abspath(base)), bf)
            if not os.path.exists(cand):
                raise FileNotFoundError(
                    f"births_file {bf!r} is relative, so it is read beside "
                    f"the deck: {cand} does not exist. Put the file next to "
                    f"the deck or give an absolute path (the working folder "
                    f"is not searched).")
            bf = cand
        s = type(s).from_dict({**s.to_dict(), "births_file": bf})
        import csv
        REQUIRED = ("x", "y", "z", "vx", "vy", "vz")   # tob defaults to 0
        rows = []
        with open(s.births_file, encoding="utf-8") as fh:
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
                f"masses = {want}. Masses are assigned by contiguous block "
                f"(ion i -> mz_list[i//n_ions]), so a mismatch mislabels "
                f"every ion past the first block. Set n_ions to match the "
                f"file, or regenerate the file.")
        return out

    _zd = s.zero_direction_note()
    if _zd:
        # the same text as the advisory, so a caller of this generator who
        # never sees advisories still hears it
        import warnings
        warnings.warn(_zd, stacklevel=2)

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
            pass                    # born AT the centre: p0 is the position
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
                # bounded: at >= 68% acceptance, running out of tries
                # means gaussian_layout let through what it should refuse
                for _try in range(GAUSSIAN_BIRTH_TRIES):
                    dpos = rng.normal() * sig_g[k]
                    if abs(dpos) <= trc_g[k]:
                        break
                else:
                    raise RuntimeError(
                        f"generate_births: no draw of the truncated "
                        f"Gaussian on axis {'xyz'[k]} (sigma "
                        f"{sig_g[k]:g} mm, truncation {trc_g[k]:g} mm) "
                        f"fell inside the truncation in "
                        f"{GAUSSIAN_BIRTH_TRIES} tries; gaussian_layout "
                        f"should have refused this beam")
                pos[k] += dpos
        else:
            # 'file' returned above (births_file); anything else is not a
            # distribution this generator knows. It used to fall through
            # and be born as a POINT source at the centre.
            from ion_gym.io.sim_spec import SOURCE_DISTRIBUTIONS
            raise ValueError(
                f"generate_births: source.distribution "
                f"{s.distribution!r} is not one this generator places "
                f"(known: {list(SOURCE_DISTRIBUTIONS)}; 'file' needs "
                f"source.births_file)")
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
            # A zero direction leaves the directed term at zero: the energy
            # is IGNORED, announced once per generation above. The energy is
            # still DRAWN, so the random stream -- and every later birth --
            # is what every earlier flight of such a deck used.
            if _n > 0.0:
                v = v + sp * (d / _n)
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
        # dc_effective, not e.dc: add-mode (float) DC groups contribute
        # here; set-mode ladders were already resolved into e.dc
        A = A + spec.dc_effective(e) * fa
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

# build_needs_solve: a cached 3-D build is still scheduled OFF the UI thread
# when its drive compose (one full-grid gradient pass per drive group, plus
# the static field) exceeds this many node-passes -- about two passes over a
# multi-million-node grid, i.e. no longer "instant" (was a literal).
COMPOSE_HEAVY_NODE_PASSES = 5_000_000


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


SOLID_3D_BUILDERS = ("stl3d", "scene3d", "shapes3d")


def solved_3d_cache_key(spec: SimSpec):
    """The disk-cache key dict under which a full 3-D route (STL, native
    scene3d or extruded shapes) keeps its solved bases, or None for the
    2-D routes (planar and r-z keep theirs in io.basis_cache). The same
    key the builder uses, so it covers STL content and solver code."""
    r = build_route(spec)
    if r.builder not in SOLID_3D_BUILDERS:
        return None
    from ion_gym.physics.build_stl3d import _bases_cache_key
    if r.builder == "scene3d":
        from ion_gym.physics.build_scene3d import _geom_bytes
        return _bases_cache_key(spec, geom_bytes=_geom_bytes,
                                tag="scene3d-v1")
    if r.builder == "shapes3d":
        from ion_gym.physics.build_shapes3d import _geom_bytes
        return _bases_cache_key(spec, geom_bytes=_geom_bytes,
                                tag="shapes3d-v1")
    return _bases_cache_key(spec)


def solved_field_key(spec: SimSpec):
    """The disk-cache key (hex) under which `spec`'s solved field is kept,
    on any route: a full 3-D route's own key (solved_3d_cache_key), else
    the planar/r-z key (io.basis_cache). It covers the geometry, the STL
    contents and the solver code, so it is what a saved field is matched
    by (io.field_io)."""
    from ion_gym.io import basis_cache, fa_cache
    kd = solved_3d_cache_key(spec)
    return basis_cache.key(spec) if kd is None else fa_cache.spec_key(kd)


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
    if r.builder in SOLID_3D_BUILDERS:
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
        # These are ion_gym's own modules: imported, never guarded (an
        # `except ImportError: return True` here answered a broken install
        # with a guess).
        from ion_gym.io import fa_cache
        arrs, _ = fa_cache.load(solved_3d_cache_key(spec))
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
            heavy = (n_groups >= 1
                     and nodes * (n_groups + 1) > COMPOSE_HEAVY_NODE_PASSES)
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


def preview_cost(spec: SimSpec):
    """What the pre-solve electrode preview of `spec` costs at its own
    pitch: {"work": shape-raster work units (raster2d.raster_work),
    "voxels": 3-D voxel count or None for a 2-D route}.

    Dispatched on the SAME BuildRoute as build_run. Planar and r-z routes
    rasterize their shapes on the 2-D grid; shapes3d rasterizes each
    shape on the x-y cross-section and extrudes it, so its work is the
    cross-section raster, with the voxel count gated separately; the
    STL and scene routes
    voxelize meshes or analytic solids, gated by voxel count alone. A
    route without a cheap preview source costs nothing here."""
    from ion_gym.io.lattice import gu_nodes
    from ion_gym.physics.raster2d import raster_work
    g = spec.geometry
    h = float(g.mm_per_gu)
    r = build_route(spec)
    if r.builder in ("planar", "rz", "shapes3d"):
        nodes = (gu_nodes(g.width_mm, h, axis="x",
                          what="width_mm domain extent")
                 * gu_nodes(g.height_mm, h, axis="y",
                            what="height_mm domain extent"))
        work = raster_work(g.electrodes, nodes)
        # the extrusion into 3-D is a broadcast, priced by the voxel
        # budget (measured: SLIM 3D at 17.6M voxels in 2.5 s), not by
        # shape count
        voxels = (int(sizing_for(spec).n_voxels)
                  if r.builder == "shapes3d" else None)
        return {"work": int(work), "voxels": voxels}
    if r.builder in ("stl3d", "scene3d"):
        return {"work": 0, "voxels": int(sizing_for(spec).n_voxels)}
    return {"work": 0, "voxels": None}


def _preview_over(cost):
    """(work ratio, voxel ratio) against the two preview budgets."""
    rw = cost["work"] / PREVIEW_WORK_BUDGET
    rv = ((cost["voxels"] / PREVIEW_VOXEL_BUDGET)
          if cost["voxels"] is not None else 0.0)
    return rw, rv


def preview_pitch(spec: SimSpec):
    """(pitch_mm, note) for the electrode preview of `spec`.

    Within both preview budgets: (spec's own pitch, None), and the
    preview is the solver's raster. Over either: the coarsest-needed
    pitch that fits, and a note saying the preview is drawn at that
    pitch while the solve keeps its own (draw coarser and label it,
    rather than skip or block). Work scales as
    pitch^-2 (the 2-D raster) and voxels as pitch^-3, so the first guess
    is exact up to set_pitch's lattice adjustment; the search then grows
    the pitch by PREVIEW_PITCH_STEP until it fits, and refuses with the
    numbers if PREVIEW_PITCH_TRIES steps do not."""
    import copy
    h = float(spec.geometry.mm_per_gu)
    cost = preview_cost(spec)
    rw, rv = _preview_over(cost)
    if rw <= 1.0 and rv <= 1.0:
        return h, None
    hp = h * max(rw ** 0.5, rv ** (1.0 / 3.0), 1.0)
    hp = float(f"{hp:.3g}")
    for _ in range(PREVIEW_PITCH_TRIES):
        trial = copy.deepcopy(spec)
        set_pitch(trial, hp)
        rw2, rv2 = _preview_over(preview_cost(trial))
        if rw2 <= 1.0 and rv2 <= 1.0:
            over = []
            if rw > 1.0:
                over.append(f"{rw:.1f}x the shape-raster budget "
                            f"({cost['work']:.2e} of "
                            f"{PREVIEW_WORK_BUDGET:.0e} units)")
            if rv > 1.0:
                over.append(f"{rv:.1f}x the voxel budget "
                            f"({cost['voxels'] / 1e6:.1f}M of "
                            f"{PREVIEW_VOXEL_BUDGET / 1e6:.0f}M)")
            note = (f"electrode preview drawn at {hp:g} mm/gu; the solve "
                    f"uses {h:g} mm/gu. At the solve pitch the preview "
                    f"would be " + " and ".join(over) + ".")
            return hp, note
        hp = float(f"{hp * PREVIEW_PITCH_STEP:.3g}")
    raise ValueError(
        f"preview_pitch: no preview pitch up to {hp:g} mm/gu fits the "
        f"preview budgets for {spec.name!r} (solve pitch {h:g} mm/gu, "
        f"work {cost['work']:.2e} units, voxels {cost['voxels']})")


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


def preflight(spec: SimSpec):
    """The refusals build_run makes before it builds anything, after
    resolving the derived values it resolves. Returns (recording_error or
    None, [validate() refusals]); both empty means build_run proceeds to
    build. `ion-gym validate` reads this, so a deck it passes is one
    build_run accepts. Refusals that need the built geometry (a birth in
    metal, an electrode with no nodes) are found only by building."""
    # DC ladders are DERIVED voltages: resolve them before anything reads
    # electrode.dc. Doing it here (one door, all builders) means a builder
    # cannot forget, and a group member's dc can never be stale.
    spec.resolve_dc_groups()
    # An unset step (dt_ns None = derive from the drives) is resolved
    # HERE, the same one-door way. It used to be a side effect of
    # spec.advisories(), which must only report.
    if spec.integration.dt_ns is None:
        spec.integration.set_derived_dt(spec.derive_dt_ns())
    # ONE DOOR for every route: a recording that cannot cover the
    # declared flight is refused here, before anything is built, rather
    # than truncating mid-flight on whichever builder happens to run.
    from ion_gym.io.sim_spec import check_recording_capacity
    return (check_recording_capacity(spec.integration) or None,
            list(spec.validate()))


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

    WHAT IT WRITES ON `spec`. The build resolves derived values onto the
    spec it is given, and callers read them there afterwards:
      * DC-ladder members' dc: the voltages their group derives
        (resolve_dc_groups); a member's dc is never an input.
      * integration.dt_ns, when null: the step derived from the drives
        (a save still writes null).
      * integration.record_channels: e_x/e_y/e_z when a non-r-z deck
        records e_axial/e_radial (a save writes the channels the deck
        asked for).
      * source._drawn_seed, in random mode: the seed this build drew
        (in memory only; io.run_provenance records it with a run).
    Nothing else the deck says is changed; tests/test_build_run_writes.py
    holds that over the shipped decks.
    """
    # RANDOM SEED, ONE DRAW PER BUILD: a random-mode spec (source.seed
    # null) draws its base once in resolve_run_seed and caches it on the
    # spec, which births and every per-ion kernel seed then share. Clearing
    # it here makes each build a fresh run, so building the same spec
    # object twice no longer repeats the first run (pooled runs were
    # duplicates); within a build, everything shares the one printed seed.
    if spec.source.seed is None and hasattr(spec.source, "_drawn_seed"):
        del spec.source._drawn_seed
    _rec_err, errs = preflight(spec)
    if _rec_err:
        raise ValueError(_rec_err)
    # SDS outside its regime WARNS rather than refuses: the note is
    # one of spec.advisories() below, so this print and the app's status
    # line carry the same text from one computation per spec.
    if errs:
        raise ValueError("invalid spec: " + "; ".join(errs))
    # RADIAL CHANNELS are DERIVED AFTER FLIGHT about the deck's axis
    # (no kernel computes them; validate() above
    # refused a missing axis). r-z keeps its kernel's values: its axis is
    # built in and they already are about it. Elsewhere the field channels
    # need the recorded field components, which are added here, announced.
    from ion_gym.io.sim_spec import RADIAL_CHANNELS
    _rc = list(spec.integration.record_channels or [])
    _derive_radial = (spec.geometry.symmetry.coords != "rz"
                      and any(c in RADIAL_CHANNELS for c in _rc))
    if _derive_radial and any(c in ("e_axial", "e_radial") for c in _rc):
        # marked as added by the build: recorded, but not written back
        # into the deck by a save
        _add = spec.integration.add_derived_channels(("e_x", "e_y", "e_z"))
        if _add:
            advise(f"[RECORD] e_axial/e_radial are derived after flight "
                   f"from the field components: also recording {_add}",
                   stacklevel=2)
    # advisories are WARNINGS, not refusals: the configuration is legal
    # but ill-advised; state the reason and proceed. Raised as
    # IonGymAdvisory against the caller's line, so Python's warning
    # filter governs repeats (by default each text once per calling line:
    # a sweep calling build_run per point states it once) and a caller
    # can collect, silence or escalate them. SimSpec.advisories() keeps
    # the full list, undeduplicated.
    for _adv in spec.advisories():
        advise(f"[ADVISORY] {_adv}", stacklevel=2)
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
        advise("[RECORD] " + "; ".join(x.replace("**", "")
                                       for x in _rv["lines"]),
               stacklevel=2)
    # ONE classification, all consumers: build_run dispatches on the SAME
    # BuildRoute the app is allowed to ask about (build_route above), so the
    # dimensionality the UI reports and the builder actually taken cannot
    # drift apart.  This is dispatch on declared geometry, not a
    # correlate -- coords, depth_mm (the SOLVE domain's z extent), e.stl and
    # spec.builder are all declared fields.
    out = _dispatch_route(spec, verbose=verbose, solve_dtype=solve_dtype)
    # TUNNELLING needs the built metal and births, so it is checked here,
    # for every route, rather than among the spec's own advisories. The
    # note rides on the model (the app lists it with the advisories).
    from ion_gym.physics.tunnelling import tunnelling_note
    _tn = tunnelling_note(spec, out[0], out[3])
    out[0].build_advisories = [_tn] if _tn else []
    if _tn:
        advise(f"[ADVISORY] {_tn}", stacklevel=2)
    if not _derive_radial:
        return out
    model, fly, cols, births = out
    frame = spec.radial_frame()
    from ion_gym.io.records import apply_radial_frame

    def fly_radial(i, *a, **k):
        traj, summary = fly(i, *a, **k)
        if traj is not None:
            apply_radial_frame(traj, cols, frame)
        return traj, summary
    # keep whatever the route's fly function carries (attributes the
    # ensemble driver or the app read)
    fly_radial.__dict__.update(getattr(fly, "__dict__", {}))
    return model, fly_radial, cols, births


def _dispatch_route(spec, *, verbose, solve_dtype):
    """The route builder build_run dispatches to (split out so build_run
    can post-process the flight records for every route in one place)."""
    r = build_route(spec)
    if r.builder == "stl3d":
        # full 3-D STL scene: real 3-D solve + 3-D fly (voxelize, Rung-2)
        from ion_gym.physics.build_stl3d import build_stl3d_run
        return build_stl3d_run(spec, verbose=verbose)
    if r.builder == "rz":
        # 2-D cylindrical (funnel/IMS/round einzel). The funnel example
        # still reuses the shipped field array; a general r-z geometry solves its
        # own bases (the r-z builder rung).
        return _build_cylindrical(spec, verbose=verbose)
    if r.builder == "stl2d":
        from ion_gym.physics.build_stl import build_stl_run
        # ground_border=None lets build_stl_run auto-decide from the
        # rasterized masks (open geometries like a quadrupole get a
        # grounded frame; walled lenses don't need one).
        return build_stl_run(spec, verbose=verbose, ground_border=None)
    if r.builder == "planar":
        # 2-D Cartesian slice (planar einzel, wedge pusher, PCB elements).
        from ion_gym.physics.build_planar import build_planar_run
        return build_planar_run(spec, verbose=verbose,
                                solve_dtype=solve_dtype)
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


def _build_cylindrical(spec: SimSpec, verbose=False):
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
        return build_rz_run(spec, verbose=verbose)
    stl_names = [e.name for e in spec.geometry.electrodes if e.stl]
    raise ValueError(
        f"r-z build: electrodes reference STL files ({stl_names}) but the "
        f"STL->r-z rasterizer is not implemented, and the retired "
        f"pre-solved-bases route no longer exists. "
        f"Author the electrodes as inline shapes in the (z, r) plane — see "
        f"examples/ion_funnel_rz.json.")
