"""
ion_gym.build_stl
-----------------
The STL -> electrode path. An electrode can carry an `stl` reference
(a file in the spec's stl_dir) instead of inline shapes; this module
turns those STLs into the same electrode MASKS the native solvers
consume, so STL and inline-shape electrodes are interchangeable.

Two regimes, matching the builder taxonomy:
  * 2-D (planar coords='xyz' depth=0, or cylindrical 'rz'): voxelize the
    STL on a thin grid and SLICE at the symmetry plane to get the 2-D
    cross-section mask. This lets a CAD electrode be prototyped in a fast
    2-D study (the quad cross-section, a shaped einzel aperture).
  * 3-D (coords='xyz' depth>0): full voxelization -> 3-D masks for the
    (future) 3-D solver.

Uses the Rung-2-validated voxelizer (voxelize.py, matched to reference STL
import at +-5.5 um). The slice plane is chosen from the geometry: for a
planar x-y slice we cut the voxel grid at its mid-z; for r-z we cut at
mid of the mirror axis. Because the 2-D solve is uniform along the cut
direction, one representative slice is exact for an extruded/axisymmetric
electrode and a faithful approximation for a gently varying one.

The STL EXAMPLE instruments (four-rod quadrupole, three-plate einzel)
that used to live here were device generators, not route machinery. They
were retired on the 2026-08-18 pattern (generate once, ship the data,
delete the Python): their meshes and decks are test fixtures under
tests/fixtures/stl_quad and tests/fixtures/stl_einzel3d (D6b/D6c,
2026-10-16).
"""

import numpy as np


def _require_trimesh():
    """Import trimesh with a clear, actionable message. Kept under this
    name because notebook 02 imports it; the ONE definition is
    io.stl_resolve._require_trimesh (L-546)."""
    from ion_gym.io.stl_resolve import _require_trimesh as _one
    return _one()


from ion_gym.io.sim_spec import SimSpec


def unsupported_drive_features(spec):
    """Declared drive features the 2-D STL route does NOT apply: NONE,
    as of 2026-09-16 (L-455). The single signed-B fold (every drive as
    sin at the lowest frequency) is superseded: this route assembles the
    same per-group channels as the planar build
    (build_planar.assemble_drive_groups) and flies the same kernel, so
    waveforms, phases, frequencies, offsets, duty and multi-group
    membership are honoured identically. Kept as the route's contract
    statement for the field export door."""
    return []




def _grid_counts_2d(g):
    """(nx, ny) of the stl2d voxel grid, through THE counting
    function. The stl2d route counts CELLS (its historical
    convention — round(mm/h) with no +1, a voxel-count grid rather than
    the node-centred inclusive span the planar/r-z/3-D routes use); for
    a conforming extent round == cells so every conforming deck is
    byte-identical, and a non-conforming extent is refused with the two
    nearest conforming extents instead of being silently rounded."""
    from ion_gym.io.lattice import gu_cells
    h = g.mm_per_gu
    return (gu_cells(g.width_mm, h, axis="x",
                     what="width_mm domain extent"),
            gu_cells(g.height_mm, h, axis="y",
                     what="height_mm domain extent"))


# --------------------------------------------------- STL -> 2-D masks
def stl_masks_2d(spec: SimSpec, verbose=False):
    """Voxelize the spec's STL electrodes and slice to 2-D masks on the
    solver grid. Returns {electrode_index: 2-D bool mask (nx, ny)}.
    Requires every electrode to have an `stl` (this is the STL path)."""
    _require_trimesh()      # availability probe; module unused here
    from ion_gym.physics.voxelize import voxelize_meshes

    g = spec.geometry
    h = g.mm_per_gu
    nx, ny = _grid_counts_2d(g)
    # a thin slab in the cut direction: enough voxels to capture the
    # cross-section robustly, sliced at the middle.
    nz = 5
    from ion_gym.io.stl_resolve import require_stls, load_mesh
    stl_dir = require_stls(spec)      # preflight: refuse-with-diagnostic

    # GridSpec IS the named owner of (nx, ny, nz, mm_per_gu); the ad-hoc
    # _Grid attribute bag this replaced was copy-pasted between the two
    # STL builders.
    from ion_gym.physics.scene3d import GridSpec
    grid = GridSpec(nx=nx, ny=ny, nz=nz, mm_per_gu=h)

    meshes = {}
    for idx, el in enumerate(g.electrodes, start=1):
        if not el.stl:
            raise ValueError(f"stl_masks_2d: electrode {el.name!r} has no "
                             f"stl reference")
        # SINGLE MESH-INGEST POINT: load_mesh applies the deck's
        # declared frame_offset_mm; this builder only scales mm -> gu.
        m = load_mesh(spec, el, stl_dir=stl_dir)
        m.apply_scale(1.0 / h)
        # centre the mesh in the thin z-slab so the mid-slice cuts it
        m.apply_translation([0, 0, nz / 2 - m.centroid[2]])
        meshes[idx] = m
        if verbose:
            print(f"  loaded {el.stl} -> {len(m.faces)} faces")

    lab = voxelize_meshes(meshes, grid)
    kmid = nz // 2
    masks = {}
    for idx in range(1, len(g.electrodes) + 1):
        masks[idx] = (lab[:, :, kmid] == idx)
    return masks


_STL_BUILD_CACHE = {}


def clear_memory_cache():
    """Drop the in-process STL basis cache (_STL_BUILD_CACHE). The disk
    cache is separate (fa_cache.clear_all). Returns the number of
    entries dropped — a visible, reported clear, not a silent wipe.

    THIS DID NOT EXIST while build_planar and build_rz both had one, so
    the app's "clear cache" button purged two of the three unbounded
    basis caches and left this one holding every STL geometry ever built
    in the session. Each entry is a full per-electrode basis set, so on
    a sweep across geometries it is the largest thing in the process.
    """
    n = len(_STL_BUILD_CACHE)
    _STL_BUILD_CACHE.clear()
    return n


def _stl_geom_signature(spec, ground_border):
    """Cache key capturing everything voxelize+solve depends on: the STL
    files + mtimes, grid, grounding, and the declared CAD->build offset.
    Voltages/RF are NOT in the key — they re-weight cheaply from cached
    bases (fast-adjust invariant).

    The STL folder is stl_resolve.resolve_stl_dir, THE folder the meshes
    are loaded from (L-557): this used Path(stl_dir) or '.', against the
    CWD, while loading resolves a relative stl_dir against the spec's own
    location -- so the key could stat different files, or none, and a
    missing file keyed as mtime -1.0, which never changes, so an edited
    STL did not invalidate the cached bases. A file that cannot be
    stat'ed now refuses (the loader's preflight has already required it).
    frame_offset_mm moves metal (it is geometry, keyed on disk by
    basis_cache) and is keyed here too."""
    from ion_gym.io.stl_resolve import resolve_stl_dir
    g = spec.geometry
    sig = [round(g.width_mm, 6), round(g.height_mm, 6),
           round(g.mm_per_gu, 6), bool(ground_border),
           tuple(round(float(v), 9) for v in (g.frame_offset_mm or ()))]
    p = resolve_stl_dir(spec)
    for el in g.electrodes:
        if el.stl:
            f = p / el.stl
            try:
                mt = f.stat().st_mtime
            except OSError as e:
                raise FileNotFoundError(
                    f"stl2d cache key: cannot stat electrode {el.name!r} "
                    f"STL {str(f)!r} ({e}) -- the key must name the file "
                    f"the build loads") from e
            sig.append((el.name, el.stl, round(mt, 3)))
    return tuple(sig)


def _stl_bases(spec, ground_border, verbose):
    """Voxelize + solve the per-electrode bases — the EXPENSIVE part (SOR
    on the grid). Cached on geometry so repeated builds (voltage tweaks,
    view switches, fly) reuse it instead of re-solving. Returns
    (bases, masks2d, ele, bands, nx, ny, ground_border)."""
    import time
    # F401-cleanup fallout fix: build_planar merely
    # RE-EXPORTED solve_bases; the cleanup removed the locally-unused
    # import and broke this hidden indirection. Import from the OWNER.
    from ion_gym.physics.solver3d import solve_bases

    g = spec.geometry
    h = g.mm_per_gu
    nx, ny = _grid_counts_2d(g)

    t0 = time.time()
    masks2d = stl_masks_2d(spec, verbose=verbose)
    if ground_border is None:
        spany = any((np.where(m.any(0))[0].min() <= 0.02 * ny and
                     np.where(m.any(0))[0].max() >= 0.98 * ny)
                    for m in masks2d.values() if m.any())
        ground_border = not spany
        if verbose:
            print(f"[stl] auto ground_border={ground_border} "
                  f"({'open' if ground_border else 'enclosed'})")
    t_vox = time.time() - t0

    key = _stl_geom_signature(spec, ground_border)
    cached = _STL_BUILD_CACHE.get(key)
    if cached is not None:
        if verbose:
            print(f"[stl] CACHE HIT ({nx}x{ny}, voxelize {t_vox:.2f}s, "
                  f"solve skipped)")
        bases, ele, bands = cached
        return bases, masks2d, ele, bands, nx, ny, ground_border

    # solver3d is correct at nz=1 (size-1 z-axis == exact 2-D Laplace;
    # test_solver3d_nz1.py), so a single z-plane is right and fastest.
    masks = {i: m[:, :, None] for i, m in masks2d.items()}
    if ground_border:
        frame = np.zeros((nx, ny), bool)
        frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
        for m in masks2d.values():
            frame &= ~m
        masks[10_000] = frame[:, :, None]
    t0 = time.time()
    # ImportError ONLY (the same fix build_planar got -- this
    # sibling was missed).  The old `except Exception` around import+solve
    # ALSO caught multigrid THROWING and silently recomputed the field
    # with a DIFFERENT SOLVER (SOR) -- the exact silent-substitution the
    # except-policy gate documents.  "Falls back if multigrid is
    # UNAVAILABLE" is an ImportError; anything else is a solver bug and
    # must surface.  Announced unconditionally (a fallback is
    # never under `verbose`).
    # the STL routes' ONE solver setting (L-552: 1e-4 and 1.9 were literals
    # here, a copy of the values build_stl3d names)
    from ion_gym.physics.build_stl3d import STL3D_SOLVE_TOL, STL3D_SOR_OMEGA
    try:
        from ion_gym.physics.multigrid3d import solve_bases_mg
    except ImportError as e:
        print(f"[stl] multigrid unavailable ({e}); solving by SOR")
        bases3 = solve_bases(masks, mirror=(False, False, False),
                             tol=STL3D_SOLVE_TOL, stencil="ghost_linear",
                             verbose=verbose, omega=STL3D_SOR_OMEGA)
    else:
        bases3 = solve_bases_mg(masks, mirror=(False, False, False),
                                tol=STL3D_SOLVE_TOL, stencil="ghost_linear",
                                verbose=verbose)
    bases = {i: b[:, :, 0] for i, b in bases3.items()}
    t_solve = time.time() - t0

    # ele carries the electrode INDEX per voxel (0 = vacuum), not a
    # boolean — same convention and same fix as build_stl3d (which got
    # this when the int16-labels doctrine landed; this 2-D path was
    # missed, so every 2-D STL model draped at electrode #1's voltage
    # in any per-electrode consumer). Boolean consumers are unaffected:
    # they all test `ele > 0.5` / `.any()`. Found by the Phase B A-2
    # sweep via scene_from_simspec's refuse-with-diagnostic.
    ele = np.zeros((nx, ny), np.int16)
    for i in sorted(masks2d):
        ele[masks2d[i]] = i
    bands = []
    for idx in range(1, len(g.electrodes) + 1):
        m = masks2d.get(idx)
        if m is None or not m.any():
            bands.append(None)
            continue
        xi, yi = np.where(m)
        bands.append((xi.min() * h, xi.max() * h,
                      yi.min() * h, yi.max() * h))
    if verbose:
        print(f"[stl] solved {len(bases)} bases on {nx}x{ny}: voxelize "
              f"{t_vox:.2f}s + solve {t_solve:.2f}s (cached for reuse)")
    _STL_BUILD_CACHE[key] = (bases, ele, bands)
    return bases, masks2d, ele, bands, nx, ny, ground_border


def stl_will_solve(spec, ground_border=None):
    """True if build_stl_run would SOLVE (cache miss) vs re-weight cached
    bases. Cheap: voxelizes (fast) to resolve the geometry signature, then
    checks the cache. Used by the UI to decide whether to show a spinner."""
    g = spec.geometry
    _, ny = _grid_counts_2d(g)
    # Narrow + REPORT.  `except Exception: return True` was a
    # silent guess -- the `_will_solve` archetype the except-policy gate
    # documents.  The conservative answer stands for the EXPECTED probe
    # failures -- a missing/unreadable STL (OSError) or an unparseable
    # mesh (ValueError) -- but it is ANNOUNCED, and anything else
    # propagates: the same failure surfaces loudly in the build itself.
    try:
        masks2d = stl_masks_2d(spec)
    except (OSError, ValueError) as e:
        print(f"[stl] will-solve probe failed ({e!r}); assuming a fresh "
              f"solve")
        return True
    if ground_border is None:
        spany = any((np.where(m.any(0))[0].min() <= 0.02 * ny and
                     np.where(m.any(0))[0].max() >= 0.98 * ny)
                    for m in masks2d.values() if m.any())
        ground_border = not spany
    return _stl_geom_signature(spec, ground_border) not in _STL_BUILD_CACHE


def build_stl_run(spec: SimSpec, verbose=False, ground_border=False):
    """SimSpec with STL electrodes (2-D) -> (model, fly_fn, cols, births)
    via the planar builder, using STL-sliced masks instead of inline
    shapes.

    ground_border: for OPEN geometries (a quadrupole's four rods don't
    enclose the domain), the free-Neumann edges leave the field floating
    (one basis fills the domain). A grounded border frame held at 0 gives
    the solve a reference. Enclosed geometries (einzel plates spanning the
    domain) don't need it."""
    from ion_gym.physics.build_planar import (PlanarModel, _grad2d, make_planar_fly_fn)
    from ion_gym.physics.sim_build import generate_births
    import time

    g = spec.geometry
    h = g.mm_per_gu
    nx, ny = _grid_counts_2d(g)

    bases, masks2d, ele0, bands, nx, ny, ground_border = _stl_bases(
        spec, ground_border, verbose)
    _t_assemble = time.time()

    # assemble A (DC + drive offsets) + per-GROUP drive potentials through
    # the ONE shared 2-D assembly (build_planar.assemble_drive_groups):
    # every waveform, phase, frequency, offset, duty and multi-group
    # membership the planar kernel honours is honoured here identically.
    # SUPERSEDED in this change (L-455): the single signed-B fold that
    # flew every drive as sin at the lowest frequency, and its >2-phase
    # warning — the general channel kernel flies the drives as declared.
    from ion_gym.physics.build_planar import assemble_drive_groups
    A, drives = assemble_drive_groups(spec, bases, v_basis=1e4)
    # back-compat Bk: the sin drives as (B, freq_hz, phase_deg) tuples
    Bk = [(B, gg.frequency_hz, gg.phase_deg) for B, gg in drives
          if gg.waveform == "sin"]
    Ex, Ey = _grad2d(np.ascontiguousarray(A), h)
    model = PlanarModel(A, Bk, ele0, Ex, Ey, h, spec, el_bands=bands,
                        drives=drives)
    # Axial extent comes from the SPEC DECLARATION only (an earlier
    # version measured the mesh z-bounds, and the app
    # quad's rods are 1 mm cross-section TOKENS — it drew 1 mm rods.
    # Meshes may be measured at SPEC-CREATION time (spec_from_upload
    # writes what it measured into the JSON, visible and editable);
    # at draw time the JSON is the only authority.)
    ax = spec.geometry.axial_extent_mm
    model.z_extent_mm = (float(ax[0]), float(ax[1])) if ax else None
    model.el_masks = masks2d       # per-electrode masks -> labeled display
    births = generate_births(spec)
    fly_fn, cols = make_planar_fly_fn(model, births, spec)
    if verbose:
        print(f"[stl] assembled A/B + fly fn in "
              f"{time.time() - _t_assemble:.3f}s "
              f"({len(births)} ions)")
    return model, fly_fn, cols, births
