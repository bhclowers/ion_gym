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

Also provides make_quad_stls(): writes four-rod quadrupole STLs and a
matching SimSpec, as the STL example.
"""

import math
from pathlib import Path

import numpy as np


def _require_trimesh():
    """Import trimesh with a clear, actionable message. trimesh is the
    STL/CAD path's mesh library (reading STLs, voxelizing) — the whole
    STL->electrode feature needs it. Only the STL examples/paths touch it;
    the inline-shape planar/rz/funnel builders don't."""
    try:
        import trimesh
        return trimesh
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "the STL path needs the 'trimesh' package "
            "(pip install trimesh). The einzel-STL example's bored plates "
            "also need a boolean backend (pip install manifold3d); the "
            "quadrupole example needs only trimesh. Inline-shape examples "
            "(planar/round einzel, funnel) don't require trimesh."
        ) from e


from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec, SourceSpec,
                      CollisionSpec, IntegrationSpec, ViewSpec, RFGroupSpec)
from ion_gym.io.sim_spec import (SymmetrySpec)
from ion_gym.io.lattice import cover_extent_mm
from ion_gym.physics.collision3d import KG_AMU, E_CHG


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


# ---------------------------------------------------- quad STL example
def make_quad_stls(out_dir, r0_mm=3.84, rod_r_mm=None, length_mm=1.0,
                   sections=96):
    """Write four cylindrical-rod STLs for a 2-D quadrupole cross-section
    (rods parallel to the extrusion direction, so a mid-slice gives four
    circles). r0 = inscribed field radius (centre to rod surface); the rod
    radius follows the classic round-rod ratio R = 1.148*r0 that best
    approximates the ideal hyperbolic field (ion_playground /
    convention), unless rod_r_mm is given. Rod centres sit at r0 + R from
    the axis. sections = circle tessellation (higher = rounder STL, though
    the voxel grid is usually the binding resolution). Returns
    ({index: filename}, centre_offset_mm)."""
    trimesh = _require_trimesh()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rod_r_mm is None:
        rod_r_mm = 1.148 * r0_mm         # classic round-rod ratio
    centre = r0_mm + rod_r_mm            # rod-centre offset from axis
    # place the domain so the axis (0,0) is at the grid centre later; here
    # STLs are in physical mm with the axis at (cx, cy) supplied by spec.
    positions = {1: (centre, 0.0), 2: (-centre, 0.0),
                 3: (0.0, centre), 4: (0.0, -centre)}
    files = {}
    for idx, (px, py) in positions.items():
        rod = trimesh.creation.cylinder(radius=rod_r_mm, height=length_mm,
                                        sections=sections)
        rod.apply_translation([px, py, 0.0])
        fn = f"quad_rod_{idx}.stl"
        rod.export(str(out / fn))
        files[idx] = fn
    return files, centre


def _q_to_rf_amp(q, mz, r0_mm, freq_hz, charge=1):
    """RF amplitude V (volts, 0-peak) for Mathieu q on an ideal quadrupole:
    q = 4 e V / (m r0^2 Omega^2)  ->  V = q m r0^2 Omega^2 / (4 e). Follows
    ion_playground's aq_to_uv / the standard mass-filter relation."""
    m = mz * KG_AMU
    omega = 2.0 * math.pi * freq_hz
    r0 = r0_mm / 1000.0
    return q * m * r0 ** 2 * omega ** 2 / (4.0 * charge * E_CHG)


def quad_stl_spec(stl_dir, rf_freq=2.0e6, mz_stable=100.0,
                  mz_unstable=30.0, r0_mm=3.84, axial_ke_ev=8.0,
                  q_stable=0.40, rf_amp=None, pitch_mm=0.2):
    """An axial-transport quadrupole driven from STL rods. The transverse
    field is a validated 2-D solve (four rods, RFA on x-rods and RFB=RFA
    inverted on y-rods — the classic quadrupole saddle). Ions are injected
    near-axis with axial KE and drift down the transport axis (z) while the
    RF confines them radially: exactly the standard quad-transport picture.
    Because an ideal quad's axial motion decouples (no z-force), the axial
    drift z = vz*t is exact on top of the 2-D transverse dynamics.

    Geometry follows the ion_playground / node-centred convention: r0 = 3.84 mm
    inscribed radius, round rods with
    R = 1.148*r0. The RF amplitude is set ANALYTICALLY from the Mathieu
    relation V = q m r0^2 Omega^2 / (4 e) for a target q on the stable
    mass (rather than an empirical field fit), so the physics is
    transparent. At q_stable=0.40 for m/z 100, the lighter m/z 30 sits at
    q = 0.40*(100/30) ~ 1.33 (> the 0.908 stability limit) and is lost.
    View the transport in the xz plane. Build with build_stl_run(spec,
    ground_border=True).

    pitch_mm is the solve pitch AND the lattice the domain span is
    counted in -- one value, used for both, so the span cannot drift out
    of conformance with the grid it is solved on."""
    files, centre = make_quad_stls(stl_dir, r0_mm=r0_mm)
    rod_r = 1.148 * r0_mm
    if rf_amp is None:
        rf_amp = _q_to_rf_amp(q_stable, mz_stable, r0_mm, rf_freq)
    # LATTICE RULE: the domain span is a LATTICE quantity, so it is an
    # exact integer number of cells at this spec's pitch. It was
    # 2.2 * (centre + rod_r) -- a raw float off r0_mm, which at the
    # declared 0.2 mm pitch gave 27.8446 mm = 139.223 cells and a spec the
    # lattice loader refuses. Rounded UP (cover-up), so the domain still clears
    # the rods by at least the 2.2x factor this example intends; rounding
    # down could bring a wall inside that clearance. The rods are centred
    # on span/2 below, so they stay centred in the conforming domain.
    span = cover_extent_mm(2.2 * (centre + rod_r), pitch_mm)
    # DECLARED axial extent: the rod STLs are 1 mm cross-section tokens
    # (make_quad_stls), so the display span is derived from the example's
    # OWN declared dynamics — the axial distance the stable mass covers in
    # t_max (v = sqrt(2*KE*e/m)), i.e. the transport length this example
    # actually exercises. No invented constant; it lands in the spec JSON
    # where it is visible and editable.
    _v_ax = math.sqrt(2.0 * axial_ke_ev * E_CHG
                      / (mz_stable * KG_AMU)) * 1e-3     # mm/us
    _t_max_us = 30.0                                     # = IntegrationSpec below
    axial_extent = [0.0, round(_v_ax * _t_max_us, 1)]
    # STL rod coordinates are centred on (0,0); the solver grid's origin
    # is at the corner, so the axis belongs at (span/2, span/2). The
    # MUTATION PATH THAT USED TO LIVE HERE -- load each fixture, translate
    # it, re-export over the original bytes -- is DELETED: it made
    # deck correctness depend on hidden fixture state, and the one time
    # pristine CAD exports shipped, half the quadrupole fell off-domain
    # and a notebook died end-to-end. Placement is now DECLARED on the
    # spec below and applied at stl_resolve.load_mesh, the single mesh-
    # ingest point; the fixtures stay pristine, origin-centred exports.

    rf_groups = [RFGroupSpec("RFA", frequency_hz=rf_freq,
                             amplitude_v=rf_amp, phase_deg=0.0),
                 RFGroupSpec("RFB", frequency_hz=rf_freq,
                             amplitude_v=rf_amp, phase_deg=180.0)]
    els = []
    for idx in range(1, 5):
        grp = "RFA" if idx in (1, 2) else "RFB"
        els.append(ElectrodeSpec(name=f"rod_{idx}", stl=files[idx],
                                 dc=0.0, rf_groups=([grp] if grp else [])))
    return SimSpec(
        geometry=GeometrySpec(
            width_mm=span, height_mm=span, mm_per_gu=float(pitch_mm),
            symmetry=SymmetrySpec(coords="xyz"), stl_dir=str(stl_dir),
            axial_extent_mm=axial_extent,
            frame_offset_mm=[span / 2, span / 2, 0.0],
            electrodes=els, rf_groups=rf_groups),
        source=SourceSpec(distribution="box", box_mm=[1.5, 1.5, 0.5],
                          n_ions=60,
                          x0_mm=span / 2, y0_mm=span / 2 + 0.3,
                          z0_mm=0.0, axis="z", direction=[0.0, 0.0, 1.0],
                          ke_lo=axial_ke_ev, ke_hi=axial_ke_ev,
                          mz_list=[mz_stable, mz_unstable],
                          tob_span_us=0.0),
        collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(dt_ns=2.0, t_max_us=30.0, rec_every=2,
                                    record_channels=["speed", "radius"]),
        view=ViewSpec(mode="2d", planes=["xz"]),
        name="quadrupole (STL rods, axial transport)",
        notes="Two masses flown: mz_stable stays bounded, mz_unstable "
              "grows and hits a rod — the Mathieu stable/unstable demo.")


def make_einzel3d_stls(out_dir, bore_mm=3.0, plate_t_mm=1.5, gap_mm=2.0,
                       outer_mm=14.0):
    """Three apertured plates (an einzel lens) as STLs — a GENUINELY 3-D
    geometry: the on-axis potential varies strongly along z (the classic
    einzel dip), so the mid-z slice is NOT representative and the full 3-D
    solve is required. Outer plates grounded, centre plate the lens
    electrode. Returns ({index: filename}, z_centres_mm, total_depth_mm)."""
    trimesh = _require_trimesh()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z0 = 2.0                                   # entrance drift before plate 1
    files = {}
    z_centres = []
    z = z0
    for idx in range(1, 4):
        zc = z + plate_t_mm / 2
        plate = trimesh.creation.box(extents=[outer_mm, outer_mm,
                                              plate_t_mm])
        bore = trimesh.creation.cylinder(radius=bore_mm, height=plate_t_mm*3,
                                         sections=64)
        plate = plate.difference(bore)         # annular plate with round bore
        plate.apply_translation([outer_mm / 2 + 1.0, outer_mm / 2 + 1.0, zc])
        fn = f"einzel3d_plate_{idx}.stl"
        plate.export(str(out / fn))
        files[idx] = fn
        z_centres.append(zc)
        z += plate_t_mm + gap_mm
    total_depth = z + z0                       # exit drift after plate 3
    return files, z_centres, total_depth


def einzel3d_spec(stl_dir, lens_v=-800.0, beam_ke_ev=30.0, bore_mm=3.0,
                  pitch_mm=0.2):
    """3-D einzel lens from STL plates: outer plates grounded, centre plate
    at lens_v. A beam launched along +z is focused by the axially-varying
    field — a geometry where the 2-D mid-slice is qualitatively wrong, so it
    exercises the full 3-D path. Build via build_stl3d_run (routed by
    depth_mm>0)."""
    files, zc, depth = make_einzel3d_stls(stl_dir, bore_mm=bore_mm)
    outer = 14.0
    # LATTICE RULE: span and depth are LATTICE quantities. `depth` comes back from
    # make_einzel3d_stls as the meshes' own axial extent -- an arbitrary
    # float -- and was written straight into depth_mm, giving 72.5 cells
    # at 0.2 mm and a spec this package's own loader refuses. Both are
    # covered up through the shared helper. pitch_mm is a
    # named argument driving BOTH the counting and the grid, so the two
    # cannot drift apart.
    span = cover_extent_mm(outer + 2.0, pitch_mm)
    depth = cover_extent_mm(depth, pitch_mm)
    els = [ElectrodeSpec(name=f"plate_{i}", stl=files[i],
                         dc=(lens_v if i == 2 else 0.0))
           for i in (1, 2, 3)]
    return SimSpec(
        geometry=GeometrySpec(
            width_mm=span, height_mm=span, depth_mm=depth,
            mm_per_gu=float(pitch_mm),
            symmetry=SymmetrySpec(coords="xyz"), stl_dir=str(stl_dir),
            electrodes=els),
        source=SourceSpec(distribution="disc", n_ions=12, r_mm=1.0,
                          x0_mm=span / 2, y0_mm=span / 2, z0_mm=0.5,
                          axis="z", direction=[0.0, 0.0, 1.0],
                          ke_lo=beam_ke_ev, ke_hi=beam_ke_ev,
                          mz_list=[100.0], tob_span_us=0.0),
        collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(dt_ns=1.0, t_max_us=20.0, rec_every=2,
                                    record_channels=["radius", "ke_ev"]),
        view=ViewSpec(mode="2d", planes=["xz"]),
        name="einzel lens — STL plates (full 3-D)",
        notes="Three apertured STL plates; centre plate focuses the beam. "
              "The on-axis field varies along z, so this needs the full 3-D "
              "solve (mid-z slice is not representative).")


_STL_BUILD_CACHE = {}


def _stl_geom_signature(spec, ground_border):
    """Cache key capturing everything voxelize+solve depends on: the STL
    files + mtimes, grid, grounding. Voltages/RF are NOT in the key —
    they re-weight cheaply from cached bases (fast-adjust invariant)."""
    g = spec.geometry
    sig = [round(g.width_mm, 6), round(g.height_mm, 6),
           round(g.mm_per_gu, 6), bool(ground_border)]
    p = Path(g.stl_dir) if g.stl_dir else Path(".")
    for el in g.electrodes:
        if el.stl:
            try:
                mt = (p / el.stl).stat().st_mtime
            except OSError:
                mt = -1.0
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
    try:
        from ion_gym.physics.multigrid3d import solve_bases_mg
    except ImportError as e:
        print(f"[stl] multigrid unavailable ({e}); solving by SOR")
        bases3 = solve_bases(masks, mirror=(False, False, False), tol=1e-4,
                             stencil="ghost_linear", verbose=verbose, omega=1.9)
    else:
        bases3 = solve_bases_mg(masks, mirror=(False, False, False), tol=1e-4,
                                stencil="ghost_linear", verbose=verbose)
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

    A = np.zeros((nx, ny))
    # collect per-(freq,phase) RF bases, then FOLD into one signed B the
    # way build_rz does: phase 0 -> +, phase 180 -> -. The planar tracer
    # flies a single B (A + sin(wt)*B); for the two-phase quad/funnel this
    # signed fold is exact and gives the proper saddle (x-rods and y-rods
    # in antiphase). >2 distinct phases (travelling wave) would need the
    # multi-B tracer — flagged, not silently mis-flown.
    freqs = set()
    phase_groups = {}          # (freq,phase) -> summed amp*fa
    for idx, el in enumerate(g.electrodes, start=1):
        fa = bases[idx] / 1e4
        A = A + el.dc * fa
        amp, freq, phase = g.electrode_rf(el)
        if amp != 0.0 and freq != 0.0:
            freqs.add(round(freq, 3))
            k2 = (round(freq, 3), round(phase, 3))
            phase_groups.setdefault(k2, np.zeros((nx, ny)))
            phase_groups[k2] += amp * fa
    Bk = []
    if phase_groups:
        f0 = sorted(freqs)[0]
        phases = sorted({k[1] for k in phase_groups})
        B = np.zeros((nx, ny))
        for (freq, phase), basis in phase_groups.items():
            sign = 1.0 if (phase % 360.0) < 90.0 or (phase % 360.0) >= 270.0 \
                else -1.0
            B = B + sign * basis
        Bk = [(B, f0, 0.0)]
        if len(phases) > 2:
            import warnings
            warnings.warn(
                "STL build: >2 RF phases folded into a two-phase B — the "
                "single-B planar tracer approximates it. Multi-B "
                "travelling-wave tracer is the flagged extension.")
    Ex, Ey = _grad2d(np.ascontiguousarray(A), h)
    model = PlanarModel(A, Bk, ele0, Ex, Ey, h, spec, el_bands=bands)
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
