# ion_gym — module map, linkage, and verification

## Linkage diagram

```
                       ┌──────────────── UI layer ────────────────┐
                       │ sim_app   (main Panel dashboard)         │
                       │  ├─ pe_view     (PE-surface tab)         │
                       │  ├─ stats       (impact-statistics card) │
                       │  ├─ stl_upload  (STL geometry sub-tab)   │
                       │  └─ last_flight / prewarm / telemetry    │
                       │ edit/     (browser geometry editor,      │
                       │            /editor + /flight viewers)    │
                       └───────────────────┬──────────────────────┘
                                           │
                          sim_build  (dispatch: build_run/…)
              ┌───────────┬───────────┬────┴──────┬─────────────┬────────────┐
        build_planar   build_rz   build_stl   build_stl3d   build_scene3d /
        (2-D planar)   (r-z cyl)  (STL 2-D)   (STL 3-D)     build_shapes3d
              │            │          │            │         (analytic 3-D)
       ┌──────┴────┐   solver2d   voxelize     voxelize +          │
  solver3d  multigrid3d (cyl 1/r) (Rung-2)     solver3d /     multigrid3d
  (3-D SOR) (V-cycle)              scene3d     multigrid3d    + tracer3d
                                                   │
                                             tracer3d (3-D channel kernel)
                                                   │
                                          collision3d + sds (HS / SDS gas)

  spec layer:   sim_spec (SimSpec — single source of truth) · symmetry
                (verified folds) · flight_conditions (re-weight w/o re-solve)
  staged:       staged_flight (multi-FA assemblies, declared seams) ·
                stations (detect/record planes) · drift_extension
                (exact ballistic flight to declared out-of-domain planes)
  I/O layer:    spec_io / assembly (documents) · fa_cache + basis_cache
                (content-hashed solved bases) · field_io · births · records
  legacy fly:   ionbench (reference-tracer equivalence path) ·
                tracer_numba (its 2-D field build and sampler)
  drivers:      ensemble_driver (threaded fly + storage)
```

## Core modules

Names in these tables are module basenames. Each lives in one subpackage:
`ion_gym.io` holds sim_spec, fa_cache, basis_cache and stl_upload;
`ion_gym.ui` holds sim_app and stats_card; `ion_gym.viz` holds viz_core,
pe_view and design_maps; `edit/` and `cad/` are `ion_gym.edit` and
`ion_gym.cad`; every other module in the tables is in `ion_gym.physics`
(so `sim_build` is `ion_gym.physics.sim_build`). `docs/PUBLIC_API.md`
gives the import path of every name scripts should use.

| module | purpose | verified by |
|---|---|---|
| `sim_spec` | Declarative SimSpec (geometry/source/gas/integration/bounds/stations); JSON round-trip; advisories (sub-cell source, voxelized-metal O(h)); the single source of truth | round-trip in app flows; every gate builds from it |
| `symmetry` | Declared symmetry (mirror/translation) VERIFIED per-problem before folding | refusal tests (false planes refused); `test_planar_fold` |
| `sim_build` | `build_run`/`build_needs_solve`/`build_route` dispatch: spec → correct builder | every end-to-end gate routes through it; `test_route_conformance` |
| `build_planar` | 2-D planar raster + multigrid solve + planar fly | einzel external-reference cross-validation (`validate_einzel_planar`); Mathieu quad gates |
| `build_rz` | Cylindrical (r-z) solve (1/r term) + r-z fly | funnel cross-validation (`validate_einzel_rz`); Kingdon log law 0.000% (internal validation record) |
| `build_stl` | STL → 2-D slice masks → planar solve | `test_stl3d` mid-z == 2-D bit-level; `test_stl_rods` |
| `build_stl3d` | Full 3-D STL: voxelize → `solve_bases_mg` → `fly3d` | `test_stl3d` gates; einzel focus-vs-voltage sweep |
| `build_scene3d`, `build_shapes3d` | Analytic 3-D geometry (CSG scenes; per-shape extrusion) | `test_scene_resolution`; `test_shape_extrusion` |
| `solver2d` | 2-D Laplace (planar + cylindrical), edge-ghost boundary handling | `test_solver_analytic`; `test_solver_convergence` |
| `solver3d` | 3-D SOR (validated kernel), `optimal_omega`, theta face-fractions | `test_solver3d_analytic`; `test_solver3d_nz1` |
| `multigrid3d` | V-cycle: full-cycle criterion + verified symmetric-subspace projection | box vs tight-SOR truth 7.9e-6 V; `test_multigrid` |
| `tracer`, `tracer3d` | Planar/r-z and 3-D multi-channel kernels (arbitrary per-electrode RF) | `test_multidrive`/`test_multidrive3d` vs frozen references |
| `staged_flight`, `stations` | Multi-FA assemblies: per-stage lattices, declared seams, face-identity bounds; detect/record station crossings | `test_staged_flight`; `test_assembly_json` |
| `flight_conditions` | Re-weight voltages/drives/ion/gas without re-solving (basis reuse; refuses geometry-key changes) | `internal` fc gate; `test_solve_orchestrator` |
| `drift_extension` | Exact ballistic flight to declared stations/bounds beyond the solved domain; refuses gas / hot edges by name | einzel analytic hand-check (internal validation record) |
| `collision3d`, `sds` | HS + SDS collision models (the same 3-D gas on every route); SDS jump statistics regenerated by `tools/regen_sds_tables.py` (see `docs/SDS_MODEL.md`) | `validate_collisions_c1`; `test_planar_gas_3d`; `test_sds_transport`; `test_sds_vs_hs`; `test_sds_regen_tables` |
| `voxelize`, `scene3d`, `rasterize3d` | STL voxelizer (Rung-2), analytic CSG scenes, field-array → masks | Rung-2 STL import ±5.5 µm (internal validation record) |
| `fa_cache`, `basis_cache` | Content-hashed solved-basis caches keyed on every geometry-affecting field | `test_fa_cache`; `test_stl_cache` |
| `ensemble_driver` | Threaded fly, storage toggles, NPZ export | `test_reproducible_fly`; `test_per_ion_mass` |
| `optimize` | OptimizeSpec (dotted SimSpec paths + bounds + metric) → CMA-ES over any buildable geometry | `test_optimize` (einzel focus recovery) |
| `flyer_plane` | Canonical width-plane flyer with first-event electrode-hit detection (EXIT/HIT/LOST/STUCK) | reference-tracer equivalence fixture; no gate in `tests/` imports it |
| `ballistic` | Field-free motion along one axis added to a two-axis deck (speed from energy, ballistic position, transverse walk); every value an argument | `test_ballistic` (speed vs SI kinetic energy, linear position, seeded draw, mass-free walk) |
| `mathieu` | Mathieu/Hill stability exponent; RF amplitude for a target q (`rf_amplitude_for_q`) | notebook `V02_mathieu_stability`; the HCD study notebook (RF from q) |
| `ionbench`, `tracer_numba` | Legacy reference-tracer equivalence path; `tracer_numba` holds only the 2-D field build and jitted bilinear sampler it shares with `tracer_tdep` | `test_ionbench_native_equiv`; `test_fa_cache` (field build) |
| `sim_app` | Panel dashboard: control tabs, plot tabs, status bar, voxel-doctrine display, exports | `test_ui_smoke`; `test_solve_orchestrator`; `validate_ui_exercise` |
| `pe_view`, `stats_card`, `stl_upload` | PE-surface tab, impact-statistics card (planes/stations figures, computed by `physics.stats`), STL upload panel | `test_pe_view`/`test_pe_surface_tab`/`test_stats`/`test_stl_upload` |
| `edit/` (policy, session, viewer, serve) | Browser geometry editor: per-route EditPolicy, targeted-patch commit (zero edits → original bytes), /editor + /flight servers | `test_edit_frame` (extrude-axis frame, clipping, shipped decks); `test_editor_refusal_rollback` |
| `cad/` (fit_geometry, stl_channels, cadquery_emit) | Fit analytic scenes from voxel masks; build an STL-channel deck from a CadQuery registry; emit CAD | `test_cadquery_emit`; fit_geometry and stl_channels have no gate in `tests/` |

## imaging / reporting

| module | role |
|--------|------|
| `viz_core.py` | **The** image/report framework. `Scene` (bodies / rays / field slabs / bounds / provenance) ← adapter `scene_from_simspec` → renderers `render_mpl` (multi-axis xy/xz/yz, per-group legend), `to_plotly`/`to_plotly_views`, `report()`, `stats_figure` (planes/stations). Metal is decomposed from the SOLVER'S OWN mask (display == solver input); fields are the solver's own arrays. Every panel is stamped SOLVER-DERIVED or SCHEMATIC. Owns the zoom policy: plotly axes are never scale-anchored. | `test_viz_core`; `validate_examples_draw` |
| `pe_view.py`, `design_maps.py` | PE-surface rendering; design-space maps | `test_pe_view`; `validate_pe_surface` |

New subsystems get pictures by writing an ADAPTER to `Scene`, not another
plotting script.

## Other modules

The supporting modules, by package. Most are reached through the modules
above rather than called directly.

**`ion_gym`**
- `cli`: the `ion-gym` command (`dashboard`, `validate`, `edit`, `--version`).
- `advisories`: `IonGymAdvisory`, the warning category for legal but
  ill-advised configurations and for fallbacks taken on the deck's behalf.
- `progress`: quote-then-run and progress reporting for long notebook cells.

**`ion_gym.io`**
- `paths`: every path ion_gym reads or writes (examples, the user data
  root, outputs, caches); `docs/DATA_LAYOUT.md` describes it.
- `spec_io`: one loader for any JSON a user hands ion_gym, deck or assembly.
- `assembly`: declarative multi-FA placement documents.
- `deck_params`: shows a deck's inherited physics and overrides it by name.
- `births`: writes the birth table that `distribution: "file"` reads.
- `field_io`: portable solved fields; `field_cycle_io`: the applied
  E-field exported over a time window.
- `records`: name-addressed access to trajectory columns.
- `results_bank`: ensemble results banked as self-describing `.npz` files.
- `run_provenance`: what a saved run needs in order to be flown again.
- `stl_resolve`: STL file resolution and preflight for STL decks.
- `lattice`: the conversion from mm to grid units.
- `jcad_export`: native-shape decks as JupyterCAD `.jcad` documents.

**`ion_gym.physics`**
- `constants`: physical constants, the one authority.
- `ccs`: per-ion hard-sphere cross sections (CCS fit or fixed).
- `beam_spec`: declarative ion-source formats and distributions.
- `ion_envelope`: the per-ion birth and fate record shared by every route;
  `birth_domain` reports an ion born outside the solved domain.
- `raster2d`: the 2-D raster shared by the planar routes; `interp`:
  generic grid interpolation.
- `tracer_rz` (r-z), `tracer_tw2d` (2-D multi-channel, travelling wave),
  `tracer_tdep` (fast-adjust superposition with voltages(t)): the
  tracers beside `tracer` and `tracer3d`. `tracer3d_v61_frozen` is the
  frozen 3-D reference `test_multidrive3d` compares against.
- `ionbench_numba`: numba port of the multi-FA flight in `ionbench`;
  `plane_gate`: terminate/record/port planes for the width-plane flyer,
  and packet replay between benches; `beam_interface`: a beam measured at
  a named port plane, handed from producer to consumer.
- `transporter`: validity gate for the periodic transporter plane pair.
- `tunnelling`: whether one integrator step can carry an ion through metal.
- `numba_rng`: a flight leaves numba's random state as it found it.
- `solve_code`: the identity of the code that solves a field, part of
  the field cache key.
- `sizing`: the solve-cost estimate for every geometry source.
- `stats`: impact and fate statistics; `traj_stats`: thermal and
  confinement statistics over trajectory ensembles.
- `w1_wiley`: the on-axis, piecewise-uniform-field model of a plate stack
  (Wiley–McLaren generalised to N fields).
- `cell_board`, `cell_screen`: parametric SLIM cross sections and their
  resumable Sobol screen.
- `run_export`: flies one deck for a run figure.

**`ion_gym.ui`**
- `serve`: the dashboard route table (`/`, `/editor`, `/flight`).
- `exports`: the Runs tab's prepared CSV, NPZ and HTML, served as
  ordinary HTTP downloads at `/export/<token>` on the dashboard server.
- `last_flight`: the server-wide last flight that `/flight` shows.
- `prewarm`: startup prewarm; `telemetry`: process-memory telemetry for
  long sessions; `widget_options`: keeps Panel selectors' options and
  value consistent.

**`ion_gym.viz`**
- `ensemble_figs`: ensemble-statistics figures.
- `run_figures`: the Impact Analysis view of one flight and its print export.
- `stability_figure`: computed (a, q) charts with flown survival markers.
- `workflow_figure`: one deck from unit-potential solutions through
  applied voltages to recorded flights.
- `deck_scheme`: an abbreviated deck listing beside the instrument.
- `nb_panels`: the notebooks' instrument intro panel.

**`ion_gym.edit`**
- `editor_panel` (numeric editor), `outline` and `contours` (render
  primitives and DC contours for the 2-D views), `flight_view` (the
  `/flight` page), `serve` (the editor page factory).

## Standalone validation scripts (`tests/validate_*.py`, run directly)
Reproducible evidence for: collisions C1, einzel (planar and r-z), example
rendering, PE surface, quadrupole Q3, STL einzel, and a scripted UI
exercise. The broader external-reference cross-validation history lives
in the internal validation record and does not ship.
