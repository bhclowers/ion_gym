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
  legacy fly:   ionbench / tracer_numba / tracer_rf2d (reference-tracer
                equivalence path)
  drivers:      ensemble_driver (threaded fly + storage)
```

## Core modules

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
| `collision3d`, `sds` | HS + SDS collision models; independently regenerated SDS tables | `validate_collisions_c1`; `test_sds_regen_tables` |
| `voxelize`, `scene3d`, `rasterize3d` | STL voxelizer (Rung-2), analytic CSG scenes, field-array → masks | Rung-2 STL import ±5.5 µm (internal validation record) |
| `fa_cache`, `basis_cache` | Content-hashed solved-basis caches keyed on every geometry-affecting field | `test_fa_cache`; `test_stl_cache` |
| `ensemble_driver` | Threaded fly, storage toggles, NPZ export | `test_reproducible_fly`; `test_per_ion_mass` |
| `optimize` | OptimizeSpec (dotted SimSpec paths + bounds + metric) → CMA-ES over any buildable geometry | `test_optimize` (einzel focus recovery) |
| `flyer_plane` | Canonical width-plane flyer with first-event electrode-hit detection (EXIT/HIT/LOST/STUCK) | reference-tracer equivalence fixture; `test_numba_equiv` |
| `ionbench`, `tracer_numba`, `tracer_rf2d` | Legacy reference-tracer equivalence path | `test_ionbench_native_equiv`; `test_numba_equiv` |
| `sim_app` | Panel dashboard: control tabs, plot tabs, Status tab, voxel-doctrine display, exports | `test_ui_smoke`; `test_solve_orchestrator`; `validate_ui_exercise` |
| `pe_view`, `stats`, `stl_upload` | PE-surface tab, impact-statistics card (planes/stations figures), STL upload panel | `test_pe_view`/`test_pe_surface_tab`/`test_stats`/`test_stl_upload` |
| `edit/` (policy, session, viewer, serve) | Browser geometry editor: per-route EditPolicy, targeted-patch commit (zero edits → original bytes), /editor + /flight servers | `edit_gate` battery (byte-identity across in-scope decks) |
| `cad/` (fit_geometry, quad_scene_build, cadquery_emit) | Fit analytic scenes from meshes; emit CAD | `test_cadquery_emit` |

## imaging / reporting

| module | role |
|--------|------|
| `viz_core.py` | **The** image/report framework. `Scene` (bodies / rays / field slabs / bounds / provenance) ← adapter `scene_from_simspec` → renderers `render_mpl` (multi-axis xy/xz/yz, per-group legend), `to_plotly`/`to_plotly_views`, `report()`, `stats_figure` (planes/stations). Metal is decomposed from the SOLVER'S OWN mask (display == solver input); fields are the solver's own arrays. Every panel is stamped SOLVER-DERIVED or SCHEMATIC. Owns the zoom policy: plotly axes are never scale-anchored. | `test_viz_core`; `validate_examples_draw` |
| `pe_view.py`, `deck_v.py`, `design_maps.py` | PE-surface rendering; deck visual summaries; design-space maps | `test_pe_view`; `validate_pe_surface` |

New subsystems get pictures by writing an ADAPTER to `Scene`, not another
plotting script.

## Standalone validation scripts (`tests/validate_*.py`, run directly)
Reproducible evidence for: collisions C1, einzel (planar and r-z), example
rendering, PE surface, quadrupole Q3, STL einzel, and a scripted UI
exercise. The broader external-reference cross-validation history lives
in the internal validation record and does not ship.
