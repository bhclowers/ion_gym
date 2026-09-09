"""tiers.py — WHICH GATES RUN WHEN.

Measured on one core. The suite is ~25-35 min, and the cost is REAL
PHYSICS (solves and flights), not overhead: numba already caches to disk, imports
are 0.2 s, and the basis cache is 580 K. Caching cannot save this; only choosing
what to run can.

The distribution is brutally skewed, which is what makes tiering work:

    4 tests in test_bench_M_PA3    151 s   (of a 192 s group; the other 76
                                            tests in it cost ~40 s COMBINED)
    validate_mirror_E2            ~300 s   (21,875 rays x pressure points)
    test_export_frame_agnostic     113 s   (of a 166 s structural set)
    everything else                 fast

TIERS
-----
  1  STRUCTURAL  (~50 s)   Code contracts: does the software still mean what it
                           says?  Except-policy, provenance, viz, UI smoke,
                           cache keys, labels, rasters, orchestration.  Runs on
                           EVERY edit.  All six defects of the v145 audit would
                           have been caught here.
  2  PHYSICS     (~5 min)  Analytic and golden gates: solver convergence and
                           order, einzel goldens, multigrid, W1/W2/W3, Q3.  Runs
                           before any claim about a NUMBER.
  3  INSTRUMENT  (~20 min) Full-instrument flights and acceptance sweeps:
                           bench_M_PA*, mirror_E2, coopt, immersion, gbstack.
                           RELEASE ONLY.

THE TRAP THIS MUST NOT REPEAT
-----------------------------
v144: 32 of 38 validators were dark behind a green pytest, because a tier that
is not run and NOT VISIBLE is indistinguishable from a tier that passed.  So:

  * THE NO-SILENCE RULE BINDS THE RUNNER TOO.  `run_gates.py --tier N` PRINTS every gate it
    skips, by name and tier.  A skipped gate that announces itself is a
    schedule; a skipped gate that is silent is a lie.
  * The release gate runs ALL tiers.  `--tier` is a development convenience and
    is never the thing that certifies a release.
  * A gate NOT LISTED HERE is REFUSED, not defaulted into a tier.  Defaulting an
    unclassified gate to tier 3 would silently exile it from every dev run;
    defaulting it to tier 1 would silently slow every dev run to a crawl.  Both
  are silent-default violations (a catch-all `else` absorbing the unknown case).  Add the
    gate here, deliberately, or the runner refuses to start.
"""

TIERS = {1: "STRUCTURAL", 2: "PHYSICS", 3: "INSTRUMENT"}

# gate stem (no .py) -> tier
GATE_TIER = {
    # ---- 1. STRUCTURAL: code contracts, no physics claim ----------------
    # CLASSIFIED 2026-09-08 after the runner REFUSED to run at all: 12 of
    # the 38 discovered script gates had no entry here, and the refusal
    # fires before anything executes, so `run_gates.py --tier N` -- the
    # release gate included -- had been dead. Runtimes below are measured,
    # not estimated, and are the reason for each tier.
    "test_lint":                 1,   # 0 s  lint policy
    "test_field_io":             1,   # 0 s  field save/load portability
    "test_cadquery_emit":        1,   # 0 s  CadQuery script emitter
    # 1 s. Certifies a ROUTE CONTRACT (every route fills the declared
    # channels), which is structure, not a number. NOTE its docstring says
    # "tier 2 of the parity plan" -- that is the PARITY PLAN's numbering,
    # not GATE_TIER's; the two are unrelated scales and the wording is a
    # trap for the next reader. Move it to 2 if the intent was otherwise.
    "test_route_conformance":    1,
    "test_ke_axis_channels":     1,   # 3 s  per-axis KE record channels
    "test_assembly_json":        1,   # 6 s  assembly JSON front end
    "test_except_policy":        1,
    "test_dependency_audit":     1,     # K1-K9 placement ratchet (AST scan, sub-second)
    "test_provenance":           1,
    "test_ele_labels":           1,
    "test_polygon_raster":       1,
    "test_scene_resolution":     1,
    "test_ui_smoke":             1,
    "test_fa_cache":             1,
    "test_stl_cache":            1,
    "test_solve_orchestrator":   1,
    "test_pe_view":              1,
    "test_pe_surface_tab":       1,
    "test_spec_loader":          1,
    "test_export_frame_agnostic": 1,   # 113 s — the tier-1 outlier; see note
    "test_reproducible_fly":     1,
    "test_stl_assembly":         1,
    "test_numba_equiv":          1,
    "test_per_ion_mass":         1,

    # ---- 2. PHYSICS: analytic / golden, certifies a NUMBER ---------------
    "test_trap_physics":         2,   # 1 s   trap physics against a deck
    "test_multidrive_compose":   2,   # 14 s  drive-channel composer
    "test_ionbench_native_equiv": 2,  # 17 s  native kernel equivalence
    "test_solver_analytic":      2,
    "test_solver_convergence":   2,
    "test_solver3d_nz1":         2,
    "validate_einzel_planar":    2,
    "validate_einzel_rz":        2,
    "validate_funnel_native":    2,
    "validate_funnel_f0":        2,
    "validate_collisions_c1":    2,
    "validate_collisions_f1":    2,
    # q3_2d reclassified 2 -> 3. Not broken — SLOW: it repeats
    # full 3-basis solves (~2 min each) and hit the 600 s tier-2 timeout at
    # a checkpoint timeout. It certifies an instrument-level number and
    # belongs with the release-only sweeps.
    # SANDBOX NOTE: it could not complete in a constrained sandbox at
    # all — root cause: basis_cache.store EXISTED (uncalled);
    # build_planar_model never banked its solves, so each ~285 s attempt
    # restarted cold. FIXED: solves now store on
    "validate_pe_surface":       2,
    "validate_stl_einzel":       2,
    # Native successor to a retired gate. Kernel-equivalence
    # bars at machine precision on an in-gate solve3d fixture — physics-
    # bearing solve + JIT, ~1-3 min, so tier 2 not tier 1.
    "test_multifa_native_equiv": 2,
    # buncher demotion — native forward-carry for tracer_tdep
    # (frozen golden chained to the validate_buncher anchor).
    "test_tdep_native":          2,
    # Caught by the runner's REFUSAL, not by me — which is the point of the
    # refusal. Was RED on a clean tree until Scene.check()
    # refused the gate's deliberate full-frame overhang (a later guard
    # forbidding the Rung-1 authoring pattern it exercises). Fixed at root
    # cause — overhang is now a DECLARED GridSpec policy — and the gate
    # declares it. Tier 2: it certifies the RUNG-1 emittance fit, a NUMBER.
    "test_rung1_fit_emit":       2,

    # ---- 3. INSTRUMENT: full flights + acceptance sweeps, RELEASE ONLY ----
    # Minutes each, on the same "reclassified 2 -> 3, not broken, SLOW"
    # grounds as the entries below: cost, not doubt about what they prove.
    "test_staged_flight":        3,   # >60 s   multi-region staged flight
    "validate_examples_draw":    3,   # ~2.5 min every shipped deck draws
    "validate_ui_exercise":      3,   # >4 min  drives the whole UI
    # GROUP A — certified-instrument-number gates. BACK BURNER:
    # not required for the UI to be functional, and they will
    # be re-derived anyway once the INPUT ION BEAM parameters are defined.
    # Left classified tier-3 (release-only) so the runner still names them;
    # they are simply not a checkpoint blocker this cycle.
    "validate_buncher":          3,     # REPAIRED (tracer_tdep; buncher fixture supplied)
    "validate_collisions_c2":    3,     # REPAIRED (tracer_gas; quad_upload aliased)
    "validate_mesh_legs":        3,     # REPAIRED (quad_scene_build landed)
    "validate_native_fly":       3,     # REPAIRED
    "validate_solver3d_quad":    3,     # REPAIRED
    "validate_voxelizer":        3,     # REPAIRED
    "validate_quad3d":           3,     # RUNNABLE + PASS: Gate 0 inverse 3.3e-15, Gate 1a 8.23mV, Gate 2 10 ions worst dTOF 1.99ns. Gate 1b legitimately SKIP (CSV has no dE/dx cols) — named, not silent.
    "validate_stl_field":        3,     # RUNNABLE + PASS: quad_stl_beamwin.npz recovered, + iob/CSV. R2-Fa 7.5mV, R2-Fc mid-transport 0.012%, 5/5 ions PASS. LOCAL FIXTURE.
    "validate_stl_import":       3,     # RUNNABLE (imported STL fixture supplied): R2-G PASS, 5/5 electrodes, FAR=0.
    # RETIRED gates are NOT classified here. The runner discovers gates by
    # globbing tests/ only (never tests/retired/), so a retired gate can never
    # appear in a plan; a GATE_TIER entry for one is a dead classification that
    # misrepresents the schedule to a human reader (it did — it inflated the
    # tier-3 count and read as live coverage). Coverage for each is accounted
    # for in tests/retired/RETIRED.md, and the live successors are classified
    # above: test_multifa_native_equiv (for the retired numba-equiv test +
    # validate_tof_multifa, GAP M-1), validate_collisions_c2 (for
    # validate_collisions_c3). validate_octupole / test_octupole_rf have no
    # successor and are recorded as retired with their history.
    # MOVED OUT 2026-09-08: validate_q3_2d went to internal/tests_q3/ with
    # the Q3 study itself (an internal project that was shipping publicly).
    # Its classification goes with it. It is also the gate that made
    # `run_gates.py` structurally unable to pass: documented as Phase-1
    # 21/23 with two open geometry items (flush.y, refuse_centre), it
    # exits non-zero BY DESIGN, so a release certification including it
    # could never be green.
    # Gates that live outside this tree are classified beside
    # themselves, merged
    # below when that tree is present (the public tree may
    # not reference internals). On a public checkout the runner globs
    # tests/ only, so no internal gate can appear in a plan anyway.
}

from pathlib import Path as _Path
_extra = _Path(__file__).resolve().parents[1] / "internal" / \
    "tests_oaTOF" / "tiers_internal.py"
if _extra.exists():
    _ns = {}
    exec(_extra.read_text(), _ns)
    GATE_TIER.update(_ns["GATE_TIER_INTERNAL"])

# test_export_frame_agnostic is 113 s of a ~50 s tier — it is the one structural
# gate that pays a physics price, because it AGNOSTICISM-checks by re-solving.
# It stays in tier 1 (frame agnosticism is a CODE contract and the export path
# is exactly what a fix "for one configuration" breaks), but it is
# the first candidate if tier 1 needs to get faster. Do not move it silently.


def tier_of(stem: str) -> int:
    """REFUSE an unclassified gate — never default it into a tier."""
    if stem not in GATE_TIER:
        raise KeyError(
            f"gate {stem!r} is not classified in tests/tiers.py. Every gate "
            f"declares its tier; the runner does not guess. Defaulting it to "
            f"tier 3 would silently exile it from every dev run (the v144 "
            f"failure exactly); defaulting it to tier 1 would silently slow "
            f"every dev run. Add it to GATE_TIER deliberately.")
    return GATE_TIER[stem]
