"""
test_ui_smoke.py -- press every button, toggle every widget, on TWO unalike
specs, and assert nothing raises.

WHY THIS EXISTS
    Three UI crashes shipped in a row, each found by the user running the app:
      * Checkbox(description=...)  -> TypeError on Panel 1.9.3
      * pe_figure_3d(kw..., pos)   -> SyntaxError
      * mmode referenced before assignment -> UnboundLocalError on Compute
    Every one of them was "verified" with ast.parse and a few direct function
    calls. ast.parse proves a file is syntactically Python. It cannot execute a
    callback, cannot see an unbound local on a branch, and knows nothing about
    which kwargs the installed Panel accepts.

    The only thing that catches this class of bug is INVOKING THE CALLBACK the
    button is wired to -- which is what this does. It is deliberately dumb: no
    physics assertions, no gates. It asks one question: does the UI raise?

    Coverage is by CALLBACK, not by widget, so a new button is only covered
    when it is added here. If a handler is not in this file, it is not tested.

Run:  python3 test_ui_smoke.py     (standalone; not pytest)
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import panel as pn
pn.extension("plotly")

import numpy as np

FAILED = []
PASSED = []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        FAILED.append((name, e))
        print(f"  [FAIL] {name}\n         {type(e).__name__}: {e}")
        if "-v" in sys.argv:
            traceback.print_exc()


# ---------------------------------------------------------------- fixtures
def spec_planar():
    """2-D r-z einzel: INLINE shapes, no external file.

    INLINE, deliberately: the fixture must not depend on any file that is not
    in the repo.  (The example this replaces, the STL funnel, pointed at
    funnel-1..17.stl in a directory that was never checked in, so it failed
    everywhere except the machine that authored it.  It was removed in v144;
    this note records why a smoke fixture is never allowed to reach outside
    the tree.)

    Unalike to the 3-D fixture in every way that matters: 2-D, r-z coords, DC
    only, no dc_groups, no RF, inline shapes instead of a GeomScene.
    """
    from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec, ShapeSpec,
                          SourceSpec)
    from ion_gym.io.sim_spec import SymmetrySpec
    els = []
    for k, (x0, dc) in enumerate(((4.0, 0.0), (9.0, -250.0), (14.0, 0.0))):
        els.append(ElectrodeSpec(
            name=f"L{k + 1}", dc=dc,
            shapes=[ShapeSpec("rect", {"x_mm": x0, "y_mm": 3.0,
                                       "width_mm": 2.0, "height_mm": 3.0})]))
    geo = GeometrySpec(width_mm=20.0, height_mm=6.0, mm_per_gu=0.2,
                       symmetry=SymmetrySpec(coords="rz"), electrodes=els)
    s = SimSpec(geometry=geo, source=SourceSpec(seed=0), name="smoke_einzel_rz")
    s.source.n_ions = 3
    s.source.distribution = "point"
    s.source.x0_mm, s.source.y0_mm = 1.0, 0.5
    s.source.direction = [1.0, 0.0, 0.0]
    s.source.ke_lo = s.source.ke_hi = 20.0
    s.source.mz_list = [115.0]
    s.integration.dt_ns = 5.0
    s.integration.t_max_us = 20.0
    return s


def spec_scene3d():
    """3-D analytic CSG with an RF drive AND a DC ladder. Unalike in every
    way that matters: builder, dimensionality, groups, RF."""
    from ion_gym.physics.scene3d import GeomScene, GridSpec, Electrode, Shape, Box3D
    from ion_gym.physics.build_scene3d import simspec_from_scene
    els = [
        Electrode(index=1, name="ROD", voltage="RF_A",
                  shapes=[Shape(within=[Box3D(1, 5, 0, 3, 7, 30)])]),
        Electrode(index=2, name="ROD2", voltage="RF_A",
                  shapes=[Shape(within=[Box3D(9, 5, 0, 11, 7, 30)])]),
    ]
    for k in range(3):
        z0 = 2.0 + k * 9.0
        els.append(Electrode(index=3 + k, name=f"E{3 + k}", voltage=10.0 - 4 * k,
                             shapes=[Shape(within=[
                                 Box3D(5, 1, z0, 7, 3, z0 + 6)])]))
    sc = GeomScene(grid=GridSpec(nx=2, ny=2, nz=2, mm_per_gu=1.0),
               electrodes=els, units="mm",
               name="smoke_rf_ladder").at_resolution(1.0, margin_mm=1.0)
    s = simspec_from_scene(sc, dc_group="LADDER")
    # DECLARED seed: simspec_from_scene defaults seed=None
    # = fresh entropy; seed 0 matches the planar twin above.
    s.source.seed = 0
    for g in s.geometry.rf_groups:
        g.frequency_hz, g.amplitude_v, g.phase_deg = 2.0e6, 100.0, 0.0
    s.source.n_ions = 3
    s.source.distribution = "point"
    s.source.x0_mm, s.source.y0_mm, s.source.z0_mm = 6.0, 6.0, 3.0
    s.source.direction = [0, 0, 1]
    s.source.mz_list = [115.0]
    s.integration.dt_ns = 20.0
    s.integration.t_max_us = 30.0
    return s


FIXTURES = [("einzel_rz_2d", spec_planar), ("scene3d_rf_ladder", spec_scene3d)]


def run_for(label, make_spec):
    from ion_gym.ui.sim_app import SimApp
    print(f"\n=== {label} ===")

    # CONSTRUCTION IS ITSELF A TEST. It used to sit outside check(), so a
    # widget kwarg the installed Panel rejects (Checkbox(description=...) on
    # 1.9.3) took the whole harness down with a traceback instead of being
    # reported as one failure. The thing most likely to break is the thing
    # that must be caught, not the thing that kills the reporter.
    holder = {}

    def construct():
        holder["app"] = SimApp(make_spec())
        holder["app"].panel()
    check("construct + panel()", construct)
    app = holder.get("app")
    if app is None:
        print("  ... construction failed; skipping the rest of this fixture")
        return

    # ---- every DISPLAY toggle, in both states, on every view plane -------
    def toggles():
        for plane in ("xy", "xz", "yz"):
            app.w_plane.value = plane
            for fills in (True, False):
                for labels in (True, False):
                    app.w_elfill.value = fills
                    app.w_ellabel.value = labels
                    app._draw_background()
        app.w_lock.value = True
        app._draw_background()
        app.w_lock.value = False
        app._draw_background()
    check("view planes x electrode fills/labels x aspect lock", toggles)

    def outlines_survive_fills_off():
        """The v127 bug: turning fills OFF blanked the electrodes in xz/yz,
        because only the xy path drew an outline. A crash test cannot see
        that -- nothing raises, the metal just disappears. So COUNT the
        outlines: with fills off there must be zero fills and a nonzero
        number of outlines, in EVERY plane.

        COUNTING CORRECTED. This counted only `contour` traces
        and layout shapes, and FAILED on xz for two fixtures for days --
        wrongly. `mask_outline_mm` draws an outline as a scatter POLYLINE,
        deliberately: a plotly 0.5-contour of a boundary-flush mask is an
        open, clipped line, which is why that path exists at all. So the
        electrodes were drawn, named e1/e2/e3, and the gate called them
        invisible.

        A gate must recognise every legitimate way of drawing the thing it
        checks, or it reports a defect in the renderer that is really a
        defect in itself -- and the recorded 'fix' for it would
        have changed working code. Outlines are therefore counted by
        IDENTITY (a line trace named after an electrode) rather than by
        trace type. Named-only, so trajectory scatter can never satisfy
        it.
        """
        app.w_elfill.value = False
        # Iterate the planes the geometry ACTUALLY HAS, not a hardcoded three.
        # An r-z model has no xz/yz -- its plane is r-z -- so demanding electrode
        # outlines there demanded them from views that cannot exist.  The
        # selector is now driven by capability (sim_app._planes_for_spec), so
        # this asks only for planes the app is willing to offer.
        app._draw_background()
        # The label the RENDERER draws, which is not always the deck's
        # electrode name: model.el_masks is keyed by 1-based index and
        # el_mask_fills labels those "e{k}". Match what is on screen.
        _mk = list((getattr(app._model, "el_masks", None) or {}).keys())
        el_names = ({e.name for e in app.spec.geometry.electrodes}
                    | {f"e{k}" for k in _mk} | {str(k) for k in _mk})
        for plane in list(app.w_plane.options):
            app.w_plane.value = plane
            app._draw_background()
            fig = app.pane.object
            kinds = [tr.type for tr in (fig.data or [])]
            # the r-z END-ON view draws its rings as layout SHAPES, not traces;
            # counting only traces would call a correctly-drawn view empty.
            n_shapes = len(getattr(fig.layout, "shapes", ()) or ())
            n_poly = sum(
                1 for tr in (fig.data or [])
                if tr.type == "scatter"
                and "lines" in str(getattr(tr, "mode", "") or "")
                and getattr(tr, "name", None) in el_names)
            n_out = kinds.count("contour") + n_shapes + n_poly
            n_fill = kinds.count("heatmap")
            assert n_out > 0, (
                f"{plane}: fills OFF and NOTHING drawn for the electrodes — "
                f"they are invisible (traces: {kinds}, shapes: {n_shapes}, "
                f"named outlines: {n_poly})")
            assert n_fill == 0, f"{plane}: fills OFF but {n_fill} fill traces"
        app.w_elfill.value = True
    check("electrode OUTLINES survive fills=off in every plane",
          outlines_survive_fills_off)

    def drawn_extent_matches_axes():
        """The v135 transposition: the xz view reported axes (z, x) but the
        electrode projection still branched on `la == "x"`, so it rendered the
        yz projection under xz's labels — a 140 mm ladder drawn up the 15 mm
        axis. Nothing raised; the picture was simply a lie.

        So assert the DATA agrees with the LABELS: whatever axis the view says
        is horizontal, the drawn geometry must span that axis's real extent.
        """
        g = app.spec.geometry
        if g.depth_mm <= 0:
            return                       # 2-D: no z extent to confuse
        span = {"x": g.width_mm, "y": g.height_mm, "z": g.depth_mm}
        app.w_elfill.value = False
        for plane in ("xy", "xz", "yz"):
            app.w_plane.value = plane
            app._draw_background()
            la, lb = app._plane_cols()[-2:]
            xs, ys = [], []
            for tr in (app.pane.object.data or []):
                if tr.type == "contour" and tr.x is not None:
                    xs += [float(np.min(tr.x)), float(np.max(tr.x))]
                    ys += [float(np.min(tr.y)), float(np.max(tr.y))]
            if not xs:
                continue
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            # the drawn span on each axis must be consistent with THAT axis
            assert w <= span[la] + 1.0, (
                f"{plane}: horizontal axis is {la} (domain {span[la]:.1f} mm) "
                f"but the geometry drawn spans {w:.1f} mm — TRANSPOSED")
            assert h <= span[lb] + 1.0, (
                f"{plane}: vertical axis is {lb} (domain {span[lb]:.1f} mm) "
                f"but the geometry drawn spans {h:.1f} mm — TRANSPOSED")
        app.w_elfill.value = True
    check("drawn geometry matches the axis labels (not transposed)",
          drawn_extent_matches_axes)

    def shading():
        app.w_showfield.value = True
        for mode in app.w_fieldmode.options:
            app.w_fieldmode.value = mode
            app._draw_background()
        app.w_showfield.value = False
        app._draw_background()
    check("every shading type (incl. PE 3D)", shading)

    # ---- the buttons ----------------------------------------------------
    check("Recompute field", lambda: app._on_recompute())
    check("Clear ions", lambda: app._on_clear())

    def fly():
        # HARNESS LIMIT, STATED: _solve_then has two paths. Cached geometry ->
        # build inline, synchronous. Fresh solve -> background thread, result
        # delivered by a Bokeh periodic callback -- which NEVER FIRES headless,
        # because there is no Bokeh document. So we warm the basis cache first,
        # which forces the synchronous path and gives us a real model to drive
        # the rest of the UI with. The async delivery path itself is therefore
        # NOT covered by this test; only a browser can cover it.
        from ion_gym.physics.sim_build import build_run
        build_run(app.spec, verbose=False)          # warm fa_cache
        app._draw_background()                      # now takes the fast path
        assert app._model is not None, (
            "no model after a warmed build — the inline path is broken")
        assert hasattr(app._model, "pe_surface"), "model exposes no pe_surface"
    check("solve (inline path; async delivery NOT covered)", fly)

    # ---- PE Surface tab: EVERY combination ------------------------------
    def pe_tab():
        t = app._pe_tab
        planes = ["xy", "xz", "yz"] if app.spec.geometry.depth_mm > 0 else ["xy"]
        # a dict-Select's .options yields LABELS; the widget's VALUE is the
        # dict value. Iterating .options set the widget to a label and the
        # crash that produced was mine, not the app's.
        comps = (list(t.w_comp.options.values())
                 if isinstance(t.w_comp.options, dict) else t.w_comp.options)
        for plane in planes:
            t.w_plane.value = plane
            for comp in comps:
                for metal in ("mask", "dc", "barrier"):
                    t.w_comp.value = comp
                    t.w_metal.value = metal
                    t.compute()              # THE button that raised
                    s = str(t.status.object or "")
                    if "error" in s.lower():
                        raise RuntimeError(f"{plane}/{comp}/{metal}: {s}")
                    if "no solved field" in s:
                        raise RuntimeError(
                            f"{plane}/{comp}/{metal}: compute() bailed early "
                            f"— this path was never exercised: {s}")
        for tc in (1, 4):
            t.w_trust.value = tc
            t.compute()
    check("PE tab: planes x components x metal conventions x trust", pe_tab)

    # ---- DC groups ------------------------------------------------------
    def dc_groups():
        if not app.spec.geometry.dc_groups:
            return                              # funnel has none: nothing to do
        w = app._dcg_widgets["LADDER"]
        w["v_in"].value = 40.0
        w["v_out"].value = 0.0
        app._refresh_dc_derived()
        mem = [e for e in app.spec.geometry.electrodes if e.dc_group]
        assert abs(mem[0].dc - 40.0) < 1e-9, f"v_in not applied: {mem[0].dc}"
        assert abs(mem[-1].dc - 0.0) < 1e-9, f"v_out not applied: {mem[-1].dc}"
        assert w["derived"].object, "derived voltages not displayed"
        app._on_autonumber_dc()                 # needs a solved model
    check("DC group: set ends, derived readout, auto-number", dc_groups)

    def add_group():
        app.w_dcg_new.value = "SMOKE_GRP"
        app._on_add_dc_group()
        assert any(g.name == "SMOKE_GRP" for g in app.spec.geometry.dc_groups)
    check("add DC group", add_group)

    # ---- Config tab: JSON round trip + resolution ------------------------
    def json_tab():
        txt = app.spec.to_json()
        app.w_json.value = txt
        app._on_apply_json()
        app._refresh_sizing()
        assert app.w_sizing.object, "no cost estimate rendered"
        app.w_pitch.value = float(app.spec.geometry.mm_per_gu) * 2.0
        app._on_apply_pitch()
    check("Config: apply JSON, estimate cost, apply resolution", json_tab)

    # ---- Config tab STRUCTURE: the restructure must actually hold ---------
    # The Config tab was one flat column with the cost readout BELOW the JSON
    # box -- so you committed to a geometry, then scrolled down to find out
    # what it cost. It is now name+notes pinned on top, then Load/Save/Runs
    # subtabs, with the estimator sitting WITH the loaders. None of that is
    # verified by calling handlers, so it is asserted structurally: a layout
    # regression is invisible to every other test in this file.
    def config_structure():
        names = [t.name for t in app.w_cfgtabs]
        assert names == ["Load", "Save", "Runs"], names

        def walk(o):
            yield o
            for c in getattr(o, "objects", []) or []:
                yield from walk(c)

        load = list(walk(app.w_cfgtabs[0]))
        # the cost estimator and its controls live ON the Load tab, with the
        # loaders -- that is the whole point of the restructure
        for w, nm in ((app.w_sizing, "cost readout"),
                      (app.w_pitch, "pitch"),
                      (app.w_examples, "example selector"),
                      (app.w_json, "JSON box")):
            assert any(o is w for o in load), f"{nm} is not on the Load tab"
        # save lives on its OWN tab -- it used to sit next to the examples,
        # which made "load an example" and "save as" read as one control
        save = list(walk(app.w_cfgtabs[1]))
        assert any(o is app.w_download for o in save), "download not on Save"
        assert not any(o is app.w_download for o in load), \
            "download is still on the Load tab"
        # name + notes are PINNED above the subtabs, not inside one
        for t in app.w_cfgtabs:
            assert not any(o is app.w_name for o in walk(t)), \
                "name is buried inside a subtab"
    check("Config: subtab structure (cost with the loaders)", config_structure)

    # ---- selecting an example STAGES it; it must NOT apply ---------------
    def example_staging():
        before = app.spec
        opts = [o for o in app.w_examples.options if o != "(keep current)"]
        if not opts:
            return
        app.w_examples.value = opts[0]           # fires _on_example_selected
        assert app.spec is before, \
            "selecting an example APPLIED it — it must only stage"
        assert app.w_sizing.object, "no cost shown for the staged example"
        staged = app.w_json.value
        app._on_load_example()                   # now commit
        assert app.w_json.value == staged or app.spec is not before, \
            "Load example did not apply the staged spec"
    check("Config: example stages (prices) before it commits", example_staging)

    # ---- staged multi-FA assembly: load, switch stage, fly --------------
    # Coverage is by CALLBACK (see this file's header), and these three
    # handlers are new. Every defect they caught in development was
    # invisible to parsing and to imports: a missing module-level import,
    # state wiped by a control rebuild, and a widget's value being the
    # wrong authority. Only invoking them finds that class.
    def assembly_load_switch_fly():
        from ion_gym.io.paths import repo_root
        # SHIPPED assembly fixture (the
        # public tree may not reference internals; the old fixture was an
        # internal superseded instrument, so a public checkout silently
        # skipped this route). The demo assembly ships, so absence is now
        # a failure, not a skip.
        inst = (repo_root() / "examples" / "funnel_hexapole"
                / "instrument.json")
        assert inst.exists(), \
            f"shipped assembly fixture missing: {inst}"
        app.w_json.value = inst.read_text()
        assert app._sniff_assembly(app.w_json.value), \
            "a staged instrument was not recognised as one"
        app._on_apply_json()                     # fires the assembly route
        names = list(app.w_stage.options)
        assert len(names) >= 2, f"expected >=2 stages, got {names}"
        assert not app.w_fly_assembly.disabled, \
            "Fly assembly stayed disabled after an assembly loaded"
        assert app._assembly_specs, \
            "assembly state did not survive the control rebuild"
        # STAGE SWITCH, per the session-5 contract (GATE AMENDED
        # the dropdown is SELECTION-ONLY —
        # _on_stage_selected records a choice and *Set FA View*
        # (_apply_fa_view) applies it; the old auto-switch was retired
        # as the 89-rebuild reentrancy class. This gate lagged that
        # change AND its old detector was fixture-dependent twice over:
        # names[-1] is now "Full Assembly" (whole-assembly subject,
        # which leaves the live spec untouched BY DESIGN), and the
        # electrode-count proxy is blind on any fixture whose stages
        # share a count — the shipped public demo's do. The detector
        # below holds for EVERY assembly: applying a different REAL
        # stage must install a different live spec object and record
        # that stage as live.
        stage_names = [n for n in names if n in app._assembly_specs]
        assert len(stage_names) >= 2, \
            f"expected >=2 real stages, got {stage_names}"
        spec_before = app.spec
        target = (stage_names[-1]
                  if getattr(app, "_live_stage_name", None) != stage_names[-1]
                  else stage_names[0])
        app.w_stage.value = target               # select (records only)
        assert app.spec is spec_before, \
            "selecting a stage APPLIED it — selection must only record"
        app._apply_fa_view()                     # the *Set FA View* press
        assert app._live_stage_name == target, \
            (f"Set FA View did not make {target!r} live "
             f"(live: {getattr(app, '_live_stage_name', None)!r})")
        assert app.spec is not spec_before, \
            "switching stage did not change the live spec"
        app._on_fly_assembly()                   # LAUNCH (async contract)
        # HEADLESS JOIN (gate amendment): the flight
        # moved to a worker thread + periodic poll (session-5 hang fix),
        # and the periodic callback never fires without a browser — this
        # file's declared headless boundary. So the gate cranks the REAL
        # poll by hand until the worker reports done: the same delivery
        # path a browser runs, minus only the timer. Progress is printed
        # as it streams (a silent multi-minute cell is a hang to the
        # reader).
        import time as _time
        st = app._afly
        _deadline = _time.time() + 600.0         # cold 3-D compile bound
        _last = None
        while _time.time() < _deadline:
            app._afly_poll()                     # drain + finish-on-done
            cur = str(app.status.object)
            if cur != _last:
                print(f"    [fly assembly] {cur[:90]}", flush=True)
                _last = cur
            if st["done"]:
                break
            _time.sleep(0.5)
        assert st["done"], "assembly flight did not complete in 600 s"
        app._afly_poll()                         # final drain after done
        msg = str(app.status.object)
        assert "failed" not in msg.lower(), f"assembly flight failed: {msg}"
        assert "arrived" in msg or "fate" in msg, \
            f"flight reported no outcome: {msg}"
    check("Config: staged assembly loads, switches stage, and flies",
          assembly_load_switch_fly)

    def assembly_refuses_bad_input():
        """The refusals are the point: a stage that names a path, or a
        stage list with duplicate names, must not load quietly."""
        import json as _json
        bad = {"schema": "ion_gym.assembly/1",
               "stages": [{"name": "a", "spec": "somewhere/spec.json"}],
               "beam": {"mz": 1000.0, "p0_mm": [0, 0, 0],
                        "v0_mm_us": [0, 0, 1]}}
        try:
            app._load_assembly_text(_json.dumps(bad))
            raise AssertionError("a path-valued stage spec was accepted")
        except TypeError:
            pass
        dup = {"schema": "ion_gym.assembly/1",
               "stages": [{"name": "a", "spec": {}}, {"name": "a", "spec": {}}],
               "beam": {"mz": 1000.0, "p0_mm": [0, 0, 0],
                        "v0_mm_us": [0, 0, 1]}}
        try:
            app._load_assembly_text(_json.dumps(dup))
            raise AssertionError("duplicate stage names were accepted")
        except (ValueError, KeyError):
            pass
    check("Config: staged assembly refuses a path spec and duplicate names",
          assembly_refuses_bad_input)

    def fly_assembly_without_one():
        """Pressing Fly with no assembly loaded must say so, not raise."""
        app._assembly_doc = None
        app._on_fly_assembly()
        assert "no assembly" in str(app.status.object).lower(), \
            f"unhelpful status with no assembly: {app.status.object}"
    check("Config: Fly assembly with nothing loaded reports, not raises",
          fly_assembly_without_one)

    check("Reset app", lambda: app._on_reset())


for label, mk in FIXTURES:
    run_for(label, mk)


def run_station_and_retune_guards():
    """Standalone section for the 2026-09-12 UI fixes (station
    write-through + retune-in-place). Tiny inline decks — nothing here
    may trigger a heavy solve. Each refusal path added by the fix gets
    its adversarial check (chartered: a guard ships with its test)."""
    from ion_gym.io.sim_spec import SimSpec
    from ion_gym.ui.sim_app import SimApp
    doc = {"name": "guard-tiny", "geometry": {
            "width_mm": 12.0, "height_mm": 8.0, "mm_per_gu": 0.5,
            "electrodes": [
                {"name": f"E{i+1}", "dc": 0.0, "shapes": [
                    {"type": "rect", "x_mm": 1.0 + 2.0 * i, "y_mm": 1.0,
                     "width_mm": 1.0, "height_mm": 1.0}]}
                for i in range(4)]},
           "source": {"x0_mm": 6.0, "y0_mm": 5.0, "mz_list": [100.0],
                      "n_ions": 2, "seed": 3},
           "stations": [
            {"name": "DET1", "kind": "detect", "on_hit": "pass",
             "axis": "x", "pos_mm": 6.0, "window": {}},
            {"name": "DET2", "kind": "detect", "on_hit": "pass",
             "axis": "x", "pos_mm": 9.0, "window": {}}]}
    print("\n=== station_writethrough_and_retune_guards ===")
    app = SimApp(SimSpec.from_dict(doc))
    app._station_editor()

    def station_writethrough_survives_sync():
        """THE REPORTED DEFECT: an on_hit flip with no Apply press must
        register in the spec and survive the next editor sync instead of
        silently reverting to the spec's old value."""
        app.w_stn_onhit.value = "splat"
        st = next(s for s in app.spec.stations if s.name == "DET1")
        assert st.on_hit == "splat", "write-through did not reach the spec"
        app._station_sync_pick()          # the old resetter
        assert app.w_stn_onhit.value == "splat", \
            "editor sync snapped the widget back"
        st = next(s for s in app.spec.stations if s.name == "DET1")
        assert st.on_hit == "splat", "editor sync reverted the spec"
        import json as _json
        jd = _json.loads(app.w_json.value)
        assert next(s for s in jd["stations"]
                    if s["name"] == "DET1")["on_hit"] == "splat", \
            "JSON box does not carry the write-through"
        assert [s.name for s in app.spec.stations] == ["DET1", "DET2"], \
            "station edit shuffled deck order"
    check("station: on_hit write-through survives editor sync",
          station_writethrough_survives_sync)

    def station_load_is_not_an_edit():
        """Picking a station populates the editor programmatically; that
        load must not itself write (or the read-back would masquerade as
        an edit and mark the document modified)."""
        app.w_stn_pick.value = "DET2"
        st = next(s for s in app.spec.stations if s.name == "DET2")
        assert st.on_hit == "pass" and app.w_stn_onhit.value == "pass", \
            "programmatic load wrote through"
    check("station: programmatic load does not write", 
          station_load_is_not_an_edit)

    def station_invalid_window_reported_not_written():
        """ADVERSARIAL on the new refusal: a mid-edit invalid window
        (lo >= hi) must be reported and NOT written; the next valid
        change writes."""
        app.w_stn_w1_on.value = True                 # valid -5..5 writes
        app.w_stn_w1_lo.value = 7.0                  # lo > hi: refuse
        st = next(s for s in app.spec.stations if s.name == "DET2")
        assert "not written" in str(app.w_stn_msg.object), \
            "invalid window was not reported"
        assert (st.window or {}).get("y", [-5, 5])[0] == -5.0, \
            "invalid window was WRITTEN"
        app.w_stn_w1_hi.value = 9.0                  # valid again
        st = next(s for s in app.spec.stations if s.name == "DET2")
        assert (st.window or {}).get("y") == [7.0, 9.0], \
            "recovered valid window did not write"
    check("station: invalid window reported, not written",
          station_invalid_window_reported_not_written)

    def station_new_needs_apply():
        """'(new)' + field edits must not create a station per keystroke
        — creation is the explicit Apply action."""
        app.w_stn_pick.value = "(new)"
        n0 = len(app.spec.stations)
        app.w_stn_pos.value = 3.3
        assert len(app.spec.stations) == n0, \
            "a field edit under '(new)' created a station"
    check("station: '(new)' creates only via Apply", station_new_needs_apply)

    def station_edit_leaves_build_sig():
        """Stations are fly-time config (the fly closure re-reads the
        live spec); a station edit must not change the build signature,
        or every station edit re-runs the channel compose."""
        sig_a = app._build_sig(app.spec)
        app.w_stn_pick.value = "DET1"
        app.w_stn_onhit.value = "pass"
        sig_b = app._build_sig(app.spec)
        assert sig_a == sig_b, "a station edit changed the build sig"
    check("station: edits leave the build signature unchanged",
          station_edit_leaves_build_sig)

    def retune_updates_in_place():
        """A retune is a VALUE change: the spec and the per-group editor
        widgets update in place, with NO control-column rebuild (the
        rebuild was the scroll-reset + stale-widget freeze) and the
        builder boxes keeping what the user typed."""
        app.w_tw_members.value = list(app.w_tw_members.options)[:4]
        app.w_tw_nphase.value = 4
        app.w_tw_amp.value = 30.0
        app.w_tw_freq.value = 20.0
        app._on_build_tw()                # structural: rebuild expected
        gnames = [g.name for g in app.spec.geometry.rf_groups]
        assert len(gnames) == 4, f"ladder build failed: {gnames}"
        tabs_children = [id(t) for t in app.tabs]
        amp_box = app.w_tw_amp
        grp_amp = {n: app._grp_widgets[n]["amp"] for n in gnames}
        app.w_tw_amp.value = 55.0
        app.w_tw_freq.value = 35.0
        app._on_retune_tw()
        assert all(abs(g.amplitude_v - 55.0) < 1e-9
                   and abs(g.frequency_hz - 35000.0) < 1e-9
                   for g in app.spec.geometry.rf_groups), \
            "retune did not reach the spec"
        assert all(app._grp_widgets[n]["amp"] is grp_amp[n]
                   and abs(app._grp_widgets[n]["amp"].value - 55.0) < 1e-9
                   for n in gnames), \
            "group editors were rebuilt or not refreshed"
        assert [id(t) for t in app.tabs] == tabs_children, \
            "retune rebuilt the control column"
        assert app.w_tw_amp is amp_box and app.w_tw_amp.value == 55.0, \
            "builder boxes were rebuilt/reset by retune"
    check("retune: values update in place, no column rebuild",
          retune_updates_in_place)

    def retune_divergence_refuses():
        """ADVERSARIAL on the new refusal: a group with no editor row
        (editor/spec structural divergence) must surface a diagnostic on
        the status line, never a silent partial refresh."""
        gname = app.spec.geometry.rf_groups[0].name
        saved = app._grp_widgets.pop(gname)
        try:
            app._on_retune_tw()
            msg = str(app.status.object)
            assert "no editor row" in msg and "refusing" in msg, \
                f"divergence not diagnosed: {msg}"
        finally:
            app._grp_widgets[gname] = saved
    check("retune: editor/spec divergence refuses with a diagnostic",
          retune_divergence_refuses)


run_station_and_retune_guards()

print("\n" + "=" * 66)
print(f"PASSED {len(PASSED)}   FAILED {len(FAILED)}")
for n, e in FAILED:
    print(f"  FAIL {n}: {type(e).__name__}: {e}")
print("=" * 66)
sys.exit(1 if FAILED else 0)
