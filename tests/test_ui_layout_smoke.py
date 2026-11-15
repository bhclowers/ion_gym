"""
test_ui_layout_smoke.py -- the dashboard layout after the UI cleanup,
checked headlessly, plus the regressions found while building it.

WHY THIS EXISTS
    The cleanup moved status into a bottom status bar, made failures open a
    dialog, put the control tabs on a vertical rail inside a draggable split,
    and regrouped loading, saving and runs under a Load tab. Each of those is
    a STRUCTURE (which container holds which widget) that no physics gate and
    no handler-invoking check sees: test_ui_smoke presses buttons, it does not
    ask where they live. Two defects found on the way are pinned here too:
    the r-z xz view read its axis LABELS as deck axes (the RF drift cell's
    detector drew across the transverse axis), and the marked examples
    shipped detector stations with empty windows.

    Deliberately headless, like test_ui_smoke: what a browser shows (the
    split dragging, the modal painting) is the maintainers' browser check. This gate
    asks whether the objects are wired the way the browser check assumes.

    The examples named in MARKED_DETECTOR_DECKS are the maintainers' list:
    those decks must carry a bounded detector window so the
    Impact Analysis tab has a face to draw.

Run:  python3 tests/test_ui_layout_smoke.py     (standalone; not pytest)
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path
import sys
import traceback

import panel as pn

FAILED = []
PASSED = []

# The maintainers' marked examples (dropdown names). A 2-D planar deck
# has one transverse axis in the plane; every other deck has two.
MARKED_DETECTOR_DECKS = (
    "Drift Tube (Mason-Schamp)", "Einzel Lens (r-z)",
    "Hexapole (3D, native shapes)", "Ion Funnel (r-z)",
    "Quadrupole (3D, STL)", "Reflectron oa-TOF (2D, refined)",
    "Reflectron oa-TOF (2D)", "RF Drift Cell", "SLIM (3D)")
DETECTOR_NAME = "detector"

# Top-level control tabs in workflow order.
EXPECTED_TABS = ["Load", "Ion Source", "Voltages", "Physics", "Display",
                 "Geometry Import", "Config"]
# Impact Analysis reads results, so it sits with the plots, right after
# Analysis.
EXPECTED_PLOT_TABS_HEAD = ["View", "Analysis", "Impact Analysis"]
EXPECTED_LOAD_SUBTABS = ["Deck", "Save", "Runs"]
EXPECTED_DECK_SUBTABS = ["Example", "User JSON", "Multi FA"]


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


def walk(o):
    """Every object reachable through Panel layout children."""
    yield o
    for c in getattr(o, "objects", None) or []:
        yield from walk(c)


def contains(container, obj):
    return any(x is obj for x in walk(container))


def tab_names(tabs):
    return list(tabs._names)


def main():
    pn.extension("plotly", "floatpanel", "modal")
    from ion_gym.ui import sim_app as S
    from ion_gym.ui.sim_app import SimApp, _example_specs
    from panel_splitjs import HSplit

    holder = {}

    def construct():
        holder["app"] = SimApp()
        holder["page"] = holder["app"].panel()
    check("construct the default app and its page", construct)
    if "app" not in holder:
        return
    app, page = holder["app"], holder["page"]

    # ---------------------------------------------------------- layout
    def page_order():
        objs = list(page.objects)
        splits = [o for o in objs if isinstance(o, HSplit)]
        assert len(splits) == 1, f"expected one HSplit, found {len(splits)}"
        i_split = objs.index(splits[0])
        i_bar = next(i for i, o in enumerate(objs) if o is app._status_bar)
        assert i_bar == i_split + 1, \
            "status bar is not the row directly under the split"
        assert any(o is app._error_modal for o in objs), \
            "error dialog is not on the page"
        assert any(o is app._history_host for o in objs), \
            "history host is not on the page"
    check("page: top bar / split / status bar, dialog and history host",
          page_order)

    def split_minimum():
        split = next(o for o in page.objects if isinstance(o, HSplit))
        lo = split.min_size[0]
        need = S.CONTROL_COL_PX + S.TAB_RAIL_PX
        assert lo >= need, f"left minimum {lo} < fitted column {need}"
        assert contains(split, app.tabs), "control tabs are not in the split"
    check("split: control pane minimum holds the fitted column", split_minimum)

    def rail():
        assert app.tabs.tabs_location == "left", app.tabs.tabs_location
        assert tab_names(app.tabs) == EXPECTED_TABS, tab_names(app.tabs)
    check("control tabs: vertical rail, workflow order, no Status tab", rail)

    def impact_home():
        names = tab_names(app.plot_tabs)
        assert names[:3] == EXPECTED_PLOT_TABS_HEAD, names
        wrap = app.plot_tabs.objects[names.index("Impact Analysis")]
        assert contains(wrap, app._impact_col)
        assert not contains(app.tabs, app._impact_col), \
            "Impact Analysis is also in the control tabs (two parents)"
        btns = [w for w in wrap[0].objects
                if isinstance(w, pn.widgets.Button) and w.name == "?"]
        assert len(btns) == 1, f"{len(btns)} help buttons on Impact"
    check("Impact Analysis: plot side after Analysis, with its help", impact_home)

    def status_home():
        assert contains(app._status_bar, app.status), \
            "status line is not in the status bar"
        assert not contains(app.tabs, app.status), \
            "status line is also inside the control tabs (two parents)"
        assert contains(app._status_bar, app.w_advisory)
        assert contains(app._status_bar, app._history_btn)
    check("status bar holds status, advisories and History", status_home)

    def load_structure():
        load_tab = app.tabs.objects[EXPECTED_TABS.index("Load")]
        assert contains(load_tab, app.w_cfgtabs)
        assert tab_names(app.w_cfgtabs) == EXPECTED_LOAD_SUBTABS, \
            tab_names(app.w_cfgtabs)
        assert tab_names(app._load_inner) == EXPECTED_DECK_SUBTABS, \
            tab_names(app._load_inner)
        assert contains(app._load_inner, app._multifa_tab)
        assert contains(app._load_inner, app.w_examples)
        assert contains(app._load_inner, app.w_json)
        assert contains(load_tab, app.w_name) and \
            not contains(app.w_cfgtabs, app.w_name), \
            "name must sit above the Load sub-tabs, not inside one"
    check("Load tab: Deck/Save/Runs, Deck has Example/User JSON/Multi FA",
          load_structure)

    def help_buttons():
        from ion_gym.ui.help_text import TAB_HELP, help_for
        assert any(o is app._help_modal for o in page.objects), \
            "help dialog is not on the page"
        for i, title in enumerate(EXPECTED_TABS):
            wrap = app.tabs.objects[i]
            btns = [w for w in wrap[0].objects
                    if isinstance(w, pn.widgets.Button) and w.name == "?"]
            assert len(btns) == 1, f"{title}: {len(btns)} help buttons"
            app._help_modal.open = False
            btns[0].clicks += 1
            assert app._help_modal.open, f"{title}: help dialog not opened"
            assert app._help_body.object == TAB_HELP[title], \
                f"{title}: dialog shows another tab's note"
        app._help_modal.open = False
        try:
            help_for("No Such Tab")
        except KeyError:
            pass
        else:
            raise AssertionError("help_for accepted an unknown tab title")
    check("help: one ? per control tab opens that tab's note; unknown "
          "title refuses", help_buttons)

    def help_survives_reload():
        before = list(app.tabs.objects)
        app._rebuild_for_new_spec(box_from_loaded=True)
        after = list(app.tabs.objects)
        assert [id(o) for o in before] == [id(o) for o in after], \
            "a reload replaced the help wrappers (two-parent risk)"
        assert contains(after[EXPECTED_TABS.index("Ion Source")],
                        app._ionsrc_col)
    check("help wrappers persist across a reload", help_survives_reload)

    def field_chip():
        top = page.objects[0]
        assert contains(top, app.field_chip), "field chip not in the top bar"
        saved = app._model
        try:
            app._model = None
            app._field_chip()
            assert "geometry preview" in app.field_chip.object
            app._model = object()
            app._field_chip()
            assert "field built" in app.field_chip.object
        finally:
            app._model = saved
            app._field_chip()
    check("field chip: top bar, preview vs built follows the model",
          field_chip)

    def deck_diff_list_matches_writer():
        import inspect, re as _re
        src = inspect.getsource(S.SimApp._widgets_into_spec)
        read = set(_re.findall(r"self\.(w_[A-Za-z0-9_]+)", src))
        grouped = {"w_bnd"}          # enumerated from its dict
        missing = read - set(S.SPEC_WIDGET_ATTRS) - grouped
        extra = set(S.SPEC_WIDGET_ATTRS) - read
        assert not missing, f"writer reads controls not listed: {missing}"
        assert not extra, f"listed controls the writer never reads: {extra}"
    check("deck diff: SPEC_WIDGET_ATTRS is exactly what the writer reads",
          deck_diff_list_matches_writer)

    def deck_diff_edit_and_revert():
        w = app.w_T
        v0 = w.value
        assert app.deck_chip.object == "", repr(app.deck_chip.object)
        try:
            w.value = v0 + 10
            assert "1 changed since load" in app.deck_chip.object, \
                app.deck_chip.object
            assert all(w.styles.get(k) == v for k, v in
                       S.DECK_DIFF_STYLE.items()), w.styles
        finally:
            w.value = v0
        assert app.deck_chip.object == "", repr(app.deck_chip.object)
        assert not any(k in (w.styles or {}) for k in S.DECK_DIFF_STYLE)
    check("deck diff: an edit outlines its control and counts; reverting "
          "clears both", deck_diff_edit_and_revert)

    def config_contents():
        cfg = app.tabs.objects[EXPECTED_TABS.index("Config")]
        assert contains(cfg, app.w_workers) and contains(cfg, app.w_ping)
        assert not contains(cfg, app.w_cfgtabs), \
            "Load/Save/Runs still inside Config"
    check("Config tab: machine resources and diagnostics only",
          config_contents)

    def deck_tail_rule():
        inner, tail = app._load_inner, app._deck_tail
        members = app._deck_tail_members
        saved_active, saved_doc = inner.active, app._assembly_doc
        i_mfa = EXPECTED_DECK_SUBTABS.index("Multi FA")
        try:
            app._assembly_doc = None
            inner.active = i_mfa
            assert len(tail.objects) == 0, \
                "cost card shown on an empty Multi FA sub-tab"
            inner.active = 0
            assert [id(o) for o in tail.objects] == [id(o) for o in members]
            app._assembly_doc = {"stages": []}
            inner.active = i_mfa
            assert [id(o) for o in tail.objects] == [id(o) for o in members], \
                "cost card hidden although an instrument is loaded"
        finally:
            app._assembly_doc = saved_doc
            inner.active = saved_active
            app._sync_deck_tail_visibility()
    check("Deck sub-tab: cost card hidden only on an empty Multi FA",
          deck_tail_rule)

    # ---------------------------------------------------------- dialogs
    def error_dialog():
        app._error_modal.open = False
        app._report_error("first failure", "**first:** detail one")
        assert app._error_modal.open, "dialog did not open"
        assert app.status.object == "**first:** detail one"
        app._report_error("second failure", "**second:** detail two")
        assert len(app._error_body.objects) == 2, \
            "second failure did not stack in the same dialog"
        txt = app._error_copy_text.value
        assert "detail one" in txt and "detail two" in txt, txt
        app._error_modal.open = False
        assert app._error_items == [], "closing did not clear the dialog"
    check("error dialog: opens, stacks, copies, clears on close",
          error_dialog)

    def report_exception():
        app._error_modal.open = False
        try:
            raise ValueError("planted")
        except ValueError as e:
            app._report_exception("planted failure", e)
        assert app._error_modal.open and "planted" in str(app.status.object)
        app._error_modal.open = False
    check("error dialog: _report_exception carries the exception text",
          report_exception)

    def history_panel():
        app._history_host[:] = []
        app._on_open_history()
        assert len(app._history_host) == 1
        first = app._history_host[0]
        app._on_open_history()
        assert len(app._history_host) == 1 and app._history_host[0] is first, \
            "a second press stacked another panel"
        first.status = "closed"
        assert len(app._history_host) == 0, "closed panel stayed on the page"
        app._on_open_history()
        assert len(app._history_host) == 1 and app._history_host[0] is not first
        assert sum(contains(f, app._status_log_pane)
                   for f in app._history_host) == 1
        app._history_host[0].status = "closed"
    check("history panel: one at a time, closes off the page, reopens",
          history_panel)

    def history_log():
        n0 = len(app._status_log)
        app.status.object = "**layout smoke** history probe"
        assert len(app._status_log) == n0 + 1 or \
            len(app._status_log) == S.STATUS_LOG_MAX
        assert "history probe" in app._status_log[0]
    check("status writes land in the session history", history_log)

    # ---------------------------------------------------------- reset
    def reset_restores():
        assert app.reset_btn.name == "Reset", app.reset_btn.name
        initial = app.spec.name
        specs = _example_specs()
        other = next(n for n in specs if n != initial
                     and "r-z" in n and "Funnel" not in n)
        app.spec = specs[other]()
        app._rebuild_for_new_spec(solve=False, box_from_loaded=True)
        assert app.spec.name != initial
        app._on_reset()       # headless: no document, so it runs inline
        assert app.spec.name == initial, (app.spec.name, initial)
        assert app.pane.loading is False, "reset left the spinner on"
    check("Reset: label, restores the initial deck, clears the spinner",
          reset_restores)

    def reset_failure_dialog():
        """ADVERSARIAL: a rebuild that raises inside Reset must clear the
        spinner and open the error dialog."""
        app._error_modal.open = False
        real = app._rebuild_for_new_spec

        def boom(*a, **k):
            raise RuntimeError("planted reset failure")
        app._rebuild_for_new_spec = boom
        try:
            app._on_reset()
        finally:
            app._rebuild_for_new_spec = real
        assert app.pane.loading is False, "spinner outlived the failure"
        assert app._error_modal.open, "failed Reset opened no dialog"
        assert "planted reset failure" in str(app.status.object)
        app._error_modal.open = False
        app._on_reset()       # leave the app on its initial deck
    check("Reset: a failed rebuild clears the spinner and opens the dialog",
          reset_failure_dialog)

    # ---------------------------------------------------------- axes
    def deck_axes_rz():
        assert app.spec.geometry.coords == "rz"
        assert app._deck_axes("z", "x") == ("x", "y")
        assert app._deck_axes("x", "y") == ("x", "y")
        assert app._deck_axes("z", "y") == ("z", "y")
    check("deck axes: r-z xz view draws the deck's (x, y) plane",
          deck_axes_rz)

    def deck_axes_refuses():
        try:
            app._deck_axes("q", "x")
        except ValueError:
            return
        raise AssertionError("an unknown label pair was not refused")
    check("deck axes: an unknown label pair is refused", deck_axes_refuses)

    def deck_axes_3d():
        saved = app.spec
        try:
            app.spec = _example_specs()["Hexapole (3D, native shapes)"]()
            assert app._deck_axes("z", "x") == ("z", "x")
        finally:
            app.spec = saved
    check("deck axes: 3-D labels are the deck axes", deck_axes_3d)

    def rz_detector_vertical():
        """The RF drift cell regression: in the r-z xz view the detector
        (axis x) must draw as a VERTICAL segment at its x position."""
        import plotly.graph_objects as go
        from ion_gym.viz import viz_core as V
        spec = _example_specs()["RF Drift Cell"]()
        st = next(s for s in spec.stations if s.name == DETECTOR_NAME)
        saved = app.spec
        try:
            app.spec = spec
            da, db = app._deck_axes("z", "x")
        finally:
            app.spec = saved
        fig = go.Figure()
        n = V.station_overlays(fig, spec, da, db, label=False)
        assert n >= 1
        tr = next(t for t in fig.data if DETECTOR_NAME in (t.name or ""))
        assert list(tr.x) == [st.pos_mm, st.pos_mm], list(tr.x)
        lo, hi = st.window["y"]
        assert list(tr.y) == [lo, hi], list(tr.y)
    check("r-z xz view: RF drift cell detector is vertical at its x",
          rz_detector_vertical)

    # ---------------------------------------------------------- decks
    def detector_windows():
        specs = _example_specs()
        missing = [n for n in MARKED_DETECTOR_DECKS if n not in specs]
        assert not missing, f"marked decks not in the example menu: {missing}"
        bad = []
        for n in MARKED_DETECTOR_DECKS:
            spec = specs[n]()
            st = [s for s in spec.stations if s.name == DETECTOR_NAME]
            if len(st) != 1:
                bad.append(f"{n}: {len(st)} detector stations")
                continue
            st = st[0]
            planar = (spec.geometry.coords != "rz"
                      and not float(spec.geometry.depth_mm or 0.0))
            trans = [a for a in "xyz" if a != st.axis]
            if planar:
                trans = [a for a in trans if a != "z"]
            win = st.window or {}
            for a in trans:
                if a not in win:
                    bad.append(f"{n}: no {a} window")
                elif not win[a][0] < win[a][1]:
                    bad.append(f"{n}: {a} window {win[a]} is not lo < hi")
        assert not bad, "; ".join(bad)
    check("marked examples: detector windows bounded on every transverse "
          "axis", detector_windows)

    # ---------------------------------------------------------- viewer
    def plasma_default():
        from ion_gym.edit import flight_view as F
        from pathlib import Path
        assert F.DEFAULT_CMAP == "plasma", F.DEFAULT_CMAP
        js = Path(F.__file__).with_name("viewer.py").read_text(
            encoding="utf-8")
        assert "st.cmap || 'plasma'" in js, \
            "viewer JS fallback is not the declared default"
    check("3D Flight Viewer: plasma is the default colormap", plasma_default)


if __name__ == "__main__":
    main()
    print("\n" + "=" * 66)
    print(f"PASSED {len(PASSED)}   FAILED {len(FAILED)}")
    for n, e in FAILED:
        print(f"  FAIL {n}: {type(e).__name__}: {e}")
    print("=" * 66)
    sys.exit(1 if FAILED else 0)
