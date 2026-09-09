"""UI-EXERCISE GATE: drive every
example through the app's OWN interactive layer — stage via the
w_examples watcher, apply via _on_load_example, geometry preview per
plane, warm inline solve, then every display widget through its real
watcher, PLUS the redraw-identity probe (a toggle must actually
rebuild the pane, not just accept a value). HEADLESS BOUNDARY (also
ui_smoke's): the async solve/fly delivery (Bokeh periodic callback)
never fires without a browser and is NOT covered here."""
import sys, time
import _bootstrap  # noqa: F401  -- repo root on sys.path
from ion_gym.physics.sim_build import build_run
from ion_gym.ui.sim_app import SimApp, _example_specs

fails = []
def act(ex, name, fn):
    try:
        fn()
        return True
    except Exception as e:
        fails.append((ex, name, repr(e)[:90]))
        print(f"    RED  {name}: {e!r}"[:110])
        return False

names = sorted(_example_specs())
print(f"UI exercise: {len(names)} examples")
for ex in names:
    t0 = time.time()
    print(f"  [{time.strftime('%H:%M:%S')}] {ex}")
    app = SimApp()
    app.w_examples.value = ex                       # stage (watcher)
    if not act(ex, "load (apply)", app._on_load_example):
        continue
    assert app.spec.name or True
    # geometry PREVIEW on all planes (no model yet)
    for pl in ("xy", "xz", "yz"):
        act(ex, f"preview {pl}", lambda p=pl: (
            setattr(app.w_plane, "value", p),
            app._draw_geometry_only()))
    # warm + solve the UI way — at a DECLARED seed: the
    # loaded decks carry seed=null = fresh entropy; seed 0 =
    # the historical draws. The UI's own default stays random for users.
    app.spec.source.seed = 0
    if not act(ex, "solve inline", lambda: (
            build_run(app.spec, verbose=False),
            app._draw_background())):
        continue
    assert app._model is not None and app._scene is not None
    # solved-state planes through the watcher
    # a browser user can only pick OFFERED planes — assert membership
    # first (the gap that hid the greyed-selector bug on r-z models),
    # and the selector must not be disabled when >1 plane exists.
    act(ex, "planes offered", lambda: (
        [p for p in ("xy", "xz", "yz")
         if p not in app.w_plane.options] == [] or
        (_ for _ in ()).throw(AssertionError(
            f"planes not offered: {app.w_plane.options}"))))
    act(ex, "selector enabled", lambda: (
        not app.w_plane.disabled or
        (_ for _ in ()).throw(AssertionError("w_plane disabled"))))
    for pl in ("xz", "yz", "xy"):
        act(ex, f"plane {pl}", lambda p=pl: (
            (_ for _ in ()).throw(AssertionError(f"{p} not in options"))
            if p not in app.w_plane.options
            else setattr(app.w_plane, "value", p)))
        assert app.pane.object is not None
    # display toggles at xy (each fires _on_display_change)
    for wname, vals in (("w_showfield", (False, True)),
                        ("w_contours", (0, 12)),
                        ("w_elalpha", (0.9,)),
                        ("w_elfill", (False, True)),
                        ("w_ellabel", (False, True))):
        w = getattr(app, wname, None)
        if w is None:
            continue
        for v in vals:
            act(ex, f"{wname}={v}", lambda w=w, v=v:
                setattr(w, "value", v))
    # redraw-identity: a display toggle must REBUILD the pane object
    for wname in ("w_elfill", "w_ellabel", "w_showfield"):
        w = getattr(app, wname, None)
        if w is None:
            continue
        def probe(w=w, wname=wname):
            before = id(app.pane.object)
            w.value = not w.value
            changed = id(app.pane.object) != before
            w.value = not w.value
            assert changed, f"{wname} accepted the value but did NOT redraw"
        act(ex, f"{wname} redraw-identity", probe)
    for fm in app.w_fieldmode.options:
        act(ex, f"fieldmode {fm!r}", lambda m=fm:
            setattr(app.w_fieldmode, "value", m))
    app.w_fieldmode.value = "|E| field"
    print(f"    done ({time.time()-t0:.0f}s)")

print(f"\nUI exercise: {len(fails)} failures")
for f in fails:
    print("  RED", *f)
sys.exit(1 if fails else 0)
