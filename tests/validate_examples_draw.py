"""EXAMPLES-DRAW GATE: every examples/*.json builds,
solves (warm-cache pattern from test_ui_smoke), and _base_figure
returns a non-empty figure for all three planes. This is the gate the
A-3 mutation check showed to be the ONLY one catching a disabled
renderer (ui_smoke missed it)."""
import sys, time
import _bootstrap  # noqa: F401  -- repo root on sys.path
from ion_gym.physics.sim_build import build_run
from ion_gym.ui.sim_app import SimApp, _example_specs

failures = []
specs = _example_specs()
print(f"A-2 sweep: {len(specs)} examples")
for name, mk in sorted(specs.items()):
    t0 = time.time()
    try:
        app = SimApp(mk())   # spec at construction — the app's real
                             # load path builds controls FOR this spec
        # DECLARED seed: most shipped decks carry
        # seed=null = fresh entropy; seed 0 = the
        # historical draws this gate ran under. Decks stay null for users.
        app.spec.source.seed = 0
        build_run(app.spec, verbose=False)      # warm fa_cache (ui_smoke)
        app._draw_background()                  # warmed -> inline path (ui_smoke pattern)
        assert app._model is not None, "model not built"
        assert app._scene is not None, "Phase A scene not built"
        for la, lb in (("x", "y"), ("x", "z"), ("y", "z")):
            fig = app._base_figure(app._model, la, lb)
            nt = len(fig.data) + len(fig.layout.shapes or ())
            assert nt >= 1, f"{la}{lb}: empty figure"
        print(f"  [{time.strftime('%H:%M:%S')}] OK  {name}  "
              f"({time.time()-t0:.0f}s)")
    except Exception as e:
        failures.append((name, repr(e)[:100]))
        print(f"  [{time.strftime('%H:%M:%S')}] RED {name}: {e!r}"[:120])
    finally:
        # Release the app's telemetry threads. One app per deck is
        # deliberate -- this gate exercises the app's REAL load path --
        # but without this the threads live for the whole sweep, and 17
        # decks meant 17 heartbeats and 17 watchdogs printing over each
        # other and over the gate's own output. finally, so a RED deck
        # releases them too.
        app = locals().get("app")
        if app is not None:
            app.close()
print(f"\nA-2: {len(specs)-len(failures)}/{len(specs)} examples draw "
      f"on all three planes")
sys.exit(1 if failures else 0)
