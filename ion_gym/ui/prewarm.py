"""Startup prewarm for the interactive app.

Telemetry hang dumps (9 pulse-stale events) split
three ways: 5x the numba dispatcher compiling / cache-loading the r-z
kernel on the FIRST fly (importlib._path_stat under load_overload), 3x
scipy splu factorizing a large grid (that IS the solve; the app's Fly
path already runs it on a worker thread), and 1x pe_figure_3d stuck in a
lazy plotly import. The compile and import classes are pure first-use
cost: they recur every time the kernel cache is invalidated (fresh
container, any kernel signature change) and land as a 15-60 s freeze on
whatever thread first touches them.

This module pays those costs in a background daemon thread at app
startup, on a postage-stamp problem, so the first user click finds warm
kernels and loaded modules. It changes no physics and no numbers: the
same compilations happen either way; only WHEN moves.

Not handled here: the splu class (genuine compute, already off the UI
thread on the Fly path) and the PE tabs' on-UI-thread compute() wiring
(it should adopt the app's existing worker+poll
pattern, but that touches the two-parent-Bokeh history).
"""
import threading
import time


def _tiny_rz_spec():
    """Smallest r-z deck that exercises the flight kernel signatures:
    2 x 1 mm at 0.2 mm/gu (11 x 6 nodes), one electrode, one ion, 0.05 us.
    Values are warm-up scaffolding, not physics choices — nothing is
    measured on this deck."""
    from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                     ShapeSpec, SourceSpec, CollisionSpec,
                                     IntegrationSpec, SymmetrySpec)
    return SimSpec(
        name="prewarm (throwaway)",
        geometry=GeometrySpec(
            width_mm=2.0, height_mm=1.0, mm_per_gu=0.2,
            symmetry=SymmetrySpec(coords="rz"),
            electrodes=[ElectrodeSpec(name="w", dc=-1.0, shapes=[ShapeSpec(
                type="rect", params={"x_mm": 0.8, "y_mm": 0.6,
                                     "width_mm": 0.4, "height_mm": 0.2})])]),
        source=SourceSpec(n_ions=1, distribution="point", x0_mm=0.1,
                          y0_mm=0.1, z0_mm=0.0, axis="x",
                          direction=[1.0, 0.0, 0.0], ke_lo=1.0, ke_hi=1.0,
                          mz_list=[100.0]),
        collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(dt_ns=1.0, t_max_us=0.05, rec_every=1))


def prewarm(verbose: bool = True) -> None:
    """Compile the r-z flight kernels and import plotly, synchronously.
    Raises nothing it can anticipate silently: failures print loudly with
    the cause — a broken prewarm must be visible, not a mystery slow
    first click."""
    t0 = time.time()
    try:
        from ion_gym.physics.sim_build import build_run
        spec = _tiny_rz_spec()
        _model, fly, _cols, _births = build_run(spec, verbose=False)
        fly(0)                                   # compiles _fly_rec_full
        import plotly.graph_objects  # noqa: F401  (pe_figure_3d's import)
        if verbose:
            print(f"[prewarm] r-z kernels compiled + plotly loaded in "
                  f"{time.time() - t0:.1f} s (background; first click "
                  f"will not pay this)", flush=True)
    except Exception as e:                        # noqa: BLE001 — reported,
        # not swallowed: prewarm runs on a daemon thread where an
        # uncaught exception dies invisibly; printing the cause IS the
        # surfacing. The app remains fully functional (first click just
        # pays the compile), so this must never take the UI down.
        print(f"[prewarm] FAILED ({type(e).__name__}: {e}) — first "
              f"fly/figure will pay the compile/import cost instead",
              flush=True)


def start_prewarm_thread() -> threading.Thread:
    """Fire-and-forget daemon prewarm; returns the thread for tests."""
    t = threading.Thread(target=prewarm, name="ion_gym-prewarm",
                         daemon=True)
    t.start()
    return t
