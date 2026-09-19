"""
test_pe_surface_tab.py — gate the compute-once / self-invalidating PE
Surface tab.

Uses the r-z einzel (the honest rotationally-symmetric lens). Asserts:
  * pre-solve guard (no model -> prompt, no crash),
  * compute builds a surface figure with draped trajectories and caches
    it with a signature,
  * a no-op refresh KEEPS the cache,
  * each surface-affecting change (mass, a voltage, resolution) DESTROYS
    the cache and prompts recompute,
  * component reduction: for a DC-only device effective == DC.
"""
from ion_gym.viz import viz_core as V
import _bootstrap  # noqa: F401  -- repo root on sys.path
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from bokeh.document import Document
from ion_gym.ui.sim_app import SimApp
# Examples come from JSON, not from a package builder.
from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec


def einzel_rz_spec(lens_v=-120.0, mm_per_gu=0.1, n_ions=12):
    """The SHIPPED einzel deck, retuned. Voltage-only changes reweight
    cached bases; the geometry is the deck's."""
    import pathlib
    sp = SimSpec.from_json(str(pathlib.Path(repo_root()) / "examples"
                               / "einzel_round_r-z.json"))
    # DECLARED seed: the shipped deck carries seed=null =
    # fresh entropy; seed 0 = the historical deterministic draws.
    sp.source.seed = 0
    sp.geometry.mm_per_gu = mm_per_gu
    sp.source.n_ions = n_ions
    for e in sp.geometry.electrodes:
        if e.name in ("lens", "centre"):
            e.dc = float(lens_v)
    return sp
from ion_gym.physics.sim_build import build_run
from ion_gym.physics.ensemble_driver import run
from ion_gym.viz.pe_view import compute_component


def main():
    app = SimApp(einzel_rz_spec(lens_v=-150.0, n_ions=14))
    app.panel().get_root(doc=Document())
    # AMENDED: the PE Surface tab's ONE home
    # is the view-side plot_tabs (built by app.panel()). Its brief second
    # home in the left controls tabs made a two-parent Bokeh model — the
    # view copy went dead in the browser.
    app.panel()   # composes plot_tabs
    assert "PE Surface" not in [t for t in app.tabs._names], \
        "PE Surface must NOT be in the controls tabs (two-parent bug)"
    assert "PE Surface" in [t for t in app.plot_tabs._names]
    pe = app._pe_tab

    # pre-solve guard
    saved, app._model = app._model, None
    pe.compute()
    assert "no solved field" in pe.status.object.lower()
    app._model = saved

    # solve + fly so there are trajectories to drape
    m, f, c, b = build_run(app.spec)
    app._model, app._cols = m, c
    res = run(len(b), f, check_every=6, decimate=1)
    app._runs["t"] = res
    app._active = "t"
    app._redraw(res.results)

    pe.compute()
    assert pe._fig is not None and pe._sig is not None
    kinds = [t.type for t in pe.pane.object.data]
    assert "surface" in kinds, "no PE sheet"
    assert kinds.count("scatter3d") >= len(res.results), "no draped ions"
    sig0 = pe._sig

    pe.refresh()
    assert pe._sig == sig0 and pe._fig is not None, "no-op refresh dropped cache"

    pe.w_mz.value = 500.0
    assert pe._fig is None and "parameters changed" in pe.status.object, \
        "mass change did not invalidate"

    pe.compute()
    assert pe._fig is not None and pe._sig != sig0

    app.spec.geometry.electrodes[1].dc = -120.0
    pe.refresh()
    assert pe._fig is None, "voltage change did not invalidate"

    pe.compute()
    pe.w_res.value = 80
    assert pe._fig is None, "resolution change did not invalidate"

    # scaling mode change also invalidates; title reflects device+component
    pe.compute()
    assert "PE surface" in V.plotly_title(pe._fig)   # title lives in meta (L-462 layout)
    # DC einzel: no RF caveat
    assert not any("Mathieu" in (a.text or "")
                   for a in pe._fig.layout.annotations)
    pe.w_scalemode.value = "asinh"
    assert pe._fig is None, "scale-mode change did not invalidate"

    # DC-only device: effective == DC everywhere
    _, _, PEe, _ = compute_component(app._model, 500.0, 1, "effective")
    _, _, PEd, _ = compute_component(app._model, 500.0, 1, "dc")
    assert np.allclose(PEe, PEd), "einzel (no RF) must have effective==DC"

    print("  guard, compute+cache, no-op keep, mass/voltage/resolution "
          "invalidate, DC==effective  OK")


if __name__ == "__main__":
    main()
    print("\nPE SURFACE TAB GATE: ALL PASS")
