"""nb_panels — the notebook instrument intro panel, banked or live.

WHY: every teaching
and validation notebook opens by SHOWING the instrument it flies, drawn
from the solver's own mask. Those intro figures were displayed from the
banked panel PNGs under notebooks/out/_panels/, which are regenerable
state and do not ship — so on a public checkout every notebook's first
figure raised FileNotFoundError. The fix is a single framework entry
point with two paths:

* FAST PATH — the banked panel PNG, when present (a dev tree, or any
  tree where the panels have been rebuilt): display it directly.
* SELF-CONTAINED PATH — build the deck, fly a few example ions, and
  render through viz_core, exactly the content the bank carries,
  generated live. DISPLAY == SOLVER INPUT holds on both paths; the live
  path costs a few seconds on first run and is served by the solve cache
  afterwards.

Contract (ion-gym-render skill): config-driven, returns nothing but
displays through IPython at the caller's moment; no import side effects;
labels carry the operating point via the scene title.
"""
from pathlib import Path


def show_instrument_panel(deck, *, banked=None, height=520, n_ions=24,
                          title=None, views=None, seed=0):
    """Display a notebook's instrument intro panel.

    deck    : repo-relative path to the JSON deck of record.
    banked  : banked panel filename (e.g. "panel_einzel.png") to prefer
              when notebooks/out/_panels/ carries it; None disables the
              fast path.
    height  : displayed pixel height (aspect preserved).
    n_ions  : example ions flown on the live path (display-only; the
              seed is pinned so the figure is deterministic — stated
              here because the deck itself may be unseeded). The default
              is a BUNDLE, not a token few: three paths cannot show the
              spread a diffusive or space-charge-free device produces,
              and an intro figure that hides the spread misleads. On a
              multi-m/z deck this is per m/z, so a 3-mass packet flies
              3x this many. Cost at 24 is a few seconds on the measured
              decks (drift tube ~5 s, funnel ~15 s, the rest under 1 s).
    title   : overrides the scene title (default: deck name + operating
              hint).
    views   : forwarded to render_mpl (None = the route's default,
              multi-axis for 3-D scenes).
    """
    from IPython.display import Image as _PNG, display as _display
    from ion_gym.io.paths import repo_root

    root = Path(repo_root())
    if banked:
        png = root / "notebooks" / "out" / "_panels" / banked
        if png.exists():
            _display(_PNG(str(png), height=height))
            return

    # ---- self-contained path: build, fly, render -----------------------
    import io as _io
    import matplotlib.pyplot as _plt
    from ion_gym.io.sim_spec import SimSpec
    from ion_gym.physics.sim_build import build_run
    from ion_gym.viz.viz_core import scene_from_simspec, render_mpl

    spec = SimSpec.from_json(str(root / deck))
    spec.source.n_ions = int(n_ions)
    spec.source.seed = int(seed)
    model, fly, cols, births = build_run(spec)
    trajs, fates = [], []
    for i in range(births.shape[0]):
        tr, st = fly(i)
        if tr is not None and len(tr):
            trajs.append(tr)
            fates.append(str(st.get("kind", "")))
    sc = scene_from_simspec(
        spec, model, field="phi", trajs=trajs, fates=fates,
        title=title or (f"{spec.name} — deck {Path(deck).name}, "
                        f"{len(trajs)} example ion(s), rendered live"))
    fig = render_mpl(sc) if views is None else render_mpl(sc, views=views)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    _display(_PNG(buf.getvalue(), height=height))
    _plt.close(fig)
