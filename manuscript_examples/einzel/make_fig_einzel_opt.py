"""Manuscript figure: the shipped einzel deck before and after voltage
optimization (CMA-ES through ion_gym.physics.optimize).

Flies examples/einzel_round_r-z.json (Abdulla & Hussein 2020 three-tube
lens) at a deliberately detuned centre-tube bias, hands the detuned spec
to the optimizer with the RMS-spot objective, re-flies the winner, and
renders the two flights as one two-panel r-z figure through the
framework renderer (viz_core.draw_view, the per-panel body of
render_mpl). The metrics table the manuscript quotes prints to stdout
and lands in a JSON sidecar next to the figure.

The device is axisymmetric and solved in r-z, so the single unfolded
r-z cross-section is the principal plane; there is no third axis to
show. Run from anywhere: paths are repo-rooted.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np


from ion_gym.io.paths import examples_dir, outputs_dir    # noqa: E402
from ion_gym.io.sim_spec import SimSpec                    # noqa: E402
from ion_gym.physics.sim_build import build_run            # noqa: E402
from ion_gym.physics.optimize import OptimizeSpec, optimize  # noqa: E402
from ion_gym.physics.stats import auto_transmitted_fate    # noqa: E402
from ion_gym.viz.viz_core import (apply_house_style, draw_view,  # noqa: E402
                                  scene_from_simspec, save_figure)

# ---------------- operating point (named, not buried) ----------------
DECK = "einzel_round_r-z.json"  # the shipped example, unmodified
SEED = 0              # pinned run seed (the deck ships seed null; a
                      # certified figure needs a reproducible flight)
V_DETUNED = 20.0      # deliberately underfocused centre-tube bias [V]
V_BOUNDS = (5.0, 90.0)   # search interval [V]; the upper bound stays
                      # below the ~100 V retarding limit of a 100 eV beam
MAX_EVALS = 40        # ensemble flights the search may spend
OPT_SEED = 1          # CMA-ES seed: the search itself is reproducible
METRIC = "spot_size"  # RMS spot radius at the declared plane + loss penalty
LENS_PATH = "geometry.electrodes[1].dc"  # the one free parameter
FONT_PT = 12.0        # page-width legibility


def fly_and_measure(spec_in):
    """(metrics, model, trajectories, fates): the SAME flights the
    figure shows are the flights the metrics score."""
    model, fly, _cols, births = build_run(spec_in)
    trajs, fates, summaries = [], [], []
    for i in range(len(births)):
        tr, s = fly(i)
        summaries.append(s)
        if tr is not None and len(tr):
            trajs.append(np.asarray(tr))
            fates.append(str(s.get("kind", "")))
    ok = auto_transmitted_fate(spec_in)
    arrived = [s for s in summaries if s["kind"] == ok]
    if not arrived:
        raise RuntimeError(
            f"no transmitted ions at V2 = "
            f"{spec_in.geometry.electrodes[1].dc:+.1f} V -- the figure "
            f"cannot be made from an empty exit plane")
    radii = [abs(s["r_end"]) for s in arrived]
    metrics = {
        "V2_v": float(spec_in.geometry.electrodes[1].dc),
        "transmission": len(arrived) / len(summaries),
        "rms_spot_mm": float(np.sqrt(np.mean(np.square(radii)))),
        "max_r_mm": float(np.max(radii)),
    }
    return metrics, model, trajs, fates


def main():
    spec = SimSpec.from_json(str(examples_dir() / DECK))
    spec.source.seed = SEED
    # The shipped deck declares a transparent detector station for the
    # app's Impact Analysis tab. A detect/pass station alters no flight,
    # and its drawn reference line and label crowd the print figure's
    # right edge, so the figure copies fly without it.
    spec.stations = []
    errs = spec.validate()
    if errs:
        raise SystemExit("deck refused:\n  " + "\n  ".join(errs))

    spec_before = copy.deepcopy(spec)
    spec_before.geometry.electrodes[1].dc = V_DETUNED
    before, model_b, trajs_b, fates_b = fly_and_measure(spec_before)

    search = OptimizeSpec(params=[LENS_PATH], lo=[V_BOUNDS[0]],
                          hi=[V_BOUNDS[1]], metric=METRIC,
                          max_evals=MAX_EVALS, seed=OPT_SEED)
    t0 = time.time()
    result = optimize(spec_before, search)
    t_opt = time.time() - t0
    v_best = float(result.best_params[LENS_PATH])

    spec_after = copy.deepcopy(spec)
    spec_after.geometry.electrodes[1].dc = v_best
    after, model_a, trajs_a, fates_a = fly_and_measure(spec_after)

    record = {
        "deck": DECK, "seed": SEED, "metric": METRIC,
        "bounds_v": list(V_BOUNDS), "max_evals": MAX_EVALS,
        "opt_seed": OPT_SEED, "n_evals": int(result.n_evals),
        "wall_s": round(t_opt, 1),
        "v_best_v": v_best,
        "audit_um": float(result.audit_value) * 1e3,
        "audit_sd_um": float(result.audit_std) * 1e3,
        "before": before, "after": after,
        "ion": {"mz": spec.source.mz_list[0],
                "ke_ev": spec.source.ke_lo,
                "n_ions": spec.source.n_ions},
    }
    for k, v in record.items():
        print(f"{k:12s}: {v}")

    # ---------------- the two-panel figure ----------------
    # No figure or panel titles: the document caption carries the
    # operating point and the objective. Panels
    # are lettered A and B, each metric sits in a small box on its own
    # panel, and the ion paths cycle the Okabe-Ito palette.
    apply_house_style()
    from ion_gym.viz.viz_core import OKABE_ITO
    # panel letters share the size and placement of the drift cell figure
    # (run_figures draws its A/B/C at PRINT_LABEL_PT, bold, offset -30/+6)
    from ion_gym.viz.run_figures import PRINT_LABEL_PT
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    rc = {"font.size": FONT_PT, "axes.labelsize": FONT_PT,
          "xtick.labelsize": FONT_PT - 1, "ytick.labelsize": FONT_PT - 1}
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True)
        panels = (
            (axes[0], "A", spec_before, model_b, trajs_b, fates_b, before),
            (axes[1], "B", spec_after, model_a, trajs_a, fates_a, after),
        )
        for ax, letter, sp, model, trajs, fates, m in panels:
            scene = scene_from_simspec(sp, model, field="phi",
                                       trajs=trajs, fates=fates)
            ray_cols = [OKABE_ITO[k % len(OKABE_ITO)]
                        for k in range(len(trajs))]
            draw_view(ax, scene, "xy", ray_colors=ray_cols)
            ax.set_title("")
            ax.set_xlabel("axial position (mm)")
            ax.set_ylabel("y (mm)")
            ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                        xytext=(-30, 6), textcoords="offset points",
                        fontsize=PRINT_LABEL_PT, fontweight="bold",
                        va="bottom", ha="left")
            spot = m["rms_spot_mm"] * 1e3
            spot_txt = (f"{spot / 1e3:.2f} mm" if spot >= 1e3
                        else f"{spot:.0f} \u00b5m")
            # the box sits inside the field-free drift region, clear of
            # the dashed bounding plane at the right axis edge, and its
            # text runs two points below the axis labels
            ax.text(0.90, 0.94,
                    f"V2 {m['V2_v']:+.1f} V\nRMS spot {spot_txt}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=FONT_PT - 2,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="0.4", alpha=0.9))
        axes[0].label_outer()
        fig.tight_layout()
    OUT_DIR = outputs_dir("manuscript_examples", "einzel")
    png = OUT_DIR / "fig_einzel_opt.png"
    save_figure(fig, str(png))
    with open(OUT_DIR / "fig_einzel_opt.json", "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
