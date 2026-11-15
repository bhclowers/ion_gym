"""SI figure: the einzel objective landscape around the optimizer result.

Sweeps the centre-tube bias of examples/einzel_round_r-z.json across the
optimizer's full search interval (panel A) and at fine steps around the
coarse minimizer (panel B), flying the deck's own ensemble at run seed 0
for every point and plotting the RMS spot radius of the transmitted
ions. The stated bias comes from the deck as distributed and the
returned bias from the banked optimizer record written by
make_fig_einzel_opt.py, so every drawn mark equals a solver output.

No figure or panel titles: the document caption carries the operating
point, matching the other SI figures. Run from
anywhere: paths are repo-rooted.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from ion_gym.io.paths import examples_dir, outputs_dir    # noqa: E402
from ion_gym.io.sim_spec import SimSpec                    # noqa: E402
from ion_gym.viz.viz_core import (apply_house_style,       # noqa: E402
                                  save_figure)
from make_fig_einzel_opt import fly_and_measure            # noqa: E402

# ---------------- operating point (named, not buried) ----------------
DECK = "einzel_round_r-z.json"  # the included example, unmodified
SEED = 0                 # pinned run seed, the same the Results figure flies
V_BOUNDS = (5.0, 90.0)   # the optimizer's search interval [V]
N_COARSE = 25            # panel A: points across the full interval
FINE_STEP_V = 1.0        # panel B: step around the coarse minimizer [V]
FINE_HALF_SPAN_V = 10.0  # panel B: half span around the coarse minimizer [V]
FONT_PT = 12.0           # page-width legibility
OPT_RECORD_PARTS = ("manuscript_examples", "einzel", "fig_einzel_opt.json")  # the banked search


def sweep(spec, biases):
    """RMS spot radius (mm) at each centre-tube bias, flying the deck's
    own ensemble; the transmission is checked point by point because a
    spot radius from a partially transmitted ensemble is a different
    quantity and must not enter the curve unnoticed."""
    radii = []
    for v2 in biases:
        sp = copy.deepcopy(spec)
        sp.geometry.electrodes[1].dc = float(v2)
        m, _model, _trajs, _fates = fly_and_measure(sp)
        if m["transmission"] < 1.0:
            raise RuntimeError(
                f"V2 = {v2:+.1f} V transmits {m['transmission']:.0%} of the "
                f"ensemble: the landscape is defined for full transmission, "
                f"so widen the exclusion or narrow the interval rather than "
                f"plotting a mixed quantity")
        radii.append(m["rms_spot_mm"])
        print(f"V2 {v2:+7.2f} V : RMS spot {m['rms_spot_mm'] * 1e3:8.1f} um")
    return np.asarray(radii)


def main():
    OPT_RECORD = outputs_dir(*OPT_RECORD_PARTS)
    if not OPT_RECORD.exists():
        raise SystemExit(
            f"no banked optimizer record at {OPT_RECORD}: run "
            f"make_fig_einzel_opt.py first; the landscape figure refuses "
            f"to restate the returned bias from anywhere but that record")
    rec = json.loads(OPT_RECORD.read_text())
    v_returned = float(rec["v_best_v"])

    spec = SimSpec.from_json(str(examples_dir() / DECK))
    v_stated = float(spec.geometry.electrodes[1].dc)  # the deck's tune
    spec.source.seed = SEED
    spec.stations = []   # the transparent detector station alters no flight
    errs = spec.validate()
    if errs:
        raise SystemExit("deck refused:\n  " + "\n  ".join(errs))

    v_coarse = np.linspace(V_BOUNDS[0], V_BOUNDS[1], N_COARSE)
    print(f"coarse sweep: {N_COARSE} points across {V_BOUNDS} V")
    r_coarse = sweep(spec, v_coarse)
    v_min = float(v_coarse[int(np.argmin(r_coarse))])

    v_fine = v_min + np.arange(-FINE_HALF_SPAN_V, FINE_HALF_SPAN_V + 0.5,
                               FINE_STEP_V)
    print(f"fine sweep: {v_fine.size} points at {FINE_STEP_V:g} V steps "
          f"around {v_min:+.1f} V")
    r_fine = sweep(spec, v_fine)

    record = {
        "deck": DECK, "seed": SEED, "bounds_v": list(V_BOUNDS),
        "v_stated_v": v_stated, "v_returned_v": v_returned,
        "coarse": {"v2_v": v_coarse.tolist(),
                   "rms_spot_mm": r_coarse.tolist()},
        "fine": {"v2_v": v_fine.tolist(), "rms_spot_mm": r_fine.tolist()},
    }

    # ---------------- the two-panel figure ----------------
    apply_house_style()
    from ion_gym.viz.run_figures import PRINT_LABEL_PT
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    rc = {"font.size": FONT_PT, "axes.labelsize": FONT_PT,
          "xtick.labelsize": FONT_PT - 1, "ytick.labelsize": FONT_PT - 1}
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
        panels = ((axes[0], "A", v_coarse, r_coarse),
                  (axes[1], "B", v_fine, r_fine))
        for ax, letter, vv, rr in panels:
            ax.semilogy(vv, rr, marker="o", color="#0072B2")
            ax.axvline(v_stated, ls="--", color="#E69F00",
                       label=f"stated {v_stated:+.0f} V")
            ax.axvline(v_returned, ls=":", color="#009E73",
                       label=f"returned {v_returned:+.1f} V")
            ax.set_xlabel(r"$V_2$ (V)")
            ax.set_ylabel("RMS spot radius (mm)")
            ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                        xytext=(-30, 6), textcoords="offset points",
                        fontsize=PRINT_LABEL_PT, fontweight="bold",
                        va="bottom", ha="left")
        # one legend, on the panel whose upper left is empty
        axes[0].legend(loc="upper left", fontsize=FONT_PT - 2,
                       framealpha=0.9)
        fig.tight_layout()
    OUT_DIR = outputs_dir("manuscript_examples", "einzel")
    png = OUT_DIR / "fig_einzel_landscape.png"
    save_figure(fig, str(png))
    with open(OUT_DIR / "fig_einzel_landscape.json", "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
