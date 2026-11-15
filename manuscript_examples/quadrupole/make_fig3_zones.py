"""Manuscript Figure 3: the first stability zone under the sinusoidal
drive and the two-level square drive at two duty cycles, from the
one-period transfer matrix. No field is solved and no ion is flown.

This is the committed producer of the doc's Figure 3 (previously drawn
only inside digital_stability.ipynb, whose teaching version carries a
suptitle; the manuscript figure carries none, per the
review, and its operating information lives in the document caption).
Same kernels, same margin convention as the notebook: color is log10 of
the larger of |tr M|/2 over the x and y axes, boundary at
|tr M|/2 = 1, with the a = 0 zone edges bisected on the transfer matrix.

Run from anywhere; outputs land in outputs_dir("manuscript_examples", "quadrupole").
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ion_gym.io.paths import outputs_dir  # noqa: E402
from ion_gym.physics.mathieu import (hill_half_trace_njit,  # noqa: E402
                                     meissner_half_trace)
from ion_gym.viz.viz_core import (apply_house_style,        # noqa: E402
                                  save_figure)
import quad_lib as QL                                       # noqa: E402

# ---------------- the figure's declared parameters --------------------
N_GRID = 161          # (a, q) map resolution per axis
A_SPAN = 0.50         # |a| extent; the duty-shifted region reaches -0.45
Q_SPAN = 1.25         # q extent
DUTY_SHIFTED = 0.4    # the off-0.5 duty of the third panel
FONT_PT = 11.0        # per the review: page-width legible
EDGE_TOL = 1e-6       # bisection tolerance on the a = 0 edges


def a0_edge(trace_fn, q_in, q_out, tol=EDGE_TOL):
    """Bisect |tr M|/2 = 1 between a stable and an unstable q."""
    if not abs(trace_fn(q_in)) < 1.0:
        raise ValueError(f"bisection start q = {q_in} is not stable")
    if not abs(trace_fn(q_out)) > 1.0:
        raise ValueError(f"bisection start q = {q_out} is not unstable")
    while abs(q_out - q_in) > tol:
        mid = 0.5 * (q_in + q_out)
        if abs(trace_fn(mid)) < 1.0:
            q_in = mid
        else:
            q_out = mid
    return 0.5 * (q_in + q_out)


def main():
    aa = np.linspace(-A_SPAN, A_SPAN, N_GRID)
    qq = np.linspace(0.0, Q_SPAN, N_GRID)
    m_sin = np.zeros((N_GRID, N_GRID))
    m_dig = np.zeros((N_GRID, N_GRID))
    m_dig2 = np.zeros((N_GRID, N_GRID))
    for i, a in enumerate(aa):
        for j, q in enumerate(qq):
            m_sin[i, j] = np.log10(max(abs(hill_half_trace_njit(a, q)),
                                       abs(hill_half_trace_njit(-a, q)),
                                       1e-12))
            m_dig[i, j] = np.log10(max(
                abs(meissner_half_trace(a, q, 0.5)),
                abs(meissner_half_trace(-a, q, 0.5)), 1e-12))
            m_dig2[i, j] = np.log10(max(
                abs(meissner_half_trace(a, q, DUTY_SHIFTED)),
                abs(meissner_half_trace(-a, q, 1.0 - DUTY_SHIFTED)),
                1e-12))

    edge_sin = a0_edge(lambda q: hill_half_trace_njit(0.0, q), 0.85, 0.95)
    edge_dig = a0_edge(lambda q: meissner_half_trace(0.0, q, 0.5),
                       0.65, 0.78)

    def two_axis_trace(q):
        return max(abs(meissner_half_trace(0.0, q, DUTY_SHIFTED)),
                   abs(meissner_half_trace(0.0, q, 1.0 - DUTY_SHIFTED)))

    runs = QL.a0_stable_bands(lambda q: two_axis_trace(q) < 1.0,
                              1e-4, 1.2, n=1201)
    if not runs:
        raise ValueError(f"duty {DUTY_SHIFTED}: no stable q on a = 0")
    run_lo, run_hi = runs[0]
    scan_step = (1.2 - 1e-4) / 1200
    run_mid = 0.5 * (run_lo + run_hi)
    shift_lo = a0_edge(two_axis_trace, run_mid, run_lo - scan_step)
    shift_hi = a0_edge(two_axis_trace, run_mid, run_hi + scan_step)

    record = {"edge_sin": edge_sin, "edge_dig": edge_dig,
              "duty_shifted": DUTY_SHIFTED,
              "shift_band": [shift_lo, shift_hi],
              "n_grid": N_GRID, "a_span": A_SPAN, "q_span": Q_SPAN}
    for k, v in record.items():
        print(f"{k:12s}: {v}")

    apply_house_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    vmin = min(m_sin.min(), m_dig.min(), m_dig2.min())
    vmax = np.percentile(np.concatenate([m_sin, m_dig, m_dig2]), 60)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6), sharey=True,
                             layout="constrained")
    panels = (
        (axes[0], m_sin, "sinusoidal drive", (edge_sin,), 0.035),
        (axes[1], m_dig, "two-level square, duty 0.5", (edge_dig,), 0.035),
        (axes[2], m_dig2, f"two-level square, duty {DUTY_SHIFTED}",
         (shift_lo, shift_hi), 0.085),
    )
    for ax, m, title, edges, label_a in panels:
        pm = ax.pcolormesh(qq, aa, m, cmap="RdBu_r", norm=norm,
                           shading="auto", rasterized=True)
        ax.contour(qq, aa, m, levels=[0.0], colors="k", linewidths=1.2)
        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        for edge in edges:
            ax.plot(edge, 0.0, "o", color="black", ms=6,
                    markerfacecolor="white", markeredgewidth=1.5, zorder=5)
        label = (f"q = {edges[0]:.4f}" if len(edges) == 1
                 else f"q = {edges[0]:.4f} to {edges[1]:.4f}")
        QL.halo_text(ax, edges[-1] + 0.025, label_a, label,
                     fontsize=FONT_PT, fontweight="semibold", color="black")
        ax.grid(False)
        ax.set_title(title, fontsize=FONT_PT + 1)
        ax.set_xlabel("q", fontsize=FONT_PT + 1)
        ax.tick_params(labelsize=FONT_PT)
    axes[0].set_ylabel("a", fontsize=FONT_PT + 1)
    cb = fig.colorbar(pm, ax=axes, shrink=0.9)
    cb.set_label("log10 max(|tr M|/2) over x, y", fontsize=FONT_PT)
    cb.ax.tick_params(labelsize=FONT_PT)

    OUT_DIR = outputs_dir("manuscript_examples", "quadrupole")
    png = OUT_DIR / "fig3_zones.png"
    save_figure(fig, str(png))
    with open(OUT_DIR / "fig3_zones.json", "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
