"""Stability charts checked in flight: a computed (a, q) chart with flown
survival markers, and a flown transmission scan across a computed band.

stability_flight_figure draws two panels for print. Panel A is a stability
chart computed from a transfer matrix, given as a scalar margin on an (a, q)
grid (stable where the margin is below zero), with one marker per flown
ion showing survival or loss. Panel B is a transmission scan, the surviving
fraction of a set of flights at each value of a swept coordinate, with the
computed stable band shaded behind it.

Both panels are generic. The chart may belong to any drive and any zone,
and the scan may sweep any coordinate across any band, so the renderer
holds no physics. The caller supplies the arrays the solver and the flights
produced, and the operating notes printed under each panel.

Import-clean: importing this module draws nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ion_gym.viz.run_figures import (NOTE_LINE_IN, PRINT_FONT_PT,
                                     PRINT_LABEL_PT, PRINT_WIDTH_IN)
from ion_gym.viz.viz_core import (OKABE_ITO, VizError,
                                  _mpl_headless_if_needed, _mpl_release)

# ---- named defaults (house palette, Okabe-Ito) ---------------------------
STABLE_FILL = OKABE_ITO[5]       # sky blue: computed stable region / band
STABLE_ALPHA = 0.35
BOUNDARY_COLOR = OKABE_ITO[7]    # black: the margin = 0 contour
SURVIVED_COLOR = OKABE_ITO[2]    # bluish green: flown ion survived
LOST_COLOR = OKABE_ITO[3]        # vermillion: flown ion lost
SCAN_COLOR = OKABE_ITO[0]        # blue: flown transmission
MARKER_PT = 4.5                  # flown markers in both panels
PANEL_H_IN = 2.6                 # plotting height of both panels
NOTE_LINES = 5                   # lines reserved under each panel
GAP_IN = 0.75                    # between the two panels
MARGIN_L_IN = 0.55
MARGIN_R_IN = 0.28
MARGIN_T_IN = 0.62               # letters and the legends above each panel
XLABEL_IN = 0.42                 # x tick labels and axis title


@dataclass
class StabilityChart:
    """Panel A input. margin[i, j] at (a[i], q[j]), stable where < 0.
    q_flown, a_flown, survived : one entry per flown ion."""
    q: np.ndarray
    a: np.ndarray
    margin: np.ndarray
    q_flown: np.ndarray
    a_flown: np.ndarray
    survived: np.ndarray
    note: str
    q_label: str = "$q$"
    a_label: str = "$a$"
    survived_label: str = "flown, survived"
    lost_label: str = "flown, lost"
    region_label: str = "computed stable region"


@dataclass
class BandScan:
    """Panel B input. transmission[k] at x[k], each in [0, 1]; band is
    the computed stable interval (lo, hi) in the units of x."""
    x: np.ndarray
    transmission: np.ndarray
    band: tuple
    note: str
    x_label: str
    y_label: str = "Transmission"
    scan_label: str = "flown"
    band_label: str = "computed stable band"


def _check_chart(c):
    q, a, m = (np.asarray(c.q, float), np.asarray(c.a, float),
               np.asarray(c.margin, float))
    if m.shape != (a.size, q.size):
        raise VizError(f"StabilityChart: margin shape {m.shape} does not "
                       f"match (len(a), len(q)) = ({a.size}, {q.size})")
    if not np.isfinite(m).all():
        raise VizError("StabilityChart: margin holds non-finite values; "
                       "the chart would draw a boundary through them")
    if not (m < 0).any():
        raise VizError("StabilityChart: no stable cell (margin < 0) on the "
                       "grid; there is no region to draw")
    qf, af, sv = (np.asarray(c.q_flown, float).ravel(),
                  np.asarray(c.a_flown, float).ravel(),
                  np.asarray(c.survived).ravel())
    if not (qf.size == af.size == sv.size) or qf.size == 0:
        raise VizError(f"StabilityChart: flown arrays must be equal and "
                       f"non-empty, got q {qf.size}, a {af.size}, "
                       f"survived {sv.size}")
    if sv.dtype != bool:
        raise VizError(f"StabilityChart: survived must be bool, got "
                       f"{sv.dtype}")
    return q, a, m, qf, af, sv


def _check_scan(s):
    x, t = np.asarray(s.x, float), np.asarray(s.transmission, float)
    if x.shape != t.shape or x.size < 2:
        raise VizError(f"BandScan: x {x.shape} and transmission {t.shape} "
                       f"must match with at least two points")
    if ((t < 0) | (t > 1)).any() or not np.isfinite(t).all():
        raise VizError("BandScan: transmission must lie in [0, 1]; got "
                       f"min {np.nanmin(t)}, max {np.nanmax(t)}")
    lo, hi = (float(v) for v in s.band)
    if not lo < hi:
        raise VizError(f"BandScan: band ({lo}, {hi}) is empty")
    return x, t, lo, hi


def _wrap_note(text, width_in, font_pt):
    import textwrap
    if not str(text).strip():
        raise VizError("operating note is empty; a panel without its "
                       "operating point is not a result")
    chars = int(width_in * 72.0 / (0.52 * font_pt))
    lines = textwrap.wrap(str(text), chars, break_long_words=False)
    if len(lines) > NOTE_LINES:
        raise VizError(f"operating note needs {len(lines)} lines at "
                       f"{font_pt} pt in {width_in:.2f} in; NOTE_LINES "
                       f"allows {NOTE_LINES}")
    return "\n".join(lines)


def stability_flight_figure(chart: StabilityChart, scan: BandScan, *,
                            width_in=PRINT_WIDTH_IN, font_pt=PRINT_FONT_PT,
                            label_pt=PRINT_LABEL_PT,
                            panel_h_in=PANEL_H_IN, operating_note=True,
                            letters=("A", "B")):
    """Two-panel print figure of a computed chart and a computed band,
    each checked by flights. Every font is in points and the figure is
    laid out at width_in. Returns the Figure; the caller saves it."""
    q, a, m, qf, af, sv = _check_chart(chart)
    x, t, lo, hi = _check_scan(scan)
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    panel_w = (width_in - MARGIN_L_IN - MARGIN_R_IN - GAP_IN) / 2.0
    if panel_w <= 1.0:
        raise VizError(f"width_in {width_in} leaves {panel_w:.2f} in per "
                       f"panel; widen the figure or shrink GAP_IN")
    note_pt = font_pt - 1
    note_h = (NOTE_LINES * NOTE_LINE_IN + 0.08) if operating_note else 0.0
    fig_h = MARGIN_T_IN + panel_h_in + XLABEL_IN + note_h
    y0 = (XLABEL_IN + note_h) / fig_h
    with plt.rc_context({"font.size": font_pt, "axes.labelsize": font_pt,
                         "xtick.labelsize": font_pt,
                         "ytick.labelsize": font_pt,
                         "legend.fontsize": font_pt - 0.5}):
        fig = plt.figure(figsize=(width_in, fig_h))
        x_a = MARGIN_L_IN / width_in
        x_b = (MARGIN_L_IN + panel_w + GAP_IN) / width_in
        w = panel_w / width_in
        h = panel_h_in / fig_h
        axA = fig.add_axes([x_a, y0, w, h])
        axB = fig.add_axes([x_b, y0, w, h])

        # ---- A: computed region, its boundary, flown markers --------------
        stable = (m < 0).astype(float)
        axA.contourf(q, a, stable, levels=[0.5, 1.5], colors=[STABLE_FILL],
                     alpha=STABLE_ALPHA)
        axA.contour(q, a, m, levels=[0.0], colors=[BOUNDARY_COLOR],
                    linewidths=0.9)
        axA.plot(qf[sv], af[sv], "o", ls="none", color=SURVIVED_COLOR,
                 ms=MARKER_PT, mec="white", mew=0.5)
        axA.plot(qf[~sv], af[~sv], "x", ls="none", color=LOST_COLOR,
                 ms=MARKER_PT, mew=1.2)
        axA.set_xlim(q.min(), q.max())
        axA.set_ylim(a.min(), a.max())
        axA.set_xlabel(chart.q_label)
        axA.set_ylabel(chart.a_label)
        axA.legend(handles=[
            Patch(facecolor=STABLE_FILL, alpha=STABLE_ALPHA,
                  edgecolor=BOUNDARY_COLOR, lw=0.9,
                  label=chart.region_label),
            Line2D([], [], marker="o", ls="none", color=SURVIVED_COLOR,
                   mec="white", ms=MARKER_PT, label=chart.survived_label),
            Line2D([], [], marker="x", ls="none", color=LOST_COLOR,
                   mew=1.2, ms=MARKER_PT, label=chart.lost_label)],
            loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
            frameon=False, handlelength=1.0, borderaxespad=0.2,
            columnspacing=0.8, handletextpad=0.4)

        # ---- B: computed band and flown transmission ----------------------
        axB.axvspan(lo, hi, color=STABLE_FILL, alpha=STABLE_ALPHA, lw=0)
        axB.plot(x, t, "-o", color=SCAN_COLOR, ms=MARKER_PT, lw=1.0,
                 mec="white", mew=0.5)
        axB.set_xlim(x.min(), x.max())
        axB.set_ylim(-0.03, 1.03)
        axB.set_xlabel(scan.x_label)
        axB.set_ylabel(scan.y_label)
        axB.legend(handles=[
            Patch(facecolor=STABLE_FILL, alpha=STABLE_ALPHA,
                  label=scan.band_label),
            Line2D([], [], marker="o", color=SCAN_COLOR, mec="white",
                   ms=MARKER_PT, lw=1.0, label=scan.scan_label)],
            loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
            frameon=False, handlelength=1.6, borderaxespad=0.2,
            columnspacing=1.0)

        for ax, lab in zip((axA, axB), letters):
            ax.annotate(lab, xy=(0.0, 1.0), xycoords="axes fraction",
                        xytext=(-30, 30), textcoords="offset points",
                        fontsize=label_pt, fontweight="bold",
                        va="bottom", ha="left")
        if operating_note:
            for x_in, note in ((MARGIN_L_IN, chart.note),
                               (MARGIN_L_IN + panel_w + GAP_IN, scan.note)):
                fig.text(x_in / width_in, (note_h - 0.04) / fig_h,
                         _wrap_note(note, panel_w, note_pt),
                         fontsize=note_pt, color="0.30", ha="left",
                         va="top", linespacing=1.25)
    _mpl_release(fig)
    return fig
