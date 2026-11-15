"""Figures of one flight: the Impact Analysis view and its print export.

impact_figure builds the interactive Plotly figure of the workbench's
Impact Analysis tab: the impact cross-section on one surface and the 1-D
histogram of one coordinate of the arrivals. publication_run_figure draws
the same flight for print: (A) the instrument with example trajectories
through viz_core.draw_view, (B) the impact cross-section, and (C) the
arrival histogram, laid out at a stated width with every font in points.

Both read the arrivals from physics.stations.impact_hits, the function the
workbench uses, so a printed figure shows exactly what the workbench shows
for the same flight.

Import-clean: importing this module draws nothing.
"""
from __future__ import annotations

import numpy as np

from ion_gym.physics.ion_envelope import ion_label
from ion_gym.viz.viz_core import (VizError, _mpl_headless_if_needed,
                                  _mpl_release, draw_view,
                                  scene_from_simspec, view_extent,
                                  views_for)

# ---- Impact Analysis tab (interactive) ---------------------------------
# The gap between the cross-section and the histogram is set in PIXELS: it
# must hold the upper plot's tick labels and x-axis title plus the lower
# plot's subplot title (about 95 px at the default fonts). A fractional
# vertical_spacing of 0.16 on a 460 px figure left ~60 px, so the x-axis
# title of the upper plot ran into the title of the lower one. The default
# marker colours are the ones the tab drew with before the colour pickers
# existed, so an untouched picker changes nothing.
IMPACT_FIG_PX = 560
IMPACT_GAP_PX = 120
IMPACT_MARGIN_T_PX = 40
IMPACT_MARGIN_B_PX = 44
IMPACT_SCATTER_COLOR = "#1f77b4"
IMPACT_HIST_COLOR = "#2ca02c"
IMPACT_N_BINS = 48

# hit-dict key and unit for each axis a caller may name
HIT_KEY = {"x": "x", "y": "y", "z": "z", "t": "t_us"}
HIT_UNIT = {"x": "mm", "y": "mm", "z": "mm", "t": "µs"}

# ---- print export --------------------------------------------------------
PRINT_WIDTH_IN = 7.0        # JASMS two-column figure width (17.8 cm)
PRINT_FONT_PT = 10.0        # every label, tick, legend and note
                            # (raised from 8.0: 8 pt reads too small at
                            # page width)
PRINT_LABEL_PT = 12.0       # panel letters
PRINT_N_PATHS = 12          # example trajectories drawn in panel A
PRINT_MARKER_PT = 2.2       # impact marker size in panel B
PRINT_N_BINS = 40           # histogram bins in panel C
PRINT_A_MAX_H_IN = 3.0      # tallest panel A (in) before it is narrowed
NOTE_LINES = 3              # lines reserved for the operating note
NOTE_LINE_IN = 0.17         # height of one note line at PRINT_FONT_PT - 1


def _surface_hits(hits, surface):
    sel = [h for h in hits if h.get("station") == surface]
    if not sel:
        names = sorted({h.get("station") for h in hits})
        raise VizError(f"no arrivals on surface {surface!r}; surfaces with "
                       f"arrivals in this flight: {names}")
    return sel


def _check_axes(plane, hist_axis):
    if len(plane) != 2 or any(a not in "xyz" for a in plane):
        raise VizError(f"plane {plane!r}: give two of x, y, z, e.g. 'yz'")
    if hist_axis not in HIT_KEY:
        raise VizError(f"hist_axis {hist_axis!r}: one of {sorted(HIT_KEY)}")


def impact_figure(hits, *, surface, plane="yz", hist_axis="t",
                  scatter_color=IMPACT_SCATTER_COLOR,
                  hist_color=IMPACT_HIST_COLOR, mz_colors=None,
                  n_bins=IMPACT_N_BINS, charge=None, label_mode="mass"):
    """Interactive impact cross-section and histogram for one surface.

    hits : list from physics.stations.impact_hits. surface : the station
    or boundary-face label to draw. A flight with more than one value of
    'mz' (the ion MASS in Da) draws one coloured series per value, and
    mz_colors (value -> colour, the app's species convention) is then
    required; a single-value flight draws in scatter_color and
    hist_color. charge and label_mode name each series through
    ion_envelope.ion_label (charge None prints the mass alone). Returns
    the Plotly figure.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _check_axes(plane, hist_axis)
    sel = _surface_hits(hits, surface)
    a1, a2 = plane[0], plane[1]
    plot_px = IMPACT_FIG_PX - IMPACT_MARGIN_T_PX - IMPACT_MARGIN_B_PX
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=IMPACT_GAP_PX / plot_px,
        subplot_titles=(f"impact · {plane}", f"signal · {hist_axis}"))
    # PER-m/z SPLIT: a merged, unlabeled histogram over a mixed packet
    # shows anonymous multi-spikes that read as a defect, so a mixed
    # packet draws one labeled, colored series per value in both panels.
    mzs = sorted({h.get("mz") for h in sel if h.get("mz") is not None})
    if len(mzs) > 1:
        if mz_colors is None:
            raise VizError(f"{len(mzs)} values of mz on {surface!r}: pass "
                           f"mz_colors so each series has its colour")
        for mzv in mzs:
            grp = [h for h in sel if h.get("mz") == mzv]
            col = mz_colors.get(mzv)
            if col is None:
                raise VizError(f"mz_colors has no colour for {mzv!r}")
            lbl = f"{ion_label(mzv, charge, label_mode)} (n={len(grp)})"
            fig.add_trace(go.Scatter(
                x=[float(h[HIT_KEY[a1]]) for h in grp],
                y=[float(h[HIT_KEY[a2]]) for h in grp], mode="markers",
                marker=dict(size=5, color=col, opacity=0.6),
                name=lbl, legendgroup=lbl, showlegend=True,
                hovertemplate=f"{a1} %{{x:.3f}}<br>{a2} %{{y:.3f}}"
                              f"<extra>{lbl}</extra>"), row=1, col=1)
            fig.add_trace(go.Histogram(
                x=[float(h[HIT_KEY[hist_axis]]) for h in grp],
                nbinsx=n_bins, marker=dict(color=col), opacity=0.75,
                name=lbl, legendgroup=lbl, showlegend=False), row=2, col=1)
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Scatter(
            x=[float(h[HIT_KEY[a1]]) for h in sel],
            y=[float(h[HIT_KEY[a2]]) for h in sel], mode="markers",
            marker=dict(size=5, color=scatter_color, opacity=0.6),
            name="impacts", showlegend=False,
            hovertemplate=f"{a1} %{{x:.3f}}<br>{a2} %{{y:.3f}}"
                          f"<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Histogram(
            x=[float(h[HIT_KEY[hist_axis]]) for h in sel], nbinsx=n_bins,
            marker=dict(color=hist_color), name="signal",
            showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text=f"{a1} (mm)", row=1, col=1)
    fig.update_yaxes(title_text=f"{a2} (mm)", row=1, col=1)
    fig.update_xaxes(title_text=f"{hist_axis} ({HIT_UNIT[hist_axis]})",
                     row=2, col=1)
    fig.update_yaxes(title_text="ions", row=2, col=1)
    fig.update_layout(height=IMPACT_FIG_PX, showlegend=(len(mzs) > 1),
                      margin=dict(l=54, r=16, t=IMPACT_MARGIN_T_PX,
                                  b=IMPACT_MARGIN_B_PX),
                      title=dict(text=f"{surface} · {len(sel)} ions",
                                 x=0.01, font=dict(size=12)))
    return fig


def _operating_note(spec, n_flown, n_hits, surface, ion_label, n_drawn):
    """The flight's operating point, from the deck, including how many
    paths panel A draws and how often they were recorded relative to the
    integration step (panel A draws straight segments between records,
    so motion faster than the record interval is not shown there)."""
    g = spec.geometry
    bits = []
    for grp in g.rf_groups:
        bits.append(f"{grp.name} {2 * grp.amplitude_v:g} Vpp, "
                    f"{grp.frequency_hz / 1e6:g} MHz, {grp.phase_deg:g}°")
    for grp in g.dc_groups:
        vin, vout = getattr(grp, "v_in", None), getattr(grp, "v_out", None)
        if vin is not None and vout is not None:
            bits.append(f"{grp.name} {vin:.4g} to {vout:.4g} V")
    c = spec.collisions
    bits.append(f"{c.gas} {c.P_torr:g} Torr, {c.T_k:g} K, "
                f"{c.model.upper()}" if c.enabled else "vacuum")
    if ion_label:
        bits.append(ion_label)
    bits.append(f"{n_hits} of {n_flown} ions on {surface}")
    it = spec.integration
    rec_us = it.rec_every * it.dt_ns * 1e-3
    bits.append(f"panel A: {n_drawn} of {n_flown} ions, one record every "
                f"{rec_us:.3g} \u00b5s ({it.rec_every} steps of dt = "
                f"{it.dt_ns:g} ns)")
    return "; ".join(bits)


def publication_run_figure(spec, model, trajs, hits, *, surface,
                           n_flown, plane="yz", hist_axis="t",
                           ion_label=None, width_in=PRINT_WIDTH_IN,
                           font_pt=PRINT_FONT_PT, n_paths=PRINT_N_PATHS,
                           path_color="#56B4E9", scatter_color="#7B2D8E",
                           hist_color="#444444", n_bins=PRINT_N_BINS,
                           operating_note=True, view=None,
                           surface_label=None, series=None):
    """Print figure of one flight: (A) instrument and example trajectories,
    (B) impact cross-section on `surface`, (C) histogram of `hist_axis`.

    spec, model : the deck and its built model. trajs : kernel record
    arrays of the flown ions (the first n_paths are drawn in A). hits :
    physics.stations.impact_hits of the same flight. n_flown : ions born.
    ion_label : how the ion is named on the figure; the caller states it,
    because the meaning of mz_list depends on the route (on the r-z route
    it carries the ion mass in Da). operating_note : print the deck's
    operating point under the panels. view : the panel-A view; None takes
    the single view of a planar or r-z deck and refuses a 3-D deck, which
    would need several views. surface_label : how the surface is named in
    the operating note (default: the surface key itself, e.g. 'x=+62').
    series : for a flight with more than one value of hits' 'mz', a dict
    value -> (label, colour); panels B and C then draw one labelled
    series per value. A mixed flight without series is refused, because
    pooling the masses would merge them into one unlabelled histogram.

    Returns the matplotlib Figure; saving is the caller's decision.
    """
    _check_axes(plane, hist_axis)
    sel = _surface_hits(hits, surface)
    if len(trajs) < 1:
        raise VizError("no trajectories to draw in panel A")
    _mpl_headless_if_needed()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    shown = [np.asarray(t, float) for t in trajs[:n_paths]]
    sc = scene_from_simspec(spec, model, field=None, title=spec.name,
                            trajs=shown)
    if view is None:
        vs = views_for(sc, None)
        if len(vs) != 1:
            raise VizError(
                f"deck {spec.name!r} is 3-D ({len(vs)} views); panel A has "
                f"room for one, so pass view= for each figure or use "
                f"render_mpl for the multi-axis drawing")
        view = vs[0]
    a1, a2 = plane[0], plane[1]
    mzs = sorted({h.get("mz") for h in sel if h.get("mz") is not None})
    if len(mzs) > 1:
        if series is None:
            raise VizError(
                f"{len(mzs)} values of mz on {surface!r} ({mzs}): pass "
                f"series={{value: (label, colour)}} so each mass is drawn "
                f"as its own labelled series")
        missing = [m for m in mzs if m not in series]
        if missing:
            raise VizError(f"series has no entry for mz {missing}")
        groups = [(series[m][0], series[m][1],
                   [h for h in sel if h.get("mz") == m]) for m in mzs]
    else:
        groups = [(None, None, sel)]

    def col(g, k):
        return np.array([float(h[HIT_KEY[k]]) for h in g])

    rc = {"font.size": font_pt, "axes.labelsize": font_pt,
          "axes.titlesize": font_pt, "xtick.labelsize": font_pt,
          "ytick.labelsize": font_pt, "legend.fontsize": font_pt,
          "font.family": "sans-serif", "axes.spines.top": False,
          "axes.spines.right": False}
    with mpl.rc_context(rc):
        # panel A height from the drawn extent at the full width
        e0, e1, f0, f1 = view_extent(sc, view)
        # panel A spans the width, true scale; a compact instrument (a
        # quad cross section) is capped in height and narrowed to keep its
        # aspect, so the figure does not grow to the height of a page
        ax_w = width_in - 0.75
        a_h = ax_w * (f1 - f0) / max(e1 - e0, 1e-9)
        if a_h > PRINT_A_MAX_H_IN:
            a_h = PRINT_A_MAX_H_IN
            ax_w = a_h * (e1 - e0) / max(f1 - f0, 1e-9)
        bc_h = 2.1
        note_h = NOTE_LINE_IN * NOTE_LINES if operating_note else 0.0
        fig_h = a_h + bc_h + 1.65 + note_h
        fig = plt.figure(figsize=(width_in, fig_h))
        top = 1.0 - 0.32 / fig_h   # room for the panel A letter
        axA = fig.add_axes([0.60 / width_in, top - a_h / fig_h,
                            ax_w / width_in, a_h / fig_h])
        draw_view(axA, sc, view, field=False,
                  ray_colors=[path_color] * len(shown))
        axA.set_title("")
        for ln in axA.get_lines():
            ln.set_linewidth(0.6)
        for t in axA.texts:
            t.set_fontsize(font_pt - 1)
        axA.xaxis.label.set_size(font_pt)
        axA.yaxis.label.set_size(font_pt)
        axA.tick_params(labelsize=font_pt)

        bc_y = (0.55 + note_h) / fig_h
        axB = fig.add_axes([0.60 / width_in, bc_y,
                            2.9 / width_in, bc_h / fig_h])
        edges = np.histogram_bin_edges(col(sel, hist_axis), n_bins)
        for lab, c, g in groups:
            axB.scatter(col(g, a1), col(g, a2), s=PRINT_MARKER_PT ** 2,
                        color=c or scatter_color, alpha=0.55,
                        linewidths=0, label=lab)
        axB.set_aspect("equal", adjustable="datalim")
        axB.set_xlabel(f"{a1} (mm)")
        axB.set_ylabel(f"{a2} (mm)")
        axC = fig.add_axes([4.15 / width_in, bc_y,
                            (width_in - 4.35) / width_in, bc_h / fig_h])
        for lab, c, g in groups:
            axC.hist(col(g, hist_axis), bins=edges, color=c or hist_color,
                     alpha=(0.6 if lab else 1.0), label=lab)
        if groups[0][0] is not None:
            axC.legend(frameon=False)
        axC.set_xlabel(f"{hist_axis} ({HIT_UNIT[hist_axis]})")
        axC.set_ylabel("ion counts")
        # counts are whole ions: no fractional ticks on a small flight
        from matplotlib.ticker import MaxNLocator
        axC.yaxis.set_major_locator(MaxNLocator(integer=True))
        # letters at a fixed offset in points from each panel's top-left
        # corner, so a tall or short panel places its letter the same way
        for ax, lab in ((axA, "A"), (axB, "B"), (axC, "C")):
            ax.annotate(lab, xy=(0.0, 1.0), xycoords="axes fraction",
                        xytext=(-30, 6), textcoords="offset points",
                        fontsize=PRINT_LABEL_PT, fontweight="bold",
                        va="bottom", ha="left")
        # drive-group legend for panel A, from the bodies' outline colours
        from matplotlib.lines import Line2D
        from ion_gym.viz.viz_core import STYLE
        grps = sorted({b.group for b in sc.bodies if b.group})
        if grps:
            pal = STYLE["groups"]
            axA.legend(handles=[Line2D([], [], color=pal[k % len(pal)],
                                       lw=1.6, label=g)
                                for k, g in enumerate(grps)],
                       loc="lower right", bbox_to_anchor=(1.0, 1.0),
                       ncol=len(grps), frameon=False, borderaxespad=0.1,
                       handlelength=1.4, columnspacing=1.0)
        if operating_note:
            import textwrap
            # wrap to the figure width: ~0.55 em per character
            chars = int((width_in - 0.8) * 72.0 / (0.55 * (font_pt - 1)))
            lines = textwrap.wrap(_operating_note(spec, n_flown, len(sel),
                                                  surface_label or surface,
                                                  ion_label, len(shown)),
                                  chars, break_long_words=False)
            if len(lines) > NOTE_LINES:
                raise VizError(f"operating note needs {len(lines)} lines "
                               f"at {font_pt - 1} pt; NOTE_LINES allows "
                               f"{NOTE_LINES}")
            fig.text(0.60 / width_in, 0.10 / fig_h, "\n".join(lines),
                     fontsize=font_pt - 1, color="0.30", ha="left",
                     va="bottom", linespacing=1.3)
    _mpl_release(fig)
    return fig
