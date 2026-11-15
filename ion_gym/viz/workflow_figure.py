"""Workflow figure: deck, unit-potential solutions, applied voltages, their
time-dependent sum, the integrator, and the recorded flights of one deck.

Every panel is drawn from solver products of the deck it is given, so the
figure is a picture of what the code did rather than a diagram of it.
Panel B shows unit-potential solutions from the solver ('basis:<name>'),
panel C the group voltages from physics.build_planar.drive_voltage (the
sim_spec contract V = amplitude * w(t) + offset), panel D the summed
potential at chosen RF phases ('phi@<deg>'), and panel F the kernel records
of flights the caller supplies. The text in panels A and E is composed from
the deck's own fields, so a displayed count or step equals the solver input.

Layout: two rows read left to right. The top row holds the deck (A), the
unit-potential solutions (B), and the applied voltages (C). One arrow,
labelled with the weighted sum, wraps from the end of the top row to the
summed potential (D), followed by the integrator (E) and the recorded
flights (F).

Scope: planar decks, because each map is one cross section. A 3-D deck
would need several views per map, so a 3-D deck is refused by name rather
than shown in a single projection.

Import-clean: importing this module draws nothing.
"""
from __future__ import annotations

import numpy as np

from ion_gym.viz.viz_core import (FATE_NAMES, OKABE_ITO, STYLE, VizError,
                                  _mpl_headless_if_needed, _mpl_release,
                                  body_electrode_name, draw_view,
                                  project, scene_from_simspec)

# Named layout defaults (inches and points).
WORKFLOW_WIDTH_IN = 7.0     # JASMS two-column figure width (17.8 cm)
MAP_IN = 1.10               # side of one square field map (axes box)
CBAR_IN = 0.36              # room right of a map for its colourbar ticks
BOX_W_IN = 1.30             # width of a text box (panels A and E)
BOX_LINESPACING = 1.32      # line pitch of box text, multiple of TEXT_PT
WAVE_W_IN = 1.05            # width of the waveform axes (panel C)
GAP_IN = 0.28               # minimum horizontal gap holding an arrow
ROW_GAP_IN = 1.12           # vertical gap between the two rows
MARGIN_L_IN = 0.40          # left margin (y tick labels of the first map)
TEXT_PT = 6.5               # box text, ticks, axis labels
LABEL_PT = 9.0              # panel letters
N_PERIODS = 2.0             # RF periods drawn in panel C
N_WAVE_PTS = 400            # samples per waveform curve in panel C
LEGEND_WRAP = 20            # characters per line of a fate in panel F
# Flight colours in panel F: house palette entries not used by the first
# two drive groups, so a path is never mistaken for an electrode outline.
FLIGHT_COLORS = (OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5], OKABE_ITO[6])
# Panel B: the electrode held at 1 V is filled; grounded electrodes are
# outlined in grey, since drive groups play no part in a unit solution.
BASIS_ON_COLOR = "#C8102E"    # red outline and fill of the 1 V electrode
BASIS_ON_ALPHA = 0.45          # fill opacity, so contours stay visible
BASIS_ON_LABEL_PT = 7.5        # size of the '1 V' label on that electrode
BASIS_OFF_COLOR = "0.55"
BIRTH_MARKER_PT = 6.0        # marker size of the ion start in panel F


def _deck_lines(spec):
    """Panel A text, composed from the deck's fields."""
    g = spec.geometry
    s = spec.source
    c = spec.collisions
    n_el = len(g.electrodes)
    lines = ["ion_gym deck (JSON)",
             f"{n_el} electrode{'s' if n_el != 1 else ''}",
             f"grid pitch {g.mm_per_gu:g} mm"]
    for grp in g.rf_groups:
        lines.append(f"{grp.name}: {grp.waveform}, {grp.amplitude_v:g} V (0-p)")
        lines.append(f"   {grp.frequency_hz / 1e6:g} MHz, "
                     f"phase {grp.phase_deg:g}\u00b0")
    for grp in g.dc_groups:
        lines.append(f"{grp.name}: DC group")
    mz = ", ".join(f"{m:g}" for m in s.mz_list)
    from ion_gym.physics.ion_envelope import charge_str
    lines.append(f"ions: {mz} Da, {charge_str(s.charge)}")
    lines.append(f"start: {s.distribution} at")
    lines.append(f"   x = {s.x0_mm:g} mm")
    lines.append(f"   y = {s.y0_mm:g} mm")
    ke = (f"{s.ke_lo:g} eV" if s.ke_lo == s.ke_hi
          else f"{s.ke_lo:g} to {s.ke_hi:g} eV")
    lines.append(f"   KE = {ke}")
    lines.append(f"gas: {c.gas}, {c.P_torr:g} Torr" if c.enabled
                 else "gas: none (vacuum)")
    return lines


def _integrator_lines(spec):
    """Panel E text, composed from the deck's integration and collision
    settings."""
    it = spec.integration
    c = spec.collisions
    lines = ["Integrator", "Runge-Kutta,", "fourth order",
             f"fixed step, dt = {it.dt_ns:g} ns",
             f"t_max = {it.t_max_us:g} µs"]
    lines.append(f"collisions: {c.model.upper()}" if c.enabled
                 else "collisions: off")
    return lines


def _text_box(fig, rect, lines, label):
    import matplotlib.patches as mpatches
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.05",
        transform=ax.transAxes, facecolor="white", edgecolor="0.25",
        linewidth=0.9, clip_on=False))
    t = ax.text(0.07, 1.0 - 0.09 / (rect[3] * fig.get_figheight()),
                "\n".join(lines), transform=ax.transAxes, fontsize=TEXT_PT,
                va="top", ha="left", linespacing=BOX_LINESPACING)
    ax.text(0.0, 1.04, label, transform=ax.transAxes, fontsize=LABEL_PT,
            fontweight="bold", va="bottom", ha="left")
    # refuse text that overruns its box: a clipped or overhanging line is
    # a figure that misstates the deck
    r = fig.canvas.get_renderer()
    tb = t.get_window_extent(r)
    bb = ax.get_window_extent(r)
    if tb.x1 > bb.x1 - 2 or tb.y0 < bb.y0 + 2:
        raise VizError(
            f"panel {label} text overruns its box by "
            f"{max(tb.x1 - bb.x1, 0) / fig.dpi:.2f} in (width) and "
            f"{max(bb.y0 - tb.y0, 0) / fig.dpi:.2f} in (height); widen "
            f"BOX_W_IN or shorten the lines")
    return ax


def _box_height_in(lines):
    return len(lines) * TEXT_PT * BOX_LINESPACING / 72.0 + 0.22


def _draw_map(ax, sc, field, ray_colors=None, colors=None):
    """draw_view into ax, then thin and size the colourbar draw_view
    appended (draw_view adds the colourbar axes to the figure)."""
    from matplotlib.ticker import MaxNLocator
    before = set(ax.figure.axes)
    draw_view(ax, sc, "xy", field=field, ray_colors=ray_colors,
              colors=colors)
    for cb_ax in set(ax.figure.axes) - before:
        cb_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        cb_ax.tick_params(labelsize=TEXT_PT - 0.5, length=1.5, pad=1.0)
        cb_ax.set_ylabel("")
    ax.tick_params(labelsize=TEXT_PT, length=2.0, pad=1.5)
    ax.xaxis.label.set_size(TEXT_PT)
    ax.yaxis.label.set_size(TEXT_PT)
    ax.xaxis.labelpad = 1.5
    ax.yaxis.labelpad = 1.0


def _letter(ax, label, dx=-0.32):
    ax.text(dx, 1.06, label, transform=ax.transAxes, fontsize=LABEL_PT,
            fontweight="bold", va="bottom", ha="left")


def workflow_figure(spec, *, basis_electrodes, phases_deg, flights,
                    model=None, width_in: float = WORKFLOW_WIDTH_IN):
    """Six-panel workflow figure of one planar deck. Returns the Figure;
    saving and display are the caller's decision.

    basis_electrodes : names of the two electrodes whose unit-potential
        solutions panel B shows (one per map), or one name.
    phases_deg : the RF phase (one entry) at which panel D shows the summed
        potential. Sinusoidal drives only, as in 'phi@<deg>'.
    flights : list of (record_array, status_dict, label) for panel F, where
        record_array is the kernel's (t, x, y, z, ...) record array as fly()
        returns it and status_dict holds the kernel's 'kind' fate code.
    model : the built planar model; None builds it from spec once and every
        panel reuses it.
    """
    if not 1 <= len(basis_electrodes) <= 2:
        raise ValueError(f"basis_electrodes takes one or two names, got "
                         f"{basis_electrodes!r}")
    if len(phases_deg) != 1:
        raise ValueError(f"phases_deg takes one phase in this layout, got "
                         f"{phases_deg!r}")
    if not flights:
        raise ValueError("flights is empty: panel F shows recorded flights, "
                         "so at least one is required")
    if len(flights) > len(FLIGHT_COLORS):
        raise ValueError(f"{len(flights)} flights, but FLIGHT_COLORS names "
                         f"{len(FLIGHT_COLORS)} distinct colours")
    known = [e.name for e in spec.geometry.electrodes]
    bad = [n for n in basis_electrodes if n not in known]
    if bad:
        raise VizError(f"basis_electrodes {bad} are not electrodes of deck "
                       f"{spec.name!r} (electrodes: {known})")
    if not spec.geometry.rf_groups:
        raise VizError(f"deck {spec.name!r} declares no drive groups, so "
                       f"panels C and D have no time dependence to show")
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    from ion_gym.physics.build_planar import build_planar_model, drive_voltage
    if model is None:
        model = build_planar_model(spec)

    def scene(field, trajs=None, fates=None):
        sc = scene_from_simspec(spec, model, field=field, title=spec.name,
                                trajs=trajs, fates=fates)
        if sc.is_3d():
            raise VizError(
                f"deck {spec.name!r} is 3-D; the workflow figure draws one "
                f"cross section per map and would show it in a single "
                f"projection, which the multi-axis rule forbids")
        return sc

    deck_lines = _deck_lines(spec)
    int_lines = _integrator_lines(spec)
    nb = len(basis_electrodes)

    # ---- horizontal positions, top row: A | B (nb maps) | C
    # fixed widths: edge margins, box A, the y-label room left of the first
    # map (YLAB_IN) and of panel C (0.45 in), the maps with colourbars, and
    # panel C. The remaining width is shared by the gaps (the gap between
    # the two maps counts 0.6 of a full gap: it holds no arrow).
    edge = 0.10
    ylab = 0.30
    fixed = (2 * edge + BOX_W_IN + ylab + nb * (MAP_IN + CBAR_IN)
             + 0.45 + WAVE_W_IN)
    n_gaps = 2 + 0.6 * (nb - 1)
    gap = (width_in - fixed) / n_gaps
    if gap < GAP_IN:
        raise VizError(f"top row needs {fixed + n_gaps * GAP_IN:.2f} in at "
                       f"MAP_IN = {MAP_IN} in, wider than the {width_in} in "
                       f"figure; reduce MAP_IN or widen")
    xa = edge
    xb = [xa + BOX_W_IN + gap + ylab]
    for _k in range(1, nb):
        xb.append(xb[-1] + MAP_IN + CBAR_IN + 0.6 * gap)
    xc = xb[-1] + MAP_IN + CBAR_IN + gap + 0.45

    # ---- bottom row: D | E | F (+ legend)
    xd = MARGIN_L_IN + 0.05
    xe = xd + MAP_IN + CBAR_IN + 0.55
    xf = xe + BOX_W_IN + 0.75
    if xf + MAP_IN + 0.10 > width_in:
        raise VizError(f"bottom row overruns the {width_in} in width "
                       f"(needs {xf + MAP_IN + 0.10:.2f} in)")

    # ---- vertical positions
    bot_y = 0.42                      # x labels of the bottom maps
    top_y = bot_y + MAP_IN + ROW_GAP_IN
    fig_h = top_y + MAP_IN + 0.45
    fig = plt.figure(figsize=(width_in, fig_h))

    def axes_in(x, y, w, h):
        return fig.add_axes([x / width_in, y / fig_h, w / width_in,
                             h / fig_h])

    def arrow(p0, p1, connection="arc3", text=None, text_xy=None):
        fig.add_artist(FancyArrowPatch(
            (p0[0] / width_in, p0[1] / fig_h),
            (p1[0] / width_in, p1[1] / fig_h), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=8, lw=0.9, color="0.25",
            connectionstyle=connection, shrinkA=0, shrinkB=0))
        if text:
            fig.text(text_xy[0] / width_in, text_xy[1] / fig_h, text,
                     fontsize=TEXT_PT + 0.5, ha="center", va="center",
                     bbox=dict(fc="white", ec="none", pad=1.0))

    top_mid = top_y + MAP_IN / 2.0
    bot_mid = bot_y + MAP_IN / 2.0

    # A: deck
    ha = _box_height_in(deck_lines)
    a_top = fig_h - 0.27              # room above for the panel letter
    y_wrap = bot_y + MAP_IN + 0.52    # height of the wrap-around arrow
    if a_top - ha < y_wrap + 0.08:
        raise VizError(
            f"panel A needs {ha:.2f} in of height and would reach the "
            f"wrap-around arrow at {y_wrap:.2f} in; shorten the deck lines "
            f"or raise ROW_GAP_IN")
    a_mid = a_top - ha / 2.0
    _text_box(fig, [xa / width_in, (a_top - ha) / fig_h,
                    BOX_W_IN / width_in, ha / fig_h], deck_lines, "A")

    # B: unit-potential solutions
    b_axes = []
    for k, name in enumerate(basis_electrodes):
        ax = axes_in(xb[k], top_y, MAP_IN + CBAR_IN, MAP_IN)
        sc = scene(f"basis:{name}")
        solved = [body_electrode_name(b.name) == name for b in sc.bodies]
        if sum(solved) != 1:
            raise VizError(f"basis panel {name!r}: expected one drawn body "
                           f"named {name!r}, found {sum(solved)}")
        edge = {i: (BASIS_ON_COLOR if on else BASIS_OFF_COLOR)
                for i, on in enumerate(solved)}
        _draw_map(ax, sc, True, colors=(dict(edge), edge))
        for b, on in zip(sc.bodies, solved):
            qs = [project(pp, "xy") for pp in b.polys]
            if on:
                for q in qs:
                    ax.fill(q[:, 0], q[:, 1], color=BASIS_ON_COLOR,
                            alpha=BASIS_ON_ALPHA, zorder=3.5, lw=0)
            qa = np.vstack(qs)
            ax.text(float(qa[:, 0].mean()), float(qa[:, 1].mean()),
                    "1 V" if on else "0 V",
                    fontsize=BASIS_ON_LABEL_PT if on else TEXT_PT - 0.5,
                    fontweight="bold" if on else "normal",
                    ha="center", va="center", zorder=7,
                    color="black" if on else "0.30")
        ax.set_title(f"φ$_{{\\mathrm{{{name}}}}}$ (V/V)",
                     fontsize=TEXT_PT, pad=2.5)
        if k > 0:
            ax.set_ylabel("")
        b_axes.append(ax)
    _letter(b_axes[0], "B")

    # C: applied voltages
    f0 = spec.geometry.rf_groups[0].frequency_hz
    t_us = np.linspace(0.0, N_PERIODS / f0 * 1e6, N_WAVE_PTS)
    wave_h = 0.85
    axc = axes_in(xc, top_mid - wave_h / 2 - 0.05, WAVE_W_IN, wave_h)
    pal = STYLE["groups"]
    for k, grp in enumerate(spec.geometry.rf_groups):
        axc.plot(t_us, drive_voltage(grp, t_us), lw=1.0,
                 color=pal[k % len(pal)], label=grp.name)
    t_mark = phases_deg[0] / 360.0 / f0 * 1e6
    axc.axvline(t_mark, color="0.40", lw=0.7, ls=":")
    axc.set_xlabel("t (µs)", fontsize=TEXT_PT, labelpad=1.5)
    axc.set_ylabel("V$_i$(t) (V)", fontsize=TEXT_PT, labelpad=1.0)
    axc.tick_params(labelsize=TEXT_PT, length=2.0, pad=1.5)
    axc.legend(fontsize=TEXT_PT, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(spec.geometry.rf_groups),
               handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
    _letter(axc, "C", dx=-0.36)

    # D: summed potential at the chosen phase
    deg = phases_deg[0]
    axd = axes_in(xd, bot_y, MAP_IN + CBAR_IN, MAP_IN)
    _draw_map(axd, scene(f"phi@{deg:g}"), True)
    axd.set_title(f"φ (V) at Ωt = {deg:g}°",
                  fontsize=TEXT_PT, pad=2.5)
    _letter(axd, "D")

    # E: integrator
    he = _box_height_in(int_lines)
    _text_box(fig, [xe / width_in, (bot_mid - he / 2) / fig_h,
                    BOX_W_IN / width_in, he / fig_h], int_lines, "E")

    # F: recorded flights, coloured from FLIGHT_COLORS
    trajs = [np.asarray(tr, float) for tr, _st, _lab in flights]
    fates = [FATE_NAMES.get(int(st["kind"]), f"fate {st['kind']}")
             for _tr, st, _lab in flights]
    fcols = list(FLIGHT_COLORS[:len(flights)])
    axf = axes_in(xf, bot_y, MAP_IN, MAP_IN)
    _draw_map(axf, scene(None, trajs=trajs, fates=fates), False,
              ray_colors=fcols)
    axf.set_title("")
    for line in axf.get_lines():
        line.set_linewidth(1.0)
    from ion_gym.physics.sim_build import generate_births
    starts = np.asarray(generate_births(spec), float)[:, :3]
    q0 = project(starts, "xy")
    axf.plot(q0[:, 0], q0[:, 1], ls="none", marker="+",
             ms=BIRTH_MARKER_PT, mew=1.1, color=STYLE["births"], zorder=6)
    import textwrap
    handles = [Line2D([], [], color=fcols[k], lw=1.4,
                      label=lab + "\n" + "\n".join(
                          textwrap.wrap(fates[k], LEGEND_WRAP)))
               for k, (_tr, _st, lab) in enumerate(flights)]
    handles.append(Line2D([], [], ls="none", marker="+",
                          ms=BIRTH_MARKER_PT, mew=1.1,
                          color=STYLE["births"], label="ion start"))
    axf.legend(handles=handles, fontsize=TEXT_PT, frameon=False,
               loc="center left", bbox_to_anchor=(1.03, 0.5),
               handlelength=1.2, labelspacing=1.0)
    _letter(axf, "F")

    # arrows (inch coordinates measured from the panel edges)
    arrow((xa + BOX_W_IN + 0.04, min(a_mid, top_mid)),
          (xb[0] - 0.36, min(a_mid, top_mid)))
    arrow((xb[-1] + MAP_IN + CBAR_IN + 0.16, top_mid),
          (xc - 0.44, top_mid))
    # wrap: from below panel C, down and left to above panel D
    xc_mid = xc + WAVE_W_IN / 2.0
    fig.add_artist(plt.Line2D(
        [xc_mid / width_in, xc_mid / width_in, (xd + 0.6) / width_in],
        [(top_y - 0.28) / fig_h, y_wrap / fig_h, y_wrap / fig_h],
        transform=fig.transFigure, color="0.25", lw=0.9))
    arrow((xd + 0.6, y_wrap), (xd + 0.6, bot_y + MAP_IN + 0.22),
          text="φ(t) = Σ$_i$ V$_i$(t) φ$_i$",
          text_xy=(0.5 * (xd + 0.6 + xc_mid), y_wrap))
    arrow((xd + MAP_IN + CBAR_IN + 0.06, bot_mid), (xe - 0.06, bot_mid))
    arrow((xe + BOX_W_IN + 0.06, bot_mid), (xf - 0.40, bot_mid))
    _mpl_release(fig)
    return fig
