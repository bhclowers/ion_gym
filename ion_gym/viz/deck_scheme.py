"""Deck scheme: an abbreviated deck listing beside the instrument it defines.

A scheme of this kind shows a reader what an ion_gym deck looks like and
which lines of it define which part of the instrument. Panel A is an excerpt
of the deck written by io.spec_io.deck_excerpt from SimSpec.to_dict(), so
every number shown is the number the solver receives. Panel B draws the
electrodes from the solver's own mask through viz_core.scene_from_simspec
and viz_core.draw_view (the per-panel code of render_mpl), and marks the
birth positions returned by physics.sim_build.generate_births, the sampler
the integrator uses. A 3-D deck gets one panel per view from views_for, so
the multi-axis rule holds without a separate code path.

Lines of the excerpt that define a drawn electrode, a drive group, or the
ion source are banded in the colour that element is drawn with in panel B.
The colours come from viz_core._body_colors and STYLE, the same authority
render_mpl uses, so the listing and the drawing cannot disagree.

Import-clean: importing this module draws nothing.
"""
from __future__ import annotations

import numpy as np

from ion_gym.io.spec_io import deck_excerpt
from ion_gym.viz.viz_core import (STYLE, VizError, _body_colors,
                                  _mpl_headless_if_needed, _mpl_release,
                                  body_electrode_name,
                                  draw_view, project, scene_from_simspec,
                                  view_extent, views_for)

# Named layout defaults (inches and points), documented rather than buried.
SCHEME_WIDTH_IN = 7.0        # JASMS two-column figure width (17.8 cm)
LISTING_FONT_PT = 6.5        # monospace size of the excerpt
LISTING_LINE_SPACING = 1.22  # line pitch as a multiple of the font size
BAND_ALPHA = 0.22            # opacity of the colour band behind a block
BIRTH_MARKER_PT = 7.0        # marker size of a birth position
TICK_FONT_PT = 7.5           # tick labels and electrode names in panel B


def _tag_color(tag, spec, el_color, grp_color, births_color):
    """Band colour for one excerpt row, or None. Rows under
    ('geometry', 'electrodes', i) take electrode i's drawn outline colour,
    rows under ('geometry', 'rf_groups', j) take group j's colour, and rows
    under ('source',) take the birth-marker colour."""
    if len(tag) >= 3 and tag[:2] == ("geometry", "electrodes"):
        return el_color.get(spec.geometry.electrodes[tag[2]].name)
    if len(tag) >= 3 and tag[:2] == ("geometry", "rf_groups"):
        return grp_color.get(spec.geometry.rf_groups[tag[2]].name)
    if len(tag) >= 1 and tag[0] == "source":
        return births_color
    return None


def deck_scheme_figure(spec, keep_paths, *, model=None, views=None,
                       births: bool = True, panel_labels=("A", "B"),
                       width_in: float = SCHEME_WIDTH_IN,
                       font_pt: float = LISTING_FONT_PT,
                       inline_width: int = 70,
                       label_electrodes: bool = True,
                       legend_loc: str = "upper left"):
    """Two-part scheme of one deck. Returns the matplotlib Figure; saving
    and display are the caller's decision.

    spec : SimSpec of a planar, r-z, or 3-D deck that scene_from_simspec
        accepts.
    keep_paths : key tuples passed to io.spec_io.deck_excerpt, naming the
        deck lines to show.
    model : an already built model, to skip the field build; None builds it.
    views : views to draw in panel B; None takes views_for(scene), which
        gives every orthogonal view of a 3-D deck.
    births : mark the birth positions of generate_births(spec).
    label_electrodes : write each electrode's deck name on its drawing.
    legend_loc : matplotlib legend location inside the first drawing panel
        (drive groups and the ion birth marker).
    """
    if len(panel_labels) != 2:
        raise ValueError(f"panel_labels needs two entries (listing, "
                         f"drawing), got {panel_labels!r}")
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    rows = deck_excerpt(spec.to_dict(), keep_paths,
                        inline_width=inline_width)
    scene = scene_from_simspec(spec, model, field=None, title=spec.name)
    vs = views_for(scene, views)
    cols, edges = _body_colors(scene.bodies)
    el_color = {body_electrode_name(b.name): edges[i]
                for i, b in enumerate(scene.bodies)}
    grp_names = sorted({b.group for b in scene.bodies if b.group})
    pal = STYLE["groups"]
    grp_color = {g: pal[k % len(pal)] for k, g in enumerate(grp_names)}
    # a group that drives no drawn body still gets a defined colour, so its
    # excerpt lines are banded rather than silently left plain
    for g in spec.geometry.rf_groups:
        grp_color.setdefault(g.name, pal[len(grp_color) % len(pal)])
    births_color = STYLE["births"]

    pts = None
    if births:
        from ion_gym.physics.sim_build import generate_births
        pts = np.asarray(generate_births(spec), float)[:, :3]
        if pts.size == 0:
            raise VizError(f"deck {spec.name!r}: generate_births returned "
                           f"no ions, so there is no source to mark; pass "
                           f"births=False to draw the scheme without one")

    # ---- size: the listing sets the height; drawing panels share the rest
    line_in = font_pt * LISTING_LINE_SPACING / 72.0
    list_h_in = line_in * (len(rows) + 1)
    longest = max(len(t) for t, _ in rows)
    list_w_in = 0.60 * font_pt / 72.0 * (longest + 2)
    draw_w_in = width_in - list_w_in - 0.9
    if draw_w_in < 2.0:
        raise VizError(
            f"deck excerpt is {longest} characters wide at {font_pt} pt, "
            f"leaving {draw_w_in:.2f} in for the drawing at a {width_in} in "
            f"figure width; shorten the keep paths, lower inline_width, or "
            f"widen the figure")
    ars = []
    for v in vs:
        a0, a1, b0, b1 = view_extent(scene, v)
        ars.append(max(b1 - b0, 1e-9) / max(a1 - a0, 1e-9))
    draw_h_in = draw_w_in * sum(ars) + 0.55 * len(vs)
    fig_h = max(list_h_in, draw_h_in) + 0.5
    fig = plt.figure(figsize=(width_in, fig_h))

    left_frac = list_w_in / width_in
    ax_list = fig.add_axes([0.0, 0.0, left_frac, 1.0])
    ax_list.set_axis_off()
    ax_list.set_xlim(0.0, list_w_in)
    ax_list.set_ylim(fig_h, 0.0)
    y = 0.35
    char_w = 0.60 * font_pt / 72.0
    for text, tag in rows:
        c = _tag_color(tag, spec, el_color, grp_color, births_color)
        if c is not None:
            ax_list.add_patch(Rectangle(
                (0.04, y - 0.5 * line_in), char_w * (longest + 1), line_in,
                facecolor=c, edgecolor="none", alpha=BAND_ALPHA, zorder=1))
        ax_list.text(0.08, y, text, family="monospace", fontsize=font_pt,
                     va="center", ha="left", zorder=2)
        y += line_in
    ax_list.text(0.04, 0.12, panel_labels[0], fontsize=10,
                 fontweight="bold", va="center", ha="left")

    # ---- drawing panels, stacked, axial coordinate horizontal (draw_view)
    x0 = left_frac + 0.55 / width_in
    w = draw_w_in / width_in
    top = 1.0 - 0.30 / fig_h
    for k, v in enumerate(vs):
        h = draw_w_in * ars[k] / fig_h
        ax = fig.add_axes([x0, top - h, w, h])
        top -= h + 0.55 / fig_h
        draw_view(ax, scene, v, field=False, colors=(cols, edges))
        ax.set_title("")
        ax.tick_params(labelsize=TICK_FONT_PT)
        if label_electrodes:
            # deck name at the centroid of each body's drawn vertices in
            # this view, so a reader can match the excerpt's "name" lines
            # to the metal
            for b in scene.bodies:
                q = np.vstack([project(pp, v) for pp in b.polys])
                ax.text(float(q[:, 0].mean()), float(q[:, 1].mean()),
                        body_electrode_name(b.name), fontsize=TICK_FONT_PT,
                        ha="center", va="center", zorder=7)
        if pts is not None:
            q = project(pts, v)
            ax.plot(q[:, 0], q[:, 1], ls="none", marker="+",
                    ms=BIRTH_MARKER_PT, mew=1.2, color=births_color,
                    zorder=6)
        if k == 0:
            ax.text(-0.02, 1.02, panel_labels[1], transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="bottom", ha="right")
    # legend: one entry per drive group plus the source, in panel B
    handles = [Line2D([], [], color=grp_color[g], lw=2.0, label=g)
               for g in grp_names]
    if pts is not None:
        handles.append(Line2D([], [], ls="none", marker="+",
                              ms=BIRTH_MARKER_PT, mew=1.2,
                              color=births_color, label="ion birth"))
    if handles:
        fig.axes[1].legend(handles=handles, loc=legend_loc, fontsize=7,
                           frameon=False, handlelength=1.6)
    _mpl_release(fig)
    return fig
