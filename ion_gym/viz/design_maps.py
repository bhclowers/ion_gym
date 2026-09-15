"""
design_maps.py -- parameter-space maps and 1-D design profiles.

Framework extension.  viz_core.Scene renders
world-frame (mm) instruments; a DESIGN MAP lives in parameter space
(e.g. wave amplitude vs wave velocity) and a DESIGN PROFILE is a 1-D
analytic curve with engineering marks on it.  Neither fits Scene's
world-frame contract, and per the rendering doctrine (R2: build it once)
they get a shared renderer here instead of inline matplotlib in a project.

Contract (same as every ion_gym renderer):
  * config-driven -- axes, labels, data all come in as arguments; no
    instrument constants live here (R1);
  * returns the Figure; never calls show()/savefig() itself;
  * no import side effects;
  * labels carry the operating point: `subtitle` is REQUIRED on both
    entry points because a design map without its operating point is
    not a result (certified-numbers doctrine).

Public surface
--------------
    ProfileCurve                 -- one labelled curve for render_profiles
    render_profiles              -- 1-D curves + vertical/horizontal marks
    render_param_map             -- classified 2-D map + contour overlays
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field as dfield

import numpy as np


def _wrap_sub(subtitle: str, width: int = 88) -> str:
    """Operating-point lines are long by doctrine (they carry the tune);
    wrap them so the figure shows ALL of it instead of truncating."""
    return "\n".join(textwrap.wrap(str(subtitle), width=width))


def _wrap_title(title: str, fig=None, fontsize: int = 9) -> str:
    """Wrap the TITLE too. Only the subtitle was wrapped, so a long title
    -- and titles carry the device and its tune by doctrine -- ran off both
    ends of the axes. Width is derived from the actual
    figure width at the actual font size, not a magic character count."""
    if fig is not None:
        w_in = fig.get_size_inches()[0]
        width = max(40, int(w_in * 72.0 / (fontsize * 0.55)))
    else:
        width = 88
    return "\n".join(textwrap.wrap(str(title), width=width,
                                   break_long_words=False))

from ion_gym.viz.viz_core import (VizError, _mpl_headless_if_needed,
                                  _mpl_release)


@dataclass
class ProfileCurve:
    """One labelled 1-D curve.

    x, y   : same length; units carried by the axis labels of the call.
    style  : optional matplotlib kwargs (color, ls, lw); empty -> cycle.
    """
    label: str
    x: np.ndarray
    y: np.ndarray
    style: dict = dfield(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.y = np.asarray(self.y, float)
        if self.x.shape != self.y.shape or self.x.ndim != 1:
            raise VizError(f"profile {self.label!r}: x{self.x.shape} and "
                           f"y{self.y.shape} must be equal-length 1-D")


def render_profiles(curves, *, xlabel, ylabel, title, subtitle,
                    vmarks=(), hmarks=(), logy=False, logx=False,
                    figsize=(7.0, 4.6)):
    """1-D design profiles with the marks a reader quotes numbers off.

    curves : list[ProfileCurve]
    vmarks : [(x, label)] vertical reference lines (e.g. a wall position)
    hmarks : [(y, label)] horizontal reference lines (e.g. kT)
    subtitle: the operating point, stated on the figure (required).
    """
    if not curves:
        raise VizError("render_profiles: no curves given -- an empty design "
                       "profile is a blank claim, refuse instead of drawing it")
    if not str(subtitle).strip():
        raise VizError("render_profiles: subtitle (operating point) is "
                       "required -- a profile without its tune is not a result")
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for c in curves:
        ax.plot(c.x, c.y, label=c.label, **c.style)
    for xv, lbl in vmarks:
        ax.axvline(float(xv), color="#555555", ls="--", lw=0.9)
        ax.annotate(f" {lbl}", (float(xv), ax.get_ylim()[1]),
                    ha="left", va="top", fontsize=8, rotation=90,
                    color="#333333")
    for yv, lbl in hmarks:
        ax.axhline(float(yv), color="#888888", ls=":", lw=0.9)
        ax.annotate(f" {lbl}", (ax.get_xlim()[0], float(yv)),
                    ha="left", va="bottom", fontsize=8, color="#333333")
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{_wrap_title(title, ax.figure)}\n{_wrap_sub(subtitle)}", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _mpl_release(fig)
    return fig


def render_param_map(xvals, yvals, class_grid, class_names, class_colors, *,
                     xlabel, ylabel, title, subtitle, contours=(),
                     marks=(), logx=False, logy=False, figsize=(7.4, 5.4),
                     legend_loc="upper right"):
    """Classified parameter-space map with quantitative contour overlays.

    xvals, yvals : 1-D axes of the map (len nx, ny)
    class_grid   : (ny, nx) int array indexing into class_names/class_colors
    class_names  : label per class index (drawn as a legend)
    class_colors : one colour per class index
    contours     : [(grid(ny,nx), levels, label, color)] overlays
    marks        : [(x, y, label)] annotated reference points
    subtitle     : operating point / fixed parameters (required).
    """
    xvals = np.asarray(xvals, float)
    yvals = np.asarray(yvals, float)
    cg = np.asarray(class_grid)
    if cg.shape != (len(yvals), len(xvals)):
        raise VizError(f"render_param_map: class_grid {cg.shape} does not "
                       f"match axes ({len(yvals)}, {len(xvals)})")
    if len(class_names) != len(class_colors):
        raise VizError("render_param_map: class_names and class_colors "
                       "lengths differ")
    lo, hi = int(cg.min()), int(cg.max())
    if lo < 0 or hi >= len(class_names):
        raise VizError(f"render_param_map: class index range [{lo},{hi}] "
                       f"outside the {len(class_names)} declared classes -- "
                       "an undeclared class would render as a lie")
    if not str(subtitle).strip():
        raise VizError("render_param_map: subtitle (operating point) is "
                       "required -- a map without its tune is not a result")
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    cmap = ListedColormap(list(class_colors))
    norm = BoundaryNorm(np.arange(len(class_names) + 1) - 0.5,
                        len(class_names))
    ax.pcolormesh(xvals, yvals, cg, cmap=cmap, norm=norm, shading="nearest")
    for grid, levels, label, color in contours:
        g = np.asarray(grid, float)
        if g.shape != cg.shape:
            raise VizError(f"contour {label!r}: shape {g.shape} does not "
                           f"match map {cg.shape}")
        cs = ax.contour(xvals, yvals, g, levels=levels,
                        colors=color, linewidths=1.0)
        ax.clabel(cs, fontsize=7, fmt=label + "=%g")
    for xm, ym, lbl in marks:
        ax.plot([float(xm)], [float(ym)], marker="o", ms=6, mfc="none",
                mec="black", mew=1.4)
        ax.annotate(f" {lbl}", (float(xm), float(ym)), fontsize=8,
                    ha="left", va="bottom")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{_wrap_title(title, ax.figure)}\n{_wrap_sub(subtitle)}", fontsize=9)
    ax.legend(handles=[Patch(fc=c, label=n) for n, c
                       in zip(class_names, class_colors)],
              fontsize=8, loc=legend_loc, framealpha=0.9)
    fig.tight_layout()
    _mpl_release(fig)
    return fig


def render_scalar_map(xvals, yvals, values, *, xlabel, ylabel, title,
                      subtitle, cbar_label, invalid_mask=None,
                      invalid_note="", marks=(), figsize=(7.4, 5.4),
                      cmap="viridis"):
    """Continuous scalar over a parameter plane (colorbar), with an
    optional INVALID mask rendered grey and stated on the figure --
    an invalid cell hidden as just-another-colour would lie."""
    xvals = np.asarray(xvals, float)
    yvals = np.asarray(yvals, float)
    v = np.asarray(values, float).copy()
    if v.shape != (len(yvals), len(xvals)):
        raise VizError(f"render_scalar_map: values {v.shape} does not "
                       f"match axes ({len(yvals)}, {len(xvals)})")
    if not str(subtitle).strip():
        raise VizError("render_scalar_map: subtitle (operating point) is "
                       "required")
    if invalid_mask is not None:
        m = np.asarray(invalid_mask, bool)
        if m.shape != v.shape:
            raise VizError("render_scalar_map: invalid_mask shape mismatch")
        if m.any() and not str(invalid_note).strip():
            raise VizError("render_scalar_map: invalid cells present but "
                           "invalid_note empty -- grey without a stated "
                           "reason is a hidden branch")
        v[m] = np.nan
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#bdbdbd")
    pm = ax.pcolormesh(xvals, yvals, np.ma.masked_invalid(v), cmap=cm,
                       shading="nearest")
    fig.colorbar(pm, ax=ax, label=cbar_label)
    for xm, ym, lbl in marks:
        ax.plot([float(xm)], [float(ym)], marker="o", ms=6, mfc="none",
                mec="black", mew=1.4)
        ax.annotate(f" {lbl}", (float(xm), float(ym)), fontsize=8,
                    ha="left", va="bottom")
    t = f"{_wrap_title(title)}\n{_wrap_sub(subtitle)}"
    if invalid_mask is not None and np.asarray(invalid_mask).any():
        t += f"\n[grey = {invalid_note}]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(t, fontsize=9)
    fig.tight_layout()
    _mpl_release(fig)
    return fig
