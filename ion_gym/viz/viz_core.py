"""
viz_core.py  --  the ONE image / report framework for ion_gym
=============================================================
Standing requirements this module exists to satisfy
(treat as binding):

  R1  AGNOSTIC.  Nothing here may know an instrument constant.  A renderer
      that only works for one frozen configuration is a renderer that will
      fail the next geometry.  Every scene is built from what the SOURCE
      OBJECT itself reports (a SimSpec's own domain box and electrode
      shapes; a Bench's own placed instances), never from a table of
      numbers typed in here.  The gate scans this file for instrument
      constants and refuses them.

  R2  BUILD IT ONCE.  Fields, schematics and ion paths all render through
      the same Scene -> View -> Renderer path, so the next subsystem gets
      pictures for free instead of a fresh one-off script.

  R3  MULTI-AXIS.  Anything with a 3-D component is presented in more than
      one projection (xy / xz / yz), because a single slice through a
      3-D instrument is a claim, not a picture.  A scene that is honestly
      planar says so (it carries slab_mm = 0) and renders one panel with
      that stated on it -- it does not fake a depth it never solved.

  R4  ZOOM POLICY (M-VIZ-Z).  Interactive (plotly) panels are NEVER
      scale-anchored.  See free_aspect_axes() for the mechanism and the
      root cause; the short version is that a scaleanchor makes the box
      zoom RIGID -- plotly reshapes whatever rectangle you drag -- and a
      long thin instrument is exactly the thing you need to zoom into a
      long thin rectangle.  True 1:1 scale is delivered by SIZING THE
      PANEL to the data box, which leaves the drag rectangle free.

  R5  PROVENANCE.  Every panel carries a stamp: SOLVER-DERIVED (geometry
      and field come from the same arrays the kernel flies through) or
      SCHEMATIC (a concept sketch, not evidence).  Mixing the two silently
      is how a drawing starts lying about an instrument.

Public surface
--------------
    Body, Rays, FieldSlab, Lineout, Scene        -- the data
    scene_from_simspec, scene_from_bench,
    scene_from_bodies                            -- the adapters (agnostic)
    lineout                                      -- cut a field, read a number
    project, views_for                           -- multi-axis machinery
    render_mpl, report                           -- static figures / reports
    free_aspect_axes, apply_zoom_policy,
    scene3d_layout, assert_free_zoom             -- the plotly zoom policy
    interactive_panel                                    -- interactive panel
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field as dfield
from typing import Sequence

import hashlib
from collections import OrderedDict
import math
import numpy as np

# ------------------------------------------------------------------ stamps
SOLVER = "SOLVER-DERIVED"      # geometry/field are the kernel's own arrays
SCHEMATIC = "SCHEMATIC"        # concept sketch -- not evidence
PROVENANCE = (SOLVER, SCHEMATIC)

VIEWS = ("xy", "xz", "yz")

# Vertical gap between stacked panels in render_mpl(layout="column"), as a
# fraction of the mean panel height. Matplotlib's 0.2 default assumes a
# one-line title and no xlabel; our panels carry a TWO-line title (view +
# slice position) above and an axis name-with-unit below, so at the default
# the lower panel's title lands on top of the upper panel's x-label
# (measured on a real 3-D geometry figure). Named rather than
# inlined so the reason travels with the number; the equal-aspect
# re-convergence pass below grows the figure to keep every panel's true
# data aspect, so widening this costs page height, never distortion.
COLUMN_PANEL_HSPACE = 0.42

# CONVENTION A: THE Z AXIS IS HORIZONTAL.
#
# 'xz' draws (z, x).  'yz' draws (z, y).  One table, no builder switch, no
# per-spec declaration, no guessing -- z is horizontal wherever it appears.
#
# The history is worth keeping, because two wrong answers preceded this one:
#
#   1. I first wrote this table as (2,0)/(2,1) -- z horizontal -- then RETRACTED
#      it on the grounds that the planar and r-z builders lay the beam along x.
#   2. I replaced it with a per-builder `axial_of_spec()` switch.  That switch
#      was WRONG FOR TWO OF THE FOUR 3-D EXAMPLES: the STL quadrupole and the
#      SLIM tetramer both transport down z and both report depth_mm == 0.0, so
#      every available discriminator (builder, coords, depth_mm) misclassified
#      at least one example.  A rule with a per-configuration exception is not a
#      convention, it is a lookup table that happens to be right about the cases
#      someone tested.
#
# The premise of (1) was itself the defect: an r-z spec stores PHYSICAL Z in the
# `width_mm` / x slot (example_einzel_rz: `width_mm=ZLEN`).  Its axial axis was
# never really x -- x was a misnomer for z.  That misnomer is exactly why the
# funnel and the SLIM 3-D example disagreed about which way the electrodes ran.
# Name the axial axis z everywhere and the disagreement dissolves; there is
# nothing left to declare.
_AXES = {"xy": (0, 1), "xz": (2, 0), "yz": (2, 1)}
_LABEL = {0: "x", 1: "y", 2: "z"}
# Station kind -> colour: ONE convention for every renderer (mpl panels,
# perspective, interactive). Bounds keep #9467bd dashed; stations are
# SOLID so "where the flight may end" and "where the instrument detects"
# never look alike.
_STATION_COLORS = {"detect": "#2ca02c", "record": "#17becf"}



class VizError(RuntimeError):
    """Refuse-with-diagnostic: a picture that would misrepresent the model."""


# -------------------------------------------------------------------- data
@dataclass
class Body:
    """A piece of metal (or any solid) as world-frame polygons.

    polys: list of (K,3) arrays, mm, closed implicitly.  A planar model
    supplies flat polygons in z = 0 and declares slab_mm on the Scene; a
    true 3-D model supplies its own faces.  The renderer never invents a
    third dimension the source did not give it.

    provenance: PER-BODY, and this is not decoration.  Q3 lesson: one honest
    figure of a 2-D solve of a long cell contains BOTH kinds of object at
    once.  The rods are translationally invariant, so extruding the solved
    cross-section along the axis is a true statement about them.  The finger
    ladder is NOT -- it has structure along the axis that a 2-D solve never
    saw, so the same extrusion is a drawing, not evidence.  A single
    scene-level stamp forces you to lie about one of them.  None = inherit the
    scene's stamp.
    """
    name: str
    polys: list
    volt: float | None = None
    # OUTLINE body ("chunky rods"): the polys are boundary
    # LOOPS of the electrode mask, drawn as unfilled paths — one clean
    # line per loop instead of one filled patch per voxel row.
    outline: bool = False
    # GRID electrode (grids are hard to see in the UI,
    # they might not know that they need to be tuned"). A transmission
    # grid occupies no cells in the ele label array, so the mask path
    # draws NOTHING for it -- a tunable electrode with no visual
    # existence. grid=True bodies are built from the DECLARED shapes and
    # rendered dashed, so a grid reads as "a plane you tune and ions
    # cross", visibly distinct from solid metal.
    grid: bool = False
    kind: str = "electrode"
    owner: str = ""             # which instance/field-array it came from
    provenance: str | None = None
    group: str | None = None      # RF/drive group: what distinguishes an
                                  # electrode when DC cannot (every RF device)

    def points(self) -> np.ndarray:
        return (np.vstack([np.asarray(p, float).reshape(-1, 3)
                           for p in self.polys])
                if self.polys else np.zeros((0, 3)))

    def stamp_in(self, scene) -> str:
        return self.provenance or scene.provenance


@dataclass
class Rays:
    """Ion paths.  paths: list of (N,3) mm.  fates: matching status names."""
    paths: list
    fates: list = dfield(default_factory=list)
    label: str = "ions"

    def points(self) -> np.ndarray:
        return (np.vstack([np.asarray(p, float).reshape(-1, 3)
                           for p in self.paths])
                if self.paths else np.zeros((0, 3)))


@dataclass
class FieldSlab:
    """A scalar field sampled on a regular grid in ONE plane.

    plane   : 'xy' | 'xz' | 'yz'   -- which projection it belongs to
    extent  : (a0, a1, b0, b1) mm in that plane's own two axes
    values  : (nb, na) array, indexed [b, a] (imshow order)
    quantity: e.g. 'phi (V)', '|E| (V/mm)'
    at      : the fixed coordinate of the third axis (mm), stated on the fig
    """
    plane: str
    extent: tuple
    values: np.ndarray
    quantity: str = "phi (V)"
    at: float = 0.0


@dataclass
class Lineout:
    """A 1-D cut through a field, with the marks that make it readable.

    Q3 lesson: the map panel shows you WHERE the field is; the line-out is
    what you actually READ A NUMBER OFF.  The coupling coefficient in that
    report was quoted from the on-axis value of a basis map -- so the figure
    has to carry that cut, with the value marked on it, or the number in the
    caption is unsupported by the picture printed next to it.

    s      : distance along the cut (mm), 0 at p0
    values : the sampled field
    marks  : [(value, label)] horizontal reference lines -- the quoted number
    """
    label: str
    s: np.ndarray
    values: np.ndarray
    quantity: str = ""
    marks: list = dfield(default_factory=list)


@dataclass
class Dim:
    """A dimension callout: the arrow and the number on a geometry review.

    Q3 lesson.  A geometry review is not a field picture with the field turned
    off -- its whole content is "this edge is 2.652 mm from that one, and here
    is why I read the paper that way."  Strip the callouts and you have a grey
    blob that reviews nothing.  So the framework carries them, and the next
    subsystem's dimensional review is a list of Dims instead of another 150
    lines of ax.annotate.

    view : which projection it belongs to ('xy' | 'xz' | 'yz')
    p0,p1: endpoints IN THAT VIEW's two coordinates (mm)
    label: the number, and what it is
    derived: True if the number is INFERRED, not stated by the source.  It
             renders differently, because "the paper says 2.5" and "I worked
             out 1.152 from two things the paper says" are not the same claim
             and a reviewer must be able to see which is which.
    """
    view: str
    p0: tuple
    p1: tuple
    label: str
    derived: bool = False


@dataclass
class Scene:
    """Everything a renderer needs, in ONE world frame (mm)."""
    title: str
    bodies: list = dfield(default_factory=list)
    rays: list = dfield(default_factory=list)          # list[Rays]
    # Per-plane METAL slices: the geometry counterpart of `fields`. When a
    # view has one, the panel draws the CUT through the instrument in that
    # plane instead of the projected body outlines — a projection of a long
    # rod stack is a solid block that tells you nothing about the aperture.
    metal: list = dfield(default_factory=list)         # list[FieldSlab]
    fields: list = dfield(default_factory=list)        # list[FieldSlab]
    lineouts: list = dfield(default_factory=list)      # list[Lineout]
    dims: list = dfield(default_factory=list)          # list[Dim]
    # DECLARED KILL BOUNDS: the spec's enabled BoundsSpec planes, as
    # (axis, coord_mm, label). These are where a flight is ALLOWED to end;
    # a figure that hides them shows a device with no exit. The interactive
    # view already draws them (sim_app, dashed #9467bd) — the static
    # renderer must agree, so the Scene carries them and render_mpl draws
    # them in the same convention.
    bound_marks: list = dfield(default_factory=list)   # list[(axis, mm, lbl)]
    # DECLARED STATIONS (a declared detector was not showing on
    # detector"): the spec's StationSpec planes — detectors and record
    # planes WITH their transmission windows. A bound is where a flight is
    # merely allowed to end; a detect station is the instrument's exit,
    # and a figure that hides it shows an analyzer with no detector. Each
    # entry: {"name", "kind", "axis", "pos_mm", "window": {ax: (lo, hi)}}.
    # Drawn as a finite SEGMENT/FACE over the window (solid, kind-
    # coloured), visibly distinct from the infinite dashed bounds.
    station_marks: list = dfield(default_factory=list)
    bounds: tuple | None = None      # (x0,x1,y0,y1,z0,z1) mm; None -> derived
    slab_mm: float = 0.0             # declared depth of a planar model (0 = none)
    provenance: str = SOLVER
    notes: list = dfield(default_factory=list)
    frame: str = "world"
    # DISPLAY names of the three WORLD axes. scene_transpose swaps scene
    # DATA into the canonical frame and swaps these names the same way, so
    # every panel keeps speaking the DECK's coordinate names (a transposed
    # r-z trap shows its axial "x" vertical, still labelled x).
    axis_names: tuple = ("x", "y", "z")

    # ---- derived
    def __post_init__(self):
        if self.provenance not in PROVENANCE:
            raise VizError(f"provenance must be one of {PROVENANCE}, "
                           f"got {self.provenance!r}")

    def points(self) -> np.ndarray:
        chunks = [b.points() for b in self.bodies] + [r.points()
                                                      for r in self.rays]
        chunks = [c for c in chunks if len(c)]
        return np.vstack(chunks) if chunks else np.zeros((0, 3))

    def box(self) -> tuple:
        if self.bounds is not None:
            return tuple(float(v) for v in self.bounds)
        P = self.points()
        if not len(P):
            raise VizError(f"scene {self.title!r} is empty: nothing to draw "
                           "and no bounds declared")
        lo, hi = P.min(0), P.max(0)
        if self.slab_mm > 0:                 # a declared slab has real depth
            lo[2] = min(lo[2], -0.5 * self.slab_mm)
            hi[2] = max(hi[2], +0.5 * self.slab_mm)
        pad = 0.02 * max(float((hi - lo).max()), 1e-9)
        lo, hi = lo - pad, hi + pad
        return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]),
                float(lo[2]), float(hi[2]))

    def is_3d(self) -> bool:
        x0, x1, y0, y1, z0, z1 = self.box()
        span = max(x1 - x0, y1 - y0, 1e-12)
        return (z1 - z0) > 1e-6 * span

    def stamps(self) -> set:
        """Every provenance actually present in this scene."""
        return {b.stamp_in(self) for b in self.bodies} | {self.provenance}

    def stamp(self) -> str:
        st = self.stamps()
        if len(st) > 1:
            n = sum(1 for b in self.bodies
                    if b.stamp_in(self) == SCHEMATIC)
            return (f"MIXED  ({n} of {len(self.bodies)} bodies are SCHEMATIC "
                    "-- hatched; the rest are as flown)")
        s = self.provenance
        if self.provenance == SOLVER:
            s += "  (geometry + field as flown)"
        else:
            s += "  (concept sketch -- NOT evidence)"
        return s


# ---------------------------------------------------------------- projection
def model_planes(model) -> tuple:
    """Which principal planes does THIS model actually have?

    Root cause of a real crash.  pe_view.compute_component decided a
    model's dimensionality by CALLING pe_surface(plane=...) and catching
    TypeError.  That is not a capability probe, it is a signature probe, and it
    lies twice over:

      * any TypeError raised INSIDE pe_surface -- a real bug, a bad mz, a None
        array -- was swallowed and re-reported as a geometric claim,
        "this model has no 'xz' plane".  The user is told about their geometry
        when what actually happened is that their solver threw.
      * PlanarModel.pe_surface has no `plane` kwarg either, so it took the same
        branch.  The probe never detected dimensionality at all.

    And the message it raised -- "a 2-D model has only xy" -- was false for the
    very model that triggered it: an r-z model's plane is r-z, not xy.

    So the model DECLARES, and nobody guesses.  A model that does not declare
    is refused LOUDLY (doctrine C: a loader misses loudly, it never coerces) --
    the alternative is to assume ("xy",) and hand a 3-D model a silent lie.
    """
    pl = getattr(model, "PLANES", None)
    if pl is None:
        raise VizError(
            f"{type(model).__name__} does not declare PLANES. Every model must "
            f"state which principal planes it has -- guessing is how "
            f"'this model has no xz plane' got raised at a user whose solver "
            f"had actually thrown a TypeError. Add a PLANES class attribute.")
    pl = tuple(pl)
    if not pl:
        raise VizError(f"{type(model).__name__}.PLANES is empty")
    return pl


def extrude(poly2d, a0, a1, axis: int = 2) -> list:
    """Cross-section -> prism: the two end caps AND the side quads.

    R3 (multi-axis) has a failure mode that looks like success: hand the
    renderer only the cross-section and the xy panel is perfect while the xz
    and yz panels quietly show a LINE.  The body is still "in" every view, so
    nothing raises -- it has simply lost two of its three dimensions.  A body
    that is translationally invariant along an axis has an honest silhouette
    in all three, and this is the one call that produces it.

    poly2d : (K,2) in the two axes OTHER than `axis`, in ascending axis order
             (axis=2 -> (x,y);  axis=0 -> (y,z);  axis=1 -> (x,z))
    a0, a1 : the extent along `axis` (mm)
    """
    if axis not in (0, 1, 2):
        raise VizError(f"extrude axis {axis!r}; expected 0, 1 or 2")
    P = np.asarray(poly2d, float).reshape(-1, 2)
    if len(P) < 3:
        raise VizError(f"extrude needs >=3 points, got {len(P)}")
    if float(a1) == float(a0):
        raise VizError(f"zero-length extrusion at {a0}: a prism with no depth "
                       f"is a polygon; pass it as one rather than pretending")
    i, j = [k for k in range(3) if k != axis]
    caps = []
    for a in (a0, a1):
        C = np.zeros((len(P), 3))
        C[:, i], C[:, j], C[:, axis] = P[:, 0], P[:, 1], float(a)
        caps.append(C)
    polys = [caps[0], caps[1]]
    n = len(P)
    for k in range(n):
        m = (k + 1) % n
        polys.append(np.array([caps[0][k], caps[0][m], caps[1][m], caps[1][k]]))
    return polys


def _drop_collinear(poly, tol=1e-9):
    """Remove points that lie on the segment between their neighbours.

    Exact simplification of a CLOSED loop: the boundary is unchanged, only
    the number of points describing it. Used before `extrude` so a plate
    whose outline was walked cell by cell raises one side face per genuine
    edge rather than one per grid cell. A loop that reduces below a triangle
    is returned untouched -- `extrude` refuses it with its own diagnostic,
    which is a better message than anything this helper could give.
    """
    P = np.asarray(poly, float).reshape(-1, 2)
    if len(P) < 4:
        return P
    keep = []
    n = len(P)
    for k in range(n):
        a, b, c = P[(k - 1) % n], P[k], P[(k + 1) % n]
        if abs(np.cross(b - a, c - b)) > tol:
            keep.append(b)
    return np.asarray(keep) if len(keep) >= 3 else P


def project(P, view: str) -> np.ndarray:
    """(N,3) mm -> (N,2) in the named projection, oriented per Convention A."""
    if view not in _AXES:
        raise VizError(f"unknown view {view!r}; known: {VIEWS}")
    i, j = _AXES[view]
    P = np.asarray(P, float).reshape(-1, 3)
    return np.c_[P[:, i], P[:, j]]


def view_labels(view: str, scene: "Scene" = None) -> tuple:
    if view not in _AXES:
        raise VizError(f"unknown view {view!r}; known: {VIEWS}")
    i, j = _AXES[view]
    names = getattr(scene, "axis_names", None) or (_LABEL[0], _LABEL[1],
                                                   _LABEL[2])
    return f"{names[i]} (mm)", f"{names[j]} (mm)"


def view_extent(scene: Scene, view: str) -> tuple:
    x0, x1, y0, y1, z0, z1 = scene.box()
    lo = (x0, y0, z0)
    hi = (x1, y1, z1)
    i, j = _AXES[view]
    ext = (lo[i], hi[i], lo[j], hi[j])
    # REFUSE-WITH-DIAGNOSTIC (name the views that WOULD work:
    # yz and it doesn't render properly at all"). A planar model has NO
    # extent along z, so an axial view's data box collapses to zero width
    # and every consumer downstream — the zoom policy, the pane sizing,
    # the heatmap axes — silently produced a degenerate panel that looked
    # broken rather than saying why. Pre-existing (v206 had no guard); the
    # check lives HERE because view_extent is the one function both
    # renderers ask for a panel's box, so no consumer can forget it.
    span = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-12)
    for k, (a, b) in ((i, (lo[i], hi[i])), (j, (lo[j], hi[j]))):
        if (b - a) <= 1e-6 * span:
            raise VizError(
                f"{scene.title!r}: view {view!r} needs the "
                f"{_LABEL[k]} axis, but this scene has no extent along it "
                f"({_LABEL[k]} = {a:g} mm exactly). This is a 2-D "
                "cross-section: it was solved in one plane, so there is no "
                f"axial structure to show. Available view(s): "
                f"{', '.join(views_for(scene, None))}. "
                "TO SEE THE OTHER VIEWS anyway, do what the app does and "
                "supply the third dimension from the route's DECLARED "
                "symmetry: voxel_views_2d(model.ele, h_mm, symmetry='rz'|"
                "'planar') returns {'3d', 'rz', 'end-on'} for r-z "
                "(revolution) and {'3d', 'xy'} for planar (slab "
                "extrusion). Nothing is re-solved and no plane is "
                "invented — the 2-D labels ARE the computation. For real "
                "axial structure (a device that varies along z), solve a "
                "3-D model.")
    return ext



def scene_transpose(scene: Scene, axes: tuple = ("x", "y")) -> Scene:
    """Return a NEW Scene with two WORLD axes swapped — any geometry.

    Display-only by construction: the scene's data (bodies, rays, metal and
    field slabs, bounds, kill-bound marks, dims) is swapped into the
    canonical frame, and `axis_names` is swapped the SAME way, so every
    panel keeps speaking the source deck's coordinate names. Solver arrays
    are never touched — a Scene is already a copy of what was solved.

    Introduced for Paul-trap figures (endcaps
    top/bottom as the trap papers draw them; deck-specific choice, NOT a new
    default), but written for any axis pair and any geometry because a
    one-family display hack is exactly the rework this module exists to
    prevent. Involution: scene_transpose(scene_transpose(sc, a), a) == sc.
    """
    import copy as _copy
    _idx = {"x": 0, "y": 1, "z": 2}
    if (len(axes) != 2 or axes[0] not in _idx or axes[1] not in _idx
            or axes[0] == axes[1]):
        raise VizError(f"scene_transpose: axes must be two distinct of "
                       f"x/y/z, got {axes!r}")
    i, j = _idx[axes[0]], _idx[axes[1]]
    perm = [0, 1, 2]
    perm[i], perm[j] = perm[j], perm[i]
    _letter = {v: k for k, v in _idx.items()}
    ax_map = {_letter[a]: _letter[perm[a]] for a in range(3)}

    def _pts(a):
        a = np.asarray(a, float).copy()
        if a.ndim == 2 and a.shape[1] == 3:
            a[:, [i, j]] = a[:, [j, i]]
            return a
        raise VizError(f"scene_transpose: expected (N,3) points, got shape "
                       f"{a.shape} — refusing to guess the layout.")

    def _slab(sl: FieldSlab) -> FieldSlab:
        ia, ib = _AXES[sl.plane]                     # (horizontal, vertical)
        na, nb = perm[ia], perm[ib]
        new_plane = [v for v in VIEWS if set(_AXES[v]) == {na, nb}]
        if len(new_plane) != 1:
            raise VizError(f"scene_transpose: no canonical plane for axes "
                           f"{na},{nb} from {sl.plane!r}")
        new_plane = new_plane[0]
        vals = np.asarray(sl.values).copy()
        ext = tuple(sl.extent)
        if _AXES[new_plane] == (na, nb):             # orientation preserved
            pass
        elif _AXES[new_plane] == (nb, na):           # horizontal<->vertical
            vals = vals.T.copy()
            ext = (ext[2], ext[3], ext[0], ext[1])
        else:
            raise VizError("scene_transpose: unreachable plane orientation")
        return FieldSlab(new_plane, ext, vals, sl.quantity, at=sl.at)

    sc = _copy.copy(scene)
    sc.bodies = []
    for b in scene.bodies:
        nb_ = _copy.copy(b)
        nb_.polys = [_pts(pp) for pp in b.polys]
        _ol = getattr(b, "outline", None)
        if (_ol is not None and not isinstance(_ol, (bool, int, float))
                and np.asarray(_ol).ndim == 2
                and np.asarray(_ol).shape[1] == 3):
            nb_.outline = _pts(_ol)
        sc.bodies.append(nb_)
    sc.rays = []
    for r in scene.rays:
        nr = _copy.copy(r)
        nr.paths = [_pts(pp) for pp in r.paths]
        sc.rays.append(nr)
    sc.metal = [_slab(m) for m in scene.metal]
    sc.fields = [_slab(f) for f in scene.fields]
    sc.dims = []
    for d in scene.dims:
        nd = _copy.copy(d)
        for fname, val in list(vars(nd).items()):
            arr = np.asarray(val) if not isinstance(val, str) else None
            if arr is not None and arr.ndim == 2 and arr.shape[-1] == 3:
                setattr(nd, fname, _pts(val))
            elif (arr is not None and arr.ndim == 1 and arr.shape == (3,)
                  and arr.dtype != object):
                v3 = arr.astype(float).copy()
                v3[[i, j]] = v3[[j, i]]
                setattr(nd, fname, type(val)(v3) if isinstance(
                    val, (tuple, list)) else v3)
        sc.dims.append(nd)
    # kill-bound marks: canonical letter follows the DATA; the human label
    # keeps the deck's own wording (axis_names keeps the panels agreeing).
    sc.bound_marks = [(ax_map.get(a, a), mm, lbl)
                      for (a, mm, lbl) in scene.bound_marks]
    # stations transpose exactly as bounds do: the plane axis AND every
    # window axis follow the data into the canonical frame.
    sc.station_marks = [dict(s, axis=ax_map.get(s["axis"], s["axis"]),
                             window={ax_map.get(a, a): w
                                     for a, w in s["window"].items()})
                        for s in scene.station_marks]
    if scene.bounds is not None:
        lo = [scene.bounds[0], scene.bounds[2], scene.bounds[4]]
        hi = [scene.bounds[1], scene.bounds[3], scene.bounds[5]]
        lo[i], lo[j] = lo[j], lo[i]
        hi[i], hi[j] = hi[j], hi[i]
        sc.bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    names = list(getattr(scene, "axis_names", ("x", "y", "z")))
    names[i], names[j] = names[j], names[i]
    sc.axis_names = tuple(names)
    sc.notes = list(scene.notes) + [
        f"display transposed: world {axes[0]}<->{axes[1]} "
        f"(scene_transpose; solver frame untouched)"]
    return sc


def views_for(scene: Scene, views: Sequence[str] | None = None) -> tuple:
    """R3.  A 3-D scene is shown in THREE projections by default; a scene
    that is honestly planar (no z extent, no declared slab) gets the one
    panel it has evidence for.  Explicit `views` always wins."""
    if views is not None:
        for v in views:
            if v not in _AXES:
                raise VizError(f"unknown view {v!r}; known: {VIEWS}")
        return tuple(views)
    return VIEWS if scene.is_3d() else ("xy",)


# ----------------------------------------------------------------- adapters
def _rect_polys(x0, x1, y0, y1, z0=0.0, z1=0.0):
    """One rectangle in the width plane; if z1>z0 also the two other faces,
    so an extruded slab projects honestly in xz / yz."""
    face = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])
    if z1 <= z0:
        return [face]
    back = face.copy()
    back[:, 2] = z1
    return [face, back,
            np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
            np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]])]


def _rz_full(arr, axis):
    """Mirror a stored half-plane about r = 0 into the FULL cross-section —
    the convention build_rz and the interactive r-z view already display.
    The shared r = 0 row is not duplicated, so the result is 2n-1 wide."""
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(None, 0, -1)
    return np.concatenate([arr[tuple(sl)], arr], axis=axis)


# fate codes the 3-D/planar kernels return (tracer3d.fly3d `kind`)
FATE_NAMES = {0: "struck metal or left the domain", 1: "left the field box",
              2: "still in flight at t_max", 3: "crossed a declared plane"}


def world_off(model):
    """The model's field->WORLD frame offset: world_off_mm (mirror +
    declared origin composed, the signed-frame convention) with the
    pure-mirror fallback for models that predate signed origins. ONE
    helper so no consumer can pick the stored frame by accident again —
    a phi_slice_3d straggler and a frame audit
    (describe_fates + five PE-tab sites) all made exactly that pick.
    Deliberate pure-mirror consumers (the staged fields pack +
    staged_flight reader, where Region-local frames are the contract
    and deck origins ride the poses) read mirror_off_mm explicitly and
    say why. scene_from_model3d was listed here as deliberate until the
    frame certification measured its metal 7.80 mm from its own
    overlaid trajectories on a signed hexapole deck — it now
    reads the pack's world_off_mm like every other display consumer."""
    return (getattr(model, "world_off_mm", None)
            or getattr(model, "mirror_off_mm", (0.0, 0.0, 0.0)))


def describe_fates(spec, model, summaries, max_named=3):
    """Plain-language account of HOW a population of ions ended.

    A fate code alone ("kind 0") does not tell you whether a run went well.
    This names the electrode an ion ended on where that can be determined
    from the geometry, and says explicitly that a timeout means the ion was
    STILL FLYING at the cap — which in a confining device is the intended
    outcome, not a truncated result. Returns a list of sentences.
    """
    import collections
    if not summaries:
        return []
    _tail = []
    # DELIVERED SPREAD: the completion print reaches every
    # headless/notebook run -- the environment where the sub-cell
    # actually lived (four sessions against a 0.19-cell artifact, no
    # stats card in sight). One authority, appended as the last sentence;
    # a failure REPORTS instead of eating the fates account.
    try:
        from ion_gym.physics.stats import delivered_spread_text
        _tail.append(delivered_spread_text(spec))
    except (ImportError, ValueError, AttributeError, TypeError, IndexError) as e:
        # ImportError if stats import fails; others from delivered_spread_text
        # (no births, missing symmetry, numpy operations). Charter: report.
        _tail.append(f"(delivered spread unavailable: {e})")
    # grid pitch: planar/r-z models expose mm_per_gu, 3-D models h_mm —
    # the same either/or the rest of this module uses. Reading only one of
    # them silently produced h = 0 and attributed every strike to nothing.
    h = float(getattr(model, "mm_per_gu", None)
              or getattr(model, "h_mm", None) or 0.0)
    ele = np.asarray(getattr(model, "ele", np.zeros((0, 0))))
    # WORLD frame (frame audit): summaries record
    # world-mm end points; mirror-only indexing misnamed struck
    # electrodes on signed-origin 3-D decks by origin/h voxels.
    moff = tuple(world_off(model))
    _is_rz = (getattr(getattr(spec.geometry, "symmetry", None), "coords", None)
              == "rz")
    # Label -> name. `index` is OPTIONAL and is frequently None, in which
    # case the voxeliser labels electrodes 1..N in declaration order — so a
    # plain getattr(el, "index", k) maps every electrode to the key None and
    # no strike is ever named.
    names = {}
    for k, el in enumerate(getattr(spec.geometry, "electrodes", []), start=1):
        lab = getattr(el, "index", None)
        names[int(lab) if lab is not None else k] = (
            getattr(el, "name", None) or f"electrode {k}")

    def _hit(su):
        """(status, electrode name) for one ion's end point.

        The kernel stops an ion AT the surface it is about to strike, so the
        recorded end point usually sits in the last vacuum cell rather than
        inside the metal. Looking only at that one cell therefore names
        nothing; a small neighbourhood search finds the surface the ion
        actually arrived at. status is 'metal', 'outside' or 'vacuum'.
        """
        if h <= 0 or ele.size == 0:
            return ("vacuum", None)
        if _is_rz and ele.ndim == 2:
            # r-z: axis 0 is axial, axis 1 is RADIUS >= 0 measured from u0.
            # A trajectory carries a SIGNED transverse coordinate (the ion
            # crosses the axis), so the radial index must come from the
            # magnitude — indexing with the signed value lands off-grid and
            # attributes every strike to nothing.
            # model.ele is the STORED HALF, indexed j = r/h from the axis.
            # (model.u0 is the origin of the MIRRORED full array the kernel
            # is handed, not of this one — subtracting it here put every
            # lookup off the end of the grid.)
            r = math.hypot(float(su.get("y_end", 0.0)),
                           float(su.get("z_end", 0.0)))
            idx = [int(round(float(su.get("x_end", 0.0)) / h)),
                   int(round(r / h))]
        else:
            idx = []
            for a, key in zip(range(ele.ndim), ("x_end", "y_end", "z_end")):
                v = float(su.get(key, 0.0)) - moff[a]
                idx.append(int(round(v / h)))
        if any(i < -1 or i > n for i, n in zip(idx, ele.shape)):
            return ("outside", None)          # left the field domain
        sl = tuple(slice(max(0, i - 1), min(n, i + 2))
                   for i, n in zip(idx, ele.shape))
        win = ele[sl]
        lab = win[win > 0]
        if lab.size:
            vals, counts = np.unique(lab, return_counts=True)
            return ("metal", names.get(int(vals[int(np.argmax(counts))])))
        return ("vacuum", None)

    by_kind = collections.defaultdict(list)
    for su in summaries:
        by_kind[int(su.get("kind", 2))].append(su)
    out = []
    for kind, group in sorted(by_kind.items()):
        n = len(group)
        if kind == 0:
            res = [_hit(su) for su in group]
            hits = collections.Counter(nm for st, nm in res
                                       if st == "metal" and nm)
            n_out = sum(1 for st, _ in res if st == "outside")
            n_unnamed = sum(1 for st, nm in res if st == "metal" and not nm)
            n_vac = sum(1 for st, _ in res if st == "vacuum")
            parts = []
            if hits:
                parts.append("struck " + ", ".join(
                    f"{nm} (x{c})" for nm, c in hits.most_common(max_named)))
            if n_out:
                parts.append(f"{n_out} left the field domain")
            if n_unnamed:
                parts.append(f"{n_unnamed} struck unnamed metal")
            if n_vac:
                parts.append(f"{n_vac} stopped in vacuum")
            out.append(f"{n} ion(s) terminated: " + "; ".join(parts))
        elif kind == 2:
            tmax = getattr(spec.integration, "t_max_us", None)
            out.append(f"{n} ion(s) were STILL FLYING at t_max"
                       + (f" = {tmax:g} us" if tmax else "")
                       + " — not a failure in a confining device; raise "
                         "t_max only if you expect them to leave")
        else:
            out.append(f"{n} ion(s) {FATE_NAMES.get(kind, 'ended (kind %d)' % kind)}")
    return out + _tail


def _revolve_polys(polys, n_seg=72):
    # n_seg 24 -> 72 (rings rendered end-on read as
    # 24-gons; the SOLVE is exactly axisymmetric, so any facets are
    # display-only, and 72 reads as a circle at figure resolution).
    # Named default; callers may still pass a coarser sweep.
    """Revolve axial-radial polygons about the axis into 3-D.

    An r-z model is axisymmetric: its (axial, radial) cross-section IS the
    instrument, and the solid it describes is that section swept through
    2*pi. Extruding it as a slab would be wrong — a revolved ring is not a
    rectangular prism — so the sweep is done properly here, which is what
    lets an axisymmetric device be shown in three orthogonal projections
    instead of one.

    Input polys are (K,3) with x = axial, y = radius >= 0, z = 0. Each is
    emitted at n_seg angles as a rotated copy, so every projection sees the
    true silhouette: the axial views show the section at its extreme radii,
    the transverse view shows the ring.
    """
    # QUAD-STRIP SWEEP. The previous implementation emitted
    # n_seg ROTATED PLANAR COPIES of the section; in the transverse
    # projection each copy is seen edge-on, rendering a ring as radial
    # SPOKES instead of an annulus (caught on a drift tube).
    # A revolve is a surface: sweep each section edge into quads between
    # consecutive angles, and every projection then sees the truth —
    # the transverse view a filled annulus, the axial views the correct
    # silhouette.
    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_seg) + 1)
    cs, ss = np.cos(thetas), np.sin(thetas)
    out = []
    for q in polys:
        a = np.asarray(q, float)
        k = a.shape[0]
        for i in range(k):
            x0, r0 = a[i, 0], a[i, 1]
            x1, r1 = a[(i + 1) % k, 0], a[(i + 1) % k, 1]
            for j in range(int(n_seg)):
                out.append(np.array([
                    [x0, r0 * cs[j],     r0 * ss[j]],
                    [x1, r1 * cs[j],     r1 * ss[j]],
                    [x1, r1 * cs[j + 1], r1 * ss[j + 1]],
                    [x0, r0 * cs[j + 1], r0 * ss[j + 1]]]))
    return out


def _drive_label(spec, el):
    """Compact drive text for legends: class + every driving number
    (the solver-true view states voltages)."""
    groups = {g.name: g for g in (spec.geometry.rf_groups or [])}
    parts = []
    for gname in (getattr(el, "rf_groups", None) or []):
        g = groups.get(gname)
        if g is not None:
            parts.append("RF {0:g}V@{1:g}kHz ph{2:g}".format(
                g.amplitude_v, g.frequency_hz / 1e3, g.phase_deg))
    if el.dc not in (None, 0.0):
        parts.append("DC {0:g}V".format(el.dc))
    return el.name + " [" + (" + ".join(parts) if parts else "GND") + "]"


def scene_from_simspec(spec, model=None, *, field: str | None = "phi",
                       field_mz=None, field_charge=None,
                       title: str | None = None, slab_mm: float = 0.0,
                       slab_z_mm: tuple | None = None,
                       provenance: str = SOLVER,
                       per_electrode: bool = True,
                       trajs=None, fates=None, revolve: int = 0,
                       slice_at: dict | None = None) -> Scene:
    """Adapter: any planar SimSpec -> Scene.  AGNOSTIC: geometry comes from
    the SOLVER'S OWN electrode mask (not from a whitelist of shape types the
    viewer happens to know how to draw -- that is how a drawing starts
    disagreeing with the thing that was flown), and the field comes from the
    SAME arrays build_planar hands the kernel.  DISPLAY == SOLVER INPUT.

    per_electrode (Q3 lesson): one Body per electrode, carrying its NAME and
    its VOLTAGE, taken from the int16 LABELS in `model.ele`.  Not from a
    re-rasterisation of the spec's shapes -- that is a second opinion, and two
    opinions are how a picture drifts from an array.  The identity is checked,
    not assumed: the union of the labelled rects must reproduce `ele > 0`
    exactly or this REFUSES to draw.  (`ele` is int16 precisely because
    `ele == i` on a bool mask is true for every conductor and paints them all
    at electrode #1's voltage -- see the C1-C5 cache doctrine.  This adapter
    is where that defect becomes visible, so this is where it is gated.)

    field: 'phi' | 'E' | 'basis:<electrode name>' | 'phi@<degrees>'.

    Array convention, stated because it is a classic sign error: build_planar
    indexes gx = x/h FIRST, so A/ExA/EyA/ele are [ix, iy].  A FieldSlab is
    imshow-ordered [ib, ia], so the arrays are transposed exactly once, here.
    """
    from ion_gym.physics.build_planar import build_planar_model
    if model is None:
        model = build_planar_model(spec)
    # models expose the grid pitch as mm_per_gu (planar/rz) or h_mm
    # (Stl3DModel) — the same two-name reality every consumer handles.
    # Assuming one name made scene_from_simspec fail (loudly, per Phase
    # A's guard) on every STL example. Config-agnostic, per doctrine.
    _h = (getattr(model, "mm_per_gu", None)
          or getattr(model, "h_mm", None))
    if _h is None:
        raise VizError(f"{type(model).__name__} exposes neither mm_per_gu "
                       f"nor h_mm — cannot place its voxels in mm")
    h = float(_h)
    ele = np.asarray(model.ele)
    # SYMMETRY: an r-z model stores r >= 0 only. Showing that half is showing
    # half the instrument, so unfold about the axis for display — the same
    # full cross-section build_rz and the interactive r-z view present. (A
    # mirrored 3-D scene3d model already arrives unfolded from the build, so
    # only r-z needs doing here.) DISPLAY == SOLVER INPUT still holds: the
    # reflection is exact, not interpolated.
    _rz = (getattr(getattr(spec.geometry, "symmetry", None), "coords", None)
           == "rz" and ele.ndim == 2)
    _rz_off_y = 0.0
    if _rz and not revolve:
        # flat display: mirror the stored half into the full cross-section
        _rz_off_y = -(ele.shape[1] - 1) * h
        ele = _rz_full(ele, 1)
    nx, ny = ele.shape[0], ele.shape[1]
    # build_planar is NODE-CENTRED: node i at i*h, the
    # 0 .. nx*h -- which is the width the SPEC declares -- and node (0,0) sits
    # at (h/2, h/2).  Pass that origin; do not let the renderer assume one.
    # KERNEL FRAME: the native rasterizers now sample at
    # i*h (node-centred, matching the fly kernel and the rasterizer), so node
    # (0,0) sits AT the origin and the node domain spans (n-1)*h. The
    # old ORG = (h/2, h/2) described the retired cell-centred sampling.
    # CANONICAL frame: the Scene — the R1 single
    # source of truth every geometry question consults — places bodies so
    # mirrored axes read [-H,+H] with the mirror plane at 0. The model's
    # mirror_off_mm is zero on non-mirrored axes and absent on 2-D models,
    # so every existing scene is byte-identical.
    _MOFF = tuple(getattr(model, "mirror_off_mm", (0.0, 0.0, 0.0)))
    # SIGNED FRAME: the planar model's anchor_mm carries the
    # deck's declared origin (plus any sub-pitch plane residue) — node
    # (0,0) sits AT anchor, so the scene must place bodies/fields/bounds
    # there or a signed-origin deck draws at [0, W] while flying at
    # [lo, lo+W] (display != solver frame). Legacy anchors are (0, 0):
    # every existing scene is byte-identical.
    _ANC = tuple(getattr(model, "anchor_mm", (0.0, 0.0)))
    _MOFF = (_MOFF[0] + _ANC[0], _MOFF[1] + _ANC[1], _MOFF[2])
    if _rz_off_y:
        _MOFF = (_MOFF[0], _MOFF[1] + _rz_off_y, _MOFF[2])
    ORG = (_MOFF[0], _MOFF[1])
    W, H = (nx - 1) * h, (ny - 1) * h
    if _rz and revolve:
        # the swept solid reaches +/- R out of the section plane
        z0, z1 = -(ele.shape[1] - 1) * h, (ele.shape[1] - 1) * h
    elif ele.ndim == 3:
        # A genuinely 3-D model HAS a z extent — take it from the grid. It
        # used to declare (0, 0) unless a slab was set, so Scene.box()
        # (which prefers declared bounds) reported zero depth, is_3d() said
        # False, and views_for gave ONE panel: a 3-D instrument silently
        # shown in a single projection — the exact failure the multi-axis
        # rule exists to prevent.
        z0, z1 = _MOFF[2], _MOFF[2] + (ele.shape[2] - 1) * h
    elif slab_z_mm is not None:
        # EXPLICIT z APERTURE.  slab_mm alone centres the slab on z = 0,
        # which is a true statement only for a device whose flight is
        # centred there.  A drift-axis MRT is not: its OA sits near z = 0
        # and its detector near z = Lz, so a symmetric slab draws metal
        # over a mirror image of the drift that no ion ever visits and
        # spends half the panel on it.  The extrusion is legitimate either
        # way (a z-invariant solve extrudes truthfully — Body.provenance),
        # so what is declared here is the APERTURE, not new physics.
        z0, z1 = (float(slab_z_mm[0]), float(slab_z_mm[1]))
        if not z1 > z0:
            raise VizError(
                f"scene_from_simspec: slab_z_mm must be (z0, z1) with "
                f"z1 > z0; got {slab_z_mm!r}. A non-positive aperture "
                f"would silently collapse the xz/yz panels to a line "
                f"rather than refusing.")
        if slab_mm and abs(slab_mm - (z1 - z0)) > 1e-9:
            raise VizError(
                f"scene_from_simspec: slab_mm={slab_mm:g} contradicts "
                f"slab_z_mm={slab_z_mm!r} (depth {z1 - z0:g} mm). Pass one "
                f"or the other; two disagreeing depths cannot both be drawn.")
        slab_mm = z1 - z0
    else:
        z0, z1 = ((-0.5 * slab_mm, 0.5 * slab_mm) if slab_mm > 0
                  else (0.0, 0.0))

    metal = ele > 0
    bodies = []
    if per_electrode:
        if ele.dtype == bool:
            # REFUSE-WITH-DIAGNOSTIC.  This is the C1-C5 defect arriving at the
            # display boundary, which is the only place it is visible: the solve
            # is unaffected (bases come from the per-electrode masks), so the
            # numbers stay right and only the PICTURE lies.  Degrading quietly
            # to one grey blob is how it survived three "fixes".
            raise VizError(
                f"{spec.name!r}: model.ele is a BOOLEAN mask, not int16 labels. "
                "`ele == i` is then true for every conductor and false for all "
                "i > 1, so every electrode would draw at electrode #1's voltage. "
                "Root cause is upstream (a builder, or a cache loader coercing "
                "on the way out -- see the C1-C5 rules); it is not fixable here. "
                "Pass per_electrode=False only if you genuinely want undifferen"
                "tiated metal.")
        drawn = np.zeros_like(metal)
        for idx, el in enumerate(spec.geometry.electrodes, start=1):
            m = (ele == idx)
            if not m.any():
                # No solve cells: a transmission grid (is_grid), or wholly
                # outside the domain. A grid is still a TUNABLE electrode;
                # draw it from its DECLARED rect shapes as a dashed outline
                # (display == solver input: the grid's plane and voltage
                # are solver inputs even though no cell is metal). Non-rect
                # grid shapes are skipped WITH a statement, not silently.
                if getattr(el, "is_grid", False):
                    for sh in (el.shapes or []):
                        pr = getattr(sh, "params", None) or {}
                        if getattr(sh, "type", getattr(sh, "kind", ""))                                 != "rect" or not pr:
                            print(f"[scene] grid electrode {el.name!r}: "
                                  f"non-rect shape not drawable as a grid "
                                  f"outline; skipped")
                            continue
                        gx0, gy0 = pr["x_mm"], pr["y_mm"]
                        gx1 = gx0 + pr["width_mm"]
                        gy1 = gy0 + pr["height_mm"]
                        loop = np.array([[gx0, gy0, z0], [gx1, gy0, z0],
                                         [gx1, gy1, z0], [gx0, gy1, z0]])
                        bodies.append(Body(
                            name=_drive_label(spec, el) + " (grid)",
                            polys=[loop], volt=float(el.dc),
                            group=getattr(el, "rf_group", None),
                            kind="electrode", owner=spec.name,
                            outline=True, grid=True))
                continue
            drawn |= m
            if m.ndim == 3:
                # 3-D mask (Stl3DModel, slim3d): the body's z-span comes
                # from the mask's OWN z occupancy — cell k spans
                # k*h..(k+1)*h — and the footprint is the z-projection.
                # Exact for extruded plates/rods (every 3-D builder here);
                # a z-varying profile would get its bounding z-span,
                # stated not hidden. This is what lets Scene.box()/is_3d()
                # answer honestly for 3-D models (viz R2, Phase A).
                kk = np.where(m.any(axis=(0, 1)))[0]
                # node-centred: node k owns [k*h - h/2, k*h + h/2]
                ez0 = (float(kk.min()) - 0.5) * h + _MOFF[2]
                ez1 = (float(kk.max()) + 0.5) * h + _MOFF[2]
                foot = m.any(axis=2)
            else:
                ez0, ez1, foot = z0, z1, m
            if m.ndim == 2 and not (_rz and revolve):
                # OUTLINE bodies for 2-D cross-sections. The
                # per-row rect decomposition renders a curved electrode as
                # stacked slabs; the 0.5-level contour of the SAME mask is
                # the same solver truth as one closed path per boundary
                # loop. The drawn==metal guard below is mask-based and
                # unaffected. FLAT r-z outlines too: it
                # was excluded only because "revolve consumes rects", and
                # the revolve branch consumes the outline itself now —
                # curved r-z electrodes (orbitrap spindle, ITMS
                # hyperbolae) drew as stacked row-slabs on the flat path,
                # the exact defect this branch was written to kill. A
                # rect-built r-z deck outlines to the same rectangle
                # boundary. 3-D keeps faces for projections.
                loops = mask_outline_mm(foot, h, ORG)
                if ez1 > ez0:
                    # A DECLARED SLAB gives this body real DEPTH, so it needs
                    # an honest silhouette in xz/yz as well. Pinning every
                    # loop at z0 -- what this branch once did --
                    # left the xy panel perfect while xz/yz showed a LINE at
                    # the slab's near face, which is the exact failure
                    # `extrude`'s own docstring warns about: the body is
                    # still "in" every view, so nothing raises, it has simply
                    # lost two of its three dimensions. Found on an MRT
                    # top view, where the whole mirror stack vanished while
                    # the panel note still claimed the metal was drawn.
                    # Extruding the OUTLINE rather than the per-row rects
                    # keeps the reason this branch exists in the first place:
                    # a curved electrode stays ONE boundary path and never
                    # renders as stacked slabs.
                    polys = []
                    for lp in loops:
                        if len(lp) > 1 and np.allclose(lp[0], lp[-1]):
                            lp = lp[:-1]        # extrude() closes the loop
                        # DROP COLLINEAR POINTS FIRST. mask_outline_mm walks
                        # the mask cell by cell, so a plain rectangular plate
                        # arrives as several hundred points along four
                        # straight edges. Extruding each of those raises one
                        # side quad per CELL: the silhouette is correct but
                        # it is drawn as hundreds of overlapping quads, which
                        # stack into a solid black bar and cost the renderer
                        # the same. Collinear removal is exact -- it changes
                        # no boundary, only how many points describe it -- so
                        # a rectangle becomes four faces and a curved
                        # electrode keeps every point its curvature needs.
                        lp = _drop_collinear(lp)
                        polys.extend(extrude(lp, ez0, ez1, axis=2))
                else:
                    polys = [np.c_[lp, np.full(len(lp), z0)] for lp in loops]
                bodies.append(Body(name=_drive_label(spec, el), polys=polys,
                                   volt=float(el.dc),
                                   group=getattr(el, "rf_group", None),
                                   kind="electrode", owner=spec.name,
                                   outline=True))
                continue
            if _rz and revolve:
                # Revolve the SECTION OUTLINE — the 0.5-level contour of
                # the SAME solver mask the flat branch draws — not the
                # per-row rect decomposition (orbitrap). The
                # rect route was written when every r-z deck WAS rects;
                # on a curved electrode it revolves each row-rect's
                # interior edges into internal surfaces (the "stacked
                # slabs" defect mask_outline_mm was created to kill,
                # reborn in 3-D) and emits O(height/h) times the quads
                # (119k on the 0.05 mm orbitrap spindle). The outline is
                # one closed loop per boundary of the same mask: the
                # swept surface is the electrode's true silhouette, for
                # rect-built decks it IS the rects' own edge line, and
                # the drawn==metal guard below is mask-based, unaffected.
                sect = []
                for lp in mask_outline_mm(foot, h, ORG):
                    if len(lp) > 1 and np.allclose(lp[0], lp[-1]):
                        lp = lp[:-1]      # _revolve_polys closes the loop
                    sect.append(np.c_[lp, np.zeros(len(lp))])
                grps = [_revolve_polys(sect, revolve)]
            else:
                grps = [_rect_polys(x0, x1, y0, y1, ez0, ez1)
                        for (x0, x1, y0, y1) in mask_rects_mm(foot, h, ORG)]
            bodies.append(Body(name=_drive_label(spec, el),
                               polys=[p for g in grps for p in g],
                               volt=float(el.dc),
                               group=getattr(el, "rf_group", None),
                               kind="electrode",
                               owner=spec.name))
        if not np.array_equal(drawn, metal):
            raise VizError(
                f"display mask != solver mask for {spec.name!r}: the labelled "
                f"electrodes cover {int(drawn.sum())} nodes but the solver's "
                f"metal mask has {int(metal.sum())} -- refusing to draw a "
                "picture that disagrees with the array the kernel flew "
                "through.  (Classic cause: `ele` arrived as a BOOLEAN mask, "
                "so every label but 1 is empty.)")
    else:
        grps = [_rect_polys(x0, x1, y0, y1, z0, z1)
                for (x0, x1, y0, y1) in mask_rects_mm(metal, h, ORG)]
        bodies = [Body(name=f"{spec.name}:metal",
                       polys=[p for g in grps for p in g],
                       kind="electrode", owner=spec.name)] if grps else []

    # ---- per-plane slices (3-D models): one cut per principal plane, the
    # same electrode-dense choice pe_surface/field_surface use, so the
    # picture is a SLICE of the instrument rather than everything projected
    # on top of itself.
    metal_slices, plane_fields = [], []
    if ele.ndim == 3 and hasattr(model, "_plane_slice"):
        _nz = ele.shape
        _plane_failures = []
        for _pl in ("xy", "xz", "yz"):
            # WHERE to cut. The default is the model's electrode-dense rule,
            # which is right for a board or a plate stack but wrong for a rod
            # set: "most metal" cuts THROUGH a rod, hiding the very aperture
            # the figure is about. `slice_at` lets the caller declare the cut
            # in mm (canonical frame); every panel states the position it used.
            _want = None
            if slice_at and _pl in slice_at and h > 0:
                _axp = {"xy": 2, "xz": 1, "yz": 0}[_pl]
                _want = int(round((float(slice_at[_pl]) - _MOFF[_axp]) / h))
            try:
                _ax, _k = model._plane_slice(_pl, _want)
            except (ValueError, IndexError, KeyError) as e:
                # AUDITED: was `except Exception: continue` —
                # a loop silently eating its errors. A plane can
                # legitimately be unsliceable (degenerate axis); that is
                # RECORDED and reported after the loop, and if EVERY
                # plane fails the scene refuses rather than rendering a
                # fieldless figure as if nothing happened.
                _plane_failures.append((_pl, f"{type(e).__name__}: {e}"))
                continue
            _sl = [slice(None)] * 3
            _sl[_ax] = _k
            _msk = np.asarray(ele[tuple(_sl)], float)
            _phi = np.asarray(model.A[tuple(_sl)], float)
            _co = [np.arange(n) * h + _MOFF[i] for i, n in enumerate(_nz)]
            # The slab's (a, b) axes are THE VIEW'S axes in the VIEW'S order
            # (_AXES, Convention A: axial horizontal) — not array order.
            # Array order (x<y<z) coincides for 'xy' but is SWAPPED for
            # 'xz'/'yz': emitting array order painted those cuts transposed,
            # and the true-scale panel then clipped them to the overlap
            # corner (a three-ring einzel rendered as one stray block).
            _ia, _ib = _AXES[_pl]
            _ext = (float(_co[_ia][0]), float(_co[_ia][-1]),
                    float(_co[_ib][0]), float(_co[_ib][-1]))
            _at = float(_co[_ax][_k])
            # _msk/_phi carry the residual axes in ARRAY order; a FieldSlab
            # is [ib, ia]. Transpose exactly when the residual order equals
            # (a, b) — derived from _AXES, so any future view added there
            # renders correctly without touching this.
            _T = ([i for i in range(3) if i != _ax] == [_ia, _ib])
            metal_slices.append(FieldSlab(_pl, _ext,
                                          _msk.T if _T else _msk,
                                          "metal", at=_at))
            plane_fields.append(FieldSlab(_pl, _ext,
                                          _phi.T if _T else _phi,
                                          "phi (V)", at=_at))
        if _plane_failures:
            # visible record: which cuts were skipped and why.
            print("[scene3d] plane slices skipped: "
                  + "; ".join(f"{p} ({m})" for p, m in _plane_failures))
        if _plane_failures and not plane_fields:
            raise ValueError(
                "scene_from_simspec: EVERY plane slice failed — "
                + "; ".join(f"{p}: {m}" for p, m in _plane_failures)
                + " — refusing to render a fieldless 3-D scene.")
    fields = []
    if plane_fields:
        fields = plane_fields
    elif field:
        _mz = field_mz
        if _mz is None and field in ("psi", "pe", "pseudo"):
            _lst = list(getattr(getattr(spec, "source", None), "mz_list", []) or [])
            if not _lst:
                raise VizError(
                    "field 'psi': no field_mz given and the deck's source "
                    "carries no mz_list, so there is no mass to compute a "
                    "pseudopotential for -- state one rather than assume.")
            _mz = float(_lst[0])
        _chg = field_charge
        if _chg is None:
            _chg = int(getattr(getattr(spec, "source", None), "charge", 1) or 1)
        vals, q = _field_slab_arrays(model, field, mz=_mz, charge=_chg)
        if _rz_off_y:                     # slab is [iy, ix]; y = r is axis 0
            vals = _rz_full(vals, 0)
        fields.append(FieldSlab("xy", (_MOFF[0], _MOFF[0] + W,
                                       _MOFF[1], _MOFF[1] + H), vals, q,
                                at=0.0))
    sc = Scene(title=title or f"{spec.name}",
               bodies=([] if metal_slices else bodies), fields=fields,
               metal=metal_slices,
               bounds=((_MOFF[0], _MOFF[0] + W, -H, H, z0, z1)
                       if (_rz and revolve) else
                       (_MOFF[0], _MOFF[0] + W, _MOFF[1], _MOFF[1] + H,
                        z0 + _MOFF[2], z1 + _MOFF[2])), slab_mm=slab_mm,
               provenance=provenance,
               notes=[f"domain {W:.3g} x {H:.3g} mm, h = {h:g} mm/gu "
                      f"(node-centred: node (0,0) at "
                      f"{ORG[0]:g}, {ORG[1]:g} mm)",
                      f"{len(spec.geometry.electrodes)} electrode(s); metal "
                      "drawn from the solver's own mask"])
    if slab_mm > 0:
        sc.notes.append(f"slab depth {slab_mm:g} mm DECLARED, not solved: "
                        "the xz/yz panels extrude the width-plane solve")
    # ION PATHS. A geometry figure shows where the field is; the trajectories
    # show what it DID. Columns are (t, x, y, z, ...) — a 2-D model has no z,
    # so those paths lie in the display plane at z = 0.
    if trajs is not None:
        paths = []
        for ti, tr in enumerate(trajs):
            a = np.asarray(tr, float)
            if a.ndim != 2 or not len(a):
                continue
            # CONTRACT GUARD: trajs are RAW KERNEL RECORDS,
            # columns (t, x, y, z, ...). A caller handing bare (x, y, z)
            # triples used to be sliced as (t, x, y) and drawn as
            # coordinate garbage with no error — display != flown, the
            # exact defect class this adapter exists to prevent. Records
            # have non-decreasing t; anything else refuses by name.
            if a.shape[1] < 3 or np.any(np.diff(a[:, 0]) < 0):
                raise VizError(
                    f"trajs[{ti}] is not a kernel record array: expected "
                    f"columns (t, x, y, z, ...) with non-decreasing t, "
                    f"got shape {a.shape} with col-0 range "
                    f"[{a[:, 0].min():g}, {a[:, 0].max():g}]. Pass the "
                    f"fly() record array itself, not extracted "
                    f"coordinates.")
            z = a[:, 3] if a.shape[1] > 3 else np.zeros(len(a))
            paths.append(np.c_[a[:, 1], a[:, 2], z])
        if paths:
            sc.rays = [Rays(paths=paths, fates=list(fates or []))]
            sc.notes.append(f"{len(paths)} ion path(s) overlaid, as flown")
    # DECLARED KILL BOUNDS. Enumerated by the same authority the impact
    # statistics use (stats.enabled_planes — ONE enumeration, every
    # consumer), so the figure and the stats can never disagree about which
    # planes are on. Frame note: bound coordinates are drawn AT THE DECLARED
    # VALUE, which is exactly the number the kernel compares against — r-z
    # y-bounds bind the SIGNED transverse coordinate, and canonical mirrored
    # frames are spec == display by the [-H,+H] rule, so no conversion.
    if getattr(spec, "bounds", None) is not None:
        from ion_gym.physics.stats import enabled_planes
        sc.bound_marks = [(ax, mm, f"{lbl} mm (bound)")
                          for lbl, ax, mm in enabled_planes(spec)]
        if sc.bound_marks:
            sc.notes.append(f"{len(sc.bound_marks)} declared bounding "
                            "plane(s) drawn dashed")
    # DECLARED STATIONS: taken from the deck verbatim (the same
    # StationSpec objects physics.stations consumes — ONE authority, so
    # the figure and the detection statistics can never disagree about
    # where the detector is or how big its window is).
    for st in (getattr(spec, "stations", None) or []):
        sc.station_marks.append({
            "name": st.name, "kind": st.kind, "axis": st.axis,
            "pos_mm": float(st.pos_mm),
            "window": {a: (float(lo), float(hi))
                       for a, (lo, hi) in (st.window or {}).items()}})
    if sc.station_marks:
        # The note names the AXIS and SPAN, not just the count, because a
        # deck can declare a scan ladder too dense to label individually;
        # the reader must still be able to say what is on the figure.
        _by_ax = {}
        for _s in sc.station_marks:
            _by_ax.setdefault(_s["axis"], []).append(_s["pos_mm"])
        _spans = ", ".join(
            (f"{len(v)} on {a} at {v[0]:g} mm" if len(v) == 1 else
             f"{len(v)} on {a} spanning {min(v):g} to {max(v):g} mm")
            for a, v in sorted(_by_ax.items()))
        sc.notes.append(f"{len(sc.station_marks)} declared station(s) "
                        f"drawn solid over their windows ({_spans})")
    return sc


def _field_slab_arrays(model, field, *, mz=None, charge=1):
    """[ix,iy] solver arrays -> [iy,ix] imshow arrays (one transpose, here).

    Selectors, all read off the model the kernel was handed:
      'phi'              the static field, as flown
      'E' / '|E|'        its magnitude
      'psi'              the SECULAR pseudopotential an ion of mass m/z feels:
                         q*phi_DC + q*V_pseudo with V_pseudo = q E0^2/(4 m W^2)
                         (Dehmelt adiabatic form). This is the right layer for
                         an RF device -- 'phi' at phase 0 on an RF-only deck is
                         identically zero, a picture of nothing.
                         Requires the model to expose pe_surface,
                         which is the same PE the pe_view route uses; m/z is
                         supplied by the caller, never guessed.
      'basis:<name>'     the DIMENSIONLESS response phi/V of ONE electrode with
                         every other grounded.  This is the definition of a
                         coupling coefficient, and it needs no second solve:
                         the bases are already in the model.
      'phi@<deg>'        the instantaneous potential at one RF phase, from the
                         model's OWN sin drives.  A picture of an RF field at
                         an unstated phase is not a picture of anything.
      'E@<deg>'          the magnitude of that same instantaneous field.  On
                         an RF-only device plain 'E' is identically zero (no
                         dc), so the phase is not optional here either.
    """
    # A FieldSlab is ONE PLANE by contract ("(nb, na) array"). A 3-D model's
    # model.A is (nx,ny,nz), so returning it whole handed the renderer a
    # 3-D array — the slab contract broken silently. Take the model's OWN
    # representative slice (potential_image / efield_magnitude pick the
    # electrode-dense plane and state the RF phase), so the picture is the
    # plane the model itself reports rather than an arbitrary index.
    _is3d = np.asarray(getattr(model, "A", np.empty(0))).ndim == 3
    if field in ("psi", "pe", "pseudo"):
        if not hasattr(model, "pe_surface"):
            raise VizError(
                f"field {field!r}: this model exposes no pe_surface(), so the "
                f"pseudopotential cannot be computed for it. Use 'phi' or "
                f"'E@<deg>' instead, or add pe_surface to the model -- do not "
                f"substitute a different quantity silently.")
        if mz is None:
            raise VizError(
                f"field {field!r}: the pseudopotential depends on m/z "
                f"(V_pseudo ~ 1/m), so it cannot be drawn without one. Pass "
                f"mz= from the deck being rendered.")
        _x, _y, PE, _ele = model.pe_surface(mz=float(mz), charge=int(charge))
        PE = np.asarray(PE, float).copy()
        _ele = np.asarray(_ele)
        # HALF-PLANE CONTRACT. Every selector here returns the STORED
        # frame; a declared mirror is unfolded ONCE, downstream
        # (_rz_full at
        # the scene call site). pe_surface, by its own contract, returns the
        # ALREADY-UNFOLDED cross-section for a cylindrical/mirrored model, so
        # passing it through mirrored the field a SECOND time — drawn at half
        # transverse scale (the r = 10 mm wall at |y| ~ 5) with the clip
        # plateau closing into spurious contour rings near the electrodes.
        # A quadrupole cannot have a second minimum; the picture had to be
        # wrong even though the ions were fine. When the surface spans BOTH
        # signs of the transverse coordinate while the stored array is
        # one-sided, keep the stored (non-negative) half; a genuinely
        # one-sided surface (planar devices) matches the stored shape and
        # passes through untouched; anything else is refused, never guessed.
        _A = np.asarray(getattr(model, "A", np.empty(0)))
        if _A.ndim == 2 and PE.shape != _A.shape:
            _yv = np.asarray(_y, float).ravel()
            _diff = [k in (0, 1) and PE.shape[k] != _A.shape[k]
                     for k in range(2)]
            _ax = _diff.index(True) if _diff.count(True) == 1 else -1
            _n = _A.shape[_ax] if _ax >= 0 else -1
            if (_ax >= 0 and _yv.size == PE.shape[_ax]
                    and _yv.min() < 0.0 < _yv.max()
                    and PE.shape[_ax] in (2 * _n, 2 * _n - 1)):
                _keep = np.flatnonzero(_yv >= -1e-9)
                if _yv[_keep][0] > _yv[_keep][-1]:      # store axis-outward
                    _keep = _keep[::-1]
                _sl = [slice(None), slice(None)]
                _sl[_ax] = _keep
                PE, _ele = PE[tuple(_sl)], _ele[tuple(_sl)]
            if PE.shape != _A.shape:
                raise VizError(
                    f"field {field!r}: pe_surface returned shape {PE.shape}, "
                    f"matching neither the stored grid {_A.shape} nor a "
                    f"single mirror-unfold of it (transverse span "
                    f"[{np.asarray(_y).min():g}, {np.asarray(_y).max():g}] "
                    "mm). Refusing to draw a field on a grid the solver did "
                    "not produce.")
        # PE diverges at the metal (|E|^2 -> large): mask the conductor so the
        # colour scale is set by the region ions actually occupy, and clip the
        # near-surface tail at the 90th percentile (same treatment as the
        # pe_view overlay, so the two routes agree).
        _m = np.asarray(_ele) > 0.5
        PE[_m] = np.nan
        _fin = PE[np.isfinite(PE)]
        if _fin.size:
            PE = np.minimum(PE, float(np.percentile(_fin, 90.0)))
        return PE.T, f"pseudopotential (eV, m/z {float(mz):g})"
    if field == "phi":
        if _is3d:
            _x, _y, img, _e = model.potential_image()
            return np.asarray(img, float).T, "phi (V)"
        return np.asarray(model.A, float).T, "phi (V)"
    if field in ("E", "|E|"):
        if _is3d:
            return (np.asarray(model.efield_magnitude(), float).T,
                    "|E| (V/mm)")
        if hasattr(model, "ExA"):
            Ex = np.asarray(model.ExA, float).T * 1e-3  # V/m -> V/mm
            Ey = np.asarray(model.EyA, float).T * 1e-3
            return np.hypot(Ex, Ey), "|E| (V/mm)"
        # 2-D route without ExA/EyA (the r-z model).
        # RZModel.efield_magnitude() returns ZEROS on a DC solve (its Ez
        # channels are not the composed DC field — a known limitation),
        # so |E| is the gradient of the COMPOSED potential A: the same
        # object the tracer differentiates in flight, at the solver's own
        # pitch. Config-agnostic: any 2-D model exposing A and its pitch
        # renders |E| through this path.
        A = np.asarray(model.A, float)
        h = float(getattr(model, "mm_per_gu", 0.0) or model.h_mm)
        gx, gy = np.gradient(A, h)                      # V/mm on [ix, iy]
        return np.hypot(gx, gy).T, "|E| (V/mm)"
    if isinstance(field, str) and field.startswith("basis:"):
        return _basis_slab(model, field.split(":", 1)[1])
    if isinstance(field, str) and field.startswith("phi@"):
        return _phase_slab(model, field[4:])
    if isinstance(field, str) and field.startswith(("E@", "|E|@")):
        # |E| AT AN RF PHASE (the E field plot is
        # blank -- by design?"). It was not by design, and it was not a
        # renderer fault either: plain 'E' is the magnitude of the STATIC
        # composed field, and an RF-only device has dc = 0 on every
        # electrode, so |E_static| is identically zero -- honest, and
        # useless. The force an ion actually feels comes from the
        # instantaneous field, so the phase must be named here exactly as
        # it is for 'phi@<deg>'. Gradient of the SAME composed potential,
        # at the solver's own pitch, so the two panels cannot disagree.
        phi, theta = _phase_potential(model, field.split("@", 1)[1])
        if phi.ndim == 3:
            raise VizError("'E@<deg>' on a 3-D model: ask the model for a "
                           "plane first (slice_at) -- a magnitude over a "
                           "volume is not a panel.")
        h = float(getattr(model, "mm_per_gu", 0.0) or model.h_mm)
        gx, gy = np.gradient(phi, h)                    # V/mm on [ix, iy]
        return (np.hypot(gx, gy).T,
                f"|E| (V/mm) at RF phase {theta:g} deg")
    raise VizError(f"unknown field {field!r}; use 'phi', 'E', "
                   "'basis:<electrode>', 'phi@<degrees>' or "
                   "'E@<degrees>'")


def _basis_slab(model, name):
    bases = getattr(model, "bases", None) or {}
    names = [el.name for el in model.spec.geometry.electrodes]
    if name not in names:
        raise VizError(f"no electrode named {name!r}; this model has {names}")
    idx = names.index(name) + 1                 # int16 labels are 1-based
    if idx not in bases:
        raise VizError(
            f"electrode {name!r} has no basis on this model (has "
            f"{sorted(bases)}).  A model built before bases were kept cannot "
            "produce a basis panel -- rebuild it; do not substitute a "
            "re-solve, which is a second opinion, not this model.")
    # bases are solved at v_basis = 1e4 V (build_planar's convention, and the
    # ONE place that number is allowed to live is build_planar -- read it from
    # there rather than hard-coding it here).
    from ion_gym.physics.build_planar import V_BASIS
    return np.asarray(bases[idx], float).T / V_BASIS, f"phi / V({name})"


def _phase_potential(model, deg):
    """Composed potential at one RF phase, in SOLVER array order [ix,iy].

    One composition, two consumers ('phi@<deg>' and 'E@<deg>'), so the
    potential a phase panel shows and the field magnitude drawn from it
    can never disagree.
    """
    try:
        theta = float(deg)
    except ValueError:
        raise VizError(f"'@{deg}': the phase must be a number in degrees")
    Bk = list(getattr(model, "Bk", []) or [])
    if not Bk:
        raise VizError(f"'@{deg}' asked for an RF phase, but this model has "
                       "no sin drives -- there is no phase to show. Use "
                       "'phi' or 'E'.")
    phi = np.asarray(model.A, float).copy()
    for B, _f0, ph in Bk:
        B = np.asarray(B, float)
        if B.ndim == 3:
            B = B[:, :, 0]
        phi = phi + np.sin(np.radians(theta + ph)) * B
    return phi, theta


def _phase_slab(model, deg):
    phi, theta = _phase_potential(model, deg)
    return phi.T, f"phi (V) at RF phase {theta:g} deg"

def lineout(scene_or_slab, p0, p1, *, n: int = 400, label: str = "",
            marks=()) -> Lineout:
    """Sample a FieldSlab along the segment p0 -> p1, in the slab's own plane
    coordinates (mm).  Bilinear on the solver's grid; no smoothing, no
    resampling to a prettier axis.  AGNOSTIC: it knows nothing but the extent
    the slab declares."""
    slab = (scene_or_slab.fields[0]
            if isinstance(scene_or_slab, Scene) else scene_or_slab)
    if not isinstance(slab, FieldSlab):
        raise VizError("lineout needs a FieldSlab (or a Scene that has one)")
    a0, a1, b0, b1 = slab.extent
    V = np.asarray(slab.values, float)                 # [ib, ia]
    nb, na = V.shape
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, int(n))
    P = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    fa = (P[:, 0] - a0) / max(a1 - a0, 1e-12) * (na - 1)
    fb = (P[:, 1] - b0) / max(b1 - b0, 1e-12) * (nb - 1)
    if (fa < -1e-9).any() or (fa > na - 1 + 1e-9).any() or \
       (fb < -1e-9).any() or (fb > nb - 1 + 1e-9).any():
        raise VizError(
            f"lineout {tuple(p0)} -> {tuple(p1)} leaves the slab's extent "
            f"{slab.extent}; refusing to extrapolate a field that was never "
            "solved there")
    ia = np.clip(fa.astype(int), 0, na - 2)
    ib = np.clip(fb.astype(int), 0, nb - 2)
    wa, wb = fa - ia, fb - ib
    v = ((1 - wa) * (1 - wb) * V[ib, ia] + wa * (1 - wb) * V[ib, ia + 1]
         + (1 - wa) * wb * V[ib + 1, ia] + wa * wb * V[ib + 1, ia + 1])
    # Arc length along the segment. np.hypot takes exactly two operands and
    # treats a third as its OUTPUT array, so hypot(*(p1 - p0)) on a 3-D point
    # raised "return arrays must be of ArrayType" -- a message that names
    # neither the caller's mistake nor this function's expectation. The
    # sampler itself only ever uses components 0 and 1 (it is a PLANE cut),
    # so the length must be measured in that plane too: a 3-D pair whose
    # out-of-plane components differ would otherwise get an arc length its
    # own samples do not correspond to.
    d = np.asarray(p1, float).ravel() - np.asarray(p0, float).ravel()
    if d.size < 2:
        raise VizError(
            f"lineout: endpoints must have at least 2 components, got "
            f"{d.size}; this is a cut through a plane, not along an axis")
    if d.size > 2 and abs(d[2]) > 1e-9:
        raise VizError(
            f"lineout: endpoints differ out of plane by {d[2]:.6g} mm. This "
            "samples a single solved plane, so a segment that leaves it "
            "would return values from the wrong depth. Cut within the plane, "
            "or take one lineout per plane.")
    s = t * float(np.hypot(d[0], d[1]))
    return Lineout(label=label or f"{tuple(p0)} -> {tuple(p1)}",
                   s=s, values=v, quantity=slab.quantity, marks=list(marks))


def mask_rects_mm(mask, h, origin=(0.0, 0.0)):
    """Exact rectangle decomposition of a boolean [ix,iy] mask -> world-mm
    rects (x0,x1,y0,y1).  Exact: the union of the rects IS the mask, so the
    picture cannot drift from the array the kernel flew through.

    origin = the world position of NODE (0, 0).  It is a PARAMETER because
    this repo has two grid registrations and they are half a cell apart:

      * the bench field is NODE-CENTRED.  It interpolates with gx = x/h, so
        node i sits at i*h and the domain runs 0 .. (nx-1)*h.  origin = (0, 0).
      * build_planar WAS cell-centred (rasterised on (arange+0.5)*h,
        origin = (h/2, h/2)) historically, until its sampling moved to
        the kernel frame — it is now node-centred like the bench, and the
        scene builder passes origin = (0, 0).

    This function used to hard-code the first one.  That was right for the
    bench and wrong for every planar SimSpec: the metal and the field were
    drawn CONSISTENTLY WITH EACH OTHER and half a cell off the coordinates the
    spec itself is written in -- 25 um at h = 0.05, 62 um at h = 0.125.  Both
    pictures looked fine.  Neither was registered to the geometry.  A rendered
    frame is part of the contract, so the origin is now passed, never assumed.
    Node-centred: a metal NODE owns the half-cell around it."""
    m = np.asarray(mask, bool).copy()
    ox, oy = float(origin[0]), float(origin[1])
    nx, ny = m.shape
    out = []
    for i in range(nx):
        j = 0
        while j < ny:
            if not m[i, j]:
                j += 1
                continue
            j1 = j
            while j1 + 1 < ny and m[i, j1 + 1]:
                j1 += 1
            i1 = i
            while i1 + 1 < nx and m[i1 + 1, j:j1 + 1].all():
                i1 += 1
            out.append((ox + (i - 0.5) * h, ox + (i1 + 0.5) * h,
                        oy + (j - 0.5) * h, oy + (j1 + 0.5) * h))
            m[i:i1 + 1, j:j1 + 1] = False
            j = j1 + 1
    return out


_MASK_OUTLINE_CACHE: "OrderedDict" = OrderedDict()
_MASK_OUTLINE_CACHE_MAX = 256   # bounded: ~loops per entry are tiny; 256
#                                 covers many models' electrodes at once


def mask_outline_mm(mask, h, origin=(0.0, 0.0)):
    """Closed boundary loop(s) of a boolean [ix,iy] mask -> world-mm (N,2)
    arrays, one per loop (a hole is its own loop).

    CACHED BY CONTENT (a real crash, plus the outlines should already
    be cached): the mask is static per model, but this pure
    function recomputed the contour on EVERY caller — including the
    250 ms flight tick, once per electrode. An index-space
    fix removed the meshgrid we called; a reported traceback
    proved that contourpy version synthesizes x/y and calls np.meshgrid
    INTERNALLY even on the z-only route (contourpy/__init__ line 190),
    where that conda CPython 3.12 died with SystemError in tuple
    construction — so the hot-loop exposure survived that fix. The key
    is (content digest, shape, h, origin): every geometry-affecting
    input, by construction, so an edited mask is a new entry, never a
    stale hit. Cached loops are returned as COPIES — callers may
    post-process them in place. LRU-bounded; eviction is oldest-first.

    Same node-centred registration as mask_rects_mm: node i sits at
    origin + i*h and owns the half-cell around it, so the 0.5-level
    contour of the 0/1 node field runs exactly halfway between a metal
    node and a vacuum node — the rects' own edge line, delivered as ONE
    path per boundary instead of one outlined patch per voxel row
    ("chunky rods"; a round rod decomposed into
    per-row rects reads as stacked slabs). Same solver truth: the loop is
    the mask's own level line, not a smoothed CAD import.
    """
    from contourpy import contour_generator
    mb = np.ascontiguousarray(np.asarray(mask, bool))
    key = (hashlib.blake2b(mb.tobytes(), digest_size=16).digest(),
           mb.shape, float(h),
           (float(origin[0]), float(origin[1])))
    hit = _MASK_OUTLINE_CACHE.get(key)
    if hit is not None:
        _MASK_OUTLINE_CACHE.move_to_end(key)
        return [lp.copy() for lp in hit]
    m = mb.astype(float)
    nx, ny = m.shape
    # VACUUM-PAD one node on every side (the "green wedge"
    # defect): a metal region flush against the domain edge used to
    # produce an OPEN 0.5-contour (it enters and exits at the border),
    # and fill() auto-closed it with a straight chord — drawing a
    # boundary-touching rect as a wedge. With a phantom vacuum border,
    # every region is strictly interior, every loop CLOSES, and the
    # boundary-side edge lands at origin - h/2 — exactly the half-cell
    # the flush node already owns under the stated registration, so the
    # drawn edge equals the rects' own edge line there too.
    mp = np.zeros((nx + 2, ny + 2))
    mp[1:-1, 1:-1] = m
    # CONTOUR IN INDEX SPACE, MAP TO mm OURSELVES. Passing 1-D
    # x/y to contour_generator makes contourpy call np.meshgrid internally,
    # which materialises TWO (ny+2, nx+2) float arrays per blob per redraw --
    # and on some macOS/conda builds that path died inside CPython
    # (`SystemError: bad argument to internal function`, tupleobject.c, via
    # meshgrid's tuple(x.copy() ...)) every animation tick on the drift-tube
    # deck. The grid is UNIFORM, so index coordinates carry the same
    # information: contour at node indices, then apply the affine map the
    # padding already defines (node i sits at origin + (i-1)*h). Verified
    # identical to the x/y route on three grids (interior rect, corner-flush
    # pair, 106-loop speckle): loop counts equal, max |diff| 0, 0, 4.4e-16.
    # Fewer allocations, no external meshgrid in the hot path, same
    # geometry.
    cg = contour_generator(z=mp.T)          # contourpy wants z[ny,nx]
    out = []
    for _l in cg.lines(0.5):
        if len(_l) < 3:
            continue
        q = np.asarray(_l, float).copy()
        q[:, 0] = float(origin[0]) + (q[:, 0] - 1.0) * h
        q[:, 1] = float(origin[1]) + (q[:, 1] - 1.0) * h
        out.append(q)
    _MASK_OUTLINE_CACHE[key] = tuple(lp.copy() for lp in out)
    while len(_MASK_OUTLINE_CACHE) > _MASK_OUTLINE_CACHE_MAX:
        _MASK_OUTLINE_CACHE.popitem(last=False)
    return out


def extrude_poly(poly, z0, z1):
    """A planar world polygon -> front/back/side faces of the declared slab.
    The slab is a DECLARED depth, not a solved one; the caller must say so."""
    p = np.asarray(poly, float).reshape(-1, 3)
    if z1 <= z0:
        return [p]
    front, back = p.copy(), p.copy()
    front[:, 2], back[:, 2] = z0, z1
    faces = [front, back]
    for k in range(len(p)):
        a, b = p[k], p[(k + 1) % len(p)]
        faces.append(np.array([[a[0], a[1], z0], [b[0], b[1], z0],
                               [b[0], b[1], z1], [a[0], a[1], z1]]))
    return faces


def scene_from_model3d(spec, model, *, field: str | None = "E_peak",
                       title: str | None = None, trajs=None, fates=None,
                       slice_at: dict | None = None,
                       metal_mode: str = "cut") -> Scene:
    """Adapter: a solved 3-D shapes-route model -> static multi-axis Scene.

    Closes the long-standing gap that the static renderer only understood
    planar/r-z models while every 3-D build (SLIM boards, quads) could only
    be seen in the app's interactive plotly view -- so 3-D notebooks had no
    geometry panel.

    AGNOSTIC and solver-true by construction: the metal comes from the
    SOLVER'S OWN 3-D electrode mask (fly_fields['ele']) sliced (metal_mode
    'cut', default -- a projection of a channel device fills the aperture
    solid) or max-projected ('project'); the optional field panel is |E| at
    RF PEAK PHASE (all drive amplitudes at their peaks, labeled as such --
    a pure-RF device has ~zero static field and would render blank
    otherwise); trajectories are drawn exactly as flown. Cut positions
    default to the domain centre per axis; override with slice_at
    ({'x'|'y'|'z': mm, canonical frame}) -- e.g. a SLIM electrode pattern
    lives at y = board face, not at mid-gap.

    DISPLAY == SOLVER INPUT: values are read from the arrays the kernel
    flies, shifted into the WORLD frame by the pack's composed
    world_off_mm (mirror + declared origin; pure-mirror fallback for
    packs that predate signed origins) — the same frame trajectories
    are recorded in, so the metal and the overlaid paths agree.
    """
    ff = model.fly_fields if hasattr(model, "fly_fields") else model
    ele = np.asarray(ff["ele"])
    if ele.ndim != 3:
        raise VizError("scene_from_model3d needs a 3-D solve; for planar/"
                       "r-z models use scene_from_simspec")
    h = float(ff["h_mm"])
    # WORLD frame (frame certification): the pack's
    # world_off_mm (mirror + declared origin composed, published by
    # build_stl3d since the origin_mm fix) with the pure-mirror fallback
    # for packs that predate it — the world_off convention. This site
    # read mirror_off_mm only and was classified deliberate in the
    # frame audit ("DISPLAY == SOLVER INPUT"); MEASURED on the
    # signed hexapole, that put the metal slab at stored [0, 15.6] mm
    # against canonical trajs about 0 — a 7.80 mm frame split inside
    # one figure, which is the opposite of one world frame. Unsigned
    # and mirrored-unsigned decks (every pre-existing caller) are
    # byte-identical: world_off_mm == mirror_off_mm when origin is 0.
    moff = np.asarray(ff.get("world_off_mm",
                             ff.get("mirror_off_mm", (0.0, 0.0, 0.0))),
                      float)
    n = ele.shape
    lo = moff.copy()
    hi = moff + (np.array(n) - 1) * h
    ax_of = {"x": 0, "y": 1, "z": 2}
    # slab axes FROM the renderer's own Convention A mapping (_AXES), so the
    # adapter cannot disagree with project(): first draft hardcoded xz =
    # (x, z) while Convention A draws xz as (z horizontal, x vertical) --
    # the metal painted its x-extent along the z axis (caught on the first
    # winner panel).
    plane_axes = {pl: (a, b, 3 - a - b) for pl, (a, b) in _AXES.items()}
    cuts = {}
    for axn, ai in ax_of.items():
        c_mm = None if slice_at is None else slice_at.get(axn)
        if c_mm is None:
            c_mm = 0.5 * (lo[ai] + hi[ai])
        idx = int(round((float(c_mm) - moff[ai]) / h))
        if not (0 <= idx < n[ai]):
            raise VizError(f"slice_at[{axn!r}] = {c_mm} mm is outside the "
                           f"solved domain [{lo[ai]:.2f}, {hi[ai]:.2f}]")
        cuts[ai] = (idx, moff[ai] + idx * h)
    # peak-phase drive amplitudes for the |E| panel
    ExK = np.asarray(ff.get("ExK", np.zeros((0,) + n)))
    EyK = np.asarray(ff.get("EyK", np.zeros((0,) + n)))
    EzK = np.asarray(ff.get("EzK", np.zeros((0,) + n)))
    # SIGNED peak-phase amplitudes (frame certification):
    # each channel's sin drive evaluated at the reference
    # RF peak instant wt = pi/2 is amp*sin(pi/2 + ph) = amp*cos(ph) —
    # for the supported {0, 180} phase split that is +amp / -amp, the
    # same signed composition the model's own field_surface/pe_surface
    # peak uses. The previous abs(ch_amp) put BOTH phases at +amp: on
    # the signed hexapole (ch_amp [100, 100], ch_ph [0, pi]) the
    # opposite-phase interior field canceled and the |E| panel rendered
    # ~1e-5 V/mm numerical residue LABELED as the RF peak — displayed
    # != computed. Single-phase decks (every earlier caller) have
    # cos(0) = 1 and are byte-identical.
    _amp = np.asarray(ff.get("ch_amp", np.zeros(ExK.shape[0])), float)
    _ph = np.asarray(ff.get("ch_ph", np.zeros(_amp.shape)), float)
    amps = _amp * np.cos(_ph)
    metal, fields = [], []
    for pl, (a_ax, b_ax, c_ax) in plane_axes.items():
        idx, at_mm = cuts[c_ax]
        sl = [slice(None)] * 3
        if metal_mode == "cut":
            sl[c_ax] = idx
            m2 = ele[tuple(sl)]
        elif metal_mode == "project":
            m2 = ele.max(axis=c_ax)
        else:
            raise VizError(f"metal_mode {metal_mode!r}: 'cut' or 'project'")
        # values indexed [b, a] per FieldSlab contract. Slicing away
        # c_ax leaves the surviving axes in ASCENDING index order
        # (min(a_ax, b_ax), max(...)), so the [b, a] layout needs a
        # transpose exactly when a_ax < b_ax — true only for xy under
        # Convention A's _AXES (xy=(0,1); xz=(2,0), yz=(2,1) draw the
        # axial coordinate first). The previous blanket .T was right
        # for xy and TRANSPOSED xz/yz — invisible because imshow
        # stretches any array to the stated extent, measured on the
        # signed hexapole deck: the x=0 cut
        # drew rods 2/5 as z-slabs instead of y-bands, and the |E|
        # panel striped along z where the rod-uniform field is
        # z-invariant.
        def _ba(img2):
            return img2.T if a_ax < b_ax else img2
        metal.append(FieldSlab(plane=pl,
                               extent=(lo[a_ax], hi[a_ax], lo[b_ax], hi[b_ax]),
                               values=_ba(np.asarray(m2, float)),
                               quantity="metal", at=at_mm))
        if field == "E_peak":
            slc = [slice(None)] * 3
            slc[c_ax] = idx
            ex = np.array(ff["EAx"][tuple(slc)])
            ey = np.array(ff["EAy"][tuple(slc)])
            ez = np.array(ff["EAz"][tuple(slc)])
            for k in range(ExK.shape[0]):
                ex += amps[k] * ExK[k][tuple(slc)]
                ey += amps[k] * EyK[k][tuple(slc)]
                ez += amps[k] * EzK[k][tuple(slc)]
            emag = np.sqrt(ex * ex + ey * ey + ez * ez)
            fields.append(FieldSlab(plane=pl,
                                    extent=(lo[a_ax], hi[a_ax],
                                            lo[b_ax], hi[b_ax]),
                                    values=_ba(emag),
                                    quantity="|E| at RF peak phase (V/mm)",
                                    at=at_mm))
        elif field is not None:
            raise VizError(f"field {field!r}: 'E_peak' or None on the 3-D "
                           "adapter (no composed static phi is stored)")
    sc = Scene(title=title or getattr(spec, "name", "3-D model"),
               metal=metal, fields=fields,
               bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]),
               provenance=SOLVER)
    sc.notes.append(f"3-D shapes-route solve, {n[0]}x{n[1]}x{n[2]} gu at "
                    f"{h:g} mm/gu; metal {metal_mode.upper()} slices at "
                    + ", ".join(f"{k}={cuts[ax_of[k]][1]:.2f} mm"
                                for k in ("x", "y", "z")))
    if trajs is not None:
        paths = []
        for tr in trajs:
            a = np.asarray(tr, float)
            if a.ndim == 2 and a.shape[1] >= 4 and len(a):
                paths.append(np.c_[a[:, 1], a[:, 2], a[:, 3]])
        if paths:
            sc.rays = [Rays(paths=paths, fates=list(fates or []))]
            sc.notes.append(f"{len(paths)} ion path(s) overlaid, as flown")
    return sc


def scene_from_bench(bench, trajs=None, fates=None, *, field: str | None = "E",
                     nx: int = 420, title: str | None = None,
                     slab_mm: float = 0.0, provenance: str = SOLVER) -> Scene:
    """Adapter: a multi-FA Bench -> Scene.  AGNOSTIC: every body is the
    placed instance's OWN electrode MASK, decomposed exactly and pushed
    through its OWN rigid transform; the field panel is sampled through the
    bench kernel's own field(), so the picture is field-free exactly where
    the flight is."""
    z0, z1 = (-0.5 * slab_mm, 0.5 * slab_mm) if slab_mm > 0 else (0.0, 0.0)
    bodies = []
    for pa in bench.pas:
        polys = []
        # bench_pa is NODE-CENTRED (it interpolates gx = x/h), so node (0,0)
        # is at the field array's local origin.  Stated, not defaulted.
        for (x0, x1, y0, y1) in mask_rects_mm(np.asarray(pa.ele) > 0, pa.h,
                                              (0.0, 0.0)):
            c = np.array([[x0, y0, 0.], [x1, y0, 0.],
                          [x1, y1, 0.], [x0, y1, 0.]])
            polys += extrude_poly(pa.rigid.to_world(c), z0, z1)
        if polys:
            bodies.append(Body(name=f"{pa.name}:metal", polys=polys,
                               kind="electrode", owner=pa.name))
    rays = []
    if trajs is not None:
        paths = [np.c_[t[:, 1], t[:, 2], np.zeros(len(t))]
                 for t in trajs if len(t)]
        rays = [Rays(paths=paths, fates=list(fates or []))]

    bx0, bx1, by0, by1 = bench.bounds()
    sc = Scene(title=title or f"bench: {bench.spec.name}",
               bodies=bodies, rays=rays, slab_mm=slab_mm,
               bounds=(bx0, bx1, by0, by1, z0, z1),
               provenance=provenance,
               notes=[f"{len(bench.pas)} instance(s); field-free outside "
                      "every instance box"])
    if slab_mm > 0:
        sc.notes.append(
            f"slab depth {slab_mm:g} mm DECLARED, not solved: the width-plane "
            "field is extruded for the xz/yz panels and every ion flies in "
            "the z = 0 midplane")
    if field:
        sc.fields.append(_bench_field_slab(bench, field, nx))
    for pa in bench.pas:
        sc.notes.append(
            f"{pa.name}: origin "
            f"{tuple(round(float(v), 3) for v in pa.rigid.origin_mm)} mm, "
            f"rot {pa.rigid.rotation_deg:g} deg, priority {pa.priority}, "
            f"vscale {pa.voltage_scale:g}")
    return sc


def _bench_field_slab(bench, field, nx) -> FieldSlab:
    x0, x1, y0, y1 = bench.bounds()
    ny = max(int(round(nx * (y1 - y0) / max(x1 - x0, 1e-12))), 2)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    if field in ("E", "|E|"):
        E = bench.field(P)
        vals = np.hypot(E[:, 0], E[:, 1]).reshape(X.shape) * 1e-3
        q = "|E| (V/mm)"
    elif field == "phi":
        # priority order, exactly as the kernel resolves ownership; outside
        # every instance the potential is undefined -> 0 (field-free).
        vals = np.zeros(X.size)
        todo = np.ones(X.size, bool)
        for pa in bench.pas:                      # already priority-sorted
            m = todo & pa.contains_world(P)
            if m.any():
                vals[m] = pa.potential_world(P[m])
                todo &= ~m
        vals = vals.reshape(X.shape)
        q = "phi (V)"
    else:
        raise VizError(f"unknown field {field!r}; use 'phi' or 'E'")
    return FieldSlab("xy", (x0, x1, y0, y1), vals, q, at=0.0)


def scene_from_bodies(title, bodies, rays=None, *, provenance=SCHEMATIC,
                      slab_mm=0.0, bounds=None, notes=(), dims=()) -> Scene:
    """Escape hatch for anything with polygons (STL, CadQuery, a sketch).
    Defaults to SCHEMATIC: if you did not solve it, it is not evidence.

    dims: dimension callouts.  A geometry review IS its callouts; a Dim in a
    view the scene does not draw is REFUSED rather than silently dropped -- a
    missing dimension on a dimensional review is the failure mode."""
    sc = Scene(title=title, bodies=list(bodies), rays=list(rays or []),
               slab_mm=slab_mm, bounds=bounds, provenance=provenance,
               notes=list(notes), dims=list(dims))
    for d in sc.dims:
        if d.view not in _AXES:
            raise VizError(f"Dim {d.label!r} names view {d.view!r}; "
                           f"known: {VIEWS}")
    return sc


# ---------------------------------------------------------- static renderer
# ------------------------------------------------------------------ style
# USER-DEFINED default colors.
# Mutate via set_style(); every renderer reads it, so a house style
# is set once and nothing has to be re-themed downstream.
STYLE = {
    "metal":     "#c8c8c8",   # an electrode we can say NOTHING about
    "cmap":      "coolwarm",  # DC voltage scale (diverging: sign is visible)
    "field":     "viridis",
    "groups":    ["#4c72b0", "#dd8452", "#55a868", "#c44e52",
                  "#8172b3", "#937860", "#da8bc3", "#8c8c8c"],
    "schematic": "#c8c8c8",
    "hatch":     "///",
    "dim":       "#c0392b",   # a STATED number
    "dim_derived": "#12806a",  # a number we INFERRED -- a different claim
}


def set_style(**kw):
    """Override any STYLE key.  Unknown keys are REFUSED, not absorbed: a
    silently ignored theme key is a user who thinks they set a colour and did
    not."""
    bad = set(kw) - set(STYLE)
    if bad:
        raise VizError(f"unknown style key(s) {sorted(bad)}; "
                       f"known: {sorted(STYLE)}")
    STYLE.update(kw)
    return dict(STYLE)


def _body_colors(bodies):
    """(facecolor, edgecolor) per body.  FILL = DC.  EDGE = RF/drive group.

    Both, always -- never one OR the other.  The first version colour-coded by
    DC "else" by group, and that `else` was itself the bug: on SLIM, DC DOES
    vary (Guard is biased, RF1/RF2/TW are not), so the DC branch wins and the
    two RF PHASES -- the whole point of the device -- come out identical.  "DC
    varies" does not mean "DC is sufficient".

    An RF instrument's electrodes are distinguished by potential AND by drive
    group, and a picture that can only show one of them will always be blind to
    some device.  So the two channels are orthogonal: fill carries potential,
    outline carries phase group, and neither can hide the other.

    Agnostic: the scale and the palette come from the bodies PRESENT, never
    from a table of expected voltages or group names.  Fully grey means the
    electrodes really are indistinguishable in both channels.
    """
    import matplotlib.pyplot as plt
    vs = [b.volt for b in bodies if b.volt is not None]
    spread = (max(vs) - min(vs)) if vs else 0.0
    if spread > 1e-12:
        lim = max(abs(min(vs)), abs(max(vs)), 1e-12)
        cm = plt.get_cmap(STYLE["cmap"])
        face = {i: (STYLE["metal"] if b.volt is None
                    else cm(0.5 + 0.5 * float(b.volt) / lim))
                for i, b in enumerate(bodies)}
    else:
        face = {i: STYLE["metal"] for i in range(len(bodies))}

    grps = [getattr(b, "group", None) for b in bodies]
    uniq = sorted({g for g in grps if g})
    pal = STYLE["groups"]
    idx = {g: pal[k % len(pal)] for k, g in enumerate(uniq)}
    edge = {i: idx.get(g, "#333333") for i, g in enumerate(grps)}
    return face, edge


def _color_basis(bodies) -> str:
    """What the picture is actually telling you -- stated on the figure, so a
    reader never has to guess which channel carries the information."""
    vs = [b.volt for b in bodies if b.volt is not None]
    spread = (max(vs) - min(vs)) if vs else 0.0
    uniq = sorted({getattr(b, "group", None) for b in bodies
                   if getattr(b, "group", None)})
    bits = []
    if spread > 1e-12:
        bits.append(f"fill = DC ({min(vs):g} to {max(vs):g} V)")
    if len(uniq) > 1:
        bits.append(f"outline = drive group ({', '.join(uniq)})")
    return "  |  ".join(bits) if bits else "electrodes are indistinguishable"


def _mpl_release(fig):
    """Deregister a finished figure from pyplot's manager so notebook
    inline backends do not auto-display it a second time at end-of-cell
    (the returned object's rich repr is the single display; savefig on
    the returned figure still works). Headless: harmless no-op effect.
    A real double-display defect."""
    if fig is None or not hasattr(fig, "number"):
        return          # not a pyplot-managed figure (e.g. Plotly) —
                        # nothing to deregister
    import matplotlib.pyplot as plt
    if plt.fignum_exists(fig.number):
        plt.close(fig)


def _mpl_headless_if_needed():
    """Select the Agg backend ONLY when no notebook/GUI backend is
    active. Inside Jupyter/VS Code the inline (or widget/nbagg) backend
    owns figure display — forcing Agg there silently downgrades every
    figure to a bare text repr. Headless scripts
    and the sandbox still get Agg."""
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if any(k in backend for k in ("inline", "ipympl", "widget", "nbagg")):
        return
    matplotlib.use("Agg")


# One authority for figure resolution across every framework renderer
# (default 300). Applies at FIGURE CREATION
# and at save, so displayed == saved at the same sharpness. Callers may
# still pass dpi= explicitly; changing the project default is THIS one
# edit, never a grep.
DEFAULT_DPI = 300


def save_figure(fig, path, dpi=DEFAULT_DPI):
    """Write a figure and palette-optimize it. Line art at 150 dpi is 2-4x
    larger as RGBA than as an 8-bit palette PNG with no visible difference
    (no resampling, no dpi change). Doing it HERE means a re-render cannot
    silently undo it -- a one-off optimization pass did exactly that
    """
    from pathlib import Path as _Path
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi)
    try:
        from PIL import Image as _Image
    except ImportError:
        return p                      # Pillow absent: the figure is written
    if p.suffix.lower() != ".png":
        return p
    im = _Image.open(p)
    if im.mode in ("RGBA", "RGB"):
        before = p.stat().st_size
        q = im.convert("RGB").quantize(colors=256, method=_Image.MEDIANCUT)
        q.save(p, optimize=True)
        if p.stat().st_size > before:  # no gain: keep the original encoding
            im.save(p, optimize=True)
    return p


def wrap_title(text, fig=None, fontsize=10):
    """Wrap a figure title to the figure's own width. Titles carry the
    device and its operating point by doctrine, so they are long; every
    suptitle site used to place them as one line and matplotlib simply
    clipped what did not fit."""
    import textwrap as _tw
    w_in = fig.get_size_inches()[0] if fig is not None else 7.0
    width = max(40, int(w_in * 72.0 / (fontsize * 0.55)))
    return "\n".join(_tw.wrap(str(text), width=width,
                              break_long_words=False))


def render_mpl(scene: Scene, views=None, *, field: bool = True,
               path: str | None = None, dpi: int = DEFAULT_DPI, cmap: str | None = None,
               legend: bool = True, layout: str = "row",
               station_label_max: int = 8, y_exag: float = 1.0):
    """Multi-axis static figure.  One panel per view; each panel is drawn
    true-scale (equal aspect) because a static report image has no zoom to
    constrain, unless y_exag magnifies the transverse axis -- in which case
    the factor is stamped on the panel so it can never be read as true
    scale.  If the scene carries line-outs they get a row of their own
    beneath the views.  Returns the matplotlib Figure.

    layout : "row" | "column"
        "row" (default) places panels side by side.  Panel WIDTHS then track
        each panel's own data aspect, capped, so a long thin instrument does
        not squeeze a transverse view into a stripe -- but the consequence is
        that panels do NOT share a mm-per-inch scale, so the same 3 mm gap
        looks one size in xy and another in xz (the axes
        are not to scale ... makes it hard to interpret").
        "column" stacks panels top to bottom on a COMMON mm-per-inch scale:
        every panel gets the full figure width, heights are solved from each
        panel's true aspect, and the axial coordinate stays horizontal.  Use
        it whenever sizes must be compared BETWEEN views.
    station_label_max : int
        Label stations individually only while the deck declares at most
        this many.  A scan ladder (a focus located by 41 non-destructive
        planes rather than assumed) otherwise writes 41 overlapping name
        boxes across the instrument and hides the thing being measured.
        Above the threshold every station is still DRAWN -- nothing is
        hidden -- and the panel note names the count, the axis and the
        span, so the figure still says exactly what is on it.  0 suppresses
        labels always; a large value restores per-station labels.
    """
    if station_label_max < 0:
        raise ValueError(
            f"render_mpl: station_label_max must be >= 0, got "
            f"{station_label_max!r} -- a negative threshold has no reading "
            f"(0 already means 'never label').")
    if layout not in ("row", "column"):
        raise ValueError(f"render_mpl: layout must be 'row' or 'column', "
                         f"got {layout!r} -- an unrecognised layout would "
                         f"silently fall back to a different figure than the "
                         f"caller asked for")
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cmap = cmap or STYLE["field"]
    vs = views_for(scene, views)
    # height reserved at the bottom for the provenance notes (they are drawn
    # from y=0.005 upward), used to place the legend band above them.
    _notes_h = 0.012 * min(len(scene.notes), 6) + 0.020
    _legend_h = 0.0
    # each panel is drawn true-scale, so give each one a WIDTH that matches
    # its own data aspect -- otherwise a wide view and a tall view forced
    # into equal boxes leave one of them a stripe in a sea of white.
    ars = []
    for v in vs:
        a0, a1, b0, b1 = view_extent(scene, v)
        # DRAWN aspect, not data aspect: with y_exag the panel is wider than
        # it is tall by (dx/dy)/y_exag, and sizing the box from the data
        # aspect instead leaves the drawing shrunk inside an oversized box.
        ars.append(max(a1 - a0, 1e-9)
                   / (max(b1 - b0, 1e-9) * float(y_exag)))
    # Cap the RATIO hard.  Panel width proportional to aspect means a common
    # mm-per-inch scale ACROSS panels -- which on a 635x16 mm instrument gives
    # the transverse view 1/40th of the page and makes the densest panel the
    # unreadable one.  True scale WITHIN a panel is what you read a dimension
    # off; a common scale BETWEEN panels is not required and costs everything.
    ws = [min(max(a, 0.5), 3.0) for a in ars]
    # ...and BUDGET the page.  The aspect cap bounds the RATIO between panels
    # and does nothing about the total: three axial panels of a long thin
    # instrument have aspect ~10 each, and a fixed 4.4-in panel height times
    # that sum produced a 62-inch, 9360-px figure -- a picture nothing will
    # open is not a picture.  So the width is FIXED and the height is solved
    # for, not the other way round.  This is the same failure as the sliver,
    # one convention later: R3 satisfied on paper, defeated in fact.
    n = len(vs)
    nlo = len(scene.lineouts)
    if layout == "column":
        # COMMON SCALE: one width for every panel, heights solved from each
        # panel's TRUE aspect (uncapped -- the cap exists to stop a stripe
        # in a row, and stacking has no stripe failure). A 3 mm gap is then
        # the same number of inches in every view, which is the whole point.
        fig_w = 9.0
        panel_w = fig_w - 2.4                      # axes width after labels/cbar
        hs = [max(panel_w / max(a, 1e-9), 0.9) for a in ars]
        total_h = sum(hs)
        if total_h > 22.0:                          # page budget, same rule as row
            _k = 22.0 / total_h
            hs = [h * _k for h in hs]
            total_h = 22.0
        # The furniture (title band, provenance notes) is fixed INCHES, not a
        # fraction: adding it to the figure height while letting the gridspec
        # span the whole figure made every axes box taller than its data
        # aspect, so equal-aspect drew the panel small with white bands on
        # both sides. Reserve the furniture explicitly via top/bottom.
        _top_in, _bot_in = 1.0, 0.6 + (0.012 * min(len(scene.notes), 6) * 6)
        fig_h = _top_in + _bot_in + total_h + (3.2 if nlo else 0.0)
        # dpi applies at FIGURE CREATION, not only at savefig: an inline
        # notebook display renders the Figure object itself, so building it
        # at rcParams' 100-dpi default made every displayed panel blurry
        # while the saved PNG (which re-rendered at `dpi`) was sharp.
        # Displayed must equal saved.
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = fig.add_gridspec(n + (1 if nlo else 0), 1,
                              height_ratios=hs + ([3.2] if nlo else []),
                              top=1.0 - _top_in / fig_h,
                              bottom=_bot_in / fig_h,
                              left=0.10, right=0.88,
                              hspace=COLUMN_PANEL_HSPACE)
        axes = [fig.add_subplot(gs[k, 0]) for k in range(n)]
    else:
        fig_w = 6.0 + 2.4 * n
        panel_h = max(1.5, min(4.4, (fig_w - 1.9 - 1.1 * n) / max(sum(ws), 1e-9)))
        fig = plt.figure(figsize=(fig_w, 1.6 + panel_h + (3.2 if nlo else 0.0)),
                         dpi=dpi)   # same displayed==saved rule as column
        gs = fig.add_gridspec(2 if nlo else 1, n,
                              width_ratios=ws,
                              height_ratios=[panel_h, 3.2] if nlo else [1.0])
        axes = [fig.add_subplot(gs[0, k]) for k in range(n)]
    fmap = {f.plane: f for f in scene.fields}
    cols, edges = _body_colors(scene.bodies)

    for ax, v in zip(axes, vs):
        f = fmap.get(v) if field else None
        if f is not None:
            # FieldSlab.extent is the NODE SPAN (first..last node centre) —
            # the convention its docstring declares and both other
            # consumers use (the plotly heatmap's linspace(a0,a1,n) centres
            # and the lineout sampler's (p-a0)/(a1-a0)*(n-1)). imshow's
            # `extent` is the OUTER PIXEL EDGES, so passing the node span
            # straight through drew the field half a cell off the bodies.
            # Convert: expand by half a cell on each side.
            # CONTOURS, not a filled gradient. A flood-filled field map
            # dominates the picture and hides the geometry and the ion
            # paths drawn over it; equipotentials carry the same
            # information as lines you can see through, and their spacing
            # reads as field strength directly. FieldSlab.extent is the
            # NODE SPAN, which is exactly what the contour grid wants.
            _a0, _a1, _b0, _b1 = f.extent
            _nb, _na = np.shape(f.values)
            _xs = np.linspace(_a0, _a1, _na)
            _ys = np.linspace(_b0, _b1, _nb)
            # Levels strictly INSIDE (vmin, vmax): auto levels land ON
            # the electrode potentials (the field extremes), and a level
            # AT an electrode's potential has nothing to trace but the
            # metal's constant-phi region and its raster staircase — it
            # degenerates into noise squiggles along every surface
            # (measured on a tube-stack deck).
            _vmin = float(np.nanmin(f.values))
            _vmax = float(np.nanmax(f.values))
            if _vmax > _vmin:
                _lv = np.linspace(_vmin, _vmax, 18)[1:-1]
            else:
                # a constant field has no equipotential structure to
                # draw; one interior level would be pure noise. Refuse
                # the contours, keep the panel (metal + paths still
                # carry the scene).
                _lv = []
            cset = (ax.contour(_xs, _ys, f.values, levels=_lv, cmap=cmap,
                               linewidths=0.7, alpha=0.9, zorder=1)
                    if len(_lv) else None)
            # colorbar sized to the DRAWN axes (same standing
            # rule as candidate_field_figure): with aspect-equal panes a
            # slot-fraction bar towers over the plot it annotates.
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            if cset is not None:
                _cax = make_axes_locatable(ax).append_axes(
                    "right", size="3.5%", pad=0.08)
                cb = fig.colorbar(cset, cax=_cax)
                cb.set_label(f.quantity, fontsize=8)
                cb.ax.tick_params(labelsize=7)
            else:
                # constant field: say so where the colorbar would be,
                # rather than a silent blank.
                ax.text(0.99, 0.01,
                        f"{f.quantity}: constant ({_vmin:g})",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7, color="0.4")
        no_field = bool(scene.fields) and field and f is None
        for i, b in enumerate(scene.bodies):
            # a SCHEMATIC body inside a solved scene is HATCHED, so the eye
            # cannot mistake the drawn part for the solved part.
            sk = b.stamp_in(scene) == SCHEMATIC
            # ONE PolyCollection per body per view: every
            # poly of a body carries the SAME style, and one Polygon
            # artist per poly made artist count O(n_polys) — a revolved
            # r-z body is tens of thousands of quads, and render_mpl
            # stalled minutes-long on the orbitrap's 3 views. A
            # collection draws the identical fills/edges/alpha/hatch in
            # one artist. Style semantics unchanged, per-poly.
            _qs = [project(p, v) for p in b.polys]
            if getattr(b, "outline", False):
                # OUTLINE body: boundary path only — the electrode
                # reads as a line drawing over the field.
                ax.add_collection(PolyCollection(
                    _qs, closed=True, facecolors="none",
                    edgecolors=edges[i], linewidths=1.4, zorder=3))
            else:
                # SEMI-TRANSPARENT: the field contours and the ion paths
                # behind an electrode stay readable, and overlapping bodies
                # in a projection remain distinguishable.
                ax.add_collection(PolyCollection(
                    _qs, closed=True, facecolors=cols[i],
                    edgecolors=edges[i], linewidths=1.0, alpha=0.30,
                    hatch="///" if sk else None, zorder=3))
        # METAL as a cut through this plane (semi-transparent), when the
        # scene carries one. Drawn under the ion paths, over the contours.
        _m = {mm.plane: mm for mm in getattr(scene, "metal", [])}.get(v)
        if _m is not None:
            _ma0, _ma1, _mb0, _mb1 = _m.extent
            _mv = np.asarray(_m.values, float)
            _mx = np.linspace(_ma0, _ma1, _mv.shape[1])
            _my = np.linspace(_mb0, _mb1, _mv.shape[0])
            ax.contourf(_mx, _my, (_mv > 0).astype(float),
                        levels=[0.5, 1.5], colors=["#4c72b0"], alpha=0.30,
                        zorder=3)
            ax.contour(_mx, _my, (_mv > 0).astype(float), levels=[0.5],
                       colors=["#26456e"], linewidths=0.8, zorder=3.1)
        for r in scene.rays:
            for k, p in enumerate(r.paths):
                q = project(p, v)
                ax.plot(q[:, 0], q[:, 1], lw=0.7, alpha=0.85, zorder=4,
                        color=plt.cm.plasma((k % 8) / 8.0))
        a0, a1, b0, b1 = view_extent(scene, v)
        ax.set_xlim(a0, a1)
        ax.set_ylim(b0, b1)
        # y_exag > 1 MAGNIFIES the transverse axis, the convention the
        # mirror-field literature uses (Verenchikov JASMS 2026 Fig. 2A/2C
        # both state "vertical scale magnified"). A mirror is hundreds of mm
        # long and a few mm tall on axis, so true scale renders the ion
        # channel as a hairline and the equipotentials as a smear -- the
        # panel is technically correct and shows nothing. The magnification
        # is STAMPED on the panel below, never silent, because a reader
        # measuring an angle or a gap off a distorted panel would be wrong.
        ax.set_aspect("equal" if y_exag == 1.0 else float(y_exag))
        if y_exag != 1.0:
            ax.text(0.995, 0.02, f"vertical scale x{y_exag:g}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="#b03030",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5),
                    zorder=9)
        # dimension callouts.  A DERIVED number is drawn in a different colour
        # and tagged, because "the source states 2.5" and "I inferred 1.152
        # from two things the source states" are different claims.
        for d in scene.dims:
            if d.view != v:
                continue
            col = "#00695c" if d.derived else "crimson"
            ax.annotate("", xy=tuple(d.p1), xytext=tuple(d.p0),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=0.9),
                        zorder=5)
            # offset the label PERPENDICULAR to its own arrow.  Centring it on
            # the midpoint puts the number on top of the line it measures and,
            # on a crowded transverse section, on top of the next number too.
            dx = float(d.p1[0]) - float(d.p0[0])
            dy = float(d.p1[1]) - float(d.p0[1])
            L = float(np.hypot(dx, dy)) or 1.0
            span = max(a1 - a0, b1 - b0)
            off = 0.018 * span
            mx = 0.5 * (d.p0[0] + d.p1[0]) - off * dy / L
            my = 0.5 * (d.p0[1] + d.p1[1]) + off * dx / L
            ax.text(mx, my, d.label + ("  (derived)" if d.derived else ""),
                    color=col, fontsize=6.5, ha="center", va="center", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.8))
        # DECLARED KILL BOUNDS: dashed, in the interactive view's colour
        # (#9467bd, sim_app's convention) so both surfaces speak one visual
        # language. A bound on the view's horizontal axis draws as a
        # vertical line and vice versa; a bound NORMAL to the view plane is
        # the whole panel and cannot be drawn as a line, so it is skipped
        # in that view (it still draws in the views that contain its axis).
        # Axis letters from the _AXES authority — the view-name CHARS are
        # not in (horizontal, vertical) order for xz/yz (Convention A puts
        # the axial coordinate horizontal). Keying on v[0]/v[1] drew the
        # z-bound as an out-of-range horizontal line in the axial views —
        # the same array-vs-view order trap as the slab builder.
        _bha, _bva = _LABEL[_AXES[v][0]], _LABEL[_AXES[v][1]]
        for _bax, _bmm, _blab in scene.bound_marks:
            if _bax == _bha:
                ax.axvline(_bmm, color="#9467bd", lw=1.0, ls="--", zorder=4)
                ax.text(_bmm, b1 - 0.04 * (b1 - b0), _blab, color="#9467bd",
                        fontsize=6.0, ha="right", va="top", rotation=90,
                        zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.72,
                                  pad=0.6))
            elif _bax == _bva:
                ax.axhline(_bmm, color="#9467bd", lw=1.0, ls="--", zorder=4)
                ax.text(a0 + 0.04 * (a1 - a0), _bmm, _blab, color="#9467bd",
                        fontsize=6.0, ha="left", va="bottom", zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.72,
                                  pad=0.6))
        # DECLARED STATIONS: a station on the view's horizontal axis is a
        # vertical SEGMENT at pos_mm spanning its window on the vertical
        # axis (and vice versa) — solid and kind-coloured, so the
        # detector reads as hardware with a finite face, not another
        # infinite bound. No window on the in-panel axis -> full-extent
        # dash-dot (honest: the deck declared no aperture in this plane).
        # A station NORMAL to the panel is skipped here, like bounds.
        # Labels are suppressed WHOLESALE above the threshold, never
        # thinned: labelling every 5th plane would leave a reader to guess
        # which line each name belongs to, which is worse than a note that
        # states the whole ladder.
        _st_label = len(scene.station_marks) <= station_label_max
        for _st in scene.station_marks:
            _sc = _STATION_COLORS.get(_st["kind"], "#2ca02c")
            _sl = f"{_st['name']} ({_st['kind']} station)"
            if not _st_label:
                _sl = None
            if _st["axis"] == _bha:
                _w = _st["window"].get(_bva)
                if _w:
                    ax.plot([_st["pos_mm"]] * 2, list(_w), color=_sc,
                            lw=2.4, solid_capstyle="butt", zorder=5)
                else:
                    ax.axvline(_st["pos_mm"], color=_sc, lw=1.4,
                               ls="-.", zorder=5)
                if _sl:
                    ax.text(_st["pos_mm"], (_w[1] if _w
                                            else b1 - 0.08 * (b1 - b0)),
                            _sl, color=_sc, fontsize=6.0, ha="center",
                            va="bottom", zorder=6,
                            bbox=dict(fc="white", ec="none", alpha=0.72,
                                      pad=0.6))
            elif _st["axis"] == _bva:
                _w = _st["window"].get(_bha)
                if _w:
                    ax.plot(list(_w), [_st["pos_mm"]] * 2, color=_sc,
                            lw=2.4, solid_capstyle="butt", zorder=5)
                else:
                    ax.axhline(_st["pos_mm"], color=_sc, lw=1.4,
                               ls="-.", zorder=5)
                if _sl:
                    ax.text((_w[0] if _w else a0 + 0.08 * (a1 - a0)),
                            _st["pos_mm"], _sl, color=_sc, fontsize=6.0,
                            ha="left", va="bottom", zorder=6,
                            bbox=dict(fc="white", ec="none", alpha=0.72,
                                      pad=0.6))
            elif (_bha in _st["window"] and _bva in _st["window"]):
                # station NORMAL to this panel: the head-on view of its
                # face. With a window on BOTH panel axes the face is a
                # drawable rectangle (the detector as the ions see it);
                # drawn solid-outlined at the plane's colour with the
                # plane coordinate stated. A windowless normal station
                # is the whole panel and stays skipped, like bounds.
                _wa, _wb = _st["window"][_bha], _st["window"][_bva]
                from matplotlib.patches import Rectangle as _MplRect
                ax.add_patch(_MplRect(
                    (_wa[0], _wb[0]), _wa[1] - _wa[0], _wb[1] - _wb[0],
                    fill=False, edgecolor=_sc, lw=2.0, zorder=5))
                if _sl:
                    ax.text(_wa[0], _wb[1],
                            f"{_sl} face, "
                            f"{_st['axis']}={_st['pos_mm']:g} mm",
                            color=_sc, fontsize=6.0, ha="left", va="bottom",
                            zorder=6, bbox=dict(fc="white", ec="none",
                                                alpha=0.72, pad=0.6))
        la, lb = view_labels(v, scene)
        ax.set_xlabel(la, fontsize=9)
        ax.set_ylabel(lb, fontsize=9)
        _cut = ""
        if _m is not None and _m.at is not None:
            _norm = ({"xy": "z", "xz": "y", "yz": "x"}).get(v, "")
            _cut = f"\nslice at {_norm} = {_m.at:.2f} mm"
        _nm = getattr(scene, "axis_names", ("x", "y", "z"))
        _vname = _nm[_AXES[v][0]] + _nm[_AXES[v][1]]
        ax.set_title(f"{_vname} view" + _cut
                     + ("\n(no solved field in this plane)"
                        if no_field else ""), fontsize=9)

    # The per-electrode legend is deliberately NOT drawn. On a real stack
    # it listed more entries than the figure had room for, and it named
    # what the colour-basis caption already states. Electrode identity
    # belongs in the caption or the surrounding text, not stacked over the
    # instrument.
    #
    # A per-GROUP legend IS drawn for multi-assembly scenes (each
    # assembly gets a named color). Different animal from
    # the removed per-electrode list: a composed layout (quad + profiler
    # + OA) has a handful of drive groups whose OUTLINE colours are the
    # only way to tell the assemblies apart, and the caption's name list
    # does not say which colour is which. Same agnosticism as
    # _body_colors: groups and colours come from the bodies present.
    # Capped at 6 entries so the old too-many-entries failure cannot
    # recur; beyond that the caption remains the identity channel.
    _legend_h = 0.0
    _grps_present = sorted({getattr(b, "group", None)
                            for b in scene.bodies
                            if getattr(b, "group", None)})
    if legend and 2 <= len(_grps_present) <= 6:
        from matplotlib.patches import Patch
        _gpal = STYLE["groups"]
        _gidx = {g: _gpal[k % len(_gpal)]
                 for k, g in enumerate(_grps_present)}
        _handles = [Patch(facecolor="none", edgecolor=_gidx[g],
                          linewidth=2.0, label=g)
                    for g in _grps_present]
        _legend_h = 0.030 + 0.022 * ((len(_handles) + 2) // 3)
        fig.legend(handles=_handles, loc="lower center",
                   ncol=min(3, len(_handles)), fontsize=8,
                   frameon=False,
                   bbox_to_anchor=(0.5, _notes_h))
        # no legend title: the caption line already names the outline
        # channel, and a title here collides with panel x-labels
    named = [b for b in scene.bodies if b.volt is not None]

    # ---- line-out row
    if nlo:
        # the line-out row is the LAST gridspec row in either layout: index
        # it by position, not by the hardcoded row 1 that only held for the
        # side-by-side case (would have put the line-outs over panel 2 in a
        # stacked figure).
        lax = fig.add_subplot(gs[-1, :])
        for lo in scene.lineouts:
            lax.plot(lo.s, lo.values, lw=1.3, label=lo.label)
            for val, lab in lo.marks:
                lax.axhline(val, color="r", ls=":", lw=1.0)
                lax.annotate(lab, (lo.s[0], val), fontsize=7.5, color="r",
                             va="bottom")
        lax.set_xlabel("distance along the cut (mm)", fontsize=9)
        lax.set_ylabel(scene.lineouts[0].quantity, fontsize=9)
        lax.grid(alpha=0.25)
        lax.legend(fontsize=7)
        lax.set_title("line-outs (the value the number is read from)",
                      fontsize=10)

    # ---- title / caption / notes: MEASURED, not guessed.
    # These bands used to be placed at fixed figure fractions, which meant a
    # long title ran through the colour-basis caption and the notes block ran
    # through the x-axis labels. Text height depends on the string, the font
    # size, the number of lines and the figure size, so the only correct way
    # to reserve room is to draw the text, measure what it actually occupies,
    # and shrink the axes rect by that much.
    n = "\n".join(f"- {t}" for t in scene.notes[:6])
    # WRAP the title to the figure width. The band logic below measures text
    # HEIGHT, so a title taller than expected is handled -- but nothing
    # constrained WIDTH, and a one-line title wider than the canvas was
    # simply clipped at both ends (every panel whose
    # title carried its operating point lost the beginning and the end of
    # the sentence). Wrap at the measured character capacity, then let the
    # existing measure-and-shrink logic reserve the room the wrapped block
    # actually needs.
    # The pure-SOLVER stamp is dropped from the TITLE (it
    # consumed title space while the footer notes already state "metal
    # drawn from the solver's own mask" — R5 provenance stays satisfied
    # in words). SCHEMATIC and MIXED stamps remain in-title: those are
    # WARNINGS that the picture is not evidence, and a warning is never
    # demoted to a footnote.
    if scene.provenance == SOLVER and len(scene.stamps()) == 1:
        _title_txt = scene.title
    else:
        _title_txt = f"{scene.title}   [{scene.stamp()}]"
    _fig_w_in = fig.get_size_inches()[0]
    # ~0.55 * fontsize(pt) per character is the width of DejaVu Sans digits
    # and lower case at this size; 72 pt per inch.
    _chars = max(40, int(_fig_w_in * 72.0 / (11 * 0.55)))
    if len(_title_txt) > _chars:
        import textwrap as _tw
        _title_txt = "\n".join(_tw.wrap(_title_txt, _chars,
                                        break_long_words=False))
    t_title = fig.text(0.5, 0.995, _title_txt,
                       fontsize=11, ha="center", va="top")
    t_notes = fig.text(0.01, 0.006, n, fontsize=7, ha="left", va="bottom",
                       color="#444")

    def _band(artist):
        """Figure-fraction bbox of a text artist, after a real draw."""
        bb = artist.get_window_extent(fig.canvas.get_renderer())
        return bb.transformed(fig.transFigure.inverted())

    fig.canvas.draw()                      # required before measuring
    _pad = 0.012
    _top = _band(t_title).y0 - _pad
    if named:
        # the caption goes UNDER the measured title, never across it
        t_cap = fig.text(0.01, _top, _color_basis(scene.bodies), fontsize=7,
                         color="#444", ha="left", va="top")
        fig.canvas.draw()
        _top = _band(t_cap).y0 - _pad
    _bot = _band(t_notes).y1 + _pad
    # tight_layout places the axes BOX inside the rect, but with a colorbar
    # present its tick and axis labels can still hang below — which is how
    # the notes ended up under the x-axis labels. So converge on the real
    # DECORATED extent: lay out, measure what the axes actually occupy
    # (get_tightbbox includes tick labels, axis labels and titles), and give
    # back whatever still intrudes into a band.
    for _ in range(5):
        fig.tight_layout(rect=(0, _bot, 1, _top))
        fig.canvas.draw()
        _r = fig.canvas.get_renderer()
        _boxes = [a.get_tightbbox(_r).transformed(fig.transFigure.inverted())
                  for a in fig.axes]
        _lo = min(b.y0 for b in _boxes)
        _hi = max(b.y1 for b in _boxes)
        _need_b = (_band(t_notes).y1 + _pad) - _lo
        _need_t = _hi - _top
        if _need_b <= 1e-3 and _need_t <= 1e-3:
            break
        _bot += max(_need_b, 0.0)
        _top -= max(_need_t, 0.0)
    else:
        print(f"[viz] layout did not converge for {scene.title!r}; the "
              f"notes/title bands may crowd the panels — reduce the number "
              f"of notes or enlarge the figure")
    if layout == "column":
        # Equal-aspect panels letterbox when their axes BOX is proportionally
        # wider than the data: the drawing shrinks and white bands appear at
        # the sides. The band heights above are decided after the figure size
        # is chosen, so the reserved furniture is only an estimate. Measure
        # the converged boxes and grow the figure by exactly the height the
        # widest shortfall needs, then re-converge once.
        fig.canvas.draw()
        _grow = 0.0
        for _ax, _v in zip(axes, vs):
            _a0, _a1, _b0, _b1 = view_extent(scene, _v)
            _data_ar = max(_a1 - _a0, 1e-9) / max(_b1 - _b0, 1e-9)
            _bb = _ax.get_position()
            _w_in, _h_in = _bb.width * fig_w, _bb.height * fig.get_figheight()
            _need_h = _w_in / _data_ar
            _grow = max(_grow, _need_h - _h_in)
        if _grow > 0.05:
            _new_h = min(fig.get_figheight() + _grow, 24.0)
            fig.set_size_inches(fig_w, _new_h)
            fig.tight_layout(rect=(0, _bot * (fig_h / _new_h),
                                   1, 1 - (1 - _top) * (fig_h / _new_h)))
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=dpi)
    _mpl_release(fig)
    return fig


def distribution_compare(panels, *, title, operating_point, bins=30,
                         figsize_per_panel=(3.6, 3.6), dpi=DEFAULT_DPI):
    """Two-sample (or k-sample) distribution comparison figure — the L5
    statistical-contract renderer (collisional layers are
    compared as DISTRIBUTIONS, never per-ion).

    panels : list of dicts, each
        {"name": str, "unit": str, "samples": {label: 1-D array}, and
         optionally "bins": int, "center": "mean", "delta_scale": float,
         "delta_unit": str}
        Every sample in a panel is drawn as a density step-histogram on
        shared bin edges; the legend carries n, mean and std per sample.

        "center": "mean" subtracts EACH sample's own mean before
        binning, so samples whose means are far apart relative to their
        widths can still be compared. Without it, two 1 ns peaks whose
        means differ by 100 ns share an axis 100 ns wide and each
        collapses into a single bin -- the comparison renders as two
        bars and shows nothing about shape. The absolute means are not
        lost: they stay in the legend, and the axis says so. Use it when
        the question is "which is narrower", not "which arrives first".
        "delta_scale"/"delta_unit" rescale that centred axis (e.g. 1e3
        and "ns" for a sample held in us).
    title : figure title (what is being compared).
    operating_point : REQUIRED string placed under the title — gas,
        drive, m/z, seed; a distribution figure without its operating
        point is not a result.
    Returns the matplotlib figure; the caller saves/shows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not panels:
        raise VizError("distribution_compare: no panels given")
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n,
                                            figsize_per_panel[1]), dpi=dpi)
    axes = np.atleast_1d(axes)
    for ax, pan in zip(axes, panels):
        samples = pan["samples"]
        if not samples:
            raise VizError("distribution_compare: panel %r has no samples"
                           % pan.get("name"))
        centred = pan.get("center") == "mean"
        if centred and pan.get("center") not in (None, "mean"):
            raise VizError("distribution_compare: unknown center %r "
                           "(only \"mean\" is defined)" % pan.get("center"))
        scale = float(pan.get("delta_scale", 1.0)) if centred else 1.0
        dunit = pan.get("delta_unit", pan["unit"]) if centred else pan["unit"]
        prepared = {lab: ((np.asarray(v, float) - np.asarray(v, float).mean())
                          * scale if centred else np.asarray(v, float))
                    for lab, v in samples.items()}
        allv = np.concatenate(list(prepared.values()))
        edges = np.histogram_bin_edges(allv, bins=pan.get("bins", bins))
        for lab, v in samples.items():
            v = np.asarray(v, float)
            ax.hist(prepared[lab], bins=edges, density=True, histtype="step",
                    lw=1.6,
                    label="%s (n=%d, mean %.6g %s, sd %.3g %s)"
                          % (lab, v.size, v.mean(), pan["unit"],
                             v.std() * scale, dunit))
        ax.set_xlabel("%s%s (%s)"
                      % (pan["name"], " - own mean" if centred else "", dunit))
        ax.set_ylabel("density")
        # LEGEND CONVENTION: never inside the data axes. Sitting at
        # "upper right" it covered the peaks of exactly the narrow,
        # tall distributions this renderer exists to compare -- the
        # tuned sample is the tallest thing in the panel, so the legend
        # hid the result. Below the axes, one row per sample.
    fig.suptitle(wrap_title("%s\n%s" % (title, operating_point), fig),
                 fontsize=10)
    # Reserve the legend band FIRST, then place each legend against the
    # axes' FINAL position. Anchoring in axes fractions does not work
    # here: tight_layout shrinks the axes, so a fixed fraction below it
    # moves up onto the x label -- which is exactly what it did.
    fig.tight_layout(rect=(0, 0.22, 1, 0.90))
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        if not h:
            continue
        pos = ax.get_position()
        # Per-axes (not one shared): each panel's labels carry ITS OWN
        # n/mean/sd, so a single legend would caption every panel with
        # the first panel's numbers.
        fig.legend(h, lab, loc="upper center", fontsize=7, frameon=False,
                   ncol=1, bbox_to_anchor=(pos.x0 + pos.width / 2.0,
                                           pos.y0 - 0.14),
                   bbox_transform=fig.transFigure)
    return fig


def resolution_anatomy(panels, *, title, operating_point, bins=60,
                       zoom_sigma=5.0, figsize_per_panel=(4.6, 4.4),
                       dpi=DEFAULT_DPI):
    """Show HOW a resolving power is read off an arrival-time distribution.

    A printed R is a claim; this figure is the derivation, drawn on the
    same samples the number came from: the distribution, the mean it is
    referenced to, the width that divides it, and the arithmetic joining
    them. It exists because "R = 9200" tells a reader nothing about
    whether the ensemble supports three digits.

    panels : list of dicts, each
        {"name": str, "unit": str, "sample": 1-D array, and optionally
         "label": str}
    title : what is being measured.
    operating_point : REQUIRED -- tune, source, ensemble size. A
        resolution figure without its operating point is not a result.
    bins : histogram bins WITHIN the zoom window.
    zoom_sigma : half-width of the drawn window, in sigma about the mean.
        The zoom is the point: at full scale a 0.8 ns peak on a 6 us
        arrival is one pixel wide and shows nothing.

    Returns the matplotlib figure; the caller saves/shows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not panels:
        raise VizError("resolution_anatomy: no panels given")
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n,
                                            figsize_per_panel[1]), dpi=dpi)
    axes = np.atleast_1d(axes)
    for ax, pan in zip(axes, panels):
        v = np.asarray(pan["sample"], float)
        if v.size < 2:
            raise VizError(
                "resolution_anatomy: panel %r has %d sample(s); a width "
                "cannot be measured from fewer than two."
                % (pan.get("name"), v.size))
        mean, sd = float(v.mean()), float(v.std())
        if not np.isfinite(sd) or sd <= 0.0:
            raise VizError(
                "resolution_anatomy: panel %r has zero spread, so R is "
                "undefined -- every ion arrived at the same time."
                % pan.get("name"))
        # sigma -> FWHM for a Gaussian: 2*sqrt(2 ln 2). Named, not 2.3548
        # typed in: the reader should be able to see where it comes from.
        k_fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0))
        fwhm = k_fwhm * sd
        half = fwhm / 2.0
        R = mean / (2.0 * fwhm)

        # PLOTTED ABOUT THE MEAN. An absolute axis puts a sub-ns width on
        # a several-us arrival, so matplotlib falls back to an offset
        # label ("+1.399e1") and the reader has to do arithmetic to see
        # the width the figure is about. The mean itself is not hidden --
        # it is printed in the annotation box, in the sample's own unit.
        scale = float(pan.get("delta_scale", 1.0))
        dunit = pan.get("delta_unit", pan["unit"])
        d = (v - mean) * scale
        sd_d, fwhm_d, half_d = sd * scale, fwhm * scale, half * scale
        lo, hi = -zoom_sigma * sd_d, zoom_sigma * sd_d
        edges = np.linspace(lo, hi, int(bins) + 1)
        ax.hist(d, bins=edges, density=True, histtype="stepfilled",
                alpha=0.35, color="0.45")
        ax.hist(d, bins=edges, density=True, histtype="step", lw=1.4,
                color="0.15")

        # the Gaussian the width actually comes from, drawn so the
        # sigma->FWHM step is visible rather than asserted
        xs = np.linspace(lo, hi, 400)
        gauss = (1.0 / (sd_d * math.sqrt(2.0 * math.pi))) * \
            np.exp(-0.5 * (xs / sd_d) ** 2)
        ax.plot(xs, gauss, lw=1.5, color="tab:blue",
                label="Gaussian of the same sigma")
        peak = gauss.max()
        # headroom for the annotation box, which otherwise sits on the
        # distribution it describes
        ax.set_ylim(0.0, peak * 2.15)
        # generic label: the figure-level legend is shared, so a per-panel
        # VALUE here would caption every panel with the first panel's mean.
        # Each panel's own mean is printed in its annotation box.
        ax.axvline(0.0, lw=1.2, color="tab:red", label="mean arrival time")
        ax.annotate("", xy=(-half_d, peak / 2.0), xytext=(half_d, peak / 2.0),
                    arrowprops=dict(arrowstyle="<->", color="tab:orange",
                                    lw=1.6))
        ax.plot([], [], color="tab:orange", lw=1.6,
                label="FWHM = 2sqrt(2ln2)*sigma")
        # ABOVE the arrow, on a white patch: printed at half-max it sat
        # on the arrow it labels, orange on orange, and was unreadable.
        ax.text(0.0, peak * 0.60, "FWHM %.3g %s" % (fwhm_d, dunit),
                ha="center", va="bottom", fontsize=8, color="tab:orange",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.85))
        ax.text(0.02, 0.97,
                "n = %d\nmean t = %.5g %s\nR = t / (2*FWHM) = %.0f"
                % (v.size, mean, pan["unit"], R),
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                          alpha=0.9))
        ax.set_xlabel("%s - mean (%s)" % (pan["name"], dunit))
        ax.set_ylabel("density")
        if pan.get("label"):
            ax.set_title(pan["label"], fontsize=9)
        ax.set_xlim(lo, hi)
    # LEGEND CONVENTION: outside the data axes -- ONE figure-level legend
    # below, deduplicated across panels, as elsewhere in this module. A
    # per-axes legend anchored below the axes lands on the x label as
    # soon as the panel is short, which is exactly when it is needed.
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 3),
               fontsize=8, frameon=False)
    fig.suptitle(wrap_title("%s\n%s" % (title, operating_point), fig),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    return fig


def lineout_figure(traces, *, xlabel, ylabel, title, operating_point,
                   spans=None, dpi=DEFAULT_DPI, logy: bool = False,
                   logx: bool = False):
    """Overlay 1-D traces (e.g. solved axial potential vs a digitized
    literature target).  traces = [(x, y, label, style)] with style in
    {"line","dash","dots"}.  spans = [(x0, x1, label)] shaded bands
    (e.g. electrode extents).  operating_point REQUIRED.  Returns the
    Figure; the caller saves.  (Built for profile-vs-reference
    validation; generic for any such readout.)

    logy: log-scale the y axis (for the K_z resonance
    tolerance, where the quantity collapses across four decades and a
    linear axis renders the whole post-cliff branch as a flat line on
    zero).  Defaults False, so every existing caller is unaffected --
    this is a display option on the same renderer, not a second one.
    It REFUSES non-positive data rather than letting matplotlib drop
    those points silently, because a vanished point in a tolerance
    curve reads as "not measured" when it usually means "collapsed".

    logx: log-scale the x axis (for PITCH-CONVERGENCE
    ladders, where the abscissa is a halving sequence and the ORDER of
    convergence is the slope on a log-log plot -- on a linear axis the
    fine rungs pile up at the origin and the order cannot be read off).
    Same display-option-on-the-same-renderer precedent as logy above,
    and the same refusal on non-positive data for the same reason.
    """
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    sty = {"line": dict(lw=1.6), "dash": dict(lw=1.4, ls="--"),
           "dots": dict(lw=0, marker="o", ms=3)}
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=dpi)

    def _refuse_nonpositive(axis_name, index):
        import numpy as _np
        for _tr in traces:
            _a = _np.asarray(_tr[index], float)
            _bad = _a[~(_a > 0) & ~_np.isnan(_a)]
            if _bad.size:
                raise VizError(
                    f"lineout_figure(log{axis_name}=True): trace "
                    f"{_tr[2]!r} carries {_bad.size} non-positive "
                    f"value(s) (min {_bad.min():g}) which a log axis "
                    f"cannot show. Plot it linearly, or filter the "
                    f"points and say so in the caption -- do not let "
                    f"them disappear.")
    if logx:
        _refuse_nonpositive("x", 0)
        ax.set_xscale("log")
    if logy:
        _refuse_nonpositive("y", 1)
        ax.set_yscale("log")
    if spans:
        for x0, x1, lab in spans:
            ax.axvspan(x0, x1, color="#cccccc", alpha=0.45)
            ax.text(0.5 * (x0 + x1), 0.985, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6, color="#555555")
    for x, y, lab, st in traces:
        ax.plot(x, y, label=lab, **sty.get(st, sty["line"]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle(wrap_title(f"{title}\n{operating_point}", fig), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def param_map(a_vals, b_vals, Z, *, a_label, b_label, quantity,
              title, operating_point, levels=None, cmap=None,
              mark_box=None, dpi=DEFAULT_DPI, log10: bool = False):
    """2-D parameter map: scalar Z[b, a] over a swept (a, b) grid, with
    contour overlay -- the acceptance-map / stability-map renderer
    (built for MR-TOF acceptance mapping; the ion-
    stability maps consume the same entry point).

    Z is imshow-ordered [b, a] (rows = b).  NaN cells (lost ions /
    unevaluated) render as hatched grey, visibly distinct from any
    finite value.  mark_box=(a0,a1,b0,b1) outlines a declared region
    (e.g. the optimization bundle) on top of the measured map.
    operating_point is REQUIRED: a map without its tune is not a
    result.  Returns the Figure; the caller saves."""
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    import numpy as _np
    Z = _np.asarray(Z, float)
    if Z.shape != (len(b_vals), len(a_vals)):
        raise VizError(f"param_map: Z shape {Z.shape} != "
                       f"({len(b_vals)}, {len(a_vals)}) from the axes")
    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=dpi)
    disp = _np.log10(Z) if log10 else Z
    ax.set_facecolor("#d9d9d9")
    ext = (min(a_vals), max(a_vals), min(b_vals), max(b_vals))
    im = ax.imshow(disp, origin="lower", aspect="auto", extent=ext,
                   cmap=cmap or STYLE["field"])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(("log10 " if log10 else "") + quantity)
    if levels is not None and _np.isfinite(disp).sum() > 4:
        cs = ax.contour(a_vals, b_vals, disp, levels=levels,
                        colors="k", linewidths=0.9)
        ax.clabel(cs, fontsize=7, fmt="%g")
    if _np.isnan(Z).any():
        ax.contourf(a_vals, b_vals, _np.isnan(Z).astype(float),
                    levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.plot([], [], ls="none", marker="s", mfc="none", mec="#555555",
                label="lost / no value (hatched)")
        ax.legend(fontsize=7, loc="upper right")
    if mark_box is not None:
        a0, a1, b0, b1 = mark_box
        ax.plot([a0, a1, a1, a0, a0], [b0, b0, b1, b1, b0],
                color="#d62728", lw=1.6, ls="--")
    ax.set_xlabel(a_label)
    ax.set_ylabel(b_label)
    fig.suptitle(wrap_title(f"{title}\n{operating_point}", fig),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def render_perspective(scene: Scene, *, elev: float = 18.0,
                       azim: float = -62.0, dpi: int = DEFAULT_DPI,
                       body_alpha: float = 0.55, figsize=(10.0, 6.0),
                       cutaway_y: float | None = None, ray_lw: float = 0.7):
    """Single 3-D PERSPECTIVE figure of a Scene (matplotlib 3-D axes).

    A 3-D perspective view.
    This SUPPLEMENTS the orthogonal multi-axis panels of
    render_mpl -- it never replaces them: a perspective view foreshortens,
    so dimensions are read off the orthogonal panels, and the doctrine
    figure for any 3-D artifact remains render_mpl's three projections.

    Bodies: a flat planar body (all z equal) inside a scene that DECLARES
    a slab is extruded over that slab for display -- the same honest
    statement render_mpl's xz/yz panels make, and the scene's slab note
    already says so.  A body with its own 3-D faces is drawn as given.
    Rays are drawn as 3-D polylines with their recorded z.  Aspect is
    isotropic (mm is mm on every axis).  Returns the Figure; saving is
    the caller's decision (contract: no savefig side effects).

    cutaway_y: if set, bodies lying ENTIRELY at y >= cutaway_y are not
    drawn, and the figure says so -- the standard section-cut that lets a
    thin enclosed instrument show its median-plane ion paths.  A cutaway
    is a stated display choice, never a silent geometry edit.
    """
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x0, x1, y0, y1, z0, z1 = scene.box()
    if scene.slab_mm > 0 and (z1 - z0) <= 1e-9:
        z0, z1 = -0.5 * scene.slab_mm, 0.5 * scene.slab_mm
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    face, edge = _body_colors(scene.bodies)   # fill = DC, edge = drive group
    _ncut = 0
    for bi, b in enumerate(scene.bodies):
        col = face.get(bi, "#888888")
        polys3 = []
        for p in b.polys:
            P = np.asarray(p, float).reshape(-1, 3)
            if len(P) < 3:
                continue
            if cutaway_y is not None and float(P[:, 1].min()) >= cutaway_y:
                _ncut += 1          # per-POLY cut: a Body may pair plates
                continue            # across the median (upper + lower)
            if np.ptp(P[:, 2]) <= 1e-9 and scene.slab_mm > 0:
                # flat section in a declared slab -> honest extrusion
                polys3.extend(extrude(P[:, :2], z0, z1))
            else:
                polys3.append(P)
        if polys3:
            pc = Poly3DCollection(polys3, facecolors=col,
                                  edgecolors="none", alpha=body_alpha)
            ax.add_collection3d(pc)
    for rays in scene.rays:
        for path in rays.paths:
            P = np.asarray(path, float).reshape(-1, 3)
            if len(P) >= 2:
                ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=ray_lw)
    for axis, mm, lbl in scene.bound_marks:
        if axis == "z":
            # kill bound drawn as a dashed frame at z = mm
            fx = [x0, x1, x1, x0, x0]
            fy = [y0, y0, y1, y1, y0]
            ax.plot(fx, fy, [mm] * 5, ls="--", lw=1.0, color="#9467bd")
            ax.text(x1, y1, mm, lbl, fontsize=7, color="#9467bd")
    # DECLARED STATIONS: the window rectangle at pos_mm on the station's
    # axis, spanned by the two window axes (an undeclared window axis
    # spans the scene — honest full aperture). Solid translucent face:
    # this is the detector the ions fly INTO, not a domain edge.
    _ext = {"x": (x0, x1), "y": (y0, y1), "z": (z0, z1)}
    for _st in scene.station_marks:
        _sc = _STATION_COLORS.get(_st["kind"], "#2ca02c")
        _a1, _a2 = [a for a in ("x", "y", "z") if a != _st["axis"]]
        _u = _st["window"].get(_a1, _ext[_a1])
        _v = _st["window"].get(_a2, _ext[_a2])
        _corners = []
        for cu, cv in ((_u[0], _v[0]), (_u[1], _v[0]),
                       (_u[1], _v[1]), (_u[0], _v[1])):
            pt = {_st["axis"]: _st["pos_mm"], _a1: cu, _a2: cv}
            _corners.append([pt["x"], pt["y"], pt["z"]])
        ax.add_collection3d(Poly3DCollection(
            [_corners], facecolors=_sc, edgecolors=_sc, alpha=0.35,
            linewidths=1.4))
        ax.text(*_corners[2], f"{_st['name']} ({_st['kind']})",
                fontsize=7, color=_sc)
    nx, ny, nz = scene.axis_names
    ax.set_xlabel(f"{nx} (mm)")
    ax.set_ylabel(f"{ny} (mm)")
    ax.set_zlabel(f"{nz} (mm)")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, max(z1 - z0, 1e-9)))
    ax.view_init(elev=elev, azim=azim)
    fig.suptitle(wrap_title(scene.title, fig), fontsize=10)
    fig.text(0.01, 0.01, scene.stamp(), fontsize=6, color="#666666")
    if scene.slab_mm > 0:
        fig.text(0.01, 0.03,
                 f"slab depth {scene.slab_mm:g} mm DECLARED, not solved "
                 "(z-invariant planar solve extruded)", fontsize=6,
                 color="#666666")
    if cutaway_y is not None and _ncut:
        fig.text(0.01, 0.05,
                 f"CUTAWAY: {_ncut} plate face(s) at y >= "
                 f"{cutaway_y:g} mm not drawn (display only; all were "
                 "solved and flown)", fontsize=6, color="#b04a4a")
    _mpl_release(fig)
    return fig


def show(fig, name=None, *, outdir=None, save=False, dpi=170,
         quiet=False):
    """DISPLAY a figure inline in the notebook; save only on request.

    The defect this exists to kill: a cell that
    calls `fig.savefig(...)` and ends there produces NOTHING visible —
    the reader is told a figure was written and has to go find it.
    Generating a figure is not displaying it.  Inline is the default
    here and saving is the option, which is the inverse of what
    hand-rolled cells keep doing.

    fig     a matplotlib Figure (render_mpl / render_perspective return
            one; so does plt.subplots).
    name    base filename, used ONLY when save=True.
    outdir  destination directory, REQUIRED when save=True -- this
            function invents no path (no repo-relative default, no
            cwd fallback), because a figure written somewhere the
            caller did not name is a figure nobody finds.
    save    write the file as well as displaying it.

    Returns the written path, or None when nothing was written.

    Note: render_mpl deregisters its figure from pyplot (so the inline
    backend cannot double-display it).  A deregistered figure still
    renders and still saves -- passing one here is the intended path.
    """
    if fig is None:
        raise VizError("show(fig=None): nothing to display. The caller "
                       "probably forgot to return the figure from its "
                       "renderer.")
    path = None
    if save:
        if not name:
            raise VizError("show(save=True) needs a `name` for the file.")
        if outdir is None:
            raise VizError(
                f"show(save=True, name={name!r}) needs an explicit "
                f"`outdir`; this function will not choose a directory "
                f"for you.")
        stem = str(name)
        if not stem.lower().endswith((".png", ".svg", ".pdf")):
            stem += ".png"
        path = os.path.join(str(outdir), stem)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=dpi)
        if not quiet:
            print(f"saved -> {path}")
    try:
        from IPython.display import display as _display, Image as _Image
        from IPython import get_ipython as _gi
    except ImportError as e:
        raise VizError(
            "show() needs IPython to display inline; import failed "
            f"({e}). In a plain script, call fig.savefig() directly or "
            "pass save=True with an outdir.") from e
    if _gi() is None:
        # A visible, reasoned skip -- not a silent no-op.
        print(f"[show] not inside an IPython kernel; figure "
              f"{name or ''!r} was not displayed"
              + (f" (written to {path})" if path else
                 " and was NOT written (save=False)"))
        return path
    # ROOT CAUSE of "figures don't show up":
    # `display(fig)` only emits a PNG if a matplotlib figure formatter
    # is registered, which it is NOT when a notebook has called
    # matplotlib.use("Agg") -- the display then silently degrades to a
    # text/plain repr ("<Figure size ...>") and the reader sees
    # nothing. So the payload is rendered HERE, explicitly, and is
    # backend-independent by construction.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    _display(_Image(data=buf.getvalue()))
    # Deregister so an inline backend cannot flush the same figure a
    # second time at end-of-cell (the double-display defect).
    _mpl_release(fig)
    return path


def report(scenes, outdir, *, name="report", views=None, field=True):
    """Multi-scene report: one PNG per scene + an index.md that states the
    provenance of every panel.  Returns the list of written paths."""
    os.makedirs(outdir, exist_ok=True)
    written, lines = [], [f"# {name}", "",
                          f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for i, sc in enumerate(scenes):
        safe = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in sc.title)[:60]
        p = os.path.join(outdir, f"{i:02d}_{safe}.png")
        fig = render_mpl(sc, views=views, field=field, path=p)
        # matplotlib is a HARD dependency of render_mpl -- which just ran, three
        # lines up.  Guarding its import with `except Exception: pass` was
        # cargo, and it would have swallowed a real failure in plt.close().
        import matplotlib.pyplot as plt
        plt.close(fig)
        written.append(p)
        lines += [f"## {sc.title}", f"**{sc.stamp()}**", "",
                  f"![{sc.title}]({os.path.basename(p)})", ""]
        lines += [f"- {t}" for t in sc.notes] + [""]
    idx = os.path.join(outdir, "index.md")
    with open(idx, "w") as fh:
        fh.write("\n".join(lines))
    written.append(idx)
    return written


# ------------------------------------------------------------- ZOOM POLICY
# NOTE: square framing (square_extent + the `square=`
# switch on apply_zoom_policy) is WITHDRAWN, not parked. It padded the
# shorter axis to force a square field of view, and it needed a second
# authority over the pane pixel box to avoid letterboxing -- two authorities
# over one box is the same defect M-VIZ-Z2 removed from free_aspect_axes.
# Framing now comes from the declared extent alone.
def free_aspect_axes(extent, *, true_scale: bool = True, px_width: int = 900,
                     px_per_mm: float | None = None, uirevision: str = "keep",
                     x_title: str = "x (mm)", y_title: str = "y (mm)") -> dict:
    """M-VIZ-Z.  Plotly layout for a 2-D panel with a FREE-ASPECT box zoom.

    ROOT CAUSE this replaces
    ------------------------
    yaxis=dict(scaleanchor="x", scaleratio=1) asks plotly to hold a fixed
    pixel-per-mm ratio on both axes AT ALL TIMES.  It is not just a first-
    paint setting: it is a live constraint, so when the user drags a zoom
    rectangle plotly RESHAPES it -- it grows the short side until the ratio
    is satisfied again.  The rectangle you get is not the rectangle you
    drew.  On this instrument that is exactly backwards: the interesting
    boxes are long and thin (a metres-long drift a few mm tall; a sub-mm
    gap inside a stack tens of mm across), and a ratio-locked zoom refuses
    to make them.

    THE FIX (root cause, not a symptom)
    -----------------------------------
    Do not constrain the axes at all.  Deliver true 1:1 scale by SIZING THE
    PANEL to the data box instead -- height = width * (dy/dx) -- so the
    first paint is undistorted, and every later drag is a free rectangle
    with whatever aspect ratio the user drew.  Double-click (plotly's
    autoscale) restores the data box, and because the panel is sized to it,
    that restores 1:1 as well: the true-scale view is one gesture away and
    is never imposed.

    true_scale=False keeps the free zoom and lets the panel fill its
    container (for a "stretch to fit" pane); nothing about the zoom changes.
    """
    a0, a1, b0, b1 = (float(v) for v in extent)
    _dx, _dy = max(a1 - a0, 1e-12), max(b1 - b0, 1e-12)
    lay = {
        "dragmode": "zoom",                     # rectangle select == zoom
        "uirevision": uirevision,               # a redraw does not steal the zoom
        "xaxis": {"range": [a0, a1], "autorange": False},
        "yaxis": {"range": [b0, b1], "autorange": False},
        # NO scaleanchor / scaleratio ANYWHERE.  That is the whole point.
    }
    if x_title is not None:
        lay["xaxis"]["title"] = x_title
    if y_title is not None:
        lay["yaxis"]["title"] = y_title
    # NOTE (M-VIZ-Z2): the figure layout carries NO width/height.
    #
    # It used to. That was a second, competing authority over the same pixel
    # box: every pane is built with sizing_mode="stretch_width" and its own
    # height, so a figure that also insisted on a fixed pixel box made the
    # two fight -- each relayout kicked off a resize round-trip and the zoom
    # felt stunted, "like it is trying to rescale". It was.
    #
    # True scale is delivered by sizing the PANE (see pane_size_for), which
    # is what this function's docstring said all along. One authority.
    return lay


def pane_size_for(extent, *, px_width: int = 900,
                  px_per_mm: float | None = None,
                  max_width: int = 1000, max_height: int = 760) -> dict:
    """Pixel box for the PANE that shows a panel of this data extent, so the
    first paint is 1:1 in mm (height = width * dy/dx).

    Apply to the pane, never to the figure:
        pn.pane.Plotly(fig, sizing_mode="fixed", **pane_size_for(extent))

    Agnostic: it knows only the data box. Panels that should stretch to fill
    a container simply do not call this -- and nothing about the free zoom
    changes either way, because the zoom freedom lives in the ABSENCE of
    scaleanchor, not in the sizing.
    """
    a0, a1, b0, b1 = (float(v) for v in extent)
    dx, dy = max(a1 - a0, 1e-12), max(b1 - b0, 1e-12)

    # ONE isotropic scale (px per mm), fitted inside a screen-sized box.
    #
    # This used to clamp the HEIGHT at a ceiling and leave the width alone,
    # which is worse than either alternative: the aspect ratio the function
    # exists to guarantee came out silently wrong (26% error on a 142 x 15 mm
    # xz view), AND a tall geometry still ran off the bottom of the window.
    # A floor or a cap on ONE axis is an anisotropic scale by another name.
    if px_per_mm is not None:
        s = float(px_per_mm)
    else:
        s = float(px_width) / dx
    s = min(s, max_width / dx, max_height / dy)      # fit the screen
    # NO minimum on either axis. A 850 x 16 mm drift region really is a thin
    # strip; "helping" by bumping the height (or the width) is an anisotropic
    # scale wearing a friendly hat, and it is the same mistake as the old
    # height clamp. If the user wants to fill the pane, they turn the aspect
    # lock OFF -- which is exactly what that control is for.
    return {"width": int(round(dx * s)), "height": int(round(dy * s))}


def apply_zoom_policy(fig, extent=None, *, true_scale: bool = True, **kw):
    """Apply M-VIZ-Z to an existing plotly Figure.  If `extent` is omitted it
    is measured from the figure's own data, so this works on any figure from
    any app (agnostic).  It also STRIPS any scaleanchor already present --
    a policy you can bypass by forgetting is not a policy."""
    if extent is None:
        extent = _fig_extent(fig)
    else:
        # NATIVE-AUTOSCALE ANCHOR (the modebar
        # "autoscale" button landed on junk in the quad transport view).
        # Plotly's autoscale ranges over TRACE data only; a view drawn
        # entirely from shapes (bounded electrode rects, bound-plane
        # lines) gives it nothing. Two invisible markers at the STATED
        # extent's corners make the button reproduce the declared framing
        # exactly — for every consumer that states an extent, not one
        # view. ("Reset axes" already restored the policy ranges; now
        # both buttons agree.)
        x0, x1, y0, y1 = (float(v) for v in extent)
        fig.add_scatter(x=[x0, x1], y=[y0, y1], mode="markers",
                        marker=dict(size=1, opacity=0.0),
                        showlegend=False, hoverinfo="skip",
                        name="_extent_anchor")
    lay = free_aspect_axes(extent, true_scale=true_scale, **kw)
    fig.update_layout(**lay)
    fig.update_yaxes(scaleanchor=None, scaleratio=None)
    fig.update_xaxes(scaleanchor=None, scaleratio=None)
    _mpl_release(fig)
    return fig


def figure_extent(fig):
    """Measure (x0,x1,y0,y1) from a plotly figure's own finite data."""
    return _fig_extent(fig)


def _fig_extent(fig):
    xs, ys = [], []
    for tr in fig.data:
        for key, sink in (("x", xs), ("y", ys)):
            v = getattr(tr, key, None)
            if v is None:
                continue
            a = np.asarray(v, dtype=object)
            a = np.array([t for t in a.ravel() if t is not None], float) \
                if a.dtype == object else np.asarray(v, float)
            a = a[np.isfinite(a)]
            if a.size:
                sink += [a.min(), a.max()]
    if not xs or not ys:
        raise VizError("cannot measure a figure extent: no finite x/y data. "
                       "Pass extent=(x0,x1,y0,y1) explicitly.")
    return (min(xs), max(xs), min(ys), max(ys))


def assert_free_zoom(fig_or_layout):
    """Gate helper: refuse a figure whose 2-D axes are scale-anchored."""
    lay = getattr(fig_or_layout, "layout", fig_or_layout)
    for ax in ("xaxis", "yaxis"):
        a = lay[ax] if not hasattr(lay, ax) else getattr(lay, ax)
        for k in ("scaleanchor", "scaleratio"):
            v = (a.get(k) if isinstance(a, dict) else getattr(a, k, None))
            if v not in (None, ""):
                raise VizError(
                    f"M-VIZ-Z violation: {ax}.{k} = {v!r}.  A scale-anchored "
                    "axis makes the box zoom rigid (plotly reshapes the drag "
                    "rectangle).  Size the panel for true scale instead -- "
                    "see viz_core.free_aspect_axes().")
    return True


def scene3d_layout(bounds, *, aspect: str = "data", uirevision="keep") -> dict:
    """Layout for a plotly 3-D scene.  aspectmode='data' keeps mm isotropic
    in the 3-D box; 3-D navigation is camera-based (orbit/dolly), so there
    is no drag rectangle to make rigid and no zoom conflict.  aspect='auto'
    stretches the box to the pane for a long thin instrument."""
    x0, x1, y0, y1, z0, z1 = (float(v) for v in bounds)
    return {"uirevision": uirevision,
            "scene": {"aspectmode": aspect,
                      "xaxis": {"title": "x (mm)", "range": [x0, x1]},
                      "yaxis": {"title": "y (mm)", "range": [y0, y1]},
                      "zaxis": {"title": "z (mm)", "range": [z0, z1]}}}


# ------------------------------------------------------- interactive panel
def interactive_panel(scene: Scene, view: str = "xy", *, field: bool = True,
              true_scale: bool = True, px_width: int = 900,
              px_per_mm: float | None = None,
              figsize: tuple | None = None,
              colorscale: str | None = None):
    """One INTERACTIVE panel of a Scene, under the M-VIZ-Z zoom policy.

    Renamed from `to_plotly` for readability: a
    public name should say what the caller GETS, not which library drew
    it. The backend is an implementation detail; `interactive_panel` and
    `render_mpl`'s static figure are two consumers of one Scene.

    figsize: (width, height) of the whole figure, **in pixels** — the
    same role matplotlib's `figsize` plays, named the same so it is
    findable. (Units differ on purpose: matplotlib counts inches because
    it targets print; this targets a browser, whose native unit is the
    pixel. Converting behind the scenes would mean guessing a DPI.)
    Omit it and the figure is sized true-scale from the data extent;
    `interactive_panels` forwards it to every panel.

    colorscale: any plotly colorscale name for the FIELD layer (None keeps
    the framework default). A display preference only — it changes no
    value, position, or geometry, so it is a caller choice, not STYLE."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fmap = {f.plane: f for f in scene.fields}
    f = fmap.get(view) if field else None
    if f is not None:
        a0, a1, b0, b1 = f.extent
        ny, nx = f.values.shape
        # CONTOUR LINES, not a filled gradient — the same doctrine as
        # render_mpl states in its own field block: a flood fill dominates
        # the panel and hides the geometry and the ion paths drawn over
        # it, while equipotential lines stay readable and their SPACING
        # reads as field strength directly (add
        # contours to plots with fields"). Hover reads the level.
        fig.add_contour(x=np.linspace(a0, a1, nx), y=np.linspace(b0, b1, ny),
                        z=f.values, colorscale=colorscale or "Viridis",
                        contours=dict(coloring="lines"),
                        line=dict(width=1.2), ncontours=26,
                        colorbar=dict(title=f.quantity),
                        hoverinfo="x+y+z")
    # METAL as a cut through this plane, when the scene carries one —
    # drawn with the existing vox_heatmap (one visual convention with the
    # app). A 3-D slice scene carries no projected bodies, so without this
    # the interactive panel showed the device with no metal at all.
    _m = {mm.plane: mm for mm in getattr(scene, "metal", [])}.get(view)
    if _m is not None:
        _ma0, _ma1, _mb0, _mb1 = _m.extent
        _mv = (np.asarray(_m.values, float) > 0).astype(float)
        vox_heatmap(fig, np.linspace(_ma0, _ma1, _mv.shape[1]),
                    np.linspace(_mb0, _mb1, _mv.shape[0]), _mv,
                    color="#4c72b0", alpha=0.35)
    for b in scene.bodies:
        if getattr(b, "outline", False):
            # OUTLINE body: one clean boundary path per loop, no fill —
            # a round rod is a circle, not a stack of slabs.
            _ln = (dict(color="#666666", width=1.4, dash="dash")
                   if getattr(b, "grid", False)
                   else dict(color="#222222", width=1.6))
            for p in b.polys:
                q = project(p, view)
                q = np.vstack([q, q[:1]])
                fig.add_scatter(x=q[:, 0], y=q[:, 1], mode="lines",
                                line=_ln,
                                name=b.name, showlegend=False,
                                hovertext=b.name, hoverinfo="text")
            continue
        for p in b.polys:
            q = project(p, view)
            q = np.vstack([q, q[:1]])
            fig.add_scatter(x=q[:, 0], y=q[:, 1], mode="lines", fill="toself",
                            fillcolor="rgba(200,200,200,0.85)",
                            line=dict(color="#222", width=1),
                            name=b.name, showlegend=False,
                            hovertext=b.name, hoverinfo="text")
    for r in scene.rays:
        for k, p in enumerate(r.paths):
            q = project(p, view)
            fig.add_scatter(x=q[:, 0], y=q[:, 1], mode="lines",
                            line=dict(width=1), showlegend=False,
                            name=f"{r.label}[{k}]")
    # DECLARED KILL BOUNDS: dashed, in sim_app's colour and style — one
    # convention across the app, the static renderer, and this panel.
    # Axis letters come from the _AXES authority (Convention A), never
    # from the view-name characters.
    _bha, _bva = _LABEL[_AXES[view][0]], _LABEL[_AXES[view][1]]
    for _bax, _bmm, _blab in scene.bound_marks:
        if _bax == _bha:
            fig.add_vline(x=_bmm, line=dict(color="#9467bd", width=1.2,
                                            dash="dash"))
        elif _bax == _bva:
            fig.add_hline(y=_bmm, line=dict(color="#9467bd", width=1.2,
                                            dash="dash"))
    # DECLARED STATIONS: same finite-window convention as render_mpl —
    # a solid kind-coloured segment over the station's window, with the
    # name on hover. One visual language across static and interactive.
    for _st in scene.station_marks:
        _sc = _STATION_COLORS.get(_st["kind"], "#2ca02c")
        _sl = f"{_st['name']} ({_st['kind']} station)"
        # extent fallback is LAZY: view_extent can refuse on field-less
        # planar panels; a station with a declared window
        # never needs it, so don't pay that failure for the common case.
        if _st["axis"] == _bha:
            _w = _st["window"].get(_bva)
            if _w is None:
                _sext = view_extent(scene, view)
                _w = (_sext[2], _sext[3])
            fig.add_scatter(x=[_st["pos_mm"]] * 2, y=list(_w),
                            mode="lines", line=dict(color=_sc, width=3),
                            name=_sl, showlegend=False,
                            hovertext=_sl, hoverinfo="text")
        elif _st["axis"] == _bva:
            _w = _st["window"].get(_bha)
            if _w is None:
                _sext = view_extent(scene, view)
                _w = (_sext[0], _sext[1])
            fig.add_scatter(x=list(_w), y=[_st["pos_mm"]] * 2,
                            mode="lines", line=dict(color=_sc, width=3),
                            name=_sl, showlegend=False,
                            hovertext=_sl, hoverinfo="text")
    la, lb = view_labels(view)
    apply_zoom_policy(fig, view_extent(scene, view), true_scale=true_scale,
                      px_width=px_width, x_title=la, y_title=lb)
    # STANDALONE SIZING (stretched renders in Jupyter).
    # In the app the PANE is the one sizing authority (pane_size_for,
    # M-VIZ-Z2) and the figure carries no size. A standalone figure — a
    # notebook cell — has no pane, so plotly falls back to container
    # sizing and the aspect is whatever the page gives it. Fix at the
    # SAME authority: apply pane_size_for's isotropic box (plus stated
    # margins) to the figure itself. `figsize=(w, h)` overrides both (a
    # user-declared box and aspect); true_scale=False keeps container
    # fill, unchanged. sim_app does not call interactive_panel, so the app's
    # pane-sized figures are untouched.
    _MARG = dict(l=60, r=120, t=48, b=50)
    if figsize is not None:
        fig.update_layout(width=int(figsize[0]), height=int(figsize[1]),
                          margin=_MARG)
    elif true_scale:
        _box = pane_size_for(view_extent(scene, view), px_width=px_width,
                             px_per_mm=px_per_mm)
        fig.update_layout(width=_box["width"] + _MARG["l"] + _MARG["r"],
                          height=_box["height"] + _MARG["t"] + _MARG["b"],
                          margin=_MARG)
    else:
        fig.update_layout(margin=_MARG)
    fig.update_layout(title=f"{scene.title} — {view} [{scene.provenance}]")
    _mpl_release(fig)
    return fig


def interactive_panels(scene: Scene, views=None, **kw) -> dict:
    """R3: the multi-axis INTERACTIVE set, keyed by view name.

    Renamed from `to_plotly_views` for readability."""
    return {v: interactive_panel(scene, v, **kw) for v in views_for(scene, views)}


# =====================================================================
# INSTRUMENT OVERLAY RENDERERS. MOVED VERBATIM from
# sim_app (its five _draw_* methods) with widget reads unwound into
# explicit kwargs: one renderer — sim_app orchestrates and
# explicit kwargs: one renderer (viz R1) — sim_app orchestrates and
# marshals widget state; viz_core draws. The bodies keep their comment
# history; each documents a fixed bug. One latent NameError (`rgba` on
# the end-on per-electrode branch) was found and fixed in the move.
# =====================================================================

EL_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
              "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f"]

def vox_heatmap(fig, x, y, img, *, color, alpha, fill=True):
    """Electrode voxels as a transparent-background heatmap PLUS an
    outline: cells that are metal in the SOLVE masks get the user's
    electrode color+alpha; all other cells fully transparent. Chunky when
    zoomed — but exactly the geometry the field was computed on (single
    source of truth).

    The OUTLINE is drawn independently of the fill. The xz/yz projections
    only ever had the fill, so turning fills off left them blank — and
    even with fills on, the metal boundary in those views was a soft alpha
    edge rather than a line. The outline is the part you cannot do
    without: it says exactly where the metal ends, which is what you need
    when reading a trajectory that grazes an electrode.
    """
    c = color.lstrip("#")
    r, g, b = (int(c[i:i+2], 16) for i in (0, 2, 4))
    z = np.asarray(img, float)
    if fill:
        a = float(alpha)
        fig.add_heatmap(
            x=x, y=y, z=z,
            colorscale=[[0.0, "rgba(0,0,0,0)"],
                        [1.0, f"rgba({r},{g},{b},{a})"]],
            zmin=0, zmax=1, showscale=False, showlegend=False,
            hoverinfo="skip")
    # outline: always, and it survives fills=off
    fig.add_contour(
        x=x, y=y, z=z,
        contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
        line=dict(color=f"rgb({max(r-60,0)},{max(g-60,0)},{max(b-60,0)})",
                  width=1.2),
        showscale=False, showlegend=False, hoverinfo="skip")


def _exposed_mesh(occ, cell_mm, origin=(0.0, 0.0, 0.0)):
    """Exposed boundary faces of a 3-D boolean occupancy -> (verts, tris).

    A voxel face is EXPOSED when the cell is occupied and its neighbor
    along that axis is not.  This is the surface of the raster itself —
    no smoothing, no re-derivation from the spec — so what renders is
    exactly the solver's occupancy, cell-accurate.  cell_mm is per-axis
    (hx, hy, hz), which lets a planar slab reuse this with hz = depth.
    Verts are duplicated per face (no dedup): simpler, and plotly Mesh3d
    is happy with it."""
    import numpy as np
    occ = np.asarray(occ, bool)
    hx, hy, hz = cell_mm
    o = np.asarray(origin, float)
    V = []
    F = []
    nv = 0
    # per (axis, side): the neighbor shift and the 4 corner offsets (in
    # cell units) of the exposed face, wound outward
    corners = {
        (0, -1): [(0,0,0),(0,1,0),(0,1,1),(0,0,1)],
        (0, +1): [(1,0,0),(1,0,1),(1,1,1),(1,1,0)],
        (1, -1): [(0,0,0),(0,0,1),(1,0,1),(1,0,0)],
        (1, +1): [(0,1,0),(1,1,0),(1,1,1),(0,1,1)],
        (2, -1): [(0,0,0),(1,0,0),(1,1,0),(0,1,0)],
        (2, +1): [(0,0,1),(0,1,1),(1,1,1),(1,0,1)],
    }
    pad = np.pad(occ, 1, constant_values=False)
    (slice(1, -1),) * 3
    for (ax, sd), cs in corners.items():
        sh = [slice(1, -1)] * 3
        sh[ax] = slice(2, None) if sd > 0 else slice(0, -2)
        exposed = occ & ~pad[tuple(sh)]
        idx = np.argwhere(exposed)
        if not len(idx):
            continue
        n = len(idx)
        for c in cs:
            V.append((idx + np.asarray(c)) * (hx, hy, hz) + o)
        base = nv + np.arange(n)
        F.append(np.stack([base, base + n, base + 2 * n], 1))
        F.append(np.stack([base, base + 2 * n, base + 3 * n], 1))
        nv += 4 * n
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    # interleave: V was appended corner-major per group; rebuild per-group
    verts = np.concatenate(V, axis=0)
    tris = np.concatenate(F, axis=0)
    return verts, tris


def _pool_any(occ, s):
    """Block-any pooling by stride s: presence-preserving decimation (a
    thin feature survives, unlike plain slicing)."""
    import numpy as np
    if s <= 1:
        return occ
    nx, ny, nz = occ.shape
    px, py, pz = (-nx) % s, (-ny) % s, (-nz) % s
    a = np.pad(occ, ((0, px), (0, py), (0, pz)), constant_values=False)
    a = a.reshape(a.shape[0]//s, s, a.shape[1]//s, s, a.shape[2]//s, s)
    return a.any(axis=(1, 3, 5))


def unfold_mirror_labels(lab, mirror):
    """Reflect a labeled voxel grid across each declared mirror axis so a
    HALF-domain scene (one board stored, the partner supplied by
    symmetry) renders as the FULL instrument. `mirror` is a subset of
    'xyz'. The stored half is the high side of the plane at index 0; the
    reflection prepends the flipped copy (dropping the shared plane) so
    the result is the physical whole. Display == solved geometry: the
    solver unfolds these same axes internally.

    A no-op when mirror is empty. Idempotent per axis (each axis unfolds
    once)."""
    import numpy as np
    lab = np.asarray(lab)
    axmap = {"x": 0, "y": 1, "z": 2}
    for ch in (mirror or ""):
        ax = axmap.get(ch)
        if ax is None:
            continue
        flip = np.flip(lab, axis=ax)
        sl = [slice(None)] * lab.ndim
        sl[ax] = slice(None, -1)        # drop the shared mirror plane
        lab = np.concatenate([flip[tuple(sl)], lab], axis=ax)
    return lab


def cad_preview(spec, *, height: int = 620, opacity: float = 0.55):
    """Interactive 3-D preview of a spec's imported STL meshes — the CAD
    truth, one colour per electrode, name on hover, aspectmode='data'.

    PROMOTED from sim_app's STL-upload preview (the 3-D CAD preview
    shows in the notebook too; interactive early and
    often) so notebooks and the app share one implementation. The
    uploader's in-place preview still draws from in-memory bytes
    pre-spec; unifying that call site onto this function is part of the
    known sim_app two-renderer debt, noted not done.

    This shows the MESH AS IMPORTED (CAD frame, mm). What the solver
    flies is the voxelized raster of it — voxel_views() shows that; the
    difference between the two IS the mesh-detail question.
    """
    import plotly.graph_objects as go
    from ion_gym.physics.build_stl import _require_trimesh
    from ion_gym.io.stl_resolve import resolve_stl_dir
    trimesh = _require_trimesh()
    stl_dir = resolve_stl_dir(spec)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
               "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    fig = go.Figure()
    n = 0
    for i, el in enumerate(spec.geometry.electrodes):
        if not getattr(el, "stl", None):
            continue
        mesh = trimesh.load(str(stl_dir / el.stl), force="mesh",
                            process=True)
        v, f = mesh.vertices, mesh.faces
        fig.add_mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=palette[i % len(palette)], opacity=opacity,
            name=el.name, showlegend=True, flatshading=True,
            hovertemplate=(f"<b>{el.name}</b><br>x %{{x:.2f}}  "
                           f"y %{{y:.2f}}  z %{{z:.2f}} mm<extra></extra>"))
        n += 1
    if n == 0:
        raise VizError(f"{spec.name!r}: no STL-bearing electrodes to "
                       "preview — cad_preview is for imported-mesh specs")
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.06),
        scene=dict(aspectmode="data", xaxis_title="x (mm)",
                   yaxis_title="y (mm)", zaxis_title="z (mm)"),
        title=dict(text=f"{spec.name} — CAD meshes as imported "
                        "(drag to orbit; hover for electrode)",
                   font=dict(size=12)))
    _mpl_release(fig)
    return fig


def voxel_views(labels, h_mm, *, names=None, voltages=None,
                title="voxel raster", origin=(0.0, 0.0, 0.0),
                max_voxels=150_000):
    """The SOLVER'S 3-D voxel raster, rendered — for any labeled grid.

    Seeing a 3-D raster of a loaded
    scene used to be impossible — the STL
    'Preview voxelization' draws the uploaded MESHES (a registration
    check), and nothing anywhere rendered the labeled voxel grid the
    solver actually consumes.  This is that renderer, and it is
    route-agnostic by construction: it takes ONLY (labels, h) — the int16
    occupancy any route produces (rasterize3d for scenes, voxelize_meshes
    for STLs, model.ele for a solved field array).  DISPLAY == SOLVER INPUT.

    Returns {"3d": Figure, "xy": Figure, "xz": Figure, "yz": Figure} —
    multi-axis by default (R1-R4).  The 3-D view is a per-label point
    cloud; above `max_voxels` occupied voxels it decimates by a uniform
    stride and SAYS SO in the title (a decimated picture that does not
    announce itself is a quiet lie).  Projections are exact (no
    decimation): per-label occupancy collapsed along the third axis via
    vox_heatmap.  Titles carry the pitch and grid — the operating point.
    Caller displays/saves; nothing is shown here."""
    import numpy as np
    import plotly.graph_objects as go
    lab = np.asarray(labels)
    if lab.ndim != 3:
        raise VizError(f"voxel_views needs a 3-D labeled grid, got "
                       f"{lab.ndim}-D {lab.shape} — for 2-D geometry the "
                       f"standard view already draws the solver labels")
    ids = [int(i) for i in np.unique(lab) if i != 0]
    if not ids:
        raise VizError("labeled grid is empty (all vacuum) — nothing to "
                       "raster")
    names = names or {}
    volts = voltages or {}
    pal = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    # SOLID exposed-face meshes, not a point cloud.
    # Face budget: pool the occupancy by block-ANY (presence-preserving)
    # until the exposed faces fit; the decimation is ANNOUNCED.
    n_occ = int((lab != 0).sum())
    o = np.asarray(origin, float)
    fig3 = go.Figure()
    max_faces = 4 * max_voxels
    total_faces = 0
    for k, i in enumerate(ids):
        occ = (lab == i)
        stride = 1
        while True:
            sub = _pool_any(occ, stride)
            # upper bound on exposed faces without building them
            est = 6 * int(sub.sum())
            if est <= max_faces // len(ids) or stride > 16:
                break
            stride += 1
        cell = (h_mm * stride,) * 3
        verts, tris = _exposed_mesh(sub, cell, origin=o)
        total_faces += len(tris)
        nm = names.get(i, f"electrode {i}")
        vtxt = f" @ {volts[i]:g} V" if i in volts else ""
        dtxt = f" [{stride}x pooled]" if stride > 1 else ""
        fig3.add_mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
            color=pal[k % len(pal)], opacity=1.0, flatshading=True,
            name=f"{nm}{vtxt}{dtxt}", showlegend=True)
    dec = f", {n_occ:,} voxels -> {total_faces:,} exposed faces"
    fig3.update_layout(
        title=f"{title} — solver voxels, h={h_mm:g} mm/gu, "
              f"grid {lab.shape[0]}x{lab.shape[1]}x{lab.shape[2]}{dec}",
        scene=dict(aspectmode="data", xaxis_title="x [mm]",
                   yaxis_title="y [mm]", zaxis_title="z [mm]"),
        margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h"))
    out = {"3d": fig3}
    ax_pairs = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
    for view, (a, b, c) in ax_pairs.items():
        f = go.Figure()
        for k, i in enumerate(ids):
            occ = np.moveaxis(lab == i, (a, b, c), (0, 1, 2)).any(axis=2)
            nx, ny = occ.shape
            x = o[a] + np.arange(nx) * h_mm
            y = o[b] + np.arange(ny) * h_mm
            vox_heatmap(f, x, y, occ.T.astype(float),
                        color=pal[k % len(pal)], alpha=0.8)
        la, lb = view[0], view[1]
        f.update_layout(title=f"{title} — {view} projection (exact, "
                              f"h={h_mm:g} mm)",
                        xaxis_title=f"{la} [mm]", yaxis_title=f"{lb} [mm]",
                        margin=dict(l=40, r=10, t=40, b=40))
        f.update_yaxes(scaleratio=1)
        out[view] = f
    return out


def voxel_views_2d_mpl(labels2d, h_mm, *, symmetry="rz", names=None,
                       voltages=None, title="voxel raster", n_theta=48,
                       max_pts=60_000):
    """Static (matplotlib) counterpart of voxel_views_2d for PNG export
    (views render as pngs -- the plotly
    path needs Chrome for static export, which offline environments lack).
    Same source of truth: the solver's own 2-D labels; the third dimension
    is the route's DECLARED symmetry. Returns {'3d','rz','end-on'} mpl
    Figures; caller saves. rz only (planar callers use render_mpl)."""
    if symmetry != "rz":
        raise VizError("voxel_views_2d_mpl: only symmetry='rz' is "
                       "implemented; planar scenes render via render_mpl")
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Wedge
    import numpy as np
    names, volts = names or {}, voltages or {}
    ids = [int(i) for i in np.unique(labels2d) if i != 0]
    cols = {i: plt.get_cmap("tab20")(k % 20) for k, i in enumerate(ids)}
    leg = [Patch(fc=cols[i],
                 label=names.get(i, f"electrode {i}")
                 + (f" \u00b7 {volts[i]:g} V" if i in volts else ""))
           for i in ids]
    nz, nr = labels2d.shape
    figs = {}
    # ---- rz: full mirrored cross-section, axial horizontal --------------
    img = np.concatenate([labels2d[:, ::-1], labels2d[:, 1:]], axis=1)
    fr, ax = plt.subplots(figsize=(11, 4))
    rgba = np.zeros(img.T.shape + (4,))
    for i in ids:
        rgba[img.T == i] = cols[i]
    ax.imshow(rgba, origin="lower", aspect="equal",
              extent=[0, nz*h_mm, -(nr-1)*h_mm, (nr-1)*h_mm])
    ax.set_xlabel("z (mm, axial)"); ax.set_ylabel("r (mm)")
    ax.set_title(f"{title} — r-z cross-section (solver labels, mirrored)",
                 fontsize=9)
    ax.legend(handles=leg, fontsize=6, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    fr.tight_layout(); figs["rz"] = fr
    # ---- end-on: exact concentric annuli (union over z) -----------------
    fe, ax = plt.subplots(figsize=(6, 6))
    occ = {}
    for i in ids:
        rows = np.where((labels2d == i).any(axis=0))[0]
        for j in rows:
            occ.setdefault(j, i)     # innermost id wins per r-bin
    for j in sorted(occ, reverse=True):
        ax.add_patch(Wedge((0, 0), (j+1)*h_mm, 0, 360, width=h_mm,
                           fc=cols[occ[j]], ec="none"))
    lim = nr*h_mm
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"{title} — end-on (union over z, exact annuli)", fontsize=9)
    ax.legend(handles=leg, fontsize=6, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    fe.tight_layout(); figs["end-on"] = fe
    # ---- 3d: transparent CUTAWAY of the revolved solid ------------------
    # (a point-cloud 3-D view reads as unhelpful dots;
    # electrodes render as revolved SURFACES, alpha-transparent, revolved
    # 270 deg so the cut plane exposes the bore, with the r-z cross-
    # section drawn solid on the cut faces.)
    f3 = plt.figure(figsize=(11, 6))
    ax = f3.add_subplot(projection="3d")
    th = np.linspace(0, 1.5*np.pi, n_theta)          # 270-degree cutaway
    for i in ids:
        mask = labels2d == i
        zi = np.where(mask.any(axis=1))[0]
        if not len(zi):
            continue
        zz = zi * h_mm
        r_in = np.array([np.argmax(mask[k]) for k in zi]) * h_mm
        r_out = np.array([nr - 1 - np.argmax(mask[k][::-1])
                          for k in zi]) * h_mm + h_mm
        for rr in (r_in, r_out):
            R, TH = np.meshgrid(rr, th, indexing="ij")
            Z = np.repeat(zz[:, None], n_theta, axis=1)
            ax.plot_surface(Z, R*np.cos(TH), R*np.sin(TH),
                            color=cols[i], alpha=0.45, linewidth=0,
                            antialiased=False, shade=True)
        # solid cross-section on the two cut faces
        for ang in (0.0, 1.5*np.pi):
            ax.plot_surface(
                np.vstack([zz, zz]).T,
                np.vstack([r_in, r_out]).T*np.cos(ang),
                np.vstack([r_in, r_out]).T*np.sin(ang),
                color=cols[i], alpha=0.9, linewidth=0, antialiased=False)
    ax.set_xlabel("z (mm, axial)"); ax.set_ylabel("x (mm)")
    ax.set_zlabel("y (mm)")
    ax.set_box_aspect((nz*h_mm, 2*nr*h_mm, 2*nr*h_mm))
    ax.view_init(elev=18, azim=-58)
    ax.set_title(f"{title} — 3-D cutaway (270\u00b0 revolve of the exact "
                 "r-z solid)", fontsize=9)
    ax.legend(handles=leg, fontsize=6, loc="center left",
              bbox_to_anchor=(1.05, 0.5))
    f3.tight_layout(); figs["3d"] = f3
    return figs


def voxel_views_2d(labels2d, h_mm, *, symmetry="rz", names=None,
                   voltages=None, title="voxel raster",
                   n_theta=64, depth_mm=None, max_voxels=150_000):
    """3-D raster for the 2-D routes (all geometries,
    including r-z").  No 3-D grid is materialized and nothing is
    re-rasterized: the 2-D int16 labels ARE the computation, and the
    third dimension is supplied by the route's DECLARED symmetry —
    revolution for 'rz', slab extrusion for 'planar'.  Every 2-D view is
    EXACT; the 3-D cloud is an azimuthal/depth SAMPLING of the exact
    solid and its title says so.

    symmetry='rz': labels are (n_axial, n_r), axis 0 = z (axial), axis 1
    = r >= 0.  Views: '3d' revolved cloud (theta samples scaled with r so
    large rings aren't sparse, globally capped at max_voxels); 'rz' the
    FULL cross-section — the solver labels mirrored about the axis, the
    same convention build_rz itself displays (axial horizontal); 'end-on'
    exact concentric annuli: a cell at radius rho is electrode i iff the
    r-bin floor(rho/h) is occupied by i at ANY z (union over z, exact to
    the grid).

    symmetry='planar': labels are (nx, ny); '3d' is a display-only slab
    extrusion of depth `depth_mm` (default 15% of the larger extent —
    NAMED default, announced in the title; the solve assumes
    translational symmetry, so depth is presentational, not physical);
    'xy' is the exact solver plane.

    Returns {view: Figure}.  Caller displays/saves."""
    import numpy as np
    import plotly.graph_objects as go
    lab = np.asarray(labels2d)
    if lab.ndim != 2:
        raise VizError(f"voxel_views_2d needs a 2-D labeled grid, got "
                       f"{lab.ndim}-D {lab.shape} — use voxel_views for "
                       f"native 3-D grids")
    if symmetry not in ("rz", "planar"):
        raise VizError(f"unknown symmetry {symmetry!r}: 'rz' | 'planar'")
    ids = [int(i) for i in np.unique(lab) if i != 0]
    if not ids:
        raise VizError("labeled grid is empty (all vacuum)")
    names = names or {}
    volts = voltages or {}
    pal = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    def _legend(i):
        nm = names.get(i, f"electrode {i}")
        return nm + (f" @ {volts[i]:g} V" if i in volts else "")

    out = {}
    fig3 = go.Figure()
    if symmetry == "rz":
        nz, nr = lab.shape
        # SOLID SURFACES, not a point cloud (the cloud
        # aliased into combs and never read as a surface).  Per electrode,
        # each z-column's occupied r-RUNS give inner/outer radius
        # envelopes taken EXACTLY from the solver labels; revolving those
        # envelopes parametrically renders the true solid of revolution.
        # NaN cells break a surface where its run is absent, and end
        # annuli close a run band where it starts/stops in z.
        th = np.linspace(0.0, 2.0 * np.pi, max(16, n_theta) + 1)
        cth, sth = np.cos(th), np.sin(th)

        def _runs(col):
            """occupied r-runs of one z-column -> [(r_lo, r_hi)] bins."""
            w = np.where(col)[0]
            if not len(w):
                return []
            brk = np.where(np.diff(w) > 1)[0]
            lo = np.concatenate([[w[0]], w[brk + 1]])
            hi = np.concatenate([w[brk], [w[-1]]])
            return list(zip(lo, hi))

        def _surf(fig, z_arr, r_of_z, color, name, legend):
            """One revolved envelope r(z) (NaN = absent) as a Surface."""
            Zg = np.repeat(z_arr[:, None], len(th), axis=1)
            Rg = np.repeat(r_of_z[:, None], len(th), axis=1)
            fig.add_surface(
                x=Zg, y=Rg * cth[None, :], z=Rg * sth[None, :],
                surfacecolor=np.zeros_like(Zg), showscale=False,
                colorscale=[[0, color], [1, color]], opacity=1.0,
                name=name, showlegend=legend, hoverinfo="name")

        def _annulus(fig, z0, r_lo, r_hi, color):
            Rg = np.array([[r_lo] * len(th), [r_hi] * len(th)])
            fig.add_surface(
                x=np.full_like(Rg, z0), y=Rg * cth[None, :],
                z=Rg * sth[None, :], surfacecolor=np.zeros_like(Rg),
                showscale=False, colorscale=[[0, color], [1, color]],
                opacity=1.0, hoverinfo="skip", showlegend=False)

        z_arr = np.arange(nz) * h_mm
        for k, i in enumerate(ids):
            col = pal[k % len(pal)]
            mask = (lab == i)
            runs_by_z = [_runs(mask[z]) for z in range(nz)]
            kmax = max((len(r) for r in runs_by_z), default=0)
            first = True
            for band in range(kmax):
                r_in = np.full(nz, np.nan)
                r_out = np.full(nz, np.nan)
                for z in range(nz):
                    if band < len(runs_by_z[z]):
                        lo, hi = runs_by_z[z][band]
                        # half-cell envelopes so touching parts touch
                        r_in[z] = max(lo - 0.5, 0.0) * h_mm
                        r_out[z] = (hi + 0.5) * h_mm
                _surf(fig3, z_arr, r_out, col, _legend(i), first)
                first = False
                if np.nanmax(np.nan_to_num(r_in)) > 0:
                    _surf(fig3, z_arr, r_in, col, _legend(i), False)
                have = ~np.isnan(r_out)
                for z in range(nz):
                    if have[z] and (z == 0 or not have[z - 1]):
                        _annulus(fig3, z_arr[z], np.nan_to_num(r_in[z]),
                                 r_out[z], col)
                    if have[z] and (z == nz - 1 or not have[z + 1]):
                        _annulus(fig3, z_arr[z], np.nan_to_num(r_in[z]),
                                 r_out[z], col)
        fig3.update_layout(
            title=f"{title} — r-z solve REVOLVED (declared symmetry), "
                  f"h={h_mm:g} mm/gu, grid {nz}x{nr}; surfaces are the "
                  f"exact run envelopes of the solver labels",
            scene=dict(aspectmode="data", xaxis_title="z axial [mm]",
                       yaxis_title="x [mm]", zaxis_title="y [mm]"),
            margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h"))
        out["3d"] = fig3
        # exact FULL cross-section: mirror about the axis (build_rz's own
        # display convention), axial horizontal
        full = np.concatenate([lab[:, :0:-1], lab], axis=1)
        r_lo = -(nr - 1) * h_mm
        f = go.Figure()
        for k, i in enumerate(ids):
            vox_heatmap(f, np.arange(nz) * h_mm,
                        r_lo + np.arange(full.shape[1]) * h_mm,
                        (full == i).T.astype(float),
                        color=pal[k % len(pal)], alpha=0.8)
        f.update_layout(title=f"{title} — full r-z cross-section (exact "
                              f"solver labels, mirrored about the axis)",
                        xaxis_title="z axial [mm]", yaxis_title="r [mm]",
                        margin=dict(l=40, r=10, t=40, b=40))
        out["rz"] = f
        # exact end-on annuli: radius-bin occupancy, union over z
        n = 2 * nr - 1
        cy = cx = (nr - 1)
        yy, xx = np.mgrid[0:n, 0:n]
        rho_bin = np.rint(np.hypot(xx - cx, yy - cy)).astype(int)
        f2 = go.Figure()
        ax_mm = (np.arange(n) - cx) * h_mm
        for k, i in enumerate(ids):
            rbins = np.zeros(nr + 2, bool)
            rbins[:nr][np.any(lab == i, axis=0)] = True
            img = rbins[np.clip(rho_bin, 0, nr + 1)]
            vox_heatmap(f2, ax_mm, ax_mm, img.astype(float),
                        color=pal[k % len(pal)], alpha=0.8)
        f2.update_layout(title=f"{title} — end-on (exact annuli: radial "
                               f"occupancy, union over z)",
                         xaxis_title="x [mm]", yaxis_title="y [mm]",
                         margin=dict(l=40, r=10, t=40, b=40))
        f2.update_yaxes(scaleratio=1)
        out["end-on"] = f2
    else:                                            # planar slab
        nx, ny = lab.shape
        depth = depth_mm if depth_mm is not None else \
            0.15 * max(nx, ny) * h_mm
        # SOLID extruded meshes via the shared exposed-face mesher on a
        # one-layer grid with hz = depth (no clouds).
        for k, i in enumerate(ids):
            occ3 = (lab == i)[:, :, None]
            verts, tris = _exposed_mesh(occ3, (h_mm, h_mm, depth))
            fig3.add_mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                color=pal[k % len(pal)], opacity=1.0, flatshading=True,
                name=_legend(i), showlegend=True)
        fig3.update_layout(
            title=f"{title} — planar solve EXTRUDED for display "
                  f"(translational symmetry; depth {depth:g} mm is "
                  f"presentational), h={h_mm:g} mm/gu",
            scene=dict(aspectmode="data", xaxis_title="x [mm]",
                       yaxis_title="y [mm]", zaxis_title="slab [mm]"),
            margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h"))
        out["3d"] = fig3
        f = go.Figure()
        for k, i in enumerate(ids):
            vox_heatmap(f, np.arange(nx) * h_mm, np.arange(ny) * h_mm,
                        (lab == i).T.astype(float),
                        color=pal[k % len(pal)], alpha=0.8)
        f.update_layout(title=f"{title} — xy (exact solver labels)",
                        xaxis_title="x [mm]", yaxis_title="y [mm]",
                        margin=dict(l=40, r=10, t=40, b=40))
        out["xy"] = f
    return out


def phi_slice_3d(model, axis, index=None):
    """(xc, yc, sl): the POTENTIAL on an xz/yz slice of a 3-D model —
    RF at peak phase, node-centred, canonical frame. ONE authority for
    both the transport-view shading (field_slice_3d's phi path) and the
    per-view equipotential contours (lines and
    shading must agree about which cut is shown, so they read the same
    arrays). Returns (None, None, None) for non-3-D models."""
    import numpy as np
    A = getattr(model, "A", None)
    h = (getattr(model, "mm_per_gu", None)
         or getattr(model, "h_mm", None))
    if A is None or getattr(A, "ndim", 0) != 3 or not h:
        return None, None, None
    # WORLD frame, not stored (contours were
    # "rotated 90 but still offset" — by exactly the deck's declared
    # origin_mm, +7.8 mm). world_off_mm composes mirror + declared
    # origin (the convention field_surface/pe_surface already
    # follow); mirror_off_mm alone leaves a signed-origin 3-D deck's
    # slice in the STORED frame while the electrodes draw in world —
    # the exact defect class already fixed for the slice slider, inherited
    # here verbatim from the old shading code.
    moff = (getattr(model, "world_off_mm", None)
            or getattr(model, "mirror_off_mm", (0.0, 0.0, 0.0)))
    phi = A
    rf_V = getattr(model, "rf_V", 0.0)
    B = getattr(model, "B", None)
    if rf_V and B is not None:
        phi = A + rf_V * B                    # RF peak snapshot
    nx, ny, nz = phi.shape
    nrm = ny if axis == "x" else nx           # suppressed axis length
    k = (nrm // 2 if index is None
         else max(0, min(nrm - 1, int(index))))
    # NODE-centred coordinates (canonical frame). These were cell-
    # centred ((arange+0.5)*h), a half-cell registration error left
    # over from the retired sampling convention — the metal and the
    # shading were drawn half a cell apart.
    if axis == "x":                           # xz: suppress y
        sl = phi[:, k, :]
        xc = np.arange(nx) * h + moff[0]
    else:                                     # yz: suppress x
        sl = phi[k, :, :]
        xc = np.arange(ny) * h + moff[1]
    yc = np.arange(nz) * h + moff[2]
    return xc, yc, sl


def field_slice_3d(fig, model, axis, index=None,
                   quantity="phi", mz=None, charge=1):
    """For a genuinely 3-D model, shade the potential on the MID-PLANE
    slice of the suppressed transverse axis in the xz/yz views (RF at
    peak phase for RF devices, like the xy view). 2-D planar models are
    z-invariant — their transport views keep the pointer to xy.

    Coordinates are emitted in the model's CANONICAL frame: mirrored axes
    read [-H,+H] with the mirror plane at 0 (model.mirror_off_mm, zero on
    non-mirrored axes — unmirrored output byte-identical). On the unfolded
    grid the suppressed-axis midpoint IS the mirror plane, so the slice
    shown is the symmetry plane itself."""
    A = getattr(model, "A", None)
    h = (getattr(model, "mm_per_gu", None)
         or getattr(model, "h_mm", None))
    if A is None or getattr(A, "ndim", 0) != 3 or not h:
        return
    plane = "xz" if axis == "x" else "yz"
    # |E| and PE come from the model's OWN plane/slice methods — the same
    # ones the PE tab uses — so the transport views can shade the field the
    # shading selector actually asks for, at ANY slice, instead of always
    # the potential at the mid-plane (a 2-D
    # slice into the apparatus, in each plane, that shows the field).
    if quantity == "efield" and hasattr(model, "field_surface"):
        xc, yc, sl, _e = model.field_surface(plane=plane, index=index)
        cscale, mid = "Viridis", None
    elif quantity == "pe" and hasattr(model, "pe_surface"):
        xc, yc, sl, _e = model.pe_surface(mz=mz, charge=charge, plane=plane,
                                          index=index)
        cscale, mid = "Viridis", None
    else:
        xc, yc, sl = phi_slice_3d(model, axis, index)
        cscale, mid = "RdBu", None
    # VIEW MAPPING (fields could read as rotated by
    # 90 degrees): every surface method returns MODEL-frame (a, b, img) in
    # axis order — a is the transverse/lower axis, b is z — while
    # Convention A draws the AXIAL coordinate HORIZONTAL in the xz/yz
    # views. This site plotted (x=a, y=b), which rotated every 3-D
    # transport-view overlay (phi, |E| and PE alike) by 90° since the
    # convention landed; the contour path exposed it. Horizontal = b
    # (z), vertical = a; img is (n_a, n_b) = (len(y), len(x)), so no
    # transpose. _view_contours applies the same mapping.
    fig.add_heatmap(x=yc, y=xc, z=sl, colorscale=cscale, zmid=mid,
                    opacity=0.5, showscale=False, hoverinfo="skip")


def plane_of(la, lb):
    """The view plane from its two axis labels, ORDER-INDEPENDENT.
    Convention A (v144) made the axial coordinate horizontal, so "xz"
    hands you la="z", lb="x" — every branch that tested `la == "x"`
    to mean "the xz view" (or "the axial coordinate") silently picked
    the wrong picture after that change (funnel rings inside the xz
    view; quad rods running up the wrong axis).
    The plane is a property of WHICH axes are shown, never their order.
    """
    s = frozenset((la, lb))
    try:
        return {frozenset("xy"): "xy", frozenset("xz"): "xz",
                frozenset("yz"): "yz"}[s]
    except KeyError:
        raise VizError(f"not a principal plane: axes {la!r}/{lb!r}")


# Above this many projected cells, a per-electrode heatmap fill ships an
# unusable JSON payload to the browser (the SLIM's 11 electrodes at
# 741x261 = >2M z-values per view wedged the tab). Rect
# decomposition below is EXACT — the same cells, as maximal rectangles —
# so display==solver-input holds at ~1/1000th the payload.
FILL_HEATMAP_MAX_CELLS = 40_000


def _bool_rects(img):
    """Maximal-rectangle decomposition of a boolean image: per-column runs
    along axis 0, merged across consecutive identical columns. EXACT
    cover of the True cells. Returns [(i0, i1, j0, j1)] inclusive."""
    rects = []
    ny = img.shape[1]
    prev_runs, j_start = None, 0
    def runs_of(col):
        out, i = [], 0
        n = len(col)
        while i < n:
            if col[i]:
                k = i
                while k + 1 < n and col[k + 1]:
                    k += 1
                out.append((i, k))
                i = k + 1
            else:
                i += 1
        return out
    for j in range(ny + 1):
        cur = runs_of(img[:, j]) if j < ny else None
        if cur != prev_runs:
            if prev_runs:
                for (i0, i1) in prev_runs:
                    rects.append((i0, i1, j_start, j - 1))
            prev_runs, j_start = cur, j
    return rects


def el_mask_fills(fig, masks, h, coords, la, lb, *, alpha, fill, label,
                  palette=None, axial_extent=None, origin=(0.0, 0.0, 0.0),
                  mirror=None):
    """Per-electrode display straight from the imported/solve masks, each
    electrode in its own colour and LABELED with its number (matching the
    Voltages tab), so voltages can be assigned unambiguously. Handles 2-D
    (planar / r-z) and 3-D masks in every view plane. Returns True if it
    drew anything.

    `axial_extent` (mm lo, hi): a 2-D mask has no axial
    structure, but the MODEL that produced it may still know how far its
    bodies extend along z (an STL solve knows its mesh z-bounds). When
    given, the transport-view broadcast draws BOUNDED rectangles over that
    span instead of infinite bands -- the "rods extend to infinity"
    disease, fixed at the ONE renderer for every configuration. Masks with
    real 3-D structure ignore it: their projections are finite by
    construction.

    `mirror` (dict {display-axis: mode}):
    the masks are stored FOLDED along the named display axis (half the
    array); the full array is the even unfold concat([flip, self]). The
    renderer unfolds transiently and ONLY when that axis is PRESERVED in
    the view -- a projection that suppresses the folded axis is identical
    folded or unfolded, so the half is projected directly and no memory
    is spent. `origin` describes the full (unfolded) array, so placement
    is unchanged. Generic: any builder storing a declared-symmetric mask
    gets half the RAM through this one path, no builder-specific code."""
    import numpy as np
    palette = EL_PALETTE if palette is None else palette
    mirror = mirror or {}
    show_fill = bool(fill)
    show_lab = bool(label)
    a = float(alpha) if show_fill else 0.0
    drew = False

    def _blob(img, xs, ys, col):
        """Draw ONE projected/sliced electrode: fill if asked, OUTLINE
        ALWAYS.

        There were three separate places drawing electrodes (xy slice, the
        _vox_heatmap projections, and this per-electrode loop) and only the
        first had an outline. So 'turn fills off, keep outlines' worked in
        xy and blanked xz/yz -- I fixed _vox_heatmap and this path still
        drew a bare heatmap at alpha 0. One renderer now; the outline is
        not optional, because it is the thing that says where the metal
        ENDS, which is what you need when reading a grazing trajectory."""
        r_, g_, b_ = (int(col[k:k + 2], 16) for k in (1, 3, 5))
        # showlegend=False: electrodes are the SET, not a data series.
        # They were adding 35 entries to the legend -- and still adding
        # them with fills OFF, since a legend entry is a property of the
        # trace, not of whether you can see it. The fates are the legend.
        if show_fill:
            fig.add_heatmap(
                x=xs, y=ys, z=img.T.astype(float),
                colorscale=[[0, "rgba(0,0,0,0)"],
                            [1, f"rgba({r_},{g_},{b_},{a})"]],
                zmin=0, zmax=1, showscale=False, name=lab,
                showlegend=False,
                hovertemplate=lab + "<extra></extra>")
        # OUTLINE via mask_outline_mm (on a dense SLIM
        # screenshot): a plotly 0.5-contour of a boundary-flush mask is an
        # OPEN line — electrodes touching the domain edge drew "clipped".
        # The vacuum-padded tracer closes every loop, and the boundary
        # edge lands at the half-cell the flush node owns (display ==
        # solver registration; same fix as the mpl paths).
        _px, _py = [], []
        for _lp in mask_outline_mm(img, h,
                                   (float(xs[0]), float(ys[0]))):
            _px += list(_lp[:, 0]) + [None]
            _py += list(_lp[:, 1]) + [None]
        fig.add_scatter(
            x=_px, y=_py, mode="lines",
            line=dict(color=f"rgb({r_},{g_},{b_})", width=1.3),
            showlegend=False, hoverinfo="skip", name=lab)
    # numeric keys sort numerically (e1..e4, e10000 LAST — the str-sort
    # regression put the border basis between e1 and e2 and shuffled every
    # electrode's colour); named keys sort lexically after numbers.
    for i, n in enumerate(sorted(
            masks, key=lambda k: (isinstance(k, str), k))):
        m = masks[n]
        c = palette[i % len(palette)]
        # named masks (str keys, e.g. "tw1"/"rf1"/"gnd") keep their names;
        # numbered masks keep the eN convention matching the Voltages tab
        lab = n if isinstance(n, str) else f"e{n}"
        if m.ndim == 3 and m.shape[2] > 1:
            # 3-D: project onto the view plane, DERIVED FROM THE AXIS NAMES.
            #
            # This used to branch on `la == "x"` to pick which array axis
            # to collapse. When the xz view started reporting (z, x) --
            # axial-horizontal -- `la` became "z", so xz fell through to
            # the yz branch: BOTH views rendered the yz projection, drawn
            # under xz's labels. That is the transposed picture (a 140 mm
            # ladder running up the 15 mm axis).
            #
            # The projection is a property of WHICH AXES the view shows,
            # not of the order they happen to be listed in. Derive it.
            AX = {"x": 0, "y": 1, "z": 2}
            drop = ({"x", "y", "z"} - {la, lb}).pop()
            # MIRROR UNFOLD (Opus task 3): a mask stored folded along a
            # display axis is unfolded here ONLY if that axis survives the
            # projection. If the folded axis is the one being dropped, the
            # union over it is identical folded or unfolded, so we skip the
            # unfold entirely (the transient array is never built). The
            # unfold is the even reflection concat([flip, self]) that the
            # declared origin already accounts for.
            m_src = m
            for _ax, _mode in mirror.items():
                if _ax != drop and _ax in AX:
                    _ai = AX[_ax]
                    m_src = np.concatenate(
                        [np.flip(m_src, axis=_ai), m_src], axis=_ai)
            img = m_src.any(axis=AX[drop])      # remaining axes, ascending
            kept = sorted(({"x", "y", "z"} - {drop}), key=AX.get)
            if (kept[0], kept[1]) != (la, lb):  # view wants them the other
                img = img.T                     # way round -> transpose
            if not img.any():
                continue
            # KERNEL FRAME: the fly
            # kernel samples node i at i*h (gx = px/mm), and em/
            # equipotential contours draw in that frame via extent().
            # These fills used (i+0.5)*h — every fill and colored edge
            # sat half a cell high/right of the field truth. The BLACK
            # em contour was the solver; the fills were the same mask
            # drawn in the wrong frame.
            # `origin`: masks may live in a frame offset from
            # the plot origin (a mirror-unfolded gap axis centred on the
            # physical mid-plane, an STL build-frame offset). Generic, per
            # axis — the same concept off_mm already encodes on models.
            # after the transpose above, axis 0 IS la and axis 1 IS lb —
            # index the origin by the PLOT axes, not the pre-transpose
            # `kept` order
            xs = np.arange(img.shape[0]) * h + float(origin[AX[la]])
            ys = np.arange(img.shape[1]) * h + float(origin[AX[lb]])
            if img.size > FILL_HEATMAP_MAX_CELLS:
                # EXACT rect cover instead of a browser-wedging heatmap
                r_, g_, b_ = (int(c[k:k + 2], 16) for k in (1, 3, 5))
                px, py = [], []
                for (i0, i1, j0, j1) in _bool_rects(img):
                    x0, x1 = xs[i0] - h / 2, xs[i1] + h / 2
                    y0, y1 = ys[j0] - h / 2, ys[j1] + h / 2
                    px += [x0, x1, x1, x0, x0, None]
                    py += [y0, y0, y1, y1, y0, None]
                fig.add_scatter(
                    x=px, y=py, mode="lines", fill="toself",
                    fillcolor=f"rgba({r_},{g_},{b_},{a})",
                    line=dict(color=f"rgb({r_},{g_},{b_})", width=1.0),
                    name=lab, showlegend=False,
                    hovertemplate=lab + "<extra></extra>")
            else:
                _blob(img, xs, ys, c)
            if show_lab:
                # centroid of the LARGEST connected body:
                # the mean over ALL pixels put every multi-body
                # electrode's label at the plot centre, stacked and
                # unreadable on dense decks — same defect the
                # mpl path had.
                try:
                    from scipy.ndimage import label as _cclabel
                    _lbl, _n = _cclabel(img)
                    if _n > 1:
                        _big = int(np.argmax(np.bincount(
                            _lbl.ravel())[1:])) + 1
                        _use = _lbl == _big
                    else:
                        _use = img
                except ImportError:
                    # scipy optional: fall back to the all-pixel mean
                    # (single-body electrodes are unaffected either way)
                    _use = img
                ii, jj = np.where(_use)
                fig.add_annotation(
                    x=float(xs[ii].mean()), y=float(ys[jj].mean()),
                    text=f"<b>{lab}</b>", showarrow=False,
                    font=dict(color=c, size=13))
            drew = True
            continue
        m2 = m[:, :, 0] if m.ndim == 3 else m
        if not m2.any():
            continue
        if frozenset((la, lb)) == frozenset("xy"):
            # cross-section / r-z field plane: (axial|x, r|y) — set form,
            # order-independent (see plane_of)
            # KERNEL frame + the caller's declared origin:
            # this branch ignored `origin` entirely while the transposed
            # branch above honoured it, so a signed-frame deck drew its
            # electrode masks a whole origin away from its own shapes --
            # the ghost ladder in the UI View tab. Measured before the
            # fix: masks started at x = -0.23 mm against shapes at
            # -226.39 mm. origin defaults to (0,0,0), so every legacy
            # deck is unchanged. r-z keeps its own mirroring below; only
            # the offset is added here.
            _AXI = {"x": 0, "y": 1, "z": 2}
            xs = np.arange(m2.shape[0]) * h + float(origin[_AXI[la]])
            ys = np.arange(m2.shape[1]) * h + float(origin[_AXI[lb]])
            if coords == "rz":
                # mirror across the axis like potential_image does
                ys_f = np.concatenate([-ys[::-1], ys[1:]])
                img = np.concatenate([m2[:, ::-1], m2[:, 1:]], axis=1)
            else:
                ys_f, img = ys, m2
            _blob(img, xs, ys_f, c)
            if show_lab:
                ii, jj = np.where(m2)
                fig.add_annotation(
                    x=float(xs[ii].mean()), y=float(ys[jj].mean()),
                    text=f"<b>{lab}</b>", showarrow=False,
                    font=dict(color=c, size=13))
            drew = True
        elif coords == "rz" and frozenset((la, lb)) == frozenset("yz"):
            # end-on rings: this electrode's own radial annulus (yz view
            # of an axisymmetric body, whichever order the labels came)
            rows = m2.any(axis=0)
            rs = np.arange(m2.shape[1]) * h        # kernel frame
            r_in = float(rs[rows].min() - h / 2)
            r_out = float(rs[rows].max() + h / 2)
            for r0 in (max(r_in, 0.0), r_out):
                if r0 <= 0:
                    continue
                fig.add_shape(type="circle", x0=-r0, y0=-r0, x1=r0, y1=r0,
                              layer="below", line=dict(color=c,  # was `rgba`: a NameError
                                                       # latent on this
                                                       # branch (Phase B
                                                       # found it)
                                                       width=1.6))
            ang = -np.pi / 2 + i * (2 * np.pi / max(len(masks), 1))
            fig.add_annotation(x=float(r_out * np.cos(ang)),
                               y=float(r_out * np.sin(ang)),
                               text=f"<b>{lab}</b>", showarrow=False,
                               font=dict(color=c, size=13))
            drew = True
        else:
            # transport view of a 2-D mask: per-electrode transverse runs.
            # DERIVED, never assumed (same doctrine as the 3-D projection
            # branch above): the transverse coordinate is whichever shown
            # label is not "z"; its mask axis is 0 for x, 1 for y; and it
            # is drawn on WHICHEVER PLOT AXIS carries its label — under
            # Convention A (z horizontal) that is the VERTICAL axis, so
            # the bands are hrects. `axis0 = 0 if la == "x" else 1` +
            # unconditional vrects put the quad's rods on the z axis
            # spanning all x.
            trans = ({la, lb} - {"z"})
            if len(trans) != 1:
                raise VizError(f"transport branch with axes {la!r}/{lb!r}"
                               f" — no single transverse coordinate")
            trans = trans.pop()
            axis0 = 0 if trans == "x" else 1
            prof = m2.any(axis=1 - axis0)
            # kernel frame + the caller's frame origin (signed frames:
            # transport bands drew transverse runs at i*h, ignoring the
            # origin the caller composed -- signed-frame decks banded at
            # [0, extent] while everything else sat at [lo, lo+extent])
            _t0 = origin[axis0]
            cs = np.arange(len(prof)) * h + _t0
            vertical = (lb == trans)   # transverse on the plot's y axis
            j = 0
            while j < len(prof):
                if prof[j]:
                    k = j
                    while k + 1 < len(prof) and prof[k + 1]:
                        k += 1
                    if vertical:
                        if axial_extent is not None:
                            # bounded: the model declared its bodies' axial
                            # span -- draw a finite rectangle, not an
                            # infinite band
                            fig.add_shape(type="rect",
                                          x0=float(axial_extent[0]),
                                          x1=float(axial_extent[1]),
                                          y0=cs[j] - h/2, y1=cs[k] + h/2,
                                          fillcolor=c, opacity=a,
                                          line_width=0, layer="below")
                        else:
                            fig.add_hrect(y0=cs[j] - h/2, y1=cs[k] + h/2,
                                          fillcolor=c, opacity=a,
                                          line_width=0, layer="below")
                        fig.add_annotation(y=float((cs[j] + cs[k]) / 2),
                                           xref="paper", x=0.03 + 0.05 * i,
                                           text=f"<b>{lab}</b>",
                                           showarrow=False,
                                           font=dict(color=c, size=12))
                    else:
                        if axial_extent is not None:
                            fig.add_shape(type="rect",
                                          y0=float(axial_extent[0]),
                                          y1=float(axial_extent[1]),
                                          x0=cs[j] - h/2, x1=cs[k] + h/2,
                                          fillcolor=c, opacity=a,
                                          line_width=0, layer="below")
                        else:
                            fig.add_vrect(x0=cs[j] - h/2, x1=cs[k] + h/2,
                                          fillcolor=c, opacity=a,
                                          line_width=0, layer="below")
                        fig.add_annotation(x=float((cs[j] + cs[k]) / 2),
                                           yref="paper", y=0.97 - 0.05 * i,
                                           text=f"<b>{lab}</b>",
                                           showarrow=False,
                                           font=dict(color=c, size=12))
                    j = k + 1
                else:
                    j += 1
            drew = True
    if drew and coords == "rz" and frozenset((la, lb)) == frozenset("yz"):
        rmax = max((np.arange(masks[n].shape[-1]) * h).max()
                   for n in masks)
        pad = 0.1 * rmax
        fig.update_xaxes(range=[-rmax - pad, rmax + pad])
        fig.update_yaxes(range=[-rmax - pad, rmax + pad])
        # M-VIZ-Z: the ranges are already square, so a square PANEL makes
        # the rings round without anchoring the axes (which would freeze
        # the zoom aspect).  viz_core owns the policy.
        apply_zoom_policy(
            fig, (-rmax - pad, rmax + pad, -rmax - pad, rmax + pad),
            true_scale=True, px_width=560, x_title=None, y_title=None)
    return drew


def transport_bands(fig, model, geo_coords, axis, la=None, lb=None, *,
                    color, alpha, fill, label, palette=None):
    """Electrode geometry in a transport (xz/yz) view.

    la/lb are the view's ACTUAL horizontal/vertical axis names. They used
    to be assumed (transverse across, z up) and hard-coded as (axis, "z").
    For 3-D geometry the view is now axial-horizontal — (z, x) — so the
    assumption drew the geometry transposed: a 140 mm ladder running up
    the 15 mm axis. Never assume an axis you were handed.

    GENERALIZED: a SLIM deck got z-resolved fills and the
    quad did not, because that capability rode a builder-specific `vox2d`
    side channel).  el_mask_fills has ALWAYS known how to project 3-D
    masks per view, per electrode, labeled; the vox2d branch was a
    monochrome parallel copy of it (K4) and is REMOVED.  The renderer now
    measures the bodies it is given (D2), for ANY builder:

      * `el_masks3d` -- named 3-D masks a model declares when its
        el_masks must stay 2-D for the cross-section view (the SLIM's xy
        field is SOLVED on the 2-D confinement masks; its transport
        geometry is the imported voxel set).  Pitch may differ from the
        display model's (`el_masks3d_h_mm`).
      * `el_masks` with 3-D values (the 3-D STL / scene3d path) -- same
        projection, same call.
      * `el_masks` with 2-D values -- transverse broadcast, CLIPPED to
        the model's declared `z_extent_mm` when it has one: a 2-D solve
        of finite bodies knows its axial extent even though its field
        does not.  No more rods to infinity."""
    if la is None or lb is None:            # legacy callers
        la, lb = axis, "z"
    hh = (getattr(model, "mm_per_gu", None)
          or getattr(model, "h_mm", None))
    m3 = getattr(model, "el_masks3d", None)
    if m3 and hh:
        h3 = getattr(model, "el_masks3d_h_mm", None) or hh
        o3 = getattr(model, "el_masks3d_origin_mm", (0.0, 0.0, 0.0))
        mir3 = getattr(model, "el_masks3d_mirror", None)
        if el_mask_fills(fig, m3, h3, geo_coords, la, lb, alpha=alpha,
                         fill=fill, label=label, palette=palette,
                         origin=o3, mirror=mir3):
            return
    elm = getattr(model, "el_masks", None)
    if elm and hh:
        # SIGNED FRAME: planar models carry the deck
        # frame origin in anchor_mm; drawing bodies without it put metal
        # at [0, extent] while rays/records flew at [lo, lo+extent] --
        # the deck appeared twice/offset in the UI View tab. Compose the
        # anchor exactly as scene_from_simspec does; legacy anchors are
        # (0, 0), byte-identical.
        _mo2 = getattr(model, "mirror_off_mm", (0.0, 0.0, 0.0))
        _anc = getattr(model, "anchor_mm", (0.0, 0.0))
        if el_mask_fills(fig, elm, hh, geo_coords, la, lb, alpha=alpha,
                         fill=fill, label=label, palette=palette,
                         axial_extent=getattr(model, "z_extent_mm", None),
                         origin=(_mo2[0] + _anc[0], _mo2[1] + _anc[1],
                                 _mo2[2])):
            return
    # Generic fallback: PROJECT the actual solve mask (model.ele — the
    # voxelized metal the field was computed on, whether from spec
    # shapes or uploaded STLs) over the suppressed transverse axis.
    # display == solve input; no bounding boxes, no straddle heuristics.
    ele = getattr(model, "ele", None)
    h = (getattr(model, "mm_per_gu", None)
         or getattr(model, "h_mm", None))
    if ele is None or h is None:
        return
    if ele.ndim == 3:
        # full 3-D mask: project to the transport plane and voxel-fill
        nx, ny, nz = ele.shape
        _mo = getattr(model, "mirror_off_mm", (0.0, 0.0, 0.0))
        _an = getattr(model, "anchor_mm", (0.0, 0.0))
        _mo = (_mo[0] + _an[0], _mo[1] + _an[1], _mo[2])
        # NODE-centred: node i sits AT i*h (the kernel registration
        # every other coordinate build here uses). The (i+0.5)*h form was
        # the retired cell-centred convention and drew this projected metal
        # half a cell off its own field.
        if axis == "x":              # xz: plot-x = x, plot-y = z
            img = ele.any(axis=1).T
            xc = np.arange(nx) * h + _mo[0]
        else:                        # yz: plot-x = y, plot-y = z
            img = ele.any(axis=0).T
            xc = np.arange(ny) * h + _mo[1]
        yc = np.arange(nz) * h + _mo[2]
        vox_heatmap(fig, xc, yc, img, color=color, alpha=alpha, fill=fill)
        return
    # 2-D slice mask (z-invariant geometry): contiguous transverse runs
    # extruded along the transport axis — exactly the slice model's
    # z-invariance made visible.
    prof = ele.any(axis=1) if axis == "x" else ele.any(axis=0)
    _an2 = getattr(model, "anchor_mm", (0.0, 0.0))
    _a0 = _an2[0] if axis == "x" else _an2[1]
    coords = np.arange(len(prof)) * h + _a0  # node-centred + frame origin
    c = color.lstrip("#")
    r, g, b = (int(c[i:i+2], 16) for i in (0, 2, 4))
    fill_on = fill
    a = float(alpha) if fill_on else 0.0
    edge = f"rgb({max(r-60,0)},{max(g-60,0)},{max(b-60,0)})"
    i = 0
    while i < len(prof):
        if prof[i]:
            j = i
            while j + 1 < len(prof) and prof[j + 1]:
                j += 1
            fig.add_vrect(x0=coords[i] - h / 2, x1=coords[j] + h / 2,
                          fillcolor=f"rgb({r},{g},{b})", opacity=a,
                          line=dict(color=edge, width=1.2),  # outline
                          layer="below")
            i = j + 1
        else:
            i += 1


def rz_rings(fig, model, geometry, *, color, alpha, fill, label,
             palette=None):
    """End-on (yz) view of an axisymmetric r-z geometry: concentric
    circles at each electrode's bore/outer radius. If the model carries
    imported per-electrode masks, those ARE the solve input —
    draw and LABEL from them; otherwise rasterize each electrode with the
    SAME electrode_mask the solver uses. Per electrode, never a union
    projection."""
    elm = getattr(model, "el_masks", None)
    if elm:
        h = (getattr(model, "mm_per_gu", None)
             or getattr(model, "h_mm", None))
        if h and el_mask_fills(fig, elm, h, "rz", "y", "z", alpha=alpha,
                           fill=fill, label=label, palette=palette):
            return True
    from ion_gym.physics.raster2d import (electrode_mask)
    from ion_gym.io.lattice import gu_cells
    g = geometry
    h = g.mm_per_gu
    # THE counting function (A7): same cell count as the historical
    # round() for every conforming deck; non-conforming decks refuse
    # here too, so the overlay can never draw a lattice the solver
    # would not build.
    nx = gu_cells(g.width_mm, h, axis="x", what="width_mm domain extent")
    ny = gu_cells(g.height_mm, h, axis="y", what="height_mm domain extent")
    xs = np.arange(nx) * h       # kernel frame
    ys = np.arange(ny) * h
    # plane_grid_views, not meshgrid: this feeds
    # electrode_mask's pure comparisons, so dense coordinate copies are
    # waste -- and the meshgrid path is what died inside numpy on some
    # build. The views also refuse a runaway grid with the numbers.
    from ion_gym.physics.raster2d import plane_grid_views
    X, Y = plane_grid_views(xs, ys, "electrode overlay")
    c = color.lstrip("#")
    cr, cg, cb = (int(c[i:i+2], 16) for i in (0, 2, 4))
    a = float(alpha)
    rmax = 0.0
    drew = False
    for el in g.electrodes:
        if not el.shapes:
            continue
        m = electrode_mask(el, X, Y)
        rows = m.any(axis=0)
        if not rows.any():
            continue
        r_in = float(ys[rows].min() - h / 2)
        r_out = float(ys[rows].max() + h / 2)
        rmax = max(rmax, r_out)
        for r0 in (max(r_in, 0.0), r_out):
            if r0 <= 0:
                continue
            fig.add_shape(type="circle", xref="x", yref="y",
                          x0=-r0, y0=-r0, x1=r0, y1=r0, layer="below",
                          line=dict(color=f"rgba({cr},{cg},{cb},{a})",
                                    width=1.2))
            drew = True
    if drew and rmax > 0:
        pad = 0.08 * rmax
        fig.update_xaxes(range=[-rmax - pad, rmax + pad])
        fig.update_yaxes(range=[-rmax - pad, rmax + pad])
        # M-VIZ-Z: the ranges are already square, so a square PANEL makes
        # the rings round without anchoring the axes (which would freeze
        # the zoom aspect).  viz_core owns the policy.
        apply_zoom_policy(
            fig, (-rmax - pad, rmax + pad, -rmax - pad, rmax + pad),
            true_scale=True, px_width=560, x_title=None, y_title=None)
    return drew


def efield_heat(fig, model, z, r_full, img):
    """|E| heatmap for the field (xy) plane — MOVED VERBATIM from
    sim_app._base_figure (Phase B). Shape mismatch means the image was
    produced on the half-plane: mirror across the axis exactly as
    potential_image does."""
    E = model.efield_magnitude()
    if E.shape == img.shape:
        # coords already correct from potential_image
        fig.add_heatmap(x=z, y=r_full, z=E.T,
                        colorscale="Inferno", opacity=0.55,
                        showscale=False)
    else:
        zf, rf2 = model.extent()
        rr = np.concatenate([-rf2[::-1], rf2[1:]])
        Ef = np.concatenate([E[:, ::-1], E[:, 1:]], axis=1)
        fig.add_heatmap(
            x=zf if Ef.shape[0] == len(zf) else z,
            y=rr, z=Ef.T, colorscale="Inferno",
            opacity=0.55, showscale=False)


def potential_contours(fig, z, r_full, img, n_contours):
    """Blue equipotential line set — MOVED VERBATIM (Phase B)."""
    if n_contours > 0:
        fig.add_contour(
            x=z, y=r_full, z=img.T, showscale=False,
            contours=dict(coloring="lines", start=float(img.min()),
                          end=float(img.max()),
                          size=max((img.max()-img.min())/n_contours, 1e-6)),
            line=dict(width=1, color="#4c78a8"))


def grid_overlays(fig, spec, la, lb, *, label=True):
    """Dashed outlines for TRANSMISSION GRIDS (is_grid electrodes), drawn
    from their DECLARED rect shapes (grids occupy no
    cells in the ele/metal masks, so every mask-driven metal renderer
    omits them -- a tunable electrode the user cannot see and so does not
    know to tune). Only the (x, y) plane carries a 2-D spec's declared
    rects; other view planes have no grid extent to draw and are skipped
    by construction, not by error."""
    ax = {la, lb}
    if ax != {"x", "y"}:
        return 0
    # rz specs declare rects in (axial x, r y) with r >= 0; the display
    # mirrors r about 0, so each grid gets its mirrored twin too --
    # otherwise the overlay tells only half the truth the metal renderer
    # tells (agnosticism check).
    _mirror_r = getattr(spec.geometry, "coords", "") == "rz"
    n = 0
    for el in spec.geometry.electrodes:
        if not getattr(el, "is_grid", False):
            continue
        for sh in (el.shapes or []):
            pr = getattr(sh, "params", None) or {}
            if getattr(sh, "type", "") != "rect" or not pr:
                print(f"[viz] grid electrode {el.name!r}: non-rect shape "
                      f"has no grid-outline rendering; skipped")
                continue
            gx0, gy0 = float(pr["x_mm"]), float(pr["y_mm"])
            gx1, gy1 = gx0 + float(pr["width_mm"]), gy0 + float(pr["height_mm"])
            loops = [(gy0, gy1)]
            if _mirror_r and gy1 > 0:
                loops.append((-gy1, -max(gy0, 0.0)))
            for gy0_, gy1_ in loops:
                xs = [gx0, gx1, gx1, gx0, gx0]
                ys = [gy0_, gy0_, gy1_, gy1_, gy0_]
                if la == "y":           # transposed view
                    xs, ys = ys, xs
                fig.add_scatter(x=xs, y=ys, mode="lines",
                                line=dict(color="#666666", width=1.4,
                                          dash="dash"),
                                name=f"{el.name} (grid)", showlegend=False,
                                hovertext=f"{el.name} (grid, {el.dc:g} V)",
                                hoverinfo="text")
            # SAME CLASS AS ELECTRODE LABELS (grid labels
            # ignored the w_ellabel toggle solid electrodes obey). 
            # is the SAME state el_mask_fills receives; dashes and hover
            # stay regardless, only the text obeys the switch. Labeled
            # once (the +r copy), not per mirrored twin.
            # (Solid electrodes are labeled from
            # mask centroids, which grids lack, so the label rides with
            # the declared-shape outline; the dc shown is read from the
            # spec at draw time -- display == solver input.)
            if label:
                cx, cy = 0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1)
                if la == "y":
                    cx, cy = cy, cx
                fig.add_annotation(x=cx, y=cy,
                                   text=(f"<b>{el.name}</b> · "
                                         f"{el.dc:g} V"),
                                   showarrow=False, yshift=10,
                                   font=dict(color="#666666", size=11))
            n += 1
    return n


def station_overlays(fig, spec, la, lb, *, label=True, autorange=None):
    """Draw DECLARED STATIONS (detect / probe planes) on a plotly view.

    Generalizable by construction (station windows
    are generalizable, never instrument-specific): nothing here
    knows what instrument it is looking at. A station declares an axis,
    a position on that axis, and an optional per-axis window; this draws
    that declaration in whatever view plane is on screen, for any deck
    with any number of stations. The matplotlib renderer has drawn these
    already; the plotly UI had no equivalent, which is why the
    detector was invisible in the View tab and detection statistics
    could not be tied to anything on screen (L-63c).

    Geometry rules, all read off the declaration:
      * station axis IN the view plane -> a segment ACROSS the other
        axis, spanning the window on that axis (the detector face).
      * station axis NORMAL to the plane -> the face is edge-on; drawn
        as a dashed rectangle of its in-plane window, or skipped when
        it declares none (a reported skip, never a silent one).
      * a window absent on an axis means "unbounded there": the segment
        spans the view, which is what the detector actually accepts.

    Returns the number of stations drawn. `autorange` is the
    (lo, hi) fallback span for unbounded axes; when None the drawn
    extent falls back to the geometry's own extent on that axis.
    """
    sts = list(getattr(spec, "stations", None) or [])
    if not sts:
        return 0
    g = spec.geometry
    org = getattr(g, "origin_mm", None) or (0.0, 0.0)
    ext = {}
    for i, ax in enumerate("xy"):
        o = float(org[i]) if len(org) > i else 0.0
        span = {"x": g.width_mm, "y": g.height_mm}[ax]
        ext[ax] = (o, o + span)
    ext["z"] = (0.0, float(getattr(g, "depth_mm", 0.0) or 0.0))
    if autorange is not None:
        ext[la] = (float(autorange[0]), float(autorange[1]))

    def _span(ax, win):
        """Window on `ax` if declared, else the view's own extent."""
        if win and ax in win:
            lo, hi = win[ax]
            return float(lo), float(hi)
        return ext.get(ax, (0.0, 0.0))

    KIND_COLOR = {"detect": "#2ca02c", "probe": "#1f77b4"}
    n = 0
    for st in sts:
        ax = getattr(st, "axis", None)
        pos = float(getattr(st, "pos_mm", 0.0))
        win = {a: (float(lo), float(hi))
               for a, (lo, hi) in (getattr(st, "window", None)
                                   or {}).items()}
        kind = getattr(st, "kind", "detect")
        color = KIND_COLOR.get(kind, "#d62728")
        name = getattr(st, "name", kind)
        if ax == la or ax == lb:
            other = lb if ax == la else la
            o0, o1 = _span(other, win)
            if ax == la:
                xs, ys = [pos, pos], [o0, o1]
            else:
                xs, ys = [o0, o1], [pos, pos]
            fig.add_scatter(x=xs, y=ys, mode="lines",
                            line=dict(color=color, width=4),
                            name=f"{name} ({kind})", showlegend=False,
                            hovertext=(f"{name}: {kind} station, "
                                       f"{ax}={pos:g} mm, window "
                                       f"{other}=[{o0:g}, {o1:g}] mm"),
                            hoverinfo="text")
            if label:
                fig.add_annotation(
                    x=(pos if ax == la else 0.5 * (o0 + o1)),
                    y=(0.5 * (o0 + o1) if ax == la else pos),
                    text=f"<b>{name}</b>", showarrow=False,
                    yshift=(0 if ax == la else 10),
                    xshift=(10 if ax == la else 0),
                    font=dict(color=color, size=11))
            n += 1
            continue
        # Face normal to this panel: edge-on. Draw its in-plane window
        # so the user still sees WHERE it accepts, or say why not.
        if not (la in win or lb in win):
            print(f"[viz] station {name!r} is normal to the {la}{lb} "
                  f"plane and declares no {la}/{lb} window; nothing to "
                  f"draw in this view (try the plane containing "
                  f"{ax!r})")
            continue
        a0, a1 = _span(la, win)
        b0, b1 = _span(lb, win)
        fig.add_scatter(x=[a0, a1, a1, a0, a0], y=[b0, b0, b1, b1, b0],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dot"),
                        name=f"{name} ({kind}, edge-on)",
                        showlegend=False,
                        hovertext=(f"{name}: {kind} station at "
                                   f"{ax}={pos:g} mm (normal to this "
                                   f"view); window shown"),
                        hoverinfo="text")
        if label:
            fig.add_annotation(x=0.5 * (a0 + a1), y=0.5 * (b0 + b1),
                               text=f"<b>{name}</b> ({ax}={pos:g})",
                               showarrow=False,
                               font=dict(color=color, size=11))
        n += 1
    return n


def station_stats(spec, trajs, cols, *, mz=None):
    """Per-station detection statistics for ANY deck.

    One authority with the figure: both this and station_overlays read
    the same declared StationSpec, so the numbers in the stats panel and
    the marks on the plot can never disagree about where the detector is
    or how wide its window is.

    Returns a list of dicts, one per declared station, each with n_hit,
    n_total, arrival-time mean/FWHM, resolution R = t/(2 dt), and the
    reflection-family ordinals seen. Stations that nothing reached
    report n_hit = 0 with a reason -- never omitted.
    """
    from ion_gym.physics.stations import first_detection
    import numpy as _np
    out = []
    sts = list(getattr(spec, "stations", None) or [])
    for st in sts:
        ts, fam = [], set()
        for tr in trajs:
            # name= selects THIS station: with several declared, the
            # unnamed call would silently report the first one for all
            # of them.
            det = first_detection(tr, cols, spec, name=st.name)
            if det is None:
                continue
            ts.append(float(det["t_us"]))
            if det.get("k") is not None:
                fam.add(int(det["k"]))
        row = dict(name=st.name, kind=getattr(st, "kind", "detect"),
                   axis=st.axis, pos_mm=float(st.pos_mm),
                   window={a: (float(lo), float(hi)) for a, (lo, hi)
                           in (st.window or {}).items()},
                   n_total=len(trajs), n_hit=len(ts),
                   k_families=sorted(fam))
        if ts:
            a = _np.asarray(ts)
            t_mean = float(a.mean())
            # quantile FWHM: the same estimator the study audits use
            lo, hi = _np.quantile(a, [0.1173, 0.8827])
            fwhm_ns = float((hi - lo) * 1e3)
            row.update(t_us=t_mean, fwhm_ns=fwhm_ns,
                       R=(t_mean * 1e3 / (2.0 * fwhm_ns)
                          if fwhm_ns > 0 else float("inf")))
        else:
            row["reason"] = ("no ion reached this station's window "
                             "within t_max")
        out.append(row)
    return out


def stats_figure(rows, *, kind="stations", height=300, title=None):
    """Plotly figure for the statistics card: 'stations' or 'planes'.

    WHY IT LIVES HERE. Rendering goes through the framework, not
    hand-rolled at the call site (do not hand-roll matplotlib
    for one-off plots" / the ion-gym-render contract). It sits beside
    `station_stats` deliberately: that function is already the ONE
    authority the figure and the panel share, so a plot built from its
    rows cannot disagree with the table above it.

    `kind="stations"` takes `station_stats` rows and draws hits and
    resolution per declared station on twin axes -- the two numbers an
    operator actually compares between detectors. `kind="planes"` takes
    the plane rows the stats table builds and draws hit counts.

    A station or plane that NOTHING reached is DRAWN, at zero, with its
    reason in the hover. Dropping it would read as "no such detector"
    when the truth is "nothing arrived" -- the same rule
    `stations_markdown` follows for its table.
    """
    import plotly.graph_objects as go
    rows = list(rows or [])
    if not rows:
        return None
    if kind not in ("stations", "planes"):
        raise ValueError(
            f"stats_figure: kind must be 'stations' or 'planes', got "
            f"{kind!r} -- an unrecognised kind would silently draw the "
            f"wrong quantity")

    names = [str(r.get("name", "?")) for r in rows]
    hits = [int(r.get("n_hit", 0) or 0) for r in rows]
    hover = []
    for r in rows:
        if r.get("n_hit"):
            hover.append(
                f"{r.get('name')}<br>hits {r.get('n_hit')}/"
                f"{r.get('n_total')}<br>t {r.get('t_us', float('nan')):.4f} us"
                f"<br>FWHM {r.get('fwhm_ns', float('nan')):.4f} ns"
                f"<br>R {r.get('R', float('nan')):,.0f}")
        else:
            hover.append(f"{r.get('name')}<br>0/{r.get('n_total', 0)}"
                         f"<br>{r.get('reason', 'nothing arrived')}")
    fig = go.Figure()
    fig.add_bar(x=names, y=hits, name="hits", marker_color="#4C78A8",
                hovertext=hover, hoverinfo="text")
    if kind == "stations":
        rs = [float(r["R"]) if r.get("n_hit") and r.get("R") not in
              (None, float("inf")) else None for r in rows]
        if any(v is not None for v in rs):
            fig.add_scatter(x=names, y=rs, name="R = t/2dt", yaxis="y2",
                            mode="markers+lines", marker_color="#E45756")
            fig.update_layout(yaxis2=dict(title="R", overlaying="y",
                                          side="right", showgrid=False))
    fig.update_layout(
        title=title or ("Per-station detection" if kind == "stations"
                        else "Per-plane impacts"),
        height=height, margin=dict(l=50, r=50, t=40, b=40),
        yaxis=dict(title="ions"), showlegend=True,
        legend=dict(orientation="h", y=1.12))
    return fig


def metal_boundary(fig, z, r_full, em):
    """The electrode-boundary contour (em at 0.5) — the line that says
    where the metal ends. MOVED VERBATIM (Phase B)."""
    # closed loops via the vacuum-padded tracer: the 0.5
    # add_contour drew boundary-flush metal as an OPEN line ("clipped"
    # electrodes at the domain edge).
    h_ = float(z[1] - z[0]) if len(z) > 1 else 1.0
    px, py = [], []
    for lp in mask_outline_mm(em, h_, (float(z[0]), float(r_full[0]))):
        px += list(lp[:, 0]) + [None]
        py += list(lp[:, 1]) + [None]
    fig.add_scatter(x=px, y=py, mode="lines",
                    line=dict(color="#333", width=1.4),
                    showlegend=False, hoverinfo="skip")


def rz_axial_heat(fig, z, r_full, img):
    """r-z potential shading in the (axial, r) plane — MOVED VERBATIM
    (Phase B)."""
    fig.add_heatmap(x=z, y=r_full, z=img.T,
                    colorscale="RdBu", opacity=0.5,
                    showscale=False, hoverinfo="skip")


def phase_portrait(results, cols, spec, *, axes=("x", "y"),
                   steady_frac=0.25, max_pts_per_panel=20000,
                   color_by="ion", dpi=DEFAULT_DPI):
    """Steady-window phase-space panels (u vs v_u), one per axis, side by
    side — the confinement-quality readout (x-vx, y-vy compactness).

    Config-driven: samples pooled over the trailing steady window
    (traj_stats.steady_slice — the same window every reported metric
    uses, so the picture and the numbers can never disagree). Panel
    annotations carry T_var, eps and sigma from traj_stats; the suptitle
    carries the operating point. Returns the matplotlib Figure; saving is
    the caller's decision.

    color_by: "ion" (per-ion hue, default) | "time" (progress along the
    window). Points decimated uniformly to max_pts_per_panel — decimation
    affects the PICTURE only; annotated numbers come from the FULL
    steady-window statistics.
    """
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from ion_gym.physics import traj_stats as _ts

    t_max = float(spec.integration.t_max_us)
    rep = _ts.confinement_report(results, cols, spec,
                                 steady_frac=steady_frac, axes=axes)
    fig, axs = plt.subplots(1, len(axes), figsize=(5.2 * len(axes), 4.6),
                            dpi=dpi)
    axs = np.atleast_1d(axs)
    from ion_gym.physics.stats import mz_of_results
    mzs = mz_of_results(spec, len(results))
    multi = len(set(mzs)) > 1
    for ax_name, ax in zip(axes, axs):
        pooled = _ts._pooled_axis(results, cols, ax_name,
                                  steady_frac, t_max)
        if pooled is None:
            raise VizError("no steady-window samples on axis " + ax_name)
        u, v, t, k = pooled
        if len(u) > max_pts_per_panel:
            sel = np.linspace(0, len(u) - 1, max_pts_per_panel).astype(int)
            u, v, t, k = u[sel], v[sel], t[sel], k[sel]
        if multi:
            # colour = species, so the per-mass clouds are separable
            sp_of = np.array([mzs[int(i)] for i in k])
            for m_da in sorted(set(mzs)):
                s = sp_of == m_da
                ax.scatter(u[s], v[s], s=2.0, alpha=0.5, linewidths=0,
                           label="m/z {0:g}".format(m_da))
            lines = []
            for m_da, blk in rep["species"].items():
                tv = blk.get("T_var_" + ax_name)
                lines.append("{0:g}: {1} K".format(
                    m_da, "-" if tv is None else format(tv, ".0f")))
            ax.set_title(ax_name + "  T_var " + "  ".join(lines),
                         fontsize=8.5)
        else:
            c = k if color_by == "ion" else t
            ax.scatter(u, v, c=c, s=2.0, cmap="viridis", alpha=0.5,
                       linewidths=0)
            ax.set_title(
                "{a}: T_var {t:.0f} K  eps {e:.3g} mm*mm/us  "
                "sigma {s:.2f} mm".format(
                    a=ax_name, t=rep["T_var_" + ax_name],
                    e=rep["eps_" + ax_name], s=rep["sigma_" + ax_name]),
                fontsize=9)
        ax.set_xlabel(ax_name + " [mm]")
        ax.set_ylabel("v" + ax_name + " [mm/us]")
        ax.grid(alpha=0.25)
    if multi:
        # LEGEND CONVENTION: outside the data axes —
        # one figure-level legend below, deduplicated across panels.
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=max(len(l), 1),
                   fontsize=8, markerscale=3, frameon=False)
    fig.suptitle("steady window (last {f:.0%} of {tm:g} us)  "
                 "retention(worst) {r:.0%}  clearance p5(worst) "
                 "{c:.2f} mm\n{op}".format(
                     f=steady_frac, tm=t_max, r=rep["retention"],
                     c=rep["clearance_p5"], op=rep["operating_point"]),
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07 if multi else 0, 1, 0.90))
    _mpl_release(fig)
    return fig


def convergence_plot(bank_rows, *, dpi=DEFAULT_DPI, ylabel="objective"):
    """Campaign convergence from result-bank rows (dicts with 'kind',
    'eval', 'value', optional 'std'). Eval values as points, running best
    as a line, audit (fresh-seed) value as a marked level with its error
    bar. Penalty-tier values (>=1e3, the infeasible ladder) are shown on
    a symlog axis so the feasible landscape stays readable. Returns the
    Figure; saving is the caller's decision."""
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt

    evs = [r for r in bank_rows if r.get("kind") == "eval"]
    if not evs:
        raise VizError("bank contains no eval rows")
    xs = np.array([r["eval"] for r in evs], float)
    ys = np.array([r["value"] for r in evs], float)
    best = np.minimum.accumulate(ys)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=dpi)
    ax.plot(xs, ys, ".", ms=4, alpha=0.55, label="evaluation")
    ax.plot(xs, best, "-", lw=1.6, label="running best")
    for r in bank_rows:
        if r.get("kind") == "audit":
            ax.errorbar([xs.max()], [r["value"]], yerr=[r.get("std", 0.0)],
                        fmt="s", ms=6, capsize=4,
                        label="fresh-seed audit of best")
    if ys.max() >= 1.0e3:
        ax.set_yscale("symlog", linthresh=1.0e3)
    # A5: a dimensionless axis SAYS SO -- a missing unit and a unitless
    # quantity must not look the same. The abscissa is an evaluation
    # ordinal, so it is labelled as one.
    ax.set_xlabel("evaluation (index, dimensionless)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    # LEGEND CONVENTION: never inside the data axes.
    ax.legend(fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _mpl_release(fig)
    return fig


def _screen_rows(bank_rows):
    """Evaluated screen rows split by outcome; shared by the screen
    renderers so every view classifies candidates identically."""
    scr = [r for r in bank_rows if r.get("kind") == "screen"]
    ok = [r for r in scr if r.get("status") == "ok"]
    if not scr:
        raise VizError("no screen rows in the bank")
    def evidenced(r):
        g = (r.get("sigma_growth_x"), r.get("sigma_growth_y"))
        return (None not in g) and max(g) <= 1.25 and r["value"] < 1.0e3
    def feasible(r):
        return r["value"] < 1.0e3
    return scr, ok, evidenced, feasible


def screen_parcoords(bank_rows, *, max_value=None, height=520):
    """Interactive parallel-coordinates map of a Sobol screen bank —
    one line per evaluated candidate across every search dimension plus
    the objective. Drag a range on any axis to filter; the objective
    axis is clamped below the penalty tier by default so the feasible
    landscape stays readable (max_value overrides). Returns a Plotly
    Figure (notebook-interactive)."""
    import plotly.graph_objects as go
    scr, ok, evidenced, feasible = _screen_rows(bank_rows)
    if not ok:
        raise VizError("screen bank has no evaluated rows to map")
    names = ok[0]["names"]
    cap = float(max_value) if max_value is not None else 1.0e3
    vals = [min(r["value"], cap) for r in ok]
    dims = [dict(label="objective [K]", values=vals)]
    for j, nm in enumerate(names):
        dims.append(dict(label=nm, values=[r["x"][j] for r in ok]))
    fig = go.Figure(go.Parcoords(
        line=dict(color=vals, colorscale="Viridis_r",
                  showscale=True,
                  colorbar=dict(title="objective [K]")),
        dimensions=dims))
    n_ev = sum(1 for r in ok if evidenced(r))
    fig.update_layout(
        height=height,
        title=("screen map: {0} evaluated, {1} feasible, {2} evidenced "
               "(objective clamped at {3:g})").format(
                   len(ok), sum(1 for r in ok if feasible(r)),
                   n_ev, cap))
    _mpl_release(fig)
    return fig


def screen_scatter(bank_rows, x_name, y_name, *, height=520):
    """Interactive scatter of a screen bank on two chosen dimensions.
    Color = objective (clamped below the penalty tier), symbol =
    outcome class (evidenced / feasible-unplateaued / constraint-
    failed), full row in the hover. Returns a Plotly Figure."""
    import plotly.graph_objects as go
    scr, ok, evidenced, feasible = _screen_rows(bank_rows)
    if not ok:
        raise VizError("screen bank has no evaluated rows to map")
    names = ok[0]["names"]
    for nm in (x_name, y_name):
        if nm not in names:
            raise VizError("dimension {0!r} not in the space (have: {1})"
                           .format(nm, ", ".join(names)))
    jx, jy = names.index(x_name), names.index(y_name)
    classes = (("evidenced", lambda r: evidenced(r), "circle"),
               ("feasible, cloud still expanding",
                lambda r: feasible(r) and not evidenced(r), "diamond"),
               ("constraint-failed", lambda r: not feasible(r), "x"))
    fig = go.Figure()
    for label, sel, symbol in classes:
        rows = [r for r in ok if sel(r)]
        if not rows:
            continue
        hover = ["idx {0}<br>value {1:.1f} K<br>ret {2}<br>clr {3}<br>"
                 "growth x/y {4}/{5}".format(
                     r["index"], r["value"],
                     r.get("retention_worst"),
                     r.get("clearance_p5_worst"),
                     r.get("sigma_growth_x"), r.get("sigma_growth_y"))
                 for r in rows]
        fig.add_scatter(
            x=[r["x"][jx] for r in rows], y=[r["x"][jy] for r in rows],
            mode="markers", name=label, text=hover,
            hoverinfo="text",
            marker=dict(symbol=symbol, size=9,
                        color=[min(r["value"], 1.0e3) for r in rows],
                        colorscale="Viridis_r", showscale=(symbol ==
                                                          "circle"),
                        colorbar=dict(title="objective [K]"),
                        line=dict(width=0.5, color="#333")))
    fig.update_layout(
        height=height, xaxis_title=x_name, yaxis_title=y_name,
        title="screen map: {0} vs {1}".format(x_name, y_name),
        # LEGEND CONVENTION: the colorbar owns the
        # right margin; the legend goes horizontally BELOW the plot so
        # the two never overlap.
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    x=0.5, xanchor="center"),
        margin=dict(b=90))
    _mpl_release(fig)
    return fig


def deck_views_mpl(spec, *, title="", show_symmetry=True,
                   figsize=(13.0, 4.2), label_fontsize=6):
    """Multi-axis matplotlib DECLARATION view of one deck
    (a deck presented as matplotlib at true scale).

    Three panels — x–z, x–y (via spec_schematic_ax, the single mpl
    declaration authority), y–z — electrode bodies from the deck's own
    shape declarations with z from each shape's extrude block, DC in
    the legend, isotropic axes, declared mirror planes drawn dash-dot.
    Returns the figure; saving/showing is the caller's decision.
    """
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    g = spec.geometry
    fig, (ax_xz, ax_xy, ax_yz) = plt.subplots(1, 3, figsize=figsize)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
               "#17becf", "#8c564b", "#e377c2"]
    for idx, el in enumerate(g.electrodes):
        col = palette[idx % len(palette)]
        first = True
        for sh in el.shapes:
            p = sh.params or {}
            if sh.type != "rect":
                raise ValueError(
                    f"deck_views_mpl: shape type {sh.type!r} on "
                    f"electrode {el.name!r} not yet drawable — extend "
                    f"the renderer, never skip a body silently")
            ex = p.get("extrude", {}) or {}
            zlo = float(ex.get("lo_mm", 0.0))
            zhi = float(ex.get("hi_mm", g.depth_mm or 0.0))
            x0, y0 = float(p["x_mm"]), float(p["y_mm"])
            w, h = float(p["width_mm"]), float(p["height_mm"])
            lbl = (f"{el.name} ({el.dc:+.0f} V)"
                   if first and el.dc is not None else None)
            ax_xz.add_patch(Rectangle((x0, zlo), w, zhi - zlo,
                                      facecolor=col, alpha=0.55,
                                      edgecolor=col, label=lbl))
            ax_yz.add_patch(Rectangle((y0, zlo), h, zhi - zlo,
                                      facecolor=col, alpha=0.55,
                                      edgecolor=col))
            first = False
    spec_schematic_ax(ax_xy, spec, label_fontsize=label_fontsize)
    pl = getattr(g.symmetry, "planes", {}) or {}
    pm = getattr(g.symmetry, "plane_mm", {}) or {}
    if show_symmetry:
        if pl.get("y") == "mirror":
            yv = float(pm.get("y", 0.0))
            for ax in (ax_xy,):
                ax.axhline(yv, ls="-.", lw=1.0, color="#444",
                           alpha=0.8)
            ax_yz.axvline(yv, ls="-.", lw=1.0, color="#444", alpha=0.8)
        if pl.get("z") == "mirror":
            zv = float(pm.get("z", 0.0))
            for ax in (ax_xz, ax_yz):
                ax.axhline(zv, ls="-.", lw=1.0, color="#444",
                           alpha=0.8)
    for ax, xl, yl in ((ax_xz, "x (mm)", "z (mm)"),
                       (ax_xy, "x (mm)", "y (mm)"),
                       (ax_yz, "y (mm)", "z (mm)")):
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal")               # true scale, always
        ax.autoscale_view()
    ax_xz.set_title("x–z"); ax_xy.set_title("x–y (beam plane view)")
    ax_yz.set_title("y–z")
    # FIGURE-level legend in its own reserved band (legends
    # must never be obscured) — an axes-anchored
    # legend collided with the x–z tick labels whenever the panel
    # aspect squeezed the axes; a bottom band cannot collide with
    # anything because tight_layout is told the band exists.
    handles, labels = ax_xz.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(labels), 4), fontsize=7,
                   frameon=False)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.93))
    return fig


def spec_schematic_ax(ax, spec, *, label_fontsize=6):
    """Draw one spec's DECLARED geometry (its shape declarations, mm,
    true scale) onto an existing axes, rails colored by drive class:
    RF phase 0 / RF phase 180 / DC-biased / grounded. This is the
    DECLARATION view — for generated cell-board specs the shapes ARE
    the single source the voxelizer measures. Solver-true rendering
    (scene_from_simspec) remains the authority for any physics claim."""
    from matplotlib.patches import Rectangle
    phase_of = {g.name: g.phase_deg for g in
                (spec.geometry.rf_groups or [])}
    palette = {"rf0": "#d62728", "rf180": "#1f77b4",
               "dc": "#2ca02c", "gnd": "#7f7f7f"}
    for el in spec.geometry.electrodes:
        if el.rf_groups:
            ph = phase_of.get(el.rf_groups[0], 0.0)
            cls = "rf0" if abs(ph) < 90.0 else "rf180"
        elif el.dc not in (None, 0.0):
            cls = "dc"
        else:
            cls = "gnd"
        for s in el.shapes:
            d = s.to_dict()
            if d.get("type") != "rect":
                raise VizError(
                    "spec_schematic_ax draws declared rects only; "
                    "electrode {0!r} has a {1!r} shape — use the "
                    "solver-true scene for this spec".format(
                        el.name, d.get("type")))
            ax.add_patch(Rectangle((d["x_mm"], d["y_mm"]),
                                   d["width_mm"], d["height_mm"],
                                   facecolor=palette[cls],
                                   edgecolor="#222", linewidth=0.3))
    ax.set_xlim(0, spec.geometry.width_mm)
    ax.set_ylim(0, spec.geometry.height_mm)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=label_fontsize)
    return palette


def screen_gallery(bank_rows, space, template, *, n=12, pick="best",
                   ncols=4, dpi=140):
    """Small-multiple gallery of screened candidate GEOMETRIES —
    declared shapes, no solves, so browsing hundreds is instant.

    pick: "best" = the n lowest-objective feasible rows;
          "spread" = n rows quantile-spaced across the feasible
          objective range (the range of what was screened, not just
          the winners).
    Panel titles: bank index / objective / channel WxH / frequency.
    Returns the matplotlib Figure."""
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from ion_gym.physics.cell_screen import build_spec

    rows = [r for r in bank_rows if r.get("kind") == "screen"
            and r.get("status") == "ok" and r["value"] < 1.0e3]
    if not rows:
        raise VizError("no feasible screen rows to draw")
    rows.sort(key=lambda r: r["value"])
    if pick == "best":
        chosen = rows[:n]
    elif pick == "spread":
        idx = np.unique(np.linspace(0, len(rows) - 1,
                                    min(n, len(rows))).astype(int))
        chosen = [rows[i] for i in idx]
    else:
        raise VizError("pick must be 'best' or 'spread', got "
                       + repr(pick))
    nrows = int(math.ceil(len(chosen) / ncols))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(3.1 * ncols, 2.5 * nrows), dpi=dpi)
    axs = np.atleast_1d(axs).ravel()
    palette = None
    for ax, r in zip(axs, chosen):
        spec = build_spec(space.to_param(np.asarray(r["x"])), template)
        palette = spec_schematic_ax(ax, spec)
        nm = dict(zip(r["names"], r["x"]))
        ax.set_title(
            "#{0}  {1:.0f} K  {2:.1f}x{3:.1f} mm  {4:.2f} MHz".format(
                r["index"], r["value"], nm["channel_w_mm"],
                nm["channel_h_mm"], nm["frequency_hz"] / 1e6),
            fontsize=7)
    for ax in axs[len(chosen):]:
        ax.axis("off")
    if palette:
        from matplotlib.patches import Patch
        fig.legend(handles=[
            Patch(color=palette["rf0"], label="RF phase 0"),
            Patch(color=palette["rf180"], label="RF phase 180"),
            Patch(color=palette["dc"], label="DC-biased"),
            Patch(color=palette["gnd"], label="grounded")],
            loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("screened candidates — DECLARED geometry ({0}); "
                 "solver-true view: show_candidate(index)".format(pick),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _mpl_release(fig)
    return fig


def drive_summary(spec):
    """Per-electrode drive lines from the spec declarations: class
    (RF / DC / RF+DC / grounded) with every number that drives it —
    amplitude, frequency, phase, DC bias. Returns a list of strings;
    callers print or annotate. Declaration view: these are the values
    the solver receives (display == computation for the drive)."""
    groups = {g.name: g for g in (spec.geometry.rf_groups or [])}
    lines = []
    for el in spec.geometry.electrodes:
        parts = []
        for gname in (el.rf_groups or []):
            g = groups.get(gname)
            if g is None:
                raise VizError(
                    "electrode {0!r} references undeclared RF group "
                    "{1!r}".format(el.name, gname))
            parts.append("RF {0:g} V0-p @ {1:g} kHz ph {2:g} deg".format(
                g.amplitude_v, g.frequency_hz / 1e3, g.phase_deg))
        if el.dc not in (None, 0.0):
            parts.append("DC {0:g} V".format(el.dc))
        if not parts:
            parts.append("grounded (0 V)")
        cls = ("RF+DC" if el.rf_groups and el.dc not in (None, 0.0)
               else "RF" if el.rf_groups
               else "DC" if el.dc not in (None, 0.0) else "GND")
        lines.append("{0:12s} [{1:5s}] {2}".format(
            el.name, cls, " + ".join(parts)))
    return lines


def spec_run_table(spec) -> str:
    """Compact run-parameter table for one spec:
    RF amplitude range + frequencies, DC range, m/z set, ion count,
    and the timeout/integration parameters — the numbers the solver
    and flyer receive, in one glance. Returns a formatted string."""
    rfs = list(spec.geometry.rf_groups or [])
    dcs = [e.dc for e in spec.geometry.electrodes
           if e.dc not in (None, 0.0)]
    lines = ["run parameters"]
    if rfs:
        amps = [g.amplitude_v for g in rfs]
        freqs = sorted({g.frequency_hz for g in rfs})
        phases = sorted({g.phase_deg for g in rfs})
        lines.append(
            "  RF     : {0} group(s), {1:g}-{2:g} V0-p, "
            "f {3} kHz, phases {4} deg".format(
                len(rfs), min(amps), max(amps),
                "/".join("{0:g}".format(f / 1e3) for f in freqs),
                "/".join("{0:g}".format(p) for p in phases)))
    else:
        lines.append("  RF     : none (DC-only)")
    if dcs:
        lines.append("  DC     : {0} biased electrode(s), "
                     "{1:g} to {2:g} V".format(len(dcs), min(dcs),
                                               max(dcs)))
    else:
        lines.append("  DC     : none biased (all 0 V)")
    lines.append("  ions   : m/z {0}, n={1}, {2:g} e".format(
        "/".join("{0:g}".format(m) for m in (spec.source.mz_list or [])),
        spec.source.n_ions, spec.source.charge))
    lines.append("  gas    : {0} at {1:g} Torr, {2:g} K".format(
        spec.collisions.gas, spec.collisions.P_torr, spec.collisions.T_k))
    lines.append("  timing : t_max {0:g} us (timeout fate), dt {1:g} ns, "
                 "record every {2} steps".format(
                     spec.integration.t_max_us, spec.integration.dt_ns,
                     spec.integration.rec_every))
    lines.append("  grid   : {0:g} x {1:g} mm at {2:g} mm/gu".format(
        spec.geometry.width_mm, spec.geometry.height_mm,
        spec.geometry.mm_per_gu))
    return "\n".join(lines)


def candidate_field_figure(spec, model=None, *, dpi=DEFAULT_DPI,
                           heatmap=False, cmap=None):
    """Solver-true field of ONE candidate with electrodes COLOR-CODED BY
    DRIVE CLASS and every voltage STATED on the figure.
    Fill = drive class (RF phase 0 / RF phase 180 / DC+ /
    DC- / grounded), not the continuous DC scale — so the two RF phases
    read at a glance; each electrode is annotated with its own drive.
    phi contours are the SOLVED field (instantaneous, RF at phase 0).
    2-D planar is a single true plane (xy) — the multi-axis rule is for
    3-D geometry; a planar board has one physical plane and stretching
    it into orthogonal panes would invent structure. Contract: returns
    the Figure, takes the spec it draws (no hardcoded values), title
    carries the operating point. LAYOUT RULE (standing
    for ALL renderers): annotations, legends, and colorbars must NEVER
    obscure axes, tick labels, or data — a label that hides the axis it
    explains is a defect, not a style choice. Voltage labels sit ON
    their electrode bodies (largest poly), the legend sits above the
    axes, and the colorbar is sized to the drawn axes.
    """
    _mpl_headless_if_needed()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    scene = scene_from_simspec(spec, model=model, field="phi",
                               title=spec.name)
    groups = {g.name: g for g in (spec.geometry.rf_groups or [])}

    def _class_of(el):
        ph = None
        for gname in (getattr(el, "rf_groups", None) or []):
            g = groups.get(gname)
            if g is not None:
                ph = g.phase_deg
                break
        if ph is not None:
            return "rf0" if abs(((ph + 90) % 360) - 90) < 90 else "rf180"
        if el.dc is None or abs(el.dc) < 1e-9:
            return "gnd"
        return "dcpos" if el.dc > 0 else "dcneg"

    palette = {"rf0": "#d62728", "rf180": "#1f77b4", "dcpos": "#2ca02c",
               "dcneg": "#9467bd", "gnd": "#9e9e9e"}
    label = {"rf0": "RF phase 0", "rf180": "RF phase 180",
             "dcpos": "DC +", "dcneg": "DC -", "gnd": "grounded"}

    # figure height FOLLOWS the domain aspect: a fixed tall
    # figure around an aspect-equal squat domain was mostly whitespace.
    g = spec.geometry
    _h = 8.5 * (g.height_mm / max(g.width_mm, 1e-9))
    fig, ax = plt.subplots(figsize=(8.5, min(9.0, max(3.4, _h + 1.9))),
                           dpi=dpi)
    # solved phi contours underneath
    fs = next((f for f in scene.fields if f.plane == "xy"), None)
    if fs is None and scene.fields:
        fs = scene.fields[0]
    if fs is None:
    # FIGURES RENDER ONLY SOLVED STATE: this
        # function's contract is a solver-true field figure. A scene with
        # no field means no solve backs the picture — rendering electrodes
        # alone would manufacture false evidence. Refuse, loudly.
        raise ValueError(
            "candidate_field_figure: spec {0!r} produced a scene with NO "
            "solved field — refusing to render a field figure without its "
            "solve.".format(getattr(spec, "name", "?")))
    # the raise above guarantees fs exists — no conditional drawing path.
    # (A vestigial `if fs is not None:` here would be a dead alternative
    # path, the display-side sibling of `except: pass`.)
    xs = np.linspace(fs.extent[0], fs.extent[1], fs.values.shape[1])
    ys = np.linspace(fs.extent[2], fs.extent[3], fs.values.shape[0])
    _cmap = cmap or STYLE["field"]
    if heatmap:
        # optional filled view under the contours (the SLIM
        # notebook): same solver-true values, same colour scale as
        # the contour lines — one legend, no second convention.
        ax.pcolormesh(xs, ys, fs.values, cmap=_cmap, alpha=0.55,
                      shading="auto", zorder=0)
    cs = ax.contour(xs, ys, fs.values, levels=18, cmap=_cmap,
                    linewidths=0.8, alpha=0.85)
    # colorbar sized to the DRAWN axes: with aspect="equal" on a squat
    # domain the axes shrink but a slot-fraction colorbar does not — it
        # towered over the plot. axes_locatable tracks
    # the axes height exactly.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="3.5%",
                                              pad=0.12)
    cb = fig.colorbar(cs, cax=cax)
    # label DERIVED from the field object + the spec's actual drive state,
    # never restated: fs.quantity is what the scene sampled; the phase-0
    # qualifier applies only when a drive exists (the display composition
    # is the instantaneous field at t=0 — scene_from_simspec docstring).
    if spec.geometry.rf_groups:
        cb.set_label(fs.quantity + ", RF at phase 0")
    else:
        cb.set_label(fs.quantity)
    # electrodes filled by drive class, annotated with voltage
    present = set()
    for el, body in zip(spec.geometry.electrodes, scene.bodies):
        cls = _class_of(el)
        present.add(cls)
        for poly in body.polys:
            arr = np.asarray(poly)
            ax.fill(arr[:, 0], arr[:, 1], facecolor=palette[cls],
                    edgecolor="#222", linewidth=0.5, zorder=3)
        # label position: centroid of the LARGEST poly — ON the
        # electrode. The old mean-over-all-polys put every multi-shape
        # electrode's label at the plot centre, stacking them into an
    # unreadable collision mid-gap.
        big = max(body.polys, key=lambda p: len(np.asarray(p)))
        arrb = np.asarray(big)
        cx = float(arrb[:, 0].mean())
        cy = float(arrb[:, 1].mean())
        g = groups.get((getattr(el, "rf_groups", None) or [None])[0])
        txt = []
        if g is not None:
            txt.append("{0:g}V@{1:g}k".format(g.amplitude_v,
                                              g.frequency_hz / 1e3))
        if el.dc not in (None, 0.0):
            txt.append("{0:+.1f}Vdc".format(el.dc))
        if txt:
            ax.annotate("\n".join(txt), (cx, cy), ha="center",
                        va="center", fontsize=5.5, zorder=4,
                        color="#111",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    # legend OUTSIDE the data region (standing convention)
    handles = [Patch(color=palette[c], label=label[c])
               for c in ("rf0", "rf180", "dcpos", "dcneg", "gnd")
               if c in present]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=len(handles),
              fontsize=8, frameon=False)
    op = getattr(spec, "name", "candidate")
    fig.suptitle("{0} — solver-true field; electrodes by drive class, "
                 "voltages stated".format(op), fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    _mpl_release(fig)
    return fig


def temperature_estimator_figure(v_by_axis, ke_by_axis, m_kg, *,
                                 fit_from_pctl=20.0, n_bins=80,
                                 title=None):
    """The two-estimator temperature figure: per axis,
    a velocity histogram with the Gaussian at T_var (top) and a log-count
    KE histogram with Maxwell-Boltzmann curves at T_var and T_slope
    (bottom), each panel annotated with both temperatures. This is the
    committed framework version of notebook 05's Stage-E3 plot — the
    notebook and the GUI call THIS, not a private copy. fit_from_pctl is
    the exposed tail-fit parameter. Returns a Plotly Figure.
    """
    from plotly.subplots import make_subplots
    from ion_gym.physics.traj_stats import (two_estimator_report,
                                            KB_EV)

    rep = two_estimator_report(v_by_axis, ke_by_axis, m_kg,
                               fit_from_pctl=fit_from_pctl, n_bins=n_bins)
    barc = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}

    def mb_counts(E_grid, T_K, N, bin_width):
        kT = KB_EV * T_K
        with np.errstate(divide="ignore", invalid="ignore"):
            p = (np.pi * kT) ** -0.5 * E_grid ** -0.5 * np.exp(-E_grid / kT)
        return N * bin_width * p

    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.16,
                        subplot_titles=("v_x", "v_y", "v_z",
                                        "KE_x", "KE_y", "KE_z"))
    for c, ax in enumerate(("x", "y", "z"), start=1):
        est = rep.get(ax)
        if est is None:
            for r in (1, 2):
                # place via row/col only. Passing xref/yref="x1 domain" is
                # INVALID in plotly — the first subplot axis is "x domain"
                # (no digit); only axes 2+ carry a number. row/col already
                # resolve the correct paper-domain ref, so we don't build
        # the string by hand (a thermal assessment
                # crashed on the 2-D-solve annotation with exactly this).
                fig.add_annotation(text="no {0}-motion<br>(2-D solve)"
                                   .format(ax), showarrow=False,
                                   x=0.5, y=0.5, xref="x domain",
                                   yref="y domain",
                                   font=dict(size=12, color="#888"),
                                   row=r, col=c)
            continue
        v = np.asarray(v_by_axis[ax], float)
        ke = np.asarray(ke_by_axis[ax], float)
        # ROBUST DISPLAY RANGES (measured on an 8-rail run): a few
        # RF-micromotion outliers (10-100 eV against a <0.1 eV thermal
        # bulk) stretched both axes until the bulk was a single bin and
        # the M-B curve underflowed the log axis to 1e-312. Histograms
        # now span the central 99.5% (v) / [0, p99.5] (KE); the clip is
        # ANNOTATED with its excluded count, never silent.
        _vlo, _vhi = np.percentile(v, [0.25, 99.75])
        _vpad = 0.05 * (_vhi - _vlo) or 1.0
        _vsel = (v >= _vlo - _vpad) & (v <= _vhi + _vpad)
        _n_vout = int(v.size - _vsel.sum())
        v = v[_vsel]
        # top: velocity histogram + Gaussian at T_var
        vc, ve = np.histogram(v, bins=n_bins)
        vctr = 0.5 * (ve[:-1] + ve[1:])
        vbw = ve[1] - ve[0]
        mu, var = v.mean(), np.var(v)
        fig.add_bar(x=vctr, y=vc, marker_color=barc[ax], opacity=0.55,
                    showlegend=False, row=1, col=c)
        vg = np.linspace(ve[0], ve[-1], 300)
        gy = est["n"] * vbw * (1.0 / np.sqrt(2*np.pi*var)) \
            * np.exp(-(vg-mu)**2 / (2*var))
        fig.add_scatter(x=vg, y=gy, mode="lines",
                        line=dict(color="black", width=2),
                        name="Gaussian @ T_var", legendgroup="gv",
                        showlegend=(c == 1), row=1, col=c)
        fig.update_xaxes(title_text="v_{0} (mm/us)".format(ax),
                         row=1, col=c)
        fig.update_yaxes(title_text="count" if c == 1 else None,
                         row=1, col=c)
        # bottom: KE log-count histogram + M-B at T_var and T_slope
        kp = ke[ke > 0]
        _khi = float(np.percentile(kp, 99.5)) if kp.size else 1.0
        _n_kout = int((kp > _khi).sum())
        kp = kp[kp <= _khi]
        kc, ke_e = np.histogram(kp, bins=n_bins)
        kctr = 0.5 * (ke_e[:-1] + ke_e[1:])
        kbw = ke_e[1] - ke_e[0]
        fig.add_bar(x=kctr, y=kc, marker_color=barc[ax], opacity=0.55,
                    showlegend=False, row=2, col=c)
        Eg = np.linspace(max(ke_e[0], kbw), ke_e[-1], 300)
        for T_K, nm, dash, cc, grp in (
                (est["T_var"], "M-B @ T_var", "solid", "#000000", "mv"),
                (est["T_slope"], "M-B @ T_slope", "dash", "#d62728", "ms")):
            if np.isfinite(T_K) and T_K > 0:
                _my = mb_counts(Eg, T_K, kp.size, kbw)
                # plot the curve only where it is above a fraction of a
                # count: an exp() underflowing to 1e-312 dragged the log
                # axis down until the data was unreadable.
                _keep = _my >= 0.3
                fig.add_scatter(x=Eg[_keep], y=_my[_keep],
                                mode="lines",
                                line=dict(color=cc, width=2, dash=dash),
                                name=nm, legendgroup=grp,
                                showlegend=(c == 1), row=2, col=c)
        fig.add_annotation(
            xref="x{0} domain".format(3+c), yref="y{0} domain".format(3+c),
            x=0.97, y=0.97, xanchor="right", yanchor="top",
            showarrow=False, align="right",
            font=dict(size=10, color="#333"),
            text=("T_var {0:.0f} K<br>T_slope {1:.0f} K{2}".format(
                est["T_var"], est["T_slope"],
                "<br>top 0.5% beyond axis (n={0})".format(_n_kout)
                if _n_kout else "")), row=2, col=c)
        fig.update_xaxes(title_text="kinetic energy (eV)", row=2, col=c)
        fig.update_yaxes(type="log", title_text="count" if c == 1
                         else None, row=2, col=c)
    fig.update_layout(
        height=760,
        title_text=title or ("two temperature estimators: velocity-"
                             "variance (top) vs energy-tail slope "
                             "(bottom); tail fit from {0:g}th pctl"
                             .format(fit_from_pctl)),
        bargap=0, margin=dict(l=70, r=20, t=70, b=110),
        legend=dict(orientation="h", yanchor="top", y=-0.14,
                    xanchor="center", x=0.5, font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1))
    return fig


def assembly_stage_draw_mode(spec, rot_deg):
    """'drawn' or 'placeholder': the ONE authority for whether
    assembly_overview renders this stage's true cross-sections or a
    dashed bounding placeholder. The overview's branch logic and the
    UI's rotated-stage warning both ask HERE, so the warning can never
    claim a placeholder for a stage the renderer actually draws (a
    funnel once warned as a placeholder after the r-z branch
    learned to draw it).

    Rules mirror the renderer exactly: an r-z stage (bounded solid of
    revolution about local x) is DRAWN whenever its posed axis lands on
    a world axis, any rotation; every other stage is drawn only without
    out-of-plane rotation (about-x/-y), because a z-invariant planar
    slab tilted out of plane has no drawable cross-section."""
    r = list(rot_deg or [0.0, 0.0, 0.0])[:3] + [0.0, 0.0, 0.0]
    rx, ry, rz = (float(v) for v in r[:3])
    coords = getattr(getattr(spec.geometry, "symmetry", None),
                     "coords", None)
    if coords == "rz":
        cry, crz = math.radians(ry), math.radians(rz)
        cy_, sy_ = math.cos(cry), math.sin(cry)
        cz_, sz_ = math.cos(crz), math.sin(crz)
        # world direction of LOCAL X under intrinsic X-Y-Z Euler: the
        # first column of Rz@Ry@Rx does not involve the x-rotation.
        a = (cz_ * cy_, sz_ * cy_, -sy_)
        k = max(range(3), key=lambda i: abs(a[i]))
        return ("drawn" if abs(abs(a[k]) - 1.0) <= 1e-9
                else "placeholder")
    return ("placeholder" if (abs(rx) > 1e-9 or abs(ry) > 1e-9)
            else "drawn")


def assembly_overview(stages, *, seams=None, trajs=None, traj_style=None, height=560,
                      dpi=DEFAULT_DPI, plane="all", fill_electrodes=True,
                      detections=None):
    """Every stage of a staged instrument drawn in ONE world frame.

    stages: [(name, spec, pose_offset_mm, rot_deg)]. Each stage's
    electrode cross-sections are transformed by that stage's pose and
    drawn together, so the question "are the stages actually where I
    think they are" is answered by looking rather than by reading three
    coordinate systems and doing the arithmetic in your head.

    MULTI-AXIS by directive: an assembly is a 3-D object even when every
    stage solves in 2-D, because the poses place them in a shared 3-D
    frame. A single projection can hide a stage displaced along the
    suppressed axis -- which is exactly the misplacement this view exists
    to catch -- so xz and xy are drawn side by side with the axial
    coordinate horizontal in both.

    Seams are drawn as dashed lines: a handoff plane is where the
    instrument's correctness lives, and an overview that showed the metal
    but not the seams would omit the part worth checking.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not stages:
        raise VizError("assembly_overview: no stages to draw")
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
               "#17becf", "#8c564b", "#e377c2"]

    def _rgba(hexcol, a):
        h = hexcol.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{a})"
    # ONE PLANE AT A TIME when asked: the full assembly view is
    # linked to the buttons on
    # the top of the view pane, never smashed onto one view.
    # plane='all' keeps the 3-panel contact sheet for notebooks and
    # library callers; 'xz'/'xy'/'yz' draws that projection alone at
    # full width, and every drawing site below routes through _col/_add
    # so a single authority decides what lands where.
    # CONVENTION A — THE AXIAL COORDINATE GOES HORIZONTAL: the z
    # axis is the
    # abscissa whenever a z axis is present. This map is the ONE
    # orientation authority for every panel: (horizontal dim,
    # vertical dim). z-carrying panels put z on the abscissa, matching
    # the single-FA view's _plane_cols — the app and this overview must
    # not draw the same instrument two different ways. Drawing sites
    # below still author Scatters in first-named-dim order (x for xz,
    # x for xy, y for yz); _add() maps that to this orientation at the
    # one boundary, and the vline/hline sites consult this map.
    _pnl_dims = {"xz": ("z", "x"), "xy": ("x", "y"), "yz": ("z", "y")}
    _titles = {"xz": "z–x  (transport plane, z horizontal)",
               "xy": "x–y  (cross-section)", "yz": "z–y"}
    if plane not in ("all", "xz", "xy", "yz"):
        raise VizError(f"assembly_overview: unknown plane {plane!r}; "
                       f"one of all/xz/xy/yz")
    if plane == "all":
        _panel_col = {"xz": 1, "xy": 2, "yz": 3}
        fig = make_subplots(rows=1, cols=3, shared_xaxes=False,
                            subplot_titles=tuple(_titles[p]
                                                 for p in ("xz", "xy", "yz")))
    else:
        _panel_col = {p: (1 if p == plane else None)
                      for p in ("xz", "xy", "yz")}
        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=(_titles[plane],))

    def _add(trace, panel):
        c = _panel_col[panel]
        if c is None:
            return
        # THE ORIENTATION BOUNDARY: sites author (first-named dim,
        # second-named dim); z-carrying panels render z horizontal per
        # _pnl_dims, so xz and yz swap here — once, for every trace.
        if panel in ("xz", "yz"):
            trace.x, trace.y = trace.y, trace.x
        fig.add_trace(trace, row=1, col=c)

    # legend entries attach to the FIRST DRAWN panel — pinning them to
    # xz left the legend EMPTY in xy/yz single-plane modes (caught by
    # the plane-mode drive; a legend that exists only in one projection
    # is a defect of the same family as the global drew==0 flag).
    _first_drawn = next(p for p in ("xz", "xy", "yz")
                        if _panel_col[p] is not None)

    # Assembly-wide z range: the declared drift window of whichever
    # stage has one. A stage with no z bound is drawn across all of it.
    z_lo_all, z_hi_all = 0.0, 1.0
    for _n, _sp, _o, _r in stages:
        b = getattr(_sp, "bounds", None)
        if b is not None and getattr(b, "z_max_on", False):
            z_hi_all = max(z_hi_all, float(getattr(b, "z_max", 1.0)))
        if b is not None and getattr(b, "z_min_on", False):
            z_lo_all = min(z_lo_all, float(getattr(b, "z_min", 0.0)))

    drew = 0
    rotated = []                 # stages drawn as placeholders, warned below
    for si, (name, spec, off, rot) in enumerate(stages):
        # A stage could be drawn but missing from the
        # legend): `showlegend=(drew == 0)` was a GLOBAL first-trace
        # flag, so only the FIRST stage ever got a legend row. One
        # entry PER STAGE is the intent; this per-stage flag delivers it.
        _stage_legend_pending = True
        col = palette[si % len(palette)]
        ox, oy, oz = (list(off) + [0.0, 0.0, 0.0])[:3]
        rx, ry, rz = ((list(rot) + [0.0, 0.0, 0.0])[:3] if rot
                      else (0.0, 0.0, 0.0))
        _out_of_plane = abs(float(rx)) > 1e-9 or abs(float(ry)) > 1e-9
        _in_plane = abs(float(rz)) > 1e-9
        # ---- r-z STAGE (mixed-geometry assemblies) ----
        # An r-z stage is a bounded SOLID OF REVOLUTION about its LOCAL
        # X axis (the r-z kernel's axial axis), so a 90-degree pose is
        # not the "infinite slab tilted out of plane" case the
        # placeholder below exists for -- its cross-sections are exact
        # in every panel whenever the posed axis lands on a world axis.
        # Panels containing the axis draw the ring pair (+/-[r_in,r_out]
        # about the axis over the axial span); the perpendicular panel
        # draws the bore and outer circles.
        _coords = getattr(getattr(spec.geometry, "symmetry", None),
                          "coords", None)
        if _coords == "rz":
            _crx, _cry, _crz = (math.radians(float(v))
                                for v in (rx, ry, rz))
            _cxr, _sxr = math.cos(_crx), math.sin(_crx)
            _cyr, _syr = math.cos(_cry), math.sin(_cry)
            _czr, _szr = math.cos(_crz), math.sin(_crz)
            _Rm = [[_czr * _cyr,
                    _czr * _syr * _sxr - _szr * _cxr,
                    _czr * _syr * _cxr + _szr * _sxr],
                   [_szr * _cyr,
                    _szr * _syr * _sxr + _czr * _cxr,
                    _szr * _syr * _cxr - _czr * _sxr],
                   [-_syr, _cyr * _sxr, _cyr * _cxr]]
            _a = [_Rm[0][0], _Rm[1][0], _Rm[2][0]]   # world dir of local x
            _k = max(range(3), key=lambda i: abs(_a[i]))
            if assembly_stage_draw_mode(spec, rot) == "placeholder":
                rotated.append((name, list(rot)))    # honest placeholder
                continue
            _sgn = 1.0 if _a[_k] > 0 else -1.0
            _t3 = (ox, oy, oz)
            _panels = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}
            for el in spec.geometry.electrodes:
                for sh in (el.shapes or []):
                    p = sh.params if hasattr(sh, "params") else {}
                    if sh.type != "rect":
                        continue
                    _ax0 = float(p.get("x_mm", 0.0))
                    _axw = float(p.get("width_mm", 0.0))
                    _rin = float(p.get("y_mm", 0.0))
                    _rout = _rin + float(p.get("height_mm", 0.0))
                    _A0, _A1 = sorted((_sgn * _ax0 + _t3[_k],
                                       _sgn * (_ax0 + _axw) + _t3[_k]))
                    for _pnl, (_hd, _vd) in _panels.items():
                        if _panel_col[_pnl] is None:
                            continue
                        _fk = (dict(fill="toself",
                                    fillcolor=_rgba(col, 0.35))
                               if fill_electrodes else {})
                        if _k in (_hd, _vd):
                            _md_ = _vd if _k == _hd else _hd
                            _c = _t3[_md_]
                            for _lo, _hi in ((_c - _rout, _c - _rin),
                                             (_c + _rin, _c + _rout)):
                                _hh = ([_A0, _A1, _A1, _A0, _A0]
                                       if _k == _hd else
                                       [_lo, _hi, _hi, _lo, _lo])
                                _vv = ([_lo, _lo, _hi, _hi, _lo]
                                       if _k == _hd else
                                       [_A0, _A0, _A1, _A1, _A0])
                                _add(go.Scatter(
                                    x=_hh, y=_vv, mode="lines",
                                    line=dict(color=col, width=1.5), **_fk,
                                    name=name, legendgroup=name,
                                    showlegend=(_stage_legend_pending
                                                and _pnl == _first_drawn),
                                    hovertemplate=(
                                        f"<b>{name} · {el.name}</b>"
                                        f"<extra></extra>")), panel=_pnl)
                                _stage_legend_pending = False
                        else:
                            _ch, _cv = _t3[_hd], _t3[_vd]
                            for _r, _dash in ((_rout, "solid"),
                                              (_rin, "dot")):
                                if _r <= 0:
                                    continue
                                _th = [i * math.tau / 64 for i in range(65)]
                                _add(go.Scatter(
                                    x=[_ch + _r * math.cos(t) for t in _th],
                                    y=[_cv + _r * math.sin(t) for t in _th],
                                    mode="lines",
                                    line=dict(color=col, width=1.5,
                                              dash=_dash),
                                    **(_fk if _dash == "solid" else {}),
                                    name=name, legendgroup=name,
                                    showlegend=(_stage_legend_pending
                                                and _pnl == _first_drawn),
                                    hoverinfo="skip"), panel=_pnl)
                                _stage_legend_pending = False
                        drew += 1
            continue
        if assembly_stage_draw_mode(spec, rot) == "placeholder":
            # PLACEHOLDER, and now ONLY for rotations that genuinely
            # cannot be drawn as a cross-section (a drawing
            # alone would already exist in plotly — correct for the
            # in-plane
            # case, handled below; this branch is the residue).
            #
            # A 2-D stage's metal is z-INVARIANT (depth_mm = 0), so a
            # rotation about x or y tilts an INFINITE slab out of the
            # drawing plane. Its intersection with the xy plane is not
            # the rotated rectangle -- it is the projection of an
            # unbounded solid, which is not a cross-section and cannot
            # be drawn as one without inventing the z extent the deck
            # leaves unmodelled.
            #
            # State the limitation precisely. The POSE IS KNOWN EXACTLY:
            # the instrument declares offset_mm and rot_deg per stage and
            # the flight honours both. Nothing about the model is
            # missing -- the inverse of the error where a drawing
            # asserted a quantity the model lacked.
            rotated.append((name, list(rot)))
            _xs, _ys = [], []
            for el in spec.geometry.electrodes:
                for sh in (el.shapes or []):
                    p = sh.params if hasattr(sh, "params") else {}
                    if sh.type != "rect":
                        continue
                    _x0 = float(p.get("x_mm", 0.0))
                    _y0 = float(p.get("y_mm", 0.0))
                    _xs += [_x0, _x0 + float(p.get("width_mm", 0.0))]
                    _ys += [_y0, _y0 + float(p.get("height_mm", 0.0))]
            if _xs:
                bx0, bx1 = min(_xs) + ox, max(_xs) + ox
                by0, by1 = min(_ys) + oy, max(_ys) + oy
                _lbl = f"{name}: TILTED {rot} — NOT DRAWN TO POSE"
                for _pnl, _ha, _hb, _lo, _hi in (
                        ("xz", bx0, bx1, z_lo_all, z_hi_all),
                        ("xy", bx0, bx1, by0, by1),
                        ("yz", by0, by1, z_lo_all, z_hi_all)):
                    _add(go.Scatter(
                        x=[_ha, _hb, _hb, _ha, _ha],
                        y=[_lo, _lo, _hi, _hi, _lo],
                        mode="lines",
                        line=dict(color=col, width=2, dash="dot"),
                        name=_lbl, legendgroup=name,
                        showlegend=(_pnl == _first_drawn),
                        hoverinfo="text",
                        text=_lbl), panel=_pnl)
                if _panel_col["xy"] is not None:
                    _sfx = ("" if plane != "all" and plane == "xy"
                            else "2") if plane == "all" else ""
                    fig.add_annotation(
                        x=0.5 * (bx0 + bx1), y=0.5 * (by0 + by1),
                        xref=f"x{_sfx}", yref=f"y{_sfx}",
                        showarrow=False,
                        text=f"<b>{name}</b><br>TILTED — placeholder",
                        font=dict(size=9, color="#b00020"))
            continue
        # NOTE: geometry.origin_mm is the SOLVE GRID's anchor, not an
        # offset on shape coordinates -- a shape's x_mm/y_mm are already
        # in the deck's own frame. Adding it here double-counted the
        # offset and drew the OA's plates at y -36.2..-3.8 mm when they
        # straddle zero at +-16.2. Only the stage POSE moves a stage.
        # EXCEPTION, stated with its evidence (mixed-geometry assemblies,
        # the NATIVE 3-D routes (shapes3d/stl3d) fly
        # staged regions in the STORED frame -- tracer3d indexes the
        # array at position/h and applies world_off_mm for DIAGNOSTIC
        # REPORTING ONLY -- so for those routes what the flight sees is
        # shape - origin_mm, and drawing the raw frame displaced the
        # hexapole by its full origin. Frame unification is
        # OPEN; when it closes, this shift and the stage poses change
        # together, in one place each.
        # Discriminated by build_route — the ONE classification build_run
        # itself dispatches on ("dispatch on declared geometry, not a
        # correlate"), so what this figure shifts and what the solver
        # shifts cannot drift apart.
        from ion_gym.physics.sim_build import build_route as _build_route
        _b = _build_route(spec).builder
        _fx, _fy = ((spec.geometry.origin_mm or (0.0, 0.0))[:2]
                    if _b in ("shapes3d", "stl3d", "scene3d")
                    else (0.0, 0.0))
        for el in spec.geometry.electrodes:
            for sh in (el.shapes or []):
                p = sh.params if hasattr(sh, "params") else {}
                if sh.type == "ellipse":
                    # cross-sections of an extruded ellipse: bands over
                    # the extrude span in the axial panels, the true
                    # ellipse outline in xy. Out-of-plane rotations of
                    # these stages fall to the placeholder branch above.
                    _ecx = float(p.get("cx_mm", 0.0)) - float(_fx) + ox
                    _ecy = float(p.get("cy_mm", 0.0)) - float(_fy) + oy
                    _erx = float(p.get("rx_mm", 0.0))
                    _ery = float(p.get("ry_mm", _erx))
                    _ext = p.get("extrude")
                    if not _ext:
                        continue        # z-invariant ellipse: not a case
                    #                     any current deck authors; skip
                    #                     is visible via the refusal if
                    #                     a whole stage is undrawable
                    if _ext.get("axis", "z") != "z":
                        raise VizError(
                            f"assembly_overview: stage {name!r} ellipse "
                            f"extrudes along {_ext.get('axis')!r}; only "
                            f"z-extrudes are drawable — refusing rather "
                            f"than drawing it as z-invariant")
                    _zl = float(_ext["lo_mm"]) + oz
                    _zh = float(_ext["hi_mm"]) + oz
                    _fk = (dict(fill="toself",
                                fillcolor=_rgba(col, 0.35))
                           if fill_electrodes else {})
                    for _pnl, _c0, _rr in (("xz", _ecx, _erx),
                                           ("yz", _ecy, _ery)):
                        _add(go.Scatter(
                            x=[_c0 - _rr, _c0 + _rr, _c0 + _rr,
                               _c0 - _rr, _c0 - _rr],
                            y=[_zl, _zl, _zh, _zh, _zl],
                            mode="lines",
                            line=dict(color=col, width=1.5), **_fk,
                            name=name, legendgroup=name,
                            showlegend=(_stage_legend_pending
                                        and _pnl == _first_drawn),
                            hoverinfo="skip"), panel=_pnl)
                        _stage_legend_pending = False
                    _th = [i * math.tau / 48 for i in range(49)]
                    _exx = [_ecx + _erx * math.cos(t) for t in _th]
                    _eyy = [_ecy + _ery * math.sin(t) for t in _th]
                    if _in_plane:
                        _ar = math.radians(float(rz))
                        _ca2, _sa2 = math.cos(_ar), math.sin(_ar)
                        _exx, _eyy = (
                            [ox + _ca2 * (px - ox) - _sa2 * (py - oy)
                             for px, py in zip(_exx, _eyy)],
                            [oy + _sa2 * (px - ox) + _ca2 * (py - oy)
                             for px, py in zip(_exx, _eyy)])
                    _add(go.Scatter(
                        x=_exx, y=_eyy, mode="lines",
                        line=dict(color=col, width=1),
                        **(dict(fill="toself", fillcolor=col, opacity=0.35)
                           if fill_electrodes else {}),
                        name=name, legendgroup=name,
                        showlegend=(_stage_legend_pending
                                    and _first_drawn == "xy"),
                        hovertemplate=(f"<b>{name} · {el.name}</b><br>"
                                       f"x %{{x:.2f}} y %{{y:.2f}} mm"
                                       f"<extra></extra>")), panel="xy")
                    _stage_legend_pending = False
                    drew += 1
                    continue
                if sh.type == "cutout":
                    # a bore: not metal — drawn as a dotted outline in
                    # xy only, so the aperture reads as an aperture.
                    for _ch in (p.get("children") or []):
                        if _ch.get("type") != "ellipse":
                            continue
                        _bcx = float(_ch.get("cx_mm", 0.0)) - float(_fx) + ox
                        _bcy = float(_ch.get("cy_mm", 0.0)) - float(_fy) + oy
                        _brx = float(_ch.get("rx_mm", 0.0))
                        _bry = float(_ch.get("ry_mm", _brx))
                        _th = [i * math.tau / 48 for i in range(49)]
                        _add(go.Scatter(
                            x=[_bcx + _brx * math.cos(t) for t in _th],
                            y=[_bcy + _bry * math.sin(t) for t in _th],
                            mode="lines",
                            line=dict(color=col, width=1, dash="dot"),
                            name=name, legendgroup=name, showlegend=False,
                            hoverinfo="skip"), panel="xy")
                    continue
                if sh.type != "rect":
                    continue
                x0 = float(p.get("x_mm", 0.0)) - float(_fx) + ox
                y0 = float(p.get("y_mm", 0.0)) - float(_fy) + oy
                w = float(p.get("width_mm", 0.0))
                h = float(p.get("height_mm", 0.0))
                # x-z panel. A 2-D stage declares NO z extent (depth_mm
                # = 0), so its metal is z-INVARIANT: infinite, by
                # construction. Drawing a thin slab here would invent a
                # clearance the model does not have and make the stage
                # look safely out of the drift path -- the opposite of
                # what a placement check is for. The band is therefore
                # drawn across the FULL z range of the assembly and
                # hatched-open at both ends, and the caption says the
                # extent is undeclared.
                # A DECLARED metal depth bounds the band.
                # Per-electrode metal_depth_mm overrides the stage's;
                # 0/None keeps the honest undeclared behaviour above
                # (full-span band + caption). The declared band is drawn
                # closed and slightly heavier: it is a real extent, not
                # an open-ended unknown.
                _md = getattr(el, "metal_depth_mm", None)
                if _md is None:
                    _md = float(getattr(spec.geometry,
                                        "metal_depth_mm", 0.0) or 0.0)
                # A shapes3d shape DECLARES its z extent in its
                # extrude block — drawing it as a z-invariant full-span
                # band displayed geometry the deck does not have (the
                # exit3d stage rendered exactly like its planar
                # neighbours). The extrude is real declared metal: drawn
                # closed, heavier, at its true pose-offset bounds. A
                # non-z extrude axis REFUSES by name rather than being
                # silently drawn as z-invariant.
                _ext = (sh.params or {}).get("extrude") \
                    if hasattr(sh, "params") else None
                if _ext:
                    _ax = _ext.get("axis", "z")
                    if _ax != "z":
                        raise VizError(
                            f"assembly_overview: stage {name!r} shape "
                            f"extrudes along {_ax!r}; only z-extrudes "
                            f"are drawable in the x–z panel — refusing "
                            f"rather than drawing it as z-invariant")
                    zlo = float(_ext["lo_mm"]) + oz
                    zhi = float(_ext["hi_mm"]) + oz
                    _lw = 1.5
                elif _md and _md > 0:
                    zlo, zhi = oz - 0.5 * float(_md), oz + 0.5 * float(_md)
                    _lw = 1.5
                else:
                    zlo, zhi = z_lo_all, z_hi_all
                    _lw = 1.0
                # FILLS (electrode fills must also
                # show in the full assembly view): declared-extent
                # bands fill like the xy cross-section; a z-invariant
                # full-span band fills FAINTLY — the metal is real in
                # every one of those z, and an outline-only column read
                # as empty space.
                # fills follow the app's Display toggle (session 7);
                # OUTLINES always stay, so metal is unambiguous either way
                _fk = (dict(fill="toself",
                            fillcolor=_rgba(col, 0.35 if _lw > 1.0
                                            else 0.10))
                       if fill_electrodes else {})
                _add(go.Scatter(
                    x=[x0, x0 + w, x0 + w, x0, x0],
                    y=[zlo, zlo, zhi, zhi, zlo],
                    mode="lines", line=dict(color=col, width=_lw),
                    **_fk,
                    name=name, legendgroup=name,
                    showlegend=(_stage_legend_pending
                                and _first_drawn == "xz"),
                    hoverinfo="skip"), panel="xz")
                # y–z panel (a 3-D deck enables
                # the xy, yz and xz views) —
                # same z band logic as x–z, with the shape's y extent.
                _add(go.Scatter(
                    x=[y0, y0 + h, y0 + h, y0, y0],
                    y=[zlo, zlo, zhi, zhi, zlo],
                    mode="lines", line=dict(color=col, width=_lw),
                    **_fk,
                    name=name, legendgroup=name,
                    showlegend=(_stage_legend_pending
                                and _first_drawn == "yz"),
                    hoverinfo="skip"), panel="yz")
                # IN-PLANE ROTATION IS EXACT AND IS JUST CORNER MATHS
                # A rotation about z keeps a
                # z-invariant cross-section IN the xy plane, so the true
                # outline is the four corners rotated about the stage's
                # pose origin. No renderer feature is needed for this and
                # declining to draw it was over-conservative.
                _cx = [x0, x0 + w, x0 + w, x0, x0]
                _cy = [y0, y0, y0 + h, y0 + h, y0]
                if _in_plane:
                    _a = math.radians(float(rz))
                    _ca, _sa = math.cos(_a), math.sin(_a)
                    _rxs, _rys = [], []
                    for _px, _py in zip(_cx, _cy):
                        # rotate about the stage's POSE ORIGIN (ox, oy),
                        # which is the point the pose places -- rotating
                        # about the shape's own corner would move the
                        # stage as well as turn it.
                        _dx, _dy = _px - ox, _py - oy
                        _rxs.append(ox + _ca * _dx - _sa * _dy)
                        _rys.append(oy + _sa * _dx + _ca * _dy)
                    _cx, _cy = _rxs, _rys
                _fk2 = (dict(fill="toself", fillcolor=col, opacity=0.35)
                        if fill_electrodes else {})
                _add(go.Scatter(
                    x=_cx, y=_cy,
                    mode="lines", line=dict(color=col, width=1),
                    **_fk2,
                    name=name, legendgroup=name,
                    showlegend=(_stage_legend_pending
                                and _first_drawn == "xy"),
                    hovertemplate=(f"<b>{name} · {el.name}</b><br>"
                                   f"x %{{x:.2f}} y %{{y:.2f}} mm"
                                   f"<extra></extra>")), panel="xy")
                _stage_legend_pending = False
                drew += 1
    if not drew:
        raise VizError(
            "assembly_overview: no drawable cross-sections found in any "
            "stage. This overview draws rect and z-extruded ellipse "
            "shapes, and r-z stages posed with their axis on a world "
            "axis; polygon and STL shapes it cannot draw — it reports "
            "that rather than showing an empty frame that reads as "
            "'nothing is there'.")

    # DETECT/monitor stations. A placement check that showed the metal
    # but not the detector omits the plane the whole flight is aimed at.
    for si, (name, spec, off, rot) in enumerate(stages):
        ox = (list(off) + [0.0, 0.0, 0.0])[0]
        for stn in (getattr(spec, "stations", None) or []):
            if getattr(stn, "axis", None) != "x":
                continue
            xs = float(getattr(stn, "pos_mm", 0.0)) + ox
            kind = getattr(stn, "kind", "monitor")
            colr = "#2ca02c" if kind == "detect" else "#7f7f7f"
            # DRAWN ONLY ACROSS ITS DECLARED WINDOW. A full-height line
            # asserts that the station is active everywhere it crosses,
            # which reads as "detection happens at every crossing" -- the
            # exact misreading this figure once invited. A real MRT
            # detector is live only over z 199.07..226.85 and y +-6.94,
            # and an ion crossing x = -21.296 anywhere else is NOT
            # detected. An undeclared window IS the full span, and only
            # then is a full-height line the truth.
            _win = getattr(stn, "window", None) or {}
            _wz = _win.get("z") or [z_lo_all, z_hi_all]
            _wy = _win.get("y")
            for _pnl, (_lo, _hi) in (
                    ("xz", (float(_wz[0]), float(_wz[1]))),
                    ("xy", (float(_wy[0]), float(_wy[1]))
                     if _wy else (None, None))):
                if _panel_col[_pnl] is None:
                    continue
                if _lo is None:
                    # station axis is x: vertical line where x is the
                    # panel's horizontal dim, horizontal line where x
                    # is vertical (the xz panel, now z-horizontal).
                    if _pnl_dims[_pnl][0] == "x":
                        fig.add_vline(x=xs, line=dict(color=colr, width=2,
                                                      dash="dot"),
                                      row=1, col=_panel_col[_pnl])
                    else:
                        fig.add_hline(y=xs, line=dict(color=colr, width=2,
                                                      dash="dot"),
                                      row=1, col=_panel_col[_pnl])
                    continue
                _add(go.Scatter(
                    x=[xs, xs], y=[_lo, _hi], mode="lines",
                    line=dict(color=colr, width=3),
                    name=f"{getattr(stn, 'name', kind)}",
                    legendgroup="stations",
                    showlegend=(_pnl == _first_drawn),
                    hoverinfo="text",
                    text=f"{getattr(stn, 'name', kind)} ({kind}) "
                         f"active {_lo:.1f}..{_hi:.1f} mm"),
                    panel=_pnl)
            # y–z panel: the station's ACTIVE AREA is the y×z window at
            # x = pos — in this panel it is a rectangle, and drawing it
            # as one shows the reader the face the flight is aimed at.
            # With either window undeclared, that edge spans the panel.
            _ry = ([float(_wy[0]), float(_wy[1])] if _wy
                   else None)
            _rz = [float(_wz[0]), float(_wz[1])]
            if _ry is not None:
                _add(go.Scatter(
                    x=[_ry[0], _ry[1], _ry[1], _ry[0], _ry[0]],
                    y=[_rz[0], _rz[0], _rz[1], _rz[1], _rz[0]],
                    mode="lines",
                    line=dict(color=colr, width=2, dash="dot"),
                    name=f"{getattr(stn, 'name', kind)}",
                    legendgroup="stations",
                    showlegend=(_first_drawn == "yz"),
                    hoverinfo="text",
                    text=f"{getattr(stn, 'name', kind)} ({kind}) window "
                         f"y {_ry[0]:.1f}..{_ry[1]:.1f}, "
                         f"z {_rz[0]:.1f}..{_rz[1]:.1f} mm"),
                    panel="yz")
            if _panel_col["xz"] is not None:
                # x is the VERTICAL dim of the flipped xz panel, so the
                # station label rides the y coordinate at the left edge.
                _cxz = _panel_col["xz"]
                fig.add_annotation(
                    x=0.01, y=xs, xref="paper",
                    yref=("y" if _cxz == 1 else f"y{_cxz}"),
                    showarrow=False, xanchor="left",
                    text=f"{getattr(stn, 'name', kind)} ({kind})",
                    font=dict(color=colr, size=10))

    # (_pnl_dims — the orientation authority — is defined once, above.)
    for _si_s, s in enumerate(seams or []):
        # AXIS-AWARE (seams read as unmarked on a z-seam
        # instrument): a seam is a plane at constant value on ONE world
        # axis — historically always x, so this block drew
        # vlines at x and a z-axis seam was silently undrawable. The
        # seam dict now carries "axis" ("x" default, back-compat) and
        # "value_mm" (falling back to the legacy "x_mm" key), and each
        # panel draws the seam on whichever of its two dimensions
        # carries that axis — vertical line when the axis is the
        # panel's horizontal dimension, horizontal line when vertical.
        _sax = s.get("axis", "x")
        _sval = float(s.get("value_mm", s.get("x_mm", 0.0)))
        _lbl = s.get("label", "seam")
        for _pnl, (_hd, _vd) in _pnl_dims.items():
            _c = _panel_col[_pnl]
            if _c is None:
                continue
            if _sax == _hd:
                fig.add_vline(x=_sval, line=dict(
                    color="#b07a2c", width=1.5, dash="dash"),
                    row=1, col=_c)
                fig.add_annotation(
                    x=_sval, y=0.99,
                    xref=("x" if _c == 1 else f"x{_c}"), yref="paper",
                    showarrow=False, textangle=-90, xanchor="left",
                    yanchor="top", text=_lbl,
                    font=dict(color="#b07a2c", size=10))
            elif _sax == _vd:
                fig.add_hline(y=_sval, line=dict(
                    color="#b07a2c", width=1.5, dash="dash"),
                    row=1, col=_c)
                fig.add_annotation(
                    x=0.99, y=_sval, xref="paper",
                    yref=("y" if _c == 1 else f"y{_c}"),
                    showarrow=False, xanchor="right", yanchor="bottom",
                    text=_lbl,
                    font=dict(color="#b07a2c", size=10))
    _ax_titles = {_p: (f"{_hd} (mm)", f"{_vd} (mm)")
                  for _p, (_hd, _vd) in _pnl_dims.items()}
    for _pnl, (_xt, _yt) in _ax_titles.items():
        _c = _panel_col[_pnl]
        if _c is None:
            continue
        fig.update_xaxes(title_text=_xt, row=1, col=_c)
        # FREE ZOOM EVERYWHERE (the zoom rectangle
        # must not have a fixed aspect ratio, in
        # all views). The xy equal-aspect anchor RE-INTRODUCED the
        # documented M-VIZ-Z regression (an anchored axis reshapes the
        # drag rectangle to 1:1); this figure now carries NO
        # scaleanchor on any panel, same rule as every served view.
        fig.update_yaxes(title_text=_yt, row=1, col=_c)
    # ION PATHS. Drawn AFTER the metal so they sit on
    # top of it: the question this view answers is where the ions go
    # relative to the electrodes, and a path hidden behind a filled
    # electrode answers it backwards.
    #
    # Coordinates arrive in the WORLD frame already (fly_staged maps each
    # region's trace through that region's pose), so nothing is
    # transformed here -- which is the point. A second transform in the
    # renderer would be a second place for the frame convention to drift
    # from the flight's, exactly the frame-mismatch defect class.
    #
    # These are a SUBSET by policy (staged_flight.trace_keep_count) and
    # the caption says how many of how many, so a reader can never take
    # the drawn count for the flown count.
    _n_traj = 0
    # TRACE STYLE FROM THE DISPLAY TAB (trace colors
    # are governed by the Display tab, as they should
    # be). traj_style carries the tab's live values: width, alpha,
    # decim, color_by ("fate" | "solid color" | anything else), solid.
    # Assembly traces bank positions only, so channel/m-z colourings the
    # single-stage view supports are NOT expressible here — an
    # unexpressible mode falls back to fate AND SAYS SO in the title
    # (silent fallback is display lying). Fate mode keeps the lost/
    # arrived distinction (lost heavier + opaque) scaled by the tab's
    # alpha.
    _ts = dict(traj_style or {})
    _tw = float(_ts.get("width", 1.0))
    _ta = float(_ts.get("alpha", 0.45))
    _td = max(1, int(_ts.get("decim", 1)))
    _tmode = _ts.get("color_by", "fate")
    _tsolid = _ts.get("solid", "#1f77b4")
    # TRAJECTORY STYLE: lines/dots/both is
    # a Display-tab condition and was never passed, so the assembly view
    # drew lines whatever the tab said. Same mode map and 3-px markers
    # as the single-FA view (_redraw), so the two views cannot draw one
    # setting two ways. An unknown style REFUSES into the title, never
    # silently — the single-FA path would KeyError, this one says why.
    _draw_mode = {"lines": "lines", "dots": "markers",
                  "lines+dots": "lines+markers"}.get(
                      _ts.get("mode", "lines"))
    _style_notes = []
    if _draw_mode is None:
        _style_notes.append(
            f"trajectory style {_ts.get('mode')!r} is not one of "
            f"lines/dots/lines+dots — drawn as lines")
        _draw_mode = "lines"
    _have_mz = any(_t.get("mz") is not None for _t in (trajs or []))
    if _tmode == "m/z" and not _have_mz:
        _style_notes.append("colour-by m/z: these traces carry no "
                            "per-ion mass (pre-fix flight or literal "
                            "packet) — coloured by fate")
        _tmode = "fate"
    elif _tmode not in ("fate", "solid color", "m/z"):
        _style_notes.append(
            f"colour-by {_tmode!r} needs per-record data the assembly "
            f"traces do not carry — coloured by fate")
        _tmode = "fate"
    _MZ_PAL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
               "#17becf", "#8c564b", "#e377c2"]
    _mz_seen = sorted({float(_t["mz"]) for _t in (trajs or [])
                       if _t.get("mz") is not None})
    _mz_col = {m: _MZ_PAL[k % len(_MZ_PAL)]
               for k, m in enumerate(_mz_seen)}
    for _t in (trajs or []):
        _lost = _t.get("fate") not in ("detected", "completed")
        if _tmode == "solid color":
            _colr = _tsolid
        elif _tmode == "m/z":
            _colr = _mz_col.get(float(_t.get("mz", 0.0)), "#7f7f7f")
        else:
            _colr = "#b00020" if _lost else "#1f77b4"
        _op_lost, _op_ok = min(1.0, _ta * 2.0), _ta
        for _rg in _t.get("regions", []):
            _x = np.asarray(_rg["x"], float)[::_td]
            _y = np.asarray(_rg["y"], float)[::_td]
            _z = np.asarray(_rg["z"], float)[::_td]
            _nm = (f"ion {_t.get('i')}"
                   + (f" — LOST ({_t.get('fate')})" if _lost else ""))
            # NO legend entries for ion paths (an unlabelled
            # "ion 0" is meaningless, so ion trajectories stay out of
            # the legend). The drawn/flown count already lives in the
            # title, which is the honest disclosure; a legend row per
            # path was noise that scaled with the packet.
            for _pnl2, _hh2, _vv2 in (("xz", _x, _z), ("xy", _x, _y),
                                      ("yz", _y, _z)):
                _add(go.Scatter(
                    x=_hh2, y=_vv2, mode=_draw_mode,
                    line=dict(color=_colr, width=_tw),
                    marker=dict(color=_colr, size=3),
                    opacity=(_op_lost if _lost else _op_ok),
                    name=_nm, legendgroup="ions",
                    showlegend=False, hoverinfo="skip"), panel=_pnl2)
        _n_traj += 1

    # TITLE IS A COMPACT DECLARED STRING (figure
    # legibility): "the operating point on a figure is a COMPACT declared
    # string ... never a prose blob". The caveats that used to live here
    # -- undeclared z, rotated placeholders, the drawn-vs-flown count --
    # turned this into three sentences of running text that wrapped over
    # the legend and made the figure unreadable. They are CONDITIONS OF
    # THE RUN, so they belong on the status line, which is where every
    # other condition of a run is reported. The figure states what it is
    # and its operating point; nothing else.
    bits = [f"assembly · {len(stages)} stages"]
    bits.extend(_style_notes)
    if _tmode == "m/z" and _mz_seen:
        for _m in _mz_seen:
            _add(go.Scatter(x=[None], y=[None], mode="lines",
                            line=dict(color=_mz_col[_m], width=3),
                            name=f"m/z {_m:g}", legendgroup="mz",
                            showlegend=True, hoverinfo="skip"),
                 panel=_first_drawn)
    if _n_traj:
        _flown = trajs[0].get("n_flown") if trajs else None
        bits.append(f"{_n_traj}/{_flown} paths" if _flown
                    else f"{_n_traj} paths")
    if rotated:
        bits.append(f"{len(rotated)} tilted (placeholder)")
    # DETECTION MARKERS (detection markers are
    # not kill zones, just pass-through): one green
    # diamond per REGISTERED crossing, drawn from the flight's arrival
    # records — the station stays non-destructive by
    # design and the trace continues past; the marker is what makes
    # the registration visible to the eye.
    if detections:
        _dx = [float(d[0]) for d in detections]
        _dy = [float(d[1]) for d in detections]
        _dz = [float(d[2]) for d in detections]
        # IMPACT MARKER SIZE/SYMBOL are Display-tab conditions
        # and were once hardcoded here. traj_style carries them
        # from the tab; the old diamond/5 stay as the defaults so
        # library callers without a traj_style draw unchanged.
        _isz = float(_ts.get("impact_size", 5))
        _isym = _ts.get("impact_symbol", "diamond")
        for _pnl, _hx, _hy in (("xz", _dx, _dz), ("xy", _dx, _dy),
                               ("yz", _dy, _dz)):
            _add(go.Scatter(
                x=_hx, y=_hy, mode="markers",
                marker=dict(symbol=_isym, size=_isz, color="#1a7f37",
                            line=dict(width=0.5, color="#0b3d1c")),
                name=f"detections ({len(detections)})",
                legendgroup="detections",
                showlegend=(_pnl == _first_drawn),
                hoverinfo="text",
                text=[f"detected at ({a:.2f}, {b:.2f}) mm — station is "
                      f"pass-through" for a, b in zip(_hx, _hy)]),
                panel=_pnl)

    _n_decl = sum(
        1 for _nm, sp, _o, _r in stages
        if (float(getattr(sp.geometry, "metal_depth_mm", 0.0) or 0.0) > 0
            or any(getattr(e, "metal_depth_mm", None)
                   for e in sp.geometry.electrodes)
            or any((sh.params or {}).get("extrude")
                   for e in sp.geometry.electrodes
                   for sh in (e.shapes or []) if hasattr(sh, "params"))))
    if _n_decl == len(stages):
        bits.append("z declared")
    elif _n_decl == 0:
        bits.append("z undeclared")
    else:
        bits.append(f"z declared {_n_decl}/{len(stages)} stages")
    fig.update_layout(
        height=height, margin=dict(l=50, r=20, t=52, b=96),
        # Legend BELOW the figure: at y=1.02 it shared the
        # top band with the title and the subplot titles, and at served
        # browser widths they collided. Below
        # the x-axis there is no other occupant at ANY width; the
        # standing convention (legends never overlap data)
        # holds by construction rather than by spacing luck.
        legend=dict(orientation="h", yanchor="top", y=-0.14,
                    xanchor="left", x=0, font=dict(size=9)),
        title=dict(text=" · ".join(bits), x=0.01, font=dict(size=12)))
    return fig
