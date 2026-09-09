"""Render primitives for the editor viewer.

WHY THIS IS PYTHON AND NOT JAVASCRIPT: the conventions that can silently
diverge (rotation sign and centre, ellipse sampling, cutout->hole
assignment) live HERE, verifiable headlessly against the rasterizer.
The JS side receives resolved polygons and extrudes/lathes them — a dumb
renderer cannot invent a second convention.

CONVENTIONS (measured against the tree):
  * rect `rotation_deg` draws the rectangle rotated CCW (+) about the
    SHAPE CENTRE.  Measured directly on `raster2d._rect_mask`: a +30 deg
    rect's occupied cells have principal axis at +29.96 deg, and the
    +30-rotated corner is inside while the -30 one is not.  The outline
    below applies R(+rot) about (x0+w/2, y0+h/2) to the four corners.
  * extrude axis: 'z' ONLY, matching `build_shapes3d`, which refuses any
    other axis rather than silently re-axising.
    The viewer refuses the same way — parity with the builder, not a
    viewer opinion.
  * planar with a DECLARED extent renders over `axial_extent_mm` if
    present else symmetric [-depth/2, +depth/2].  (This convention was
    previously cited here as "the same fallback extrude3d uses";
    an earlier `extrude3d` module was DELETED as an orphan with no
    importer anywhere in the tree, so the convention is stated here
    rather than pointed at.)

A cutout child becomes a HOLE of a positive shape only when its
footprint bbox lies inside that shape's bbox and (when extruded) its
z-range lies inside the shape's.  Anything that cannot be assigned is
NOT dropped: it ships as a `ghost` entry with the reason, and the
reason is repeated in `reported` so the legend states it (a skipped
item is a reported item).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ion_gym.edit.policy import EditRefusal
from ion_gym.edit.session import EditSession, normalize_color

# ellipse -> polygon sampling for display; display fidelity only (the
# rasterizer keeps using the implicit equation).
ELLIPSE_SEGMENTS = 64


# --------------------------------------------------------------------------
# single-shape outlines
# --------------------------------------------------------------------------
def shape_outline(shape: dict) -> List[List[float]]:
    """Closed CCW-ish polygon [[x, y], ...] for one positive shape dict.
    Refuses unknown types by name (a silent default would render an
    unknown shape as nothing)."""
    t = shape.get("type")
    if t == "rect":
        x0 = float(shape["x_mm"]); y0 = float(shape["y_mm"])
        w = float(shape["width_mm"]); h = float(shape["height_mm"])
        rot = float(shape.get("rotation_deg", 0.0) or 0.0)
        cx, cy = x0 + w / 2.0, y0 + h / 2.0
        ca, sa = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        out = []
        for px, py in ((x0, y0), (x0 + w, y0),
                       (x0 + w, y0 + h), (x0, y0 + h)):
            dx, dy = px - cx, py - cy
            out.append([cx + dx * ca - dy * sa, cy + dx * sa + dy * ca])
        return out
    if t == "ellipse":
        cx = float(shape["cx_mm"]); cy = float(shape["cy_mm"])
        rx = float(shape["rx_mm"]); ry = float(shape["ry_mm"])
        return [[cx + rx * math.cos(2 * math.pi * k / ELLIPSE_SEGMENTS),
                 cy + ry * math.sin(2 * math.pi * k / ELLIPSE_SEGMENTS)]
                for k in range(ELLIPSE_SEGMENTS)]
    if t == "polygon":
        pts = shape.get("points_mm") or []
        if len(pts) < 3:
            raise EditRefusal(
                f"polygon with {len(pts)} points_mm cannot be outlined")
        return [[float(p[0]), float(p[1])] for p in pts]
    raise EditRefusal(
        f"shape type {t!r} has no outline rule — refusing rather than "
        f"rendering it as nothing")


def _bbox(outline: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in outline]; ys = [p[1] for p in outline]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_inside(inner, outer, tol: float = 1e-9) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _extrude_of(shape: dict) -> Optional[dict]:
    ex = shape.get("extrude")
    if ex is None:
        return None
    if not isinstance(ex, dict):
        raise EditRefusal(f"extrude must be a dict, got "
                          f"{type(ex).__name__}")
    ax = ex.get("axis")
    if ax != "z":
        raise EditRefusal(
            f"extrude axis {ax!r}: the shapes3d builder supports 'z' only "
            f"and refuses others (build_shapes3d H8); the viewer refuses "
            f"the same rather than drawing a body the builder cannot build")
    return {"axis": "z", "lo_mm": float(ex["lo_mm"]),
            "hi_mm": float(ex["hi_mm"])}


# --------------------------------------------------------------------------
# electrode -> solids + ghosts + reports
# --------------------------------------------------------------------------
def electrode_primitives(shapes: List[dict]) -> dict:
    """Resolve one electrode's shape list into renderable primitives.

    Returns {"solids": [...], "ghosts": [...], "reported": [...]} where a
    solid is {"outline", "holes", "extrude"} and a ghost carries a
    "reason".  Every non-assignment is reported, never dropped.
    """
    positives: List[dict] = []
    reported: List[str] = []
    ghosts: List[dict] = []
    cut_children: List[dict] = []

    for j, sh in enumerate(shapes):
        if sh.get("type") == "cutout":
            kids = sh.get("children") or []
            if not kids:
                reported.append("cutout with no children: nothing to "
                                "subtract (reported, not dropped)")
            cut_children.extend((j, k) for k in kids)
        else:
            positives.append({
                "shape_index": j,           # RAW index for pick-to-select
                "outline": shape_outline(sh),
                "holes": [],
                "extrude": _extrude_of(sh),
            })

    for cut_idx, kid in cut_children:
        k_out = shape_outline(kid)
        k_ex = _extrude_of(kid)
        k_bb = _bbox(k_out)
        host = None
        for pos in positives:
            if not _bbox_inside(k_bb, _bbox(pos["outline"])):
                continue
            pe, ke = pos["extrude"], k_ex
            if (pe is None) != (ke is None):
                continue
            if pe is not None and not (ke["lo_mm"] >= pe["lo_mm"] - 1e-9
                                       and ke["hi_mm"] <= pe["hi_mm"]
                                       + 1e-9):
                continue
            host = pos
            break
        if host is not None:
            host["holes"].append(k_out)
        else:
            reason = (f"cutout child ({kid.get('type')}) not contained in "
                      f"any positive shape footprint/extent — drawn as an "
                      f"unassigned ghost, NOT subtracted")
            reported.append(reason)
            ghosts.append({"shape_index": cut_idx, "outline": k_out,
                           "holes": [], "extrude": k_ex,
                           "reason": reason})

    return {"solids": positives, "ghosts": ghosts, "reported": reported}


def shift_shape(shape: dict, dx: float = 0.0, dy: float = 0.0,
                dz: float = 0.0) -> dict:
    """Return a COPY of a shape dict translated by (dx, dy) in-plane and
    dz along the extrude axis.  One authority for the move operation:
    the gizmo drag, the gate, and any future nudge tool all shift a
    shape through here.  Refuses unknown types by name."""
    import copy as _copy
    out = _copy.deepcopy(shape)
    t = out.get("type")
    if t == "rect":
        out["x_mm"] = float(out["x_mm"]) + dx
        out["y_mm"] = float(out["y_mm"]) + dy
    elif t == "ellipse":
        out["cx_mm"] = float(out["cx_mm"]) + dx
        out["cy_mm"] = float(out["cy_mm"]) + dy
    elif t == "polygon":
        out["points_mm"] = [[float(a2) + dx, float(b2) + dy]
                            for a2, b2 in out["points_mm"]]
    elif t == "cutout":
        out["children"] = [shift_shape(k, dx, dy, dz)
                           for k in (out.get("children") or [])]
        return out
    else:
        raise EditRefusal(f"no shift rule for shape type {t!r}")
    ex = out.get("extrude")
    if isinstance(ex, dict) and dz:
        ex["lo_mm"] = float(ex["lo_mm"]) + dz
        ex["hi_mm"] = float(ex["hi_mm"]) + dz
    return out


def render_primitives(session: EditSession) -> dict:
    """Per-electrode primitives + document extents for the viewer.

    Colors come from the parsed spec's electrodes (the same authority
    viz_core draws with), not re-invented."""
    g = session.spec.geometry
    out_els = []
    all_x: List[float] = []
    all_y: List[float] = []
    for i, el in enumerate(g.electrodes):
        ok, reason = session.electrode_editable(i)
        raw_shapes = session.doc["geometry"]["electrodes"][i] \
            .get("shapes", [])
        prim = electrode_primitives(raw_shapes)
        if not ok:
            prim["reported"].append(reason)
        for s in prim["solids"] + prim["ghosts"]:
            bb = _bbox(s["outline"])
            all_x += [bb[0], bb[2]]; all_y += [bb[1], bb[3]]
        out_els.append({
            "index": i,
            "name": el.name,
            "color": normalize_color(el.color),
            "is_grid": bool(el.is_grid),
            "editable": ok,
            **prim,
        })
    if all_x:
        bbox2d = [min(all_x), min(all_y), max(all_x), max(all_y)]
    else:
        bbox2d = [0.0, 0.0, float(g.width_mm), float(g.height_mm)]
    return {"electrodes": out_els, "bbox2d": bbox2d,
            "pitch_mm": float(g.mm_per_gu)}
