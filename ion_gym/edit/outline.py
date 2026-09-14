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
  * extrude axis: ANY SINGLE axis of x|y|z (2026-09-12; supersedes
    the z-only refusal). The JS still extrudes along its own z only —
    that stays true — but the ShapeSpec convention stores outlines in
    the extrude axis's cyclic in-plane coordinates, so a whole deck
    extruded along world A renders exactly by handing outlines through
    unchanged and permuting the world-framed payload (trajectories,
    detections, mirror axes, axis labels) by the same cycle: viewer
    (x, y, z) displays world (*EXTRUDE_INPLANE_AXES[A], A). The frame
    rides the payload (`prims['frame']`, policy.viewer_frame); it is
    the identity for z. Decks MIXING extrude axes have no single frame
    and refuse by name at render_primitives (the builder still builds
    them; deck_multiview still draws them).
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

from ion_gym.edit.policy import (EditRefusal, deck_extrude_axis,
                                 viewer_frame)
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


def _clip_half(outline: List[List[float]],
               axis_idx: int) -> List[List[float]]:
    """Clip a closed polygon to coord[axis_idx] >= 0 — the stored half
    of a declared mirror plane (signed frame: the plane is at zero).

    This is the display-side twin of the rasterizer's clip to the
    stored domain: a shape may legally straddle the plane in the spec
    (the mirror supplies the other half), but the STORED geometry — and
    therefore the drawn solid, and any hole path THREE.js must
    triangulate inside it — is the >= 0 half. Sutherland–Hodgman
    against one line; a polygon already inside comes back with its
    vertex list UNCHANGED (no intersections are inserted), so every
    non-straddling deck renders byte-identically. An empty result
    (shape entirely in the mirrored half) is the caller's to report.
    """
    out: List[List[float]] = []
    n = len(outline)
    for i in range(n):
        a, b = outline[i], outline[(i + 1) % n]
        ain, bin_ = a[axis_idx] >= 0.0, b[axis_idx] >= 0.0
        if ain:
            out.append(a)
        if ain != bin_:
            t = a[axis_idx] / (a[axis_idx] - b[axis_idx])
            pt = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
            out.append(pt)
    return out


def _bbox_inside(inner, outer, tol: float = 1e-9) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _signed_area(poly: List[List[float]]) -> float:
    """Shoelace signed area: > 0 CCW (an outer contour in the boolean
    backend's convention), < 0 CW (a hole)."""
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def _point_in_poly(pt, poly: List[List[float]]) -> bool:
    """Ray-cast point-in-polygon (even-odd). Used only to assign a
    subtraction result's holes to their outer contours; the backend
    emits holes strictly interior to their outers, so edge grazing is
    not a live case there."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


def _manifold():
    """The 2-D boolean backend — an injectable seam so the gate can
    exercise the backend-absent degrade path. manifold3d is a DECLARED
    dependency (requirements.txt: trimesh's boolean backend, 'needed
    for bored plates'), so it is normally present."""
    import manifold3d
    return manifold3d


def _boolean_subtract(host_outline, host_holes, kid_outline):
    """(host outline + holes) minus kid, via manifold3d.CrossSection.

    For a cutout child that CROSSES its host's boundary (an open-sided
    bore: the StepWave's truncated plates), the true stored
    cross-section is a boolean difference — a THREE.Shape hole cannot
    cross its outline, so the hole model structurally cannot express
    it. Returns (solids, None) with solids = [(outline, holes), ...]
    (several when the kid SPLITS the host), or (None, reason) when the
    subtraction cannot be performed or would be a lie — the caller
    keeps the reported-ghost behavior, loud, never silent:
      * backend not importable (named; it is a declared dependency);
      * the kid touches nothing (a no-op subtraction would silently
        vanish the cutout with zero geometric effect);
      * the kid consumes the whole host (nothing stored to draw);
      * a result hole cannot be assigned to an outer contour.
    EvenOdd fill makes contour orientation irrelevant on input (user
    polygons carry no orientation contract); the OUTPUT convention is
    the backend's: CCW outers, CW holes — mapped by signed area.
    """
    try:
        _m = _manifold()
    except ImportError as e:
        return None, (f"boundary-crossing cutout needs the manifold3d "
                      f"boolean backend (a declared dependency; "
                      f"requirements.txt) — import failed here "
                      f"({e}); drawn as an unassigned ghost instead")
    cs = _m.CrossSection
    host = cs([host_outline] + list(host_holes), _m.FillRule.EvenOdd)
    diff = host - cs([kid_outline], _m.FillRule.EvenOdd)
    if abs(diff.area() - host.area()) < 1e-12:
        return None, ("cutout child overlaps its host's bounding box "
                      "but subtracts no area — drawn as an unassigned "
                      "ghost rather than silently vanishing")
    outers: List[tuple] = []
    holes: List[List[List[float]]] = []
    for p in diff.to_polygons():
        pts = [[float(q[0]), float(q[1])] for q in p]
        if _signed_area(pts) > 0:
            outers.append((pts, []))
        else:
            holes.append(pts)
    if not outers:
        return None, ("cutout child consumes the entire shape — "
                      "nothing stored to draw; reported, not dropped")
    for h in holes:
        for o_pts, o_holes in outers:
            if _point_in_poly(h[0], o_pts):
                o_holes.append(h)
                break
        else:
            # every else produces an outcome: an unassignable hole
            # means the mapping back to outline+holes would be wrong —
            # refuse the whole subtraction to the reported-ghost path.
            return None, ("boolean subtraction produced a hole contour "
                          "not containable in any outer contour — "
                          "unmappable to the viewer's outline+holes "
                          "model; drawn as an unassigned ghost")
    return outers, None


def _extrude_of(shape: dict) -> Optional[dict]:
    ex = shape.get("extrude")
    if ex is None:
        return None
    if not isinstance(ex, dict):
        raise EditRefusal(f"extrude must be a dict, got "
                          f"{type(ex).__name__}")
    ax = ex.get("axis")
    if ax not in ("x", "y", "z"):
        raise EditRefusal(
            f"extrude axis {ax!r} is not one of 'x'|'y'|'z' — the "
            f"shape cannot be outlined")
    # ANY single axis renders (2026-09-12, supersedes the z-only
    # refusal that stood here): outlines are stored in the extrude
    # axis's cyclic in-plane coordinates (the ShapeSpec convention),
    # so the JS extrudes them along ITS z unchanged and the payload
    # frame (policy.viewer_frame, attached by render_primitives)
    # relabels the viewer axes to the world names. Decks MIXING
    # extrude axes still refuse, by name, at the deck-level scan in
    # render_primitives — one frame per payload.
    return {"axis": ax, "lo_mm": float(ex["lo_mm"]),
            "hi_mm": float(ex["hi_mm"])}


# --------------------------------------------------------------------------
# electrode -> solids + ghosts + reports
# --------------------------------------------------------------------------
def electrode_primitives(shapes: List[dict],
                         inplane_clip: Tuple[int, ...] = ()) -> dict:
    """Resolve one electrode's shape list into renderable primitives.

    Returns {"solids": [...], "ghosts": [...], "reported": [...]} where a
    solid is {"outline", "holes", "extrude"} and a ghost carries a
    "reason".  Every non-assignment is reported, never dropped.

    `inplane_clip` names the in-plane coordinate indices (0/1) that
    carry a declared mirror plane at zero: outlines are clipped to the
    stored >= 0 half there (_clip_half), exactly as the rasterizer
    stores them — a shape may straddle the plane in the spec (the
    mirror supplies the image), but the drawn solid and any hole path
    THREE.js triangulates inside it are the stored half. Empty clips
    are reported by name, never silently dropped. No clip axes (the
    default, and every non-mirrored deck) leaves every outline
    byte-identical.
    """
    def _clipped(outline: List[List[float]], what: str):
        for ax in inplane_clip:
            outline = _clip_half(outline, ax)
            if not outline:
                return None, (f"{what} lies entirely in the mirrored "
                              f"half (in-plane coord {ax} < 0): the "
                              f"stored fraction draws nothing of it — "
                              f"reported, not dropped")
        return outline, None

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
            _out, _why = _clipped(shape_outline(sh),
                                  f"shape {j} ({sh.get('type')})")
            if _out is None:
                reported.append(_why)
                continue
            positives.append({
                "shape_index": j,           # RAW index for pick-to-select
                "outline": _out,
                "holes": [],
                "extrude": _extrude_of(sh),
            })

    def _extrude_compatible(pe, ke) -> bool:
        """A kid can only bore a host it shares extrusion with: same
        None-ness, same axis, kid range within host range."""
        if (pe is None) != (ke is None):
            return False
        if pe is None:
            return True
        return (ke["axis"] == pe["axis"]
                and ke["lo_mm"] >= pe["lo_mm"] - 1e-9
                and ke["hi_mm"] <= pe["hi_mm"] + 1e-9)

    def _bbox_overlap(a, b) -> bool:
        return not (a[2] < b[0] or b[2] < a[0]
                    or a[3] < b[1] or b[3] < a[1])

    for cut_idx, kid in cut_children:
        k_out, _why = _clipped(shape_outline(kid),
                               f"cutout child ({kid.get('type')})")
        if k_out is None:
            reported.append(_why)
            continue
        k_ex = _extrude_of(kid)
        k_bb = _bbox(k_out)
        host = None
        for pos in positives:
            if not _bbox_inside(k_bb, _bbox(pos["outline"])):
                continue
            if not _extrude_compatible(pos["extrude"], k_ex):
                continue
            host = pos
            break
        if host is not None:
            host["holes"].append(k_out)
            continue
        # BOUNDARY SUBTRACTION (2026-09-12, Brian's browser pass on
        # the StepWave: 112 truncated plates rendered as SOLID slabs
        # because their open-sided bores could only ghost). A kid that
        # crosses its host's boundary is a boolean difference of the
        # outline itself — the hole model cannot express it. Candidate
        # = the ONE extrude-compatible positive whose footprint
        # overlaps; ambiguity, a no-op, or a failed mapping keeps the
        # reported-ghost path with the reason named. This path never
        # fires on a deck whose cutouts all host as holes (measured:
        # zero ghosts across every shipped example), so certified
        # z-deck payloads are untouched.
        cands = [pos for pos in positives
                 if _extrude_compatible(pos["extrude"], k_ex)
                 and _bbox_overlap(k_bb, _bbox(pos["outline"]))]
        if len(cands) == 1:
            solids, _why2 = _boolean_subtract(
                cands[0]["outline"], cands[0]["holes"], k_out)
            if solids is not None:
                first_o, first_h = solids[0]
                cands[0]["outline"] = first_o
                cands[0]["holes"] = first_h
                for extra_o, extra_h in solids[1:]:
                    # the kid SPLIT the host: each piece is a solid of
                    # the same raw shape (picking either selects it)
                    positives.append({
                        "shape_index": cands[0]["shape_index"],
                        "outline": extra_o, "holes": extra_h,
                        "extrude": cands[0]["extrude"]})
                continue
            reason = (f"cutout child ({kid.get('type')}): {_why2}")
        elif len(cands) > 1:
            reason = (f"cutout child ({kid.get('type')}) overlaps "
                      f"{len(cands)} extrude-compatible shapes — "
                      f"ambiguous subtraction target; drawn as an "
                      f"unassigned ghost, NOT subtracted")
        else:
            reason = (f"cutout child ({kid.get('type')}) not contained "
                      f"in any positive shape footprint/extent — drawn "
                      f"as an unassigned ghost, NOT subtracted")
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
    viz_core draws with), not re-invented.

    The returned dict carries `frame` (policy.viewer_frame): which
    WORLD axis each viewer axis displays. It is the identity for
    z-extruded and extrude-less decks — their payloads are unchanged
    except for this additive key — and the cyclic permutation for an
    x/y-extruded deck. A deck mixing extrude axes refuses by name
    here (deck_extrude_axis), before anything renders."""
    g = session.spec.geometry
    axis = deck_extrude_axis(
        (el.name, session.doc["geometry"]["electrodes"][i]
         .get("shapes", []))
        for i, el in enumerate(g.electrodes))
    frame = viewer_frame(axis)
    # DECLARED MIRRORS on the deck's in-plane WORLD axes clip outlines
    # to the stored >= 0 half (the signed-frame convention: plane at
    # zero) — the display-side twin of the rasterizer's stored-domain
    # clip, so drawn equals stored even for shapes that straddle the
    # plane (the StepWave's bores are centred ON their world-z plane,
    # which is the stored in-plane y here). A mirror on the EXTRUDE
    # axis needs no in-plane clip. No declared mirrors: no clip, every
    # outline byte-identical.
    inplane_clip = tuple(
        i for i, w in enumerate(frame["axes"][:2])
        if g.symmetry.planes.get(w) == "mirror")
    out_els = []
    all_x: List[float] = []
    all_y: List[float] = []
    for i, el in enumerate(g.electrodes):
        ok, reason = session.electrode_editable(i)
        raw_shapes = session.doc["geometry"]["electrodes"][i] \
            .get("shapes", [])
        prim = electrode_primitives(raw_shapes, inplane_clip=inplane_clip)
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
            "pitch_mm": float(g.mm_per_gu),
            "frame": frame}
