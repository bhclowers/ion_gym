"""Native-shape decks -> JupyterCAD ``.jcad`` documents (L-466).

The REVERSE of the CAD->STL import path: a deck whose electrodes are
declared as native shapes (rect / circular ellipse / cutout, with
extrude descriptors) becomes a parametric JupyterCAD document the PI
can open, edit and re-export. Emitted against jcad schema 3.0.0 and
validated (see spec_to_jcad's validate flag) by constructing the
official pydantic model jupytercad_core ships, which carries
``extra='forbid'`` on every object -- a stricter check than any GUI
smoke test.

WHAT MAPS, exactly:
  rect            -> Part::Box   (any extrude axis unrotated; rotation
                                  about the extrude axis for axis "z")
  ellipse rx==ry  -> Part::Cylinder (any extrude axis)
  polygon         -> Sketcher::SketchObject (a closed loop of
                     Part::GeomLineSegment in world mm) extruded by
                     Part::Extrusion along the extrude axis
  electrode       -> its positive shapes fused (Part::MultiFuse when
                     more than one), then every cutout CHILD subtracted
                     as a Part::Cut chain -- exactly build_shapes3d's
                     union-then-subtract semantics
                     (the final body is always NAMED for the
                     electrode; intermediates are _s<i> / _fuse)
  declared mirror -> the mirrored half is EMITTED as real objects
                     (name__mirror_<axis>, one suffix per plane so
                     two planes give four unique bodies), because CAD
                     wants the whole
                     instrument; stated in the document metadata.

WHAT REFUSES, by name (never approximated):
  true ellipses (rx != ry -- jcad sketch geometry has circles and line
  segments only), rotated rects on x/y extrude axes, shapes with no
  extrude descriptor (a 2-D deck is not a solid), and unknown types.

Frames: shape coordinates are the deck's STORED mm frame; the extrude
cross-section plane follows sim_spec's CYCLIC order -- axis a maps the
shape's (first, second) in-plane coordinates onto world axes
(a+1) % 3 and (a+2) % 3. Colors follow each electrode's declared
color. Group membership is recorded per object in metadata (jcad
objects carry no free-form tags).

STEP (jcad_to_step): a .jcad document -> a STEP assembly, by evaluating
the document's own object graph in CadQuery. The STEP is therefore
derived from the jcad, never re-derived from the deck: ONE mapping
authority (spec_to_jcad) feeds both files, so they cannot disagree.
Because it reads any jcad document, it also converts a file edited in
the JupyterCAD GUI. Every ROOT object (one no other object consumes)
becomes one named solid in the assembly, hidden or not; a hidden root is
exported and reported, never dropped. CadQuery is an OPTIONAL author-
time dependency (extra "cad"): imported inside the function and refused
with the install hint if absent, so importing this module needs nothing.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_AXIS_I = {"x": 0, "y": 1, "z": 2}

# cylinder local axis is +Z; Placement (Axis, Angle) rotating +Z onto
# each world axis. Verified numerically in the L-466 probe.
_CYL_ROT = {"z": ([0.0, 0.0, 1.0], 0.0),
            "x": ([0.0, 1.0, 0.0], 90.0),
            "y": ([1.0, 0.0, 0.0], -90.0)}


# Named default palette for colors="cycle": matplotlib's tab10, as hex,
# so neighbouring electrodes in deck order are visually distinct. Pass
# palette= to use another; entries must be '#rrggbb'.
CYCLE_PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

# A mirrored copy is named <name>__mirror_<axis>, one suffix per plane, so
# two declared planes give four UNIQUE bodies (X, X__mirror_y,
# X__mirror_z, X__mirror_y__mirror_z). mirror_base() recovers X.
MIRROR_SUFFIX = "__mirror"


def mirror_suffix(axis):
    return f"{MIRROR_SUFFIX}_{axis}"


def mirror_base(name):
    """The electrode name behind a (possibly multiply) mirrored copy."""
    return name.split(MIRROR_SUFFIX, 1)[0]


def _hex(color):
    """Electrode [r, g, b] (0-255) or '#rrggbb' -> '#rrggbb';
    None -> jcad grey."""
    if not color:
        return "#808080"
    if isinstance(color, str):
        return _check_hex(color, "color")
    r, g, b = (int(c) for c in color[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _check_hex(value, where):
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(value)):
        raise ValueError(f"{where}: {value!r} is not a '#rrggbb' color")
    return str(value).lower()


def cycled_colors(names, palette=CYCLE_PALETTE):
    """Assign palette colors cyclically by BASE name (a mirrored copy,
    see mirror_base, shares its twin's color), in order of first
    appearance. The single authority for colors="cycle" in both
    spec_to_jcad and jcad_to_step, so a cycled .jcad and a STEP cycled
    from any document of the same deck color every electrode alike."""
    pal = [_check_hex(c, "palette entry") for c in palette]
    if not pal:
        raise ValueError("palette is empty")
    out = {}
    for name in names:
        base = mirror_base(name)
        if base not in out:
            out[base] = pal[len(out) % len(pal)]
    return out


def _placement(pos, axis=(0.0, 0.0, 1.0), angle=0.0):
    return {"Position": [float(v) for v in pos],
            "Axis": [float(v) for v in axis],
            "Angle": float(angle)}


def _in_plane(axis, first, second, along):
    """CYCLIC frame: place the shape's (first, second) in-plane values
    and the along-axis value onto world xyz."""
    a = _AXIS_I[axis]
    out = [0.0, 0.0, 0.0]
    out[(a + 1) % 3] = float(first)
    out[(a + 2) % 3] = float(second)
    out[a] = float(along)
    return out


def _shape_objects(shape, ele_name, color, idx):
    """One ShapeSpec -> a list of (name, jcad_object) pairs plus the
    name of the pair that IS the electrode's solid (for cut chains).
    Refuses unsupported geometry naming the electrode and shape."""
    where = f"electrode {ele_name!r} shape[{idx}] ({shape.type})"
    ext = shape.extrude()
    if shape.type in ("rect", "ellipse", "polygon") and ext is None:
        raise ValueError(
            f"{where}: no extrude descriptor -- a 2-D deck is not a "
            f"solid; only 3-D native-shape decks export to jcad")
    if shape.type == "rect":
        p = shape.params
        w = float(p["width_mm"])
        h = float(p["height_mm"])
        rot = float(p.get("rotation_deg", 0.0) or 0.0)
        lo, hi = float(ext["lo_mm"]), float(ext["hi_mm"])
        a = _AXIS_I[ext["axis"]]
        dims = [0.0, 0.0, 0.0]
        dims[(a + 1) % 3] = w
        dims[(a + 2) % 3] = h
        dims[a] = hi - lo
        corner = _in_plane(ext["axis"], p["x_mm"], p["y_mm"], lo)
        if rot == 0.0:
            plc = _placement(corner)
        elif ext["axis"] == "z":
            # rect rotates about its CENTER (raster2d convention); a
            # jcad Box rotates about its Placement corner, so the
            # corner is re-derived from the rotated center offset
            cx = float(p["x_mm"]) + w / 2.0
            cy = float(p["y_mm"]) + h / 2.0
            th = math.radians(rot)
            ox = -w / 2.0 * math.cos(th) + h / 2.0 * math.sin(th)
            oy = -w / 2.0 * math.sin(th) - h / 2.0 * math.cos(th)
            plc = _placement([cx + ox, cy + oy, lo],
                             axis=[0.0, 0.0, 1.0], angle=rot)
        else:
            raise ValueError(
                f"{where}: rotation_deg={rot:g} on extrude axis "
                f"{ext['axis']!r} has no jcad mapping here (rotation "
                f"is supported on axis 'z' only)")
        name = f"{ele_name}_s{idx}"
        return [(name, {
            "name": name, "visible": True, "shape": "Part::Box",
            "parameters": {"Length": dims[0], "Width": dims[1],
                           "Height": dims[2], "Color": _hex(color),
                           "Placement": plc}})], name
    if shape.type == "ellipse":
        p = shape.params
        rx, ry = float(p["rx_mm"]), float(p["ry_mm"])
        if abs(rx - ry) > 1e-12:
            raise ValueError(
                f"{where}: true ellipse (rx {rx:g} != ry {ry:g} mm) "
                f"has no jcad primitive; refusing rather than "
                f"approximating")
        lo, hi = float(ext["lo_mm"]), float(ext["hi_mm"])
        axis_v, ang = _CYL_ROT[ext["axis"]]
        base = _in_plane(ext["axis"], p["cx_mm"], p["cy_mm"], lo)
        name = f"{ele_name}_s{idx}"
        return [(name, {
            "name": name, "visible": True, "shape": "Part::Cylinder",
            "parameters": {"Radius": rx, "Height": hi - lo,
                           "Angle": 360.0, "Color": _hex(color),
                           "Placement": _placement(base, axis_v, ang)}}
                 )], name
    if shape.type == "polygon":
        pts = [tuple(map(float, q)) for q in shape.params["points_mm"]]
        if len(pts) < 3:
            raise ValueError(f"{where}: polygon needs >= 3 vertices")
        if pts[0] == pts[-1]:
            pts = pts[:-1]           # stored closed; segments re-close
        lo, hi = float(ext["lo_mm"]), float(ext["hi_mm"])
        world = [_in_plane(ext["axis"], u, v, lo) for u, v in pts]
        segs = []
        for k in range(len(world)):
            a3 = world[k]
            b3 = world[(k + 1) % len(world)]
            segs.append({"TypeId": "Part::GeomLineSegment",
                         "StartX": a3[0], "StartY": a3[1],
                         "StartZ": a3[2], "EndX": b3[0],
                         "EndY": b3[1], "EndZ": b3[2]})
        sk_name = f"{ele_name}_s{idx}_sketch"
        ex_name = f"{ele_name}_s{idx}"
        direction = [0.0, 0.0, 0.0]
        direction[_AXIS_I[ext["axis"]]] = 1.0
        ident = _placement([0.0, 0.0, 0.0])
        return [(sk_name, {
            "name": sk_name, "visible": False,
            "shape": "Sketcher::SketchObject",
            "parameters": {"AttachmentOffset": ident, "Geometry": segs,
                           "Color": _hex(color), "Placement": ident}}),
                (ex_name, {
            "name": ex_name, "visible": True, "shape": "Part::Extrusion",
            "parameters": {"Base": sk_name, "Dir": direction,
                           "LengthFwd": hi - lo, "LengthRev": 0.0,
                           "Solid": True, "Color": _hex(color),
                           "Placement": ident},
            "dependencies": [sk_name]})], ex_name
    raise ValueError(f"{where}: shape type {shape.type!r} has no jcad "
                     f"mapping (supported: rect, circular ellipse, "
                     f"polygon, cutout)")


def _electrode_objects(e, color):
    """One electrode -> its jcad objects, following build_shapes3d:
    positive shapes UNION (MultiFuse when more than one), cutout
    children SUBTRACT (Cut chain). Returns the object list; only the
    electrode's final solid is visible."""
    positives = []
    tools = []
    for i, sh in enumerate(e.shapes):
        if sh.type == "cutout":
            for k, ch in enumerate(sh.children):
                objs, nm = _shape_objects(ch, f"{e.name}_cut{i}",
                                          color, k)
                tools += [(objs, nm)]
        else:
            positives.append(_shape_objects(sh, e.name, color, i))
    if not positives:
        raise ValueError(f"electrode {e.name!r}: only cutout shapes -- "
                         f"nothing to subtract from")
    out = []
    # The electrode's FINAL body always carries the electrode's name, on
    # every path (one shape, several fused, cutouts or not), so the
    # document and anything built from it (jcad_to_step) can say which
    # part is which electrode. Intermediates keep their _s<i>/_fuse names.
    if len(positives) == 1 and not tools:
        (pairs, solid) = positives[0]
        objs = [o for _, o in pairs]
        for o in objs:
            if o["name"] == solid:
                o["name"] = e.name
        return objs
    for pairs, _nm in positives:
        for _n, o in pairs:
            o["visible"] = False
            out.append(o)
    if len(positives) > 1:
        solid = e.name if not tools else f"{e.name}_fuse"
        out.append({"name": solid, "visible": not tools,
                    "shape": "Part::MultiFuse",
                    "parameters": {"Shapes": [nm for _p, nm in positives],
                                   "Refine": False, "Color": _hex(color),
                                   "Placement": _placement([0, 0, 0])},
                    "dependencies": [nm for _p, nm in positives]})
    else:
        solid = positives[0][1]
    for k, (objs, nm) in enumerate(tools):
        for _n, o in objs:
            o["visible"] = False
            out.append(o)
        cut_name = e.name if k == len(tools) - 1 else f"{e.name}_cut{k}"
        out.append({"name": cut_name, "visible": k == len(tools) - 1,
                    "shape": "Part::Cut",
                    "parameters": {"Base": solid, "Tool": nm,
                                   "Refine": False, "Color": _hex(color),
                                   "Placement": _placement([0, 0, 0])},
                    "dependencies": [solid, nm]})
        solid = cut_name
    return out


def _mirror_object(obj, axis):
    """Mirrored COPY of a jcad object about plane axis=0. Supported for
    the objects this module emits; anything else refuses."""
    a = _AXIS_I[axis]
    sfx = mirror_suffix(axis)
    m = json.loads(json.dumps(obj))
    m["name"] = obj["name"] + sfx
    prm = m["parameters"]
    plc = prm["Placement"]
    if m["shape"] == "Part::Box":
        if plc["Angle"] not in (0, 0.0) and a != 2:
            raise ValueError(f"{obj['name']}: mirrored rotated box on "
                             f"axis {axis!r} is not supported")
        span = {0: "Length", 1: "Width", 2: "Height"}[a]
        plc["Position"][a] = -(plc["Position"][a] + prm[span])
        if plc["Angle"] and a == 2:
            plc["Position"][2] = -(plc["Position"][2] + prm["Height"])
    elif m["shape"] == "Part::Cylinder":
        av = plc["Axis"]
        along = abs(av[a]) > 0.5 and plc["Angle"] in (0.0, 90.0, -90.0)
        if a == 2 and plc["Angle"] == 0.0:
            plc["Position"][2] = -(plc["Position"][2] + prm["Height"])
        elif along:
            plc["Position"][a] = -(plc["Position"][a] + prm["Height"])
        else:
            plc["Position"][a] = -plc["Position"][a]
    elif m["shape"] == "Part::Cut":
        m["parameters"]["Base"] = prm["Base"] + sfx
        m["parameters"]["Tool"] = prm["Tool"] + sfx
        m["dependencies"] = [d + sfx for d in obj.get(
            "dependencies", [])]
    elif m["shape"] == "Sketcher::SketchObject":
        L = "XYZ"[a]
        for seg in prm["Geometry"]:
            seg["Start" + L] = -seg["Start" + L]
            seg["End" + L] = -seg["End" + L]
    elif m["shape"] == "Part::Extrusion":
        m["parameters"]["Base"] = prm["Base"] + sfx
        m["parameters"]["Dir"] = [(-v if k == a else v)
                                  for k, v in enumerate(prm["Dir"])]
        m["dependencies"] = [d + sfx for d in obj.get(
            "dependencies", [])]
    elif m["shape"] == "Part::MultiFuse":
        m["parameters"]["Shapes"] = [n + sfx for n in prm["Shapes"]]
        m["dependencies"] = [d + sfx for d in obj.get(
            "dependencies", [])]
    else:
        raise ValueError(f"{obj['name']}: no mirror rule for "
                         f"{m['shape']}")
    return m


def _mirror_axes(g):
    sym = getattr(g, "symmetry", None)
    planes = dict(getattr(sym, "planes", None)
                  or (sym.get("planes", {}) if isinstance(sym, dict)
                      else {}))
    return [ax for ax, kind in planes.items() if kind == "mirror"]


def final_body_names(spec, *, expand_mirror=True, skipped=()):
    """The final-body names spec_to_jcad writes for this deck: one per
    electrode (minus `skipped`), times every declared mirror plane when
    expanded. The single authority for checking a written document."""
    names = [e.name for e in spec.geometry.electrodes
             if e.name not in set(skipped)]
    if expand_mirror:
        for ax in _mirror_axes(spec.geometry):
            names += [n + mirror_suffix(ax) for n in list(names)]
    return names


def spec_to_jcad(spec, path, *, expand_mirror=True, validate=True,
                 on_unsupported="refuse", colors="deck",
                 palette=CYCLE_PALETTE):
    """Export a native-shapes SimSpec to a JupyterCAD document at
    ``path``. Returns the written Path. Refuses unsupported geometry
    by electrode and shape name (module docstring lists the limits).

    expand_mirror: emit the mirrored half of a declared mirror plane as
        real objects (CAD wants the whole instrument); False keeps only
        the stored half.
    validate: construct jupytercad_core's official IJCadContent pydantic
        model (extra='forbid') from the emitted JSON -- schema
        validation without a GUI. Requires jupytercad-core; refuses
        with the install hint if absent rather than skipping silently.
    on_unsupported: "refuse" (default) raises on the first electrode
        with unmappable geometry; "skip" OMITS such electrodes, prints
        each with its reason, and records them in the document's
        metadata["skipped"] -- a skipped item is a reported item, never
        a silent hole.
    colors: "deck" (default) uses each electrode's declared color;
        "cycle" assigns ``palette`` cyclically in deck (basis) order via
        cycled_colors, so neighbouring electrodes differ. Every object
        of an electrode, and its mirrored twin, share one color. The
        mode is recorded in the document metadata.
    """
    if on_unsupported not in ("refuse", "skip"):
        raise ValueError(f"on_unsupported={on_unsupported!r}: use "
                         f"'refuse' or 'skip'")
    if colors not in ("deck", "cycle"):
        raise ValueError(f"colors={colors!r}: use 'deck' or 'cycle'")
    g = spec.geometry
    cycle = (cycled_colors([e.name for e in g.electrodes], palette)
             if colors == "cycle" else None)
    objects = []
    groups = {}
    skipped = {}
    for e in g.electrodes:
        if not getattr(e, "shapes", None):
            raise ValueError(
                f"electrode {e.name!r} declares no native shapes -- "
                f"this exporter covers native-shape decks only (STL "
                f"decks already live in CAD)")
        try:
            objs = _electrode_objects(
                e, cycle[e.name] if cycle is not None else e.color)
        except ValueError as err:
            if on_unsupported == "refuse":
                raise
            skipped[e.name] = str(err)
            print(f"[jcad_export] SKIPPED electrode {e.name!r}: {err}",
                  flush=True)
            continue
        objects += objs
        for o in objs:
            groups[o["name"]] = ",".join(
                getattr(e, "rf_groups", []) or []) or "DC"
    mirrors = _mirror_axes(g)
    if expand_mirror:
        for ax in mirrors:
            objects += [_mirror_object(o, ax) for o in list(objects)]
    # Checked AFTER mirror expansion: every object name, generated or
    # mirrored, must be unique or the document is ambiguous.
    seen = set()
    for o in objects:
        if o["name"] in seen:
            raise ValueError(
                f"jcad object name {o['name']!r} would be written twice "
                f"(an electrode name collides with a generated or mirrored "
                f"object name); rename the electrode")
        seen.add(o["name"])
    meta = {"generator": "ion_gym.io.jcad_export (L-466)",
            "deck": str(getattr(spec, "name", "") or ""),
            "units": "mm",
            "colors": colors,
            "mirror_planes": ",".join(mirrors) or "none",
            "mirror_expanded": str(bool(expand_mirror and mirrors)),
            "groups": json.dumps(groups)}
    if skipped:
        meta["skipped"] = json.dumps(skipped)
    doc = {"schemaVersion": "3.0.0", "objects": objects,
           "metadata": meta}
    if validate:
        try:
            from jupytercad_core.schema import IJCadContent
        except ImportError as err:
            raise ImportError(
                "validate=True needs jupytercad-core (pip install "
                "jupytercad-core); pass validate=False to emit "
                "unvalidated") from err
        IJCadContent(**doc)          # extra='forbid': schema-strict
    path = Path(path)
    path.write_text(json.dumps(doc, indent=1))
    return path


# jcad Placement: rotation of Angle degrees about Axis through the
# ORIGIN, then translation by Position (FreeCAD convention).
# cq.Location(t, axis, angle) composes the same T*R.
def _cq_location(cq, placement, where):
    for key in ("Position", "Axis", "Angle"):
        if key not in placement:
            raise ValueError(f"{where}: Placement lacks {key!r}")
    axis = [float(v) for v in placement["Axis"]]
    if math.hypot(*axis) == 0.0:
        raise ValueError(f"{where}: Placement Axis is the zero vector")
    return cq.Location(cq.Vector(*map(float, placement["Position"])),
                       cq.Vector(*axis), float(placement["Angle"]))


def _is_identity(placement):
    return (all(float(v) == 0.0 for v in placement["Position"])
            and float(placement["Angle"]) == 0.0)


def _sketch_face(cq, obj, tol_mm):
    """Sketcher::SketchObject of Part::GeomLineSegment -> one planar
    face. The segments must form ONE closed loop, in order, end to start
    within tol_mm; anything else refuses by name (arcs, several loops or
    a gap are not something this reader invents)."""
    name = obj["name"]
    prm = obj["parameters"]
    off = prm.get("AttachmentOffset")
    if off is not None and not _is_identity(off):
        raise ValueError(f"sketch {name!r}: non-identity AttachmentOffset "
                         f"has no mapping here; refusing")
    segs = prm.get("Geometry") or []
    if len(segs) < 3:
        raise ValueError(f"sketch {name!r}: {len(segs)} segments cannot "
                         f"close a face")
    edges = []
    for k, seg in enumerate(segs):
        kind = seg.get("TypeId")
        if kind != "Part::GeomLineSegment":
            raise ValueError(f"sketch {name!r} segment {k}: geometry "
                             f"{kind!r} is not supported (line segments "
                             f"only)")
        start = [float(seg["Start" + c]) for c in "XYZ"]
        end = [float(seg["End" + c]) for c in "XYZ"]
        nxt = segs[(k + 1) % len(segs)]
        nxt_start = [float(nxt["Start" + c]) for c in "XYZ"]
        gap = math.dist(end, nxt_start)
        if gap > tol_mm:
            raise ValueError(f"sketch {name!r}: segment {k} ends "
                             f"{gap:.3g} mm from the start of segment "
                             f"{(k + 1) % len(segs)} (tolerance "
                             f"{tol_mm:g} mm); not one closed loop")
        edges.append(cq.Edge.makeLine(cq.Vector(*start),
                                      cq.Vector(*nxt_start)))
    face = cq.Face.makeFromWires(cq.Wire.assembleEdges(edges))
    return face.moved(_cq_location(cq, prm["Placement"],
                                   f"sketch {name!r}"))


def _build_jcad_object(cq, name, table, memo, tol_mm):
    """Evaluate one jcad object (recursively through its inputs) to a
    CadQuery shape. Supported: exactly the vocabulary spec_to_jcad
    emits; any other shape type refuses naming the object."""
    if name in memo:
        return memo[name]
    if name not in table:
        raise ValueError(f"jcad object {name!r} is referenced but not "
                         f"defined in the document")
    obj = table[name]
    kind = obj.get("shape")
    prm = obj.get("parameters", {})
    where = f"jcad object {name!r} ({kind})"
    if kind == "Part::Box":
        shp = cq.Solid.makeBox(float(prm["Length"]), float(prm["Width"]),
                               float(prm["Height"]))
        shp = shp.moved(_cq_location(cq, prm["Placement"], where))
    elif kind == "Part::Cylinder":
        shp = cq.Solid.makeCylinder(float(prm["Radius"]),
                                    float(prm["Height"]),
                                    angleDegrees=float(prm.get("Angle",
                                                               360.0)))
        shp = shp.moved(_cq_location(cq, prm["Placement"], where))
    elif kind == "Sketcher::SketchObject":
        shp = _sketch_face(cq, obj, tol_mm)
    elif kind == "Part::Extrusion":
        if prm.get("Solid") is not True:
            raise ValueError(f"{where}: Solid is not True; a surface "
                             f"extrusion is not a conductor")
        base = _build_jcad_object(cq, prm["Base"], table, memo, tol_mm)
        if not isinstance(base, cq.Face):
            raise ValueError(f"{where}: Base {prm['Base']!r} is not a "
                             f"sketch face")
        direction = cq.Vector(*map(float, prm["Dir"]))
        if direction.Length == 0.0:
            raise ValueError(f"{where}: Dir is the zero vector")
        direction = direction.normalized()
        fwd = float(prm.get("LengthFwd", 0.0))
        rev = float(prm.get("LengthRev", 0.0))
        if fwd + rev <= 0.0:
            raise ValueError(f"{where}: LengthFwd + LengthRev = "
                             f"{fwd + rev:g} mm; nothing to extrude")
        start = base.translate(direction * (-rev)) if rev else base
        shp = cq.Solid.extrudeLinear(start.outerWire(), start.innerWires(),
                                     direction * (fwd + rev))
        shp = shp.moved(_cq_location(cq, prm["Placement"], where))
    elif kind == "Part::MultiFuse":
        parts = [_build_jcad_object(cq, n, table, memo, tol_mm)
                 for n in prm["Shapes"]]
        if len(parts) < 2:
            raise ValueError(f"{where}: needs >= 2 Shapes, got "
                             f"{len(parts)}")
        shp = parts[0].fuse(*parts[1:])
        if prm.get("Refine"):
            shp = shp.clean()
        shp = shp.moved(_cq_location(cq, prm["Placement"], where))
    elif kind == "Part::Cut":
        base = _build_jcad_object(cq, prm["Base"], table, memo, tol_mm)
        tool = _build_jcad_object(cq, prm["Tool"], table, memo, tol_mm)
        shp = base.cut(tool)
        if prm.get("Refine"):
            shp = shp.clean()
        shp = shp.moved(_cq_location(cq, prm["Placement"], where))
    else:
        raise ValueError(f"{where}: no STEP mapping for shape type "
                         f"{kind!r} (supported: Part::Box, Part::Cylinder, "
                         f"Sketcher::SketchObject, Part::Extrusion, "
                         f"Part::MultiFuse, Part::Cut)")
    memo[name] = shp
    return shp


def _consumed_names(obj):
    """Names an object takes as INPUT: its dependencies plus the
    parameter fields that reference other objects."""
    prm = obj.get("parameters", {})
    used = set(obj.get("dependencies", []) or [])
    for key in ("Base", "Tool"):
        if isinstance(prm.get(key), str):
            used.add(prm[key])
    used.update(prm.get("Shapes", []) or [])
    return used


def jcad_to_step(jcad, step_path, *, close_tol_mm=1e-6,
                 colors="document", palette=CYCLE_PALETTE):
    """Write a STEP assembly from a JupyterCAD document; return a census.

    jcad: a path to a .jcad file, or the already-parsed document dict.
    step_path: output .step path. Written unconditionally -- the caller
        owns the overwrite policy.
    close_tol_mm: largest allowed gap between one sketch segment's end
        and the next one's start. spec_to_jcad writes exactly closed
        loops, so the default is float-noise sized; a GUI-edited file
        with a real gap refuses rather than being silently closed.
    colors: "document" (default) keeps each root's Color from the jcad;
        "cycle" recolors the parts with ``palette`` via cycled_colors in
        document order (a mirrored twin shares its part's color), for
        any jcad -- including one written with colors="deck".

    Every ROOT object becomes one assembly part named after it and
    colored by its Color. Each part must be a valid solid of positive
    volume or the call refuses naming it. A part may hold several
    disjoint solids (an electrode made of separate metal pieces that
    share one voltage, e.g. a tile slotted around a rail); n_solids
    counts them. Returns one dict per part: name, visible, n_solids,
    color (as written), volume_mm3, bbox_mm ((xmin, ymin, zmin), (xmax, ymax, zmax)).
    """
    try:
        import cadquery as cq
    except ImportError as err:
        raise ImportError(
            "jcad_to_step needs CadQuery (pip install cadquery, or the "
            "ion_gym 'cad' extra)") from err
    if colors not in ("document", "cycle"):
        raise ValueError(f"colors={colors!r}: use 'document' or 'cycle'")
    if isinstance(jcad, dict):
        doc = jcad
    else:
        doc = json.loads(Path(jcad).read_text())
    objs = doc.get("objects")
    if not objs:
        raise ValueError("jcad document has no objects")
    table = {}
    for o in objs:
        if o["name"] in table:
            raise ValueError(f"jcad object name {o['name']!r} is defined "
                             f"twice")
        table[o["name"]] = o
    consumed = set()
    for o in objs:
        consumed |= _consumed_names(o)
    roots = [o for o in objs if o["name"] not in consumed]
    if not roots:
        raise ValueError("jcad document has no root objects (every "
                         "object is consumed by another)")
    deck_name = str(doc.get("metadata", {}).get("deck", "") or "")
    assy = cq.Assembly(name=deck_name or Path(step_path).stem)
    cycle = (cycled_colors([o["name"] for o in roots], palette)
             if colors == "cycle" else None)
    memo = {}
    census = []
    for o in roots:
        shp = _build_jcad_object(cq, o["name"], table, memo, close_tol_mm)
        if not shp.isValid():
            raise ValueError(f"root {o['name']!r}: the built shape is "
                             f"not a valid solid")
        vol = shp.Volume()
        if vol <= 0.0:
            raise ValueError(f"root {o['name']!r}: volume {vol:g} mm^3; "
                             f"not a solid")
        if not o.get("visible", True):
            print(f"[jcad_to_step] root {o['name']!r} is HIDDEN in the "
                  f"document; exported anyway (a root is a final body)",
                  flush=True)
        if cycle is not None:
            color = cycle[mirror_base(o["name"])]
        else:
            color = _check_hex(o.get("parameters", {}).get(
                "Color", "#808080"), f"root {o['name']!r} Color")
        rgb = [int(color[k:k + 2], 16) / 255.0 for k in (1, 3, 5)]
        assy.add(shp, name=o["name"], color=cq.Color(*rgb))
        bb = shp.BoundingBox()
        census.append({"name": o["name"],
                       "visible": bool(o.get("visible", True)),
                       "n_solids": len(shp.Solids()),
                       "color": color,
                       "volume_mm3": vol,
                       "bbox_mm": ((bb.xmin, bb.ymin, bb.zmin),
                                   (bb.xmax, bb.ymax, bb.zmax))})
    assy.export(str(step_path), "STEP")
    return census
