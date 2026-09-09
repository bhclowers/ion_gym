"""Editing session: load a spec document, expose an editor scene, commit
targeted patches.

THE COMMIT CONTRACT: the editor writes what it owns and
nothing else.  A commit is a TARGETED PATCH into the raw loaded JSON —
only `geometry.electrodes[i].shapes` moves; every other key in the document
survives byte-for-byte in structure because the editor never re-serializes
what it did not touch.  With zero edits the ORIGINAL BYTES come back
verbatim, which is what protects live-but-unmodelled keys (_display_name,
legacy rf fields) without needing to know they exist.

CONVENTIONS PINNED HERE (verified against the tree):
  * rect rotation is about the SHAPE CENTRE (raster2d.py:58,
    cx = x0 + w/2) — carried in the scene so a front end cannot invent a
    corner-rotation convention that diverges from the rasterizer.
  * shape field names are the raster2d vocabulary: rect(x_mm, y_mm,
    width_mm, height_mm[, rotation_deg]), ellipse(cx_mm, cy_mm, rx_mm,
    ry_mm), polygon(points_mm), cutout(children), extrude{axis, lo_mm,
    hi_mm}.
  * signed frame: a mirrored plane is at coordinate zero.

STL is refused PER ELECTRODE, not per route: build_route sends coords='rz'
to the rz builder regardless of STL, so an rz deck may carry an STL-backed
electrode — that electrode is non-editable with the reason named, while its
neighbours edit normally.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List, Optional, Tuple

from ion_gym.io.basis_cache import geometry_key_dict
from ion_gym.io.basis_cache import key as geometry_key_hash
from ion_gym.io.sim_spec import SimSpec
from ion_gym.edit.policy import (EditRefusal,
                                 GHOST_EXTENT_FRACTION, clamps_for,
                                 policy_for_document)

EDITOR_SCENE_SCHEMA = "ion_gym.editor_scene/1"

# standing fabrication floors (warnings at apply, never silent,
# never a refusal — a deliberate thin feature is the user's call):
PLATE_FLOOR_MM = 0.5
# off-lattice tolerance relative to the pitch, for the snap warning
PITCH_TOL_FRACTION = 1e-6


def normalize_color(value) -> list:
    """Stored electrode color -> [r, g, b] ints 0..255.

    Decks store color BOTH ways: a 3-list of ints and a "#rrggbb" hex
    string (both are found on shipped decks).  One
    normalizer, used by the scene AND the primitives, so the two can
    never disagree.  Anything else refuses by name."""
    if isinstance(value, str):
        h = value.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            try:
                return [int(h[k:k + 2], 16) for k in (0, 2, 4)]
            except ValueError as e:
                raise EditRefusal(
                    f"color {value!r} is not valid hex") from e
        raise EditRefusal(f"color string {value!r} is neither #rgb nor "
                          f"#rrggbb")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [max(0, min(255, int(c))) for c in value]
        except (TypeError, ValueError) as e:
            raise EditRefusal(
                f"color {value!r} has non-numeric components") from e
    raise EditRefusal(
        f"color {value!r}: expected [r, g, b] or a hex string")


# --------------------------------------------------------------------------
# structural diff — the gate's measuring instrument, so it lives with the
# session it measures (one module owns the patch AND the ruler for it).
# --------------------------------------------------------------------------
def structural_diff(a, b, path: str = "") -> List[Tuple[str, str]]:
    """All (kind, path) differences between two parsed JSON values.

    kind is 'added' (in b only), 'dropped' (in a only), 'changed', or
    'len' (list length changed).  An empty list means structurally
    identical documents regardless of formatting.
    """
    out: List[Tuple[str, str]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}/{k}"
            if k not in a:
                out.append(("added", p))
            elif k not in b:
                out.append(("dropped", p))
            else:
                out.extend(structural_diff(a[k], b[k], p))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(("len", path))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(structural_diff(x, y, f"{path}[{i}]"))
    elif a != b:
        out.append(("changed", path))
    return out


# --------------------------------------------------------------------------
# clamp checking: the lowest stored coordinate of a shape along an axis.
# --------------------------------------------------------------------------
def _shape_lo(shape: dict, axis: str) -> float:
    """Lowest extent of a shape dict along an in-plane axis ('x'|'y').

    Refuses unknown shape types by name — a silent default here would let
    a new shape type bypass every clamp (match/case has no silent default).
    """
    t = shape.get("type")
    if t == "rect":
        lo = float(shape["x_mm"] if axis == "x" else shape["y_mm"])
        # rotation about the centre can swing a corner below the stored
        # lo; be conservative: use the rotated bounding box.
        rot = float(shape.get("rotation_deg", 0.0) or 0.0)
        if rot:
            import math
            w = float(shape["width_mm"]); h = float(shape["height_mm"])
            cx = float(shape["x_mm"]) + w / 2.0
            cy = float(shape["y_mm"]) + h / 2.0
            ca = abs(math.cos(math.radians(rot)))
            sa = abs(math.sin(math.radians(rot)))
            half = (w * ca + h * sa) / 2.0 if axis == "x" \
                else (w * sa + h * ca) / 2.0
            lo = (cx if axis == "x" else cy) - half
        return lo
    if t == "ellipse":
        if axis == "x":
            return float(shape["cx_mm"]) - float(shape["rx_mm"])
        return float(shape["cy_mm"]) - float(shape["ry_mm"])
    if t == "polygon":
        pts = shape.get("points_mm") or []
        if not pts:
            raise EditRefusal("polygon with no points_mm — nothing to clamp "
                              "and nothing to rasterize")
        idx = 0 if axis == "x" else 1
        return min(float(p[idx]) for p in pts)
    if t == "cutout":
        kids = shape.get("children") or []
        if not kids:
            raise EditRefusal("cutout with no children — an empty hole is "
                              "not a declaration")
        return min(_shape_lo(k, axis) for k in kids)
    raise EditRefusal(
        f"shape type {t!r} has no lower-extent rule; clamps cannot be "
        f"checked, so the edit is refused rather than passed unverified")


def _check_clamps(shape: dict, clamps: List[dict], el_name: str) -> None:
    """Refuse a shape (recursing into cutout children for extrude) that
    violates any clamp, naming the clamp's reason and the offending value."""
    for c in clamps:
        if c["target"] == "shape":
            lo = _shape_lo(shape, c["axis"])
            if lo < c["min"]:
                raise EditRefusal(
                    f"electrode {el_name!r}: shape reaches "
                    f"{c['axis']} = {lo:g} mm, below the clamp "
                    f"{c['axis']} >= {c['min']:g} — {c['reason']}")
        elif c["target"] == "extrude":
            for sub, ex in _iter_extrudes(shape):
                if ex.get("axis") == c["axis"] \
                        and float(ex["lo_mm"]) < c["min"]:
                    raise EditRefusal(
                        f"electrode {el_name!r}: extrude lo_mm = "
                        f"{float(ex['lo_mm']):g} on axis {c['axis']!r}, "
                        f"below the clamp >= {c['min']:g} — {c['reason']}")
        else:
            raise EditRefusal(
                f"clamp with unknown target {c['target']!r} — refusing "
                f"rather than skipping an unenforceable constraint")


def _iter_extrudes(shape: dict):
    """Yield (shape_dict, extrude_dict) for a shape and any cutout children."""
    ex = shape.get("extrude")
    if isinstance(ex, dict):
        yield shape, ex
    for kid in (shape.get("children") or []):
        yield from _iter_extrudes(kid)


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------
class EditSession:
    """One loaded document + its policy + an ordered list of shape edits.

    Construction refuses (EditRefusal) anything out of scope; every
    operation refuses invalid input by name.  Nothing here mutates the
    original document — `to_document_bytes()` builds the patched copy.
    """

    def __init__(self, doc_bytes: bytes, name: str = "<memory>"):
        self.name = name
        self.original_bytes = bytes(doc_bytes)
        try:
            self.doc = json.loads(self.original_bytes)
        except json.JSONDecodeError as e:
            raise EditRefusal(f"{name}: not valid JSON: {e}") from e
        self.policy, self.spec = policy_for_document(self.doc)
        self.clamps = clamps_for(self.spec, self.policy)
        self._check_consistency()
        # ordered edit ops: dicts {op, el, [idx], [shape]}
        self._ops: List[dict] = []
        # redo stack: ops popped by undo_last; ANY new op clears it (a
        # redo after divergence would replay onto a different document)
        self._redo: List[dict] = []
        # committed-state cache: every read of the document as it
        # WOULD SAVE goes through committed_doc(); _rev bumps on
        # every queue/undo/redo so the cache can never be stale.
        # (v420 defect: pickers/fields read the ORIGINAL doc, so
        # added shapes/electrodes were invisible to editing.)
        self._rev = 0
        self._committed_cache = (-1, None)

    # -- loading ----------------------------------------------------------
    @classmethod
    def load(cls, path) -> "EditSession":
        p = Path(path)
        return cls(p.read_bytes(), name=p.name)

    def _check_consistency(self) -> None:
        """The raw document and the parsed spec must agree on the electrode
        list, because patches target RAW indices while validation reads the
        SPEC — a mismatch would edit one electrode and validate another."""
        raw = self.doc.get("geometry", {}).get("electrodes", [])
        parsed = self.spec.geometry.electrodes
        if len(raw) != len(parsed):
            raise EditRefusal(
                f"{self.name}: raw document has {len(raw)} electrodes but "
                f"the parsed spec has {len(parsed)} — the loader folded or "
                f"dropped entries, so raw-index patching would mistarget")
        for i, (r, p) in enumerate(zip(raw, parsed)):
            if r.get("name") != p.name:
                raise EditRefusal(
                    f"{self.name}: electrode {i} is {r.get('name')!r} in "
                    f"the raw document but {p.name!r} in the parsed spec — "
                    f"order changed in parsing; raw-index patching would "
                    f"mistarget")

    # -- editability ------------------------------------------------------
    def electrode_editable(self, i: int) -> Tuple[bool, Optional[str]]:
        """(editable, reason-if-not) for COMMITTED electrode i — pending
        adds are editable before any save.  STL is per-electrode: an
        in-scope route may still carry a mesh-backed electrode."""
        els = self.committed_electrodes()
        if not (0 <= i < len(els)):
            raise EditRefusal(
                f"electrode index {i} out of range 0..{len(els) - 1}")
        el = els[i]
        if el.get("stl"):
            return False, (f"electrode {el.get('name')!r} is STL-backed "
                           f"({el.get('stl')!r}): meshes are owned by "
                           f"CAD (L-193 ruling); parametric shapes only")
        return True, None

    def _require_editable(self, i: int) -> str:
        ok, reason = self.electrode_editable(i)
        if not ok:
            raise EditRefusal(reason)
        return str(self.committed_electrodes()[i].get("name"))

    # -- validation -------------------------------------------------------
    def _validate_shape(self, shape: dict, el_name: str) -> None:
        if not isinstance(shape, dict):
            raise EditRefusal(
                f"electrode {el_name!r}: a shape must be a dict, got "
                f"{type(shape).__name__}")
        t = shape.get("type")
        if t not in self.policy.editable_shape_types:
            raise EditRefusal(
                f"electrode {el_name!r}: shape type {t!r} is not in the "
                f"{self.policy.route} policy vocabulary "
                f"{self.policy.editable_shape_types}")
        has_ex = any(True for _ in _iter_extrudes(shape))
        if has_ex and not self.policy.supports_extrude:
            raise EditRefusal(
                f"electrode {el_name!r}: shape carries an extrude block "
                f"but the {self.policy.route} route has no per-shape "
                f"extent ({self.policy.out_of_plane})")
        _check_clamps(shape, self.clamps, el_name)

    # -- edit operations --------------------------------------------------
    def set_shape(self, el_index: int, shape_index: int,
                  shape: dict) -> None:
        """Replace shape j of electrode i with a validated shape dict."""
        el_name = self._require_editable(el_index)
        shapes = self.committed_electrodes()[el_index] \
            .get("shapes", [])
        if not (0 <= shape_index < len(shapes)):
            raise EditRefusal(
                f"electrode {el_name!r}: shape index {shape_index} out of "
                f"range 0..{len(shapes) - 1}")
        self._validate_shape(shape, el_name)
        self._queue({"op": "set", "el": el_index,
                          "idx": shape_index,
                          "shape": copy.deepcopy(shape)})

    def add_shape(self, el_index: int, shape: dict) -> None:
        """Append a validated shape to electrode i."""
        el_name = self._require_editable(el_index)
        self._validate_shape(shape, el_name)
        self._queue({"op": "add", "el": el_index,
                          "shape": copy.deepcopy(shape)})

    def remove_shape(self, el_index: int, shape_index: int) -> None:
        """Remove shape j of electrode i.  Removing the LAST shape is
        refused: an electrode with no shapes and no STL changes the build
        route itself (planar with no shapes anywhere is unbuildable), and
        deleting a whole electrode is a different, deliberate operation."""
        el_name = self._require_editable(el_index)
        shapes = self.committed_electrodes()[el_index] \
            .get("shapes", [])
        if not (0 <= shape_index < len(shapes)):
            raise EditRefusal(
                f"electrode {el_name!r}: shape index {shape_index} out of "
                f"range 0..{len(shapes) - 1}")
        if len(shapes) <= 1:
            raise EditRefusal(
                f"electrode {el_name!r}: removing its last shape would "
                f"leave a conductor with no geometry — delete the "
                f"electrode deliberately instead (not a Slice 0 operation)")
        self._queue({"op": "remove", "el": el_index,
                          "idx": shape_index})

    @property
    def edit_count(self) -> int:
        return len(self._ops)

    # -- the editor scene (what a front end consumes) ---------------------
    def editor_scene(self) -> dict:
        """A JSON-able description of what is editable and under what
        conventions.  Shapes are PASSTHROUGH copies of the raw document's
        dicts — the scene annotates, it does not re-enumerate fields, so a
        schema addition survives untouched."""
        g = self.spec.geometry
        raw_els = self.doc["geometry"]["electrodes"]
        els = []
        for i, el in enumerate(g.electrodes):
            ok, reason = self.electrode_editable(i)
            entry = {
                "index": i,
                "name": el.name,
                "editable": ok,
                "is_grid": bool(el.is_grid),
                "color": normalize_color(el.color),
                "shapes": copy.deepcopy(raw_els[i].get("shapes", [])),
            }
            if not ok:
                entry["refusal"] = reason
            els.append(entry)
        return {
            "schema": EDITOR_SCENE_SCHEMA,
            "document": self.name,
            "route": self.policy.route,
            "policy": {
                "editable_shape_types":
                    list(self.policy.editable_shape_types),
                "in_plane_axes": list(self.policy.in_plane_axes),
                "in_plane_labels": list(self.policy.in_plane_labels),
                "supports_extrude": self.policy.supports_extrude,
                "viewports": list(self.policy.viewports),
            },
            "pitch_mm": float(g.mm_per_gu),
            "symmetry": {
                "coords": g.symmetry.coords,
                "planes": dict(g.symmetry.planes),
                "ghost_rule": "the stored fraction is editable; every "
                              "mirror image is drawn as a non-editable "
                              "ghost (mirrored plane at coordinate zero)",
            },
            "out_of_plane": self._out_of_plane_block(),
            "clamps": copy.deepcopy(self.clamps),
            "conventions": {
                "rotation_center": "shape_center",
                "rotation_center_authority": "raster2d _rect (cx = x0 + "
                                             "w/2, cy = y0 + h/2)",
                "frame": "signed; any mirrored plane at coordinate zero",
            },
            "electrodes": els,
        }

    def _out_of_plane_block(self) -> dict:
        g = self.spec.geometry
        kind = self.policy.out_of_plane
        if kind == "revolved":
            return {"kind": "revolved",
                    "note": "r-z: the out-of-plane direction is the "
                            "azimuth; extent is 2*pi and is not a "
                            "parameter"}
        if kind == "per_shape_extrude":
            return {"kind": "per_shape_extrude",
                    "depth_mm": float(g.depth_mm),
                    "axial_extent_mm": (list(g.axial_extent_mm)
                                        if g.axial_extent_mm else None)}
        if kind == "undetermined":
            if g.depth_mm and g.depth_mm > 0.0:
                return {"kind": "declared_depth",
                        "depth_mm": float(g.depth_mm)}
            if g.axial_extent_mm:
                return {"kind": "declared_axial",
                        "axial_extent_mm": list(g.axial_extent_mm)}
            ghost = GHOST_EXTENT_FRACTION * min(float(g.width_mm),
                                                float(g.height_mm))
            return {"kind": "ghost",
                    "extent_mm": ghost,
                    "display_only": True,
                    "note": "planar with neither depth_mm nor "
                            "axial_extent_mm declared: this extent is a "
                            "DISPLAY placeholder (10% of the smaller "
                            "in-plane dimension, PI 2026-08-25). It is "
                            "never written to the spec, never editable, "
                            "and never a dimension readout"}
        raise EditRefusal(
            f"out_of_plane kind {kind!r} has no scene rule — a policy "
            f"grew a value this method does not handle")

    def _queue(self, op: dict) -> None:
        """THE door for new ops: appends and clears the redo stack."""
        self._ops.append(op)
        self._redo.clear()
        self._rev += 1

    # -- Slice 3: creation, undo, snap, warnings --------------------------
    def add_electrode(self, name: str, shape: dict, *,
                      is_grid: bool = False, dc: float = 0.0,
                      color=None) -> None:
        """Append a new electrode with one initial validated shape (our
        own remove rule forbids shapeless conductors, so creation starts
        with geometry).  Duplicate names refuse: the basis->electrode
        mapping and dc_group membership key off names."""
        if not name or not str(name).strip():
            raise EditRefusal("electrode name must be non-empty")
        name = str(name).strip()
        existing = [e.get("name")
                    for e in self.committed_electrodes()]
        if name in existing:
            raise EditRefusal(
                f"electrode name {name!r} already exists — names key the "
                f"basis mapping and group membership, so a duplicate "
                f"would be two authorities for one conductor")
        self._validate_shape(shape, name)
        el = {"name": name, "shapes": [copy.deepcopy(shape)],
              "is_grid": bool(is_grid), "dc": float(dc)}
        if color is not None:
            el["color"] = normalize_color(color)
        self._queue({"op": "add_electrode", "electrode": el})

    def set_color(self, el_index: int, color) -> None:
        """Set one electrode's stored display color.  Allowed on ANY
        electrode including STL-backed ones: color is display metadata,
        not mesh geometry, so the mesh-ownership refusal does not
        apply."""
        els = self.spec.geometry.electrodes
        if not (0 <= el_index < len(els)):
            raise EditRefusal(
                f"electrode index {el_index} out of range "
                f"0..{len(els) - 1}")
        self._queue({"op": "set_color", "el": el_index,
                     "color": normalize_color(color)})

    def set_color_all(self, color) -> None:
        """Set every electrode's stored display color in one op."""
        self._queue({"op": "set_color_all",
                     "color": normalize_color(color)})

    def undo_last(self) -> dict:
        """Pop and return the most recent edit op; refuses when there is
        nothing to undo (a no-op undo hides a user mistake)."""
        if not self._ops:
            raise EditRefusal("nothing to undo — no edits are queued")
        op = self._ops.pop()
        self._redo.append(op)
        self._rev += 1
        return op

    def redo_last(self) -> dict:
        """Re-queue the most recently undone op; refuses when there is
        nothing to redo.  New edits clear the redo stack, so a redo
        always replays onto the document state it was popped from."""
        if not self._redo:
            raise EditRefusal("nothing to redo — no undone edits are "
                              "pending (new edits clear the redo stack)")
        op = self._redo.pop()
        self._ops.append(op)   # NOT _queue: must not clear the stack
        self._rev += 1
        return op

    def committed_doc(self) -> dict:
        """The document AS IT WOULD SAVE (original + queued ops),
        parsed.  THE read surface for pickers, fields, and drag
        bases — reading the original instead made added geometry
        invisible (v420)."""
        if self._committed_cache[0] == self._rev:
            return self._committed_cache[1]
        doc = json.loads(self.to_document_bytes())
        self._committed_cache = (self._rev, doc)
        return doc

    def committed_electrodes(self) -> list:
        return self.committed_doc().get("geometry", {}) \
            .get("electrodes", [])

    def snap_to_pitch(self, value: float) -> float:
        """Nearest pitch-lattice value.  The UI shows the snapped number
        BEFORE apply, so displayed == committed (the control proposes,
        the number decides)."""
        h = float(self.spec.geometry.mm_per_gu)
        return round(round(float(value) / h) * h, 12)

    def shape_warnings(self, shape: dict) -> List[str]:
        """Named, non-blocking warnings: fabrication floors and
        off-lattice coordinates.  Warnings, not refusals — a deliberate
        thin or off-pitch feature is the user's call, but it is never a
        silent one (closed-edge raster: a box exactly N intervals wide
        occupies N+1 nodes, so sub-pitch edges land by tolerance)."""
        out: List[str] = []
        t = shape.get("type")
        if t == "rect":
            for k in ("width_mm", "height_mm"):
                v = float(shape.get(k, 0.0))
                if 0 < v < PLATE_FLOOR_MM:
                    out.append(f"{k} = {v:g} mm is below the "
                               f"{PLATE_FLOOR_MM:g} mm plate floor "
                               f"(standing ruling); rasterization at "
                               f"pitch {self.spec.geometry.mm_per_gu:g} "
                               f"mm may thin or drop it")
        if t == "ellipse":
            for k in ("rx_mm", "ry_mm"):
                v = float(shape.get(k, 0.0))
                if 0 < 2 * v < PLATE_FLOOR_MM:
                    out.append(f"2*{k} = {2*v:g} mm is below the "
                               f"{PLATE_FLOOR_MM:g} mm plate floor "
                               f"(standing ruling)")
        h = float(self.spec.geometry.mm_per_gu)
        tol = h * PITCH_TOL_FRACTION
        num_keys = {"rect": ("x_mm", "y_mm", "width_mm", "height_mm"),
                    "ellipse": ("cx_mm", "cy_mm", "rx_mm", "ry_mm"),
                    "polygon": (), "cutout": ()}.get(t, ())
        for k in num_keys:
            v = float(shape.get(k, 0.0))
            snapped = self.snap_to_pitch(v)
            if abs(v - snapped) > tol:
                out.append(f"{k} = {v:g} mm is off the {h:g} mm pitch "
                           f"lattice (nearest {snapped:g}); edges "
                           f"classify by the closed-edge tolerance "
                           f"raster")
        ex = shape.get("extrude")
        if isinstance(ex, dict):
            for k in ("lo_mm", "hi_mm"):
                v = float(ex.get(k, 0.0))
                snapped = self.snap_to_pitch(v)
                if abs(v - snapped) > tol:
                    out.append(f"extrude {k} = {v:g} mm off the {h:g} mm "
                               f"pitch lattice (nearest {snapped:g})")
        for kid in (shape.get("children") or []):
            out.extend(self.shape_warnings(kid))
        return out

    def geometry_key_short(self) -> str:
        """Current document's geometry key via THE cache authority
        (basis_cache.key -> fa_cache.spec_key), never a second hash."""
        return geometry_key_hash(self.spec)

    # -- commit -----------------------------------------------------------
    def to_document_bytes(self) -> bytes:
        """The document to save.  ZERO edits -> the ORIGINAL BYTES,
        verbatim — the editor does not reformat what it did not touch.
        With edits: the raw document patched ONLY at
        geometry.electrodes[*].shapes, then serialized."""
        if not self._ops:
            return self.original_bytes
        doc = json.loads(self.original_bytes)
        els = doc["geometry"]["electrodes"]
        # STRICTLY IN ORDER: every op's indices are in the committed
        # frame at its point in the sequence (removals apply inline —
        # the old deferred/descending machinery assumed original-frame
        # indices and mis-targeted after adds or repeated removals).
        for op in self._ops:
            if op["op"] == "add_electrode":
                els.append(copy.deepcopy(op["electrode"]))
                continue
            if op["op"] == "set_color_all":
                for el in els:
                    el["color"] = list(op["color"])
                continue
            el = els[op["el"]]
            if op["op"] == "set_color":
                el["color"] = list(op["color"])
                continue
            shapes = el.setdefault("shapes", [])
            if op["op"] in ("set", "remove"):
                j = op["idx"]
                if not (0 <= j < len(shapes)):
                    raise EditRefusal(
                        f"electrode {op['el']}: op {op['op']!r} targets "
                        f"shape {j}, out of range 0..{len(shapes) - 1} "
                        f"at its point in the sequence")
                if op["op"] == "set":
                    shapes[j] = copy.deepcopy(op["shape"])
                else:
                    del shapes[j]
            elif op["op"] == "add":
                shapes.append(copy.deepcopy(op["shape"]))
            else:
                raise EditRefusal(
                    f"unknown edit op {op['op']!r} in the queue — the "
                    f"session recorded something it cannot apply")
        return (json.dumps(doc, indent=2) + "\n").encode()

    def verify_commit(self) -> dict:
        """Prove the patch did exactly what it claims (the gate calls this;
        Slice 3's Apply will too).  Returns a report dict; raises
        EditRefusal if the patched document violates the contract."""
        out_bytes = self.to_document_bytes()
        orig = json.loads(self.original_bytes)
        patched = json.loads(out_bytes)
        diff = structural_diff(orig, patched)
        # ownership, precisely: shape lists of existing electrodes, plus
        # the electrodes LIST itself (an add changes its length; the diff
        # cannot descend past a length mismatch).  An electrode's dc,
        # name, or group fields are NOT the editor's to change.
        stray = [(k, p) for k, p in diff
                 if not (p == "/geometry/electrodes"
                         or ("/geometry/electrodes[" in p
                             and ("/shapes" in p
                                  or "/color" in p)))]
        if stray:
            raise EditRefusal(
                f"commit touched paths outside electrode shapes: "
                f"{stray[:6]} — the targeted-patch contract is violated")
        try:
            new_spec = SimSpec.from_dict(patched)
        except Exception as e:
            raise EditRefusal(
                f"patched document no longer parses as a SimSpec: "
                f"{type(e).__name__}: {e}") from e
        key_before = json.dumps(geometry_key_dict(self.spec),
                                sort_keys=True)
        key_after = json.dumps(geometry_key_dict(new_spec),
                               sort_keys=True)
        # key coherence against GEOMETRY-affecting diffs only:
        # color is display metadata and must not claim a geometry
        # change (a color-only edit keeps the solved field valid)
        geo_diff = [q for _, q in diff
                    if "/shapes" in q or q == "/geometry/electrodes"]
        shapes_changed = bool(geo_diff)
        key_changed = key_before != key_after
        if shapes_changed != key_changed:
            raise EditRefusal(
                f"geometry key {'did not change' if not key_changed else 'changed'} "
                f"while the shape diff says "
                f"{'edits exist' if shapes_changed else 'no edits'} — "
                f"displayed geometry and solver geometry disagree")
        return {"diff_paths": [p for _, p in diff],
                "geometry_key_changed": key_changed,
                "n_ops": len(self._ops)}
