"""Per-route editing policy for the geometry editor.

WHY PER-ROUTE POLICIES AND NOT ONE RULE: the
build routes genuinely mean different things — planar has one undetermined
out-of-plane value, r-z has NO such quantity (the revolution is 2*pi), and
shapes3d declares extent per shape.  A unified rule could only be written by
inventing a fictional common "extent" concept, and that fiction is where
collisions come from.  Each rule below has exactly one owner.

WHY build_route() AND NOT OUR OWN CLASSIFIER (one authority):
`sim_build.BuildRoute` exists because six sim_app call sites answered
"is this 3-D?" from depth_mm > 0 and were wrong twice.  This module must
not become the seventh site: the policy is SELECTED by the same routing
build_run() dispatches on, and the editor core contains ZERO route checks —
every route-dependent fact is data on the policy.

SCOPE:
  * editable routes: planar, rz, shapes3d
  * stl2d / stl3d: REFUSED — meshes are owned by CAD; this tool edits
    parametric shapes.
  * scene3d: PARKED — zero shipped decks exercise it; an editing path with
    no fixture to falsify it is unverifiable.
  * multi-FA instrument documents: REFUSED (pending the stage
    export / re-inline path that would make them editable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ion_gym.io.sim_spec import EXTRUDE_INPLANE_AXES, SimSpec
from ion_gym.physics.sim_build import build_route

# Display-only ghost slab for a planar deck that declares NEITHER depth_mm
# nor axial_extent_mm: fraction of the SMALLER in-plane dimension
# (obviously-a-placeholder beats plausibly-wrong; never written
# back to the spec, never editable, never shown in a dimension readout).
GHOST_EXTENT_FRACTION = 0.10


class EditRefusal(ValueError):
    """A document or operation the editor declines, with the reason named.

    Every refusal in the edit package raises this (or a subclass) so a
    caller can distinguish "the editor declines this by policy" from a
    genuine malfunction.  The message always names what did not line up.
    """


# ---------------------------------------------------------------------
# VIEWER FRAME (2026-09-12, the x-extrusion fix): the edit viewer's JS
# extrudes resolved in-plane polygons along ITS z only. That is not a
# limit of the builder (build_shapes3d accepts any axis) and it need
# not limit the viewer either, because the ShapeSpec convention already
# makes the fix a pure relabeling: a deck extruded along world axis A
# stores its outlines in the cyclic in-plane coordinates
# EXTRUDE_INPLANE_AXES[A], so handing those outlines to the JS
# unchanged and permuting everything world-framed (trajectories,
# detections, mirror-plane axes, axis labels) by the SAME cycle draws
# the deck exactly — viewer (x, y, z) displays world
# (*EXTRUDE_INPLANE_AXES[A], A). For A == 'z' the map is the identity
# and every payload is unchanged, which is what keeps the certified
# z-extrusion paths untouched. This holds for ANY conforming deck, not
# one configuration, because it is derived from the schema's own
# in-plane convention rather than from any geometry in front of us.
# A deck mixing extrude axes has no single such frame and REFUSES by
# name (the builder still builds it; deck_multiview still draws it).
# ---------------------------------------------------------------------

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def deck_extrude_axis(named_shapes) -> str:
    """The ONE world extrude axis of a deck, from (owner_name,
    [shape dicts]) pairs — raw document shapes, cutout children walked.

    A shape with no extrude descriptor counts as 'z': that is the
    builder's full-depth z-slab default (build_shapes3d) and the
    planar/r-z convention, so extrude-less decks land on the identity
    frame and behave byte-identically to before this function existed.
    Mixed axes refuse by name; an empty deck is 'z' (identity).
    """
    seen: dict = {}          # axis -> [owner names], first-seen order

    def _note(ax: str, owner: str) -> None:
        seen.setdefault(ax, [])
        if owner not in seen[ax]:
            seen[ax].append(owner)

    def _walk(owner: str, shape: dict) -> None:
        t = shape.get("type")
        if t in ("cutout", "group"):
            # CONTAINER nodes render nothing themselves — a cutout is
            # a grouping of subtractions, not a solid — so an absent
            # extrude on the wrapper is not a z-slab declaration
            # (counting it as one framed every extruded deck as
            # "mixed": found on the StepWave, whose cutout wrappers
            # tripped exactly this). Only their children carry axes.
            for kid in (shape.get("children") or []):
                _walk(owner, kid)
            return
        ex = shape.get("extrude")
        if ex is None:
            _note("z", owner)
        else:
            ax = ex.get("axis")
            if ax not in _AXIS_INDEX:
                raise EditRefusal(
                    f"{owner}: extrude axis {ax!r} is not one of "
                    f"'x'|'y'|'z' — the shape cannot be framed")
            _note(ax, owner)
        for kid in (shape.get("children") or []):
            _walk(owner, kid)

    for owner, shapes in named_shapes:
        for sh in (shapes or []):
            _walk(owner, sh)
    if len(seen) > 1:
        detail = "; ".join(
            f"{ax!r} on [{', '.join(names[:6])}"
            + (f", … {len(names) - 6} more" if len(names) > 6 else "")
            + "]"
            for ax, names in sorted(seen.items()))
        raise EditRefusal(
            f"mixed extrude axes cannot share one viewer frame: {detail}. "
            f"A shape with no extrude block counts as 'z' (the builder's "
            f"full-depth slab default). The deck still builds and flies; "
            f"re-author to one axis to view it here, or use "
            f"deck_multiview for the schematic.")
    return next(iter(seen), "z")


def viewer_frame(axis: str) -> dict:
    """Frame descriptor for the payload: which WORLD axis each viewer
    axis displays. Identity for 'z'."""
    if axis not in _AXIS_INDEX:
        raise EditRefusal(f"viewer_frame: extrude axis {axis!r} is not "
                          f"one of 'x'|'y'|'z'")
    a0, a1 = EXTRUDE_INPLANE_AXES[axis]
    return {"extrude_axis": axis, "axes": [a0, a1, axis]}


def viewer_axis_indices(axis: str):
    """(i0, i1, i2): world-component indices that become viewer
    (x, y, z). Identity (0, 1, 2) for 'z' — callers may skip the
    permutation entirely in that case."""
    a0, a1 = EXTRUDE_INPLANE_AXES[axis]
    return (_AXIS_INDEX[a0], _AXIS_INDEX[a1], _AXIS_INDEX[axis])



@dataclass(frozen=True)
class EditPolicy:
    """Everything route-dependent, stated as data.

    The editor core consults these fields instead of testing the route:
    if an `if route == ...` appears in the session or (later) the renderer,
    that is the smell build_route() was created to remove.
    """
    route: str                          # build_route().builder this serves
    editable_shape_types: Tuple[str, ...]
    in_plane_axes: Tuple[str, str]      # stored coordinate names, in order
    in_plane_labels: Tuple[str, str]    # what those axes MEAN on this route
    out_of_plane: str                   # 'undetermined'|'revolved'|'per_shape_extrude'
    supports_extrude: bool              # may shapes carry an extrude block?
    viewports: Tuple[str, ...]          # viewer layout, per route (data,
                                        # so the JS side has no route checks)


_VOCAB = ("rect", "ellipse", "polygon", "cutout")

POLICIES: Dict[str, EditPolicy] = {
    "planar": EditPolicy(
        route="planar",
        editable_shape_types=_VOCAB,
        in_plane_axes=("x", "y"),
        in_plane_labels=("x", "y"),
        out_of_plane="undetermined",
        supports_extrude=False,
        viewports=("persp", "xy"),
    ),
    "rz": EditPolicy(
        route="rz",
        editable_shape_types=_VOCAB,
        in_plane_axes=("x", "y"),
        in_plane_labels=("axial", "r"),
        out_of_plane="revolved",
        supports_extrude=False,
        viewports=("persp", "rx"),
    ),
    "shapes3d": EditPolicy(
        route="shapes3d",
        editable_shape_types=_VOCAB,
        in_plane_axes=("x", "y"),
        in_plane_labels=("x", "y"),
        out_of_plane="per_shape_extrude",
        supports_extrude=True,
        viewports=("persp", "xy", "xz", "yz"),
    ),
}


def clamps_for(spec: SimSpec, policy: EditPolicy) -> List[dict]:
    """Hard geometric constraints for this spec under this policy, as data.

    Each clamp: {"target": "shape"|"extrude", "axis": <coord>,
                 "min": <mm>, "reason": <named>}.
    The session enforces them at apply; the front end also carries them
    so a drag can be clamped live.  Signed-frame rule:
    any mirrored plane sits at coordinate zero, so mirror clamps are >= 0.
    """
    out: List[dict] = []
    if policy.route == "rz":
        out.append({
            "target": "shape", "axis": "y", "min": 0.0,
            "reason": "r-z route: r >= 0 — the axis is a boundary of the "
                      "domain, not a free edge",
        })
        return out
    # WHICH STORED COORDINATE a declared WORLD mirror binds is set by
    # the deck's extrude axis (2026-09-12, found via the StepWave: a
    # world-z mirror on an x-extruded deck binds the stored in-plane
    # y_mm, not the extrude range — the previous code assumed
    # z-extrusion and emitted the extrude clamp for ANY z mirror).
    # Derivation, one authority (EXTRUDE_INPLANE_AXES): for deck
    # extrude axis A, world axis w == A is the extrude coordinate;
    # w in EXTRUDE_INPLANE_AXES[A] is the stored in-plane coordinate
    # at that position. For A == 'z' this reproduces the old mapping
    # exactly (x/y -> shape clamps, z -> extrude clamp), so every
    # existing deck's clamps are byte-identical.
    deck_axis = deck_extrude_axis(
        (el.name, [sh.to_dict() for sh in el.shapes])
        for el in spec.geometry.electrodes)
    inplane_world = EXTRUDE_INPLANE_AXES[deck_axis]
    planes = spec.geometry.symmetry.planes
    for axis, kind in planes.items():
        if kind != "mirror":
            continue
        if axis in inplane_world:
            stored = policy.in_plane_axes[inplane_world.index(axis)]
            out.append({
                "target": "shape", "axis": stored, "min": 0.0,
                "reason": f"declared mirror in {axis}: the stored fraction "
                          f"lives at {axis} >= 0 (mirrored plane at "
                          f"coordinate zero); the image is a ghost, not "
                          f"an edit target"
                          + (f" — {axis} is the stored in-plane "
                             f"'{stored}' coordinate on this "
                             f"{deck_axis}-extruded deck"
                             if deck_axis != "z" else ""),
            })
        elif axis == deck_axis and policy.supports_extrude:
            out.append({
                "target": "extrude", "axis": deck_axis, "min": 0.0,
                "reason": f"declared mirror in {deck_axis}: extrude "
                          f"lo_mm >= 0 (mirrored plane at coordinate "
                          f"zero)",
            })
        else:
            # DELIBERATE NO-CLAMP, stated (charter §8: every else has
            # an outcome): the mirror axis is the extrude coordinate of
            # a route that does not support extrudes (e.g. a planar
            # deck's z mirror — depth_mm is the stored half-domain).
            # There is no stored shape coordinate to clamp; the build
            # honors the mirror, and the editor has nothing to guard.
            pass
    return out


def policy_for_document(doc: object) -> Tuple[EditPolicy, SimSpec]:
    """Classify a raw loaded JSON document; return (policy, spec) or refuse.

    Document-level refusals fire BEFORE SimSpec parsing, because the parser
    fails a multi-FA file with a bare KeyError that names nothing.
    """
    if not isinstance(doc, dict):
        raise EditRefusal(
            f"not a JSON object at top level (got {type(doc).__name__}); "
            f"a simulation spec is a dict with a 'geometry' block")
    if "stages" in doc:
        raise EditRefusal(
            "multi-FA instrument document (has 'stages'): out of editor "
            "scope by ruling (PI 2026-08-25). Edit a single-FA spec; "
            "the stage export / re-inline path is L-195")
    if "geometry" not in doc:
        raise EditRefusal(
            f"not a single-FA simulation spec: no 'geometry' block "
            f"(top-level keys: {sorted(doc.keys())[:8]})")
    try:
        spec = SimSpec.from_dict(doc)
    except EditRefusal:
        raise
    except Exception as e:
        raise EditRefusal(
            f"document does not parse as a SimSpec: "
            f"{type(e).__name__}: {e}") from e
    try:
        route = build_route(spec)
    except ValueError as e:
        # build_route's own named refusal (e.g. a retired builder name);
        # keep its wording — it is the one authority on classification.
        raise EditRefusal(f"build_route refuses this spec: {e}") from e
    b = route.builder
    if b in POLICIES:
        return POLICIES[b], spec
    if b in ("stl2d", "stl3d"):
        raise EditRefusal(
            f"route {b!r}: geometry is STL-backed. Meshes are owned by "
            f"the CAD tool that made them (L-193, PI 2026-08-25); this "
            f"editor edits parametric shapes only")
    if b == "scene3d":
        raise EditRefusal(
            "route 'scene3d': editing PARKED (L-194) — zero shipped decks "
            "exercise this route, so an editing path here has no fixture "
            "that can falsify it. Re-opens when a scene3d deck exists")
    if b == "unwired":
        raise EditRefusal(
            "route 'unwired': build_run() itself has no builder for this "
            "spec (xyz, depth_mm > 0, no inline shapes, no STL) — nothing "
            "to edit that could ever be built")
    raise EditRefusal(
        f"route {b!r} is not covered by any editing policy; known "
        f"policies: {sorted(POLICIES)} — this is a code defect, not a "
        f"document defect")
