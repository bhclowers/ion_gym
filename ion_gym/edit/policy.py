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

from ion_gym.io.sim_spec import SimSpec
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
    planes = spec.geometry.symmetry.planes
    for axis, kind in planes.items():
        if kind != "mirror":
            continue
        if axis in policy.in_plane_axes:
            out.append({
                "target": "shape", "axis": axis, "min": 0.0,
                "reason": f"declared mirror in {axis}: the stored fraction "
                          f"lives at {axis} >= 0 (mirrored plane at "
                          f"coordinate zero); the image is a ghost, not "
                          f"an edit target",
            })
        elif axis == "z" and policy.supports_extrude:
            out.append({
                "target": "extrude", "axis": "z", "min": 0.0,
                "reason": "declared mirror in z: extrude lo_mm >= 0 "
                          "(mirrored plane at coordinate zero)",
            })
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
