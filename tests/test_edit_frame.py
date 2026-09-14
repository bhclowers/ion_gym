"""test_edit_frame.py — the edit viewer's extrude-axis frame (L-432).

COMMISSIONED by Brian 2026-09-12 (the session that landed the fix):
the standing convention "every refusal or guard gets an adversarial
test in the same change" applied to the payload-frame permutation.
Before this file the edit subsystem had ZERO standing pytest coverage;
every L-432 property lived in session drives that die with their
container.

What it pins (all headless, ZERO field solves — sessions and payloads
only walk shapes):

  F1  frame authority: viewer_frame/viewer_axis_indices are the exact
      cyclic map of the ShapeSpec convention; z is the identity.
  F2  deck_extrude_axis: single axis resolved; CONTAINER nodes
      (cutout/group) contribute no axis (the wrapper-counts-as-z bug
      framed the StepWave against itself); extrude-less RENDERABLE
      shapes count as z; MIXED axes refuse naming electrodes + axes.
  F3  mirror clip (_clip_half): identity vertex list for a polygon
      already inside; straddling polygons clip to the >= 0 half;
      fully-mirrored shapes come back empty (the caller reports).
  F4  an x-extruded deck renders end-to-end: frame (y, z, x), the
      plane-straddling bore hosts as a clipped hole, the
      boundary-crossing bore ghosts WITH a report (never dropped),
      and the world-z mirror clamp binds the stored in-plane 'y'.
  F5  shipped-deck identity: every editor-accepted examples/ deck
      resolves the identity frame and unchanged z-mirror clamp form —
      the property that replaced the one-time byte-diff against the
      pristine v516 extract.
  F6  ortho label composition (the ESM's wlab, replicated): identity
      frame reproduces the pre-fix label strings character for
      character; the x frame names the world planes. The JS itself
      cannot run under pytest; this pins the CONTRACT the ESM
      implements, and the browser pass is the second witness.
  F7  /flight payload: trajectory points and detection markers
      permute world -> viewer on an x-deck and pass through
      byte-equal on a z-deck (the identity path is skipped, not
      applied).
  F8  StepWave cross-check (LOUD SKIP when internal/ is absent, per
      the cleanup-contract Q3 rule): 140 electrodes, frame (y,z,x),
      28 subtracted bores + 112 reported open-sided ghosts — the
      split recomputed from the raw deck, not hardcoded trust.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

import _bootstrap  # noqa: F401  (roots imports on the repo)

from ion_gym.edit.outline import _clip_half, render_primitives
from ion_gym.edit.policy import (EditRefusal, deck_extrude_axis,
                                 viewer_axis_indices, viewer_frame)
from ion_gym.edit.session import EditSession
from ion_gym.io.sim_spec import EXTRUDE_INPLANE_AXES

REPO = Path(__file__).resolve().parent.parent


def _sarea(poly) -> float:
    """Shoelace signed area — the test's OWN copy (a gate must not
    certify a function with that function)."""
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def _pip(pt, poly) -> bool:
    """Ray-cast point-in-polygon — the test's own copy, same reason."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


# ---------------------------------------------------------------------
# fixtures: authored IN-FILE so the gate is self-contained (no fixture
# files to drift). Small enough that a session can read them whole.
# ---------------------------------------------------------------------
def _xdeck() -> dict:
    """Minimal x-extruded deck with a declared world-z mirror and two
    bores: one fully inside its plate (must HOST as a hole) and one
    crossing the plate edge (must GHOST with a report). Stored (x, y)
    are world (y, z); the mirror plane is stored y = 0."""
    ex = {"axis": "x", "lo_mm": 1.0, "hi_mm": 2.0}
    return {
        "name": "xdeck synthetic (test_edit_frame)",
        "geometry": {
            "width_mm": 10.0,        # world x (the extrude axis)
            "height_mm": 8.0,        # world y (stored in-plane x)
            "depth_mm": 6.0,         # world z half-domain (mirrored)
            "mm_per_gu": 0.5,
            "symmetry": {"planes": {"z": "mirror"}},
            "electrodes": [
                {"name": "HOSTED", "dc": 0.0, "shapes": [
                    {"type": "rect", "x_mm": 0.0, "y_mm": 0.0,
                     "width_mm": 8.0, "height_mm": 6.0,
                     "extrude": dict(ex)},
                    {"type": "cutout", "children": [
                        # centred ON the mirror plane (stored y = 0):
                        # straddles it exactly as the StepWave bores do
                        {"type": "ellipse", "cx_mm": 4.0, "cy_mm": 0.0,
                         "rx_mm": 2.0, "ry_mm": 2.0,
                         "extrude": dict(ex)}]},
                ]},
                {"name": "OPENBORE", "dc": 0.0, "shapes": [
                    {"type": "rect", "x_mm": 0.0, "y_mm": 0.0,
                     "width_mm": 5.0, "height_mm": 6.0,
                     "extrude": dict(ex)},
                    {"type": "cutout", "children": [
                        # pokes past the plate's stored-x edge at 5.0
                        {"type": "ellipse", "cx_mm": 4.0, "cy_mm": 0.0,
                         "rx_mm": 2.0, "ry_mm": 2.0,
                         "extrude": dict(ex)}]},
                ]},
            ]},
        "source": {"x0_mm": 0.5, "y0_mm": 0.5, "z0_mm": 0.5,
                   "mz_list": [100.0], "n_ions": 1},
    }


def _zdeck() -> dict:
    """The same construction extruded along z (identity frame). The
    mirror moves WITH the meaning, not the letter: the x deck's world-z
    plane was IN-PLANE (stored y); the identity-frame twin therefore
    declares y:mirror so the same straddling-bore clip is exercised —
    plus z:mirror, so the extrude-clamp leg is pinned in the same
    fixture. (The first cut of this file kept z:mirror alone and the
    bore honestly ghosted: with no in-plane plane there is nothing to
    clip against, and the bore really does poke below its plate.)"""
    d = copy.deepcopy(_xdeck())
    d["name"] = "zdeck synthetic (test_edit_frame)"
    d["geometry"]["symmetry"] = {"planes": {"y": "mirror",
                                            "z": "mirror"}}
    for el in d["geometry"]["electrodes"]:
        for sh in el["shapes"]:
            for node in [sh] + (sh.get("children") or []):
                if "extrude" in node:
                    node["extrude"]["axis"] = "z"
    return d


def _sess(doc: dict) -> EditSession:
    return EditSession(json.dumps(doc).encode(), name=doc["name"])


def _wlab(h: str, v: str) -> str:
    """The ESM's wlab(), replicated verbatim in Python (viewer.py):
    plane name = the two world letters sorted, arrows = horizontal,
    vertical. Change one side and this gate names the divergence."""
    return "".join(sorted([h, v])) + f"  ({h}\u2192, {v}\u2191)"


# ---------------------------------------------------------------------
# F1 — frame authority
# ---------------------------------------------------------------------
def test_F1_frame_is_the_schema_cycle_and_z_is_identity():
    for ax, (a0, a1) in EXTRUDE_INPLANE_AXES.items():
        assert viewer_frame(ax) == {"extrude_axis": ax,
                                    "axes": [a0, a1, ax]}
    assert viewer_frame("z")["axes"] == ["x", "y", "z"]
    assert viewer_axis_indices("z") == (0, 1, 2)
    assert viewer_axis_indices("x") == (1, 2, 0)
    assert viewer_axis_indices("y") == (2, 0, 1)
    with pytest.raises(EditRefusal):
        viewer_frame("w")


# ---------------------------------------------------------------------
# F2 — deck axis resolution and the mixed refusal
# ---------------------------------------------------------------------
def test_F2_container_nodes_carry_no_axis():
    # a cutout WRAPPER without an extrude key must not vote z: this is
    # the bug that framed the StepWave as mixed against itself
    shapes = [{"type": "rect", "x_mm": 0, "y_mm": 0, "width_mm": 1,
               "height_mm": 1,
               "extrude": {"axis": "x", "lo_mm": 0, "hi_mm": 1}},
              {"type": "cutout", "children": [
                  {"type": "ellipse", "cx_mm": 0.5, "cy_mm": 0.5,
                   "rx_mm": 0.2, "ry_mm": 0.2,
                   "extrude": {"axis": "x", "lo_mm": 0, "hi_mm": 1}}]}]
    assert deck_extrude_axis([("E", shapes)]) == "x"


def test_F2_extrudeless_renderable_shape_counts_as_z():
    shapes = [{"type": "rect", "x_mm": 0, "y_mm": 0, "width_mm": 1,
               "height_mm": 1}]
    assert deck_extrude_axis([("E", shapes)]) == "z"
    assert deck_extrude_axis([]) == "z"          # empty deck: identity


def test_F2_mixed_axes_refuse_naming_electrodes_and_axes():
    named = [("EX", [{"type": "rect", "x_mm": 0, "y_mm": 0,
                      "width_mm": 1, "height_mm": 1,
                      "extrude": {"axis": "x", "lo_mm": 0,
                                  "hi_mm": 1}}]),
             ("EZ", [{"type": "rect", "x_mm": 0, "y_mm": 0,
                      "width_mm": 1, "height_mm": 1,
                      "extrude": {"axis": "z", "lo_mm": 0,
                                  "hi_mm": 1}}])]
    with pytest.raises(EditRefusal) as e:
        deck_extrude_axis(named)
    msg = str(e.value)
    for token in ("mixed extrude axes", "EX", "EZ", "'x'", "'z'"):
        assert token in msg, (token, msg)


def test_F2_mixed_deck_refuses_at_session_load():
    doc = _xdeck()
    doc["geometry"]["electrodes"][1]["shapes"][0]["extrude"]["axis"] = "z"
    doc["geometry"]["electrodes"][1]["shapes"][1]["children"][0][
        "extrude"]["axis"] = "z"
    with pytest.raises(EditRefusal) as e:
        _sess(doc)
    assert "mixed extrude axes" in str(e.value)


# ---------------------------------------------------------------------
# F3 — the mirror half-plane clip
# ---------------------------------------------------------------------
def test_F3_clip_identity_for_contained_polygons():
    poly = [[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]
    assert _clip_half(poly, 1) == poly           # SAME vertex values
    assert _clip_half(poly, 0) == poly


def test_F3_clip_straddling_polygon_to_the_stored_half():
    sq = [[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]]
    got = np.array(_clip_half(sq, 1))
    assert got[:, 1].min() >= -1e-12
    assert abs(got[:, 1].max() - 1.0) < 1e-12
    # area halves exactly (shoelace)
    x, y = got[:, 0], got[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert abs(area - 2.0) < 1e-12               # was 4.0


def test_F3_fully_mirrored_polygon_clips_to_empty():
    below = [[0.0, -2.0], [1.0, -2.0], [1.0, -1.0], [0.0, -1.0]]
    assert _clip_half(below, 1) == []


# ---------------------------------------------------------------------
# F4 — the x-deck end to end (render + clamps)
# ---------------------------------------------------------------------
def test_F4_xdeck_renders_hosts_clips_and_reports():
    sess = _sess(_xdeck())
    prims = render_primitives(sess)
    assert prims["frame"] == {"extrude_axis": "x",
                              "axes": ["y", "z", "x"]}
    els = {e["name"]: e for e in prims["electrodes"]}

    hosted = els["HOSTED"]
    assert len(hosted["solids"]) == 1 and not hosted["ghosts"]
    hole = np.array(hosted["solids"][0]["holes"][0])
    assert hole[:, 1].min() >= -1e-12            # clipped at the plane
    assert abs(hole[:, 1].max() - 2.0) < 1e-9    # bore radius survives
    assert hosted["solids"][0]["extrude"]["axis"] == "x"

    openb = els["OPENBORE"]
    # boundary-crossing bore: SUBTRACTED at the outline level (the
    # 2026-09-12 browser-pass fix) — one concave simply-connected
    # solid, no ghost, no warning. The bore mouth is genuinely open:
    # points inside the old bore-∩-plate region are OUTSIDE the metal.
    assert len(openb["solids"]) == 1
    assert not openb["ghosts"] and not openb["reported"]
    sol = openb["solids"][0]
    assert not sol["holes"]                      # simply connected
    assert len(sol["outline"]) > 4               # concave: rect + arc
    assert not _pip((4.0, 0.5), sol["outline"])  # bore interior: open
    assert not _pip((4.9, 0.2), sol["outline"])  # past the plate edge
    assert _pip((1.0, 4.0), sol["outline"])      # plate metal remains
    assert _pip((4.9, 5.0), sol["outline"])      # above the bore
    # area: rect minus the (clipped bore ∩ plate) region — bounded
    # between "full half-disc removed" and "nothing removed"
    a = abs(_sarea(sol["outline"]))
    rect, halfdisc = 5.0 * 6.0, math.pi * 2.0 ** 2 / 2
    assert rect - halfdisc < a < rect, a


def test_F4_world_z_mirror_binds_stored_inplane_y_on_an_x_deck():
    sess = _sess(_xdeck())
    assert sess.clamps == [{
        "target": "shape", "axis": "y", "min": 0.0,
        "reason": sess.clamps[0]["reason"]}]
    r = sess.clamps[0]["reason"]
    assert "mirror in z" in r and "'y'" in r and "x-extruded" in r


def test_F4_z_twin_keeps_the_extrude_clamp_and_identity_frame():
    sess = _sess(_zdeck())
    prims = render_primitives(sess)
    assert prims["frame"] == {"extrude_axis": "z",
                              "axes": ["x", "y", "z"]}
    # both clamp legs, in declaration order: the in-plane y mirror is
    # the PRE-FIX shape-clamp form byte-for-byte; the z mirror is the
    # pre-fix extrude clamp byte-for-byte.
    assert [(c["target"], c["axis"], c["min"]) for c in sess.clamps] \
        == [("shape", "y", 0.0), ("extrude", "z", 0.0)]
    assert "x-extruded" not in sess.clamps[0]["reason"]
    # the y-straddling bore hosts via the same clip, identity frame
    els = {e["name"]: e for e in prims["electrodes"]}
    assert len(els["HOSTED"]["solids"][0]["holes"]) == 1


# ---------------------------------------------------------------------
# F5 — every shipped editor-accepted deck stays on the identity frame
# ---------------------------------------------------------------------
def test_F5_shipped_decks_identity_frame_and_clamp_form():
    seen = 0
    for p in sorted((REPO / "examples").glob("*.json")):
        try:
            sess = EditSession.load(p)
        except EditRefusal:
            continue                     # STL decks: out of editor scope
        prims = render_primitives(sess)
        assert prims["frame"] == {"extrude_axis": "z",
                                  "axes": ["x", "y", "z"]}, p.name
        for c in sess.clamps:
            assert c["axis"] in ("x", "y", "z"), (p.name, c)
        seen += 1
    assert seen >= 15, f"only {seen} shipped decks reached the editor " \
                       f"— the sweep lost its subject"


# ---------------------------------------------------------------------
# F6 — label composition contract (the ESM's wlab, second witness =
#      the browser pass)
# ---------------------------------------------------------------------
def test_F6_identity_labels_reproduce_the_prefix_strings():
    old = {"xy": "xy  (x\u2192, y\u2191)",
           "xz": "xz  (x\u2192, z\u2191)",
           "yz": "yz  (z\u2192, y\u2191)"}
    A = viewer_frame("z")["axes"]
    assert {"xy": _wlab(A[0], A[1]), "xz": _wlab(A[0], A[2]),
            "yz": _wlab(A[2], A[1])} == old


def test_F6_x_frame_labels_name_the_world_planes():
    A = viewer_frame("x")["axes"]                # ['y', 'z', 'x']
    assert _wlab(A[0], A[1]) == "yz  (y\u2192, z\u2191)"
    assert _wlab(A[0], A[2]) == "xy  (y\u2192, x\u2191)"
    assert _wlab(A[2], A[1]) == "xz  (x\u2192, z\u2191)"


# ---------------------------------------------------------------------
# F7 — /flight payload permutation
# ---------------------------------------------------------------------
def _record(spec: dict) -> dict:
    pts = np.array([[10.0, 1.0, 2.0], [20.0, 1.5, 2.5],
                    [30.0, 2.0, 3.0]])           # world (x, y, z)
    return {"subject": {"kind": "single", "spec": spec},
            "paths": [{"pts": pts, "label": "ion 0",
                       "t_us": np.array([0.0, 1.0, 2.0])}],
            "n_paths": 1, "n_flown": 1, "stamp": 1,
            "site": "test_edit_frame", "when": "now",
            "detections": [(5.0, 6.0, 7.0)]}


def test_F7_xdeck_paths_and_detections_permute_world_to_viewer():
    import ion_gym.edit.flight_view as fv
    payload, refusal = fv._compose(_record(_xdeck()), 10)
    assert refusal is None
    assert payload["trajs"]["paths"][0]["pts"] == [
        [1.0, 2.0, 10.0], [1.5, 2.5, 20.0], [2.0, 3.0, 30.0]]
    assert payload["trajs"]["detections"] == [[6.0, 7.0, 5.0]]
    assert payload["prims"]["frame"]["axes"] == ["y", "z", "x"]


def test_F7_zdeck_passes_through_byte_equal():
    import ion_gym.edit.flight_view as fv
    rec = _record(_zdeck())
    payload, refusal = fv._compose(rec, 10)
    assert refusal is None
    assert payload["trajs"]["paths"][0]["pts"] == \
        rec["paths"][0]["pts"].tolist()
    assert payload["trajs"]["detections"] == [[5.0, 6.0, 7.0]]


# ---------------------------------------------------------------------
# F8 — StepWave cross-check (internal fixture: LOUD SKIP when absent)
# ---------------------------------------------------------------------
def test_F8_stepwave_all_140_bores_render_as_metal_absence():
    deck = (REPO / "internal" / "studies" / "stepwave" / "decks"
            / "conjoined_stepwave.json")
    if not deck.is_file():
        pytest.skip("internal/ fixture absent on this checkout: "
                    "internal/studies/stepwave/decks/"
                    "conjoined_stepwave.json — the StepWave cross-check "
                    "needs the dev tree (Q3 loud-skip rule)")
    sess = EditSession.load(deck)
    prims = render_primitives(sess)
    assert prims["frame"] == {"extrude_axis": "x",
                              "axes": ["y", "z", "x"]}
    els = prims["electrodes"]
    assert len(els) == 140
    # EVERY bore is metal absence now: 28 fully-contained bores as
    # holes, 112 open-sided bores as boundary subtractions — zero
    # ghosts, zero warnings (the 2026-09-12 browser-pass finding:
    # 112 ghosted bores drew a solid wall).
    n_holes = sum(len(s["holes"]) for e in els for s in e["solids"])
    n_sub = sum(1 for e in els for s in e["solids"]
                if len(s["outline"]) > 4 and not s["holes"])
    n_ghost = sum(len(e["ghosts"]) for e in els)
    n_rep = sum(len(e["reported"]) for e in els)
    assert (n_holes, n_sub, n_ghost, n_rep) == (28, 112, 0, 0), \
        (n_holes, n_sub, n_ghost, n_rep)
    # the hole/subtraction split equals the raw deck's containment
    # truth — recomputed, never trusted:
    expect = 0
    for el in sess.doc["geometry"]["electrodes"]:
        rect = next(s for s in el["shapes"] if s["type"] == "rect")
        kid = next(s for s in el["shapes"]
                   if s["type"] == "cutout")["children"][0]
        expect += (kid["cx_mm"] + kid["rx_mm"]
                   <= rect["x_mm"] + rect["width_mm"] + 1e-9
                   and kid["cx_mm"] - kid["rx_mm"]
                   >= rect["x_mm"] - 1e-9)
    assert n_holes == expect == 28
    # spot-check a truncated plate (L007): the bore region is OPEN
    l007 = next(e for e in els if e["name"] == "L007")
    out = l007["solids"][0]["outline"]
    assert not _pip((9.5, 0.5), out)      # bore centreline: open
    assert not _pip((16.2, 0.1), out)     # open mouth past the edge
    assert _pip((1.0, 10.0), out)         # plate metal remains
    assert prims["bbox2d"] == [0.0, 0.0, 28.25, 14.25]
    assert sess.clamps[0]["target"] == "shape" \
        and sess.clamps[0]["axis"] == "y"


# ---------------------------------------------------------------------
# F9 — boundary-subtraction degrade paths (every guard exercised)
# ---------------------------------------------------------------------
def test_F9_backend_absent_degrades_to_reported_ghost(monkeypatch):
    import ion_gym.edit.outline as ol

    def _no_backend():
        raise ImportError("manifold3d blocked by the gate")
    monkeypatch.setattr(ol, "_manifold", _no_backend)
    prims = render_primitives(_sess(_xdeck()))
    openb = {e["name"]: e for e in prims["electrodes"]}["OPENBORE"]
    assert len(openb["ghosts"]) == 1
    assert "manifold3d" in openb["ghosts"][0]["reason"]
    assert any("manifold3d" in r for r in openb["reported"])


def test_F9_noop_overlap_kid_stays_a_reported_ghost():
    # kid bbox overlaps the plate but the geometry is disjoint: a
    # subtraction that removes nothing must NOT silently vanish it
    doc = _xdeck()
    doc["geometry"]["electrodes"] = [doc["geometry"]["electrodes"][1]]
    kid = doc["geometry"]["electrodes"][0]["shapes"][1]["children"][0]
    # corner geometry: bboxes overlap (4.5..5 x 5.5..6) but the disc's
    # nearest approach to the plate corner (5, 6) is 2.121 > r = 2, so
    # the subtraction removes nothing
    kid["cx_mm"], kid["cy_mm"] = 6.5, 7.5
    kid["rx_mm"] = kid["ry_mm"] = 2.0
    prims = render_primitives(_sess(doc))
    el = prims["electrodes"][0]
    assert len(el["ghosts"]) == 1
    assert "subtracts no area" in el["ghosts"][0]["reason"]


def test_F9_kid_consuming_the_host_is_reported():
    doc = _xdeck()
    doc["geometry"]["electrodes"] = [doc["geometry"]["electrodes"][1]]
    kid = doc["geometry"]["electrodes"][0]["shapes"][1]["children"][0]
    kid["cx_mm"], kid["cy_mm"] = 2.5, 3.0
    kid["rx_mm"] = kid["ry_mm"] = 50.0        # swallows the plate
    prims = render_primitives(_sess(doc))
    el = prims["electrodes"][0]
    assert len(el["ghosts"]) == 1
    assert "consumes the entire shape" in el["ghosts"][0]["reason"]


def test_F9_kid_splitting_the_host_yields_two_solids():
    # a bar cutting the plate in two: the subtraction must emit BOTH
    # pieces (same shape_index), never drop one
    doc = _xdeck()
    doc["geometry"]["electrodes"] = [{
        "name": "SPLIT", "dc": 0.0, "shapes": [
            {"type": "rect", "x_mm": 0.0, "y_mm": 0.0,
             "width_mm": 8.0, "height_mm": 6.0,
             "extrude": {"axis": "x", "lo_mm": 1.0, "hi_mm": 2.0}},
            {"type": "cutout", "children": [
                {"type": "rect", "x_mm": 3.5, "y_mm": -1.0,
                 "width_mm": 1.0, "height_mm": 9.0,
                 "extrude": {"axis": "x", "lo_mm": 1.0,
                             "hi_mm": 2.0}}]}]}]
    prims = render_primitives(_sess(doc))
    el = prims["electrodes"][0]
    assert not el["ghosts"] and not el["reported"]
    assert len(el["solids"]) == 2
    assert {s["shape_index"] for s in el["solids"]} == {0}
    areas = sorted(abs(_sarea(s["outline"])) for s in el["solids"])
    assert abs(areas[0] - 3.5 * 6.0) < 1e-9   # both halves survive
    assert abs(areas[1] - 3.5 * 6.0) < 1e-9
