"""`/flight` — read-only 3-D view of the LAST FLIGHT.

The page: a separate browser tab showing the flown
geometry with the last trajectories, a clear option, a refusal for
planar geometries, and a selector for how many ions to show.

Design facts this module rests on:

* Each browser tab is its OWN Panel session, so the flight arrives via
  the server-global slot (`ion_gym.ui.last_flight`) — published by every
  sim_app fly site, integrity-verified on every read, clearable by name.
* The record is SELF-CONTAINED (it carries the spec/instrument document
  that flew), so this page draws the geometry of THE FLIGHT, never
  whatever the dashboard holds now — staleness impossible by
  construction.
* PLANAR HANDLING (this SUPERSEDED an earlier whole-refusal):
  a single-FA planar subject
  RENDERS — electrodes get the editor's display-only ghost slab (10% of
  the smaller in-plane dimension, the editor convention, disclosed in
  the legend) and trajectories draw at their real banked coordinates,
  i.e. on the plane they were simulated in. In a MIXED assembly the
  planar STAGES likewise draw as disclosed display-only slabs while
  non-planar stages draw their declared extents. NOTE: this
  header previously still described the retired refusal and a status
  report was given FROM the header instead of the code — the header is
  part of the contract and must move in the same change as the rule.
* Rendering goes through the SAME EditorViewer as the editor (one
  renderer; gizmo and selection off), so lathe/extrude conventions and
  the camera memo carry over. The memo keys on the record stamp: a
  Refresh of the same flight keeps your view; a new flight reframes.
"""
from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
import panel as pn

from ion_gym.ui import last_flight

DEFAULT_SHOW = 50          # matches assembly_overview's default path count
MAX_PTS_PER_PATH = 2000    # per-path record decimation for payload size


def _route_of(spec_dict: dict) -> str:
    """The builder name from `build_route()` — the ONE route authority
    (policy.py's one-authority rule). The first implementation compared the
    BuildRoute DATACLASS to the string "planar", which never matched,
    so the planar refusal was dead on arrival — caught by the headless
    drive, banked here so the comparison stays on the named field."""
    from ion_gym.io.sim_spec import SimSpec
    from ion_gym.physics.sim_build import build_route
    return build_route(SimSpec.from_dict(spec_dict)).builder


def _stage_prims(spec_dict: dict, name: str, offset):
    """One stage's render primitives, pose-offset into the world frame.
    In-plane outlines shift by (dx, dy); extrude ranges by dz. Returns
    (electrodes, reported) — a refusal is reported, never silent."""
    from ion_gym.edit.session import EditSession
    from ion_gym.edit.outline import render_primitives
    dx, dy, dz = (list(offset) + [0.0, 0.0, 0.0])[:3]
    sess = EditSession(json.dumps(spec_dict).encode(),
                       name=f"stage:{name}")
    prims = render_primitives(sess)
    for el in prims["electrodes"]:
        el["name"] = f"{name}:{el['name']}"
        for entry in el.get("solids", []) + el.get("ghosts", []):
            entry["outline"] = [[p[0] + dx, p[1] + dy]
                                for p in entry["outline"]]
            entry["holes"] = [[[p[0] + dx, p[1] + dy] for p in h]
                              for h in entry.get("holes", [])]
            if entry.get("extrude"):
                entry["extrude"] = {
                    "lo_mm": entry["extrude"]["lo_mm"] + dz,
                    "hi_mm": entry["extrude"]["hi_mm"] + dz}
    return prims["electrodes"]


def _compose(record: dict, n_show: int, style: Optional[dict] = None,
             electrode_colors: str = "deck",
             fa_colors: Optional[dict] = None,
             quarter: bool = False, ghost: bool = False):
    """(payload, subject_refusal_or_None). The payload is the
    EditorViewer contract: scene + prims (+ trajs).
    electrode_colors: 'deck' keeps each deck's own colors; 'stage'
    paints every electrode of a stage one palette color (assemblies —
    different electrode colors per FA). For a
    single-FA subject the deck colors are the per-FA color, so 'stage'
    is a no-op there by construction, not by a silent branch.
    quarter/ghost are INDEPENDENT display-only flags:
    checkboxes, combinable — supersedes the 3-state
    solid/quarter/ghost mode). ghost renders every solid translucent;
    quarter asks the viewer to clip the +x/+y quadrant about the
    electrode bbox centre via prims['quarter_cut'] — the ESM's
    material clipping planes cut EVERY stage's metal identically
    (r-z annuli AND native 3-D rods/plates), which is what retires the
    r-z-only C-ring construction that used to live in the r-z branch
    below. The flown geometry is uncut either way; the viewer legend
    states both."""
    subject = record["subject"]

    if subject["kind"] == "single":
        spec = subject["spec"]
        route = _route_of(spec)
        # PLANAR REFUSAL SUPERSEDED: a minimum width is applied to
        # the electrodes and the trajectories live on the plane in
        # which they are flown. A planar subject now RENDERS through
        # the editor's own planar path, whose out_of_plane ghost slab
        # IS that convention (10% of the smaller in-plane dimension,
        # display-only, disclosed in the legend — one
        # authority). Trajectories draw at their real banked
        # coordinates. The earlier refusal is retired, not routed
        # around.
        from ion_gym.edit.session import EditSession
        from ion_gym.edit.outline import render_primitives
        sess = EditSession(json.dumps(spec).encode(), name="flight-subject")
        scene = sess.editor_scene()
        # the editor's planar policy views persp+xy (an EDITING choice);
        # flight trajectories are honestly 3-D, so this page shows all
        # four — the slab ghost carries its own display-only disclosure.
        scene.setdefault("policy", {})["viewports"] = [
            "persp", "xy", "xz", "yz"]
        prims = render_primitives(sess)
        _ov = (fa_colors or {}).get("(subject)")
        if _ov:
            for el in prims["electrodes"]:
                el["color"] = list(_ov)
    else:
        doc = subject["doc"]
        electrodes: List[dict] = []
        stage_notes: List[str] = []
        drew_any = False
        for st in doc.get("stages", []):
            nm = st.get("name", "?")
            pose = st.get("pose", {}) or {}
            rot = pose.get("rot_deg", [0.0, 0.0, 0.0]) or [0, 0, 0]
            off = pose.get("offset_mm", [0.0, 0.0, 0.0]) or [0, 0, 0]
            try:
                route = _route_of(st["spec"])
            except Exception as e:
                stage_notes.append(f"stage {nm}: unreadable spec — "
                                   f"{type(e).__name__}: {e}")
                continue
            if route == "planar":
                # SLAB GHOSTS, NOT REFUSAL:
                # a planar stage draws as a display-only slab whose
                # thickness is 10% of its smaller in-plane dimension —
                # the SAME convention the editor uses for planar
                # undeclared-z (one authority). Rendered as
                # GHOSTS (translucent) so declared 3-D metal and
                # display-only slabs cannot be confused, and the legend
                # says so per stage. Trajectories draw at their real
                # banked coordinates regardless.
                try:
                    els_p = _stage_prims(st["spec"], nm, off)
                except Exception as e:
                    stage_notes.append(f"stage {nm}: {type(e).__name__}: {e}")
                    continue
                _xy = [p for el in els_p
                       for en in el.get("solids", [])
                       for p in en["outline"]]
                if not _xy:
                    stage_notes.append(f"stage {nm}: planar with no "
                                       f"drawable outlines")
                    continue
                _w = max(p[0] for p in _xy) - min(p[0] for p in _xy)
                _h = max(p[1] for p in _xy) - min(p[1] for p in _xy)
                _t = 0.10 * min(_w, _h)
                _dz = (list(off) + [0.0, 0.0, 0.0])[2]
                for el in els_p:
                    ghosts = el.get("ghosts", [])
                    for en in el.get("solids", []):
                        en["extrude"] = {"lo_mm": _dz - 0.5 * _t,
                                         "hi_mm": _dz + 0.5 * _t}
                        ghosts.append(en)
                    el["ghosts"] = ghosts
                    el["solids"] = []
                electrodes.extend(els_p)
                drew_any = True
                stage_notes.append(
                    f"stage {nm}: planar — drawn as a DISPLAY-ONLY slab "
                    f"({_t:.2f} mm = 10% of its smaller in-plane "
                    f"dimension, the L-193 convention); its z extent is "
                    f"NOT declared geometry")
                continue
            if route == "rz":
                # r-z STAGE, DRAWN (an r-z stage used to be
                # skipped). A posed r-z stage whose axial axis (LOCAL X,
                # the r-z kernel convention) lands on WORLD Z is exactly
                # expressible in this viewer's primitive vocabulary: each
                # (axial, radial) rect revolves to an ANNULUS — an
                # outline circle with a bore hole — extruded over its
                # world-z axial span. Any other posed axis is refused by
                # name (this viewer extrudes along z only).
                from ion_gym.physics.staged_flight import _rot_matrix
                import numpy as _np
                _R = _rot_matrix(rot if any(abs(float(v)) > 1e-9
                                            for v in rot) else None)
                _a = _R @ _np.array([1.0, 0.0, 0.0])
                if abs(abs(_a[2]) - 1.0) > 1e-9:
                    stage_notes.append(
                        f"stage {nm}: r-z with its axis posed along "
                        f"{_np.round(_a, 3).tolist()} — this viewer "
                        f"extrudes along z only; pose the axial axis on "
                        f"world z to draw it")
                    continue
                _sgn = 1.0 if _a[2] > 0 else -1.0
                _t3 = (list(off) + [0.0, 0.0, 0.0])[:3]
                _g = st["spec"]["geometry"]
                _th = _np.linspace(0, 2 * _np.pi, 48, endpoint=False)
                for _ei, _el in enumerate(_g.get("electrodes", [])):
                    _solids = []
                    for _sh in (_el.get("shapes") or []):
                        if _sh.get("type") != "rect":
                            continue
                        _x0 = float(_sh["x_mm"]); _dx = float(_sh["width_mm"])
                        _ri = float(_sh["y_mm"])
                        _ro = _ri + float(_sh["height_mm"])
                        _lo, _hi = sorted((_sgn * _x0 + _t3[2],
                                           _sgn * (_x0 + _dx) + _t3[2]))
                        # Always the FULL annulus. The quarter cutaway
                        # that used to build a 270-degree C-ring polygon
                        # HERE cut only r-z stages (the quarter cut
                        # must extend across the entire
                        # assembly) — it is superseded by the viewer's
                        # clipping planes (see prims['quarter_cut']
                        # below), which cut every route's meshes
                        # identically, concave-triangulation-free.
                        _outl = [[_t3[0] + _ro * float(_np.cos(a)),
                                  _t3[1] + _ro * float(_np.sin(a))]
                                 for a in _th]
                        _hole = ([[[_t3[0] + _ri * float(_np.cos(a)),
                                    _t3[1] + _ri * float(_np.sin(a))]
                                   for a in _th]] if _ri > 0 else [])
                        _solids.append({"outline": _outl, "holes": _hole,
                                        "extrude": {"lo_mm": _lo,
                                                    "hi_mm": _hi}})
                    if _solids:
                        electrodes.append({
                            "index": _ei, "name": f"{nm}:{_el['name']}",
                            "color": list(_el.get("color")
                                          or (31, 119, 180)),
                            "is_grid": bool(_el.get("is_grid")),
                            "editable": False, "solids": _solids,
                            "ghosts": [], "reported": []})
                        drew_any = True
                stage_notes.append(
                    f"stage {nm}: r-z drawn as revolved annuli (axial "
                    f"local x posed on world z)")
                continue
            if any(abs(float(r)) > 1e-9 for r in rot):
                stage_notes.append(
                    f"stage {nm}: rotated pose {rot} — refused "
                    f"(only r-z stages posed axially on world z have a "
                    f"drawable rotation in this viewer)")
                continue
            # STORED FRAME for the native 3-D routes (trajectories
            # drew offset — a rod bundle drew 7.8 mm
            # off the beam). The flight and every other renderer apply
            # the pose to the STORED frame (shape − origin_mm; tracer3d
            # flies the stored frame, world_off_mm is diagnostic-only),
            # while this viewer shifted the RAW deck frame. Same
            # build_route discrimination as assembly_overview; the signed-frame
            # note applies here identically.
            _off2 = list(off) + [0.0, 0.0, 0.0]
            if route in ("shapes3d", "stl3d", "scene3d"):
                _org = ((st["spec"].get("geometry") or {})
                        .get("origin_mm") or (0.0, 0.0))
                _off2 = [_off2[0] - float(_org[0]),
                         _off2[1] - float(_org[1]), _off2[2]]
            try:
                electrodes.extend(_stage_prims(st["spec"], nm, _off2[:3]))
                drew_any = True
            except Exception as e:
                stage_notes.append(f"stage {nm}: {type(e).__name__}: {e}")
        if not drew_any and not record["paths"]:
            return None, (
                "nothing to draw: every stage refused ("
                + "; ".join(stage_notes) + ") and the record carries no "
                "trajectories.")
        all_xy = [p for el in electrodes
                  for entry in el.get("solids", []) + el.get("ghosts", [])
                  for p in entry["outline"]]
        if fa_colors:
            for el in electrodes:
                _stg = str(el["name"]).split(":", 1)[0]
                if _stg in fa_colors:
                    el["color"] = list(fa_colors[_stg])
        if electrode_colors == "stage":
            # one palette color per stage; the prefix "stage:" naming
            # from _stage_prims is the grouping key. Colors are the
            # project palette so /flight matches the app's stage hues.
            _pal = [(31, 119, 180), (255, 127, 14), (44, 160, 44),
                    (214, 39, 40), (148, 103, 189), (23, 190, 207),
                    (140, 86, 75), (227, 119, 194)]
            _order = []
            for el in electrodes:
                _stg = str(el["name"]).split(":", 1)[0]
                if _stg not in _order:
                    _order.append(_stg)
                if _stg not in (fa_colors or {}):   # override wins
                    el["color"] = list(_pal[_order.index(_stg) % len(_pal)])
        bbox = ([min(p[0] for p in all_xy), min(p[1] for p in all_xy),
                 max(p[0] for p in all_xy), max(p[1] for p in all_xy)]
                if all_xy else [0.0, 0.0, 1.0, 1.0])
        # (ghost/quarter handling moved AFTER the subject branches — it
        # applies to single-FA subjects too, where the old placement
        # made the page-level control a silent no-op.)
        # per-stage refusals ride the reported channel of a pseudo
        # entry so the ESM legend (which reads e.reported) states them
        electrodes.append({"index": -1, "name": "assembly",
                           "color": [120, 120, 130], "is_grid": False,
                           "editable": False, "solids": [], "ghosts": [],
                           "reported": stage_notes})
        scene = {"schema": 1, "document": doc.get("name", "assembly"),
                 "route": "assembly", "out_of_plane": {},
                 "policy": {"viewports": ["persp", "xy", "xz", "yz"]}}
        prims = {"electrodes": electrodes, "bbox2d": bbox,
                 "pitch_mm": 0.0}

    # DISPLAY-ONLY ELECTRODE FLAGS, both subject kinds:
    # ghost + quarter are independent checkboxes, combinable; the old
    # 3-state mode also applied only to assemblies, leaving the control
    # a silent no-op on a single-FA subject).
    if ghost:
        # TRANSLUCENT ELECTRODES: every solid renders as a ghost so the
        # trajectories inside are visible; the viewer legend discloses
        # ghosts as display-only.
        for el in prims["electrodes"]:
            el["ghosts"] = el.get("ghosts", []) + el.get("solids", [])
            el["solids"] = []
    if quarter:
        # QUARTER CUTAWAY: the viewer clips the +x/+y quadrant of EVERY
        # electrode mesh about one centre — derived from the drawn
        # metal's own xy bounding box, never a baked-in axis, so it
        # holds for any assembly or single FA without a declared beam
        # axis. With no drawable metal there is nothing to cut and no
        # centre to derive: the flag is dropped WITH A REPORT rather
        # than clipping about a guessed point.
        _qxy = [p for el in prims["electrodes"]
                for entry in el.get("solids", []) + el.get("ghosts", [])
                for p in entry["outline"]]
        if _qxy:
            prims["quarter_cut"] = {
                "x_mm": 0.5 * (min(p[0] for p in _qxy)
                               + max(p[0] for p in _qxy)),
                "y_mm": 0.5 * (min(p[1] for p in _qxy)
                               + max(p[1] for p in _qxy))}
        else:
            prims["electrodes"].append({
                "index": -1, "name": "quarter cutaway",
                "color": [120, 120, 130], "is_grid": False,
                "editable": False, "solids": [], "ghosts": [],
                "reported": ["quarter cutaway requested but this record "
                             "has no drawable electrode geometry — "
                             "nothing was cut"]})

    # trajectories: evenly strided ion subset (spans
    # the retained set, not its head), per-path record decimation, and
    # a legend that states shown/retained/flown so a subset never reads
    # as the packet.
    paths = record["paths"]
    n_show = max(0, int(n_show))
    if paths and n_show and n_show < len(paths):
        idx = np.linspace(0, len(paths) - 1, n_show).round().astype(int)
        shown = [paths[i] for i in sorted(set(idx.tolist()))]
    else:
        shown = list(paths)

    # PARAMETER COLORING (a color selector by ion
    # parameter — KE or velocity — as in the main view). Speed and
    # time are DERIVABLE from what the record banks (pts + t_us):
    # speed = |Δpts|/Δt per segment in mm/µs, assigned per point. KE is
    # NOT drawable yet — it needs per-ion mass, which the record does
    # not bank; that is a record-schema addition, not a coloring
    # option, so it is absent rather than faked. A record without t_us
    # REFUSES the scheme by name and falls back to the per-ion ramp —
    # stated in the legend, never silent.
    _scheme = (style or {}).get("color_by", "")
    _want_param = _scheme in ("speed", "time", "ke")
    _param_note = ""
    _raw_vals = []
    if _want_param:
        _m = record.get("m_amu")
        if any(p.get("t_us") is None for p in shown):
            _param_note = (f"{_scheme} coloring unavailable: this "
                           f"record lacks t_us — using per-ion ramp")
            _want_param = False
            style = dict(style); style["color_by"] = "ion"
        elif _scheme == "ke" and _m is None:
            _param_note = ("KE coloring unavailable: this record banks "
                           "no single packet mass (mixed or undeclared "
                           "m/z) — using per-ion ramp")
            _want_param = False
            style = dict(style); style["color_by"] = "ion"
        else:
            for p in shown:
                t = np.asarray(p["t_us"], float)
                if _scheme == "time":
                    _raw_vals.append(t.copy())
                else:
                    d = np.diff(np.asarray(p["pts"], float), axis=0)
                    dt = np.maximum(np.diff(t), 1e-12)
                    seg = np.linalg.norm(d, axis=1) / dt   # mm/us
                    v = np.r_[seg[:1], seg]
                    if _scheme == "ke":
                        # KE_eV = 0.5 m v^2 / e; v[mm/us] = 1e3 m/s.
                        # Constants from collision3d — one authority.
                        from ion_gym.physics.collision3d import (E_CHG,
                                                                 KG_AMU)
                        v = (0.5 * _m * KG_AMU * (v * 1e3) ** 2) / E_CHG
                    _raw_vals.append(v)
            _all = np.concatenate(_raw_vals) if _raw_vals else np.zeros(1)
            _vmin, _vmax = float(_all.min()), float(_all.max())

    out_paths = []
    for _pi, p in enumerate(shown):
        pts = p["pts"]
        vals = _raw_vals[_pi] if _want_param else None
        if len(pts) > MAX_PTS_PER_PATH:
            stride = int(np.ceil(len(pts) / MAX_PTS_PER_PATH))
            keep = np.r_[np.arange(0, len(pts) - 1, stride),
                         len(pts) - 1]          # endpoint always kept
            pts = pts[keep]
            if vals is not None:
                vals = vals[keep]               # lockstep with pts
        entry = {"pts": pts.tolist(), "label": p["label"]}
        if vals is not None:
            _spread = (_vmax - _vmin) or 1.0
            entry["vals"] = ((vals - _vmin) / _spread).tolist()
        out_paths.append(entry)

    legend = (f"{len(out_paths)} shown of {record['n_paths']} retained "
              f"({record['n_flown']} flown) — record #{record['stamp']} "
              f"{record['site']} {record['when']}"
              + (f" · {record['note']}" if record.get("note") else ""))
    if _want_param and out_paths:
        _unit = {"time": "µs", "speed": "mm/µs", "ke": "eV"}[_scheme]
        legend += (f" · colored by {_scheme} {_vmin:.4g}–{_vmax:.4g} "
                   f"{_unit} (blue→red, range over shown paths)")
    if _param_note:
        legend += f" · {_param_note}"
    if not out_paths:
        legend += " · NO PATHS in this record (trace storage was off?)"
    _det = record.get("detections")
    if _det:
        legend += (f" · {len(_det)} detection crossings marked "
                   f"(station is pass-through)")
    payload_trajs = {"paths": out_paths, "legend": legend}
    if _det:
        payload_trajs["detections"] = [list(d) for d in _det]
    if style:
        payload_trajs["style"] = dict(style)
    return {"scene": scene, "prims": prims, "trajs": payload_trajs}, None


def _compose_style(color_by: str, cmap: str, color: str,
                   width_px: float, opacity: float) -> dict:
    """The trajectory style block the ESM consumes. Values are clamped
    HERE so displayed equals drawn — the JS applies them verbatim.
    color_by: ion | solid | speed | time | ke.  cmap: the ramp used by
    every non-solid mode (cool-warm | plasma | viridis)."""
    return {"color_by": color_by,
            "cmap": cmap,
            "color": color,
            "width_px": float(min(8.0, max(0.5, width_px))),
            "opacity": float(min(1.0, max(0.05, opacity)))}


def flight_page(title: str = "3D Flight Viewer") -> pn.Column:
    """One /flight session: slot snapshot + n-ions selector + Clear +
    Refresh. Rebuilt per browser session; Refresh re-reads the slot."""
    from ion_gym.edit.viewer import EditorViewer

    w_n = pn.widgets.IntInput(name="ions to show", value=DEFAULT_SHOW,
                              start=1, step=10, width=120)
    b_refresh = pn.widgets.Button(name="Refresh", width=90)
    b_clear = pn.widgets.Button(name="Clear last flight",
                                button_type="danger", width=140)
    status = pn.pane.Markdown("", sizing_mode="stretch_width")
    holder = pn.Column(sizing_mode="stretch_width")
    # trajectory style. Width is real: Line2 fat
    # lines, because WebGL ignores LineBasicMaterial.linewidth — a
    # plain width knob would silently do nothing on most platforms.
    w_scheme = pn.widgets.Select(
        name="color trajectories by",
        options={"ion (ramp per ion)": "ion",
                 "solid color": "solid",
                 "speed (mm/µs)": "speed",
                 "time (µs)": "time",
                 "kinetic energy (eV)": "ke"},
        value="ion", width=170)
    w_cmap = pn.widgets.Select(
        name="colormap",
        options={"cool → warm": "cool-warm", "plasma": "plasma",
                 "viridis": "viridis"},
        value="cool-warm", width=130)
    # CHECKBOXES, not a 3-state mode: ghost and quarter are
    # checkboxes so they can be applied
    # simultaneously. Quarter clips the +x/+y quadrant out of EVERY
    # stage's metal — r-z annuli and native 3-D alike — about the drawn
    # metal's bbox centre (display only; the flown geometry is uncut).
    # Ghost renders all electrodes translucent. Both flags compose.
    # (No description= — Checkbox lacks it on the supported Panel
    # build; the sim_app note on _mkw records the same constraint.)
    w_quarter = pn.widgets.Checkbox(name="quarter cutaway (display only)",
                                    value=False, width=200)
    w_ghost = pn.widgets.Checkbox(name="ghost (translucent)",
                                  value=False, width=160)
    w_ecolor = pn.widgets.Select(
        name="electrode colors",
        options={"deck colors": "deck", "per stage (assemblies)": "stage"},
        value="deck", width=170)
    # PER-FA COLOR OVERRIDE (a selector
    # for the FA plus a button to apply a color to a given FA).
    # Overrides layer on top of the mode above and win per stage; the
    # page holds them until Reset. The FA list follows the record.
    fa_colors: dict = {}
    w_fa = pn.widgets.Select(name="FA / stage", options=["(subject)"],
                             value="(subject)", width=140)
    w_fa_color = pn.widgets.ColorPicker(name="color", value="#4472c4",
                                        width=70)
    b_fa_apply = pn.widgets.Button(name="Apply color to FA", width=140)
    b_fa_reset = pn.widgets.Button(name="Reset colors", width=110)
    w_color = pn.widgets.ColorPicker(name="solid", value="#c83c3c",
                                     width=70, disabled=True)
    w_width = pn.widgets.FloatInput(name="line width (px)", value=2.0,
                                    start=0.5, end=8.0, step=0.5,
                                    width=120)
    w_alpha = pn.widgets.FloatInput(name="opacity", value=0.85,
                                    start=0.05, end=1.0, step=0.05,
                                    width=100)

    def _style() -> dict:
        return _compose_style(w_scheme.value, w_cmap.value, w_color.value,
                              w_width.value or 2.0, w_alpha.value or 0.85)

    def _draw(_=None):
        try:
            rec = last_flight.snapshot()
        except RuntimeError as e:          # corrupted slot: named, never drawn
            holder[:] = []
            status.object = f"**{e}**"
            return
        if rec is None:
            holder[:] = []
            status.object = ("**no flight banked** — fly something in "
                             "the dashboard, then press Refresh.")
            return
        _fas = (["(subject)"] if rec["subject"]["kind"] == "single"
                else [s.get("name", "?") for s in
                      rec["subject"]["doc"].get("stages", [])])
        if list(w_fa.options) != _fas:
            w_fa.options = _fas
            if w_fa.value not in _fas:
                w_fa.value = _fas[0]
        payload, refusal = _compose(rec, w_n.value or DEFAULT_SHOW,
                                    style=_style(),
                                    electrode_colors=w_ecolor.value,
                                    quarter=w_quarter.value,
                                    ghost=w_ghost.value,
                                    fa_colors=fa_colors)
        if refusal is not None:
            holder[:] = []
            status.object = (f"**record #{rec['stamp']} "
                             f"({rec['site']}, {rec['when']}): "
                             f"{refusal}**")
            return
        # Camera-memo key (the flight perspective once
        # changed on refresh of the flight deck): keyed by the
        # record STAMP, every new flight was a new key and reframed the
        # view. The pose belongs to the INSTRUMENT — same subject keeps
        # the user's perspective across flights; a different subject
        # (different scale) still gets fresh framing.
        _subj = rec["subject"]
        if _subj["kind"] == "single":
            _skey = (_subj.get("spec") or {}).get("name") or "single"
        else:
            _doc = _subj.get("doc") or {}
            _skey = (_doc.get("name")
                     or "+".join(s.get("name", "?")
                                 for s in _doc.get("stages", []))
                     or "staged")
        v = EditorViewer(payload=payload, sizing_mode="stretch_width",
                         canvas_w=1100, canvas_h=520, selected=-1,
                         selected_shape=-1, gizmo_enabled=False,
                         doc_id=f"flight:{_subj['kind']}:{_skey}")
        holder[:] = [v]
        status.object = (f"record **#{rec['stamp']}** · {rec['site']} · "
                         f"{rec['when']} · {rec['n_paths']} paths "
                         f"retained of {rec['n_flown']} flown")

    def _clear(_):
        msg = last_flight.clear("cleared from the /flight tab")
        holder[:] = []
        status.object = f"**{msg}**"

    def _hex_rgb(h):
        h = h.lstrip("#")
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)]

    def _fa_apply(_):
        fa_colors[w_fa.value] = _hex_rgb(w_fa_color.value)
        _draw()

    def _fa_reset(_):
        fa_colors.clear()
        _draw()

    b_fa_apply.on_click(_fa_apply)
    b_fa_reset.on_click(_fa_reset)
    b_refresh.on_click(_draw)
    b_clear.on_click(_clear)
    w_n.param.watch(lambda _e: _draw(), "value")
    for _w in (w_scheme, w_cmap, w_color, w_width, w_alpha, w_ecolor,
               w_quarter, w_ghost):
        _w.param.watch(lambda _e: _draw(), "value")
    w_scheme.param.watch(
        lambda e: setattr(w_color, "disabled", e.new != "solid"), "value")
    _draw()

    return pn.Column(
        pn.pane.Markdown(f"## {title}"),
        pn.Row(w_n, b_refresh, b_clear, status),
        pn.Row(w_scheme, w_cmap, w_color, w_width, w_alpha),
        pn.Row(w_ecolor, w_quarter, w_ghost, w_fa, w_fa_color,
               b_fa_apply, b_fa_reset),
        holder, sizing_mode="stretch_width")
