"""Numeric geometry editor.

Selection + numeric editing and creation, NO gizmo: pick an electrode
and shape, edit its parameters (and extrude lo/hi on shapes3d) in typed
fields, add shapes and electrodes from the policy vocabulary, undo, and
save.  Apply runs the Slice 0 targeted-patch commit and its
verify_commit proof every time; refusals and warnings are shown by
name; the geometry-key banner is always visible and turns to a warning
the moment the document diverges from its loaded key.

Rules carried here:
  * the field decides: values are snapped to the pitch lattice BEFORE
    apply when "snap to pitch" is on, and the snapped number is written
    back into the widget so displayed == committed.  Snap off -> the
    off-lattice value goes through with a NAMED warning, never silently.
  * a same-value Apply queues NOTHING (a no-op apply is
    byte-identical), and the status says so.
  * this editor launches NO solve.  The banner states the key change
    and that solved fields for the old key are invalid; re-solving
    happens in sim_app / notebooks under the X1 quote, not here.
  * Save writes the committed bytes to the named path and RELOADS the
    session from what was written, so further edits stand on saved
    truth, never on a diverged in-memory copy.

Serve:  through the editor demo entry point.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import panel as pn

from ion_gym.edit.policy import EditRefusal
from ion_gym.edit.session import EditSession
from ion_gym.edit.outline import shift_shape
from ion_gym.edit.viewer import viewer_for_session, DEFAULT_WIDTH

# editor canvas: STRETCH policy — the viewer
# tracks its container width; this is only the pre-layout fallback.
# Height stays declared.
EDITOR_CANVAS_W = 1100
EDITOR_CANVAS_H = 640

# per-type numeric fields (rotation_deg is never snapped: it is not a
# length on the pitch lattice)
_FIELDS: Dict[str, List[str]] = {
    "rect": ["x_mm", "y_mm", "width_mm", "height_mm", "rotation_deg"],
    "ellipse": ["cx_mm", "cy_mm", "rx_mm", "ry_mm"],
}
_NO_SNAP = {"rotation_deg"}

# optional keys with a defined absent-default: the form must not
# MATERIALIZE them on a no-op apply (writing rotation_deg: 0.0 into a
# shape that never had it is a phantom edit — found by the E2 check)
_OPTIONAL_DEFAULTS = {"rotation_deg": 0.0}

_NEW_SHAPE_SPAN_PITCHES = 8   # default size of a created shape


class EditorApp:
    """One document, one session, one viewer, one commit path."""

    def __init__(self, path, *, upload=None):
        # `upload=(name, data)` loads from a BROWSER UPLOAD (the OS
        # file dialog — same FileInput as the main
        # panel). The dialog delivers bytes + filename, never a disk
        # path, so an upload-origin document has self.path = None and
        # an EMPTY save path: Save already writes to whatever the
        # save-path box says (Save-As semantics), and inventing a
        # destination silently would be a guessed value, i.e. a bug.
        if upload is not None:
            up_name, up_data = upload
            self.path = None
            self.session = EditSession(bytes(up_data), name=up_name)
        else:
            self.path = Path(path)
            self.session = EditSession.load(self.path)
        self.loaded_key = self.session.geometry_key_short()

        self.banner = pn.pane.Alert("", alert_type="success",
                                    sizing_mode="stretch_width")
        self.status = pn.pane.Markdown("", width=DEFAULT_WIDTH)
        self.viewer_holder = pn.Column(
            sizing_mode="stretch_width")
        self._suppress_pick = False

        self.w_el = pn.widgets.Select(name="electrode", options=[],
                                      width=280)
        self.w_sh = pn.widgets.Select(name="shape", options=[], width=280)
        self.w_snap = pn.widgets.Checkbox(name="snap to pitch",
                                          value=True)
        self.w_contours = pn.widgets.Checkbox(
            name="contours (DC potential; solves the 2-D field on demand)",
            value=False)
        self.w_contours.param.watch(
            lambda _e: self._refresh_viewer(), "value")
        self.w_gizmo = pn.widgets.Checkbox(
            name="gizmo (drag the selected shape; snaps to pitch)",
            value=True)
        self.w_gizmo.param.watch(self._on_gizmo_toggle, "value")
        self.fields = pn.Column()
        self._field_widgets: Dict[str, pn.widgets.Widget] = {}

        self.w_apply = pn.widgets.Button(name="Apply",
                                         button_type="primary", width=90)
        self.w_undo = pn.widgets.Button(name="Undo", width=70)
        self.w_redo = pn.widgets.Button(name="Redo", width=70)
        self.w_remove = pn.widgets.Button(name="Remove shape", width=110)

        self.w_new_type = pn.widgets.Select(
            name="new shape type",
            options=[t for t in
                     self.session.policy.editable_shape_types
                     if t != "cutout"],
            width=140)
        self.w_add_shape = pn.widgets.Button(
            name="Add shape", width=170,
            description="Extends the SELECTED electrode: more "
                        "metal on the SAME conductor at the same "
                        "potential (e.g. a SLIM electrode built "
                        "from several ellipses and rects).")
        self.w_new_el = pn.widgets.TextInput(name="new electrode name",
                                             width=180)
        self.w_add_el = pn.widgets.Button(
            name="Add electrode", width=120,
            description="Creates a NEW conductor with its own "
                        "potential, dc value and tuning degree "
                        "of freedom, holding one shape of the "
                        "chosen type.")

        self.w_color = pn.widgets.ColorPicker(
            name="electrode color", value="#c83c3c", width=140)
        self.w_color_one = pn.widgets.Button(
            name="Color electrode", width=130)
        self.w_color_all = pn.widgets.Button(
            name="Color all", width=90)
        self.w_save_path = pn.widgets.TextInput(
            name="save path",
            value=str(self.path) if self.path is not None else "",
            placeholder=("uploaded document — type a destination path "
                         "to save"),
            width=460)
        self.w_save = pn.widgets.Button(name="Save",
                                        button_type="success", width=80)

        self.w_apply.on_click(self._on_apply)
        self.w_undo.on_click(self._on_undo)
        self.w_redo.on_click(self._on_redo)
        self.w_remove.on_click(self._on_remove)
        self.w_add_shape.on_click(self._on_add_shape)
        self.w_add_el.on_click(self._on_add_electrode)
        self.w_save.on_click(self._on_save)
        self.w_color_one.on_click(self._on_color_one)
        self.w_color_all.on_click(self._on_color_all)
        self.w_el.param.watch(self._on_pick_el, "value")
        self.w_sh.param.watch(self._on_pick_sh, "value")

        self._refresh_viewer()
        self._rebuild_pickers()
        self._refresh_banner()

    # -- helpers ----------------------------------------------------------
    def _say(self, text: str) -> None:
        self.status.object = text

    def _refuse(self, e: Exception) -> None:
        self._say(f"**REFUSED:** {e}")

    def _shapes_of(self, el_i: int) -> list:
        # COMMITTED state (v420 defect: reading the original doc
        # made added shapes invisible and stale after edits)
        return self.session.committed_electrodes()[el_i] \
            .get("shapes", [])

    def _refresh_banner(self) -> None:
        n = self.session.edit_count
        key = self.loaded_key
        if n == 0:
            self.banner.alert_type = "success"
            self.banner.object = (
                f"geometry key **{key}** — document matches its loaded "
                f"key; no edits queued")
        else:
            preview = EditSession(self.session.to_document_bytes(),
                                  name=self.session.name)
            if preview.geometry_key_short() == key:
                self.banner.alert_type = "primary"
                self.banner.object = (
                    f"{n} display-only edit op(s) queued (colors) — "
                    f"geometry key **{key}** unchanged; solved "
                    f"fields remain valid")
            else:
                self.banner.alert_type = "warning"
                self.banner.object = (
                    f"**GEOMETRY CHANGED** — {n} edit op(s) queued "
                    f"on top of key **{key}**: any solved field for "
                    f"that key is invalid once saved. This editor "
                    f"launches no solve; re-solve in sim_app / a "
                    f"notebook (X1 quote applies there).")

    def _refresh_viewer(self) -> None:
        # a fresh session view of the COMMITTED bytes: what you see is
        # what would be saved, holes/ghosts re-resolved by the same
        # Python authorities as always
        preview = EditSession(self.session.to_document_bytes(),
                              name=self.session.name)
        sel = self.w_el.value if self.w_el.value is not None else -1
        sel_sh = self.w_sh.value if self.w_sh.value is not None \
            else -1
        cont = None
        if getattr(self, "w_contours", None) is not None \
                and self.w_contours.value:
            from ion_gym.edit.contours import contour_primitives
            cont = contour_primitives(preview)
            if "refused" not in cont:
                self._say(f"contours: {len(cont['polylines'])} "
                          f"line(s), field build "
                          f"{cont['build_s']} s")
        v = viewer_for_session(
            preview,
            canvas_w=EDITOR_CANVAS_W, canvas_h=EDITOR_CANVAS_H,
            selected=sel, selected_shape=sel_sh,
            gizmo_enabled=bool(getattr(self, "w_gizmo", None)
                               and self.w_gizmo.value),
            contours=cont,
            doc_id=(str(self.path) if self.path is not None
                    else f"upload:{self.session.name}"))
        # each rebuild is a NEW component: re-wire the JS -> Python
        # channels or picks and drags from the fresh canvas vanish
        v.param.watch(self._on_canvas_pick, "picked")
        v.param.watch(self._on_gizmo_drag, "drag")
        self.viewer_holder[:] = [v]

    def has_unsaved_edits(self) -> bool:
        """True when queued ops diverge the document from its
        loaded bytes; the demo's document-switch guard asks here."""
        return self.session.edit_count > 0

    def _rebuild_pickers(self, keep=None) -> None:
        els = self.session.committed_electrodes()
        opts = {}
        for i, el in enumerate(els):
            ok, _ = self.session.electrode_editable(i)
            label = f"{i}: {el.get('name')}"
            opts[label + ("" if ok else "  [STL]")] = i
        self.w_el.options = opts
        if keep is not None and keep in opts.values():
            self.w_el.value = keep
        self._on_pick_el(None)

    # -- selection --------------------------------------------------------
    def _on_pick_el(self, _event) -> None:
        el_i = self.w_el.value
        els_now = self.session.committed_electrodes()
        if el_i is not None and not (0 <= el_i < len(els_now)):
            # selection points past the committed list (e.g. the undo of
            # an add_electrode removed it): fall back to the last valid
            # electrode rather than indexing blindly (gate-found crash)
            fallback = len(els_now) - 1 if els_now else None
            self.w_el.value = fallback     # re-enters this watcher
            return
        if el_i is not None:
            from ion_gym.edit.session import normalize_color
            raw = self.session.committed_electrodes()[el_i]
            c = normalize_color(raw.get("color", [200, 60, 60]))
            self.w_color.value = "#%02x%02x%02x" % tuple(c)
            # by design: geometry is only addable
            # THROUGH an electrode — the button names its target
            self.w_add_shape.name = (
                f"Add shape to {raw.get('name')}")
            self.w_add_shape.disabled = False
        else:
            self.w_add_shape.name = "Add shape (select an "\
                                    "electrode)"
            self.w_add_shape.disabled = True
        # live highlight: sync the selection to
        # the mounted viewer; the JS restyles without a rebuild
        if len(self.viewer_holder):
            v = self.viewer_holder[0]
            if hasattr(v, "selected"):
                v.selected = el_i if el_i is not None else -1
                sh = self.w_sh.value
                v.selected_shape = sh if sh is not None else -1
        if el_i is None:
            self.w_sh.options = {}
            self.fields[:] = []
            return
        shapes = self._shapes_of(el_i)
        self.w_sh.options = {f"{j}: {s.get('type')}": j
                             for j, s in enumerate(shapes)}
        if shapes:
            self.w_sh.value = 0
        self._on_pick_sh(None)

    def _on_pick_sh(self, _event) -> None:
        self._field_widgets = {}
        self.fields[:] = []
        el_i, sh_j = self.w_el.value, self.w_sh.value
        if el_i is None or sh_j is None:
            return
        ok, reason = self.session.electrode_editable(el_i)
        if not ok:
            self.fields[:] = [pn.pane.Markdown(f"**not editable:** "
                                               f"{reason}")]
            return
        shapes = self._shapes_of(el_i)
        if not (0 <= sh_j < len(shapes)):
            return
        shape = shapes[sh_j]
        t = shape.get("type")
        widgets: List = []
        if t in _FIELDS:
            for k in _FIELDS[t]:
                w = pn.widgets.FloatInput(
                    name=k, value=float(shape.get(k, 0.0)),
                    step=float(self.session.spec.geometry.mm_per_gu),
                    width=140)
                self._field_widgets[k] = w
                widgets.append(w)
        elif t in ("polygon", "cutout"):
            key = "points_mm" if t == "polygon" else "children"
            w = pn.widgets.TextAreaInput(
                name=f"{t} {key} (JSON)",
                value=json.dumps(shape.get(key, []), indent=1),
                height=140, width=420)
            self._field_widgets[key] = w
            widgets.append(w)
        else:
            self.fields[:] = [pn.pane.Markdown(
                f"**shape type {t!r} has no editing form** (policy "
                f"vocabulary: "
                f"{self.session.policy.editable_shape_types})")]
            return
        if self.session.policy.supports_extrude:
            ex = shape.get("extrude") or {}
            w_on = pn.widgets.Checkbox(name="extrude (axis z)",
                                       value=bool(ex))
            w_lo = pn.widgets.FloatInput(
                name="extrude lo_mm", value=float(ex.get("lo_mm", 0.0)),
                width=140)
            w_hi = pn.widgets.FloatInput(
                name="extrude hi_mm", value=float(ex.get("hi_mm", 0.0)),
                width=140)
            self._field_widgets["_ex_on"] = w_on
            self._field_widgets["_ex_lo"] = w_lo
            self._field_widgets["_ex_hi"] = w_hi
            widgets += [w_on, w_lo, w_hi]
        self.fields[:] = widgets

    # -- build the shape dict from the widgets ---------------------------
    def _shape_from_fields(self, base: dict) -> dict:
        shape = json.loads(json.dumps(base))
        snap = self.w_snap.value
        for k, w in self._field_widgets.items():
            if k.startswith("_ex_"):
                continue
            if k in ("points_mm", "children"):
                try:
                    shape[k] = json.loads(w.value)
                except json.JSONDecodeError as e:
                    raise EditRefusal(f"{k}: not valid JSON: {e}") from e
            else:
                v = float(w.value)
                if snap and k not in _NO_SNAP:
                    v = self.session.snap_to_pitch(v)
                    w.value = v          # displayed == committed
                if (k in _OPTIONAL_DEFAULTS and k not in base
                        and v == _OPTIONAL_DEFAULTS[k]):
                    continue             # keep the key absent
                shape[k] = v
        if self.session.policy.supports_extrude \
                and "_ex_on" in self._field_widgets:
            if self._field_widgets["_ex_on"].value:
                lo = float(self._field_widgets["_ex_lo"].value)
                hi = float(self._field_widgets["_ex_hi"].value)
                if snap:
                    lo = self.session.snap_to_pitch(lo)
                    hi = self.session.snap_to_pitch(hi)
                    self._field_widgets["_ex_lo"].value = lo
                    self._field_widgets["_ex_hi"].value = hi
                if hi <= lo:
                    raise EditRefusal(
                        f"extrude hi_mm ({hi:g}) must exceed lo_mm "
                        f"({lo:g}) — a zero or negative slab is not a "
                        f"body")
                shape["extrude"] = {"axis": "z", "lo_mm": lo, "hi_mm": hi}
            else:
                shape.pop("extrude", None)
        return shape

    # -- actions ----------------------------------------------------------
    def _on_apply(self, _event) -> None:
        el_i, sh_j = self.w_el.value, self.w_sh.value
        if el_i is None or sh_j is None:
            self._say("**nothing selected**")
            return
        try:
            current = self._shapes_of(el_i)[sh_j]
            shape = self._shape_from_fields(current)
            if shape == current:
                self._say("no changes — nothing queued (no-op apply "
                          "leaves the document byte-identical)")
                return
            warns = self.session.shape_warnings(shape)
            self.session.set_shape(el_i, sh_j, shape)
            rep = self.session.verify_commit()
        except EditRefusal as e:
            self._refuse(e)
            return
        lines = [f"applied: {len(rep['diff_paths'])} path(s) changed, "
                 f"geometry key changed = {rep['geometry_key_changed']}"]
        lines += [f"&#9888; {w}" for w in warns]
        self._say("<br>".join(lines))
        self._refresh_banner()
        self._refresh_viewer()

    def _on_undo(self, _event) -> None:
        try:
            op = self.session.undo_last()
        except EditRefusal as e:
            self._refuse(e)
            return
        self._say(f"undid: {op['op']}")
        self._refresh_banner()
        self._refresh_viewer()
        self._rebuild_pickers(keep=self.w_el.value)

    def _on_redo(self, _event) -> None:
        try:
            op = self.session.redo_last()
        except EditRefusal as e:
            self._refuse(e)
            return
        self._say(f"redid: {op['op']}")
        self._refresh_banner()
        self._refresh_viewer()
        self._rebuild_pickers(keep=self.w_el.value)

    def _on_remove(self, _event) -> None:
        el_i, sh_j = self.w_el.value, self.w_sh.value
        if el_i is None or sh_j is None:
            self._say("**nothing selected**")
            return
        try:
            self.session.remove_shape(el_i, sh_j)
            rep = self.session.verify_commit()
        except EditRefusal as e:
            self._refuse(e)
            return
        self._say(f"shape removed ({len(rep['diff_paths'])} diff "
                  f"path(s))")
        self._refresh_banner()
        self._refresh_viewer()

    def _default_shape(self, t: str) -> dict:
        g = self.session.spec.geometry
        h = float(g.mm_per_gu)
        span = _NEW_SHAPE_SPAN_PITCHES * h
        cx = self.session.snap_to_pitch(float(g.width_mm) / 2)
        cy = self.session.snap_to_pitch(float(g.height_mm) / 2)
        if t == "rect":
            s = {"type": "rect", "x_mm": cx - span / 2,
                 "y_mm": cy - span / 2, "width_mm": span,
                 "height_mm": span}
        elif t == "ellipse":
            s = {"type": "ellipse", "cx_mm": cx, "cy_mm": cy,
                 "rx_mm": span / 2, "ry_mm": span / 2}
        elif t == "polygon":
            s = {"type": "polygon", "points_mm":
                 [[cx, cy], [cx + span, cy], [cx, cy + span]]}
        else:
            raise EditRefusal(f"no default for shape type {t!r}")
        if self.session.policy.supports_extrude:
            ax = self.session.spec.geometry.axial_extent_mm
            lo, hi = (ax if ax else (0.0, span))
            s["extrude"] = {"axis": "z", "lo_mm": float(lo),
                            "hi_mm": float(hi)}
        return s

    def _on_add_shape(self, _event) -> None:
        el_i = self.w_el.value
        if el_i is None:
            self._say("**no electrode selected**")
            return
        try:
            shape = self._default_shape(self.w_new_type.value)
            warns = self.session.shape_warnings(shape)
            self.session.add_shape(el_i, shape)
            self.session.verify_commit()
        except EditRefusal as e:
            self._refuse(e)
            return
        self._refresh_banner()
        self._rebuild_pickers(keep=el_i)
        shapes = self._shapes_of(el_i)
        # select the shape just added so its fields appear and
        # Apply edits IT (now real: pickers read committed state)
        if shapes:
            self.w_sh.value = len(shapes) - 1
            self._on_pick_sh(None)
        self._refresh_viewer()
        name = self.session.committed_electrodes()[el_i] \
            .get("name")
        self._say("<br>".join(
            [f"shape added to electrode {name!r} and selected — "
             f"edit its numbers, then Apply"]
            + [f"&#9888; {w}" for w in warns]))

    def _on_add_electrode(self, _event) -> None:
        try:
            shape = self._default_shape(self.w_new_type.value)
            self.session.add_electrode(self.w_new_el.value, shape)
            self.session.verify_commit()
        except EditRefusal as e:
            self._refuse(e)
            return
        self._refresh_banner()
        self._rebuild_pickers()
        new_i = len(self.session.committed_electrodes()) - 1
        self.w_el.value = new_i          # select what was created
        self._refresh_viewer()
        self._say(f"electrode {self.w_new_el.value!r} added with "
                  f"one {self.w_new_type.value} and selected — "
                  f"edit it, then Save when ready")

    def _on_gizmo_toggle(self, _event) -> None:
        if len(self.viewer_holder):
            v = self.viewer_holder[0]
            if hasattr(v, "gizmo_enabled"):
                v.gizmo_enabled = bool(self.w_gizmo.value)

    def _on_canvas_pick(self, event) -> None:
        """A click in any viewport selects the hit shape's electrode and
        shape; the pickers follow, which drives highlight and gizmo."""
        d = event.new or {}
        if self._suppress_pick or "el" not in d:
            return
        el_i, sh_i = int(d["el"]), int(d.get("sh", -1))
        self._suppress_pick = True
        try:
            if el_i in (self.w_el.options.values()
                        if isinstance(self.w_el.options, dict)
                        else self.w_el.options):
                self.w_el.value = el_i
            if sh_i >= 0 and sh_i in (
                    self.w_sh.options.values()
                    if isinstance(self.w_sh.options, dict)
                    else self.w_sh.options):
                self.w_sh.value = sh_i
                self._on_pick_sh(None)
        finally:
            self._suppress_pick = False

    def _on_gizmo_drag(self, event) -> None:
        """The gizmo PROPOSED a delta; the commit door DECIDES.  The
        moved shape goes through the same set_shape as typed values —
        snap policy, warnings, clamps, undo/redo, banner all identical.
        A refusal reverts the visual by rebuilding the viewer from the
        committed bytes."""
        d = event.new or {}
        if "el" not in d:
            return
        el_i, sh_i = int(d["el"]), int(d["sh"])
        dx, dy = float(d.get("dx", 0.0)), float(d.get("dy", 0.0))
        dz = float(d.get("dz", 0.0))
        try:
            current = self._shapes_of(el_i)[sh_i]
            moved = shift_shape(current, dx=dx, dy=dy, dz=dz)
            if self.w_snap.value:
                moved = self._snap_shape(moved)
            if moved == current:
                self._refresh_viewer()   # snap collapsed the drag
                self._say("drag snapped back to the original position")
                return
            warns = self.session.shape_warnings(moved)
            self.session.set_shape(el_i, sh_i, moved)
            rep = self.session.verify_commit()
        except (EditRefusal, IndexError) as e:
            self._refuse(e)
            self._refresh_viewer()       # revert the dangling visual
            return
        lines = [f"gizmo move applied (dx={dx:g}, dy={dy:g}"
                 + (f", dz={dz:g}" if dz else "")
                 + f"): geometry key changed = "
                   f"{rep['geometry_key_changed']}"]
        lines += [f"&#9888; {w}" for w in warns]
        self._say("<br>".join(lines))
        self._refresh_banner()
        self._refresh_viewer()
        self._on_pick_sh(None)           # numeric fields show the move

    def _snap_shape(self, shape: dict) -> dict:
        """Snap a shape's positional numerics to the pitch lattice —
        the same field-decides policy the numeric form applies."""
        import copy as _copy
        out = _copy.deepcopy(shape)
        t = out.get("type")
        keys = {"rect": ("x_mm", "y_mm"),
                "ellipse": ("cx_mm", "cy_mm")}.get(t, ())
        for k in keys:
            out[k] = self.session.snap_to_pitch(out[k])
        if t == "polygon":
            out["points_mm"] = [
                [self.session.snap_to_pitch(a),
                 self.session.snap_to_pitch(b)]
                for a, b in out["points_mm"]]
        if t == "cutout":
            out["children"] = [self._snap_shape(k)
                               for k in (out.get("children") or [])]
        ex = out.get("extrude")
        if isinstance(ex, dict):
            ex["lo_mm"] = self.session.snap_to_pitch(ex["lo_mm"])
            ex["hi_mm"] = self.session.snap_to_pitch(ex["hi_mm"])
        return out

    def _apply_color(self, everywhere: bool) -> None:
        try:
            if everywhere:
                self.session.set_color_all(self.w_color.value)
            else:
                el_i = self.w_el.value
                if el_i is None:
                    self._say("**no electrode selected**")
                    return
                self.session.set_color(el_i, self.w_color.value)
            rep = self.session.verify_commit()
        except EditRefusal as e:
            self._refuse(e)
            return
        scope = "all electrodes" if everywhere \
            else f"electrode {self.w_el.value}"
        self._say(f"color {self.w_color.value} applied to {scope} "
                  f"(geometry key changed = "
                  f"{rep['geometry_key_changed']})")
        self._refresh_banner()
        self._refresh_viewer()

    def _on_color_one(self, _event) -> None:
        self._apply_color(everywhere=False)

    def _on_color_all(self, _event) -> None:
        self._apply_color(everywhere=True)

    def _on_save(self, _event) -> None:
        if not self.w_save_path.value.strip():
            self._refuse(EditRefusal(
                "no save path — this document was loaded from a browser "
                "upload (bytes only, the dialog gives no disk path); "
                "type a destination in the save-path box"))
            return
        target = Path(self.w_save_path.value)
        try:
            self.session.verify_commit()
            data = self.session.to_document_bytes()
            target.write_bytes(data)
            reloaded = EditSession.load(target)
        except (EditRefusal, OSError) as e:
            self._refuse(e)
            return
        self.session = reloaded
        self.path = target
        self.loaded_key = self.session.geometry_key_short()
        self._say(f"saved {len(data)} bytes to {target} and reloaded — "
                  f"geometry key now **{self.loaded_key}**")
        self._refresh_banner()
        self._refresh_viewer()
        self._rebuild_pickers()

    # -- layout -----------------------------------------------------------
    def panel(self) -> pn.Column:
        left = pn.Column(
            self.w_el, self.w_sh, self.w_snap, self.w_contours,
            self.w_gizmo,
            self.fields,
            pn.Row(self.w_apply, self.w_undo, self.w_redo,
                   self.w_remove),
            pn.pane.Markdown("---"),
            pn.Row(self.w_color, self.w_color_one,
                   self.w_color_all),
            pn.Row(self.w_new_type, self.w_add_shape),
            pn.Row(self.w_new_el, self.w_add_el),
            pn.pane.Markdown("---"),
            self.w_save_path, self.w_save,
            width=470)
        # banner at the BOTTOM;
        # still always on screen and still the loud stale channel
        return pn.Column(
            pn.Row(left, self.viewer_holder,
                   sizing_mode="stretch_width"),
            self.status, self.banner,
            sizing_mode="stretch_width")


def edit_document(path):
    """UI-path entry: the editor, or the named refusal as a pane."""
    try:
        return EditorApp(path).panel()
    except EditRefusal as e:
        return pn.pane.Markdown(
            f"### Editor refuses this document\n**{e}**", width=700)
