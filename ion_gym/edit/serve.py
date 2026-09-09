"""Editor page factory and server entry.

Integration: the editor lives in its OWN
browser tab, opened from a button in the dashboard, and browses the
repository's example decks.  Coupling to sim_app is file-mediated —
edit -> Save -> load in the dashboard, where the geometry-key machinery
already refuses stale solved fields — so no dashboard state is touched
and no widget can ever acquire two parents.

This module is the ONE home of the editor page (deck browser + unsaved-
edit switch guard); the editor demo entry point is a thin wrapper
that pins the acceptance decks and the browser checklist.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import panel as pn

import ion_gym


def repo_root() -> Path:
    """The working-tree root, derived from the package location (no
    hardcoded machine paths — standing rule)."""
    return Path(ion_gym.__file__).resolve().parents[1]


def available_decks() -> List[str]:
    """Repo-relative deck paths the browser offers NATIVELY: `examples/`
    ONLY (internal example folders are not
    exposed in the picker; internal decks load via the path/folder box
    like any other external document).  A missing directory is REPORTED
    via the page (installed-without-repo layouts), never silently empty;
    loading a deck the editor cannot edit refuses by name in-page, which
    is the designed behavior for e.g. the STL quads."""
    root = repo_root()
    out: List[str] = []
    p = root / "examples"
    if p.is_dir():
        out.extend(sorted(str(f.relative_to(root))
                          for f in p.glob("*.json")))
    return out


def editor_page(title: str = "ion_gym geometry editor",
                deck_options: Optional[Sequence[str]] = None,
                initial: Optional[str] = None) -> pn.Column:
    """Build one editor page: deck browser, free-path loader, unsaved-
    edit switch guard, and the EditorApp itself.  Layout order is part
    of the contract (the gate drives it): [title, Row(pick, path_in,
    load_btn, file_in, status), guard_row, holder] — `file_in` is the
    OS file dialog (bytes + filename, no disk path) and `status` is the
    info channel; both ride INSIDE the Row so the gate's positional
    indices (app[1][0] = pick, app[2] = guard_row, app[3] = holder)
    hold.  Called per browser session, so every tab gets its own
    EditorApp."""
    from ion_gym.edit.editor_panel import EditorApp
    from ion_gym.edit.policy import EditRefusal

    root = repo_root()
    options = list(deck_options) if deck_options else available_decks()
    missing_note = ""
    if not options:
        missing_note = (f"no examples/ folder found under {root} — "
                        f"use the path/folder box below")
    if initial is None:
        initial = options[0] if options else ""
    if initial and initial not in options:
        options = [initial] + options

    pick = pn.widgets.Select(name="document", options=options,
                             value=initial or None, width=460)
    path_in = pn.widgets.TextInput(
        name="load a deck by path, or a FOLDER to browse its *.json "
             "(repo-relative or absolute)",
        value="", width=460)
    load_btn = pn.widgets.Button(name="Load path / folder", width=130)
    state = {"app": None, "doc": initial, "reverting": False,
             "pending": None}
    holder = pn.Column(sizing_mode="stretch_width")

    def _resolve(rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else root / p

    def _mount(rel):
        # `rel` is a path string, or ('upload', name, data) from the
        # OS file dialog (FileInput — same picker as the main panel;
        # it delivers bytes + filename, never a disk path).
        try:
            if isinstance(rel, tuple) and rel[0] == "upload":
                app = EditorApp(None, upload=(rel[1], rel[2]))
            else:
                app = EditorApp(_resolve(rel))
        except (EditRefusal, OSError) as e:
            state["app"] = None
            holder[:] = [pn.pane.Markdown(
                f"### Editor refuses this document\n**{e}**", width=700)]
            return
        state["app"] = app
        holder[:] = [app.panel()]

    # UNSAVED-EDIT GUARD: switching documents with
    # queued ops reverts the picker and offers the choice BY NAME —
    # silently dropping edits is the failure mode this removes.
    # `status` is the separate INFO channel: folder
    # listings and path errors report there, so guard_row never shows
    # its Save/Discard buttons without a real pending switch — showing
    # them armed with pending=None was a latent _do_switch(None) crash.
    guard_msg = pn.pane.Markdown("")
    b_save = pn.widgets.Button(name="Save, then switch",
                               button_type="success", width=150)
    b_drop = pn.widgets.Button(name="Discard edits and switch",
                               button_type="danger", width=200)
    guard_row = pn.Row(guard_msg, b_save, b_drop, visible=False)
    status = pn.pane.Markdown("", width=460)

    def _do_switch(rel):
        if isinstance(rel, tuple) and rel[0] == "upload":
            state["doc"] = f"(upload) {rel[1]}"
        else:
            state["doc"] = rel
        state["pending"] = None
        guard_row.visible = False
        if state["doc"] in (pick.options or []):
            state["reverting"] = True
            pick.value = state["doc"]
            state["reverting"] = False
        _mount(rel)

    def _on_save_then_switch(_):
        app, rel = state["app"], state["pending"]
        if rel is None:
            guard_row.visible = False
            status.object = "**no switch pending** — nothing to save into"
            return
        if app is not None:
            app._on_save(None)
            if app.has_unsaved_edits():
                guard_msg.object = ("**save refused** (see the editor "
                                    "status line) — still on "
                                    f"`{state['doc']}`")
                return
        _do_switch(rel)

    def _on_drop_then_switch(_):
        rel = state["pending"]
        if rel is None:
            guard_row.visible = False
            status.object = "**no switch pending** — nothing to discard for"
            return
        _do_switch(rel)

    b_save.on_click(_on_save_then_switch)
    b_drop.on_click(_on_drop_then_switch)

    def _request_switch(target):
        app = state["app"]
        if app is not None and app.has_unsaved_edits():
            state["pending"] = target
            tgt_label = (f"(upload) {target[1]}"
                         if isinstance(target, tuple) else target)
            guard_msg.object = (
                f"**{app.session.edit_count} unsaved edit op(s)** on "
                f"`{state['doc']}` — save or discard before switching "
                f"to `{tgt_label}`:")
            guard_row.visible = True
            if state["doc"] in (pick.options or []):
                state["reverting"] = True
                pick.value = state["doc"]      # revert; guard decides
                state["reverting"] = False
            return
        _do_switch(target)

    def _on_pick(event):
        if state["reverting"] or event.new is None:
            return
        _request_switch(event.new)

    def _on_load_path(_):
        target = path_in.value.strip()
        if not target:
            status.object = "**type a path first**"
            return
        resolved = _resolve(target)
        # FOLDER semantics: a directory repopulates
        # the document picker with that folder's *.json — the localhost
        # equivalent of a folder dialog (a browser cannot hand the
        # server a native OS file path). Entries outside the repo are
        # listed ABSOLUTE so _resolve round-trips them; a repo-internal
        # folder inside the repo lists repo-relative.
        if resolved.is_dir():
            decks = sorted(resolved.glob("*.json"))
            if not decks:
                status.object = (f"**no *.json in** `{resolved}` — "
                                 f"picker unchanged")
                return
            def _label(p: Path) -> str:
                try:
                    return str(p.relative_to(root))
                except ValueError:
                    return str(p)
            opts = [_label(p) for p in decks]
            current = state["doc"]
            if current and current not in opts:
                opts = [current] + opts   # keep the open doc pickable
            state["reverting"] = True
            pick.options = opts
            pick.value = current if current in opts else None
            state["reverting"] = False
            status.object = (f"**{len(decks)} deck(s) from** "
                             f"`{resolved}` — pick one above")
            return
        if not resolved.is_file():
            status.object = f"**not a file or folder:** `{resolved}`"
            return
        status.object = ""
        _request_switch(target)

    pick.param.watch(_on_pick, "value")
    load_btn.on_click(_on_load_path)

    # OS FILE DIALOG (the same file picker as the
    # main panel widget" — same widget here). FileInput delivers bytes
    # + filename; the loaded document has NO disk path, so its
    # save-path box starts EMPTY and Save refuses by name until a
    # destination is typed (Save has always written to that box —
    # Save-As semantics, nothing invented).
    file_in = pn.widgets.FileInput(accept=".json", width=220)

    def _on_upload(event):
        if not event.new:
            return
        name = file_in.filename or "upload.json"
        status.object = f"**loaded from dialog:** `{name}` (no disk path)"
        _request_switch(("upload", name, bytes(event.new)))

    file_in.param.watch(_on_upload, "value")

    if initial:
        _mount(initial)
    elif missing_note:
        holder[:] = [pn.pane.Markdown(f"**{missing_note}**", width=700)]

    return pn.Column(
        pn.pane.Markdown(f"## {title}"),
        pn.Row(pick, path_in, load_btn, file_in, status),
        guard_row, holder,
        sizing_mode="stretch_width")


def serve_editor(port: int = 5007, show: bool = True):
    """Serve the editor alone (the `ion-gym edit` entry point)."""
    pn.extension()
    return pn.serve(editor_page, port=port, show=show,
                    title="ion_gym geometry editor")
