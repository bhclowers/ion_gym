"""THE dashboard route table, in one place.

The dashboard is three pages on ONE server: `/` (the app), `/editor`
(the geometry editor) and `/flight` (the 3-D flight viewer), plus the
`/export/<token>` file downloads (ui.exports). The app's
own buttons link to `/editor` and `/flight` as relative paths, so they
work only if all three were registered on the server that served `/`.

This module exists because that table used to live inside `cli.py` and
nowhere else. `ion-gym dashboard` registered all three; a notebook doing
`app.panel().show()` started a server with only `/`, and the two buttons
answered **404** — the app promising a page the launch method never
created. One table, both launch paths, no way for them to disagree.

Imports of `ion_gym.edit` are deferred to call time: the editor pulls in
the browser layer, and importing this module must not.
"""
from __future__ import annotations

# THE GEOMETRY EDITOR IS SWITCHED OFF while it is reworked. With this
# False the dashboard shows its link greyed out, `/editor` serves a
# notice instead of the editor, and `ion-gym edit` refuses. The editor
# code and its tests are untouched; set True to restore all three.
EDITOR_ENABLED = False
EDITOR_OFF_NOTE = ("The geometry editor is temporarily disabled while it "
                   "is being reworked.")


def editor_off_page():
    """What `/editor` serves while the editor is switched off."""
    import panel as pn
    return pn.Column(pn.pane.Markdown(
        f"## Geometry editor\n\n{EDITOR_OFF_NOTE} Edit the deck's JSON, or "
        f"its STL files, and load it in the dashboard."), margin=20)


def dashboard_routes(app=None, spec=None):
    """{path: page} for the full dashboard — pass to `pn.serve`.

    app  : an existing SimApp. Built from `spec` when omitted.
    spec : starting deck for a freshly built app; None = the app's own
           default. Ignored when `app` is given, rather than silently
           overriding an app the caller already configured.

    `/editor` and `/flight` are per-session FACTORIES, not built pages:
    each browser session gets its own, so two tabs cannot end up sharing
    one widget tree.
    """
    from ion_gym.ui.sim_app import SimApp
    from ion_gym.edit.serve import editor_page
    from ion_gym.edit.flight_view import flight_page
    if app is None:
        app = SimApp(spec) if spec is not None else SimApp()
    # THE SERVER starts the server-scoped telemetry: the RSS heartbeat
    # and the loop-stall watchdog. SimApp does NOT start them in its
    # constructor -- see SimApp.start_telemetry -- so a gate, test or
    # notebook that merely builds an app spawns no threads at all (17
    # apps in one gate previously meant 17 of each, printing over one
    # another). Started here, and also armed lazily by the solve/flight
    # paths, so an app served some other way still gets its hang dump.
    # Both calls are idempotent.
    app.start_telemetry()
    return {"/": app.panel(),
            "/editor": editor_page if EDITOR_ENABLED else editor_off_page,
            "/flight": flight_page}


def serve_dashboard(app=None, spec=None, *, port=5006, show=True,
                    on_quit=None, **kw):
    """Serve the full dashboard — `/`, `/editor` and `/flight` together.

    Use this instead of `app.panel().show()` anywhere the editor and
    flight-viewer buttons should work: from a notebook, a script, or the
    `ion-gym dashboard` entry point, which all route through here.

    Extra keyword arguments are passed to `pn.serve` unchanged. Pass
    `threaded=True` from a notebook: `pn.serve` otherwise BLOCKS the
    calling cell, and every cell below it would sit unrun behind a
    server that only stops when the kernel does.

    `on_quit`: a no-argument callable that ends the process, given only
    by a launcher that owns it (`ion-gym dashboard`). It arms the app's
    Quit button; without it the button stays disabled, because quitting
    a notebook's server process would end the user's kernel.

    The server also carries `/export/<token>` (ui.exports): the Runs
    tab's prepared CSV, NPZ and HTML download from there as ordinary
    files, at any size, instead of through the page's connection.
    """
    import panel as pn
    from ion_gym.ui import exports
    pn.extension("plotly")
    kw["extra_patterns"] = list(kw.get("extra_patterns", [])) + exports.routes()
    exports.mark_served()
    if app is None:
        from ion_gym.ui.sim_app import SimApp
        app = SimApp(spec) if spec is not None else SimApp()
    if on_quit is not None:
        app.enable_quit(on_quit)
    return pn.serve(dashboard_routes(app=app), port=port, show=show, **kw)
