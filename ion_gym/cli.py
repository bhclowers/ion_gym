"""
ion_gym command line.

Two equivalent spellings after `pip install -e .`:

    ion-gym dashboard                  # console script
    python -m ion_gym dashboard        # module form

Options:
    ion-gym dashboard --port 5007      # serve on another port
    ion-gym dashboard --no-show        # don't open a browser
    ion-gym dashboard path/to/spec.json    # open on a specific spec

This replaces the README's old `python -m panel serve <path-to-
sim_app.py>` incantation with the app's own sanctioned entry point.
"""
import argparse
import sys


def _dashboard(args):
    from ion_gym.ui.sim_app import SimApp

    spec = None
    if args.spec:
        from ion_gym.io.sim_spec import SimSpec
        spec = SimSpec.from_json(args.spec)     # refuses missing paths by name
    app = SimApp(spec)
    # /editor and /flight ride on the SAME server: the dashboard buttons
    # open them in new browser tabs, and each visit builds a fresh
    # per-session page. The route table is ui.serve's, not a second copy
    # here -- a notebook launching the app any other way used to get a
    # server with only "/", so those buttons answered 404.
    from ion_gym.ui.serve import serve_dashboard
    serve_dashboard(app=app, port=args.port, show=not args.no_show,
                    title="ion_gym",
                    websocket_max_message_size=200 * 1024 * 1024)
    return 0


def _edit(args):
    from ion_gym.edit.serve import serve_editor
    serve_editor(port=args.port, show=not args.no_show)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ion-gym",
        description="ion_gym — ion-optics simulation toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dashboard", aliases=["app"],
                       help="launch the interactive Panel dashboard")
    d.add_argument("spec", nargs="?", default=None,
                   help="optional spec JSON to open with")
    d.add_argument("--port", type=int, default=5006)
    d.add_argument("--no-show", action="store_true",
                   help="don't open a browser tab")
    d.set_defaults(fn=_dashboard)

    e = sub.add_parser("edit",
                       help="launch the geometry editor alone")
    e.add_argument("--port", type=int, default=5007)
    e.add_argument("--no-show", action="store_true",
                   help="don't open a browser tab")
    e.set_defaults(fn=_edit)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
