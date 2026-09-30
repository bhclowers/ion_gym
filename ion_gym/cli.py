"""
ion_gym command line.

Two equivalent spellings after `pip install -e .`:

    ion-gym dashboard                  # console script
    python -m ion_gym dashboard        # module form

Options:
    ion-gym dashboard --port 5007      # serve on another port
    ion-gym dashboard --no-show        # don't open a browser
    ion-gym dashboard path/to/spec.json    # open on a specific spec
    ion-gym dashboard --mem-holders 8000   # name array holders past 8 GB
    ion-gym dashboard --mem-holders        # holders on every heartbeat

This replaces the README's old `python -m panel serve <path-to-
sim_app.py>` incantation with the app's own sanctioned entry point.
"""
import argparse
import sys


def _dashboard(args):
    from ion_gym.ui.sim_app import SimApp

    if args.mem_holders is not None:
        # Arm the memory-holder instrumentation BEFORE the app exists so
        # the very first heartbeat obeys it. Bare `--mem-holders` (or any
        # value <= 0) means "always": full VERBOSE telemetry, holders on
        # every reading. A positive value is a threshold in MB: the
        # heartbeat stays one quiet line until the process footprint
        # crosses it, then the crossing reading carries the holder
        # breakdown (see telemetry.HOLDERS_ABOVE_MB).
        from ion_gym.ui import telemetry
        if args.mem_holders <= 0:
            telemetry.VERBOSE[0] = True
        else:
            telemetry.HOLDERS_ABOVE_MB[0] = float(args.mem_holders)

    if args.mem_referrers is not None:
        # The referrer walk only runs on readings that gather holders
        # (telemetry.snapshot gates it on the same condition), so this
        # flag is meaningless alone — refuse with the pairing named
        # rather than silently never printing (no hidden branches).
        if args.mem_holders is None:
            raise SystemExit(
                "--mem-referrers requires --mem-holders: referrers are "
                "walked only on readings that gather holders (add "
                "--mem-holders for every heartbeat, or --mem-holders "
                "<MB> for the crossing reading)")
        from ion_gym.ui import telemetry
        telemetry.REFERRER_TARGET[0] = args.mem_referrers

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
    d.add_argument("--mem-holders", type=float, nargs="?", const=0.0,
                   default=None, metavar="MB",
                   help="print who is holding array memory: bare flag = on "
                        "every heartbeat (verbose telemetry); with a value "
                        "= only once process footprint exceeds MB "
                        "(e.g. --mem-holders 8000)")
    d.add_argument("--mem-referrers", metavar="CLASS", default=None,
                   help="with --mem-holders: also walk WHO still points at "
                        "live instances of this class (e.g. Stl3DModel) or "
                        "at dicts carrying a key (dict:res); printed under "
                        "the holder line")
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
