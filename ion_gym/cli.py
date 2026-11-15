"""
ion_gym command line.

Two equivalent spellings after `pip install -e .`:

    ion-gym dashboard                  # console script
    python -m ion_gym dashboard        # module form

Options:
    ion-gym --version                  # the installed ion_gym version
    ion-gym validate path/to/spec.json # check a deck, no GUI (exit 0/1/2)
    ion-gym dashboard --port 5007      # serve on another port
    ion-gym dashboard --no-show        # don't open a browser
    ion-gym dashboard path/to/spec.json    # open on a specific spec
    ion-gym dashboard --mem-holders-above 8000   # name array holders past 8 GB
    ion-gym dashboard --mem-holders              # holders on every heartbeat

A mistake on the command line (a deck that cannot be opened, a port in
use) is reported in one line on stderr with exit status 2, the status
argparse uses for usage errors; a traceback and status 1 mean a fault in
ion_gym itself.

This replaces the README's old `python -m panel serve <path-to-
sim_app.py>` incantation with the app's own sanctioned entry point.
"""
import argparse
import errno
import os
import signal
import sys

# exit status of a command-line mistake (argparse's own usage-error status)
USAGE_ERROR = 2


class UsageError(Exception):
    """A command-line mistake: reported in one line, exit USAGE_ERROR."""


def _load_deck(path):
    """The deck at `path`, or a UsageError naming why it cannot be opened.
    Missing file (OSError), malformed JSON or a refused deck (ValueError,
    which MissingPhysicsKey is), a missing required block (KeyError)."""
    from ion_gym.io.sim_spec import SimSpec
    try:
        return SimSpec.from_json(path)
    except (OSError, ValueError, KeyError) as e:
        hint = ""
        if _is_number(path):
            hint = (" (a number here is read as the deck path; for a "
                    "memory threshold use --mem-holders-above MB)")
        raise UsageError(f"cannot open deck {path!r}: "
                         f"{type(e).__name__}: {e}{hint}") from e


def _is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def _quit_this_process():
    """The dashboard's Quit, confirmed: end `ion-gym dashboard` exactly as
    Ctrl-C in its terminal does. Panel's SIGINT handler stops the server
    loop, the blocking serve returns, and so does this command; every
    thread the app starts is a daemon, so nothing keeps the process
    alive after that."""
    os.kill(os.getpid(), signal.SIGINT)


def _serve(start, port):
    """Run a server start; a port already in use is a UsageError."""
    try:
        return start()
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        raise UsageError(f"port {port} is already in use (another ion-gym "
                         f"or server is on it); pass --port with a free "
                         f"port") from e


def _dashboard(args):
    if args.mem_referrers is not None and not (args.mem_holders
                                               or args.mem_holders_above):
        # The referrer walk only runs on readings that gather holders
        # (telemetry.snapshot gates it on the same condition), so this
        # flag is meaningless alone — refuse with the pairing named
        # rather than silently never printing (no hidden branches).
        raise UsageError(
            "--mem-referrers requires --mem-holders or "
            "--mem-holders-above: referrers are walked only on readings "
            "that gather holders")
    if args.mem_holders and args.mem_holders_above is not None:
        raise UsageError("--mem-holders (every heartbeat) and "
                         "--mem-holders-above MB (past a threshold) are "
                         "alternatives; pass one")
    if args.mem_holders_above is not None and args.mem_holders_above <= 0:
        raise UsageError(f"--mem-holders-above {args.mem_holders_above:g}: "
                         f"the threshold must be > 0 MB (for every "
                         f"heartbeat pass --mem-holders)")
    # the deck first: a mistake there is reported before anything is built
    spec = _load_deck(args.spec) if args.spec else None

    # Arm the memory-holder instrumentation BEFORE the app exists so the
    # very first heartbeat obeys it. --mem-holders: full VERBOSE
    # telemetry, holders on every reading. --mem-holders-above MB: the
    # heartbeat stays one quiet line until the process footprint crosses
    # MB, then the crossing reading carries the holder breakdown (see
    # telemetry.HOLDERS_ABOVE_MB).
    if args.mem_holders or args.mem_holders_above is not None:
        from ion_gym.ui import telemetry
        if args.mem_holders:
            telemetry.VERBOSE[0] = True
        else:
            telemetry.HOLDERS_ABOVE_MB[0] = float(args.mem_holders_above)
        if args.mem_referrers is not None:
            telemetry.REFERRER_TARGET[0] = args.mem_referrers

    from ion_gym.ui.sim_app import SimApp
    app = SimApp(spec)
    # /editor and /flight ride on the SAME server: the dashboard buttons
    # open them in new browser tabs, and each visit builds a fresh
    # per-session page. The route table is ui.serve's, not a second copy
    # here -- a notebook launching the app any other way used to get a
    # server with only "/", so those buttons answered 404.
    from ion_gym.ui.serve import serve_dashboard
    _serve(lambda: serve_dashboard(
        app=app, port=args.port, show=not args.no_show, title="ion_gym",
        on_quit=_quit_this_process,
        websocket_max_message_size=200 * 1024 * 1024), args.port)
    print("ion-gym: dashboard stopped")
    return 0


def _validate(args):
    """Check a deck without the GUI: the refusals build_run would make
    before building, and the advisories. Exit 0 valid, 1 refused, 2 a
    deck that cannot be opened."""
    spec = _load_deck(args.spec)
    from ion_gym.physics.sim_build import preflight
    rec_err, errs = preflight(spec)
    refusals = ([rec_err] if rec_err else []) + errs
    for note in spec.advisories():
        print(f"advisory: {note}")
    for r in refusals:
        print(f"REFUSED: {r}")
    if refusals:
        print(f"{args.spec}: INVALID ({len(refusals)} refusal(s))")
        return 1
    print(f"{args.spec}: VALID (checks made before a build; a birth in "
          f"metal or an empty electrode is found only by building)")
    return 0


def _edit(args):
    from ion_gym.ui.serve import EDITOR_ENABLED, EDITOR_OFF_NOTE
    if not EDITOR_ENABLED:
        raise UsageError(f"{EDITOR_OFF_NOTE} Edit the deck's JSON, or its "
                         f"STL files, and open it with `ion-gym dashboard`.")
    from ion_gym.edit.serve import serve_editor
    _serve(lambda: serve_editor(port=args.port, show=not args.no_show),
           args.port)
    return 0


def main(argv=None):
    # Silence OpenMP's informational "omp_set_nested deprecated" notice
    # (numba/MKL threading; not an error, no effect on results). Set HERE,
    # at the process entry point and before anything imports numba --
    # sim_app used to set it at IMPORT time, so importing the package from
    # a notebook changed that kernel's environment.
    import os
    os.environ.setdefault("KMP_WARNINGS", "0")
    import ion_gym
    ap = argparse.ArgumentParser(
        prog="ion-gym",
        description="ion_gym — ion-optics simulation toolkit")
    ap.add_argument("--version", action="version",
                    version=f"ion-gym {ion_gym.__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dashboard", aliases=["app"],
                       help="launch the interactive Panel dashboard")
    d.add_argument("spec", nargs="?", default=None,
                   help="optional spec JSON to open with")
    d.add_argument("--port", type=int, default=5006)
    d.add_argument("--no-show", action="store_true",
                   help="don't open a browser tab")
    # Two flags, not one with an optional value: `--mem-holders deck.json`
    # read the deck path as the value and failed on it as a number.
    d.add_argument("--mem-holders", action="store_true",
                   help="print who is holding array memory on every "
                        "heartbeat (verbose telemetry)")
    d.add_argument("--mem-holders-above", type=float, default=None,
                   metavar="MB",
                   help="print who is holding array memory once the process "
                        "footprint exceeds MB (e.g. --mem-holders-above "
                        "8000)")
    d.add_argument("--mem-referrers", metavar="CLASS", default=None,
                   help="with --mem-holders or --mem-holders-above: also "
                        "walk WHO still points at live instances of this "
                        "class (e.g. Stl3DModel) or at dicts carrying a key "
                        "(dict:res); printed under the holder line")
    d.set_defaults(fn=_dashboard)

    v = sub.add_parser("validate",
                       help="check a deck without the GUI (exit 0 valid, "
                            "1 refused, 2 unreadable)")
    v.add_argument("spec", help="the deck JSON to check")
    v.set_defaults(fn=_validate)

    e = sub.add_parser("edit",
                       help="launch the geometry editor alone "
                            "(temporarily disabled)")
    e.add_argument("--port", type=int, default=5007)
    e.add_argument("--no-show", action="store_true",
                   help="don't open a browser tab")
    e.set_defaults(fn=_edit)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except UsageError as e:
        print(f"ion-gym: error: {e}", file=sys.stderr)
        return USAGE_ERROR


if __name__ == "__main__":
    sys.exit(main())
