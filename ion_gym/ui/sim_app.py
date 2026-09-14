"""
ion_gym.sim_app
---------------
The GENERAL interactive ionbench: one Panel app that flies ANY SimSpec
(planar einzel, r-z einzel, IMS, funnel) by driving build_run(spec) —
not a hardcoded geometry. Supersedes the funnel-only FunnelApp.

Requirements met:
  * consumes build_run(spec) so einzel / IMS / funnel all load the same
    interface;
  * FREE-ASPECT zoom: box-zoom with variable size AND variable aspect
    (dragmode='zoom', scrollZoom, autorange on both axes, aspect lock
    OFF by default and user-toggleable);
  * Start / Stop at the TOP of the app;
  * controls organized into TABS (Source / Voltages / Gas / Integration
    / Display / Config);
  * trajectories PERSIST: a completed run is stored and can be reloaded
    into the view, re-colored, re-styled, and exported — not one-and-done.

Load a spec three ways: SimApp(spec), SimApp.from_json(path), or one of
the built-in examples via the Config tab.
"""

import io
import json
import os
import threading
import time

# Silence OpenMP's informational "omp_set_nested deprecated" notice (from
# numba/MKL threading). It's not an error and doesn't affect results; this
# just keeps the console clean. Set before numba/MKL initialize.
os.environ.setdefault("KMP_WARNINGS", "0")

import ion_gym
import numpy as np

try:
    import panel as pn
    import plotly.graph_objects as go
    _HAVE = True
except ImportError:
    _HAVE = False

from ion_gym.io.sim_spec import (SimSpec, OPTIONAL_CHANNELS,
                                set_pitch)
from ion_gym.io.spec_io import load_any_spec
from ion_gym.physics.sim_build import build_run, build_needs_solve, build_route
from ion_gym.physics.ensemble_driver import run_threaded
from ion_gym.viz.viz_core import VizError
from ion_gym.viz import viz_core as V

# bound for the headless (no-server) build wait — a pathological-stall
# guard, NOT a solve timeout; a real solve can take minutes, so this is
# generous. The server path never uses it (it schedules a poll).
_BUILD_JOIN_TIMEOUT_S = 900.0

# Fate tables come from THE single authority
# (physics.ion_envelope): this module's own copies stopped at
# 3, so station fates 5/6 had no name, no colour and no impact
# marker anywhere in the UI even though the kernels emit them.
from ion_gym.physics.ion_envelope import (  # noqa: E402
    FATE_COLOR as _FATE_COLOR, FATE_NAME as _FATE_NAME)
# Control-column geometry (a duty box was once cut off by
# the plot). A drive-group row is the widest thing the left column holds,
# so the column width is DERIVED from that row instead of being a literal
# -- change a widget width here and the column follows.
DRIVE_W = dict(wave=120, amp=120, freq=140, phase=130, duty=90)
PANEL_MARGIN_PX = 20          # Panel's default per-widget margins
VIEW_PX = 560                 # pane height when the view is not size-locked
CONTROL_COL_PX = (sum(DRIVE_W.values())
                  + PANEL_MARGIN_PX * len(DRIVE_W) + 30)
# Panel/Bokeh give a widget with no declared width this many pixels. It is
# the reason three bare Buttons in a Row need 960 px: named here so the
# control-column fitter can reason about undeclared widths instead of
# guessing (three unwidthed buttons is not a "wide row" anyone authored).
DEFAULT_WIDGET_PX = 300
# Separator between the control column and the plots. Named because they are
# design decisions, not magic: GUTTER_PX is the clear space on EACH side of
# the rule, so the controls and the figure's axis labels cannot sit flush
# against each other and read as one surface.
SEPARATOR_COLOR = "#d3d7dc"
GUTTER_PX = 12
# Field-contour count the Display tab starts at, and the value the
# one-click contours off/on button restores. ONE name
# for both so the button can never restore a different default than
# the slider ships with.
CONTOURS_DEFAULT = 16
# Status tab history depth. Since the status feedback lives in its own tab
# rather than the always-visible top bar, a message that
# lands while another tab is active would otherwise be overwritten unseen;
# the tab keeps a rolling session log of the latest messages. Named, not a
# magic number in the body.
STATUS_LOG_MAX = 100


def _mz_color_map(mz_values):
    """THE m/z -> colour convention for the app (PI 2026-09-13).

    Plotly's Dark24, imported from the installed package rather than
    copied as literals: 24 well-separated hues, so a many-mass deck gets
    discrete levels instead of an 8-colour cycle repeating.

    Callers pass the SPEC's declared mz_list, not the masses that
    happen to appear in a result set. Keying off what was detected
    would shift every colour when one mass records no hits, so the
    trajectories and the detector histogram would disagree about which
    colour means which mass.
    """
    from plotly.colors import qualitative as _q
    pal = list(_q.Dark24)
    return {m: pal[k % len(pal)] for k, m in enumerate(sorted(set(mz_values)))}

# One-time numba tracer compile on the FIRST fly of a session -- already
# stated in the flying status; the sizing readout states it
# too so the pre-run estimate does not over-promise.
NUMBA_JIT_FIRST_FLY_S = 39.0   # [measured on an einzel fly]


# ------------------------------------------------- control-column containment
def required_width(obj, default_widget_px: int = DEFAULT_WIDGET_PX) -> int:
    """Pixels this layout object needs to render without overflowing.

    A `pn.Row` lays its children out on ONE line at their natural widths and
    does not wrap, so its requirement is the SUM of theirs; every other
    container stacks, so its requirement is the MAX.  A leaf with no declared
    width takes Panel's default (`default_widget_px`) -- undeclared is not
    zero, and treating it as zero is how a row of three bare Buttons reads as
    "narrow" while needing 960 px.
    """
    kids = list(getattr(obj, "objects", None) or [])
    margin = getattr(obj, "margin", None)
    if margin is None:
        pad = PANEL_MARGIN_PX // 2
    elif isinstance(margin, (int, float)):
        pad = 2 * int(margin)
    elif len(margin) == 2:
        pad = 2 * int(margin[1])
    elif len(margin) == 4:
        pad = int(margin[1]) + int(margin[3])
    else:
        raise ValueError(f"required_width: unreadable margin {margin!r} on "
                         f"{type(obj).__name__}")
    if kids:
        sub = [required_width(k, default_widget_px) for k in kids]
        need = sum(sub) if isinstance(obj, pn.Row) else max(sub)
    else:
        w = getattr(obj, "width", None)
        need = int(w) if w else default_widget_px
    declared = getattr(obj, "width", None)
    if declared:
        need = max(need, int(declared))
    return need + pad


def fit_to_column(root, budget_px: int, *, label: str = "column"):
    """Make every row under `root` fit `budget_px`, in place.

    ROOT CAUSE: no side bar tab UI element may spread
    into the graph").  The left column is a fixed-width box in a `pn.Row`
    beside the plots.  A `pn.Row` inside it does not wrap and Panel does not
    clip, so a row whose children need more than the column's width simply
    paints across the boundary and lands on top of the figure -- which is
    exactly what the Voltages group-editor row (1020 px in a 730 px column)
    and the Config > Runs field-loader row did.

    The fix is a container swap, not width arithmetic: an over-budget
    `pn.Row` becomes a `pn.FlexBox`, which holds the same children in the
    same order but WRAPS them onto the next line when they do not fit.  That
    is why this is configuration-agnostic -- it makes no assumption about
    which widgets are present, so a tab added later (or one contributed by
    another module, as the geometry-import and STL-upload panels are) is contained on the
    same terms without anyone re-deriving a pixel budget.  Nothing is
    clipped or hidden: content that will not fit sideways goes downward, and
    the column already scrolls vertically as one surface.

    Returns the list of `(path, needed_px)` it rewrapped.  Every swap is
    a reported decision, not a silent reshuffle.
    """
    rewrapped: list[tuple[str, int]] = []

    def walk(node, path):
        kids = list(getattr(node, "objects", None) or [])
        for i, kid in enumerate(kids):
            here = f"{path}/{type(kid).__name__}[{i}]"
            need = required_width(kid, DEFAULT_WIDGET_PX)
            if isinstance(kid, pn.Row) and need > budget_px:
                node[i] = pn.FlexBox(*kid.objects,
                                     sizing_mode="stretch_width")
                rewrapped.append((here, need))
                walk(node[i], here)
                continue
            walk(kid, here)

    walk(root, label)
    for path, need in rewrapped:
        print(f"[sim_app] {label}: wrapped a {need}px row into {budget_px}px "
              f"at {path} (pn.Row -> pn.FlexBox)")
    over = required_width(root, DEFAULT_WIDGET_PX) - budget_px
    if over > 0:
        # A single leaf wider than the whole column cannot be wrapped. Say so
        # by name rather than letting it paint over the plot unannounced.
        print(f"[sim_app] WARNING: {label} still needs "
              f"{required_width(root, DEFAULT_WIDGET_PX)}px against a "
              f"{budget_px}px budget ({over}px over) — a single widget is "
              f"wider than the column; give it a width or split it.")
    return rewrapped


def _axial_axis(geom, ndim):
    """Which axis of a MASK array is the transport (axial) direction?
    None if the mask has no axial axis at all.

    Convention A (THE Z AXIS IS HORIZONTAL) says the axial coordinate is z --
    and, crucially, that **an r-z spec stores physical z in the `width_mm`/x
    slot**.  `x` was always a misnomer for `z` there.  So:

      * r-z          -> axis 0   (the width/x slot IS z)
      * 3-D x-y-z    -> axis 2   (z, as written)
      * 2-D x-y      -> NONE.  The cross-section has no axial axis; z is the
                        uniform direction and does not appear in the mask.
                        There is no gradient to measure, and measuring one
                        along y would be a number with no meaning.

    This is stated in ONE place because the alternative is what we found:
    `argwhere(m)[:, -1]` -- "the last axis is the axial one" -- which is true
    for the 3-D case, false for r-z, and meaningless for a planar cross-section,
    and which silently printed a V/cm measured ACROSS THE RADIUS.
    """
    if geom.coords == "rz":
        return 0
    if ndim >= 3:
        return 2
    return None


# Drive waveforms selectable in the GUI. Adding triangle/sawtooth later is
# a one-line kernel add (tracer3d._wave_eval) + one entry here — the
# composer and tracer already treat waveform as data.
WAVEFORM_OPTS = ["sin", "square", "table"]


def _example_specs():
    """Built-in examples: EVERY ONE IS A JSON FILE IN examples/.  Nothing else.

    RULE: ALL examples
    must be JSON-only with NO special wiring — no dedicated builder
    modules, no special cases in sim_build, no fixed-path caches.  NOW OR
    EVER.  A geometry that cannot be expressed as declarative JSON on a
    generic route is not an example yet; it goes to examples_quarantine/
    (not scanned here) until the generic capability exists.

    Every example imports a json.
    There should be no special builder case for them.  That's not agnostic."

    What this replaces: seven registrations, each hand-written, several wrapped
    in `except Exception: pass` (so a release that dropped a module shipped an
    app with a SILENTLY shorter menu), two of them importing example specs out
    of build_stl.py -- a CORE module shipping examples, the mirror image of
    examples shipping core.  The menu was a pile of special cases, and every
    special case was a place the app and the loader could disagree.

    Now: one loader, the SAME `SimSpec.from_json` a user gets when they load
    their own file.  If an example loads, the loader works; if the loader
    breaks, every example breaks LOUDLY and at once.  There is no path an
    example can take that a user's own JSON cannot.

    STL-backed examples keep their STLs in a per-example folder named by the
    stl_dir.  They used to bake an ABSOLUTE /tmp path into the spec -- the same
    disease as the removed funnel STL example, which named files that existed
    only on the authoring machine.  Relative to the example file, resolved at
    load: it works from any cwd, on any machine.

    Errors are LOUD.  An example that will not load is a bug, not an absence.
    """
    import json
    from pathlib import Path
    from ion_gym.io.spec_io import load_any_spec
    from ion_gym.io import paths

    d = paths.repo_root() / "examples"
    if not d.is_dir():
        raise RuntimeError(f"examples/ not found at {d} -- the app ships its "
                           f"examples as JSON and there are none")

    def _load(path):
        def build():
            j = json.loads(path.read_text())
            j.pop("_display_name", None)
            g = j.get("geometry", {})
            sd = g.get("stl_dir")
            if sd and not Path(sd).is_absolute():
                g["stl_dir"] = str((d / sd).resolve())   # relative -> real
            # load_any_spec, not SimSpec.from_json: the SAME door routes a
            # single-FA spec, a scene3d, OR a declarative multi-FA assembly
            # (kind:"assembly" -> flattened to one runnable spec). An
            # example takes no path a user's own JSON cannot.
            return load_any_spec(json.dumps(j))
        return build

    specs = {}
    for p in sorted(d.glob("*.json")):
        j = json.loads(p.read_text())
        # GATE-ONLY EXAMPLES: a spec can be a fixture a
        # gate depends on without being something a user should
        # meet in the menu (a hard-mirror reflectron is a negative
        # control). "_ui_hidden": true keeps the FILE -- gates read it by
        # path -- and drops only the menu entry. Still JSON-only: no
        # special wiring, no quarantine move, no gate breakage.
        if j.get("_ui_hidden") is True:
            continue
        # DISPLAY NAME: _display_name (curated menu label) -> name (the
        # deck's own title) -> filename stem. A deck missing the curated
        # label used to show as snake_case filename in the menu (browser
        # report) -- the deck's declared name is the obvious
        # second authority.
        name = j.get("_display_name") or j.get("name") or p.stem
        specs[name] = _load(p)
    if not specs:
        raise RuntimeError(f"no example JSON in {d}")
    return specs


def _mkw(cls, **kw):
    """Construct a Panel widget, dropping kwargs this Panel build does not
    support.

    Panel added `description` (hover tooltips) to widget classes at different
    versions and NOT uniformly -- Select has it in builds where Checkbox does
    not. Hard-coding the argument therefore works on the author's machine and
    raises TypeError on the user's, which is exactly what happened. Filtering
    against the class's own declared parameters makes the call self-checking:
    the widget renders on every build, and merely loses its tooltip on one
    that cannot show it.
    """
    try:
        allowed = set(cls.param.values())
    except (AttributeError, TypeError):
        # A param class that cannot enumerate its own parameters.  NARROW: if
        # anything ELSE goes wrong here it is a bug in ion_gym, not a widget
        # without introspection, and it must not silently DISABLE the filter
        # (which is what `except Exception` did -- unknown kwargs then reached
        # the widget and failed somewhere further away).
        allowed = None
    if allowed is not None:
        dropped = [k for k in kw if k not in allowed]
        for k in dropped:
            kw.pop(k)
    return cls(**kw)


# THE SUBJECT SENTINEL. The app's stage selector is
# its global SUBJECT: it names what is DRAWN, what is SOLVED and what the
# tabs EDIT. `<whole assembly>` is a legitimate value of that subject, so
# it lives in the selector's options beside the stage names rather than
# behind its own button.
#
# Declared here, once, because it is compared against in the draw path,
# the solve path and the edit path. Three inline copies of a magic string
# is three places for a typo to become a silent fall-through to
# single-stage behaviour -- which would look like the selector being
# ignored. The angle brackets keep it from ever colliding with a stage
# name a deck could legally declare.
WHOLE_ASSEMBLY = "<whole assembly>"
# LIVE flight redraw duty cycle (from hang dumps). The
# 200 ms tick stays the heartbeat cadence for the flight chip and
# status line; the heavy plotly/stats/publish redraw waits at least
# LIVE_REDRAW_MIN_S after the PREVIOUS one COMPLETED, stretched to
# LIVE_REDRAW_DUTY x its measured cost so a slow figure (field shading,
# big packets) can never occupy more than ~1/LIVE_REDRAW_DUTY of the
# loop. At duty 4 the loop is free >= 75% of the time between draws.
LIVE_REDRAW_MIN_S = 1.5
LIVE_REDRAW_DUTY = 4.0
# /flight publish cadence DURING a flight (/flight went
# stale mid-flight after publish left the per-tick path — the earlier
# per-tick publishing was accidentally load-bearing for viewer liveness).
# 296 ms per publish at 5 s cadence is ~6% duty; completion still
# publishes unconditionally.
LIVE_PUBLISH_MIN_S = 5.0
# LIVE display point budget (the live redraw can read as slow):
# SDS/diffusive paths carry 10-100x the points of ballistic
# ones, and the browser must re-render every displayed point each
# streamed update. The LIVE view thins each drawn path to keep total
# displayed points under this budget — display only, disclosed on the
# figure, never applied to stored/final draws or statistics.
LIVE_MAX_DRAW_PTS = 150_000


class SimApp:
    def __init__(self, spec: SimSpec = None):
        if not _HAVE:
            raise ImportError("pip install panel plotly")
        pn.extension("plotly", "tabulator")
        # Pay numba kernel compilation + plotly's import in the background
        # NOW, so the first Fly / Compute click doesn't stall 15-60 s on a
        # cold cache (6 of 9 events in a hang dump).
        from ion_gym.ui.prewarm import start_prewarm_thread
        start_prewarm_thread()
        if spec is None:
            # refuse-with-diagnostic: there is no second fallback to hide
            # behind.  If the default example cannot be built, say why.
            # Loaded as JSON DATA (the app's own examples-are-JSON doctrine),
            # not by importing an examples module: core never imports
            # examples/projects.
            from ion_gym.io import paths
            from ion_gym.io.sim_spec import SimSpec as _SimSpec
            _fj = paths.repo_root() / "examples" / "ion_funnel_rz.json"
            try:
                spec = _SimSpec.from_json(_fj.read_text())
            except Exception as e:
                raise ImportError(
                    "SimApp() was given no spec and the default example "
                    f"({_fj.name}) failed to load: {type(e).__name__}: {e}") from e
        self.spec = spec
        self._initial_spec_json = self.spec.to_json()   # for Reset
        self.tabs = None                                # persistent container
        self._model = None
        self._scene = None
        self._cols = None
        self._handle = None
        self._stop = False
        self._pcb = None
        self._solve_gen = 0      # solve generation; newer supersedes older
        self._runs = {}          # name -> final EnsembleProgress (reloadable)
        self._active = None      # currently displayed run name

        # TELEMETRY IS BUILT HERE AND STARTED BY THE SERVER, not in the
        # constructor. Both of these only MEAN anything when a browser
        # session exists: the heartbeat logs RSS against a running app,
        # and the watchdog reports a stalled EVENT LOOP. Started
        # unconditionally, every SimApp anyone constructs spawns two
        # daemon threads that nothing ever stops -- and the gates build
        # one app per deck, so a 17-deck run ended with ~17 of each,
        # printing over one another (interleaved [mem] lines with
        # out-of-order timestamps, and one "NO PULSE YET" note per
        # watchdog per 15 s, in a harness with no server to stall).
        #
        # So: constructed here, owned here, and started by
        # ui.serve.serve_dashboard -- the one place that knows a server
        # is coming. A notebook, a gate or a test that builds an app now
        # costs nothing and prints nothing. `close()` stops them again.
        from ion_gym.ui.telemetry import MemoryHeartbeat, LoopStallWatchdog
        self._heartbeat = MemoryHeartbeat(
            interval_s=30.0,
            context_fn=lambda: dict(runs=len(self._runs)))
        self._watchdog = LoopStallWatchdog(stall_s=15.0)
        self._watchdog_pcb = None

        self._build_controls()
        self._build_run_controls()
        # NOT silent.  This is what puts the ladder's DERIVED voltages on
        # screen; swallowing it means the numbers simply are not there and
        # nobody is told.  An absent number is a wrong figure.
        self._refresh_dc_derived()     # show the ladder's voltages at once
        # do NOT build the field at construction. Even a CACHE HIT on the
        # bases still runs compose_drive_channels (the field-aware gradient
        # over the whole grid — slow for a big 3-D example), so any auto-
        # build on load can freeze the app (a dense load borks
        # everything). Show the geometry preview; the field builds when the
        # user presses Recompute/Fly.
        self.w_dt.param.watch(lambda _e: self._update_dt_advice(),
                              "value")
        self._update_dt_advice()
        self._draw_geometry_only()

    def _ensure_watchdog_pulse(self):
        """Arm the 1 s loop-side pulse once a server session exists.

        Called from the same places that successfully arm other periodic
        callbacks (solve poll, flight tick) — the reliable signal that
        pn.state can schedule. Idempotent; outside a server session the
        watchdog simply never fires, which is correct (a script cannot
        drop a websocket)."""
        import panel as pn
        doc = pn.state.curdoc
        if doc is None:
            return
        # A SESSION EXISTS -> start the server-scoped telemetry. This is
        # the honest trigger: earlier (at construction, or at serve time
        # before a tab connects) the watchdog can only report "NO PULSE
        # YET" and the heartbeat can only log an idle process. Doing it
        # here also covers EVERY serving path, not just serve_dashboard,
        # so an app served some other way still gets its hang dump.
        # Idempotent.
        self.start_telemetry()
        # Re-arm when the SESSION changed:
        # a periodic callback dies with its session, but the stale handle
        # kept this method returning early, so a fresh tab never got a
        # pulse and the watchdog reported dead-session stalls forever.
        if self._watchdog_pcb is not None:
            if getattr(self, "_watchdog_doc", None) is doc:
                return                      # same live session: armed
            self._watchdog_pcb = None       # session died; re-arm below
        try:
            self._watchdog_pcb = pn.state.add_periodic_callback(
                self._watchdog.pulse, 1000)
            self._watchdog_doc = doc
            self._watchdog.pulse()          # fresh session = fresh pulse
        except RuntimeError:
            # cannot schedule here (no running session loop yet); a later
            # arming site will succeed. NOT silent-forever: every arming
            # site retries until one lands.
            pass

    # ------------------------------------------------------ from JSON
    @classmethod
    def from_json(cls, path):
        return cls(SimSpec.from_json(path))

    # ------------------------------------------------- controls (tabs)
    def _source_spec(self):
        """The spec whose SOURCE the Ion Source tab edits.

        PINNED SOURCE FA: in an assembly,
        *Set Fly Parameters* pins an FA, and the Ion Source tab binds to
        THAT stage's spec object in _assembly_specs — the same single
        authority the assembly flies from, so no copy exists to drift —
        while the LIVE spec remains the display/solve subject. This is
        what makes fly-parameters and view independent axes in fact: the
        session-5 mechanism made the source FA live, so *Set FA View*
        (which also installs the live spec) silently retargeted the
        source panel at whatever was displayed. Outside an assembly, or
        with no pin, this is self.spec — bit-identical behaviour.
        """
        pin = getattr(self, "_fly_src_pin", None)
        specs = getattr(self, "_assembly_specs", None) or {}
        if pin and pin in specs:
            return specs[pin]
        return self.spec

    def _build_controls(self):
        s = self.spec
        src, col, integ = self._source_spec().source, s.collisions, \
            s.integration
        # --- Source tab
        self.w_n = pn.widgets.IntInput(
            name="ions per m/z", start=1, value=src.n_ions, width=140,
            description="Ions flown PER m/z value. Total flown = this x the "
                        "number of m/z entries. E.g. 50 with 4 masses flies "
                        "200 ions (50 of each).")
        self.w_ion_total = pn.pane.Markdown("", width=280)
        # MASTER RANDOMNESS CONTROL (flights must be
        # random by default): every ion source starts RANDOM. This checkbox is
        # the ONE authority for seeded vs unseeded -- unchecked (the
        # default) draws fresh entropy per run and prints the drawn seed
        # so any run can be pinned; checked pins the run to the seed
        # beside it. It lives in the shared Ion Source header because it
        # governs births, birth times AND collision draws.
        self.w_seeded = pn.widgets.Checkbox(
            name="seeded run", value=(src.seed is not None))
        self.w_seedval = pn.widgets.IntInput(
            name="seed", width=100,
            value=(int(src.seed) if src.seed is not None else 42),
            disabled=(src.seed is None),
            description="Pins the whole run: births, birth times and "
            "collision draws. Ion i always draws the same numbers, so "
            "reruns are bit-identical and two tunes are compared on "
            "common random numbers. Unchecked (default): fresh entropy "
            "per run, drawn seed printed.")
        self.w_seeded.param.watch(
            lambda e: setattr(self.w_seedval, "disabled", not e.new),
            "value")
        self.w_dist = pn.widgets.Select(
            name="birth distribution", value=src.distribution,
            options=["point", "disc", "line", "grid", "box",
                     "gaussian"],
            description="Spatial pattern the ions are seeded over: a single "
            "point, a filled disc, a line, a grid, or a BOX (random uniform "
            "placement inside a rectangular volume -- avoids the skew of a "
            "contrived single-point source; extents set below).")
        self.w_x0 = pn.widgets.FloatInput(
            name="start x (mm)", value=src.x0_mm, width=140,
            description="Birth position of the ion cloud centre, x (mm). "
            "In r-z geometry x is the axis.")
        self.w_y0 = pn.widgets.FloatInput(
            name="start y (mm)", value=src.y0_mm, width=140,
            description="Birth position of the ion cloud centre, y (mm). "
            "In r-z geometry y is the radius.")
        self.w_r = pn.widgets.FloatInput(
            name="birth cloud size (mm)", start=0.0, value=src.r_mm,
            width=140, description="Disc radius / line half-length / grid "
            "half-width of the birth cloud.")
        b0 = list(getattr(src, "box_mm", [1.0, 1.0, 1.0])) + [0.0] * 3
        _box_help = ("Full extent (mm) of the birth box along this axis, "
                     "centred on (start x, y, z); each ion is placed "
                     "uniformly at random inside. 0 collapses that axis.")
        self.w_boxx = pn.widgets.FloatInput(name="box dx (mm)", value=b0[0],
                                            start=0.0, width=90, step=0.1,
                                            description=_box_help)
        self.w_boxy = pn.widgets.FloatInput(name="box dy (mm)", value=b0[1],
                                            start=0.0, width=90, step=0.1,
                                            description=_box_help)
        self.w_boxz = pn.widgets.FloatInput(name="box dz (mm)", value=b0[2],
                                            start=0.0, width=90, step=0.1,
                                            description=_box_help)
        # GAUSSIAN spatial beam: per-axis FWHM + MANDATORY
        # truncation half-widths. display == solver input: these are
        # exactly the numbers gaussian_layout() hands the birth draw.
        f0 = list(getattr(src, "fwhm_mm", None) or [0.0, 0.0, 0.0])
        _fw_help = ("Gaussian FWHM (mm) of the birth cloud along this "
                    "axis (distribution=gaussian). 0 = no spread on "
                    "this axis; all three 0 refuses (that is a point).")

        self.w_fwx = pn.widgets.FloatInput(name="FWHM x (mm)", value=f0[0],
                                           start=0.0, width=90, step=0.05,
                                           description=_fw_help)
        self.w_fwy = pn.widgets.FloatInput(name="FWHM y (mm)", value=f0[1],
                                           start=0.0, width=90, step=0.05,
                                           description=_fw_help)
        self.w_fwz = pn.widgets.FloatInput(name="FWHM z (mm)", value=f0[2],
                                           start=0.0, width=90, step=0.05,
                                           description=_fw_help)
        # TRUNCATION IS NOT A UI CONTROL: it is a
        # deck/API declaration. The UI derives an effectively-untruncated
        # 3-sigma window per active axis when it writes a gaussian source
        # (the spec SHOWS the derived value -- display == solver input);
        # a deck-declared trunc_mm is PRESERVED, never stomped.
        self.w_ke_lo = pn.widgets.FloatInput(
            name="KE min (eV)", value=src.ke_lo, width=140,
            description="Lower bound of the birth kinetic energy; each ion "
            "draws uniformly in [min, max]. Set equal for monoenergetic.")
        self.w_ke_hi = pn.widgets.FloatInput(
            name="KE max (eV)", value=src.ke_hi, width=140,
            description="Upper bound of the birth kinetic energy.")
        # SOURCE TEMPERATURE (there was no way to define the 300 K
        # number in the UI: temperature_k rode
        # along invisibly from the loaded JSON, with no widget and no
        # _sync_spec write-back, so the ONE physics knob the refined oa-TOF
        # turns on was undisplayable and unsettable. display == solver
        # input applies to the source too.)
        self.w_temp = pn.widgets.FloatInput(
            name="temperature (K)", value=getattr(src, "temperature_k", 0.0),
            start=0.0, width=140,
            description="Isotropic Maxwell-Boltzmann thermal spread of the "
            "birth velocities: each lab axis gets an independent Gaussian "
            "kick with sigma = sqrt(kT/m), SUPERPOSED on the directed beam "
            "KE, the per-axis dv FWHM, and the drift terms (a beam AT a "
            "temperature). 0 = off: an idealized, perfectly cold beam with "
            "NO thermal scatter. In a TOF that deletes the turn-around "
            "term — usually the resolution-limiting aberration of an OA "
            "source — so a 0 K TOF resolution is optimistic by "
            "construction. Real beams are also rarely isotropic: for an "
            "OA-MRT use the per-axis dv FWHM (m/s) instead, which sets "
            "each axis's effective temperature independently.")
        self.w_z0 = pn.widgets.FloatInput(
            name="start z (mm)", value=getattr(src, "z0_mm", 0.0), width=140,
            description="Birth position z (mm) — relevant for 3-D scenes.")
        # launch DIRECTION (KE is the speed ALONG this vector; it is
        # normalized, so magnitude is irrelevant). Default +x. Exposed as
        # all three components so the user can fly axially (z), radially,
        # or at any angle — not just the +x default.
        d0 = list(src.direction) + [0.0, 0.0, 0.0]
        _dir_help = ("Launch direction vector (the birth KE is directed "
                     "along this; it is normalized, so only the direction "
                     "matters, not the magnitude). e.g. (0,0,1) fires along "
                     "z.")
        self.w_dirx = pn.widgets.FloatInput(name="launch dir x", value=d0[0],
                                            width=90, step=0.1,
                                            description=_dir_help)
        self.w_diry = pn.widgets.FloatInput(name="launch dir y", value=d0[1],
                                            width=90, step=0.1,
                                            description=_dir_help)
        self.w_dirz = pn.widgets.FloatInput(name="launch dir z", value=d0[2],
                                            width=90, step=0.1,
                                            description=_dir_help)
        self.w_mz = pn.widgets.TextInput(
            name="ion mass list (Da, comma-separated)",
            value=", ".join(f"{m:g}" for m in src.mz_list),
            placeholder="e.g. 100, 200, 500")
        # CHARGE STATE (PI directive 2026-09-09): the ion description is
        # mass in Da + a signed integer charge, default 1. The list above
        # carries MASSES (the mz_list field name is historical); charge
        # is NOT folded into the masses anywhere.
        self.w_charge = pn.widgets.IntInput(
            name="charge state (signed z)", value=int(src.charge),
            step=1, width=140,
            description="Signed integer charge state for every ion in "
            "the packet (spec source.charge). 0 refuses at fly time.")
        # A second access point for clear-ions ON the
        # Source subtab (same handler as the top bar — one behavior).
        self.w_clear_ions_src = pn.widgets.Button(
            name="Clear ions", button_type="default", width=110,
            description="Clear the flown ions from the view (same as the "
            "top-bar Clear ions).")
        self.w_clear_ions_src.on_click(self._on_clear)
        self.w_n.param.watch(self._on_ion_count_change, "value")
        self.w_mz.param.watch(self._on_ion_count_change, "value")
        # The Ion Source tab is a
        # shared HEADER (ions per m/z + m/z list — they govern Basic AND
        # Advanced) above four sub-tabs: Basic (source parameters),
        # Advanced (the beam declaration), Stations (the detector
        # station editor), Stats (the persistent stats card).
        ionsrc_header = pn.Column(
            pn.pane.Markdown(
                (f"**Ion Source pinned to FA "
                 f"`{getattr(self, '_fly_src_pin', None)}`** — edits "
                 f"apply to that stage regardless of the displayed FA "
                 f"(*Set Fly Parameters* re-pins).")
                if (getattr(self, "_fly_src_pin", None)
                    and getattr(self, "_fly_src_pin", None)
                    in (getattr(self, "_assembly_specs", None) or {}))
                else "", margin=(0, 5)),
            pn.Row(self.w_n,
                   pn.Column(pn.Spacer(height=6),
                             pn.Row(self.w_seeded, self.w_seedval)),
                   self.w_ion_total),
            self.w_mz,
            self.w_charge,
            pn.pane.Markdown(
                "*Each m/z is flown with its OWN block of "
                "`ions per m/z` ions — e.g. 50 with 3 masses flies "
                "150, 50 of each. These govern Basic and Advanced.*",
                styles={"font-size": "11px"}))
        # ONE PARAMETER PANEL PER DISTRIBUTION: the
        # basic tab used to show every distribution's widgets at once --
        # box extents, disc radius, and gaussian FWHMs all visible
        # regardless of the selected type. Now the block below the
        # selector swaps to exactly the selected distribution's
        # parameters, and switching to a type whose parameters are all
        # zero seeds VIABLE defaults (a gaussian of FWHM 0 is a refusal,
        # not a starting point).
        self._dist_params = pn.Column(sizing_mode="stretch_width")

        def _dist_rows(dist):
            if dist == "point":
                return [pn.pane.Markdown(
                    "*point source — all ions born exactly at the centre "
                    "(no size parameters)*",
                    styles={"font-size": "11px"})]
            if dist in ("disc", "line", "grid"):
                if not self.w_r.value:
                    self.w_r.value = 1.0
                return [self.w_r]
            if dist == "box":
                if not any((self.w_boxx.value, self.w_boxy.value,
                            self.w_boxz.value)):
                    self.w_boxx.value = 1.0
                    self.w_boxy.value = 1.0
                return [pn.Row(self.w_boxx, self.w_boxy, self.w_boxz)]
            if dist == "gaussian":
                if not any((self.w_fwx.value, self.w_fwy.value,
                            self.w_fwz.value)):
                    self.w_fwx.value = 1.0
                    self.w_fwy.value = 1.0
                return [pn.Row(self.w_fwx, self.w_fwy, self.w_fwz),
                        pn.pane.Markdown(
                            "*wings kept to ±3σ automatically; custom "
                            "truncation is a deck/API setting "
                            "(source.trunc_mm), not a UI control*",
                            styles={"font-size": "11px"})]
            return []

        def _on_dist(*_a):
            self._dist_params.objects = _dist_rows(self.w_dist.value)
        self.w_dist.param.watch(_on_dist, "value")
        _on_dist()

        ionsrc_basic = pn.Column(self.w_dist,
                                 pn.Row(self.w_x0, self.w_y0, self.w_z0),
                                 self._dist_params,
                                 pn.Row(self.w_ke_lo, self.w_ke_hi,
                                        self.w_temp),
                                 pn.pane.Markdown(
                                     "**launch direction** (KE is along "
                                     "this vector):"),
                                 pn.Row(self.w_dirx, self.w_diry,
                                        self.w_dirz),
                                 self.w_clear_ions_src)
        # BUILD-ONCE (L-192 family, field-hit 2026-09-11: add-DC-group /
        # TW-group edits froze the visible tab until a tab switch). The
        # Advanced and Stations sub-tabs hold PERSISTENT columns
        # (_beam_col, _station_col); a fresh pn.Column wrapper around
        # them each build re-parented the persistent halves — the exact
        # two-parent transient the stats.card/STL-panel fix removed, one
        # wrapper level down. Advanced needs no wrapper at all; Stations
        # keeps its static caption in a once-built column.
        ionsrc_adv = self._beam_panel()
        if getattr(self, "_stations_col", None) is None:
            # Reference text lives in a COLLAPSED accordion (PI request
            # 2026-09-12): it is reference prose, not a control, and as
            # a permanent wall of text above the editor it was being
            # scrolled past — the impact_plane aperture/patch
            # distinction below cost a session to rediscover. The
            # editor itself stays visible and un-nested.
            _kinds_md = pn.pane.Markdown(
                "**record** — transparent tally; crossings are logged, "
                "flight unchanged.\n\n"
                "**detect** — also transparent in flight; the FIRST "
                "window crossing is the arrival time and later motion "
                "is ignored by analysis.\n\n"
                "**detect / on_hit='pass'** — log only; never touches "
                "the flight. Crossings come post-hoc from the "
                "trajectory.\n\n"
                "**detect / on_hit='splat'** — a detector PATCH: "
                "crossings INSIDE the window ABSORB the ion (fate "
                "'station detect', the detection event); outside "
                "passes. Empty window = full-plane detector.\n\n"
                "**impact_plane** — a physical PLATE: crossings "
                "OUTSIDE the window splat (fate 'station impact "
                "plane'), inside passes. Empty window = wall. "
                "`on_hit` does not apply to this kind.\n\n"
                "**The window inverts when you change kind.** The same "
                "numbers mean a detector patch under `detect` and an "
                "APERTURE under `impact_plane` — a window drawn on the "
                "beam absorbs as a detector and passes everything as a "
                "plate. To stop a beam with `impact_plane`, clear the "
                "window (= solid wall) or set one that EXCLUDES the "
                "beam.\n\n"
                "Bounding planes (Bounds tab) are absolute whole-plane "
                "kills and cannot make a splat window.")
            self._stations_col = pn.Column(
                pn.Accordion(("Station kinds — what each one does",
                              _kinds_md),
                             active=[], sizing_mode="stretch_width",
                             # Panel's default accordion header is bold
                             # and larger than the controls around it,
                             # which made this reference panel shout at
                             # the editor it belongs to (PI 2026-09-13).
                             # Normal weight, inherited size.
                             stylesheets=[
                                 ".accordion-header button, "
                                 ".card-header button, "
                                 ".bk-btn { font-weight: 400; "
                                 "font-size: 1em; }"]),
                self._station_editor())
        ionsrc_stations = self._stations_col

        # --- Voltages tab: session RF groups + per-electrode DC & group
        # DC and RF are independent — each electrode has a DC value AND an
        # optional drive-group assignment. Groups are defined once here
        # (waveform/amplitude/frequency/phase); a travelling wave is just
        # many groups at stepped phases. Waveform is a first-class DROPDOWN
        # sin/square/table now; triangle/sawtooth are a
        # one-line kernel add away.
        self._grp_widgets = {}
        self._drive_pick = {}
        # --- drive-template loader: point at another
        # json and pull ONLY its rf/dc groups into this session, wiping the
        # current groups and clearing every electrode binding so the user
        # re-assigns. Geometry, physics, integration, ion source untouched.
        self.w_tmpl_upload = _mkw(
            pn.widgets.FileInput, accept=".json",
            description="Load ONLY the voltage/drive groups (rf + dc) from "
                        "another spec's json. Wipes the current drive "
                        "groups and clears all electrode assignments — "
                        "geometry, physics, integration and the ion source "
                        "are left untouched.")
        self.w_tmpl_upload.param.watch(self._on_load_drive_template,
                                       "value")
        self.w_tmpl_status = pn.pane.Markdown(
            "_load a drive template: pulls rf + dc groups from another "
            "json, clears electrode assignments for re-binding_")
        grp_rows = [
            pn.pane.Markdown("**Load drive template** (voltage groups "
                             "only, from another json):"),
            self.w_tmpl_upload,
            self.w_tmpl_status,
            pn.layout.Divider(),
            pn.pane.Markdown("**Drive groups** (define waveform + "
                             "amplitude/frequency/phase once; "
                             "assign electrodes below):")]
        for grp in s.geometry.rf_groups:
            wave = pn.widgets.Select(
                name=f"{grp.name} waveform", width=DRIVE_W["wave"],
                options=WAVEFORM_OPTS,
                value=grp.waveform if grp.waveform in WAVEFORM_OPTS
                else "sin",
                description="Drive waveform for this group. 'table' is the "
                "general sampled shape (arbitrary); analytic shapes are "
                "sin/square (triangle/sawtooth extensible).")
            amp = pn.widgets.FloatInput(
                name=f"{grp.name} amplitude (V)", value=grp.amplitude_v,
                width=DRIVE_W["amp"], description="Drive amplitude, 0-to-peak volts "
                "(a value of 100 here means 200 V peak-to-peak).")
            freq = pn.widgets.FloatInput(
                name=f"{grp.name} frequency (Hz)", value=grp.frequency_hz,
                width=DRIVE_W["freq"], description="Drive frequency for this group, "
                "in Hz. For a stepped travelling wave this is the per-group "
                "rate.")
            ph = pn.widgets.FloatInput(
                name=f"{grp.name} phase (deg)", value=grp.phase_deg,
                width=DRIVE_W["phase"], description="Phase offset of this group's "
                "waveform. A travelling wave is groups stepped in phase; "
                "two confinement rails run 180 deg apart.")
            duty = pn.widgets.FloatInput(
                name=f"{grp.name} duty", value=float(getattr(grp, "duty",
                                                             0.5)),
                start=0.01, end=0.99, step=0.05, width=DRIVE_W["duty"],
                description="Square waves only: fraction of the period "
                "HIGH (0.5 = 11110000 across 8 stepped phases; 0.25 = "
                "11000000).")
            # STEP-PULSE DELAY (this extends to time-lag
            # focusing). The instrument case for a table waveform is the
            # 2-point step 0 -> 1 at tau (delayed extraction); tau is the
            # tuning knob, so it gets a widget. A LONGER table stays
            # read-only here (summarised, widget disabled) -- a stated
            # refusal beats a half-editor. When the user flips a group's
            # waveform to 'table' with no table yet, _sync_spec CREATES the
            # canonical step [0, tau] -> [0, 1] hold, so a pulse can be
            # authored entirely from the GUI.
            _tt = list(getattr(grp, "table_t_us", []) or [])
            _is_step = grp.waveform == "table" and len(_tt) in (0, 2)
            tau = pn.widgets.FloatInput(
                name=f"{grp.name} pulse delay τ (µs)",
                value=(_tt[-1] if len(_tt) == 2 else 1.0), start=0.0,
                step=0.1, width=170, disabled=not _is_step,
                description="Step-table drives only: the waveform holds 0 "
                "until τ, then 1 (× amplitude). This is Wiley-McLaren "
                "delayed extraction when it drives the pusher/extraction "
                "plates. NOTE: τ is co-tuned with the mirror detune — "
                "changing τ alone moves the velocity focus off the "
                "detector. Non-step tables are read-only here.")
            _tsum = ("" if _is_step else
                     f"table: {len(_tt)} pts, t {_tt[0]:g}..{_tt[-1]:g} µs "
                     f"(read-only)" if _tt else "")
            tau_row = pn.Row(tau, pn.pane.Markdown(_tsum, width=220))
            def _tau_gate(ev, _tau=tau, _n=len(_tt)):
                _tau.disabled = not (ev.new == "table" and _n in (0, 2))
            wave.param.watch(_tau_gate, "value")
            self._grp_widgets[grp.name] = dict(amp=amp, freq=freq, phase=ph,
                                               wave=wave, duty=duty, tau=tau)
            # INVERTED PICKER for the drive group: choose
            # members from the group side. An electrode can be in several
            # drive groups AND a DC group at once, so this is additive.
            _elopts = [f"e{i+1} — {e.name}"
                       for i, e in enumerate(s.geometry.electrodes)]
            _mem = [f"e{i+1} — {e.name}"
                    for i, e in enumerate(s.geometry.electrodes)
                    if grp.name in (e.rf_groups or [])]
            dpick = pn.widgets.MultiChoice(
                name=f"{grp.name} — members", options=_elopts, value=_mem,
                width=360,
                description="Electrodes driven by this group. Same as the "
                            "per-electrode 'drive groups' box; an electrode "
                            "may be in several drive groups and a DC group.")
            dpick.param.watch(self._on_drive_member_pick, "value")
            self._drive_pick[grp.name] = dpick
            grp_rows.append(pn.Column(pn.Row(wave, amp, freq, ph, duty),
                                      tau_row, dpick))
        # create a NEW drive group (name + waveform) — closing a GUI gap:
        # you couldn't define a group or pick its waveform here.
        self.w_grp_new = _mkw(pn.widgets.TextInput, name="new group name",
                              value="", width=150, placeholder="e.g. RF3")
        self.w_grp_wave = _mkw(pn.widgets.Select, name="waveform",
                               width=120, options=WAVEFORM_OPTS,
                               value="sin")
        self.w_grp_add = _mkw(pn.widgets.Button, name="+ add drive group",
                              width=150)
        self.w_grp_add.on_click(self._on_add_rf_group)
        # BULK removal (one by one is painful):
        # pick any number of groups; one click detaches + removes them
        # all, each detachment reported.
        self.w_grp_del_pick = _mkw(pn.widgets.MultiChoice,
                                   name="remove group(s)", width=230,
                                   options=[g.name for g in
                                            s.geometry.rf_groups],
                                   value=[])
        self.w_grp_del = _mkw(pn.widgets.Button, name="- remove selected",
                              width=130)
        self.w_grp_del.on_click(self._on_remove_rf_group)
        grp_rows.append(pn.Row(self.w_grp_new, self.w_grp_wave,
                               self.w_grp_add, self.w_grp_del_pick,
                               self.w_grp_del))
        # --- travelling-wave builder: create N phase-
        # stepped groups + cyclic electrode assignment in ONE action, instead
        # of wiring TW0..TWn by hand. Pick the ladder electrodes IN ORDER; the
        # phase steps one electrode per segment (that ordering IS the wave). ---
        el_opts_tw = [f"e{i+1} — {e.name}"
                      for i, e in enumerate(s.geometry.electrodes)]
        self.w_tw_members = _mkw(
            pn.widgets.MultiChoice, name="TW ladder electrodes (in order)",
            options=el_opts_tw, value=[], width=360,
            description="Ordered electrodes the wave travels along. Order is "
                        "the wave direction; phase steps one electrode per "
                        "segment. Assignment is cyclic (electrode j -> phase "
                        "j mod N), so any number of segments needs only N "
                        "groups.")
        self.w_tw_nphase = _mkw(pn.widgets.IntInput, name="phases (N)",
                                value=8, start=2, width=90,
                                description="Number of distinct phases "
                                            "(4 and 8 are common SLIM).")
        self.w_tw_wave = _mkw(pn.widgets.Select, name="waveform", width=110,
                              options=["square", "sin"], value="square",
                              description="square = classic stepped SLIM TW; "
                                          "sin = sinusoidal TW.")
        # TW BUILDER DEFAULTS (PI 2026-09-13): 10 kHz, 10 V, 0 V offset
        # — a bipolar +/-10 V drive rather than the old unipolar 0..50 V
        # (25/25/25). These are the STARTING POINT for a new ladder only;
        # an existing deck's groups are never re-defaulted.
        self.w_tw_freq = _mkw(pn.widgets.FloatInput, name="TW freq (kHz)",
                              value=10.0, width=110)
        self.w_tw_amp = _mkw(pn.widgets.FloatInput, name="amp (V)",
                             value=10.0, width=90)
        self.w_tw_off = _mkw(pn.widgets.FloatInput, name="offset (V)",
                             value=0.0, width=90,
                             description="DC offset of the drive. 0 = "
                                         "bipolar, +/-amp about ground. For "
                                         "a unipolar 0..V SLIM drive set "
                                         "amp V/2 and offset V/2 (e.g. amp "
                                         "25, offset 25 -> 0..50 V).")
        self.w_tw_prefix = _mkw(pn.widgets.TextInput, name="prefix",
                                value="TW", width=80)
        self.w_tw_build = _mkw(pn.widgets.Button,
                               name="⟳ build travelling wave", width=200,
                               button_type="primary")
        self.w_tw_build.on_click(self._on_build_tw)
        # RETUNE the whole ladder from the same amp/freq/waveform/offset
        # controls — one action for all N phase groups (hand-
        # editing 8 groups is painful). Preserves per-group phase.
        self.w_tw_retune = _mkw(
            pn.widgets.Button, name="⇄ retune all TW phases", width=200,
            description="Apply the amp / freq / waveform / offset above to "
                        "EVERY phase group of the existing ladder at once, "
                        "keeping each group's phase. Use after 'build "
                        "travelling wave' to change the drive with one edit "
                        "instead of N.")
        self.w_tw_retune.on_click(self._on_retune_tw)
        grp_rows.append(pn.Column(
            pn.pane.Markdown("###### travelling-wave builder"),
            self.w_tw_members,
            pn.Row(self.w_tw_nphase, self.w_tw_wave, self.w_tw_freq,
                   self.w_tw_amp, self.w_tw_off, self.w_tw_prefix),
            pn.Row(self.w_tw_build, self.w_tw_retune)))
        ["(none)"] + [g.name for g in s.geometry.rf_groups]

        # --- DC groups (resistor-divider ladders) ------------------------
        # A ladder is fed at two ends and interpolated across its members, so
        # sweeping an axial field is a TWO-NUMBER edit instead of 33. The
        # member dc is DERIVED (SimSpec.resolve_dc_groups) -- so a member's DC
        # box below goes read-only, because a value you can type AND a value
        # the ladder computes would be two sources of truth for one voltage.
        self._dcg_widgets = {}
        dcg_rows = [pn.pane.Markdown(
            "**DC groups** — feed a chain at both ends; members are "
            "interpolated by their *number*:")]
        self.w_dcg_new = _mkw(pn.widgets.TextInput, name="new group name",
                              value="", width=150,
                              placeholder="e.g. LADDER")
        self.w_dcg_kind = _mkw(
            pn.widgets.Select, name="kind", width=110,
            options=["ladder", "uniform"], value="ladder",
            description="ladder = two-end interpolated divider (set DC in / "
                        "DC out and a number per member). uniform = one "
                        "voltage on every member (no numbering needed).")
        self.w_dcg_add = _mkw(pn.widgets.Button, name="+ add DC group",
                              width=130)
        self.w_dcg_add.on_click(self._on_add_dc_group)
        self.w_dcg_del_pick = _mkw(pn.widgets.MultiChoice,
                                   name="remove DC group(s)", width=230,
                                   options=[g.name for g in
                                            s.geometry.dc_groups],
                                   value=[])
        self.w_dcg_del = _mkw(pn.widgets.Button, name="- remove selected",
                              width=130)
        self.w_dcg_del.on_click(self._on_remove_dc_group)
        self.w_dcg_auto = _mkw(
            pn.widgets.Button, name="auto-number by z", width=150,
            description="Number the members of each group in ascending "
                        "centroid-z, straight from the SOLVE masks. Needs a "
                        "solved field; refuses rather than guessing.")
        self.w_dcg_auto.on_click(self._on_autonumber_dc)
        el_opts = [f"e{i+1} — {e.name}"
                   for i, e in enumerate(s.geometry.electrodes)]
        self._el_opt_of = {f"e{i+1} — {e.name}": i
                           for i, e in enumerate(s.geometry.electrodes)}
        for grp in s.geometry.dc_groups:
            is_uni = getattr(grp, "uniform", False)
            if is_uni:
                vin = _mkw(pn.widgets.FloatInput,
                           name=f"{grp.name} — DC (V) [uniform]",
                           value=grp.v_in, width=180,
                           description="One voltage on EVERY member of this "
                                       "uniform group.")
                vout = None
            else:
                vin = _mkw(pn.widgets.FloatInput,
                           name=f"{grp.name} — DC in (V)", value=grp.v_in,
                           width=150,
                           description="Applied to the member with the LOWEST "
                                       "number.")
                vout = _mkw(pn.widgets.FloatInput,
                            name=f"{grp.name} — DC out (V)", value=grp.v_out,
                            width=150,
                            description="Applied to the member with the "
                                        "HIGHEST number. A RE-WEIGHT, not a "
                                        "re-solve: cache hit.")
            # INVERTED PICKER: choose members from the
            # group side. Coexists with the per-electrode DC-group dropdown;
            # both write el.dc_group, kept in sync on rebuild.
            members = [f"e{i+1} — {e.name}"
                       for i, e in enumerate(s.geometry.electrodes)
                       if e.dc_group == grp.name]
            pick = _mkw(pn.widgets.MultiChoice,
                        name=f"{grp.name} — members", options=el_opts,
                        value=members, width=360,
                        description="Electrodes in this DC group. Editing here "
                                    "is the same as the per-electrode 'DC "
                                    "group' dropdown — an electrode may also "
                                    "be in a drive group at the same time.")
            derived = pn.pane.Markdown("", sizing_mode="stretch_width",
                                       margin=(0, 0, 4, 6))
            self._dcg_widgets[grp.name] = dict(v_in=vin, v_out=vout,
                                               derived=derived, pick=pick,
                                               uniform=is_uni)
            vin.param.watch(self._refresh_dc_derived, "value")
            if vout is not None:
                vout.param.watch(self._refresh_dc_derived, "value")
            pick.param.watch(self._on_dc_member_pick, "value")
            row = pn.Row(vin) if vout is None else pn.Row(vin, vout)
            dcg_rows.append(pn.Column(row, pick, derived))
        dcg_rows.append(pn.Row(self.w_dcg_new, self.w_dcg_kind, self.w_dcg_add,
                               self.w_dcg_del_pick, self.w_dcg_del,
                               self.w_dcg_auto))
        dc_group_opts = ["(none)"] + [g.name for g in s.geometry.dc_groups]

        self._v_widgets = {}
        rows = [pn.pane.Markdown("**Per-electrode** — DC always applies; "
                                 "assign an RF group or (none). A DC-group "
                                 "member's DC is DERIVED from the ladder "
                                 "(read-only) — set its *number* instead:")]
        for i, el in enumerate(s.geometry.electrodes):
            in_ladder = el.dc_group is not None
            dcw = _mkw(
                pn.widgets.FloatInput,
                name=("DC (V) — from ladder" if in_ladder else "DC (V)"),
                value=el.dc, width=150, disabled=in_ladder,
                description="Static (DC) potential on this electrode. When "
                            "the electrode belongs to a DC group this is "
                            "DERIVED from the group's in/out and its number.")
            grp_choices = [g.name for g in s.geometry.rf_groups]
            grpw = _mkw(
                pn.widgets.MultiChoice, name="drive groups", width=180,
                options=grp_choices,
                value=[g for g in el.group_names() if g in grp_choices],
                description="Assign this electrode to zero or MORE drive "
                            "groups (multi-select). An electrode can carry "
                            "a confinement RF and a travelling-wave AC at "
                            "once, plus its DC.")
            dcgw = _mkw(
                pn.widgets.Select, name="DC group", width=130,
                options=dc_group_opts,
                value=(el.dc_group if el.dc_group in dc_group_opts
                       else "(none)"),
                description="Ladder membership. DC and RF groups are "
                            "independent — an electrode can be in both.")
            idxw = _mkw(
                pn.widgets.IntInput, name="number", width=90,
                value=(el.dc_index if el.dc_index is not None else 0),
                description="Position in the ladder. EXPLICIT: order is never "
                            "parsed from the name (E10 would sort before E9). "
                            "DC in lands on the lowest, DC out on the "
                            "highest.")
            self._v_widgets[i] = {"dc": dcw, "group": grpw,
                                  "dc_group": dcgw, "dc_index": idxw}
            # LIVE JSON SYNC: edits reflect in the Config
            # JSON immediately, not only after solve/fly. Cheap: re-sync +
            # rewrite the textarea + refresh the summary.
            for _w in (dcw, grpw, dcgw, idxw):
                _w.param.watch(self._on_live_edit, "value")
            # electrode name on its OWN line (full width) so long assembly
            # names can't overlap the DC/RF-group widget labels beneath them
            rows.append(pn.Column(
                pn.pane.Markdown(f"**e{i + 1} — {el.name}**",
                                 margin=(4, 0, -6, 2), width=560),
                pn.Row(dcw, grpw), pn.Row(dcgw, idxw),
                margin=(0, 0, 4, 0)))
        # UNCONSTRAINED height (a fixed 600 px +
        # inner scroll fought the sidebar's own scrolling — nested
        # scrollbars). The tab now grows to its content; the sidebar
        # scrolls as one surface.
        volt_tab = pn.Column(*grp_rows,
                             pn.layout.Divider(), *dcg_rows,
                             pn.layout.Divider(), *rows)

        # --- Gas tab
        self.w_gas_on = pn.widgets.Checkbox(name="collisions enabled",
                                            value=col.enabled)
        self.w_col_model = pn.widgets.Select(
            name="collision model", value=getattr(col, "model", "hs"),
            options={"hard-sphere (HS)": "hs",
                     "diffusion (SDS)": "sds"},
            description="HS = discrete elastic hard-sphere collisions. "
            "SDS = the Statistical Diffusion model (mobility drift + "
            "diffusion), matched to tabulated ion mobilities.")
        self.w_gas = pn.widgets.Select(
            name="buffer gas", value=col.gas,
            options=["He", "H2", "N2", "air", "Ar", "CO2", "Kr", "Xe"])
        self.w_T = pn.widgets.FloatInput(
            name="gas temperature (K)", value=col.T_k,
            description="Buffer-gas temperature; sets the thermal velocity "
            "ions relax toward.")
        self.w_P = pn.widgets.FloatInput(name="P (Torr)",
                                         value=getattr(col, "P_torr", 1.0))
        self.w_sigma = pn.widgets.LiteralInput(
            name="HS cross-section σ (m²)", value=col.sigma_m2,
            type=(int, float),
            description="Hard-sphere ion–neutral collision cross-section. "
            "Used only by the HS model; larger σ = more frequent "
            "collisions. Typical N₂ value ≈ 2.3e-18 m².")
        self.w_gdiam = pn.widgets.FloatInput(
            name="SDS gas diameter (nm)",
            value=getattr(col, "gas_diam_nm", 0.366),
            description="Neutral gas molecule diameter, used only by the SDS "
            "(diffusion) model to set the mean free path. Air ≈ 0.366 nm.")
        # dt-adequacy advisor: text beside the pressure
        # telling the user what time step the CURRENT gas + RF settings
        # demand. Computed with the SAME functions the kernel uses
        # (_mfp_mm, c_star/c_bar, _sds_ion_params) — advice ==
        # computation, no second physics.
        self.w_dt_advice = pn.pane.Markdown("", width=560)
        gas_tab = pn.Column(self.w_gas_on, self.w_col_model, self.w_gas,
                            self.w_T, self.w_P, self.w_dt_advice,
                            self.w_sigma, self.w_gdiam)
        for _w in (self.w_gas_on, self.w_col_model, self.w_gas, self.w_T,
                   self.w_P, self.w_sigma, self.w_gdiam):
            _w.param.watch(lambda _e: self._update_dt_advice(), "value")

        # --- Integration tab
        self.w_dt = pn.widgets.FloatInput(
            name="time step Δt (ns)", value=integ.dt_ns,
            description="Integration time step. Smaller = more accurate but "
            "slower; it must resolve the fastest RF (rule of thumb ≈ 1/20 of "
            "the RF period, so ≈0.06 µs at 800 kHz).")
        self.w_tmax = pn.widgets.FloatInput(
            name="max flight time (µs)", value=integ.t_max_us,
            description="Longest an ion is flown. An ion still airborne at "
            "this time stops with fate 'time out'.")
        self.w_rec = pn.widgets.IntInput(
            name="record every N steps", value=integ.rec_every,
            description="Trajectory sampling interval: one point is stored "
            "every N integration steps. Larger = fewer stored points (smaller "
            "files, faster plots); it does NOT change the physics, only how "
            "densely the path is saved.")
        self.w_maxrec = pn.widgets.IntInput(
            name="max recorded points per ion", value=int(
                getattr(integ, "max_records", 100000)), step=50000,
            description="Trajectory buffer cap. When a flight needs more "
            "points than this the RECORDING stops but the ion keeps "
            "flying — the app warns before the fly and names the true "
            "stopping point afterwards. Raise this (or 'record every N') "
            "to capture a whole long flight.")
        self.w_chan = pn.widgets.MultiChoice(
            name="extra recorded quantities", value=list(integ.record_channels),
            options=list(OPTIONAL_CHANNELS.keys()),
            description="Additional per-point values to store along each "
            "trajectory (e.g. field magnitude, collision count) for colouring "
            "or export.")
        self.w_store_traj = pn.widgets.Checkbox(
            name="store trajectories (off = fly for fates only, low memory)",
            value=getattr(self, "_store_traj_val", True))
        self.w_autoclear = pn.widgets.Checkbox(
            name="clear previous stored runs on each Fly",
            # DEFAULT ON: a Fly starts from a clean run
            # store; untick to accumulate overlays deliberately.
            value=getattr(self, "_autoclear_val", True))
        self.w_clearruns = pn.widgets.Button(
            name="Clear stored runs now", button_type="default", width=180)
        self.w_clearruns.on_click(self._on_clear_runs)
        # Solved-field cache clear (bug D): disk + in-memory bases, surfaced
        # prominently in the Config > Runs maintenance row.
        self.w_clearcache = pn.widgets.Button(
            name="Clear field cache", button_type="warning", width=170)
        self.w_clearcache.on_click(self._on_clear_cache)
        integ_tab = pn.Column(
            self.w_dt, self.w_tmax, self.w_rec, self.w_maxrec, self.w_chan,
            pn.layout.Divider(),
            pn.pane.Markdown("**long-run storage** — trajectories can be "
                             "large; turn storage off to just fly ions and "
                             "keep only their fates. (*Clear stored runs is "
                             "on Config › Runs › maintenance.*)"),
            self.w_store_traj, self.w_autoclear)

        # --- Bounds tab (impact planes; all off by default)
        bnd = s.bounds
        self.w_bnd = {}
        bnd_rows = [pn.pane.Markdown(
            "Bounding/impact planes — an ion crossing an ENABLED plane "
            "terminates (fate: bounding plane). All off by default. In "
            "r-z, x is the axis and y the radius.")]
        for axis, lo_lbl, hi_lbl in [("x", "x min", "x max"),
                                     ("y", "y min", "y max"),
                                     ("z", "z min", "z max")]:
            on_lo = pn.widgets.Checkbox(
                name=f"{lo_lbl} on", value=getattr(bnd, f"{axis}_min_on"),
                width=90)
            v_lo = pn.widgets.FloatInput(
                name=f"{lo_lbl} (mm)", value=getattr(bnd, f"{axis}_min"),
                width=140)
            on_hi = pn.widgets.Checkbox(
                name=f"{hi_lbl} on", value=getattr(bnd, f"{axis}_max_on"),
                width=90)
            v_hi = pn.widgets.FloatInput(
                name=f"{hi_lbl} (mm)", value=getattr(bnd, f"{axis}_max"),
                width=140)
            self.w_bnd[axis] = dict(min_on=on_lo, min=v_lo,
                                    max_on=on_hi, max=v_hi)
            bnd_rows.append(pn.Row(on_lo, v_lo, on_hi, v_hi, width=620))
        bounds_tab = pn.Column(*bnd_rows, width=630)

        # --- Display tab
        #
        # EVERY widget here is PERSISTENT (a repeated
        # report of "Display settings not applied to assembly traces").
        # _build_controls re-runs on every spec load, Set-FA-View and
        # source-FA pin, and these used to be plain constructions — so
        # each rebuild REBOUND self.w_* to fresh widgets at their
        # DEFAULTS. Whichever copy the served layout held, the user's
        # choices were discarded or orphaned, while the draw path (which
        # honours the style dict) looked correct in any test that
        # rebuilt the layout — the exact verification trap _persistent's
        # docstring names. Identity-stable widgets + wire-once watchers
        # close both halves at the root.
        _disp = lambda w: w.param.watch(self._on_display_change, "value")
        self._persistent(
            "w_contours", lambda: pn.widgets.IntSlider(
                name="field contour lines (0 = off)", start=0, end=40,
                value=CONTOURS_DEFAULT), wire=_disp)
        # One-click contours off/on (a toggle
        # next to the contour count that
        # switches between off and the default). A stateless
        # Button, not a Toggle: the SLIDER stays the single value
        # authority (a Toggle would hold a second copy of on/off that
        # goes stale the moment the slider is dragged to 0 by hand).
        # The click writes the slider, whose existing watcher redraws.
        def _wire_ctoggle(btn):
            def _flip(_evt):
                self.w_contours.value = (
                    0 if self.w_contours.value > 0 else CONTOURS_DEFAULT)
            btn.on_click(_flip)
        self._persistent(
            "w_contours_toggle", lambda: pn.widgets.Button(
                name="contours off/on", width=110, align="end",
                button_type="default"), wire=_wire_ctoggle)
        self._persistent(
            "w_showfield", lambda: pn.widgets.Checkbox(
                name="shade field / potential", value=False), wire=_disp)
        # Where the xz/yz shading is cut: mm along the axis NOT shown (y for
        # xz, x for yz) in the canonical frame (mirror plane at 0). At the
        # minimum it uses the automatic slice. The xy view is unaffected.
        # (Persistent: its start/end are re-ranged per plane at draw time,
        # which mutates the SAME object — no rebuild dependence.)
        self._persistent(
            "w_viewslice", lambda: pn.widgets.FloatSlider(
                name="xz/yz shading slice (mm; auto at min)",
                start=-1.0, end=0.0, step=0.05, value=-1.0, width=230),
            wire=_disp)
        self._persistent(
            "w_fieldmode", lambda: pn.widgets.Select(
                name="shading type", width=200,
                options=["|E| field", "PE surface (effective)",
                         "PE 3D landscape (node-centred)"],
                value="|E| field",
                description="What the shading shows: instantaneous field "
                "magnitude |E|, the effective RF pseudopotential (PE) as a "
                "2-D map, or the PE as a 3-D landscape."),
            wire=lambda w: (
                _disp(w),
                # Picking a shading TYPE while the master switch is off
                # silently did nothing (browser pass: "PE does not
                # render"). The pick IS the intent, so it enables the
                # switch (which redraws via its own watcher). Wired ONCE,
                # here, so rebuilds cannot stack duplicates.
                w.param.watch(
                    lambda e: (setattr(self.w_showfield, "value", True)
                               if not self.w_showfield.value else None),
                    "value")))
        # (PE surface m/z lives on the PE Surface tab — PeSurfaceTab.w_mz.
        # A second copy here (Display tab) was a reported duplicate;
        # the draw path reads self._pe_tab.w_mz.)
        self._persistent(
            "w_lock", lambda: pn.widgets.Checkbox(
                name="lock aspect ratio (equal x/y scale)", value=False),
            wire=_disp)
        self._persistent(
            "w_verbose", lambda: pn.widgets.Checkbox(
                name="verbose build log", value=True))
        self._persistent(
            "w_trajmode", lambda: pn.widgets.Select(
                name="trajectory style", value="lines",
                options=["lines", "dots", "lines+dots"],
                description="Draw each ion path as connected lines, "
                "individual recorded points, or both."), wire=_disp)
        self._persistent(
            "w_width", lambda: pn.widgets.FloatSlider(
                name="line width", start=0.2, end=4.0, value=0.8,
                step=0.1), wire=_disp)
        # Does the max of 25 hold for a regular FA?
        # yes — the single-FA view uses the same declared policy as the
        # assembly and /flight (floor 25, 25% of the packet). This knob
        # OVERRIDES it per session: 0 = the policy; N = draw exactly
        # min(N, packet). Display only — statistics always use all.
        self._persistent(
            "w_maxpaths", lambda: _mkw(
                pn.widgets.IntInput, name="paths drawn (0 = auto 25%)",
                value=0, start=0, end=100000, step=25, width=150,
                description=("How many ion paths to draw, evenly "
                             "strided across the packet. 0 keeps the "
                             "automatic policy: all up to 25, then 25% "
                             "of the packet. Statistics and impact "
                             "markers always use every ion.")),
            wire=_disp)
        self._persistent(
            "w_alpha", lambda: pn.widgets.FloatSlider(
                name="opacity", start=0.05, end=1.0, value=0.5,
                step=0.05), wire=_disp)
        self._persistent(
            "w_colorby", lambda: pn.widgets.Select(
                name="colour traces by", value="m/z",
                options=["fate", "m/z", "solid color"]
                + list(OPTIONAL_CHANNELS.keys()),
                description="Colour each trajectory by its outcome (fate), "
                "by m/z (one colour per mass, with a legend), a single "
                "colour, or a recorded channel."), wire=_disp)
        self._persistent(
            "w_solidcolor", lambda: pn.widgets.ColorPicker(
                name="trace color", value="#1f77b4"),
            wire=lambda w: (
                _disp(w),
                # THE PICK IS THE INTENT (a trace
                # color set purple while colour-by read "fate" — a live
                # control silently ignored). Editing the
                # color switches colour-by to "solid color", the same
                # convention as shading-type enabling the master switch;
                # the colour-by watcher then redraws with the new color.
                w.param.watch(
                    lambda e: (setattr(self.w_colorby, "value",
                                       "solid color")
                               if self.w_colorby.value != "solid color"
                               else None), "value")))
        self._persistent(
            "w_plane", lambda: pn.widgets.Select(
                name="view plane", value="xy", options=["xy", "xz", "yz"],
                description="Which 2-D plane to project the trajectories "
                "and field onto for display."), wire=_disp)
        self._persistent(
            "w_decim", lambda: pn.widgets.IntSlider(
                name="plot every Nth point (display only)", start=1,
                end=20, value=3), wire=_disp)
        self._persistent(
            "w_impactsize", lambda: pn.widgets.FloatSlider(
                name="impact marker size", start=2, end=16, value=7),
            wire=_disp)
        self._persistent(
            "w_impactsym", lambda: pn.widgets.Select(
                name="impact marker symbol", value="circle",
                options=["circle", "x", "cross", "diamond", "star",
                         "square"],
                description="Marker shape drawn where an ion terminates on "
                "metal or a bounding plane."), wire=_disp)
        self._persistent(
            "w_elcolor", lambda: pn.widgets.ColorPicker(
                name="electrode color", value="#9aa0a6", width=120),
            wire=_disp)
        self._persistent(
            "w_elfill", lambda: _mkw(
                pn.widgets.Checkbox, name="electrode fills", value=False,
                description="Shaded electrode bodies. Turn OFF to see ion "
                "trajectories through dense geometry — the OUTLINES stay, "
                "so the metal is still unambiguous."), wire=_disp)
        self._persistent(
            "w_ellabel", lambda: _mkw(
                pn.widgets.Checkbox, name="electrode labels", value=True,
                description="The e1/e2/E3... tags. Turn OFF for a "
                "35-conductor ladder, where the labels stack on each other "
                "and on the beam."), wire=_disp)
        self._persistent(
            "w_elalpha", lambda: pn.widgets.FloatSlider(
                name="electrode alpha", start=0.1, end=1.0, value=0.6,
                step=0.05, width=180), wire=_disp)
        self._persistent(
            "w_field_btn",
            lambda: pn.widgets.Button(name="Show field (no ions)"),
            wire=lambda w: w.on_click(lambda e: self._show_field()))
        # FLIGHT CHIP (for some flights it is
        # unclear whether the flight is still going or
        # frozen). An ALWAYS-VISIBLE heartbeat in the TOP BUTTON
        # ROW — the status line moved to the Status tab, so a
        # user watching the View tab during a long flight had no live
        # signal at all. One widget, one container: it lives in
        # panel()'s top row only; persistent so control rebuilds cannot
        # orphan it. Content comes ONLY from _fly_chip().
        self._persistent(
            "fly_chip", lambda: pn.pane.Markdown(
                "", margin=(8, 10, 0, 10),
                styles={"white-space": "nowrap"}))
        # NOTE: pn.widgets.Checkbox does NOT accept `description` on the
        # supported Panel version (it is a TypeError at construction --
        # every other description= in this file is on IntInput / FloatInput
        # / Select / Button / MultiChoice, which do accept it). Help text
        # for checkboxes goes in the adjacent Markdown instead.
        #
        # SQUARE VIEW REMOVED (it did not work smoothly).
        # There is now NO under-plot control row at all, so nothing sits
        # between the pane and the control column to bleed across it.
        # BUILD-ONCE: every child here is a _persistent widget (or static
        # markdown); a fresh Column around them each build re-parented
        # all ~22 on every rebuild — the measured freeze trigger for the
        # 2026-09-11 field report (add group -> visible tab dead until a
        # tab switch re-rendered).
        if getattr(self, "_disp_col", None) is None:
            self._disp_col = pn.Column(
                pn.pane.Markdown("#### field overlay", margin=(6, 0, 0, 0)),
                self.w_plane,
                pn.Row(self.w_contours, self.w_contours_toggle),
                self.w_showfield,
                self.w_fieldmode, self.w_viewslice, self.w_lock,
                self.w_verbose,
                self.w_trajmode, self.w_width, self.w_maxpaths,
                self.w_alpha, self.w_colorby,
                self.w_solidcolor, self.w_decim, self.w_impactsize,
                self.w_impactsym, pn.Row(self.w_elfill, self.w_ellabel),
                pn.Row(self.w_elcolor, self.w_elalpha),
                self.w_field_btn)
        disp_tab = self._disp_col
        # Watchers for every display widget above are wired ONCE at first
        # creation (the wire= of each _persistent call). The per-rebuild
        # watch loop that lived here would now attach a DUPLICATE watcher
        # on every _build_controls re-run — the doubly-wired-callback
        # failure _persistent's docstring warns about — so it is gone,
        # not merely emptied.

        # --- Analysis tab: plot any recorded channel vs any other. Built
        # ONCE (like the PE tab) so the right-side plot_tabs keeps a live
        # reference to self.analysis_pane; only the channel OPTIONS refresh
        # when the spec changes. (Rebuilding it here would leave plot_tabs
        # pointing at a detached pane, so "Plot" would do nothing.)
        # axis options: the recorded columns PLUS the always-derivable
        # channels (speed/ke_ev/radius are functions of the base trajectory,
        # so they are plottable for every run without being recorded — see
        # _traj_column). Field channels stay opt-in via the Integration tab.
        chan_opts = self.spec.column_names()
        for _d in ("speed", "ke_ev", "radius", "e_field", "e_axial"):
            if _d not in chan_opts:
                chan_opts = chan_opts + [_d]
        if getattr(self, "analysis_pane", None) is None:
            self.w_ax = pn.widgets.Select(
                name="x axis", options=chan_opts, width=300,
                value="t" if "t" in chan_opts else chan_opts[0],
                description="Recorded channel for the horizontal axis.")
            self.w_ay = pn.widgets.Select(
                name="y axis", options=chan_opts, width=300,
                value="x" if "x" in chan_opts else chan_opts[-1],
                description="Recorded channel for the vertical axis.")
            self.w_amode = pn.widgets.Select(
                name="data", width=200,
                options={"trajectory (all points)": "traj",
                         "per-ion endpoints": "end",
                         "histogram of x (endpoints)": "hist",
                         "histogram of x (trajectories)": "hist_traj"},
                value="traj",
                description="Plot every recorded point, one point per ion at "
                "its final state, or a histogram of the x-axis quantity over "
                "per-ion endpoints (y axis is ignored).")
            self.w_bins = pn.widgets.IntInput(
                name="bins", value=20, start=2, end=200, width=90,
                description="Histogram bin count. A good rule of thumb is "
                "about sqrt(number of ions); the default 20 suits ~50-400 "
                "ions. Only used in histogram mode.")
            self.w_agroup = pn.widgets.Select(
                name="colour / trace by", width=200,
                options=["ion", "all together", "fate", "m/z"], value="ion",
                description="ion: a separate colour per ion (cycling "
                "palette). all together: one series. fate: colour by "
                "outcome. m/z: one colour per m/z value — each ion is "
                "coloured by its OWN flown m/z (stable palette, sorted "
                "numerically, shown in the legend); works in all three "
                "plot modes.")
            self.w_astyle = pn.widgets.Select(
                name="style", width=180,
                options=["lines", "markers", "lines+markers"],
                value="lines")
            self.w_ascheme = pn.widgets.Select(
                name="colour scheme", width=160,
                options=["Plotly", "D3", "G10", "T10", "Dark24", "Set1",
                         "Set2", "Bold", "Safe", "Vivid"],
                value="Plotly",
                description="Palette for line/marker colours — applies to "
                "per-ion cycling, m/z colouring (stable assignment over "
                "sorted m/z), and the all-together series. Fate colours "
                "stay fixed: they are semantic (impact/alive/timeout), not "
                "aesthetic.")
            # NOTE (caught by the bridged UI gate): panel's
            # FloatSlider does NOT accept description= (Select/MultiChoice
            # do) — passing it crashed Analysis-tab construction from
            # v284. Guidance lives in the name instead.
            # NAME COLLISION FIX (the alpha slider did
            # nothing): this slider was assigned to self.w_alpha,
            # OVERWRITING the main-view opacity slider bound at
            # construction — the analysis redraw then read whichever
            # object won, and this one had no watcher. Distinct name +
            # its own watcher.
            self.w_an_alpha = pn.widgets.FloatSlider(
                name="alpha (opacity of lines/markers/bars)",
                start=0.05, end=1.0, step=0.05, value=0.7, width=220)
            self.w_an_alpha.param.watch(
                lambda e: self._on_analysis_plot(), "value_throttled")
            self.w_amz = pn.widgets.MultiChoice(
                name="m/z filter", options=[], value=[], width=260,
                description="Empty = ALL m/z flown (default). Pick one or "
                "more values to plot only those ions. Options come from "
                "the displayed run's FLOWN ions (each ion's own recorded "
                "m/z), not from the editor — so a reloaded older run "
                "offers what it actually flew.")
            self.w_plot_btn = pn.widgets.Button(
                name="Plot", button_type="primary", width=90)
            self.w_plot_btn.on_click(self._on_analysis_plot)
            self.analysis_pane = pn.pane.Plotly(
                height=560, sizing_mode="stretch_width")
            # --- Thermal assessment (two-estimator temperatures) ---
            # Uses the SAME framework functions as notebook 05
            # (traj_stats.two_estimator_report + viz_core.
            # temperature_estimator_figure). The tail-fit percentile is
            # exposed: T_slope depends on where the
            # tail fit starts, so it is a control, not a buried default.
            self.w_therm_pctl = pn.widgets.FloatSlider(
                name="T_slope tail-fit percentile", start=0.0, end=60.0,
                step=5.0, value=20.0, width=280)
            self.w_therm_bins = pn.widgets.IntInput(
                name="bins", value=80, start=10, end=300, width=90,
                description="Histogram bins for the velocity and energy "
                "distributions.")
            self.w_therm_btn = pn.widgets.Button(
                name="Assess temperature", button_type="primary",
                width=170)
            self.w_therm_btn.on_click(self._on_thermal_assess)
            self.thermal_pane = pn.pane.Plotly(
                height=620, sizing_mode="stretch_width")
            self.thermal_table = pn.pane.Markdown(
                "_Fly or load a run, then press **Assess temperature**._")
            self._analysis_tab = pn.Column(
                pn.pane.Markdown(
                    "**Analysis** — plot any recorded quantity against any "
                    "other for the displayed run (e.g. x vs t, ke_ev vs t, "
                    "vz vs z). Add extra channels on the Integration tab."),
                pn.Row(self.w_ax, self.w_ay),
                pn.Row(self.w_amode, self.w_agroup, self.w_astyle,
                       self.w_bins, self.w_amz),
                pn.Row(self.w_ascheme, self.w_an_alpha),
                self.w_plot_btn, self.analysis_pane,
                sizing_mode="stretch_width")
            # --- THERMAL tab (thermal analysis lives
            # in its own tab, with hot-spot analysis) --------------
            self.w_hot_thresh = pn.widgets.FloatInput(
                name="hot threshold (eV; blank = 10 kT)", value=None,
                width=200)
            self.w_hot_cell = pn.widgets.FloatInput(
                name="cell size (mm)", value=0.12, width=120)
            self.w_hot_nions = pn.widgets.IntInput(
                name="ions", value=40, start=1, width=90)
            self.w_hot_btn = pn.widgets.Button(
                name="Map hot spots", button_type="primary", width=150)
            self.w_hot_btn.on_click(self._on_hotspot_map)
            self.hotspot_pane = pn.pane.Plotly(
                height=460, sizing_mode="stretch_width")
            self.hotspot_note = pn.pane.Markdown(
                "_Fly or load a run, then press **Map hot spots**._")
            self._thermal_tab = pn.Column(
                pn.pane.Markdown(
                    "**Thermal assessment** — two temperature estimators "
                    "for the displayed run: T_var (velocity variance, "
                    "drift-subtracted) and T_slope (Maxwell-Boltzmann "
                    "energy-tail slope). When they disagree the "
                    "distribution is non-thermal (a hot tail on a cooler "
                    "core). The tail-fit percentile sets where the "
                    "Maxwell-Boltzmann log-tail slope is fit: for a "
                    "non-thermal distribution T_slope depends on this "
                    "cut, so it is stated, not fixed."),
                pn.Row(self.w_therm_pctl, self.w_therm_bins,
                       self.w_therm_btn),
                self.thermal_table, self.thermal_pane,
                pn.layout.Divider(),
                pn.pane.Markdown(
                    "**Hot spots** — each spatial cell shows the MAXIMUM "
                    "recorded KE it ever hosted (a single hot sample "
                    "cannot hide under thousands of cold ones); cells "
                    "below the threshold are gray, and the colour range "
                    "runs to the true maximum — the extremes are the "
                    "signal."),
                pn.Row(self.w_hot_nions, self.w_hot_cell,
                       self.w_hot_thresh, self.w_hot_btn),
                self.hotspot_note, self.hotspot_pane,
                sizing_mode="stretch_width")
        else:
            # refresh channel options for the new spec, preserving selection
            for w in (self.w_ax, self.w_ay):
                keep = w.value
                w.options = chan_opts
                w.value = keep if keep in chan_opts else chan_opts[0]

        # --- Config tab (examples, save/load, reload past runs)
        self.w_examples = pn.widgets.Select(
            name="load example", options=["(keep current)"]
            + list(_example_specs().keys()))
        self.w_loadex = pn.widgets.Button(name="Load example",
                                          button_type="primary")
        self.w_loadex.on_click(self._on_load_example)
        # SELECTING an example STAGES it into the JSON box; it does not apply.
        # The estimator already reports for the JSON in the box (_editor_spec),
        # so the cost of an example appears the moment you pick it -- BEFORE
        # you commit. Previously the cost only appeared after loading, which
        # is exactly backwards: you found out what it cost by paying it.
        self.w_examples.param.watch(self._on_example_selected, "value")
        # --- Staged multi-FA assembly controls.
        # An instrument.json lists stages that each solve in their OWN grid
        # and hand the ion across a seam, so it is NOT one SimSpec and the
        # single-spec loader refuses it by design. The app handles it by
        # showing ONE STAGE AT A TIME: selecting a stage makes that stage's
        # spec the live spec, so every existing view, solve and diagnostic
        # works on it unchanged. Nothing about the assembly needs its own
        # rendering path.
        #
        # Selecting a stage does NOT solve anything. Loading an instrument
        # solves only what the user asks to look at, because solving every
        # stage on load would charge for the whole instrument to inspect
        # one plate.
        self._persistent(
            "w_stage",
            lambda: pn.widgets.Select(
                name="assembly stage", options=["(no assembly loaded)"],
                disabled=True),
            wire=lambda w: w.param.watch(self._on_stage_selected, "value"))
        # Its OWN upload, beside the stage controls. The spec uploader in
        # section 2 already handles an instrument correctly (it routes
        # through _on_apply_json, which sniffs), but nothing told anyone
        # that, and the section-1b text pointed at a JSON box further down
        # the page. A 61 kB instrument is not something anyone pastes.
        # A working path nobody can find is not a working path.
        self._persistent(
            "w_instrument",
            lambda: pn.widgets.FileInput(accept=".json"),
            wire=lambda w: w.param.watch(self._on_upload, "value"))
        self._persistent(
            "w_fly_assembly",
            lambda: pn.widgets.Button(
                name="Fly assembly (all stages)", button_type="primary",
                disabled=True),
            wire=lambda w: w.on_click(self._on_fly_assembly))
        # Lives beside the main Fly button, not buried in Config.
        # Its value SURVIVES a control rebuild: _build_controls re-runs on
        # every spec load and stage swap, so a freshly constructed widget
        # would silently reset the user's choice each time they changed
        # stage -- and the symptom would be the Fly button quietly going
        # back to flying one stage.
        self._persistent(
            "w_view_assembly",
            lambda: pn.widgets.Button(
                name="Set Assembly View", button_type="default",
                disabled=True),
            wire=lambda w: w.on_click(self._on_view_assembly))
        # SET FA VIEW: ONE button applying
        # the FA View dropdown — "Full Assembly" (the default) or an FA.
        self._persistent(
            "w_set_fa_view",
            lambda: pn.widgets.Button(
                name="Set FA View", button_type="default",
                disabled=True),
            wire=lambda w: w.on_click(self._apply_fa_view))
        self._persistent(
            "w_loaded_name",
            lambda: pn.pane.Markdown("*(no instrument loaded)*"))
        # MULTI FA FLIGHT PARAMETERS: pick the FA whose ion
        # parameters populate the Ion Source tab. The instrument's own
        # default flag is beam.from_stage — shown as the initial value.
        self._persistent(
            "w_fly_src",
            lambda: pn.widgets.Select(name="flight-parameters FA",
                                      options=[], disabled=True))
        self._persistent(
            "w_set_fly_params",
            lambda: pn.widgets.Button(
                name="Set Fly Parameters", button_type="default",
                disabled=True),
            wire=lambda w: w.on_click(self._on_set_fly_params))
        self._persistent(
            "w_fly_mode",
            lambda: pn.widgets.Checkbox(
                name="Fly whole assembly", value=False, disabled=True))
        # SOLVE TARGET (which stage gets solved on Recompute needs
        # a selector to tell
        # the solver). Recompute acts on ONE stage's
        # geometry, so it needs a named target and the user needs to see
        # which. Defaults to following the displayed subject, because the
        # common case is "solve what I am looking at" and a target that
        # silently differs from the view is the ambiguity this removes.
        self._persistent(
            "w_assembly_info",
            lambda: pn.pane.Markdown(
                "*No instrument loaded.* Upload an instrument JSON above; "
                "these controls stay disabled until one is."))
        self._persistent(
            "w_solve_target",
            lambda: pn.widgets.Select(
                name="solve", options=["(displayed stage)"],
                width=150, disabled=True))
        # _build_controls RE-RUNS on every spec load (via
        # _rebuild_for_new_spec), so this state is PRESERVED across a
        # rebuild rather than initialised here. Initialising it here wiped
        # the loaded assembly the instant the first stage was shown --
        # the stage displayed correctly and the selector then reported no
        # assembly loaded, which looks like a rendering glitch rather
        # than lost state.
        self._assembly_doc = getattr(self, "_assembly_doc", None)
        self._assembly_specs = getattr(self, "_assembly_specs", {})
        if getattr(self, "_station_col", None) is not None:
            # The station editor reads the LIVE spec; a stage switch or a
            # plain load just changed it, so the picker must follow -- a
            # picker naming the previous spec's stations is stale
            # state wearing a different widget.
            self._station_sync_pick()
        self._stage_swapping = getattr(self, "_stage_swapping", False)
        if self._assembly_specs:
            # THE SUBJECT'S OPTIONS. `<whole assembly>` is an
            # ordinary entry in this list, not a separate button, because
            # a button is a one-shot action and its result is dropped by
            # the next redraw -- which is exactly the reported
            # symptom: switching xy/xz/yz silently returned the last
            # sub-stage. As a CHOICE it survives every redraw, because
            # the redraw path reads the selection instead of remembering
            # what was last clicked.
            _names = list(self._assembly_specs) + [WHOLE_ASSEMBLY]
            self.w_stage.options = ["Full Assembly"] + _names
            _tgt = ["(displayed stage)"] + list(self._assembly_specs)
            if list(self.w_solve_target.options) != _tgt:
                self.w_solve_target.options = _tgt
            self.w_solve_target.disabled = False
            self.w_stage.disabled = False
            self.w_fly_assembly.disabled = False
            self.w_fly_mode.disabled = False
            self.w_view_assembly.disabled = False
            self.w_set_fa_view.disabled = False
            _keep = getattr(self, "_assembly_stage", None) or _names[0]
            if _keep == WHOLE_ASSEMBLY:
                _keep = "Full Assembly"
            # Restoring the selection must NOT re-enter the swap. Showing
            # a stage rebuilds the controls, which lands here, which sets
            # this value, which fires the watcher, which shows a stage:
            # measured 89 rebuilds for ONE switch (57 s, against 0.64 s
            # for a single rebuild). The guard is what makes the restore
            # a display update rather than a new swap.
            self._stage_swapping = True
            try:
                self.w_stage.value = _keep if _keep in _names else _names[0]
            finally:
                self._stage_swapping = False
        self.w_json = pn.widgets.TextAreaInput(
            name="spec JSON", height=200, value=self.spec.to_json())
        # summary table under the JSON: legible digest of
        # the staged spec — updates live so a changed JSON is visible at a
        # glance (electrodes, symmetry, drive/DC groups, size, pitch, ions).
        self.w_spec_summary = pn.pane.Markdown(
            "", sizing_mode="stretch_width", height=230,
            styles={"overflow-y": "auto"})
        self.w_applyjson = pn.widgets.Button(name="Apply JSON")
        # visible busy indicator for the synchronous rebuild (large imports
        # take a beat and blocked the UI silently).
        self.w_apply_busy = pn.indicators.LoadingSpinner(
            value=False, size=20, name="")
        self.w_applyjson.on_click(self._on_apply_json)

        # --- solve resolution + cost, for a LOADED JSON ------------------
        # An STL import has had a pitch control and an honest cost readout
        # since day one; a JSON-loaded geometry had neither, so the only way
        # to change its solve resolution was to edit mm_per_gu in the text
        # box and hope. Same estimator (sizing.py), same contract: the USER
        # picks the pitch, the app reports what it will cost. No pitch is
        # ever proposed -- guessing one for someone is how a 0.5 mm gap ends
        # up resolved by two cells.
        self.w_pitch = _mkw(
            pn.widgets.FloatInput, name="solve resolution (mm/gu)",
            value=self.spec.geometry.mm_per_gu, step=0.01, start=0.001,
            end=10.0, width=200,
            description="Grid pitch for the field solve. Smaller = finer = "
                        "costlier (nodes scale as pitch^-3 in 3-D).")
        self.w_minfeat = _mkw(
            pn.widgets.FloatInput, name="smallest feature (mm, optional)",
            value=0.0, step=0.05, start=0.0, end=100.0, width=220,
            description="The narrowest gap/slot you care about. Used ONLY to "
                        "warn when the pitch cannot resolve it (<~3 cells).")
        # MIRROR SYMMETRY (scene3d geometry). Declaring a mirror axis says
        # the stored grid is HALF the physical domain with the mirror plane
        # at the low-index face; the solver applies a reflecting BC and the
        # field unfolds to the whole. Multiple axes are allowed (e.g. x+y
        # halves the solve twice). Only meaningful for scene3d specs; it is
        # a GEOMETRY change, so editing it invalidates the cached field.
        self.w_mirror = pn.widgets.MultiChoice(
            name="mirror symmetry axes", value=[], options=["x", "y", "z"],
            width=220,
            description="Reflecting-boundary symmetry planes. Each axis "
            "listed means the stored grid is HALF the domain (mirror plane "
            "at the low face) — the solver halves the work per axis and the "
            "field unfolds to the full instrument. Declare an axis ONLY when "
            "the geometry is truly symmetric about that plane and the "
            "electrodes sit against it. Changing this re-solves.")
        self.w_mirror.param.watch(self._on_mirror_change, "value")
        # INITIALISE from the current spec so a shipped mirror (SLIM's 'y')
        # is reflected in the control AT CONSTRUCTION — otherwise the empty
        # widget default gets written back by the first _sync_spec and
        # SILENTLY DROPS the mirror, changing the geometry and the cache key
        # (a dense load froze — build_needs_solve saw the 'y' key
        # while the sync'd solve used the ''-mirror key: a re-solve every
        # time, and a wrong half-domain geometry).
        _isc = getattr(self.spec, "scene", None)
        if _isc and isinstance(_isc.get("grid"), dict):
            self.w_mirror.value = [a for a in "xyz"
                                   if a in (_isc["grid"].get("mirror") or "")]
        else:
            _isym = getattr(self.spec.geometry, "symmetry", None)
            _ipl = getattr(_isym, "planes", {}) if _isym else {}
            self.w_mirror.value = [a for a in "xyz"
                                   if (_ipl or {}).get(a) == "mirror"]
        self.w_sizing = pn.pane.Markdown("", sizing_mode="stretch_width")
        # FIELD-BUILD OPTIONS (GeometrySpec.field_method / channel_dtype).
        # Both change the composed field, so editing them re-composes (a
        # geometry change). Defaults match current behavior.
        self.w_fieldmethod = pn.widgets.Select(
            name="field method", width=220,
            options={"electrode-aware (accurate at surfaces)":
                     "electrode_aware",
                     "plain gradient (faster, off-surface)":
                     "plain_gradient"},
            value=getattr(self.spec.geometry, "field_method",
                          "electrode_aware"),
            description="How E is differenced from the potential. Electrode-"
            "aware corrects the field at metal surfaces (essential when ions "
            "ride close to electrodes, e.g. a surface-born TOF source). Plain "
            "gradient skips that pass — faster, and correct when ions stay "
            "off the surfaces (e.g. SLIM central-gap transport).")
        self.w_chandtype = pn.widgets.Select(
            name="channel precision", width=220,
            options={"float64 (full)": "float64",
                     "float32 (half memory)": "float32"},
            value=getattr(self.spec.geometry, "channel_dtype", "float64"),
            description="Stored precision of the drive-channel field stacks. "
            "float32 halves their memory (matters as grids grow); the "
            "gradient is always computed in float64, only storage changes.")
        self.w_fieldmethod.param.watch(self._on_fieldopt_change, "value")
        self.w_chandtype.param.watch(self._on_fieldopt_change, "value")
        self.w_apply_pitch = pn.widgets.Button(
            name="Apply resolution", button_type="primary")
        self.w_apply_pitch.on_click(self._on_apply_pitch)
        # Liveness probe: one click prints a heartbeat
        # line to the SERVER console (stdout), so a wedged browser tab can
        # be told apart from a wedged kernel. Touches no state.
        self.w_ping = pn.widgets.Button(
            name="ping → stdout", width=110,
            description="Print one line to the SERVER console. If the UI "
                        "feels wedged, a ping that appears means the "
                        "kernel is alive and the browser tab is the "
                        "problem; no line means the server is busy or "
                        "stuck. Touches no state and cannot disturb a "
                        "run.")
        self.w_ping.on_click(self._on_ping)
        # Auto memory heartbeat (bug A): daemon-thread RSS logger for
        # catching the long-session leak, toggled on demand.
        # Toggle takes no `description` in this Panel version (Button
        # does), so its tooltip rides alongside as a TooltipIcon rather
        # than being dropped.
        self.w_autolog = pn.widgets.Toggle(name="mem autolog", width=110)
        self.w_autolog_tip = pn.widgets.TooltipIcon(
            value="mem autolog: log this process's memory (RSS) on a "
                  "heartbeat to the server console, for catching growth "
                  "over a long session. Diagnostic only — it records "
                  "usage, it does not limit it (that is the record RAM "
                  "budget in Config).")
        self.w_autolog.param.watch(self._on_autolog, "value")
        self.w_pitch.param.watch(self._refresh_sizing, "value")
        self.w_minfeat.param.watch(self._refresh_sizing, "value")
        self._json_guard = False
        self.w_json.param.watch(self._on_json_edited, "value")

        # AUTOSCALE ON VIEW CHANGE. plotly keeps pan/zoom across redraws when
        # `uirevision` is unchanged -- which is what you want while iterating
        # inside one view, and exactly what you DON'T want when switching to a
        # different plane: you arrive at the new view still zoomed into a box
        # that meant something in the old one, with no idea where you are.
        # Bumping a counter on every view change makes uirevision differ, so
        # plotly discards the stale zoom and the explicitly-set domain ranges
        # take effect: the new view opens FIT TO THE GEOMETRY.
        self._view_rev = 0

        # AUTOSCALE ON VIEW CHANGE — plane-keyed, watcher-order-INDEPENDENT.
        # Earlier this bumped a counter via a watcher registered AFTER the
        # redraw watcher, so on a plane switch the redraw ran with the OLD
        # revision: uirevision was unchanged and plotly kept the stale zoom
        # (changing the view always autoscales). Instead the
        # revision is derived from the plane itself at draw time, below
        # (_uirev), so a new plane ALWAYS yields a new uirevision and the
        # explicit geometry range takes effect — no reliance on watcher
        # firing order. w_lock/w_fieldmode still bump the counter for their
        # own (non-plane) view changes.
        def _bump_view(_e):
            self._view_rev += 1
        # w_plane changes are handled in _uirev (draw-time, order-safe);
        # lock/field-mode are non-plane view changes that still need a bump.
        for _w in (self.w_lock, self.w_fieldmode):
            _w.param.watch(_bump_view, "value")
        self.w_cfgname = pn.widgets.TextInput(
            name="save as (filename)", value="sim_spec.json", width=260,
            description="Filename for the downloaded configuration. Include "
            "the .json extension (added if omitted).")
        self.w_download = pn.widgets.FileDownload(
            filename="sim_spec.json", label="Download spec",
            callback=self._spec_bytes)
        # SAVE THE INSTRUMENT (changed parameters need a way
        # to be saved). Downloads the LOADED
        # assembly document with every stage's spec re-serialized from
        # the LIVE _assembly_specs objects — the same objects stage
        # edits and the pinned Ion Source tab write into — so what
        # downloads is what would fly. Wrapper facts (name, notes,
        # poses, exits, beam) are preserved from the loaded doc.
        # Geometry stays INLINE by design: one file is
        # the instrument.
        # disabled is DERIVED from current state, because this widget is
        # (re)created on EVERY control rebuild — a load-time enable was
        # clobbered by the next rebuild and the button stayed grey with
        # an instrument loaded.
        _has_asm = bool(getattr(self, "_assembly_doc", None))
        _asm_nm = ((getattr(self, "_assembly_doc", None) or {}).get("name")
                   or "instrument").strip() or "instrument"
        # USER-EDITABLE EXPORT NAME. The field is the
        # ONE authority for the download filename: loading an
        # instrument writes the doc's own name into it (visible
        # provenance — displayed equals actual), and any user edit wins
        # from then on. PERSISTENT so the choice survives control
        # rebuilds; the button below is deliberately recreated per
        # rebuild (state-derived enable), so it re-reads the field.
        self._persistent(
            "w_instr_name", lambda: pn.widgets.TextInput(
                name="instrument export filename", value=_asm_nm,
                placeholder="instrument", width=260,
                description="Filename for Download instrument; .json is "
                "appended if missing. Loading an instrument resets this "
                "to the document's own name."),
            wire=lambda w: w.param.watch(self._sync_instr_filename,
                                         "value"))
        _fn = (self.w_instr_name.value or "instrument").strip() \
            or "instrument"
        if not _fn.lower().endswith(".json"):
            _fn += ".json"
        # the FileDownload below is deliberately fresh each build
        # (state-derived enable); the PERSISTENT filename field must not
        # share a fresh container with it (re-parent -> freeze), so the
        # pair lives in a once-built column and only the button is
        # swapped in place.
        self.w_instrument_dl = pn.widgets.FileDownload(
            filename=_fn, label="Download instrument",
            callback=self._instrument_bytes, disabled=not _has_asm)
        if getattr(self, "_instr_dl_col", None) is None:
            self._instr_dl_col = pn.Column(self.w_instr_name,
                                           self.w_instrument_dl)
        else:
            self._instr_dl_col[1] = self.w_instrument_dl

        def _sync_cfgname(evt):
            nm = (evt.new or "sim_spec").strip()
            if not nm.lower().endswith(".json"):
                nm += ".json"
            self.w_download.filename = nm
        self.w_cfgname.param.watch(_sync_cfgname, "value")
        # ONE LOAD DOOR, and it carries the whole deck. A spec and the
        # STL files it names are ONE unit, so they are selected together
        # HERE rather than through a second path/browse control. The
        # browser hands over BYTES and hides the folder they came from,
        # so a relative stl_dir has nothing to anchor against; the answer
        # is to take the meshes too (io.stl_resolve.install_stl_payload),
        # never to ask the user where the file lives.
        self.w_upload = pn.widgets.FileInput(accept=".json,.stl",
                                             multiple=True)
        self.w_upload.param.watch(self._on_upload, "value")
        # Load results and refusals render HERE, beside the widget the
        # user is looking at. The status pane lives on another tab, and a
        # refusal written only there reads as "nothing happened" (L-422
        # lesson, re-learned on this very door 2026-09-11).
        self.w_load_msg = pn.pane.Markdown("", sizing_mode="stretch_width")
        # reloadable runs
        self.w_runsel = pn.widgets.Select(name="stored runs", options=[])
        self.w_reload = pn.widgets.Button(name="Reload run into view")
        self.w_reload.on_click(self._on_reload_run)
        self.w_export = pn.widgets.FileDownload(
            filename="trajectories.csv", label="Export displayed run",
            callback=self._export_bytes)
        self.w_export_npz = pn.widgets.FileDownload(
            filename="trajectories.npz", label="Download run (NPZ)",
            callback=self._export_npz_bytes)
        self.w_export_html = pn.widgets.FileDownload(
            filename="run_view.html", label="Export view + run (HTML)",
            callback=self._export_view_html,
            description="Self-contained HTML of the CURRENT view (field / PE "
            "surface + ion trajectories) with the full run metadata "
            "(geometry, voltages, gas, integration, fates) embedded.")
        # ---------------- solved-field / trajectory persistence ----------
        # SPEC_field_portability.md.  Named self-validating npz per solved
        # geometry; picker is MATCH-ANNOTATED (the app matches for the
        # user); field load HARD-REFUSES on geometry mismatch; trajectory
        # load overlays with a warning.  Labels are LOCAL (registry).
        self.w_savefield = pn.widgets.Button(
            name="Save solved field", button_type="default",
            description="Save the current geometry's solved bases as one "
            "named .npz in the fields folder (self-validating; safe to "
            "rename or send).")
        self.w_savefield.on_click(self._on_save_field)
        self.w_fieldpick = pn.widgets.Select(name="saved fields",
                                             options=[], size=1)
        self.w_loadfield = pn.widgets.Button(
            name="Load field", description="Validate the selected field "
            "against the LOADED geometry and reinstate it into the cache. "
            "Refuses on any mismatch — a wrong-geometry field is silently "
            "wrong physics.")
        self.w_loadfield.on_click(self._on_load_field)
        self.w_bootfield = pn.widgets.Button(
            name="Open saved field", description="One-file session open "
            "(PI 2026-08-06): reconstruct the geometry + operating "
            "point from the file's embedded spec, load it into the "
            "editor as if you had pasted the JSON, then validate and "
            "reinstate the bases. Refuses pre-bootstrap saves (no "
            "embedded spec) with the reason.")
        self.w_bootfield.on_click(self._on_bootstrap_field)
        self.w_fieldscan = pn.widgets.Button(name="↻", width=40,
                                             description="Rescan the "
                                             "fields folder.")
        self.w_fieldscan.on_click(lambda e: self._refresh_field_picker())
        self.w_fieldlabel = pn.widgets.TextInput(
            name="label", width=200, placeholder="display label…",
            description="Local display label for the selected field "
            "(stored in the registry; never rewrites the npz).")
        self.w_relabel = pn.widgets.Button(name="Relabel",
                                           description="Apply the label "
                                           "to the selected field "
                                           "(local registry only).")
        self.w_relabel.on_click(self._on_relabel_field)
        self.w_savetraj = pn.widgets.Button(
            name="Save trajectories (reloadable here)",
            description="Save the ACTIVE run's ion paths to this app's "
            "store as a named .traj.npz; reload it with 'Load as run' "
            "or open it in notebook 05. Stays server-side (see the save "
            "path in the Status tab). For a file to take elsewhere, use "
            "'Download run (NPZ)' above.")
        self.w_savetraj.on_click(self._on_save_traj)
        self.w_trajpick = pn.widgets.Select(name="saved trajectories",
                                            options=[], size=1)
        self.w_loadtraj = pn.widgets.Button(
            name="Load as run", description="Load the selected trajectory "
            "file as a stored run (overlays even across geometries, with "
            "a warning).")
        self.w_loadtraj.on_click(self._on_load_traj)
        self._refresh_field_picker()
        self.w_name = pn.widgets.TextInput(name="name",
                                           value=self.spec.name)
        self.w_notes = pn.widgets.TextAreaInput(
            name="notes", height=120, value=self.spec.notes,
            placeholder="Free-text notes saved with the spec "
            "(design intent, tune settings, provenance)...")
        # ---------------- Config tab -------------------------------------
        # Was one flat column: name/notes, examples, JSON+save/load, cost,
        # runs -- five sections deep, with the COST READOUT AT THE BOTTOM,
        # below the JSON box, while the examples sat at the top. So you
        # committed to a geometry and then scrolled down to learn what it
        # would cost. And "load an example" sat visually next to "save as",
        # which made the two read as one control.
        #
        # Now: name + notes PINNED at the top (they describe the configuration
        # you currently have), and three subtabs for the three things you
        # actually do -- Load, Save, Runs. The cost estimator lives WITH the
        # loaders, because it is a decision aid for loading, not a postscript.
        cost_card = pn.Card(
            pn.Row(self.w_pitch, self.w_minfeat, self.w_apply_pitch),
            self.w_mirror,
            pn.Row(self.w_fieldmethod, self.w_chandtype),
            self.w_sizing,
            title="solve resolution & cost — for the spec staged below",
            collapsed=False, sizing_mode="stretch_width")

        load_tab = pn.Column(
            pn.pane.Markdown(
                "### 1 · start from a built-in example\n"
                "*Selecting one STAGES it in the JSON box below and prices it. "
                "It is not loaded until you press the button.*"),
            pn.Row(self.w_examples, pn.Column(pn.Spacer(height=18),
                                              self.w_loadex)),
            cost_card,
            pn.layout.Divider(),
            # "1b · staged multi-stage instrument" pointer REMOVED entirely
            # The assembly widgets live on the Multi FA
            # tab and are never duplicated here (one widget in two
            # containers is the two-parent Bokeh defect).
            pn.pane.Markdown(
                "### 2 · or upload a spec file / paste JSON\n"
                "*`Apply JSON` commits whatever is in the box. To keep a spec "
                "for later, use the **Save** tab — an example is something you "
                "load, not something you write to.*\n\n"
                "*An **STL deck** is its .json plus its .stl meshes: select "
                "them **together** (Cmd/Ctrl-click) the first time. After "
                "that, the .json alone reloads it — the meshes are kept and "
                "re-verified by hash.*"),
            self.w_upload,
            self.w_load_msg,
            self.w_json,
            pn.Row(self.w_applyjson, self.w_apply_busy),
            pn.pane.Markdown("##### spec summary *(live)*"),
            self.w_spec_summary,
            name="Load", sizing_mode="stretch_width")

        # BUILD-ONCE (the freeze chain must be unbroken to the root): the
        # persistent _instr_dl_col re-parented whenever this tab column
        # was rebuilt fresh around it. The column is now built once; its
        # deliberately-fresh members (name field, spec download) are
        # swapped IN PLACE each build — a fresh widget in a stable
        # container renders; a persistent widget in a fresh container
        # orphans.
        if getattr(self, "_save_col", None) is None:
            self._save_col = pn.Column(
                pn.pane.Markdown(
                    "#### save this configuration\n"
                    "*Writes the spec **currently loaded** (name, notes, "
                    "geometry, voltages, source, gas, integration) to a "
                    "JSON file. This is not related to the built-in "
                    "examples on the Load tab. STL decks loaded through "
                    "the upload door save with `stl_dir: \".\"` — keep "
                    "the JSON with its .stl files, or reload the JSON "
                    "alone and the app finds the installed meshes by "
                    "manifest.*"),
                self.w_cfgname,
                self.w_download,
                pn.pane.Markdown(
                    "*With an instrument loaded, **Download instrument** "
                    "writes the whole assembly — every stage's CURRENT "
                    "parameters (voltage, source and gas edits included) "
                    "inline in one file, ready to reload or share.*"),
                self._instr_dl_col,
                name="Save", sizing_mode="stretch_width")
        else:
            self._save_col[1] = self.w_cfgname
            self._save_col[2] = self.w_download
        save_tab = self._save_col

        runs_tab = pn.Column(
            pn.pane.Markdown("#### stored runs (reloadable)"),
            self.w_runsel,
            pn.Row(self.w_reload, self.w_export),
            pn.Row(self.w_export_npz, self.w_export_html),
            pn.layout.Divider(),
            pn.pane.Markdown(
                "#### solved fields & trajectories\n*Save a solved field "
                "as one named, self-validating .npz (rename/send freely); "
                "loading validates it against the LOADED geometry and "
                "refuses on mismatch. Trajectories overlay across "
                "geometries with a warning.*"),
            pn.Row(self.w_savefield, self.w_savetraj),
            pn.Row(self.w_fieldpick, self.w_fieldscan, self.w_loadfield,
                   self.w_bootfield),
            pn.Row(self.w_fieldlabel, self.w_relabel),
            pn.Row(self.w_trajpick, self.w_loadtraj),
            pn.pane.Markdown(
                "#### maintenance\n*Free memory / force a fresh solve. Each "
                "clear reports what it freed. (**Clear ions** is on the top "
                "bar.)*"),
            pn.Row(self.w_clearruns, self.w_clearcache),
            name="Runs", sizing_mode="stretch_width")

        # Named Columns, not (name, obj) tuples: pn.Tabs stores tuple names in
        # a private `_names`, so the tab's own .name stays a generated id like
        # "Column02419" and nothing downstream can read it back.
        # BUILD-ONCE Tabs (last link of the freeze chain): a fresh
        # pn.Tabs here re-parented the persistent _save_col every
        # rebuild. The Tabs object persists; children are reassigned in
        # place — fresh tabs re-render, the persistent one keeps its
        # parent.
        if getattr(self, "w_cfgtabs", None) is None:
            self.w_cfgtabs = pn.Tabs(load_tab, save_tab, runs_tab,
                                     dynamic=False)
        else:
            self.w_cfgtabs[:] = [load_tab, save_tab, runs_tab]
        # BUILD-ONCE (final link of the freeze chain to the root): the
        # persistent w_cfgtabs re-parented while this wrapper was fresh.
        # Direct children of self.tabs are stable (the Ion Source column
        # proves it), so the chain ends here: build the Config column
        # once, swap its fresh members (name/notes) in place.
        self._ensure_machine_widgets()
        _cfg_head = [pn.pane.Markdown("### current configuration"),
                     self.w_name, self.w_notes]
        if getattr(self, "_config_col", None) is not None:
            for _i, _w in enumerate(_cfg_head):
                self._config_col[_i] = _w
            config_tab = self._config_col
        else:
            config_tab = self._config_col = pn.Column(
            *_cfg_head,
            pn.layout.Divider(),
            # MACHINE RESOURCES (PI 2026-09-13): properties of this
            # installation, not of the deck — they do not travel with a
            # spec and do not change the physics. Same widget OBJECTS as
            # before the move, so every reader (_on_start's n_workers,
            # the record-volume quote) is unchanged.
            pn.Card(pn.Row(self.w_workers, self.w_ram_budget),
                    title="Compute Resources",
                    collapsed=False, sizing_mode="stretch_width"),
            pn.layout.Divider(),
            self.w_cfgtabs,
            # Version note: read from the ONE authority,
            # ion_gym.__version__ -- never a literal here, so the tab can
            # never disagree with the installed package or the zip name.
            # Placed in Config because that is where examples are loaded,
            # and version-coupled examples (e.g. the refined oa-TOF, which
            # needs >= v329 birth semantics) fail HERE first.
            pn.pane.Markdown(
                f"<span style='color:#888'>ion_gym v{ion_gym.__version__}"
                f" · examples: {__import__('ion_gym.io.paths', fromlist=['paths']).repo_root() / 'examples'}"
                "</span>"),
            sizing_mode="stretch_width")

        # STL upload tab: persistent across reloads (rebuilding it would
        # drop staged files); commit follows the _rebuild_for_new_spec
        # discipline — geometry shows immediately, solve on Recompute/Fly.
        if getattr(self, "_stl_tab", None) is None:
            from ion_gym.io.stl_upload import StlUploadPanel
            def _commit(spec):
                # Preserve the user's CURRENT ion-source / gas / integration
                # settings — a new geometry must not reset them to defaults
                # (the user explicitly does not want to fall back to the
                # funnel/default source on every STL import).
                spec.source = self.spec.source
                spec.collisions = self.spec.collisions
                spec.integration = self.spec.integration
                self._pre_stl_spec = self.spec        # for restore on clear
                self.spec = spec
                self._rebuild_for_new_spec(solve=False)
                # FLAG the dimensionality assumption rather than hiding it:
                # STL import currently builds a 2-D x-y cross-section
                # (planar slice); ions coast along z. Exact only for
                # z-invariant geometry (straight multipoles); z-varying
                # structure is NOT represented. Full 3-D STL solve is a
                # roadmap item.
                if build_route(spec).field_dims == 3:
                    self.status.object = (
                        "**STL committed for FULL 3-D solve** (voxelize + "
                        "solver3d + fly3d). Ion-source settings preserved. "
                        "Press Recompute / Fly — the 3-D solve takes "
                        "longer; watch the cost estimate on the STL tab.")
                else:
                    self.status.object = (
                        "**STL committed as a 2-D x-y cross-section** "
                        "(planar slice; ions coast along z). Exact for "
                        "z-invariant geometry (straight multipoles); "
                        "z-variation is NOT modelled — tick 'solve full "
                        "3-D' on the STL tab for the true 3-D solve. "
                        "Ion-source settings preserved.")

            def _clear_stl():
                # Restore the pre-STL spec so the app returns to a working
                # geometry, and drop any active run.
                if getattr(self, "_pre_stl_spec", None) is not None:
                    self.spec = self._pre_stl_spec
                    self._pre_stl_spec = None
                    self._rebuild_for_new_spec(solve=False)
                self._runs.clear()
                self._active = None
                self.status.object = ("**cleared** — STLs + cache purged; "
                                      "reverted to the previous geometry")

            self._stl_tab = StlUploadPanel(on_commit=_commit,
                                           on_clear=_clear_stl)

        if getattr(self, "_pe_tab", None) is None:
            from ion_gym.viz.pe_view import PeSurfaceTab
            self._pe_tab = PeSurfaceTab(
                get_model=lambda: getattr(self, "_model", None),
                get_spec=lambda: self.spec,
                get_results=lambda: (self._runs[self._active].results
                                     if self._active in self._runs else None),
                get_run_id=lambda: self._active)
            self._pe_tab.w_mz.value = float(self.spec.source.mz_list[0])

        # Field Slice tab: a flat sliced heatmap+contour view of |E| / φ /
        # PE on any plane, with colormap/log/range and contour controls
        # Compute-on-press like the PE tab.
        if getattr(self, "_fs_tab", None) is None:
            from ion_gym.viz.pe_view import FieldSliceTab
            self._fs_tab = FieldSliceTab(
                get_model=lambda: getattr(self, "_model", None),
                get_spec=lambda: self.spec)
            self._field_slice_tab = self._fs_tab.panel()

        # Stats card lives in the first tab; created once (persistent
        # across in-place tab rebuilds), updated on every fly.
        if getattr(self, "stats", None) is None:
            from ion_gym.physics.stats import stats_card
            self.stats = stats_card(collapsed=False)

        # MULTI-FA TAB. Every multi-FA control lives
        # here, in one place, instead of being scattered across the top
        # row and the Config tab. The widgets are the SAME objects
        # (`_persistent`) referenced from one container -- Panel
        # renders a widget in exactly one place, so listing them here
        # MOVES them and there is never a second copy to disagree with
        # the first.
        # THE MULTI FA TAB IS BUILT ONCE AND REUSED (root
        # cause, pinned by a console capture): every
        # stage/view change routes through _rebuild_for_new_spec ->
        # _build_controls -> `self.tabs[:] = new_tabs`, and rebuilding
        # this Column fresh each time re-parented the PERSISTENT
        # multi-FA widgets (old Column still holding them while the new
        # one claimed them) -- the two-parent condition, fired
        # dynamically on every swap. The browser trace shows it exactly:
        # _apply_json_patch -> update_children -> a fresh widget render;
        # the on-screen Select is then an orphaned DOM copy, dead until
        # a tab switch re-renders. Every child here is a persistent
        # widget or a static caption, so there is NOTHING spec-dependent
        # to rebuild: one construction is the correct lifetime, and
        # reusing the same Column keeps the Tabs diff from ever touching
        # these widgets again.
        if getattr(self, "_multifa_tab", None) is None:
            # LAYOUT:
            # Choose File + loaded-filename string; FA View dropdown
            # (Full Assembly default) + ONE *Set FA View* button;
            # Multi FA Flight Parameters selector + *Set Fly
            # Parameters*; solve target; info. ONE Fly button lives in
            # the top row and flies what is displayed.
            self._multifa_tab = pn.Column(
                pn.pane.Markdown("### Multi-FA instrument"),
                self.w_instrument,
                self.w_loaded_name,
                pn.pane.Markdown("**FA View** — default: Full Assembly"),
                self.w_stage,
                self.w_set_fa_view,
                pn.pane.Markdown(
                    "**Multi FA Flight Parameters** — which FA's ion "
                    "parameters the Ion Source tab shows/edits. The "
                    "instrument's default source FA is its declared "
                    "`beam.from_stage`."),
                self.w_fly_src,
                self.w_set_fly_params,
                pn.pane.Markdown("**solve target** — which stage "
                                 "*Recompute* acts on"),
                self.w_solve_target,
                self.w_assembly_info,
                width=CONTROL_COL_PX - 20)
        multifa_tab = self._multifa_tab
        # BUILD-ONCE WRAPPERS (the other half of the
        # load-time freeze): a fresh
        # container around a PERSISTENT object re-parents it on every
        # spec load (the two-parent transient). Two such objects
        # lived in the control tabs: stats.card (Ion Source) and the
        # STL panel's widgets (Geometry Import — StlUploadPanel.panel()
        # builds a fresh Column around persistent widgets each call).
        # Their WRAPPERS are now built once; spec-dependent content
        # (the Source column) is swapped INSIDE the persistent wrapper,
        # so the persistent halves are never re-parented again.
        # G1(a): immediate write-through — wired IDEMPOTENTLY, per
        # widget. The old per-build loop stacked a duplicate doc-write
        # watcher on every PERSISTENT widget each rebuild (Advanced /
        # Stations hold build-once columns), and a wire-once guard
        # misses the spec-dependent widgets those columns swap in on a
        # deck load. The rule that is true in both directions: a widget
        # carries this watcher exactly once, checked on the widget
        # itself.
        def _wire_spec_widget(w):
            for _lst in w.param.watchers.get("value", {}).values():
                for _ws in _lst:
                    if getattr(_ws, "fn", None) == \
                            self._on_spec_widget_change:
                        return
            w.param.watch(self._on_spec_widget_change, "value")
        for _cont in (ionsrc_header, ionsrc_basic, ionsrc_adv,
                      ionsrc_stations, gas_tab, integ_tab, bounds_tab):
            for _wdg in self._walk_value_widgets(_cont):
                _wire_spec_widget(_wdg)
        _ionsrc_children = [("Basic", ionsrc_basic),
                            ("Advanced", ionsrc_adv),
                            ("Stations", ionsrc_stations),
                            ("Stats", self.stats.card)]
        if getattr(self, "_ionsrc_tabs", None) is None:
            self._ionsrc_tabs = pn.Tabs(*_ionsrc_children)
            self._ionsrc_col = pn.Column(ionsrc_header,
                                         self._ionsrc_tabs)
        else:
            self._ionsrc_tabs[:] = _ionsrc_children
            self._ionsrc_col[0] = ionsrc_header
        if getattr(self, "_geomimp_tabs", None) is None:
            self._stl_panel = self._stl_tab.panel()   # built ONCE
            self._geomimp_tabs = pn.Tabs(("STL Upload", self._stl_panel))
        new_tabs = [
            ("Ion Source", self._ionsrc_col),
            ("Voltages", volt_tab),
            ("Physics", pn.Tabs(
                ("Gas", gas_tab), ("Integration", integ_tab),
                ("Bounds", bounds_tab))),
            ("Display", disp_tab),
            ("Impact Analysis", self._impact_tab()),
            # PE Surface lives ONLY in plot_tabs (the view side).
            # Restoring it HERE (when
            # closing test_pe_surface_tab) put one panel in two containers:
            # a two-parent Bokeh model — the browser bound this copy and
            # the view-side copy went dead, shedding dropped-patch
            # warnings. The test now asserts the plot_tabs home.
            # STL upload STAYS (a mesh is a shape); the geometry-import tab
            # went with the reader stack.
            ("Geometry Import", self._geomimp_tabs),
            # Status feedback in its own left tab,
            # moved out of the top bar. Placed here so the standing tab
            # order holds: Multi FA second-to-last,
            # Config last.
            ("Status", self._status_tab()),
            # SECOND-TO-LAST by convention.
            ("Multi FA", multifa_tab),
            ("Config", config_tab)]

        if getattr(self, "tabs", None) is None:
            # first build: create the persistent container
            self.tabs = pn.Tabs(*new_tabs, width=CONTROL_COL_PX,
                                sizing_mode="fixed")
        else:
            # reload: repopulate IN PLACE so the layout already handed to
            # the notebook shows the new controls (the bug was rebuilding
            # a NEW Tabs object the displayed layout never saw)
            active = self.tabs.active
            self.tabs[:] = new_tabs
            self.tabs.active = min(active, len(new_tabs) - 1)
        # CONTAINMENT BELONGS HERE, NOT IN panel() (the
        # sidebar still crossed into the plot after
        # LOADING AN EXAMPLE).  panel() runs once, at construction; the line
        # above repopulates the same container on every spec load with FRESH
        # pn.Row objects, so a containment pass done at layout time is
        # discarded by the first reload.  Fitting at BUILD time covers the
        # first build and every rebuild by construction -- there is no second
        # place a tab can enter the column from.
        fit_to_column(self.tabs, CONTROL_COL_PX, label="control column")
        # initial summary population (safe if the pane exists)
        self._refresh_spec_summary()

    def _status_tab(self):
        """Status feedback in its OWN left tab (it
        previously sat in the top bar, height-capped so it would not
        cover the top-right buttons).

        This method OWNS `self.status`: the pane is created here, once,
        and `_build_run_controls` no longer creates it -- a second
        creation would orphan this tab's copy and every later
        `self.status.object = ...` write would land on a pane no layout
        shows. Built once and reused across `_build_controls` reloads
        (persistent-container rule).

        Because the tab is only visible when selected, the latest message
        alone would be lossy: anything written while another tab is
        active gets overwritten unseen. A watcher on `status.object`
        keeps a rolling session log (newest first, STATUS_LOG_MAX deep)
        so transient messages remain readable after the fact. Identical
        consecutive writes coalesce (param fires on change only) --
        acceptable: a repeat carries no new information.
        """
        if getattr(self, "_status_col", None) is not None:
            return self._status_col
        self.status = pn.pane.Markdown(
            "**idle**", sizing_mode="stretch_width",
            styles={"overflow-wrap": "anywhere"})
        from collections import deque
        self._status_log = deque(maxlen=STATUS_LOG_MAX)
        self._status_log_pane = pn.pane.Markdown(
            "_no messages yet this session_", sizing_mode="stretch_width",
            styles={"overflow-wrap": "anywhere"})
        def _log_status(evt):
            stamp = time.strftime("%H:%M:%S")
            self._status_log.appendleft(f"`{stamp}`\n\n{evt.new}")
            self._status_log_pane.object = "\n\n---\n\n".join(
                self._status_log)
        self.status.param.watch(_log_status, "object")
        self._ensure_machine_widgets()
        self.workers_line = pn.pane.Markdown(
            "_no flight this session_", sizing_mode="stretch_width")
        self._status_col = pn.Column(
            pn.pane.Markdown("#### current"),
            self.status,
            pn.layout.Divider(),
            pn.pane.Markdown("#### workers"),
            # The worker-count and record-RAM CONTROLS moved to the
            # Config tab (PI 2026-09-13): they describe the MACHINE, not
            # the instrument — they belong with the other
            # per-installation settings rather than being re-decided
            # beside every flight. The READOUT stays here, where the
            # flight it describes is reported.
            self.workers_line,
            pn.layout.Divider(),
            # SESSION DIAGNOSTICS (PI 2026-09-13). Moved here from the
            # run-controls row: neither touches a deck or a run, and
            # both answer "is this session healthy?", which is what the
            # Status tab is for.
            pn.pane.Markdown("#### diagnostics"),
            pn.Row(self.w_ping, self.w_autolog, self.w_autolog_tip),
            pn.layout.Divider(),
            pn.pane.Markdown(
                f"#### history *(this session, newest first, "
                f"last {STATUS_LOG_MAX})*"),
            self._status_log_pane,
            width=CONTROL_COL_PX - 20)
        return self._status_col

    def _ensure_machine_widgets(self):
        """Create the MACHINE-resource widgets (worker threads,
        record RAM) exactly once.

        They are displayed in the Config tab (PI 2026-09-13) but
        were historically constructed by the status column, which
        is built AFTER it — so Config referenced them before they
        existed. Construction lives here, idempotent through
        _persistent, and BOTH builders call it: whichever runs
        first creates them, the other finds them. One owner, no
        ordering dependency between two layout builders.
        """
        # WORKERS block (the worker
        # queue variables are exposed in their own tab). The count is a
        # THROUGHPUT choice, not physics, so it lives here and never in
        # a deck; it is read at LAUNCH (mid-flight edits apply to the
        # next Fly). The live line is written only by _tick from
        # EnsembleProgress telemetry.
        from ion_gym.physics.ensemble_driver import default_workers
        self._persistent(
            "w_workers", lambda: _mkw(
                pn.widgets.IntInput, name="worker threads",
                value=default_workers(), start=1,
                end=max(1, (os.cpu_count() or 2)), step=1, width=120,
                description=("Ion-flight threads for the next Fly. "
                             "Default leaves two cores for the app; "
                             "1 = the sequential legacy path. Results "
                             "are seed-identical at any count.")))
        # MEMORY BUDGET, beside the worker count because it is the same
        # KIND of choice: a throughput/resource decision about this
        # machine and this session, never physics, never a deck field.
        # A fraction of total RAM is a guess about what else is running;
        # on a workstation doing nothing else, 32 of 48 GB is a fine
        # answer and no fraction would ever produce it. Declared here,
        # it becomes the refusal ceiling and the warning sits at half.
        # 0 = fall back to the machine-derived fraction.
        from ion_gym.physics.sizing import (system_memory_gb,
                                            RECORD_REFUSE_FRACTION)
        _ram = system_memory_gb()
        _default_budget = round(_ram * RECORD_REFUSE_FRACTION) if _ram else 0
        self._persistent(
            "w_ram_budget", lambda: _mkw(
                pn.widgets.IntInput, name="max record RAM (GB)",
                value=int(_default_budget), start=0, step=1, width=140,
                description=(
                    "Memory the trajectory record may use. The Fly "
                    "refuses above this and warns at half of it, using "
                    "the quote shown in the sizing readout. 0 = derive "
                    "from system RAM ("
                    + (f"{_ram:.0f} GB detected" if _ram
                       else "undetectable on this platform")
                    + "). Storage only: rec_every changes what is kept, "
                      "never what is computed.")))

    def _build_run_controls(self):
        self.start_btn = pn.widgets.Button(name="Fly",
                                           button_type="success",
                                           width=90)
        self.stop_btn = pn.widgets.Button(name="Stop",
                                          button_type="danger", width=90)
        self.clear_btn = pn.widgets.Button(name="Clear ions",
                                           button_type="default", width=90)
        self.recompute_btn = pn.widgets.Button(name="Recompute field",
                                               button_type="default",
                                               width=130)
        self.reset_btn = pn.widgets.Button(name="Reset app",
                                           button_type="warning", width=90)
        self.view_xy = pn.widgets.Button(name="xy", width=50)
        self.view_xz = pn.widgets.Button(name="xz", width=50)
        self.view_yz = pn.widgets.Button(name="yz", width=50)
        self.view_xy.on_click(lambda e: self._set_plane("xy"))
        self.view_xz.on_click(lambda e: self._set_plane("xz"))
        self.view_yz.on_click(lambda e: self._set_plane("yz"))
        # self.status is created by _status_tab() (which _build_controls
        # calls before this method runs) -- it lives in the left "Status"
        # tab, not the top bar. Do NOT recreate it
        # here: a second pane would orphan the tab's copy and every later
        # write would go to a pane nothing displays. The relocation also
        # retires the old height cap -- in its own tab the pane may grow;
        # there are no top-bar neighbours to overlap.
        self.start_btn.on_click(self._on_start)
        self.stop_btn.on_click(self._on_stop)
        self.clear_btn.on_click(self._on_clear)
        self.recompute_btn.on_click(self._on_recompute)
        self.reset_btn.on_click(self._on_reset)
        self.pane = pn.pane.Plotly(
            height=VIEW_PX, sizing_mode="stretch_width", align="center",
            config={"scrollZoom": True, "displayModeBar": True})

    def _on_clear(self, _=None):
        """Clear the displayed trajectories/impacts back to just the field
        (stored runs are kept; use the Config tab to reload one)."""
        self._active = None
        # ASSEMBLY SUBJECT (Clear ions on the Full
        # Assembly view stomped it with the live stage's y-z background
        # — and there was no way to clear assembly trajectories at all).
        # Clearing clears the SUBJECT'S ions and keeps the subject: the
        # assembly stays the assembly, minus its traces and impacts.
        if self._subject_is_assembly():
            self._assembly_traces = None
            self._impact_hits = []
            self._redraw_subject()
            self.stats.clear()
            self.status.object = ("**cleared** — assembly trajectories "
                                  "and impacts removed (stored runs "
                                  "kept); geometry redrawn")
            return
        self._draw_background()
        self.stats.clear()
        self.status.object = "**cleared** — field only (stored runs kept)"

    def _on_recompute(self, _=None):
        """Force a fresh field solve/re-weight from the current controls
        and redraw (also picks up a geometry change).

        RECOMPUTE HAS A STAGE TARGET. The controls
        edit ONE spec, so a solve is meaningful only against one stage.
        With the whole assembly as the subject there is no such spec, and
        silently solving whichever stage happened to be live last would
        report "field recomputed" for a stage the user is not looking at
        -- a wrong answer wearing a success message. It declines and says
        which selection would make the button meaningful.
        """
        _tgt = getattr(getattr(self, "w_solve_target", None), "value", None)
        if self._assembly_specs and _tgt and _tgt != "(displayed stage)":
            # An EXPLICIT target: solve that stage even if another is
            # displayed. This is what makes Recompute meaningful under the
            # whole-assembly subject, where there is no single live spec --
            # it could previously only decline. Showing what is being
            # solved is deliberate: a solve reported for a stage the user
            # cannot see is the ambiguity this selector exists to remove.
            if _tgt != getattr(self, "_assembly_stage", None):
                self.w_stage.value = _tgt
        elif self._subject_is_assembly():
            self.status.object = (
                "**Recompute needs one stage.** Pick one in *solve*, or "
                "select a stage as the subject.")
            return
        self._sync_spec()
        # a geometry change invalidates cached bases only if the geometry
        # key differs; the builders handle that. The completion message is
        # set by the done callback AFTER the (possibly async) build — never
        # before, or a slow solve reads as "complete" while still running.
        _tgt = (f" (stage **{self._assembly_stage}**)"
                if self._assembly_specs else "")
        self._draw_background(
            done_msg=f"**field recomputed** from current settings{_tgt}")

    def _on_reset(self, _=None):
        """Reset the app to the spec it was created with; drop stored
        runs and displayed ions."""
        self._fly_chip("idle")
        if self._handle is not None and not self._handle.done:
            self._handle.stop("reset")
        # Reset ALWAYS recovers the run machinery: a wedged or dead
        # handle cannot survive it.
        self._handle = None
        self._runs.clear()
        self._active = None
        # Reset ADOPTS A DIFFERENT MODEL, so it drops the instrument too
        # (stale-state class; found by audit alongside the
        # example-button fix). Without this, instrument -> Reset leaves
        # _assembly_specs populated and _assembly_stage on WHOLE_ASSEMBLY
        # while the displayed model is the initial single spec — the
        # stage selector then offers stages the model does not have.
        # Unconditional here, unlike the example path: from_json on the
        # spec the app was constructed with cannot fail in a way that
        # leaves a working instrument worth keeping.
        self._clear_assembly_state()
        self.spec = SimSpec.from_json(self._initial_spec_json)
        self._rebuild_for_new_spec()
        self.w_runsel.options = []
        self.status.object = "**reset** to initial spec"

    # ------------------------------------------------------ layout
    def panel(self):
        # NOTE: a pn.state.onload hook does NOT work here as a
        # per-session trigger. panel() is evaluated ONCE, at serve time,
        # for the single shared app -- there is no session context to
        # defer into, so the callback runs immediately. The server starts
        # telemetry explicitly instead (ui.serve.serve_dashboard).
        # ping sits TOP RIGHT: the liveness probe must
        # stay reachable when the rest of the UI is wedged, so it gets the
        # corner, after a stretching spacer.
        # SINGLE-FA CONTROLS ONLY. The multi-FA widgets live in the
        # Multi FA tab and NOWHERE ELSE: listing a widget in two
        # containers gives the Bokeh model two parents, the browser binds
        # one and the OTHER GOES DEAD -- which is exactly what shipped in
        # v408 (every assembly button dead in the browser, while headless
        # tests, which drive widgets programmatically rather than through
        # the DOM, kept passing). The PE-Surface comment below documents
        # this same defect class before; the lesson is now a
        # rule: ONE widget, ONE container, no exceptions.
        # geometry-editor link: opens /editor in a
        # NEW BROWSER TAB.  A plain anchor, built HERE and owned by this
        # row only (ONE widget, ONE container); coupling to the
        # editor is file-mediated, so nothing else in the app changes.
        editor_link = pn.pane.HTML(
            '<a href="/editor" target="_blank" title="Open the geometry '
            'editor in a new browser tab (served at /editor by '
            'ion-gym dashboard)" style="display:inline-block;'
            'padding:5px 10px;border:1px solid #888;border-radius:4px;'
            'text-decoration:none;color:inherit;white-space:nowrap">'
            'Geometry editor &#8599;</a>', margin=(4, 6))
        # flight-viewer link: same pattern —
        # a plain anchor to /flight in a NEW TAB; coupling is the
        # server-global last-flight slot, no dashboard state touched.
        flight_link = pn.pane.HTML(
            '<a href="/flight" target="_blank" title="Open the 3D Flight Viewer '
            'in a new browser tab (last flown '
            'trajectories; served at /flight by ion-gym dashboard)" '
            'style="display:inline-block;'
            'padding:5px 10px;border:1px solid #888;border-radius:4px;'
            'text-decoration:none;color:inherit;white-space:nowrap">'
            '3D Flight Viewer &#8599;</a>', margin=(4, 6))
        # status is NOT here: it has its own left tab.
        top = pn.Row(self.start_btn,
                     self.stop_btn, self.clear_btn,
                     self.recompute_btn, self.reset_btn,
                     self.fly_chip,
                     pn.layout.HSpacer(), editor_link, flight_link,
                     # ping / mem-autolog moved to the Status tab (PI
                     # 2026-09-13): they are session DIAGNOSTICS, and
                     # they belong with the health readouts rather than
                     # beside the run controls.
                     sizing_mode="stretch_width")
        views = pn.Row(pn.pane.Markdown("**view:**", width=45),
                       self.view_xy, self.view_xz, self.view_yz)
        # Left: controls (fixed width). Right: ALL plots share one footprint
        # via a Tabs container — the main field/trajectory view (with its
        # xy/xz/yz plane buttons), the Analysis plot, and the PE-surface
        # plot. Only the active plot renders, so they never fight for space.
        #
        # THE COLUMN IS A HARD EDGE, not a suggestion (the
        # sidebar could bleed into the plot). Three things make it one, and all three
        # are needed -- the earlier passes each supplied only one:
        #   1. fit_to_column at BUILD time (see _build_controls) wraps rows
        #      that are wider than the column, so the overflow is small;
        #   2. self.tabs carries the width itself, so long labels and
        #      Markdown WRAP at the boundary instead of setting their own
        #      natural width (a fixed-width Column does not constrain a
        #      child that declares no width -- it just leaves it sticking
        #      out, which is the "Per-electrode ..." line crossing the
        #      divider);
        #   3. overflow-x here, so anything that still does not fit is
        #      contained by the browser rather than painted over the figure.
        # overflow-x alone would CLIP a control (the original cut-off duty
        # box, restaged), which is why 1 and 2 come first: by the time the
        # browser sees this box there is nothing left to clip.
        left = pn.Column(self.tabs, width=CONTROL_COL_PX,
                         styles={"overflow-x": "auto"})
        # VISIBLE SEPARATOR (there needs to be a UI
        # separator between left and right). A rule plus a gutter on
        # each side: the rule says where the controls end, the gutter stops
        # a control and an axis label sitting flush against each other and
        # reading as one cluttered surface. pn.layout.Divider is horizontal,
        # so the vertical rule is a Spacer with a background.
        divider = pn.Spacer(width=1, sizing_mode="stretch_height",
                            margin=(0, GUTTER_PX),
                            styles={"background": SEPARATOR_COLOR})
        main_view = pn.Column(views, self.pane, sizing_mode="stretch_width")
        if getattr(self, "_raster_tab", None) is None:
            self._raster_btn = pn.widgets.Button(
                name="Compute 3D View", button_type="primary",
                width=200,
                description="Render the labeled voxel grid the SOLVER "
                "consumes (route-agnostic: scene CSG, STL voxelization). "
                "Multi-axis: 3-D cloud + exact xy/xz/yz projections.")
            self._raster_btn.on_click(self._on_raster)
            self.w_raster_electrodes = pn.widgets.Checkbox(
                name="show electrodes", value=True)
            self.w_raster_legend = pn.widgets.Checkbox(
                name="show legend", value=False)
            self.w_raster_legend.param.watch(
                lambda e: self._apply_raster_visibility(), "value")
            self.w_raster_electrodes.param.watch(
                lambda e: self._apply_raster_visibility(), "value")
            self._raster_panes = pn.Tabs(sizing_mode="stretch_width")
            self._raster_status = pn.pane.Markdown(
                "_the solver's voxel occupancy — press Compute after "
                "loading a 3-D geometry (scene or STL)_")
            self._raster_tab = pn.Column(
                pn.Row(self._raster_btn, self.w_raster_electrodes,
                       self.w_raster_legend),
                self._raster_status,
                self._raster_panes, sizing_mode="stretch_width")
        if getattr(self, "_cache_tab", None) is None:
            self._build_cache_tab()
        self.plot_tabs = pn.Tabs(
            ("View", main_view),
            ("Analysis", self._analysis_tab),
            ("Thermal", self._thermal_tab),
            ("PE Surface", self._pe_tab.panel()),
            ("Field Slice", self._field_slice_tab),
            ("Raster", self._raster_tab),
            ("Cache", self._cache_tab),
            sizing_mode="stretch_width")
        body = pn.Row(left, divider, self.plot_tabs,
                      sizing_mode="stretch_width")
        return pn.Column(top, body, sizing_mode="stretch_width")

    # ------------------------------------------------- spec assembly
    def _write_mirror(self, axes):
        """Write mirror axes to whichever location THIS spec's solver reads
        (matching build_stl3d._declared_mirror_axes): a scene-bearing spec
        uses scene.grid.mirror; an STL/CAD spec (e.g. Q3) uses
        geometry.symmetry.planes. Returns the string form written, or None
        if the spec has no place for it."""
        sc = getattr(self.spec, "scene", None)
        if sc and isinstance(sc.get("grid"), dict):
            m = "".join(axes)
            sc["grid"]["mirror"] = m
            return m
        sym = getattr(self.spec.geometry, "symmetry", None)
        if sym is not None and hasattr(sym, "planes"):
            planes = dict(getattr(sym, "planes", {}) or {})
            for a in ("x", "y", "z"):
                planes[a] = "mirror" if a in axes else "none"
            sym.planes = planes
            return "".join(a for a in "xyz" if a in axes)
        return None

    def _on_fieldopt_change(self, _=None):
        """Write the field-build options onto the geometry. Both change the
        composed field, so the next build re-composes — announced. Guarded
        against firing during construction (before self.status)."""
        if not hasattr(self, "status"):
            return
        g = self.spec.geometry
        g.field_method = self.w_fieldmethod.value
        g.channel_dtype = self.w_chandtype.value
        # the JSON box must never go stale w.r.t. these options: a later
        # Apply JSON re-parses the BOX, and a stale box silently reverted
        # them (gradient/float32 would not stick).
        self.w_json.value = self.spec.to_json()
        self._refresh_spec_summary()
        self.status.object = (
            f"**field method = {g.field_method}, channel = "
            f"{g.channel_dtype}** — the next Solve/Fly re-composes the field.")
        # the cost card these selectors live in must re-price immediately:
        # both options change the compose time and the channel memory
        self._refresh_sizing()

    def _on_mirror_change(self, _=None):
        """Write the chosen mirror axes onto the spec. A mirror change is a
        geometry change, so the next build re-solves — expected and
        announced. Works for scene3d (scene.grid.mirror) and STL/CAD
        (geometry.symmetry.planes) specs alike."""
        # may fire while the widget is being INITIALISED in __init__, before
        # self.status exists — skip the announcement then (the value is set
        # directly on the spec elsewhere at that point).
        if not hasattr(self, "status"):
            return
        m = self._write_mirror(self.w_mirror.value)
        if m is not None:
            self.status.object = (f"**mirror set to {m or '(none)'}** — the "
                                  "next Solve/Fly will re-solve the field "
                                  "for this symmetry.")

    def _err_status(self, prefix, e):
        """Format an exception for the status line WITH a short traceback so
        config-time failures are diagnosable, not just a one-line message
        """
        import traceback
        frames = traceback.format_exc().strip().splitlines()
        tail = "\n".join(frames[-5:]) if len(frames) > 1 else str(e)
        return f"**{prefix}:** {type(e).__name__}: {e}\n```\n{tail}\n```"

    def _on_build_tw(self, _=None):
        """Build a travelling wave from the picker in one action: N phase
        groups + cyclic assignment via build_travelling_wave. Reports the
        created groups; refuses (with a message) if fewer than 2 electrodes
        or 2 phases were chosen rather than making a degenerate wave."""
        try:
            from ion_gym.io.sim_spec import build_travelling_wave
            self._sync_spec()
            picks = self.w_tw_members.value or []
            order = [self._el_opt_of[l] for l in picks if l in self._el_opt_of]
            names = [self.spec.geometry.electrodes[i].name for i in order]
            if len(names) < 2:
                self.status.object = ("**pick >=2 ladder electrodes** for the "
                                      "travelling wave (in order).")
                return
            n = int(self.w_tw_nphase.value)
            if n < 2:
                self.status.object = "**phases (N) must be >= 2.**"
                return
            groups = build_travelling_wave(
                self.spec, names, n,
                frequency_hz=float(self.w_tw_freq.value) * 1e3,
                amplitude_v=float(self.w_tw_amp.value),
                offset_v=float(self.w_tw_off.value),
                waveform=self.w_tw_wave.value,
                prefix=(self.w_tw_prefix.value or "TW").strip())
            self.w_json.value = self.spec.to_json()
            self._suspend_live = True
            try:
                self._rebuild_for_new_spec()
            finally:
                self._suspend_live = False
            self._refresh_spec_summary()
            self.status.object = (
                f"**built {len(groups)}-phase {self.w_tw_wave.value} "
                f"travelling wave** ({', '.join(groups)}) across "
                f"{len(names)} electrodes, cyclic phase assignment.")
        except Exception as e:
            self.status.object = self._err_status("build TW", e)

    def _on_retune_tw(self, _=None):
        """Retune every phase group of the existing TW ladder from the one
        set of amp/freq/waveform/offset controls, preserving per-group
        phase (retune_travelling_wave). One edit instead of N; refuses with
        a message if no ladder with this prefix exists rather than silently
        doing nothing."""
        try:
            from ion_gym.io.sim_spec import retune_travelling_wave
            self._sync_spec()
            prefix = (self.w_tw_prefix.value or "TW").strip()
            names = retune_travelling_wave(
                self.spec,
                amplitude_v=float(self.w_tw_amp.value),
                frequency_hz=float(self.w_tw_freq.value) * 1e3,
                waveform=self.w_tw_wave.value,
                offset_v=float(self.w_tw_off.value),
                prefix=prefix)
            self.w_json.value = self.spec.to_json()
            # IN-PLACE editor refresh (2026-09-12, Brian: adjusting TW
            # groups scrolled the UI to the top, and the retune controls
            # went unresponsive until a tab switch). The old path called
            # _rebuild_for_new_spec() — the full control-column rebuild —
            # for a VALUE-only change: tabs[:] replacement re-rendered
            # the whole column (scroll lost), left the on-screen widgets
            # of the active tab stale until a tab switch forced a
            # re-render (the L-192 class), and re-created the TW builder
            # boxes at their hard-coded defaults, discarding what the
            # user had just typed. A retune changes VALUES of existing
            # groups, never structure, so the per-group editor widgets
            # are updated in place; _rebuild_for_new_spec remains for
            # the structural edits (build ladder, add/remove group,
            # member reassignment).
            for g in self.spec.geometry.rf_groups:
                w = self._grp_widgets.get(g.name)
                if w is None:
                    raise RuntimeError(
                        f"retune: group {g.name!r} has no editor row — "
                        f"the group editors and the spec have diverged "
                        f"structurally; reload the deck (refusing a "
                        f"silent partial refresh)")
                w["amp"].value = float(g.amplitude_v)
                w["freq"].value = float(g.frequency_hz)
                w["wave"].value = g.waveform
            self._refresh_spec_summary()
            self.status.object = (
                f"**retuned {len(names)} TW phase groups** "
                f"({', '.join(names)}) to amp {self.w_tw_amp.value:g} V, "
                f"{self.w_tw_freq.value:g} kHz, {self.w_tw_wave.value} — "
                f"phases preserved. Fly reuses the field (drive-only "
                f"change).")
        except ValueError as e:
            self.status.object = (
                f"**no travelling wave to retune** — {e}. Build one first "
                f"with the builder above.")
        except Exception as e:
            self.status.object = self._err_status("retune TW", e)

    def _on_dc_member_pick(self, event=None):
        """Inverted assignment: the group's member multiselect changed.
        Rewrite el.dc_group from the picker sets, number any newly-added
        members (append after the group's current max index), and rebuild
        so the per-electrode dropdowns match. Coexists with per-electrode
        assignment — this is the same field, edited from the group side."""
        if getattr(self, "_suspend_live", False):
            return
        try:
            self._sync_spec()
            s = self.spec
            want = {}          # electrode idx -> group name (last picker wins)
            for gname, w in self._dcg_widgets.items():
                for lab in (w["pick"].value or []):
                    idx = self._el_opt_of.get(lab)
                    if idx is not None:
                        want[idx] = gname
            for i, el in enumerate(s.geometry.electrodes):
                new_g = want.get(i)
                if new_g != el.dc_group:
                    el.dc_group = new_g
                    if new_g is not None and el.dc_index is None:
                        sib = [e.dc_index for e in s.geometry.electrodes
                               if e.dc_group == new_g and e.dc_index is not None]
                        el.dc_index = (max(sib) + 1) if sib else 0
                    if new_g is None:
                        el.dc_index = None
            self.spec = s
            self.w_json.value = s.to_json()
            # IN-PLACE sync: rebuilding the tab on every
            # pick scrolled the view to the top and left the fired widget
            # detached (unresponsive until a tab switch). Update the per-
            # electrode dropdowns and sibling pickers directly instead.
            self._suspend_live = True
            try:
                for i, el in enumerate(s.geometry.electrodes):
                    w = self._v_widgets.get(i)
                    if w is None:
                        continue
                    tgt = el.dc_group if el.dc_group else "(none)"
                    if w["dc_group"].value != tgt:
                        w["dc_group"].value = tgt
                    tgt_idx = 0 if el.dc_index is None else int(el.dc_index)
                    if w["dc_index"].value != tgt_idx:
                        w["dc_index"].value = tgt_idx
                    w["dc"].disabled = el.dc_group is not None
                for gname, gw in self._dcg_widgets.items():
                    mem = [f"e{i+1} — {e.name}"
                           for i, e in enumerate(s.geometry.electrodes)
                           if e.dc_group == gname]
                    if list(gw["pick"].value or []) != mem:
                        gw["pick"].value = mem
            finally:
                self._suspend_live = False
            self._refresh_dc_derived()
            self._refresh_spec_summary()
        except Exception as e:
            self.status.object = self._err_status("member pick", e)

    def _on_drive_member_pick(self, event=None):
        """Inverted assignment for DRIVE (RF) groups: the group's member
        multiselect changed. An electrode may be in SEVERAL drive groups and
        also a DC group — this only edits rf_groups membership for the one
        group whose picker fired, leaving other memberships intact."""
        if getattr(self, "_suspend_live", False):
            return
        try:
            self._sync_spec()
            s = self.spec
            for gname, w in getattr(self, "_drive_pick", {}).items():
                chosen = {self._el_opt_of.get(lab)
                          for lab in (w.value or [])}
                for i, el in enumerate(s.geometry.electrodes):
                    has = gname in (el.rf_groups or [])
                    if i in chosen and not has:
                        el.rf_groups = list(el.rf_groups or []) + [gname]
                    elif i not in chosen and has:
                        el.rf_groups = [g for g in el.rf_groups if g != gname]
            self.spec = s
            self.w_json.value = s.to_json()
            # IN-PLACE sync — same rationale as the DC picker above.
            self._suspend_live = True
            try:
                for i, el in enumerate(s.geometry.electrodes):
                    w = self._v_widgets.get(i)
                    if w is None:
                        continue
                    tgt = list(el.rf_groups or [])
                    if list(w["group"].value or []) != tgt:
                        w["group"].value = tgt
                for gname, dp in getattr(self, "_drive_pick", {}).items():
                    mem = [f"e{i+1} — {e.name}"
                           for i, e in enumerate(s.geometry.electrodes)
                           if gname in (e.rf_groups or [])]
                    if list(dp.value or []) != mem:
                        dp.value = mem
            finally:
                self._suspend_live = False
            self._refresh_spec_summary()
        except Exception as e:
            self.status.object = self._err_status("drive member pick", e)

    def _on_ion_count_change(self, _=None):
        """Live total-ions readout: ions per m/z x number of masses."""
        try:
            n = int(self.w_n.value)
            toks = [t for t in str(self.w_mz.value).replace(";", ",").split(",")
                    if t.strip()]
            nmz = max(1, len(toks))
            self.w_ion_total.object = (
                f"**= {n*nmz} ions total** ({n} x {nmz} m/z)")
        except (ValueError, TypeError, AttributeError) as e:
            # WAS `except Exception:` writing "" — the ONE genuinely
            # silent handler found by the full 39-handler audit of
            # Blanking the readout is a substitution:
            # the user sees no ion total and is told nothing, so an
            # unparseable m/z list looks identical to a field that has
            # not been filled in yet. NARROWED to the exceptions this
            # actually expects -- str()/split() on a bad widget value and
            # int() on a non-numeric count -- so an unrelated failure is
            # no longer hidden behind it, and it now SAYS what went
            # wrong.
            self.w_ion_total.object = (
                f"**= ?** — cannot read the m/z list or ion count: "
                f"{type(e).__name__}: {e}")

    def _on_live_edit(self, _=None):
        """Reflect a per-electrode / group voltage or membership edit into
        the Config JSON and the summary table immediately. Guarded so a
        transient half-edited state (e.g. a just-added group not yet in the
        widget maps) reports instead of crashing the callback."""
        if getattr(self, "_suspend_live", False):
            return
        try:
            self._sync_spec()                # reads widgets -> self.spec
            self.w_json.value = self.spec.to_json()
            self._refresh_spec_summary()
        except Exception as e:               # never let a watcher die silently
            if hasattr(self, "status"):
                self.status.object = f"**live sync:** {e}"

    def _refresh_spec_summary(self):
        """Rebuild the Config-tab summary table from the current spec."""
        if not hasattr(self, "w_spec_summary"):
            return
        try:
            from ion_gym.io.spec_io import spec_summary_rows
            rows = spec_summary_rows(self.spec)
            body = "\n".join(f"| {k} | {v} |" for k, v in rows)
            self.w_spec_summary.object = (
                "| field | value |\n|---|---|\n" + body)
        except Exception as e:
            self.w_spec_summary.object = f"*summary unavailable: {e}*"

    def _sync_spec(self):
        s = self.spec
        # Source WRITES go where the Ion Source tab READS: the pinned
        # source FA's spec when pinned (see _source_spec), else the live
        # spec — identical objects outside an assembly. Geometry,
        # collisions and integration stay on the LIVE spec: they belong
        # to the displayed/solved subject.
        ssrc = self._source_spec().source
        self._write_mirror(self.w_mirror.value)   # scene OR symmetry.planes
        if hasattr(self, "w_fieldmethod"):
            s.geometry.field_method = self.w_fieldmethod.value
            s.geometry.channel_dtype = self.w_chandtype.value
        ssrc.n_ions = self.w_n.value
        ssrc.distribution = self.w_dist.value
        ssrc.x0_mm = self.w_x0.value
        ssrc.y0_mm = self.w_y0.value
        ssrc.r_mm = self.w_r.value
        ssrc.box_mm = [self.w_boxx.value, self.w_boxy.value,
                       self.w_boxz.value]
        _fw = [self.w_fwx.value, self.w_fwy.value, self.w_fwz.value]
        if (self.w_dist.value == "gaussian" or any(_fw)
                or ssrc.fwhm_mm is not None):
            ssrc.fwhm_mm = _fw
            # trunc: deck/API territory. Preserve a declared value;
            # otherwise derive 3 sigma per active axis (effectively
            # untruncated, satisfies the mandatory-trunc contract, and
            # the written spec SHOWS it).
            if ssrc.trunc_mm is None:
                _sig = 1.0 / 2.3548200450309493      # FWHM -> sigma
                ssrc.trunc_mm = [round(3.0 * f * _sig, 6) if f else 0.0
                                 for f in _fw]
        ssrc.ke_lo = self.w_ke_lo.value
        ssrc.ke_hi = self.w_ke_hi.value
        ssrc.temperature_k = self.w_temp.value
        ssrc.z0_mm = self.w_z0.value
        d = [self.w_dirx.value, self.w_diry.value, self.w_dirz.value]
        if any(abs(c) > 1e-12 for c in d):        # ignore an all-zero entry
            ssrc.direction = d
        masses, bad = [], []
        for tok in str(self.w_mz.value).replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                masses.append(float(tok))
            except ValueError:
                bad.append(tok)
        if bad:
            # This USED to be `except ValueError: pass` -- a typo'd mass was
            # silently DROPPED, so the run flew a DIFFERENT ion set than the
            # box shows (and if every token was bad, the PREVIOUS list, with
            # the box still showing the new one).  Display == solver input
            # applies to the ion source too.
            self.status.object = (
                f"**mz list:** could not parse {', '.join(map(repr, bad))} — "
                f"flying {masses if masses else 'the previous list'} "
                f"(fix the box to match what you want flown)")
        if masses:
            ssrc.mz_list = masses         # keep old list if input empty
        if getattr(self, "w_charge", None) is not None \
                and self.w_charge.value is not None:
            # written verbatim, INCLUDING 0 — the builders' charge=0
            # refusal is the diagnostic; a silent 0->1 here would mask it
            ssrc.charge = int(self.w_charge.value)
        for name, gw in self._grp_widgets.items():
            for grp in s.geometry.rf_groups:
                if grp.name == name:
                    grp.amplitude_v = gw["amp"].value
                    grp.frequency_hz = gw["freq"].value
                    grp.phase_deg = gw["phase"].value
                    if "wave" in gw:
                        grp.waveform = gw["wave"].value
                    if "duty" in gw:
                        grp.duty = float(gw["duty"].value)
                    # step-pulse write-back (see the tau widget note):
                    # 2-point step -> update tau; empty table under
                    # waveform 'table' -> CREATE the canonical step;
                    # longer tables -> untouched (widget was disabled).
                    if "tau" in gw and grp.waveform == "table":
                        _t = list(grp.table_t_us or [])
                        _tv = float(gw["tau"].value)
                        if len(_t) == 2 and _tv > _t[0]:
                            grp.table_t_us = [_t[0], _tv]
                        elif len(_t) == 0:
                            grp.table_t_us = [0.0, max(_tv, 1e-6)]
                            grp.table_v = [0.0, 1.0]
                            grp.interp = "hold"
        if self._v_widgets and (max(self._v_widgets)
                                >= len(s.geometry.electrodes)):
            raise RuntimeError(
                f"_sync_spec: per-electrode editor has rows for "
                f"{max(self._v_widgets) + 1} electrodes but the live "
                f"spec has {len(s.geometry.electrodes)} — the editor "
                f"is stale relative to the spec (a load rebuilt one "
                f"but not the other). Reload the spec; if this "
                f"recurs, the load path that got here skipped "
                f"_rebuild_for_new_spec")
        for i, w in self._v_widgets.items():
            el = s.geometry.electrodes[i]
            gv = w["group"].value
            el.rf_groups = list(gv) if gv else []
            dg = w["dc_group"].value
            el.dc_group = None if dg == "(none)" else dg
            el.dc_index = (int(w["dc_index"].value)
                           if el.dc_group is not None else None)
            if el.dc_group is None:
                el.dc = w["dc"].value          # authored
            # else: DERIVED below — never read from the (disabled) box
        for name, w in getattr(self, "_dcg_widgets", {}).items():
            for grp in s.geometry.dc_groups:
                if grp.name == name:
                    grp.v_in = float(w["v_in"].value)
                    if w.get("v_out") is not None:
                        grp.v_out = float(w["v_out"].value)
                    elif getattr(grp, "uniform", False):
                        grp.v_out = grp.v_in
        s.resolve_dc_groups()                  # ladders/uniform -> member dc
        s.collisions.enabled = self.w_gas_on.value
        s.collisions.gas = self.w_gas.value
        s.collisions.T_k = self.w_T.value
        s.collisions.set_pressure_torr(self.w_P.value)   # both fields, atomically
        s.collisions.__post_init__()
        s.collisions.sigma_m2 = self.w_sigma.value
        s.collisions.model = self.w_col_model.value
        s.collisions.gas_diam_nm = self.w_gdiam.value
        s.integration.dt_ns = self.w_dt.value
        s.integration.t_max_us = self.w_tmax.value
        s.integration.rec_every = self.w_rec.value
        s.integration.max_records = int(self.w_maxrec.value)
        # seed policy: None = random per run, int = fixed/CRN
        # seed policy (master control): unchecked = None = random per
        # run; checked = the pinned integer.
        ssrc.seed = (int(self.w_seedval.value) if self.w_seeded.value
                     else None)
        s.integration.record_channels = list(self.w_chan.value)
        s.name = self.w_name.value
        s.notes = self.w_notes.value
        for axis, w in self.w_bnd.items():
            setattr(s.bounds, f"{axis}_min_on", w["min_on"].value)
            setattr(s.bounds, f"{axis}_min", w["min"].value)
            setattr(s.bounds, f"{axis}_max_on", w["max_on"].value)
            setattr(s.bounds, f"{axis}_max", w["max"].value)
        self.w_json.value = s.to_json()

    # --------------------------------------------------- rendering
    def _update_dt_advice(self):
        """dt-adequacy text beside the pressure. Two
        physical clocks bound the step, both computed from the SAME
        kernel functions (no second physics):
        * RF: the fastest enabled drive period must be resolved —
          advise dt <= T_RF/50.
        * HS collisions: occurrence is a Poisson draw per step with
          P = 1-exp(-v dt/lambda(v)); keeping P <~ 0.1 (dt <= tau/10,
          tau = lambda/v at the characteristic ion speed) preserves the
          exponential free-path statistics — larger steps undercount
          multiple collisions.
        * SDS: the mobility damping time 1/damping must be resolved —
          advise dt <= 0.1/damping.
        Errors render IN the pane, never silently."""
        try:
            import math as _m
            from ion_gym.physics.collision3d import (_mfp_mm, gas_mass,
                                                     KB, KG_AMU, E_CHG)
            s = self.spec
            lines = []
            recs = []
            fs = [g.frequency_hz for g in s.geometry.rf_groups
                  if getattr(g, "amplitude_v", 0.0)]
            if fs:
                t_rf_ns = 1e9 / max(fs)
                recs.append(t_rf_ns / 50.0)
                lines.append(f"fastest RF period **{t_rf_ns:.1f} ns** "
                             f"(resolve: dt &le; {t_rf_ns/50:.2f} ns)")
            if self.w_gas_on.value:
                T = float(self.w_T.value)
                P_pa = float(self.w_P.value) * 133.322
                mz = float(self._source_spec().source.mz_list[0])
                mg = gas_mass(self.w_gas.value)
                c_star = _m.sqrt(2 * KB * T / (mg * KG_AMU)) / 1000.0
                c_bar = _m.sqrt(8 * KB * T / (_m.pi * mg * KG_AMU)) / 1000.0
                ke = max(float(self._source_spec().source.ke_hi), 0.0)
                v_th = _m.sqrt(8 * KB * T
                               / (_m.pi * mz * KG_AMU)) / 1000.0
                v_ke = _m.sqrt(2 * ke * E_CHG / (mz * KG_AMU)) / 1000.0
                v_ch = max(v_th, v_ke, 1e-9)
                if self.w_col_model.value == "sds":
                    from ion_gym.physics.sds import (
                        ion_params as _sds_ion_params,
                        load_massdata as _sds_load_mass)
                    import os
                    from ion_gym.physics import sds as _sds_mod
                    _sdir = os.path.dirname(os.path.abspath(
                        _sds_mod.__file__))
                    md = _sds_load_mass(os.path.join(_sdir, _sds_mod.MOBILITY_FILE))
                    prm = _sds_ion_params(mz, 1.0, mg,
                                          float(self.w_gdiam.value), T,
                                          float(self.w_P.value), md)
                    t_damp_ns = 1e3 / prm["damping"]   # damping in 1/us
                    recs.append(0.1 * t_damp_ns)
                    lines.append(
                        f"SDS damping time **{t_damp_ns:.1f} ns** "
                        f"(resolve: dt &le; {0.1*t_damp_ns:.2f} ns)")
                else:
                    lam = _mfp_mm(v_ch, T, P_pa,
                                  float(self.w_sigma.value), c_star, c_bar)
                    tau_ns = 1e3 * lam / v_ch
                    recs.append(tau_ns / 10.0)
                    lines.append(
                        f"mean free path **{lam:.3f} mm** at "
                        f"v&#8776;{v_ch:.2f} mm/&micro;s &rarr; collision "
                        f"time **{tau_ns:.1f} ns** (Poisson sampling: "
                        f"dt &le; {tau_ns/10:.2f} ns)")
            if not lines:
                self.w_dt_advice.object = (
                    "*no RF drive or collisions enabled — dt is "
                    "unconstrained by gas/RF physics*")
                return
            rec = min(recs)
            cur = float(self.spec.integration.dt_ns
                        if not hasattr(self, "w_dt")
                        else self.w_dt.value)
            ok = cur <= rec * 1.0001
            mark = "&#9989;" if ok else "&#9888;&#65039;"
            self.w_dt_advice.object = (
                "**&Delta;t guidance** &mdash; " + "; ".join(lines)
                + f". Recommended **dt &le; {rec:.2f} ns**; current "
                  f"dt = {cur:g} ns {mark}")
        except Exception as e:
            self.w_dt_advice.object = f"**dt guidance error:** {e}"

    def _el_style(self):
        """Widget state -> renderer kwargs. sim_app marshals; viz_core
        draws (Phase B, one renderer)."""
        return dict(
            color=self.w_elcolor.value,
            alpha=float(self.w_elalpha.value),
            fill=(True if getattr(self, "w_elfill", None) is None
                  else bool(self.w_elfill.value)),
            label=bool(self.w_ellabel.value),
            palette=self._EL_PALETTE)

    def _base_figure(self, model, la="x", lb="y"):
        fig = go.Figure()
        # The 2-D solve defines the FIELD in the xy (axis-transverse)
        # plane only. In xz (also axis-transverse for an r-z/planar
        # geometry) the electrode outline is identical, so we show it; in
        # yz (both transverse) an axisymmetric electrode projects to
        # concentric rings. Field shading/equipotentials only in xy.
        field_plane = ({la, lb} == {"x", "y"})   # order-independent
        # The PE landscape has its OWN plane selector (w_pe_plane) and is a
        # standalone 3-D figure -- it must not be gated on the 2-D VIEW plane
        # (w_plane). It was, so selecting an xz/yz *view* silently dropped you
        # back to the field render and the PE plane looked like it "did not
        # work". Two independent controls, one of them secretly vetoing the
        # other. (I fixed this same gate in _redraw and missed this copy.)
        if (self.w_showfield.value
                and self.w_fieldmode.value.startswith("PE 3D")
                and hasattr(model, "pe_surface")):
            from ion_gym.viz.pe_view import pe_figure_3d
            mzq = float(self._pe_tab.w_mz.value or self.spec.source.mz_list[0])
            # the PE options live on the PE Surface tab — read them from
            # there rather than keeping a second copy of the same controls
            pl, mm = self._pe_opts()
            return pe_figure_3d(model, mz=mzq,
                                charge=int(self.spec.source.charge),
                                plane=pl, metal_mode=mm,
                                electrode_dc=(self._electrode_dc()
                                              if mm != "mask" else None),
                                trust_cells=int(self._pe_tab.w_trust.value))
        # For an RF device (Stl3DModel carries rf_V) the DC potential is ~0,
        # so snapshot the field at RF PEAK phase — the quad saddle and its
        # contours become visible instead of a blank DC map. Models without
        # rf_V (planar/SLIM, which fold RF differently) are unchanged.
        #
        # This USED to be an `except TypeError` sniff around the rf_phase
        # kwarg — the SAME capability-probe-by-exception D1 killed three
        # times in pe_view: a TypeError raised INSIDE potential_image (a real
        # bug) was swallowed and re-run WITHOUT the phase, silently showing a
        # blank DC map for an RF device. The contract is now uniform: every
        # model accepts `rf_phase`, and PlanarModel REFUSES a non-None one
        # with a diagnostic rather than being probed.
        # RF-PEAK is the one phase convention for EVERY view (as
        # exposed by an identity check: this fetch was gated on
        # the xy view, so the xz view of the SAME r-z plane shaded and
        # contoured the phase-0 image while xy showed the peak — two
        # pictures of one plane). field_slice_3d already documents peak
        # phase; now the 2-D image follows the physics, not the view.
        if getattr(model, "rf_V", 0):
            z, r_full, img, em = model.potential_image(rf_phase=np.pi / 2)
        else:
            z, r_full, img, em = model.potential_image()
        if field_plane:
            if self.w_showfield.value:
                pe_mode = (self.w_fieldmode.value.startswith("PE")
                           and hasattr(model, "pe_surface"))
                if pe_mode:
                    # effective (adiabatic) potential-energy landscape for
                    # the selected ion mass: DC + RF Dehmelt pseudopotential.
                    # node-centred: PE heatmap + red PE equipotentials
                    # (pe_view.pe_overlay_2d) — the old faint-blue raw-
                    # potential contours over a 60% heatmap were hard to see.
                    from ion_gym.viz.pe_view import pe_overlay_2d
                    mzq = float(self._pe_tab.w_mz.value or
                                self.spec.source.mz_list[0])
                    pe_overlay_2d(fig, model, mz=mzq,
                                  n_contours=max(self.w_contours.value, 8))
                    # NAME WHAT WAS COMPOSED: this caption
                    # said "effective RF pseudopotential" unconditionally,
                    # so a DC-only deck (the einzel lens in the manual's
                    # Quick Start) was labelled as carrying a
                    # pseudopotential term it does not have -- the
                    # Dehmelt sum is empty with no drives, and the map is
                    # plain electrostatic potential energy. Derived from
                    # the model rather than from the route, so every
                    # model type answers for itself: planar carries
                    # `drives`, r-z and 3-D carry `rf_V`.
                    _has_rf = bool(getattr(model, "rf_V", 0)) or bool(
                        getattr(model, "drives", ()))
                    _pe_txt = (
                        f"effective RF pseudopotential (adiabatic "
                        f"approximation), m/z {mzq:g}" if _has_rf else
                        f"electrostatic potential energy (no RF drive "
                        f"declared), m/z {mzq:g}")
                    fig.add_annotation(
                        text=_pe_txt,
                        xref="paper", yref="paper", x=0.5, y=1.06,
                        showarrow=False,
                        font=dict(color="#888", size=11))
                else:
                    V.efield_heat(fig, model, z, r_full, img)
            elm = getattr(model, "el_masks", None)
            hh = (getattr(model, "mm_per_gu", None)
                  or getattr(model, "h_mm", None))
            if elm and hh:
                st = self._el_style()
                # DRAW FRAME: a planar model carries no
                # `mirror_off_mm`, so the old (0,0,0) fallback drew the
                # electrode masks in the KERNEL frame -- the ghost
                # ladder beside the real instrument. el_mask_fills now
                # applies `origin` on this path too (it previously
                # honoured it only in the transposed branch), so the
                # deck's declared origin is the right fallback. It is
                # (0,0) on every legacy deck.
                _do = list(getattr(self.spec.geometry, "origin_mm", None)
                           or ())
                _fallback = tuple(float(_do[i]) if i < len(_do) else 0.0
                                  for i in range(3))
                V.el_mask_fills(fig, elm, hh,
                                self.spec.geometry.coords, "x", "y",
                                alpha=st["alpha"], fill=st["fill"],
                                label=st["label"], palette=st["palette"],
                                origin=(getattr(model, "world_off_mm",
                                                None)
                                        or getattr(model, "mirror_off_mm",
                                                   None)
                                        or _fallback))
            self._view_contours(fig, model, la, lb, z, r_full, img)
            V.metal_boundary(fig, z, r_full, em)
        elif V.plane_of(la, lb) == "xz":
            # xz plane (SET-derived: Convention A hands la="z", lb="x", so
            # `la == "x"` was never true and this whole branch fell through
            # to the yz/rings else — the funnel-rings-inside-xz screenshot).
            # For an r-z geometry this is axis-vs-transverse and
            # the electrode footprint matches xy. For a PLANAR geometry the
            # electrodes are translationally invariant along the transport
            # axis (z), so each projects to a band on the transverse (x)
            # axis. We draw the OFF-AXIS rods (those not straddling the
            # centre) as the channel walls — e.g. a quad's x-rods appear as
            # two side bands with the beam wiggling between them.
            if self.spec.geometry.coords == "rz":
                if self.w_showfield.value:
                    # the r-z field IS defined in this (axial, r) plane —
                    # shade it here just like xy
                    V.rz_axial_heat(fig, z, r_full, img)
                V.metal_boundary(fig, z, r_full, em)
                elm = getattr(model, "el_masks", None)
                hh = getattr(model, "mm_per_gu", None)
                if elm and hh:
                    # labeled per-electrode fills in the (axial, r) plane
                    st = self._el_style()
                    V.el_mask_fills(fig, elm, hh, "rz", "x", "y",
                                    alpha=st["alpha"], fill=st["fill"],
                                    label=st["label"],
                                    palette=st["palette"])
            else:
                if self.w_showfield.value:      # gate moved out of the
                    _q, _ix = self._view_shading(model, "xz")
                    V.field_slice_3d(fig, model, "x", index=_ix,
                                     quantity=_q, mz=self._pe_mz())
                V.transport_bands(fig, self._model,
                                  self.spec.geometry.coords, axis="x",
                                  la=la, lb=lb, **self._el_style())
        else:
            # yz plane. r-z axisymmetric electrode -> its metal occupies an
            # annulus in r (bore -> outer); by symmetry it also fills
            # [-outer, -bore]. Shade both as bands so the aperture (the
            # inner bore edge, near the beam) is visible — drawing only the
            # outer radius leaves it off-screen and looks electrode-less.
            if self.spec.geometry.coords == "rz":
                # end-on: concentric ring circles (was two radial bands,
                # which read as an aperture but not the ring structure)
                V.rz_rings(fig, self._model, self.spec.geometry,
                           **self._el_style())
            else:
                if self.w_showfield.value:
                    _q, _ix = self._view_shading(model, "yz")
                    V.field_slice_3d(fig, model, "y", index=_ix,
                                     quantity=_q, mz=self._pe_mz())
                V.transport_bands(fig, self._model,
                                  self.spec.geometry.coords, axis="y",
                                  la=la, lb=lb, **self._el_style())
        # the quad/planar field is transverse (xy); if the user asked to
        # see it from a transport view, point them to xy rather than
        # leaving the plot mysteriously field-less.
        if (self.w_showfield.value and not field_plane
                and self.spec.geometry.coords == "xyz"):
            fig.add_annotation(
                text="field is in the xy cross-section — switch to xy",
                xref="paper", yref="paper", x=0.5, y=1.06,
                showarrow=False, font=dict(color="#888", size=12))
        if V.plane_of(la, lb) in ("xz", "yz"):
            # equipotential lines in EVERY
            # view, from the same cut the shading reads; the xy branch
            # called the authority above.
            self._view_contours(fig, model, la, lb, z, r_full, img)
        # PROJECTION WARNING (offered, but with
        # a warning). A 2-D solve has no third dimension. Its transport
        # views are therefore PROJECTIONS -- the electrodes are drawn as
        # they would be if translated along the suppressed axis, which is
        # exactly what the model assumes, but it is an assumption and the
        # picture does not otherwise announce it. Saying so matters most
        # for the MRT, where the xz view (x-oscillation against z-drift) is
        # the most informative picture available AND the one whose axial
        # extent is inferred rather than solved.
        if (V.plane_of(la, lb) in ("xz", "yz")
                and getattr(self._model, "ele", None) is not None
                and getattr(self._model.ele, "ndim", 3) == 2):
            _ax = "z" if V.plane_of(la, lb) == "xz" else "z"
            fig.add_annotation(
                text=(f"projection: this is a 2-D solve — geometry is "
                      f"assumed invariant along {_ax}, not solved there"),
                xref="paper", yref="paper", x=0.5, y=-0.14,
                showarrow=False, font=dict(color="#b07a2c", size=11))
        yaxis = dict(title=f"{lb} (mm)", autorange=True)
        # Aspect: the planar cross-section (xy) must be true-shape so round
        # rods render round — auto-lock it. A genuinely 3-D geometry
        # (depth>0 STL, or the SLIM) is true-shape in EVERY view (all axes
        # are mm), so lock those too — otherwise plotly stretches axes to
        # fill and the quad-in-3-D looks skewed. 2-D transport drifts (a
        # planar quad's arbitrary z-length) still honour the lock checkbox
        # only, since equal-scaling an arbitrary drift makes a sliver.
        # Ask the builder's OWN routing, not a depth_mm correlate.  The
        # `or builder == "slim3d"` special case that used to sit here was
        # papering over depth_mm being wrong about the SLIM's field (3-D
        # solve, depth_mm == 0); build_route answers the field question
        # directly, and it is the SAME classification build_run dispatches on.
        _rt = build_route(self.spec)
        # M-VIZ-Z: true shape is delivered by SIZING the panel (viz_core),
        # never by a scaleanchor -- an anchored axis is a LIVE constraint, so
        # plotly reshapes any zoom rectangle the user drags (it grows the
        # short side).  On a long thin instrument the rectangles that matter
        # are long and thin, so the lock has to go.  Double-click still
        # restores the true-scale view.
        # The CHECKBOX decides. auto_lock used to be OR'd in here, so for any
        # 3-D geometry the user could not turn true-scale OFF -- and a
        # 142 x 15 mm device at 1:1 is a 190 x 760 px sliver, which is exactly
        # what "totally wonked" looks like. auto_lock now only seeds the
        # widget's DEFAULT (see _build_controls); after that the user owns it.
        true_shape = bool(self.w_lock.value)
        # bounding planes relevant to this view's axes
        b = self.spec.bounds
        axis_bounds = {"x": [(b.x_min_on, b.x_min), (b.x_max_on, b.x_max)],
                       "y": [(b.y_min_on, b.y_min), (b.y_max_on, b.y_max)],
                       "z": [(b.z_min_on, b.z_min), (b.z_max_on, b.z_max)]}
        for on, val in axis_bounds.get(la, []):
            if on:
                fig.add_vline(x=val, line=dict(color="#9467bd", width=1.2,
                              dash="dash"))
        for on, val in axis_bounds.get(lb, []):
            if on:
                fig.add_hline(y=val, line=dict(color="#9467bd", width=1.2,
                              dash="dash"))
        fig.update_layout(
            height=560, xaxis_title=f"{la} (mm)", yaxis=yaxis,
            xaxis=dict(autorange=True), dragmode="zoom",
            margin=dict(l=50, r=10, t=30, b=40),
            # zoom persists WITHIN a plane but resets when the plane
            # changes ("keep" alone froze the previous plane's zoom, which
            # made view switches look broken/empty)
            uirevision=self._uirev(la, lb),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0))
        # M-VIZ-Z (see viz_core.free_aspect_axes): free-aspect box zoom;
        # true scale, when asked for, comes from the panel's own size.
        # (apply_zoom_policy moved BELOW, where the extent is known -- see there)
        # autoscale to the model bounds for the chosen axes: transverse
        # axes span the geometry; the transport axis takes the true axial
        # extent when the model declares it (z_extent_mm — the generic
        # attribute every builder can set; replaced the vox2d/outlines3d
        # side channel), else autoranges.
        _zext = getattr(model, "z_extent_mm", None)

        def _rng(lbl):
            # USER BOUNDS FIRST: if the Bounds tab has min/max enabled for
            # this axis, the view honours it — this is the scene-extent
            # control (e.g. clamp a z-invariant STL quad's infinite bands to
            # the physical 120 mm rod length). Falls back to geometry/model
            # extents.
            bw = self.w_bnd.get(lbl)
            if bw is not None and bw["min_on"].value and bw["max_on"].value:
                lo, hi = float(bw["min"].value), float(bw["max"].value)
                if hi > lo:
                    pad = 0.04 * (hi - lo)
                    return (lo - pad, hi + pad)
            if lbl == "z" and _zext is not None:
                lo, hi = _zext
            else:
                dr = self._domain_range(lbl)      # ONE rule (r-z mirrored)
                if dr is None:
                    return None
                lo, hi = dr
            pad = 0.04 * (hi - lo)
            return (lo - pad, hi + pad)

        ra, rb = _rng(la), _rng(lb)
        if ra is not None:
            fig.update_xaxes(range=list(ra))
        if rb is not None:
            fig.update_yaxes(range=list(rb))

        # ZOOM POLICY, with the extent PASSED, not guessed.
        #
        # This used to be called ~30 lines earlier, with no extent, so
        # apply_zoom_policy had to MEASURE the extent off the figure's traces --
        # and in a plane where the electrodes are drawn as shapes rather than
        # scatter (xz with fills off), there is no finite x/y data to measure and
        # it raised VizError.  That exception was swallowed by
        # `except Exception: pass  # a picture must never break a solve`, so the
        # zoom policy SILENTLY DID NOT APPLY and nobody ever saw the error.
        #
        # The caller knew the extent the whole time -- `_rng` is right above.
        # Guessing a quantity the caller can state is the defect; the swallow
        # only hid it.
        from ion_gym.viz import viz_core
        ext = ((ra[0], ra[1], rb[0], rb[1]) if (ra and rb) else None)
        # UNION WITH THE DRAWN GEOMETRY (an STL quad
        # showed only the rods' inner faces until autoscale). Imported
        # bodies can extend OUTSIDE the declared solve domain -- the
        # rods' outer halves do -- so framing to the domain alone clips
        # metal that is on screen. Union so the first paint shows
        # everything drawn; autoscale then agrees instead of correcting.
        # (Fields still exist only inside the domain; this is framing.)
        if ext is not None:
            try:
                dx0, dx1, dy0, dy1 = viz_core.figure_extent(fig)
                ext = (min(ext[0], dx0), max(ext[1], dx1),
                       min(ext[2], dy0), max(ext[3], dy1))
            except viz_core.VizError:
                pass   # nothing drawn to measure: the declared extent stands
        try:
            viz_core.apply_zoom_policy(
                fig, extent=ext, true_scale=true_shape, px_width=900,
                uirevision=self._uirev(la, lb),
                x_title=None, y_title=None)
            # NOTE: nothing sizes the pane here. Square framing used to,
            # which made this block and _size_pane below two authorities
            # over one pixel box -- the second silently overwrote the
            # first. _size_pane is the only writer again.
        except viz_core.VizError as e:
            # A z-invariant geometry in a transport view can draw ONLY
            # shapes (vrects) with an axis whose honest extent does not
            # exist until ions are flown — there is nothing to measure
            # and nothing truthful to state. Autorange IS the truthful
            # answer there. Reported, never swallowed, never fatal
            # to the picture. (Found by the Phase B A-2 sweep on the
            # SLIM 2-D example's xz view.)
            print(f"zoom policy skipped for {la}{lb}: {e}")
        # TRUE SCALE lives on the PANE, not the figure (M-VIZ-Z2). When the
        # pixel sizing moved out of free_aspect_axes, this wiring was never
        # done -- so `true_shape` silently became a no-op and every locked
        # view rendered STRETCHED (axes correct, shapes wrong). Sizing the
        # pane to the data box restores 1:1 mm on first paint, and the zoom
        # stays free (no scaleanchor anywhere), so a drag rectangle is still
        # whatever rectangle the user drew, and double-click restores 1:1.
        # GRIDS: the mask-driven metal renderers above
        # cannot draw a transmission grid (no cells), so users could not
        # see that G1/G2/mirror-entrance exist, let alone that they are
        # tuned. Dashed declared-shape overlays, from the one sanctioned
        # helper.
        V.grid_overlays(fig, self.spec, la, lb, label=bool(self.w_ellabel.value))
        # DECLARED STATIONS: the detector had no
        # plotly equivalent of the mpl renderer's station marks, so the
        # View tab showed an instrument with no exit and Rp could not be
        # read off it. Deck-driven, instrument-agnostic.
        V.station_overlays(fig, self.spec, la, lb,
                           label=bool(self.w_ellabel.value))
        self._size_pane(true_shape, ra, rb)
        return fig

    def _finish_preview(self, fig, la, lb, ra=None, rb=None):
        """Size the pane and apply the stated ranges to a geometry-preview
        figure, then show it.  EVERY preview branch still ends here -- the
        single finish point is worth keeping even though the square framing
        it was introduced for is gone.

        Ranges are applied only when the caller STATES them.  When it does
        not, plotly autoscales: the app measuring a span off the figure and
        presenting it as a declared range was framing invented by the
        display, which is the shape of defect `display == solver input`
        exists to prevent.
        """
        self.pane.sizing_mode = "stretch_width"
        self.pane.width = None
        self.pane.height = VIEW_PX
        fig.update_layout(autosize=True, width=None, height=VIEW_PX)
        if ra:
            fig.update_xaxes(range=list(ra))
        if rb:
            fig.update_yaxes(range=list(rb))
        self.pane.object = fig

    def _uirev(self, la, lb, prefix="keep"):
        """Plotly uirevision that forces AUTOSCALE on every view change.
        plotly keeps pan/zoom while uirevision is constant; we want a fresh
        fit whenever the plane changes (including RETURNING to a plane you
        previously zoomed). We detect the plane change HERE, at draw time,
        and bump the counter — so it does not depend on whether the
        _bump_view watcher fired before or after this redraw (it fires
        after, which is why keying on the plane string alone left the stale
        zoom: returning to 'xy' reused 'keep-xy-N'). Non-plane view changes
        (lock, field mode) bump via their watchers."""
        plane = f"{la}{lb}"
        if getattr(self, "_uirev_plane", None) != plane:
            self._uirev_plane = plane
            self._view_rev = getattr(self, "_view_rev", 0) + 1
        return f"{prefix}-{plane}-{getattr(self, '_view_rev', 0)}"

    def _size_pane(self, true_shape, ra, rb):
        pane = getattr(self, "pane", None)
        if pane is None:
            return
        try:
            # Only touch the pane when the size ACTUALLY changes. Writing
            # width/height on every redraw makes Panel resize the pane, which
            # makes Plotly relayout, which throws away the user's wheel-zoom
            # -- the view snaps back and reads as a spurious "autoscale".
            if true_shape and ra and rb:
                from ion_gym.viz import viz_core
                sz = viz_core.pane_size_for((ra[0], ra[1], rb[0], rb[1]),
                                            px_width=900)
                if (pane.sizing_mode != "fixed"
                        or pane.width != sz["width"]
                        or pane.height != sz["height"]):
                    pane.sizing_mode = "fixed"
                    pane.width = sz["width"]
                    pane.height = sz["height"]
            elif pane.sizing_mode != "stretch_width":
                pane.sizing_mode = "stretch_width"
                pane.width = None
                pane.height = VIEW_PX
        except (ValueError, TypeError, AttributeError) as e:
            # Sizing must not break a solve -- but it must not fail SILENTLY
            # either: a pane that never resizes renders every locked view
            # stretched, and that is exactly the class of bug that shipped.
            print(f"[sim_app] WARNING: pane sizing failed ({e}); "
                  f"the view may render stretched")

    def _refresh_dc_derived(self, _=None):
        """Show what the ladder ACTUALLY puts on each electrode.

        The member DC boxes are disabled (derived, not authored) -- but they
        were also never UPDATED, so after changing DC in/DC out there was no
        way to read the resulting voltages anywhere in the GUI. A derived
        value you cannot see is not much better than one you cannot trust.
        Writes the resolved voltage back into every member's (read-only) box
        and prints the ends + step next to the group.
        """
        s = self.spec
        for name, w in getattr(self, "_dcg_widgets", {}).items():
            for grp in s.geometry.dc_groups:
                if grp.name == name:
                    grp.v_in = float(w["v_in"].value)
                    if w.get("v_out") is not None:
                        grp.v_out = float(w["v_out"].value)
                    elif getattr(grp, "uniform", False):
                        grp.v_out = grp.v_in     # uniform: mirror
        s.resolve_dc_groups()
        for i, el in enumerate(s.geometry.electrodes):
            if el.dc_group is not None and i in self._v_widgets:
                self._v_widgets[i]["dc"].value = round(float(el.dc), 6)
        for name, w in getattr(self, "_dcg_widgets", {}).items():
            mem = sorted((e for e in s.geometry.electrodes
                          if e.dc_group == name),
                         key=lambda e: (e.dc_index if e.dc_index is not None
                                        else 0))
            if w.get("uniform"):
                if mem:
                    w["derived"].object = (
                        f"_uniform: {len(mem)} member(s) at "
                        f"{mem[0].dc:g} V_")
                else:
                    w["derived"].object = "_uniform: no members yet_"
                continue
            if len(mem) < 2:
                w["derived"].object = (
                    "_ladder needs >=2 members "
                    "(or switch to a uniform group)_" if mem
                    else "_no members_")
                continue
            step = (mem[1].dc - mem[0].dc)
            grad = ""
            # THE LADDER GRADIENT.  Three defects sat in this block, two of them
            # under an `except Exception: pass`:
            #
            # (a) `masks.get(...)` was indexed `[:, -1]` -- the LAST axis -- as if
            #     that were always the axial one.  It is for a 3-D (nx,ny,nz) mask.
            #     It is NOT for r-z, where Convention A says the physical z lives
            #     in the WIDTH/x slot (axis 0) and the last axis is r.  An imported-array
            #     r-z import with a DC ladder therefore had its V/cm measured
            #     ACROSS THE RADIUS and printed as if it were down the axis.
            # (b) `if masks:` -- `el_masks` is set by build_stl / build_stl3d /
            #     the retired import route and NEVER by build_planar or build_rz,
            #     NATIVELY built spec the gradient silently did not appear.  Not
            #     "approximate": absent, with no word said.
            # (c) a 2-D xyz cross-section has NO axial axis in the mask at all
            #     (z is the uniform direction).  There is no gradient to compute,
            #     and computing one along y would be a number with no meaning.
            #
            # DECLARE the axis; refuse when there isn't one.
            masks = getattr(self._model, "el_masks", None)
            if not masks:
                grad = (" · _gradient unavailable: this model does not expose "
                        "per-electrode masks (native builders do not set "
                        "`el_masks`)_")
            else:
                m0 = masks.get(mem[0].basis)
                m1 = masks.get(mem[-1].basis)
                if m0 is None or m1 is None:
                    grad = (" · _gradient unavailable: no mask for "
                            f"`{mem[0].basis}`/`{mem[-1].basis}`_")
                else:
                    ax = _axial_axis(s.geometry, m0.ndim)
                    if ax is None:
                        grad = (" · _no axial gradient: this is a 2-D x-y "
                                "cross-section; z is the uniform direction_")
                    else:
                        import numpy as np
                        h = s.geometry.mm_per_gu
                        zc = [float(np.argwhere(m)[:, ax].mean() * h)
                              for m in (m0, m1)]
                        span_mm = abs(zc[1] - zc[0])
                        if span_mm <= 0:
                            grad = (" · _no axial gradient: both ends of the "
                                    "ladder sit at the same axial station_")
                        else:
                            grad = (f" · **{(mem[0].dc - mem[-1].dc) / span_mm * 10:.4g} "
                                    f"V/cm** over {span_mm:.1f} mm "
                                    f"(axial = mask axis {ax})")
            w["derived"].object = (
                f"**{name}**: {mem[0].name} = {mem[0].dc:.4g} V → "
                f"{mem[-1].name} = {mem[-1].dc:.4g} V "
                f"({len(mem)} taps, step {step:.4g} V){grad}")

    def _pe_opts(self):
        """(plane, metal_mode) — from the PE Surface tab, the ONE place the
        PE options live. Keeping a duplicate set on the display tab would be
        two controls for one setting."""
        t = getattr(self, "_pe_tab", None)
        if t is None:
            return "xy", "mask"
        return t.w_plane.value, t.w_metal.value

    def _electrode_dc(self):
        """{electrode label -> DC volts}, resolved (so a DC-ladder member
        reports the voltage the ladder actually gives it, not a stale one)."""
        s = self.spec
        s.resolve_dc_groups()
        out = {}
        for i, el in enumerate(s.geometry.electrodes):
            out[el.basis if el.basis else i + 1] = float(el.dc)
        return out

    def _pe_mz(self):
        """m/z for PE shading — the PE tab's selector, falling back to the
        source's first mass (the same value the xy PE overlay uses)."""
        tab = getattr(self, "_pe_tab", None)
        v = getattr(getattr(tab, "w_mz", None), "value", None)
        return float(v or self.spec.source.mz_list[0])

    def _view_contours(self, fig, model, la, lb, z, r_full, img):
        """Equipotential lines for ANY view:
        the contour slider governs every view, and lines always read
        the SAME cut the shading reads. Per view:
          xy  — the solved primary plane (today's behaviour);
          xz  — r-z/planar solved-in-this-plane routes reuse the
                IDENTICAL (axial, radial) image (free by symmetry);
                3-D routes contour the phi slice the shading shows;
          yz  — r-z synthesises the transverse map from the radial
                potential profile at the slice station (axisymmetry);
                3-D routes contour the phi slice.
        Views where the route defines no field picture (a planar
        model's z-invariant transport views) draw none — the SAME rule
        the shading already follows, stated here rather than silent.
        Synthesised/sliced images are CACHED per (model, plane, slice)
        so switching views never recomputes; a new solve is a new
        model object, so the cache self-invalidates.
        """
        nc = self.w_contours.value
        if nc <= 0:
            return
        plane = V.plane_of(la, lb)
        if plane == "xy":
            V.potential_contours(fig, z, r_full, img, nc)
            return
        coords = self.spec.geometry.coords
        is3d = getattr(getattr(model, "A", None), "ndim", 0) == 3
        cache = getattr(self, "_view_field_cache", None)
        if cache is None or cache.get("model") is not model:
            cache = self._view_field_cache = {"model": model}
        if coords == "rz" and not is3d:
            if plane == "xz":
                # same physical plane as xy by axisymmetry — same lines
                V.potential_contours(fig, z, r_full, img, nc)
                return
            # yz: concentric equipotentials from the radial profile at
            # the slice station (auto = axial midpoint, matching the
            # shading's convention)
            _q, ix = self._view_shading(model, "yz")
            key = ("rz_yz", ix)
            got = cache.get(key)
            if got is None:
                import numpy as _np
                kx = (img.shape[0] // 2 if ix is None
                      else max(0, min(img.shape[0] - 1, int(ix))))
                prof_r = _np.asarray(r_full, float)
                prof_v = _np.asarray(img[kx, :], float)
                half = prof_r >= 0
                rr = _np.hypot(prof_r[:, None], prof_r[None, :])
                vv = _np.interp(rr, prof_r[half], prof_v[half])
                got = cache[key] = (prof_r, prof_r, vv,
                                    float(_np.asarray(z)[kx]))
            ry, rz2, vv, station = got
            V.potential_contours(fig, ry, rz2, vv, nc)
            fig.add_annotation(
                name="contour_slice_note",
                text=f"equipotentials at the x = {station:.2f} mm slice",
                xref="paper", yref="paper", x=1.0, y=1.045,
                xanchor="right", yanchor="bottom", showarrow=False,
                font=dict(size=11, color="#888"))
            return
        if is3d:
            axis = "x" if plane == "xz" else "y"
            _q, ix = self._view_shading(model, plane)
            key = ("phi3d", plane, ix)
            got = cache.get(key)
            if got is None:
                got = cache[key] = V.phi_slice_3d(model, axis, ix)
            xc, yc, sl = got
            if sl is not None:
                # view mapping per field_slice_3d's note: horizontal =
                # z (yc), vertical = transverse (xc); potential_contours
                # wants img (len(X), len(Y)) = (n_z, n_a) = sl.T
                V.potential_contours(fig, yc, xc, sl.T, nc)
        # planar transport views: z-invariant model, no field picture —
        # same rule as the shading; nothing to draw is the documented
        # outcome, not a fall-through.

    def _view_shading(self, model, plane):
        """(quantity, index) for the xz/yz shading: what the shading-type
        selector asks for, cut where the View-tab slice slider says. The
        transport views used to shade the POTENTIAL at a hard-coded
        mid-plane, ignoring the selector — so '|E| field' produced a
        potential map and there was no way to cut elsewhere."""
        q = ("efield" if self.w_fieldmode.value.startswith("|E|")
             else "pe" if self.w_fieldmode.value.startswith("PE")
             else "phi")
        idx = None
        try:
            h = float(getattr(model, "h_mm", 0.0)) or 0.0
            ax = {"xy": 2, "xz": 1, "yz": 0}[plane]
            n = model.A.shape[ax]
            # world_off_mm composes mirror + declared origin (signed
            # frames); mirror_off_mm alone left a signed-origin 3-D
            # deck's slice slider in the stored frame.
            off = (getattr(model, "world_off_mm", None)
                   or getattr(model, "mirror_off_mm", (0.0, 0.0, 0.0)))[ax]
            lo, hi = off, off + (n - 1) * h
            step = max(h, (hi - lo) / 200.0)
            if abs(self.w_viewslice.end - hi) > 1e-9:   # re-range per plane
                self.w_viewslice.start = lo - step
                self.w_viewslice.end = hi
                self.w_viewslice.step = step
            if self.w_viewslice.value > self.w_viewslice.start and h > 0:
                idx = int(round((self.w_viewslice.value - off) / h))
        except Exception as e:
            print(f"[view] slice range unavailable ({e!r}); using the "
                  f"automatic slice")
        return q, idx

    def _mirror_display_offset(self):
        """Per-axis canonical-frame offset (mm): field mm + offset places
        each declared mirror plane at 0, so a mirrored axis reads [-H,+H]
        (the r-z radial convention
        generalised; field-array symmetry centres at local 0). Reads the
        composed world offset (world_off_mm, with the pure-mirror
        fallback) — the docstring said mirror_off_mm long after the code
        moved to the composed offset; corrected in a frame
        audit. Zero on non-mirrored axes and for pre-build/2-D models."""
        out = {"x": 0.0, "y": 0.0, "z": 0.0}
        model = getattr(self, "_model", None)
        # Prefer the composed world offset (mirror + declared origin,
        # signed frames); a 3-D model without it falls back to the pure
        # mirror offset, and planar models carry anchor_mm through the
        # Scene instead.
        mo = (getattr(model, "world_off_mm", None)
              or getattr(model, "mirror_off_mm", None)) if model else None
        if mo is not None:
            out["x"], out["y"], out["z"] = (float(v) for v in mo)
        return out

    def _domain_range(self, lbl):
        """Declared view extent for one axis, in mm.  ONE rule, both views.

        r-z geometry is DESCRIBED on the half-plane (y = radius >= 0) but is a
        body of revolution, and both the solved view and the preview DRAW the
        mirrored body. So the declared y extent is [-H, +H], not [0, H].

        There were two copies of this logic -- one here, one in
        _draw_geometry_only -- and they disagreed: the solved view pinned
        y to [0, H] and then update_yaxes() OVERRODE autorange with it, so a
        freshly loaded funnel opened cropped to its top half in a half-height
        box, and only 'autoscale' (which throws the explicit range away) put
        it right. Duplicated view logic is how that survives a fix: patch one
        copy, the other keeps lying. One rule now.
        """
        g = self.spec.geometry
        # SCENE BOX (unfolded, [0,2H] frame) + canonical offset: mirrored
        # axes read [-H,+H] with the mirror plane at 0, matching the drawn
        # geometry, trajectories, PE and analysis (one canonical
        # frame everywhere; offset zero on non-mirrored axes).
        sc = getattr(self, "_scene", None)
        if sc is not None and sc.is_3d():
            try:
                x0, x1, y0, y1, z0, z1 = sc.box()
                # the Scene is built in the canonical frame (mirror at 0)
                # by scene_from_simspec — use its box directly.
                box = {"x": (x0, x1), "y": (y0, y1), "z": (z0, z1)}[lbl]
                return float(box[0]), float(box[1])
            except (AttributeError, ValueError, KeyError):
                # AUDITED: narrowed — an incomplete/pre-build
                # Scene falls through to the spec-based rule below (the
                # documented equivalent); structural errors now raise.
                pass
        # pre-build fallback: spec extents. A declared 3-D mirror makes the
        # axis canonical [-H,+H] (the stored extent IS the half H) — same
        # rule as r-z below, so the pre-solve view matches the solved one.
        try:
            from ion_gym.physics.build_stl3d import _declared_mirror_axes
            _mx = dict(zip("xyz", _declared_mirror_axes(self.spec)))
        except (AttributeError, TypeError, ValueError):
            # AUDITED: narrowed — incomplete spec means no
            # declared mirrors; anything structural raises.
            _mx = {}
        if _mx.get(lbl):
            _ext = {"x": g.width_mm, "y": g.height_mm,
                    "z": g.depth_mm}[lbl]
            return -_ext, _ext
        # DECLARED ORIGIN: a signed-frame deck has its
        # domain at [origin, origin+extent], not [0, extent]. Assuming
        # zero cropped the axes off the drawn geometry on every
        # signed-frame deck. origin_mm is [0,0] on legacy decks, so
        # every previously-correct view is unchanged.
        _org = getattr(g, "origin_mm", None) or (0.0, 0.0)
        _ox = float(_org[0]) if len(_org) > 0 else 0.0
        _oy = float(_org[1]) if len(_org) > 1 else 0.0
        if lbl == "x":
            return _ox, _ox + g.width_mm
        if lbl == "y":
            if g.symmetry.coords == "rz":
                return -g.height_mm, g.height_mm
            return _oy, _oy + g.height_mm
        if lbl == "z":
            if g.depth_mm > 0:
                # depth_mm answering ITS OWN question: the SOLVE domain's
                # z extent — the honest range for a declared-3-D solve.
                return 0.0, g.depth_mm
            # depth_mm == 0 on a z-transporting geometry (STL quadrupole,
            # 3-D SLIM): the body's z extent is NOT IN THE SPEC — it lives
            # in the bodies: ask the Scene (viz
            # R2, Scene.box()), which is built from those bodies. No scene
            # yet (pre-solve) -> None: autorange stays the truthful answer.
            sc = getattr(self, "_scene", None)
            if sc is not None and sc.is_3d():
                x0, x1, y0, y1, z0, z1 = sc.box()
                return float(z0), float(z1)
            return None
        return None

    def _field_sig(self, spec):
        """Signature of everything the SOLVED FIELD depends on: geometry
        (via _geom_sig) plus the drive (DC values, rf groups). Excludes
        source and integration — changing the ion start point, dt, t_max,
        or record stride does NOT change the field, so those must take the
        zero-work path (reuse the built field, just re-fly) instead of
        falling through to a recompose. Changing the ion
        start or integration params triggered a spurious recompose because
        the full-spec sig below moved even though the field was identical.

        This is deliberately NOT the same as _geom_sig: a DRIVE change DOES
        change the field (and is served cheaply by the reweight compose),
        so the drive belongs in the field sig even though it is absent from
        the geometry sig."""
        import hashlib
        import json
        gsig = self._geom_sig(spec)
        if gsig is None:
            return None
        try:
            j = json.loads(spec.to_json())
        except Exception as e:
            # AUDITED: conservative substitution (no signature
            # -> fresh build, correct-but-slow) now REPORTS instead of
            # hiding serialisation bugs.
            print(f"[sim_app] drive-signature unavailable "
                  f"({type(e).__name__}: {e}) — forcing fresh build")
            return None
        g = j.get("geometry", {})
        drive = {"electrodes": [{"dc": e.get("dc"),
                                 "dc_group": e.get("dc_group"),
                                 "rf_groups": e.get("rf_groups")}
                                for e in g.get("electrodes", [])],
                 "rf_groups": g.get("rf_groups")}
        payload = {"geom": gsig, "drive": drive}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _build_sig(self, spec):
        """A signature of everything a field build depends on. When it is
        unchanged, the last build is still valid and _solve_then reuses it
        rather than re-running voxelize + the (heavy) channel compose. It
        must cover EVERY field-affecting input: geometry, resolution,
        mirror/symmetry, and the full drive (DC + rf groups). We derive it
        from the spec's own serialisation, which already carries all of
        these — so the signature can never drift from what the builder
        actually reads (the failure mode of a hand-listed field set).

        STATIONS ARE EXCLUDED (2026-09-12): they are fly-time config,
        not build input — the fly closure re-evaluates the kernel plane
        list from the LIVE spec on every call (build_stl3d fly_fn ->
        _plane_list(); 2-D routes apply them post-hoc), which the
        station-edit probe demonstrated on the reuse branch (edited
        splat honored by a REUSED fly_fn). Hashing them made every
        station edit read as a drive change and re-ran the whole
        channel compose — minutes on a 140-electrode deck — for an
        edit the build never consumes.
        """
        import hashlib
        try:
            d = spec.to_dict()
        except Exception as e:
            # if the spec cannot serialise, treat every call as a fresh
            # build (correct-but-slow) rather than reuse a stale field.
            # AUDITED: now REPORTS the substitution.
            print(f"[sim_app] spec signature unavailable "
                  f"({type(e).__name__}: {e}) — treating as fresh build")
            return None
        d.pop("stations", None)
        return hashlib.sha256(
            json.dumps(d, sort_keys=True).encode()).hexdigest()

    def _geom_sig(self, spec):
        """Signature of GEOMETRY ONLY — the inputs that force a re-SOLVE of
        the per-electrode bases (shapes, pitch, dims, mirror/symmetry, STL
        refs), with all drive values (DC, rf amplitudes/phases) excluded.
        Lets the reuse diagnostic distinguish a genuine re-solve (geometry
        moved) from a mere re-compose (only the drive changed, bases cache
        will HIT) — so the message stops implying a solve that will not
        happen (it tries to resolve, then catches a cache).
        Derived from the geometry block's own serialisation, minus the
        per-electrode drive fields."""
        import hashlib
        import json
        try:
            g = json.loads(spec.to_json()).get("geometry", {})
        except Exception as e:
            # AUDITED: reports (see drive-signature note).
            print(f"[sim_app] geometry-signature unavailable "
                  f"({type(e).__name__}: {e}) — forcing fresh build")
            return None
        for el in g.get("electrodes", []):
            for drive_key in ("dc", "dc_group", "dc_index", "rf_groups"):
                el.pop(drive_key, None)
        g.pop("rf_groups", None)
        return hashlib.sha256(
            json.dumps(g, sort_keys=True).encode()).hexdigest()

    def _solve_then(self, on_ready):
        """Build the field, then call on_ready(model, fly, cols, births).

        If the field is already cached, builds inline (instant, no
        spinner). If it needs a fresh solve (seconds), runs the build on a
        background thread and shows a spinner — the solver kernel releases
        the GIL, so the UI stays live and the spinner animates.

        Overlap-safe: each call bumps a GENERATION. A newer request
        supersedes older in-flight ones — an old solve's result is dropped
        (never delivered late over a newer field), and each poll owns ITS
        OWN periodic-callback handle, so an old poll finishing can never
        stop a newer solve's poll (that shared-handle clobber was how
        rapid tweak-tweak-recompute left the app silently dead)."""
        self._sync_spec()
        errs = self.spec.validate()
        if errs:
            self.status.object = "**spec errors:** " + "; ".join(errs)
            return
        _advs = self.spec.advisories()
        if _advs:
            # warned, not blocked: the build proceeds;
            # the reason is stated where the user is looking.
            self.status.object = ("⚠ **advisory** — "
                                  + " | ".join(_advs))
        spec = self.spec
        vb = self.w_verbose.value
        # one-click save offer (explicit save plus an
        # offer after finishing): after ANY successful build, highlight
        # the Save-field button if this geometry has no saved field yet.
        _user_ready = on_ready

        def on_ready(*res):
            self._offer_field_save()
            # remember THIS build so a later action on the SAME spec reuses
            # it instead of rebuilding. The rebuild was the bug: pressing
            # Fly after a solve re-ran the whole build (voxelize + the heavy
            # channel compose) even though the field was already in hand —
            # so a cached deck re-composed on every Fly (it
            # re-resolved on Fly though already solved). There is no
            # [vmg] in that log because nothing SOLVED; the cost is the
            # per-group gradient compose, which the reuse now skips.
            self._built = dict(sig=self._build_sig(spec),
                               gsig=self._geom_sig(spec), res=tuple(res))
            _user_ready(*res)

        # REUSE: if the last build's FIELD signature matches, the solved
        # field is still valid even if the source or integration changed —
        # hand the stored result straight back (the fly closure reads the
        # live spec for source/integration, so re-flying needs no rebuild).
        prior = getattr(self, "_built", None)
        _sig = self._build_sig(spec)
        _fsig = self._field_sig(spec)
        _prior_fsig = prior.get("fsig") if prior else None
        _field_reuse = (prior is not None and _fsig is not None
                        and _prior_fsig == _fsig)
        # REUSE DIAGNOSTICS (a deck that "says solving" every Fly — is it
        # actually recomputing?). Report the decision and, on a miss, WHY
        # (no prior build, or the signature changed) so a spurious rebuild
        # is traceable to the exact spec field that shifted.
        if self.w_verbose.value:
            if prior is None:
                print("[reuse] no prior build -> BUILD", flush=True)
            elif _sig is None:
                print("[reuse] spec did not serialise -> BUILD", flush=True)
            elif prior["sig"] == _sig:
                print(f"[reuse] sig match {_sig[:12]} -> REUSE "
                      "(no solve, no compose)", flush=True)
            elif _field_reuse:
                print(f"[reuse] FIELD sig match {_fsig[:12]} -> REUSE "
                      "field (source/integration changed only — re-fly, "
                      "no solve, no compose)", flush=True)
            else:
                prior_gsig = prior.get("gsig")
                now_gsig = self._geom_sig(spec)
                if (prior_gsig is not None and now_gsig is not None
                        and prior_gsig == now_gsig):
                    print(f"[reuse] sig CHANGED {prior['sig'][:12]} -> "
                          f"{_sig[:12]} -> RECOMPOSE; only the DRIVE changed "
                          "(geometry identical) — bases cache will HIT, no "
                          "re-solve, just the channel compose", flush=True)
                else:
                    print(f"[reuse] sig CHANGED {prior['sig'][:12]} -> "
                          f"{_sig[:12]} -> BUILD; GEOMETRY changed since the "
                          "last build — a re-solve is expected", flush=True)
        # RANDOM MODE NEVER REUSES BIRTHS: the stored
        # res is (model, fly_fn, cols, BIRTHS); handing it back replays the
        # identical ensemble, which is exactly "same impact points every
        # Fly". With seed None we fall through to build_run — the bases
        # cache makes that a compose, not a solve, and a FRESH run seed is
        # drawn (and printed) by resolve_run_seed.
        if (prior is not None and _sig is not None and prior["sig"] == _sig
                and spec.source.seed is not None):
            self.status.object = "**reused the solved field** (no recompute)."
            _user_ready(*prior["res"])
            return
        if (prior is not None and _sig is not None and prior["sig"] == _sig
                and self.w_verbose.value):
            print("[reuse] sig matched but seed is RANDOM -> rebuilding "
                  "births (compose only; bases cache hits)", flush=True)
        if spec.source.seed is None and hasattr(spec.source, "_drawn_seed"):
            del spec.source._drawn_seed     # new Fly -> new draw

        self._solve_gen = getattr(self, "_solve_gen", 0) + 1
        gen = self._solve_gen
        # Re-arm the cooperative solve-stop flag: a PRIOR Stop leaves
        # multigrid3d._STOP set, which would otherwise abort this fresh
        # solve immediately.
        from ion_gym.physics import multigrid3d
        multigrid3d.clear_stop()

        # ALL builds go off-thread (a heavy 3-D build dropped the GUI).
        # There USED to be an inline "fast path" here for cache-hit builds
        # ("no spinner flicker"). That equated cache-hit with fast, and for
        # a full SLIM 3-D model a cache hit still loads and re-weights
        # eleven large 3-D bases — tens of seconds ON THE TORNADO EVENT
        # LOOP. A blocked loop misses the websocket keepalive, the browser
        # closes the connection, Bokeh tears the session down, and the next
        # click dies on `assert self._session is not None` (an exact
        # traceback). The off-thread path below already shows its spinner
        # only after 0.12 s, so genuinely fast builds never flicker — the
        # inline branch bought nothing and could kill the session.
        # slow path: solve off-thread, spinner after a short delay
        needs_solve = build_needs_solve(spec)   # for the honest status text
        st = {"done": False, "res": None, "err": None, "spin": False,
              "tb": None, "out": ""}
        own = {"pcb": None}          # THIS solve's callback handle
        t0 = time.time()

        def work():
            # TEE stdout/stderr from the build: the solver's convergence
            # reports stay LIVE on the real streams (
            # capture-only silenced verbose output), while the buffer is
            # kept so a failure can show the tail in the UI.
            import contextlib
            import io
            import sys as _sys
            import traceback as _tb

            class _Tee(io.TextIOBase):
                def __init__(self, live, sink):
                    self._live = live
                    self._sink = sink

                def write(self, s):
                    self._live.write(s)
                    self._live.flush()
                    self._sink.write(s)
                    return len(s)

                def flush(self):
                    self._live.flush()

            buf = io.StringIO()
            tee_out = _Tee(_sys.stdout, buf)
            tee_err = _Tee(_sys.stderr, buf)
            try:
                with contextlib.redirect_stdout(tee_out), \
                        contextlib.redirect_stderr(tee_err):
                    # Pass the DECLARED budget through. Without it the
                    # app's guard would use the user's ceiling while
                    # build_run used the machine fraction, so a declared
                    # 40 GB on a 48 GB box would pass the click and then
                    # be refused by the library underneath it -- two
                    # ceilings for one decision. None = the widget is at
                    # 0, which means "derive from RAM" in both places.
                    _w = getattr(self, "w_ram_budget", None)
                    _budget = (float(_w.value)
                               if (_w is not None and _w.value) else None)
                    st["res"] = build_run(spec, verbose=vb,
                                          record_budget_gb=_budget)
            except Exception as e:
                st["err"] = e
                st["tb"] = _tb.format_exc()
            finally:
                st["out"] = buf.getvalue()
            st["done"] = True

        worker = threading.Thread(target=work, daemon=True)
        worker.start()

        def _stop_own():
            if own["pcb"] is not None:
                try:
                    own["pcb"].stop()      # already-stopped callback: benign
                except (ValueError, RuntimeError):
                    pass
                own["pcb"] = None

        def poll():
            if self._solve_gen != gen:
                # superseded by a newer request: stop quietly, drop result.
                # MUST also clear the spinner THIS solve turned on — else a
                # dropped solve leaves pane.loading=True and the spinner
                # spins forever with no completion (field solving
                # halts and the spinner just keeps spinning). The newer
                # solve manages its own spinner.
                _stop_own()
                if st["spin"]:
                    try:
                        self.pane.loading = False
                    except (AttributeError, RuntimeError):
                        pass
                return
            if not st["done"]:
                # H9e: a worker that DIES without setting done (e.g. an
                # OOM kill or a C-level crash in a numba kernel that no
                # Python `except` can catch) would otherwise leave this
                # poll ticking forever with the spinner stuck (field
                # solve halts, spinner keeps spinning). Detect the dead
                # worker and report instead of spinning.
                if not worker.is_alive():
                    _stop_own()
                    if st["spin"]:
                        try:
                            self.pane.loading = False
                        except (AttributeError, RuntimeError):
                            pass
                    self.status.object = (
                        "**build stopped unexpectedly** — the solve thread "
                        "ended without a result (most likely out of memory "
                        "on this geometry; the field cache and console may "
                        "have more). Nothing was computed.")
                    return
                if not st["spin"] and time.time() - t0 > 0.12:
                    st["spin"] = True
                    self.pane.loading = True
                # ticking notice so a long solve visibly shows progress.
                # The heavy-solve warning is GENERIC: it asks
                # the route's own cost estimate once (sizing_for — the
                # same authority the sizing readout uses) instead of the
                # builder=="slim3d" special case that sat here.
                el = time.time() - t0
                # Throttle: rewrite the notice on the
                # first tick and then every ~10 s -- a per-second counter
                # is churn, not information, on a minutes-long solve.
                last = st.get("_note_t")
                if last is not None and el - last < 10.0:
                    return
                st["_note_t"] = el
                extra = st.get("_slow_note")
                if extra is None:
                    from ion_gym.physics.sim_build import sizing_for
                    from ion_gym.physics.sizing import fmt_time
                    _est = sizing_for(spec).est_solve_s
                    extra = (f" — the first solve for this geometry is "
                             f"heavy (est. ~{fmt_time(_est)})"
                             if _est > 60 else "")
                    st["_slow_note"] = extra
                verb = ("solving field" if needs_solve
                        else "loading cached field")
                self.status.object = (
                    f"⏳ **{verb}… {el:0.0f}s elapsed** (one-time for "
                    f"this geometry; voltage & view changes after this are "
                    f"instant){extra}")
                return
            # finished (and still the current generation)
            _stop_own()
            if st["spin"]:
                try:
                    self.pane.loading = False
                except (AttributeError, RuntimeError):
                    pass        # panel guard: pane torn down mid-solve
            if st["err"] is not None:
                e = st["err"]
                from ion_gym.physics.multigrid3d import SolveInterrupted
                if isinstance(e, SolveInterrupted):
                    # a requested stop is a clean outcome, not a failure
                    self.status.object = ("**solve stopped** — no field "
                                          "computed; press Go to restart")
                else:
                    parts = [f"**build error:** {type(e).__name__}: {e}"]
                    out = (st.get("out") or "").strip()
                    if out:
                        tail = "\n".join(out.splitlines()[-12:])
                        parts.append("\n\n*build output (tail):*\n```\n"
                                     + tail + "\n```")
                    tb = st.get("tb")
                    if tb:
                        frames = tb.strip().splitlines()
                        parts.append("\n*traceback (last frames):*\n```\n"
                                     + "\n".join(frames[-6:]) + "\n```")
                    self.status.object = "".join(parts)
                return
            on_ready(*st["res"])

        # Drive the poll on Panel's event loop when there is a server
        # session. Earlier this pre-checked asyncio.get_running_loop(), but
        # a Bokeh button callback does NOT necessarily have a running
        # asyncio loop in its calling frame even when the server has one —
        # so the check returned False, we fell to the BLOCKING busy-wait
        # below, and the whole UI froze with no spinner, no status ticks,
        # and stdout buffered (a heavy solve — no stdout, no spinner,
        # no status). The reliable signal that we are in a server session
        # is pn.state.curdoc; and add_periodic_callback itself raises
        # cleanly when it cannot schedule, so we TRY it and only busy-wait
        # if it genuinely cannot run.
        started = False
        in_server = False
        try:
            in_server = pn.state.curdoc is not None
        except (RuntimeError, AttributeError):
            # AUDITED: narrowed — outside a server session the
            # probe legitimately answers False.
            in_server = False
        if in_server:
            try:
                own["pcb"] = pn.state.add_periodic_callback(poll, 60)
                self._ensure_watchdog_pulse()
                started = True
                # kick an immediate first tick so the spinner/status appear
                # without waiting a full period
                poll()
            except (RuntimeError, ValueError):
                # no schedulable document after all — fall through
                started = False
        if not started:
            # No server session to schedule a poll on (headless/notebook).
            # Wait for the build thread with a BOUNDED join rather than an
            # open-ended `while not done: sleep` — that spin had no exit if
            # the worker died before setting the flag (no unbounded
            # sleeps). The worker captures its own exception into st["err"],
            # so a definite timeout here surfaces a stuck build instead of
            # hanging forever. The join returns the moment the build
            # finishes; the cap only bounds a pathological stall.
            worker.join(timeout=_BUILD_JOIN_TIMEOUT_S)
            if worker.is_alive():
                self.status.object = (
                    f"**build did not finish within "
                    f"{_BUILD_JOIN_TIMEOUT_S:.0f}s** — it may still be "
                    "running in the background; check the console. This "
                    "path only runs without a live server session.")
                return
            poll()

    def _planes_for_spec(self):
        """Which view planes THIS geometry actually has.

        Same defect, different widget.  pe_view's plane selector offered
        ["xy","xz","yz"] regardless of model, and pressing 'yz' on an r-z model
        blew up in the bokeh event loop -- that was a real crash, and
        it was fixed by DECLARING capability (model.PLANES) and driving the
        selector from it.  This selector was never fixed, and it fails more
        quietly: an r-z model has NO xz/yz plane -- its plane is r-z -- so those
        views draw NOTHING AT ALL, with fills on or off, and say nothing.  The
        electrodes are simply absent.

        A control that can request an impossible state is the bug.
        Offer what exists.
        """
        # AMENDED: the r-z restriction
        # below was RIGHT when written — the r-z xz/yz views drew nothing.
        # The renderer has since grown honest r-z transport views (labeled
        # fills + axial heat in xz, end-on rings in yz), and the
        # examples-draw gate PROVES every example draws all three planes.
        # The stale declaration was disabling the selector on every r-z
        # model, which is why the browser showed a greyed selector and
        # nothing but xy. Capability is still declared, not sniffed — the
        # declaration just now matches the renderer, and the gate is the
        # proof that keeps it honest.
        return ["xy", "xz", "yz"]

    def _sync_planes(self):
        opts = self._planes_for_spec()
        if list(self.w_plane.options) != opts:
            self.w_plane.options = opts
            if self.w_plane.value not in opts:
                self.w_plane.value = opts[0]
        self.w_plane.disabled = (len(opts) == 1)

    def _station_stats(self, results):
        """Per-station detection stats for the Stats card, from the SAME
        declared stations the View tab draws. [] when the deck declares
        none (most decks) or nothing has been flown. A failure reports
        and yields [] -- a stats table is never worth losing a drawn
        flight over."""
        try:
            if not (getattr(self.spec, "stations", None) or []):
                return []
            trajs = [r.traj for r in results
                     if getattr(r, "traj", None) is not None
                     and len(r.traj)]
            if not trajs or not self._cols:
                return []
            return V.station_stats(self.spec, trajs, self._cols)
        except (AttributeError, KeyError, ValueError) as e:
            print(f"[sim_app] station statistics unavailable "
                  f"({type(e).__name__}: {e}); the Stats card omits the "
                  f"station table this run")
            return []

    def _rebuild_scene(self, model):
        """The app builds a Scene — the
        R1 single-source-of-truth for every geometry question (viz R2:
        dimensionality and extents come from the BODIES). Failure is loud
        but never blocks a solve."""
        try:
            from ion_gym.viz.viz_core import scene_from_simspec
            self._scene = scene_from_simspec(self.spec, model)
        except Exception as e:
            self._scene = None
            print(f"scene build FAILED ({e!r}) — geometry questions fall "
                  f"back to spec/autorange until the next solve")

    def _draw_background(self, done_msg=None, allow_solve=True):
        self._sync_planes()   # a control must not offer an impossible plane
        # DON'T AUTO-SOLVE ON LOAD. If drawing the field would require a
        # fresh (slow) solve and the caller did not explicitly ask for one,
        # show the geometry preview instead and let the user press
        # Solve/Fly. Auto-solving here froze the app when a heavy example
        # (SLIM 3-D, ~4 min for 11 bases) was loaded — the constructor and
        # every example-load call this, so loading such an example looked
        # like a total hang (loading a dense example borked the
        # entire app). A cached field still draws inline (fast), so this
        # only changes the miss case.
        if not allow_solve:
            try:
                if build_needs_solve(self.spec):
                    self._draw_geometry_only()
                    self.status.object = (
                        "**geometry preview** — this field isn't solved yet; "
                        "press *Recompute field* or *Fly* to solve.")
                    return
            except Exception as e:
                # if we cannot tell, fall through to the normal path rather
                # than hide the geometry. AUDITED: reports.
                print(f"[sim_app] solve-necessity check failed "
                      f"({type(e).__name__}: {e}) — drawing normally")
        def done(model, fly, cols, births):
            self._model, self._cols, self._births = model, cols, births
            self._rebuild_scene(model)
            try:
                # respect the selected view plane (an xy-only background
                # made the view appear locked until ions were flown)
                _, _, _, _, la, lb = self._plane_cols()
                self.pane.object = self._base_figure(model, la, lb)
            except Exception as e:
                self.status.object = self._err_status("draw error", e)
                return
            if done_msg:
                self.status.object = done_msg
            if getattr(self, "_pe_tab", None) is not None:
                self._pe_tab.refresh()
            if getattr(self, "_fs_tab", None) is not None:
                self._fs_tab.refresh()
            if getattr(self, "_cache_tab", None) is not None:
                self._refresh_cache_tab()
        self._solve_then(done)

    def _show_field(self, _=None):
        self._draw_background()
        self.status.object = "**field preview** (no ions) — press Fly to add ions"

    def _on_display_change(self, _=None):
        # any display edit invalidates the banked live figure: the next
        # live tick sees a signature mismatch and rebuilds once.
        self._disp_rev = getattr(self, "_disp_rev", 0) + 1
        # SUBJECT FIRST (electrodes in the
        # multi-FA view must conform to the display settings): with the
        # assembly displayed, a display toggle redraws the ASSEMBLY —
        # previously this path both ignored the toggle for that view
        # and stomped it with a single-stage figure.
        if self._subject_is_assembly():
            self._redraw_subject()
            return
        # redraw active run if present; else redraw the already-built field
        # in the new plane (no re-solve); else show the geometry preview so
        # views can be checked before any solve
        if self._active and self._active in self._runs:
            self._redraw(self._runs[self._active].results)
        elif getattr(self, "_model", None) is not None:
            # This USED to be `except Exception: self._draw_geometry_only()`
            # -- and geometry-only is a DIFFERENT ANSWER, not a degraded one: it is
            # a plausible, well-formed picture WITH NO FIELD IN IT.  A user who
            # switched plane and got a figure has no way to know the field failed
            # to draw.  Absence is an answer.
            #
            # The identical call 20 lines above (`_redraw`) reports `**draw
            # error:**`.  Two copies of one call, one reporting and one hiding, is
            # exactly the duplicated-view-logic disease `_axis_range`'s own
            # docstring complains about: patch one copy, the other keeps lying.
            _, _, _, _, la, lb = self._plane_cols()
            try:
                self.pane.object = self._base_figure(self._model, la, lb)
            except (VizError, ValueError, TypeError, AttributeError,
                    KeyError, IndexError) as e:
                self.status.object = (
                    f"**draw error ({la}{lb}):** {type(e).__name__}: {e} — the "
                    f"field is NOT shown. (The geometry-only preview is a "
                    f"different picture, not a degraded one; press Recompute "
                    f"or pick a plane the model has.)")
        else:
            self._draw_geometry_only()

    def _ci(self, name):
        return (self._cols.index(name) if self._cols and name in self._cols
                else None)

    def _plane_cols(self):
        """(traj column a, traj column b, summary key a, summary key b,
        axis label a, axis label b) for the selected view plane. Traj
        columns: 1=x 2=y 3=z. Retained runs re-project instantly because
        every step stores x, y AND z."""
        p = self.w_plane.value
        # THE AXIAL COORDINATE GOES HORIZONTAL -- FOR 3-D GEOMETRY ONLY.
        #
        # A 3-D guide is long in z: plotting x across and z UP puts the long
        # axis on the short screen axis, and at 1:1 that is an 80 x 760 px
        # sliver (the Q3 screenshot). Beamlines are drawn beam-left-to-right,
        # and then 1:1 gives a usable 900 x 95 strip.
        #
        # A 2-D (planar / r-z) geometry is ALREADY axial-in-x, and its drawing
        # branches key off `la == "x"` to mean "the axial coordinate" -- so
        # CONVENTION A: THE Z AXIS IS HORIZONTAL.  One mapping,
        # unconditionally -- the same one viz_core uses, because the app and the
        # report must not draw the same instrument two different ways.
        #
        # This USED to branch on `depth_mm > 0`, with the comment "the swap is a
        # fix for one specific pathology, so it applies only where that pathology
        # exists".  That is the defect exactly: a conditional keyed on a CORRELATE rather
        # than a declared property.  And the correlate is WRONG -- the STL
        # quadrupole and the SLIM tetramer both transport down z and BOTH report
        # depth_mm == 0.0, so the "2-D" branch was being taken for two genuinely
        # 3-D instruments.  There is no pathology to confine; there is one axis
        # convention.
        # Plane axes by CHANNEL NAME, not column number.
        # This map is the ONE declared source of the plane->channel pairing;
        # trajectory access goes through TrajRecord with these names.
        m = {"xy": ("x", "y", "x_end", "y_end", "x", "y"),
             "xz": ("z", "x", "z_end", "x_end", "z", "x"),
             "yz": ("z", "y", "z_end", "y_end", "z", "y")}
        if self.spec.geometry.coords == "rz":
            # DECLARED, not guessed: an r-z spec stores the physical
            # AXIAL coordinate in the x slot (see _axial_axis) -- so traj
            # column 1 IS the axial position and columns 2,3 are the two
            # transverse Cartesian components (r_end = hypot(col2, col3)).
            # The xz view for r-z draws the model's own (axial, r) plane
            # (see the coords=="rz" branch of _base_figure), so the axial
            # column goes on the HORIZONTAL plot axis (Convention A).  The
            # old fixed map put column 3 (a TRANSVERSE component) horizontal
            # and column 1 (the axial) vertical, which drew every r-z
            # trajectory rotated 90 degrees against its own electrodes.
            # yz (end-on) keeps (3, 2): both columns transverse -- correct.
            # axial lives in the "x" CHANNEL by r-z declaration; the first
            # transverse component in "y". Same aliasing, now by name.
            m["xz"] = ("x", "y", "x_end", "y_end", "z", "x")
        return m[p]

    def _redraw(self, results, live=False):
        """Draw a result set into the main pane.

        live=False (stored runs, plane changes, the FINAL post-flight
        draw): full figure build + stats + impact + /flight publish —
        the historical behaviour, unchanged.

        live=True (the in-flight tick, where a cheaper draw
        way"): STREAMING. The first live call builds the figure once and
        BANKS it with a group->trace-index map; every later call
        computes the same batched groups (ONE helper, so live and
        stored can never draw one policy two ways) and assigns the new
        arrays INTO the existing traces, then triggers the pane — no
        go.Figure reconstruction, no re-validation of the static base,
        no /flight publish (that belongs to completion; it alone cost
        296 ms/tick). A display-settings change or a new model bumps
        _disp_rev / the model id in the banked signature and forces one
        full rebuild. Measured before: full redraw ~3.9 s at 120 ions;
        the streaming update targets tens of ms."""
        ca, cb, ka, kb, la, lb = self._plane_cols()
        # NOTE: the PE landscape is no longer gated on la=="x" and lb=="y".
        # That test hard-wired the assumption that the PE only exists in the
        # xy plane, which was true only because pe_surface could not slice
        # anything else. It can now; the plane is the user's choice.
        if (self.w_showfield.value
                and self.w_fieldmode.value.startswith("PE 3D")
                and hasattr(self._model, "pe_surface")):
            from ion_gym.viz.pe_view import pe_figure_3d
            mzq = float(self._pe_tab.w_mz.value or self.spec.source.mz_list[0])
            pl, mm = self._pe_opts()
            self.pane.object = pe_figure_3d(
                self._model, mz=mzq, results=results,
                decimate=self.w_decim.value, plane=pl, metal_mode=mm,
                electrode_dc=(self._electrode_dc() if mm != "mask" else None),
                trust_cells=int(self._pe_tab.w_trust.value))
            self.stats.update(results, self.spec,
                              stations=self._station_stats(results))
            if getattr(self, "stats_impact", None) is not None:
                self.stats_impact.update(
                    results, self.spec,
                    stations=self._station_stats(results))
            self._impact_from_results(results)
            if not live:
                self._publish_last_flight(
                    results, site="fly single-stage (PE 3D view)")
            return
        decim = self.w_decim.value
        cby = self.w_colorby.value
        by_mz = (cby == "m/z")
        cidx = (self._ci(cby)
                if cby not in ("fate", "solid color", "m/z") else None)
        solid = self.w_solidcolor.value if cby == "solid color" else None
        # per-result mass (contiguous blocks: ion i -> mz_list[i//n_ions])
        from ion_gym.physics.sim_build import mz_of
        mz_list = list(self.spec.source.mz_list)
        mz_color = _mz_color_map(mz_list)
        _mz_fallbacks = [0]
        def _mz_of_result(i):
            try:
                return mz_of(self.spec, i)
            except (ValueError, IndexError):
                # AUDITED: narrowed to the mz_of refusal class
                # and COUNTED — the count is surfaced on the status line
                # (silently mis-colouring ions is display lying).
                _mz_fallbacks[0] += 1
                return mz_list[0] if mz_list else 0.0
        vmin = vmax = None
        if cidx is not None:
            from ion_gym.io.records import TrajRecord
            allv = np.concatenate([TrajRecord(r.traj, self._cols)[cby]
                                   for r in results if r.traj is not None])
            if allv.size:
                vmin, vmax = float(allv.min()), float(allv.max())
        mode = {"lines": "lines", "dots": "markers",
                "lines+dots": "lines+markers"}[self.w_trajmode.value]
        from ion_gym.io.records import TrajRecord
        # DRAW A DECLARED SUBSET, BATCHED (a request to fly
        # 1000 ions must be honoured). Two costs
        # killed the loop: one plotly trace PER ION (per-trace Python
        # validation is ~ms each, so 1000 traces >> the tick) and no cap
        # at all on this view while the assembly and /flight already
        # subset by policy. Both fixed at the root:
        # (1) SUBSET by the ONE declared policy (trace_keep_count —
        #     floor 25, 25% of the packet, evenly strided so the drawn
        #     set spans the packet), DISCLOSED on the figure title so a
        #     subset can never read as the packet. Statistics, impact
        #     markers, stats cards and the /flight publish below still
        #     use EVERY result.
        # (2) BATCH the drawn paths into ONE scatter per colour group
        #     with None separators (plotly skips None coordinates), so
        #     trace count is O(colour groups), not O(ions).
        from ion_gym.physics.staged_flight import trace_keep_count
        _n_all = len(results)
        _override = int(getattr(getattr(self, "w_maxpaths", None),
                                "value", 0) or 0)
        _keep = (min(_n_all, max(1, _override)) if _override > 0
                 else trace_keep_count(_n_all))
        if _keep < _n_all:
            _di = np.linspace(0, _n_all - 1, _keep).round().astype(int)
            _draw_idx = sorted(set(int(v) for v in _di))
        else:
            _draw_idx = list(range(_n_all))
        # Groups accumulate NUMPY arrays with NaN separators, not lists
        # with None: plotly treats NaN exactly as a gap, but assigning a
        # LIST into an existing trace makes plotly's change-tracking
        # compare it ELEMENT BY ELEMENT (measured: 1.7M scalar
        # comparisons, 2.4 s per streaming update at 120 ions), while an
        # ndarray takes the vectorised fast path in both validation and
        # equality — and ships as a binary buffer instead of JSON.
        _groups = {}   # key -> dict(x=[arr...], y=[arr...], ...)

        def _grp(key, **meta):
            g = _groups.get(key)
            if g is None:
                g = _groups[key] = dict(x=[], y=[], cvals=[], **meta)
            return g

        # LIVE point budget: total points across the drawn subset stay
        # under LIVE_MAX_DRAW_PTS via an EXTRA display stride on top of
        # the user's decim. Display-only (stored/final draws and every
        # statistic use full fidelity), and disclosed on the figure.
        _thin = 1
        if live:
            _tot = sum(len(results[i].traj) for i in _draw_idx
                       if results[i].traj is not None)
            _eff = max(1, _tot // max(1, decim))
            if _eff > LIVE_MAX_DRAW_PTS:
                _thin = int(np.ceil(_eff / LIVE_MAX_DRAW_PTS))
        for i in _draw_idx:
            r = results[i]
            t = r.traj
            if t is None:
                continue
            trec = TrajRecord(t, self._cols)
            _st = decim * _thin
            td = t[::_st] if _st > 1 else t
            if TrajRecord(td, self._cols)["t"][-1] != trec["t"][-1]:
                # keep the true endpoint when decimation would drop it
                td = np.vstack([td, t[-1]])
            drec = TrajRecord(td, self._cols)
            if by_mz:
                # live parallel results arrive in COMPLETION order; the
                # ion's identity is r.index, never its list position.
                m = _mz_of_result(getattr(r, "index", i))
                col = mz_color.get(m, "#888")
                g = _grp(("mz", m), color=col, name=f"m/z {m:g}",
                         legendgroup=f"mz{m:g}", showlegend=True,
                         kind="path")
            elif cidx is None:
                col = (solid if solid is not None
                       else _FATE_COLOR.get(r.summary.get("kind", 2),
                                            "#888"))
                g = _grp(("solid", col), color=col, name=None,
                         legendgroup=None, showlegend=False, kind="path")
            else:
                g = _grp(("chan",), color=None, name=None,
                         legendgroup=None, showlegend=False,
                         kind="channel")
                g["cvals"].append(np.append(
                    np.asarray(drec[cby], float), np.nan))
            g["x"].append(np.append(np.asarray(drec[ca], float), np.nan))
            g["y"].append(np.append(np.asarray(drec[cb], float), np.nan))
        _E = np.empty(0)
        for g in _groups.values():
            g["x"] = np.concatenate(g["x"]) if g["x"] else _E
            g["y"] = np.concatenate(g["y"]) if g["y"] else _E
            g["cvals"] = (np.concatenate(g["cvals"])
                          if g["cvals"] else _E)
        # fate-marker arrays come from EVERY result (never the subset)
        def _fate_xy(code):
            xs = np.array([r.summary[ka] for r in results
                           if r.summary.get("kind") == code], float)
            ys = np.array([r.summary[kb] for r in results
                           if r.summary.get("kind") == code], float)
            return xs, ys

        def _add_group(fig, g):
            if g["kind"] == "channel":
                fig.add_scatter(
                    x=g["x"], y=g["y"], mode="markers",
                    marker=dict(size=3, color=g["cvals"],
                                colorscale="Viridis", cmin=vmin, cmax=vmax,
                                showscale=False),
                    opacity=self.w_alpha.value, showlegend=False,
                    hoverinfo="skip")
            else:
                fig.add_scatter(
                    x=g["x"], y=g["y"], mode=mode,
                    line=dict(color=g["color"], width=self.w_width.value),
                    marker=dict(color=g["color"], size=3),
                    opacity=self.w_alpha.value,
                    showlegend=bool(g["showlegend"]),
                    name=g["name"], legendgroup=g["legendgroup"],
                    hoverinfo="skip")

        def _fate_trace(fig, code, xs, ys):
            fig.add_scatter(
                x=xs, y=ys, mode="markers", name=_FATE_NAME[code],
                marker=dict(color=_FATE_COLOR[code],
                            size=self.w_impactsize.value,
                            symbol=self.w_impactsym.value,
                            line=dict(width=0.5, color="white")))

        def _disclosure_text():
            """Subset disclosure belongs in the Status tab, not on the
            graph. It previously drew as a figure annotation; it now
            writes to the Status tab (whose rolling log keeps it
            readable after later messages), and only WHEN THE TEXT
            CHANGES — a per-tick rewrite of identical text would bury
            the flight-progress log in duplicates. Wording extended
            when the live point budget thins."""
            if _keep >= _n_all and _thin == 1:
                return None
            parts = []
            if _keep < _n_all:
                parts.append(
                    f"showing {len(_draw_idx)} of {_n_all} ion paths "
                    f"(evenly spaced sample; statistics and impact "
                    f"markers use all {_n_all})")
            if _thin > 1:
                parts.append(f"live view thinned {_thin}x for speed - "
                             f"full detail on completion")
            return " · ".join(parts)

        def _apply_disclosure():
            txt = _disclosure_text()
            if txt != getattr(self, "_disclosure_last", None):
                self._disclosure_last = txt
                if txt is not None:
                    self.status.object = f"**path display** — {txt}"

        _sig = (id(self._model), la, lb,
                getattr(self, "_disp_rev", 0),
                getattr(self, "_view_rev", 0))

        def _path_dict(g):
            if g["kind"] == "channel":
                return dict(type="scatter", x=g["x"], y=g["y"],
                            mode="markers",
                            marker=dict(size=3, color=g["cvals"],
                                        colorscale="Viridis", cmin=vmin,
                                        cmax=vmax, showscale=False),
                            opacity=self.w_alpha.value, showlegend=False,
                            hoverinfo="skip")
            return dict(type="scatter", x=g["x"], y=g["y"], mode=mode,
                        line=dict(color=g["color"],
                                  width=self.w_width.value),
                        marker=dict(color=g["color"], size=3),
                        opacity=self.w_alpha.value,
                        showlegend=bool(g["showlegend"]),
                        name=g["name"], legendgroup=g["legendgroup"],
                        hoverinfo="skip")

        def _fate_dict(code, xs, ys):
            return dict(type="scatter", x=xs, y=ys, mode="markers",
                        name=_FATE_NAME[code],
                        marker=dict(color=_FATE_COLOR[code],
                                    size=self.w_impactsize.value,
                                    symbol=self.w_impactsym.value,
                                    line=dict(width=0.5, color="white")))

        def _disclosure_ann():
            # RETIRED as an annotation (Status tab, not
            # the graph); the streaming path calls the status writer and
            # contributes nothing to layout.annotations.
            _apply_disclosure()
            return []

        _bank = getattr(self, "_live_bank", None)
        if live and _bank is not None and _bank["sig"] == _sig:
            # ---- STREAMING = ATOMIC DICT REPLACEMENT (a
            # 'Trace index 59 out of range' + dead Stop/Reset + figures
            # that "look animated" after completion). The previous
            # design mutated the banked go.Figure in place and
            # param-triggered it; Panel's Plotly pane round-trips
            # browser style events (restyle_data) against ITS copy, so
            # once trace counts diverged every echoed event referenced
            # missing indices, the watcher raised inside the session's
            # event batch, and the raise took the Stop/Reset clicks
            # riding the same pipeline down with it. A wholesale dict
            # assignment replaces data and layout ATOMICALLY (and skips
            # plotly validation entirely — dict figures are ~free to
            # compose; the base trace dicts are serialized once at bank
            # time and reused by reference).
            fig_dict = {
                "data": (_bank["base_data"]
                         + [_path_dict(g) for g in _groups.values()]
                         + [_fate_dict(c, *_fate_xy(c))
                            for c in sorted(_FATE_NAME)
                            if _fate_xy(c)[0].size]),
                "layout": {**_bank["base_layout"],
                           "annotations": (_bank["base_anns"]
                                           + _disclosure_ann())},
            }
            self.pane.object = fig_dict
        else:
            # ---- FULL BUILD (stored / final / first live tick) -------
            fig = self._base_figure(self._model, la, lb)
            if live:
                # bank the base ONCE, as plain dicts: traces and layout
                # reused by reference in every streamed frame.
                bj = fig.to_plotly_json()
                self._live_bank = dict(
                    sig=_sig, base_data=list(bj["data"]),
                    base_layout=dict(bj["layout"]),
                    base_anns=list(bj["layout"].get("annotations")
                                   or []))
            for key, g in _groups.items():
                _add_group(fig, g)
            # EVERY known fate, from the single table — not a hardcoded
            # (0,1,2,3). Station absorptions (5 impact plane, 6 detect)
            # were being dropped here, so an ion stopped by a detector
            # patch left no impact marker and the working splat looked
            # like it had done nothing (2026-09-12).
            for code in sorted(_FATE_NAME):
                xs, ys = _fate_xy(code)
                if xs.size:
                    _fate_trace(fig, code, xs, ys)
            _apply_disclosure()
            self.pane.object = fig
            if not live:
                # a stored/final draw retires the live bank: the flight
                # is over or the user is browsing runs.
                self._live_bank = None
        if _mz_fallbacks[0]:
            self.status.object = (
                f"**m/z colour caveat** — {_mz_fallbacks[0]} ion(s) have no "
                f"recorded m/z (pre-v280 run); coloured as the first "
                f"species. Re-fly to record per-ion m/z.")
        self.stats.update(results, self.spec,
                          stations=self._station_stats(results))
        if getattr(self, "stats_impact", None) is not None:
            self.stats_impact.update(
                results, self.spec,
                stations=self._station_stats(results))
        self._impact_from_results(results)
        import time as _t2
        if not live:
            self._publish_last_flight(results, site="fly single-stage")
            self._live_pub_ts = _t2.time()
        elif (_t2.time() - getattr(self, "_live_pub_ts", 0.0)
                >= LIVE_PUBLISH_MIN_S):
            # keep /flight LIVE at a bounded cadence (regression fix,
        # never per tick (296 ms each), and never not at all.
            self._publish_last_flight(results, site="fly single-stage")
            self._live_pub_ts = _t2.time()
        if getattr(self, "_pe_tab", None) is not None:
            self._pe_tab.refresh()
        if getattr(self, "_fs_tab", None) is not None:
            self._fs_tab.refresh()

    # ------------------------------------------------------ run
    def _on_start(self, _=None):
        # ONE FLY BUTTON (two fly buttons were unclear;
        # there should be one fly
        # button"). Fly flies WHAT IS DISPLAYED: assembly subject ->
        # whole instrument (its beam derives from the designated
        # beam.from_stage FA); single-FA subject -> that stage from the
        # Ion tab. The old checkbox/second-button pair is retired — the
        # ambiguity argument is resolved by routing on the
        # visible subject instead of hidden toggle state: what you see
        # is what flies.
        if self._subject_is_assembly():
            return self._on_fly_assembly()
        if self._handle is not None and not self._handle.done:
            return
        # RECORD GUARD, before anything is built or flown: the cost of the
        # record is arithmetic on values already in hand, so a run that
        # cannot fit in memory is refused here rather than discovered in
        # swap with the UI unable to answer.
        if self.w_store_traj.value and not self._record_guard(self.spec):
            return
        self._stop = False
        # persist storage prefs across spec rebuilds
        self._store_traj_val = self.w_store_traj.value
        self._autoclear_val = self.w_autoclear.value
        if self.w_autoclear.value:
            self._runs.clear()
            self.w_runsel.options = []

        def done(model, fly, cols, births):
            self._model, self._cols = model, cols
            self._rebuild_scene(model)
            # keep the Analysis-tab channel choosers in sync with what this
            # run actually records — PLUS the always-derivable channels
            # (speed/ke_ev/radius from position+velocity; e_field/e_axial
            # from the recorded e_x/e_y/e_z). Without the derived ones the
            # post-run refresh would offer e_x/e_y/e_z but DROP e_field/
            # e_axial (they are not recorded columns), the mirror of the
            # pre-run state — so a user could never plot |E|.
            _derived = ["speed", "ke_ev", "radius", "e_field", "e_axial"]
            _opts = list(cols) + [d for d in _derived if d not in cols]
            for w in (self.w_ax, self.w_ay):
                cur = w.value
                w.options = _opts
                if cur in _opts:
                    w.value = cur
                # NO SWALLOW.  This used to be `except Exception: pass`, which
                # meant a column-name mismatch between the tracer's recorded
                # channels and the analysis selectors left the OLD options in
                # place, silently -- so the Analysis tab plotted a channel the
                # run does not have, or refused to offer one it does.
            n = len(births)
            keep = self.w_store_traj.value
            note = "" if keep else " (fates only — trajectories not stored)"
            self.status.object = (
                f"**flying {n} ions...**{note} (a first fly in a fresh "
                f"session pauses ~30-60 s compiling the tracer before "
                f"ion 1 — Stop takes effect after that)")
            self._handle = run_threaded(
                n, fly, check_every=max(5, n // 20),
                decimate=1, keep_traj=keep,
                n_workers=int(getattr(self, "w_workers", None).value
                              if getattr(self, "w_workers", None)
                              is not None else 1))
            self._fly_chip("begin", total=n)
            # a new flight streams into a FRESH banked figure: stale
            # groups/counts from the previous flight must not survive.
            self._live_bank = None
            try:
                self._pcb = pn.state.add_periodic_callback(self._tick, 200)
                self._ensure_watchdog_pulse()
            except RuntimeError:
                # as above: no server/document -> poll on a thread instead.
                self._pcb = None
                threading.Thread(target=self._poll_thread,
                                 daemon=True).start()

        self._solve_then(done)

    def _on_clear_runs(self, _=None):
        """Drop all stored runs and free their trajectory memory."""
        self._runs.clear()
        self._active = None
        self.w_runsel.options = []
        # This USED to be `self._base_figure(self._model)` inside an
        # `except Exception: pass`.  `_base_figure(model, la="x", lb="y")` has
        # DEFAULTS, so the call did not fail -- it silently redrew in xy no matter
        # which plane the user was looking at, and on an r-z model xy is a plane
        # the model DOES NOT HAVE (the plane-capability leak, 4th copy: pe_view,
        # the selector, `_set_plane`, and here).  The swallow then hid whatever
        # that produced, leaving the STALE figure on screen under a status line
        # that said the clear had succeeded.
        #
        # The plane is knowable -- `_plane_cols()` is right there.  Guessing a
        # quantity the caller can state is the defect; the swallow only hid it.
        if getattr(self, "_model", None) is not None:
            _, _, _, _, la, lb = self._plane_cols()
            self.pane.object = self._base_figure(self._model, la, lb)
        else:
            self._draw_geometry_only()
        self.status.object = "**stored runs cleared** — trajectory memory freed."

    # -------------------------------------- solved-field / traj persistence
    def _offer_field_save(self):
        """After a successful build: highlight Save-field iff this
        geometry has no saved field yet (an offer, never a nag)."""
        from ion_gym.io import field_io
        try:
            rows = field_io.scan_fields(spec=self.spec)
            have = any(r["matches"] for r in rows)
        except (OSError, ValueError) as e:
            # a broken fields dir must not break the solve flow — but it
            # is REPORTED, not swallowed
            self.status.object = f"**fields folder unreadable:** {e}"
            return
        self.w_savefield.button_type = "default" if have else "success"

    def _refresh_field_picker(self):
        """Match-annotated pickers: the app matches for the user (SPEC
        §Discoverability). Unreadable files are visible rows, not drops."""
        from ion_gym.io import field_io
        from ion_gym.io import paths as _paths
        rows = field_io.scan_fields(spec=self.spec)
        opts = {}
        for r in rows:
            if r["problem"]:
                opts[f"⚠ {os.path.basename(r['path'])} — unreadable"] = \
                    r["path"]
            elif r["matches"]:
                opts[f"✓ {r['label']} — matches this geometry"] = r["path"]
            else:
                opts[f"✗ {r['label']} — different geometry "
                     f"({r['spec_name']})"] = r["path"]
        self.w_fieldpick.options = opts
        self._field_rows = {r["path"]: r for r in rows}
        d = str(_paths.fields_dir())
        topts = {}
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.endswith(".traj.npz"):
                    topts[name] = os.path.join(d, name)
        self.w_trajpick.options = topts

    def _on_save_field(self, _=None):
        """Explicit save (the default). Pulls the solved bases from
        the disk cache — no re-solve; if the geometry was never solved
        (or its builder doesn't cache), refuse with the reason."""
        from ion_gym.io import basis_cache, field_io
        self._sync_spec()
        bases, ele = basis_cache.load(self.spec)
        if bases is None:
            self.status.object = (
                "**nothing to save** — no solved bases in the cache for "
                "the current geometry. Solve first (Go / preview field); "
                "note: cache-backed builders (planar, 3-D, r-z) store "
                "bases.")
            return
        label = (self.w_fieldlabel.value or "").strip() or None
        try:
            path = field_io.save_field(self.spec, bases, ele, label=label)
        except OSError as e:
            self.status.object = f"**field save failed:** {e}"
            return
        self.w_savefield.button_type = "default"
        self._refresh_field_picker()
        self.status.object = (f"**field saved** → `{os.path.abspath(path)}`"
                              f" in the fields folder — self-validating; "
                              f"rename or send freely.")

    def _on_load_field(self, _=None):
        """HARD-REFUSE load: validates the selected npz
        against the LOADED geometry by embedded key; on success the bases
        are reinstated into the cache (next build is a hit)."""
        from ion_gym.io import field_io
        path = self.w_fieldpick.value
        if not path:
            self.status.object = "**no field selected** — pick one (↻ to rescan)."
            return
        self._sync_spec()
        try:
            field_io.load_field(path, self.spec)
        except field_io.FieldMismatch as e:
            self.status.object = f"**field refused:** {e}"
            return
        except (OSError, ValueError) as e:
            self.status.object = f"**field unreadable:** {e}"
            return
        self.status.object = (
            "**field loaded** — validated against the loaded geometry and "
            "reinstated into the cache. Press **Go** (or preview the "
            "field): it will hit the cache, no re-solve.")

    def _on_bootstrap_field(self, _=None):
        """One-file session open. Order matters and is
        deliberate: (1) reconstruct the spec from the file's embedded
        _meta (refuses pre-bootstrap saves with the reason); (2) put its
        JSON in the EDITOR and apply through the same door as a pasted
        spec — the editor stays the single configuration authority, no
        side-channel spec injection; (3) run the UNCHANGED hard-refuse
        load, whose every integrity check still fires (against the
        reconstructed spec it is a self-consistency proof of the file)."""
        import json
        from ion_gym.io import field_io
        path = self.w_fieldpick.value
        if not path:
            self.status.object = ("**no field selected** — pick one "
                                  "(↻ to rescan).")
            return
        try:
            spec, meta = field_io.read_spec_from_field(path)
        except field_io.FieldMismatch as e:
            self.status.object = f"**cannot bootstrap:** {e}"
            return
        except (OSError, ValueError) as e:
            self.status.object = f"**field unreadable:** {e}"
            return
        self.w_json.value = json.dumps(meta["spec"], indent=2)
        self._on_apply_json()
        if getattr(self, "spec", None) is None:
            return                      # apply already reported its error
        try:
            field_io.load_field(path, self.spec)
        except field_io.FieldMismatch as e:
            self.status.object = f"**field refused after bootstrap:** {e}"
            return
        except (OSError, ValueError) as e:
            self.status.object = f"**field unreadable:** {e}"
            return
        self.status.object = (
            f"**session opened from file** — spec "
            f"`{meta.get('spec_name') or '?'}` loaded into the editor, "
            f"field validated and reinstated into the cache. Press "
            f"**Go**: it will hit the cache, no re-solve.")

    def _on_relabel_field(self, _=None):
        """Relabel = LOCAL registry edit (never rewrites the npz)."""
        from ion_gym.io import field_io
        path = self.w_fieldpick.value
        label = (self.w_fieldlabel.value or "").strip()
        if not path or not label:
            self.status.object = ("**relabel needs both** a selected "
                                  "field and a label.")
            return
        row = getattr(self, "_field_rows", {}).get(path)
        if not row or not row.get("key"):
            self.status.object = ("**cannot relabel** — the selected "
                                  "entry is unreadable (no key).")
            return
        field_io.set_label(row["key"], label)
        self._refresh_field_picker()
        self.status.object = f"**relabeled** → {label} (local registry)."

    def _on_save_traj(self, _=None):
        from ion_gym.io import field_io
        if self._active not in self._runs:
            self.status.object = ("**no run to save** — Fly some ions "
                                  "first.")
            return
        res = self._runs[self._active].results
        try:
            path = field_io.save_trajectories(
                res, self.spec,
                label=(self.w_fieldlabel.value or "").strip() or None)
        except OSError as e:
            self.status.object = f"**trajectory save failed:** {e}"
            return
        self._refresh_field_picker()
        meta = field_io.read_trajectory_meta(path)
        skipped = (f" ({meta['n_skipped']} ions had no recorded "
                   f"trajectory and were skipped)"
                   if meta.get("n_skipped") else "")
        self.status.object = (f"**trajectories saved** → "
                              f"`{os.path.abspath(path)}`{skipped}.")

    def _on_load_traj(self, _=None):
        """Load a .traj.npz as a STORED RUN — plugs into the existing
        reloadable-runs machinery, so overlay/compare costs no new plot
        code. Geometry mismatch WARNS and proceeds."""
        from ion_gym.io import field_io
        from ion_gym.physics.ensemble_driver import (IonResult,
                                                     EnsembleProgress)
        path = self.w_trajpick.value
        if not path:
            self.status.object = "**no trajectory file selected**."
            return
        self._sync_spec()
        try:
            trajs, meta, warn = field_io.load_trajectories(path, self.spec)
        except (field_io.TrajectoryFormatError, OSError, ValueError) as e:
            self.status.object = f"**trajectories unreadable:** {e}"
            return
        summs = meta.get("summaries") or [{}] * len(trajs)
        results = [IonResult(index=i, traj=trajs[i],
                             summary=(summs[n] if n < len(summs) else {}))
                   for n, i in enumerate(sorted(trajs))]
        n_term = sum(1 for r in results
                     if r.summary.get("kind") == 0)
        fin = EnsembleProgress(done=len(results), total=len(results),
                               elapsed_s=0.0, n_terminated=n_term,
                               results=results)
        name = (f"loaded: {meta.get('label') or meta.get('spec_name')} "
                f"[{time.strftime('%H:%M:%S')}]")
        self._runs[name] = fin
        self._active = name
        self._sync_analysis_mz()
        self.w_runsel.options = list(self._runs.keys())
        self.w_runsel.value = name
        self._redraw(results)
        msg = (f"**trajectories loaded** as run '{name}' "
               f"({len(results)} ions).")
        if warn:
            msg += f" ⚠ {warn}."
        self.status.object = msg

    def _apply_raster_visibility(self):
        """Toggle electrode voxels in the already-computed raster figs
        WITHOUT recomputing — flips trace visibility (Plotly Mesh3d /
        Scatter / Surface all honor .visible). Cheap and instant."""
        vis = bool(getattr(self, "w_raster_electrodes", None)
                   and self.w_raster_electrodes.value)
        panes = getattr(self, "_raster_panes", None)
        if panes is None:
            return
        leg = bool(getattr(self, "w_raster_legend", None)
                   and self.w_raster_legend.value)
        # _raster_panes (a pn.Tabs/Column) COERCES (title, pane) tuples:
        # iterating yields bare panes (a real bokeh crash).
        items = list(getattr(panes, "objects", panes))
        for entry in items:
            pane = entry[1] if isinstance(entry, tuple) else entry
            fig = getattr(pane, "object", None)
            if fig is None:
                continue
            for tr in fig.data:
                tr.visible = True if vis else "legendonly"
            fig.update_layout(showlegend=leg)
            pane.object = fig

    def _clear_raster(self):
        """Wipe the Raster tab (a stale raster from the
        PREVIOUS geometry displayed under a new example is a lie)."""
        if getattr(self, "_raster_panes", None) is not None:
            self._raster_panes[:] = []
            self._raster_status.object = (
                "_the solver's voxel occupancy — press Compute after "
                "loading a 3-D geometry (scene or STL)_")

    def _on_raster(self, _=None):
        if self._subject_is_assembly():
            self.status.object = ("**Raster is not available for a multi-FA instrument** — it operates on the single displayed FA. Use *Set FA View* first.")
            return
        """3-D voxel raster of the LOADED geometry — the labels the solver
        consumes, per route's OWN rasterizer (never a second opinion):
        scene3d -> rasterize3d.rasterize; stl3d -> stl_masks_3d. 2-D
        routes are refused with the reason (their standard view already
        draws the solver labels)."""
        from ion_gym.viz.viz_core import voxel_views, VizError
        self._sync_spec()
        spec = self.spec
        try:
            if spec.scene is not None:
                from ion_gym.physics.scene3d import GeomScene
                from ion_gym.physics.rasterize3d import rasterize
                sc = GeomScene.from_json(dict(spec.scene))
                lab = rasterize(sc)
                # UNFOLD the declared mirror(s) so the raster shows the
                # FULL instrument, not the stored half (a
                # SLIM showed only the top board). The solver unfolds these
                # internally; the display must match the solved geometry.
                from ion_gym.viz.viz_core import unfold_mirror_labels
                lab = unfold_mirror_labels(lab, getattr(sc.grid,
                                                          "mirror", ""))
                h = sc.grid.mm_per_gu
                names = {e.index: e.name for e in sc.electrodes}
                volts = {e.index: e.voltage for e in sc.electrodes
                         if isinstance(e.voltage, (int, float))}
                title = sc.name or spec.name
            elif spec.builder == "stl3d" or (
                    spec.geometry.electrodes and
                    all(e.stl for e in spec.geometry.electrodes)):
                import numpy as np
                from ion_gym.physics.build_stl3d import stl_masks_3d
                masks = stl_masks_3d(spec)
                first = next(iter(masks.values()))
                lab = np.zeros(first.shape, np.int16)
                for i, m in sorted(masks.items()):
                    lab[m] = i
                h = spec.geometry.mm_per_gu
                names = {i + 1: e.name for i, e in
                         enumerate(spec.geometry.electrodes)}
                volts = {i + 1: e.dc for i, e in
                         enumerate(spec.geometry.electrodes)}
                title = spec.name
            else:
                # 2-D route (planar / r-z): the 3rd dimension is the
                # DECLARED symmetry — revolve r-z, extrude planar. The
                # labels come from the built model (a
                # raster for ALL geometries incl. r-z).
                from ion_gym.viz.viz_core import voxel_views_2d
                model = getattr(self, "_model", None)
                if model is None or not hasattr(model, "ele"):
                    self._raster_status.object = (
                        "**no built model yet** — press Go or Recompute "
                        "field first; the raster shows the solver's own "
                        "labels, which exist once the geometry is built.")
                    return
                sym = getattr(self.spec.geometry.symmetry, "coords", "xyz")
                mode = "rz" if sym == "rz" else "planar"
                names = {i + 1: e.name for i, e in
                         enumerate(spec.geometry.electrodes)}
                volts = {i + 1: e.dc for i, e in
                         enumerate(spec.geometry.electrodes)}
                import numpy as np
                ele = np.asarray(model.ele)
                if ele.ndim == 3 and ele.shape[2] > 1:
                    # a LOADED imported 3-D geometry lands here (not a
                    # scene, electrodes carry no .stl), so its labels are
                    # already a native 3-D grid — render with voxel_views,
                    # NOT voxel_views_2d, which rejects a 3-D grid (the
                    # raster refused on a loaded json).
                    figs = voxel_views(ele, spec.geometry.mm_per_gu,
                                       names=names, voltages=volts,
                                       title=spec.name)
                    self._raster_panes[:] = [
                        (k, pn.pane.Plotly(v, sizing_mode="stretch_width",
                                           height=620))
                        for k, v in figs.items()]
                    self._apply_raster_visibility()
                    self._raster_status.object = (
                        "**raster computed** (native 3-D grid from the "
                        "loaded geometry).")
                    return
                figs = voxel_views_2d(
                    model.ele, spec.geometry.mm_per_gu, symmetry=mode,
                    names=names, voltages=volts, title=spec.name)
                self._raster_panes[:] = [
                    (k, pn.pane.Plotly(v, sizing_mode="stretch_width",
                                       height=620))
                    for k, v in figs.items()]
                self._apply_raster_visibility()
                self._raster_status.object = (
                    f"**raster computed** ({mode}: 3-D via declared "
                    f"symmetry; 2-D views exact solver labels).")
                return
            figs = voxel_views(lab, h, names=names, voltages=volts,
                               title=title)
        except VizError as e:
            self._raster_status.object = f"**raster refused:** {e}"
            return
        except Exception as e:
            # UI boundary: report, never a silent dead button
            self._raster_status.object = (f"**raster error:** "
                                          f"{type(e).__name__}: {e}")
            return
        self._raster_panes[:] = [
            (k, pn.pane.Plotly(figs[k], sizing_mode="stretch_width",
                               height=620))
            for k in ("3d", "xy", "xz", "yz")]
        self._apply_raster_visibility()
        occ = "; ".join(f"{k}" for k in figs)
        self._raster_status.object = (
            f"**raster computed** — views: {occ}. This is the exact int16 "
            f"occupancy the solver receives (h={h:g} mm/gu).")

    # ---------------------------------------------- Cache tab
    # (export default: the user-facing outputs dir when present, else cwd)
    def _build_cache_tab(self):
        """A tab that shows WHERE the cache lives, WHAT is in it and how
        big each item is, with selective + full clear.
        make the disk footprint of a model visible and prunable, and move
        the clear controls out of Config › Runs to their own home."""
        import panel as pn
        from ion_gym.io.fa_cache import DEFAULT_ROOT
        self._cache_root = os.path.expanduser(DEFAULT_ROOT)
        self._cache_loc = pn.pane.Markdown("")
        import pandas as _pd
        # Tabulator, not a static DataFrame pane (ruled 2026-09-11):
        # every column header click-sorts (date, size, name — the
        # "rapid sort" ask), and clicking a row IS the selection —
        # it drives the picker below and the export button's label.
        self._cache_table = pn.widgets.Tabulator(
            _pd.DataFrame(columns=["key", "produced_by", "descriptor",
                                   "size_MB", "arrays", "shape",
                                   "modified"]),
            selectable=1, disabled=True, show_index=False,
            pagination=None, height=300, sizing_mode="stretch_width")
        self._cache_table.param.watch(self._on_cache_table_select,
                                      "selection")
        self._cache_pick = pn.widgets.Select(
            name="entry to remove", options=["(refresh first)"], width=380)
        self._cache_refresh_btn = pn.widgets.Button(
            name="↻ refresh", width=120)
        self._cache_refresh_btn.on_click(lambda e: self._refresh_cache_tab())
        self._cache_remove_btn = pn.widgets.Button(
            name="✕ remove selected", button_type="warning", width=170)
        self._cache_remove_btn.on_click(self._on_cache_remove_one)
        self._cache_export_btn = pn.widgets.Button(
            name="⇩ export — nothing selected", disabled=True,
            button_type="primary", width=360)
        self._cache_export_btn.on_click(self._on_cache_export_one)
        # the button NAMES what it will export (ruled 2026-09-11): a
        # generic label on a destructive-adjacent action hides intent
        self._cache_pick.param.watch(self._sync_cache_export_btn,
                                     "value")
        self._cache_export_dir = pn.widgets.TextInput(
            name="export to", value=str(_default_export_dir()), width=380)
        self._cache_clear_btn = pn.widgets.Button(
            name="🗑 clear ALL field cache", button_type="danger", width=210)
        self._cache_clear_btn.on_click(self._on_cache_clear_all)
        self._cache_mem = pn.pane.Markdown("")
        self._cache_status = pn.pane.Markdown("_press refresh to inventory "
                                              "the cache_")
        self._cache_tab = pn.Column(
            self._cache_loc,
            pn.Row(self._cache_refresh_btn, self._cache_clear_btn),
            self._cache_table,
            pn.Row(self._cache_pick, self._cache_remove_btn),
            pn.Row(self._cache_export_dir, self._cache_export_btn),
            self._cache_mem,
            self._cache_status,
            sizing_mode="stretch_width")
        self._refresh_cache_tab()

    def _refresh_cache_tab(self, _=None):
        import pandas as pd
        from ion_gym.io.fa_cache import entries
        root = self._cache_root
        try:
            inv = entries(root=root)
        except OSError as e:
            self._cache_status.object = f"**cache unreadable** — {e}"
            return
        total = sum(e["bytes"] for e in inv)
        self._cache_loc.object = (
            f"**Cache location:** `{root}`  \n"
            f"**{len(inv)} entries, {total / 1e6:.1f} MB total on disk**")
        import time as _t
        def _prov(e):
            # DESCRIPTIVE, many-to-one by design:
            # every named spec that stored or resolved this geometry, in
            # first-seen order — never one pretended origin, never part
            # of the key or of validation. Pre-provenance entries show a
            # dash until next touched.
            seen = list(dict.fromkeys(e.get("produced_by") or []))
            return ", ".join(seen) if seen else "—"
        rows = [dict(key=e["key"][:12],
                     produced_by=_prov(e),
                     descriptor=e.get("descriptor") or "— (pre-v341 "
                     "solve; re-solve to describe)",
                     size_MB=round(e["bytes"] / 1e6, 2),
                     arrays=e["arrays"],
                     shape=("×".join(str(s) for s in e["shape"])
                            if e["shape"] else "—"),
                     modified=_t.strftime("%Y-%m-%d %H:%M",
                                          _t.localtime(e["mtime"])))
                for e in inv]
        self._cache_table.value = (pd.DataFrame(rows) if rows
                                    else pd.DataFrame(
                                        columns=["key", "produced_by",
                                                 "descriptor", "size_MB",
                                                 "arrays", "shape",
                                                 "modified"]))
        def _pick_label(e):
            names = list(dict.fromkeys(e.get("produced_by") or []))
            who = names[0] + ("…" if len(names) > 1 else "") if names \
                else (e["spec_name"] or "—")
            desc = e.get("descriptor")
            return (f"{e['key'][:12]} — {who}"
                    + (f" — {desc}" if desc else "")
                    + f" ({e['bytes'] / 1e6:.1f} MB)")
        self._cache_pick.options = ([_pick_label(e) for e in inv]
                                    or ["(cache empty)"])
        self._cache_key_of = {_pick_label(e): e["key"] for e in inv}
        self._cache_label_of_key12 = {e["key"][:12]: _pick_label(e)
                                      for e in inv}
        self._sync_cache_export_btn()
        header = ("**In-memory caches** (freed on 'clear ALL' or when the "
                  "app restarts):\n")
        self._cache_mem.object = header + "\n".join(
            self._mem_cache_lines())
        self._cache_status.object = "_inventory current_"

    def _on_cache_table_select(self, event=None):
        """A clicked table row selects that entry in the picker (and
        therefore names it on the export button). Sorting the table
        never desyncs this: the selected DATAFRAME row's key column is
        matched, not a positional index."""
        try:
            df = self._cache_table.selected_dataframe
        except (AttributeError, IndexError):
            return
        if df is None or not len(df):
            return
        lab = getattr(self, "_cache_label_of_key12", {}).get(
            str(df.iloc[0]["key"]))
        if lab:
            self._cache_pick.value = lab

    def _sync_cache_export_btn(self, event=None):
        sel = self._cache_pick.value
        key = getattr(self, "_cache_key_of", {}).get(sel)
        if not key:
            self._cache_export_btn.name = "⇩ export — nothing selected"
            self._cache_export_btn.disabled = True
            return
        short = sel if len(sel) <= 46 else sel[:43] + "…"
        self._cache_export_btn.name = f"⇩ export {short} (+ spec json)"
        self._cache_export_btn.disabled = False

    def _on_cache_export_one(self, _=None):
        """Export the selected cache entry as a portable field bundle
        plus its producing-spec JSON. Loud on
        every failure path; a legacy entry without a descriptor exports
        arrays-only with a NOTE (its spec was never recorded)."""
        sel = self._cache_pick.value
        key = getattr(self, "_cache_key_of", {}).get(sel)
        if not key:
            self._cache_status.object = ("**export:** select an entry "
                                         "first (refresh, then pick)")
            return
        from ion_gym.io.field_io import export_field_bundle
        try:
            npz, sj = export_field_bundle(key, self._cache_export_dir.value,
                                          root=self._cache_root)
        except (OSError, ValueError, FileNotFoundError) as e:
            self._cache_status.object = f"**export FAILED:** {e}"
            return
        self._cache_status.object = (
            f"**exported** `{npz}`"
            + (f" **+** `{sj}`" if sj else
               " — *no spec JSON: entry predates descriptors; re-solve "
               "once to record it*"))

    def _mem_cache_lines(self):
        lines = []
        try:
            from ion_gym.physics.build_stl3d import _COMPOSE_GRAD_CACHE
            nb = 0
            for slot in _COMPOSE_GRAD_CACHE.values():
                for trip in slot.get("grads", {}).values():
                    for a in trip:
                        nb += getattr(a, "nbytes", 0)
            lines.append(f"- compose gradients: {len(_COMPOSE_GRAD_CACHE)} "
                         f"geometry, {nb / 1e6:.1f} MB")
        except Exception as e:
            lines.append(f"- compose gradients: unreadable ({e})")
        return lines

    def _on_cache_remove_one(self, _=None):
        from ion_gym.io.fa_cache import remove
        label = self._cache_pick.value
        key = getattr(self, "_cache_key_of", {}).get(label)
        if key is None:
            self._cache_status.object = "_nothing to remove (refresh first)_"
            return
        ok, freed = remove(key)
        if ok:
            self._cache_status.object = (
                f"**removed** `{key[:12]}` — freed {freed / 1e6:.1f} MB")
        else:
            self._cache_status.object = (
                f"**not found** `{key[:12]}` — already gone?")
        self._refresh_cache_tab()

    def _on_cache_clear_all(self, _=None):
        # reuse the audited clear (disk + in-memory), then re-inventory
        self._on_clear_cache()
        try:
            from ion_gym.physics.build_stl3d import _COMPOSE_GRAD_CACHE
            _COMPOSE_GRAD_CACHE.clear()
        except (ImportError, AttributeError):
            # AUDITED: stated skip (see above).
            pass
        self._refresh_cache_tab()

    def _on_clear_cache(self, _=None):
        """Purge every solved-field cache: the disk basis cache
        (fa_cache) AND the in-process planar/r-z caches. A visible,
        reported clear — it names what it freed, never a silent wipe
        (bug D). The next solve re-solves from scratch."""
        from ion_gym.io.fa_cache import clear_all
        from ion_gym.physics.build_planar import (
            clear_memory_cache as _clear_planar)
        from ion_gym.physics.build_rz import (
            clear_memory_cache as _clear_rz)
        mem = _clear_planar() + _clear_rz()
        try:
            n, freed = clear_all()
        except OSError as e:
            self.status.object = (f"**cache partly cleared** — {mem} "
                                  f"in-memory dropped; disk cache: {e}")
            return
        self.status.object = (
            f"**field cache cleared** — {n} disk entries "
            f"({freed / 1e6:.1f} MB) + {mem} in-memory dropped; the next "
            f"solve recomputes.")

    # units for every analysis channel, so axis labels state them (the
    # e-field is V/mm, NOT V/m — the recorded gradient is V per mm; showing
    # the wrong unit or none was the reported ambiguity).
    _CHAN_UNITS = {
        "t": "us", "x": "mm", "y": "mm", "z": "mm",
        "vx": "mm/us", "vy": "mm/us", "vz": "mm/us",
        "speed": "mm/us", "ke_ev": "eV", "radius": "mm",
        "e_field": "V/mm", "e_axial": "V/mm", "e_radial": "V/mm",
        "e_x": "V/mm", "e_y": "V/mm", "e_z": "V/mm", "n_col": "count",
    }

    def _chan_label(self, name):
        u = self._CHAN_UNITS.get(name)
        return f"{name} ({u})" if u else name

    def _traj_column(self, traj, name, cols, mz):
        """Return one analysis column from a trajectory array by NAME.

        Base channels (t,x,y,z,vx,vy,vz) are read directly. The velocity/
        position-DERIVED optional channels are COMPUTED here from those base
        columns — exactly (display == computation: they are functions of the
        recorded state, not a second recording that could drift):
            speed  = |v|                       (mm/us)
            ke_ev  = 1/2 m v^2                  (eV, m from this ion's m/z)
            radius = sqrt(x^2 + y^2)            (mm, transverse to z)
        Field channels (e_field, e_axial, ...) are NOT derivable from the
        trajectory alone (they need the field sampled at each step, which the
        3-D tracer does not currently store per point) — asking for one
        returns None so the caller can report it cleanly rather than plot
        zeros.
        """
        import numpy as np
        from ion_gym.io.records import TrajRecord
        rec = TrajRecord(traj, cols)
        if name in rec:
            return rec[name]
        have = set(rec.keys())
        if name == "speed" and {"vx", "vy", "vz"} <= have:
            return np.sqrt(rec["vx"] ** 2 + rec["vy"] ** 2
                           + rec["vz"] ** 2)
        if name == "ke_ev" and {"vx", "vy", "vz"} <= have:
            # v in mm/us -> m/s (*1e3); KE = 1/2 m v^2 / e, m = mz*amu
            AMU, E = 1.66053906660e-27, 1.602176634e-19
            v2 = ((rec["vx"] ** 2 + rec["vy"] ** 2
                   + rec["vz"] ** 2) * 1e6)      # (m/s)^2
            return 0.5 * (float(mz) * AMU) * v2 / E
        if name == "radius" and {"x", "y"} <= have:
            return np.sqrt(rec["x"] ** 2 + rec["y"] ** 2)
        # FIELD channels: e_x/e_y/e_z are recorded base columns now; e_field
        # (magnitude) and e_axial (the transport-axis component, x) derive
        # from them exactly.
        if name == "e_field" and {"e_x", "e_y", "e_z"} <= have:
            return np.sqrt(rec["e_x"] ** 2 + rec["e_y"] ** 2
                           + rec["e_z"] ** 2)
        if name == "e_axial" and "e_x" in rec:
            return rec["e_x"]                    # transport axis is x
        return None

    def _on_hotspot_map(self, _=None):
        """Hot-spot map for the DISPLAYED run: each spatial cell = the
        MAXIMUM recorded KE it hosted (a lone hot sample cannot hide
        under overplotted cold ones), floor at the stated threshold
        (default 10 kT at the gas temperature), ceiling at the TRUE
        maximum — the extremes are the signal. Same convention as the
        notebook cell; one implementation of the physics
        via binned max-statistics."""
        import numpy as np
        if self._active not in self._runs:
            self.hotspot_note.object = "**no displayed run** — Fly first."
            return
        results = self._runs[self._active].results
        cols = self._cols or self.spec.column_names()
        if "ke_ev" not in cols:
            self.hotspot_note.object = (
                "**this run recorded no `ke_ev` channel** — add it on "
                "the Integration tab and re-fly.")
            return
        from ion_gym.io.records import TrajRecord
        n_show = int(self.w_hot_nions.value or 40)
        X, Y, K = [], [], []
        for r in results[:n_show]:
            if r.traj is None or not len(r.traj):
                continue
            rec = TrajRecord(r.traj, cols)
            X.append(rec["x"])
            Y.append(rec["y"])
            K.append(rec["ke_ev"])
        if not X:
            self.hotspot_note.object = "**displayed run holds no samples**"
            return
        X, Y, K = (np.concatenate(a) for a in (X, Y, K))
        T_k = float(getattr(self.spec.collisions, "T_k", 300.0) or 300.0)
        thr = self.w_hot_thresh.value
        thr = float(thr) if thr else 10 * 8.617333262e-5 * T_k
        cell = float(self.w_hot_cell.value or 0.12)
        try:
            from scipy.stats import binned_statistic_2d
        except ImportError as e:
            self.hotspot_note.object = (f"**scipy required for the "
                                        f"binned max-statistic**: {e}")
            return
        nx = max(4, int(round((X.max() - X.min()) / cell)))
        ny = max(4, int(round((Y.max() - Y.min()) / cell)))
        st = binned_statistic_2d(X, Y, K, statistic="max", bins=[nx, ny])
        Z = st.statistic.T
        n_hot = int(np.nansum(Z >= thr))
        import plotly.graph_objects as go
        Zp = np.where(np.isnan(Z), None, Z)
        fig = go.Figure(go.Heatmap(
            x=0.5 * (st.x_edge[:-1] + st.x_edge[1:]),
            y=0.5 * (st.y_edge[:-1] + st.y_edge[1:]),
            z=Zp, zmin=thr, zmax=float(np.nanmax(Z)),
            colorscale="Inferno",
            colorbar=dict(title=f"max KE (eV)<br>floor {thr:.3g}")))
        # electrode outlines for context, closed loops from the model
        model = getattr(self, "_model", None)
        if model is not None and hasattr(model, "potential_image"):
            from ion_gym.viz.viz_core import mask_outline_mm
            xs, ys, _, em = model.potential_image()
            px, py = [], []
            for lp in mask_outline_mm(
                    em, self.spec.geometry.mm_per_gu,
                    (float(xs[0]), float(ys[0]))):
                px += list(lp[:, 0]) + [None]
                py += list(lp[:, 1]) + [None]
            fig.add_scatter(x=px, y=py, mode="lines",
                            line=dict(color="#333", width=1.2),
                            showlegend=False, hoverinfo="skip")
        fig.update_layout(margin=dict(l=10, r=10, t=36, b=10),
                          title=(f"hot spots — {n_hot} cells >= "
                                 f"{thr:.3g} eV (cell {cell:g} mm, "
                                 f"{min(n_show, len(results))} ions)"))
        # This panel carried a live
        # scaleanchor, so a drag rectangle was RESHAPED to 1:1 -- the
        # long-thin boxes that matter here could not be drawn. Same
        # policy as every other 2-D panel: free zoom, true scale from
        # pane sizing, autoscale restores.
        from ion_gym.viz import viz_core
        try:
            viz_core.apply_zoom_policy(fig, true_scale=True, px_width=760)
        except viz_core.VizError as e:
            print(f"[hotspot] zoom policy skipped: {e}")
        self.hotspot_pane.object = fig
        self.hotspot_note.object = (
            f"**{n_hot} hot cell(s)** at or above {thr:.3g} eV "
            f"(threshold = {'user-set' if self.w_hot_thresh.value else f'10 kT at {T_k:g} K'}); "
            f"gray/empty = visited but cool or never visited.")

    def _on_thermal_assess(self, _=None):
        if self._subject_is_assembly():
            self.status.object = ("**Thermal assessment is not available for a multi-FA instrument** — it operates on the single displayed FA. Use *Set FA View* first.")
            return
        """Two-estimator temperature assessment for the active run, via
        the framework (traj_stats.two_estimator_report +
        viz_core.temperature_estimator_figure) — the SAME code path as
        notebook 05. The tail-fit percentile comes from w_therm_pctl."""
        import numpy as np
        from ion_gym.physics.traj_stats import (two_estimator_report,
                                                slowest_rf_period_us,
                                                KG_AMU)
        from ion_gym.viz.viz_core import temperature_estimator_figure
        if self._active not in self._runs:
            self.status.object = ("**no run to assess** — fly or load a "
                                  "run first.")
            return
        results = self._runs[self._active].results
        cols = self._cols or self.spec.column_names()
        if not all(c in cols for c in ("vx", "vy", "vz")):
            self.thermal_table.object = (
                "**this run has no vx/vy/vz channels** — enable velocity "
                "recording on the Integration tab and re-fly.")
            return
        # m/z per ion from each result's summary (fallback to spec)
        mz0 = float(self.spec.source.mz_list[0])
        V = {"x": [], "y": [], "z": []}
        KE = {"x": [], "y": [], "z": []}
        T = {"x": [], "y": [], "z": []}
        for r in results:
            if r.traj is None or not len(r.traj):
                continue
            mz = float(r.summary.get("mz", mz0))
            vx = self._traj_column(r.traj, "vx", cols, mz)
            vy = self._traj_column(r.traj, "vy", cols, mz)
            vz = self._traj_column(r.traj, "vz", cols, mz)
            ke = self._traj_column(r.traj, "ke_ev", cols, mz)
            # PREFER recorded per-axis KE (ke_x/ke_y/ke_z) when present:
            # some runs record those but NOT vx/vy/vz (e.g. this imported deck's
            # record_channels), leaving the velocity columns zero -> the
            # estimator saw "no motion" though the ions flew.
            # Per-axis KE is the recorded observable; recover
            # per-axis |v| from it (sign is unresolved from KE alone, but
            # T_var uses v^2 and T_slope uses KE, so the sign is immaterial
            # to both temperatures). Fall back to velocities when KE_i
            # channels are absent.
            has_keaxis = all(c in cols for c in ("ke_x", "ke_y", "ke_z"))
            v_present = bool(np.any(vx) or np.any(vy) or np.any(vz))
            if has_keaxis:
                _kx = self._traj_column(r.traj, "ke_x", cols, mz)
                _ky = self._traj_column(r.traj, "ke_y", cols, mz)
                _kz = self._traj_column(r.traj, "ke_z", cols, mz)
                # DEFENSE: runs recorded BEFORE assemble_record
                # carry ke_* schema slots that were silently ZERO-filled.
                # Preferring those zeros over real velocities reproduced
                # the "no motion" bug. Recorded KE is trusted only when it
                # is actually populated; otherwise fall through to the
                # velocity split.
                if not (np.any(_kx) or np.any(_ky) or np.any(_kz)):
                    has_keaxis = v_present is False and has_keaxis
            if has_keaxis:
                # PREFER the recorded per-axis KE — ground truth, not a
                # reconstruction (assemble_record now fills these for every
                # new run; the populated-check above shields old runs).
                kex, key, kez = _kx, _ky, _kz
                KE["x"].append(kex)
                KE["y"].append(key)
                KE["z"].append(kez)
                if v_present:
                    # real velocities recorded -> use them for T_var
                    V["x"].append(vx)
                    V["y"].append(vy)
                    V["z"].append(vz)
                else:
                    # no velocities: reconstruct signed v from KE_i so T_var
                    # is unbiased (random symmetric sign; the true velocity
                    # distribution is zero-mean symmetric — verified 296 K
                    # vs 300 true; unsigned |v| gives a biased 108 K).
                    AMU, ECH = 1.66053906660e-27, 1.602176634e-19
                    inv = 1.0 / (float(mz) * AMU)
                    _rng = np.random.default_rng(
                        abs(hash(("kesign", r.index))) % 2 ** 32)
                    for axk, kev in (("x", kex), ("y", key), ("z", kez)):
                        mag = np.sqrt(np.clip(kev, 0, None)
                                      * 2 * ECH * inv) / 1e3
                        V[axk].append(
                            mag * _rng.choice((-1.0, 1.0), size=mag.shape))
            else:
                # no per-axis KE channels: split total ke_ev by v^2 share
                s2 = vx ** 2 + vy ** 2 + vz ** 2
                with np.errstate(invalid="ignore", divide="ignore"):
                    fx = np.where(s2 > 0, vx ** 2 / s2, 0.0)
                    fy = np.where(s2 > 0, vy ** 2 / s2, 0.0)
                    fz = np.where(s2 > 0, vz ** 2 / s2, 0.0)
                V["x"].append(vx)
                V["y"].append(vy)
                V["z"].append(vz)
                KE["x"].append(ke * fx)
                KE["y"].append(ke * fy)
                KE["z"].append(ke * fz)
            tt = self._traj_column(r.traj, "t", cols, mz)
            T["x"].append(tt)
            T["y"].append(tt)
            T["z"].append(tt)
        if not V["x"]:
            self.thermal_table.object = "**no trajectories in this run.**"
            return
        V = {k: np.concatenate(v) for k, v in V.items()}
        KE = {k: np.concatenate(v) for k, v in KE.items()}
        T = {k: np.concatenate(v) for k, v in T.items()}
        m_kg = mz0 * KG_AMU
        pctl = float(self.w_therm_pctl.value)
        nb = int(self.w_therm_bins.value)
        # secular/micromotion split needs the RF period; bath from the
        # spec's collision temperature so the excess column is real.
        try:
            rf_period = slowest_rf_period_us(self.spec)
        except ValueError:
            rf_period = None            # DC-only run: no split, reported
        bath = float(getattr(self.spec.collisions, "T_k", 298.0))
        rep = two_estimator_report(V, KE, m_kg, fit_from_pctl=pctl,
                                   n_bins=nb, t_by_axis=T,
                                   rf_period_us=rf_period, bath_K=bath)
        # DIAGNOSE 'no motion': if every axis came back None, the velocity
        # arrays were empty or ~all-zero — say WHICH, and the likely cause,
        # instead of three silent dashes (the table read all '—'
        # on an imported run whose ions clearly flew). SDS carries no thermal
        # velocity by design; HS/vacuum records real velocities.
        if all(rep.get(ax) is None for ax in ("x", "y", "z")):
            import numpy as _np
            stat = []
            for ax in ("x", "y", "z"):
                a = _np.asarray(V[ax], float)
                k = _np.asarray(KE[ax], float)
                rms = float(_np.sqrt(_np.mean(a * a))) if a.size else 0.0
                kpos = int((k > 0).sum())
                kmean = float(k[k > 0].mean()) if kpos else 0.0
                stat.append(
                    "{0}: n={1}, v_rms={2:.3e} mm/us, KE>0 count={3}, "
                    "mean KE(>0)={4:.3e} eV".format(ax, a.size, rms,
                                                    kpos, kmean))
            model_kind = str(getattr(self.spec.collisions, "model", "?"))
            gas_on = bool(getattr(self.spec.collisions, "enabled", False))
            cause = ("SDS gas model carries NO thermal velocity by design "
                     "(drift + positional diffusion) — use HS for a "
                     "temperature read"
                     if (gas_on and model_kind == "sds")
                     else "velocities recorded are ~zero — check that "
                     "vx/vy/vz recording is on and the run is not "
                     "endpoint-only")
            self.thermal_table.object = (
                "**no resolvable motion on any axis.**\n\n"
                + "\n".join("- " + s for s in stat)
                + "\n\n_Likely cause: {0}._".format(cause))
            self.status.object = ("**temperature: no motion** — see the "
                                  "Analysis panel for per-axis velocity "
                                  "stats and the likely cause.")
            return
        # table
        def _f(x):
            return "—" if x is None else "{0:.0f}".format(x)
        lines = ["| axis | T_var (K) | T_sec (K) | T_mic (K) | T_mean (K) "
                 "| T_slope (K) | excess vs bath (K) | drift |",
                 "|---|---|---|---|---|---|---|---|"]
        for ax in ("x", "y", "z"):
            e = rep.get(ax)
            if e is None:
                lines.append("| {0} | — | — | — | — | — | — | (no motion) |"
                             .format(ax))
            else:
                lines.append(
                    "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |"
                    .format(ax, _f(e["T_var"]), _f(e.get("T_sec")),
                            _f(e.get("T_mic")), _f(e["T_mean"]),
                            _f(e["T_slope"]), _f(e.get("excess_K")),
                            "yes" if e["drift"] else "no"))
        note = ("\n\n_All temperatures are ABSOLUTE (K), not excess. "
                "T_var (secular) = T_sec; the full velocity spread is "
                "T_sec + T_mic, so a T_var below the {1:g} K bath is "
                "accounted for by micromotion (T_mic), NOT sub-bath "
                "cooling. **excess vs bath** = T_var - {1:g} K is the "
                "heating figure the audits quote. T_var vs T_slope "
                "disagreement flags a non-thermal tail; tail fit from "
                "the {0:g}th percentile (reliable band ~10-35%)._"
                .format(pctl, bath))
        # ride-height section: WHERE the cloud sits
        # relative to the carpets — T_mic is set by ride position, so the
        # standoff is the observable that explains the temperatures above.
        try:
            from ion_gym.physics.traj_stats import ride_height_report
            rh = ride_height_report(results, cols, self.spec)
            if rh.get("standoff_mean_mm") is not None:
                lines.append("")
                lines.append("**Ride height** (standoff from nearest "
                             "carpet; gap {0:.1f} mm):".format(rh["gap_mm"]))
                lines.append(
                    "mean {0:.2f} mm | p5 {1:.2f} | median {2:.2f} | "
                    "within 1 mm: {3:.0%} | within 0.5 mm: {4:.0%} | "
                    "mid-half of gap: {5:.0%}".format(
                        rh["standoff_mean_mm"], rh["standoff_p5_mm"],
                        rh["standoff_p50_mm"], rh["frac_within_1mm"],
                        rh["frac_within_half_mm"], rh["mid_fraction"]))
                lines.append(
                    "*mid-fraction ~1 = mid-channel (cold basin); "
                    "large within-1mm fraction = RF-riding (T_mic from "
                    "pseudopotential at the ride position).*")
            else:
                lines.append("")
                lines.append("**Ride height:** " +
                             str(rh.get("note", "unavailable")))
        except ValueError as e:
            lines.append("")
            lines.append("**Ride height:** not measurable for this "
                         "geometry — " + str(e))
        self.thermal_table.object = "\n".join(lines) + note
        try:
            fig = temperature_estimator_figure(
                V, KE, m_kg, fit_from_pctl=pctl, n_bins=nb,
                title="two temperature estimators — {0} — {2} "
                      "(tail fit {1:g}th pctl)".format(
                          self._active, pctl,
                          self.spec.drive_summary()))
        except Exception as e:
            # surface the ACTUAL failure (no silent/opaque fail) —
            # the figure renders fine on standard 3-D data in isolation, so
            # a break here is data-shape-specific and its message is the
            # diagnostic we need to see.
            import traceback
            tb = traceback.format_exc().strip().splitlines()
            self.thermal_table.object = (
                "**temperature figure failed** — {0}: {1}\n\n```\n{2}\n```"
                .format(type(e).__name__, e, "\n".join(tb[-4:])))
            self.status.object = "**temperature assessment failed** (see "\
                                 "the Analysis panel for the traceback)."
            return
        self.thermal_pane.object = fig
        self.status.object = ("**temperature assessed** for {0} "
                              "(tail pctl {1:g}).".format(self._active,
                                                          pctl))

    def _sync_analysis_mz(self):
        """Refresh the Analysis m/z-filter options from the DISPLAYED run's
        flown ions (each result's own summary mz — what actually flew, not
        the editor's mz_list). Preserves any still-valid selection; a
        selection whose m/z is absent from the new run is dropped, so a
        stale filter can never silently empty the plot."""
        opts = []
        if self._active in self._runs:
            seen = {}
            for r in self._runs[self._active].results:
                mz = r.summary.get("mz")
                if mz is not None:
                    seen[f"{float(mz):g}"] = True
            opts = sorted(seen, key=float)
        self.w_amz.options = opts
        self.w_amz.value = [v for v in self.w_amz.value if v in opts]

    @staticmethod
    def _scheme_colors(name):
        """Named qualitative palette -> colour list. The Select is the
        only caller, so an unknown name is a programming error — refuse
        with the name rather than silently recolouring."""
        import plotly.colors as pc
        pal = getattr(pc.qualitative, name, None)
        if not pal:
            raise ValueError(f"unknown colour scheme '{name}'")
        return list(pal)

    def _mz_palette(self, results, palette=None):
        """Stable m/z -> colour mapping for the displayed results: unique
        flown m/z values sorted numerically, assigned from the ACTIVE
        colour scheme (cycled past its length). Sorted assignment keeps
        a given m/z the same colour across replots of the same run and
        scheme."""
        pal = palette or self._scheme_colors(self.w_ascheme.value)
        mzs = sorted({float(r.summary["mz"]) for r in results
                      if r.summary.get("mz") is not None})
        return {mz: pal[i % len(pal)] for i, mz in enumerate(mzs)}

    def _analysis_mz_filter(self, results):
        """Apply the m/z selector: empty selection = ALL (default).
        Returns (filtered_results, label). Filtering matches each ion's
        OWN recorded m/z — the same value the tracer flew (mz_of
        authority) — never an index-based guess."""
        sel = [float(v) for v in (self.w_amz.value or [])]
        if not sel:
            return results, "all m/z"
        kept = [r for r in results
                if r.summary.get("mz") is not None
                and any(abs(float(r.summary["mz"]) - s) < 1e-9 for s in sel)]
        label = "m/z " + ", ".join(f"{s:g}" for s in sorted(sel))
        return kept, label

    def _analysis_plot_assembly(self):
        """Analysis for an ASSEMBLY flight.

        Endpoint modes read the flight's per-ion arrival records
        (x/y/z mm, tof µs, vx/vy/vz mm/µs, derived speed and ke_ev);
        trajectory mode reads the retained world-frame paths (x/y/z/
        t_us per point — assemblies do not retain per-point velocities,
        so velocity channels REFUSE BY NAME there rather than plotting
        something else). Uses the tab's own axis/mode/alpha controls;
        the caption carries the operating point (arrivals/flown).
        """
        import plotly.graph_objects as go
        per = getattr(self, "_assembly_per_ion", None)
        if not per:
            self.status.object = ("**no assembly flight to analyze** — "
                                  "press Fly with the assembly displayed "
                                  "first.")
            return
        ax, ay = self.w_ax.value, self.w_ay.value
        mode = self.w_amode.value
        _alpha = float(self.w_an_alpha.value)
        _bm = (self._assembly_doc.get("beam") or {})

        def _end_chan(p, nm):
            if nm in ("x", "y", "z", "vx", "vy", "vz"):
                return p.get(nm)
            if nm in ("t", "tof", "t_us"):
                return p.get("tof", p.get("t_us"))
            if nm == "speed":
                v = [p.get(k) for k in ("vx", "vy", "vz")]
                if any(c is None for c in v):
                    return None
                return float(np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
            if nm == "ke_ev":
                s = _end_chan(p, "speed")
                mz = p.get("mz")
                if s is None or mz is None:
                    return None
                from ion_gym.physics.collision3d import E_CHG, KG_AMU
                # mz carries the ion MASS in Da (ruled convention,
                # 2026-09-09); the former *charge factor here treated it
                # as m/z and doubled the KE of a 2+ ion. KE[eV] = E_J/e
                # regardless of charge — flight-summary parity (tracer3d).
                return (0.5 * float(mz) * KG_AMU
                        * (s * 1e3) ** 2) / E_CHG
            return "__unknown__"

        _END = "x, y, z, tof, vx, vy, vz, speed, ke_ev"
        _TRAJ = "x, y, z, t_us"
        if mode == "traj":
            tr = getattr(self, "_assembly_traces", None) or []
            if not tr:
                self.status.object = ("**no retained paths** — the trace "
                                      "policy kept none for this flight.")
                return
            _ok = {"x": "x", "y": "y", "z": "z", "t": "t_us",
                   "t_us": "t_us", "tof": "t_us"}
            if ax not in _ok or ay not in _ok:
                self.status.object = (
                    f"**{ax}/{ay} not recorded per-point for an assembly "
                    f"flight** — trajectory channels: {_TRAJ}. Endpoint "
                    f"modes additionally offer: {_END}.")
                return
            # COLOUR / TRACE BY, honoured in trajectory mode (the
            # m/z analysis plotting did not honour it — the
            # rainbow was plotly's default per-trace cycle; this branch
            # never read w_agroup at all, while its own help text
            # promised m/z "works in all three plot modes"). Traces
            # carry their flown mz; the palette and the m/z filter are
            # the SAME authorities the endpoint modes use.
            _grp = self.w_agroup.value
            _sel = [float(v) for v in (self.w_amz.value or [])]
            if _sel:
                _tr_f = [t for t in tr if t.get("mz") is not None
                         and any(abs(float(t["mz"]) - s) < 1e-9
                                 for s in _sel)]
            else:
                _tr_f = list(tr)
            _note = ""
            if _grp == "m/z" and not any(t.get("mz") is not None
                                         for t in _tr_f):
                _note = (" · m/z colouring unavailable: these paths "
                         "bank no per-ion mass (pre-fix flight) — "
                         "per-ion colours shown")
                _grp = "ion"
            _pal = self._scheme_colors(self.w_ascheme.value)
            _mzs = sorted({float(t["mz"]) for t in _tr_f
                           if t.get("mz") is not None})
            _mz_col = {m: _pal[i % len(_pal)]
                       for i, m in enumerate(_mzs)}
            _seen = set()
            fig = go.Figure()
            for _t in _tr_f:
                xs = np.concatenate([np.asarray(r[_ok[ax]], float)
                                     for r in _t.get("regions", [])])
                ys = np.concatenate([np.asarray(r[_ok[ay]], float)
                                     for r in _t.get("regions", [])])
                _kw = dict(opacity=_alpha, showlegend=False,
                           name=f"ion {_t.get('i')}")
                if _grp == "m/z" and _t.get("mz") is not None:
                    _m = float(_t["mz"])
                    _kw.update(line=dict(color=_mz_col[_m]),
                               name=f"m/z {_m:g}",
                               legendgroup=f"mz{_m:g}",
                               showlegend=_m not in _seen)
                    _seen.add(_m)
                elif _grp == "fate":
                    _lost = _t.get("fate") not in ("detected", "completed")
                    _kw.update(line=dict(
                        color="#b00020" if _lost else "#1f77b4"),
                        name=("lost" if _lost else "arrived"),
                        legendgroup=("lost" if _lost else "ok"),
                        showlegend=_lost not in _seen)
                    _seen.add(_lost)
                elif _grp == "all together":
                    _kw.update(line=dict(color=_pal[0]))
                fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines",
                                           **_kw))
            _flt = (f" · filtered to m/z "
                    f"{', '.join(f'{v:g}' for v in sorted(_sel))}"
                    if _sel else "")
            _cap = (f"assembly · trajectory · {len(_tr_f)} retained "
                    f"paths of {len(per)} flown{_flt}{_note}")
        else:
            vals = [(_end_chan(p, ax), _end_chan(p, ay)) for p in per]
            if any(v == "__unknown__" for pair in vals for v in pair):
                self.status.object = (
                    f"**{ax}/{ay} not recorded for assembly arrivals** — "
                    f"endpoint channels: {_END}; trajectory mode offers: "
                    f"{_TRAJ}.")
                return
            pairs = [(a, b) for a, b in vals
                     if a is not None and b is not None]
            _dropped = len(vals) - len(pairs)
            if not pairs:
                # WHY it is empty matters. This used to say "lost ions
                # have no arrival state" unconditionally, which was
                # actively wrong on a 200/200 flight: the arrival record
                # simply did not carry the channel. Separate the two
                # causes and name the one that applies.
                _arrived = [p for p in per if p.get("arrived")]
                _missing = sorted(
                    {nm for nm in (ax, ay)
                     if _arrived and _end_chan(_arrived[0], nm) is None})
                if _missing:
                    self.status.object = (
                        f"**assembly arrivals do not carry "
                        f"{', '.join(_missing)}** — {len(_arrived)} of "
                        f"{len(per)} ions arrived, so this is not a loss "
                        f"problem: the flight's arrival record has no such "
                        f"field. Recorded per arrival: {_END}.")
                else:
                    self.status.object = (
                        f"**no arrivals carry both {ax} and {ay}** — "
                        f"{len(_arrived)} of {len(per)} ions arrived; lost "
                        f"ions have no arrival state.")
                return
            xs = np.asarray([p[0] for p in pairs], float)
            ys = np.asarray([p[1] for p in pairs], float)
            if mode == "hist":
                fig = go.Figure(go.Histogram(x=xs, nbinsx=60,
                                             opacity=_alpha))
                fig.update_yaxes(title_text="ions")
            else:
                fig = go.Figure(go.Scattergl(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=5, opacity=_alpha),
                    showlegend=False))
            _cap = (f"assembly · endpoints · {len(pairs)} arrivals of "
                    f"{len(per)} flown"
                    + (f" · {_dropped} lacking a channel (named, not "
                       f"silently dropped)" if _dropped else ""))
        fig.update_xaxes(title_text=ax)
        if mode != "hist":
            fig.update_yaxes(title_text=ay)
        fig.update_layout(height=560, margin=dict(l=50, r=20, t=40, b=44),
                          title=dict(text=_cap, x=0.01,
                                     font=dict(size=12)))
        self.analysis_pane.object = fig
        self.status.object = f"**analysis drawn** — {_cap}"

    def _on_analysis_plot(self, _=None):
        if self._subject_is_assembly():
            # Wired, not refused. Assembly flights record different
            # quantities
            # than single-stage runs, so this is its own honest branch.
            return self._analysis_plot_assembly()
        """Scatter/line of any recorded channel vs any other for the active
        run. Trajectory mode uses every stored point; endpoint mode uses one
        point per ion."""
        import plotly.graph_objects as go
        import numpy as np
        if self._active not in self._runs:
            self.status.object = "**no run to plot** — Fly some ions first."
            return
        results = self._runs[self._active].results
        results, _mz_label = self._analysis_mz_filter(results)
        if not results:
            self.status.object = (
                "**m/z filter matched no ions in this run** — the selected "
                "value(s) were not flown here; clear the filter (empty = "
                "all) or pick from the offered options.")
            return
        cols = self._cols or self.spec.column_names()
        ax, ay = self.w_ax.value, self.w_ay.value
        # a per-run m/z for ke_ev (derived); take the active run's first ion
        _run = self._runs[self._active]
        _mz = 1.0
        for _r in _run.results:
            _mz = float(_r.summary.get("mz", 1.0))
            break
        # derivable-or-present: speed/ke_ev/radius from position+velocity;
        # e_field (|E|) and e_axial from the recorded e_x/e_y/e_z columns.
        _DERIVED = ("speed", "ke_ev", "radius", "e_field", "e_axial")

        def _have(nm):
            return nm in cols or nm in _DERIVED
        if not _have(ax) or not _have(ay):
            self.status.object = (f"**{ax}/{ay} not plottable for this run** "
                                  f"— have: {', '.join(cols)}; derivable: "
                                  f"{', '.join(_DERIVED)}")
            return
        mode = self.w_astyle.value
        grp = self.w_agroup.value
        # Read the ANALYSIS tab's own slider — before the
        # collision fix this read whichever object had last claimed
        # self.w_alpha, so the visible Analysis slider changed nothing.
        _alpha = float(self.w_an_alpha.value)
        _scheme = self._scheme_colors(self.w_ascheme.value)
        if grp == "m/z" and not any(r.summary.get("mz") is not None
                                    for r in results):
            self.status.object = (
                "**this run carries no per-ion m/z** — it was flown or "
                "saved before per-ion m/z recording (v280). Re-fly to get "
                "m/z colouring; the other colour modes work on this run.")
            return
        fig = go.Figure()
        if self.w_amode.value in ("hist", "hist_traj"):
            # histogram of the x-axis quantity: per-ion ENDPOINTS ("hist")
            # or EVERY recorded sample of every ion ("hist_traj" -- a
            # residence profile along the chosen quantity; by
            # design trajectories are Analysis, not Impact). One
            # aligned per-SAMPLE collection serves both.
            _samples = self.w_amode.value == "hist_traj"
            vals, fates, mz_vals = [], [], []
            for r in results:
                if r.traj is not None and len(r.traj):
                    _rmz = r.summary.get("mz", _mz)
                    _ca = self._traj_column(r.traj, ax, cols, _rmz)
                    if _ca is None:
                        continue
                    take = _ca if _samples else _ca[-1:]
                    vals.extend(float(v) for v in take)
                    fates.extend([r.summary.get("kind", 2)] * len(take))
                    mz_vals.extend([float(_rmz) if _rmz is not None
                                    else float("nan")] * len(take))
            nb = int(self.w_bins.value)
            if grp == "m/z":
                cmap = self._mz_palette(results)
                for mz, cname in cmap.items():
                    mv = [v for v, m in zip(vals, mz_vals) if m == mz]
                    if mv:
                        fig.add_histogram(x=mv, nbinsx=nb, opacity=_alpha,
                                          name=f"m/z {mz:g}",
                                          marker=dict(color=cname))
                fig.update_layout(barmode="overlay")
            elif grp == "fate":
                for code, cname in _FATE_COLOR.items():
                    fv = [v for v, f in zip(vals, fates) if f == code]
                    if fv:
                        fig.add_histogram(x=fv, nbinsx=nb, opacity=_alpha,
                                          name=_FATE_NAME.get(code, str(code)),
                                          marker=dict(color=cname))
                fig.update_layout(barmode="overlay")
            else:
                fig.add_histogram(x=vals, nbinsx=nb, opacity=_alpha,
                                  marker=dict(color=_scheme[0]))
            fig.update_layout(height=460, xaxis_title=self._chan_label(ax),
                              yaxis_title="count",
                              title=f"{ax} distribution ({len(vals)} ions, "
                              f"{nb} bins, {_mz_label}) — {self._active}",
                              margin=dict(l=55, r=15, t=45, b=45))
            self.analysis_pane.object = fig
            self.status.object = f"**histogram** of {ax} ({nb} bins)."
            return
        if self.w_amode.value == "end":
            xs, ys, fates, mzs_end = [], [], [], []
            for r in results:
                if r.traj is not None and len(r.traj):
                    _ca = self._traj_column(r.traj, ax, cols,
                                            r.summary.get("mz", _mz))
                    _cb = self._traj_column(r.traj, ay, cols,
                                            r.summary.get("mz", _mz))
                    if _ca is None or _cb is None:
                        continue
                    # true point of stoppage when the summary has it (a
                    # truncated record's last row is NOT where the ion
                    # ended)
                    _sx = r.summary.get(f"{ax}_end")
                    _sy = r.summary.get(f"{ay}_end")
                    xs.append(float(_sx) if _sx is not None
                              else float(_ca[-1]))
                    ys.append(float(_sy) if _sy is not None
                              else float(_cb[-1]))
                    fates.append(r.summary.get("kind", 2))
                    mzs_end.append(float(r.summary.get("mz", _mz)))
            if grp == "m/z":
                cmap = self._mz_palette(results)
                for mz, cname in cmap.items():
                    px = [x for x, m in zip(xs, mzs_end) if m == mz]
                    py = [y for y, m in zip(ys, mzs_end) if m == mz]
                    if px:
                        fig.add_scatter(x=px, y=py, mode="markers",
                                        opacity=_alpha,
                                        name=f"m/z {mz:g}",
                                        marker=dict(color=cname))
            elif grp == "fate":
                for code, cname in _FATE_COLOR.items():
                    px = [x for x, f in zip(xs, fates) if f == code]
                    py = [y for y, f in zip(ys, fates) if f == code]
                    if px:
                        fig.add_scatter(x=px, y=py, mode="markers",
                                        opacity=_alpha,
                                        name=_FATE_NAME.get(code, str(code)),
                                        marker=dict(color=cname))
            else:
                fig.add_scatter(x=xs, y=ys, mode="markers", opacity=_alpha,
                                marker=dict(color=_scheme[0]))
        else:
            allx, ally = [], []
            mz_groups = {}
            n_ion_traces = 0
            # WebGL renderer for trajectory plots: an SVG scatter chokes the
            # browser at ~10^5 points / many traces (plotting all ions
            # of the funnel hangs — ~100 ions x ~1000 pts). scattergl draws
            # the same data on the GPU and stays responsive into the millions
            # of points. Per-ion mode also caps the number of separate traces
            # (each trace is browser overhead) and decimates within a trace.
            _MAX_ION_TRACES = 400
            _MAX_PTS_PER_TRACE = 5000
            for r in results:
                if r.traj is None or not len(r.traj):
                    continue
                x = self._traj_column(r.traj, ax, cols,
                                      r.summary.get("mz", _mz))
                y = self._traj_column(r.traj, ay, cols,
                                      r.summary.get("mz", _mz))
                if x is None or y is None:
                    continue
                if grp == "ion":
                    if n_ion_traces >= _MAX_ION_TRACES:
                        continue          # cap traces; reported below
                    if len(x) > _MAX_PTS_PER_TRACE:
                        step = len(x) // _MAX_PTS_PER_TRACE + 1
                        x = x[::step]
                        y = y[::step]
                    fig.add_scattergl(x=x, y=y, mode=mode,
                                      line=dict(width=1), opacity=_alpha,
                                      name=f"ion {r.index}", showlegend=False)
                    n_ion_traces += 1
                elif grp == "m/z":
                    mz_groups.setdefault(
                        float(r.summary.get("mz", _mz)), []).append((x, y))
                else:
                    allx.append(x)
                    ally.append(y)
            if grp == "m/z" and mz_groups:
                # one trace PER m/z (few traces, GPU-drawn): each ion's
                # points join its OWN flown-m/z series; NaN separators keep
                # per-ion lines from joining across ions in line styles.
                cmap = self._mz_palette(results)
                for mz in sorted(mz_groups):
                    seg = []
                    for x, y in mz_groups[mz]:
                        if len(x) > _MAX_PTS_PER_TRACE:
                            step = len(x) // _MAX_PTS_PER_TRACE + 1
                            x = x[::step]
                            y = y[::step]
                        seg.append((x, y))
                    cx = np.concatenate(
                        [np.append(x, np.nan) for x, _ in seg])
                    cy = np.concatenate(
                        [np.append(y, np.nan) for _, y in seg])
                    fig.add_scattergl(
                        x=cx, y=cy, mode=mode, line=dict(width=1),
                        opacity=_alpha, marker=dict(size=3),
                        name=f"m/z {mz:g}",
                        line_color=cmap.get(mz), showlegend=True)
            elif grp not in ("ion", "m/z") and allx:
                cx = np.concatenate(allx)
                cy = np.concatenate(ally)
                fig.add_scattergl(x=cx, y=cy, mode="markers",
                                  marker=dict(size=3, color=_scheme[0],
                                              opacity=_alpha),
                                  showlegend=False)
        fig.update_layout(height=460, colorway=_scheme,
                          xaxis_title=self._chan_label(ax),
                          yaxis_title=self._chan_label(ay),
                          title=f"{ay} vs {ax} ({_mz_label}) — {self._active}",
                          margin=dict(l=55, r=15, t=45, b=45))
        self.analysis_pane.object = fig
        if self.w_amode.value == "traj" and not any(
                r.traj is not None for r in results):
            self.status.object = ("**this run stored no trajectories** "
                                  "(storage was off) — only endpoints/fates "
                                  "are available.")
        elif (self.w_amode.value == "traj" and grp == "ion"
              and n_ion_traces >= _MAX_ION_TRACES):
            self.status.object = (
                f"**plotted {ay} vs {ax}** — showing the first "
                f"{_MAX_ION_TRACES} ions (trace cap for responsiveness). "
                "Use 'all together' to see every ion's points in one series, "
                "or 'fate' to colour by outcome.")
        else:
            self.status.object = f"**plotted** {ay} vs {ax}."

    def _poll_thread(self):
        # Fallback poller for the ion flight when there is no server session
        # to schedule _tick on. The exit is DEFINITE: it ends when the run
        # finishes (_handle.done) OR the worker thread is no longer alive.
        # Relying on .done alone could spin forever if the worker died
        # before setting its final progress (no unbounded waits) —
        # so we also check the thread liveness and tick once more to flush
        # whatever final state exists.
        h = self._handle
        while (h is not None and not h.done
               and h._thread is not None and h._thread.is_alive()):
            self._tick()
            h._thread.join(timeout=0.2)   # bounded wait; returns on finish
        self._tick()

    def _tick(self):
        h = self._handle
        if h is None:
            return
        if self._stop:
            h.stop("user pressed Stop")
        p = h.latest
        if p is not None:
            self._fly_chip("progress", done=p.done, total=p.total)
            if getattr(self, "workers_line", None) is not None:
                _queued = max(0, p.total - p.done - p.in_flight)
                self.workers_line.object = (
                    f"**{p.n_workers}** thread(s) · **{p.in_flight}** "
                    f"in flight · **{_queued}** queued · "
                    f"**{p.done}/{p.total}** done · {p.rate_hz:.1f} "
                    f"ions/s")
            # LIVE-REDRAW THROTTLE (all three
            # stale-pulse snapshots caught THE LOOP inside _redraw's
            # plotly construction while the flight worker ran fine — at
            # 1000 ions a 200 ms tick spent seconds per redraw, starving
            # the websocket so even the chip never flushed). The chip and
            # the status line are the per-tick heartbeat; the heavy
            # figure/stats/publish pipeline is DUTY-CYCLED: the clock
            # stamps at redraw COMPLETION and the next one waits
            # max(LIVE_REDRAW_MIN_S, LIVE_REDRAW_DUTY x last cost) — a
            # fixed interval alone reopened the gate immediately whenever
            # one redraw ran longer than the interval (measured ~5 s at
            # 120 ions with field shading), saturating the loop anyway.
            # The FINAL redraw below is ungated — the stored run always
            # draws.
            import time as _t
            _cost = getattr(self, "_live_draw_cost", 0.0)
            if (_t.time() - getattr(self, "_live_draw_ts", 0.0)
                    >= max(LIVE_REDRAW_MIN_S, LIVE_REDRAW_DUTY * _cost)):
                _d0 = _t.time()
                self._redraw(p.results, live=True)
                self._live_draw_cost = _t.time() - _d0
                self._live_draw_ts = _t.time()
            self.status.object = (
                f"**{p.done}/{p.total}** ions — {p.rate_hz:.0f}/s"
                + (f", ETA {p.eta_s:.0f}s" if p.done < p.total else "")
                + (f" — STOPPED: {p.stop_reason}" if p.stopped_early
                   else ""))
        if h.done:
            # A worker error is a FINISHED run that says what killed it
            # (the death was console-only and the
            # app wedged). Status carries the exception; partial results,
            # if any ions flew, are stored below as usual.
            if h.error is not None:
                self._fly_chip("error", note=type(h.error).__name__)
                tb_tail = "\n".join(
                    (h.error_tb or "").strip().splitlines()[-3:])
                self.status.object = (
                    f"**flight error:** {type(h.error).__name__}: "
                    f"{h.error}\n\n```\n{tb_tail}\n```\n"
                    f"Fix the source/spec and press Fly again — the run "
                    f"machinery has recovered.")
            fin = h.final
            if fin is not None:
                name = f"{self.spec.name} [{time.strftime('%H:%M:%S')}]"
                self._runs[name] = fin
                self._active = name
                self._sync_analysis_mz()
                self.w_runsel.options = list(self._runs.keys())
                self.w_runsel.value = name
                self._redraw(fin.results)
                plate = sum(1 for r in fin.results
                            if r.summary.get("kind") == 0)
                self._fly_chip("stopped" if fin.stopped_early
                               else "done",
                               done=fin.done, total=fin.total,
                               note=(fin.stop_reason
                                     if fin.stopped_early else None))
                # INFORMATIVE COMPLETION LINE (the completion
                # line must actually inform): the fate
                # breakdown by the legend's own names, wall time, and
                # the median tof when the summaries bank one — the
                # numbers a fly is run FOR — plus the /flight stamp
                # the publish tap just recorded (it no longer
                # append-chains onto status). Stats card carries the
                # full detail; this is the headline.
                _kinds = {}
                for r in fin.results:
                    k = r.summary.get("kind")
                    if k is not None:
                        _kinds[int(k)] = _kinds.get(int(k), 0) + 1
                _fates = ", ".join(
                    f"{n} {_FATE_NAME.get(k, f'kind {k}')}"
                    for k, n in sorted(_kinds.items(), key=lambda kv: -kv[1]))
                _tofs = [float(r.summary["tof"]) for r in fin.results
                         if r.summary.get("tof") is not None]
                _tof = (f" · median tof {float(np.median(_tofs)):.4g} µs"
                        if _tofs else "")
                _stamp = getattr(self, "_last_flight_stamp", None)
                _bank = f" · /flight #{_stamp}" if _stamp is not None else ""
                _tail = (f" ({_fates}){_tof} · stored as '{name}' "
                         f"(reloadable){_bank}")
                if fin.stopped_early:
                    # A stopped flight SAYS SO,
                    # first word, not as a suffix on "done".
                    self.status.object = (
                        f"**STOPPED after {fin.done}/{fin.total} ions** "
                        f"({fin.stop_reason}) — {plate} on metal/bounds"
                        + _tail)
                else:
                    self.status.object = (
                        f"**done: {fin.done} ions in "
                        f"{fin.elapsed_s:.1f} s**" + _tail)
            if self._pcb is not None:
                try:
                    self._pcb.stop()
                except (RuntimeError, ValueError):
                    pass        # panel guard: callback already stopped
                self._pcb = None

    def _set_plane(self, plane):
        """Quick view-plane switch (the top-row buttons). Sets the Display
        tab's plane widget, which triggers a redraw of the active run."""
        # A planar cross-section only lacks z-structure if the ions don't
        # drift axially. With axial KE (direction has a z-component) the
        # geometry IS a transport section and xz/yz show the wiggle down
        # the axis with the rods as channel walls — so only warn when
        # there's no axial motion (e.g. an einzel).
        s = self.spec.source
        axial = (len(s.direction) > 2 and abs(s.direction[2]) > 1e-9
                 and (s.ke_hi > 0 or s.temperature_k > 0))
        _rt = build_route(self.spec)
        from ion_gym.physics.sim_build import has_3d_transport
        if has_3d_transport(self.spec) and plane in ("xz", "yz"):
            # route predicate, not a builder name: any device
            # whose display field is a 2-D cross-section of 3-D transport
            # gets this explainer
            self.status.object = (
                f"**{plane} view** — z is the transport axis; ions travel "
                f"down z while the transverse section confines them "
                f"(field shading lives in the xy cross-section).")
        elif (plane in ("xz", "yz") and _rt.coords == "xyz"
                and _rt.field_dims == 2 and not axial):
            self.status.object = (
                f"**note:** this is a planar (x-y) geometry with no axial "
                f"drift — z is the uniform direction, so the {plane} view "
                f"carries no structure. Use xy here.")
        self.w_plane.value = plane
        # THE SUBJECT SURVIVES A PLANE CHANGE. This
        # used to end here, having already drawn a single stage further
        # up -- so choosing xy/xz/yz while looking at the whole assembly
        # silently returned the last sub-stage, and the only way back was
        # the Config tab. The plane now says HOW to draw; the subject says
        # WHAT. Guarded on the assembly case so a plain single-FA session
        # takes exactly its previous path and pays nothing.
        if self._subject_is_assembly():
            self._redraw_subject()

    def _on_stop(self, _=None):
        self._stop = True
        # A long FIELD SOLVE runs on a background thread through the
        # multigrid solver, which polls a cooperative stop flag at every
        # V-cycle and between bases (multigrid3d._STOP). Stop used to wire
        # ONLY to the fly handle, so during a solve (handle None/done) the
        # button did nothing. Signal BOTH: the solver
        # halts at its next cycle boundary and raises SolveInterrupted,
        # caught in the solve thread and shown as "stopped" (not an error).
        from ion_gym.physics import multigrid3d
        multigrid3d.request_stop()
        solving = self._handle is None or self._handle.done
        if self._handle is not None and not self._handle.done:
            self._handle.stop("user pressed Stop")
            # the press is ACKNOWLEDGED NOW; the halt lands after the
            # ion currently integrating finishes (per-ion check in the
            # driver). Silence here read as a dead button.
            self.status.object = ("**stopping** — halting after the "
                                  "current ion...")
        elif solving:
            self.status.object = ("**stopping** — halting the field solve "
                                  "at the next cycle boundary...")

    # ------------------------------------------------- config actions
    def _on_load_example(self, _=None):
        name = self.w_examples.value
        specs = _example_specs()
        if name in specs:
            if self._handle is not None and not self._handle.done:
                self._handle.stop("loading new spec")
            try:
                # spec construction can do real work (an STL example
                # generates + reads meshes via trimesh); a failure here
                # must surface as a status message, not crash the session.
                # ONE boundary for the whole load: this used
                # to be TWO consecutive broad handlers -- the builder call,
                # then _rebuild_for_new_spec -- reporting the IDENTICAL
                # message.  A duplicated boundary is a K4 defect and put
                # sim_app one over its except-policy budget.  Merged;
                # behaviour preserved: self.spec is still assigned only
                # after the builder succeeds, and nothing followed the
                # second handler.
                new_spec = specs[name]()
                # A NEW SPEC DROPS THE OLD INSTRUMENT. This was
                # once on _on_apply_json only; the
                # example button assigns self.spec directly and was missed,
                # so loading a single FA after viewing an instrument left
                # _assembly_specs populated and _assembly_stage still on
                # WHOLE_ASSEMBLY. _build_controls then kept the stage
                # machinery enabled for stages the loaded model does not
                # have, and the view refused with "select an FA" on a spec
                # that has no FAs to select. Same stale-state defect,
                # one path down.
                #
                # ORDER DIFFERS FROM _on_apply_json ON PURPOSE. There the
                # clear is unconditional and BEFORE the branch, because the
                # incoming document may be either kind and they must not
                # mix. Here the builder can fail and the handler returns
                # with self.spec untouched, so clearing first would drop a
                # working instrument on behalf of a load that never
                # happened. Clear once the new spec exists, immediately
                # before it is adopted.
                self._clear_assembly_state()
                _prev_spec = self.spec
                self.spec = new_spec
                try:
                    self._rebuild_for_new_spec(solve=False)
                except Exception:
                    # NEVER leave the app torn (spec swapped, controls
                    # stale): a later recompute would crash far from
                    # the cause (IndexError in _sync_spec, Brian
                    # 2026-09-11). Roll back to the working spec,
                    # rebuild IT, and surface the real traceback on
                    # the console as well as the status pane.
                    import traceback as _tb
                    _tb.print_exc()
                    self.spec = _prev_spec
                    self._rebuild_for_new_spec(solve=False)
                    raise
            except Exception as e:
                self.status.object = (
                    f"**couldn't load '{name}':** {e} — previous "
                    f"spec restored; full traceback on the console")
                return

    def _rebuild_for_new_spec(self, solve=False):
        # A NEW SPEC IS A NEW SUBJECT: retire flight-derived overlays
        # (detections/impacts, assembly traces, live figure bank) so the
        # previous instrument's arrivals cannot draw on this one
        # (same rule as the
        # assembly-doc load path).
        self._clear_run_overlays("spec load")
        # Rebuild controls for the new electrode set. We do NOT solve the
        # field here by default: a new geometry means a full native solve
        # (~seconds) that would block the UI thread and make the app look
        # hung (clicking Load again just queues another solve). Instead we
        # show the geometry outline immediately and let the user press
        # 'Recompute field' or 'Fly' to trigger the solve.
        # A NEW SPEC IS A NEW VIEW (loading an
        # imported deck kept the PREVIOUS example's zoom — "first
        # view doesn't cover the width, autoscale looks weird"). The
        # uirevision counter bumps on plane changes but nothing bumped it
        # on a SPEC change, so plotly faithfully preserved a viewport
        # sized for the old geometry. Bump here: every load autoscales.
        self._view_rev = getattr(self, "_view_rev", 0) + 1
        self._build_controls()
        # reflect the loaded scene's declared mirror into the control (so a
        # SLIM example that ships mirror='y' shows y ticked, and editing it
        # round-trips); inert for non-scene specs.
        _sc = getattr(self.spec, "scene", None)
        if _sc and isinstance(_sc.get("grid"), dict):
            self.w_mirror.value = [a for a in "xyz"
                                   if a in (_sc["grid"].get("mirror") or "")]
        else:
            _sym = getattr(self.spec.geometry, "symmetry", None)
            _pl = getattr(_sym, "planes", {}) if _sym else {}
            self.w_mirror.value = [a for a in "xyz"
                                   if (_pl or {}).get(a) == "mirror"]
        self.w_json.value = self.spec.to_json()
        self._active = None
        self._model = None            # force geometry preview until re-solved
        self._scene = None             # scene lives and dies with the model
        self._built = None             # drop the reuse cache: new geometry
        # LARGE-GEOMETRY FEEDBACK: rendering a big json
        # (e.g. surround SLIM) takes 10-30 s with no sign of life. Set the
        # plot pane's loading overlay + an honest status FIRST, then run
        # the heavy draw on the NEXT document tick so the browser paints
        # the spinner before the render blocks. Falls back to an immediate
        # draw when no live document exists (tests, scripted use).
        _n_ele = len(getattr(self.spec.geometry, "electrodes", []) or [])
        self.status.object = (
            f"**loading {self.spec.name}…** rendering geometry "
            f"({_n_ele} electrodes) — large models can take ~10-30 s.")
        self.pane.loading = True

        def _finish_render():
            try:
                if solve:
                    self._draw_background()
                else:
                    self._draw_geometry_only()
                self.status.object = (
                    f"**loaded:** {self.spec.name} — press *Recompute "
                    f"field* to solve, or *Fly* to run.")
            except Exception as e:
                # the render failing must SAY so, not spin forever
                self.status.object = (
                    f"**geometry render failed** — "
                    f"{type(e).__name__}: {e}")
            finally:
                self.pane.loading = False

        _doc = None
        try:
            import panel as pn
            _doc = getattr(pn.state, "curdoc", None)
        except (RuntimeError, AttributeError, ImportError):
            # AUDITED: narrowed environment probe.
            _doc = None
        if _doc is not None and hasattr(_doc, "add_next_tick_callback"):
            _doc.add_next_tick_callback(_finish_render)
        else:
            _finish_render()

    _EL_PALETTE = V.EL_PALETTE   # single source (K10): viz_core owns it

    def _draw_geometry_only(self):
        """Show just the electrode outlines for the current spec, in the
        SELECTED view plane, without solving the field — instant, so loading
        never blocks and the geometry can be checked in xy/xz/yz first. For
        the 3-D SLIM the transport (xz/yz) views use the exact imported
        outlines; xy uses the across×gap footprint raster."""
        try:
            import plotly.graph_objects as go
            g = self.spec.geometry
            _, _, _, _, la, lb = self._plane_cols()
            fig = go.Figure()
            # pre-solve 3-D transport geometry, dispatched by the ROUTE
            # (sim_build.preview_masks3d) — the `is_slim3d` special case
            # that sat here is gone; a new 3-D builder gets a
            # preview by adding a route entry, not by editing the UI
            from ion_gym.physics.sim_build import (build_route,
                                                   preview_masks3d)
            mk3 = preview_masks3d(self.spec)

            # ROUTE-DISPATCHED xy footprint (found on a
            # native SLIM spec): the 2-D shape raster below is valid exactly
            # for routes whose shapes ARE x-y cross-sections (planar, rz,
            # the slim3d confinement plane). A shapes3d spec carries
            # EXTRUDED shapes — possibly non-z axes, possibly
            # partial-depth cutouts — so its xy view must come from the
            # SAME 3-D masks the solve uses (mk3, single source of
            # truth), like every other plane. Before this dispatch the
            # 2-D raster refused loudly here (electrode_mask's extrude
            # guard) — correct refusal, wrong path.
            _is_shapes3d = build_route(self.spec).builder == "shapes3d"
            if V.plane_of(la, lb) == "xy" and not _is_shapes3d:
                # transverse footprint: raster the electrode shapes
                # (counts through THE counting function —
                # identical for conforming decks, refusal otherwise)
                # NODES, not cells. This used gu_cells and then rastered
                # `arange(n)`, which is one point SHORT in each direction:
                # a domain of N cells has N+1 nodes, so the preview
                # stopped a full cell inside the solved domain. Measured
                # Measured against anchored_grid, the axes the solve
                # actually builds: oa_12plate drew 929 x 400 spanning
                # x[0, 92.8] where the solve is 930 x 401 spanning
                # x[0, 92.9]; other planar decks
                # were short by one node on both axes too. A preview that
                # draws a different domain than the one solved is the
                # "displayed equals solver input" invariant broken, even
                # though it only ever looked like a slightly cropped
                # picture. This block had never executed under test (the
                # v436 uncertainty register listed it as NOT EXERCISED).
                from ion_gym.io.lattice import gu_nodes
                h = g.mm_per_gu
                nx = gu_nodes(g.width_mm, h, axis="x",
                              what="width_mm domain extent")
                ny = gu_nodes(g.height_mm, h, axis="y",
                              what="height_mm domain extent")
                # DECK FRAME, not kernel frame (signed-frame
                # decks previewed as a SECOND ladder
                # offset from the solved one, looking like electrodes
                # "extending to infinity"). Shapes are stored in the
                # deck's signed frame -- e.g. CAP at x_mm = -226.4 with
                # origin_mm = [-226.4, -19.9] -- so a raster starting at
                # 0 both mis-samples electrode_mask and draws the result
                # a whole origin off the solved view. origin_mm is [0,0]
                # on every legacy deck, so this is a no-op for them.
                # None means "no anchor declared" (the STARTUP default
                # spec ships that way) and is the default anchor, not a
                # crash — long pre-existing, found in a
                # startup report.
                _og = g.origin_mm if g.origin_mm is not None else (0.0,
                                                                   0.0)
                ox, oy = (float(_og[0]), float(_og[1]))
                xs = ox + np.arange(nx) * h
                ys = oy + np.arange(ny) * h
                # plane_grid_views, not meshgrid: zero-copy
                # views for a pure comparison, and it REFUSES a runaway
                # grid with the numbers rather than dying inside numpy.
                from ion_gym.physics.raster2d import (electrode_mask,
                                                      plane_grid_views)
                X, Y = plane_grid_views(xs, ys, "preview footprint")
                ele = np.zeros((nx, ny), bool)
                for el in g.electrodes:
                    # grids draw DASHED (below), not as solid metal: the
                    # raster is "what the solver treats as metal", and a
                    # grid is not that -- preview and solved view now tell
                    # the same story about them.
                    if el.shapes and not getattr(el, "is_grid", False):
                        ele |= electrode_mask(el, X, Y)
                # r-z geometry is described on the HALF-plane (y = radius >= 0)
                # but it IS a body of revolution: the SOLVED view mirrors it
                # across the axis (see viz_core.el_mask_fills /
                # potential_image), so
                # the PREVIEW must mirror it too. It did not -- which is why
                # a freshly loaded funnel showed only its top half in a
                # half-height box (aspect wrong), and snapped to the full
                # picture the moment you solved. Two display paths, one
                # convention: they agree now.
                ys_v, ele_v = ys, ele
                if g.symmetry.coords == "rz":
                    ys_v = np.concatenate([-ys[::-1], ys[1:]])
                    ele_v = np.concatenate([ele[:, ::-1], ele[:, 1:]], axis=1)
                if ele_v.any():
                    V.metal_boundary(fig, xs, ys_v, ele_v)
                V.grid_overlays(fig, self.spec, la, lb, label=bool(self.w_ellabel.value))
                V.station_overlays(fig, self.spec, la, lb,
                                   label=bool(self.w_ellabel.value))
            elif mk3 is not None:
                # transport view: NAMED electrode voxels straight from the
                # solve masks (single source of truth — display == solved
                # metal), drawn by the ONE renderer with per-electrode
                # colours + labels, exactly like every other example
                masks3, h3, org3, mir3 = mk3
                _st = self._el_style()
                V.el_mask_fills(fig, masks3, h3, "xyz", la, lb,
                                alpha=_st["alpha"], fill=_st["fill"],
                                label=_st["label"], palette=_st["palette"],
                                origin=org3, mirror=mir3)
            elif (V.plane_of(la, lb) == "yz"
                  and self.spec.geometry.symmetry.coords == "rz"):
                # axisymmetric end-on view: concentric ring circles
                if V.rz_rings(fig, self._model, self.spec.geometry,
                              **self._el_style()):
                    fig.update_layout(
                        xaxis_title="y (mm)",
                        yaxis=dict(title="z (mm)"),
                        margin=dict(l=50, r=10, t=30, b=40),
                        uirevision=self._uirev(la, lb, "geo"))
                    self._finish_preview(fig, "y", "z")
                    return
            # per-plane autoscale (transverse spans geometry; transport
            # spans the masks' true axial extent when the domain rule has
            # no answer)
            _zext = None
            if mk3 is not None:
                _m3, _h3, _o3, _mir3 = mk3
                _nz = next(iter(_m3.values())).shape[2]
                # UNFOLD-AWARE (found on a surround-SLIM load
                # report): the stored masks are the FOLDED half on a
                # declared-mirror axis, but el_mask_fills draws the
                # unfold — so the extent must count the full body
                # ((n-1)·h per half, node extent not node count), or the
                # +z board sits outside the initial view until manual
                # autoscale.
                _span = (_nz - 1) * _h3
                if _mir3 and "z" in _mir3:
                    _span *= 2
                _zext = (_o3[2], _o3[2] + _span)

            def _rng(lbl):
                dr = self._domain_range(lbl)      # ONE rule (mirrors incl.)
                # dr WINS whenever it has an answer: the old
                # `_zext`-override preferred the stored-half extent over
                # the canonical [-H,+H] frame on a declared z-mirror —
                # the exact range-vs-drawing disagreement _domain_range's
                # own docstring exists to prevent. The mask extent is the
                # FALLBACK for depth==0 z-transport geometries, where the
                # body's z extent is not in the spec and dr is honestly
                # None pre-solve.
                if dr is not None:
                    lo, hi = dr
                elif lbl == "x":
                    lo, hi = 0.0, g.width_mm
                elif lbl == "y":
                    lo, hi = 0.0, g.height_mm
                elif lbl == "z" and _zext is not None:
                    lo, hi = _zext
                else:
                    return None
                pad = 0.04 * (hi - lo)
                return [lo - pad, hi + pad]

            fig.update_layout(
                height=560, xaxis_title=f"{la} (mm)",
                yaxis=dict(title=f"{lb} (mm)"), dragmode="zoom",
                margin=dict(l=50, r=10, t=30, b=40),
                # ONE autoscale authority (switching
                # views kept the stale zoom): the plane-string-only
                # revision this replaced is the exact documented _uirev
                # failure mode — RETURNING to a plane reused its string
                # and plotly restored the old zoom.
                uirevision=self._uirev(la, lb, "geo"))
            self._finish_preview(fig, la, lb, _rng(la), _rng(lb))
        except Exception as e:
            # the STATUS names the failure; the CONSOLE keeps the
            # traceback (a startup report where the status
            # line alone buried where the None came from).
            import traceback
            traceback.print_exc()
            self.status.object = f"**geometry preview error:** {e}"

    def _on_add_rf_group(self, _=None):
        """Add a drive group with a chosen waveform. Rebuilds the Voltages
        tab so the new group's amp/freq/phase/waveform row and the
        electrode multi-select pick it up."""
        from ion_gym.io.sim_spec import RFGroupSpec
        name = (self.w_grp_new.value or "").strip()
        if not name:
            self.status.object = "**name the drive group first**"
            return
        self._sync_spec()
        s = self.spec
        if any(g.name == name for g in s.geometry.rf_groups):
            self.status.object = f"**drive group {name!r} already exists**"
            return
        wave = self.w_grp_wave.value
        if wave not in WAVEFORM_OPTS:
            self.status.object = f"**unknown waveform {wave!r}**"
            return
        s.geometry.rf_groups.append(
            RFGroupSpec(name=name, waveform=wave, amplitude_v=0.0))
        self.spec = s
        self.w_json.value = s.to_json()
        self._rebuild_for_new_spec()
        self.status.object = (
            f"**drive group {name!r} ({wave}) added** — set its amplitude/"
            f"frequency/phase, then assign electrodes to it below.")

    def _on_add_dc_group(self, _=None):
        """Add a DC group. Rebuilds the Voltages tab so the new group's
        in/out and the per-electrode membership dropdowns pick it up."""
        from ion_gym.io.sim_spec import DCGroupSpec
        name = (self.w_dcg_new.value or "").strip()
        if not name:
            self.status.object = "**name the DC group first**"
            return
        self._sync_spec()
        s = self.spec
        if any(g.name == name for g in s.geometry.dc_groups):
            self.status.object = f"**DC group {name!r} already exists**"
            return
        kind = getattr(self, "w_dcg_kind", None)
        uniform = bool(kind and kind.value == "uniform")
        s.geometry.dc_groups.append(DCGroupSpec(name=name, uniform=uniform))
        self.spec = s
        self.w_json.value = s.to_json()
        self._rebuild_for_new_spec()
        if uniform:
            self.status.object = (
                f"**uniform DC group {name!r} added** — assign electrodes and "
                f"set its single voltage (all members share it).")
        else:
            self.status.object = (
                f"**DC ladder {name!r} added** — assign electrodes, give each "
                f"a number, then set DC in / DC out.")

    def _on_remove_rf_group(self, _=None):
        """Remove the selected drive group. Members are DETACHED first (their
        rf_groups reference is cleared) so no electrode is left pointing at a
        group that no longer exists — a dangling reference would drive the
        solve with an undefined waveform. The detachment is reported, never
        silent."""
        names = list(getattr(self, "w_grp_del_pick", None)
                     and self.w_grp_del_pick.value or [])
        if not names:
            self.status.object = "**pick drive group(s) to remove**"
            return
        self._sync_spec()
        s = self.spec
        missing = [n for n in names
                   if not any(g.name == n for g in s.geometry.rf_groups)]
        if missing:
            self.status.object = f"**no drive group(s) named {missing}**"
            return
        detached = []
        for el in s.geometry.electrodes:
            if el.rf_groups and any(n in el.rf_groups for n in names):
                el.rf_groups = [g for g in el.rf_groups
                                if g not in names]
                detached.append(el.name)
        s.geometry.rf_groups = [g for g in s.geometry.rf_groups
                                if g.name not in names]
        self.spec = s
        self.w_json.value = s.to_json()
        self._rebuild_for_new_spec()
        det = (" — detached " + ", ".join(sorted(set(detached)))
               ) if detached else ""
        self.status.object = (
            f"**{len(names)} drive group(s) removed** "
            f"({', '.join(names)}){det}. Detached electrodes are now "
            f"DC-only at their dc value; reassign them if they should "
            f"still be driven.")

    def _on_remove_dc_group(self, _=None):
        """Remove the selected DC group. Members are DETACHED first: their
        dc_group/dc_index are cleared and their dc is FROZEN at the ladder-
        derived value it had (resolve_dc_groups already wrote it), so the
        geometry keeps solving with the same voltages it showed — nothing
        silently jumps to 0 V. The detachment is reported, never silent."""
        names = list(getattr(self, "w_dcg_del_pick", None)
                     and self.w_dcg_del_pick.value or [])
        if not names:
            self.status.object = "**pick DC group(s) to remove**"
            return
        self._sync_spec()
        s = self.spec
        missing = [n for n in names
                   if not any(g.name == n for g in s.geometry.dc_groups)]
        if missing:
            self.status.object = f"**no DC group(s) named {missing}**"
            return
        try:
            s.resolve_dc_groups()      # freeze members at derived values
        except Exception as e:
            self.status.object = (f"**cannot resolve before removal: "
                                  f"{e}** — fix the group(s) first.")
            return
        detached = []
        for el in s.geometry.electrodes:
            if el.dc_group in names:
                el.dc_group = None
                el.dc_index = None
                detached.append(f"{el.name} (frozen at {el.dc:+.3g} V)")
        s.geometry.dc_groups = [g for g in s.geometry.dc_groups
                                if g.name not in names]
        self.spec = s
        self.w_json.value = s.to_json()
        self._rebuild_for_new_spec()
        det = (" — " + "; ".join(detached)) if detached else ""
        self.status.object = (
            f"**{len(names)} DC group(s) removed** ({', '.join(names)})"
            f"{det}. Members keep their last derived voltage as a plain "
            f"DC; edit them individually now.")

    def _on_autonumber_dc(self, _=None):
        """Number each group's members by ascending centroid z, read from the
        SOLVE masks.

        Refuses when there is no solved model: the ordering has to come from
        the geometry the solver actually used, not from a name, a list
        position, or a guess. If it cannot be derived it is not offered."""
        import numpy as np
        masks = getattr(self._model, "el_masks", None)
        if not masks:
            self.status.object = ("**auto-number needs a solved field** — the "
                                  "order is read from the solve masks, not "
                                  "guessed from names. Recompute, then retry.")
            return
        self._sync_spec()
        s = self.spec
        h = s.geometry.mm_per_gu
        by_group = {}
        for i, el in enumerate(s.geometry.electrodes):
            if el.dc_group is None:
                continue
            m = masks.get(el.basis if el.basis else i + 1)
            if m is None or not m.any():
                continue
            zc = float(np.argwhere(m)[:, -1].mean() * h)
            by_group.setdefault(el.dc_group, []).append((zc, i))
        if not by_group:
            self.status.object = "**no DC-group members to number**"
            return
        msg = []
        for gname, rows in by_group.items():
            for n, (_zc, i) in enumerate(sorted(rows), start=1):
                s.geometry.electrodes[i].dc_index = n
                self._v_widgets[i]["dc_index"].value = n
            msg.append(f"{gname}: {len(rows)} members numbered 1..{len(rows)}")
        s.resolve_dc_groups()
        self.spec = s
        self.w_json.value = s.to_json()
        self.status.object = "**auto-numbered by z** — " + "; ".join(msg)

    def _on_example_selected(self, evt=None):
        self._clear_raster()
        """Selecting an example STAGES it into the JSON box and prices it.
        It does NOT apply it -- self.spec is untouched until the button is
        pressed. Writing w_json fires _on_json_edited, which re-runs the
        estimator against the editor spec, so the cost of the example appears
        immediately, before any commitment.

        Building an example spec can do real work (the STL example rasterises),
        so this is wrapped: a failure to stage must not take the app down, and
        must SAY so rather than leaving a stale cost from the previous pick.
        """
        name = self.w_examples.value
        if not name or name == "(keep current)":
            return
        try:
            spec = _example_specs()[name]()
            self.w_json.value = spec.to_json()      # -> _on_json_edited -> cost
            self.status.object = (
                f"**staged** *{name}* — cost shown above. "
                f"Press **Load example** to apply it.")
        except Exception as e:
            self.w_sizing.object = (
                f"**could not stage `{name}`:** {e}\n\n"
                f"*The cost above, if any, is from the previous selection — "
                f"do not read it as this example's.*")
            self.status.object = f"**example error:** {e}"

    def _on_apply_json(self, _=None):
        # rebuilding the per-electrode controls for a large geometry (e.g. a
        # 35-conductor import) takes a beat and blocks the UI thread; without
        # a notice it reads as a hang. Announce before,
        # and confirm after, so the wait is legible.
        self.status.object = "**loading spec** — building controls…"
        if hasattr(self, "w_apply_busy"):
            self.w_apply_busy.value = True
        try:
            # LOUD LOAD: capture the unknown-key
            # warnings the spec loaders now emit and surface them IN
            # the status pane — stderr alone is invisible in the app.
            import warnings as _warnings
            with _warnings.catch_warnings(record=True) as _wrec:
                # UserWarning only: that is the loud-load category; a
                # stray ResourceWarning from elsewhere is not a spec
                # problem and must not masquerade as one.
                _warnings.simplefilter("ignore")
                _warnings.simplefilter("always", UserWarning)
                # A staged assembly is not a SimSpec and load_any_spec
                # refuses it (correctly). Route it BEFORE that refusal:
                # the app can show it, one stage at a time.
                # A NEW LOAD DROPS THE OLD INSTRUMENT FIRST.
                # Unconditionally, and BEFORE the branch: whether the
                # incoming document is an assembly or a plain spec, the
                # previous instrument's stages must not survive into it.
                # Old FAs persisting in these containers is what left an
                # ion funnel showing `stage: mrt`.
                self._clear_assembly_state()
                if self._sniff_assembly(self.w_json.value):
                    n_st = self._load_assembly_text(self.w_json.value)
                    self._beam_sync_banner()
                    self._clear_beam_panel()
                    # DEFAULT VIEW = FULL ASSEMBLY, applied LAST
                    # (an earlier hook ran inside the loader
                    # and
                    # later calls could outdraw it; nothing runs after
                    # this line).
                    self._stage_sync_guard = True
                    try:
                        self.w_stage.value = "Full Assembly"
                    finally:
                        self._stage_sync_guard = False
                    self._on_view_assembly()
                    _fn = getattr(self.w_instrument, "filename", "") or \
                        "(pasted JSON)"
                    self.w_loaded_name.object = f"**loaded:** `{_fn}`"
                    _src_fa = (self._assembly_doc.get("beam")
                               or {}).get("from_stage", "?")
                    self.w_assembly_info.object = (
                        f"**Instrument loaded:** {n_st} stage(s) — "
                        f"{', '.join(self._assembly_specs)}. Default "
                        f"view: the WHOLE ASSEMBLY; **Fly** flies what "
                        f"is displayed (assembly beam derives from FA "
                        f"`{_src_fa}`).")
                    self.status.object = (
                        f"**staged assembly loaded** — {n_st} stage(s): "
                        f"{', '.join(self._assembly_specs)}. Showing the "
                        f"WHOLE ASSEMBLY (beam source FA `{_src_fa}`); "
                        f"*Set FA View* shows one stage.")
                    return
                self.spec = load_any_spec(self.w_json.value)
            n_el = len(self.spec.geometry.electrodes)
            self._rebuild_for_new_spec()
            # a loaded spec reports its cost AS AUTHORED, before anyone
            # changes anything
            self.w_pitch.value = float(self.spec.geometry.mm_per_gu)
            self._refresh_sizing()
            _msgs = sorted({str(w.message) for w in _wrec})
            if _msgs:
                self.status.object = (
                    f"**spec loaded** — {n_el} electrodes — "
                    f"**⚠️ {len(_msgs)} spec warning(s):** "
                    + " · ".join(_msgs))
            else:
                self.status.object = (f"**spec loaded** — {n_el} "
                                      "electrodes, controls rebuilt.")
        except Exception as e:
            self.status.object = f"**JSON error:** {e}"
        finally:
            if hasattr(self, "w_apply_busy"):
                self.w_apply_busy.value = False

    # ---------------------------------------------------------------
    # Staged multi-FA assembly support
    # ---------------------------------------------------------------
    def _sniff_assembly(self, txt):
        """True if the text is a STAGED assembly (stages + poses + seams),
        not a single spec and not the coaxial `assembly` schema, which
        flattens to one SimSpec and needs none of this."""
        from ion_gym.io.spec_io import sniff
        try:
            return sniff(txt) == "staged_assembly"
        except (ValueError, TypeError):
            # Unreadable text is not an assembly; let the normal loader
            # produce the real diagnostic rather than reporting a sniff
            # failure as though it were a schema verdict.
            return False

    def _load_assembly_text(self, txt):
        """Parse an instrument document into per-stage specs. Solves
        NOTHING: a stage is solved only when the user looks at it or flies
        the assembly."""
        doc = json.loads(txt)
        stages = doc.get("stages") or []
        if not stages:
            raise ValueError("staged assembly declares no stages")
        specs = {}
        for st in stages:
            if "name" not in st or "spec" not in st:
                raise ValueError(f"stage needs 'name' and 'spec': {st}")
            raw = st["spec"]
            if isinstance(raw, str):
                raise TypeError(
                    f"stage {st['name']!r}: 'spec' is a path ({raw!r}). "
                    f"Stage geometry must be INLINE — one file is the "
                    f"instrument.")
            if st["name"] in specs:
                raise ValueError(
                    f"two stages are both named {st['name']!r}; stage "
                    f"names select which geometry is displayed, so a "
                    f"duplicate makes the selector ambiguous.")
            specs[st["name"]] = load_any_spec(json.dumps(raw))
        self._clear_run_overlays("instrument load")
        self._assembly_doc = doc
        self._assembly_specs = specs
        names = list(specs)
        self.w_stage.options = ["Full Assembly"] + names
        self.w_fly_src.options = names
        self.w_fly_src.disabled = False
        self.w_set_fly_params.disabled = False
        if getattr(self, "w_instrument_dl", None) is not None:
            self.w_instrument_dl.disabled = False
            _nm = (doc.get("name") or "instrument").strip() \
                or "instrument"
            # ONE AUTHORITY: the editable field carries the name; its
            # watcher (_sync_instr_filename) writes the button's
            # filename. Setting the field here (not the button) keeps
            # displayed == actual and hands the user the load's default
            # to edit. A load RESETS any previous edit on purpose — a
            # stale name from the last instrument would mislabel this
            # one; assigning the same value is a no-op for the watcher.
            if getattr(self, "w_instr_name", None) is not None:
                self.w_instr_name.value = _nm
            else:
                self.w_instrument_dl.filename = _nm + ".json"
        _from = (doc.get("beam") or {}).get("from_stage")
        if _from in names:
            self.w_fly_src.value = _from
        # PIN DEFAULT: the Ion Source tab opens bound to
        # the instrument's own declared source FA, independent of the
        # displayed stage from the first click. No from_stage -> no pin
        # (single-beam instruments bind to the live spec as ever).
        self._fly_src_pin = _from if _from in names else None
        self.w_stage.disabled = False
        self.w_fly_assembly.disabled = False
        self.w_fly_mode.disabled = False
        self.w_view_assembly.disabled = False
        self.w_set_fa_view.disabled = False
        # THE COUNT DEFAULTS TO THE WHOLE DECLARED PACKET (a deck
        # showed only 8 ions of a larger packet). w_n
        # initialises from whatever spec was DISPLAYED before the load,
        # and the narrowing lever then silently subset a 200-ion
        # instrument down to that stale number. Narrowing stays available
        # -- but as a deliberate act after load, never a leftover.
        _bm = doc.get("beam") or {}
        _n_decl = len(_bm.get("ions") or []) or 0
        if not _n_decl and _bm.get("from_stage") in specs:
            _n_decl = int(getattr(
                specs[_bm["from_stage"]].source, "n_ions", 0) or 0)
        if _n_decl > 0 and getattr(self, "w_n", None) is not None:
            self.w_n.value = _n_decl
        # Show the first stage's CONTROLS. Assigning .value fires the
        # watcher, which does the swap — one code path for "loaded" and
        # "user switched", so the two cannot drift.
        # controls come up on the first FA; the VIEW lands on the
        # assembly in the caller, LAST, so no banner/panel call after
        # this function can undo the default (an earlier
        # ordering let a later call leave 'oa' on screen).
        self._loading_doc = True
        try:
            self._show_stage(names[0])
        finally:
            self._loading_doc = False
        self._doc_dirty = False
        return len(names)

    _stage_sync_guard = False

    _live_stage_name = None      # the stage whose spec the tabs EDIT
    _doc_dirty = False
    _loading_doc = False

    def _mark_doc_modified(self, reason: str):
        """The FIRST divergence
        stamps the in-memory document — provenance (when/why/source
        file) and certification_stale — and the multiFA header shows
        MODIFIED until Save-As. Certified numbers cannot be quoted
        against edited physics; the stamp travels with every save and
        every fly (the document is what flies)."""
        if not self._assembly_doc:
            return
        first = not self._doc_dirty
        self._doc_dirty = True
        prov = self._assembly_doc.setdefault("provenance", {})
        prov.setdefault("source_file", getattr(
            self.w_instrument, "filename", "") or "(pasted JSON)")
        prov["modified_in_session"] = __import__(
            "datetime").datetime.now().isoformat(timespec="seconds")
        prov["certification_stale"] = True
        prov.setdefault("reasons", [])
        if reason not in prov["reasons"]:
            prov["reasons"].append(reason)
        if first and hasattr(self, "w_loaded_name"):
            _fn = prov["source_file"]
            self.w_loaded_name.object = (
                f"**loaded:** `{_fn}` · **MODIFIED** — differs from the "
                f"loaded file; certified numbers are STALE for this "
                f"edited document.")

    def _write_through_live_stage(self, reason: str = "spec edit"):
        """EVERY per-stage spec edit
        writes through to the retained document immediately — the live
        stage's spec IS the document's stage, one regime, no special
        cases. No-ops while a document is loading (programmatic widget
        sets are not edits) and without an assembly (nothing to write
        through to)."""
        if self._loading_doc or not self._assembly_doc:
            return
        nm = self._live_stage_name
        if not nm or nm not in (self._assembly_specs or {}):
            return
        self._sync_spec()
        for st in self._assembly_doc.get("stages", []):
            if st.get("name") == nm:
                st["spec"] = self.spec.to_dict()
                break
        else:
            raise ValueError(
                f"write-through: live stage {nm!r} is not in the "
                f"document's stages — the live spec and the document "
                f"have diverged structurally; refusing to guess")
        self._mark_doc_modified(f"{reason} (stage {nm})")

    def _on_spec_widget_change(self, _evt=None):
        self._write_through_live_stage()

    @staticmethod
    def _walk_value_widgets(container):
        out = []
        stack = [container]
        while stack:
            o = stack.pop()
            if hasattr(o, "__iter__") and not isinstance(o, (str, bytes)):
                try:
                    stack.extend(list(o))
                except TypeError as e:
                    raise TypeError(
                        f"_walk_value_widgets: container {type(o).__name__} "
                        f"iterates but not listably: {e}") from e
            if hasattr(o, "param") and "value" in getattr(
                    o, "param", ()) and hasattr(o, "value"):
                out.append(o)
        return out

    def _record_chip_suffix(self) -> str:
        """The large-record warning, carried on the live chip.

        A warned flight DOES launch, so a warning written once would be
        overwritten by the next chip update within 200 ms. Riding on the
        chip instead keeps it visible for exactly as long as the flight
        it applies to -- which is when a 9 GB record matters.
        """
        note = getattr(self, "_record_note", None)
        return f" · {note}" if note else ""

    def _fly_chip(self, phase, done=None, total=None, note=None,
                  activity_ts=None):
        """THE one writer of the always-visible flight chip.

        Phases: 'begin' (a flight launched; starts the clock),
        'progress' (periodic; done/total optional while solving),
        'done', 'stopped', 'error' (terminal; text persists until the
        next begin), 'idle' (clear — Reset app). The heartbeat is
        'last activity N s ago': activity is the moment the done-count
        last CHANGED (or an explicit worker timestamp), so a genuinely
        frozen flight shows a GROWING age while a long single ion shows
        a growing age with a running elapsed clock — decidable at a
        glance, from any tab.
        """
        import time as _t
        _PHASES = ("begin", "progress", "done", "stopped", "error", "idle",
                   "refused")
        if phase not in _PHASES:
            # validated at ENTRY: the before-begin guard below returns
            # early, and an unknown phase slipping through it would
            # render a stale chip that reads as a frozen flight.
            raise ValueError(
                f"_fly_chip: unknown phase {phase!r} — one of "
                f"{'/'.join(_PHASES)}.")
        now = _t.time()
        st = getattr(self, "_fly_chip_state", None)
        if phase == "begin":
            self._fly_chip_state = st = dict(t0=now, act=now, done=0,
                                             total=total)
            self.fly_chip.object = ("✈ **flight launched** · 0 s"
                                    + (f" · 0/{total} ions" if total
                                       else "")
                                    + self._record_chip_suffix())
            return
        if phase == "idle":
            self._fly_chip_state = None
            self.fly_chip.object = ""
            self._record_note = None
            return
        if phase == "refused":
            # TERMINAL, and rendered WITHOUT a preceding 'begin' -- a
            # refused flight never launched, so there is no clock to
            # report. It lands here because the chip is the one thing
            # visible from EVERY tab: a refusal written only to the
            # status line is a press that appears to do nothing, with
            # the reason on a tab the user is not looking at.
            self._fly_chip_state = None
            self.fly_chip.object = ("\u26d4 **flight refused** \u2014 "
                                    + (note or "see Status tab"))
            return
        if st is None:
            # a progress/terminal write with no begin: a real sequencing
            # bug upstream — surface it rather than guessing a clock.
            self.fly_chip.object = f"⚠ flight chip: {phase!r} before begin"
            return
        if activity_ts is not None:
            st["act"] = max(st["act"], float(activity_ts))
        if done is not None and int(done) != st["done"]:
            st["done"] = int(done)
            st["act"] = now
        if total is not None:
            st["total"] = int(total)
        el = now - st["t0"]
        age = now - st["act"]
        cnt = (f"{st['done']}/{st['total']} ions" if st["total"]
               else "working")
        if phase == "progress":
            self.fly_chip.object = (
                f"✈ **{cnt}** · {el:.0f} s"
                + (f" · last activity {age:.0f} s ago" if age >= 2 else "")
                + self._record_chip_suffix())
        elif phase == "done":
            self.fly_chip.object = f"✓ **{cnt}** in {el:.0f} s"
        elif phase == "stopped":
            self.fly_chip.object = (f"■ **STOPPED {cnt}** at {el:.0f} s"
                                    + (f" — {note}" if note else ""))
        elif phase == "error":
            self.fly_chip.object = (f"✖ **flight failed** at {el:.0f} s"
                                    + (f" — {note}" if note else "")
                                    + " · see Status tab")
        else:
            # UNREACHABLE ON PURPOSE (the else states its
            # outcome): the phase set is validated at entry and
            # begin/idle returned above, so only progress/done/stopped/
            # error reach this chain. Raising here means the entry
            # validation and this chain have drifted apart.
            raise RuntimeError(
                f"_fly_chip: phase {phase!r} passed entry validation "
                f"but no branch renders it — the phase set and this "
                f"chain have drifted apart.")

    def _clear_run_overlays(self, why):
        """Retire every overlay DERIVED FROM A FLIGHT when the subject
        changes (a mixed-geometry load drew 100 green
        'detections' at x~78.5 — the PREVIOUS rf-drift flight's
        arrivals riding into the new instrument's view, because nothing
        scoped _impact_hits to the run that produced it). Detections,
        assembly traces, and the live figure bank belong to a flight of
        a PARTICULAR subject; a new document must start visually clean.
        Stored runs are NOT touched — they remain reloadable from the
        run selector, which re-populates these overlays deliberately.
        """
        self._impact_hits = []
        self._assembly_traces = None
        self._live_bank = None
        # a subject change also outdates the banked live-figure
        # signature indirectly (model id), but stating it costs nothing
        print(f"[overlays] cleared on {why}", flush=True)

    def _sync_instr_filename(self, evt=None):
        """The instrument-export field is the ONE filename authority
        every edit lands on the download button here.
        Empty input falls back to 'instrument' — a visible named
        default, not a silent empty filename — and .json is appended
        when missing so the exported file always reloads."""
        if getattr(self, "w_instrument_dl", None) is None:
            return
        nm = ((evt.new if evt is not None else self.w_instr_name.value)
              or "instrument").strip() or "instrument"
        if not nm.lower().endswith(".json"):
            nm += ".json"
        self.w_instrument_dl.filename = nm

    def _subject_is_assembly(self):
        """True when the app's SUBJECT is the whole assembly.

        One predicate, consulted by the draw, solve and edit paths, so
        those three can never disagree about what the user selected.
        """
        return (bool(self._assembly_specs)
                and getattr(self, "_assembly_stage", None) == WHOLE_ASSEMBLY)

    def _redraw_subject(self):
        """Draw whatever the SUBJECT currently names, at the current plane.

        THE single redraw entry for an assembly. Every path that
        wants the main pane refreshed calls this instead of drawing
        directly, which is what makes the assembly view survive a plane
        change: the plane buttons no longer decide WHAT is drawn, only
        HOW, and the subject decides what.
        """
        if self._subject_is_assembly():
            self._on_view_assembly()
        else:
            self._draw_background()

    def _persistent(self, attr, factory, wire=None):
        """Create a widget ONCE and hand back the same object thereafter.

        THE BUG THIS EXISTS TO KILL. `_build_controls` re-runs on
        every spec load and every stage swap, and it used to REBIND these
        attributes to brand-new widget objects. A served Panel session
        built its layout once, so the widget on the user's SCREEN stayed
        the original while `self.w_*` pointed at a replacement. Ticking
        the on-screen box wrote to an orphan nobody read: measured,
        "displayed is self.w_fly_mode: False" from the moment an
        instrument loaded, which is exactly why *Fly whole assembly* did
        nothing.

        Carrying a `value` across the rebuild (what this code did before)
        does not fix it. It preserves the STATE while still swapping the
        IDENTITY, so the copy is correct and the widget the user is
        touching is still the wrong one.

        Note the verification trap this defect hides behind: calling
        `panel()` again in a test REBUILDS the layout and picks up the new
        widget, so the app looks healthy. It has to be driven through a
        layout captured ONCE, the way a server holds it.

        `wire` runs only on first creation, so watchers and click handlers
        are never attached twice — a doubly-wired callback fires twice per
        interaction, which on `_show_stage` is how a re-entrant
        rebuild loop started.
        """
        w = getattr(self, attr, None)
        if w is not None:
            return w
        w = factory()
        setattr(self, attr, w)
        if wire is not None:
            wire(w)
        return w

    def _clear_assembly_state(self):
        """Drop every trace of a previously loaded instrument.

        THE DEFECT THIS FIXES. Assembly state was only ever SET,
        never cleared, so loading a plain spec left `_assembly_doc` and
        `_assembly_specs` populated: the stage selector kept offering the
        OLD instrument's stages, the solve target still named a stage that
        no longer existed, and *View whole assembly* silently did nothing
        because the stages it named were not the loaded model. This showed
        exactly this -- an ion funnel loaded with `stage: mrt` still
        showing.

        A control offering a choice that cannot be honoured is worse than
        a disabled one: it reports a state the app is not in, which is the
        "displayed equals computed" invariant failing in the widget layer.
        So this resets the DATA and the CONTROLS together -- clearing one
        without the other is how they drifted apart in the first place.
        """
        self._assembly_doc = None
        self._assembly_specs = {}
        self._assembly_stage = None
        self._fly_src_pin = None       # pin dies with the assembly
        if getattr(self, "w_instrument_dl", None) is not None:
            self.w_instrument_dl.disabled = True
        self._assembly_traces = None
        self._assembly_rot_warning = ""
        if getattr(self, "w_stage", None) is not None:
            self._stage_swapping = True
            try:
                self.w_stage.options = ["(no assembly loaded)"]
                self.w_stage.value = "(no assembly loaded)"
            finally:
                self._stage_swapping = False
            self.w_stage.disabled = True
        for _a, _opts in (("w_solve_target", ["(displayed stage)"]),):
            _w = getattr(self, _a, None)
            if _w is not None:
                _w.options = _opts
                _w.value = _opts[0]
                _w.disabled = True
        for _a in ("w_view_assembly", "w_fly_assembly"):
            _w = getattr(self, _a, None)
            if _w is not None:
                _w.disabled = True
        if getattr(self, "w_fly_mode", None) is not None:
            # FALSE unless an instrument is loaded. Leaving it
            # ticked with no assembly means the app's main verb routes to
            # a flight that cannot run.
            self.w_fly_mode.value = False
            self.w_fly_mode.disabled = True
        if getattr(self, "w_assembly_info", None) is not None:
            self.w_assembly_info.object = (
                "*No instrument loaded.* Upload an instrument JSON above; "
                "these controls stay disabled until one is.")
        for _b in (getattr(self, "view_xy", None),
                   getattr(self, "view_xz", None),
                   getattr(self, "view_yz", None)):
            if _b is not None:
                _b.disabled = False
        self._clear_beam_panel()

    def _impact_tab(self):
        """Impact / detector analysis: what the ion signal would LOOK like.

        Built once and repopulated, like `stats`, so the served layout
        keeps one object.
        """
        if getattr(self, "_impact_col", None) is None:
            from ion_gym.physics.stats import stats_card
            # A SECOND stats_card INSTANCE, fed the same results as the
            # Ion Source one (the stats table is repeated
            # here). Re-using the SAME card object would put one Bokeh
            # model in two tabs -- the exact two-parent defect
            # that killed every assembly button in v408. Two instances,
            # one update site.
            self.stats_impact = stats_card(collapsed=False)
            # ONE dropdown for every impact surface (
            # "treat the boundaries and the detector stations the same
            # where there's one plot and a drop down"). Stations and
            # enabled boundary faces share the stream; face labels come
            # from enabled_planes() -- the SAME naming authority the
            # stats table uses, so the dropdown and the table can never
            # disagree about what a face is called.
            self.w_impact_station = pn.widgets.Select(
                name="surface (station / boundary face)",
                options=["(no flight yet)"], disabled=True)
            self.w_impact_plane = pn.widgets.Select(
                name="cross-section plane", options=["yz", "xy", "xz"],
                value="yz", disabled=True)
            self.w_impact_hist_axis = pn.widgets.Select(
                name="histogram axis", options=["t", "y", "z", "x"],
                value="t", disabled=True)

            self.w_impact_pane = pn.pane.Plotly(height=460,
                                                sizing_mode="stretch_width")
            self.w_impact_msg = pn.pane.Markdown(
                "*No flight yet.* Fly the instrument; arrivals at any "
                "declared station or enabled boundary face appear here "
                "as a 2-D impact cross-section and a 1-D histogram — "
                "pick the surface in the dropdown.")
            for _w in (self.w_impact_station, self.w_impact_plane,
                       self.w_impact_hist_axis):
                _w.param.watch(lambda *_a: self._draw_impact(), "value")
            self._impact_col = pn.Column(
                pn.pane.Markdown("### Impact analysis"),
                self.stats_impact.card,
                self.w_impact_msg,
                pn.Row(self.w_impact_station),
                pn.Row(self.w_impact_plane, self.w_impact_hist_axis),
                self.w_impact_pane,
                width=CONTROL_COL_PX - 20)
        return self._impact_col

    def _draw_impact(self):
        """2-D impact cross-section + 1-D histogram for the chosen station.

        WHAT THIS IS FOR: a resolution number says how sharp the peak is;
        it does not say what the signal LOOKS like. The cross-section is
        where the ions land on the detector face, and the histogram is the
        recorded trace along one axis -- with `t` the default, because a
        time-of-flight instrument's signal IS the arrival-time
        distribution.

        Uses the retained per-ion arrivals from the last flight. If the
        run kept no arrivals, it SAYS so rather than drawing an empty
        frame, which reads as "nothing landed" when the truth is "nothing
        was measured".
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        _all = getattr(self, "_impact_hits", None) or []
        # FILTER TO THE SELECTED SURFACE. This once drew EVERY
        # hit under the selected surface's title -- a multi-surface deck
        # pooled all arrivals into one mislabeled plot. Found while
        # unifying stations and boundary faces into one dropdown.
        _sel = self.w_impact_station.value
        hits = [h for h in _all if h.get("station") == _sel]
        if not _all:
            self.w_impact_msg.object = (
                "*No arrivals to analyse.* Fly the instrument; ions that "
                "cross a declared station or an enabled boundary face "
                "appear here.")
            self.w_impact_pane.object = None
            return
        if not hits:
            self.w_impact_msg.object = (
                f"*No arrivals on **{_sel}** this flight* — other "
                f"surfaces in the dropdown have data.")
            self.w_impact_pane.object = None
            return
        _ax = {"x": "x", "y": "y", "z": "z", "t": "t_us"}
        pl = self.w_impact_plane.value or "yz"
        a1, a2 = pl[0], pl[1]
        u = [float(h[_ax[a1]]) for h in hits]
        v = [float(h[_ax[a2]]) for h in hits]
        hax = self.w_impact_hist_axis.value or "t"
        hv = [float(h[_ax[hax]]) for h in hits]

        fig = make_subplots(
            rows=2, cols=1, vertical_spacing=0.16,
            subplot_titles=(f"impact · {pl}", f"signal · {hax}"))
        # PER-m/z SPLIT (a histogram read incorrectly for a
        # planar detector station): the NUMBERS were verified exact
        # (feed == per-ion first_detection to machine precision, y on
        # the plane), but a MERGED unlabeled histogram over a mixed-m/z
        # packet shows anonymous multi-spikes that read as a defect —
        # while the fate table above even promises "see per-m/z tabs".
        # A mixed packet now draws one labeled, colored series PER m/z
        # in both the cross-section and the signal; a single-m/z run
        # keeps the plain single-series form.
        _mzs = sorted({h.get("mz") for h in hits
                       if h.get("mz") is not None})
        if len(_mzs) > 1:
            # SAME convention as the trajectories, keyed off the
            # DECLARED mz_list so a mass with no hits does not shift
            # every other mass's colour between the two figures.
            _mzmap = _mz_color_map(self.spec.source.mz_list)
            for _im, _mzv in enumerate(_mzs):
                _sel = [h for h in hits if h.get("mz") == _mzv]
                _cu = [float(h[_ax[a1]]) for h in _sel]
                _cv = [float(h[_ax[a2]]) for h in _sel]
                _ch = [float(h[_ax[hax]]) for h in _sel]
                _col = _mzmap.get(_mzv, "#7f7f7f")
                _lbl = f"m/z {_mzv:g} (n={len(_sel)})"
                fig.add_trace(go.Scatter(
                    x=_cu, y=_cv, mode="markers",
                    marker=dict(size=5, color=_col, opacity=0.6),
                    name=_lbl, legendgroup=_lbl, showlegend=True,
                    hovertemplate=f"{a1} %{{x:.3f}}<br>{a2} %{{y:.3f}}"
                                  f"<extra>{_lbl}</extra>"),
                    row=1, col=1)
                fig.add_trace(go.Histogram(
                    x=_ch, nbinsx=48, marker=dict(color=_col),
                    opacity=0.75, name=_lbl, legendgroup=_lbl,
                    showlegend=False), row=2, col=1)
            fig.update_layout(barmode="overlay")
        else:
            fig.add_trace(go.Scatter(
                x=u, y=v, mode="markers",
                marker=dict(size=5, color="#1f77b4", opacity=0.6),
                name="impacts", showlegend=False,
                hovertemplate=f"{a1} %{{x:.3f}}<br>{a2} %{{y:.3f}}"
                              f"<extra></extra>"), row=1, col=1)
            fig.add_trace(go.Histogram(
                x=hv, nbinsx=48, marker=dict(color="#2ca02c"),
                name="signal", showlegend=False), row=2, col=1)
        _u = "us" if hax == "t" else "mm"
        fig.update_xaxes(title_text=f"{a1} (mm)", row=1, col=1)
        # free zoom, in all views —
        # the anchored axis reshaped the drag rectangle to 1:1 here too.
        fig.update_yaxes(title_text=f"{a2} (mm)", row=1, col=1)
        fig.update_xaxes(title_text=f"{hax} ({_u})", row=2, col=1)
        fig.update_yaxes(title_text="ions", row=2, col=1)
        fig.update_layout(height=460, showlegend=(len(_mzs) > 1),
                          margin=dict(l=54, r=16, t=40, b=44),
                          title=dict(
                              text=(f"{self.w_impact_station.value} · "
                                    f"{len(hits)} ions"),
                              x=0.01, font=dict(size=12)))
        self.w_impact_pane.object = fig
        self.w_impact_msg.object = (
            f"**{len(hits)} ions** at station "
            f"**{self.w_impact_station.value}**.")

    def _ui_seed(self):
        """The run seed per the MASTER control: the pinned value when
        'seeded run' is checked, otherwise fresh entropy drawn now and
        PRINTED (an unrecorded random seed is an unreproducible run)."""
        if self.w_seeded.value:
            return int(self.w_seedval.value)
        import secrets
        drawn = secrets.randbits(31)
        print(f"[seed] random mode: drew {drawn} — tick 'seeded run' and "
              f"enter it to reproduce", flush=True)
        return drawn

    def _station_editor(self):
        """Detector/monitor station editor, bottom of the Ion Source tab.

        WHY IT EXISTS: there was no way to DEFINE a
        detector plane from the UI at all — bounds exist, but a bound is
        an absolute domain kill and cannot represent a detector (the
        StationSpec docstring records that). Until now a station
        could only be typed into raw JSON.

        CONTRACT: reads and populates from the loaded spec; *Apply
        station* writes back into the LIVE spec, the JSON box, and — for
        an assembly — the retained instrument document, so *Fly assembly*
        uses it and a save persists it. One station edited at a time,
        selected by name; a new name creates a new station.

        Built once (`_persistent` reasoning): one object,
        one container.
        """
        if getattr(self, "_station_col", None) is not None:
            return self._station_col
        self.w_stn_pick = pn.widgets.Select(name="station", options=["(new)"],
                                            width=150)
        self.w_stn_name = pn.widgets.TextInput(name="name", value="DETECTOR",
                                               width=130)
        self.w_stn_kind = pn.widgets.Select(name="kind",
                                            options=["detect",
                                                     "impact_plane"],
                                            value="detect", width=90)
        self.w_stn_onhit = pn.widgets.Select(
            name="on hit (detect)", options=["pass", "splat"],
            value="pass", width=110,
            description="detect only: pass = log and continue; "
                        "splat = absorbing detector (fate 6). An "
                        "impact_plane ignores this (aperture "
                        "passes, plate splats by definition).")
        self.w_stn_axis = pn.widgets.Select(name="axis",
                                            options=["x", "y", "z"],
                                            value="x", width=60)
        self.w_stn_pos = pn.widgets.FloatInput(name="pos (mm)", value=0.0,
                                               width=110)
        # Window over the two transverse axes. Labels follow the chosen
        # axis so the user is never editing "w1" blind.
        self.w_stn_w1_on = pn.widgets.Checkbox(name="window y", value=False)
        self.w_stn_w1_lo = pn.widgets.FloatInput(name="min", value=-5.0,
                                                 width=95)
        self.w_stn_w1_hi = pn.widgets.FloatInput(name="max", value=5.0,
                                                 width=95)
        self.w_stn_w2_on = pn.widgets.Checkbox(name="window z", value=False)
        self.w_stn_w2_lo = pn.widgets.FloatInput(name="min", value=-5.0,
                                                 width=95)
        self.w_stn_w2_hi = pn.widgets.FloatInput(name="max", value=5.0,
                                                 width=95)
        self.w_stn_apply = pn.widgets.Button(name="Apply station",
                                             button_type="primary", width=110)
        self.w_stn_delete = pn.widgets.Button(name="Delete", width=70)
        self.w_stn_msg = pn.pane.Markdown("", styles={"font-size": "11px"})
        self.w_stn_pick.param.watch(lambda *_: self._station_load(), "value")
        self.w_stn_axis.param.watch(lambda *_: self._station_axis_labels(),
                                    "value")
        # G1(a) ONE REGIME, stations included (2026-09-12, Brian: "setting
        # a detector station to splat doesn't register — defaults back to
        # pass"): every OTHER spec widget writes through on change, so a
        # flipped Select here that silently required Apply reverted on the
        # next editor sync — the un-applied value looked applied and then
        # vanished. A VALUE edit on an EXISTING picked station now writes
        # through immediately; renaming and creating ('(new)') stay behind
        # Apply, because a half-typed name must not rename or spawn a
        # station per keystroke.
        for _w in (self.w_stn_kind, self.w_stn_onhit, self.w_stn_axis,
                   self.w_stn_pos, self.w_stn_w1_on, self.w_stn_w1_lo,
                   self.w_stn_w1_hi, self.w_stn_w2_on, self.w_stn_w2_lo,
                   self.w_stn_w2_hi):
            _w.param.watch(self._on_station_field_change, "value")
        self.w_stn_apply.on_click(self._on_station_apply)
        self.w_stn_delete.on_click(self._on_station_delete)
        self._station_col = pn.Column(
            pn.pane.Markdown("**Detector / monitor stations** — a plane at "
                             "`axis = pos` with optional windows over the "
                             "other axes. `detect` absorbs at first "
                             "in-window crossing; `record` logs and lets "
                             "the ion continue."),
            pn.Row(self.w_stn_pick, self.w_stn_name, self.w_stn_kind,
                   self.w_stn_onhit),
            pn.Row(self.w_stn_axis, self.w_stn_pos),
            pn.Row(self.w_stn_w1_on, self.w_stn_w1_lo, self.w_stn_w1_hi),
            pn.Row(self.w_stn_w2_on, self.w_stn_w2_lo, self.w_stn_w2_hi),
            pn.Row(self.w_stn_apply, self.w_stn_delete),
            self.w_stn_msg)
        self._station_sync_pick()
        return self._station_col

    def _station_other_axes(self):
        ax = self.w_stn_axis.value or "x"
        return [a for a in ("x", "y", "z") if a != ax]

    def _station_axis_labels(self):
        o1, o2 = self._station_other_axes()
        self.w_stn_w1_on.name = f"window {o1}"
        self.w_stn_w2_on.name = f"window {o2}"

    def _station_sync_pick(self):
        """Repopulate the picker from the LIVE spec's stations."""
        stns = list(getattr(self.spec, "stations", None) or [])
        opts = [getattr(st, "name", f"station{i}")
                for i, st in enumerate(stns)] + ["(new)"]
        self._station_loading = True
        try:
            self.w_stn_pick.options = opts
            self.w_stn_pick.value = opts[0]
            self._station_load()
        finally:
            self._station_loading = False

    def _station_load(self):
        """Populate the editor from the picked station (or defaults).

        Programmatic: sets widget values from the spec, so it runs under
        the _station_loading guard — a LOAD is not an edit, and the
        write-through watchers must not re-write (or redraw) what was
        just read.
        """
        _outer = getattr(self, "_station_loading", False)
        self._station_loading = True
        try:
            self._station_load_inner()
        finally:
            self._station_loading = _outer

    def _station_load_inner(self):
        """Populate the editor from the picked station (or defaults)."""
        nm = self.w_stn_pick.value
        stns = list(getattr(self.spec, "stations", None) or [])
        st = next((s_ for s_ in stns if getattr(s_, "name", "") == nm), None)
        if st is None:
            # RESET TO DEFAULTS, not "leave whatever was there":
            # the editor must be clear on a new load unless the JSON
            # declares a station). Stale field values from the previous
            # spec would read as a station the new deck does not have --
            # stale state in the editor's clothing.
            self.w_stn_name.value = "DETECTOR"
            self.w_stn_kind.value = "detect"
            self.w_stn_onhit.value = "pass"
            self.w_stn_axis.value = "x"
            self.w_stn_pos.value = 0.0
            for w_ in (self.w_stn_w1_on, self.w_stn_w2_on):
                w_.value = False
            for w_, v_ in ((self.w_stn_w1_lo, -5.0), (self.w_stn_w1_hi, 5.0),
                           (self.w_stn_w2_lo, -5.0), (self.w_stn_w2_hi, 5.0)):
                w_.value = v_
            self.w_stn_msg.object = ""
            self._station_axis_labels()
            return
        self.w_stn_name.value = st.name
        self.w_stn_kind.value = st.kind
        if st.kind == "detect":
            self.w_stn_onhit.value = st.on_hit or "pass"
        self.w_stn_axis.value = st.axis
        self.w_stn_pos.value = float(st.pos_mm)
        self._station_axis_labels()
        o1, o2 = self._station_other_axes()
        for o, won, wlo, whi in ((o1, self.w_stn_w1_on, self.w_stn_w1_lo,
                                  self.w_stn_w1_hi),
                                 (o2, self.w_stn_w2_on, self.w_stn_w2_lo,
                                  self.w_stn_w2_hi)):
            win = (st.window or {}).get(o)
            won.value = win is not None
            if win is not None:
                wlo.value, whi.value = float(win[0]), float(win[1])

    def _station_from_editor(self):
        from ion_gym.io.sim_spec import StationSpec
        win = {}
        o1, o2 = self._station_other_axes()
        for o, won, wlo, whi in ((o1, self.w_stn_w1_on, self.w_stn_w1_lo,
                                  self.w_stn_w1_hi),
                                 (o2, self.w_stn_w2_on, self.w_stn_w2_lo,
                                  self.w_stn_w2_hi)):
            if won.value:
                lo, hi = float(wlo.value), float(whi.value)
                if not lo < hi:
                    raise ValueError(
                        f"window {o}: min ({lo}) must be < max ({hi})")
                win[o] = [lo, hi]
        nm = (self.w_stn_name.value or "").strip()
        if not nm:
            raise ValueError("a station needs a name; it is how detections "
                             "are attributed and how this editor finds it "
                             "again")
        return StationSpec(name=nm, kind=self.w_stn_kind.value,
                           on_hit=(self.w_stn_onhit.value
                                   if self.w_stn_kind.value
                                   == "detect" else None),
                           axis=self.w_stn_axis.value,
                           pos_mm=float(self.w_stn_pos.value), window=win)

    def _station_writeback(self):
        """Spec -> JSON box -> assembly document, in that order.

        The JSON box always reflects the live spec after an edit, so what
        the user saves is what the flight uses. For an assembly the same
        stations are written into the retained instrument document's
        stage — the document is what *Fly assembly* flies, so an
        edit that stopped at the live spec would VANISH from the very
        flight it was made for.
        """
        self.w_json.value = self.spec.to_json()
        # LIVE stage, not the displayed SUBJECT (found while
        # wiring this): _assembly_stage can be WHOLE_ASSEMBLY since the
        # subject-mode change, and matching it against stage names
        # SILENTLY skipped this write whenever the assembly was on
        # screen — the edit vanished from the very flight it was for.
        if self._assembly_doc and getattr(self, "_live_stage_name", None):
            nm = self._live_stage_name
            for st in self._assembly_doc.get("stages", []):
                if st.get("name") == nm:
                    st["spec"]["stations"] = [
                        s_.to_dict() for s_ in
                        (getattr(self.spec, "stations", None) or [])]
                    break

    def _on_station_field_change(self, _evt=None):
        """G1(a) one regime, stations included: a VALUE edit (kind,
        on_hit, axis, position, windows) on an EXISTING picked station
        writes through immediately — spec, JSON box, and the retained
        instrument document — exactly like every other spec widget.
        Before this, the editor was Apply-gated while the rest of the
        app wrote through on change, so a flipped on_hit Select looked
        applied and then silently reverted to the spec's old value on
        the next editor sync ("setting a detector station to splat
        doesn't register — defaults back to pass", 2026-09-12).

        Renaming and creating ('(new)') stay behind Apply: a half-typed
        name must not rename or spawn a station per keystroke. A
        mid-edit INVALID state (window lo >= hi while typing) is
        REPORTED and not written — the next valid change writes.
        No-ops during programmatic loads (a load is not an edit)."""
        if getattr(self, "_station_loading", False):
            return
        nm = self.w_stn_pick.value
        stns = list(getattr(self.spec, "stations", None) or [])
        if nm == "(new)" or all(getattr(s_, "name", "") != nm
                                for s_ in stns):
            return              # creation is an explicit Apply action
        try:
            new = self._station_from_editor()
        except ValueError as e:
            self.w_stn_msg.object = (
                f"**not written:** {e} — the station keeps its last "
                f"valid value until this is fixed.")
            return
        if new.name != nm:
            return              # rename in progress: Apply territory
        # ORDER-PRESERVING in-place replace: station order is deck
        # content; an edit must not shuffle it.
        self.spec.stations = [new if getattr(s_, "name", "") == nm
                              else s_ for s_ in stns]
        self._station_writeback()
        self._mark_doc_modified("station edit")
        self.w_stn_msg.object = (
            f"**station `{nm}` updated** — in the spec, the JSON box, "
            f"and the instrument document (write-through; Apply is only "
            f"needed to rename or create). Re-fly to see it in Impact "
            f"Analysis.")
        # Redraw so the View overlay tracks the committed spec. With
        # stations excluded from _build_sig this is a sig-match REUSE:
        # no solve, no compose — just the overlay.
        self._redraw_subject()

    def _on_station_apply(self, _=None):
        try:
            new = self._station_from_editor()
        except ValueError as e:
            self.w_stn_msg.object = f"**refused:** {e}"
            return
        stns = list(getattr(self.spec, "stations", None) or [])
        if any(getattr(s_, "name", "") == new.name for s_ in stns):
            # ORDER-PRESERVING for an existing name (matches the
            # write-through path — an edit must not shuffle deck order);
            # only a genuinely NEW station appends.
            stns = [new if getattr(s_, "name", "") == new.name else s_
                    for s_ in stns]
        else:
            stns = stns + [new]
        self.spec.stations = stns
        self._station_writeback()
        self._station_sync_pick()
        self.w_stn_pick.value = new.name
        self._mark_doc_modified("station edit")
        _tgt = (f" (stage **{self._live_stage_name}** of the instrument)"
                if self._assembly_doc else "")
        self.w_stn_msg.object = (
            f"**station `{new.name}` applied**{_tgt} — in the spec, the "
            f"JSON box, and the instrument document. Re-fly to see it in "
            f"Impact Analysis.")
        # Redraw NOW. The View tab draws declared stations; leaving the
        # old picture up until the user happens to switch views shows a
        # spec that is not the one on screen (displayed != committed).
        self._redraw_subject()

    def _on_station_delete(self, _=None):
        nm = self.w_stn_pick.value
        if nm == "(new)":
            self.w_stn_msg.object = "*nothing selected to delete*"
            return
        stns = [s_ for s_ in (getattr(self.spec, "stations", None) or [])
                if getattr(s_, "name", "") != nm]
        self.spec.stations = stns
        self._station_writeback()
        self._station_sync_pick()
        self.w_stn_msg.object = (
            f"**station `{nm}` deleted** — a deliberate removal is a "
            f"reported one, not a silent absence.")
        self._redraw_subject()


    def _publish_last_flight(self, results, *, site: str) -> None:
        """Bank a SINGLE-STAGE flight in the server-global
        last-flight slot for the /flight tab.

        Route parity (A2): planar, r-z, 3-D and tw2d flights all
        complete through the two sites that call this — the tap reads
        retained trajectories, it touches no fly wrapper or kernel. For
        r-z the trace has no azimuth (axisymmetric), so pts lie in the
        phi=0 meridian plane with z = 0 and the record says so; the
        /flight tab separately REFUSES planar subjects
        — the record still banks, the refusal is the viewer's to state.

        A publish failure REPORTS on the status line and never breaks
        the fly that produced the results: the flight happened; only
        the banking failed, and silently losing THAT fact is the
        failure mode."""
        try:
            from ion_gym.ui import last_flight
            # ONCE PER RESULT SET (eight 'banked #N'
            # stamps for one flight): every stored redraw — plane
            # change, post-flight refresh — re-published the SAME
            # results and churned the record stamp. Results lists are
            # append-only during a run, so (identity, length) equal to
            # the last publish means identical content: skip, the
            # banked record is already current. A live list that has
            # GROWN republishes (length differs); a reloaded older run
            # republishes (identity differs).
            _pub_key = (id(results), len(list(results)))
            if getattr(self, "_last_pub_key", None) == _pub_key:
                return
            cx, cy = self._ci("x"), self._ci("y")
            cz, ct = self._ci("z"), self._ci("t_us")
            if ct is None:
                # KE once coloured the whole trace: the
                # tracer records its time column as 't' (µs) — the
                # 't_us'-only lookup returned None, no time was banked,
                # and every parameter scheme (speed/time/KE) refused
                # into the per-ion ramp. Index 0 is falsy, so this is a
                # None-check, never an `or` chain.
                ct = self._ci("t")
            if cx is None or cy is None:
                raise ValueError(
                    f"trace columns lack x/y (cols={self._cols}) — "
                    f"cannot bank a spatial path")
            note = ""
            if cz is None:
                note = ("r-z flight: no azimuth (axisymmetric) — paths "
                        "drawn in the phi=0 meridian plane, z=0")
            # SUBSET BY THE ONE DECLARED POLICY (trace_keep_count), the
            # same one the dashboard view uses. This path used to bank
            # EVERY ion at full sample resolution, which at 1000 ions x
            # 400k samples is ~13 GB built here and copied again by
            # last_flight._freeze_path -- on the document-lock
            # coroutine, so all of it was UI-freeze time, for a viewer
            # that draws a subset anyway. Evenly strided so the banked
            # set SPANS the packet rather than taking the first N, and
            # DISCLOSED in the record note so a subset can never read as
            # the packet.
            #
            # Scope: the VIEWER's copy only. Statistics, impact markers,
            # the stats cards and both npz save paths still use every
            # result at full float64 -- subsetting what is drawn never
            # subsets what is measured or saved.
            from ion_gym.physics.staged_flight import trace_keep_count
            _flown = list(results)
            _keep = trace_keep_count(len(_flown))
            _idx = (range(len(_flown)) if _keep >= len(_flown)
                    else np.linspace(0, len(_flown) - 1, _keep).astype(int))
            paths = []
            for i in _idx:
                r = _flown[int(i)]
                traj = getattr(r, "traj", None)
                if traj is None or not len(traj):
                    continue
                a = np.asarray(traj, float)
                z = a[:, cz] if cz is not None else np.zeros(len(a))
                pts = np.column_stack([a[:, cx], a[:, cy], z])
                p = {"pts": pts, "label": f"ion {int(i)}"}
                if ct is not None:
                    p["t_us"] = a[:, ct]
                paths.append(p)
            if _keep < len(_flown):
                _sub = (f"showing {len(paths)} of {len(_flown)} flown ion "
                        f"paths (evenly strided across the packet); "
                        f"statistics and saved files use every ion")
                note = f"{note} — {_sub}" if note else _sub
            # ONE mass for the packet, or none: mz_list with a single
            # entry (x charge) is a declared mass; several entries mean
            # per-ion masses this tap cannot attribute — m stays None
            # and the KE scheme refuses by name downstream.
            _mz = list(getattr(self.spec.source, "mz_list", []) or [])
            # mz_list carries MASSES (ruled convention 2026-09-09);
            # the former *charge factor published a doubled mass at z=2
            _m = float(_mz[0]) if len(_mz) == 1 else None
            rec = last_flight.publish(
                site=site,
                subject={"kind": "single", "spec": self.spec.to_dict()},
                paths=paths, n_flown=len(list(results)), note=note,
                m_amu=_m)
            # The stamp rides the COMPLETION message instead of
            # append-chaining onto whatever status is showing (a
            # chained ' · /flight banked #5 · #6 · …'
            # read as noise). Success is otherwise quiet; the Status
            # tab's rolling log already timestamps everything.
            self._last_pub_key = _pub_key
            self._last_flight_stamp = rec["stamp"]
        except Exception as e:
            # A publish failure still REPORTS — as its own
            # message, not an append: the rolling log preserves the
            # message it would have covered.
            self.status.object = (f"**/flight NOT banked:** "
                                  f"{type(e).__name__}: {e}")

    def _publish_last_flight_assembly(self, traces, n_flown: int) -> None:
        """Assembly site: world-frame per-region traces
        concatenated per ion; the record carries the retained instrument
        DOCUMENT (self-contained — the tab draws the geometry that
        flew). Same failure contract as the single-stage tap."""
        try:
            from ion_gym.ui import last_flight
            paths = []
            for t in traces or []:
                xs, ys, zs, ts = [], [], [], []
                for rg in t.get("regions", []):
                    xs.append(np.asarray(rg["x"], float))
                    ys.append(np.asarray(rg["y"], float))
                    zs.append(np.asarray(rg["z"], float))
                    ts.append(np.asarray(rg["t_us"], float))
                if not xs:
                    continue
                pts = np.column_stack([np.concatenate(xs),
                                       np.concatenate(ys),
                                       np.concatenate(zs)])
                paths.append({"pts": pts,
                              "t_us": np.concatenate(ts),
                              "label": f"ion {t.get('i', '?')} "
                                       f"[{t.get('fate', '?')}]"})
            _b = (self._assembly_doc or {}).get("beam", {}) or {}
            _m = (float(_b["mz"])              # beam mz carries MASS (Da)
                  if isinstance(_b.get("mz"), (int, float)) else None)
            _det = [(float(p["x"]), float(p.get("y", 0.0)),
                     float(p.get("z", 0.0)))
                    for p in (getattr(self, "_impact_hits", None) or [])
                    if p.get("x") is not None]
            rec = last_flight.publish(
                site="fly assembly",
                subject={"kind": "assembly", "doc": self._assembly_doc},
                paths=paths, n_flown=int(n_flown), m_amu=_m,
                detections=(_det or None))
            # Same discipline as the single-stage tap:
            # the stamp rides the completion message via
            # _last_flight_stamp; success does not append-chain onto
            # the visible status.
            self._last_flight_stamp = rec["stamp"]
        except Exception as e:
            self.status.object = (f"**/flight NOT banked:** "
                                  f"{type(e).__name__}: {e}")

    def _impact_from_results(self, results):
        """Feed the Impact Analysis tab from a SINGLE-STAGE flight.

        THE GAP THIS CLOSES: only the assembly fly
        path populated `_impact_hits`, so a plain FA with a declared
        station showed NOTHING in Impact Analysis no matter how many ions
        crossed it -- the ions went through, the tab stayed empty, and the
        two together read as "no signal" when the truth was "nobody
        looked".

        Uses the SAME `station_hits` machinery the Stats card uses (one
        crossing convention, not two). Detect semantics: an ion's FIRST
        in-window crossing is its event; `record` stations log every
        in-window crossing, because a transparent plane's signal IS the
        multiple passes.
        """
        try:
            stns = list(getattr(self.spec, "stations", None) or [])
            if not self._cols:
                return
            # NO early return on "no stations" any more: boundary faces
            # feed this tab too (unified surfaces).
            from ion_gym.physics.stations import station_hits
            hits = []
            for r in results:
                traj = getattr(r, "traj", None)
                if traj is None or not len(traj):
                    continue
                for st in stns:
                    hs = [h for h in station_hits(traj, self._cols, st)
                          if h.get("in_window")]
                    if not hs:
                        continue
                    take = (hs[:1] if (st.kind == "detect" and
                                       getattr(st, "on_hit", None)
                                       == "splat") else hs)
                    for h in take:
                        hits.append(dict(station=st.name, x=h.get("x"),
                                         y=h.get("y"), z=h.get("z"),
                                         t_us=h.get("t_us"),
                                         mz=r.summary.get("mz")))
            # BOUNDARY FACES JOIN THE SAME STREAM:
            # an ion that terminated on an enabled bound is an impact on
            # that face, in exactly the shape the plot consumes. The
            # summary carries the EXACT terminal state (x/y/z_end, tof),
            # so no decimation slop; the face an ion belongs to is the
            # enabled plane its pinned coordinate sits on (half-pitch
            # tolerance), labeled by enabled_planes() -- the stats
            # table's own naming authority.
            from ion_gym.physics.stats import enabled_planes
            # SIDE-AWARE face assignment. Termination happens ON
            # CROSSING, so the terminal coordinate sits AT OR PAST the
            # bound by up to a step -- a symmetric half-pitch window
            # dropped most ejected ions (measured: 1 of 6 assigned on the
            # quad deck). A face claims an ion when the pinned coordinate
            # is at-or-beyond it (small inward tolerance for exact
            # landings); with several candidates the LARGEST penetration
            # wins -- that is the face that ended the flight.
            _h = float(getattr(self.spec.geometry, "mm_per_gu", 0.0) or 0.0)
            _tol = max(_h / 2.0, 1e-6)
            _b = getattr(self.spec, "bounds", None)
            _axi = {"x": "x_end", "y": "y_end", "z": "z_end"}
            _faces = []
            for lbl, ax, val in enabled_planes(self.spec):
                side = ("min" if (_b is not None and
                                  getattr(_b, f"{ax}_min_on", False) and
                                  float(getattr(_b, f"{ax}_min")) ==
                                  float(val)) else "max")
                _faces.append((lbl, str(ax), float(val), side))
            for r in results:
                sm = getattr(r, "summary", {}) or {}
                if sm.get("kind") not in (1, 3):
                    continue
                best = None   # (penetration, label)
                for lbl, ax, val, side in _faces:
                    key = _axi.get(ax)
                    if key is None or sm.get(key) is None:
                        continue
                    c = float(sm[key])
                    pen = (c - val) if side == "max" else (val - c)
                    if pen >= -_tol and (best is None or pen > best[0]):
                        best = (pen, lbl)
                if best is not None:
                    hits.append(dict(
                        station=best[1], x=sm.get("x_end"),
                        y=sm.get("y_end"), z=sm.get("z_end"),
                        t_us=sm.get("tof"), mz=sm.get("mz")))
            self._impact_hits = hits
            _names = sorted({h["station"] for h in hits})
            if _names:
                self.w_impact_station.options = _names
                if self.w_impact_station.value not in _names:
                    self.w_impact_station.value = _names[0]
                for _w in (self.w_impact_station, self.w_impact_plane,
                           self.w_impact_hist_axis):
                    _w.disabled = False
            self._draw_impact()
        except (AttributeError, KeyError, ValueError) as e:
            # Report-and-continue, same contract as _station_stats: an
            # impact view is never worth losing a drawn flight over, but
            # the failure is NAMED, not passed.
            print(f"[sim_app] impact analysis unavailable "
                  f"({type(e).__name__}: {e}); the tab keeps its last "
                  f"contents this run")


    # ---------------- beam declaration panel -----------------------
    def _beam_panel(self):
        """Declare the source: FORMAT chooses fields, DISTRIBUTION fills them.

        These controls edit the INSTRUMENT'S
        BEAM when an instrument is loaded, and a stage's own source
        otherwise -- the tab means "the beam that flies". The banner names
        which, because the same controls silently changing meaning with
        load state is the ambiguity B trades for seamlessness, and the
        banner is what buys it back.

        Everything here RESETS when the configuration changes
        (`_clear_beam_panel`, wired to the same load path).
        A format or a field surviving from the previous instrument would
        describe a packet the new one does not have.
        """
        if getattr(self, "_beam_col", None) is not None:
            return self._beam_col
        from ion_gym.physics.beam_spec import (POSITION_FORMATS,
                                               VELOCITY_FORMATS,
                                               DISTRIBUTIONS)
        self.w_beam_banner = pn.pane.Markdown("")
        # SINGLE ION-COUNT AUTHORITY: the number of
        # ions is set ONCE, in the Basic tab ('ions per m/z'). This
        # control is a read-only mirror so the Advanced tab still SHOWS
        # the count it will fly, but two editable definitions of the same
        # number cannot disagree.
        self.w_beam_pf = pn.widgets.Select(
            name="position format", options=list(POSITION_FORMATS),
            value="xyz", width=170)
        self.w_beam_vf = pn.widgets.Select(
            name="velocity format", options=list(VELOCITY_FORMATS),
            value="direction_ke", width=170)
        self.w_beam_axis = pn.widgets.Select(
            name="propagation axis", options=["x", "y", "z"], value="x",
            width=130, visible=False)
        self._beam_fields = pn.Column()      # rebuilt per format
        self.w_beam_derived = pn.pane.Markdown("", styles={"font-size": "11px"})
        self.w_beam_apply = pn.widgets.Button(
            name="Regenerate beam", button_type="primary", width=150)
        self.w_beam_msg = pn.pane.Markdown("", styles={"font-size": "11px"})
        self._beam_scalar_w = {}
        self._beam_plane_w = {}
        self._beam_dists = list(DISTRIBUTIONS)
        # WHICH dropdown changed is load-bearing: the partner follows the
        # one the user touched. Without that fact the method cannot tell
        # "entering an envelope" from "leaving one" and drags the format
        # back either way.
        self.w_beam_pf.param.watch(
            lambda *_: self._beam_sync_fields(changed="pf"), "value")
        self.w_beam_vf.param.watch(
            lambda *_: self._beam_sync_fields(changed="vf"), "value")
        self.w_beam_apply.on_click(self._on_beam_regenerate)
        self.w_beam_derive = pn.widgets.Button(
            name="Derive from current packet", width=200)
        self.w_beam_derive.on_click(self._on_beam_derive)
        self._beam_col = pn.Column(
            pn.pane.Markdown("**Beam declaration**"),
            self.w_beam_banner,
            pn.Row(self.w_beam_axis),
            pn.Row(self.w_beam_pf, self.w_beam_vf),
            self._beam_fields,
            self.w_beam_derived,
            pn.Row(self.w_beam_apply, self.w_beam_derive),
            self.w_beam_msg)
        self._beam_sync_fields()
        self._beam_sync_banner()
        return self._beam_col

    def _beam_sync_banner(self):
        """Name the subject these controls edit. See _beam_panel."""
        if getattr(self, "w_beam_banner", None) is None:
            return
        if getattr(self, "_assembly_doc", None):
            self.w_beam_banner.object = (
                "*editing: **the instrument's beam** — this is what "
                "`Fly assembly` flies. Regenerate writes it into the "
                "instrument document.*")
        else:
            self.w_beam_banner.object = (
                "*editing: **this spec's source** — no instrument is "
                "loaded, so the beam belongs to the spec in front of you.*")

    def _beam_scalar_row(self, key):
        """One scalar: its distribution and that distribution's parameters.

        Built from the distribution list, not per-scalar special cases --
        no scalar is privileged, which is the whole point of separating
        format from distribution (an axis cannot be presumed to
        carry a particular kind of spread).
        """
        w_kind = pn.widgets.Select(name=key, options=self._beam_dists,
                                   value="single", width=150)
        w_a = pn.widgets.FloatInput(name="value", value=0.0, width=95)
        w_b = pn.widgets.FloatInput(name="—", value=0.0, width=95,
                                    visible=False)
        def _labels(*_):
            k = w_kind.value
            if k == "single":
                w_a.name, w_b.visible = "value", False
            elif k == "uniform":
                w_a.name, w_b.name, w_b.visible = "min", "max", True
            elif k == "gaussian":
                w_a.name, w_b.name, w_b.visible = "mean", "sigma", True
            elif k == "grid":
                w_a.name, w_b.name, w_b.visible = "min", "max", True
            elif k == "maxwellian":
                w_a.name, w_b.visible = "temperature (K)", False
            elif k == "list":
                w_a.name, w_b.visible = "(use JSON for lists)", False
        w_kind.param.watch(_labels, "value")
        _labels()
        self._beam_scalar_w[key] = (w_kind, w_a, w_b)
        return pn.Row(w_kind, w_a, w_b)

    def _beam_plane_row(self, ax):
        w_on = pn.widgets.Checkbox(name=f"plane {ax}", value=True)
        w_s = pn.widgets.FloatInput(name="size (mm)", value=1.0, width=100)
        w_d = pn.widgets.FloatInput(name="divergence (mrad)", value=10.0,
                                    width=140)
        w_f = pn.widgets.FloatInput(name="focus at (mm)", value=0.0,
                                    width=115)
        for w_ in (w_s, w_d, w_f):
            w_.param.watch(lambda *_: self._beam_sync_derived(), "value")
        self._beam_plane_w[ax] = (w_on, w_s, w_d, w_f)
        return pn.Row(w_on, w_s, w_d, w_f)

    def _beam_sync_fields(self, changed=None):
        """Render exactly the fields the chosen formats require.

        The field list comes from `FORMAT_SCALARS` in beam_spec -- the
        generator's own table -- so the UI can never offer a field the
        generator does not read, or omit one it requires.
        """
        from ion_gym.physics.beam_spec import FORMAT_SCALARS
        pf, vf = self.w_beam_pf.value, self.w_beam_vf.value
        # An envelope is a JOINT statement about position and velocity, so
        # the two dropdowns move TOGETHER -- into it and, equally, OUT of
        # it. The first version only forced INTO the envelope, so leaving
        # it ping-ponged (set position to xyz, the still-envelope velocity
        # dragged it back) and the panel could never be reset. Found by
        # the panel's own test, not by reading.
        #
        # `_beam_pairing` guards the re-entrancy: assigning the partner
        # re-enters this method, and without the guard the two
        # assignments chase each other.
        if getattr(self, "_beam_pairing", False):
            return
        if (pf == "beam_envelope") != (vf == "beam_envelope"):
            self._beam_pairing = True
            try:
                if changed == "pf":
                    # the user moved POSITION; velocity follows it, whether
                    # that means entering the envelope or leaving it
                    self.w_beam_vf.value = (
                        "beam_envelope" if pf == "beam_envelope"
                        else "direction_ke")
                elif changed == "vf":
                    self.w_beam_pf.value = (
                        "beam_envelope" if vf == "beam_envelope" else "xyz")
                else:
                    # programmatic sync with no origin: default to the
                    # non-envelope pair rather than guessing an intent
                    self.w_beam_pf.value = "xyz"
                    self.w_beam_vf.value = "direction_ke"
            finally:
                self._beam_pairing = False
            pf, vf = self.w_beam_pf.value, self.w_beam_vf.value
        self._beam_scalar_w = {}
        self._beam_plane_w = {}
        rows = []
        if pf == "beam_envelope":
            self.w_beam_axis.visible = True
            rows.append(pn.pane.Markdown(
                "*size = half-width here · divergence = half-angle · "
                "focus at = distance to the waist (0 = you are at it, "
                "+ = converging downstream).*",
                styles={"font-size": "11px"}))
            for ax in ("x", "y", "z"):
                rows.append(self._beam_plane_row(ax))
            rows.append(self._beam_scalar_row("ke_ev"))
        else:
            self.w_beam_axis.visible = False
            for key in (FORMAT_SCALARS[pf] + FORMAT_SCALARS[vf]):
                rows.append(self._beam_scalar_row(key))
        rows.append(self._beam_scalar_row("tob_us"))
        self._beam_fields.objects = rows
        self._beam_sync_derived()

    def _beam_sync_derived(self):
        """Derived read-out: emittance first, Twiss beneath and read-only.

        Emittance is the INVARIANT -- size and divergence trade against
        each other as the beam drifts while their product at the waist
        does not -- so it is the number worth watching. Twiss is shown
        because anyone who thinks in beta/alpha should be able to read
        them, not because anyone should need to.
        """
        if not self._beam_plane_w:
            self.w_beam_derived.object = ""
            return
        from ion_gym.physics.beam_spec import (envelope_to_twiss,
                                               BeamSpecError)
        out = []
        for ax, (w_on, w_s, w_d, w_f) in self._beam_plane_w.items():
            if not w_on.value:
                continue
            try:
                eps, beta, alpha = envelope_to_twiss(
                    w_s.value, w_d.value, w_f.value)
            except BeamSpecError as e:
                out.append(f"**{ax}: refused** — {e}")
                continue
            out.append(f"**{ax}**  ε = {eps * 1e3:.4g} mm·mrad "
                       f"*(invariant)*  ·  β {beta:.4g} m, α {alpha:+.4g}")
        self.w_beam_derived.object = "<br>".join(out)

    def _beam_declaration(self):
        """Assemble the declaration dict from the rendered fields."""
        decl = {"n": int(self.w_n.value),
                "seed": self._ui_seed(),
                "position_format": self.w_beam_pf.value,
                "velocity_format": self.w_beam_vf.value,
                "scalars": {}}
        for key, (w_kind, w_a, w_b) in self._beam_scalar_w.items():
            k = w_kind.value
            if k == "single":
                d = {"kind": "single", "value": float(w_a.value)}
            elif k in ("uniform", "grid"):
                d = {"kind": k, "min": float(w_a.value),
                     "max": float(w_b.value)}
            elif k == "gaussian":
                d = {"kind": "gaussian", "mean": float(w_a.value),
                     "sigma": float(w_b.value)}
            elif k == "maxwellian":
                d = {"kind": "maxwellian",
                     "temperature_k": float(w_a.value)}
            else:
                raise ValueError(
                    f"{key}: the {k!r} distribution is declared in JSON, "
                    f"not in this panel")
            if k in ("uniform", "gaussian", "maxwellian"):
                d["seed"] = self._ui_seed()
            decl["scalars"][key] = d
        if self.w_beam_pf.value == "beam_envelope":
            decl["axis"] = self.w_beam_axis.value
            # CARRY THE FIELDS THE PANEL DOES NOT SHOW. A derived
            # declaration knows the plane's centre, its MEAN ANGLE and its
            # measured shape, and the axial sign; the panel exposes only
            # size/divergence/focus. Rebuilding from the visible fields
            # alone silently dropped the mean angle -- deleting the
            # analyzer's z-drift and losing every ion (0/5000, caught by
            # this panel's own test). The visible fields override; the
            # invisible ones are preserved.
            _d = getattr(self, "_derived_decl", None) or {}
            if _d.get("axis_sign") is not None:
                decl["axis_sign"] = _d["axis_sign"]
            _dp = _d.get("planes") or {}
            decl["planes"] = {}
            for ax, (w_on, w_s, w_d, w_f) in self._beam_plane_w.items():
                if not w_on.value:
                    continue
                base = dict(_dp.get(ax) or {})
                base.update({"size_mm": float(w_s.value),
                             "divergence_mrad": float(w_d.value),
                             "focus_mm": float(w_f.value),
                             "seed": self._ui_seed()})
                decl["planes"][ax] = base
            for _k in ("ke_ev", "tob_us"):
                # the derived KE/tob distributions likewise carry shape
                # (uniform vs gaussian) the panel's simple rows cannot
                if (_k in (_d.get("scalars") or {})
                        and _k not in self._beam_scalar_w):
                    decl["scalars"][_k] = _d["scalars"][_k]
            _ax_off = f"{self.w_beam_axis.value}_mm"
            if _ax_off in (_d.get("scalars") or {}):
                decl["scalars"].setdefault(_ax_off, _d["scalars"][_ax_off])
        return decl

    def _on_beam_regenerate(self, _=None):
        """Realise the declaration and write it into the document.

        Spec -> JSON box -> instrument document, the same contract the
        station editor uses. The document is what `Fly assembly` flies
        so a regenerated packet that stopped short of it would
        not be the packet that flies.

        The declaration is STAMPED alongside the rows (`beam.source`).
        An inline packet with no record of what produced it is a number
        without an operating point -- and now that `beam.ions` can be
        overwritten, provenance is the only thing that makes a banked
        result reproducible.
        """
        from ion_gym.physics.beam_spec import generate_packet, BeamSpecError
        if not getattr(self, "_assembly_doc", None):
            self.w_beam_msg.object = (
                "**No instrument loaded.** These controls describe the "
                "beam an instrument flies; load one in the Multi FA tab.")
            return
        beam = self._assembly_doc.get("beam") or {}
        mz = float(beam.get("mz", 0) or 0)
        if mz <= 0:
            self.w_beam_msg.object = (
                "**The instrument declares no m/z**, and a beam cannot be "
                "generated without one: energy, speed and temperature all "
                "convert through mass.")
            return
        try:
            decl = self._beam_declaration()
            rows = generate_packet(decl, mass_amu=mz)
        except (BeamSpecError, ValueError) as e:
            self.w_beam_msg.object = f"**refused:** {e}"
            return
        prev = len(beam.get("ions") or [])
        _unstamped = bool(beam.get("ions")) and not beam.get("source")
        beam["ions"] = rows
        beam["source"] = decl
        self._assembly_doc["beam"] = beam
        self.w_beam_msg.object = (
            f"**beam regenerated — {len(rows)} ions** (was {prev}), "
            f"written into the instrument document with its generator "
            f"parameters and seed {decl['seed']}. *Fly assembly* now flies "
            f"this packet; save the instrument to keep it."
            + ("  \n**Note: the packet you replaced carried no generator "
               "stamp** — it was declared as literal rows, so it cannot be "
               "regenerated from this panel. If it was a certified tune, "
               "reload the instrument from disk to get it back."
               if _unstamped else ""))

    def _clear_beam_panel(self):
        """Reset the beam panel on a configuration change.

        Same stale-state discipline: a format or a field surviving from
        the previous instrument describes a packet the new one does not
        have, and a control that describes something absent is worse than
        a disabled one.
        """
        if getattr(self, "_beam_col", None) is None:
            return
        self.w_beam_msg.object = ""
        # Both formats set under the pairing guard, so leaving an
        # envelope does not drag the partner back (see _beam_sync_fields).
        self._beam_pairing = True
        try:
            self.w_beam_pf.value = "xyz"
            self.w_beam_vf.value = "direction_ke"
        finally:
            self._beam_pairing = False
        self.w_beam_axis.value = "x"
        beam = (getattr(self, "_assembly_doc", None) or {}).get("beam") or {}
        rows = beam.get("ions") or []
        if rows:
            self.w_beam_msg.object = (
                f"*document packet: {len(rows)} ions; count and seed are "
                f"governed by the header controls ('ions per m/z', "
                f"'seeded run').*")
        src = beam.get("source")
        if isinstance(src, dict):
            # The document remembers HOW its packet was made: reopen
            # showing the fields the user typed, not a back-conversion.
            for attr, key in ((self.w_beam_pf, "position_format"),
                              (self.w_beam_vf, "velocity_format")):
                if src.get(key):
                    attr.value = src[key]
            if src.get("axis"):
                self.w_beam_axis.value = src["axis"]
            if src.get("seed") is not None:
                # a document that declares its seed IS the seeded option:
                # reflect it on the master control.
                self.w_seedval.value = int(src["seed"])
                self.w_seeded.value = True
        self._beam_sync_fields()
        self._beam_sync_banner()


    def _on_beam_derive(self, _=None):
        """Fit a declaration to the packet the instrument already carries.

        WHY: a packet declared as literal rows cannot be resampled --
        there is no distribution to draw more from. Deriving turns the
        rows into a DESCRIPTION so the same beam can be drawn at any N,
        which is what building a statistically useful Impact Analysis
        histogram needs.

        THE RESIDUAL IS REPORTED LOUDLY AND IS NOT A FORMALITY. A linear
        phase-space ellipse is three numbers per plane; a packet carrying
        aberration is a curved filament that no three numbers describe.
        Measured on a certified packet: residual 0.59 in y, and
        regenerating from the fit gives R about 20% OPTIMISTIC because the
        idealised ellipse focuses better than the real beam. That is a
        finding to act on, not a number to bury.
        """
        from ion_gym.physics.beam_spec import (derive_declaration,
                                               BeamSpecError)
        doc = getattr(self, "_assembly_doc", None)
        beam = (doc or {}).get("beam") or {}
        rows = beam.get("ions") or []
        if not rows:
            self.w_beam_msg.object = (
                "**No packet to derive from.** Load an instrument that "
                "declares `beam.ions`.")
            return
        try:
            decl = derive_declaration(rows, mass_amu=float(beam["mz"]),
                                      seed=self._ui_seed())
        except (BeamSpecError, KeyError, ValueError) as e:
            self.w_beam_msg.object = f"**derive refused:** {e}"
            return
        self._beam_pairing = True
        try:
            self.w_beam_pf.value = "beam_envelope"
            self.w_beam_vf.value = "beam_envelope"
        finally:
            self._beam_pairing = False
        self.w_beam_axis.value = decl["axis"]
        self._beam_sync_fields()
        for ax, p_ in decl["planes"].items():
            if ax in self._beam_plane_w:
                w_on, w_s, w_d, w_f = self._beam_plane_w[ax]
                w_on.value = True
                w_s.value = float(p_["size_mm"])
                w_d.value = float(p_["divergence_mrad"])
                w_f.value = float(p_["focus_mm"])
        self._derived_decl = decl
        _ke = (decl.get("scalars") or {}).get("ke_ev") or {}
        if "ke_ev" in self._beam_scalar_w and _ke.get("kind"):
            w_kind, w_a, w_b = self._beam_scalar_w["ke_ev"]
            w_kind.value = _ke["kind"]
            if _ke["kind"] == "single":
                w_a.value = float(_ke["value"])
            elif _ke["kind"] == "uniform":
                w_a.value, w_b.value = float(_ke["min"]), float(_ke["max"])
            elif _ke["kind"] == "gaussian":
                w_a.value, w_b.value = float(_ke["mean"]), float(_ke["sigma"])
        res = decl["derived_from"]["residual_frac"]
        worst = max(res.values()) if res else 0.0
        warn = ""
        if worst > 0.3:
            warn = (f"  \n**⚠ the ellipse describes this packet only "
                    f"approximately** (worst residual {worst:.2f} — 0 is a "
                    f"perfect ellipse). Regenerating from it produces a "
                    f"MORE IDEAL beam than the original, so resolution "
                    f"will read optimistically. Measured on the certified "
                    f"OA-MRT packet: about +20% in R. Use it for "
                    f"statistics and distribution shape, NOT to re-certify "
                    f"a tune.")
        self.w_beam_msg.object = (
            f"**derived from {len(rows)} ions** — axis {decl['axis']}, "
            f"residuals " + ", ".join(f"{k} {v:.2f}" for k, v in res.items())
            + ". Set the ion count and press *Regenerate beam*." + warn)
        self._beam_sync_derived()

    def _sync_plane_controls(self):
        """Enable the plane buttons only where they GOVERN something.

        Single-stage: untouched. Capability there is declared per route by
        `_planes_for_spec` / `_sync_planes`, and a planar deck deliberately
        still OFFERS xz/yz carrying the projection warning --
        "offered, but with a warning". Disabling them
        here would silently reverse that.

        Whole assembly: DISABLED, because `assembly_overview` is
        inherently multi-axis -- it draws the xz and xy panels together by
        directive, since an assembly is a 3-D object even when every stage
        solves in 2-D and a single projection could hide a stage displaced
        along the suppressed axis. There is no third state for a plane
        button to select, so leaving them live would be a control that
        appears to do something and does not.

        INTERIM, and recorded as such: the better answer is for
        the overview to honour the selected plane and choose which
        orthogonal PAIR to show, which keeps the buttons meaningful and
        keeps the multi-axis directive. Disabling is the honest stopgap,
        not the design.
        """
        _assembly = self._subject_is_assembly()
        for _b in (getattr(self, "view_xy", None),
                   getattr(self, "view_xz", None),
                   getattr(self, "view_yz", None)):
            if _b is not None:
                _b.disabled = _assembly

    def _on_set_fly_params(self, _=None):
        """*Set Fly Parameters*: PIN the chosen FA as the Ion Source
        tab's binding (superseding the
        session-5 make-it-live mechanism). The tab binds to the pinned
        stage's spec object in _assembly_specs — the same single
        authority the assembly flies from, no copy to drift — and the
        LIVE spec (display/solve subject) is NOT touched, so *Set FA
        View* no longer retargets the source panel: fly parameters and
        view are independent axes in fact, not just in a docstring. The
        instrument's default source FA is its own declared
        beam.from_stage; count edits beyond the declared packet
        regenerate the stamped beam at fly time."""
        fa = self.w_fly_src.value
        if not fa or fa not in (self._assembly_specs or {}):
            self.status.object = ("**pick a flight-parameters FA "
                                  "first** — the instrument's default "
                                  "is its beam.from_stage.")
            return
        self._fly_src_pin = fa
        self._build_controls()     # re-entrant refresh: tabs update in
        #                            place, source widgets now read the
        #                            pin; view and live spec untouched
        _src = (self._assembly_doc.get("beam") or {}).get("from_stage")
        self.status.object = (
            f"**Ion Source PINNED to FA `{fa}`** (instrument default "
            f"source FA: `{_src}`). Edits there apply to that stage "
            f"regardless of which FA is displayed; the ion COUNT beyond "
            f"the declared packet regenerates the stamped beam at fly "
            f"time.")

    def _apply_fa_view(self, _=None):
        """*Set FA View*: 'Full Assembly' -> the assembly
        subject; an FA name -> that stage's view + controls."""
        v = self.w_stage.value
        if v in (None, "", "Full Assembly"):
            self._on_view_assembly()
            return
        if v not in (self._assembly_specs or {}):
            self.status.object = (f"**unknown FA {v!r}** — the picker "
                                  f"offers Full Assembly plus: "
                                  f"{', '.join(self._assembly_specs or [])}")
            return
        self._show_stage(v)

    def _show_stage(self, name):
        _was_loading = self._loading_doc
        self._loading_doc = True
        try:
            return self._show_stage_inner(name)
        finally:
            self._loading_doc = _was_loading

    def _show_stage_inner(self, name):
        """Make the SUBJECT live: one stage's spec, or the whole assembly.

        For a stage name this makes that stage the live spec, so every
        existing view and solve applies to it with no assembly-specific
        rendering path.

        For `<whole assembly>` there is no single live spec to install --
        the assembly is not a spec, it is a set of posed ones -- so the
        live spec is left untouched and only the drawn subject changes.
        That asymmetry is deliberate and is why the solve and edit paths
        below REFUSE the assembly subject by name rather than quietly
        acting on whichever stage happened to be live last: acting on a
        stage the user did not select is worse than declining.
        """
        if name == WHOLE_ASSEMBLY:
            self._assembly_stage = name
            self._sync_assembly_notices()
            self._live_stage_name = name
            self._sync_plane_controls()
            self._redraw_subject()
            return
        spec = self._assembly_specs.get(name)
        if spec is None:
            raise KeyError(
                f"stage {name!r} is not in the loaded assembly "
                f"({sorted(self._assembly_specs)})")
        if hasattr(self, "w_apply_busy"):
            self.w_apply_busy.value = True
        self._stage_swapping = True
        try:
            self.spec = spec
            self._assembly_stage = name
            self._sync_assembly_notices()
            self._live_stage_name = name
            self._sync_plane_controls()
            self._rebuild_for_new_spec()
            self.w_pitch.value = float(self.spec.geometry.mm_per_gu)
            self._refresh_sizing()
        finally:
            self._stage_swapping = False
            if hasattr(self, "w_apply_busy"):
                self.w_apply_busy.value = False

    def _on_stage_selected(self, evt=None):
        """SELECTION ONLY: the FA View dropdown
        records a choice; *Set FA View* applies it. The old auto-switch
        fired the full stage swap from every programmatic resync, which
        is the reentrancy class the 89-rebuild incident lived in — a
        button-applied model retires it. Callers that need a swap call
        _show_stage or _apply_fa_view directly."""
        return

    # Analysis was REMOVED from this set when option A wired it for
    # assembly flights — a banner calling a working tool
    # unavailable is the same defect as silence about a broken one.
    _ASM_NOTICE_TABS = ("Thermal", "PE Surface",
                        "Field Slice", "Raster")

    def _sync_assembly_notices(self):
        """A visible per-tab notice while a multi-FA instrument is the
        subject (the user is notified when
        these don't work for a multiFA json"). Inserted at the top of
        each affected tab when the subject becomes the assembly,
        removed when an FA view is set — never silent either way."""
        if not hasattr(self, "plot_tabs"):
            return
        is_asm = self._subject_is_assembly()
        for lbl in self._ASM_NOTICE_TABS:
            if lbl not in self.plot_tabs._names:
                continue
            cont = self.plot_tabs[self.plot_tabs._names.index(lbl)]
            if not hasattr(cont, "insert") or not hasattr(cont, "pop"):
                continue
            has = (len(cont) > 0
                   and getattr(cont[0], "_asm_notice", False))
            if is_asm and not has:
                note = pn.pane.Markdown(
                    "⚠ **Multi-FA instrument displayed** — this tool "
                    "is not available for a multi-FA flight; it operates "
                    "on the single displayed FA. Use **Set FA View** to "
                    "analyze one FA. *(Analysis, Impact Analysis and "
                    "the Stats card DO cover assembly flights.)*",
                    styles={"background": "#fff4e5", "padding": "6px",
                            "border-radius": "4px"})
                note._asm_notice = True
                cont.insert(0, note)
            elif (not is_asm) and has:
                cont.pop(0)

    def _on_view_assembly(self, _=None):
        """Draw every stage in ONE world frame, so placement is checkable
        by eye rather than by reading three coordinate systems.

        This is a MODE, not just a look — it sets
        the SUBJECT to the whole assembly, which is what the single Fly
        button routes on ("what you see is what flies") and what a plane
        change redraws. The earlier "look, not a mode" reading is
        superseded: nothing ever set the subject to WHOLE_ASSEMBLY, so
        the subject predicate could never be true and assembly state
        lived in a checkbox instead of in what was on screen.
        """
        if not self._assembly_doc:
            self.status.object = ("**no assembly loaded** — upload an "
                                  "instrument JSON first.")
            return
        self._assembly_stage = WHOLE_ASSEMBLY
        self._sync_assembly_notices()
        try:
            stages, seams = [], []
            for st in (self._assembly_doc.get("stages") or []):
                nm = st["name"]
                spec = self._assembly_specs.get(nm)
                if spec is None:
                    raise KeyError(f"stage {nm!r} was not parsed")
                pose = st.get("pose") or {}
                # ONE POSE AUTHORITY. This used to read
                # `origin_mm` while the flight read `offset_mm`, so an
                # off-origin stage would DRAW displaced and FLY coaxial.
                # Parsing through the loader's own `_pose_from` means the
                # picture is built from exactly what the flight will
                # honour -- including its refusal of an unknown key --
                # rather than from a second reading of the same document.
                from ion_gym.physics.staged_flight import _pose_from
                _p = _pose_from(pose)
                stages.append((nm, spec, list(_p.offset_mm), _p.rot_deg))
                ex = st.get("exit")
                # AXIS-AWARE (a z-seam instrument showed
                # "0 seam(s) marked" — only x-axis exits were collected).
                if ex and ex.get("axis") in ("x", "y", "z"):
                    seams.append({"axis": ex.get("axis"),
                                  "value_mm": ex.get("value_mm", 0.0),
                                  "label": f"seam after {nm}"})
            # ONE PLANE AT A TIME, DRIVEN BY THE VIEW BUTTONS:
            # the full assembly view
            # needs to be linked to the buttons on the top of the view
            # pane. Don't smash them all on one view"). w_plane already
            # re-draws this subject on change, so the buttons
            # were live — they just weren't consulted here.
            _pl = self.w_plane.value if self.w_plane.value in (
                "xy", "xz", "yz") else "xz"
            _det = [(p["x"], p.get("y", 0.0), p.get("z", 0.0))
                    for p in (getattr(self, "_impact_hits", None) or [])
                    if p.get("x") is not None]
            self.pane.object = V.assembly_overview(
                stages, seams=seams,
                trajs=getattr(self, "_assembly_traces", None),
                traj_style=dict(
                    width=float(self.w_width.value),
                    alpha=float(self.w_alpha.value),
                    decim=int(self.w_decim.value),
                    color_by=self.w_colorby.value,
                    solid=self.w_solidcolor.value,
                    # The Display tab's
                    # REMAINING trace conditions now travel too —
                    # trajectory style (lines/dots/both) and the impact
                    # marker size/symbol were never passed, so the
                    # assembly view drew hardcoded lines and diamonds
                    # whatever the tab said.
                    mode=self.w_trajmode.value,
                    impact_size=float(self.w_impactsize.value),
                    impact_symbol=self.w_impactsym.value),
                detections=(_det or None),
                plane=_pl,
                fill_electrodes=(True if getattr(self, "w_elfill", None)
                                 is None else bool(self.w_elfill.value)))
            _rot = [nm for nm, _s, _o, r in stages
                    if V.assembly_stage_draw_mode(_s, r) == "placeholder"]
            _rotmsg = ""
            if _rot:
                # The figure title carries this too, but a warning only in
                # the title is a warning a user can scroll past while
                # reading placement off the picture. Say it where every
                # other outcome of this action is reported.
                _rotmsg = (
                    f" **⚠ {len(_rot)} ROTATED STAGE(S) — "
                    f"{', '.join(_rot)} — DRAWN AS PLACEHOLDERS.** Their "
                    f"pose is known exactly and IS flown; this renderer "
                    f"cannot draw a rotated cross-section, so the dashed "
                    f"box is a bounding placeholder, not the true "
                    f"footprint. Do not read placement from it (L-162).")
            # Retained, because `_on_stage_selected` writes the status
            # AFTER this runs (it calls the redraw, then reports the
            # subject) and would otherwise clobber the warning -- leaving
            # it visible only in the figure title, which is the surface a
            # user reading placement is least likely to check.
            self._assembly_rot_warning = _rotmsg
            self.status.object = (
                f"**assembly view** — {len(stages)} stage(s) in the world "
                f"frame, {len(seams)} seam(s) marked. Both panels put x "
                f"horizontal; a stage displaced in y or z shows in the "
                f"panel that carries it. Pick a stage above to go back to "
                f"the single-stage view." + _rotmsg)
        except Exception as e:
            self.status.object = f"**assembly view failed:** {e}"

    def _on_fly_assembly(self, _=None):
        """Fly the whole instrument — LAUNCH ONLY. The solve + fly run on
        a WORKER THREAD; a periodic callback drains progress into the
        status line (and the console) and runs the completion on the
        loop. Root cause found in a hang dump: this
        used to execute fly_packet — including a MINUTES-long cold numba
        compile of the 3-D kernel — inside the websocket handler on the
        tornado event loop. The entire server froze: no patches, no
        status, no pulse; the browser gave up and the 'change views back
        and forth' ritual was the user manually flushing a dead UI. A
        blocked loop is not a slow fly; it is the absence of an app."""
        if not self._assembly_doc:
            self.status.object = ("**no assembly loaded** — paste an "
                                  "instrument JSON and press Apply.")
            return
        st = getattr(self, "_afly", None)
        if st is not None and not st.get("done"):
            self.status.object = ("**an assembly flight is already "
                                  "running** — progress in this line.")
            return
        # WIDGETS -> SPECS FIRST (a 3-mass m/z list set on
        # the Ion tab flew as 25 ions of the doc's single mass). The
        # session-7 sync moved n_ions explicitly but every OTHER source
        # edit (m/z list, distribution, KE...) lived only in widgets
        # until _sync_spec ran — which the single-stage fly does and this
        # path never did. Source writes land on the PINNED FA.
        self._sync_spec()
        import collections
        import threading
        if hasattr(self, "w_apply_busy"):
            self.w_apply_busy.value = True
        self._afly = {"msgs": collections.deque(), "done": False,
                      "error": None, "mode": None, "out": None,
                      "res": None, "extras": {},
                      "n_req": (int(self.w_n.value)
                                if getattr(self, "w_n", None) is not None
                                else 0)}
        self.status.object = ("**assembly fly launched** — solving "
                              "stages on a worker thread; progress "
                              "streams here. The first 3-D solve "
                              "compiles kernels and can take minutes "
                              "cold; the app stays live.")
        self._fly_chip("begin",
                       total=(self._afly["n_req"] or None))
        threading.Thread(target=self._afly_compute, daemon=True).start()
        import panel as _pn
        self._afly_cb = _pn.state.add_periodic_callback(
            self._afly_poll, period=400)

    def _afly_say(self, msg):
        """Thread-safe progress: banked for the poll callback AND
        printed to the console (there was no robust reporting in the
        UI or console')."""
        print(f"[fly assembly] {msg}", flush=True)
        import time as _t
        self._afly["last_ts"] = _t.time()
        self._afly["msgs"].append(msg)

    def _afly_compute(self):
        """WORKER THREAD: document staging, solves, seam checks, beam
        prep (incl. the stamped-packet auto-extend) and the fly itself.
        Touches NO widgets and NO panes — plain dict/status text only;
        every UI consequence runs in _afly_finish on the loop."""
        st = self._afly
        _say = self._afly_say
        import tempfile
        import time as _time
        from ion_gym.physics.staged_flight import (load_assembly,
                                                   fly_packet,
                                                   fly_staged, check_seam)
        try:
                # FLY-PARAMETERS SYNC (setting an FA as
                # the fly parameters and then setting the ion count,
            # I don't get 500 ions flown"). A from_stage beam is PRODUCED
            # by flying the source FA from its declared source — so the
            # Ion tab's edits to that FA (Set Fly Parameters made it the
            # live spec) must reach the FLOWN document, and a requested
            # count above the declared n raises the source's n_ions in
            # BOTH the live spec and the document. Displayed = flown; the
            # raise is said, never silent.
            _bm0 = self._assembly_doc.get("beam") or {}
            _src_fa = _bm0.get("from_stage")
            if _src_fa and _src_fa in (self._assembly_specs or {}):
                _live = self._assembly_specs[_src_fa]
                _n_src = int(getattr(_live.source, "n_ions", 0) or 0)
                if st["n_req"] and int(st["n_req"]) > _n_src:
                    _live.source.n_ions = int(st["n_req"])
                    _say(f"beam source FA '{_src_fa}': n_ions raised "
                         f"{_n_src} → {st['n_req']} (Ion-tab count) — "
                         f"births produced at the requested n")
                for _stg in self._assembly_doc.get("stages", []):
                    if _stg.get("name") == _src_fa:
                        _stg["spec"]["source"] = _live.source.to_dict()
                        break
                # the headline beam mass follows the source's first mass
                # (the loader generates per-ion masses from the same
                # source — one authority); a stale scalar here labelled
                # every provenance line with a mass that no longer flew
                _bm0["mz"] = float(_live.source.mz_list[0])
            # The JSON box is NOT the authority here: showing a stage
            # rewrites the box with that STAGE'S spec, so by the time this
            # runs the box holds one stage, not the instrument. The
            # retained document is what was loaded, so it is what flies.
            with tempfile.NamedTemporaryFile("w", suffix=".json",
                                             delete=False) as fh:
                json.dump(self._assembly_doc, fh)
                tmp = fh.name
            t0 = _time.time()
            _say("document staged — solving + posing stages (first 3-D "
                 "solve compiles kernels; minutes cold, seconds warm)")
            regions, beam = load_assembly(tmp)
            _say(f"{len(regions)} stage region(s) solved + posed "
                 f"({_time.time() - t0:.1f} s) — checking seams")
            seams = []
            for a, b in zip(regions[:-1], regions[1:]):
                if a.exit is not None:
                    _ok, _e, rep = check_seam(a, b, float(beam["mz"]))
                    seams.append(rep)
            if "ions" in beam:
                # TRACES FOR THE VIEW. The count comes from the
                # instrument's declared policy, not a literal here, so the
                # picture the app draws matches what the document asks for.
                from ion_gym.physics.staged_flight import (trace_policy,
                                                           trace_keep_count)
                # ION COUNT IS A UI LEVER NOW
                # (narrowing). The Ion tab's count selects HOW MANY
                # of the instrument's inline ions fly, by even stride so
                # the subset spans the declared distribution rather than
                # its head. The PHYSICS of the packet stays the
                # document's: the app selects from declared births, it
                # does not invent new ones -- asking for MORE ions than
                # the instrument declares refuses by name instead of
                # duplicating rows, because two ions with identical birth
                # state fly identical vacuum paths and would silently
                # double-count the same trajectory in every histogram.
                # At n = full packet this reproduces the banked numbers
                # bit-for-bit.
                _rows = beam["ions"]
                # ONE COUNT, TWO ROLES. The PACKET's size is declared in
                # the beam panel (Regenerate beam); *ions per m/z* only
                # NARROWS that packet for a quick look. Two independent
                # counts was a wart -- regenerating 500 while the Ion tab
                # still said 200 flew 200, and the assistant papered over
                # it in its own test by setting both.
                _n_req = int(st["n_req"]) if st["n_req"] else len(_rows)
                # MULTI-MASS PACKETS (a 75-entry mass
                # table refused against a 25-row narrowed packet). The
                # Ion-tab count is PER m/z — its own label — so with a
                # per-ion mass table the requested TOTAL is count x
                # masses, narrowing strides WITHIN each contiguous mass
                # block (the blocks convention), and the mass table is
                # narrowed with the rows so they can never misalign.
                # Growth is already handled upstream: the source-count
                # sync raises n_ions and the loader regenerates births,
                # so a larger request never reaches this block with a
                # mass table. Scalar (single-mass / literal) packets
                # take the unchanged path below.
                _mzs_tab = beam.get("mz_per_ion")
                if _mzs_tab is not None:
                    _n_mass = max(1, len(set(float(v) for v in _mzs_tab)))
                    _blk = max(1, len(_rows) // _n_mass)
                    _tot_req = _n_req * _n_mass
                    if _tot_req < len(_rows):
                        _per = max(1, min(_n_req, _blk))
                        _keep_i = []
                        for _b in range(_n_mass):
                            _base = _b * _blk
                            _keep_i += ([_base] if _per == 1 else
                                        [_base + int(round(k * (_blk - 1)
                                                    / (_per - 1)))
                                         for k in range(_per)])
                        _keep_i = list(dict.fromkeys(_keep_i))
                        beam = dict(beam,
                                    ions=[_rows[k] for k in _keep_i],
                                    mz_per_ion=[_mzs_tab[k]
                                                for k in _keep_i])
                        _say(f"packet narrowed for a quick look: "
                             f"{_per} of {_blk} ions per mass x "
                             f"{_n_mass} masses = {len(_keep_i)} flown "
                             f"(Ion-tab count is per m/z)")
                    elif _tot_req > len(_rows):
                        _say(f"flying all {len(_rows)} declared ions "
                             f"({_n_req}/mass x {_n_mass} = {_tot_req} "
                             f"requested): raise the source FA's ions "
                             f"per m/z and re-fly — the source sync "
                             f"regenerates births at load.")
                elif _n_req > len(_rows):
                    # EXTENDING THE PACKET IS NOW AUTOMATIC when the
                    # packet is GENERATOR-STAMPED (a deck
                    # would not fly more than
                    # 200 ions). A stamped
                    # packet carries its declaration + seed, so a larger
                    # one is produced by the SAME declared generator at
                    # the requested n — produced, not duplicated (two
                    # identical birth rows fly identical vacuum paths
                    # and double-count every histogram, which is why
                    # duplication stays refused). The regenerated packet
                    # and its declaration are WRITTEN INTO the retained
                    # document: what the document says is what flew. An
                    # UNSTAMPED literal packet cannot be extended and
                    # refuses by name, exactly as before.
                    _decl = (beam.get("source")
                             or (self._assembly_doc.get("beam") or {}
                                 ).get("source"))
                    if _decl:
                        from ion_gym.physics.beam_spec import (
                            generate_packet, BeamSpecError)
                        try:
                            _prev = len(_rows)
                            _mz = float(
                                (self._assembly_doc.get("beam") or {}
                                 ).get("mz"))
                            _rows = generate_packet(_decl, mass_amu=_mz,
                                                    n=_n_req)
                            _newdecl = dict(_decl, n=_n_req)
                            _bm = self._assembly_doc.setdefault("beam", {})
                            _bm["ions"] = _rows
                            _bm["source"] = _newdecl
                            beam = dict(beam, ions=_rows)
                            _say(
                                f"**beam regenerated at {_n_req} ions** "
                                f"(was {_prev}) from the packet's own "
                                f"generator (seed {_newdecl['seed']}) — "
                                f"written into the document.")
                        except (BeamSpecError, ValueError, TypeError) as e:
                            _say(
                                f"**cannot extend the packet:** "
                                f"{type(e).__name__}: {e} — flying the "
                                f"declared {len(_rows)}.")
                            _n_req = len(_rows)
                    else:
                        _say(
                            f"**flying all {len(_rows)} declared ions** "
                            f"({_n_req} requested): this packet is "
                            f"LITERAL rows with no generator stamp, so "
                            f"it cannot be extended — regenerate it in "
                            f"**Ion Source → Beam declaration** to get "
                            f"a stamped one.")
                        _n_req = len(_rows)
                if _mzs_tab is None and _n_req < len(_rows):
                    _idx = [int(round(k * (len(_rows) - 1)
                                      / max(1, _n_req - 1)))
                            for k in range(_n_req)] if _n_req > 1 else [0]
                    beam = dict(beam, ions=[_rows[k] for k in
                                            dict.fromkeys(_idx)])
                _frac, _floor = trace_policy(self._assembly_doc)
                _keep = trace_keep_count(len(beam["ions"]), _frac, _floor)
                _say(f"flying {len(beam['ions'])} ions "
                     f"({_keep} traces kept) — 3-D kernels compile on "
                     f"first use")
                def _rows_tracked(rows):
                    # PER-ION HEARTBEAT for the flight chip: the worker
                    # writes plain dict fields; the poll renders them on
                    # the loop. len() is safe — rows is the packet list.
                    st["n_fly"] = len(rows)
                    for _ri, _row in enumerate(rows):
                        st["i_fly"] = _ri
                        st["last_ts"] = _time.time()
                        yield _row
                    st["i_fly"] = len(rows)
                out = fly_packet(regions, beam, keep_traces=_keep,
                                 tracker=_rows_tracked)
                _say(f"flight done ({_time.time() - t0:.1f} s total) — "
                     f"finishing on the UI loop")
                st["mode"] = "packet"
                st["out"] = out
                st["extras"] = dict(beam=beam, regions=regions,
                                    seams=seams, t0=t0,
                                    dt=_time.time() - t0)
            else:
                _say("flying single declared ion (p0 path)")
                res = fly_staged(regions, float(beam["mz"]), beam["p0_mm"],
                                 beam["v0_mm_us"],
                                 tob_us=float(beam.get("tob_us", 0.0)))

                st["mode"] = "p0"
                st["res"] = res
                st["extras"] = dict(beam=beam, regions=regions,
                                    seams=seams, t0=t0,
                                    dt=_time.time() - t0)
        except Exception as e:
            st["error"] = f"{type(e).__name__}: {e}"
        finally:
            st["done"] = True

    def _afly_poll(self):
        """LOOP: drain progress into the status line; on completion run
        the finish (all UI) and stand down."""
        st = getattr(self, "_afly", None)
        if st is None:
            return
        while st["msgs"]:
            self.status.object = st["msgs"].popleft()
        if not st["done"]:
            self._fly_chip("progress", done=st.get("i_fly"),
                           total=st.get("n_fly"),
                           activity_ts=st.get("last_ts"))
            return
        try:
            if st["error"] is not None:
                self._fly_chip("error", note=str(st["error"])[:40])
                self.status.object = (f"**assembly flight failed:** "
                                      f"{st['error']}")
            else:
                try:
                    self._afly_finish(st)
                    self._fly_chip("done", done=st.get("i_fly"),
                                   total=st.get("n_fly"))
                except Exception as e:
                    # the FLIGHT succeeded; the UI completion did not.
                    # Naming both is the difference between a losable
                    # callback traceback and a report the user can act
                    # on (the traces are banked; a redraw shows them).
                    self.status.object = (
                        f"**assembly flight FINISHED but the UI "
                        f"completion failed:** {type(e).__name__}: {e} "
                        f"— traces are banked; press a plane button to "
                        f"redraw.")
                    raise
        finally:
            if getattr(self, "_afly_cb", None) is not None:
                try:
                    self._afly_cb.stop()
                except Exception as e:
                    print(f"[fly assembly] callback stop: {e}", flush=True)
                self._afly_cb = None
            if hasattr(self, "w_apply_busy"):
                self.w_apply_busy.value = False

    def _afly_finish(self, st):
        """LOOP: the completion — traces, /flight publish, redraw,
        impact + stats feeds, and the headline status. Verbatim the
        pre-threading completion block; only its thread moved."""
        import time as _time
        # NOTE: `beam = st["extras"]["beam"]` was bound here
        # and never read — removed (F841). The provenance block below
        # reads self._assembly_doc.get("beam") instead, i.e. the doc's
        # CURRENT beam rather than the one this flight actually flew.
        # That is a live difference if the doc is edited while a flight
        # is in flight; it is REPORTED, not changed here, because
        # switching it is a behaviour change and not a lint fix.
        regions = st["extras"]["regions"]
        seams = st["extras"]["seams"]
        t0 = st["extras"]["t0"]
        if st["mode"] == "packet":
            out = st["out"]
            _n_flown = out["n"]
            self._assembly_traces = [dict(t, n_flown=_n_flown)
                                     for t in out.get("traces", [])]
            # Bank this flight for the /flight tab. World-
            # frame regions concatenated per ion; the record carries
            # the INSTRUMENT DOC ITSELF so the tab always draws the
            # geometry that flew, never current dashboard state.
            # RETAINED FOR THE ANALYSIS TAB, moved ABOVE the
            # /flight publish, which banks _impact_hits
            # as THIS flight's detection markers — set-after-use banked
            # the PREVIOUS flight's):
            # assembly flights have per-ion endpoint records, not
            # single-stage result objects — the Analysis tab's assembly
            # branch reads these.
            self._assembly_per_ion = list(out["per_ion"])
            self._impact_hits = [p for p in out["per_ion"]
                                 if p.get("arrived")
                                 and p.get("x") is not None]
            self._publish_last_flight_assembly(
                self._assembly_traces, _n_flown)
            # DRAW THE FLIGHT WHERE THE USER IS LOOKING:
            # after Fly or Fly assembly
            # the trajectories must show up without having to look at
            # an individual sub-FA and then go back"). This
            # completion path fed impact, stats and the status line
            # but never redrew the VIEW — the on-screen assembly
            # figure stayed pre-fly. The sub-FA round trip "worked"
            # only because _show_stage calls _redraw_subject; do it
            # here directly when the assembly is what's displayed.
            if getattr(self, "_assembly_stage", None) == WHOLE_ASSEMBLY:
                self._redraw_subject()
            # Feed the Impact Analysis tab. Arrivals carry their
            # interpolated crossing state, so the cross-section is the
            # detector face as the flight measured it.
            _st = sorted({p.get("station") for p in self._impact_hits
                          if p.get("station")})
            if _st:
                self.w_impact_station.options = _st
                self.w_impact_station.value = _st[0]
                for _w in (self.w_impact_station, self.w_impact_plane,
                           self.w_impact_hist_axis):
                    _w.disabled = False
            self._draw_impact()
            dt = _time.time() - t0
            # FEED THE STATS TAB. The card already computes FWHM and R
            # from per-ion `tof` grouped by m/z, so an assembly flight
            # only has to present its arrivals in the same record shape
            # a normal run produces. Reporting the resolution solely in
            # this status line put the assembly's headline number
            # somewhere the app's own statistics panel could not see it,
            # which made the two disagree by silence.
            from ion_gym.physics.stats import auto_transmitted_fate
            _final = regions[-1]
            _fate_ok = auto_transmitted_fate(
                self._assembly_specs.get(_final.name, self.spec))
            _lost_fate = 2          # held/timeout: did not arrive
            results = [
                {"kind": (_fate_ok if p["arrived"] else _lost_fate),
                 "tof": (p.get("tof_us") if p["arrived"] else None),
                 "mz": p.get("mz")}
                for p in out["per_ion"]]
            self._assembly_results = results
            _flown_mz = sorted({float(p["mz"]) for p in out["per_ion"]
                                if p.get("mz") is not None})
            if _flown_mz and getattr(self, "w_amz", None) is not None:
                self.w_amz.options = [f"{m:g}" for m in _flown_mz]
            self.stats.update(results,
                              self._assembly_specs.get(_final.name,
                                                       self.spec))
            self.stats_impact.update(results,
                                     self._assembly_specs.get(
                                         _final.name, self.spec))
            msg = (f"**assembly flown** ({dt:.1f} s) — "
                   f"{out['n_arrived']}/{out['n']} arrived")
            # BEAM PROVENANCE. An assembly flies the
            # INSTRUMENT'S inline packet, never the Ion tab's source:
            # the Ion tab edits the displayed STAGE's source, and no
            # stage source is consulted here. Saying which packet flew
            # is the difference between a number and a result -- a
            # user who has just retuned births and sees only
            # "200/200, R = 90,769" has no way to tell their edits
            # were not used.
            # Provenance comes from the beam itself, not from an
            # assumption frozen in a string: a from_stage beam is
            # GENERATED from the named stage's source, and a
            # narrowed count is named as the Ion tab's doing --
            # the previous text said "NOT the Ion tab" in exactly
            # the case where the Ion tab set the count.
            _prov = (self._assembly_doc.get("beam") or {}) \
                if getattr(self, "_assembly_doc", None) else {}
            if _prov.get("from_stage"):
                _src_txt = (f"generated from stage "
                            f"'{_prov['from_stage']}' source")
            else:
                _src_txt = "the instrument's inline packet"
            _n_all = len(_prov.get("ions") or []) or None
            if _n_all is None and _prov.get("from_stage"):
                _sp2 = self._assembly_specs.get(_prov["from_stage"])
                _n_all = int(getattr(getattr(_sp2, "source", None),
                                     "n_ions", 0) or 0) or None
            if _n_all and out['n'] < _n_all:
                msg += (f" · beam: {out['n']} of {_n_all} declared "
                        f"({_src_txt}; narrowed by *ions per m/z*)")
            else:
                msg += f" · beam: {_src_txt} ({out['n']} ions)"
            if out["n_lost"]:
                # Losses are named, never folded into the headline.
                msg += (f", **{out['n_lost']} lost**: "
                        + "; ".join(f"{v}x {k}"
                                    for k, v in out["losses"].items()))
            ks = sorted({p.get("k") for p in out["per_ion"]
                         if p["arrived"] and p.get("k") is not None})
            if ks:
                msg += f" · fold order {ks}"
            import math as _math
            if _math.isfinite(out.get("R", float("nan"))):
                msg += (f" · T {out['T_us']:.4f} us · FWHM "
                        f"{out['fwhm_ns']:.4f} ns · **R "
                        f"{out['R']:,.0f}**")
            else:
                msg += f" · {out.get('note', 'R undefined')}"
        else:
            res = st["res"]
            dt = _time.time() - t0
            det = res.get("detection")
            # BANK THE TRACE (without it there are no trajectories
            # showing where ions are dying). The flight result carries the
            # full world-frame path — including the PARTIAL path of a
            # terminated ion, which is precisely the diagnostic — but
            # this branch only ever built a status string, so a single-
            # beam assembly flight drew nothing. Same record shape and
            # same detected-time cut as the packet path, one ion.
            import numpy as _np
            _rg_out = []
            for rg in res.get("regions", []):
                _t = _np.asarray(rg["t_us"], float)
                _cut = rg.get("detected_t_us")
                _m = (_t <= float(_cut)) if _cut is not None \
                    else slice(None)
                _rg_out.append(dict(
                    name=rg["name"],
                    x=_np.asarray(rg["x"], float)[_m],
                    y=_np.asarray(rg["y"], float)[_m],
                    z=_np.asarray(rg["z"], float)[_m], t_us=_t[_m]))
            self._assembly_traces = [dict(i=0, fate=res.get("fate"),
                                          regions=_rg_out, n_flown=1)]
            if getattr(self, "_assembly_stage", None) == WHOLE_ASSEMBLY:
                self._redraw_subject()
            msg = (f"**assembly flown** ({dt:.1f} s) — single ion, "
                   f"fate {res.get('fate')}")
            if det:
                msg += (f" · detected at {det['t_us']:.4f} us, "
                        f"k = {det['k']}")
            msg += ("  ·  a single ion has no time spread, so this "
                    "reports arrival only — R needs a packet beam.")
        if seams:
            msg += "  \n" + "  \n".join(seams)
        self.status.object = msg

    def _instrument_bytes(self):
        """Serialize the LOADED assembly with every stage's CURRENT spec
        (the live _assembly_specs objects, so voltage/source/gas edits
        ride along) back into ONE inline instrument.json. Refuses with a
        visible status when no assembly is loaded — the button is
        disabled then, but a callback must not silently produce an empty
        file if reached anyway."""
        import io as _io, json as _json
        doc = getattr(self, "_assembly_doc", None)
        specs = getattr(self, "_assembly_specs", None) or {}
        if not doc or not specs:
            self.status.object = ("**no instrument loaded** — Download "
                                  "instrument needs a loaded assembly.")
            return _io.BytesIO(b"")
        out = _json.loads(_json.dumps(doc))      # deep copy of wrapper
        for st in out.get("stages", []):
            sp = specs.get(st.get("name"))
            if sp is None:
                raise ValueError(
                    f"stage {st.get('name')!r} is in the document but "
                    f"not in the live stage set — the save would drop "
                    f"its edits silently. Reload the instrument.")
            st["spec"] = _json.loads(sp.to_json())
        return _io.BytesIO(
            _json.dumps(out, indent=1).encode("utf-8"))

    def _editor_spec(self):
        """The spec CURRENTLY IN THE TEXT BOX -- which is not necessarily
        self.spec.

        self.spec is the spec that is LOADED (and possibly solved). The text
        box is what the user is looking at. Conflating the two is why the
        estimator reported the old geometry's cost after you pasted a new
        JSON, and why Apply-resolution wrote the OLD spec back over your
        pasted text. The editor is the authority for anything the editor
        shows; self.spec only changes when something is applied.
        """
        return load_any_spec(self.w_json.value)

    def _on_ping(self, _=None):
        """Heartbeat -> BOTH the server console (stdout) and the visible
        status line, so a user can confirm the kernel is alive without
        access to the console (ping had stopped doing anything —
        it only printed to stdout, invisible in the browser). Prints enough
        state to localize a wedge (plane, model, RSS, figure/GC counts).
        Deliberately boring: no side effects on the run."""
        from ion_gym.ui import telemetry
        snap = telemetry.snapshot(
            spec=repr(self.spec.name), plane=self.w_plane.value,
            model=("yes" if self._model is not None else "no"))
        line = telemetry.format_line(snap)
        print(line, flush=True)
        import time as _t
        self.status.object = (f"**ping {_t.strftime('%H:%M:%S')}** — kernel "
                              f"alive. `{line}`")

    def _on_autolog(self, _=None):
        """Toggle the daemon-thread memory heartbeat (bug A observability).
        It logs RSS + leak indicators every 30 s on its OWN thread, so it
        keeps recording the growth curve even if the Panel event loop
        wedges — which is how you capture the leak leading INTO the drop."""
        from ion_gym.ui import telemetry
        mon = getattr(self, "_mem_monitor", None)
        if mon is None:
            mon = telemetry.MemoryHeartbeat(
                interval_s=30.0,
                context_fn=lambda: {
                    "spec": repr(self.spec.name),
                    "plane": self.w_plane.value,
                    "model": ("yes" if self._model is not None else "no")})
            self._mem_monitor = mon
        if mon.running:
            mon.stop()
            self.status.object = "**memory autolog OFF**"
        else:
            mon.start()
            self.status.object = ("**memory autolog ON** — RSS logged to "
                                  "stdout every 30 s (survives a UI wedge)")

    def _record_volume(self, spec):
        """Record-volume estimate for `spec`, through the ONE cost model
        (physics.sizing). Uses the BUILT record width when this session
        has one, so the quote is the width the kernel actually writes
        rather than a derivation from the declared channels."""
        from ion_gym.physics.sizing import record_volume
        n_cols = len(self._cols) if getattr(self, "_cols", None) else None
        # 0 (or the widget absent, e.g. before the Status tab is built)
        # means "no declared budget" -> the machine fraction decides.
        _w = getattr(self, "w_ram_budget", None)
        budget = float(_w.value) if (_w is not None and _w.value) else None
        return record_volume(spec, n_cols=n_cols, budget_gb=budget)

    def _record_quote_md(self, spec):
        """The record quote as markdown, loud in proportion to the tier."""
        try:
            v = self._record_volume(spec)
        except (ValueError, TypeError, AttributeError) as e:
            # NARROW, and these are the whole set: ValueError for a
            # non-positive dt or declared budget, TypeError/AttributeError
            # for a spec missing integration/source fields. A blanket
            # catch would report an unrelated bug in record_volume as
            # "estimate unavailable" and hide it.
            return f"**record estimate unavailable:** {type(e).__name__}: {e}"
        body = "\n\n".join(v["lines"])
        if v["tier"] == "ok":
            return body
        head = ("### \u26a0\ufe0f LARGE TRAJECTORY RECORD"
                if v["tier"] == "warn" else
                "### \u26d4 TRAJECTORY RECORD TOO LARGE FOR THIS MACHINE")
        levers = "\n".join(f"- {x}" for x in v["levers"])
        return f"{head}\n\n{body}\n\n**Cheapest fixes:**\n{levers}"

    def _record_guard(self, spec) -> bool:
        """True if the fly may proceed. Refuses LOUDLY and by name when
        the record would not fit, and warns on the status line when it
        is merely large -- the numbers are the same ones the sizing
        readout shows, from the same call."""
        try:
            v = self._record_volume(spec)
        except (ValueError, TypeError, AttributeError) as e:
            # Same narrow set as the readout above. An estimate that
            # cannot be computed is reported, never treated as
            # permission: the fly still proceeds, because refusing on a
            # broken estimator would be worse.
            self.status.object = (f"**record estimate failed** "
                                  f"({type(e).__name__}: {e}) — flying "
                                  f"without a record quote.")
            return True
        # The chip is the ALWAYS-VISIBLE surface (top button row, every
        # tab); the status line carries the full diagnosis. Both are
        # written from here, from the same numbers, so they cannot
        # disagree -- and a user who never opens the Status tab still
        # learns that the press did something and why.
        if v["tier"] == "refuse":
            _ceiling = (f"your declared budget of {v['budget_gb']:.0f} GB"
                        if v.get("budget_gb")
                        else (f"the limit for a {v['ram_gb']:.0f} GB machine"
                              if v["ram_gb"] else "the fallback limit"))
            self.status.object = (
                f"**REFUSED — the trajectory record would be "
                f"{v['gb']:.1f} GB**, over {_ceiling} "
                f"({v['refuse_gb']:.1f} GB). "
                f"{v['n_ions']:,} ion(s) x {v['samples_per_ion']:,} samples "
                f"x {v['n_cols']} channels. Nothing was flown. "
                + " ".join(v["levers"][:3])
                + " rec_every is STORAGE ONLY: tof, fate, impact position "
                  "and collision count are identical at any rec_every.")
            print("[record guard] REFUSED: " +
                  "; ".join(x.replace("**", "") for x in v["lines"]),
                  flush=True)
            self._record_note = None
            _ceil_short = (f"your {v['budget_gb']:.0f} GB budget"
                           if v.get("budget_gb")
                           else f"the {v['refuse_gb']:.0f} GB limit")
            self._fly_chip(
                "refused",
                note=(f"trajectory record would be {v['gb']:.1f} GB, over "
                      f"{_ceil_short}. Nothing was flown \u2014 raise "
                      f"rec_every, or the budget on the Status tab."))
            return False
        if v["tier"] == "warn":
            self.status.object = (
                f"**\u26a0\ufe0f large record: {v['gb']:.1f} GB** "
                f"({v['n_ions']:,} ions x {v['samples_per_ion']:,} samples "
                f"x {v['n_cols']} channels), over half of "
                + (f"your {v['budget_gb']:.0f} GB budget"
                   if v.get("budget_gb") else "what this machine should hold")
                + " — flying anyway. Raise rec_every to shrink it; it is a "
                  "STORAGE setting, not a physics one.")
            print("[record guard] WARNING: " +
                  "; ".join(x.replace("**", "") for x in v["lines"]),
                  flush=True)
            # rides on the live chip for the duration of this flight
            self._record_note = (f"\u26a0 large record {v['gb']:.1f} GB")
        else:
            self._record_note = None
        return True

    def start_telemetry(self) -> None:
        """Start the RSS heartbeat and the loop-stall watchdog.

        Called by the server (ui.serve.serve_dashboard), because both are
        server-scoped: a heartbeat with no session logs an idle process,
        and a stall watchdog with no loop to stall reports "NO PULSE YET"
        forever. Idempotent -- both start() calls are.
        """
        self._heartbeat.start()
        self._watchdog.start()

    def close(self) -> None:
        """Release what this app owns: the two telemetry threads.

        The threads are daemons, so this is not needed for process exit.
        It is needed by anything that builds MANY apps -- gates, tests,
        notebooks -- where without it the threads accumulate one pair per
        app for the life of the process. Idempotent, and safe to call on
        an app whose telemetry was never started.
        """
        self._heartbeat.stop()
        self._watchdog.stop()

    def _refresh_sizing(self, _=None):
        """Report what a solve at the chosen pitch will cost, FOR THE JSON IN
        THE WINDOW. Reports only -- never solves, never mutates a spec."""
        try:
            from ion_gym.physics.sim_build import sizing_for
            sp = self._editor_spec()
            mf = float(self.w_minfeat.value or 0.0) or None
            # Route-dispatched (sim_build.sizing_for): the estimate is for
            # the solve build_run will ACTUALLY run, not for the spec's own
            # domain -- the 3-D SLIM's confinement-plane spec estimated as a
            # ~1 s 2-D solve while the routed solve is ~6 min of 3-D
            # multigrid.
            p = sizing_for(sp, pitch=float(self.w_pitch.value),
                           min_feature_mm=mf,
                           field_method=self.w_fieldmethod.value,
                           channel_dtype=self.w_chandtype.value)
            hdr = ("" if sp.name == self.spec.name
                   else f"*(estimating **{sp.name}** — the JSON in the "
                        f"window, not the loaded spec)*\n\n")
            scope = (f"\n\n*Scope: the field solve only. The first fly of a "
                     f"session adds a one-time numba JIT compile "
                     f"(~{NUMBA_JIT_FIRST_FLY_S:.0f} s measured); the fly "
                     f"itself is extra.*")
            # The RECORD is the other half of the cost and used to be
            # quoted nowhere: t_max, dt, rec_every and n_ions are all
            # known here, and their product is what the machine has to
            # hold. Quoting the solve but not the record is how a run
            # that needs 38 GB gets started by accident.
            self.w_sizing.object = (hdr + p.markdown() + "\n\n"
                                    + self._record_quote_md(sp) + scope)
        except Exception as e:
            self.w_sizing.object = f"**sizing error:** {e}"

    def _on_json_edited(self, _=None):
        """The text changed: re-sync the pitch widget to the JSON's own
        mm_per_gu and re-estimate. Never touches self.spec."""
        if getattr(self, "_json_guard", False):
            return
        try:
            sp = self._editor_spec()
            self._json_guard = True
            try:
                self.w_pitch.value = float(sp.geometry.mm_per_gu)
            finally:
                self._json_guard = False
            # summary reflects the JSON in the BOX (staged), so a changed
            # or pasted JSON is legible before Apply.
            try:
                from ion_gym.io.spec_io import spec_summary_rows
                rows = spec_summary_rows(sp)
                self.w_spec_summary.object = (
                    "| field | value |\n|---|---|\n"
                    + "\n".join(f"| {k} | {v} |" for k, v in rows))
            except (ValueError, KeyError, TypeError, AttributeError):
                # AUDITED: narrowed like the enclosing handler
                # — fires per keystroke on half-typed JSON; quiet by
                # design, structural errors raise.
                pass
        except (ValueError, KeyError, TypeError, AttributeError):
            # NARROW, and legitimately quiet: this fires on EVERY keystroke in
            # the JSON editor, and mid-edit text is invalid JSON almost by
            # definition (json.JSONDecodeError is a ValueError).  The
            # authoritative parse with a REPORTED error is _on_apply_json;
            # this handler only keeps the pitch widget in sync when the text
            # happens to be parseable.
            self.w_spec_summary.object = "*(JSON not parseable yet)*"
        self._refresh_sizing()

    def _on_apply_pitch(self, _=None):
        """Set the pitch ON THE JSON IN THE WINDOW, apply it, rebuild.

        Previously this took self.spec (the already-loaded geometry), set its
        pitch, and wrote it back over the text box -- so pasting a new JSON
        and hitting Apply silently restored the old one.
        """
        try:
            h = float(self.w_pitch.value)
            if h <= 0:
                raise ValueError(f"resolution must be > 0, got {h}")
            sp = self._editor_spec()          # the JSON in the window
            # SET THE PITCH THROUGH set_pitch, not by assignment.
            # Under the lattice rule the extents are counted in
            # cells, so `sp.geometry.mm_per_gu = h` alone left the deck
            # REFUSABLE: measured on the certified oa_12plate deck, four
            # of five ordinary pitches were rejected by the loader and
            # surfaced here as "resolution error", as though the user had
            # mistyped. set_pitch re-derives the domain at the new pitch.
            # ELECTRODE GEOMETRY IS NEVER TOUCHED — metal stays in mm and
            # rasterizes; only the vacuum walls move, by under a cell.
            changes = set_pitch(sp, h)
            self._json_guard = True
            try:
                self.w_json.value = sp.to_json()
            finally:
                self._json_guard = False
            self.spec = sp
            self._rebuild_for_new_spec()
            self._refresh_sizing()
            # The adjustment is REPORTED, always, with no threshold — a
            # domain that moved silently is the defect class this guards
            # against, and "only mention it if it exceeds X" is how the
            # second instance hides behind the first.
            moved = [c.describe() for c in changes if c.moved()]
            adj = ("" if not moved else
                   "\n\ndomain covered to the new lattice (metal unchanged): "
                   + "; ".join(moved))
            self.status.object = (
                f"**{sp.name}: resolution {h:g} mm/gu** — bases re-solve on "
                f"the next run (a voltage-only change would not).{adj}")
        except Exception as e:
            self.status.object = f"**resolution error:** {e}"

    def _on_upload(self, event):
        """Load a spec .json -- plus, for an STL deck, the .stl files it
        references, selected in the same shot. See the widget comment:
        the upload carries bytes, not the folder they sit in, so the
        meshes travel WITH the spec or the deck cannot resolve them."""
        if not event.new:
            return
        self._clear_raster()
        try:
            # multiple=True yields lists; a lone value (programmatic
            # caller, older widget) is the same payload with one entry,
            # normalized here so nothing downstream special-cases it.
            blobs = event.new if isinstance(event.new, list) else [event.new]
            names = self.w_upload.filename
            if not isinstance(names, list):
                names = [names] if names else []
            if len(names) != len(blobs):
                raise ValueError(
                    f"upload returned {len(blobs)} file(s) but "
                    f"{len(names)} filename(s) -- cannot pair them, so "
                    f"nothing was loaded")
            pairs = list(zip(names, blobs))
            jsons = {n: b for n, b in pairs if str(n).lower().endswith(".json")}
            stls = {n: b for n, b in pairs if str(n).lower().endswith(".stl")}
            other = [str(n) for n, _ in pairs
                     if not str(n).lower().endswith((".json", ".stl"))]
            if len(jsons) != 1:
                raise ValueError(
                    f"select exactly ONE spec .json (plus its .stl files "
                    f"for an STL deck); got {len(jsons)}: "
                    f"{sorted(jsons) if jsons else '(none)'}")
            import json as _json
            from ion_gym.io.paths import repo_root
            from ion_gym.io.stl_resolve import install_stl_payload
            text = next(iter(jsons.values())).decode()
            doc = _json.loads(text)
            # Install BEFORE apply: _on_apply_json builds and draws, and a
            # draw against an unresolvable stl_dir is exactly the refusal
            # this door exists to prevent.
            where, notes = install_stl_payload(
                doc, stls, repo_root() / "uploads" / "decks")
            if where is not None:
                text = _json.dumps(doc, indent=1)
            self.w_json.value = text
            self._on_apply_json()
            if other:
                notes = list(notes) + [
                    f"**ignored {len(other)} non-spec file(s):** "
                    f"{', '.join(other)}"]
            done = str(self.status.object)
            if notes:
                done = done + "\n\n" + "\n\n".join(notes)
                self.status.object = done
            # the pane BESIDE the widget carries the outcome; the status
            # pane alone is invisible from the Load tab
            self.w_load_msg.object = done
        except Exception as e:
            msg = self._err_status("upload", e)
            self.status.object = msg
            self.w_load_msg.object = msg

    def _on_load_drive_template(self, event):
        """Pull ONLY the rf/dc drive groups from an uploaded json into the
        live session (load_drive_template): wipes the current groups and
        clears every electrode assignment, leaving geometry/physics/
        integration/source untouched. Reports exactly what changed so the
        destructive swap is never silent."""
        if not event.new:
            return
        try:
            from ion_gym.io.sim_spec import SimSpec, load_drive_template
            source = SimSpec.from_json(event.new.decode())
        except Exception as e:
            self.w_tmpl_status.object = (
                f"**template load failed** — could not parse json: {e}")
            return
        try:
            self._sync_spec()
            rep = load_drive_template(self.spec, source)
            self.w_json.value = self.spec.to_json()
            self._suspend_live = True
            try:
                self._rebuild_for_new_spec(solve=False)
            finally:
                self._suspend_live = False
            self._refresh_spec_summary()
            rf = ", ".join(rep["rf_loaded"]) or "(none)"
            dc = ", ".join(rep["dc_loaded"]) or "(none)"
            self.w_tmpl_status.object = (
                f"**template loaded** — {len(rep['rf_loaded'])} rf + "
                f"{len(rep['dc_loaded'])} dc groups replace the previous "
                f"{len(rep['rf_removed'])} rf + {len(rep['dc_removed'])} "
                f"dc.\n\n- rf: {rf}\n- dc: {dc}\n\n"
                f"**{rep['electrodes_cleared']} of "
                f"{rep['electrodes_total']} electrodes had assignments "
                f"cleared** — re-assign them below.")
            self.status.object = (
                "**drive template applied** — assign electrodes to the new "
                "groups in the Voltages tab.")
        except Exception as e:
            self.w_tmpl_status.object = self._err_status(
                "load drive template", e)

    def _spec_bytes(self):
        self._sync_spec()
        # SAVED copy is normalized for portability: a cache-absolute
        # stl_dir (from the upload door's mesh install) becomes "." in
        # the file. The LIVE spec keeps its absolute dir -- the session
        # must keep resolving; only the download is rewritten, and the
        # rewrite is stated on the status line, never silent.
        import json as _json
        from ion_gym.io.paths import repo_root
        from ion_gym.io.stl_resolve import portable_stl_dir
        doc = _json.loads(self.spec.to_json())
        note = portable_stl_dir(doc, repo_root() / "uploads" / "decks")
        if note:
            self.status.object = f"**saved spec:** {note}"
        return io.BytesIO(_json.dumps(doc, indent=1).encode())

    def _on_reload_run(self, _=None):
        name = self.w_runsel.value
        if name in self._runs:
            self._active = name
            self._sync_analysis_mz()
            self._redraw(self._runs[name].results)
            self.status.object = f"**reloaded:** {name}"

    def _export_view_html(self):
        """Self-contained HTML of the current view (field/PE + trajectories)
        with the full run metadata embedded as a collapsible panel — a
        shareable, archivable snapshot of one simulation."""
        import io as _io
        import json
        import datetime

        def _fig_html(fig):
            # D-block: this was a CAPABILITY SNIFF BY EXCEPTION -- call
            # fig.to_html and, on any failure, assume the plotly version is old
            # and re-render through pio.  It would equally have caught a genuine
            # error INSIDE the render and silently produced a DIFFERENT figure
            # (no width/height) in the report.  pio.to_html is the stable API and
            # is present in every plotly that ion_gym supports; use it always.
            import plotly.io as pio
            return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                               default_width="900px", default_height="600px")

        fig = self.pane.object
        if fig is None:
            return _io.BytesIO(b"<html><body>No view to export - "
                               b"Fly some ions first.</body></html>")
        # main view (field/trajectories) + the PE surface if one has been
        # generated on the PE tab — both in one self-contained file.
        sections = [("Field / trajectories view", _fig_html(fig))]
        pe_fig = getattr(getattr(self, "_pe_tab", None), "_fig", None)
        if pe_fig is not None and pe_fig is not fig:
            sections.append(("PE surface", _fig_html(pe_fig)))
        plot_html = "".join(
            f'<h3>{t}</h3><div>{h}</div>' for t, h in sections)
        # metadata: full spec + run summary
        meta = {"exported": datetime.datetime.now().isoformat(timespec="seconds"),
                "name": self.spec.name,
                "view_plane": self.w_plane.value,
                "field_shading": (self.w_fieldmode.value
                                  if self.w_showfield.value else "off"),
                "pe_mz": self._pe_tab.w_mz.value,
                "spec": self.spec.to_dict()}
        if self._active in self._runs:
            res = self._runs[self._active].results
            fates = {}
            for r in res:
                k = r.summary.get("kind", -1)
                fates[k] = fates.get(k, 0) + 1
            fate_name = {0: "hit electrode", 1: "left domain",
                         2: "timed out", 3: "bounding plane"}
            meta["run"] = {
                "label": self._active, "n_ions": len(res),
                "fates": {fate_name.get(k, str(k)): v
                          for k, v in fates.items()}}
        meta_json = json.dumps(meta, indent=2)
        title = self.spec.name
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title} — ion_gym run</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
h2{{font-weight:600}} h3{{font-weight:600;margin-top:22px}}
details{{margin-top:18px}}
pre{{background:#f5f5f7;padding:14px;border-radius:8px;overflow:auto;
font-size:12px;line-height:1.4}}
summary{{cursor:pointer;font-weight:600;color:#555}}</style></head>
<body><h2>{title}</h2>
<div>{plot_html}</div>
<details open><summary>Run metadata (geometry, voltages, gas, fates)</summary>
<pre>{meta_json}</pre></details>
<p style="color:#999;font-size:11px">Generated by ion_gym.</p>
</body></html>"""
        buf = _io.BytesIO(html.encode("utf-8"))
        buf.seek(0)
        return buf

    def _export_npz_bytes(self):
        """Compact per-ion NPZ: one array per ion (columns = channels) plus a
        summary table, far smaller than CSV for long trajectories.

        PRECISION: trajectories are cast to float32 here (~7 significant
        digits), NOT the float64 the kernel records. This docstring used
        to claim it "preserves full float precision", which was wrong and
        is the kind of wrong that only shows up in someone's arrival-time
        difference. The trade is deliberate — this is the in-browser
        download, where size is the binding constraint — but it is a
        trade, so it is stated: for plotting and inspection float32 is
        ample; for sub-nanosecond timing differences or anything
        differentiated along a path, use **Save trajectories**, which
        writes the same paths at full float64 to disk. The summary table
        stays float64 in both.

        Per-ion m/z is stored (both a plain `mz_da` array and a `summary`
        column) so absolute KE / temperature is recoverable downstream —
        the Save-trajectories path already did this; the export path did
        not, which left exported runs massless."""
        if self._active not in self._runs:
            return io.BytesIO(b"")
        from ion_gym.physics.sim_build import mz_of
        arrs = {}
        summ_rows = []
        mz_list = []
        for r in self._runs[self._active].results:
            if r.traj is not None:
                arrs[f"ion_{r.index:04d}"] = r.traj.astype(np.float32)
            # per-ion m/z: prefer the value the run recorded, else derive it
            # from the spec (same source as save_trajectories); NaN only if
            # genuinely unknown, so a reader can tell "massless" from "0".
            mz = r.summary.get("mz")
            if mz is None:
                try:
                    mz = float(mz_of(self.spec, int(r.index)))
                except (ValueError, TypeError, IndexError, AttributeError):
                    mz = np.nan
            mz_list.append(float(mz) if mz is not None else np.nan)
            summ_rows.append([r.index, r.summary.get("kind", -1),
                              r.summary.get("tof", np.nan),
                              r.summary.get("x_end", np.nan),
                              r.summary.get("y_end", np.nan),
                              r.summary.get("z_end", np.nan),
                              mz_list[-1]])
        buf = io.BytesIO()
        np.savez_compressed(
            buf, columns=np.array(self._cols),
            summary=np.array(summ_rows, float),
            summary_cols=np.array(["ion", "fate", "tof",
                                   "x_end", "y_end", "z_end", "mz_da"]),
            mz_da=np.array(mz_list, float),
            **arrs)
        buf.seek(0)
        return buf

    def _export_bytes(self):
        import pandas as pd
        if self._active not in self._runs:
            return io.BytesIO(b"")
        frames = []
        for r in self._runs[self._active].results:
            if r.traj is None:
                continue
            df = pd.DataFrame(r.traj, columns=self._cols)
            df.insert(0, "ion", r.index)
            df["fate"] = r.summary["kind"]
            frames.append(df)
        out = (pd.concat(frames, ignore_index=True) if frames
               else pd.DataFrame())
        return io.BytesIO(out.to_csv(index=False).encode())

    def results_dataframe(self, run_name=None):
        """The displayed (or named) run's channels as a DataFrame."""
        import pandas as pd
        name = run_name or self._active
        if name not in self._runs:
            return pd.DataFrame()
        frames = []
        for r in self._runs[name].results:
            if r.traj is None:
                continue
            df = pd.DataFrame(r.traj, columns=self._cols)
            df.insert(0, "ion", r.index)
            df["fate"] = r.summary["kind"]
            frames.append(df)
        return (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame())

def _default_export_dir():
    """Default field-export destination: the user-facing outputs dir in
    managed sandboxes, else ./exported_fields under the cwd."""
    import os
    cand = "/mnt/user-data/outputs"
    if os.path.isdir(cand) and os.access(cand, os.W_OK):
        return cand
    return os.path.abspath("exported_fields")
