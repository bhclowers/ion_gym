"""Process-memory telemetry for diagnosing the long-session leak/wedge.

The app wedges after ~10-15 min with RSS in the multi-GB range; the UI
goes partly unresponsive, so a heartbeat driven by the Panel event loop
can stall WITH it and capture nothing at the moment that matters. This
monitor runs on its OWN daemon thread, independent of the event loop, so
it keeps logging the RSS trajectory right up to (and through) a wedge —
answering "is there another way to capture when it does?".

Boring by design: read RSS, count a couple of leak indicators, print one
line, sleep. No app state is mutated. `rss_mb()` is also the single
correct-units RSS reader for the manual ping (resource.ru_maxrss is KB on
Linux but BYTES on macOS — the old ping mislabeled it; psutil removes the
ambiguity)."""
from __future__ import annotations

import gc
import os
import sys
import threading
import time


# Set to a type name to have every heartbeat report WHAT REFERS TO it.
# A list so it can be retargeted from a running process without reload:
#   from ion_gym.ui import telemetry; telemetry.REFERRER_TARGET[0] = "X"
# Costs a gc.get_referrers walk per reading, so it is OFF by default.
REFERRER_TARGET = [None]

# Console volume. MINIMAL is the default because the heartbeat prints on
# a 30 s timer for the life of the session and the diagnostic form --
# holders plus a referrer block -- runs to six lines a reading, which
# buries the [compose]/[reuse]/[seed] lines the app needs the console
# for. VERBOSE is opt-in, for a hunt.
#   from ion_gym.ui import telemetry
#   telemetry.VERBOSE[0] = True
#   telemetry.REFERRER_TARGET[0] = "Stl3DModel"   # or "dict:res"
VERBOSE = [False]


def _macos_phys_footprint_mb():
    """macOS phys_footprint in MB, or None if unavailable.

    THE NUMBER ACTIVITY MONITOR SHOWS, and the one that predicts a
    freeze. psutil's rss maps to Mach `resident_size`, which EXCLUDES
    COMPRESSED PAGES: macOS compresses inactive pages instead of
    swapping, so a page you allocated and stopped touching leaves
    resident_size while still being held against physical memory.
    Measured 2026-09-14: the heartbeat read 463 MB while Activity
    Monitor showed the same process at 20 GB, and killing it returned
    20 GB. Three conclusions in this investigation were drawn from
    resident_size and two of them were wrong in opposite directions.

    Read via libproc proc_pid_rusage(RUSAGE_INFO_V0), whose struct is
    a 16-byte uuid followed by ten uint64s; ri_phys_footprint is the
    eighth of those. Returns None on ANY failure rather than a
    substitute number — the caller labels which metric it got, so a
    fallback is never mistaken for the real thing.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        class _RUsageV0(ctypes.Structure):
            _fields_ = [("ri_uuid", ctypes.c_uint8 * 16),
                        ("ri_fields", ctypes.c_uint64 * 10)]

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        buf = _RUsageV0()
        rc = libc.proc_pid_rusage(ctypes.c_int(os.getpid()),
                                  ctypes.c_int(0),
                                  ctypes.byref(buf))
        if rc != 0:
            return None
        return buf.ri_fields[7] / (1024.0 ** 2)
    except Exception:
        return None


def mem_reading() -> tuple:
    """(value_mb, label). Footprint where it is the right quantity.

    macOS  -> phys_footprint, labelled 'footprint'
    else   -> psutil rss, labelled 'rss'
    neither-> resource.ru_maxrss, labelled 'PEAK' because ru_maxrss is a
              HIGH-WATER MARK that cannot fall; reporting it as current
              is how a wrong number gets acted on.
    """
    fp = _macos_phys_footprint_mb()
    if fp is not None:
        return fp, "footprint"
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 ** 2), "rss"
    except Exception:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ru = ru / (1024.0 ** 2) if sys.platform == "darwin" else ru / 1024.0
        return ru, "PEAK"


def rss_mb() -> float:
    """Memory in MB, scalar. KEPT SCALAR DELIBERATELY: sim_app's
    _mem_cache_lines() calls this and formats it with :.0f, so returning
    the (value, label) tuple broke the app at import of the Cache tab.
    Callers that need the label use mem_reading(); this stays the shape
    its existing callers expect."""
    return mem_reading()[0]


def array_holders(top=8):
    """[(holder_type, mb, n_arrays)] — WHO is holding array memory.

    THE WALK THAT REACHES WHAT THE OTHERS CANNOT. A numeric ndarray is
    not GC-tracked, and CPython UNTRACKS a dict whose values are all
    untracked, so neither a global array scan nor a container walk finds
    arrays held in a cache dict or an instance __dict__ (both measured,
    both returned 0 while a gigabyte was held). But an INSTANCE is
    tracked, and its __dict__ is reachable THROUGH it -- so walking
    objects and charging their ndarray-valued attributes reaches exactly
    the category that has stayed invisible all session.

    Attributed to the HOLDER'S TYPE, so the answer is a name to go fix,
    not another total. Dedup by array id; views skipped via .base so a
    slice is never charged against the buffer it borrows.
    """
    try:
        import numpy as np
    except ImportError:
        return []
    by_type, seen = {}, set()

    def charge(holder, v):
        if type(v) is not np.ndarray or v.base is not None:
            return
        if id(v) in seen:
            return
        seen.add(id(v))
        mb, n = by_type.get(holder, (0.0, 0))
        by_type[holder] = (mb + v.nbytes / (1024.0 ** 2), n + 1)

    for o in gc.get_objects():
        try:
            d = getattr(o, "__dict__", None)
            if isinstance(d, dict) and d:
                nm = type(o).__name__
                for v in list(d.values()):
                    charge(nm, v)
                    if isinstance(v, dict):
                        for vv in list(v.values()):
                            charge(nm + ".dict", vv)
                    elif isinstance(v, (list, tuple)):
                        for vv in list(v):
                            charge(nm + ".seq", vv)
            elif isinstance(o, dict):
                for v in list(o.values()):
                    charge("<dict>", v)
            elif isinstance(o, (list, tuple)):
                for v in list(o):
                    charge("<seq>", v)
        except Exception:
            continue
    rows = sorted(((mb, n, k) for k, (mb, n) in by_type.items()),
                  reverse=True)
    return [(k, mb, n) for mb, n, k in rows[:top]]


def _live_figures() -> int:
    """Count live Plotly Figure objects — the prime suspect for a per-
    render leak (a replot that leaves the previous figure referenced).
    Zero when plotly isn't imported; never raises."""
    try:
        import plotly.graph_objects as go
        return sum(1 for o in gc.get_objects()
                   if isinstance(o, go.Figure))
    except Exception:
        return -1


def referrers_of(type_name, max_objs=3, depth=2):
    """Who still points at the leaked instances of `type_name`.

    THE LAST LINK. array_holders() names the OBJECT holding the memory
    (measured: 20 live Stl3DModel instances, 16.0 GB). It cannot say why
    they are still alive. gc.get_referrers() on one of them names what
    refers to it, which is the thing to go and fix.

    Referrers are described, never returned, because a returned referrer
    is itself a new reference that would keep the leak alive while you
    look at it. Frames are reported with their FUNCTION NAME: a referrer
    that is a frame means a live call is holding the object, and a
    referrer that is a cell means a CLOSURE captured it -- a Panel
    callback defined per flight with the model in scope keeps that model
    for as long as the widget lives, which is the leading hypothesis.
    A dict referrer is reported with the attribute name the object sits
    under, plus the type that owns that dict, so 'SimApp._model' or
    'dict in _reuse_cache' comes out directly rather than 'a dict'.

    Bounded: at most `max_objs` instances examined and `depth` levels
    followed, so a diagnostic cannot walk the whole heap.
    """
    out = []
    # "dict:KEY" targets PLAIN DICTS CARRYING THAT KEY, because the thing
    # we now need to trace is one: the per-build state dict st = {done,
    # res, err, ...} that pins a full model through st["res"]. Targeting
    # by type name cannot reach it -- every dict in the process is named
    # "dict", so the ordinary path would examine an arbitrary three of
    # hundreds of thousands. The key IS the identity here.
    if type_name.startswith("dict:"):
        want = type_name.split(":", 1)[1]
        targets = [o for o in gc.get_objects()
                   if isinstance(o, dict) and want in o][:max_objs]
    else:
        targets = [o for o in gc.get_objects()
                   if type(o).__name__ == type_name][:max_objs]
    if not targets:
        return [f"no live {type_name} found"]
    out.append(f"{type_name}: examining "
               f"{len(targets)} of the live instances")

    def describe(r, obj):
        tn = type(r).__name__
        if tn == "frame":
            return f"frame in {r.f_code.co_name}() — a live call holds it"
        if tn == "cell":
            return "cell — CAPTURED BY A CLOSURE"
        if isinstance(r, dict):
            key = next((k for k, v in list(r.items()) if v is obj), None)
            owners = [o for o in gc.get_referrers(r)
                      if getattr(o, "__dict__", None) is r]
            if owners:
                return f"{type(owners[0]).__name__}.{key} (attribute)"
            return f"dict[{key!r}]"
        if isinstance(r, (list, tuple, set)):
            return f"{tn} of len {len(r)}"
        return tn

    for i, obj in enumerate(targets, 1):
        seen = {id(obj)}
        level = [obj]
        for d in range(depth):
            nxt, lines = [], []
            for o in level:
                for r in gc.get_referrers(o):
                    if id(r) in seen or r is level or r is targets:
                        continue
                    seen.add(id(r))
                    lines.append(describe(r, o))
                    nxt.append(r)
            if not lines:
                break
            uniq = sorted(set(lines))
            out.append(f"  [{i}] depth {d + 1}: " + "; ".join(uniq[:6]))
            level = nxt[:12]
            del nxt
        del level, seen
    del targets
    return out


def snapshot(collect: bool = True, **context) -> dict:
    """One telemetry reading: RSS, live-figure count before AND after a
    cyclic collection, total GC object count, plus caller context.

    WHY THE COLLECT (2026-09-14). Measured on a live session: RSS climbed
    ~365 MB per flight while `figures` rose by exactly 4 and never fell,
    `gc_objects` stayed flat at ~630k, and dropping every stored run
    freed 30 MB out of 10.4 GB. Four is the number of Plotly panes in the
    app, so each flight leaves the previous four figures alive, each
    holding its copy of the plotted arrays.

    Plotly figures contain REFERENCE CYCLES (traces and layout hold
    back-references to the parent), so they are never freed by
    refcounting — only by the cyclic collector, which nothing in the app
    calls. That gives two possibilities with completely different fixes,
    and this reading separates them in ONE LINE:

      figs=22->6   collectable. They were garbage all along and the app
                   simply never collected. Fix is a collect on replot.
      figs=22->22  NOT collectable. Something still holds a live
                   reference (Panel's model registry, the Bokeh
                   document). Fix is a real teardown on replot.

    The collect costs a pause proportional to the heap — noticeable on a
    multi-GB process, which is why it is a parameter and not forced. It
    runs on the monitor's own daemon thread, never the Panel event loop.
    """
    _mem, _lbl = mem_reading()
    figs_pre = _live_figures()
    n_collected = gc.collect() if collect else None
    snap = {"t": time.strftime("%H:%M:%S"),
            "rss_mb": _mem, "rss_label": _lbl,
            "figures_pre": figs_pre,
            "figures": _live_figures() if collect else figs_pre,
            "collected": n_collected,
            "gc_objects": len(gc.get_objects()),
            "holders": array_holders() if VERBOSE[0] else None,
            "referrers": (referrers_of(REFERRER_TARGET[0])
                          if (VERBOSE[0] and REFERRER_TARGET[0])
                          else None)}
    snap.update(context)
    return snap


def format_line(snap: dict, delta_mb: float | None = None) -> str:
    d = f" Δ{delta_mb:+.0f}MB" if delta_mb is not None else ""
    ctx = " ".join(f"{k}={v}" for k, v in snap.items()
                   if k not in ("t", "rss_mb", "figures", "figures_pre",
                                "collected", "gc_objects",
                                "rss_label", "holders",
                                "referrers"))
    pre, post = snap.get("figures_pre"), snap["figures"]
    figs = (f"{post}" if pre is None or pre == post
            else f"{pre}->{post}")
    col = ("" if snap.get("collected") is None
           else f" gc={snap['collected']:,}")
    head = (f"[mem] {snap['t']} {snap.get('rss_label', 'rss')}"
            f"~{snap['rss_mb']:.0f}MB{d} "
            f"figs={figs}{col} objs={snap['gc_objects']:,}"
            + (f" | {ctx}" if ctx else ""))
    # WHO is holding it, printed under the line. The whole point of this
    # build: a total tells you there is a problem, a holder tells you
    # where to go.
    if not VERBOSE[0]:
        return head
    ref = snap.get("referrers") or []
    hold = snap.get("holders") or []
    if hold:
        head += "\n       holders: " + "; ".join(
            f"{k} {mb:.0f}MB x{n}" for k, mb, n in hold if mb >= 1.0)
    for line in ref:
        head += "\n       " + line
    return head


class MemoryHeartbeat:
    """Daemon-thread RSS logger. Survives a Panel event-loop wedge because
    it does not run on that loop. `context_fn` (optional) returns a dict of
    app state (plane, model, spec) to tag each line — called from the
    monitor thread, so it must only READ cheap attributes.

    Start/stop are idempotent. The thread is a daemon: it never blocks
    process exit, and there is nothing to clean up if the app is killed."""

    def __init__(self, interval_s: float = 30.0, context_fn=None):
        self.interval_s = float(interval_s)
        self._context_fn = context_fn
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_rss: float | None = None

    def _run(self):
        while not self._stop.wait(self.interval_s):
            ctx = {}
            if self._context_fn is not None:
                try:
                    ctx = self._context_fn() or {}
                except Exception as e:
                    # never let a context read kill the monitor; report it
                    ctx = {"context_error": type(e).__name__}
            snap = snapshot(**ctx)
            delta = (None if self._last_rss is None
                     else snap["rss_mb"] - self._last_rss)
            self._last_rss = snap["rss_mb"]
            print(format_line(snap, delta), flush=True)

    def start(self) -> bool:
        """Begin logging. Returns True if newly started, False if already
        running."""
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._last_rss = None
        self._thread = threading.Thread(
            target=self._run, name="mem-heartbeat", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> bool:
        """Stop logging. Returns True if it was running."""
        if self._thread is None or not self._thread.is_alive():
            return False
        self._stop.set()
        return True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class LoopStallWatchdog:
    """Dump every thread's stack when the event loop stalls.

    The GUI-drop failure mode (seen on a 3-D refly): some
    callback blocks the Tornado event loop long enough that the websocket
    keepalive is missed, the browser closes the connection, and the session
    is torn down — after which every click dies on Bokeh's
    `assert self._session is not None`. The console errors show the
    AFTERMATH; what diagnosis needs is what the loop was DOING while
    blocked, which nothing captures by default.

    Mechanism (same independence argument as MemoryHeartbeat): the app
    arms a periodic callback ON the event loop that calls `pulse()` every
    second; this watchdog runs on its OWN daemon thread and, when the last
    pulse goes stale by more than `stall_s`, uses faulthandler to write
    every thread's current stack to `dump_path` (default
    outputs/diagnostics/ion_gym_hang_dump.txt) — once per stall, with a
    banner naming the stall length. The dump's MainThread section is the
    exact line the loop was stuck on, which turns "it hangs sometimes"
    into a file to read.

    Boring by design: no app state mutated; arming it costs one timestamp
    write per second.
    """

    def __init__(self, stall_s: float = 15.0, dump_path=None):
        """dump_path=None (default) resolves to
        repo_root()/outputs/diagnostics/ion_gym_hang_dump.txt.

        It used to default to the BARE filename "ion_gym_hang_dump.txt",
        so a hang dump landed in whatever directory the app happened to be
        launched from -- littering the repo root, and worse, going
        somewhere unpredictable when launched from elsewhere.
        outputs/ is the declared home for generated files
        and diagnostics/ keeps crash residue out of the way of
        deliverables. An explicit path is still honoured.
        """
        from ion_gym.io.paths import outputs_dir
        self.stall_s = float(stall_s)
        self.dump_path = (str(outputs_dir("diagnostics",
                                          "ion_gym_hang_dump.txt"))
                          if dump_path is None else str(dump_path))
        self._last = time.time()
        self._dumped_this_stall = False
        self._pulse_tid = None   # thread id of the loop that beats pulse()
        self._thread = None
        self._stop = threading.Event()

    def pulse(self):
        """Called from the EVENT LOOP (periodic callback). A fresh pulse
        also re-arms the one-dump-per-stall latch, and records WHICH
        thread beat it — the loop self-identifies, so the stall verdict
        never has to guess the loop thread from stack contents."""
        self._last = time.time()
        self._dumped_this_stall = False
        self._pulse_tid = threading.get_ident()

    def _verdict(self):
        """Verdict on a stale pulse, read off the PULSE OWNER's own live
        stack. Exactly three causes are possible and each has a distinct
        signature on that one thread:

          1. The thread is GONE from sys._current_frames(): the server /
             loop thread terminated (server stopped, process winding down).
          2. The thread is PARKED — innermost frame is a `select` invoked
             by asyncio's `_run_once`: the loop is alive but running no
             callback, so the pulse source (the session's periodic
             callback) was torn down: tab closed, websocket dropped.
          3. Anything else: the loop is EXECUTING, and the stack printed
             here is, by construction, the exact code holding it.

        Environment-agnosticism is the design requirement (this will
        not always run on the machine that wrote it).
        The previous classifier tested `"ion_gym" in filename` across all
        threads, which false-positived on ANY interpreter installed in a
        conda env *named* ion_gym (every stdlib frame lives under
        .../envs/ion_gym/...), and false-verdicted BLOCKED on an idle
        server launched via `python -m ion_gym` (cli.py frames parked
        beneath select forever). This verdict contains no path test, no
        package-location assumption, and no launch-style dependence: the
        thread is identified by pulse() itself, and the idle test is on
        frame FUNCTION identity — `select` called from `_run_once` —
        which is how both the selector loop (Linux/macOS) and the
        proactor loop (Windows) park between callbacks."""
        import sys
        import traceback
        if self._pulse_tid is None:
            return (True,
                    "NO PULSE YET — the watchdog is armed but no server "
                    "session has beaten pulse(): normal between server "
                    "start and the first tab connecting. A stall verdict "
                    "is meaningless with nothing to stall.\n")
        frame = sys._current_frames().get(self._pulse_tid)
        if frame is None:
            return (True,
                    f"PULSE OWNER THREAD GONE (thread "
                    f"{self._pulse_tid:#x} not alive) — the loop thread "
                    "that was beating pulse() has terminated: the server "
                    "stopped or the process is winding down. The "
                    "timestamp above is when it died.\n")
        code = frame.f_code
        caller = frame.f_back
        parked = (code.co_name == "select"
                  and caller is not None
                  and caller.f_code.co_name == "_run_once")
        if parked:
            return (True,
                    "PULSE OWNER IDLE (loop parked in select under "
                    "_run_once) — the loop is alive but running no "
                    "callbacks, so the PULSE SOURCE DIED (the server "
                    "session owning the 1 s callback ended: tab closed, "
                    "websocket dropped), NOT a blocked loop. The "
                    "timestamp above is when the session died. A new "
                    "session's pulse re-arms the watchdog.\n")
        stack = "".join(traceback.format_stack(frame))
        return (False,
                "PULSE OWNER EXECUTING (a BLOCKED loop or a long "
                "compute) — the stack below is the code holding the "
                "loop right now; its innermost frames are the culprit:\n"
                + stack)

    def _run(self):
        import faulthandler
        while not self._stop.wait(1.0):
            stale = time.time() - self._last
            if stale > self.stall_s and not self._dumped_this_stall:
                self._dumped_this_stall = True
                benign, verdict = self._verdict()
                try:
                    with open(self.dump_path, "a") as fh:
                        fh.write(
                            f"\n===== PULSE STALE {stale:.0f}s "
                            f"(threshold {self.stall_s:.0f}s) at "
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
                            f"VERDICT: {verdict}")
                        if benign:
                            # A real dump carried three entries, all
                            # self-diagnosed benign (pre-first-session,
                            # then two tab-closes):
                            # a benign verdict gets ONE line — the
                            # verdict IS the diagnostic; the stacks
                            # contain a parked selector and nothing
                            # else, and a file full of them reads as
                            # hangs that never happened. Still written,
                            # never silent.
                            fh.write("(benign — no stack dump: nothing "
                                     "was blocked)\n")
                        else:
                            fh.write("Full all-thread stacks follow:\n")
                            fh.flush()
                            faulthandler.dump_traceback(file=fh,
                                                        all_threads=True)
                    if benign:
                        print(f"[watchdog] note: pulse quiet {stale:.0f}s "
                              f"— {verdict.splitlines()[0]}", flush=True)
                    else:
                        print(f"[watchdog] event loop stalled "
                              f"{stale:.0f}s — all-thread stack dump "
                              f"appended to {self.dump_path}", flush=True)
                except OSError as e:
                    # The one thing this thread must not do is die silently:
                    # a watchdog that stops watching without saying so is
                    # worse than none.
                    print(f"[watchdog] could not write {self.dump_path}: "
                          f"{type(e).__name__}: {e}", flush=True)

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="loop-stall-watchdog")
        self._thread.start()
        return True

    def stop(self) -> bool:
        if self._thread is None:
            return False
        self._stop.set()
        return True
