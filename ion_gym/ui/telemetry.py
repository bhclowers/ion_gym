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
import threading
import time


def rss_mb() -> float:
    """Resident set size in MB, cross-platform. psutil when present
    (unambiguous bytes); else resource.ru_maxrss with the per-platform
    unit applied (Linux KB, macOS/BSD bytes) instead of assuming one."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 ** 2)
    except Exception:
        import resource
        import sys
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss units differ by platform; this is the documented split.
        if sys.platform == "darwin":
            return ru / (1024.0 ** 2)      # bytes -> MB
        return ru / 1024.0                 # KB -> MB (Linux)


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


def snapshot(**context) -> dict:
    """One telemetry reading: RSS, live-figure count, total GC object
    count, plus any caller context (plane, model, spec name)."""
    snap = {"t": time.strftime("%H:%M:%S"),
            "rss_mb": rss_mb(),
            "figures": _live_figures(),
            "gc_objects": len(gc.get_objects())}
    snap.update(context)
    return snap


def format_line(snap: dict, delta_mb: float | None = None) -> str:
    d = f" Δ{delta_mb:+.0f}MB" if delta_mb is not None else ""
    ctx = " ".join(f"{k}={v}" for k, v in snap.items()
                   if k not in ("t", "rss_mb", "figures", "gc_objects"))
    return (f"[mem] {snap['t']} rss~{snap['rss_mb']:.0f}MB{d} "
            f"figs={snap['figures']} objs={snap['gc_objects']:,}"
            + (f" | {ctx}" if ctx else ""))


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
