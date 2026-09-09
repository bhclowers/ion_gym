"""
ion_gym.ensemble_driver
-----------------------
A thin, read-only orchestration layer over the (njit) fly kernels. It
does NOT touch the kernels' physics — it calls a user-supplied fly_fn
once per ion, between flights (the only place Python regains control,
since an njit loop cannot be interrupted mid-flight), and there handles:
  * progress reporting (a lightweight EnsembleProgress per batch),
  * clean early termination (a viability predicate + a cooperative
    stop flag + KeyboardInterrupt), always returning PARTIAL results so
    an aborted run is diagnostic, not wasted,
  * decimation of stored trajectories for display (full arrays stay
    available to stats; the viewer gets a strided copy).

Design invariants (why it is shaped this way):
  - Kernels stay headless and gate-pure. If a view needs a quantity,
    the KERNEL records it; the driver never calls back into a kernel to
    synthesize data.
  - Granularity is the BATCH, not the ion: checking a Python predicate
    after every ion is wasteful, so check_every batches the interrupt
    poll (and, later, the UI push). Default 25.
  - Threading-ready without threading yet: run() is the synchronous
    path; run_threaded() launches the SAME generator in a worker and
    exposes a poll()-able handle. Both share _iter_batches, so there is
    one code path for the physics loop. Kernels intended for the
    threaded path should be built njit(nogil=True) so the worker
    releases the GIL during flight (a kernel property, not a driver
    concern) — the driver already avoids holding locks across a flight.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class IonResult:
    """One ion's outcome. traj is the DECIMATED (n_dec, k) record copy
    for display; summary carries whatever scalars the fly_fn exposed
    (kind, tof, n_collisions, impact coords, ...). full is the undecimated
    record IF keep_full is set (off by default to bound memory)."""
    index: int
    traj: Optional[np.ndarray]
    summary: dict
    full: Optional[np.ndarray] = None


@dataclass
class EnsembleProgress:
    """Emitted once per batch (and at completion/abort). Cheap to build,
    safe to hand to a UI thread (plain scalars + the results list ref)."""
    done: int
    total: int
    elapsed_s: float
    n_terminated: int              # kind == 0 (impact) count so far
    stopped_early: bool = False
    stop_reason: str = ""
    results: list = field(default_factory=list)
    # worker-pool telemetry: sequential runs report 1 / 0 so consumers read
    # one shape for both paths.
    n_workers: int = 1
    in_flight: int = 0

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 1.0

    @property
    def rate_hz(self) -> float:
        return self.done / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def eta_s(self) -> float:
        r = self.rate_hz
        return (self.total - self.done) / r if r > 0 else float("nan")


def _decimate(traj: Optional[np.ndarray], stride: int):
    """Strided view->copy of an (n, k) record. stride<=1 keeps all;
    always retains the LAST row (impact/exit state is diagnostic)."""
    if traj is None or stride <= 1 or traj.shape[0] <= 2:
        return None if traj is None else traj.copy()
    idx = np.arange(0, traj.shape[0], stride)
    if idx[-1] != traj.shape[0] - 1:
        idx = np.append(idx, traj.shape[0] - 1)
    return traj[idx].copy()


class StopToken:
    """Cooperative, thread-safe stop flag. The UI (or a signal handler)
    calls .stop(reason); the driver polls it once per batch."""

    def __init__(self):
        self._e = threading.Event()
        self.reason = ""

    def stop(self, reason: str = "user requested"):
        self.reason = reason
        self._e.set()

    @property
    def stopped(self) -> bool:
        return self._e.is_set()


def default_workers() -> int:
    """Worker count when the caller doesn't say: every core but two,
    floor one — the server loop and the OS keep breathing while the
    ensemble saturates the rest. A deck never declares this (it is
    throughput, not physics)."""
    import os
    return max(1, (os.cpu_count() or 2) - 2)


def _iter_batches(
    n_ions: int,
    fly_fn: Callable[[int], tuple],
    *,
    check_every: int = 25,
    decimate: int = 1,
    keep_full: bool = False,
    keep_traj: bool = True,
    viability: Optional[Callable[[list, EnsembleProgress], Optional[str]]]
    = None,
    stop_token: Optional[StopToken] = None,
    n_workers: int = 1,
):
    """The single shared physics loop, as a generator yielding an
    EnsembleProgress after every batch of `check_every` ions.

    fly_fn(i) -> (traj_or_None, summary_dict). traj is an (n, k) record
    (whatever the kernel emitted — the driver is shape-agnostic); summary
    must include 'kind' if termination stats/viability use it.

    viability(results_so_far, progress) -> reason:str to ABORT, or None
    to continue. Evaluated once per batch, so it sees >= check_every
    ions before it can veto — enough statistics to judge, e.g., "0%
    transmission and beam not shrinking".

    n_workers > 1 fans fly_fn over a thread pool. Correct by
    MEASUREMENT,
    not hope: per-ion streams are seeded seed_base + i at kernel entry
    and numba's np.random state is THREAD-LOCAL (verified concurrently
    bit-equal to sequential), so a pinned seed gives results
    identical to n_workers=1 regardless of completion order; kernels
    are njit(nogil=True) across every route so threads truly overlap.
    While flying, results arrive in COMPLETION order — each IonResult
    carries .index and consumers needing position must use it — and the
    FINAL yield is sorted by index, so stored runs are byte-identical
    to sequential ones. Stop keeps the per-ion latency contract:
    nothing new is submitted, only in-flight ions finish. A failed ion
    is never dropped: the first worker exception halts submission,
    in-flight ions drain, and the error re-raises with its ion named.
    """
    n_workers = max(1, int(n_workers))
    if n_workers > 1:
        yield from _iter_batches_parallel(
            n_ions, fly_fn, check_every=check_every, decimate=decimate,
            keep_full=keep_full, keep_traj=keep_traj,
            viability=viability, stop_token=stop_token,
            n_workers=n_workers)
        return
    results: list = []
    t0 = time.perf_counter()
    n_term = 0
    stop_reason = ""
    stopped = False
    try:
        for i in range(n_ions):
            # STOP LATENCY IS PER-ION, NOT PER-BATCH (as observed in a
            # browser, Stop "did nothing" — the token was only
            # consulted at check_every yield boundaries, minutes away on
            # slow 3-D ions). A bool read per ion is free; check_every
            # governs PROGRESS EMISSION only. Remaining latency = the
            # current ion's own integration, stated not hidden.
            if stop_token is not None and stop_token.stopped:
                prog = EnsembleProgress(
                    done=i, total=n_ions,
                    elapsed_s=time.perf_counter() - t0,
                    n_terminated=n_term, results=results)
                prog.stopped_early = True
                prog.stop_reason = stop_token.reason
                yield prog
                return
            traj, summary = fly_fn(i)
            if summary.get("kind", -1) == 0:
                n_term += 1
            results.append(IonResult(
                index=i,
                traj=(_decimate(traj, decimate) if keep_traj else None),
                summary=summary,
                full=(traj.copy() if (keep_full and keep_traj
                                      and traj is not None) else None),
            ))
            at_boundary = ((i + 1) % check_every == 0) or (i + 1 == n_ions)
            if at_boundary:
                prog = EnsembleProgress(
                    done=i + 1, total=n_ions,
                    elapsed_s=time.perf_counter() - t0,
                    n_terminated=n_term, results=results)
                if stop_token is not None and stop_token.stopped:
                    stopped, stop_reason = True, stop_token.reason
                if not stopped and viability is not None:
                    r = viability(results, prog)
                    if r:
                        stopped, stop_reason = True, r
                if stopped:
                    prog.stopped_early = True
                    prog.stop_reason = stop_reason
                    yield prog
                    return
                yield prog
    except KeyboardInterrupt:
        yield EnsembleProgress(
            done=len(results), total=n_ions,
            elapsed_s=time.perf_counter() - t0, n_terminated=n_term,
            stopped_early=True, stop_reason="KeyboardInterrupt",
            results=results)
        return


def _iter_batches_parallel(
    n_ions, fly_fn, *, check_every, decimate, keep_full, keep_traj,
    viability, stop_token, n_workers):
    """The n_workers>1 body of _iter_batches; contract documented there.

    HAND-ROLLED DAEMON POOL, not ThreadPoolExecutor (so workers die
    appropriately when the app is
    stopped). Executor threads are non-daemon and joined at
    interpreter exit, so quitting the app mid-flight would BLOCK until
    in-flight ions finished — minutes for slow ions. Daemon workers
    vanish instantly at process exit; Stop/Reset semantics are
    unchanged and sharpened: a stop skips queued-but-unstarted ions
    (workers re-check the token before starting each ion), so only
    ions truly mid-integration finish.
    """
    import queue as _queue
    results: list = []
    t0 = time.perf_counter()
    n_term = 0
    stop_reason = ""
    stopped = False
    failure = None                        # (ion_index, exception)
    work: "_queue.SimpleQueue" = _queue.SimpleQueue()
    done_q: "_queue.SimpleQueue" = _queue.SimpleQueue()
    halt = threading.Event()              # failure OR stop: start no ion

    def _worker_loop():
        while True:
            i = work.get()
            if i is None:
                return
            if halt.is_set() or (stop_token is not None
                                 and stop_token.stopped):
                done_q.put((i, None, None, "skipped"))
                continue
            try:
                traj, summary = fly_fn(i)
                done_q.put((i, traj, summary, None))
            except BaseException as e:    # noqa: BLE001 -- carried + re-raised
                done_q.put((i, None, None, e))

    threads = [threading.Thread(target=_worker_loop, daemon=True,
                                name=f"ion-worker-{k}")
               for k in range(n_workers)]
    for t in threads:
        t.start()

    def _bank(i, traj, summary):
        nonlocal n_term
        if summary.get("kind", -1) == 0:
            n_term += 1
        results.append(IonResult(
            index=i,
            traj=(_decimate(traj, decimate) if keep_traj else None),
            summary=summary,
            full=(traj.copy() if (keep_full and keep_traj
                                  and traj is not None) else None)))

    def _progress(in_flight):
        p = EnsembleProgress(
            done=len(results), total=n_ions,
            elapsed_s=time.perf_counter() - t0,
            n_terminated=n_term, results=results)
        p.n_workers = n_workers
        p.in_flight = in_flight
        return p

    try:
        next_i = 0
        outstanding = 0
        last_emit = 0
        while True:
            want_stop = (halt.is_set()
                         or (stop_token is not None
                             and stop_token.stopped))
            while (not want_stop and failure is None
                   and next_i < n_ions and outstanding < n_workers * 2):
                work.put(next_i)
                next_i += 1
                outstanding += 1
            if outstanding == 0:
                break
            i, traj, summary, err = done_q.get()
            outstanding -= 1
            if err == "skipped":
                pass                       # counted below as not-done
            elif err is not None:
                if failure is None:
                    failure = (i, err)
                halt.set()
            else:
                _bank(i, traj, summary)
            emit = (len(results) - last_emit >= check_every
                    or outstanding == 0 or want_stop
                    or failure is not None)
            if not emit:
                continue
            last_emit = len(results)
            prog = _progress(in_flight=outstanding)
            if stop_token is not None and stop_token.stopped:
                stopped, stop_reason = True, stop_token.reason
            if not stopped and failure is None and viability is not None:
                r = viability(results, prog)
                if r:
                    stopped, stop_reason = True, r
                    if stop_token is not None:
                        stop_token.stop(r)
                    halt.set()
            if (stopped or failure is not None) and outstanding:
                continue                   # drain silently, then finish
            if stopped:
                results.sort(key=lambda x: x.index)
                prog = _progress(in_flight=0)
                prog.stopped_early = True
                prog.stop_reason = stop_reason
                yield prog
                return
            if failure is None:
                yield prog
        if failure is not None:
            i, e = failure
            results.sort(key=lambda x: x.index)
            raise RuntimeError(
                f"ensemble worker failed on ion {i} "
                f"({len(results)}/{n_ions} completed before the "
                f"failure): {type(e).__name__}: {e}") from e
        results.sort(key=lambda x: x.index)
        prog = _progress(in_flight=0)
        if stopped:
            prog.stopped_early = True
            prog.stop_reason = stop_reason
        yield prog
    finally:
        # retire the pool: daemon threads exit on the sentinel; if the
        # generator is abandoned mid-flight the sentinels still land
        # and any ion mid-integration finishes into a queue nobody
        # reads — harmless, and at process exit daemon threads vanish.
        for _ in threads:
            work.put(None)

def run(
    n_ions: int,
    fly_fn: Callable[[int], tuple],
    *,
    check_every: int = 25,
    decimate: int = 1,
    keep_full: bool = False,
    viability=None,
    stop_token: Optional[StopToken] = None,
    on_progress: Optional[Callable[[EnsembleProgress], None]] = None,
):
    """Synchronous run. Drains the batch generator, invoking on_progress
    (e.g. a tqdm update or a FigureWidget refresh) each batch. Returns
    the final EnsembleProgress (with .results, .stopped_early)."""
    last = None
    for prog in _iter_batches(
            n_ions, fly_fn, check_every=check_every, decimate=decimate,
            keep_full=keep_full, viability=viability,
            stop_token=stop_token):
        last = prog
        if on_progress is not None:
            on_progress(prog)
        # An on_progress hook may set the stop_token in response to what
        # it just saw; honour it BEFORE the next batch runs, and reflect
        # it on the returned progress so the caller sees a clean abort at
        # exactly this ion count (no extra batch of wasted flights).
        if (stop_token is not None and stop_token.stopped
                and not prog.stopped_early):
            prog.stopped_early = True
            prog.stop_reason = stop_token.reason
            return prog
    if last is None:
        last = EnsembleProgress(0, n_ions, 0.0, 0)
    return last


class RunHandle:
    """Threading-ready handle. run_threaded() returns this immediately;
    the worker drives the SAME generator. The caller polls .latest for
    the most recent EnsembleProgress (safe: progress objects are only
    read here and the worker only reassigns the reference), calls
    .stop() to request clean early termination, and .join() to await the
    final result. Kernels on this path should be njit(nogil=True) so the
    worker actually runs in parallel with the poller."""

    def __init__(self):
        self.latest: Optional[EnsembleProgress] = None
        self.final: Optional[EnsembleProgress] = None
        # A worker that DIED is a finished run with an error, never a
        # forever-pending one (an uncaught fly_fn
        # exception left .done False permanently, so every later Fly saw
        # a phantom run-in-progress and the app was wedged past Reset).
        self.error: Optional[BaseException] = None
        self.error_tb: Optional[str] = None
        self._stop = StopToken()
        self._thread: Optional[threading.Thread] = None

    def stop(self, reason: str = "user requested"):
        self._stop.stop(reason)

    @property
    def done(self) -> bool:
        return self.final is not None or self.error is not None

    def join(self, timeout: Optional[float] = None):
        if self._thread is not None:
            self._thread.join(timeout)
        return self.final


def run_threaded(
    n_ions: int,
    fly_fn: Callable[[int], tuple],
    *,
    check_every: int = 25,
    decimate: int = 1,
    keep_full: bool = False,
    keep_traj: bool = True,
    viability=None,
    n_workers: int = 1,
) -> RunHandle:
    """Launch the ensemble in a background thread; return a RunHandle to
    poll/stop/join. Same generator, same semantics as run() — only the
    consumption is off-thread. A user-supplied stop_token is merged with
    the handle's own, so handle.stop() works regardless."""
    handle = RunHandle()

    def _worker():
        last = None
        try:
            for prog in _iter_batches(
                    n_ions, fly_fn, check_every=check_every,
                    decimate=decimate, keep_full=keep_full,
                    keep_traj=keep_traj, n_workers=n_workers,
                    viability=viability, stop_token=handle._stop):
                handle.latest = last = prog
            handle.final = last
        except Exception as e:            # noqa: BLE001 -- recorded, surfaced
            # Not a swallow: the error is RECORDED on the handle (which
            # marks the run done so the UI can recover), the traceback is
            # printed loudly for the console, and partial progress is
            # kept so what DID fly is inspectable.
            import traceback
            handle.error = e
            handle.error_tb = traceback.format_exc()
            handle.final = last
            print(handle.error_tb, flush=True)

    t = threading.Thread(target=_worker, daemon=True)
    handle._thread = t
    t.start()
    return handle


# ------------------------------------------------------------ viability
# Reusable predicates. Each returns a reason string to abort, or None.

def viability_transmission(min_fraction: float, after: int = 50):
    """Abort if, after >= `after` ions, the terminated (impact, kind==0)
    fraction is below min_fraction — e.g. a funnel/cooler config that is
    losing everything to the walls. (For a funnel, impact on the exit
    PLATE is success, so invert min_fraction accordingly per problem.)"""
    def _v(results, prog: EnsembleProgress):
        if prog.done < after:
            return None
        frac = prog.n_terminated / prog.done
        if frac < min_fraction:
            return (f"transmission {frac:.0%} < {min_fraction:.0%} "
                    f"after {prog.done} ions")
        return None
    return _v


def viability_beam_shrinks(radius_of, after: int = 50,
                           min_ratio: float = 0.9):
    """Abort if the beam is not being confined: compares mean final
    radius of the last third of processed ions to the first third; if it
    has GROWN (ratio > 1/min_ratio), the trap/cooler is not working.
    radius_of(summary)->float extracts a final radius from each summary.
    """
    def _v(results, prog: EnsembleProgress):
        if prog.done < after:
            return None
        r = np.array([radius_of(x.summary) for x in results])
        r = r[np.isfinite(r)]
        if len(r) < after:
            return None
        k = len(r) // 3
        early, late = r[:k].mean(), r[-k:].mean()
        if early > 0 and late / early > 1.0 / min_ratio:
            return (f"beam not confined: final radius grew "
                    f"{late/early:.2f}x across the run")
        return None
    return _v
