"""Figure-dispatcher retention harness — chartered by the PI 2026-09-29.

The defect it guards against (L-516, field-confirmed the same night):
Bokeh retains a STOPPED periodic callback for the session's lifetime,
so any closure chain riding the worker+poll pattern is immortal.
Whatever those closures captured is pinned with them — measured live on
the split-float tetramer as one whole Stl3DModel (~1.6 GB) per PE
compute (the prep pack `p` rode the apply closure) plus each dispatch's
previous pane figure, climbing 7.2 -> 11.9 GB in a two-minute session.

The fix under test: every reference the finish path consumes lives in
the dispatcher's state dict and is POPPED OUT at delivery, so the
retained chain keeps only an emptied dict. Pinned here, for BOTH
dispatchers (pe_view._run_off_doc and SimApp._dispatch_figure):

  * under emulated Bokeh retention (a callback registry that never
    drops a stopped callback), the delivered payload and everything
    the caller's closures captured are FREED at delivery — proven by
    weakref, with the callback still retained;
  * delivery happens exactly once, and a late tick after stop is a
    benign no-op (documented skip, not a swallow);
  * the error path releases the same way and reports exactly once;
  * _dispatch_figure releases base_obj (the previous pane figure) —
    the secondary leak of one stale figure pair per redraw;
  * the headless no-server paths are unchanged: synchronous delivery,
    error reporting intact.

The emulation is the MECHANISM, not a mock of Bokeh's API surface:
pn.state.curdoc is patched truthy and add_periodic_callback returns a
handle whose stop() keeps the closure — exactly the retention the
referrer walk measured. If Panel ever stops retaining stopped
callbacks these tests still pass; if a dispatcher regresses to keeping
consumer refs in closure cells or a plain local, the weakref stays
alive and the test names which dispatcher leaked.
"""
import gc
import threading
import time
import types
import unittest.mock as um
import weakref

import panel as pn
import pytest

from ion_gym.ui.sim_app import SimApp
from ion_gym.viz import pe_view

# The worker threads here do no numerics; this bound only turns a hung
# worker into a named failure instead of a hung suite.
WORKER_WAIT_S = 5.0


class _Payload:
    """Stand-in for the heavy freight (an Stl3DModel, a figure): any
    plain object a weakref can watch."""


class _RetainedPCB:
    """A periodic-callback handle whose stop() does NOT drop the
    closure — the measured Bokeh behavior this harness emulates."""

    def __init__(self, fn):
        self.fn = fn

    def stop(self):
        # retention on purpose: self.fn survives, exactly like a
        # stopped bokeh PeriodicCallback held by the session
        pass


class _Retainer:
    """Session-lifetime callback registry: everything ever registered
    stays reachable, so anything a callback chain pins stays pinned."""

    def __init__(self):
        self.callbacks = []

    def add(self, fn, period):
        pcb = _RetainedPCB(fn)
        self.callbacks.append(pcb)
        return pcb

    def tick_all(self):
        for pcb in list(self.callbacks):
            pcb.fn()


@pytest.fixture()
def retained_server():
    """Patch pn.state into 'server session with retained callbacks'."""
    reg = _Retainer()
    with um.patch.object(pn.state, "add_periodic_callback",
                         side_effect=reg.add), \
         um.patch.object(type(pn.state), "curdoc",
                         new_callable=um.PropertyMock,
                         return_value=object()):
        yield reg


def _min_app():
    """A SimApp shell with exactly what _dispatch_figure touches —
    pane, status, watchdog hook — and none of the widget tree."""
    app = SimApp.__new__(SimApp)
    app.pane = types.SimpleNamespace(object=None, loading=False)
    app.status = types.SimpleNamespace(object="")
    app._ensure_watchdog_pulse = lambda: None
    return app


def _assert_freed(ref, who):
    gc.collect()
    assert ref() is None, (
        f"{who}: freight still pinned by the retained callback chain "
        f"after delivery — a consumer ref stayed in a closure or the "
        f"state dict")


# ---------------------------------------------------------------------------
# pe_view._run_off_doc
# ---------------------------------------------------------------------------

def _apply_with_prep_pack(delivered):
    """The L-516 shape, built in its own scope: apply closes over a
    prep pack `p` holding the model — the exact capture the referrer
    walk named (SimpleNamespace.model). Only the weakref leaves this
    factory, so once the dispatcher drops its consumer refs nothing
    outside the retained callback chain can pin the model."""
    p = types.SimpleNamespace(model=_Payload())

    def apply_fn(res):
        delivered.append((res, p.model is not None))

    return apply_fn, weakref.ref(p.model)


def test_run_off_doc_releases_capture_under_retention(retained_server):
    delivered = []
    apply_fn, ref = _apply_with_prep_pack(delivered)

    pe_view._run_off_doc(lambda: "surf",
                         apply_fn,
                         lambda e, tb: pytest.fail(f"error path hit: {e}"))
    del apply_fn                     # the dispatcher holds the only ref now
    # drive the retained callback until the apply lands, then late-tick
    t0 = time.monotonic()
    while not delivered and time.monotonic() - t0 < WORKER_WAIT_S:
        retained_server.tick_all()
        time.sleep(0.01)
    assert delivered == [("surf", True)], delivered
    retained_server.tick_all()               # late tick: no re-delivery
    assert len(delivered) == 1, "apply ran more than once"
    _assert_freed(ref, "pe_view._run_off_doc")
    assert retained_server.callbacks, "harness invalid: nothing retained"


class _Boom(RuntimeError):
    pass


def _raising_build(hold_s=0.0):
    """A build that captures freight and raises — the error-path
    counterpart of the prep-pack capture. Only the weakref escapes."""
    freight = _Payload()
    gate = threading.Event()

    def build():
        gate.wait(hold_s)
        raise _Boom(f"carrying {type(freight).__name__}")

    return build, weakref.ref(freight)


def test_run_off_doc_error_path_releases_under_retention(retained_server):
    errors = []
    build, ref = _raising_build()

    def on_error(e, tb):
        errors.append(type(e).__name__)
        assert "_Boom" in tb

    pe_view._run_off_doc(build, lambda res: pytest.fail("apply on error"),
                         on_error)
    del build
    t0 = time.monotonic()
    while not errors and time.monotonic() - t0 < WORKER_WAIT_S:
        retained_server.tick_all()
        time.sleep(0.01)
    retained_server.tick_all()
    assert errors == ["_Boom"], "error reported other than exactly once"
    _assert_freed(ref, "pe_view._run_off_doc (error path)")


def test_run_off_doc_headless_sync_paths():
    """No server session: synchronous contract unchanged."""
    got, errs = [], []
    pe_view._run_off_doc(lambda: "S", got.append,
                         lambda e, tb: errs.append(e))
    assert got == ["S"] and errs == []
    pe_view._run_off_doc(lambda: 1 / 0, got.append,
                         lambda e, tb: errs.append(type(e).__name__))
    assert errs == ["ZeroDivisionError"]
    assert got == ["S"], "apply ran on the error path"


# ---------------------------------------------------------------------------
# SimApp._dispatch_figure
# ---------------------------------------------------------------------------

def _slow_build(payload, hold_s=0.2):
    """Miss the 50 ms fast path so the dispatch takes the periodic-
    callback route — the only route with retention."""
    gate = threading.Event()

    def build():
        gate.wait(hold_s)
        return payload

    return build


def test_dispatch_figure_releases_payload_under_retention(retained_server):
    app = _min_app()
    payload = _Payload()
    ref = weakref.ref(payload)
    published = []

    app._dispatch_figure("retention probe", _slow_build(payload),
                         publish=published.append)
    t0 = time.monotonic()
    while not published and time.monotonic() - t0 < WORKER_WAIT_S:
        retained_server.tick_all()
        time.sleep(0.02)
    assert published and published[0] is payload
    retained_server.tick_all()
    assert len(published) == 1, "publish ran more than once"
    del payload, published[:]
    _assert_freed(ref, "SimApp._dispatch_figure (payload)")
    assert not app.pane.loading


def test_dispatch_figure_releases_base_obj_under_retention(retained_server):
    """The secondary leak: each dispatch pinned the PREVIOUS pane
    figure through its identity snapshot."""
    app = _min_app()
    old_fig = _Payload()
    app.pane.object = old_fig                 # the figure being replaced
    ref = weakref.ref(old_fig)
    published = []

    app._dispatch_figure("base_obj probe", _slow_build(_Payload()),
                         publish=lambda fig: (
                             setattr(app.pane, "object", fig),
                             published.append(True)))
    t0 = time.monotonic()
    while not published and time.monotonic() - t0 < WORKER_WAIT_S:
        retained_server.tick_all()
        time.sleep(0.02)
    assert published == [True]
    retained_server.tick_all()
    del old_fig
    _assert_freed(ref, "SimApp._dispatch_figure (base_obj)")


def test_dispatch_figure_error_path_releases_under_retention(
        retained_server):
    app = _min_app()
    build, ref = _raising_build(hold_s=0.2)   # miss the 50 ms fast path

    app._dispatch_figure("err probe", build,
                         publish=lambda fig: pytest.fail("published error"))
    del build
    t0 = time.monotonic()
    while ("err probe error" not in str(app.status.object)
           and time.monotonic() - t0 < WORKER_WAIT_S):
        retained_server.tick_all()
        time.sleep(0.02)
    assert "err probe error" in str(app.status.object)
    retained_server.tick_all()               # late tick after error
    _assert_freed(ref, "SimApp._dispatch_figure (error path)")
    assert not app.pane.loading


def test_dispatch_figure_headless_sync_paths():
    app = _min_app()
    out = []
    app._dispatch_figure("headless", lambda: 42, publish=out.append)
    assert out == [42]
    app._dispatch_figure("headless err", lambda: 1 / 0,
                         publish=out.append)
    assert "headless err error" in str(app.status.object)
    assert out == [42], "published on the error path"
