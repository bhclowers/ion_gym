"""
test_solve_orchestrator.py — the app's async solve must be overlap-safe.

The reported failure: load the funnel, tweak a DC, tweak the contour
count, hit Recompute — nothing happens, no spinner, Fly dead. Cause: every
rebuild took the threaded path (build_needs_solve was wrongly True for
cached r-z), and overlapping solves shared ONE periodic-callback handle,
so an old poll finishing STOPPED the new solve's poll — the new result was
computed but never delivered. These gates lock the fix.

Gates (with a fake 0.3 s build and a fake event loop driving polls):
  O-1 SINGLE: one slow solve delivers its result and stops its poll.
  O-2 OVERLAP: two overlapping solves — the older is superseded (result
      dropped), the newer delivers, and NO periodic callback leaks.
  O-3 STATUS: Recompute's "recomputed" message appears only AFTER the
      build completes, never while it is still running.
  O-4 CACHED-RZ: build_needs_solve is False for a cached r-z geometry
      (funnel tweaks go inline — no thread, no poll, no race).
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ion_gym.ui import sim_app as SA
from ion_gym.ui.sim_app import SimApp, _example_specs


class FakePCB:
    def __init__(self, fn):
        self.fn = fn
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeState:
    def __init__(self, polls):
        self._polls = polls
        # sim_app's orchestrator grew an
        # in_server probe (pn.state.curdoc is not None) AFTER this gate
        # was written; without a curdoc the fake silently steered every
        # solve down the non-server path and polls stayed empty
        # (polls[0] IndexError, red since v275). The fake must present
        # the interface it fakes.
        self.curdoc = object()

    def add_periodic_callback(self, fn, period):
        p = FakePCB(fn)
        self._polls.append(p)
        return p


def _drive(polls, until, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for p in list(polls):
            if not p.stopped:
                p.fn()
        if until():
            return True
        time.sleep(0.03)
    return False


def main():
    ok = True
    polls = []
    delivered = []

    real_state = SA.pn.state
    real_build = SA.build_run
    real_will = SA.build_needs_solve

    def fake_build(spec, verbose=False, **kw):
        # **kw ON PURPOSE. This gate certifies the SOLVE ORCHESTRATOR --
        # threading, polling, delivery -- and nothing about build_run's
        # parameter list. Pinned to the exact signature, it broke the
        # moment build_run gained record_budget_gb: a concurrency gate
        # failing for a reason that has nothing to do with concurrency,
        # the same coupling the note below describes for menu keys.
        time.sleep(0.3)
        return ("MODEL", "FLY", ["t"], [0, 1])

    async def run_gates():
        nonlocal ok
        app = SimApp()
        # AGNOSTIC: this gate tests the SOLVE ORCHESTRATOR -- the threading,
        # the polling, the delivery -- not any particular geometry.  It used to
        # name an example by string key, so deleting the planar einzel (a
        # product decision about the EXAMPLE MENU) broke a gate about
        # CONCURRENCY.  A gate coupled to a menu entry fails for reasons that
        # have nothing to do with what it certifies.  Take whatever is there.
        specs = _example_specs()
        assert specs, "no example specs registered at all"
        app.spec = specs[next(iter(specs))]()
        app._build_controls()
        # drawing is OUT OF SCOPE here (this gate certifies threading/
        # polling/delivery); the fake model is not drawable and the
        # strict viz contract would rightly refuse it.
        app._redraw = lambda *a, **k: None
        # same scope statement for the field/PE tabs: their refresh
        # inspects the (fake) model via the strict viz contract.
        for _tab in ("_fs_tab", "_pe_tab"):
            if getattr(app, _tab, None) is not None:
                getattr(app, _tab).refresh = lambda *a, **k: None
        SA.pn.state = FakeState(polls)
        SA.build_run = fake_build
        SA.build_needs_solve = lambda spec: True     # force threaded path

        # O-1 single solve delivers
        app._solve_then(lambda *r: delivered.append(r))
        got = _drive(polls, lambda: len(delivered) == 1)
        g1 = got and polls[0].stopped
        ok &= g1
        print(f"O-1 single: delivered={len(delivered)}, poll stopped="
              f"{polls[0].stopped}  ->  {'PASS' if g1 else 'FAIL'}")

        # O-2 overlap: second supersedes first
        polls.clear()
        delivered.clear()
        # The REUSE path (added after this gate) hands
        # an unchanged spec's stored build back INLINE — correct product
        # behaviour, but it means no overlap ever exists. Clear the
        # stored build so both submissions genuinely go threaded, which
        # is the concurrency this section certifies.
        app._built = None
        app._solve_then(lambda *r: delivered.append(("A",) + r))
        time.sleep(0.05)
        app._solve_then(lambda *r: delivered.append(("B",) + r))
        _drive(polls, lambda: all(p.stopped for p in polls))
        tags = [d[0] for d in delivered]
        g2 = tags == ["B"] and all(p.stopped for p in polls)
        ok &= g2
        print(f"O-2 overlap: delivered={tags} (want ['B']), leaked polls="
              f"{sum(not p.stopped for p in polls)}  ->  "
              f"{'PASS' if g2 else 'FAIL'}")

        # O-3 recompute status honest
        app._built = None   # force the threaded path (see O-2)
        polls.clear()
        app._base_figure = lambda m, la="x", lb="y": SA.go.Figure()
        app.status.object = "before"
        app._on_recompute()
        early = app.status.object
        _drive(polls, lambda: "recomputed" in str(app.status.object),
               timeout=5.0)
        g3 = ("recomputed" not in str(early)
              and "recomputed" in str(app.status.object))
        ok &= g3
        print(f"O-3 status: during-build={early!r:.40} final="
              f"{str(app.status.object)!r:.40}  ->  "
              f"{'PASS' if g3 else 'FAIL'}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_gates())
    finally:
        SA.pn.state = real_state
        SA.build_run = real_build
        SA.build_needs_solve = real_will

    # O-4 cached r-z is inline (real functions, real cache)
    from ion_gym.physics.sim_build import build_run, build_needs_solve
    # select the funnel by PATTERN, not by menu key (this gate's own
    # doctrine: coupling to a menu string breaks a concurrency gate for
    # product reasons; the key changed when the funnel config
    # replaced the default example).
    _keys = [k for k in _example_specs() if "funnel" in k.lower()]
    assert len(_keys) >= 1, "no funnel example registered"
    spec = _example_specs()[_keys[0]]()
    # DECLARED seed: the funnel deck carries seed=null =
    # fresh entropy; seed 0 = the historical draws. Also keeps
    # O-4's cache key stable across the two build calls below.
    spec.source.seed = 0
    build_run(spec)                     # solve + cache
    g4 = not build_needs_solve(spec)
    ok &= g4
    print(f"O-4 cached rz inline: build_needs_solve={build_needs_solve(spec)} "
          f"(want False)  ->  {'PASS' if g4 else 'FAIL'}")

    print("\nSOLVE ORCHESTRATOR GATES:", "ALL PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
