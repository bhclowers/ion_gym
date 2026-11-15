"""Fly one deck for a run figure: build, time one ion, fly the batch,
collect the impact arrivals.

The split into plan_flight and fly_flight is the notebook quote rule:
plan_flight builds the deck and measures one ion, so a notebook can
print the projected time in its own cell and fly the batch in the next.
Arrivals come from physics.stations.impact_hits, the function the
workbench's Impact Analysis tab uses, so a banked flight and the
workbench agree. Bank with io.results_bank.bank_flight and draw with
viz.run_figures.publication_run_figure.

Import-clean: importing this module builds and flies nothing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

# progress lines printed over one batch (the ensemble reports per check)
PROGRESS_LINES = 10


@dataclass
class FlightPlan:
    """A built deck ready to fly, with the measured time of one ion."""
    spec: object
    model: object
    fly_fn: object
    cols: list
    n_ions: int
    per_ion_s: float

    def projected_s(self, n_workers=1):
        """Projected wall time of the batch on n_workers threads, from
        the one measured ion, assuming ideal scaling across workers (ions
        fly independently; shared memory bandwidth makes real scaling
        somewhat worse, so read it as a lower bound)."""
        return self.per_ion_s * self.n_ions / max(1, int(n_workers))


def plan_flight(spec, *, record_budget_gb=None):
    """Build `spec` and time one ion. The first flight compiles the
    kernel, so it runs untimed; the second is the measured one (both are
    flown again in fly_flight, so neither is a result). Refuses a deck
    that births fewer than two ions, which leaves nothing to time.
    record_budget_gb goes to build_run (None derives it from RAM)."""
    from ion_gym.physics.sim_build import build_run
    model, fly_fn, cols, births = build_run(
        spec, record_budget_gb=record_budget_gb)
    n = len(births)
    if n < 2:
        raise ValueError(f"plan_flight: the deck births {n} ion(s); two "
                         f"are needed to time a warm flight")
    fly_fn(0)
    t0 = time.perf_counter()
    fly_fn(1)
    return FlightPlan(spec=spec, model=model, fly_fn=fly_fn,
                      cols=list(cols), n_ions=n,
                      per_ion_s=time.perf_counter() - t0)


def fly_flight(plan, *, label="flight", n_workers=None):
    """Fly every ion of `plan` with progress, then collect the impact
    arrivals. Returns (results, hits). Refuses an ensemble that returns
    fewer ions than it was asked for.

    n_workers : threads to fly on; None takes
    ensemble_driver.default_workers() (every core but two), the rule the
    workbench uses. A seeded deck gives the same results for any count.
    """
    from ion_gym.physics import ensemble_driver as ed
    from ion_gym.physics.stations import impact_hits

    if n_workers is None:
        n_workers = ed.default_workers()
    n_workers = int(n_workers)
    if n_workers < 1:
        raise ValueError(f"fly_flight: n_workers={n_workers}; give 1 or "
                         f"more, or None for default_workers()")
    print(f"flying {plan.n_ions} ions on {n_workers} worker thread(s)",
          flush=True)

    printed = {"done": -1}

    def on_progress(p):
        # one flushed line per batch: works headless and in nbclient. The
        # parallel path yields its final count twice (once as the last
        # batch, once sorted by ion index), so a repeated count is not
        # printed again
        if p.done == printed["done"]:
            return
        printed["done"] = p.done
        eta = (p.elapsed_s / p.done * (p.total - p.done)) if p.done else None
        print(f"  [{p.done}/{p.total}] {label}: elapsed {p.elapsed_s:.0f} s"
              + (f", eta {eta:.0f} s" if eta is not None else ""),
              flush=True)

    prog = ed.run(plan.n_ions, plan.fly_fn, keep_full=True,
                  check_every=max(1, plan.n_ions // PROGRESS_LINES),
                  on_progress=on_progress, n_workers=n_workers)
    results = list(prog.results)
    if len(results) != plan.n_ions:
        raise RuntimeError(f"fly_flight: ensemble returned {len(results)} "
                           f"of {plan.n_ions} ions (stopped early: "
                           f"{prog.stopped_early})")
    return results, impact_hits(plan.spec, plan.cols, results)


def _sysctl(key):
    """One macOS sysctl value as text, or an explicit 'unknown (...)'."""
    import subprocess
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, check=True, timeout=5)
    except FileNotFoundError:
        return "unknown (no sysctl)"
    except subprocess.CalledProcessError as e:
        return f"unknown (sysctl {key} exit {e.returncode})"
    except subprocess.TimeoutExpired:
        return f"unknown (sysctl {key} timed out)"
    return out.stdout.strip() or f"unknown (sysctl {key} empty)"


def machine_info():
    """The machine a timing was measured on, as a dict of plain strings:
    cpu model, logical cores, OS, and the Python, numba, and NumPy
    versions. On macOS the performance and efficiency core counts are
    included, because an ensemble spread over both kinds does not scale
    like one spread over identical cores. A field that cannot be read is
    reported as 'unknown (reason)', never left out."""
    import os
    import platform
    import numba
    import numpy as np
    info = {"os": platform.platform(),
            "python": platform.python_version(),
            "numba": numba.__version__, "numpy": np.__version__,
            "logical_cores": str(os.cpu_count())}
    system = platform.system()
    if system == "Darwin":
        info["cpu"] = _sysctl("machdep.cpu.brand_string")
        info["performance_cores"] = _sysctl("hw.perflevel0.physicalcpu")
        info["efficiency_cores"] = _sysctl("hw.perflevel1.physicalcpu")
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                names = [ln.split(":", 1)[1].strip() for ln in fh
                         if ln.startswith("model name")]
        except OSError as e:
            names = []
            info["cpu"] = f"unknown (/proc/cpuinfo: {e})"
        if names:
            info["cpu"] = names[0]
        elif "cpu" not in info:
            info["cpu"] = "unknown (no model name in /proc/cpuinfo)"
    else:
        info["cpu"] = platform.processor() or f"unknown ({system})"
    return info
