"""
ion_gym.optimize
----------------
Declarative optimization over SimSpec parameters with CMA-ES.

An OptimizeSpec names WHAT to vary (dotted paths into a SimSpec, e.g.
"geometry.electrodes[1].dc"), the bounds, and WHICH metric of the flown
ensemble to minimize. The driver patches a deep-copied spec, builds it
through the SAME build_run dispatch the app uses (so every geometry class —
planar, r-z, STL, 3-D, imported — is optimizable), flies the ensemble, and
scores it. Deterministic per candidate (fixed seeds via fly_fn(i)), so CMA-ES
sees a noise-free objective unless collisions are enabled.

Metrics (objective is always MINIMIZED):
  * "exit_radius"   mean final transverse radius of transmitted ions
                    (fate==1), about the geometry centre. + penalty for
                    non-transmitted ions. The einzel/beam-focus objective.
  * "spot_size"     RMS of final (x, y) about their own centroid
                    (transmitted only, same penalty).
  * "loss_fraction" fraction of ions NOT transmitted.
  * "tof_spread"    std of time-of-flight of transmitted ions (us).
  * callable        any user function f(results, cols, spec) -> float.

Caching note: geometry-affecting parameters trigger a re-solve per candidate;
voltage-only parameters re-weight cached bases, so voltage optimizations are
fast. Prefer voltage/drive parameters in the loop where possible.
"""
from __future__ import annotations
import copy
import json
import math
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

import numpy as np


# ------------------------------------------------------------ param paths
_TOKEN = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)(?:\[(\d+)\])?")


def _resolve(obj, path):
    """Walk 'a.b[2].c' to (parent, final_attr_or_index_setter)."""
    parts = path.split(".")
    cur = obj
    for k, p in enumerate(parts):
        m = _TOKEN.fullmatch(p)
        if not m:
            raise ValueError(f"bad path token {p!r} in {path!r}")
        name, idx = m.group(1), m.group(2)
        last = (k == len(parts) - 1)
        nxt = getattr(cur, name)
        if idx is not None:
            i = int(idx)
            if last:
                return (nxt, i, "index")
            nxt = nxt[i]
        elif last:
            return (cur, name, "attr")
        cur = nxt
    raise ValueError(f"empty path {path!r}")


def set_param(spec, path, value):
    holder, key, kind = _resolve(spec, path)
    if kind == "index":
        holder[key] = value
    else:
        setattr(holder, key, float(value))


def get_param(spec, path):
    holder, key, kind = _resolve(spec, path)
    return holder[key] if kind == "index" else getattr(holder, key)


# A params entry is ONE search variable. It may be a single dotted path or
# a SEQUENCE of paths that all receive the same value (a TIED parameter —
# e.g. two opposite-phase RF rails whose amplitude must move together).
# General: any spec, any paths; not tied to a particular configuration.
def _param_paths(entry):
    return [entry] if isinstance(entry, str) else list(entry)


def _param_key(entry):
    """Stable display/dict key for a params entry."""
    return entry if isinstance(entry, str) else " | ".join(entry)


# ------------------------------------------------------------ metrics
def _transmitted(results, spec=None):
    """Ions that made it, per the SPEC'S OWN definition of "made it".

    Config-agnostic by design: this once hardcoded fate 1 (boundary
    exit). Any spec that declares a bounding plane terminates its
    transmitted ions with fate 3 instead, so on those configurations
    every candidate scored zero-transmitted and fell into the metric's
    penalty branch — the optimizer was then ranking penalties, not
    physics, and would happily "converge" on nonsense. `stats` already
    owns this decision (auto_transmitted_fate); defer to it so the
    optimizer and the reported statistics can never disagree about which
    ions counted. spec=None keeps the old literal for a caller that has
    no spec to hand, and says so rather than guessing silently.
    """
    if spec is None:
        want = 1
    else:
        from ion_gym.physics.stats import auto_transmitted_fate
        want = auto_transmitted_fate(spec)
    return [r for r in results if r.summary.get("kind") == want]


def _final_xy(r, cols):
    from ion_gym.io.records import TrajRecord
    end = TrajRecord(r.traj, cols).row(-1)
    return end["x"], end["y"]


def metric_exit_radius(results, cols, spec):
    ok = _transmitted(results, spec)
    n = max(len(results), 1)
    penalty = 10.0 * (1.0 - len(ok) / n)      # losing ions must never win
    if not ok:
        return 100.0 + penalty
    cx = spec.geometry.width_mm / 2.0
    cy = spec.geometry.height_mm / 2.0
    rs = [math.hypot(x - cx, y - cy) for x, y in
          (_final_xy(r, cols) for r in ok)]
    return float(np.mean(rs)) + penalty


def metric_spot_size(results, cols, spec):
    ok = _transmitted(results, spec)
    n = max(len(results), 1)
    penalty = 10.0 * (1.0 - len(ok) / n)
    if not ok:
        return 100.0 + penalty
    xy = np.array([_final_xy(r, cols) for r in ok])
    return float(np.sqrt(((xy - xy.mean(0)) ** 2).sum(1).mean())) + penalty


def metric_loss_fraction(results, cols, spec):
    n = max(len(results), 1)
    return 1.0 - len(_transmitted(results)) / n


def metric_tof_spread(results, cols, spec):
    ok = _transmitted(results, spec)
    n = max(len(results), 1)
    penalty = 1e3 * (1.0 - len(ok) / n)
    if len(ok) < 2:
        return 1e3 + penalty
    from ion_gym.io.records import TrajRecord
    ends = [TrajRecord(r.traj, cols).row(-1)["t"] for r in ok]
    return float(np.std(ends)) + penalty


METRICS = {"exit_radius": metric_exit_radius,
           "spot_size": metric_spot_size,
           "loss_fraction": metric_loss_fraction,
           "tof_spread": metric_tof_spread}


# ------------------------------------------------------------ spec
@dataclass
class OptimizeSpec:
    params: list                            # per-variable: dotted path OR sequence of tied paths
    lo: Sequence[float]                     # per-param bounds
    hi: Sequence[float]
    metric: Union[str, Callable] = "exit_radius"
    n_ions: int = 0                         # 0 = keep the spec's n_ions
    sigma0: float = 0.3                     # CMA initial step (norm. units)
    max_evals: int = 120
    popsize: int = 0                        # 0 = CMA default
    seed: int = 1
    verbose: bool = False
    # ---- stochastic-objective policy (collisions on => noisy metric) ----
    # Rotating common-random-numbers: every candidate in a GENERATION shares
    # one seed base (paired ranking — noise cancels in the comparisons CMA
    # actually makes), and the base ROTATES each generation (the search
    # cannot overfit a single noise draw). The reported optimum is then
    # re-evaluated on FRESH seeds never used in the search (audit) so the
    # returned value is an unbiased estimate, not a lucky draw.
    n_repeats: int = 1        # repeats averaged per evaluation (seed-strided)
    crn: str = "rotate"       # "rotate" (default) | "fixed" (one base always)
    n_final: int = 5          # fresh-seed audit repeats of the best point
    bank_path: str = ""       # optional JSONL result bank: one line per
                              # evaluation (kind=eval) + one audit line
                              # (kind=audit), appended crash-safe, so a
                              # long campaign is persisted as it runs.
    checkpoint_path: str = "" # optional pickle: CMA state + counters,
                              # written after EVERY generation. A rerun
                              # with the same path RESUMES; a completed
                              # run returns its stored result and does
                              # no work. (Sandbox tool calls die ~285 s;
                              # local runs get crash-safety for free.)
    time_budget_s: float = 0  # >0: return (complete=False) before
                              # STARTING a generation that would exceed
                              # this wall budget. Requires checkpoint_path.
    spec_factory: object = None   # optional callable(values)->SimSpec.
                              # GEOMETRY MODE: replaces the dotted-path
                              # apply entirely; `params` then only names
                              # the vector components for reporting, and
                              # lo/hi still bound the search. A factory
                              # ValueError = infeasible candidate ->
                              # infeasible_value (recorded, not raised).
    infeasible_value: float = 1.0e9   # objective value assigned to
                              # factory-refused candidates; must exceed
                              # every constraint-penalty tier.
    x0: object = None         # optional starting point (PHYSICAL units,
                              # length = len(params)); None = domain
                              # midpoint. Basin-seeded refinements pass
                              # the screen winner here.

    def __post_init__(self):
        assert len(self.params) == len(self.lo) == len(self.hi)


@dataclass
class OptimizeResult:
    best_params: dict
    best_value: float               # best SEARCH value (CRN seeds)
    n_evals: int
    history: list = field(default_factory=list)   # (value, {param: val})
    audit_value: float = float("nan")   # fresh-seed mean at best_params
    audit_std: float = float("nan")     # fresh-seed std  at best_params
    complete: bool = True           # False: budget hit; rerun to resume


def _bank_write(path, row):
    """Append one JSON line. Crash-safe (open/write/close per line); an
    unwritable bank is a real failure, not a warning."""
    row = dict(row, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------ driver
class _R:                                       # minimal result adapter
    __slots__ = ("traj", "summary")

    def __init__(self, t, s):
        self.traj, self.summary = t, s


def _evaluate(base_spec, ospec, values, seed_base=None, n_repeats=None):
    """One objective evaluation. seed_base sets the FULL stochastic draw
    (births + collision RNG, every builder derives from source.seed);
    n_repeats > 1 averages seed-strided repeats."""
    from ion_gym.physics.sim_build import build_run
    reps = n_repeats if n_repeats is not None else max(ospec.n_repeats, 1)
    fn = ospec.metric if callable(ospec.metric) else METRICS[ospec.metric]
    vals = []
    for r in range(reps):
        if ospec.spec_factory is not None:
            # GEOMETRY MODE: the factory owns vector -> spec. It may
            # REFUSE (ValueError subclass, e.g. CellBoardInfeasible);
            # a refusal is a recorded infeasible evaluation with the
            # declared penalty — never a crash, never a silent skip.
            try:
                sp = ospec.spec_factory(values)
            except ValueError as e:
                if ospec.verbose:
                    print("[optimize] infeasible candidate: "
                          + str(e)[:70])
                return float(ospec.infeasible_value), 0.0
        else:
            sp = copy.deepcopy(base_spec)
            for entry, v in zip(ospec.params, values):
                for path in _param_paths(entry):
                    set_param(sp, path, float(v))
        if ospec.n_ions:
            sp.source.n_ions = int(ospec.n_ions)
        if seed_base is not None:
            sp.source.seed = int(seed_base + r * 10007)   # prime stride
        model, fly, cols, births = build_run(sp)
        results = []
        for i in range(len(births)):
            t, s = fly(i)
            results.append(_R(t, s))
        vals.append(float(fn(results, cols, sp)))
    return float(np.mean(vals)), float(np.std(vals))


def optimize(base_spec, ospec: OptimizeSpec) -> OptimizeResult:
    """CMA-ES over the named SimSpec parameters (normalized to [0,1] per
    bound pair). Returns the best parameter dict and the search history."""
    try:
        import cma
    except ImportError as e:      # refuse with the remedy, not a stack trace
        raise RuntimeError(
            "CMA-ES optimization needs the 'cma' package, which is not "
            "installed in this environment. Install it with "
            "`pip install cma` (pure Python, no build step). Everything "
            "else in ion_gym runs without it.") from e
    lo = np.asarray(ospec.lo, float)
    hi = np.asarray(ospec.hi, float)
    span = hi - lo
    ndim = len(lo)
    pad = ndim == 1        # CMA's bounds transform needs dimension >= 2:
                           # pad 1-D problems with an ignored dummy coord

    def denorm(z):
        z = np.asarray(z)[:ndim]
        return lo + np.clip(z, 0.0, 1.0) * span

    if ospec.x0 is not None:
        z0 = ((np.asarray(ospec.x0, float) - lo)
              / np.where(span > 0, span, 1.0))
        if z0.shape != (ndim,):
            raise ValueError("x0 length {0} != ndim {1}".format(
                z0.shape, ndim))
        if (z0 < -1e-9).any() or (z0 > 1 + 1e-9).any():
            raise ValueError("x0 outside [lo, hi] bounds")
        x0 = np.clip(np.concatenate([z0, [0.5]]) if pad else z0, 0.0, 1.0)
    else:
        x0 = np.full(ndim + (1 if pad else 0), 0.5)
    opts = {"bounds": [0.0, 1.0], "seed": ospec.seed,
            "maxfevals": ospec.max_evals, "verbose": -9}
    if ospec.popsize:
        opts["popsize"] = ospec.popsize
    t_start = time.time()
    if ospec.time_budget_s and not ospec.checkpoint_path:
        raise ValueError("time_budget_s needs checkpoint_path — a budgeted "
                         "return without a checkpoint would lose the run.")
    ck = ospec.checkpoint_path
    if ck and os.path.exists(ck):
        with open(ck, "rb") as f:
            state = pickle.load(f)
        if state.get("done"):
            return state["result"]          # completed earlier: no work
        es = state["es"]
        history = state["history"]
        n_evals = state["n_evals"]
        gen = state["gen"]
    else:
        es = cma.CMAEvolutionStrategy(x0, ospec.sigma0, opts)
        history = []
        n_evals = 0
        gen = 0
    while not es.stop():
        if (ospec.time_budget_s
                and time.time() - t_start > ospec.time_budget_s):
            with open(ck, "wb") as f:
                pickle.dump({"done": False, "es": es, "history": history,
                             "n_evals": n_evals, "gen": gen}, f)
            print("[optimize] budget reached at gen {0}, {1} evals — "
                  "checkpointed, rerun to resume".format(gen, n_evals))
            return OptimizeResult(best_params={}, best_value=float("nan"),
                                  n_evals=n_evals, history=history,
                                  complete=False)
        zs = es.ask()
        # rotating CRN: one seed base per GENERATION (shared by every
        # candidate -> paired ranking); rotates unless crn == "fixed".
        seed_base = ospec.seed + (gen * 1009 if ospec.crn == "rotate" else 0)
        vals = []
        for z in zs:
            v, _ = _evaluate(base_spec, ospec, denorm(z),
                             seed_base=seed_base)
            vals.append(v)
            n_evals += 1
            pdict = dict(zip(map(_param_key, ospec.params),
                             map(float, denorm(z))))
            history.append((v, pdict))
            if ospec.bank_path:
                _bank_write(ospec.bank_path,
                            {"kind": "eval", "gen": gen, "eval": n_evals,
                             "seed_base": int(seed_base),
                             "n_repeats": int(max(ospec.n_repeats, 1)),
                             "value": float(v), "params": pdict})
            if ospec.verbose:
                print(f"  gen {gen} eval {n_evals}: {v:.5g} at "
                      f"{[f'{q:.4g}' for q in denorm(z)]}")
        es.tell(zs, vals)
        gen += 1
        if ck:
            with open(ck, "wb") as f:
                pickle.dump({"done": False, "es": es, "history": history,
                             "n_evals": n_evals, "gen": gen}, f)
    zbest = np.clip(es.result.xbest, 0.0, 1.0)
    best = denorm(zbest)
    # fresh-seed audit: seeds from a range the search never touched, so the
    # reported number is an estimate of the true objective, not a noise fit.
    audit_mean, audit_std = _evaluate(
        base_spec, ospec, best, seed_base=ospec.seed + 999_983,
        n_repeats=max(ospec.n_final, 1))
    best_pdict = dict(zip(map(_param_key, ospec.params), map(float, best)))
    if ospec.bank_path:
        _bank_write(ospec.bank_path,
                    {"kind": "audit", "eval": n_evals,
                     "seed_base": int(ospec.seed + 999_983),
                     "n_repeats": int(max(ospec.n_final, 1)),
                     "value": float(audit_mean), "std": float(audit_std),
                     "params": best_pdict})
    result = OptimizeResult(
        best_params=best_pdict,
        best_value=float(es.result.fbest), n_evals=n_evals,
        history=history, audit_value=audit_mean, audit_std=audit_std)
    if ck:
        with open(ck, "wb") as f:
            pickle.dump({"done": True, "result": result}, f)
    return result
