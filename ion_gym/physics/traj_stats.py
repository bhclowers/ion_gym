"""
traj_stats.py — trajectory-ensemble thermal & confinement statistics.

Formalizes the notebook-05 estimators (var_temperature, secular_micro)
into ONE importable implementation so the optimizer, notebooks, and any
gate consume the same math (two writers of the same statistic always
drift apart). Operates on the (results, cols, spec) triple produced by
`sim_build.build_run` + fly and received by `physics.optimize` metrics.

Conventions (all pinned, none assumed):
- Framework units mm / us / Da / V / K / Torr; velocities are converted to
  m/s only inside the temperature estimators.
- STEADY WINDOW: statistics use samples with
  t >= (1 - steady_frac) * t_max_us (default: the last quarter), so the
  launch transient does not pollute steady-state readouts. Every report
  states the window.
- T_var = m * Var(v) / kB per axis — variance of VELOCITY, never of KE
  (Var(KE) maps to no temperature). Drift-robust.
- Secular / micromotion split: per-RF-period window means (secular) vs
  residual (micromotion), period taken from the SLOWEST declared RF group
  (covers every faster drive's cycle). Refused, not guessed, when the
  spec declares no RF group.
- EMITTANCE: pooled-sample RMS emittance over the steady window,
  eps_u = sqrt(<u^2><v_u^2> - <u v_u>^2) about pooled means, in mm*mm/us.
  This pools time samples of a steady-state cloud rather than one time
  slice; reports say "pooled".
- CLEARANCE: distance (mm) from each recorded sample to the nearest metal
  cell, via a Euclidean distance transform of the spec's OWN rasterized
  electrode masks on the anchored grid (`anchored_grid` +
  `electrode_mask` — the identical code path the builder voxelizes with,
  so displayed clearance == solver geometry).
- E2 EXPOSURE: mean over steady samples of e_field^2 [(V/mm)^2]. In the
  collisional regime (nu >> omega) drive power into an ion is
  P ~ q K E^2, so this is the drive-heating surrogate under test. It is
  reported as None (with a stated reason) when the run did not record the
  'e_field' channel; the OBJECTIVE factory, by contrast, refuses.

Fate convention for confinement (stats.py pin): 2 = timeout = still
confined at t_max; 0 = impact. `confined_fate` is a parameter.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

# Universal constants (CODATA), named — not magic numbers.
KB_J = 1.380649e-23            # J/K
KB_EV = 8.617333262e-5         # eV/K
KG_AMU = 1.66053906660e-27     # kg per Da
MMUS_TO_MS = 1.0e3             # mm/us -> m/s


# ------------------------------------------------------------ primitives
def var_temperature(v_mmus, m_kg: float) -> float:
    """Drift-subtracted temperature T = m*Var(v)/kB (v in mm/us)."""
    v = np.asarray(v_mmus, float) * MMUS_TO_MS
    return float(m_kg * np.var(v) / KB_J)


def secular_micro(t_us, v_mmus, period_us: float):
    """Split velocity into per-RF-period window means (secular) and the
    residual (micromotion). Returns (secular, micro) arrays or
    (None, None) when the record is too short to window."""
    t = np.asarray(t_us, float)
    v = np.asarray(v_mmus, float)
    if len(t) < 4 or not period_us or period_us <= 0.0:
        return None, None
    win = np.floor((t - t[0]) / period_us).astype(int)
    sec = np.empty_like(v)
    for w in np.unique(win):
        sel = win == w
        sec[sel] = v[sel].mean()
    return sec, v - sec


def slowest_rf_period_us(spec) -> float:
    """Longest RF period among declared groups (covers all drives).
    Refuses when the spec declares no RF group — a confinement thermal
    split without a drive period is a caller error, not a default."""
    groups = list(spec.geometry.rf_groups or [])
    freqs = [g.frequency_hz for g in groups if g.frequency_hz and
             g.frequency_hz > 0.0]
    if not freqs:
        raise ValueError(
            "spec declares no RF group with a positive frequency; "
            "secular/micromotion windowing needs the drive period. "
            "Pass period_us explicitly if this run is DC-only.")
    return 1.0e6 / min(freqs)


def steady_slice(t_us, t_max_us: float, steady_frac: float):
    """Boolean mask selecting the trailing steady window."""
    t = np.asarray(t_us, float)
    return t >= (1.0 - float(steady_frac)) * float(t_max_us)


# ------------------------------------------------------------ clearance
def metal_distance_field(spec):
    """(dist_mm, xs, ys): Euclidean distance to nearest metal on the
    spec's own anchored grid, from the same electrode_mask rasterization
    the builder uses. Planar (2-D) geometries only."""
    from scipy.ndimage import distance_transform_edt
    from ion_gym.physics.raster2d import (anchored_grid, electrode_mask,
                                          plane_grid_views)
    xs, ys, _anchor = anchored_grid(spec)
    # plane_grid_views, not meshgrid: see raster2d. NOTE the
    # indexing -- this site used indexing="xy", so the views are built on
    # the SWAPPED axis order to keep the same (ny, nx) shape.
    Y, X = plane_grid_views(ys, xs, "clearance map")
    metal = np.zeros(X.shape, bool)
    for el in spec.geometry.electrodes:
        metal |= electrode_mask(el, X, Y)
    if not metal.any():
        raise ValueError("spec rasterizes to zero metal cells — clearance "
                         "against nothing is undefined (check shapes/pitch).")
    h = float(spec.geometry.mm_per_gu)
    dist = distance_transform_edt(~metal, sampling=h)
    return dist, xs, ys


def sample_clearance(results, cols: Sequence[str], spec,
                     steady_frac: float = 0.25):
    """Per-sample distance-to-metal (mm) pooled over the ensemble's
    steady window, by nearest-node lookup on the distance field. Samples
    outside the grid clip to the edge node (boundary clearance is then a
    lower-bound estimate; confined ions never leave the grid)."""
    dist, xs, ys = metal_distance_field(spec)
    from ion_gym.io.records import TrajRecord
    t_max = float(spec.integration.t_max_us)
    h = float(spec.geometry.mm_per_gu)
    out = []
    for r in results:
        rec = TrajRecord(r.traj, cols)
        m = steady_slice(rec["t"], t_max, steady_frac)
        if not m.any():
            continue
        jx = np.clip(np.rint((rec["x"][m] - xs[0]) / h).astype(int),
                     0, len(xs) - 1)
        jy = np.clip(np.rint((rec["y"][m] - ys[0]) / h).astype(int),
                     0, len(ys) - 1)
        out.append(dist[jy, jx])
    if not out:
        raise ValueError("no steady-window samples in any trajectory — "
                         "t_max/steady_frac exclude every record "
                         f"(t_max={t_max} us, steady_frac={steady_frac}).")
    return np.concatenate(out)


# ------------------------------------------------------------ report
def _pooled_axis(results, cols, axis: str, steady_frac: float, t_max: float):
    """(u, v_u, t, ion_index) pooled over the steady window; ion_index is
    the index WITHIN the given results list."""
    from ion_gym.io.records import TrajRecord
    us, vs, ts, ks = [], [], [], []
    for k, r in enumerate(results):
        rec = TrajRecord(r.traj, cols)
        t = rec["t"]
        m = steady_slice(t, t_max, steady_frac)
        if m.any():
            us.append(rec[axis][m])
            vs.append(rec["v" + axis][m])
            ts.append(t[m])
            ks.append(np.full(m.sum(), k))
    if not us:
        return None
    return (np.concatenate(us), np.concatenate(vs),
            np.concatenate(ts), np.concatenate(ks))


def rms_emittance(u_mm, v_mmus) -> float:
    """Pooled-sample RMS emittance about pooled means, mm*mm/us."""
    u = np.asarray(u_mm, float) - np.mean(u_mm)
    v = np.asarray(v_mmus, float) - np.mean(v_mmus)
    det = np.mean(u * u) * np.mean(v * v) - np.mean(u * v) ** 2
    return float(math.sqrt(max(det, 0.0)))


def operating_point(spec) -> str:
    """One-line operating point for report labels (doctrine: every
    certified number carries its tune inline)."""
    gs = ", ".join(
        "{n} {a:g}V0-p {f:g}kHz ph{p:g}".format(
            n=g.name, a=g.amplitude_v, f=g.frequency_hz / 1e3,
            p=g.phase_deg)
        for g in (spec.geometry.rf_groups or []))
    dcs = ", ".join("{0} {1:g}V".format(e.name, e.dc)
                    for e in spec.geometry.electrodes
                    if e.dc not in (None, 0.0))
    mz = "/".join("{0:g}".format(m) for m in (spec.source.mz_list or []))
    return ("RF[{gs}] DC[{dcs}] {p:g} Torr {gas} m/z {mz} "
            "n={n} t_max {tm:g}us dt {dt:g}ns").format(
        gs=gs, dcs=dcs or "-", p=spec.collisions.P_torr,
        gas=spec.collisions.gas, mz=mz or "-", n=spec.source.n_ions,
        tm=spec.integration.t_max_us, dt=spec.integration.dt_ns)


def _species_block(sub, cols, spec, m_da, steady_frac, t_max, p_us,
                   confined_fate, axes, dist_lookup):
    """All metrics for ONE species (the sub-list of results at m/z m_da).
    Thermal fields are None (with 'note') when the species left no
    steady-window samples — an infeasible candidate is REPORTED, not a
    crash and not a silent skip."""
    kinds = [r.summary.get("kind") for r in sub]
    blk = {"mz": float(m_da), "n": len(sub),
           "retention": kinds.count(confined_fate) / max(len(sub), 1),
           "fate_counts": {int(k): kinds.count(k)
                           for k in sorted(set(kinds))}}
    m_kg = float(m_da) * KG_AMU
    excess = []
    for ax in axes:
        pooled = _pooled_axis(sub, cols, ax, steady_frac, t_max)
        if pooled is None:
            blk["T_var_" + ax] = None
            blk["note"] = ("no steady-window samples (species lost before "
                           "the window) — thermal stats undefined")
            continue
        u, v, t, k = pooled
        T_var = var_temperature(v, m_kg)
        sec_all, mic_all = [], []
        for kk in np.unique(k):
            sel = k == kk
            sc, mc = secular_micro(t[sel], v[sel], p_us)
            if sc is not None:
                sec_all.append(sc)
                mic_all.append(mc)
        blk["T_var_" + ax] = T_var
        if sec_all:
            blk["T_sec_" + ax] = var_temperature(
                np.concatenate(sec_all), m_kg)
            blk["T_mic_" + ax] = var_temperature(
                np.concatenate(mic_all), m_kg)
        blk["eps_" + ax] = rms_emittance(u, v)
        blk["sigma_" + ax] = float(np.std(u))
        excess.append(T_var - float(spec.collisions.T_k))
    blk["excess_T_K"] = float(np.mean(excess)) if excess else None

    if "e_field" in cols:
        from ion_gym.io.records import TrajRecord
        vals = []
        for r in sub:
            rec = TrajRecord(r.traj, cols)
            m = steady_slice(rec["t"], t_max, steady_frac)
            if m.any():
                vals.append(rec["e_field"][m] ** 2)
        blk["e2_exposure"] = (float(np.mean(np.concatenate(vals)))
                              if vals else None)
    else:
        blk["e2_exposure"] = None

    clr = dist_lookup(sub)
    if clr is None:
        blk["clearance_p5"] = None
        blk["clearance_p50"] = None
    else:
        blk["clearance_p5"] = float(np.percentile(clr, 5))
        blk["clearance_p50"] = float(np.percentile(clr, 50))
    return blk


def confinement_report(results, cols: Sequence[str], spec, *,
                       steady_frac: float = 0.25,
                       axes: Sequence[str] = ("x", "y"),
                       confined_fate: int = 2,
                       period_us: Optional[float] = None) -> dict:
    """Confinement/thermal metrics, PER SPECIES, for one flown ensemble.

    Structure:
      top level : n, fate_counts, steady_frac, rf_period_us, T_bath_K,
                  operating_point
      species   : {mz: block} — per-species retention, per-axis
                  T_var/T_sec/T_mic [K], eps [mm*mm/us], sigma [mm],
                  e2_exposure [(V/m)^2 planar route], clearance p5/p50
                  [mm], excess_T_K
      aggregates (the MINIMAX objective inputs):
                  retention_worst, clearance_p5_worst,
                  excess_T_worst_K, excess_T_mean_K
    Single-species decks additionally get the species block FLATTENED to
    the top level (back-compatible with earlier callers/renderers).
    A species that died before the steady window reports None thermal
    fields with a note; aggregates then reflect only species with data,
    and retention_worst carries the infeasibility.
    """
    n = len(results)
    if n == 0:
        raise ValueError("empty results — nothing to report on")
    kinds = [r.summary.get("kind") for r in results]
    t_max = float(spec.integration.t_max_us)
    # DC-only specs (no RF group with a positive frequency) are a
    # legitimate configuration: the secular/micromotion SPLIT is
    # undefined for them (no drive period exists), while T_var,
    # emittance, clearance, retention and E2 all remain defined. The
    # report therefore carries rf_period_us=None and omits T_sec/T_mic
    # — a REPORTED absence. Callers that require a period still use
    # slowest_rf_period_us, which refuses by name.
    has_rf = any(g.frequency_hz and g.frequency_hz > 0.0
                 for g in (spec.geometry.rf_groups or []))
    if period_us is not None:
        p_us = period_us
    elif has_rf:
        p_us = slowest_rf_period_us(spec)
    else:
        p_us = None            # split skipped per-species (secular_micro
                               # returns (None, None) on a None period)
    mzs = _mz_per_ion(spec, n)

    # one distance field for the whole spec; per-species lookup closure
    dist, xs, ys = metal_distance_field(spec)
    h = float(spec.geometry.mm_per_gu)
    from ion_gym.io.records import TrajRecord

    def dist_lookup(sub):
        out = []
        for r in sub:
            rec = TrajRecord(r.traj, cols)
            m = steady_slice(rec["t"], t_max, steady_frac)
            if not m.any():
                continue
            jx = np.clip(np.rint((rec["x"][m] - xs[0]) / h).astype(int),
                         0, len(xs) - 1)
            jy = np.clip(np.rint((rec["y"][m] - ys[0]) / h).astype(int),
                         0, len(ys) - 1)
            out.append(dist[jy, jx])
        return np.concatenate(out) if out else None

    rep: dict = {"n": n,
                 "fate_counts": {int(k): kinds.count(k)
                                 for k in sorted(set(kinds))},
                 "steady_frac": steady_frac, "rf_period_us": p_us,
                 "T_bath_K": float(spec.collisions.T_k),
                 "operating_point": operating_point(spec)}

    species = {}
    for m_da in sorted(set(mzs)):
        sub = [r for r, m in zip(results, mzs) if m == m_da]
        species[m_da] = _species_block(sub, cols, spec, m_da, steady_frac,
                                       t_max, p_us, confined_fate, axes,
                                       dist_lookup)
    rep["species"] = species

    rets = [b["retention"] for b in species.values()]
    clrs = [b["clearance_p5"] for b in species.values()
            if b["clearance_p5"] is not None]
    excs = [b["excess_T_K"] for b in species.values()
            if b["excess_T_K"] is not None]
    rep["retention_worst"] = float(min(rets))
    rep["clearance_p5_worst"] = float(min(clrs)) if clrs else None
    rep["excess_T_worst_K"] = float(max(excs)) if excs else None
    rep["excess_T_mean_K"] = float(np.mean(excs)) if excs else None
    # back-compat aliases (single- and multi-species)
    rep["retention"] = rep["retention_worst"]
    rep["clearance_p5"] = rep["clearance_p5_worst"]
    rep["excess_T_K"] = rep["excess_T_worst_K"]
    if len(species) == 1:
        only = next(iter(species.values()))
        for k, v in only.items():
            if k not in ("mz", "n", "fate_counts", "retention"):
                rep.setdefault(k, v)
    return rep


def _mz_per_ion(spec, n: int):
    from ion_gym.physics.stats import mz_of_results
    mzs = mz_of_results(spec, n)
    if any(m is None for m in mzs):
        raise ValueError("spec.source.mz_list is empty — temperatures need "
                         "a mass; declare m/z in the spec.")
    return mzs


# ------------------------------------------------------------ objective
def make_confinement_objective(*, retention_min: float = 0.95,
                               d_min_mm: float = 0.5,
                               steady_frac: float = 0.25,
                               axes: Sequence[str] = ("x", "y"),
                               confined_fate: int = 2) -> Callable:
    """Scalar MINIMIZED objective for physics.optimize (callable metric),
    MINIMAX over declared species: the WORST species must clear each
    tier. Lexicographic ladder, magnitude-separated:
      tier 1  worst retention < retention_min : 1e6 * (1 + shortfall)
      tier 2  worst clearance_p5 < d_min_mm   : 1e3 * (1 + deficit/d_min)
      tier 3  feasible                        : excess_T_worst_K
    Refuses when 'e_field' was not recorded (exposure stays reportable
    next to every scored candidate).
    """
    def objective(results, cols, spec) -> float:
        if "e_field" not in cols:
            raise ValueError("confinement objective requires the 'e_field' "
                             "channel; add it to record_channels.")
        rep = confinement_report(results, cols, spec,
                                 steady_frac=steady_frac, axes=axes,
                                 confined_fate=confined_fate)
        short = retention_min - rep["retention_worst"]
        if short > 0.0:
            return 1.0e6 * (1.0 + short)
        clr = rep["clearance_p5_worst"]
        deficit = d_min_mm - (clr if clr is not None else 0.0)
        if deficit > 0.0:
            return 1.0e3 * (1.0 + deficit / d_min_mm)
        return rep["excess_T_worst_K"]
    objective.__name__ = ("confinement_worst_excessT_ret{0:g}_d{1:g}mm"
                          .format(retention_min, d_min_mm))
    return objective


def slope_temperature(ke_ev, *, fit_from_pctl: float = 20.0,
                      n_bins: int = 80) -> float:
    """T_slope: temperature from the SLOPE of the log-count energy tail
    of a 1-D Maxwell-Boltzmann KE distribution. Extracted verbatim from
    the trapping notebook so the notebook and GUI share ONE
    estimator. fit_from_pctl is the tail percentile above which the
    log-linear fit is taken — it is a PARAMETER, not a buried default,
    because for a non-thermal distribution (T_var != T_slope) the fitted
    value depends on where the tail starts, so the operating point must
    be stated. Returns NaN when the tail has < 2 populated bins or the
    fitted slope is non-negative (no thermal tail to read)."""
    ke = np.asarray(ke_ev, float)
    ke_pos = ke[ke > 0]
    if ke_pos.size == 0:
        return float("nan")
    # ROBUST BINNING (measured on an 8-rail SLIM run): RF
    # micromotion near electrodes puts a handful of 10-100 eV samples
    # into an otherwise sub-0.1 eV thermal ensemble. Binning the FULL
    # range collapsed the entire thermal bulk into bin 0 and handed the
    # tail fit a scatter of count-1 outlier bins -> T_slope = nan. The
    # histogram now spans [0, p99.5] of the positive energies — a
    # DECLARED 0.5% clip (stated here and annotated on the figure), not
    # a silent one; T_var/T_mean remain moment-based over ALL samples.
    hi = float(np.percentile(ke_pos, 99.5))
    ke_fit = ke_pos[ke_pos <= hi]
    if ke_fit.size < 10:
        return float("nan")
    counts, edges = np.histogram(ke_fit, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # The fit start is the fit_from_pctl percentile OF THE POPULATED
    # ENERGY AXIS (bin centers that actually hold ions), not of the raw
    # per-ion energies. The reason: raw-energy percentiles
    # all collapse into the first bin for a peaked KE distribution, so
    # the threshold moved by less than a bin width across 10-55% and the
    # parameter had NO effect (a knob that does not change the answer is
    # a defect). Anchoring to the populated-bin span makes the percentile
    # sweep the tail as intended: 0% fits from the first populated bin,
    # 100% from the last.
    pop = counts > 0
    if pop.sum() < 3:
        return float("nan")
    lo_e = centers[pop].min()
    hi_e = centers[pop].max()
    e_start = lo_e + (hi_e - lo_e) * (float(fit_from_pctl) / 100.0)
    tail = pop & (centers >= e_start)
    if tail.sum() < 2:
        return float("nan")
    # PREFACTOR-CORRECTED FIT. The 1-D Maxwell-
    # Boltzmann energy density is p(E) = (pi kT)^-1/2 * E^-1/2 *
    # exp(-E/kT). The E^-1/2 term CURVES log(count) vs E, so a plain
    # log-linear slope reads T differently depending on where the fit
    # starts (percentile-dependent even for a THERMAL cloud). Fitting
    # log(count * sqrt(E)) removes the prefactor exactly: for a thermal
    # distribution that is a straight line of slope -1/(kT) at EVERY
    # start point, so residual percentile-dependence is now REAL non-
    # thermal structure, not a fit artifact.
    c = counts[tail].astype(float)
    e = centers[tail]
    y = np.log(c * np.sqrt(e))
    # COUNT-WEIGHTED fit. On a log axis, a bin of N
    # counts has variance ~1/N (Poisson), so sparse far-tail bins (N~1)
    # are noisy AND biased low (counts can't go negative). Weighting
    # each bin by its count makes the well-populated bins set the slope
    # and stops the sparse tail from dragging T_slope upward — the
    # residual drift the prefactor correction alone left behind.
    w = np.sqrt(c)                       # sqrt(N): the Poisson weight
    A = np.vstack([e * w, w]).T
    (slope, _icpt), *_ = np.linalg.lstsq(A, y * w, rcond=None)
    if slope >= 0:
        return float("nan")
    return float(-1.0 / (slope * KB_EV))


def two_estimator_report(v_by_axis, ke_by_axis, m_kg, *,
                         fit_from_pctl: float = 20.0, n_bins: int = 80,
                         t_by_axis=None, rf_period_us=None,
                         bath_K=None):
    """Per-axis two-estimator temperatures: T_var
    (drift-subtracted velocity variance), T_mean (drift-SENSITIVE mean
    KE), and T_slope (energy-tail fit). v_by_axis/ke_by_axis are dicts
    {'x'|'y'|'z': 1-D ensemble arrays}; m_kg is the ion mass. Axes with
    no motion (2-D solve) yield None. Returns {axis: {T_var, T_mean,
    T_slope, v_mean, n, drift}} — the SAME numbers notebook 05 prints,
    from ONE code path. Physics: T_var and T_mean AGREE on a drift-free
    axis (that agreement is the physical result); they SPLIT only under
    bulk drift. T_slope reads the tail shape independently, so
    T_var != T_slope flags a non-thermal (hot-tailed) distribution."""
    out = {}
    for ax in ("x", "y", "z"):
        v = np.asarray(v_by_axis.get(ax, []), float)
        ke = np.asarray(ke_by_axis.get(ax, []), float)
        if v.size == 0 or np.allclose(v, 0.0):
            out[ax] = None                 # no motion on this axis
            continue
        ke_pos = ke[ke > 0]
        if ke_pos.size == 0:
            out[ax] = None
            continue
        T_mean = 2.0 * ke_pos.mean() / KB_EV
        T_var = m_kg * np.var(v * MMUS_TO_MS) / KB_J
        T_slope = slope_temperature(ke, fit_from_pctl=fit_from_pctl,
                                    n_bins=n_bins)
        drift = abs(T_var - T_mean) > 0.05 * max(T_var, 1.0)
        # secular/micromotion split (needs the time axis + RF period):
        # T_var = T_sec + T_mic in energy, so a sub-bath T_var is
        # accounted for by T_mic (driven wiggle), not sub-bath cooling.
        T_sec = T_mic = None
        if t_by_axis is not None and rf_period_us:
            t = np.asarray(t_by_axis.get(ax, []), float)
            if t.size == v.size and t.size >= 4:
                sc, mc = secular_micro(t, v, rf_period_us)
                if sc is not None:
                    T_sec = m_kg * np.var(sc * MMUS_TO_MS) / KB_J
                    T_mic = m_kg * np.var(mc * MMUS_TO_MS) / KB_J
        excess = None if bath_K is None else (T_var - float(bath_K))
        out[ax] = dict(T_var=T_var, T_mean=T_mean, T_slope=T_slope,
                       T_sec=T_sec, T_mic=T_mic, excess_K=excess,
                       v_mean=float(v.mean()), n=int(ke_pos.size),
                       drift=bool(drift))
    return out


def survival_report(results, cols, spec, *, confined_fate=2,
                    n_windows=6, axes=("x", "y")):
    """Time-resolved survival for ms-scale screening.

    The pass/fail retention at one horizon cannot tell a SLOW-impactting
    design (stable to 400 us, gone by 3 ms) from a stable one — that is
    exactly what hid the 3 ms wall losses. This measures HOW ions are
    lost over the flight: it bins each ion's loss TIME (when it hits
    metal) into n_windows across [0, t_max] and returns a per-window
    surviving fraction plus a fitted loss RATE (fractional loss per ms).

    Returns {t_max_us, survival_curve [(t_us, frac_alive)],
             loss_per_ms, final_retention, half_life_ms|None}. loss_per_ms
    is the slope of -ln(frac_alive) vs t (ms): 0 = perfectly stable,
    larger = faster bleed. half_life_ms is when frac hits 0.5 (None if it
    never does). Species are pooled to the WORST (min surviving frac per
    window) so a single leaking species is not averaged away.
    """
    t_max = float(spec.integration.t_max_us)
    edges = np.linspace(0.0, t_max, n_windows + 1)
    # per species, get each ion's loss time (NaN if it survived)
    by_species = {}
    for r in results:
        mz = float(r.summary.get("mz", 0.0))
        fate = r.summary.get("kind")
        t_loss = r.summary.get("t_loss_us", None)
        if t_loss is None and fate != confined_fate:
            # lost but no recorded loss time -> treat as lost at t_max
            t_loss = t_max
        by_species.setdefault(mz, []).append(
            (fate == confined_fate, t_loss))

    # worst-species surviving fraction in each window
    curve = []
    for j in range(1, n_windows + 1):
        t_hi = edges[j]
        worst = 1.0
        for mz, ions in by_species.items():
            n = len(ions)
            if n == 0:
                continue
            alive = sum(1 for (surv, tl) in ions
                        if surv or (tl is not None and tl > t_hi))
            worst = min(worst, alive / n)
        curve.append((float(t_hi), float(worst)))

    final = curve[-1][1] if curve else 1.0
    # loss rate: slope of -ln(frac) vs t(ms), over windows with frac>0
    ts_ms = np.array([t for t, f in curve]) / 1000.0
    fr = np.array([f for t, f in curve])
    good = fr > 1e-6
    loss_per_ms = 0.0
    if good.sum() >= 2 and (fr[good] < 1.0 - 1e-9).any():
        y = -np.log(np.clip(fr[good], 1e-6, 1.0))
        A = np.vstack([ts_ms[good], np.ones(good.sum())]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0], None
        loss_per_ms = float(max(0.0, slope[0]))
    half_life_ms = None
    if loss_per_ms > 1e-9:
        hl = np.log(2.0) / loss_per_ms
        half_life_ms = float(hl)
    return dict(t_max_us=t_max, survival_curve=curve,
                loss_per_ms=loss_per_ms, final_retention=final,
                half_life_ms=half_life_ms)


def make_survival_objective(*, target_hold_ms=3.0, d_min_mm=0.5,
                            excess_T_max_K=None, steady_frac=0.25,
                            axes=("x", "y"), confined_fate=2,
                            n_windows=6, w_temp=0.05):
    """Graded ms-scale objective: rewards LONG-TIME
    survival continuously (no hard retention cliff) and keeps TEMPERATURE
    in the score, so slow-impact ranks strictly worse than stable and
    strictly better than fast-leak — the distinction the old pass/fail
    penalty destroyed.

    Minimized score, magnitude-separated tiers:
      tier 1  clearance_p5 < d_min : 1e3 * (1 + deficit/d_min)   [hard]
      tier 2  feasible-clearance   : loss_cost + w_temp * excess_T
        loss_cost = 100 * loss_per_ms * target_hold_ms
          (fraction of the deck lost over the target hold, x100 — a
           design losing 1%/ms over a 3 ms goal scores 3; a stable one 0)
        excess_T = worst-species excess temperature (K), so a cold
          design is preferred among equally-stable ones. w_temp scales
          K into the same range as loss_cost (default: 20 K ~ 1 loss pt).
      excess_T_max_K (optional) adds a hard tier if the cloud runs hotter
      than a stated ceiling (1e2 * over/ceiling), so temperature can be a
      CONSTRAINT, not just a tie-breaker, when you set it.

    Every scored candidate carries loss_per_ms, half_life_ms,
    final_retention and excess_T inline (attached to the objective call's
    last_report) so the audit reads survival AND temperature, never one
    collapsed number.
    """
    def objective(results, cols, spec) -> float:
        if "e_field" not in cols:
            raise ValueError("survival objective requires 'e_field' in "
                             "record_channels.")
        rep = confinement_report(results, cols, spec,
                                 steady_frac=steady_frac, axes=axes,
                                 confined_fate=confined_fate)
        surv = survival_report(results, cols, spec,
                               confined_fate=confined_fate,
                               n_windows=n_windows, axes=axes)
        objective.last_report = {**rep, **surv}   # inline provenance
        clr = rep["clearance_p5_worst"]
        deficit = d_min_mm - (clr if clr is not None else 0.0)
        if deficit > 0.0:
            return 1.0e3 * (1.0 + deficit / d_min_mm)
        excessT = rep["excess_T_worst_K"]
        if excess_T_max_K is not None and excessT is not None \
                and excessT > excess_T_max_K:
            return 1.0e2 * (excessT / excess_T_max_K)
        loss_cost = 100.0 * surv["loss_per_ms"] * target_hold_ms
        temp_cost = w_temp * (excessT if excessT is not None else 0.0)
        return float(loss_cost + temp_cost)
    objective.last_report = None
    objective.__name__ = ("survival_hold{0:g}ms_d{1:g}"
                          .format(target_hold_ms, d_min_mm))
    return objective


def ride_height_report(results, cols, spec, *, axis="y", confined_fate=2):
    """Where does the cloud RIDE relative to the RF carpets?
    T_mic is set by WHERE ions sit in the pseudopotential,
    so this is the observable that discriminates 'mid-channel cold' from
    'surface-riding hot' — the temperature columns only show the symptom.

    Standoff = distance from the nearest carpet surface, per recorded
    sample of surviving ions. Carpet surfaces are read from the spec's
    metal: bottom surface = max y of bottom_* shapes, top surface = min y
    of top_* shapes. If the spec has no such electrodes this REFUSES with
    a diagnostic (no silent default — a wrong surface would make every
    number wrong).

    Returns {surface_lo_mm, surface_hi_mm, gap_mm, standoff_mean_mm,
    standoff_p5_mm, standoff_p50_mm, frac_within_1mm, frac_within_half_mm,
    mid_fraction} where mid_fraction is the fraction of samples in the
    middle half of the gap (the 'cold zone'). Positions are pooled over
    surviving ions' recorded frames.
    """
    lo_tops, hi_bots = [], []
    for el in spec.geometry.electrodes:
        nm = getattr(el, "name", "") or ""
        for sh in el.shapes:
            d = sh.to_dict()
            if d.get("type") != "rect":
                continue
            y0, y1 = d["y_mm"], d["y_mm"] + d["height_mm"]
            if nm.startswith("bottom"):
                lo_tops.append(y1)
            elif nm.startswith("top"):
                hi_bots.append(y0)
    if not lo_tops or not hi_bots:
        raise ValueError(
            "ride_height_report: cannot locate carpet surfaces — spec has "
            "no bottom_*/top_* rect electrodes (found {0} bottom, {1} top). "
            "Pass a cell_board-style spec or add surface detection for this "
            "geometry.".format(len(lo_tops), len(hi_bots)))
    surf_lo = max(lo_tops)          # top face of the bottom board metal
    surf_hi = min(hi_bots)          # bottom face of the top board metal
    gap = surf_hi - surf_lo
    if gap <= 0:
        raise ValueError("ride_height_report: degenerate gap "
                         "({0:.3f} mm)".format(gap))
    from ion_gym.io.records import TrajRecord
    ys = []
    for r in results:
        if r.summary.get("kind") != confined_fate:
            continue
        if axis in cols:
            ys.append(np.asarray(TrajRecord(r.traj, cols)[axis], float))
        else:
            # legacy planar layout without a schema for this axis:
            # (t, x, y, ...) -> y at column 2. Kept as the ONE documented
            # positional fallback (the layout comment IS the map).
            ys.append(np.asarray(r.traj[:, 2], float))
    if not ys:
        return dict(surface_lo_mm=surf_lo, surface_hi_mm=surf_hi,
                    gap_mm=gap, n_samples=0, standoff_mean_mm=None,
                    standoff_p5_mm=None, standoff_p50_mm=None,
                    frac_within_1mm=None, frac_within_half_mm=None,
                    mid_fraction=None,
                    note="no surviving ions to measure")
    y = np.concatenate(ys)
    y = y[(y > surf_lo) & (y < surf_hi)]        # interior samples only
    if y.size == 0:
        return dict(surface_lo_mm=surf_lo, surface_hi_mm=surf_hi,
                    gap_mm=gap, n_samples=0, standoff_mean_mm=None,
                    standoff_p5_mm=None, standoff_p50_mm=None,
                    frac_within_1mm=None, frac_within_half_mm=None,
                    mid_fraction=None,
                    note="no interior samples (all outside the gap?)")
    standoff = np.minimum(y - surf_lo, surf_hi - y)
    mid_lo, mid_hi = surf_lo + 0.25 * gap, surf_hi - 0.25 * gap
    return dict(
        surface_lo_mm=float(surf_lo), surface_hi_mm=float(surf_hi),
        gap_mm=float(gap), n_samples=int(y.size),
        standoff_mean_mm=float(np.mean(standoff)),
        standoff_p5_mm=float(np.percentile(standoff, 5)),
        standoff_p50_mm=float(np.percentile(standoff, 50)),
        frac_within_1mm=float(np.mean(standoff < 1.0)),
        frac_within_half_mm=float(np.mean(standoff < 0.5)),
        mid_fraction=float(np.mean((y > mid_lo) & (y < mid_hi))))
