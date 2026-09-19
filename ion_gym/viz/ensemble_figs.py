"""ion_gym.viz.ensemble_figs — ensemble-statistics renderers.

Owner of figures computed ACROSS an ensemble of flights (per-ion
scalars and channel statistics), as distinct from field/PE/scene
renders (viz_core, pe_view) and single-path renders. Added 2026-09-09
for the rf_arrival_time campaign; both renderers satisfy the framework
contract: config-driven (channel names and windows are arguments, no
magic numbers), display equals solver input (values plotted are the
recorded channels verbatim), figures are RETURNED (no show/savefig
side effects), no import side effects, and every title carries the
operating point the caller supplies.
"""
from __future__ import annotations

import numpy as np

KB = 1.380649e-23          # J/K (CODATA, same constant collision3d uses)
AMU_KG = 1.66053906660e-27
MM_US_TO_M_S = 1e3         # recorded velocities are mm/us


def _fig(nrows, ncols, figsize):
    # no backend selection — display is the caller's decision (same
    # fix as viz_core.deck_plane_preview, 2026-09-10); headless
    # environments auto-select Agg themselves
    import matplotlib.pyplot as plt
    return plt.subplots(nrows, ncols, figsize=figsize)


def path_accounting_figure(results, cols, *, operating_point,
                           path_channel="path_mm", axial_channel="x",
                           tof_key="tof", figsize=(9.5, 4.0)):
    """The rejected-hypothesis figure (manuscript C2): per-ion integrated
    path length vs net axial displacement vs arrival time. If the extra
    path were axial ('multiple paths'), excess path fraction would
    correlate with arrival time; a flat cloud shows the excess is
    transverse and TIME-NEUTRAL.

    results: iterable of ensemble_driver.IonResult (traj + summary).
    cols: the fly_fn column-name list (path/axial channels read from it).
    operating_point: REQUIRED string rendered into the title.
    Returns (fig, stats_dict); raises with the missing channel by name.
    """
    for ch in (path_channel, axial_channel):
        if ch not in cols:
            raise ValueError(
                f"path_accounting_figure: channel {ch!r} not in cols "
                f"{list(cols)} — request it in "
                f"spec.integration.record_channels")
    ip, ia = cols.index(path_channel), cols.index(axial_channel)
    path, axial, tof = [], [], []
    for r in results:
        if r.traj is None or not len(r.traj):
            raise ValueError(
                f"path_accounting_figure: ion {r.index} carries no "
                f"trajectory (keep_traj off?) — this figure needs the "
                f"per-ion record")
        path.append(float(r.traj[-1, ip]))
        axial.append(abs(float(r.traj[-1, ia]) - float(r.traj[0, ia])))
        tof.append(float(r.summary[tof_key]))
    path, axial, tof = map(np.asarray, (path, axial, tof))
    if np.any(axial <= 0):
        raise ValueError(
            "path_accounting_figure: non-positive axial displacement — "
            "wrong axial_channel for this device orientation")
    excess = path / axial - 1.0
    r_corr = (float(np.corrcoef(excess, tof)[0, 1])
              if len(tof) > 2 else float("nan"))
    fig, (ax1, ax2) = _fig(1, 2, figsize)
    ax1.scatter(100.0 * excess, tof, s=12, alpha=0.7)
    ax1.set_xlabel("excess path over axial displacement [%]")
    ax1.set_ylabel(f"arrival time [{tof_key} units]")
    ax1.set_title(f"time-neutrality test (r = {r_corr:+.3f})")
    ax2.hist(100.0 * excess, bins=max(10, len(excess) // 20))
    ax2.set_xlabel("excess path [%]")
    ax2.set_ylabel("ions")
    ax2.set_title("micromotion path excess")
    fig.suptitle(f"path accounting — {operating_point}")
    fig.tight_layout()
    return fig, dict(excess_mean=float(excess.mean()),
                     excess_max=float(excess.max()),
                     corr_excess_tof=r_corr, n=len(excess))


def heating_anisotropy_figure(results, cols, *, mass_Da, operating_point,
                              components=("vx", "vy", "vz"),
                              component_labels=None,
                              steady_frac=0.5, figsize=(9.5, 4.0)):
    """Component-resolved effective temperatures (manuscript C3/anisotropy):
    T_i = m <var(v_i)> / kB over the LATE fraction of each flight
    (steady_frac names the window). RF heating concentrates in the
    confinement direction(s); the axial component staying cold is the
    figure's point — 'field heating is not uniform in xyz'.

    components: velocity channel names per route (r-z: e.g.
    ('vx','vy') with labels ('axial','radial'); 3-D: vx/vy/vz).
    Returns (fig, {label: T_kelvin}); refuses missing channels by name.
    """
    idx = []
    for ch in components:
        if ch not in cols:
            raise ValueError(
                f"heating_anisotropy_figure: channel {ch!r} not in cols "
                f"{list(cols)} — this route records {list(cols)}")
        idx.append(cols.index(ch))
    labels = list(component_labels or components)
    if len(labels) != len(components):
        raise ValueError("component_labels length must match components")
    if not (0.0 < steady_frac <= 1.0):
        raise ValueError(f"steady_frac {steady_frac} outside (0, 1]")
    m_kg = float(mass_Da) * AMU_KG
    temps = {}
    for lab, j in zip(labels, idx):
        vs = []
        for r in results:
            if r.traj is None or not len(r.traj):
                raise ValueError(
                    f"heating_anisotropy_figure: ion {r.index} carries "
                    f"no trajectory — per-sample velocities required")
            n0 = int(len(r.traj) * (1.0 - steady_frac))
            vs.append(r.traj[n0:, j] * MM_US_TO_M_S)
        v = np.concatenate(vs)
        temps[lab] = float(m_kg * v.var() / KB)
    fig, ax = _fig(1, 1, figsize)
    ax.bar(list(temps), list(temps.values()))
    ax.set_ylabel("effective temperature [K]")
    ax.set_title(
        f"component-resolved heating (late {steady_frac:.0%} window) — "
        f"{operating_point}")
    for k, (lab, t) in enumerate(temps.items()):
        ax.text(k, t, f"{t:.0f} K", ha="center", va="bottom")
    fig.tight_layout()
    return fig, temps


def atd_figure(tofs_by_label, *, operating_point, bins=30,
               tof_unit="us", figsize=(9.5, 4.2)):
    """THE arrival-time-distribution figure: overlaid step histograms,
    one per label (e.g. per RF Vpp rung), with centroid/width/skew per
    label in the legend. tofs_by_label: {label: 1-D array of arrival
    times}. Returns (fig, stats {label: dict}); stats carry centroid,
    width (sd), skew, kappa3 (unnormalized third central moment,
    tof_unit^3 — additive under independent convolution),
    tail_frac_2sd, n."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    stats = {}
    allt = np.concatenate([np.asarray(v, float)
                           for v in tofs_by_label.values()])
    if not len(allt):
        raise ValueError("atd_figure: no arrival times supplied")
    if isinstance(bins, float):
        # a FLOAT bins is a bin WIDTH in tof units (PI knob 2026-09-10)
        if bins <= 0:
            raise ValueError(f"atd_figure: bin width {bins} <= 0")
        lo, hi = float(allt.min()), float(allt.max())
        edges = np.arange(lo, hi + bins, bins)
    else:
        edges = np.histogram_bin_edges(allt, bins=bins)
    for lab, t in tofs_by_label.items():
        t = np.asarray(t, float)
        if len(t) < 3:
            raise ValueError(
                f"atd_figure: label {lab!r} has {len(t)} arrivals — "
                f"too few for a distribution")
        mu, sd = float(t.mean()), float(t.std())
        sk = float(((t - mu) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
        # kappa3: UNNORMALIZED third central moment [tof_unit^3]. Under
        # independent injection + transport, kappa2 and kappa3 ADD,
        # while normalized skew DILUTES as the gate widens (kappa2
        # grows under it) — so kappa3, not skew, is the observable
        # that untangles gate width from transport tailing (2026-09-11).
        k3 = float(((t - mu) ** 3).mean())
        tail = float((t > mu + 2 * sd).mean())
        stats[lab] = dict(centroid=mu, width=sd, skew=sk, kappa3=k3,
                          tail_frac_2sd=tail, n=len(t))
        ax.hist(t, bins=edges, histtype="step", lw=1.8,
                label=f"{lab}: mu={mu:.1f}, sd={sd:.2f}, "
                      f"skew={sk:+.2f}, n={len(t)}")
    ax.set_xlabel(f"arrival time [{tof_unit}]")
    ax.set_ylabel("ions")
    ax.legend(fontsize=8)
    ax.set_title(f"ATD — {operating_point}")
    fig.tight_layout()
    return fig, stats


def atd_shift_figure(stats_by_x, *, x_label, operating_point,
                     figsize=(9.5, 3.8)):
    """The prize summary: ATD centroid (left axis) and 2-sigma tail
    fraction (right axis) vs the swept variable (e.g. RF Vpp).
    stats_by_x: {x_value: stats dict from atd_figure}. Returns fig."""
    import matplotlib.pyplot as plt
    xs = sorted(stats_by_x)
    cen = [stats_by_x[x]["centroid"] for x in xs]
    tail = [stats_by_x[x]["tail_frac_2sd"] for x in xs]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, cen, "o-", label="centroid")
    ax.set_xlabel(x_label)
    ax.set_ylabel("ATD centroid [us]")
    ax2 = ax.twinx()
    ax2.plot(xs, tail, "s--", color="tab:red", label="tail frac (>mu+2sd)")
    ax2.set_ylabel("tail fraction", color="tab:red")
    ax.set_title(f"arrival-time shift vs {x_label} — {operating_point}")
    fig.tight_layout()
    return fig


def trajectory_panel(results, cols, *, axial_channel, transverse_channel,
                     operating_point, time_channel="t", n_show=3,
                     figsize=(9.5, 4.2)):
    """A few ions' paths: transverse vs axial (left, axial HORIZONTAL
    per doctrine) and axial position vs time (right). time_channel
    defaults to "t" — the BASE_CHANNELS name (sim_spec authority); it
    was briefly hardcoded as a guessed "t_us" and crashed the campaign
    notebook (2026-09-10) — channel names come from the spec authority
    or the caller, never from memory."""
    import matplotlib.pyplot as plt
    for ch in (axial_channel, transverse_channel, time_channel):
        if ch not in cols:
            raise ValueError(
                f"trajectory_panel: channel {ch!r} not in cols "
                f"{list(cols)}")
    ia, ir, it = (cols.index(axial_channel),
                  cols.index(transverse_channel),
                  cols.index(time_channel))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    shown = 0
    for r in results:
        if r.traj is None or not len(r.traj):
            continue
        ax1.plot(r.traj[:, ia], r.traj[:, ir], lw=0.8)
        ax2.plot(r.traj[:, it], r.traj[:, ia], lw=0.8)
        shown += 1
        if shown >= n_show:
            break
    if not shown:
        raise ValueError("trajectory_panel: no ions carried trajectories")
    ax1.set_xlabel(f"{axial_channel} [mm] (axial)")
    ax1.set_ylabel(f"{transverse_channel} [mm]")
    ax1.set_title(f"paths ({shown} ions)")
    ax2.set_xlabel(f"{time_channel} [us]")
    ax2.set_ylabel(f"{axial_channel} [mm] (axial)")
    ax2.set_title("axial position vs time")
    fig.suptitle(f"trajectories — {operating_point}")
    fig.tight_layout()
    return fig


def tw_slip_metrics(tof_us, x_net_mm, *, f_hz, lambda_mm):
    """Slip accounting for TW transport: waves APPLIED during the
    transit = f * tof; waves RIDDEN = net advance / wavelength; slips
    (rollovers) = applied - ridden; slip factor = ridden / applied
    (1.0 = surfing, 0 = fully parked). All inputs are the recorded
    values — nothing inferred."""
    tof_s = np.asarray(tof_us, float) * 1e-6
    applied = f_hz * tof_s
    ridden = np.asarray(x_net_mm, float) / float(lambda_mm)
    slips = applied - ridden
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(applied > 0, ridden / applied, np.nan)
    return dict(waves_applied=applied, waves_ridden=ridden,
                slips=slips, slip_factor=factor)


def slip_persistence_scatter(slips_first, slips_second, slips_total,
                             tof, *, operating_point,
                             figsize=(9.5, 4.0)):
    """Render the persistence panels from per-ion ARRAYS — the shape
    slip_persistence_figure computes live and the persistence banks
    store — so a banked run re-renders through exactly the code path
    that drew it. Returns (fig, stats) with rho_split, fano, n and
    the arrays; rho/fano recomputed from the arrays themselves
    (displayed equals stored)."""
    import matplotlib.pyplot as plt
    h1 = np.asarray(slips_first, float)
    h2 = np.asarray(slips_second, float)
    tot = np.asarray(slips_total, float)
    tofs = np.asarray(tof, float)
    if not (len(h1) == len(h2) == len(tot) == len(tofs)):
        raise ValueError(
            f"slip_persistence_scatter: array lengths differ "
            f"({len(h1)}/{len(h2)}/{len(tot)}/{len(tofs)})")
    if len(h1) < 3:
        raise ValueError(f"slip_persistence_scatter: {len(h1)} ions "
                         f"— too few for a correlation")
    s1, s2 = h1.std(), h2.std()
    rho = (float(np.corrcoef(h1, h2)[0, 1])
           if s1 > 0 and s2 > 0 else float("nan"))
    mtot = float(tot.mean())
    fano = (float(tot.var() / mtot) if mtot > 0
            else float("nan"))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
    axL.scatter(h1, h2, s=12, alpha=0.5, lw=0)
    lo = float(min(h1.min(), h2.min()))
    hi = float(max(h1.max(), h2.max()))
    axL.plot([lo, hi], [lo, hi], color="0.6", lw=0.8)  # equal rate
    axL.set_xlabel("slips, first half")
    axL.set_ylabel("slips, second half")
    axL.annotate(f"rho = {rho:+.3f}", xy=(0.04, 0.92),
                 xycoords="axes fraction", fontsize=9)
    axR.scatter(tot, tofs, s=12, alpha=0.5, lw=0)
    axR.set_xlabel("total slips")
    axR.set_ylabel("arrival time [us]")
    axR.annotate(f"Fano = {fano:.2f}\nn = {len(tot)}",
                 xy=(0.04, 0.86), xycoords="axes fraction", fontsize=9)
    fig.suptitle(f"slip persistence — {operating_point}", fontsize=10)
    fig.tight_layout()
    stats = dict(rho_split=rho, fano=fano, n=len(h1),
                 slips_first=h1, slips_second=h2, slips_total=tot,
                 tof=tofs)
    return fig, stats


def slip_persistence_figure(results, cols, *, f_hz, lambda_mm,
                            operating_point, axial_channel="x",
                            wrap_channel=None, wrap_len_mm=None,
                            tof_key="tof", settle_waves=0,
                            figsize=(9.5, 4.0)):
    """Per-ion slip persistence: does an ion that slips more than its
    peers in the FIRST half of its transit (split at its own
    mid-flight TIME) keep doing so in the second half?

    rho_split ~ 0 with Fano <= 1: slip is resampled luck — differences
    accumulate as WIDTH (CLT, Gaussian). rho_split -> 1 with
    Fano >> 1: persistent per-ion rates — differences compound
    linearly with path and the 1/v mapping turns them into a RIGHT
    TAIL, hardest for slow species.

    Slips per window use the tw_slip_metrics identity
    slips = f*dt - dx/lambda on the UNWRAPPED axial coordinate. On a
    wrapped transporter axis you MUST pass wrap_channel (cumulative
    pass count, e.g. "wrap_passes") and wrap_len_mm
    (accept_mm - emit_mm): unwrapped = stored + passes*wrap_len
    (tracer3d contract). Naming a wrap_channel absent from cols is
    REFUSED — a wrapped axis silently treated as straight undercounts
    every window's advance.

    settle_waves skips the first settle_waves wave periods of each
    flight before splitting — the capture transient (ions born
    near-rest locking on) otherwise loads h1-only variance and
    biases rho. Returns (fig, stats): rho_split, fano, n, n_skipped, skipped
    (indices, printed — a skipped ion is a reported ion), and the
    per-ion arrays slips_first/slips_second/slips_total/tof.
    Left panel: half vs half with the equal-rate y=x line, rho
    labeled on-panel. Right panel: total slips vs arrival, Fano
    labeled on-panel."""
    # (No local matplotlib import, unlike the other figure functions in
    # this module: this one draws nothing itself — it reduces the
    # ensemble to per-ion arrays and hands them to
    # slip_persistence_scatter, which owns the figure and its own
    # import.)
    it = cols.index("t")
    ix = cols.index(axial_channel)
    iw = None
    if wrap_channel is not None:
        if wrap_channel not in cols:
            raise ValueError(
                f"slip_persistence_figure: wrap_channel "
                f"{wrap_channel!r} not in cols {cols!r} — record it, "
                f"or omit wrap_channel only for a genuinely straight "
                f"axis")
        if wrap_len_mm is None or float(wrap_len_mm) <= 0:
            raise ValueError(
                f"slip_persistence_figure: wrap_channel given but "
                f"wrap_len_mm={wrap_len_mm!r} — pass accept_mm - "
                f"emit_mm")
        iw = cols.index(wrap_channel)
    lam = float(lambda_mm)
    h1, h2, tot, tofs, skipped = [], [], [], [], []
    for r in results:
        if r.traj is None or len(r.traj) < 4:
            skipped.append(r.index)
            continue
        t = np.asarray(r.traj[:, it], float)      # ABSOLUTE us
        x = np.asarray(r.traj[:, ix], float)
        if iw is not None:
            x = x + np.asarray(r.traj[:, iw], float) * float(wrap_len_mm)
        if t[-1] <= t[0]:
            skipped.append(r.index)
            continue
        ts = t[0] + settle_waves / f_hz * 1e6
        if ts >= t[-1]:
            skipped.append(r.index)
            continue
        xs = float(np.interp(ts, t, x))
        tm = 0.5 * (ts + t[-1])
        xm = float(np.interp(tm, t, x))           # x already unwrapped
        def _slips(t0, t1, x0, x1):
            return f_hz * (t1 - t0) * 1e-6 - (x1 - x0) / lam
        h1.append(_slips(ts, tm, xs, xm))
        h2.append(_slips(tm, t[-1], xm, x[-1]))
        tot.append(_slips(ts, t[-1], xs, x[-1]))
        tofs.append(float(r.summary[tof_key]))
    if skipped:
        print(f"slip_persistence_figure: skipped {len(skipped)} "
              f"ion(s) with no usable trajectory: {skipped[:20]}"
              + (" ..." if len(skipped) > 20 else ""))
    if len(h1) < 3:
        raise ValueError(
            f"slip_persistence_figure: only {len(h1)} usable ions "
            f"({len(skipped)} skipped) — too few for a correlation")
    fig, stats = slip_persistence_scatter(
        np.asarray(h1), np.asarray(h2), np.asarray(tot),
        np.asarray(tofs), operating_point=operating_point,
        figsize=figsize)
    stats.update(n_skipped=len(skipped), skipped=list(skipped),
                 settle_waves=int(settle_waves))
    return fig, stats


def station_crossings(results, cols, station, *, first_only=True):
    """Per-ion crossing times of a plane station, applied POST-HOC to
    recorded trajectories — the StationSpec contract for 'record' and
    'detect' kinds ('impact_plane' fates come from the kernel and need no
    helper). station: a StationSpec or a dict with axis, pos_mm and an
    optional window {axis: [lo, hi]} over the transverse axes.

    Crossing time and transverse point are linearly interpolated
    between the bracketing records, and the window is tested at the
    interpolated point — so on a serpentine, one x-plane serves many
    legs and the window picks WHICH leg's crossing counts. Returns
    (t_us, crossed, skipped): t_us is nan where no in-window crossing
    occurred (crossed False there); skipped lists ions with no usable
    trajectory, printed — a skipped ion is a reported ion. With
    first_only=False, t_us is instead a list of per-ion arrays of
    every in-window crossing (re-crossings visible, not collapsed)."""
    def _get(o, k, default=None):
        if isinstance(o, dict):
            return o.get(k, default)
        return getattr(o, k, default)
    axis = str(_get(station, "axis", "x"))
    pos = float(_get(station, "pos_mm"))
    win = _get(station, "window") or {}
    for a in (axis, "t"):
        if a not in cols:
            raise ValueError(f"station_crossings: channel {a!r} not in "
                             f"cols {list(cols)!r}")
    for a in win:
        if a not in cols:
            raise ValueError(f"station_crossings: window axis {a!r} "
                             f"not in cols {list(cols)!r} — record it")
        if a == axis:
            raise ValueError(f"station_crossings: window on the "
                             f"plane's own axis {a!r} is meaningless")
    ia, it = cols.index(axis), cols.index("t")
    iw = {a: cols.index(a) for a in win}
    out_t, out_m, out_all, skipped = [], [], [], []
    for r in results:
        if r.traj is None or len(r.traj) < 2:
            skipped.append(r.index)
            out_t.append(float("nan")); out_m.append(False)
            out_all.append(np.empty(0))
            continue
        c = np.asarray(r.traj[:, ia], float)
        t = np.asarray(r.traj[:, it], float)
        d = c - pos
        idx = np.flatnonzero(d[:-1] * d[1:] <= 0.0)
        hits = []
        for j in idx:
            den = c[j + 1] - c[j]
            f = 0.0 if den == 0.0 else (pos - c[j]) / den
            ok = True
            for a, (lo, hi) in win.items():
                v = r.traj[j, iw[a]] + f * (r.traj[j + 1, iw[a]]
                                            - r.traj[j, iw[a]])
                if not (float(lo) <= v <= float(hi)):
                    ok = False
                    break
            if ok:
                hits.append(t[j] + f * (t[j + 1] - t[j]))
                if first_only:
                    break
        out_all.append(np.asarray(hits))
        out_t.append(hits[0] if hits else float("nan"))
        out_m.append(bool(hits))
    if skipped:
        print(f"station_crossings: skipped {len(skipped)} ion(s) with "
              f"no usable trajectory: {skipped[:20]}"
              + (" ..." if len(skipped) > 20 else ""))
    if first_only:
        return np.asarray(out_t), np.asarray(out_m), skipped
    return out_all, np.asarray(out_m), skipped


def trajs_from_results(results, cols, *,
                       channels=("t", "x", "wrap_passes", "y", "z")):
    """Adapt driver IonResults to the traj-dict shape the trajectory
    analyses (slip_memory_figure, station_slip_figure) consume — the
    same shape ion_gym.io.results_bank.trajs_from_bank returns, so
    live results and banked runs share one analysis path. Every
    requested channel must be in cols (REFUSED otherwise);
    'wrap_passes' maps to key 'wrap'. Returns (trajs, tof_us,
    skipped); skipped ions are reported."""
    missing = [c for c in channels if c not in cols]
    if missing:
        raise ValueError(
            f"trajs_from_results: channel(s) {missing} not in cols "
            f"{list(cols)!r} — record them, or drop them from "
            f"`channels`")
    idx = {c: cols.index(c) for c in channels}
    keymap = {"wrap_passes": "wrap"}
    trajs, tofs, skipped = [], [], []
    for r in results:
        if r.traj is None or len(r.traj) < 2:
            skipped.append(int(r.index))
            continue
        trajs.append({keymap.get(c, c):
                      np.asarray(r.traj[:, idx[c]], float)
                      for c in channels})
        tofs.append(float(r.summary["tof"]))
    if skipped:
        print(f"trajs_from_results: skipped {len(skipped)} ion(s) "
              f"with no usable trajectory: {skipped[:20]}"
              + (" ..." if len(skipped) > 20 else ""))
    return trajs, np.asarray(tofs), skipped


def _unwrapped_x(tr, wrap_len_mm):
    """Unwrapped axial coordinate of one traj dict, with the same
    refusal semantics everywhere: a 'wrap' channel present without
    wrap_len_mm — or wrap_len_mm given without a 'wrap' channel — is
    an ERROR, because either mismatch silently corrupts every
    window's advance."""
    if "wrap" in tr:
        if wrap_len_mm is None or float(wrap_len_mm) <= 0:
            raise ValueError(
                "trajectory carries a 'wrap' channel but "
                f"wrap_len_mm={wrap_len_mm!r} — pass accept_mm - "
                "emit_mm")
        return tr["x"] + tr["wrap"] * float(wrap_len_mm)
    if wrap_len_mm is not None:
        raise ValueError(
            "wrap_len_mm given but trajectory has no 'wrap' channel "
            "— straight axis and wrap length cannot both be true")
    return tr["x"]


def _window_slips(trajs, *, f_hz, lambda_mm, wrap_len_mm,
                  n_windows, settle_waves):
    """(W, kept, skipped, window_us): W is (n_ions x n_windows) slip
    counts in equal-TIME windows after the settle skip."""
    lam = float(lambda_mm)
    rows, kept, skipped, wus = [], [], [], []
    for i, tr in enumerate(trajs):
        t = np.asarray(tr["t"], float)
        if len(t) < n_windows + 1 or t[-1] <= t[0]:
            skipped.append(i)
            continue
        x = np.asarray(_unwrapped_x(tr, wrap_len_mm), float)
        ts = t[0] + settle_waves / f_hz * 1e6
        if ts >= t[-1]:
            skipped.append(i)
            continue
        edges = np.linspace(ts, t[-1], n_windows + 1)
        xe = np.interp(edges, t, x)
        rows.append(f_hz * np.diff(edges) * 1e-6 - np.diff(xe) / lam)
        kept.append(i)
        wus.append((t[-1] - ts) / n_windows)
    if skipped:
        print(f"window slips: skipped {len(skipped)} ion(s) (too few "
              f"records or settle exceeds flight): {skipped[:20]}"
              + (" ..." if len(skipped) > 20 else ""))
    if len(rows) < 3:
        raise ValueError(f"window slips: only {len(rows)} usable "
                         f"ions ({len(skipped)} skipped) — too few")
    return (np.vstack(rows), kept, skipped, float(np.mean(wus)))


def slip_memory_figure(trajs, *, f_hz, lambda_mm, wrap_len_mm=None,
                       n_windows=8, settle_waves=0, operating_point,
                       figsize=(11.5, 3.2)):
    """How long does 'slipping more than your peers' last? Splits
    each flight into n_windows equal-time windows (after skipping
    settle_waves wave periods) and correlates per-ion slip counts
    between windows.

    Panels: (1) mean inter-window correlation vs lag — flat and high
    = frozen per-ion rates; decaying = finite memory time (the 1/e
    crossing is annotated when it exists); ~0 everywhere = resampled
    slip. (2) mean slips per window — a raised first window IS the
    capture transient, seen directly. (3) var/mean per window
    (dispersion; Poisson = 1 line drawn).

    n_windows=2 reduces to the split-half rho of
    slip_persistence_figure, so banked runs need only this function.
    Returns (fig, stats): corr_by_lag, lag_us, rho_adjacent,
    rho_split, tau_us (None when correlation never falls below 1/e
    of adjacent — reported, not invented), mean_by_window,
    dispersion_by_window, window_us, n, n_skipped, skipped."""
    import matplotlib.pyplot as plt
    if n_windows < 2:
        raise ValueError(f"slip_memory_figure: n_windows={n_windows} "
                         f"— need >= 2")
    W, kept, skipped, window_us = _window_slips(
        trajs, f_hz=f_hz, lambda_mm=lambda_mm,
        wrap_len_mm=wrap_len_mm, n_windows=n_windows,
        settle_waves=settle_waves)
    n, m = W.shape
    sd = W.std(axis=0)
    zero_var = np.flatnonzero(sd == 0)
    if len(zero_var):
        print(f"slip_memory_figure: window(s) {zero_var.tolist()} "
              f"have ZERO count variance — correlations with them "
              f"are undefined (nan), reported not hidden")
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(W.T)
    corr_by_lag = np.array([np.nanmean(np.diag(C, k))
                            for k in range(1, m)])
    lag_us = np.arange(1, m) * window_us
    half = m // 2
    a, b = W[:, :half].sum(axis=1), W[:, half:].sum(axis=1)
    rho_split = (float(np.corrcoef(a, b)[0, 1])
                 if a.std() > 0 and b.std() > 0 else float("nan"))
    rho_adj = float(corr_by_lag[0])
    tau_us = None
    if np.isfinite(rho_adj) and rho_adj > 0:
        below = np.flatnonzero(corr_by_lag < rho_adj / np.e)
        if len(below):
            tau_us = float(lag_us[below[0]])
    mean_w = W.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        disp_w = W.var(axis=0) / np.where(mean_w > 0, mean_w,
                                                  np.nan)
    fig, (axC, axM, axD) = plt.subplots(1, 3, figsize=figsize)
    axC.plot(lag_us / 1e3, corr_by_lag, "o-", ms=4)
    axC.axhline(0.0, color="0.75", lw=0.8)
    axC.set_xlabel("lag [ms]")
    axC.set_ylabel("correlation (dimensionless)")
    axC.set_title("inter-window slip correlation", fontsize=9)
    axC.annotate((f"adjacent {rho_adj:+.2f}\n"
                  f"split-half {rho_split:+.2f}\n"
                  + (f"tau ~ {tau_us/1e3:.2f} ms" if tau_us is not None
                     else "no 1/e crossing in span")),
                 xy=(0.05, 0.72), xycoords="axes fraction", fontsize=8)
    wi = np.arange(1, m + 1)
    axM.plot(wi, mean_w, "o-", ms=4)
    axM.set_ylabel("mean slips per window (waves)")
    axM.set_xlabel("window # (index)")
    axM.set_title("mean slips / window", fontsize=9)
    axD.plot(wi, disp_w, "o-", ms=4)
    axD.axhline(1.0, color="0.75", lw=0.8)
    axD.annotate(" Poisson (1)", xy=(m, 1.0), fontsize=7,
                 color="0.5", va="bottom")
    axD.set_ylabel("var/mean of slips (dimensionless)")
    axD.set_xlabel("window # (index)")
    axD.set_title("var/mean / window", fontsize=9)
    fig.suptitle(f"slip memory — {operating_point}", fontsize=10)
    fig.tight_layout()
    stats = dict(corr_by_lag=corr_by_lag, lag_us=lag_us,
                 rho_adjacent=rho_adj, rho_split=rho_split,
                 tau_us=tau_us, mean_by_window=mean_w,
                 dispersion_by_window=disp_w, window_us=window_us,
                 settle_waves=int(settle_waves), n=n,
                 n_skipped=len(skipped), skipped=list(skipped))
    return fig, stats


def station_slip_figure(trajs, *, f_hz, lambda_mm, wrap_len_mm=None,
                        settle_waves=0, operating_point,
                        figsize=(9.5, 4.0)):
    """Does transverse station carry slip rate? Per ion: total slips
    (after the settle skip) against (a) lateral station |mean(y) -
    ensemble median| and (b) across-gap RMS z. Pearson r annotated
    on each panel; a flat scatter is a direct, per-ion refutation of
    the transverse-heterogeneity mechanism in this geometry — no
    inference from exponents needed. Requires 'y' and 'z' channels
    in every traj (REFUSED otherwise). Returns (fig, stats) with the
    per-ion arrays and both r values."""
    import matplotlib.pyplot as plt
    for need in ("y", "z"):
        bad = [i for i, tr in enumerate(trajs) if need not in tr]
        if bad:
            raise ValueError(
                f"station_slip_figure: {len(bad)} traj(s) lack the "
                f"{need!r} channel (first: {bad[:5]}) — bank it or "
                f"use trajs that carry it")
    W, kept, skipped, _ = _window_slips(
        trajs, f_hz=f_hz, lambda_mm=lambda_mm,
        wrap_len_mm=wrap_len_mm, n_windows=2,
        settle_waves=settle_waves)
    tot = W.sum(axis=1)
    ybar = np.array([float(np.mean(trajs[i]["y"])) for i in kept])
    zrms = np.array([float(np.sqrt(np.mean(
        np.asarray(trajs[i]["z"], float) ** 2))) for i in kept])
    ydev = np.abs(ybar - np.median(ybar))
    def _r(u, v):
        return (float(np.corrcoef(u, v)[0, 1])
                if u.std() > 0 and v.std() > 0 else float("nan"))
    r_y, r_z = _r(ydev, tot), _r(zrms, tot)
    fig, (axY, axZ) = plt.subplots(1, 2, figsize=figsize)
    axY.scatter(ydev, tot, s=12, alpha=0.5, lw=0)
    axY.set_xlabel("lateral station |mean y - median| [mm]")
    axY.set_ylabel("total slips")
    axY.annotate(f"r = {r_y:+.3f}", xy=(0.05, 0.92),
                 xycoords="axes fraction", fontsize=9)
    axZ.scatter(zrms, tot, s=12, alpha=0.5, lw=0)
    axZ.set_xlabel("across-gap RMS z [mm]")
    axZ.annotate(f"r = {r_z:+.3f}", xy=(0.05, 0.92),
                 xycoords="axes fraction", fontsize=9)
    fig.suptitle(f"station vs slip — {operating_point}", fontsize=10)
    fig.tight_layout()
    stats = dict(r_lateral=r_y, r_gap=r_z, slips_total=tot,
                 ydev_mm=ydev, ybar_mm=ybar, zrms_mm=zrms,
                 settle_waves=int(settle_waves), n=len(tot),
                 n_skipped=len(skipped), skipped=list(skipped))
    return fig, stats


def cumulant_scaling_figure(runs, *, operating_point,
                            figsize=(13.5, 3.0)):
    """Path-length scaling of the arrival cumulants — the analytic
    discriminator between slip models. runs: {L_mm: dict with
    'tof_us' (per-ion arrivals, us) and 'slips' (per-ion total slip
    counts), optionally 'rho' (split-half correlation)}; needs >= 3
    path lengths.

    Exact identity: T = L/(f*lambda) + n/f, so kappa_m(T) =
    kappa_m(n)/f^m — arrival cumulants ARE slip-count cumulants.
    Short-memory slip: kappa2 and kappa3 extensive in L (log-log
    slopes 1, 1), Fano ~ const, rho ~ 0 — skew dies as 1/sqrt(L).
    Frozen per-ion rates: kappa2 ~ L^2, kappa3 ~ L^3, Fano ~ L,
    rho -> 1 — skew constant in L. Panels are log-log with the
    fitted exponent annotated on-panel and both model slopes drawn
    as labeled guides. kappa3's sign is annotated; a sign that
    CHANGES across L refuses the kappa3 exponent fit (reported) —
    fitting magnitudes of a sign-changing cumulant is not a slope.

    The |kappa3| panel carries the sampling noise floor: bars are
    +/- one SE (sqrt(6/n)*sd^3, the Gaussian-null kappa3 SE), and
    the exponent is fitted ONLY on points clearing 2 SE — fewer than
    3 such points refuses the fit (reported): a slope through
    statistical zeros is not a slope. The dispersion panel is
    var(n)/mean(n) with the Poisson = 1 line drawn. The plate-height
    panel is chromatography's H = L/N with N = T^2/kappa2: a normal
    column holds H constant; frozen heterogeneity grows H ~ L.
    X-ticks sit at the actual ladder L values (no minor-tick
    collisions). Returns (fig, table): table maps each L to
    dict(kappa2, kappa3, sigma_kappa3, fano, mean_slips, n, plates,
    H_mm[, rho]) plus fitted exponents (kappa2, kappa3, fano, H,
    kappa3_note) under key 'exponents'."""
    import matplotlib.pyplot as plt
    if len(runs) < 3:
        raise ValueError(f"cumulant_scaling_figure: {len(runs)} path "
                         f"lengths — need >= 3 to fit a slope")
    Ls = np.array(sorted(runs), float)
    k2l, k3l, fanol, rhol, mul, nl, table = ([], [], [], [], [],
                                             [], {})
    for L in Ls:
        d = runs[float(L)] if float(L) in runs else runs[L]
        t = np.asarray(d["tof_us"], float)
        s = np.asarray(d["slips"], float)
        if len(t) < 3:
            raise ValueError(f"cumulant_scaling_figure: L={L:g} mm "
                             f"has {len(t)} arrivals — too few")
        mu = t.mean()
        _k2 = float(t.var())
        _k3 = float(((t - mu) ** 3).mean())
        ms = float(s.mean())
        _fa = float(s.var() / ms) if ms > 0 else float("nan")
        _r = d.get("rho")
        k2l.append(_k2); k3l.append(_k3); fanol.append(_fa)
        rhol.append(_r); mul.append(float(mu)); nl.append(len(t))
        row = dict(kappa2=_k2, kappa3=_k3, fano=_fa, mean_slips=ms,
                   n=len(t))
        if _r is not None:
            row["rho"] = float(_r)
        table[float(L)] = row
    k2 = np.array(k2l); k3 = np.array(k3l); fano = np.array(fanol)
    mus = np.array(mul); ns = np.array(nl, float)
    sig3 = np.sqrt(6.0 / ns) * k2 ** 1.5   # kappa3 SE (Gaussian null)
    N_pl = mus ** 2 / k2
    H_mm = Ls / N_pl
    for _j, _L in enumerate(Ls):
        table[float(_L)].update(sigma_kappa3=float(sig3[_j]),
                                plates=float(N_pl[_j]),
                                H_mm=float(H_mm[_j]))

    def _fit(y):
        m = np.isfinite(y) & (y > 0)
        if m.sum() < 3:
            return None
        return float(np.polyfit(np.log(Ls[m]), np.log(y[m]), 1)[0])

    e2 = _fit(k2)
    sig_mask = np.abs(k3) >= 2.0 * sig3
    _signs = set(np.sign(k3[sig_mask & (k3 != 0)]).tolist())
    k3_mixed = len(_signs) > 1
    k3_note = None
    if sig_mask.sum() < 3:
        e3 = None
        k3_note = (f"below noise floor ({int(sig_mask.sum())}/"
                   f"{len(k3)} points clear 2 SE) — fit refused")
        print("cumulant_scaling_figure: kappa3 " + k3_note)
    elif k3_mixed:
        e3 = None
        k3_note = "sign mixed among significant points — fit refused"
        print("cumulant_scaling_figure: kappa3 changes SIGN across L "
              "among significant points — exponent fit refused (a "
              "magnitude fit of a sign-changing cumulant is not a "
              "slope)")
    else:
        e3 = _fit(np.where(sig_mask, np.abs(k3), np.nan))
    e_f = _fit(fano)
    have_rho = any(r is not None for r in rhol)
    ncols = 5 if have_rho else 4
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    def _panel(ax, y, name, guides):
        m = np.isfinite(y) & (y > 0)
        ax.loglog(Ls[m], y[m], "o-", ms=4)
        if m.any():
            _j = int(m.sum()) // 2          # mid anchor: guides stay
            y0, L0 = y[m][_j], Ls[m][_j]    # near the data band
            for g, lab in guides:
                ref = y0 * (Ls / L0) ** g
                ax.loglog(Ls, ref, color="0.75", lw=0.8)
                ax.annotate(lab, xy=(Ls[-1], ref[-1]), fontsize=7,
                            color="0.5", va="center")
        ax.set_xlim(Ls[0] * 0.8, Ls[-1] * 1.25)
        ax.set_xlabel("path L [mm]")
        ax.set_title(name, fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(Ls)
        ax.set_xticklabels([f"{v:.0f}" for v in Ls],
                           rotation=35, ha="right")
        ax.minorticks_off()

    _panel(axes[0], k2, "kappa2(T) [us2]",
           [(1, " slope 1"), (2, " slope 2")])
    if e2 is not None:
        axes[0].annotate(f"fit {e2:.2f}", xy=(0.05, 0.90),
                         xycoords="axes fraction", fontsize=9)
    _panel(axes[1], np.abs(k3), "|kappa3(T)| [us3]  (bars: 1 SE)",
           [(1, " slope 1"), (3, " slope 3")])
    _pos = np.abs(k3)[np.abs(k3) > 0]
    _floor = (min(_pos.min(), sig3.min()) * 0.05 if len(_pos)
              else sig3.min() * 0.05)
    axes[1].vlines(Ls, np.maximum(np.abs(k3) - sig3, _floor),
                   np.abs(k3) + sig3, color="0.6", lw=0.9)
    axes[1].set_xlim(Ls[0] * 0.8, Ls[-1] * 1.25)   # vlines on a log
    # axis inflate autoscale limits (mpl LineCollection); re-pin
    axes[1].set_ylim(_floor * 0.5,
                     float((np.abs(k3) + sig3).max()) * 4.0)
    _k3lab = (f"fit {e3:.2f} (sign {'+' if k3[-1] >= 0 else '-'})"
              if e3 is not None else
              ("below noise floor\n— fit refused" if "floor" in
               (k3_note or "") else "sign mixed\n— fit refused"))
    axes[1].annotate(_k3lab, xy=(0.05, 0.90),
                     xycoords="axes fraction", fontsize=8)
    _panel(axes[2], fano, "var(n)/mean(n)  [slip dispersion]",
           [(1, " slope 1")])
    axes[2].axhline(1.0, color="0.75", lw=0.8)
    axes[2].annotate(" Poisson (1)", xy=(Ls[-1], 1.0), fontsize=7,
                     color="0.5", va="bottom")
    if e_f is not None:
        axes[2].annotate(f"fit {e_f:.2f}", xy=(0.05, 0.90),
                         xycoords="axes fraction", fontsize=9)
    _panel(axes[3], H_mm, "plate height H = L/N [mm]",
           [(0, " const"), (1, " slope 1")])
    eH = _fit(H_mm)
    if eH is not None:
        axes[3].annotate(f"fit {eH:.2f}", xy=(0.05, 0.90),
                         xycoords="axes fraction", fontsize=9)
    if have_rho:
        rr = np.array([float(r) if r is not None else np.nan
                       for r in rhol])
        axes[4].semilogx(Ls, rr, "o-", ms=4)
        axes[4].set_ylim(-0.1, 1.05)
        axes[4].set_xlabel("path L [mm]")
        axes[4].set_title("split-half rho", fontsize=9)
        axes[4].set_xticks(Ls)
        axes[4].set_xticklabels([f"{v:.0f}" for v in Ls],
                                rotation=35, ha="right")
        axes[4].minorticks_off()
    fig.suptitle(f"cumulant scaling — {operating_point}", fontsize=10)
    fig.tight_layout()
    table["exponents"] = dict(kappa2=e2, kappa3=e3, fano=e_f,
                              H=eH, kappa3_note=k3_note)
    return fig, table


def atd_model_figure(measured_by_label, model_by_label, *, operating_point,
                     bins=60, tof_unit="us", center=False, log_y=False,
                     normalize_models=True, y_label=None,
                     figsize=(9.5, 4.2)):
    """Data-vs-model ATD figure: measured arrivals as DENSITY step
    histograms, model predictions as density curves, on shared axes.
    measured_by_label: {label: 1-D arrivals}; model_by_label:
    {label: (t, density)} — a label may appear in either or both.
    Each model curve is normalized to unit area (trapezoid) so data
    and model share the density axis. center=True shifts every label
    to its own centroid (sample mean; curve first moment), so shapes
    with very different means compare on one axis — the axis is then
    labelled as a centroid offset. log_y=True puts density on a log
    axis (the tail view); zero-count bins simply have no step there.
    normalize_models=False draws each model curve AT THE AREA IT WAS
    GIVEN (for mixture components whose area is their population
    weight); legend stats always come from the unit-area shape. Returns (fig, stats): stats as in
    atd_figure for measured labels, plus centroid/width/skew/kappa3
    computed from the curve for model labels (key 'model' True)."""
    import matplotlib.pyplot as plt
    if not measured_by_label and not model_by_label:
        raise ValueError("atd_model_figure: nothing to draw")
    fig, ax = plt.subplots(figsize=figsize)
    stats = {}
    for lab, t in measured_by_label.items():
        t = np.asarray(t, float)
        if len(t) < 3:
            raise ValueError(f"atd_model_figure: label {lab!r} has "
                             f"{len(t)} arrivals — too few")
        mu, sd = float(t.mean()), float(t.std())
        k3 = float(((t - mu) ** 3).mean())
        stats[lab] = dict(centroid=mu, width=sd, skew=k3 / sd ** 3,
                          kappa3=k3, tail_frac_2sd=float((t > mu + 2 * sd).mean()),
                          n=len(t), model=False)
        ax.hist(t - (mu if center else 0.0), bins=bins, density=True,
                histtype="step", lw=1.8,
                label=f"{lab}: mu={mu:.0f}, sd={sd:.1f}, "
                      f"skew={k3 / sd ** 3:+.3f} (n={len(t)})")
    for lab, (t, dens) in model_by_label.items():
        t = np.asarray(t, float)
        dens = np.asarray(dens, float)
        if t.ndim != 1 or t.shape != dens.shape or len(t) < 8:
            raise ValueError(f"atd_model_figure: model {lab!r} needs "
                             f"matching 1-D (t, density) with >= 8 points")
        if np.any(np.diff(t) <= 0) or np.any(dens < 0):
            raise ValueError(f"atd_model_figure: model {lab!r} t must "
                             f"increase and density be >= 0")
        area = float(np.trapezoid(dens, t))
        if area <= 0:
            raise ValueError(f"atd_model_figure: model {lab!r} has "
                             f"non-positive area {area}")
        shape = dens / area
        mu = float(np.trapezoid(t * shape, t))
        var = float(np.trapezoid((t - mu) ** 2 * shape, t))
        sd = float(np.sqrt(var))
        k3 = float(np.trapezoid((t - mu) ** 3 * shape, t))
        stats[lab] = dict(centroid=mu, width=sd, skew=k3 / sd ** 3,
                          kappa3=k3, area=area, n=None, model=True)
        dens = shape if normalize_models else dens
        ax.plot(t - (mu if center else 0.0), dens, lw=1.6, alpha=0.9,
                label=f"{lab}: mu={mu:.0f}, sd={sd:.1f}, "
                      f"skew={k3 / sd ** 3:+.3f} (model)")
    ax.set_xlabel(f"arrival time - centroid [{tof_unit}]" if center
                  else f"arrival time [{tof_unit}]")
    ax.set_ylabel(y_label or f"density [1/{tof_unit}]")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    import textwrap
    ax.set_title("\n".join(textwrap.wrap(
        f"ATD vs model — {operating_point}", width=88)), fontsize=10)
    fig.tight_layout()
    return fig, stats


def cumulant_projection_figure(projection, *, x_label, operating_point,
                               measured=None, figsize=(13.5, 3.0)):
    """Additive-cumulant projection panels: sd, kappa3, and normalized
    skew (kappa3/sd^3) vs a count or length variable. projection:
    {x: dict(sd, kappa3)} with >= 2 points, drawn as lines — the
    analytic prediction. measured (optional): {x: dict(sd, kappa3[,
    sd_se, kappa3_se])} drawn as points with error bars where SEs are
    given. Returns fig."""
    import matplotlib.pyplot as plt
    if len(projection) < 2:
        raise ValueError(f"cumulant_projection_figure: {len(projection)} "
                         f"projection points — need >= 2 for a line")
    xs = np.array(sorted(projection), float)
    sd = np.array([projection[x]["sd"] for x in xs])
    k3 = np.array([projection[x]["kappa3"] for x in xs])
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = ((axes[0], sd, "sd [us]"), (axes[1], k3, "kappa3 [us^3]"),
              (axes[2], k3 / sd ** 3, "skew = kappa3 / sd^3"))
    for ax, y, ylabel in panels:
        ax.plot(xs, y, "-", lw=1.8, label="additive cumulants")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
    if measured:
        mx = np.array(sorted(measured), float)
        msd = np.array([measured[x]["sd"] for x in mx])
        mk3 = np.array([measured[x]["kappa3"] for x in mx])
        msd_se = np.array([measured[x].get("sd_se", 0.0) for x in mx])
        mk3_se = np.array([measured[x].get("kappa3_se", 0.0) for x in mx])
        skew = mk3 / msd ** 3
        skew_se = np.abs(skew) * np.sqrt(
            np.divide(mk3_se, mk3, out=np.zeros_like(mk3_se),
                      where=mk3 != 0) ** 2
            + (3 * np.divide(msd_se, msd, out=np.zeros_like(msd_se),
                             where=msd != 0)) ** 2)
        for ax, y, yerr in ((axes[0], msd, msd_se), (axes[1], mk3, mk3_se),
                            (axes[2], skew, skew_se)):
            ax.errorbar(mx, y, yerr=np.where(yerr > 0, yerr, np.nan),
                        fmt="o", ms=5, capsize=3, label="measured")
    for ax in axes:
        ax.legend(fontsize=8)
    import textwrap
    fig.suptitle("\n".join(textwrap.wrap(
        f"cumulant projection — {operating_point}", width=130)), fontsize=10)
    fig.tight_layout()
    return fig


def trend_panels_figure(panels, *, x_label, operating_point, figsize=None):
    """Measured trends against one control variable, one panel per
    quantity, one or more series per panel. panels is an ordered
    {panel_title: spec} where spec has:
      series: {label: dict(x, y[, yerr])} — yerr absolute, either one
              array (symmetric) or a (lo, hi) pair of BOUNDS converted
              to bar lengths here;
      y_label (optional), log_y (optional bool), reference_y (optional
      horizontal guide, e.g. 1.0 for a ratio panel).
    Error bars draw only where given. Returns fig; no file side
    effects."""
    import matplotlib.pyplot as plt
    import textwrap
    if not panels:
        raise ValueError("trend_panels_figure: no panels")
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=figsize or (4.5 * n_panels, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (title, spec) in zip(axes, panels.items()):
        series = spec.get("series")
        if not series:
            raise ValueError(f"trend_panels_figure: panel {title!r} has "
                             f"no series")
        for label, data in series.items():
            x = np.asarray(data["x"], float)
            y = np.asarray(data["y"], float)
            if x.shape != y.shape:
                raise ValueError(f"trend_panels_figure: {title!r}/{label!r} "
                                 f"x {x.shape} vs y {y.shape}")
            yerr = data.get("yerr")
            if yerr is not None:
                yerr = np.asarray(yerr, float)
                if yerr.ndim == 2:
                    if yerr.shape != (2, len(x)):
                        raise ValueError(
                            f"trend_panels_figure: {title!r}/{label!r} "
                            f"bounds shape {yerr.shape}, need (2, n)")
                    yerr = np.vstack([y - yerr[0], yerr[1] - y])
                    if (yerr < 0).any():
                        raise ValueError(
                            f"trend_panels_figure: {title!r}/{label!r} "
                            f"bounds do not bracket y")
            ax.errorbar(x, y, yerr=yerr, fmt="o-", ms=4.5, lw=1.4,
                        capsize=3, label=label)
        if spec.get("reference_y") is not None:
            ax.axhline(spec["reference_y"], color="0.4", lw=0.9, ls="--")
        if spec.get("log_y"):
            ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(spec.get("y_label", title))
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("\n".join(textwrap.wrap(operating_point, width=130)),
                 fontsize=10)
    fig.tight_layout()
    return fig


def ecdf_compare_figure(measured_by_label, model_by_label, *,
                        operating_point, survival=False, log_y=False,
                        x_label="arrival time [us]", figsize=(9.5, 4.2)):
    """Cumulative comparison of sample sets and model curves with NO
    binning: measured sets draw as exact ECDF staircases (every event a
    step), models as cumulative curves integrated from (t, density)
    input. survival=True plots 1 - F (the tail view; pairs naturally
    with log_y). Sparse data belong here rather than in a histogram.
    Returns fig."""
    import matplotlib.pyplot as plt
    import textwrap
    if not measured_by_label and not model_by_label:
        raise ValueError("ecdf_compare_figure: nothing to draw")
    fig, ax = plt.subplots(figsize=figsize)
    for lab, samples in measured_by_label.items():
        samples = np.sort(np.asarray(samples, float))
        if len(samples) < 2:
            raise ValueError(f"ecdf_compare_figure: {lab!r} has "
                             f"{len(samples)} samples")
        fraction = np.arange(1, len(samples) + 1) / len(samples)
        y = 1.0 - fraction if survival else fraction
        ax.step(samples, y, where="post", lw=1.6,
                label=f"{lab} (n={len(samples)})")
    for lab, (t, dens) in model_by_label.items():
        t = np.asarray(t, float)
        dens = np.asarray(dens, float)
        if t.shape != dens.shape or np.any(np.diff(t) <= 0):
            raise ValueError(f"ecdf_compare_figure: model {lab!r} needs "
                             f"matching arrays with increasing t")
        cumulative = np.concatenate(
            [[0.0], np.cumsum(np.diff(t) * 0.5 * (dens[1:] + dens[:-1]))])
        if cumulative[-1] <= 0:
            raise ValueError(f"ecdf_compare_figure: model {lab!r} has "
                             f"non-positive area")
        cumulative = cumulative / cumulative[-1]
        ax.plot(t, 1.0 - cumulative if survival else cumulative, lw=1.6,
                alpha=0.9, label=f"{lab} (model)")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("1 - F(t)" if survival else "F(t)")
    ax.legend(fontsize=8)
    ax.set_title("\n".join(textwrap.wrap(
        f"{'survival' if survival else 'ECDF'} — {operating_point}",
        width=88)), fontsize=10)
    fig.tight_layout()
    return fig
