"""
stats.py — impact / fate statistics for ion_gym ensembles.

One source of truth for ensemble reductions: the GUI Card (sim_app) and any
gate script render from the same `compute_stats()`, so they can never
disagree.

Conventions (pinned against v51 code, not assumed):
- fate codes (THE table: physics.ion_envelope.FATE_NAME):
      0 = impact / exit (electrode)     1 = boundary exit (left domain)
      2 = timeout (still flying)       3 = bounding plane
      4 = transporter max_passes       5 = station impact plane
      6 = station detect (absorbed by a detector patch)
- summary keys: kind, tof [us], x_end, y_end, z_end, r_end [mm], n_col
- "transmitted" is EXAMPLE-DEPENDENT: the r-z einzel detector is a fate-3
  bound; the quad's success case is fate-2 drift-through. So the
  transmitted fate is a parameter, with an 'auto' heuristic: fate 3 if any
  bounding plane is enabled in the spec, else fate 2.
- mz_list runs mix masses in one ensemble; TOF stats pooled across masses
  are meaningless, so TOF / resolving power group per m/z.

This module is Panel-free (usable from gate scripts / notebooks). The app's
Card that renders these reductions lives in ion_gym/ui/stats_card.py
(moved out of this physics module 2026-10-15, L-548).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Iterable, Optional, Sequence

# Fate tables are re-exported from their single authority
# (physics.ion_envelope, where make_summary constructs the codes) so the
# many existing `from ...stats import FATE_NAME` consumers keep working
# without a second, driftable copy living here.
from ion_gym.physics.ion_envelope import (  # noqa: F401
    FATE_COLOR, FATE_NAME, fate_color, fate_name)

FWHM_PER_SIGMA = 2.35482004503  # Gaussian


# --------------------------------------------------------------- helpers
def _summaries(results: Iterable[Any]):
    return [r.summary if hasattr(r, "summary") else r for r in results]


def _mean_std(xs):
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None, 0
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0, 1
    # POPULATION sd (ddof=0), by design. FWHM/R
    # are DESCRIPTIVE statistics of the flown packet (those ions ARE the
    # population of the simulation), and every banked R in the tree is
    # ddof=0 (fly_packet, mirror_null). This module used (n - 1), so the
    # SAME flight read R = 90,769 from the library and 90,541 on the
    # Stats tab — a silent 0.25% disagreement. One convention, stated
    # here, gated in pack_release. Report n alongside any FWHM: at
    # small n the two conventions diverge hard (n = 4 -> 15%).
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, math.sqrt(var), n


def auto_transmitted_fate(spec) -> int:
    """fate 3 if any bounding plane is enabled in the spec, else fate 1.

    "Transmitted" means the ion LEFT the device the intended way: through
    a declared bounding plane when the spec has one (fate 3), else out
    through the solved-domain boundary (fate 1). An earlier change
    returned 2 (held/timeout) for no-plane specs — but a timeout is never
    transmission, and every unbounded transit device (STL einzel among
    them) then scored zero-transmitted, so transmission metrics ranked
    penalties. Known residue: for TRAP decks
    "success" is held (2), and this bounds-flag correlate cannot know a
    trap from a transit — the honest fix is a declared transmitted-fate
    on the spec; until then this
    answers the transmission question its name asks."""
    b = getattr(spec, "bounds", None)
    if b is not None:
        for ax in ("x", "y", "z"):
            for side in ("min", "max"):
                if getattr(b, f"{ax}_{side}_on", False):
                    return 3
    return 1


def mz_of_results(spec, n: int):
    """Per-ion m/z for n results: ions cycle spec.source.mz_list."""
    # No try/except: mz_list is a DECLARED SimSpec field
    # (a capability is declared or refused, never defaulted).
    # The old `except Exception: lst = []` silently substituted all-None
    # m/z for ANY failure (a .get-style sniff standing in for a
    # contract).  A legitimately empty/None mz_list still yields the
    # all-None answer below; a malformed spec now raises where the defect
    # is, not three displays later.
    lst = list(spec.source.mz_list or [])
    if not lst:
        return [None] * n
    # CONTIGUOUS BLOCKS: n_ions is per-m/z, ions flown in
    # blocks (ion i -> mz_list[i // n_per]), matching sim_build.generate_births.
    # (was i % len, the round-robin convention, now wrong.)
    n_per = max(1, int(getattr(spec.source, "n_ions", 1)))
    return [lst[(i // n_per) % len(lst)] for i in range(n)]


# --------------------------------------------------------------- reducer
@dataclass
class MzStats:
    mz: Optional[float]
    n: int
    tof_mean: Optional[float]           # us
    tof_sigma: Optional[float]          # us
    tof_fwhm: Optional[float]           # us
    resolving_power: Optional[float]    # R = t / (2*FWHM)


@dataclass
class PlaneStats:
    """Where a set of terminated ions landed, and how tightly.

    The plane NORMAL is KNOWN when the rows were grouped by a declared
    impact plane (plane_label set); the normal is that plane's axis. Only
    for an unlabeled residual group (metal impacts not on any declared
    plane) is the normal inferred as the axis with the smallest spread —
    which degrades gracefully for a 3-D metal surface (large "thickness")
    but is silently wrong across a MIXTURE of planes, the reason grouping
    is done by declared plane first.
    """
    fate: int
    n: int
    normal: str                  # plane normal ('x'|'y'|'z') — known or inferred
    normal_mean: float           # where the plane sits
    normal_sigma: float          # thickness: ~0 => a true plane
    in_plane: tuple              # ('x','y') etc
    centroid: tuple              # mean of the two in-plane coords
    sigma: tuple                 # per-axis sigma in the plane
    r_mean: float                # mean radius about the centroid
    r_rms: float                 # RMS radius (the beam-spot number)
    r_p95: float                 # 95th percentile radius: 95% of the ions
                                 # land within this distance of the centroid.
                                 # Outlier-robust, unlike r_rms.
    plane_label: str = ""        # the DECLARED plane these rows hit (B);
                                 # "" = inferred residual (electrode impact)
    ke_n: int = 0                # how many rows CARRIED ke_end; 0 means the
                                 # route recorded no termination KE and the
                                 # KE cell renders as a blank, not as 0
    ke_mean: float = 0.0         # landing kinetic energy (eV)
    ke_sigma: float = 0.0
    ke_min: float = 0.0
    ke_max: float = 0.0
    # velocity resolved AGAINST THE PLANE NORMAL (mm/us). "Transverse" has no
    # meaning until you say transverse to WHAT -- and the plane normal is
    # already inferred here, so it needs no user input and works for an exit
    # plane in any orientation.
    v_norm_mean: float = 0.0     # longitudinal (along the normal)
    v_norm_sigma: float = 0.0
    v_perp_mean: float = 0.0     # transverse (in the plane)
    v_perp_sigma: float = 0.0
    v_perp_rms: float = 0.0
    div_mrad_rms: float = 0.0    # atan2(v_perp, v_norm): beam divergence
    div_mrad_p95: float = 0.0
    # ---- ARRIVAL TIME at this plane (so a per-plane resolution can
    # be read straight off the GUI). R belongs to a LANDING GROUP, not to a
    # fate: pooled fate-0 mixes detector hits with wall impacts on any
    # geometry messier than this one. tof_n = 0 means the group's rows
    # carried no tof OR mixed more than one m/z -- a resolving power
    # across masses is not a number, so it renders as a stated blank
    # (tof_note says which), never as a fabricated value.
    tof_n: int = 0
    tof_mean: float = 0.0        # us
    tof_fwhm: float = 0.0        # us (Gaussian, 2.3548 sigma)
    tof_R: float = 0.0           # t / (2 * FWHM)
    tof_note: str = ""           # why blank, when blank


# stats_markdown's plane_filter value meaning "every plane"; public because
# the app's card (ui/stats_card.py) offers it as a selector option
ALL_PLANES = "all planes"


def enabled_planes(spec):
    """The DECLARED impact/bounding planes, from spec.bounds — each
    (label, axis, coord_mm). This is the explicit plane set the user
    turned on; stats group against it rather than inferring a plane from
    the spread of a possibly-mixed landing cloud. [] when
    no spec/bounds."""
    b = getattr(spec, "bounds", None) if spec is not None else None
    if b is None:
        return []
    out = []
    for axis in ("x", "y", "z"):
        for side in ("min", "max"):
            if getattr(b, f"{axis}_{side}_on", False):
                coord = float(getattr(b, f"{axis}_{side}"))
                out.append((f"{axis}={coord:+g}", axis, coord))
    return out


def _tof_block(rows):
    """Arrival-time stats for ONE landing group, as PlaneStats kwargs.

    A resolving power is only a number for a single mass: mixed-m/z rows
    (the pooled 'all' tab) get a stated blank with the reason, same for
    routes whose summaries carry no tof. Single-ion groups get the time
    but no width (a 1-point FWHM is not a measurement).
    """
    mzs = {s.get("mz") for s in rows if s.get("mz") is not None}
    tofs = [s.get("tof") for s in rows if s.get("tof") is not None]
    if not tofs:
        return dict(tof_n=0, tof_note="route records no tof")
    if len(mzs) > 1:
        return dict(tof_n=0, tof_note="mixed m/z — see per-m/z tabs")
    mu, sd, cnt = _mean_std(tofs)
    if cnt < 2 or not sd:
        return dict(tof_n=cnt, tof_mean=mu, tof_note="n < 2 — no width")
    fwhm = FWHM_PER_SIGMA * sd
    return dict(tof_n=cnt, tof_mean=mu, tof_fwhm=fwhm,
                tof_R=mu / (2 * fwhm))


def _plane_stats(rows, fate, normal=None, plane_label=""):
    import math
    pts = {a: [s.get(f"{a}_end") for s in rows] for a in ("x", "y", "z")}
    if not rows or any(v is None for v in pts["x"]):
        return None
    stat = {}
    for a in ("x", "y", "z"):
        mu, sd, cnt = _mean_std(pts[a])
        if cnt == 0:
            return None
        stat[a] = (mu, sd or 0.0)
    # NORMAL: when the caller grouped rows by a DECLARED plane it passes
    # that plane's axis, so the normal is KNOWN and the "infer the
    # normal from the smallest spread" step is skipped — that inference is
    # only correct for a single-plane cloud and silently wrong across a
    # mixture (B). Fall back to inference ONLY for an unlabeled residual
    # group (electrode impacts not on any declared plane).
    if normal not in ("x", "y", "z"):
        normal = min(("x", "y", "z"), key=lambda a: stat[a][1])
    ip = tuple(a for a in ("x", "y", "z") if a != normal)
    cx, cy = stat[ip[0]][0], stat[ip[1]][0]
    rs = sorted(math.hypot(px - cx, py - cy)
                for px, py in zip(pts[ip[0]], pts[ip[1]]))
    n = len(rs)
    r_mean = sum(rs) / n
    r_rms = math.sqrt(sum(r * r for r in rs) / n)
    r_p95 = rs[min(n - 1, int(0.95 * n))]
    kes = [r.get("ke_end") for r in rows if r.get("ke_end") is not None]
    # ke_n distinguishes "no KE was recorded" from "KE measured as zero":
    # a summary route that carries no ke_end (currently everything except
    # the 3-D tracer) must render a stated blank, never a fabricated
    # 0 +/- 0 -- the same principle as provenance's "unknown is a value".
    ke_n = len(kes)
    ke_mu, ke_sd, _ = _mean_std(kes) if kes else (0.0, 0.0, 0)

    # ---- velocity resolved against the inferred plane normal -------------
    vN = vP = None
    vel = [(r.get("vx_end"), r.get("vy_end"), r.get("vz_end")) for r in rows]
    vel = [v for v in vel if all(c is not None for c in v)]
    v_n_mu = v_n_sd = v_p_mu = v_p_sd = v_p_rms = d_rms = d_p95 = 0.0
    if vel and normal in ("x", "y", "z"):
        import numpy as _np
        V = _np.asarray(vel, float)
        k = {"x": 0, "y": 1, "z": 2}[normal]
        vN = V[:, k]                                  # along the normal
        vP = _np.hypot(*[V[:, j] for j in range(3) if j != k])   # in-plane
        v_n_mu, v_n_sd = float(vN.mean()), float(vN.std() if len(vN) > 1 else 0.0)
        v_p_mu, v_p_sd = float(vP.mean()), float(vP.std() if len(vP) > 1 else 0.0)
        v_p_rms = float(_np.sqrt((vP ** 2).mean()))
        with _np.errstate(invalid="ignore", divide="ignore"):
            ang = _np.abs(_np.arctan2(vP, _np.abs(vN))) * 1e3   # mrad
        ang = ang[_np.isfinite(ang)]
        if ang.size:
            d_rms = float(_np.sqrt((ang ** 2).mean()))
            d_p95 = float(_np.percentile(ang, 95.0))
    return PlaneStats(v_norm_mean=v_n_mu, v_norm_sigma=v_n_sd,
                      v_perp_mean=v_p_mu, v_perp_sigma=v_p_sd,
                      v_perp_rms=v_p_rms, div_mrad_rms=d_rms,
                      div_mrad_p95=d_p95,
                      **_tof_block(rows),
                      ke_mean=ke_mu or 0.0, ke_sigma=ke_sd or 0.0,
                      ke_min=min(kes) if kes else 0.0,
                      ke_max=max(kes) if kes else 0.0,
                      ke_n=ke_n,
                      fate=fate, n=n, normal=normal,
                      normal_mean=stat[normal][0],
                      normal_sigma=stat[normal][1], in_plane=ip,
                      plane_label=plane_label,
                      centroid=(cx, cy),
                      sigma=(stat[ip[0]][1], stat[ip[1]][1]),
                      r_mean=r_mean, r_rms=r_rms, r_p95=r_p95)


@dataclass
class Stats:
    n_total: int
    fate_counts: dict                   # code -> count
    transmitted_fate: int
    transmission: Optional[float]
    per_mz: list                        # [MzStats] over transmitted ions
    end_spread: dict                    # axis -> (mean, sigma), transmitted
    r_end_mean: Optional[float]
    r_end_sigma: Optional[float]
    n_col_mean: Optional[float]
    impact_spread: dict                  # axis -> (mean, sigma), fate 0
    n_transmitted: int
    landing: dict = None                # fate code -> PlaneStats


def compute_stats(results, *, transmitted_fate: int = 2,
                  mz: Optional[Sequence] = None,
                  planes: Optional[Sequence] = None,
                  plane_tol_mm: float = 2.0) -> Stats:
    """Reduce an ensemble (list of driver results or raw summary dicts).

    transmitted_fate: which kind counts as success for transmission and
        for the TOF / spot statistics (see module docstring / 'auto').
    mz: optional per-result m/z (same order as results) for per-mass TOF
        grouping; from `mz_of_results(spec, len(results))` in the app.
    planes: DECLARED impact planes as (label, axis, coord_mm) — from
        `enabled_planes(spec)`. When given, landing stats GROUP terminated
        ions by which declared plane they hit (each with the KNOWN normal),
        instead of pooling a fate's hits across planes and inferring one
        normal from the mixed spread (mixed-plane pooling
        mis-weighted and mis-grouped the numbers). None => legacy behavior,
        landing keyed by fate code (gate scripts / tests unchanged).
    plane_tol_mm: an ion counts as ON a declared plane when its end
        coordinate is within this of the plane; capped internally at half
        the smallest inter-plane gap so two planes can't claim one ion.
    """
    S = _summaries(results)
    n = len(S)
    mz = list(mz) if mz is not None else [None] * n

    fate_counts: dict = {}
    for s in S:
        k = s.get("kind")
        fate_counts[k] = fate_counts.get(k, 0) + 1

    ok = [(s, m) for s, m in zip(S, mz) if s.get("kind") == transmitted_fate]
    impact_rows = [s for s in S if s.get("kind") == 0]
    transmission = (len(ok) / n) if n else None

    # per-m/z TOF stats over transmitted ions
    groups: dict = {}
    for s, m in ok:
        groups.setdefault(m, []).append(s.get("tof"))
    per_mz = []
    for m in sorted(groups, key=lambda v: (v is None, v)):
        mu, sd, cnt = _mean_std(groups[m])
        fwhm = R = None
        if mu is not None and sd is not None:
            fwhm = FWHM_PER_SIGMA * sd
            R = (mu / (2 * fwhm)) if fwhm > 0 else float("inf")
        per_mz.append(MzStats(m, cnt, mu, sd, fwhm, R))

    def spread(rows, keys=("x_end", "y_end", "z_end")):
        out = {}
        for key in keys:
            mu, sd, cnt = _mean_std([s.get(key) for s in rows])
            if cnt:
                out[key[0]] = (mu, sd)
        return out

    ok_rows = [s for s, _ in ok]
    r_mu, r_sd, _ = _mean_std([s.get("r_end") for s in ok_rows])
    nc_mu, _, _ = _mean_std([s.get("n_col") for s in S])

    # WHERE ions ended. GROUPED BY DECLARED PLANE when the plane set is
    # given (B): each declared plane's hits are a separate, correctly-
    # weighted group with a KNOWN normal; ions on no declared plane
    # (metal impacts) fall into a residual group per fate with the normal
    # inferred. Without a plane set, the legacy per-fate grouping stands
    # so gate scripts and tests are unchanged.
    landing = {}
    plist = list(planes) if planes else []
    # cap the match tolerance at half the smallest inter-plane gap so no
    # ion is ambiguously claimed by two planes (config-agnostic: derived
    # from the declared planes, not a fixed guess).
    tol = float(plane_tol_mm)
    for ax in ("x", "y", "z"):
        cs = sorted(c for _, a, c in plist if a == ax)
        gaps = [b - a for a, b in zip(cs, cs[1:]) if b > a]
        if gaps:
            tol = min(tol, 0.5 * min(gaps))

    for code in sorted(fate_counts, key=lambda c: (c is None, c)):
        rows = [s for s in S if s.get("kind") == code]
        if not plist:
            ps = _plane_stats(rows, code)          # legacy: one inferred group
            if ps is not None:
                landing[code] = ps
            continue
        # assign each row to the NEAREST declared plane within tol
        by_plane = {}
        residual = []
        for s in rows:
            best = None
            for label, axis, coord in plist:
                v = s.get(f"{axis}_end")
                if v is None:
                    continue
                d = abs(v - coord)
                if best is None or d < best[0]:
                    best = (d, label, axis)
            if best is not None and best[0] <= tol:
                by_plane.setdefault((best[1], best[2]), []).append(s)
            else:
                residual.append(s)
        for (label, axis), prows in sorted(by_plane.items()):
            ps = _plane_stats(prows, code, normal=axis, plane_label=label)
            if ps is not None:
                landing[f"{code}:{label}"] = ps
        if residual:
            ps = _plane_stats(residual, code)      # electrode impacts, inferred
            if ps is not None:
                landing[f"{code}:·" if by_plane else code] = ps

    return Stats(
        n_total=n, fate_counts=fate_counts,
        transmitted_fate=transmitted_fate, transmission=transmission,
        per_mz=per_mz, end_spread=spread(ok_rows),
        r_end_mean=r_mu, r_end_sigma=r_sd, n_col_mean=nc_mu,
        impact_spread=spread(impact_rows), n_transmitted=len(ok),
        landing=landing,
    )


# --------------------------------------------------------------- render
def _f(v, p=4):
    if v is None:
        return "–"
    if v == float("inf"):
        return "∞"
    return f"{v:.{p}g}"


def stats_markdown(st: Stats, plane_filter=None) -> str:
    L = [f"**{st.n_total} ions** &nbsp; transmitted "
         f"(fate {st.transmitted_fate}, {FATE_NAME.get(st.transmitted_fate)}): "
         f"**{st.n_transmitted}**"
         + (f" = **{st.transmission * 100:.1f}%**"
            if st.transmission is not None else ""), ""]
    L.append("| fate | n | % |")
    L.append("|---|---:|---:|")
    for code in sorted(st.fate_counts,
                       key=lambda c: (c is None, c)):
        cnt = st.fate_counts[code]
        pct = 100 * cnt / st.n_total if st.n_total else 0
        L.append(f"| {code} · {FATE_NAME.get(code, '?')} | {cnt} | {pct:.1f} |")

    # ---- WHERE they ended -------------------------------------------------
    # vrows MUST be bound before the branch: it is read at `if vrows:` below,
    # which sits OUTSIDE this `if`. With no landing groups (e.g. every ion
    # timed out) the old code raised NameError: vrows. Same failure mode as
    # the `mmode` referenced-before-assignment bug.
    vrows = []
    if st.landing:
        L += ["", "**landing statistics** — where each fate terminated, "
                  "and how tightly", ""]
        L.append("| fate | n | plane | centroid (mm) | σ (mm) | "
                 "r̄ | r_rms | r₉₅ | KE (eV) |")
        L.append("|---|---:|---|---|---|---:|---:|---:|---|")
        for key in sorted(st.landing, key=lambda k: str(k)):
            p = st.landing[key]
            if (plane_filter and plane_filter != ALL_PLANES
                    and p.plane_label != plane_filter):
                continue          # user picked a single plane (B)
            a, b = p.in_plane
            planar = p.normal_sigma < 1e-6
            nrm = (f"{p.normal} = {_f(p.normal_mean, 4)}" if planar
                   else f"{p.normal} = {_f(p.normal_mean, 3)} "
                        f"± {_f(p.normal_sigma, 2)}")
            plane = (f"{p.plane_label} ({nrm})" if p.plane_label else nrm)
            L.append(
                f"| {p.fate} · {FATE_NAME.get(p.fate, '?')} | {p.n} | "
                f"{plane} | "
                f"{a}={_f(p.centroid[0], 4)}, {b}={_f(p.centroid[1], 4)} | "
                f"{a}={_f(p.sigma[0], 3)}, {b}={_f(p.sigma[1], 3)} | "
                f"{_f(p.r_mean, 3)} | {_f(p.r_rms, 3)} | {_f(p.r_p95, 3)} | "
                + (f"{_f(p.ke_mean, 3)} ± {_f(p.ke_sigma, 2)} "
                   f"[{_f(p.ke_min, 2)}–{_f(p.ke_max, 2)}] |"
                   if getattr(p, "ke_n", 0)
                   else "— (route records no termination KE) |"))
        vrows = [p for p in st.landing.values()
                 if (p.v_perp_rms or p.v_norm_mean)
                 and not (plane_filter and plane_filter != ALL_PLANES
                          and p.plane_label != plane_filter)]

        # ---- WHEN they arrived -----------------------------------------
        trows = [(k, p) for k, p in sorted(st.landing.items(),
                                           key=lambda kv: str(kv[0]))
                 if (p.tof_n or p.tof_note)
                 and not (plane_filter and plane_filter != ALL_PLANES
                          and p.plane_label != plane_filter)]
        if trows:
            L += ["", "**arrival time** — per landing group; R is only "
                      "quoted for a single-m/z group"]
            L.append("| fate | plane | n | t̄ (µs) | FWHM (ns) | "
                     "R = t/2FWHM |")
            L.append("|---|---|---:|---:|---:|---:|")
            for _, p in trows:
                where = p.plane_label or f"{p.normal}≈{_f(p.normal_mean, 4)}"
                if p.tof_note:
                    tail = (f"{_f(p.tof_mean, 6) if p.tof_n else '–'} | – | "
                            f"— ({p.tof_note}) |")
                else:
                    tail = (f"{_f(p.tof_mean, 6)} | {_f(p.tof_fwhm * 1e3)} | "
                            f"**{_f(p.tof_R, 5)}** |")
                L.append(f"| {p.fate} · {FATE_NAME.get(p.fate, '?')} | "
                         f"{where} | {p.tof_n or p.n} | {tail}")

    if vrows:
        L += ["", "**exit velocity** — resolved against each plane's normal"]
        L.append("| fate | v∥ (mm/µs) | v⊥ (mm/µs) | v⊥ rms | "
                 "divergence rms | div ₉₅ |")
        L.append("|---|---|---|---:|---:|---:|")
        for p in vrows:
            L.append(
                f"| {p.fate} · {FATE_NAME.get(p.fate, '?')} | "
                f"{_f(p.v_norm_mean, 3)} ± {_f(p.v_norm_sigma, 3)} | "
                f"{_f(p.v_perp_mean, 3)} ± {_f(p.v_perp_sigma, 3)} | "
                f"{_f(p.v_perp_rms, 3)} | "
                f"{_f(p.div_mrad_rms, 1)} mrad | "
                f"{_f(p.div_mrad_p95, 1)} mrad |")
        L += ["", "_v∥ is along the inferred plane NORMAL, v⊥ is in the "
                  "plane — 'transverse' is meaningless until you say "
                  "transverse to what. Divergence = atan2(v⊥, |v∥|)._"]

    L += ["", "_r₉₅ = 95th-percentile radius: 95% of the ions land inside "
              "it. KE is the kinetic energy AT termination._",
          "", "_plane normal is INFERRED as the axis of least spread; a "
                  "large ± on it means the hits were not planar (metal spread "
                  "over a 3-D surface), and the in-plane numbers should be "
                  "read with that in mind. r is measured about the centroid, "
                  "not the axis._"]
    if st.per_mz:
        L += ["", "| m/z | n | TOF μs | σ ns | FWHM ns | R=t/2FWHM |",
              "|---|---:|---:|---:|---:|---:|"]
        for g in st.per_mz:
            L.append(
                f"| {_f(g.mz)} | {g.n} | {_f(g.tof_mean, 6)} | "
                f"{_f(None if g.tof_sigma is None else g.tof_sigma * 1e3)} | "
                f"{_f(None if g.tof_fwhm is None else g.tof_fwhm * 1e3)} | "
                f"**{_f(g.resolving_power, 5)}** |")
    if st.end_spread:
        parts = [f"{ax}: {_f(mu)}±{_f(sd, 3)}"
                 for ax, (mu, sd) in st.end_spread.items()]
        L += ["", "**end spot (transmitted, mm):** " + " &nbsp; ".join(parts)]
    if st.r_end_mean is not None:
        L.append(f"**r_end:** {_f(st.r_end_mean)}±{_f(st.r_end_sigma, 3)} mm")
    if st.n_col_mean:
        L.append(f"**collisions/ion (all):** {_f(st.n_col_mean, 3)}")
    if st.impact_spread:
        parts = [f"{ax}: {_f(mu)}±{_f(sd, 3)}"
                 for ax, (mu, sd) in st.impact_spread.items()]
        L += ["", "**electrode-impact locations (fate 0, mm):** "
              + " &nbsp; ".join(parts)]
    return "\n".join(L)


# ---------------------------------------------------- station table (text)
def stations_markdown(rows):
    """Markdown table of per-station detection statistics.

    Instrument-agnostic: a station declares an axis, a position and a
    window, and this reports what reached it. A station nothing reached
    is shown with its reason, never dropped -- a missing row would read
    as "no such detector" when the truth is "nothing arrived".
    """
    if not rows:
        return ""
    out = ["#### Stations (declared detection planes)", "",
           "| station | kind | plane | window | hits | t (us) | "
           "FWHM (ns) | R = t/2dt | families |",
           "|---|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        win = ", ".join(f"{a}=[{lo:g}, {hi:g}]"
                        for a, (lo, hi) in sorted(r["window"].items())) \
            or "unbounded"
        hits = f"{r['n_hit']}/{r['n_total']}"
        if r["n_hit"]:
            out.append(
                f"| {r['name']} | {r['kind']} | {r['axis']}="
                f"{r['pos_mm']:g} mm | {win} | {hits} | "
                f"{r['t_us']:.3f} | {r['fwhm_ns']:.2f} | "
                f"{r['R']:,.0f} | {r['k_families'] or '-'} |")
        else:
            out.append(
                f"| {r['name']} | {r['kind']} | {r['axis']}="
                f"{r['pos_mm']:g} mm | {win} | {hits} | - | - | - | "
                f"_{r.get('reason', 'no hits')}_ |")
    out += ["", "_R here is the ensemble quantile-FWHM resolution at "
                "the station, from the same declared window the View "
                "tab draws._", ""]
    return "\n".join(out)


# --------------------------------------- delivered spread in cells
def delivered_spread_cells(spec):
    """Per-axis extent, in CELLS, of the ensemble AS DELIVERED at birth --
    the flight-side half of the sub-cell resolution check (the
    static SimSpec advisory reads only the DECLARED source
    extent and explicitly defers point and births-file sources to a flown
    measurement: this is that measurement).

    Births are REGENERATED from the spec, which the seed policy makes the
    flown ensemble bit-for-bit (source.seed defaults FIXED; a random-mode
    spec object carries the drawn seed in _drawn_seed) -- so the number
    measures delivery, not declaration, and point sources, births files,
    gaussians and boxes are all measured identically.

    Returns (spread, h): spread is an ordered {axis: cells} over the
    LATTICE-RESOLVED axes only -- planar 2-D: x, y; r-z: axial x and
    radius r = hypot(y, z); 3-D: x, y, z. The 2-D axial drift axis has no
    lattice channel BY DESIGN and is deliberately not reported (a
    standing "sub-cell" flag there would be a false alarm). Field
    sampling is (tri)linear, so structure narrower than a cell is the
    stencil's, not the field's (SOURCE_CELLS_ADVISORY rationale); this
    function reports, callers advise -- never refuse (convergence is
    demonstrated by a pitch ladder, not assumed from a threshold).
    """
    import numpy as np
    from ion_gym.physics.sim_build import generate_births, build_route
    b = np.asarray(generate_births(spec), dtype=float)
    h = float(spec.geometry.mm_per_gu)
    if b.size == 0:
        raise ValueError("delivered_spread_cells: the spec generates zero "
                         "births -- nothing was delivered to measure")
    # The lattice-resolved axes are a property of the ROUTE, so they come
    # from build_route (the one classification build_run dispatches on).
    # This used to be `if rz / else` with the else meaning "Cartesian" and
    # 3-D re-decided here by depth_mm > 0 -- a second copy of the routing
    # rule (it agrees with build_route on every current deck, but a route
    # added there would have been measured here as Cartesian, L-554).
    route = build_route(spec).builder
    spread = {}
    if route == "rz":
        spread["x"] = float(np.ptp(b[:, 0])) / h
        spread["r"] = float(np.ptp(np.hypot(b[:, 1], b[:, 2]))) / h
    elif route in ("planar", "stl2d"):
        spread["x"] = float(np.ptp(b[:, 0])) / h
        spread["y"] = float(np.ptp(b[:, 1])) / h
    elif route in ("stl3d", "scene3d", "shapes3d"):
        for k, ax in enumerate("xyz"):
            spread[ax] = float(np.ptp(b[:, k])) / h
    else:
        raise ValueError(
            f"delivered_spread_cells: route {route!r} has no declared "
            f"lattice axes here (known: rz, planar, stl2d, stl3d, scene3d, "
            f"shapes3d)")
    return spread, h


def delivered_spread_text(spec, *, markdown=False):
    """One line for the run print / stats card, from the ONE authority
    above. Sub-advisory axes (< SOURCE_CELLS_ADVISORY cells, the same
    named threshold the static check uses) are called out by name;
    markdown=True bolds the advisory for the card."""
    from ion_gym.io.sim_spec import SOURCE_CELLS_ADVISORY
    spread, h = delivered_spread_cells(spec)
    body = " · ".join(f"{a} {v:.2f}" for a, v in spread.items())
    line = f"delivered spread: {body} cells @ {h:g} mm/gu"
    low = [a for a, v in spread.items() if v < SOURCE_CELLS_ADVISORY]
    if low:
        note = (f"axis {', '.join(low)} below the "
                f"{SOURCE_CELLS_ADVISORY:g}-cell advisory — packet-derived "
                f"observables need a pitch ladder there")
        line += (f" — **{note}**" if markdown else f" [ADVISORY: {note}]")
    return line
