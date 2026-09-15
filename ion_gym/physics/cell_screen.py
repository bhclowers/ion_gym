"""
cell_screen.py — geometry/voltage search space and the
resumable Sobol screen driver.

SPACE (mirrored: mirror_tb=mirror_lr=True): one flat vector
maps to a CellBoardParam. Per space, k_bottom rails on the bottom cell
(mirrored to top) and k_side rails on the left cell (mirrored to
right); every rail carries (width, gap, signed RF amplitude, DC). All
declared bounds hold BY CONSTRUCTION (vector bounds start at the declared
minima) except pattern-fits-span, which is handled by the DERIVED
TILING RULE below.

DECLARED DERIVED RULES (parameters of the mapping, not hidden branches):
  * n_tiles is DERIVED, not searched: each wall tiles as many whole
    cells as fit its span, centered ("as many as fit"). Zero fitting
    cells => CellBoardInfeasible (recorded, never skipped).
  * channel_h SNAPS so the y mirror plane lands on a lattice node with
    a node-symmetric domain: total height 2*m*pitch (fold engages, the
    declared symmetry is collected as a solve discount). channel_w
    snaps to the pitch for cache friendliness.
  * side wall may be switched OFF by the search: the last side-rail
    slot doubles as the wall's presence via side_active in [0,1]
    (>= 0.5 = active) — walls are possible, not required, by design.

SCREEN: deterministic scrambled Sobol (seeded), evaluated at a DECLARED
screen operating point (small decks, shortened t_max, template pitch).
The JSONL bank is the checkpoint: every
row carries the Sobol index; a rerun
skips banked indices, so the same command resumes in the sandbox
(chunked) and on a local machine (to completion). Screen numbers are
MAPS, not results: anything quotable is re-scored at the fidelity pitch
and full deck by the refinement/audit stage.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ion_gym.physics.cell_board import (CellBoardParam, WallParam,
                                        RailParam, build_spec,
                                        CellBoardInfeasible,
                                        MIN_FEATURE_MM, RF_MAX_V,
                                        ENVELOPE_MM)


# Evaluation-semantics version: BUMP whenever evaluate_candidate's
# meaning changes (constraints, birth rule, objective wiring), so bank
# filenames derived from tag()+EVAL_VERSION can never mix definitions.
# v2 = RF-pairing constraint + probe birth.
# v4 = symmetric-RF space (ONE shared amplitude, per-rail phase
#      selector; the drive circuits deliver equal-magnitude phases),
#      gap dims REMOVED (isolation gap fixed at
#      125 um; wider = exposed dielectric), rail width min 0.4 mm,
#      edge absorption.
# v3 = direct-LU planar solve engine (exact) replaced multigrid
#      (tol-approximate, residual ~7e-4 of V_BASIS): field values move
#      within mg's error, so banked values are not comparable across
#      the change.
EVAL_VERSION = 4


@dataclass
class CellSpace:
    """Declared bounds of one search space. Every field is a
    named bound; nothing inside the mapping is a literal."""
    k_bottom: int = 3
    k_side: int = 1
    k_top: int = 0            # rails in the TOP cell — asym mode only
                              # (0 elsewhere; asym defaults it to
                              # k_bottom when left 0).
    wall_mode: str = "sandwich"   # "sandwich": top mirrors bottom (the
                              # SLIM class).
                              # "pressed": top wall = UNIFORM DC PLATE
                              # (searchable top_dc, +1 dim) pressing
                              # ions toward the patterned bottom — the
                              # one-sided SLIM configuration run
                              # experimentally, and the RECOMMENDED
                              # mode for surface-trap-style cells
                              # (five-wire class at k_bottom=5).
                              # "asym": top wall carries its OWN
                              # independent k_top-rail cell — the
                              # asymmetric surface-trap / merged space
                              # (contains sandwich, pressed, single as
                              # special cases; ~2x the dims).
                              # "single": top wall ABSENT (open half-
                              # space) — retained as an escape hatch;
                              # in practice a confining
                              # counter-surface is almost certainly
                              # required, so prefer "pressed".
                              # Different modes are DIFFERENT spaces:
                              # tag() differs, so banks auto-separate.
    rail_thick_mm: float = 0.2
    pitch_mm: float = 0.05            # template pitch; H snapping uses it
    w_lo_mm: float = 4.0
    h_lo_mm: float = 1.0
    rail_w_hi_mm: float = 1.5
    gap_hi_mm: float = 1.5
    side_rail_w_hi_mm: float = 6.0
    dc_lo_v: float = -10.0
    dc_hi_v: float = 10.0
    guard_lo_v: float = -30.0   # OUTER/side (guard) rails get a wider
    guard_hi_v: float = 30.0    # range: with walls off, guards do the
                                # transverse confinement.
                                # Interior DC stays +-10.

    min_rf_rails: int = 2     # RF-pairing constraint:
                              # a LONE RF-driven rail role is rejected —
                              # stability needs a null formed by >=2 RF
                              # roles (opposite-phase pair, OR same-phase
                              # pair against grounded/DC metal, per the
                              # surface-trap literature). DC-only boards
                              # (zero RF) remain legal; retention judges
                              # them. Set 0 to disable ("I could be
                              # wrong" escape hatch); rejections are
                              # RECORDED infeasible rows, so the bank
                              # shows how often the rule binds.
    f_lo_hz: float = 0.4e6
    f_hi_hz: float = 2.5e6            # declared ceiling
    envelope_mm: float = ENVELOPE_MM

    # ---- vector layout ------------------------------------------------
    def names(self) -> List[str]:
        n = ["channel_w_mm", "channel_h_mm", "frequency_hz",
             "rf_amp_v"]
        for i in range(self.k_bottom):
            n += ["b{0}_w".format(i), "b{0}_sel".format(i),
                  "b{0}_dc".format(i)]
        if self.wall_mode == "asym":
            for i in range(self._k_top()):
                n += ["t{0}_w".format(i), "t{0}_sel".format(i),
                      "t{0}_dc".format(i)]
        for i in range(self.k_side):
            n += ["s{0}_w".format(i), "s{0}_sel".format(i),
                  "s{0}_dc".format(i)]
        if self.k_side > 0:
            n.append("side_active")     # k_side=0 => wall-less, no knob
        if self.wall_mode == "pressed":
            n.append("top_dc")            # the pressing plate's bias
        return n

    def bounds(self):
        hi_wh = self.envelope_mm - 2.0 * self.rail_thick_mm
        from ion_gym.physics.cell_board import RAIL_MIN_MM
        lo = [self.w_lo_mm, self.h_lo_mm, self.f_lo_hz, 0.0]
        hi = [hi_wh, hi_wh, self.f_hi_hz, RF_MAX_V]
        for j in range(self.k_bottom):
            guard = (j == 0 or j == self.k_bottom - 1)   # outer = guard
            dlo = self.guard_lo_v if guard else self.dc_lo_v
            dhi = self.guard_hi_v if guard else self.dc_hi_v
            lo += [RAIL_MIN_MM, -1.5, dlo]
            hi += [self.rail_w_hi_mm, 1.5, dhi]
        if self.wall_mode == "asym":
            for _ in range(self._k_top()):
                lo += [RAIL_MIN_MM, -1.5, self.dc_lo_v]
                hi += [self.rail_w_hi_mm, 1.5, self.dc_hi_v]
        for _ in range(self.k_side):
            lo += [RAIL_MIN_MM, -1.5, self.guard_lo_v]
            hi += [self.side_rail_w_hi_mm, 1.5, self.guard_hi_v]
        if self.k_side > 0:
            lo.append(0.0)
            hi.append(1.0)
        if self.wall_mode == "pressed":
            lo.append(self.dc_lo_v)
            hi.append(self.dc_hi_v)
        return np.array(lo, float), np.array(hi, float)

    # ---- vector -> CellBoardParam ------------------------------------
    def to_param(self, x: Sequence[float]) -> CellBoardParam:
        x = np.asarray(x, float)
        lo, hi = self.bounds()
        if x.shape != lo.shape:
            raise ValueError("vector length {0} != space dim {1}".format(
                x.shape, lo.shape))
        if (x < lo - 1e-9).any() or (x > hi + 1e-9).any():
            j = int(np.argmax((x < lo - 1e-9) | (x > hi + 1e-9)))
            raise CellBoardInfeasible(
                "component {0} ({1}) = {2:g} outside [{3:g}, {4:g}]"
                .format(j, self.names()[j], x[j], lo[j], hi[j]))
        t = self.rail_thick_mm
        h2 = 2.0 * self.pitch_mm
        total_h = max(h2 * round((x[1] + 2.0 * t) / h2), h2)
        H = total_h - 2.0 * t
        W = self.pitch_mm * round(x[0] / self.pitch_mm)
        f = float(x[2])
        from ion_gym.physics.cell_board import RF_MIN_V as _rfmin
        amp = float(x[3])          # ONE shared RF amplitude magnitude:
                                   # the drive circuits deliver EQUAL
                                   # phases by construction. Below
                                   # RF_MIN_V the board is DC-only.
        if amp < _rfmin:
            amp = 0.0
        idx = 4

        def rails(k):
            nonlocal idx
            out = []
            for _ in range(k):
                w, sel, dc = x[idx:idx + 3]
                idx += 3
                if amp > 0.0 and sel > 0.5:
                    a = amp                    # phase 0
                elif amp > 0.0 and sel < -0.5:
                    a = -amp                   # phase 180
                else:
                    a = 0.0                    # RF off on this rail
                out.append(RailParam(float(w), MIN_FEATURE_MM,
                                     float(a), float(dc)))
            return out

        if self.wall_mode not in ("sandwich", "pressed", "asym",
                                  "single"):
            raise ValueError("wall_mode must be 'sandwich', 'pressed', "
                             "'asym' or 'single', got "
                             + repr(self.wall_mode))
        bottom = WallParam(cell=rails(self.k_bottom), n_tiles=1)
        top_asym = (WallParam(cell=rails(self._k_top()), n_tiles=1)
                    if self.wall_mode == "asym" else None)
        side = WallParam(cell=rails(self.k_side), n_tiles=1)
        if self.k_side > 0:
            side.active = bool(x[idx] >= 0.5)
            idx += 1
        else:
            side.active = False          # wall-less: guards do the edge
        top_dc = float(x[idx]) if self.wall_mode == "pressed" else None

        # derived tiling: as many whole cells as fit, centered
        tiled = [(bottom, W, "bottom"), (side, H, "left")]
        if top_asym is not None:
            tiled.insert(1, (top_asym, W, "top"))
        for wall, span, name in tiled:
            if not wall.active:
                continue
            cell_len = sum(r.width_mm + r.gap_after_mm for r in wall.cell)
            trailing = wall.cell[-1].gap_after_mm
            n = int((span + trailing + 1e-9) // cell_len)
            if n < 1:
                raise CellBoardInfeasible(
                    "{0} wall: one cell ({1:.3f} mm) exceeds span "
                    "{2:.3f} mm".format(name, cell_len - trailing, span))
            wall.n_tiles = n
        if self.min_rf_rails:
            from ion_gym.physics.cell_board import (rail_is_rf,
                                                    RF_MIN_V)
            n_rf = sum(1 for w in (bottom, top_asym, side)
                       if w is not None and w.active
                       for i in range(len(w.cell))
                       if rail_is_rf(w, i, RF_MIN_V))
            if 0 < n_rf < self.min_rf_rails:
                raise CellBoardInfeasible(
                    "unpaired RF: {0} RF-driven rail role(s) < "
                    "min_rf_rails={1} (lone RF rail lacks a stable "
                    "null partner; set min_rf_rails=0 to allow)"
                    .format(n_rf, self.min_rf_rails))
        if self.wall_mode == "sandwich":
            top, mirror_tb = None, True
        elif self.wall_mode == "asym":
            top, mirror_tb = top_asym, False
        elif self.wall_mode == "pressed":
            from ion_gym.physics.cell_board import uniform_dc_wall
            top, mirror_tb = uniform_dc_wall(W, top_dc), False
        else:                                   # "single" (open)
            top, mirror_tb = WallParam(active=False), False
        return CellBoardParam(
            channel_w_mm=W, channel_h_mm=H, bottom=bottom, left=side,
            top=top, mirror_tb=mirror_tb, mirror_lr=True,
            frequency_hz=f, rail_thick_mm=t,
            envelope_mm=self.envelope_mm)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def _k_top(self) -> int:
        return self.k_top if self.k_top else self.k_bottom

    def tag(self) -> str:
        """Short definition tag for bank filenames — different spaces
        can never share a bank by construction."""
        kt = ("t{0}".format(self._k_top())
              if self.wall_mode == "asym" else "")
        return "{0}_k{1}{2}s{3}_rf{4}".format(
            self.wall_mode, self.k_bottom, kt, self.k_side,
            self.min_rf_rails)


# ---------------------------------------------------------------- eval
def pe_min_birth(spec, *, mz: Optional[float] = None,
                 margin_mm: float = 0.3):
    '''Move the ion birth point to the EFFECTIVE-POTENTIAL MINIMUM of
    this candidate (pseudopotential + DC, median m/z by default),
    restricted to interior nodes at least margin_mm clear of metal.
    Root-cause fix: birth at the geometric channel
    center scored asymmetric-but-confining candidates infeasible as a
    BIRTH ARTIFACT (ejected before reaching the trap they had). Uses
    the already-cached solve (no extra field cost). Mutates
    spec.source.x0/y0 and returns (x0, y0) so callers record the birth
    rule in the bank. Refuses when no node clears the margin.'''
    from scipy.ndimage import distance_transform_edt
    from ion_gym.physics.build_planar import build_planar_model
    if mz is None:
        mzs = sorted(spec.source.mz_list or [])
        if not mzs:
            raise ValueError("pe_min_birth needs mz_list on the spec")
        mz = mzs[len(mzs) // 2]                  # median species
    from scipy.ndimage import minimum_filter
    model = build_planar_model(spec)
    xg, yg, pe, ele = model.pe_surface(mz=mz)
    pe = np.asarray(pe, float)
    # pe_surface layout is [ix, iy] — pinned EMPIRICALLY on the
    # non-square SLIM baseline (a shape sniff is ambiguous
    # for square domains and picked the wrong convention there).
    if pe.shape != (len(xg), len(yg)):
        raise ValueError(
            "pe_surface shape {0} != (nx,ny)=({1},{2}) — the [ix,iy] "
            "convention this rule is pinned to has changed upstream"
            .format(pe.shape, len(xg), len(yg)))
    ax_x, ax_y = 0, 1
    h = float(spec.geometry.mm_per_gu)
    dist = distance_transform_edt(np.asarray(ele) == 0, sampling=h)
    # PHYSICS OF THE RULE (revised after the center-vs-pe_min A/B
    # exposed the naive version): the GLOBAL minimum of pseudo+DC for a
    # positive ion hugs the most attractive negative-DC rail — a death
    # spot, not a trap. The trapping null is a LOCAL minimum standing
    # clear of metal. So: local minima only (3x3 plateau-tolerant),
    # clearance >= margin, and SELECT BY MAX CLEARANCE (mid-gap nulls
    # beat margin-hugging DC-attraction artifacts), tie-broken by PE.
    finite = np.isfinite(pe)
    local_min = (pe <= minimum_filter(pe, size=3)) & finite
    ok = local_min & (dist >= margin_mm)
    if not ok.any():
        raise CellBoardInfeasible(
            "no LOCAL effective-potential minimum clears the {0} mm "
            "birth margin — no interior trapping basin to be born into"
            .format(margin_mm))
    # SELECTION = BASIN PROMINENCE (2nd revision; the anchor check
    # caught max-clearance picking far-field PLATEAUS, which the
    # plateau-tolerant minimum test admits): score each clearing
    # minimum by the median PE rise on a ring around it. Plateaus
    # score ~0; a real null scores its wall height. Attraction wells
    # self-exclude upstream (their potential keeps falling toward the
    # electrode, so they are not raw local minima).
    r_ring = 0.35                                # mm, named
    cand = np.argwhere(ok)
    ring = []
    n_ring = 16
    for kk in range(n_ring):
        th = 2.0 * math.pi * kk / n_ring
        ring.append((int(round(r_ring * math.sin(th) / h)),
                     int(round(r_ring * math.cos(th) / h))))
    ring = sorted(set(ring))
    best = None
    for c in cand:
        vals = []
        for dj, di in ring:
            j, i = int(c[0]) + dj, int(c[1]) + di
            if (0 <= j < pe.shape[0] and 0 <= i < pe.shape[1]
                    and dist[j, i] >= margin_mm
                    and np.isfinite(pe[j, i])):
                vals.append(pe[j, i])
        if len(vals) < 6:
            continue                     # ring mostly in metal/margin:
                                         # not a bornable basin center
        score = float(np.median(vals) - pe[tuple(c)])
        if best is None or score > best[0]:
            best = (score, c)
    if best is None or best[0] <= 0.0:
        raise CellBoardInfeasible(
            "no PROMINENT effective-potential basin clears the birth "
            "margin (all clearing minima are flat or unringable) — "
            "nothing to be born into")
    sel = best[1]
    spec.source.x0_mm = float(xg[int(sel[ax_x])])
    spec.source.y0_mm = float(yg[int(sel[ax_y])])
    return spec.source.x0_mm, spec.source.y0_mm


def evaluate_candidate(space: CellSpace, x, template, objective, *,
                       n_ions: Optional[int] = None,
                       t_max_us: Optional[float] = None,
                       seed: int = 0,
                       birth: str = "probe") -> dict:
    """One candidate -> metric row (dict, bank-ready). Infeasible is a
    RESULT: {"status": "infeasible", "reason": ...}. Feasible rows carry
    the scalar objective value plus the report aggregates and the
    operating point."""
    try:
        p = space.to_param(x)
        spec = build_spec(p, template)
    except CellBoardInfeasible as e:
        return {"status": "infeasible", "reason": str(e)}
    if n_ions is not None:
        spec.source.n_ions = int(n_ions)
    if t_max_us is not None:
        spec.integration.t_max_us = float(t_max_us)
    spec.source.seed = int(seed)
    from ion_gym.physics.sim_build import build_run as _br

    def _probe_retention(x0, y0, n_probe=6, t_probe_us=20.0):
        sp = copy.deepcopy(spec)
        sp.source.x0_mm, sp.source.y0_mm = x0, y0
        sp.source.n_ions = n_probe
        sp.integration.t_max_us = t_probe_us
        _m, _fly, _c, _b = _br(sp)
        kinds = [_fly(k)[1].get("kind") for k in range(len(_b))]
        return kinds.count(2) / max(len(kinds), 1)

    if birth == "probe":
        # EMPIRICAL BIRTH SELECTION (3rd revision): both
        # landscape heuristics failed A/B validation in one direction
        # or the other (global min -> attraction spots; prominence ->
        # sharp near-board pockets). So the physics decides: tiny probe
        # decks fly from BOTH starts; the better-retaining start hosts
        # the full deck (tie -> center: deterministic, conservative).
        # Never worse than center-birth by construction; rescues
        # asymmetric geometries when the PE basin is the true trap.
        cx, cy = spec.source.x0_mm, spec.source.y0_mm
        starts = {"center": (cx, cy)}
        try:
            starts["pe_min"] = pe_min_birth(spec)
        except CellBoardInfeasible:
            pass                       # no clearing basin: center only
        rets = {k: _probe_retention(*v) for k, v in starts.items()}
        pick = max(sorted(starts),      # 'center' wins ties (sorted)
                   key=lambda k: (rets[k], k == "center"))
        spec.source.x0_mm, spec.source.y0_mm = starts[pick]
        birth_rec = {"birth": "probe:" + pick,
                     "x0_mm": spec.source.x0_mm,
                     "y0_mm": spec.source.y0_mm,
                     "probe_retention": rets}
    elif birth == "pe_min":
        try:
            bx, by = pe_min_birth(spec)
        except CellBoardInfeasible as e:
            return {"status": "infeasible", "reason": str(e)}
        birth_rec = {"birth": "pe_min", "x0_mm": bx, "y0_mm": by}
    elif birth == "center":
        birth_rec = {"birth": "center",
                     "x0_mm": spec.source.x0_mm,
                     "y0_mm": spec.source.y0_mm}
    else:
        raise ValueError("birth must be 'probe', 'pe_min' or "
                         "'center', got " + repr(birth))

    from ion_gym.physics.sim_build import build_run
    from ion_gym.physics.traj_stats import confinement_report

    class _R:
        __slots__ = ("traj", "summary")

        def __init__(self, tr, sm):
            self.traj, self.summary = tr, sm

    model, fly, cols, births = build_run(spec)
    results = [_R(*fly(i)) for i in range(len(births))]
    value = float(objective(results, cols, spec))
    rep = confinement_report(results, cols, spec)

    # CONFINEMENT EVIDENCE (screen-horizon guard): a cloud that is still
    # expanding at t_max is indistinguishable from a slowly-escaping one
    # — its bath-temperature score is a horizon artifact. sigma_growth =
    # steady-window sigma / early-window sigma per axis; ~1 = plateaued.
    # Evidence, not proof: the refinement's full horizon decides.
    def _pooled_sigma(f_lo, f_hi, axis):
        from ion_gym.io.records import TrajRecord
        tm = float(spec.integration.t_max_us)
        vals = []
        for r in results:
            rec = TrajRecord(r.traj, cols)
            t = rec["t"]
            m = (t >= f_lo * tm) & (t < f_hi * tm)
            if m.any():
                vals.append(rec[axis][m])
        if not vals:
            return None
        return float(np.std(np.concatenate(vals)))
    growth = {}
    for ax in ("x", "y"):
        s_early = _pooled_sigma(0.25, 0.50, ax)
        s_late = _pooled_sigma(0.75, 1.00, ax)
        growth[ax] = (None if not s_early or s_late is None
                      else s_late / s_early)
    return {"status": "ok", "value": value, **birth_rec,
            "sigma_growth_x": growth["x"], "sigma_growth_y": growth["y"],
            "retention_worst": rep["retention_worst"],
            "clearance_p5_worst": rep["clearance_p5_worst"],
            "excess_T_worst_K": rep["excess_T_worst_K"],
            "excess_T_mean_K": rep["excess_T_mean_K"],
            "operating_point": rep["operating_point"],
            "n_electrodes": len(spec.geometry.electrodes),
            "symmetry_y": spec.geometry.symmetry.planes.get("y", "none")}


# --------------------------------------------------------------- screen
def sobol_points(space: CellSpace, n: int, seed: int) -> np.ndarray:
    """Deterministic scrambled Sobol sample of the space, [n, dim]."""
    from scipy.stats import qmc
    lo, hi = space.bounds()
    eng = qmc.Sobol(d=len(lo), scramble=True, seed=seed)
    u = eng.random(n)
    return lo + u * (hi - lo)


def run_screen(space: CellSpace, template, objective, bank_path: str, *,
               n_points: int, sobol_seed: int = 3,
               n_ions: int = 60, t_max_us: float = 100.0,
               fly_seed: int = 0, time_budget_s: float = 0.0,
               verbose: bool = True) -> int:
    """Evaluate Sobol points, appending one JSON line per candidate.
    THE BANK IS THE CHECKPOINT: banked indices are skipped, so the same
    call resumes after any interruption. time_budget_s > 0 returns
    early (sandbox chunking); 0 runs to completion (local machine).
    Returns the number of indices newly evaluated this call."""
    pts = sobol_points(space, n_points, sobol_seed)
    done = set()
    if os.path.exists(bank_path):
        with open(bank_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "screen":
                    done.add(int(row["index"]))
    t0 = time.time()
    n_new = 0
    for i in range(n_points):
        if i in done:
            continue
        if time_budget_s and time.time() - t0 > time_budget_s:
            if verbose:
                print("[screen] budget reached: {0}/{1} banked — rerun "
                      "to resume".format(len(done) + n_new, n_points))
            return n_new
        row = evaluate_candidate(space, pts[i], template, objective,
                                 n_ions=n_ions, t_max_us=t_max_us,
                                 seed=fly_seed)
        row.update({"kind": "screen", "index": i,
                    "x": [float(v) for v in pts[i]],
                    "names": space.names(),
                    "screen_op": {"n_ions": n_ions, "t_max_us": t_max_us,
                                  "sobol_seed": sobol_seed,
                                  "fly_seed": fly_seed},
                    "space": space.to_dict(),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with open(bank_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        n_new += 1
        if verbose:
            tag = (row.get("reason", "")[:48] if row["status"] != "ok"
                   else "{0:.1f}".format(row["value"]))
            print("[screen {0:4d}/{1}] {2}: {3}".format(
                i, n_points, row["status"], tag))
    if verbose:
        print("[screen] complete: {0} points banked".format(n_points))
    return n_new


# --------------------------------------------------------------- basins
FEASIBLE_VALUE_MAX = 1.0e3   # objective values at/above this are the
                             # constraint-penalty ladder, never basins


def top_distinct(bank_path: str, n_basins: int, space: CellSpace, *,
                 min_dist: float = 0.15,
                 max_sigma_growth: float = 1.25) -> List[dict]:
    """Greedy pick of the best feasible screen rows separated by at
    least min_dist in the NORMALIZED parameter space — CMA restart
    seeds from distinct basins, not n copies of the same minimum."""
    lo, hi = space.bounds()
    rows = []
    n_seen = 0
    with open(bank_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "screen":
                continue
            n_seen += 1
            if r.get("status") != "ok":
                continue
            if r["value"] >= FEASIBLE_VALUE_MAX:
                continue      # constraint-penalty tier: not a basin
            keys = ["sigma_growth_" + ax for ax in ("x", "y")]
            if any(k not in r for k in keys):
                raise ValueError(
                    "screen row index {0} lacks sigma_growth fields — "
                    "bank predates the confinement-evidence guard; "
                    "re-screen rather than rank unguarded rows"
                    .format(r.get("index")))
            g = [r[k] for k in keys]
            if any(v is None for v in g):
                continue      # no steady samples: reported no-evidence row
            if max(g) > max_sigma_growth:
                continue      # still-expanding cloud: horizon artifact
            rows.append(r)
    if n_seen == 0:
        raise ValueError("bank {0} holds no screen rows at all — wrong "
                         "path or the screen has not run".format(bank_path))
    if not rows:
        return []   # screen ran, nothing passed the guards: a REPORTED
                    # state (extend the screen / raise screen t_max),
                    # not an exception
    rows.sort(key=lambda r: r["value"])
    picked = []
    for r in rows:
        z = (np.asarray(r["x"]) - lo) / (hi - lo)
        if all(np.linalg.norm(z - (np.asarray(q["x"]) - lo) / (hi - lo))
               >= min_dist for q in picked):
            picked.append(r)
        if len(picked) == n_basins:
            break
    return picked


# --------------------------------------------------------------- resets
def _retire(path: str) -> Optional[str]:
    """Move a state file aside to a timestamped .bak — visible and
    reversible, never a silent delete. Returns the new path, or None if
    the file did not exist (reported by the caller)."""
    if not os.path.exists(path):
        return None
    dst = "{0}.{1}.bak".format(path, time.strftime("%Y%m%d-%H%M%S"))
    os.rename(path, dst)
    return dst


def reset_screen(bank_path: str) -> None:
    """Reset the Sobol screen: retire the bank so every index
    re-evaluates on the next run. Prints exactly what moved where and
    how many rows it held. NOTE: to grow a screen, raise n_points
    instead (the bank extends); to change the space/objective/operating
    point, use a NEW bank filename instead of resetting."""
    n = 0
    if os.path.exists(bank_path):
        with open(bank_path) as f:
            n = sum(1 for _ in f)
    dst = _retire(bank_path)
    if dst is None:
        print("reset_screen: no bank at {0} — nothing to reset"
              .format(bank_path))
    else:
        print("reset_screen: retired {0} rows -> {1}".format(n, dst))


def reset_basin(results_dir: str, basin: int, stem: str = "refine_k3s1"
                ) -> None:
    """Reset one refinement basin: retire its bank AND its CMA
    checkpoint TOGETHER (a bank without its checkpoint, or vice versa,
    is a mixed history — the failure this function exists to prevent).
    Prints the disposition of both files."""
    for ext in (".jsonl", ".ckpt"):
        path = os.path.join(results_dir,
                            "{0}_b{1}{2}".format(stem, basin, ext))
        dst = _retire(path)
        print("reset_basin {0}: {1} -> {2}".format(
            basin, os.path.basename(path),
            os.path.basename(dst) if dst else "(did not exist)"))


def load_bank(path: str) -> list:
    """Read a JSONL bank that may be MID-WRITE by a concurrent screen or
    refinement. Exactly one unparsable FINAL line is tolerated as a torn
    in-flight write (reported, dropped — it will be complete on the next
    read); an unparsable line anywhere else is corruption and refuses
    with its line number."""
    rows = []
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:
                print("load_bank: dropped one torn in-flight final line "
                      "(concurrent writer) — re-read later for it")
                break
            raise ValueError(
                "bank {0} line {1} is corrupt (not a torn tail): {2}"
                .format(path, i + 1, e)) from e
    return rows
