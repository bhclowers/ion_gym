"""
ion_gym.sizing
--------------
ONE solve-cost estimator, for every geometry source.

Before this module the cost model lived inside stl_upload -- so an STL got
a pitch control and an honest "this will cost you N GB and M minutes"
readout, and a JSON-loaded geometry got neither.  That asymmetry is not a
property of the physics; it is an accident of which import path was
written first.  The estimator only ever needed two things:

    * a set of axis-aligned bounding boxes (what geometry occupies)
    * how many conductors there are (how many bases must be solved)

Both are available from an STL mesh list, a scene3d.GeomScene, and a
sim_spec.SimSpec alike.  So the core takes exactly those, and each front
end supplies them.  A new geometry source gets a cost readout by writing
an ADAPTER, not another estimator.

DELIBERATELY makes no assumption about the kind of optic: no feature
detection, no "recommended" pitch.  The user sets the pitch; this reports
what it will cost.  Guessing a pitch for someone is how a 0.5 mm gap gets
silently resolved by two cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

import json
import math
import os
import platform
import threading
import time

BYTES_PER_NODE = 4                     # float32 basis
# MULTIGRID throughput: SOLVED nodes per second, to tolerance, end-to-end
# (V-cycles make "sweeps" meaningless as a unit). Since L-439 this is
# only the PRIOR for the multigrid routes, used until this machine has
# a recorded multigrid solve (see "per-machine solve rates" below); the
# 2-D LU routes never use it.
MG_NODES_PER_S = 2.2e5   # [measured, then re-measured
                         #  across three grids: 236k @1M nodes,
                         #  256k @2.7M, ~O(N) linear/plateauing. Toy
                         #  plates on the PI's machine 2026-10-15 ran
                         #  ~0.8e5 at 885k nodes, so this prior can be
                         #  short -- the quote says it is uncalibrated.]

# ------------------------------------------------ per-machine solve rates
# L-439. One multigrid rate priced every route, and each solver costs
# differently (toy grids, 4 electrodes, 2026-10-15):
#   2-D LU (planar, r-z): ONE factorisation per geometry + one cheap
#       back-solve (~2-3% of the factor) per electrode. Pricing every
#       electrode as a full solve quoted planar 7.5-10.5x long.
#   3-D multigrid (stl2d, stl3d, shapes3d, scene3d): a full solve per
#       electrode whose per-node speed RISES with N (1.4 s at 14k nodes,
#       11.6 s at 885k) and is not monotone in grid shape.
# A fixed constant is right on one machine at best, so the rates are
# MEASURED ON THIS MACHINE (PI ruling 2026-10-15):
#   * every LU factorisation (solver2d.laplace_factor) and every
#     multigrid basis (multigrid3d.solve_bases_mg) records its node
#     count and seconds -- the solves the user runs anyway;
#   * the quote scales the record NEAREST in size;
#   * LU has a sub-second PROBE (LU_PROBE_SIDES) run on the first LU
#     quote with no record; multigrid cannot be probed honestly (its
#     smallest solve is >1 s and does not predict a large one), so until
#     a multigrid solve is recorded the quote uses MG_NODES_PER_S and
#     says it is not yet calibrated.
# Records live in cache_root()/SOLVE_RATES_FILE, keyed by machine, so a
# cache folder shared between machines keeps each machine's own rates.
SOLVER_OF_ROUTE = {"planar": "lu", "rz": "lu", "stl2d": "mg",
                   "stl3d": "mg", "shapes3d": "mg", "scene3d": "mg"}
SOLVE_RATES_FILE = "solve_rates.json"
SOLVE_RATES_KEEP = 200        # most recent records kept per solver
SOLVE_RATES_MIN_NODES = 10_000  # smaller solves are timer noise
# Scaling exponents (up, down) from the nearest record to the quoted N:
# t = t_rec * (N / N_rec) ** exp, `up` when N > N_rec. Each pair rounds
# toward LONG: up uses the steepest local slope measured, down the
# shallowest (toy grids 2026-10-15: LU factor slope 1.07 below 122k
# nodes, 1.37 from 360k to 1M; back-solve ~1.0-1.2; multigrid 0.5
# overall, ~1.0 at the large end).
LU_FACTOR_EXP = (1.35, 1.0)
LU_BACK_EXP = (1.25, 1.0)
MG_BASIS_EXP = (1.0, 0.5)
LU_PROBE_SIDES = (150, 250, 350)   # square toy grids; ~0.5 s in all

_RATES_LOCK = threading.Lock()
_RATES = None                 # this machine's records, loaded lazily
_RATES_WRITE_WARNED = False


def _machine_key():
    return f"{platform.node()}|{platform.machine()}|{os.cpu_count()}"


def _rates_path():
    from ion_gym.io.paths import cache_root
    return os.path.join(str(cache_root()), SOLVE_RATES_FILE)


def _load_rates():
    """This machine's records, {"lu": [...], "mg": [...]}. A missing file
    is no records; an unreadable one is announced and then ignored --
    rates are an estimate input, never physics."""
    global _RATES
    if _RATES is None:
        _RATES = {"lu": [], "mg": []}
        path = _rates_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    mine = json.load(f).get(_machine_key(), {})
                for s in _RATES:
                    _RATES[s] = list(mine.get(s, []))
            except (OSError, ValueError, AttributeError) as e:
                print(f"[sizing] solve-rate records at {path} unreadable "
                      f"({e}); quoting as if this machine had none")
    return _RATES


def _store_rates():
    """Atomic rewrite of this machine's section. A failure is announced
    once per process and the records stay in memory: a cost estimate
    must never fail a solve."""
    global _RATES_WRITE_WARNED
    path = _rates_path()
    try:
        allm = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    allm = json.load(f)
            except (OSError, ValueError):
                allm = {}
        if not isinstance(allm, dict):
            allm = {}
        allm[_machine_key()] = _RATES
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(allm, f, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        if not _RATES_WRITE_WARNED:
            print(f"[sizing] solve-rate record not saved ({e}); this "
                  f"process keeps it in memory, the next starts over")
            _RATES_WRITE_WARNED = True


def record_solve(solver, n_nodes, **seconds):
    """Record one measured solve: solver "lu" with factor_s and back_s,
    or "mg" with basis_s (seconds per electrode basis). Called by the
    solver doors themselves, so every route records the same way."""
    if solver not in ("lu", "mg"):
        raise ValueError(f"record_solve: unknown solver {solver!r}; "
                         f"known: 'lu', 'mg'")
    want = {"lu": ("factor_s", "back_s"), "mg": ("basis_s",)}[solver]
    if set(seconds) != set(want):
        raise ValueError(f"record_solve({solver!r}): needs exactly "
                         f"{want}, got {sorted(seconds)}")
    n = int(n_nodes)
    if n < SOLVE_RATES_MIN_NODES:
        return
    rec = {"n": n, "when": time.strftime("%Y-%m-%d %H:%M")}
    rec.update({k: float(v) for k, v in seconds.items()})
    with _RATES_LOCK:
        recs = _load_rates()[solver]
        recs.append(rec)
        del recs[:-SOLVE_RATES_KEEP]
        _store_rates()


def lu_probe():
    """Time solver2d.laplace_factor on LU_PROBE_SIDES square toy grids
    (two plates). laplace_factor records each one itself; this only
    supplies the grids. ~0.5 s."""
    from ion_gym.physics.solver2d import laplace_factor
    for n in LU_PROBE_SIDES:
        fixed = np.zeros((n, n), bool)
        fixed[n // 3, n // 5: 4 * n // 5] = True
        fixed[2 * n // 3, n // 5: 4 * n // 5] = True
        v = np.zeros((n, n))
        v[n // 3, n // 5: 4 * n // 5] = 1.0
        laplace_factor(fixed, 0.1)(v)


def _scale(t, n, n_rec, exps):
    up, down = exps
    return t * (n / n_rec) ** (up if n > n_rec else down)


def _nearest(recs, n):
    """Record nearest to n in log size; the most recent wins a tie."""
    best = None
    for r in recs:
        d = abs(math.log(n / r["n"]))
        if best is None or d <= best[0]:
            best = (d, r)
    return best[1]


def solve_seconds(solver, n_solved, n_electrodes):
    """(seconds, provenance line) to solve n_electrodes bases on
    n_solved nodes with this solver, on THIS machine."""
    n = max(float(n_solved), 1.0)
    k = int(n_electrodes)
    if solver == "lu":
        with _RATES_LOCK:
            have = bool(_load_rates()["lu"])
        if not have:
            print("[sizing] no LU solve recorded on this machine; timing "
                  "a sub-second probe once")
            lu_probe()
        with _RATES_LOCK:
            r = _nearest(_load_rates()["lu"], n)
        t = (_scale(r["factor_s"], n, r["n"], LU_FACTOR_EXP)
             + k * _scale(r["back_s"], n, r["n"], LU_BACK_EXP))
        return t, (f"LU: one factorisation + {k} back-solve(s), scaled "
                   f"from this machine's {r['n']:,}-node solve "
                   f"({r['when']})")
    if solver == "mg":
        with _RATES_LOCK:
            recs = list(_load_rates()["mg"])
        if not recs:
            return k * n / MG_NODES_PER_S, (
                f"multigrid: NOT YET CALIBRATED on this machine -- "
                f"default {MG_NODES_PER_S:.3g} nodes/s per basis, which "
                f"can be several times short; the first 3-D solve here "
                f"calibrates it")
        r = _nearest(recs, n)
        return k * _scale(r["basis_s"], n, r["n"], MG_BASIS_EXP), (
            f"multigrid: {k} basis solve(s), scaled from this machine's "
            f"{r['n']:,}-node solve ({r['when']})")
    raise ValueError(f"solve_seconds: unknown solver {solver!r}; known: "
                     f"'lu' (planar, r-z), 'mg' (stl2d and the 3-D routes)")


# Per-node compose rates [measured on a 3.5M-node array].
# The compose runs np.gradient once per drive channel; float32 is ~4x the
# throughput of float64 at half the memory. The electrode-aware surface
# correction (njit _fa3d_correct) adds ~1 ns/node — negligible since the
# v198 compile, but stated so the two field methods price honestly.
COMPOSE_NS_PER_NODE = {"float64": 22.5, "float32": 5.0}
EAWARE_CORRECTION_NS_PER_NODE = 1.2


@dataclass
class SizingProposal:
    """What a solve at this pitch will cost.  Reported, never chosen."""
    domain_min: np.ndarray            # (3,) mm
    domain_max: np.ndarray
    pitch: float                      # mm/gu (USER-chosen)
    dims: tuple                       # (nx, ny, nz)
    n_voxels: int
    n_electrodes: int
    mem_bases_bytes: int
    mem_solver_bytes: int
    est_solve_s: float
    fold: str = ""                    # mirror axes the SOLVE halves (planar)
    min_feature_mm: float | None = None   # smallest gap the user told us about
    # mirror axes declared on a deck that IS already the reduced domain
    # (3-D routes, scenes): reported, never divided out again (L-565)
    mirror_stored: str = ""
    solver: str = "mg"                # "lu" (planar, r-z) | "mg" (the rest)
    rate_note: str = ""               # where est_solve_s's rate came from
    warnings: List[str] = field(default_factory=list)
    # field-build options (GeometrySpec.field_method / channel_dtype): they
    # change the COMPOSE cost and the channel memory, not the Laplace solve
    n_drive_channels: int = 0         # RF/time-domain groups + static DC
    field_method: str = "electrode_aware"
    channel_dtype: str = "float64"

    @property
    def n_voxels_solved(self) -> int:
        """Nodes actually solved, after the declared symmetry fold."""
        return int(self.n_voxels / (2 ** len(set(self.fold) & set("xyz"))))

    def markdown(self) -> str:
        gb = 1024 ** 3
        d = self.domain_max - self.domain_min
        L = [f"**domain:** {d[0]:.1f} × {d[1]:.1f} × {d[2]:.1f} mm",
             f"**pitch:** {self.pitch:g} mm/gu",
             f"**grid:** {self.dims[0]} × {self.dims[1]} × {self.dims[2]} "
             f"= {self.n_voxels:,} nodes"]
        if self.fold:
            L.append(f"**fold:** mirror {self.fold} → "
                     f"{self.n_voxels_solved:,} nodes solved "
                     f"({2 ** len(set(self.fold) & set('xyz'))}× reduction)")
        if self.mirror_stored:
            L.append(f"**mirror:** {self.mirror_stored} declared — this grid "
                     f"is already the reduced domain, solved as stored "
                     f"(no further fold)")
        L += [f"**memory:** bases {self.mem_bases_bytes / gb:.2f} GB "
              f"({self.n_electrodes} × float32) + solver "
              f"~{self.mem_solver_bytes / gb:.2f} GB",
              f"**est. solve:** ~{fmt_time(self.est_solve_s)}"
              + (f" ({self.rate_note})" if self.rate_note else "")]
        if self.n_drive_channels:
            it = np.dtype(self.channel_dtype).itemsize
            mem_ch = 3 * self.n_drive_channels * self.n_voxels * it
            ns = COMPOSE_NS_PER_NODE.get(self.channel_dtype,
                                         COMPOSE_NS_PER_NODE["float64"])
            if self.field_method == "electrode_aware":
                ns += EAWARE_CORRECTION_NS_PER_NODE
            t_comp = 3 * self.n_drive_channels * self.n_voxels * ns * 1e-9
            L.append(f"**channels:** {self.n_drive_channels} drive "
                     f"channel(s) × 3 components at {self.channel_dtype} = "
                     f"{mem_ch / gb:.2f} GB; est. compose ~"
                     f"{fmt_time(t_comp)} ({self.field_method}) — runs on "
                     f"every build, cached bases included")
        if self.min_feature_mm:
            c = self.min_feature_mm / self.pitch
            L.append(f"**smallest declared feature:** "
                     f"{self.min_feature_mm:g} mm = **{c:.1f} cells**")
        for w in self.warnings:
            L.append(f"⚠️ {w}")
        return "\n\n".join(L)


def fmt_time(s):
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def propose_sizing_from_aabbs(aabb_min: Sequence, aabb_max: Sequence,
                              n_electrodes: int, *, pitch: float,
                              margin_mm: float = 2.0, fold: str = "",
                              min_feature_mm: float | None = None,
                              solver: str = "mg", mirror_stored: str = ""
                              ) -> SizingProposal:
    """The core.  aabb_min/max: (N,3) arrays of per-electrode AABBs in mm.
    solver: "lu" (planar, r-z) or "mg" (stl2d and the 3-D routes, and an
    STL upload or scene, which build 3-D) -- SOLVER_OF_ROUTE.
    fold: axes the SOLVE halves out of this domain (planar only).
    mirror_stored: axes declared mirror on a domain that is ALREADY the
    reduced part (3-D routes, scenes) -- reported, not divided out."""
    lo = np.min(np.atleast_2d(aabb_min), axis=0) - margin_mm
    hi = np.max(np.atleast_2d(aabb_max), axis=0) + margin_mm
    p = float(pitch)
    if p <= 0:
        raise ValueError(f"pitch must be > 0, got {p}")
    # Sanitised HERE, above the lattice snap below, because that snap reads it
    # to decide which axes need an even cell count. It used to be cleaned
    # further down, next to its only other use.
    fold = "".join(c for c in fold if c in "xyz")

    # LATTICE CONFORMANCE AT THE SOURCE. The domain proposed here
    # becomes a deck's width/height/depth via stl_upload.spec_from_upload
    # (extents = domain_max - domain_min), so a raw AABB+margin span --
    # an arbitrary float -- produced a spec the lattice loader REFUSES. That
    # is not a test-fixture problem: every STL upload that accepted the
    # proposed sizing produced an unbuildable deck. Measured: the
    # quadrupole-rod fixtures proposed 24.0898, 28.3133 and 24.2949 mm
    # spans, all refused.
    #
    # The span is therefore snapped to an exact integer number of cells,
    # and the extent is DERIVED from that count through the lattice owner
    # (derived_extent_mm) rather than recomputed here -- one counting
    # authority.
    #
    # COVER-UP, never trim: `hi` moves out. An AABB is the metal's own
    # bounding box, so rounding the span DOWN would clip a conductor,
    # while rounding up only adds vacuum at the outer wall -- which is
    # exactly where the remainder belongs. `lo` is left alone: it
    # is the frame origin, not a lattice quantity, and moving it would
    # shift every mesh translation downstream.
    #
    # The tolerance guard matters: a span that is ALREADY conforming must
    # not gain a spurious cell from float noise, so a value within
    # LATTICE_TOL_CELLS of an integer count is taken as that count.
    from ion_gym.io.lattice import cover_extent_mm
    # `fold` names the axes this domain will be MIRRORED on, and a
    # mirrored axis needs an EVEN cell count so its midline plane lands
    # on a node — passed through rather than assumed, so the estimate
    # sizes the domain the builder will actually solve.
    _mirrored = set(fold)
    cells = []
    for i in range(3):
        ext = cover_extent_mm(hi[i] - lo[i], p,
                              mirrored="xyz"[i] in _mirrored)
        cells.append(int(round(ext / p)))
        hi[i] = lo[i] + ext

    # dims are NODES of the inclusive span [lo, hi] -- cells + 1 -- which
    # is now exact rather than floor()'d. The old floor(span/p)+1 silently
    # absorbed the remainder, so the proposal described a domain smaller
    # than the one it reported.
    dims = tuple(c + 1 for c in cells)
    nvox = int(np.prod(dims))
    nsolved = nvox / (2 ** len(set(fold)))

    mem_b = int(nvox * BYTES_PER_NODE * n_electrodes)
    mem_s = int(nsolved * BYTES_PER_NODE * 4)
    # SOLVE-TIME ESTIMATE: priced by THE SOLVER THAT RUNS, at this
    # machine's measured rate (solve_seconds; L-439). It was one
    # multigrid constant for every route, which priced the 2-D LU
    # routes' cheap per-electrode back-solves as full solves.
    est_s, rate_note = solve_seconds(solver, nsolved, n_electrodes)

    warns = []
    span = float(np.max(hi - lo))
    if span < 1.0:
        warns.append(f"total extent {span:.3g} mm — if authored in "
                     "metres/inches the units are wrong (ion_gym assumes mm)")
    if span > 2000.0:
        warns.append(f"total extent {span:.0f} mm — check units")
    if min_feature_mm and min_feature_mm / p < 3.0:
        warns.append(
            f"the smallest declared feature ({min_feature_mm:g} mm) is only "
            f"{min_feature_mm / p:.1f} cells at this pitch — it will be "
            f"smeared. Below ~3 cells the field through it is not resolved.")
    if mem_b > 8 * 1024 ** 3:
        warns.append(f"bases need {mem_b / 1024**3:.1f} GB — consider a "
                     f"coarser pitch, a declared symmetry fold, or fewer "
                     f"conductors")
    if not fold and not mirror_stored:
        warns.append("no symmetry fold declared — if the geometry has a "
                     "mirror plane, declaring it (and letting the solver "
                     "VERIFY it) halves the cost per plane")

    return SizingProposal(domain_min=lo, domain_max=hi, pitch=p, dims=dims,
                          n_voxels=nvox, n_electrodes=int(n_electrodes),
                          mem_bases_bytes=mem_b, mem_solver_bytes=mem_s,
                          est_solve_s=est_s, fold=fold,
                          min_feature_mm=min_feature_mm, warnings=warns,
                          solver=solver, rate_note=rate_note,
                          mirror_stored="".join(
                              c for c in mirror_stored if c in "xyz"))


# ----------------------------------------------------------- adapters
def propose_sizing_scene(scene, *, pitch: float, margin_mm: float = 0.0,
                         min_feature_mm: float | None = None):
    """scene3d.GeomScene -> SizingProposal.  Domain = the box the scene's
    stored GRID covers (GeomScene.grid_extent_mm) -- the domain the solver
    actually solves, vacuum margin included. Margin 0 by default: a
    GeomScene declares its box explicitly, unlike an STL, whose box we
    must infer.

    It priced the METAL box (extent_mm) instead, so a scene's vacuum margin
    was never quoted: fixture A at 0.25 mm/gu with a 1 mm margin stores
    49x49x41 = 98,441 nodes and was quoted 41x41x33 = 55,473 (L-567). A
    scene whose grid spans nothing REFUSES rather than quoting zero."""
    sc = scene.in_mm()
    lo, hi = sc.grid_extent_mm()
    if not np.all(np.asarray(hi) - np.asarray(lo) > 0.0):
        raise ValueError(
            f"propose_sizing_scene: the scene's grid "
            f"({sc.grid.nx}x{sc.grid.ny}x{sc.grid.nz} nodes) spans no "
            f"volume; grid it (GeomScene.at_resolution) before pricing it")
    # A scene's grid is the STORED/FOLDED array frame (scene3d), so its
    # mirror axes are already cut away: reported, never divided out again.
    # Passing them as `fold` under-quoted a mirrored scene 2x per axis
    # (L-565).
    return propose_sizing_from_aabbs([lo], [hi], len(sc.electrodes),
                                     pitch=pitch, margin_mm=margin_mm,
                                     mirror_stored=sc.grid.mirror,
                                     min_feature_mm=min_feature_mm)


def propose_sizing_simspec(spec, *, pitch: float | None = None,
                           min_feature_mm: float | None = None,
                           field_method: str | None = None,
                           channel_dtype: str | None = None):
    """sim_spec.SimSpec -> SizingProposal.  pitch defaults to the spec's own
    mm_per_gu, so a loaded JSON reports its cost AS AUTHORED before anyone
    changes anything. field_method/channel_dtype default to the spec's own
    values; pass overrides to price a STAGED selection before it is applied
    (the UI's selectors), so the estimate follows the control that changes
    the cost."""
    g = spec.geometry
    p = float(pitch if pitch is not None else g.mm_per_gu)
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([g.width_mm, g.height_mm, max(g.depth_mm, 0.0)])
    # FOLD TRUTH (2a): for the PLANAR route the estimate asks THE SAME
    # authority the builder uses (planar_fold_axes over the real anchored
    # masks), so node counts halve exactly and only when the solve will
    # actually fold — prerequisite-zero proved the old kind()=="mirror"
    # halving described a fold that never ran. Every other route is
    # named in the chain below with its own builder's truth.
    sn = g.symmetry.normalized()
    mirror_stored = ""
    # THE ROUTE comes from the one authority build_run dispatches on
    # (sim_build.build_route), and each route asks ITS OWN builder's fold
    # truth (L-522). This was a hand-built planar test with an `else` that
    # priced r-z and stl2d by the 3-D mirror rule -- so a declared mirror
    # halved their quote although neither builder folds.
    from ion_gym.physics.sim_build import build_route
    route = build_route(spec).builder
    if route == "planar":
        from ion_gym.physics.raster2d import (anchored_grid,
                                              planar_fold_axes,
                                              electrode_mask,
                                              plane_grid_views)
        xs, ys, _anchor = anchored_grid(spec, pitch=p)
        # plane_grid_views, not meshgrid: see raster2d.
        X, Y = plane_grid_views(xs, ys, "sizing estimate")
        m2 = {i: electrode_mask(e, X, Y)
              for i, e in enumerate(g.electrodes, start=1)}
        fx = planar_fold_axes(sn, m2, xs, ys,
                              {"x": g.width_mm, "y": g.height_mm})
        fold = "".join("xy"[a] for a in sorted(fx))
    elif route in ("stl2d", "rz"):
        # NEITHER BUILDER FOLDS. build_stl solves the full domain
        # (mirror=(False, False, False)); build_rz VERIFIES declared mirror
        # planes (refusing a false one) and then solves every basis on the
        # full (z, r) grid. Pricing a fold neither performs under-quoted
        # the solve by 2x per declared plane.
        fold = ""
    elif route in ("stl3d", "shapes3d", "scene3d"):
        # 3-D ROUTES SOLVE THE DECLARED GRID AS STORED (L-565). A declared
        # mirror on these routes means the deck IS the reduced domain:
        # build_stl3d_run solves _grid_from_spec (the declared extent)
        # and multigrid applies the mirror as a boundary at the face. So
        # nothing is divided out. This used to halve per declared axis
        # (and an earlier comment here claimed oa3d_quarter /
        # mirror_end_* were priced unfolded "while the builder folded
        # it" -- the builder never folds). Measured with no solve:
        # slim_surround_mini (mirror z) solves 81x41x17 = 56,457 nodes
        # and was quoted 28,228. The mirror axes come from the 3-D
        # builders' own authority, _declared_mirror_axes (a POSITIONAL
        # (mx, my, mz) tuple), and are reported, not credited.
        from ion_gym.physics.build_stl3d import _declared_mirror_axes
        fold = ""
        mirror_stored = "".join(
            a for a, m in zip("xyz", _declared_mirror_axes(spec)) if m)
    else:
        raise ValueError(
            f"propose_sizing_simspec: route {route!r} has no solve-cost "
            f"rule; the estimate knows planar, stl2d, rz, stl3d, shapes3d "
            f"and scene3d. Refusing to price it as some other route.")
    prop = propose_sizing_from_aabbs([lo], [hi], len(g.electrodes),
                                     pitch=p, margin_mm=0.0, fold=fold,
                                     min_feature_mm=min_feature_mm,
                                     solver=SOLVER_OF_ROUTE[route],
                                     mirror_stored=mirror_stored)
    # price the compose: drive channels = RF/time-domain groups + static DC
    prop.n_drive_channels = 1 + len(getattr(g, "rf_groups", []) or [])
    prop.field_method = str(field_method or getattr(
        g, "field_method", "electrode_aware"))
    prop.channel_dtype = str(channel_dtype or getattr(
        g, "channel_dtype", "float64"))
    return prop


# ---------------------------------------------------------------- record
# TRAJECTORY RECORD VOLUME. The solve cost above has been quoted before
# the fact for a long time; the RECORD had no such quote, and the four
# values that set it (t_max, dt, rec_every, n_ions) are all known before
# the first ion flies. A run that needs more memory than the machine has
# is arithmetic, not bad luck: 1000 ions x 400,000 samples x 12 channels
# x 8 bytes = 38 GB, which on a 32 GB laptop wedges the process in swap
# with the UI unable to answer. Quote it, warn on it, refuse past it.
#
# rec_every is a STORAGE decision, never a physics one: the kernel steps
# at dt regardless, and the per-step accumulators (path, collision count,
# time-integrated field and KE) are summed every step and merely REPORTED
# at sample times. Measured on the drift cell: rec_every 20 / 400 / 2000
# give identical tof, fate, impact position and collision count. Only
# quantities read back OFF the stored array -- a frequency from zero
# crossings, an MSD fit, the drawn shape of a path -- care about the
# sampling interval.
BASE_RECORD_COLS = 7          # t, x, y, z, vx, vy, vz -- the shared base
RECORD_BYTES_PER_VALUE = 8    # float64 record rows

# Thresholds as a FRACTION OF SYSTEM RAM, not fixed gigabytes: the same
# deck is routine on a workstation and fatal on a laptop, and a constant
# in gigabytes would be wrong on both. The absolute floors apply only
# when the machine will not report its memory.
RECORD_WARN_FRACTION = 0.15
RECORD_REFUSE_FRACTION = 0.40
RECORD_WARN_GB_FALLBACK = 2.0
RECORD_REFUSE_GB_FALLBACK = 6.0
# When the caller DECLARES a memory budget, that budget is the refusal
# ceiling and this is where the warning sits inside it. A machine-derived
# fraction is a guess about what else is running; a declared budget is
# the operator saying what this process may have, and it wins.
RECORD_WARN_OF_BUDGET = 0.5


def system_memory_gb():
    """Total physical memory in GB, or None if the platform will not say.

    None is returned, never a guess: a fabricated memory size would turn
    the refusal below into a number nobody can check.
    """
    try:
        import psutil
        return psutil.virtual_memory().total / (1024.0 ** 3)
    except ImportError:
        pass
    try:
        import os
        return (os.sysconf("SC_PHYS_PAGES")
                * os.sysconf("SC_PAGE_SIZE")) / (1024.0 ** 3)
    except (ValueError, OSError, AttributeError):
        return None


def _record_ceiling(budget_gb):
    """(budget_gb, ram_gb, warn_gb, refuse_gb): the ONE ceiling every
    record quote is tiered against, so a single-stage and a staged quote
    on the same machine and budget cannot disagree about what fits."""
    ram = system_memory_gb()
    if budget_gb is not None:
        budget_gb = float(budget_gb)
        if not (budget_gb > 0):
            raise ValueError(
                f"record_volume: budget_gb must be a positive number of "
                f"gigabytes, got {budget_gb!r}. Pass None to derive the "
                f"limit from system RAM instead of declaring one.")
        return (budget_gb, ram, budget_gb * RECORD_WARN_OF_BUDGET,
                budget_gb)
    if ram:
        return (None, ram, ram * RECORD_WARN_FRACTION,
                ram * RECORD_REFUSE_FRACTION)
    return (None, ram, RECORD_WARN_GB_FALLBACK, RECORD_REFUSE_GB_FALLBACK)


def _ceiling_line(budget_gb, ram, warn_gb, refuse_gb):
    if budget_gb is not None:
        return (f"**budget:** {budget_gb:.0f} GB declared for this run"
                + (f" (machine has {ram:.0f} GB)" if ram else "")
                + f" — warn at {warn_gb:.1f} GB, refuse at "
                  f"{refuse_gb:.1f} GB")
    if ram:
        return (f"**machine:** {ram:.0f} GB RAM — warn at "
                f"{warn_gb:.1f} GB, refuse at {refuse_gb:.1f} GB")
    return (f"**machine:** total RAM unavailable on this platform "
            f"— falling back to fixed limits (warn "
            f"{warn_gb:.1f} GB, refuse {refuse_gb:.1f} GB)")


def record_volume(spec, *, n_cols=None, budget_gb=None):
    """What THIS run will store, before it stores any of it.

    spec   : the SimSpec about to be flown.
    n_cols : record width. Pass the built width when known (len(cols));
             otherwise it is derived as BASE_RECORD_COLS plus the
             declared record_channels, and the result says it assumed.
    budget_gb : memory this run may use, DECLARED by the operator. It
             replaces the machine-fraction ceiling entirely -- on a
             48 GB workstation running nothing else, 32 GB is a
             perfectly good answer and no fraction of total RAM will
             ever produce it. Warning then sits at
             RECORD_WARN_OF_BUDGET of the budget. None = derive from
             system RAM as before.

    Returns a dict: samples_per_ion, n_ions, n_cols, bytes, gb, ram_gb,
    tier ("ok" | "warn" | "refuse"), lines (human-readable quote), and
    levers (what to change, with the arithmetic). Computes and reports;
    it never mutates the spec and never refuses on its own -- the caller
    decides what a tier means, so the same estimate serves the sizing
    readout, the fly guard and a notebook.
    """
    integ, src = spec.integration, spec.source
    dt_us = float(integ.dt_ns) * 1e-3
    if dt_us <= 0:
        raise ValueError(f"record_volume: dt_ns must be positive, got "
                         f"{integ.dt_ns!r}")
    rec_every = max(int(getattr(integ, "rec_every", 1) or 1), 1)
    steps = float(integ.t_max_us) / dt_us
    wanted = int(steps // rec_every) + 1
    # max_records CAPS what the kernel stores, so it caps what this
    # estimate may claim. Without it the quote was the number of samples
    # the flight WOULD produce, not the number it will keep -- 4x too
    # large on a deck whose cap binds, which would refuse a run that fits.
    # A quote that is not what the solver does is the exact defect this
    # guard exists to catch.
    cap = int(getattr(integ, "max_records", 0) or 0)
    samples = min(wanted, cap) if cap > 0 else wanted
    truncated = cap > 0 and wanted > cap
    assumed = n_cols is None
    if assumed:
        n_cols = BASE_RECORD_COLS + len(
            list(getattr(integ, "record_channels", []) or []))
    n_ions = int(getattr(src, "n_ions", 0) or 0)
    n_mz = max(len(list(getattr(src, "mz_list", []) or [])), 1)
    n_flown = n_ions * n_mz          # n_ions is PER m/z
    total = samples * int(n_cols) * RECORD_BYTES_PER_VALUE * n_flown
    gb = total / (1024.0 ** 3)

    budget_gb, ram, warn_gb, refuse_gb = _record_ceiling(budget_gb)
    tier = ("refuse" if gb >= refuse_gb else
            "warn" if gb >= warn_gb else "ok")

    per_ion_mb = samples * int(n_cols) * RECORD_BYTES_PER_VALUE / (1024.0 ** 2)
    lines = [
        f"**trajectory record:** {n_flown:,} ion(s) x {samples:,} samples "
        f"x {n_cols} channels x {RECORD_BYTES_PER_VALUE} B = "
        f"**{gb:.2f} GB** ({per_ion_mb:.1f} MB per ion)"
        + ("  [record width assumed from record_channels; the built "
           "width may differ]" if assumed else ""),
        f"**sampling:** every {rec_every} steps of {integ.dt_ns:g} ns = "
        f"{rec_every * float(integ.dt_ns):g} ns per stored sample, over "
        f"{integ.t_max_us:g} us",
    ]
    if truncated:
        covered_us = samples * rec_every * float(integ.dt_ns) / 1e3
        lines.append(
            f"**TRUNCATED:** max_records={cap:,} caps the record, so only "
            f"the first {covered_us:,.0f} us of the {integ.t_max_us:g} us "
            f"flight is stored ({wanted:,} samples wanted). The size above "
            f"is what is KEPT. Raise rec_every to cover the whole flight "
            f"at the same cost, or raise max_records to keep more.")
    lines.append(_ceiling_line(budget_gb, ram, warn_gb, refuse_gb))

    levers = []
    if tier != "ok":
        for factor in (5, 20, 100):
            levers.append(
                f"rec_every {rec_every} -> {rec_every * factor} "
                f"({rec_every * factor * float(integ.dt_ns):g} ns per sample): "
                f"{gb / factor:.2f} GB")
        levers.append(f"n_ions {n_ions} -> {max(n_ions // 10, 1)}: "
                      f"{gb / 10:.2f} GB")
        levers.append(
            "rec_every is STORAGE ONLY — the integrator steps at dt "
            "either way, and tof, fate, impact position, collision count "
            "and the time-integrated channels are identical at any "
            "rec_every (measured). Only quantities read back off the "
            "stored path (a frequency from zero crossings, an MSD fit, "
            "the drawn shape) need fine sampling.")
    return dict(samples_per_ion=samples, samples_wanted=wanted,
                truncated=truncated, max_records=cap,
                n_ions=n_flown, n_cols=int(n_cols),
                bytes=int(total), gb=gb, ram_gb=ram, budget_gb=budget_gb,
                tier=tier,
                warn_gb=warn_gb, refuse_gb=refuse_gb,
                lines=lines, levers=levers, assumed_width=assumed)


# A staged packet KEEPS x, y, z, t per region for its drawn traces only
# (staged_flight.fly_packet); every other ion's region records live for
# that ion's flight and are dropped, so they are not a retained cost.
STAGED_TRACE_COLS = 4


def staged_record_volume(regions, n_traces, *, budget_gb=None):
    """What a staged packet flight will KEEP, before it flies.

    regions  : the loaded staged_flight Regions (dt_ns, t_max_us,
               rec_every, max_records -- the stage overrides already
               applied by load_assembly).
    n_traces : how many ions keep their path (trace_keep_count).

    Each region stores at most min(t_max/dt // rec_every + 1,
    max_records) samples per ion -- the same per-region count
    record_volume uses -- and a kept trace holds STAGED_TRACE_COLS of
    them in every region it crosses. Upper bound: a trace is cut at
    detection and an ion may never reach later regions. Tiered against
    the same ceiling as record_volume and returns the same keys, so one
    guard renders both.
    """
    n = int(n_traces)
    per_region, samples = [], 0
    for rg in regions:
        dt_us = float(rg.dt_ns) * 1e-3
        if dt_us <= 0:
            raise ValueError(f"staged_record_volume: region {rg.name!r} "
                             f"dt_ns must be positive, got {rg.dt_ns!r}")
        rec_every = max(int(rg.rec_every), 1)
        wanted = int(float(rg.t_max_us) / dt_us // rec_every) + 1
        cap = int(rg.max_records)
        s = min(wanted, cap) if cap > 0 else wanted
        per_region.append((rg.name, s, rec_every))
        samples += s
    total = samples * STAGED_TRACE_COLS * RECORD_BYTES_PER_VALUE * n
    gb = total / (1024.0 ** 3)
    budget_gb, ram, warn_gb, refuse_gb = _record_ceiling(budget_gb)
    tier = ("refuse" if gb >= refuse_gb else
            "warn" if gb >= warn_gb else "ok")
    lines = [
        f"**staged trajectory record:** {n:,} kept trace(s) x {samples:,} "
        f"samples over {len(per_region)} region(s) x {STAGED_TRACE_COLS} "
        f"channels x {RECORD_BYTES_PER_VALUE} B = **{gb:.2f} GB** "
        f"(upper bound)",
        "**per region:** " + ", ".join(
            f"{nm} {s:,} (rec_every {re})" for nm, s, re in per_region),
        _ceiling_line(budget_gb, ram, warn_gb, refuse_gb),
    ]
    levers = []
    if tier != "ok":
        big = max(per_region, key=lambda p: p[1])
        levers.append(f"raise rec_every on stage {big[0]!r} (it holds "
                      f"{big[1]:,} of the {samples:,} samples per trace)")
        levers.append(f"lower the instrument's trace fraction/floor: "
                      f"{n:,} traces -> {max(n // 10, 1):,} is "
                      f"{gb / 10:.2f} GB")
        # NOT the single-stage "storage only" lever: staged detect
        # crossings are found post-hoc on the stored records, so a
        # coarser rec_every can move an interpolated arrival.
    return dict(samples_per_ion=samples, n_ions=n,
                n_cols=STAGED_TRACE_COLS, bytes=int(total), gb=gb,
                ram_gb=ram, budget_gb=budget_gb, tier=tier,
                warn_gb=warn_gb, refuse_gb=refuse_gb,
                lines=lines, levers=levers, per_region=per_region)
