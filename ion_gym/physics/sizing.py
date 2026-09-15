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

BYTES_PER_NODE = 4                     # float32 basis
# MULTIGRID throughput: SOLVED nodes per second, to tolerance, end-to-end
# (V-cycles make "sweeps" meaningless as a unit). This is the ONE
# solve-cost constant — the solver is solve_bases_mg everywhere.
MG_NODES_PER_S = 2.2e5   # [measured, then re-measured
                         #  across three grids: 236k @1M nodes,
                         #  256k @2.7M, ~O(N) linear/plateauing. 2.2e5 is
                         #  the conservative floor — the estimate rounds
                         #  toward slightly-long, never short.]


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
    fold: str = ""                    # declared mirror axes actually used
    min_feature_mm: float | None = None   # smallest gap the user told us about
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
        L += [f"**memory:** bases {self.mem_bases_bytes / gb:.2f} GB "
              f"({self.n_electrodes} × float32) + solver "
              f"~{self.mem_solver_bytes / gb:.2f} GB",
              f"**est. solve:** ~{fmt_time(self.est_solve_s)}"]
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
                              min_feature_mm: float | None = None
                              ) -> SizingProposal:
    """The core.  aabb_min/max: (N,3) arrays of per-electrode AABBs in mm."""
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
    # SOLVE-TIME ESTIMATE: the actual solver is
    # MULTIGRID (solve_bases_mg), whose end-to-end throughput is
    # MG_NODES_PER_S (measured, ~O(N) linear: 236k @1M nodes, 256k @2.7M,
    # plateauing — remeasured across three grids). est_s
    # previously used a stale SOR pair (62.5k nodes/s/basis), 3.5x slower
    # than the multigrid solver that actually runs — so a 274 M-voxel /
    # ~39-electrode import estimated 47.5 h when the real multigrid cost
    # is ~13.5 h. The dead SOR
    # constants were removed with this fix: one solver, one constant, and
    # the estimate now describes the solve that runs.
    est_s = n_electrodes * nsolved / MG_NODES_PER_S

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
    if not fold:
        warns.append("no symmetry fold declared — if the geometry has a "
                     "mirror plane, declaring it (and letting the solver "
                     "VERIFY it) halves the cost per plane")

    return SizingProposal(domain_min=lo, domain_max=hi, pitch=p, dims=dims,
                          n_voxels=nvox, n_electrodes=int(n_electrodes),
                          mem_bases_bytes=mem_b, mem_solver_bytes=mem_s,
                          est_solve_s=est_s, fold=fold,
                          min_feature_mm=min_feature_mm, warnings=warns)


# ----------------------------------------------------------- adapters
def propose_sizing_scene(scene, *, pitch: float, margin_mm: float = 0.0,
                         min_feature_mm: float | None = None):
    """scene3d.GeomScene -> SizingProposal.  Domain from the scene's OWN mm
    extent (margin 0 by default: a GeomScene declares its box explicitly, unlike
    an STL, whose box we must infer)."""
    sc = scene.in_mm()
    lo, hi = sc.extent_mm()
    return propose_sizing_from_aabbs([lo], [hi], len(sc.electrodes),
                                     pitch=pitch, margin_mm=margin_mm,
                                     fold=sc.grid.mirror,
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
    # halving described a fold that never ran. Non-planar routes keep the
    # conservative rule (mirror kinds minus plane_mm axes);
    # their builders' fold truth is UNVERIFIED for those routes —
    # not guessed here.
    sn = g.symmetry.normalized()
    is_planar_route = (sn.coords == "xyz" and float(g.depth_mm or 0.0) == 0
                       and g.electrodes
                       and all(e.shapes for e in g.electrodes)
                       and not any(getattr(e, "stl", None)
                                   for e in g.electrodes))
    if is_planar_route:
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
    else:
        # NON-PLANAR ROUTES: ask THE SAME authority the 3-D builders
        # ask. This was
        #     kind(a) == "mirror" and a not in sn.plane_mm
        # — a conservative rule, justified above on the
        # grounds that the non-planar builders' fold truth was
        # UNVERIFIED. That premise no longer holds:
        # build_stl3d._declared_mirror_axes is documented as THE one
        # authority for declared mirror planes on the 3-D routes, and
        # physics.symmetry.assert_mirror_field_symmetry now proves the
        # fold on every 3-D build.
        #
        # The excluded-by-plane_mm clause was not merely conservative,
        # it was WRONG: _declared_mirror_axes reads scene.grid.mirror or
        # symmetry.planes and NEVER reads plane_mm, so a deck that
        # DECLARES plane_mm was priced as if it would not fold while the
        # builder folded it. Measured on this tree, that mis-sized three
        # of the seven mirrored 3-D decks — oa3d_quarter (y,z),
        # mirror_end_half (y), mirror_end_quarter (y,z) — by a factor of
        # 2 per axis, and those are exactly the decks a user prices
        # before committing to a long solve.
        #
        # Configuration-agnostic: every 3-D route (shapes3d, stl3d,
        # scene3d) reaches build_stl3d_run through this one function, so
        # the estimate and the build now read the same fact from the same
        # place and cannot drift.
        from ion_gym.physics.build_stl3d import _declared_mirror_axes
        # Returns a POSITIONAL (mx, my, mz) tuple of bools, not axis
        # names — zipped against "xyz" here rather than assumed.
        fold = "".join(a for a, m in zip("xyz", _declared_mirror_axes(spec))
                       if m)
    # price the compose: drive channels = RF/time-domain groups + static DC
    prop = propose_sizing_from_aabbs([lo], [hi], len(g.electrodes),
                                     pitch=p, margin_mm=0.0, fold=fold,
                                     min_feature_mm=min_feature_mm)
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

    ram = system_memory_gb()
    if budget_gb is not None:
        budget_gb = float(budget_gb)
        if not (budget_gb > 0):
            raise ValueError(
                f"record_volume: budget_gb must be a positive number of "
                f"gigabytes, got {budget_gb!r}. Pass None to derive the "
                f"limit from system RAM instead of declaring one.")
        warn_gb, refuse_gb = budget_gb * RECORD_WARN_OF_BUDGET, budget_gb
    elif ram:
        warn_gb, refuse_gb = (ram * RECORD_WARN_FRACTION,
                              ram * RECORD_REFUSE_FRACTION)
    else:
        warn_gb, refuse_gb = RECORD_WARN_GB_FALLBACK, RECORD_REFUSE_GB_FALLBACK
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
    if budget_gb is not None:
        lines.append(f"**budget:** {budget_gb:.0f} GB declared for this run"
                     + (f" (machine has {ram:.0f} GB)" if ram else "")
                     + f" — warn at {warn_gb:.1f} GB, refuse at "
                       f"{refuse_gb:.1f} GB")
    elif ram:
        lines.append(f"**machine:** {ram:.0f} GB RAM — warn at "
                     f"{warn_gb:.1f} GB, refuse at {refuse_gb:.1f} GB")
    else:
        lines.append(f"**machine:** total RAM unavailable on this platform "
                     f"— falling back to fixed limits (warn "
                     f"{warn_gb:.1f} GB, refuse {refuse_gb:.1f} GB)")

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
