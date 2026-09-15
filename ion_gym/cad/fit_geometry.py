"""
ion_gym.fit_geometry
--------------------
Recover a primitive GeomScene from per-electrode node masks (the stored/folded
field-array frame). This is how Rung 1 avoids asserting any geometry dimension from
memory: the ORIGINAL node-centred grid is the source of the dimensions, the emitted
The reconstruction is the fitted scene, and the audit closes the loop
node-exactly.

Method, per electrode:
  1. z-run segmentation: group consecutive z slices with identical occupancy
     patterns. (primitive geometry is piecewise-constant in z for
     z-aligned solids — exactly the quad's structure.)
  2. classify each run's 2-D pattern, trying in order:
       full rectangle              -> Box3D
       disc                        -> Cylinder
       annulus (concentric)        -> Cylinder minus Cylinder (within/notin)
       rectangle with disc hole    -> Box3D minus Cylinder
       rectangle with rect hole    -> Box3D minus Box3D
       disjoint components         -> recurse per component, union
  3. parameter snapping: every fitted surface parameter (radius, box edge,
     z plane) is chosen so that either it passes EXACTLY through node
     coordinates (the inclusive on-surface rule then matches our
     rasterizer's) or it keeps a finite margin from every node. Snap tiers
     prefer round values in gu — 1, 0.5, 0.25, 0.1, 0.05 — inside the
     feasible open interval (max inside distance, min outside distance);
     the interval midpoint is the fallback. z-extents snap to the integer
     node planes of the run (face exactly on the boundary node plane — the
     convention the original quad source used: detector face exactly at 474).
  4. verification: rasterize the fitted GeomScene and require node-exact
     equality with the input masks, per electrode and combined. A fit that
     does not verify is REPORTED and refused, never silently emitted.

Circle fits are brute-force over sub-grid center candidates with local
refinement — patterns are 39x39-ish, so exhaustive search is cheap and
guarantees the feasibility interval is honest (no local-minimum surprises).
Half-disc patterns from mirror folding fit naturally: the center simply
lands on (or beyond) a mirror axis and the quadrant clips it.
"""

import numpy as np
from scipy import ndimage

from ion_gym.physics.scene3d import GeomScene, GridSpec, Electrode, Shape, Cylinder, Box3D
from ion_gym.physics.rasterize3d import rasterize

SNAP_TIERS = (1.0, 0.5, 0.25, 0.1, 0.05)


# --------------------------------------------------------------- snapping
def _snap(lo, hi):
    """Pick a value in [lo, hi) (surface-inclusive at lo): prefer round gu
    values; allow exact lo (on-node surface) if lo itself is round."""
    if not (hi > lo - 1e-9):
        return None
    for t in SNAP_TIERS:
        # candidates: multiples of t in [lo - eps, hi - margin)
        k0 = np.ceil((lo - 1e-9) / t)
        k1 = np.floor((hi - 1e-6) / t)
        if k1 >= k0:
            ks = np.arange(k0, k1 + 1) * t
            return float(ks[np.argmin(np.abs(ks - 0.5 * (lo + hi)))])
    return 0.5 * (lo + hi)


def _snap_center(feas, cx, cy):
    """Try snapped (cx, cy) candidates tier by tier — BOTH floor and ceil
    multiples per axis (nearest-only rounding can jump the wrong way, e.g.
    0.55 -> 1.0 when 0.0 is the feasible round center). feas(cx, cy) returns
    a margin (> 0 feasible) — the largest-margin feasible snapped center of
    the coarsest feasible tier wins; falls back to the unsnapped center."""
    for t in SNAP_TIERS:
        cands = {(f(cx / t) * t, g(cy / t) * t)
                 for f in (np.floor, np.ceil, np.round)
                 for g in (np.floor, np.ceil, np.round)}
        best = max(((feas(a, b), a, b) for a, b in cands), key=lambda v: v[0])
        if best[0] > 1e-9:
            return float(best[1]), float(best[2])
    return cx, cy


def _fit_circle(P, region=None):
    """Fit (cx, cy, r) so that {nodes with dist <= r} == P within `region`
    (bool mask of nodes the circle is responsible for; default: everywhere).
    Returns (cx, cy, r) or None. Surface-inclusive: r may equal the max
    inside distance exactly (nodes on the surface are electrode)."""
    if region is None:
        region = np.ones_like(P, bool)
    ins = np.argwhere(P & region).astype(float)
    if len(ins) == 0:
        return None
    out = np.argwhere(~P & region).astype(float)
    x0, y0 = ins.min(0)
    x1, y1 = ins.max(0)

    def interval(cx, cy):
        di = np.sqrt((ins[:, 0] - cx) ** 2 + (ins[:, 1] - cy) ** 2)
        lo = di.max()
        if len(out):
            do = np.sqrt((out[:, 0] - cx) ** 2 + (out[:, 1] - cy) ** 2)
            hi = do.min()
        else:
            hi = lo + 1.0
        return lo, hi

    best = None
    for step, span in ((0.25, None), (0.05, 0.3), (0.01, 0.06)):
        if span is None:
            cxs = np.arange(x0 - 2, x1 + 2 + step, step)
            cys = np.arange(y0 - 2, y1 + 2 + step, step)
        else:
            bcx, bcy = best[1], best[2]
            cxs = np.arange(bcx - span, bcx + span + step / 2, step)
            cys = np.arange(bcy - span, bcy + span + step / 2, step)
        for cx in cxs:
            for cy in cys:
                lo, hi = interval(cx, cy)
                m = hi - lo
                if best is None or m > best[0]:
                    best = (m, cx, cy, lo, hi)
        if best[0] <= 0 and span is None:
            pass  # keep refining; margin may appear
    m, cx, cy, lo, hi = best
    if m <= 1e-9:
        return None
    cx, cy = _snap_center(lambda a, b: (lambda l, h: h - l)(*interval(a, b)),
                          cx, cy)
    lo, hi = interval(cx, cy)
    return (cx, cy, _snap(lo, hi))


def _fit_annulus(P):
    """Concentric annulus: all True dists in [ri, ro]; all False dists
    outside. Center search shares the circle machinery via feasibility of a
    two-radius split."""
    ins = np.argwhere(P).astype(float)
    if len(ins) == 0:
        return None
    out = np.argwhere(~P).astype(float)
    x0, y0 = ins.min(0); x1, y1 = ins.max(0)
    best = None
    for step, span in ((0.25, None), (0.05, 0.3), (0.01, 0.06)):
        if span is None:
            cxs = np.arange(x0 - 2, x1 + 2 + step, step)
            cys = np.arange(y0 - 2, y1 + 2 + step, step)
        else:
            cxs = np.arange(best[1] - span, best[1] + span + step / 2, step)
            cys = np.arange(best[2] - span, best[2] + span + step / 2, step)
        for cx in cxs:
            for cy in cys:
                di = np.sqrt((ins[:, 0] - cx) ** 2 + (ins[:, 1] - cy) ** 2)
                do = np.sqrt((out[:, 0] - cx) ** 2 + (out[:, 1] - cy) ** 2)
                tlo, thi = di.min(), di.max()
                f_in = do[do < tlo]          # hole nodes
                f_out = do[do > thi]         # exterior nodes
                if len(f_in) + len(f_out) != len(do):
                    continue                  # a False node inside the band
                ri_lo = f_in.max() if len(f_in) else 0.0
                ro_hi = f_out.min() if len(f_out) else thi + 1.0
                m = min(tlo - ri_lo, ro_hi - thi)
                if best is None or m > best[0]:
                    best = (m, cx, cy, ri_lo, tlo, thi, ro_hi)
        if best is None:
            return None
    m, cx, cy, ri_lo, tlo, thi, ro_hi = best
    if m <= 1e-9 or ri_lo <= 0:
        return None

    def feas(a, b):
        di = np.sqrt((ins[:, 0] - a) ** 2 + (ins[:, 1] - b) ** 2)
        do = np.sqrt((out[:, 0] - a) ** 2 + (out[:, 1] - b) ** 2)
        tlo_, thi_ = di.min(), di.max()
        if ((do >= tlo_) & (do <= thi_)).any():
            return -1.0
        f_in = do[do < tlo_]
        f_out = do[do > thi_]
        ril = f_in.max() if len(f_in) else 0.0
        roh = f_out.min() if len(f_out) else thi_ + 1.0
        return min(tlo_ - ril, roh - thi_) if ril > 0 else -1.0

    cx, cy = _snap_center(feas, cx, cy)
    di = np.sqrt((ins[:, 0] - cx) ** 2 + (ins[:, 1] - cy) ** 2)
    do = np.sqrt((out[:, 0] - cx) ** 2 + (out[:, 1] - cy) ** 2)
    tlo, thi = di.min(), di.max()
    ri_lo = do[do < tlo].max() if (do < tlo).any() else 0.0
    ro_hi = do[do > thi].min() if (do > thi).any() else thi + 1.0
    ri = _snap(ri_lo, tlo)          # inner surface: hole boundary
    ro = _snap(thi, ro_hi)          # outer surface
    # inner radius semantics: notin cylinder of radius r removes nodes with
    # dist <= r, so we need ri strictly BELOW the innermost True distance and
    # at-or-above the outermost hole distance -> snap in [ri_lo, tlo) then
    # nudge: r_notin must satisfy dist<=r for hole nodes only.
    if ri is None or ro is None:
        return None
    return (cx, cy, ri, ro)


def _classify_pattern(P):
    """Classify a 2-D bool pattern -> list of (within_desc, notin_desc)
    tuples, where each desc is ('circle', cx, cy, r) or
    ('rect', x1, y1, x2, y2). Returns None if unclassifiable."""
    ii, jj = np.where(P)
    x1, x2 = int(ii.min()), int(ii.max())
    y1, y2 = int(jj.min()), int(jj.max())
    bbox = np.zeros_like(P)
    bbox[x1:x2 + 1, y1:y2 + 1] = True

    # 1) full rectangle
    if (P == bbox).all():
        return [(("rect", x1, y1, x2, y2), None)]

    # 2) disc
    c = _fit_circle(P)
    if c:
        cx, cy, r = c
        # verify against the full pattern (not just bbox): recompute
        X, Y = np.meshgrid(np.arange(P.shape[0], dtype=float),
                           np.arange(P.shape[1], dtype=float), indexing="ij")
        if (((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2 + 1e-9) == P).all():
            return [(("circle", cx, cy, r), None)]

    # 3) rectangle with a hole
    hole = bbox & ~P
    if hole.any() and (P | hole == bbox).all():
        hi, hj = np.where(hole)
        # 4a) disc hole (fit circle responsible only for the bbox region)
        c = _fit_circle(hole, region=bbox)
        if c:
            cx, cy, r = c
            X, Y = np.meshgrid(np.arange(P.shape[0], dtype=float),
                               np.arange(P.shape[1], dtype=float),
                               indexing="ij")
            rec = bbox & ~((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2 + 1e-9)
            if (rec == P).all():
                return [(("rect", x1, y1, x2, y2), ("circle", cx, cy, r))]
        # 4b) rect hole
        hx1, hx2 = int(hi.min()), int(hi.max())
        hy1, hy2 = int(hj.min()), int(hj.max())
        hbox = np.zeros_like(P)
        hbox[hx1:hx2 + 1, hy1:hy2 + 1] = True
        if ((hole == hbox).all()):
            return [(("rect", x1, y1, x2, y2), ("rect", hx1, hy1, hx2, hy2))]

    # 4) annulus (tried AFTER rect-with-hole: a
    #    full-width plate with a round hole is node-identical to a huge
    #    annulus, but the rect is the true outer surface)
    a = _fit_annulus(P)
    if a:
        cx, cy, ri, ro = a
        X, Y = np.meshgrid(np.arange(P.shape[0], dtype=float),
                           np.arange(P.shape[1], dtype=float), indexing="ij")
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        if (((d2 <= ro ** 2 + 1e-9) & ~(d2 <= ri ** 2 + 1e-9)) == P).all():
            return [(("circle", cx, cy, ro), ("circle", cx, cy, ri))]

    # 5) disjoint components -> recurse
    labl, n = ndimage.label(P)
    if n > 1:
        parts = []
        for k in range(1, n + 1):
            sub = _classify_pattern(labl == k)
            if sub is None:
                return None
            parts += sub
        return parts
    return None


def _desc_to_prims(desc, z_hi, length):
    """(within_desc, notin_desc) at a z-run -> Shape. Rect descs become boxes
    spanning [z_lo, z_hi] with faces exactly on node planes; circles become
    z-cylinders on the pinned [z - L, z] convention."""
    (w, nn) = desc
    def prim(d):
        if d[0] == "circle":
            _, cx, cy, r = d
            return Cylinder(cx, cy, z_hi, r, length)
        _, a1, b1, a2, b2 = d
        return Box3D(a1, b1, z_hi - length, a2, b2, z_hi)
    sh = Shape(within=[prim(w)])
    if nn is not None:
        # notin solids get a half-gu z overhang so the subtraction never
        # leaves surface-tie ambiguity at the run's end planes
        d = prim(nn)
        if isinstance(d, Cylinder):
            d = Cylinder(d.cx, d.cy, d.z + 0.5, d.r, d.length + 1.0)
        else:
            d = Box3D(d.x1, d.y1, d.z1 - 0.5, d.x2, d.y2, d.z2 + 0.5)
        sh.notin.append(d)
    return sh


def fit_electrode(mask):
    """mask (nx,ny,nz) bool -> list[Shape], or raise with a slice report."""
    nz = mask.shape[2]
    [z for z in range(nz) if mask[:, :, z].any()]
    shapes, z = [], 0
    runs = []
    z = 0
    while z < nz:
        if not mask[:, :, z].any():
            z += 1
            continue
        z0 = z
        while z + 1 < nz and (mask[:, :, z + 1] == mask[:, :, z0]).all():
            z += 1
        runs.append((z0, z))
        z += 1
    for (z0, z1) in runs:
        P = mask[:, :, z0]
        descs = _classify_pattern(P)
        if descs is None:
            lines = ["".join(".#"[int(v)] for v in row) for row in P.T[::-1]]
            raise ValueError(
                f"unclassifiable slice pattern at z-run [{z0},{z1}]:\n"
                + "\n".join(lines))
        for d in descs:
            shapes.append(_desc_to_prims(d, float(z1), float(z1 - z0)))
    return shapes


def fit_scene(masks, grid: GridSpec, names=None, voltages=None,
              name="fitted_scene", notes="") -> GeomScene:
    """masks: {index: bool (nx,ny,nz)} -> verified GeomScene (node-exact).
    Raises if the reconstruction does not reproduce the masks exactly."""
    names = names or {}
    voltages = voltages or {}
    els = [Electrode(index=i, name=names.get(i, f"electrode_{i}"),
                     voltage=voltages.get(i, 0.0),
                     shapes=fit_electrode(m))
           for i, m in sorted(masks.items())]
    sc = GeomScene(grid=grid, electrodes=els, units="gu",
               name=name, notes=notes).check()
    lab = rasterize(sc)
    bad = {}
    for i, m in masks.items():
        d = int(((lab == i) != m).sum())
        if d:
            bad[i] = d
    if bad:
        raise ValueError(f"fit did not verify node-exactly: "
                         f"{{index: bad-node count}} = {bad}")
    return sc
