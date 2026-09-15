"""
ion_gym.raster2d — THE 2-D raster substrate (extracted from
build_planar).

build_planar accreted the route-AGNOSTIC machinery every 2-D-raster
route stands on, because planar was the first native builder — the same
naming-lie class as a device-named module hosting the universal r-z
tracer. This module is the honest home:

  * shape rasterizer: electrode_mask + rect/ellipse/polygon primitives
    (_POLY_EDGE_TOL is the shared edge convention)
  * grid: anchored_grid / _anchor_mm / _anchor_pitch / plane_grid_views
  * symmetry folding of the raster: planar_fold_axes (+ crop/reflect)
  * label decomposition: el_masks_from_labels
  * THE 2-D metal predicate: _metal_nn — the impact rule the planar and
    r-z tracers share — and refuse_birth_in_metal built on it

Every function is moved VERBATIM (byte-preserving; flight replay proven
byte-equal at extraction). Consumers import from HERE — build_planar
included; no re-exports (the solve_bases lesson: hidden indirections
break silently under cleanup).
"""
import math

import numpy as np
from numba import njit

# ---------------------------------------------------- shape rasterization
# Boundary tolerance for the closed-solid rule, in mm. Sized to swallow
# float noise (a 180 deg rotation puts sin at 1.2e-16, so a mirrored vertex
# lands ~1e-15 mm off its ideal position) while staying astronomically
# below any grid pitch, so it can never absorb a real cell.
# Sanity ceiling for a 2-D rasterizer grid, in nodes. 100x the largest
# grid any example has ever used (a coarse SLIM is ~0.2M; surround SLIM
# ~2M): past this, per-electrode bool masks alone reach multi-GB and the
# spec's pitch is almost certainly a typo. A refused build must NAME the
# numbers — on some numpy builds the failed allocation surfaces as a
# bare SystemError from tupleobject.c instead of a readable MemoryError
# (seen on a Kingdon-trap deck with a pitch typo).
_GRID_MAX_NODES_2D = 200_000_000


def _rect_mask(X, Y, p):
    # CLOSED-EDGE WITH TOLERANCE: the
    # closed rule (node on the fill edge = electrode node, the
    # convention) evaluated with _POLY_EDGE_TOL, the SAME edge tolerance
    # _polygon_mask has carried all along — rect/ellipse never got it,
    # and the miss was the mirror trio's 4.9% "skin": an edge at a
    # binary-inexact multiple of h classifies its boundary node by float
    # noise (node 240 -> 19.2+3e-15 fails <=19.2). The tolerance moves
    # every edge outward by 1e-9 mm — astronomically below any pitch, so
    # it can never absorb a real cell, and boundary classification
    # becomes noise-independent and reflection-exact.
    T = _POLY_EDGE_TOL
    x0, y0 = p["x_mm"], p["y_mm"]
    w, h = p["width_mm"], p["height_mm"]
    rot = p.get("rotation_deg", 0.0)
    cx, cy = x0 + w / 2, y0 + h / 2
    if rot:
        a = math.radians(-rot)
        ca, sa = math.cos(a), math.sin(a)
        xr = cx + (X - cx) * ca - (Y - cy) * sa
        yr = cy + (X - cx) * sa + (Y - cy) * ca
    else:
        xr, yr = X, Y
    return ((xr >= x0 - T) & (xr <= x0 + w + T)
            & (yr >= y0 - T) & (yr <= y0 + h + T))


def _ellipse_mask(X, Y, p):
    # Closed-edge with tolerance: radii inflate by _POLY_EDGE_TOL —
    # the same "edge moved outward by T mm" semantics as rect/polygon.
    # NOTE cutout convention unchanged: a cutout's inner shape inflates
    # identically, so the subtracted region also grows by T; boundary
    # OWNERSHIP (metal vs cutout) stays exactly what it was under exact
    # arithmetic — the tolerance only removes noise-dependence.
    T = _POLY_EDGE_TOL
    cx, cy = p["cx_mm"], p["cy_mm"]
    rx, ry = p["rx_mm"], p["ry_mm"]
    return (((X - cx) / (rx + T)) ** 2
            + ((Y - cy) / (ry + T)) ** 2 <= 1.0)


def _shape_mask(shape, X, Y):
    """Render a ShapeSpec to a boolean mask. Additive shapes union; a
    cutout subtracts its inner shape from the accumulated mask (applied by
    the electrode assembler, which sees shape order)."""
    t = shape.type
    if t == "rect":
        return _rect_mask(X, Y, shape.params)
    if t == "ellipse":
        return _ellipse_mask(X, Y, shape.params)
    if t == "polygon":
        return _polygon_mask(X, Y, shape.params)
    raise ValueError(f"planar rasterizer: unsupported shape {t!r}")


_POLY_EDGE_TOL = 1e-9


def _polygon_mask(X, Y, p):
    """Rasterize a polygon as a CLOSED SOLID, reflection-exactly.

    WHY NOT matplotlib's Path.contains_points: its boundary rule depends on
    edge winding direction, so a polygon and its MIRROR IMAGE classify
    on-edge sample points differently. Any geometry whose faces land on cell
    centres (a rod face at x = 2.6 mm on an 0.08 mm grid, say) then produces
    masks that are asymmetric at the 1% level -- and verify_symmetry
    correctly REFUSES to fold, on a geometry that is in truth perfectly
    symmetric. The bug is in the rasteriser, not the geometry.

    The rule here:
        inside  =  strictly inside (even-odd crossing)  OR  on the boundary
    i.e. metal is a closed set -- the same convention flyer_plane already
    uses for electrode hits (tangent contact counts). "On the boundary" is
    decided by DISTANCE to the edge segments, and distance is invariant
    under reflection, so the mask is reflection-exact by construction and
    independent of vertex order or winding direction.
    """
    pts = np.asarray(p["points_mm"], float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError(f"polygon needs >=3 (x,y) points_mm, got "
                         f"shape {pts.shape}")
    x = X.ravel()
    y = Y.ravel()
    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)

    # ---- even-odd crossing test (strict interior) ----------------------
    inside = np.zeros(x.shape, bool)
    for j in range(len(pts)):
        ax, ay, bx, by = x0[j], y0[j], x1[j], y1[j]
        if ay == by:
            continue
        straddles = (ay > y) != (by > y)
        with np.errstate(invalid="ignore", divide="ignore"):
            xint = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= straddles & (x < xint)

    # ---- boundary band (closes the solid; mirror-invariant) ------------
    ex, ey = x1 - x0, y1 - y0
    seg2 = ex * ex + ey * ey
    d2 = np.full(x.shape, np.inf)
    for j in range(len(pts)):
        if seg2[j] == 0.0:
            dj2 = (x - x0[j]) ** 2 + (y - y0[j]) ** 2
        else:
            t = ((x - x0[j]) * ex[j] + (y - y0[j]) * ey[j]) / seg2[j]
            t = np.clip(t, 0.0, 1.0)
            px = x0[j] + t * ex[j]
            py = y0[j] + t * ey[j]
            dj2 = (x - px) ** 2 + (y - py) ** 2
        d2 = np.minimum(d2, dj2)

    return (inside | (d2 <= _POLY_EDGE_TOL ** 2)).reshape(X.shape)


def plane_grid_views(as_, bs_, what="grid"):
    """Full-shape (na, nb) coordinate views over two 1-D axes, ZERO-COPY
    (np.broadcast_to). Replaces np.meshgrid for the 2-D rasterizers:
    electrode_mask's rect/ellipse tests are pure comparisons, so dense
    copies of the coordinate matrices were pure waste — and on a runaway
    grid the meshgrid allocation could die inside numpy internals with an
    unreadable SystemError. Refuses (with the numbers) past the sanity
    ceiling instead."""
    na = len(as_)
    nb = len(bs_)
    nodes = na * nb
    if nodes > _GRID_MAX_NODES_2D:
        raise MemoryError(
            f"{what}: {na} x {nb} = {nodes/1e6:.0f}M nodes exceeds the "
            f"2-D rasterizer sanity ceiling ({_GRID_MAX_NODES_2D/1e6:.0f}M)."
            f" Axis extents [{as_[0]:g}, {as_[-1]:g}] x "
            f"[{bs_[0]:g}, {bs_[-1]:g}] mm — check mm_per_gu (a too-small "
            f"pitch is the usual cause).")
    A = np.broadcast_to(np.asarray(as_, float)[:, None], (na, nb))
    B = np.broadcast_to(np.asarray(bs_, float)[None, :], (na, nb))
    return A, B


def electrode_mask(el, X, Y):
    """Compose an ElectrodeSpec's shapes in order: additive shapes union
    in, cutout shapes subtract out (ion_playground semantics).

    2-D ONLY: a shape carrying an `extrude` descriptor is REFUSED here —
    a planar (z-invariant) or r-z solve cannot honor a finite sub-range
    along a third axis, and rasterizing the cross-section while silently
    dropping the range would put metal where the spec says vacuum. The
    3-D consumer is build_shapes3d.shapes_masks_3d, which calls the
    per-type rasterizers (_shape_mask) directly with the descriptor
    already applied."""
    m = np.zeros(X.shape, bool)
    for s in el.shapes:
        inner = (s.children[0] if s.type == "cutout" and s.children
                 else None)
        for chk in (s, inner):
            if chk is not None and chk.extrude() is not None:
                e = chk.extrude()
                raise ValueError(
                    f"electrode {getattr(el, 'name', '?')!r}: shape "
                    f"{chk.type!r} carries extrude "
                    f"{e['axis']}=[{e['lo_mm']:g},{e['hi_mm']:g}] mm, "
                    f"which a 2-D build cannot honor. Extruded shapes "
                    f"require the 3-D route: coords 'xyz', depth_mm > 0, "
                    f"no STL electrodes (build_shapes3d).")
        if s.type == "cutout":
            if inner is not None:
                m &= ~_shape_mask(inner, X, Y)
        else:
            m |= _shape_mask(s, X, Y)
    return m


def el_masks_from_labels(ele, electrodes=None):
    """Per-electrode boolean masks derived from the labeled solve mask
    (int16, 0 = vacuum). THE display source for per-electrode fills and
    labels: derived from the very array the field was computed on, so
    display == solver input by construction. Returns {} for an unlabeled
    (boolean) ele rather than faking a single electrode — absence is
    honest, a wrong picture is not. (Native Planar/RZ models once
    never carried el_masks, so the browser showed no fills/labels on any
    native example.)

    `electrodes` (the spec's electrode list) keys the result by each
    electrode's DECLARED NAME instead of its integer label. The label is
    the 1-based index into that list (`enumerate(..., start=1)` in both
    the planar and r-z rasterizers), so the mapping is exact, not
    inferred. Without it the UI labelled electrodes "e1/e2/e3" while the
    deck called them "L1/L2/L3", and a reader could not match a shape in
    a figure to an entry in the deck — the identity half of "displayed
    values equal solver input". Omitting it keeps the integer
    keys, which the renderer still accepts, so no caller is forced to
    change at once.
    """
    import numpy as np
    ele = np.asarray(ele)
    if ele.dtype == bool or ele.max(initial=0) < 1:
        return {}
    out = {}
    for i in np.unique(ele):
        if i <= 0:
            continue
        k = int(i)
        if electrodes is not None:
            if not 1 <= k <= len(electrodes):
                # A label with no electrode behind it means the mask and
                # the spec disagree about how many electrodes exist. That
                # is a build defect, and silently falling back to "e{k}"
                # would hide it behind a plausible-looking label.
                raise IndexError(
                    f"el_masks_from_labels: solve mask carries label {k} "
                    f"but the spec declares {len(electrodes)} electrode(s). "
                    f"The mask and the spec disagree; the labels cannot be "
                    f"named.")
            k = electrodes[k - 1].name
        out[k] = (ele == i)
    return out


def _anchor_mm(spec, origin=(0.0, 0.0)):
    """(dx, dy): the sub-cell grid anchor RESIDUE per axis.
    For an axis with a DECLARED plane_mm, the node
    lattice is shifted by d = q - h*round(q/h), q = plane - origin
    (|d| <= h/2) so a node lies EXACTLY on the declared plane at any
    pitch — the raster becomes reflection-symmetric about it by
    construction. The residue is computed RELATIVE TO THE FRAME ORIGIN
    (signed frames): the plane must land on a node of the
    lattice that STARTS at the origin, so with origin (0,0) — every
    legacy spec — the arithmetic is byte-identical. Axes without a
    declared location anchor at 0.0. Deterministic in (plane_mm,
    origin_mm, mm_per_gu) only, all already in the geometry cache key."""
    g = spec.geometry
    h = float(g.mm_per_gu)
    sym = g.symmetry.normalized()
    out = []
    for a, o in zip(("x", "y"), origin):
        p = sym.plane_mm.get(a)
        if p is None:
            out.append(0.0)
        else:
            q = float(p) - float(o)
            out.append(q - h * round(q / h))
    return tuple(out)


def planar_fold_axes(sym, mask2d, xs, ys, extents, report=False):
    """THE fold authority — used by the builder AND by sizing, so the
    estimate is the solve that runs. Per mirror axis, folding requires:
      1. the declared plane lies ON a lattice node (guaranteed for
         declared plane_mm by the anchored grid; midline defaults only
         when the centre lands on a node),
      2. the domain is node-symmetric about that node (p == n-1-p; an
         asymmetric outer boundary breaks field symmetry because every
         edge carries the same even-reflection ghost),
      3. EVERY electrode's discrete mask is exactly mirror-equal about
         it (per-electrode: positional symmetry across different bases
         is not field symmetry).
    Returns {axis_index: plane_node}. Prints the reason for every
    declared-but-unfolded axis when report=True (a skipped optimisation
    is a reported decision)."""
    out = {}
    for ax_i, (a, coords) in enumerate((("x", xs), ("y", ys))):
        if sym.kind(a) != "mirror":
            continue
        # Midline-default
        # mirrors fold too, when the discrete preconditions below hold.
        # This moves undeclaring specs' fields at solver-tol level; that
        # movement is certified by the
        # fold==full known-answer in test_planar_fold.
        n = len(coords)
        h = float(coords[1] - coords[0]) if n > 1 else 1.0
        plane = sym.plane(a, extents[a])
        jf = (plane - float(coords[0])) / h
        p = int(round(jf))
        if abs(jf - p) > 1e-6 or not (0 <= p < n):
            if report:
                print(f"symmetry: {a}-mirror at {plane:g} mm is not on a "
                      f"lattice node (index {jf:.3f}) — solving full "
                      f"domain")
            continue
        if p != n - 1 - p:
            if report:
                print(f"symmetry: {a}-mirror plane node {p} is not the "
                      f"domain centre ({n} nodes) — outer boundaries "
                      f"would be asymmetric; solving full domain")
            continue
        # Per-basis discrete check, with the PAIRING distinction: a basis
        # whose flip equals ITSELF is foldable this slice; a basis whose
        # flip equals a DIFFERENT basis is a mirror PAIR — folding pairs
        # needs partner-derived bases (basis_lo == reflected basis_hi),
        # a flagged extension, not silently wrong zeros from an empty
        # cropped mask. Anything matching neither is raster asymmetry
        # (closed-edge float-noise skin).
        skew, pairs = [], []
        flips = {i: np.flip(m, axis=ax_i) for i, m in mask2d.items()}
        for i, m in mask2d.items():
            if np.array_equal(m, flips[i]):
                continue
            partner = [k for k, mk in mask2d.items()
                       if k != i and np.array_equal(mk, flips[i])]
            (pairs if partner else skew).append(i)
        if skew or pairs:
            if report:
                if skew:
                    print(f"symmetry: {a}-mirror declared and shape-"
                          f"exact, but discrete masks of bases {skew} "
                          f"are not mirror-equal — solving full domain "
                          f"(post-2b this indicates genuinely asymmetric "
                          f"rasterization, e.g. an asymmetric domain)")
                if pairs:
                    print(f"symmetry: {a}-mirror bases {pairs} mirror "
                          f"onto PARTNER bases, not themselves — pair-"
                          f"folding (partner-derived bases) is a flagged "
                          f"extension; solving full domain")
            continue
        out[ax_i] = p
    return out


def _fold_crop(arr, fold):
    """Crop a full-domain (nx, ny[, 1]) array to its solved half on every
    folded axis: keep [plane_node:] (the plane node becomes index 0).
    The SOLVER must then apply the even-reflection 'mirror' ghost on
    that edge (build_planar passes it per-edge) — the open-
    edge ghost_linear is NOT the mirror boundary; on a plane with
    transverse field curvature it injects an O(h^2 * phi'') BC error
    (measured 38 V on a real MRT median plane)."""
    for ax_i, p in fold.items():
        sl = [slice(None)] * arr.ndim
        sl[ax_i] = slice(p, None)
        arr = arr[tuple(sl)]
    return np.ascontiguousarray(arr)


def _fold_reflect(arr, fold):
    """Inverse of _fold_crop for a SOLVED half: rebuild the full domain
    by even reflection about the plane node (exact for a discretely
    symmetric problem; certified against the full solve in
    test_planar_fold)."""
    for ax_i, p in sorted(fold.items(), reverse=True):
        sl = [slice(None)] * arr.ndim
        sl[ax_i] = slice(p, 0, -1)          # p..1: the mirror image rows
        arr = np.concatenate([arr[tuple(sl)], arr], axis=ax_i)
    return np.ascontiguousarray(arr)


def anchored_grid(spec, pitch=None):
    """THE planar lattice authority: (xs, ys, anchor). Node i sits
    at i*h + anchor[axis]; anchored axes resize to cover the declared
    extent and node-symmetrize about the plane; un-anchored axes keep the
    historical round() count byte-identically. Used by the builder AND by
    sizing — duplicating this arithmetic is how the estimate and the
    solve drift apart (it happened once; measured, then
    unified here).

    SIGNED FRAME: a declared geometry.origin_mm [x_lo, y_lo]
    composes into the anchor, so node coordinates are stated in the
    deck's own (possibly signed) frame and the domain spans
    [lo, lo + extent]. The sub-pitch plane-anchoring residue is computed
    relative to the origin (a plane must land on a node of THIS lattice).
    origin None == (0, 0): every legacy grid is byte-identical."""
    from ion_gym.io.lattice import LATTICE_TOL_CELLS, conformance_error, gu_nodes
    g = spec.geometry
    h = float(pitch if pitch is not None else g.mm_per_gu)
    o = tuple(float(v) for v in (g.origin_mm or (0.0, 0.0)))
    sub = (_anchor_mm(spec, origin=o) if pitch is None
           else _anchor_pitch(spec, h, origin=o))

    # LATTICE RULE: a sub-pitch residue means a declared plane is NOT
    # on this lattice. The residue path below used to absorb that by
    # shifting the whole grid — plane on a node, slop smeared to BOTH
    # walls, and the actual node positions silently off the declared
    # frame. Symmetry planes are lattice quantities: refuse instead.
    # Conforming planes leave IEEE noise (|d| ~ 1e-15 mm) in `sub`; that
    # noise keeps flowing through the historical ceil arithmetic so every
    # conforming deck's grid stays byte-identical.
    sym = g.symmetry.normalized()
    for a, d in zip(("x", "y"), sub):
        if abs(d) > LATTICE_TOL_CELLS * h:
            pl = sym.plane_mm.get(a)
            raise ValueError(conformance_error(
                float(pl) - o[0 if a == "x" else 1], h, axis=a,
                what="symmetry-plane position (relative to the origin)"))

    def _n(extent, d, axis):
        # THE counting function: the conformance refusal runs on
        # EVERY axis (round+1 == cells+1, so conforming decks keep their
        # byte-identical count); the residual-`d` arithmetic below is
        # kept for the IEEE-noise case and yields the same count there.
        n_conf = gu_nodes(extent, h, axis=axis, what="domain extent")
        if d == 0.0:
            return n_conf
        return int(math.ceil((extent - d) / h - 1e-9)) + 1
    xs = np.arange(_n(g.width_mm, sub[0], "x")) * h + sub[0] + o[0]
    ys = np.arange(_n(g.height_mm, sub[1], "y")) * h + sub[1] + o[1]
    anchor = (sub[0] + o[0], sub[1] + o[1])
    return xs, ys, anchor


def _anchor_pitch(spec, h, origin=(0.0, 0.0)):
    """_anchor_mm at an explicit pitch (sizing sweeps pitches); same
    origin-relative residue (signed frames)."""
    sym = spec.geometry.symmetry.normalized()
    out = []
    for a, o in zip(("x", "y"), origin):
        pl = sym.plane_mm.get(a)
        if pl is None:
            out.append(0.0)
        else:
            q = float(pl) - float(o)
            out.append(q - h * round(q / h))
    return tuple(out)


@njit(cache=True, nogil=True)
def _metal_nn(ele, gx, gy, nx, ny):
    """Nearest-node metal test.
    The impact surface is the half-cell shell around metal nodes:
    index-INDEPENDENT (the old test bilinearly interpolated the LABEL
    array, so the kill zone scaled with an electrode's list position —
    idx 1 killed within h/2 of a face, idx 97 across ~99.5% of the
    adjacent cell), thin-electrode-safe (single-node metal stays opaque,
    unlike the 3-D all-corners rule), and the nearest-grid-unit
    convention. Outside the box: False (out-of-bounds is fate kind=1,
    tested separately)."""
    i = int(math.floor(gx + 0.5))
    j = int(math.floor(gy + 0.5))
    if i < 0 or j < 0 or i > nx - 1 or j > ny - 1:
        return False
    return ele[i, j] > 0.5


def refuse_birth_in_metal(ele, gx, gy, h, spec, i, pos_txt):
    """Nearest-node refusal: a birth inside the
    effective metal shell used to die silently at step 1 (the recorded
    r-z "launch bug" — it read as a velocity collapse with E=0 in the
    GUI). Refuse loudly, naming the ion, its user-frame position, and
    the electrode. Shared by every 2-D route wrapper (planar and r-z)
    so the convention cannot fork."""
    gi = int(math.floor(gx + 0.5))
    gj = int(math.floor(gy + 0.5))
    if (0 <= gi < ele.shape[0] and 0 <= gj < ele.shape[1]
            and ele[gi, gj] > 0):
        # labels are 1-based positions in the FULL electrode list (grid
        # electrodes keep their index but are never painted), so the name
        # map must count every electrode — counting only solids misnames
        # any solid that follows a grid in the list.
        names = {k: el.name
                 for k, el in enumerate(spec.geometry.electrodes, start=1)}
        lbl = int(ele[gi, gj])
        raise ValueError(
            f"ion {i} birth at {pos_txt} lies inside the effective "
            f"metal of electrode {names.get(lbl, lbl)!r} (nearest-node "
            f"rule, node pitch {h} mm): move the source or shrink its "
            f"radius")
