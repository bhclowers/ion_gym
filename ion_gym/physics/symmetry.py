"""
ion_gym.symmetry
----------------
The declare-and-verify symmetry system. A geometry declares two things,
kept deliberately separate because conflating them causes silent wrong
fields:

  coords : how the geometry is DESCRIBED — 'rz' (2-D cylindrical: x is the
           axis, y is radius >= 0) or 'xyz' (Cartesian, 2-D if depth=0
           else 3-D). This is a coordinate choice, not an assertion.

  planes : per-axis SYMMETRY ASSERTIONS the solver may exploit to shrink
           the domain — a map {axis: kind} with kind in
             'none'          - no symmetry; solve the full extent on it
             'mirror'        - reflection symmetry about the axis midplane
                               (the planar path folds on this only
                               through the anchored-grid machinery; the
                               solver's own mirror flag is a proven
                               no-op there)
             'translational' - the field is invariant along this axis
                               (a long straight section); solve one layer
                               and extrude
           Default is 'none' on every axis -> always-correct full solve.

CRITICAL: a declared plane is an ASSERTION, not a hint. verify_symmetry()
proves each declared mirror/translational plane against the actual
electrode masks AND (for RF) the phase pattern, and REFUSES with a clear
error if it doesn't hold. So "declare two mirror planes for a long quad"
is supported and safe: assert them, the builder checks them, and only
then reduces the solve. Break a plane (segmented rod, tilted pusher) and
the check fails loudly instead of yielding a clean-looking wrong field.

This is why the coarse cylindrical|planar|none enum was replaced: 'the
geometry is in r-z' (a coordinate choice) and 'the midplane is a mirror'
(an assertion the solver must prove) are different statements that the
old enum smashed together.
"""

# SCHEMA HOMECOMING: SymmetrySpec is spec
# DATA (coords/planes/plane_mm declarations, no solving) and lives
# with the other *Spec classes in io.sim_spec; this module keeps
# the VERIFY machinery and imports the schema from its owner —
# physics standing on io, the declared direction.
from ion_gym.io.sim_spec import SymmetrySpec, AXES


import numpy as np

def _axis_index(axis):
    return {"x": 0, "y": 1, "z": 2}[axis]


def verify_mirror_mask(mask, axis, atol_frac=0.005):
    """Is `mask` (2-D or 3-D bool array) mirror-symmetric about the
    midplane of `axis`? Returns (ok, mismatch_fraction). The midplane is
    the array-centre; an odd extent mirrors about the centre row/col, an
    even extent about the boundary between the two central rows."""
    ax = _axis_index(axis)
    if ax >= mask.ndim:
        return True, 0.0
    flipped = np.flip(mask, axis=ax)
    diff = mask ^ flipped
    frac = diff.sum() / max(mask.sum(), 1)
    return frac <= atol_frac, float(frac)


def verify_translational_mask(mask, axis, atol_frac=0.005):
    """Is `mask` invariant along `axis` (every layer identical)? Returns
    (ok, mismatch_fraction) comparing each layer to the first."""
    ax = _axis_index(axis)
    if ax >= mask.ndim:
        return True, 0.0
    first = np.take(mask, 0, axis=ax)
    n = mask.shape[ax]
    total = 0
    for k in range(1, n):
        total += (np.take(mask, k, axis=ax) ^ first).sum()
    frac = total / max(mask.sum(), 1)
    return frac <= atol_frac, float(frac)


def verify_symmetry(sym: SymmetrySpec, electrode_masks, rf_groups=None,
                    atol_frac=0.005):
    """Prove every declared mirror/translational plane against the actual
    geometry. electrode_masks: {index: bool array} (same shape). rf_groups
    (optional): list of bool arrays, one per (freq,phase) RF group — each
    must ALSO satisfy the declared symmetry for the reduction to be valid
    on the RF field, since paired rods at opposite phase break a naive
    mirror unless the pairing respects the plane.

    Returns (ok, report) where report is a list of
    (axis, kind, ok, mismatch_fraction, scope) tuples. Raises nothing —
    the caller decides whether to reduce or refuse."""
    sym = sym.normalized()
    # union mask (all metal) for geometric checks
    total = None
    for m in electrode_masks.values():
        total = m.copy() if total is None else (total | m)
    report = []
    all_ok = True
    for axis in AXES:
        kind = sym.kind(axis)
        if kind == "none":
            continue
        checker = (verify_mirror_mask if kind == "mirror"
                   else verify_translational_mask)
        ok, frac = checker(total, axis, atol_frac)
        report.append((axis, kind, ok, frac, "geometry"))
        all_ok &= ok
        # RF fields must respect the plane too (phase-aware)
        if rf_groups:
            for gi, gmask in enumerate(rf_groups):
                gok, gfrac = checker(gmask, axis, atol_frac)
                report.append((axis, kind, gok, gfrac, f"rf_group{gi}"))
                all_ok &= gok
    return all_ok, report


# ------------------------------------- shape tier: exact, on shapes.
# A NATIVE geometry's declared mirror is
# verified against its CONTINUOUS shapes, exactly. The mask-level check
# above remains for imports only (imported geometry has no inline shapes) -- it measures
# the raster, and CLOSED-EDGE FLOAT NOISE (edges at binary-inexact
# multiples of h drop/keep boundary nodes asymmetrically) puts a
# one-node skin on symmetric
# rect pairs (4.9% on the transaxial mirror's boards at h=0.08), so a
# fixed mask fraction can never
# be an exactness criterion. Shapes can. Translational planes stay on the
# mask check: a finite shape set cannot witness translational invariance.

_ROUND = 9      # canonicalization decimals == the contract's 1e-9 mm eps


def _canon_params(p):
    """Order-free, rounded canonical form of a params dict."""
    out = []
    for k in sorted(p):
        v = p[k]
        if isinstance(v, (int, float)):
            out.append((k, round(float(v), _ROUND)))
        elif isinstance(v, (list, tuple)):
            out.append((k, tuple(
                tuple(round(float(c), _ROUND) for c in q)
                if isinstance(q, (list, tuple)) else round(float(q), _ROUND)
                for q in v)))
        else:
            out.append((k, str(v)))
    return tuple(out)


def _canon_shape(shape, axis=None, plane=None):
    """Canonical tuple of a ShapeSpec; if axis/plane are given, of its
    mirror image about axis=plane. Raises on a shape type it cannot
    reflect -- refusing beats silently skipping the shape."""
    p = dict(shape.params)
    if axis is not None:
        t = shape.type
        if t == "rect":
            lo_key, sz_key = (("x_mm", "width_mm") if axis == "x"
                              else ("y_mm", "height_mm"))
            lo = float(p.get(lo_key, 0.0))
            sz = float(p.get(sz_key, 0.0))
            p[lo_key] = 2.0 * plane - (lo + sz)
        elif t == "ellipse":
            c_key = "cx_mm" if axis == "x" else "cy_mm"
            p[c_key] = 2.0 * plane - float(p.get(c_key, 0.0))
        elif t == "polygon":
            key = "points_mm" if "points_mm" in p else "points"
            pts = p.get(key)
            if pts is None:
                raise ValueError("polygon shape without points cannot be "
                                 "reflected -- refusing to verify by "
                                 "omission")
            k = 0 if axis == "x" else 1
            p[key] = [[2.0 * plane - float(q[i]) if i == k else float(q[i])
                       for i in range(len(q))] for q in pts]
        elif t in ("cutout", "group"):
            pass               # container: children carry the geometry
        else:
            raise ValueError(
                f"shape type {t!r}: no exact reflection rule -- refusing "
                f"to verify a declared mirror against a shape I cannot "
                f"reflect (extend _canon_shape)")
    kids = tuple(sorted(_canon_shape(c, axis, plane)
                        for c in shape.children))
    return (shape.type, _canon_params(p), kids)


def _electrical_id(el):
    """What must match for two mirrored electrodes to carry the same
    FIELD: potential and drive. Positional symmetry at a different
    voltage/phase is not field symmetry."""
    return (round(float(getattr(el, "dc", 0.0)), _ROUND),
            getattr(el, "rf_group", None),
            bool(getattr(el, "is_grid", False)))


def verify_symmetry_shapes(sym: SymmetrySpec, electrodes, extents):
    """Shape tier: prove every declared MIRROR plane against the continuous
    shapes, exactly.

    electrodes: iterable of ElectrodeSpec-like (name, shapes, dc,
    rf_group, is_grid). extents: {axis: extent_mm} for midline defaults.

    The multiset of (electrical_id, shape-multiset) must be invariant
    under reflection about the declared plane -- a bijection in which each
    electrode's mirror image is an electrode at the same potential/drive
    (possibly itself). Returns (ok, report); report rows are
    (axis, kind, ok, detail, scope). Raises nothing -- the caller decides
    refuse vs proceed."""
    report = []
    all_ok = True
    for axis in AXES:
        if sym.kind(axis) != "mirror":
            continue
        plane = sym.plane(axis, extents.get(axis, 0.0))
        orig, refl = [], []
        for el in electrodes:
            eid = _electrical_id(el)
            orig.append((eid, tuple(sorted(_canon_shape(s)
                                           for s in el.shapes))))
            refl.append((eid, tuple(sorted(_canon_shape(s, axis, plane)
                                           for s in el.shapes))))
        ok_axis = sorted(orig) == sorted(refl)
        if ok_axis:
            detail = f"exact about {axis}={plane:g} mm"
        else:
            bad = [f"{el.name!r}" for el, (eo, ro) in
                   zip(electrodes, zip(sorted(orig), sorted(refl)))
                   if eo != ro][:4]
            detail = (f"shapes are not mirror-images about {axis}="
                      f"{plane:g} mm at matching potentials/drives "
                      f"(first diffs near: {', '.join(bad) or 'n/a'})")
        report.append((axis, "mirror", ok_axis, detail, "shapes"))
        all_ok &= ok_axis
    return all_ok, report


def plane_rows(sym: SymmetrySpec, axis, coords, extent_mm, tol_mm=1e-6):
    """Where the declared plane on `axis` sits in a coordinate array.

    THE consumer accessor (gates, display): replaces every per-gate guess
    of "the symmetry row" (argmin against a source position was the
    slim_tetramer defect). coords: 1-D sorted node coordinates (mm).

    Returns ("node", j)           -- a node lies on the plane (|c-p|<tol);
            ("straddle", (j,j+1)) -- the plane falls between nodes j,j+1;
            raises if the plane lies outside coords' span (a declaration
            outside the sampled domain is a spec error, not a lookup miss).
    For a mirror-symmetric phi the normal E-component is antisymmetric, so
    a "straddle" consumer asserts the PAIR cancels (interpolated value at
    the plane), never a single row."""
    c = np.asarray(coords, float)
    p = float(sym.plane(axis, extent_mm))
    if not (c[0] - tol_mm <= p <= c[-1] + tol_mm):
        raise ValueError(f"declared {axis}-plane at {p:g} mm lies outside "
                         f"the sampled span [{c[0]:g}, {c[-1]:g}] mm")
    j = int(np.argmin(np.abs(c - p)))
    if abs(c[j] - p) < tol_mm:
        return ("node", j)
    j0 = j if c[j] < p else j - 1
    return ("straddle", (j0, j0 + 1))


def assert_mirror_field_symmetry(*, axis_index, plane_node, potentials,
                                 normal_E, context):
    """The physics-level symmetry gate, run after EVERY mirrored
    build (the enforcement layer between the
    loader's lattice refusal and the pair-agreement acceptance rule).

    A mirrored solve unfolds (or reflects) its stored half about a plane
    node. If the machinery is right, two things hold EXACTLY — not
    approximately, because reflection is a permutation and the central
    difference of a symmetric array at its own plane is (a - a)/2h:

      1. every solved field array is bit-symmetric about `plane_node`
         along `axis_index`;
      2. the E-field component NORMAL to the plane is identically zero
         on the plane row/slab.

    This is measured, never assumed, because a real quarter/full
    disagreement once hid for a long time: nothing ever checked the
    plane the mirror machinery claimed to be using.

    potentials: iterable of (name, ndarray) — the unfolded potentials
               (full arrays; the gate slices the plane itself).
    normal_E:  iterable of (name, ndarray) — the E component ALONG the
               mirrored axis, ALREADY SLICED to the plane row/slab by
               the caller (e.g. EAy[:, p, :] for a y-mirror at node p).
               The caller slices because only it knows its channel
               layout; the gate owns the criterion.
    Raises ValueError naming the context, axis, array, and the measured
    violation; returns None on pass (silent pass is fine here — the
    caller announces the gate ran)."""
    ax = int(axis_index)
    p = int(plane_node)
    for name, arr in potentials:
        n = arr.shape[ax]
        if not (0 <= p < n):
            raise ValueError(
                f"{context}: mirror-plane node {p} outside axis "
                f"{'xyz'[ax]} of {name} (length {n}) — the plane the "
                f"gate was handed is not on this lattice")
        if p != n - 1 - p:
            raise ValueError(
                f"{context}: plane node {p} is not the centre of axis "
                f"{'xyz'[ax]} ({n} nodes) for {name} — the unfolded "
                f"domain is not node-symmetric about its own plane "
                f"(the L-213 class: extent and plane disagree)")
        if not np.array_equal(arr, np.flip(arr, axis=ax)):
            d = np.abs(arr - np.flip(arr, axis=ax))
            raise ValueError(
                f"{context}: potential {name} is NOT bit-symmetric "
                f"about node {p} on axis {'xyz'[ax]} — max asymmetry "
                f"{float(d.max()):.3e} at {int((d > 0).sum())} nodes. "
                f"The mirrored build machinery reconstructed a field "
                f"that is not the mirror of its half; refusing to hand "
                f"it to a flight.")
    for name, row in normal_E:
        row = np.asarray(row)
        if row.size and float(np.abs(row).max()) != 0.0:
            nz = int((row != 0).sum())
            raise ValueError(
                f"{context}: {name} (the E component normal to the "
                f"{'xyz'[ax]}-mirror) is NOT identically zero on plane "
                f"node {p}: max |E| = {float(np.abs(row).max()):.6e} "
                f"at {nz}/{row.size} nodes. A symmetric potential has "
                f"an exactly-zero central difference on its own plane, "
                f"so any nonzero here is the mirror machinery breaking "
                f"symmetry (the L-212 artifact class); refusing.")


def reduction_summary(sym: SymmetrySpec):
    """Human-readable description of the solve-domain reduction a verified
    symmetry permits."""
    sym = sym.normalized()
    parts = []
    for axis in AXES:
        k = sym.kind(axis)
        if k == "mirror":
            parts.append(f"{axis}: fold to half")
        elif k == "translational":
            parts.append(f"{axis}: single layer + extrude")
    return "; ".join(parts) if parts else "full domain (no reduction)"
