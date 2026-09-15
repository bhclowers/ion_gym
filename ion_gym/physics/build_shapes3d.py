"""
ion_gym.build_shapes3d
----------------------
Solve INLINE SHAPES with extrusion (axis + range + mirror) natively in 3-D.
No imported geometry, no STL, no external solver: geometry lives in the spec JSON as 2-D
cross-sections (rect / ellipse / polygon, ShapeSpec) each carrying an
optional extrusion descriptor
    "extrude": {"axis": "x"|"y"|"z", "lo_mm": float, "hi_mm": float}
and rasterizes straight to the labeled voxel masks the shared 3-D builder
wants. This is the third mask PROVIDER beside stl_masks_3d (meshes) and
scene_masks_3d (analytic CSG): all three hand the SAME masks dict to the
SAME build_stl3d_run, so a fix in one cannot silently special-case
another.

CONVENTIONS:
  * Cross-section plane = the two axes perpendicular to `axis` in CYCLIC
    order — x->(y,z), y->(z,x), z->(x,y). The shape's own x/y params read
    as the (first, second) in-plane coordinate. The cross-section is
    rasterized by build_planar._shape_mask — the ONE 2-D authority
    (closed-solid rule, mirror-exact polygons, edge tolerance) — on
    zero-copy plane_grid_views; no meshgrid (Kingdon SystemError).
  * Shapes are authored in the STORED frame. On a declared mirror axis
    the stored frame is the non-negative half with the mirror plane at 0
    supplying the image (e.g. a half-gap SLIM deck stores z in [0,2],
    walls at [1.375,1.575], unfolds to +/-). lo_mm < 0 on a mirrored axis is
    REFUSED — a shape asymmetric about its own declared plane is a
    contradiction, and auto-folding would be a silent guess. In-plane
    coordinates that miss the stored domain are caught by the
    empty-electrode refusal below (parity with scene_masks_3d).
  * Absent descriptor = full-depth slab along DEFAULT_EXTRUDE_AXIS ("z",
    the out-of-plane axis of an x-y-authored cross-section) spanning the
    whole stored depth. Named default, per the no-buried-constants rule.
  * Mirror/frame: the solve stores the half; build_stl3d unfolds and sets
    mirror_off_mm = -0.5*(n-1)*h per mirrored axis, so display AND flight
    read the canonical [-H,+H] frame with the plane at 0 — both from the
    start, not display-only as on the earlier import fix.

Gates: tests/test_shape_extrusion.py (extrusion-range, axis-generality,
default-unchanged, mirror-about-0, cutout-volume).

Provenance: extrusion semantics are stated here in full. A planar shape
in one principal plane is swept
along the third axis between lo_mm and hi_mm inclusive, producing a prism
whose cross-section is the shape and whose axis is normal to that plane;
the sweep does not scale or rotate the cross-section. Frames follow the
canonical [-H,+H] convention.
"""
from __future__ import annotations

import json

import numpy as np

from ion_gym.io.sim_spec import SimSpec

# The out-of-plane axis assumed for a shape with NO extrude descriptor in
# a 3-D shapes build: x-y cross-sections extruded through the full stored
# depth. This is the one named default of the module; everything else is
# declared in the spec.
DEFAULT_EXTRUDE_AXIS = "z"

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _inplane_axes(axis_index):
    """Cyclic in-plane axis pair for an extrusion axis:
    x->(y,z), y->(z,x), z->(x,y)."""
    return (axis_index + 1) % 3, (axis_index + 2) % 3


def _shape_volume(shape, coords, mirror_axes, extents_mm, *, el_name,
                  shape_pos):
    """(nx,ny,nz) bool occupancy of ONE shape: its 2-D cross-section
    (build_planar._shape_mask, the single 2-D authority) filled along its
    extrusion range. Refusals name the electrode, shape, axis, and the
    numbers that did not line up."""
    from ion_gym.physics.raster2d import (_POLY_EDGE_TOL, _shape_mask, plane_grid_views)
    ext = shape.extrude()
    if ext is None:
        ai = _AXIS_INDEX[DEFAULT_EXTRUDE_AXIS]
        lo, hi = 0.0, extents_mm[ai]
    else:
        ai = _AXIS_INDEX[ext["axis"]]
        lo, hi = ext["lo_mm"], ext["hi_mm"]
        if mirror_axes[ai] and lo < 0.0:
            raise ValueError(
                f"electrode {el_name!r} shape #{shape_pos} "
                f"({shape.type!r}): extrude {ext['axis']}=[{lo:g},{hi:g}] "
                f"mm has lo_mm < 0 on the declared mirror axis "
                f"{ext['axis']!r}. The stored frame on a mirrored axis is "
                f"the non-negative half with the plane at 0 supplying the "
                f"image; author the [0,+] half (a symmetric-through-plane "
                f"shape is [0, hi]).")
        if hi < 0.0 or lo > extents_mm[ai]:
            raise ValueError(
                f"electrode {el_name!r} shape #{shape_pos} "
                f"({shape.type!r}): extrude {ext['axis']}=[{lo:g},{hi:g}] "
                f"mm lies entirely outside the stored domain "
                f"[0, {extents_mm[ai]:g}] mm on that axis — the shape "
                f"would rasterize empty. Fix the range or the domain.")
    a1, a2 = _inplane_axes(ai)
    A, B = plane_grid_views(coords[a1], coords[a2],
                            what=f"shapes3d cross-section "
                                 f"({el_name}/{shape.type})")
    xs = _shape_mask(shape, A, B)
    rng = ((coords[ai] >= lo - _POLY_EDGE_TOL)
           & (coords[ai] <= hi + _POLY_EDGE_TOL))
    # (a1, a2, ai) order, then move each axis to its canonical position
    v = xs[:, :, None] & rng[None, None, :]
    return np.moveaxis(v, (0, 1, 2), (a1, a2, ai))


def electrode_mask_3d(el, coords, mirror_axes, extents_mm):
    """Compose an ElectrodeSpec's shapes in order, in 3-D: additive
    shapes union their volumes in, cutout shapes subtract their (own,
    possibly extruded) volumes out — the electrode_mask semantics lifted
    to 3-D, with a no-extrude cutout removing through the full depth."""
    n = tuple(len(c) for c in coords)
    m = np.zeros(n, bool)
    for pos, s in enumerate(el.shapes):
        if s.type == "cutout":
            inner = s.children[0] if s.children else None
            if inner is not None:
                cut = _shape_volume(inner, coords, mirror_axes,
                                    extents_mm, el_name=el.name,
                                    shape_pos=pos)
                if not cut.any():
                    raise ValueError(
                        f"shapes3d: electrode {el.name!r} cutout #{pos} "
                        f"rasterises to ZERO nodes — the declared "
                        f"subtraction removes nothing on this grid. A "
                        f"no-op cutout is a silent declaration failure: "
                        f"either its coordinates miss the stored frame "
                        f"(shapes are authored in the STORED frame, "
                        f"non-negative; world coordinates must be "
                        f"shifted by the origin before authoring) or "
                        f"the pitch is too coarse to resolve it.")
                m &= ~cut
        else:
            sv = _shape_volume(s, coords, mirror_axes, extents_mm,
                               el_name=el.name, shape_pos=pos)
            if not sv.any():
                # A solid shape that rasterises to zero nodes was
                # silently dropped whenever a SIBLING kept the electrode
                # non-empty — the per-electrode empty refusal below this
                # never fired, and the OA exit study solved a HALF-SLOT
                # (every lower plate authored at negative stored y was
                # clipped to nothing) with no message anywhere. The
                # numbers it produced were superseded wholesale. A
                # conductor piece with zero nodes is not a smaller
                # conductor; it is an absent one, and absence is never
                # silent.
                raise ValueError(
                    f"shapes3d: electrode {el.name!r} shape #{pos} "
                    f"({s.type}) rasterises to ZERO nodes on the stored "
                    f"grid — it would be silently absent from the solve "
                    f"while its siblings keep the electrode alive. "
                    f"Shapes are authored in the STORED frame "
                    f"(non-negative; a world coordinate below the origin "
                    f"must be shifted by the origin before authoring), "
                    f"or the pitch may be too coarse to resolve this "
                    f"shape.")
            m |= sv
    return m


def shapes_masks_3d(spec: SimSpec, verbose=False):
    """{electrode index -> (nx,ny,nz) bool}, straight from the inline
    shapes on the STORED grid (coords = arange(n)*h; on a mirrored axis
    that is the non-negative half, plane at stored 0)."""
    from ion_gym.physics.build_stl3d import (_declared_mirror_axes,
                                             _grid_from_spec)
    g = spec.geometry
    nx, ny, nz, h = _grid_from_spec(spec)
    # SIGNED FRAME: shapes are authored in
    # the DECK frame. origin_mm declares the world coordinate of stored
    # node (0,0), so the raster coordinates carry it -- a shape at a
    # signed cx finds its nodes, and the zero-node refusals below
    # keep firing for genuinely-empty shapes, now tested AFTER the
    # shift. origin_mm is None -> (0,0) on legacy decks: bit-identical.
    _org = g.origin_mm or (0.0, 0.0)
    mirror_axes = _declared_mirror_axes(spec)
    for _a in (0, 1):
        if mirror_axes[_a] and float(_org[_a]) != 0.0:
            raise ValueError(
                f"shapes3d: origin_mm[{_a}] = {_org[_a]!r} on a declared "
                f"{'xy'[_a]}-mirror axis. On a mirrored axis coordinates "
                f"mean 'distance from the mirror plane' and the plane "
                f"sits at 0 by construction -- a nonzero origin there "
                f"has no defined meaning. Drop the origin on that axis "
                f"or the mirror declaration.")
    coords = (np.arange(nx) * h + float(_org[0]),
              np.arange(ny) * h + float(_org[1]),
              np.arange(nz) * h)
    extents_mm = (float(g.width_mm), float(g.height_mm),
                  float(g.depth_mm))
    if verbose:
        print(f"[shapes3d] rasterizing {len(g.electrodes)} conductors on "
              f"{nx}x{ny}x{nz} @ {h} mm/gu (inline shapes, native — "
              f"no imported geometry, no STL)")
    masks = {}
    lab = np.zeros((nx, ny, nz), np.int16)
    for idx, el in enumerate(g.electrodes, start=1):
        if el.stl:
            raise ValueError(
                f"shapes3d: electrode {el.name!r} references an STL "
                f"({el.stl!r}) — this route is inline-shapes-only. Use "
                f"the STL route, or drop the reference.")
        if not el.shapes:
            raise ValueError(
                f"shapes3d: electrode {el.name!r} has no shapes — a "
                f"Dirichlet electrode with zero volume would solve "
                f"silently as if absent.")
        m = electrode_mask_3d(el, coords, mirror_axes, extents_mm)
        if not m.any():
            raise ValueError(
                f"shapes3d: electrode {el.name!r} rasterises EMPTY on the "
                f"{nx}x{ny}x{nz} stored grid @ {h} mm/gu (stored extents "
                f"{extents_mm} mm, mirror "
                f"{''.join(a for a, on in zip('xyz', mirror_axes) if on) or 'none'}). "
                f"A conductor with zero nodes cannot hold a boundary "
                f"condition and the solve would silently proceed without "
                f"it — check the shape coordinates against the stored "
                f"frame, or use a finer pitch.")
        clash = m & (lab != 0)
        if clash.any():
            i, j, k = (int(v[0]) for v in np.where(clash))
            raise ValueError(
                f"shapes3d: electrode {idx} ({el.name!r}) overlaps "
                f"electrode {int(lab[i, j, k])} at node ({i},{j},{k}) "
                f"(+{int(clash.sum()) - 1} more) — two conductors cannot "
                f"own one voxel.")
        lab[m] = idx
        masks[idx] = m
        if verbose:
            print(f"    e{idx:<3d} {el.name:<10s} "
                  f"{int(m.sum()):>9,d} voxels")
    return masks


def _geom_bytes(spec: SimSpec) -> bytes:
    """Cache-key material: the SHAPES THEMSELVES (ordered, extrude
    included), each electrode's name and is_grid — every
    geometry-affecting field, per the basis-cache doctrine. Pitch, dims,
    and mirror are hashed by _bases_cache_key itself."""
    g = spec.geometry
    rec = [{"name": el.name, "is_grid": bool(el.is_grid),
            "shapes": [s.to_dict() for s in el.shapes]}
           for el in g.electrodes]
    # origin_mm shifts the raster coordinates (signed frame), so two
    # decks differing only in origin rasterize DIFFERENT masks -- it is
    # cache-key material, or a signed deck and its stored-frame twin
    # would collide on a stale basis (cross-session patch watch-out).
    return json.dumps({"electrodes": rec,
                       "origin_mm": list(g.origin_mm or (0.0, 0.0))},
                      sort_keys=True).encode()


def build_shapes3d_run(spec: SimSpec, verbose=False):
    """Inline extruded shapes -> (model, fly_fn, col_names, births).
    Same solver, same cache, same flyer, same mirror/canonical-frame
    machinery as the STL and scene3d paths; only the mask provider
    differs."""
    from ion_gym.physics.build_stl3d import build_stl3d_run
    return build_stl3d_run(spec, verbose=verbose, masks_fn=shapes_masks_3d,
                           geom_bytes=_geom_bytes, tag="shapes3d-v1")
