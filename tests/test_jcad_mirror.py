"""
JCAD MIRROR GATE (L-528, PI-approved 2026-10-15).

The JupyterCAD / STEP export emits the mirrored half of every declared
mirror plane as real objects. Their placement algebra was wrong for
cylinders (the 'along' test read the ROTATION axis, so x- and y-cylinders
were mirrored as each other) and for boxes rotated about z (a z mirror
applied twice, cancelling). This gate checks the whole matrix -- every
primitive the exporter emits x every orientation it emits x every mirror
plane, singly and combined -- against GEOMETRY BUILT BY CADQUERY
(jcad_to_step), not against the mirror algebra itself:

  M1  each original body's bounding box equals the deck's own declared
      metal bounds (sim_spec.shape_metal_bounds_mm + the extrude range),
      so the unmirrored export is anchored to the deck;
  M2  each mirrored body (X__mirror_<a>[__mirror_<b>...]) has the
      original's bounding box REFLECTED through every plane in its name,
      and the original's volume.

Toy deck, no field solve. Needs CadQuery (the ion_gym 'cad' extra);
skipped by name without it.

Run: python tests/test_jcad_mirror.py   (or pytest)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                 ShapeSpec, SourceSpec, SymmetrySpec,
                                 shape_metal_bounds_mm)
from ion_gym.io.jcad_export import (spec_to_jcad, jcad_to_step,
                                    mirror_base, MIRROR_SUFFIX)

BBOX_TOL_MM = 1e-5        # OCC bounding boxes of exact solids
VOL_REL_TOL = 1e-9


def _ext(axis, lo, hi):
    return {"axis": axis, "lo_mm": lo, "hi_mm": hi}


def _rect(x, y, w, h, ext, rot=0.0):
    p = {"x_mm": x, "y_mm": y, "width_mm": w, "height_mm": h,
         "extrude": ext}
    if rot:
        p["rotation_deg"] = rot
    return ShapeSpec("rect", p)


def _circ(cx, cy, r, ext):
    return ShapeSpec("ellipse", {"cx_mm": cx, "cy_mm": cy, "rx_mm": r,
                                 "ry_mm": r, "extrude": ext})


def _poly(pts, ext):
    return ShapeSpec("polygon", {"points_mm": pts, "extrude": ext})


# every primitive x every orientation the exporter emits; all placed
# off every plane and asymmetric, so a wrong mirror cannot coincide
# with the right one
ELECTRODES = [
    ("box_z", [_rect(1.0, 2.0, 3.0, 1.5, _ext("z", 0.5, 4.0))]),
    ("box_x", [_rect(1.0, 2.0, 3.0, 1.5, _ext("x", 0.5, 4.0))]),
    ("box_y", [_rect(1.0, 2.0, 3.0, 1.5, _ext("y", 0.5, 4.0))]),
    ("box_rot30", [_rect(2.0, 3.0, 4.0, 1.0, _ext("z", 1.0, 3.0),
                         rot=30.0)]),
    ("box_rot_neg70", [_rect(5.0, 1.0, 2.5, 0.75, _ext("z", 0.25, 2.0),
                             rot=-70.0)]),
    ("cyl_z", [_circ(2.0, 3.0, 0.8, _ext("z", 1.0, 11.0))]),
    ("cyl_x", [_circ(2.0, 3.0, 0.8, _ext("x", 1.0, 11.0))]),
    ("cyl_y", [_circ(2.0, 3.0, 0.8, _ext("y", 1.0, 11.0))]),
    ("poly_z", [_poly([[1.0, 1.0], [4.0, 1.5], [2.0, 3.0]],
                      _ext("z", 0.5, 2.5))]),
    ("poly_x", [_poly([[1.0, 1.0], [4.0, 1.5], [2.0, 3.0]],
                      _ext("x", 0.5, 2.5))]),
    ("holed", [_rect(1.0, 1.0, 4.0, 4.0, _ext("z", 0.5, 1.5)),
               ShapeSpec("cutout", children=[
                   _circ(3.0, 3.0, 1.0, _ext("z", 0.5, 1.5))])]),
]
PLANES = ("x", "y", "z")


def _deck():
    els = [ElectrodeSpec(name=n, dc=0.0, shapes=s) for n, s in ELECTRODES]
    return SimSpec(name="jcad mirror gate",
                   geometry=GeometrySpec(
                       width_mm=20.0, height_mm=20.0, depth_mm=20.0,
                       mm_per_gu=0.5, electrodes=els,
                       symmetry=SymmetrySpec(
                           coords="xyz",
                           planes={a: "mirror" for a in PLANES})),
                   source=SourceSpec())


def _declared_bbox(shapes):
    """World bbox of an electrode's metal from the DECK (positive shapes;
    a cutout only removes): in-plane bounds via the shared authority,
    placed on world axes in the extrude's cyclic frame, plus the range."""
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for s in shapes:
        b = shape_metal_bounds_mm(s.to_dict())
        if b is None:
            continue
        e = s.extrude()
        a = "xyz".index(e["axis"])
        first, second = (a + 1) % 3, (a + 2) % 3
        for i, (l, h) in ((first, b[0:2]), (second, b[2:4]),
                          (a, (e["lo_mm"], e["hi_mm"]))):
            lo[i] = min(lo[i], l)
            hi[i] = max(hi[i], h)
    return lo, hi


def _reflect(lo, hi, axes):
    lo, hi = list(lo), list(hi)
    for ax in axes:
        i = "xyz".index(ax)
        lo[i], hi[i] = -hi[i], -lo[i]
    return lo, hi


def _close(b1, b2):
    return all(abs(u - v) <= BBOX_TOL_MM
               for p, q in zip(b1, b2) for u, v in zip(p, q))


def _census():
    try:
        import cadquery  # noqa: F401
    except ImportError:
        return None
    with tempfile.TemporaryDirectory() as td:
        jp = spec_to_jcad(_deck(), os.path.join(td, "gate.jcad"),
                          validate=False)
        return jcad_to_step(jp, os.path.join(td, "gate.step"))


_CENSUS = None


def _get_census():
    global _CENSUS
    if _CENSUS is None:
        _CENSUS = _census()
    return _CENSUS


def _skip_if_no_cadquery():
    if _get_census() is None:
        try:
            import pytest
            pytest.skip("CadQuery not installed (ion_gym 'cad' extra)")
        except ImportError:
            print("  [SKIP] CadQuery not installed (ion_gym 'cad' extra)")
            sys.exit(0)


def test_M1_originals_match_the_deck():
    _skip_if_no_cadquery()
    by_name = {c["name"]: c for c in _get_census()}
    bad = []
    for name, shapes in ELECTRODES:
        got = by_name[name]["bbox_mm"]
        want = _declared_bbox(shapes)
        if not _close(got, want):
            bad.append(f"{name}: built {got} vs deck {want}")
    assert not bad, "original bodies off the deck geometry:\n  " + \
        "\n  ".join(bad)


def test_M2_every_mirror_is_the_reflection():
    _skip_if_no_cadquery()
    census = _get_census()
    by_name = {c["name"]: c for c in census}
    n_mirrored, bad = 0, []
    for c in census:
        name = c["name"]
        if MIRROR_SUFFIX not in name:
            continue
        n_mirrored += 1
        base = mirror_base(name)
        axes = name.split(MIRROR_SUFFIX + "_")[1:]
        want = _reflect(*by_name[base]["bbox_mm"], axes)
        if not _close(c["bbox_mm"], want):
            bad.append(f"{name}: built {c['bbox_mm']} vs reflected "
                       f"{tuple(map(tuple, want))}")
        v0 = by_name[base]["volume_mm3"]
        if abs(c["volume_mm3"] - v0) > VOL_REL_TOL * v0:
            bad.append(f"{name}: volume {c['volume_mm3']:.9g} vs "
                       f"{v0:.9g}")
    # 3 planes -> 7 mirrored copies of each electrode
    want_n = len(ELECTRODES) * (2 ** len(PLANES) - 1)
    assert n_mirrored == want_n, (f"{n_mirrored} mirrored bodies, "
                                  f"expected {want_n}")
    assert not bad, "mirrored bodies misplaced:\n  " + "\n  ".join(bad)


if __name__ == "__main__":
    failed = 0
    for fn in (test_M1_originals_match_the_deck,
               test_M2_every_mirror_is_the_reflection):
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {e}")
    print("=" * 60)
    print(f"JCAD MIRROR: {'PASS' if not failed else f'{failed} FAILED'}")
    print("=" * 60)
    sys.exit(1 if failed else 0)
