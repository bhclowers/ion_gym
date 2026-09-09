"""Stage-1 gates: generalized inline-shape extrusion (axis + range + mirror).

Shape extrusion, with three rules folded in:

  1. extrude3d migrates to the `extrude` descriptor; the ad-hoc shape-level
     z0_mm/z1_mm keys are DELETED in the same change.
  2. Frame rule: the mirror plane is at coordinate 0. Shapes are authored
     in the STORED frame; on a declared mirror axis the stored frame is the
     non-negative half and lo_mm < 0 on that axis is REFUSED.
  3. Five gates (the handoff's four + the cutout-volume gate).

The schema under test: an optional params entry on any ShapeSpec,
    "extrude": {"axis": "x"|"y"|"z", "lo_mm": float, "hi_mm": float}
Cross-section plane = the two axes perpendicular to `axis`, in cyclic
order (x->(y,z), y->(z,x), z->(x,y)); the shape's own x/y params read as
(first, second) in-plane coordinate. Absent descriptor in a 3-D shapes
build = full-depth slab along z (named default, documented in
build_shapes3d). Absent descriptor in a 2-D build = today's behavior,
byte-identical (Gate 3).

K9: every fixture in this file is self-contained; nothing is imported
from another test.

All geometries here are TINY synthetics (well under 10k nodes); the whole
file is seconds of work. The full SLIM solve never runs in-sandbox.

COVERAGE NOTE (mutation-verified): Gate 2's permutation check
proves axis-CONSISTENCY only — a globally consistent in-plane axis swap
preserves the permutation relations. Gate 1 anchors the absolute x-y
convention for axis z; the PAIR pins the full convention.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ion_gym.io.sim_spec import SimSpec, ElectrodeSpec, ShapeSpec

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- fixtures
def _tiny_spec(*, width=4.0, height=3.0, depth=3.0, pitch=0.25,
               mirror_z=False, shapes=None, electrodes=None):
    """Minimal xyz spec routed to the native shapes3d builder: inline
    shapes, no STL, depth > 0. Coordinates are the STORED frame."""
    d = {
        "name": "tiny_extrude_synthetic",
        # DECLARED seed: build_run draws births even when a
        # gate never flies; seed 0 = the historical deterministic draws.
        "source": {"seed": 0},
        "geometry": {
            "width_mm": width, "height_mm": height, "depth_mm": depth,
            "mm_per_gu": pitch,
            "symmetry": {"coords": "xyz",
                         "planes": {"z": "mirror"} if mirror_z else {}},
            "electrodes": electrodes if electrodes is not None else [
                {"name": "E1", "dc": 1.0,
                 "shapes": shapes or []},
            ],
        },
    }
    return SimSpec.from_dict(d)


def _rect(x, y, w, h, extrude=None):
    d = {"type": "rect", "x_mm": x, "y_mm": y, "width_mm": w,
         "height_mm": h}
    if extrude is not None:
        d["extrude"] = extrude
    return d


# ------------------------------------------------- Gate 1: extrusion range
def test_gate1_extrusion_range():
    """A shape with extrude z=[a,b] produces metal ONLY in [a,b] along z,
    full cross-section in x-y."""
    from ion_gym.physics.build_shapes3d import shapes_masks_3d
    from ion_gym.physics.build_stl3d import _grid_from_spec

    lo, hi = 0.5, 1.0
    spec = _tiny_spec(shapes=[_rect(1.0, 0.5, 2.0, 1.5,
                                    extrude={"axis": "z",
                                             "lo_mm": lo, "hi_mm": hi})])
    masks = shapes_masks_3d(spec)
    nx, ny, nz, h = _grid_from_spec(spec)
    m = masks[1]
    assert m.shape == (nx, ny, nz)

    zc = np.arange(nz) * h
    in_rng = (zc >= lo - 1e-9) & (zc <= hi + 1e-9)
    # outside the range: zero metal anywhere
    assert not m[:, :, ~in_rng].any(), "metal leaked outside extrude range"
    # inside the range: every slab equals the 2-D cross-section and is
    # non-empty
    xs2d = m[:, :, np.argmax(in_rng)]
    assert xs2d.any(), "cross-section rasterized empty"
    for k in np.nonzero(in_rng)[0]:
        assert np.array_equal(m[:, :, k], xs2d), \
            f"cross-section varies along the extrusion axis at k={k}"
    # cross-section is the rect: closed-solid bounds at [1,3]x[0.5,2.0]
    xc = np.arange(nx) * h
    yc = np.arange(ny) * h
    exp = ((xc[:, None] >= 1.0 - 1e-9) & (xc[:, None] <= 3.0 + 1e-9)
           & (yc[None, :] >= 0.5 - 1e-9) & (yc[None, :] <= 2.0 + 1e-9))
    assert np.array_equal(xs2d, exp)


def test_gate1_refusals():
    """Malformed descriptors refuse with named diagnostics."""
    from ion_gym.physics.build_shapes3d import shapes_masks_3d

    # lo > hi refused at the schema validator
    sh = ShapeSpec.from_dict(_rect(0, 0, 1, 1,
                                   extrude={"axis": "z", "lo_mm": 2.0,
                                            "hi_mm": 1.0}))
    with pytest.raises(ValueError, match="lo_mm"):
        sh.extrude()
    # unknown axis refused
    sh = ShapeSpec.from_dict(_rect(0, 0, 1, 1,
                                   extrude={"axis": "q", "lo_mm": 0.0,
                                            "hi_mm": 1.0}))
    with pytest.raises(ValueError, match="axis"):
        sh.extrude()
    # range entirely outside the domain refused by the builder, with numbers
    spec = _tiny_spec(shapes=[_rect(1.0, 0.5, 2.0, 1.5,
                                    extrude={"axis": "z", "lo_mm": 5.0,
                                             "hi_mm": 6.0})])
    with pytest.raises(ValueError, match="outside"):
        shapes_masks_3d(spec)
    # extruded shape in a 2-D build refused (planar/rz cannot honor it)
    from ion_gym.physics.raster2d import (electrode_mask, plane_grid_views)
    el = ElectrodeSpec.from_dict(
        {"name": "E", "shapes": [_rect(0, 0, 1, 1,
                                       extrude={"axis": "z", "lo_mm": 0.0,
                                                "hi_mm": 1.0})]})
    X, Y = plane_grid_views(np.arange(5) * 0.5, np.arange(5) * 0.5)
    with pytest.raises(ValueError, match="extrude"):
        electrode_mask(el, X, Y)


# ------------------------------------------------ Gate 2: axis generality
def test_gate2_axis_generality():
    """The SAME cross-section + range extruded along x, y, z produces
    exactly permuted masks — proves no z-hardcoding leaked in.

    Cubic domain so the three grids coincide. With the cyclic in-plane
    convention (axis a -> planes ((a+1)%3, (a+2)%3)):
        m_z[i,j,k] = XS[i,j] & R[k]
        m_x[i,j,k] = XS[j,k] & R[i]   == moveaxis(m_z, (0,1,2), (1,2,0))
        m_y[i,j,k] = XS[k,i] & R[j]   == moveaxis(m_z, (0,1,2), (2,0,1))
    """
    from ion_gym.physics.build_shapes3d import shapes_masks_3d

    ext = 3.0
    rng = {"lo_mm": 0.5, "hi_mm": 1.0}
    masks = {}
    for ax in ("x", "y", "z"):
        spec = _tiny_spec(width=ext, height=ext, depth=ext,
                          shapes=[_rect(0.75, 0.25, 1.5, 1.0,
                                        extrude=dict(axis=ax, **rng))])
        masks[ax] = shapes_masks_3d(spec)[1]
    mz = masks["z"]
    assert mz.any()
    assert np.array_equal(masks["x"],
                          np.moveaxis(mz, (0, 1, 2), (1, 2, 0)))
    assert np.array_equal(masks["y"],
                          np.moveaxis(mz, (0, 1, 2), (2, 0, 1)))


# ----------------------------------------- Gate 3: default unchanged (2-D)
# Reference sha256 of the concatenated electrode-mask bytes, recorded with
# PRE-CHANGE code (v266, this session, before any edit) on the shipped
# examples below at their own grids. Byte-identity here proves the 2-D
# rasterization path is untouched for every spec with no extrude
# descriptor. Mutation-verified: see the session decisions entry.
_REF_MASK_SHA = {
    "examples/slim_tetramer_confinement_2-d_acrossxgap.json":
        "06b43ba293aa96c6d978b6ab1f2380342015ee29b49400b5107608223dadbb84",
    "examples/paul_trap_r-z_he_cooling.json":
        # REBASELINED (deck swap; REBUILT per
        # RECOVERY_CONTRACT §4.W6): the deck is now the Finnigan ITMS
        # COMMERCIAL trap geometry (generator lives with the dev tooling).
        # Rebaselined only after MEASURING the new solver mask against the
        # papers: ring r0 = 10.00 mm (exact), endcap apex 7.90 mm = 7.83 +
        # one 0.1 mm raster pitch, on-axis ejection bore open, q_z = 0.400
        # at 455 V / 1 MHz / m/z 100. The gate caught the swap, as designed.
        "af9a3329ca2741a43414ed4ea3a33fca06d07f5fc0d6963623cecfba962ba04b",
}


def test_gate3_default_unchanged():
    from ion_gym.physics.raster2d import (anchored_grid, electrode_mask, plane_grid_views)
    for rel, ref in _REF_MASK_SHA.items():
        spec = SimSpec.from_dict(json.load(open(REPO / rel)))
        xs, ys, _anchor = anchored_grid(spec)
        X, Y = plane_grid_views(xs, ys)
        h = hashlib.sha256()
        for el in spec.geometry.electrodes:
            h.update(electrode_mask(el, X, Y).tobytes())
        assert h.hexdigest() == ref, f"2-D masks changed for {rel}"


def test_gate3_routing_unchanged():
    """No PRE-EXISTING example acquires the new route: shapes3d only
    claims the previously-unwired combination.

    GATE AMENDMENT (a native
    shapes3d example): the original assertion — NO example routes
    shapes3d — was the introduction-time config-agnosticism proof. A
    deliberate shapes3d example now ships, so the gate pins the EXACT
    expected set instead; anything else appearing on the route is still
    a routing regression."""
    from ion_gym.physics.sim_build import build_route
    expected_shapes3d = {"slim_surround_mini_3-d_native_shapes.json",
                         # BY DESIGN: the tetramer
                         # carries 192 inline rect+extrude shapes,
                         # no STLs — shapes3d is CORRECT; this set
                         # exists to catch ACCIDENTAL routing
                         # changes, and was stale (predates the
                         # tetramer example).
                         "slim_tetramer_full_3-d_transport_generic.json",
                         # ADDED: hexapole (native 3-D shapes)
                         "hexapole_native_3d.json"}
    checked = 0
    on_route = set()
    for p in sorted((REPO / "examples").glob("*.json")):
        d = json.load(open(p))
        if "geometry" not in d:
            # NAMED skip: assembly-kind files are not SimSpecs and
            # build_route does not classify them.
            assert d.get("kind") == "assembly", \
                f"{p.name}: no geometry and not an assembly — unknown kind"
            continue
        spec = SimSpec.from_dict(d)
        if build_route(spec).builder == "shapes3d":
            on_route.add(p.name)
        checked += 1
    assert on_route == expected_shapes3d, \
        f"shapes3d route set changed: {sorted(on_route)}"
    assert checked >= 10, f"only {checked} example specs checked"


# -------------------------------------------------- Gate 4: mirror about 0
def test_gate4_mirror_about_zero():
    """Extruded shape on a declared z-mirror: stored half [0,3] solves,
    the unfolded frame reads [-3,+3] with the plane at 0.000, walls land
    at +/-[1.375,1.5] (node coords inside [1.375,1.575]), and the FLIGHT
    pack carries the same frame as the display (both, from the start —
    not display-only)."""
    from ion_gym.physics.sim_build import build_route, build_run

    pitch = 0.125
    spec = _tiny_spec(width=2.0, height=1.0, depth=3.0, pitch=pitch,
                      mirror_z=True,
                      shapes=[_rect(0.25, 0.25, 1.5, 0.5,
                                    extrude={"axis": "z", "lo_mm": 1.375,
                                             "hi_mm": 1.575})])
    assert build_route(spec).builder == "shapes3d"
    model, fly_fn, cols, births = build_run(spec)

    nz_stored = int(np.floor(3.0 / pitch)) + 1
    off = model.mirror_off_mm
    assert off[0] == 0.0 and off[1] == 0.0
    assert off[2] == pytest.approx(-(nz_stored - 1) * pitch)
    # frame [-3, +3]
    metal = model.ele > 0
    nz_full = metal.shape[2]
    zc = np.arange(nz_full) * pitch + off[2]
    assert zc[0] == pytest.approx(-3.0)
    assert zc[-1] == pytest.approx(+3.0)
    # symmetric about the plane at 0; centre of metal at 0.000
    assert np.array_equal(metal, metal[:, :, ::-1])
    zmetal = zc[np.nonzero(metal.any(axis=(0, 1)))[0]]
    assert float(zmetal.mean()) == pytest.approx(0.0, abs=1e-9)
    # walls exactly at +/- the node coords inside [1.375, 1.575]
    got = sorted(round(float(z), 6) for z in zmetal)
    exp = sorted({round(s * z, 6) for s in (+1.0, -1.0)
                  for z in (1.375, 1.5)})
    assert got == exp
    # flight == display: the fly pack rides the same canonical offset
    ff = getattr(model, "fly_fields", None)
    assert ff is not None
    assert np.allclose(np.asarray(ff["mirror_off_mm"], float),
                       np.asarray(off, float))


def test_gate4_mirrored_axis_negative_lo_refused():
    """lo_mm < 0 on a declared mirror axis is a contradiction (geometry
    asymmetric about its own declared plane) — refused, never folded."""
    from ion_gym.physics.build_shapes3d import shapes_masks_3d
    spec = _tiny_spec(mirror_z=True,
                      shapes=[_rect(0.5, 0.5, 1.0, 1.0,
                                    extrude={"axis": "z", "lo_mm": -0.5,
                                             "hi_mm": 0.5})])
    with pytest.raises(ValueError, match="mirror"):
        shapes_masks_3d(spec)


# -------------------------------------------------- Gate 5: cutout volume
def test_gate5_cutout_volume():
    """A cutout with its own extrude removes metal ONLY in its range (a
    bore through a slab); a cutout with no extrude removes through full
    depth (today's 2-D semantics lifted to 3-D)."""
    from ion_gym.physics.build_shapes3d import shapes_masks_3d
    from ion_gym.physics.build_stl3d import _grid_from_spec

    slab = _rect(0.5, 0.25, 3.0, 2.5,
                 extrude={"axis": "z", "lo_mm": 0.25, "hi_mm": 2.75})
    bore = {"type": "cutout", "children": [
        {"type": "ellipse", "cx_mm": 2.0, "cy_mm": 1.5,
         "rx_mm": 0.6, "ry_mm": 0.6,
         "extrude": {"axis": "z", "lo_mm": 1.0, "hi_mm": 2.0}}]}
    spec = _tiny_spec(shapes=[slab, bore])
    m = shapes_masks_3d(spec)[1]
    _nx, _ny, nz, h = _grid_from_spec(spec)
    zc = np.arange(nz) * h
    slab_rng = (zc >= 0.25 - 1e-9) & (zc <= 2.75 + 1e-9)
    bore_rng = (zc >= 1.0 - 1e-9) & (zc <= 2.0 + 1e-9)

    ref_slab = None
    for k in range(nz):
        sl = m[:, :, k]
        if not slab_rng[k]:
            assert not sl.any()
            continue
        if ref_slab is None and not bore_rng[k]:
            ref_slab = sl.copy()
    assert ref_slab is not None and ref_slab.any()
    for k in np.nonzero(slab_rng & ~bore_rng)[0]:
        assert np.array_equal(m[:, :, k], ref_slab), \
            "slab damaged outside the cutout range"
    holed = None
    for k in np.nonzero(slab_rng & bore_rng)[0]:
        sl = m[:, :, k]
        assert sl.sum() < ref_slab.sum(), "bore removed nothing"
        assert not (sl & ~ref_slab).any()
        if holed is None:
            holed = sl.copy()
        assert np.array_equal(sl, holed)

    # no-extrude cutout: removes through the full depth of the slab
    bore_full = {"type": "cutout", "children": [
        {"type": "ellipse", "cx_mm": 2.0, "cy_mm": 1.5,
         "rx_mm": 0.6, "ry_mm": 0.6}]}
    spec2 = _tiny_spec(shapes=[slab, bore_full])
    m2 = shapes_masks_3d(spec2)[1]
    for k in np.nonzero(slab_rng)[0]:
        assert np.array_equal(m2[:, :, k], holed), \
            "full-depth cutout should bore every slab layer"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
