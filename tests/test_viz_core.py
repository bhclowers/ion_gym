"""
test_viz_core.py  --  the M-VIZ gate ladder
===========================================
  V0  AGNOSTIC: the framework carries no instrument constant, and both
      fixture geometries (deliberately unalike) render through the SAME
      code path with no special-casing.
  V1  DISPLAY == SOLVER INPUT: the drawn metal is an EXACT decomposition of
      the mask the kernel flies through (union of rects == mask, no gaps,
      no spill), and the field panel is the solver's own array (one stated
      transpose, no resample).
  V2  MULTI-AXIS: a scene with a z extent renders xy/xz/yz; a scene that is
      honestly planar renders the one panel it has evidence for and says so.
  V3  ZOOM POLICY (M-VIZ-Z): no plotly axis in this repo is scale-anchored.
      A scale-anchored axis makes the box zoom RIGID -- plotly reshapes the
      rectangle you drag -- and this instrument is long and thin, so the
      rectangles that matter are too.  The gate REFUSES an anchored figure,
      scans every app module for the anti-pattern, and proves that after
      the policy a 100:1 zoom rectangle survives unchanged.
  V4  PROVENANCE: every panel is stamped SOLVER-DERIVED or SCHEMATIC, and a
      bogus stamp is refused.
  V5  REPORT: multi-scene report writes its panels + an index that states
      provenance.
"""
import re
import glob
import json
import pathlib
import dataclasses
import numpy as np
import pytest

from ion_gym.viz import viz_core as V
from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec, ShapeSpec,
                                 SourceSpec, IntegrationSpec, BoundsSpec,
                                 CollisionSpec)
from ion_gym.io.sim_spec import SymmetrySpec

GEOS = ["geo_a", "geo_b"]
# every module in the repo is scanned except the policy owner itself
# (viz_core quotes the anti-pattern in its docstring, on purpose) and the
# gates (this file quotes it too, to prove the gate catches it).
SCAN_SKIP = {"viz_core.py", "test_viz_core.py"}


# Generic 2D test geometries, inlined on severing the dependency on the oaTOF stack's
# bench_fixtures (a core viz gate must not import a project). Plain
# multi-electrode SimSpecs -- no stack physics -- used purely as render input.
def _VACUUM():
    return CollisionSpec(enabled=False)


def _plate(name, x_mm, t_mm, H, slot_mm, dc, basis):
    rect = ShapeSpec(type="rect", params=dict(
        x_mm=x_mm, y_mm=0.0, width_mm=t_mm, height_mm=H))
    cut = ShapeSpec(type="cutout", children=[ShapeSpec(
        type="rect", params=dict(x_mm=x_mm - 0.01, y_mm=(H - slot_mm) / 2,
                                 width_mm=t_mm + 0.02, height_mm=slot_mm))])
    return ElectrodeSpec(name=name, shapes=[rect, cut], dc=dc, basis=basis)


def _geo_a(dc=(0.0, -300.0, -900.0), h=0.25):
    W, H = 24.0, 12.0
    els = [_plate("A0", 4.0, 0.6, H, 2.0, dc[0], 1),
           _plate("A1", 10.0, 0.6, H, 2.0, dc[1], 2),
           _plate("A2", 16.0, 0.6, H, 2.0, dc[2], 3)]
    g = GeometrySpec(width_mm=W, height_mm=H, depth_mm=0.0, mm_per_gu=h,
                     symmetry=SymmetrySpec(coords="xyz",
                                           planes={"x": "none", "y": "mirror",
                                                   "z": "none"}),
                     electrodes=els)
    return SimSpec(geometry=g, name="geo_a_slit3",
                   source=SourceSpec(seed=0, n_ions=1, distribution="point",
                                     x0_mm=1.0, y0_mm=H / 2, mz_list=[524.0]),
                   integration=IntegrationSpec(dt_ns=0.5, t_max_us=20.0),
                   collisions=_VACUUM(), bounds=BoundsSpec())


def _geo_b(dc=(0.0, 120.0, -450.0, -1200.0, -2000.0), h=0.2):
    # A7: the domain extent is an exact integer cell count at this
    # fixture's pitch. W was 31.7 mm, which at h = 0.2 is 158.5 cells and
    # is refused by the loader. Rounded UP to 31.8 (159 cells): the metal
    # ends at 27.4 mm (last rect x = 26.6 + 0.8 wide), so widening the
    # domain adds vacuum at the outer wall and clips nothing, which is
    # where A7 says the remainder belongs.
    W, H = 31.8, 9.4                      # 159 x 47 cells at h = 0.2
    els = []
    xs = [3.0, 8.4, 13.9, 20.1, 26.6]
    slots = [1.4, 1.8, 2.2, 2.6, 3.0]
    for i, (x, s) in enumerate(zip(xs, slots)):
        rect = ShapeSpec(type="rect", params=dict(
            x_mm=x, y_mm=0.0, width_mm=0.8, height_mm=H))
        y0 = H / 2 - s / 2 + 0.7          # aperture off-centre: breaks y-mirror
        cut = ShapeSpec(type="cutout", children=[ShapeSpec(
            type="rect", params=dict(x_mm=x - 0.01, y_mm=y0,
                                     width_mm=0.82, height_mm=s))])
        els.append(ElectrodeSpec(name=f"B{i}", shapes=[rect, cut],
                                 dc=dc[i], basis=i + 1))
    g = GeometrySpec(width_mm=W, height_mm=H, depth_mm=0.0, mm_per_gu=h,
                     symmetry=SymmetrySpec(coords="xyz",
                                           planes={"x": "none", "y": "none",
                                                   "z": "none"}),
                     electrodes=els)
    return SimSpec(geometry=g, name="geo_b_ladder5",
                   source=SourceSpec(seed=0, n_ions=1, distribution="point",
                                     x0_mm=1.0, y0_mm=H / 2 + 0.7,
                                     mz_list=[524.0]),
                   integration=IntegrationSpec(dt_ns=0.5, t_max_us=20.0),
                   collisions=_VACUUM(), bounds=BoundsSpec())


_GEOS = {"geo_a": _geo_a, "geo_b": _geo_b}


# ---------------------------------------------------------------- V0
@pytest.mark.parametrize("geo", GEOS)
def test_V0_adapters_are_agnostic(geo):
    spec = _GEOS[geo]()
    sc = V.scene_from_simspec(spec, field="E")
    assert sc.bodies and sc.fields


def test_V0_no_instrument_constants_in_the_framework():
    """A renderer that knows a number from ONE instrument is a renderer that
    will misdraw the next one."""
    src = open(V.__file__).read()
    for bad in ("34.2", "45.0", "850.3", "V_LAUNCH", "S_EXIT", "BEAM_ENTRY",
                "6126", "immersion", "reflectron"):
        assert bad not in src, f"instrument constant {bad!r} leaked into viz_core"


# ---------------------------------------------------------------- V1
@pytest.mark.parametrize("geo", GEOS)
def test_V1_drawn_metal_is_the_solver_mask_exactly(geo):
    from ion_gym.physics.build_planar import build_planar_model
    spec = _GEOS[geo]()
    m = build_planar_model(spec)
    ele = np.asarray(m.ele) > 0
    h = float(m.mm_per_gu)
    rebuilt = np.zeros_like(ele)
    ORG = 0.5 * h                      # build_planar is cell-centred
    for (x0, x1, y0, y1) in V.mask_rects_mm(ele, h, (ORG, ORG)):
        i0 = int(round((x0 - ORG) / h + 0.5))
        i1 = int(round((x1 - ORG) / h - 0.5))
        j0 = int(round((y0 - ORG) / h + 0.5))
        j1 = int(round((y1 - ORG) / h - 0.5))
        assert not rebuilt[i0:i1 + 1, j0:j1 + 1].any(), "rects overlap"
        rebuilt[i0:i1 + 1, j0:j1 + 1] = True
    assert np.array_equal(rebuilt, ele), "drawn metal != solver mask"


@pytest.mark.parametrize("geo", GEOS)
def test_V1_field_panel_is_the_solver_array(geo):
    from ion_gym.physics.build_planar import build_planar_model
    spec = _GEOS[geo]()
    m = build_planar_model(spec)
    sc = V.scene_from_simspec(spec, model=m, field="phi")
    f = sc.fields[0]
    assert f.values.shape == (m.A.shape[1], m.A.shape[0])   # one transpose
    assert np.array_equal(f.values, np.asarray(m.A, float).T)  # no resample
    # CONVENTION HISTORY — this assertion has now flipped TWICE; read
    # before flipping a third time. (1) Originally (n-1)*h, node-centred.
    # (2) Amended to n*h when the display was found half a cell off the
    # spec — the fix then declared the RASTERIZER (cell-centred) the
    # truth. (3) The deeper analysis showed the FLY KERNEL
    # (gx = px/mm, node i at i*h) is the
    # physical truth; the rasterizer was the mis-sampled side. Builders
    # now sample node-centred with n = W/h + 1 nodes, so (n-1)*h equals
    # the width the spec DECLARES — the original intent of this gate,
    # finally satisfied in ONE frame shared by metal, field, and spec.
    assert f.extent == (0.0, (m.A.shape[0] - 1) * m.mm_per_gu,
                        0.0, (m.A.shape[1] - 1) * m.mm_per_gu)


# ---------------------------------------------------------------- V2
@pytest.mark.parametrize("geo", GEOS)
def test_V2_multi_axis_for_3d_single_for_planar(geo):
    spec = _GEOS[geo]()
    planar = V.scene_from_simspec(spec, field=None)
    assert not planar.is_3d()
    assert V.views_for(planar) == ("xy",)          # no faked depth
    slab = V.scene_from_simspec(spec, field=None, slab_mm=16.0)
    assert slab.is_3d()
    assert V.views_for(slab) == ("xy", "xz", "yz")  # R3
    assert any("DECLARED, not solved" in n for n in slab.notes)
    # CONVENTION A (final): THE Z AXIS IS HORIZONTAL.
    # One table, every scene, no builder switch.  The per-builder switch this
    # replaced misclassified two of the four 3-D examples: the STL quadrupole
    # and the SLIM tetramer both transport down z and both report
    # depth_mm == 0.0, so no discriminator (builder, coords, depth_mm) was
    # right about all four.  A rule with a per-configuration exception is not
    # a convention.
    e_xy = V.view_extent(slab, "xy")
    e_xz = V.view_extent(slab, "xz")
    assert e_xz[:2] == (-8.0, 8.0)          # z is the HORIZONTAL pair
    assert e_xz[2:] == e_xy[:2]             # x is the vertical one
    assert V.view_labels("xz") == ("z (mm)", "x (mm)")
    assert V.view_labels("yz") == ("z (mm)", "y (mm)")
    assert V.view_labels("xy") == ("x (mm)", "y (mm)")
    P = np.array([[1.0, 2.0, 3.0]])
    assert tuple(V.project(P, "xz")[0]) == (3.0, 1.0)
    assert tuple(V.project(P, "yz")[0]) == (3.0, 2.0)
    with pytest.raises(V.VizError):
        V.views_for(slab, ["xq"])


@pytest.mark.parametrize("geo", GEOS)
def test_V2_multi_axis_renders(geo):
    spec = _GEOS[geo]()
    sc = V.scene_from_simspec(spec, field="E", slab_mm=16.0)
    fig = V.render_mpl(sc)
    assert len(fig.axes) >= 3          # 3 panels (+ colourbars)
    figs = V.interactive_panels(sc)
    assert set(figs) == {"xy", "xz", "yz"}
    for f in figs.values():
        V.assert_free_zoom(f)


# ---------------------------------------------------------------- V3
def test_V3_free_aspect_layout_is_not_anchored():
    """M-VIZ-Z: no scaleanchor (a live constraint reshapes the drag box).

    The FIGURE must also carry NO pixel dimensions.
    It used to carry width/height, which made it a second authority over the
    same pixel box as the pane (built stretch_width + its own height) -- the
    two fought on every relayout and the zoom felt stunted. True scale is
    delivered by sizing the PANE (pane_size_for), which is what this
    function's docstring always said. This gate now pins BOTH halves.
    """
    lay = V.free_aspect_axes((0.0, 850.0, 0.0, 16.0), px_width=800)
    assert "scaleanchor" not in str(lay) and "scaleratio" not in str(lay)
    assert lay["dragmode"] == "zoom"
    # the figure constrains NOTHING about pixels
    assert "width" not in lay and "height" not in lay
    # true scale is delivered by sizing the PANE, not the figure.
    #
    # M-VIZ-Z3: the pane must be ISOTROPIC (px/mm equal on both axes) and must
    # FIT ON A SCREEN. The old version clamped height at a ceiling and floored
    # it at 180 px while leaving the width alone -- both are anisotropic
    # scales, and they put a 26% error into the aspect ratio this function
    # exists to guarantee. An 850 x 16 mm drift region really is a 15 px
    # strip; the aspect-lock control is how a user opts out.
    def _iso(extent, **kw):
        sz = V.pane_size_for(extent, **kw)
        dx = extent[1] - extent[0]
        dy = extent[3] - extent[2]
        return sz, sz["width"] / dx, sz["height"] / dy

    sz, sx, sy = _iso((0.0, 850.0, 0.0, 16.0), px_width=800)
    assert sz["width"] == 800
    assert abs(sx - sy) / sx < 0.02, "long-thin view must stay isotropic"
    assert sz["height"] == int(round(800 * 16.0 / 850.0))   # no 180 px floor

    sq, sx, sy = _iso((-8.0, 8.0, -8.0, 8.0), px_width=600)
    assert sq["width"] == sq["height"] == 600

    # tall geometry: fits the screen, still isotropic (used to overflow)
    tall, sx, sy = _iso((0.0, 10.0, 0.0, 40.0))
    assert tall["height"] <= 760 and tall["width"] <= 1000
    assert abs(sx - sy) / sx < 0.02, "tall view must stay isotropic"


def test_V3_zoom_rectangle_keeps_its_aspect_ratio():
    """The directive, executed: drag a 100:1 rectangle and it stays a 100:1
    rectangle.  With a scaleanchor plotly would grow the short side."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=[0, 850], y=[0, 16])
    V.apply_zoom_policy(fig, (0.0, 850.0, 0.0, 16.0))
    # the user drags a long thin box (a drift region, 100:1)
    fig.update_layout(xaxis_range=[400.0, 500.0], yaxis_range=[7.0, 8.0])
    assert list(fig.layout.xaxis.range) == [400.0, 500.0]
    assert list(fig.layout.yaxis.range) == [7.0, 8.0]
    assert fig.layout.yaxis.scaleanchor is None
    ar = ((fig.layout.xaxis.range[1] - fig.layout.xaxis.range[0])
          / (fig.layout.yaxis.range[1] - fig.layout.yaxis.range[0]))
    assert abs(ar - 100.0) < 1e-9


def test_V3_policy_strips_an_existing_anchor_and_the_gate_refuses_one():
    import plotly.graph_objects as go
    bad = go.Figure()
    bad.add_scatter(x=[0, 10], y=[0, 1])
    bad.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    with pytest.raises(V.VizError):
        V.assert_free_zoom(bad)                 # the gate catches it
    V.apply_zoom_policy(bad)                    # extent measured from the data
    V.assert_free_zoom(bad)                     # and the policy removes it


def test_V3_no_app_module_anchors_an_axis():
    """Repo-wide scan: the anti-pattern must not exist anywhere, or it will
    come back the next time someone copies a plotting block."""
    offenders = []
    # Scan every CORE module -- the ion_gym package plus any module still at
    # the repo root mid-migration. The old scan globbed root *.py only; once
    # modules live under ion_gym/<subpkg>/ that misses exactly the app/viz
    # code this gate protects. Rooted on __file__ (not cwd), and never scans
    # tests/studies/tools (the original never did; the gates quote the
    # anti-pattern on purpose -- SCAN_SKIP covers viz_core + this file).
    root = pathlib.Path(__file__).resolve().parent.parent
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        is_core = rel.parts[0] == "ion_gym" or len(rel.parts) == 1
        if not is_core or "__pycache__" in rel.parts:
            continue
        if p.name in SCAN_SKIP:
            continue
        for i, line in enumerate(open(p), 1):
            # the ANTI-PATTERN is an axis actually being anchored to another
            # axis, i.e. scaleanchor="x" -- not the word in a comment
            # explaining why we do not do that.
            if re.search(r'scaleanchor\s*=\s*[\'"]', line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, ("M-VIZ-Z violations (rigid zoom):\n"
                           + "\n".join(offenders))


def test_V3_3d_scene_layout_is_camera_based():
    lay = V.scene3d_layout((0, 850, 0, 16, -8, 8))
    assert lay["scene"]["aspectmode"] == "data"
    assert "scaleanchor" not in str(lay)


# ---------------------------------------------------------------- V4
def test_V4_provenance_is_stamped_and_bogus_stamps_refused():
    spec = _GEOS["geo_a"]()
    sc = V.scene_from_simspec(spec, field=None)
    assert sc.provenance == V.SOLVER and "as flown" in sc.stamp()
    sk = V.scene_from_bodies("sketch", [V.Body("p", [np.zeros((4, 3))])])
    assert sk.provenance == V.SCHEMATIC and "NOT evidence" in sk.stamp()
    with pytest.raises(V.VizError):
        V.Scene(title="x", provenance="LOOKS_RIGHT")


# ---------------------------------------------------------------- V5
def test_V5_report_writes_panels_and_an_index(tmp_path):
    # two solver-provenance scenes -> report writes 2 panels + a stamped index
    scenes = [V.scene_from_simspec(_GEOS["geo_a"](), field="phi", slab_mm=16.0),
              V.scene_from_simspec(_GEOS["geo_b"](), field="phi", slab_mm=16.0)]
    V.report(scenes, str(tmp_path), name="M-VIZ smoke")
    assert len(glob.glob(str(tmp_path / "*.png"))) == 2
    idx = open(tmp_path / "index.md").read()
    assert V.SOLVER in idx


# ======================================================================
# V6..V10 -- the Q3 lessons, gated (v144).  Each of these was a thing the
# Q3 one-off script did BY HAND; each is now framework, so each needs a gate
# or it will be re-implemented (or re-broken) by the next subsystem.
# ======================================================================
def _spec_and_model(geo="geo_a"):
    from ion_gym.physics.build_planar import build_planar_model
    spec = _GEOS[geo]()
    return spec, build_planar_model(spec)


# ---------------------------------------------------------------- V6
@pytest.mark.parametrize("geo", GEOS)
def test_V6_registration_the_drawn_metal_sits_where_the_spec_says(geo):
    """REGISTRATION. EVERYTHING is NODE-CENTRED (node i
    at i*h): the fly kernel always was (gx = px/mm), the CSG rasterizer always was,
    and the native rasterizers now sample there too, with n = W/h + 1
    nodes so the domain [0, (n-1)*h] IS the declared width. (This gate's
    docstring previously declared build_planar cell-centred and treated
    the rasterizer as truth — see the convention-history note at V-ext.)

    The check: the field extent must be the domain the spec DECLARES —
    (n-1)*h under node-centred sizing."""
    spec, m = _spec_and_model(geo)
    h = float(m.mm_per_gu)
    ele = np.asarray(m.ele)
    nx, ny = ele.shape[:2]

    sc = V.scene_from_simspec(spec, model=m, field="phi")
    assert sc.fields[0].extent == (0.0, (nx - 1) * h, 0.0, (ny - 1) * h)
    assert sc.box()[:4] == (0.0, (nx - 1) * h, 0.0, (ny - 1) * h)

    # every drawn rect must cover exactly the cell of the node it came from:
    # node i occupies [i*h, (i+1)*h] on a cell-centred grid.
    for idx in range(1, len(spec.geometry.electrodes) + 1):
        msk = (ele == idx)
        if not msk.any():
            continue
        ii, jj = np.where(msk)
        rects = V.mask_rects_mm(msk, h, (0.5 * h, 0.5 * h))
        x0 = min(r[0] for r in rects)
        x1 = max(r[1] for r in rects)
        assert abs(x0 - ii.min() * h) < 1e-9, "left edge off by half a cell"
        assert abs(x1 - (ii.max() + 1) * h) < 1e-9, "right edge off"


def test_V6_bench_registration_is_unchanged_node_centred():
    """The same fix must NOT move the bench, which really is node-centred."""
    # node-centred registration is a property of mask_rects_mm at origin (0,0)
    # -- a rect's left edge sits half a cell BEFORE the node -- not of any
    # project's FA; a core model supplies the mask.
    spec, m = _spec_and_model("geo_a")
    ele = np.asarray(m.ele)
    h = float(m.mm_per_gu)
    r = V.mask_rects_mm(ele > 0, h, (0.0, 0.0))
    ii, _ = np.where(ele > 0)
    assert abs(min(x[0] for x in r) - (ii.min() - 0.5) * h) < 1e-9


# ---------------------------------------------------------------- V7
@pytest.mark.parametrize("geo", GEOS)
def test_V7_one_body_per_electrode_carrying_its_name_and_voltage(geo):
    spec, m = _spec_and_model(geo)
    sc = V.scene_from_simspec(spec, model=m, field=None)
    live = [el for i, el in enumerate(spec.geometry.electrodes, start=1)
            if (np.asarray(m.ele) == i).any()]
    # AMENDED: body names are DRIVE LABELS by design
    # (_drive_label: the solver-true view states
    # voltages") — 'A0 [GND]', 'A2 [DC -900V]'. The contract's own
    # title says name AND voltage; the assertion form predated the
    # label. Identity = the prefix before the drive bracket.
    assert [b.name.split(" [")[0] for b in sc.bodies] == \
        [el.name for el in live]
    assert all(" [" in b.name and b.name.endswith("]")
               for b in sc.bodies), \
        f"drive tag missing: {[b.name for b in sc.bodies]}"
    assert [b.volt for b in sc.bodies] == [el.dc for el in live]
    # and the union of the labelled bodies IS the solver's metal
    assert sum(len(b.polys) for b in sc.bodies) > 0


@pytest.mark.parametrize("geo", GEOS)
def test_V7_a_boolean_ele_is_REFUSED_not_quietly_drawn(geo):
    """The C1-C5 defect, arriving at the display boundary.  A bool `ele` makes
    `ele == 1` true for EVERY conductor: draw it and all electrodes appear at
    #1's voltage.  It must not degrade to one grey blob -- that is how it
    survived being 'fixed' three times."""
    spec, m = _spec_and_model(geo)

    class _Bool:                       # a model whose ele came back coerced
        def __init__(self, m):
            self.__dict__.update(m.__dict__)
            self.ele = np.asarray(m.ele) > 0
    with pytest.raises(V.VizError) as e:
        V.scene_from_simspec(spec, model=_Bool(m), field=None)
    assert "BOOLEAN" in str(e.value)
    # and it is refused, not worked around, unless the caller SAYS undifferentiated
    sc = V.scene_from_simspec(spec, model=_Bool(m), field=None,
                              per_electrode=False)
    assert len(sc.bodies) == 1


# ---------------------------------------------------------------- V8
@pytest.mark.parametrize("geo", GEOS)
def test_V8_basis_panel_is_the_models_own_basis_no_second_solve(geo):
    """The basis is already on the model.  The panel must BE it -- divided by
    the excitation it was solved at, and by nothing else."""
    from ion_gym.physics.build_planar import V_BASIS
    spec, m = _spec_and_model(geo)
    name = spec.geometry.electrodes[0].name
    sc = V.scene_from_simspec(spec, model=m, field=f"basis:{name}")
    f = sc.fields[0]
    assert np.array_equal(f.values,
                          np.asarray(m.bases[1], float).T / V_BASIS)
    assert f.values.max() <= 1.0 + 1e-9 and f.values.min() >= -1e-9
    assert "phi / V" in f.quantity
    with pytest.raises(V.VizError):
        V.scene_from_simspec(spec, model=m, field="basis:no_such_electrode")


def _rf_phase_spec(v_rf=150.0, f_rf_hz=2.4e6, mm_per_gu=0.25):
    """Minimal anti-phase RF quad (2 rod groups A@0 / B@180), core-built, for
    the RF-phase-panel viz test. Replaces a former reach into the q3 project:
    the core viz suite must not depend on a project (core never imports
    examples/projects)."""
    from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
        ShapeSpec, SourceSpec, CollisionSpec, IntegrationSpec, ViewSpec,
        RFGroupSpec)
    from ion_gym.io.sim_spec import SymmetrySpec
    W = H = 10.0

    def _rod(name, cx, cy, grp, col):
    # Gate repair: ElectrodeSpec's RF membership field is
        # `rf_groups` (a list) — this fixture still used the pre-rename
        # singular `rf_group=` kwarg and V9 has been un-constructable in
        # stock v206 since the schema change. Behavior asserted by the
        # gate is unchanged.
        return ElectrodeSpec(name=name, dc=0.0, rf_groups=[grp], color=col,
            shapes=[ShapeSpec("rect", {"x_mm": cx - 0.6, "y_mm": cy - 0.6,
                                       "width_mm": 1.2, "height_mm": 1.2})])
    els = [_rod("ROD_A1", 1.6, 5.0, "A", [143, 180, 217]),
           _rod("ROD_A2", 8.4, 5.0, "A", [143, 180, 217]),
           _rod("ROD_B1", 5.0, 1.6, "B", [217, 154, 143]),
           _rod("ROD_B2", 5.0, 8.4, "B", [217, 154, 143])]
    groups = [RFGroupSpec(name="A", frequency_hz=f_rf_hz, amplitude_v=v_rf,
                          phase_deg=0.0, waveform="sin"),
              RFGroupSpec(name="B", frequency_hz=f_rf_hz, amplitude_v=v_rf,
                          phase_deg=180.0, waveform="sin")]
    geo = GeometrySpec(width_mm=W, height_mm=H, mm_per_gu=mm_per_gu,
        symmetry=SymmetrySpec(coords="xyz",
                              planes={"x": "mirror", "y": "mirror", "z": "none"}),
        electrodes=els, rf_groups=groups)
    return SimSpec(geometry=geo,
        source=SourceSpec(seed=0, distribution="point", n_ions=1, x0_mm=5.0, y0_mm=5.0,
                          mz_list=[115.0]),
        collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(dt_ns=0.5, t_max_us=1.0),
        view=ViewSpec(mode="2d", planes=["xy"]),
        name="rf phase test (anti-phase quad)")


# ---------------------------------------------------------------- V9
def test_V9_rf_phase_panel_states_its_phase_and_refuses_a_dc_model():
    """An RF field drawn at an unstated phase is a picture of nothing."""
    from ion_gym.physics.build_planar import build_planar_model
    spec = _rf_phase_spec(v_rf=150.0, f_rf_hz=2.4e6, mm_per_gu=0.25)
    m = build_planar_model(spec)
    f0 = V.scene_from_simspec(spec, model=m, field="phi@0").fields[0]
    f9 = V.scene_from_simspec(spec, model=m, field="phi@90").fields[0]
    assert "90" in f9.quantity
    # at phase 0 the two anti-phase rod groups cancel; at 90 they do not
    assert np.abs(f9.values).max() > 10.0 * np.abs(f0.values).max()
    # exactly A + sum_k sin(theta + ph_k) * B_k -- no smoothing, no rescale
    want = np.asarray(m.A, float).copy()
    for B, _f, ph in m.Bk:
        want = want + np.sin(np.radians(90.0 + ph)) * np.asarray(B, float)
    assert np.allclose(f9.values, want.T, rtol=0, atol=1e-12)

    dc, mdc = _spec_and_model("geo_a")            # a DC-only model
    with pytest.raises(V.VizError) as e:
        V.scene_from_simspec(dc, model=mdc, field="phi@90")
    assert "no sin drives" in str(e.value)


# ---------------------------------------------------------------- V10
def test_V10_lineout_is_the_field_and_refuses_to_extrapolate():
    """A cut through a KNOWN field must return that field.  A cut that leaves
    the solved region must REFUSE, not extrapolate -- the Q3 report reads a
    quoted number off a line-out, so a silently invented tail is a lie in a
    caption."""
    xs = np.linspace(0.0, 10.0, 101)
    ys = np.linspace(0.0, 4.0, 41)
    Xa, _ = np.meshgrid(xs, ys)                   # phi = 3*x + 1, exactly
    slab = V.FieldSlab("xy", (0.0, 10.0, 0.0, 4.0), 3.0 * Xa + 1.0, "phi (V)")
    lo = V.lineout(slab, (0.0, 2.0), (10.0, 2.0), n=51, label="cut")
    assert np.allclose(lo.values, 3.0 * np.linspace(0, 10, 51) + 1.0, atol=1e-9)
    assert abs(lo.s[-1] - 10.0) < 1e-9
    with pytest.raises(V.VizError):
        V.lineout(slab, (0.0, 2.0), (12.0, 2.0))  # off the end of the solve


# ---------------------------------------------------------------- V11
def test_V11_a_scene_cannot_quietly_promote_a_sketch_to_evidence():
    solved = V.Body("rod", [np.zeros((4, 3))], provenance=V.SOLVER)
    sketch = V.Body("ladder", [np.zeros((4, 3))], provenance=V.SCHEMATIC)
    sc = V.scene_from_bodies("mixed", [solved, sketch], provenance=V.SOLVER,
                             bounds=(0, 1, 0, 1, 0, 1))
    assert "MIXED" in sc.stamp() and "1 of 2" in sc.stamp()
    assert V.SCHEMATIC in sc.stamps() and V.SOLVER in sc.stamps()
    clean = V.scene_from_bodies("clean", [solved], provenance=V.SOLVER,
                                bounds=(0, 1, 0, 1, 0, 1))
    assert "MIXED" not in clean.stamp()


# ---------------------------------------------------------------- V12
def test_V12_dimension_callouts_survive_and_a_bogus_view_is_refused():
    """A geometry review IS its callouts.  Dropping one silently -- or drawing
    it in a view the scene never renders -- turns a dimensional review into a
    grey blob that reviews nothing."""
    b = V.Body("blk", [np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
                                float)])
    sc = V.scene_from_bodies("review", [b], bounds=(0, 2, 0, 1, 0, 0),
                             dims=[V.Dim("xy", (0, 0), (2, 0), "2.0"),
                                   V.Dim("xy", (0, 0), (0, 1), "1.152",
                                         derived=True)])
    assert len(sc.dims) == 2
    assert sc.dims[1].derived is True
    fig = V.render_mpl(sc, field=False)
    txt = " ".join(t.get_text() for a in fig.axes for t in a.texts)
    assert "2.0" in txt and "1.152" in txt and "derived" in txt
    with pytest.raises(V.VizError):
        V.scene_from_bodies("bad", [b], bounds=(0, 2, 0, 1, 0, 0),
                            dims=[V.Dim("qq", (0, 0), (1, 1), "x")])


# ---------------------------------------------------------------- V13
def test_V13_extrude_gives_a_body_an_honest_silhouette_in_all_three_views():
    """The R3 failure that looks like success: a cross-section-only body draws
    perfectly in xy and collapses to a LINE in xz and yz.  Nothing raises; the
    body has simply lost two dimensions.  Extrusion must restore them."""
    sq = [(0., 0.), (2., 0.), (2., 1.), (0., 1.)]        # 2 (x) by 1 (y)
    polys = V.extrude(sq, 5.0, 9.0, axis=2)              # 4 mm deep in z
    P = np.vstack(polys)
    # the prism must actually occupy its declared extent on every axis
    for ax, (lo, hi) in enumerate([(0., 2.), (0., 1.), (5., 9.)]):
        assert np.isclose(P[:, ax].min(), lo) and np.isclose(P[:, ax].max(), hi)
    # and every view must show AREA, not a degenerate line
    for v in V.VIEWS:
        Q = V.project(P, v)
        span = Q.max(axis=0) - Q.min(axis=0)
        assert span.min() > 0.0, f"{v} view collapsed: span {span}"
    # closed prism: 2 caps + one quad per edge
    assert len(polys) == 2 + len(sq)
    # a different axis lifts into the correct plane
    p0 = V.extrude(sq, -1.0, 1.0, axis=0)
    assert np.isclose(np.vstack(p0)[:, 0].min(), -1.0)   # x is now the depth
    with pytest.raises(V.VizError):
        V.extrude(sq, 3.0, 3.0)                          # zero depth
    with pytest.raises(V.VizError):
        V.extrude([(0., 0.), (1., 1.)], 0.0, 1.0)        # not a polygon
    with pytest.raises(V.VizError):
        V.extrude(sq, 0.0, 1.0, axis=7)


# ---------------------------------------------------------------- V14
def test_V14_extreme_aspect_scene_stays_inside_a_page_budget():
    """A long thin instrument (the real case: 635 mm by 16 mm) must not blow
    the figure up.  The aspect cap bounds the ratio BETWEEN panels and says
    nothing about the total; a fixed panel height times a sum of ~10s produced
    a 62-inch, 9360-px figure.  A picture nothing will open is not a picture."""
    poly = [(-8., -8.), (8., -8.), (8., 8.), (-8., 8.)]
    b = V.Body("bar", V.extrude(poly, 0.0, 635.0, axis=2))
    sc = V.scene_from_bodies("long thin", [b],
                             bounds=(-8, 8, -8, 8, 0, 635))
    assert V.views_for(sc) == ("xy", "xz", "yz")       # R3 still holds
    fig = V.render_mpl(sc, field=False)
    w_in, h_in = fig.get_size_inches()
    assert w_in <= 20.0, f"figure {w_in:.1f} in wide"
    assert h_in <= 12.0, f"figure {h_in:.1f} in tall"
    assert w_in * 150 <= 8000 and h_in * 150 <= 8000   # renderable at dpi=150
    # and the axial panels must still be TRUE SCALE -- budgeting the page is
    # not a licence to distort a panel you read dimensions off.
    for ax, v in zip(fig.axes, ("xy", "xz", "yz")):
        assert ax.get_aspect() in (1.0, "equal")


# ---------------------------------------------------------------- V15
def test_V15_models_declare_their_planes_and_nothing_sniffs_a_signature():
    """A real crash.  pe_view decided a model's dimensionality by
    calling pe_surface(plane=...) and catching TypeError.  That is a SIGNATURE
    probe wearing a CAPABILITY probe's clothes, and it lied in three ways."""
    from ion_gym.physics import build_planar, build_rz, build_stl3d
    from ion_gym.viz import pe_view

    # 1. every model DECLARES; nobody guesses.
    assert build_planar.PlanarModel.PLANES == ("xy",)
    assert build_rz.RZModel.PLANES == ("rz",)          # NOT "xy" -- the old
    # (a device-named model assertion was retired with the class;
    #  the rz declaration doctrine lives on RZModel above.)
    assert build_stl3d.Stl3DModel.PLANES == ("xy", "xz", "yz")

    # 2. a model that does not declare is refused LOUDLY, never assumed planar.
    class Undeclared:
        def pe_surface(self, **kw):
            raise AssertionError("must never be reached")
    with pytest.raises(V.VizError, match="does not declare PLANES"):
        V.model_planes(Undeclared())

    # 3. THE LIE: a TypeError raised INSIDE pe_surface must NOT come back as a
    #    geometric claim.  The old code told the user "this model has no 'xz'
    #    plane" when what actually happened was that their solver threw.
    class Exploding:
        PLANES = ("xy", "xz", "yz")
        def pe_surface(self, mz=None, charge=1, plane="xy"):
            raise TypeError("unsupported operand type(s) for *: 'NoneType'")
    with pytest.raises(TypeError, match="NoneType"):
        pe_view.compute_component(Exploding(), 100.0, 1, "effective",
                                  plane="xz")

    # 4. asking a model for a plane it does not have is refused by NAME, and
    #    the message says what it DOES have.
    class RZLike:
        PLANES = ("rz",)
        def pe_surface(self, mz=None, charge=1, plane="rz"):
            raise AssertionError("must never be reached")
    with pytest.raises(ValueError, match=r"has no 'yz' plane.*'rz'"):
        pe_view.compute_component(RZLike(), 100.0, 1, "effective", plane="yz")

    # 5. and the default plane is the model's OWN first plane, not "xy".
    class RZOk:
        PLANES = ("rz",)
        def pe_surface(self, mz=None, charge=1, plane="rz"):
            assert plane == "rz"
            n = np.zeros((4, 4))
            return np.arange(4.), np.arange(4.), n, n.astype(np.int16)
    x, y, pe, ele = pe_view.compute_component(RZOk(), 100.0, 1, "effective")
    assert pe.shape == (4, 4)





# ---------------------------------------------------------------- V16
def test_V16_z_is_horizontal_with_no_per_builder_escape_hatch():
    """THE Z AXIS IS HORIZONTAL.  One table, every scene.

    This gate exists because the two previous answers were both wrong, and both
    were wrong in the SAME way -- they made orientation a function of something
    other than the view.  The per-builder switch (`axial_of_spec`) misclassified
    the STL quadrupole and the SLIM tetramer, which both transport down z and
    both report depth_mm == 0.0: no discriminator available (builder, coords,
    depth_mm) was right about all four 3-D examples.  A rule with a
    per-configuration exception is not a convention.

    So: no `axial` parameter, no `axial_of_spec`, no `Scene.axial`.  If any of
    them comes back, this fails."""
    import inspect
    for fn in (V.project, V.view_labels):
        assert "axial" not in inspect.signature(fn).parameters, (
            f"{fn.__name__} takes an axial argument again -- orientation must "
            f"be a function of the VIEW alone")
    for gone in ("axial_of_spec", "view_axes"):
        assert not hasattr(V, gone), f"{gone} is back"
    assert "axial" not in {f.name for f in dataclasses.fields(V.Scene)}

    # z horizontal, in every view that contains z, for every scene alike.
    assert V.view_labels("xz")[0] == "z (mm)"
    assert V.view_labels("yz")[0] == "z (mm)"
    P = np.array([[1.0, 2.0, 3.0]])
    assert tuple(V.project(P, "xz")[0]) == (3.0, 1.0)
    assert tuple(V.project(P, "yz")[0]) == (3.0, 2.0)
    assert tuple(V.project(P, "xy")[0]) == (1.0, 2.0)


# ---------------------------------------------------------------- V17
def test_V17_every_example_is_json_and_nothing_special_cases_a_builder():
    """Examples are JSON, full stop.  No builder special case.

    The old registry was seven hand-written registrations, several inside
    `except Exception: pass` -- so a release that dropped a module shipped an
    app with a SILENTLY shorter menu -- and two of them imported example specs
    out of build_stl.py, a CORE module shipping examples.

    An example must take NO path a user's own JSON cannot take.  That is the
    whole point: if the examples load, the loader works."""
    from pathlib import Path
    from ion_gym.ui import sim_app
    from ion_gym.io import paths
    from ion_gym.io.sim_spec import SimSpec

    d = paths.repo_root() / "examples"
    files = sorted(d.glob("*.json"))
    assert files, "no example JSON shipped"

    # The menu is not every file. A deck may ship and fly
    # (notebooks reference it) while being kept OUT of the menu via
    # _ui_hidden, and examples/ also carries assembly-format files that are
    # not SimSpecs at all. Compare the menu against what is ACTUALLY
    # menu-eligible, and name the offenders when it disagrees.
    hidden, non_spec = [], []
    for f in files:
        _deck = json.loads(f.read_text())
        if _deck.get("_ui_hidden"):
            hidden.append(f.name)
        elif "geometry" not in _deck:
            non_spec.append(f.name)
    specs = sim_app._example_specs()
    eligible = len(files) - len(hidden) - len(non_spec)
    assert len(specs) == eligible, (
        f"menu and examples/ disagree: {len(specs)} menu entries vs "
        f"{eligible} eligible files ({len(files)} total, hidden={hidden}, "
        f"non-spec={non_spec})")

    for k, f in specs.items():
        sp = f()
        assert sp.geometry.electrodes, f"{k}: no electrodes"
        # round-trips through the SAME loader a user gets
        assert SimSpec.from_json(sp.to_json()).to_json() == sp.to_json()

    # no example may bake an ABSOLUTE path -- that is the funnel-STL disease:
    # a spec that only loads on the machine that wrote it.
    for p in files:
        j = json.loads(p.read_text())
        sd = j.get("geometry", {}).get("stl_dir")
        if sd:
            assert not Path(sd).is_absolute(), f"{p.name}: absolute stl_dir {sd}"
            for e in j["geometry"]["electrodes"]:
                if e.get("stl"):
                    assert (d / sd / e["stl"]).exists(), (
                        f"{p.name} names {e['stl']} which is not shipped")

    # and the registry must not silently swallow a broken example
    src = Path(sim_app.__file__).read_text()
    body = src[src.index("def _example_specs"):src.index("def _mkw")]
    assert "except Exception:\n        pass" not in body, (
        "an example that cannot load is a BUG, not an absence")


# ---------------------------------------------------------------- V18
def test_V18_style_is_user_defined_and_rf_devices_are_not_all_grey():
    """The default colour is the user's, and an RF device must not
    come out uniformly grey.

    Colour was a function of DC ALONE, and its own docstring said a stack of
    identically-grey electrodes tells you nothing about the thing you tuned --
    then returned exactly that for every RF device, because on an RF instrument
    the interesting electrodes carry DC = 0 BY DESIGN (the STL quadrupole's four
    rods are 0.0 .. 0.0; SLIM's RF1/RF2 are both 0).  They differ by drive group
    and phase, not potential.

    The first repair was "DC, ELSE group", and that `else` was itself the bug:
    on SLIM, DC *does* vary (Guard is biased), so the DC branch wins and the two
    RF PHASES come out identical.  "DC varies" != "DC is sufficient".  So both
    channels, always: FILL = DC, EDGE = drive group."""
    old = dict(V.STYLE)
    try:
        # -- user-defined default
        V.set_style(metal="#010203")
        assert V.STYLE["metal"] == "#010203"
        plain = [V.Body("a", [], volt=None), V.Body("b", [], volt=None)]
        face, _ = V._body_colors(plain)
        assert all(c == "#010203" for c in face.values())
        # an unknown key is REFUSED, not absorbed
        with pytest.raises(V.VizError, match="unknown style key"):
            V.set_style(colour="red")

        # -- an all-DC-zero RF device is distinguished by GROUP, not left grey
        rods = [V.Body(f"rod_{i}", [], volt=0.0, group=("RFA", "RFB")[i % 2])
                for i in range(4)]
        face, edge = V._body_colors(rods)
        assert len({edge[i] for i in range(4)}) == 2, "RF phases collapsed"

        # -- and where DC varies, group STILL survives (the SLIM trap)
        slim = [V.Body("RF1", [], volt=0.0, group="RF1"),
                V.Body("RF2", [], volt=0.0, group="RF2"),
                V.Body("TW", [], volt=0.0, group=None),
                V.Body("Guard", [], volt=7.5, group=None)]
        face, edge = V._body_colors(slim)
        assert edge[0] != edge[1], "RF1/RF2 collapsed because DC happened to vary"
        assert face[3] != face[0], "the DC bias on Guard is invisible"
        basis = V._color_basis(slim)
        assert "DC" in basis and "drive group" in basis
    finally:
        V.STYLE.clear()
        V.STYLE.update(old)


# ---------------------------------------------------------------- V19
def test_V19_the_app_and_the_report_share_one_axis_convention():
    """sim_app draws trajectories; viz_core draws reports.  They must not render
    the same instrument two different ways.

    They DID.  sim_app._plane_cols branched on `depth_mm > 0` -- its own comment
    conceded "the swap is a fix for one specific pathology, so it applies only
    where that pathology exists" -- so a planar spec got x horizontal while
    viz_core gave z horizontal.  Worse, the correlate is wrong: the STL
    quadrupole and the SLIM tetramer both transport down z and BOTH report
    depth_mm == 0.0, so two genuinely 3-D instruments took the "2-D" branch.
    (A conditional keyed on a correlate rather than a declared property.)

    One convention: z horizontal.  This gate reads sim_app's table and compares
    it to viz_core's, so they cannot drift apart again.

    AMENDED with the r-z trajectory-column fix.
    The gate used to also assert _plane_cols contains ZERO `if` statements.
    That was a PROXY for the contract above, and it over-reached: the r-z
    tracer records the AXIAL coordinate in trajectory column 1 (the
    declared slot -- "an r-z spec stores physical z in the width/x
    slot", see sim_app._axial_axis), so WHICH COLUMNS feed the axes
    legitimately depends on the declared coords.  Keying the columns off
    `spec.geometry.coords` is D1 dispatch on a declared property -- not the
    correlate (`depth_mm`) this gate was written against.  What must
    never branch is the ORIENTATION: the (la, lb) labels.  The gate now
    asserts that invariant directly, and STRONGER than before: the old
    regex saw only the first dict-literal binding per view; this walks the
    AST and checks EVERY 6-tuple bound to a view key -- dict literal or
    `m['xz'] = (...)` override -- so a branch that swapped labels in ANY
    path fails.  The depth_mm ban stands unchanged."""
    import ast
    from pathlib import Path
    from ion_gym.ui import sim_app

    # Inspect the CODE, not the prose.  (My first version of this gate grepped
    # the source text and matched `depth_mm` inside its own explanatory comment
    # -- a gate that fails on its own documentation is a gate that will be
    # deleted rather than obeyed.)
    src = Path(sim_app.__file__).read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_plane_cols")
    code = ast.unparse(fn)                     # comments are gone

    assert "depth_mm" not in code, "sim_app still keys orientation off depth_mm"

    # Collect EVERY (ca, cb, ka, kb, la, lb) tuple bound to a view key,
    # wherever it is bound: {'xz': (...)} dict entries AND m['xz'] = (...)
    # subscript overrides.  A branch may change the COLUMNS (that is
    # declared-property dispatch); if any binding changes the LABELS,
    # orientation has become configuration-dependent again and this fails.
    views = ("xy", "xz", "yz")
    found = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value in views
                        and isinstance(v, ast.Tuple) and len(v.elts) == 6):
                    found.setdefault(k.value, []).append(v)
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Tuple)
                and len(node.value.elts) == 6):
            for t in node.targets:
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value in views):
                    found.setdefault(t.slice.value, []).append(node.value)

    for view in views:
        tups = found.get(view, [])
        assert tups, f"sim_app has no {view} mapping"
        for tup in tups:
            la, lb = tup.elts[4], tup.elts[5]
            assert (isinstance(la, ast.Constant)
                    and isinstance(lb, ast.Constant)), (
                f"{view}: the axis labels must be literals, not computed")
            app = (f"{la.value} (mm)", f"{lb.value} (mm)")
            assert app == V.view_labels(view), (
                f"{view}: sim_app draws {app} (one of {len(tups)} bindings "
                f"of this view), viz_core draws {V.view_labels(view)}")


# ---------------------------------------------------------------- V20
def test_V20_pe_view_keeps_electrode_identity_and_never_omits_the_overlay():
    """The reported symptom -- "electrode numbering and shading is largely
    absent" -- was THREE defects stacked in one block of pe_view:

      1. `np.asarray(ele, float) > 0.5` CAST THE int16 LABELS TO A BOOLEAN MASK.
         `ele` carries the electrode INDEX per node; that IS the electrode's
         identity, and collapsing it to metal/not-metal threw it away.  Same
         class as the basis_cache `.astype(bool)` that made every electrode draw
         at #1's voltage (doctrine C).
      2. `if ii.size < 20000` SILENTLY SKIPPED the whole overlay on a fine grid.
    Not "fewer points" -- NONE, and it said nothing.
      3. every electrode was one grey (#333): no shading, no legend.

    ...and the lot sat inside `except Exception: pass`, so if any of it failed
    the electrodes were simply not there."""
    import ast
    from pathlib import Path
    from ion_gym.viz import pe_view

    src = Path(pe_view.__file__).read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "pe_figure_3d")
    code = ast.unparse(fn)

    # 1. the labels must not be collapsed to a boolean
    assert "float) > 0.5" not in code and "ele, float)" not in code, (
        "pe_view is casting the int16 electrode labels to a boolean mask again")
    # 2. no silent size cap
    assert "20000" not in code, "the silent overlay cap is back"
    # 3. no bare/broad swallow inside the figure builder
    for h in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
        nm = "BARE" if h.type is None else ast.unparse(h.type)
        assert nm not in ("BARE", "Exception", "BaseException"), (
            f"pe_figure_3d swallows {nm} again -- an electrode that fails to "
            f"draw must say so, not vanish")

    # and functionally: one labelled, voltage-shaded trace PER electrode
    import numpy as np
    x = np.arange(20.0)
    y = np.arange(12.0)
    ele = np.zeros((20, 12), np.int16)
    ele[2:5, 2:5] = 1
    ele[10:14, 4:8] = 2
    PE = np.zeros((20, 12))
    fig = pe_view.pe_figure_3d(surface=(x, y, PE, ele),
                               electrode_dc={1: -100.0, 2: +50.0},
                               metal_mode="dc")
    names = [t.name for t in fig.data
             if getattr(t, "name", None) and str(t.name).startswith("E")]
    assert len(names) == 2, f"expected one trace per electrode, got {names}"
    assert any("-100" in n for n in names) and any("+50" in n for n in names), (
        f"the electrode's VOLTAGE must be on its label: {names}")

    # a boolean ele is REFUSED, not silently drawn as one undifferentiated blob
    with pytest.raises(ValueError, match="BOOLEAN ele"):
        pe_view.pe_figure_3d(surface=(x, y, PE, (ele > 0)),
                             electrode_dc={1: -100.0}, metal_mode="dc")
