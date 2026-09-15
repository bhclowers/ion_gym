"""
test_pe_view.py — gate the PE landscape.

Asserts figure STRUCTURE (surface sheet, mesh + draped red contours on,
trajectories draped at surface height, adiabatic caveat annotation) and
the draping MATH (trajectory z equals PE sampled at (x,y) plus the fixed
lift) on a real einzel run. The 2-D overlay path is exercised too.
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from ion_gym.physics.sim_build import build_run
from ion_gym.physics.ensemble_driver import run
from ion_gym.viz.pe_view import pe_figure_3d, pe_overlay_2d, _pe_sampler


def _einzel_spec(lens_v=-100.0, mm_per_gu=0.2, n_ions=15):
    """Planar-einzel test spec, inlined on retirement of einzel_planar_fixture.
    Duplicated per-gate on purpose: K9 forbids a test importing a fixture from
    another test, so each consumer carries its own copy."""
    from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
        ShapeSpec, SourceSpec, CollisionSpec, IntegrationSpec, ViewSpec)
    from ion_gym.io.sim_spec import SymmetrySpec
    W, H = 40.0, 20.0
    BORE, THICK, GAP = 3.0, 1.5, 8.0
    CX, CY = W / 2, H / 2
    MARGIN = 1.0
    PLATE_H = H - 2 * MARGIN
    def _plate(name, xc, dc):
        return ElectrodeSpec(name=name, dc=dc, shapes=[
            ShapeSpec("rect", {"x_mm": xc - THICK / 2, "y_mm": MARGIN,
                               "width_mm": THICK, "height_mm": PLATE_H}),
            ShapeSpec("cutout", {}, children=[
                ShapeSpec("ellipse", {"cx_mm": xc, "cy_mm": CY,
                                      "rx_mm": BORE, "ry_mm": BORE})])])
    return SimSpec(
        geometry=GeometrySpec(width_mm=W, height_mm=H, mm_per_gu=mm_per_gu,
            symmetry=SymmetrySpec(coords="xyz", planes={"y": "mirror"}),
            electrodes=[_plate("entrance", CX - GAP, 0.0),
                        _plate("lens", CX, lens_v),
                        _plate("exit", CX + GAP, 0.0)]),
        source=SourceSpec(seed=0, distribution="line", n_ions=n_ions, x0_mm=1.5,
            y0_mm=CY - 1.5, len_mm=3.0, axis="y", direction=[1.0, 0.0, 0.0],
            ke_lo=50.0, ke_hi=50.0, mz_list=[100.0], tob_span_us=0.0),
        collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(dt_ns=0.5, t_max_us=15.0, rec_every=8,
            record_channels=["speed", "ke_ev", "e_field"]),
        view=ViewSpec(mode="2d", planes=["xy"]),
        name="einzel lens (native)")



def main():
    spec = _einzel_spec(-100.0)
    spec.bounds.x_max_on = True
    spec.bounds.x_max = 30.0
    model, fly, cols, births = build_run(spec)
    res = run(len(births), fly, check_every=5, decimate=1)

    fig = pe_figure_3d(model, mz=spec.source.mz_list[0],
                       results=res.results,
                       title="PE surface — test einzel · DC only · m/z 100")
    kinds = [t.type for t in fig.data]
    assert "surface" in kinds, "PE sheet missing"
    surf = [t for t in fig.data if t.type == "surface"][0]
    assert surf.contours.z.show, "draped equipotentials off"
    assert surf.contours.x.show and surf.contours.y.show, "mesh grid off"
    n3d = kinds.count("scatter3d")
    assert n3d >= len(res.results), "trajectories not draped"
    # title present; and the Mathieu caveat must be ABSENT on a DC lens
    assert fig.layout.title.text and "einzel" in fig.layout.title.text
    # (amended: the caveat text was deliberately reworded in a
    # viz pass and no longer contains "Mathieu"; keying on the removed
    # word made the DC-absence check vacuous and the RF-presence check
    # unpassable. "adiabatic" is in the current text.)
    assert not any("adiabatic" in (a.text or "")
                   for a in fig.layout.annotations), \
        "DC lens must not show the RF adiabatic caveat"
    # caveat DOES appear when explicitly flagged (RF device)
    frf = pe_figure_3d(model, mz=100.0, show_adiabatic_caveat=True)
    assert any("adiabatic" in (a.text or "")
               for a in frf.layout.annotations)

    # scaling: linear factor and asinh both transform the axis + relabel
    from ion_gym.viz.pe_view import scale_pe
    x, y, PE, _ = model.pe_surface(mz=spec.source.mz_list[0])
    lin, lab_lin = scale_pe(PE, "linear", 2.0)
    # (amended: pe_surface grew NaN
    # metal-masking — NaN*2 = NaN is CORRECT and needs equal_nan; and
    # the label was ASCII-fied to "x2" in the viz sprint. Both are
    # assertion-side staleness, not behaviour bugs.)
    assert np.allclose(lin, PE * 2.0, equal_nan=True) and "x2" in lab_lin
    asi, lab_asi = scale_pe(PE, "asinh", 10.0)
    assert "asinh" in lab_asi
    # asinh compresses the deep end: ratio of extremes shrinks vs linear
    span_lin = np.ptp(PE)
    span_asi = np.ptp(asi)
    assert span_asi < span_lin, "asinh should compress the well depth"

    # draping math on the SCALED sheet (default linear ×1 -> PE unchanged)
    # AMENDED: the displayed sheet is the trust-radius-masked,
    # scale-prepared PEs, not the raw PE — its span here is
    # 94.79 vs raw 100.0, so a raw-sheet drape height is wrong by
    # construction. The honest contract is SELF-consistency: the drape
    # sits 2% of the DISPLAYED span above the DISPLAYED sheet — everything
    # sampled from the figure's own surface trace.
    # plotly surface stores z in (ny, nx) row convention (the product
    # passes Zs = PEs.T); transpose back for the (nx, ny) sampler.
    surf_z = np.asarray(surf.z, float).T
    samp = _pe_sampler(np.asarray(surf.x, float),
                       np.asarray(surf.y, float), surf_z)
    lift = 0.02 * (float(np.nanmax(surf_z)) - float(np.nanmin(surf_z))
                   + 1e-9)
    tr = res.results[0].traj
    want = samp(np.column_stack([tr[:, 1], tr[:, 2]])) + lift
    traj_traces = [t for t in fig.data if t.type == "scatter3d"
                   and t.mode == "lines"]
    got = np.asarray(traj_traces[0].z)
    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-9), "draping height wrong"
    print(f"  3-D: sheet+mesh+contours, title set, RF caveat gated, "
          f"linear+asinh scaling, {n3d} draped traces exact  OK")

    # 2-D overlay adds heatmap + red contour to an existing figure
    import plotly.graph_objects as go
    f2 = go.Figure()
    pe_overlay_2d(f2, model, mz=spec.source.mz_list[0])
    k2 = [t.type for t in f2.data]
    assert "heatmap" in k2 and "contour" in k2
    print("  2-D overlay: heatmap + red PE contours  OK")


if __name__ == "__main__":
    main()
    print("\nPE VIEW GATE: ALL PASS")
