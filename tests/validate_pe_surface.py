"""
validate_pe_surface.py — the effective (adiabatic) potential landscape.

PE(x) = charge*phi_DC(x) + charge*V_pseudo(x), with the Dehmelt
pseudopotential V_pseudo = q|E0|^2/(4 m Omega^2) from the cached RF basis.
DC devices roll on their static potential directly; RF devices get the
time-averaged confining well (mass-dependent — the mass filter).

Gates:
  PE-1 DC IDENTITY: for a DC-only device (planar einzel) the PE surface
       equals charge * the static potential exactly (no RF term).
  PE-2 QUAD WELL: the quad's pseudopotential is a confining well — minimum
       on the axis, monotonically rising toward the rods.
  PE-3 DEHMELT DEPTH: the well depth at r0 matches the classic analytic
       identity D = q_Mathieu * V_rf / 4 (ideal-quad relation) within the
       round-rod/grid tolerance.
  PE-4 MASS SCALING: V_pseudo scales exactly as 1/m — PE(m/z 100) is twice
       PE(m/z 200) at any off-axis probe (pure-RF device).
  PE-5 FUNNEL WALL: the funnel's pseudopotential forms an RF wall — PE
       near the ring bore is far above PE on the axis.
  PE-6 APP RENDER: the app's PE mode draws a heatmap + the adiabatic
       annotation for the quad without error.
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import sys
from pathlib import Path

import numpy as np


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


sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    ok = True
    from ion_gym.physics.sim_build import build_run

    # PE-1 DC identity (planar einzel: no RF groups)
    from ion_gym.physics.build_planar import build_planar_run
    m, f, c, b = build_planar_run(_einzel_spec(-100.0))
    x, y, pe, ele = m.pe_surface(mz=100.0)
    h1 = m.mm_per_gu
    jax = int(round((_einzel_spec(-100.0).geometry.height_mm / 2) / h1))
    dip = pe[pe.shape[0] // 2, jax]        # on-axis lens dip (not the
    g1 = np.allclose(pe, m.A) and abs(dip + 73.1) < 2.0   # electrode metal)
    ok &= g1
    print(f"PE-1 DC identity: max|PE - phi| = {np.abs(pe - m.A).max():.2e} "
          f"eV, on-axis dip {dip:.1f} eV (phi: -73.1)  ->  "
          f"{'PASS' if g1 else 'FAIL'}")

    # PE-2/3/4 on the STL quad (pure RF: rods DC=0)
    from ion_gym.physics.build_stl import quad_stl_spec, build_stl_run
    qspec = quad_stl_spec("/tmp/quad_pe")
    qm, qf, qc, qb = build_stl_run(qspec, ground_border=True)
    h = qm.mm_per_gu
    cc = qspec.geometry.width_mm / 2
    ci = int(round(cc / h))
    _, _, PE100, _ = qm.pe_surface(mz=100.0)

    # PE-2 confining well: min at axis, rising outward along +x
    rs = np.arange(0.4, 3.4, 0.4)
    vals = [PE100[ci + int(round(rr / h)), ci] for rr in rs]
    rising = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    g2 = PE100[ci, ci] < min(vals) and rising
    ok &= g2
    print(f"PE-2 quad well: axis {PE100[ci, ci]:.3f} eV, rises to "
          f"{vals[-1]:.1f} eV at r={rs[-1]:.1f}mm (monotonic: {rising})"
          f"  ->  {'PASS' if g2 else 'FAIL'}")

    # PE-3 Dehmelt depth: V_pseudo = D*(r/r0)^2 with D = q_Mathieu*V_rf/4
    # (the classic identity). Fit the quadratic coefficient over the
    # INTERIOR (r <= 2 mm, where the round-rod field matches the ideal
    # hyperbolic form) and extrapolate to r0 — probing AT the rod surface
    # would read the near-surface field, which legitimately deviates.
    r0 = 3.84
    v_rf = qspec.geometry.rf_groups[0].amplitude_v
    D_analytic = 0.40 * v_rf / 4.0
    rfit = np.arange(0.4, 2.01, 0.2)
    pfit = np.array([PE100[ci + int(round(rr / h)), ci] - PE100[ci, ci]
                     for rr in rfit])
    cquad = float(np.polyfit(rfit, pfit, 2)[0])       # eV / mm^2
    D_fit = cquad * r0 ** 2
    g3 = abs(D_fit - D_analytic) / D_analytic < 0.15
    ok &= g3
    print(f"PE-3 Dehmelt depth (interior fit): D = {D_fit:.1f} eV vs "
          f"analytic q*V/4 = {D_analytic:.1f} eV "
          f"({100 * abs(D_fit - D_analytic) / D_analytic:.0f}% off)  ->  "
          f"{'PASS' if g3 else 'FAIL'}")

    # PE-4 mass scaling: V_pseudo ∝ 1/m exactly
    _, _, PE200, _ = qm.pe_surface(mz=200.0)
    probe = (ci + int(round(1.5 / h)), ci)
    ratio = PE100[probe] / PE200[probe]
    g4 = abs(ratio - 2.0) < 1e-6
    ok &= g4
    print(f"PE-4 mass scaling: PE(m100)/PE(m200) = {ratio:.6f} "
          f"(exact 2)  ->  {'PASS' if g4 else 'FAIL'}")

    # PE-5 funnel RF wall. On-axis the funnel PE is dominated by its DC
    # transport gradient (correct), so isolate the PSEUDO part using the
    # exact 1/m scaling: PE_m - PE_2m = P_m/2 (the DC cancels), hence
    # P_m = 2*(PE_m - PE_2m). The RF wall must tower over the axis.
    # The SHIPPED funnel deck -- which is AHEAD of the retired
    # builder (it carries the DC ladder group and the corrected ring width).
    import pathlib
    from ion_gym.io.paths import repo_root
    from ion_gym.io.sim_spec import SimSpec
    fspec = SimSpec.from_json(str(pathlib.Path(repo_root()) / "examples"
                                  / "ion_funnel_rz.json"))
    fm, ff, fc, fb = build_run(fspec)
    _, _, PEa, fem = fm.pe_surface(mz=556.0)
    _, _, PEb, _ = fm.pe_surface(mz=1112.0)
    P = 2.0 * (PEa - PEb)              # pure pseudopotential for 556 Da
    kmid = P.shape[0] // 2
    jaxis = P.shape[1] // 2
    p_axis = P[kmid, jaxis]
    # VACUUM CELLS, selected by VALUE. `fem` is pe_surface's 4th return:
    # an int16 ELECTRODE-INDEX map (0 = vacuum, N = electrode N), not a
    # boolean. This read `P[kmid][~fem[kmid]]`, and on an integer array
    # `~` is a BITWISE NOT: every vacuum 0 became -1, so the expression
    # fancy-indexed column -1 hundreds of times and measured the domain
    # EDGE, where the pseudopotential is ~0. The gate therefore reported
    # "wall 0.0 eV" and failed on correct fields -- the wall is really
    # 17.2 eV in this row, 160 eV over the surface.
    if fem.dtype == bool:
        raise TypeError(
            "pe_surface's electrode map is boolean here; this gate selects "
            "vacuum by index value (== 0) and would silently select the "
            "wrong cells. Update the predicate deliberately.")
    vac = fem[kmid] == 0
    p_wall = P[kmid][vac].max() if vac.any() else P[kmid].max()
    g5 = p_wall > 5.0 and p_wall > 10 * max(abs(p_axis), 0.02)
    ok &= g5
    print(f"PE-5 funnel RF wall (pseudo only): axis {p_axis:.3f} eV vs "
          f"wall {p_wall:.1f} eV  ->  {'PASS' if g5 else 'FAIL'}")

    # PE-6 app render (quad, PE mode)
    # BY PATH, not by menu label. This read
    # _example_specs()["quadrupole — STL rods (transport)"], a CURATED
    # DISPLAY NAME; the deck's _display_name is now "Quadrupole #2 (3D,
    # STL)" and the gate died with a KeyError before reaching its own
    # assertion -- reported as a gate failure when nothing it tests had
    # changed. A menu label is presentation and may be re-worded at any
    # time; the file path is the deck's identity, and the loader is the
    # same one the menu uses.
    from ion_gym.io.paths import repo_root
    from ion_gym.io.sim_spec import SimSpec
    from ion_gym.ui.sim_app import SimApp
    app = SimApp()
    app.spec = SimSpec.from_json(str(Path(repo_root()) / "examples" /
                                     "quadrupole_stl_rods_transport.json"))
    app._build_controls()
    app._model, app._cols = qm, qc
    app.w_showfield.value = True
    app.w_fieldmode.value = "PE surface (effective)"
    # The PE m/z widget lives on the PE tab. A second copy on the
    # Display tab was removed as a duplicate, and app.w_pe_mz went
    # with it; the draw path reads self._pe_tab.w_mz, so the gate
    # sets the SAME widget the renderer reads.
    app._pe_tab.w_mz.value = 100.0
    app.w_plane.value = "xy"
    fig = app._base_figure(qm)
    heat = sum(1 for t in fig.data if t.type == "heatmap")
    notes = [a for a in (fig.layout.annotations or [])
             if "adiabatic" in str(a.text)]
    g6 = heat >= 1 and len(notes) == 1
    ok &= g6
    print(f"PE-6 app render: {heat} heatmap, adiabatic note present: "
          f"{len(notes) == 1}  ->  {'PASS' if g6 else 'FAIL'}")

    print("\nPE SURFACE (EFFECTIVE POTENTIAL):",
          "ALL PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
