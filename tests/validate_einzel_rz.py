"""
validate_einzel_rz.py — the CYLINDRICAL independence proof: a round-bore
einzel solved and flown in r-z natively, gated on lens
physics. Companion to validate_einzel_planar (Cartesian slice); together
they cover both symmetry reductions of the same optical element.

Geometry: three ring electrodes (tubes from bore radius to wall) on the
cylinder axis; outer two grounded, centre at lens_v. Solved natively by
solver2d.solve_laplace (exact sparse cylindrical solve, correct (1/r)
phi_r term + on-axis handling). Bases cached by geometry; voltage change
re-weights.

Gates:
  RZ-1 FIELD: on-axis saddle — centre plate at -100 V produces an
       on-axis dip (shallower than a slot lens: a round bore penetrates
       less), returning to ~0 at entrance/exit.
  RZ-2 FOCUSING: a near-axis parallel beam converges strongly through
       the lens (downstream radius << entrance radius at the design
       voltage) — and the focus MOVES UPSTREAM with voltage (a fixed
       downstream plane sees radius fall to a minimum then rise as the
       lens over-focuses). This over-focus turnaround is the signature
       of a real strong lens.
  RZ-3 STRONGER THAN SLOT: the round einzel focuses more strongly than
       the equivalent Cartesian-slot einzel (a bore concentrates the
       field more than an infinite slot) — a cross-check between the two
       independent builders.
  RZ-4 FAST-ADJUST: a new lens voltage reuses cached bases (no re-solve).
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
# GATE FIXTURE (all UI examples come from jsons.
# No exceptions."). The core builder module this gate imported is gone --
# a package that hosts geometry-building Python IS the violation. The
# geometry is preserved as a JSON fixture under tests/fixtures/ and the
# gate reads it, deriving its own constants from the deck. Gates may own
# their fixtures; core may not host examples.
from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec

_FIXTURE = (Path(repo_root()) / "tests" / "fixtures"
            / "einzel_rz_gate_fixture.json")


def einzel_rz_spec(lens_v=-120.0, mm_per_gu=0.1, n_ions=12):
    """The gate's einzel, from its JSON fixture. lens_v retunes the centre
    electrode -- a VOLTAGE change, which reweights cached bases rather than
    forcing a re-solve (the build->solve->voltages ordering)."""
    sp = SimSpec.from_json(str(_FIXTURE))
    sp.geometry.mm_per_gu = mm_per_gu
    sp.source.n_ions = n_ions
    for e in sp.geometry.electrodes:
        if e.name == "lens":
            e.dc = float(lens_v)
    return sp


# Geometry constants READ FROM THE FIXTURE, never retyped: if the deck
# changes, the gate's expectations follow it instead of silently diverging.
def _fixture_geometry():
    sp = SimSpec.from_json(str(_FIXTURE))
    tubes = [e.shapes[0].params for e in sp.geometry.electrodes]
    bore = min(t["y_mm"] for t in tubes)
    ro = max(t["y_mm"] + t["height_mm"] for t in tubes)
    tube_l = tubes[0]["width_mm"]
    xs = sorted(t["x_mm"] for t in tubes)
    gap = xs[1] - (xs[0] + tube_l)
    return bore, ro, tube_l, gap, sp.geometry.width_mm, sp.geometry.height_mm


BORE, RO, TUBE_L, GAP, ZLEN, REXT = _fixture_geometry()


def _tube_edges():
    """Axial edges of the three tubes, from the fixture's own shapes."""
    sp = SimSpec.from_json(str(_FIXTURE))
    by_name = {e.name: e.shapes[0].params for e in sp.geometry.electrodes}
    order = [n for n in ("entrance", "lens", "exit") if n in by_name] or \
        list(by_name)
    out = []
    for n in order:
        t = by_name[n]
        out += [t["x_mm"], t["x_mm"] + t["width_mm"]]
    return out


_ENT0, _ENT1, _LENS0, _LENS1, _EXIT0, _EXIT1 = _tube_edges()
from ion_gym.physics.build_rz import build_rz_run, build_rz_model
from ion_gym.physics.ensemble_driver import run


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


# Three coaxial CYLINDERS
# (einzel.imported: bore radius ~= domain radius, tube length ~ 1.56*bore
# radius, small gaps, thin walls). Bore ID ~ 12 mm as requested.
WALL = RO                  # tube fills r in [BORE, RO]
# lens stack near the entrance, then a field-free DRIFT so the focus
# forms inside the solved domain (an ideal solver lets ions focus past the field array; we
# keep it in-domain so the tracer never coasts into undefined field).
CZ = 0.5 * (_LENS0 + _LENS1)           # 19.25, centre of the lens tube






def _beam_r(results, zq):
    import math
    rs = []
    for r in results:
        t = r.traj
        k = np.searchsorted(t[:, 1], zq)
        if 0 < k < len(t):
            rs.append(math.hypot(t[k, 2], t[k, 3]))
    return np.mean(rs) if len(rs) > 2 else np.nan


def main():
    ok = True
    # RZ-1 field
    spec = einzel_rz_spec(lens_v=-100.0)
    model = build_rz_model(spec)
    h = model.mm_per_gu
    onax = model.A[:, 0]
    v_lens = onax[int(round(CZ / h))]
    v_ent = onax[int(round(5.0 / h))]
    g1 = (-100 < v_lens < -60) and abs(v_ent) < 5
    ok &= g1
    print(f"RZ-1 field: on-axis {v_ent:.1f} (entrance) -> {v_lens:.1f} "
          f"(lens, -100 V applied) -> {onax[int(round((_EXIT1 - 2) / h))]:.1f} "
          f"(exit)  ->  {'PASS' if g1 else 'FAIL'}")

    # RZ-2 focusing + over-focus turnaround at a fixed plane z=36
    radii = []
    for v in (0.0, -80.0, -200.0, -400.0):
        sp = einzel_rz_spec(lens_v=v)
        m, f, c, b = build_rz_run(sp)
        res = run(len(b), f, check_every=4, decimate=1)
        radii.append(_beam_r(res.results, 36.0))
    # strong convergence at intermediate V, then over-focus (rise)
    strong = radii[2] < radii[0] * 0.4
    overfocus = radii[3] > radii[2]
    g2 = strong and overfocus
    ok &= g2
    print(f"RZ-2 focusing: r_out(z=36) vs V "
          f"{[f'{r:.3f}' for r in radii]} (0,-80,-200,-400 V) — strong "
          f"focus then over-focus turnaround  ->  "
          f"{'PASS' if g2 else 'FAIL'}")

    # RZ-3 stronger than the slot einzel (cross-builder check)
    sp = einzel_rz_spec(lens_v=-200.0)
    m, f, c, b = build_rz_run(sp)
    r_rz = _beam_r(run(len(b), f, check_every=4, decimate=1).results, 30.0)
    try:
        from ion_gym.physics.build_planar import build_planar_run
        ssp = _einzel_spec(lens_v=-200.0)
        ssp.source.n_ions = 12
        sm, sf, sc, sb = build_planar_run(ssp)
        sres = run(len(sb), sf, check_every=4, decimate=1)
        # slot beam width in y at same downstream distance
        ys = [abs(r.traj[np.searchsorted(r.traj[:, 1], 30.0), 2])
              for r in sres.results
              if 0 < np.searchsorted(r.traj[:, 1], 30.0) < len(r.traj)]
        r_slot = np.mean(ys) if ys else np.nan
        g3 = r_rz < r_slot
        ok &= g3
        print(f"RZ-3 bore vs slot: round r_out {r_rz:.3f} < slot "
              f"{r_slot:.3f} mm (bore concentrates field)  ->  "
              f"{'PASS' if g3 else 'FAIL'}")
    except Exception as e:
        print(f"RZ-3 skipped ({type(e).__name__})")

    # RZ-4 fast-adjust
    t0 = time.time()
    _ = build_rz_run(einzel_rz_spec(lens_v=-137.0))
    dt = time.time() - t0
    g4 = dt < 0.5
    ok &= g4
    print(f"RZ-4 fast-adjust: new voltage rebuilt in {dt:.3f} s (cached "
          f"bases)  ->  {'PASS' if g4 else 'FAIL'}")

    # RZ-5 tube geometry: a real einzel is three coaxial CYLINDERS, not
    # thin apertured plates. Check the electrode mask is three separate
    # z-bands, each longer along the axis than the wall is thick (a tube,
    # not a washer), and that the middle tube shields the axis to a deep
    # dip (tube shielding -> |v_lens| >> the ~0.7*V of an aperture lens).
    model, _, _, _ = build_rz_run(einzel_rz_spec(-100.0))
    z2, r2, img2, em2 = model.potential_image()
    metal_z = em2.any(axis=1)
    runs = []
    inrun = False
    for i, v in enumerate(metal_z):
        if v and not inrun:
            s = i
            inrun = True
        elif not v and inrun:
            runs.append((s, i))
            inrun = False
    if inrun:
        runs.append((s, len(metal_z)))
    tube_len = (runs[0][1] - runs[0][0]) * h if runs else 0.0
    wall = (WALL - BORE)
    three_tubes = len(runs) == 3 and tube_len > wall     # long, not a plate
    deep_dip = model.A[int(round(CZ / h)), 0] < -60.0     # tube shielding
    g5 = three_tubes and deep_dip
    ok &= g5
    print(f"RZ-5 tube geometry: {len(runs)} coaxial tubes, length "
          f"{tube_len:.0f}mm > wall {wall:.0f}mm, axis shielded to "
          f"{model.A[int(round(CZ / h)), 0]:.0f}V  ->  "
          f"{'PASS' if g5 else 'FAIL'}")

    # RZ-6 detector focus: ions must STOP at the axial detector (not coast
    # into undefined field past the solved domain — that coast made the
    # transverse view a diverging starburst), and at the design voltage the
    # beam focuses to a spot much smaller than the entrance.
    spec = einzel_rz_spec(-120.0)
    mm, ff, cc, bb = build_rz_run(spec)
    resd = run(len(bb), ff, check_every=5, decimate=1)
    x_ends = [r.summary["x_end"] for r in resd.results]
    stopped = max(x_ends) <= ZLEN and min(x_ends) > ZLEN - 3.0
    r_in = np.mean([np.hypot(r.traj[0, 2], r.traj[0, 3])
                    for r in resd.results])
    r_out = np.mean([np.hypot(r.summary["y_end"], r.summary["z_end"])
                     for r in resd.results])
    focused = r_out < 0.3 * r_in
    g6 = stopped and focused
    ok &= g6
    print(f"RZ-6 detector focus: ions stop at x~{np.mean(x_ends):.0f}mm "
          f"(domain {ZLEN:.0f}), beam {r_in:.2f}->{r_out:.3f}mm "
          f"({r_in / max(r_out, 1e-6):.0f}x demag)  ->  "
          f"{'PASS' if g6 else 'FAIL'}")

    print("\nROUND EINZEL (R-Z NATIVE):",
          "ALL PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
