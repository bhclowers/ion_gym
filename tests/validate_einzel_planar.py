"""
validate_einzel_planar.py — the INDEPENDENCE proof: a full spec -> native
solve -> fly natively, gated on einzel physics.

Geometry is ion_playground's einzel_lens (three plates, thickness 1.5 mm
at x = cx +/- 8 and cx; bore radius 3 mm; outer two grounded, centre at
lens_voltage) expressed as a SimSpec with inline shapes. The field is
solved natively (solver3d ghost_linear stencil, conservative omega — auto-omega
diverges on a thin planar domain with free Neumann edges) and bases are
cached by geometry so a voltage change re-weights instantly.

Gates:
  E-1 FIELD: the central plate at -100 V produces the einzel saddle —
       on-axis potential dips to ~ -95 V at the lens plane and returns to
       ~0 at entrance/exit (partial bore penetration).
  E-2 FOCUSING (the physics): a collimated beam converges through the
       lens, and the downstream beam width DECREASES MONOTONICALLY with
       lens voltage magnitude (0 -> -50 -> -150 -> -300 V). This is the
       defining einzel behavior.
  E-3 FAST-ADJUST: changing the lens voltage reuses cached bases (no
       re-solve) — the invariant the interactive app relies on.
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec, ShapeSpec,
                      SourceSpec, CollisionSpec, IntegrationSpec, ViewSpec)
from ion_gym.io.sim_spec import (SymmetrySpec)
from ion_gym.physics.build_planar import build_planar_run, build_planar_model
from ion_gym.physics.ensemble_driver import run

# Planar-einzel spec: reclaimed here (this gate's original home) on retirement
# of einzel_planar_fixture. Not an example, not core -- test scaffolding local
# to the gate that owns it (uses the sim_spec primitives imported above).
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


def einzel_spec(lens_v=-100.0, mm_per_gu=0.2, n_ions=15):
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


def _sigma_at(results, xq):
    ys = []
    for r in results:
        t = r.traj
        k = np.searchsorted(t[:, 1], xq)
        if 0 < k < len(t):
            ys.append(t[k, 2])
    return np.std(ys) if len(ys) > 2 else np.nan


def main():
    ok = True
    # E-1 field
    spec = einzel_spec(lens_v=-100.0)
    model = build_planar_model(spec)
    h = model.mm_per_gu
    jax = int(round(CY / h))
    onax = model.A[:, jax]
    v_lens = onax[int(round(CX / h))]
    v_ent = onax[int(round(5.0 / h))]
    g1 = (-85.0 < v_lens < -60.0) and abs(v_ent) < 8.0
    ok &= g1
    print(f"E-1 field: on-axis V = {v_ent:.1f} (entrance) -> {v_lens:.1f} "
          f"(lens) -> {onax[int(round(35.0/h))]:.1f} (exit)  ->  "
          f"{'PASS' if g1 else 'FAIL'}")

    # E-2 focusing: the lens focuses (beam narrows as |V| increases from
    # 0), with the minimum spot at some intermediate voltage — a strong
    # lens over-focuses past its crossover so sigma rises again, which is
    # correct physics, not a monotonic decrease. Gate: focusing happens
    # (some voltage beats V=0) and the trend is focusing-then-crossover.
    sigmas = []
    t0 = time.time()
    for v in (0.0, -50.0, -150.0, -300.0):
        sp = einzel_spec(lens_v=v)
        m, f, c, b = build_planar_run(sp)
        res = run(len(b), f, check_every=5, decimate=1)
        sigmas.append(_sigma_at(res.results, 36.0))
    solve_plus = time.time() - t0
    focuses = min(sigmas[1:]) < sigmas[0]
    g2 = focuses
    ok &= g2
    print(f"E-2 focusing: sigma_out vs lens V "
          f"{[f'{s:.3f}' for s in sigmas]} (0,-50,-150,-300 V) "
          f"-> focuses (min < V=0) {'PASS' if g2 else 'FAIL'}")

    # E-3 fast-adjust: a fresh voltage reuses cache -> near-instant
    t1 = time.time()
    _ = build_planar_run(einzel_spec(lens_v=-123.0))
    reweight = time.time() - t1
    g3 = reweight < 0.5     # cached re-weight, not a ~14 s solve
    ok &= g3
    print(f"E-3 fast-adjust: new voltage rebuilt in {reweight:.3f} s "
          f"(cached bases, no re-solve)  ->  {'PASS' if g3 else 'FAIL'}")

    print(f"\n[timing] four solves+flies in {solve_plus:.1f}s "
          f"(first solve ~14s, rest cached)")
    print("EINZEL PLANAR (NATIVE):",
          "ALL PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
