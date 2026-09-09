"""
test_per_ion_mass.py — one run may mix ion masses.

generate_births already assigns each ion a mass from mz_list[i % len] and
sets its birth velocity from that mass (heavier ion, same KE -> slower).
This test confirms every tracer (planar, r-z, funnel) then FLIES each ion
with its own mass and acceleration, so a single ensemble can hold several
species — the basis for the quad stable/unstable demo and for isobaric /
multi-species studies.

Gates:
  P-1 BIRTH: two masses get birth speeds in the sqrt(m2/m1) ratio.
  P-2 PLANAR: the planar tracer flies the two masses distinctly.
  P-3 RZ: the r-z tracer flies the two masses distinctly (TOF differs).
  P-4 FUNNEL: the native funnel example (generic r-z route, relocated
      tracer) flies the two masses distinctly.
  P-5 RETIRED with its subject (a device-named make_fly_fn,
      deleted with the retired build half).
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import pathlib
from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec
import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ion_gym.physics.sim_build import generate_births, build_run
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



def _mean_tof(results, idxs):
    return float(np.mean([results[i].summary["tof"] for i in idxs]))


def main():
    ok = True

    # P-1 birth speeds scale as sqrt(m)
    # Load the SHIPPED deck; P-1 tests birth-speed scaling,
    # which is geometry-independent.
    spec = SimSpec.from_json(str(pathlib.Path(repo_root()) / "examples"
                                 / "einzel_round_r-z.json"))
    spec.source.mz_list = [100.0, 200.0]
    spec.source.n_ions = 8
    b = generate_births(spec)
    sp = np.hypot(b[:, 3], np.hypot(b[:, 4], b[:, 5]))
    # SAMPLING: the builder assigns
    # m/z in CONTIGUOUS BLOCKS (i // n_ions); sp[0]/sp[1] compared two
    # m100 ions and read exactly 1.000 forever.
    ratio = sp[0] / sp[8]
    g1 = abs(ratio - math.sqrt(2.0)) < 0.02
    ok &= g1
    print(f"P-1 birth: v(m100)/v(m200) = {ratio:.3f} (sqrt2=1.414)  ->  "
          f"{'PASS' if g1 else 'FAIL'}")

    # P-2 planar tracer distinct masses
    psp = _einzel_spec(-100.0)
    psp.source.mz_list = [100.0, 250.0]
    psp.source.n_ions = 8
    pm, pf, pc, pb = build_run(psp)
    pres = run(len(pb), pf, check_every=5, decimate=1)
    # compare mean speed of first recorded step (birth) via trajectories
    # blocks: 0..7 = m100, 8..15 = m250
    v100 = np.mean([np.hypot(pres.results[i].traj[0, 4],
                             pres.results[i].traj[0, 5])
                    for i in range(0, 8)])
    v250 = np.mean([np.hypot(pres.results[i].traj[0, 4],
                             pres.results[i].traj[0, 5])
                    for i in range(8, 16)])
    g2 = abs(v100 / v250 - math.sqrt(2.5)) < 0.05
    ok &= g2
    print(f"P-2 planar: v(m100)/v(m250) = {v100 / v250:.3f} "
          f"(sqrt2.5=1.581)  ->  {'PASS' if g2 else 'FAIL'}")

    # P-3 r-z tracer distinct TOF
    spec.integration.t_max_us = 200.0
    rm, rf, rc, rb = build_run(spec)
    rres = run(len(rb), rf, check_every=5, decimate=1)
    # blocks: 0..7 = m100, 8..15 = m200
    t100 = _mean_tof(rres.results, range(0, 8))
    t200 = _mean_tof(rres.results, range(8, 16))
    g3 = t200 > t100 + 1.0     # heavier is slower through the same optics
    ok &= g3
    print(f"P-3 rz: TOF m100 {t100:.1f}us < m200 {t200:.1f}us  ->  "
          f"{'PASS' if g3 else 'FAIL'}")

    # P-4 funnel_sim tracer distinct masses
    # Examples are JSON-only; the STL
    # funnel example was removed then. The native funnel (F-1 referenced)
    # drives the same tracer and serves P-4's distinct-mass purpose.
    # KEY REPAIRED: the menu keys by filename slug now; the
    # old display-name key was a pre-existing red found (and owned)
    # by the build-half retirement, whose verification this gate is.
    # KEY -> PATH: the menu is keyed by DISPLAY NAME (a deck
    # can also be _ui_hidden and absent from the menu entirely), so a gate
    # that indexes the menu by filename slug breaks whenever the gallery is
    # curated. Load the deck the gate actually means, by its repo-relative
    # path -- stable under any menu presentation.
    fsp = SimSpec.from_json(str(pathlib.Path(repo_root()) / "examples"
                                / "ion_funnel_rz.json"))
    # MASS PAIR WIDENED: with the desync
    # fixed, [300,600] separates by only ~0.5 us here — HS mobility at
    # fixed sigma goes as 1/sqrt(reduced mass), nearly saturated against
    # N2 at these masses; the old 2.6 us "separation" was partly the
    # desync artifact. [100,2000] gives a real ~10% reduced-mass lever.
    fsp.source.mz_list = [100.0, 2000.0]
    fsp.source.n_ions = 6
    fm, ff, fc, fb = build_run(fsp)
    fres = run(len(fb), ff, check_every=10, decimate=1)
    # SAMPLING REPAIRED: ions are assigned per-m/z
    # in CONTIGUOUS BLOCKS (ion i -> mz_list[i // n_ions]); the old
    # even/odd interleave compared m300 against ITSELF and passed on
    # Monte-Carlo noise (a false pass — the stale-ALL-PASS class).
    ta = _mean_tof(fres.results, range(0, 6))      # m100 block
    tb = _mean_tof(fres.results, range(6, 12))     # m2000 block
    g4 = abs(ta - tb) > 1.0
    ok &= g4
    print(f"P-4 funnel: TOF m100 {ta:.1f}us vs m2000 {tb:.1f}us "
          f"(distinct)  ->  {'PASS' if g4 else 'FAIL'}")

    # P-5 RETIRED with its subject: it tested a device-named
    # make_fly_fn's scalar-m_ion back-compat, and that API was deleted in
    # the retired build half. The
    # scalar/mz_list contract on the LIVE route is exercised by P-3/P-4
    # (build_rz's fly chooses per-ion mass the same way).

    print("\nPER-ION MASS GATES:", "ALL PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
