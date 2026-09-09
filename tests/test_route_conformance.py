"""
ROUTE CONFORMANCE GATE (tier 2 of the parity plan).

Flies ONE tiny spec through EVERY spec-routed builder and asserts the
route-INDEPENDENT per-ion contract — the envelope whose per-route
reimplementation caused the 2026-08 parity escapes (birth refusal, mass
desync, summary m/z). A change that breaks envelope parity on any route
goes red here mechanically, instead of waiting for a screenshot.

Parity enumeration (PARITY_RULE.md — the gate eats its own dogfood):
  planar   — COVERED (native 2-D spec)
  r-z      — COVERED (native rz spec)
  3-D      — COVERED via shapes3d (native inline extrusion; scene3d and
             stl3d share build_stl3d's per-ion envelope, so the envelope
             assertions transfer; their file-fixture front doors are the
             excluded part)
  tw2d     — PARTIAL: birth-refusal parity only, by direct kernel call —
             tw2d is not spec-routed; mass/summary are direct arguments,
             so R1/R3/R4 have no meaning for it

Contract cases, per covered route:
  R1  summary: every ion's summary carries kind/tof/mz/x_end/y_end/z_end,
      and summary["mz"] EQUALS sim_build.mz_of(spec, i) — the flown mass,
      from the single authority.
  R2  birth refusal: a source inside metal REFUSES loudly at birth
      (ValueError naming the birth) — never a silent step-1 splat.
  R3  determinism: with gas on, the same ion flown twice is byte-equal
      (per-ion seeding is reproducible).
  R4  channels: a requested optional channel ('speed') is genuinely
      filled for a moving ion — never silent zeros.

Runtime target: tiny grids, sub-minute wall clock (numba first-compile
dominates). K9: this file carries its own fixtures.

Run: python tests/test_route_conformance.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                 ShapeSpec, SourceSpec, CollisionSpec,
                                 IntegrationSpec)
from ion_gym.io.sim_spec import (SymmetrySpec)
from ion_gym.physics.sim_build import build_run, mz_of
from ion_gym.physics.ion_envelope import REQUIRED_SUMMARY_KEYS

FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except Exception as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y,
                              "width_mm": w, "height_mm": h})


def _spec(coords, depth=0.0, source_in_metal=False):
    """Tiny two-electrode duct, one recipe for all routes: metal walls top
    and bottom, ions born mid-channel (or INSIDE the bottom wall when
    source_in_metal, for R2)."""
    g = GeometrySpec(
        width_mm=10.0, height_mm=6.0, depth_mm=depth, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[
            ElectrodeSpec(name="top", dc=5.0,
                          shapes=[_rect(0.0, 5.0, 10.0, 1.0)]),
            ElectrodeSpec(name="bot", dc=0.0,
                          shapes=[_rect(0.0, 0.0, 10.0, 1.0)]),
        ])
    y0 = 0.5 if source_in_metal else 3.0
    return SimSpec(
        geometry=g, name=f"conformance_{coords}_{depth:g}",
        source=SourceSpec(distribution="point", n_ions=2,
                          x0_mm=2.0, y0_mm=y0,
                          z0_mm=(depth / 2.0 if depth else 0.0),
                          direction=[1.0, 0.0, 0.0],
                          ke_lo=1.0, ke_hi=1.0,
                          mz_list=[100.0, 400.0], tob_span_us=0.0, seed=11),
        collisions=CollisionSpec(enabled=True, gas="N2", T_k=300.0,
                                 P_pa=50.0, sigma_m2=2.27e-18),
        integration=IntegrationSpec(dt_ns=2.0, t_max_us=3.0, rec_every=5,
                                    record_channels=["speed"]))


ROUTES = [
    ("planar", dict(coords="xyz", depth=0.0)),
    ("r-z", dict(coords="rz", depth=0.0)),
    ("3-D shapes3d", dict(coords="xyz", depth=4.0)),
]

# Tier 3: the contract is DECLARED in ion_envelope and
# imported here — the gate tests the declared contract, not a private
# copy that could drift from it.
REQUIRED_KEYS = REQUIRED_SUMMARY_KEYS


def main():
    for label, kw in ROUTES:
        spec = _spec(**kw)
        model, fly, cols, births = build_run(spec)

        def r1(fly=fly, spec=spec, births=births, label=label):
            for i in range(len(births)):
                traj, s = fly(i)
                missing = [k for k in REQUIRED_KEYS if k not in s]
                assert not missing, f"{label} ion {i} summary missing {missing}"
                assert abs(float(s["mz"]) - mz_of(spec, i)) < 1e-12, (
                    f"{label} ion {i}: summary mz {s['mz']} != flown "
                    f"{mz_of(spec, i)} (mz_of authority)")
        check(f"R1 {label}: summary contract + mz authority", r1)

        def r3(fly=fly, label=label):
            t1, _ = fly(0)
            t2, _ = fly(0)
            assert np.array_equal(t1, t2), (
                f"{label}: ion 0 not reproducible (per-ion seeding broken)")
        check(f"R3 {label}: gas flight deterministic per ion", r3)

        def r4(fly=fly, cols=cols, label=label):
            assert "speed" in cols, f"{label}: requested channel absent"
            traj, _ = fly(0)
            sp = traj[:, cols.index("speed")]
            assert np.any(sp > 0), (
                f"{label}: 'speed' silently zero for a moving ion")
        check(f"R4 {label}: requested channel genuinely filled", r4)

        def r2(kw=kw, label=label):
            bad = _spec(source_in_metal=True, **kw)
            _m, bfly, _c, bb = build_run(bad)
            try:
                bfly(0)
            except ValueError as e:
                assert "birth" in str(e).lower(), (
                    f"{label}: refusal does not name the birth: {e}")
                return
            raise AssertionError(
                f"{label}: in-metal birth flew (silent step-1 splat class)")
        check(f"R2 {label}: in-metal birth refuses loudly", r2)

    # tw2d — PARTIAL parity by direct kernel call (see enumeration above)
    def t1():
        from ion_gym.physics.tracer_tw2d import fly_tw2d, build_tw2d_fields
        nx = ny = 20
        ele = np.zeros((nx, ny), bool)
        ele[8:12, 8:12] = True
        f = build_tw2d_fields({1: np.zeros((nx, ny))}, groups=[], assign={},
                              dc={1: 0.0}, h_mm=0.5, ele=ele)
        try:
            fly_tw2d(f, 100.0, (5.0, 5.0), (0, 0, 0), t_max_us=0.2,
                     ion_label="ion 0")
        except ValueError as e:
            assert "birth" in str(e).lower(), e
            return
        raise AssertionError("tw2d: in-metal birth flew")
    check("R2 tw2d (partial route): in-metal birth refuses loudly", t1)

    n = 3 * len(ROUTES) + len(ROUTES) + 1
    print("=" * 60)
    print(f"ROUTE CONFORMANCE: PASSED {n - len(FAILED)}   "
          f"FAILED {len(FAILED)}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
