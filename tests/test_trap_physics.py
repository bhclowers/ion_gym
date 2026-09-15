"""
Gate: trap physics — anchored to the
SHIPPED example examples/paul_trap_r-z_he_cooling.json (the gate loads
the artifact users load; no parallel geometry).

  P1  STABLE: at the shipped q_z = 0.45 (a_z = 0), VACUUM (collisions
      off — pure Mathieu motion), every ion survives the window with
      excursion bounded well inside the trap dimensions.
  P2  UNSTABLE: scaling the RF amplitude to q_z = 1.05 (> 0.908 boundary,
      z-motion exponentially growing) ejects the ions onto the electrodes
      within the same window.  P2 is the built-in negative of P1: a
      solver/tracer that always "traps" fails here.
  P3  He COOLING, MONOTONE: at the shipped He operating point (20 mTorr,
      300 K, hs, sigma 1.2e-18 m^2) the rms secular excursion decreases
      early -> late for the ensemble, by a margin (>25%).

Operating point inline (doctrine): everything geometric and the source
distribution are READ FROM THE SHIPPED DECK (see _deck_point and _spec)
— currently the Finnigan ITMS-class trap (r0 10.00 mm, z0 7.83 mm,
455 V, q_z 0.40), births on a 1.0 mm disc at 0.1-1.9 eV over one RF
cycle, dt 5 ns, r-z. q_z scales linearly in V (P2 rescales amplitude by
1.05/q_z_shipped). Flights run at the DECLARED GATE_SEED, not the
deck's seed=null (which means fresh entropy per run).

Run: python tests/test_trap_physics.py     (PASS/FAIL lines, exit code)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ion_gym.io.spec_io import load_any_spec
from ion_gym.io.sim_spec import CollisionSpec
from ion_gym.physics.sim_build import build_run

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "examples", "paul_trap_r-z_he_cooling.json")

# DECK-DERIVED OPERATING POINT. These were literals matching
# the RETIRED trap (r0 4.1 mm, z0 2.90, 77.4 V, q_z 0.45); the shipped deck
# is now the Finnigan ITMS commercial geometry (r0 10.00, z0 7.83, 455 V,
# q_z 0.40), so the gate was measuring the new trap against the old trap's
# numbers. Everything geometric is now READ FROM THE DECK -- the gate stays
# anchored to whatever artifact users actually load, which was its purpose.
def _deck_point():
    sp = load_any_spec(open(EXAMPLE).read())
    g = sp.geometry
    zc = g.width_mm / 2.0
    # r0: the ring's closest approach to the axis, from the deck's own
    # polygon, not a remembered number.
    r0 = min(min(pt[1] for pt in sh.params["points_mm"])
             for e in g.electrodes if e.name == "ring" for sh in e.shapes)
    v0 = g.rf_groups[0].amplitude_v
    f0 = g.rf_groups[0].frequency_hz
    z0 = max(abs(pt[0] - zc) for e in g.electrodes
             if e.name.startswith("endcap") for sh in e.shapes
             for pt in sh.params["points_mm"] if pt[1] < 1.0)
    mz = float(sp.source.mz_list[0])
    e_c, amu = 1.602176634e-19, 1.66053906660e-27
    d2 = (r0 * 1e-3) ** 2 + 2.0 * (z0 * 1e-3) ** 2
    import math as _m
    qz = 8.0 * e_c * v0 / (mz * amu * d2 * (2 * _m.pi * f0) ** 2)
    return zc, r0, qz


ZC, R0_MM, QZ_SHIPPED = _deck_point()
EXCURSION_FRAC = 0.85          # P1 bound as a FRACTION of r0 (was 3.5 mm
                               # on r0 4.1 = 0.85; same physics, any trap)
N_IONS = 4                     # lean ensemble; physics is per-ion
GATE_SEED = 20260905           # DECLARED seed: the
                               # shipped deck carries seed=null and, since
                               # null = fresh OS entropy per run —
                               # correct for UI/API flights, wrong for a
                               # declared gate. Every assertion here is a
                               # tail statistic on n=4 (P1 excursion max,
                               # P2 >=3/4 ejected, P3 ensemble mean drop),
                               # so an unseeded run made the gate flake
                               # rarely mid-suite (the "battery_bridge
                               # intermittent"). A certified check carries
                               # its operating point; the seed is part of
                               # the operating point.
FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except AssertionError as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {e}")


def _spec(qz, collisions, t_max_us, n_ions=N_IONS):
    s = load_any_spec(open(EXAMPLE).read())
    s.source.n_ions = n_ions
    s.collisions = collisions
    s.integration.t_max_us = t_max_us
    if abs(qz - QZ_SHIPPED) > 1e-12:
        for g in s.geometry.rf_groups:
            g.amplitude_v = round(g.amplitude_v * qz / QZ_SHIPPED, 2)
    return s


def _fly_all(s):
    # ONE choke point seeds EVERY flight this gate makes (P1/P2 via _spec,
    # P3's as-shipped load alike) — a future check added to the gate cannot
    # forget to seed, because it cannot fly except through here.
    s.source.seed = GATE_SEED
    _, fly, cols, _ = build_run(s)
    xi, yi, ti = cols.index("x"), cols.index("y"), cols.index("t")
    out = []
    for i in range(s.source.n_ions):
        traj, summ = fly(i)
        out.append((traj[:, xi], traj[:, yi], traj[:, ti],
                    summ.get("kind")))
    return out


def main():
    # ------------------------------------------------------------- P1
    def p1():
        s = _spec(QZ_SHIPPED, CollisionSpec(enabled=False), 100.0)
        res = _fly_all(s)
        for n, (z, u, t, kind) in enumerate(res):
            assert t[-1] >= 99.0, (f"ion {n} left at t={t[-1]:.1f} us "
                                   f"(kind {kind}) at stable q_z")
            r = float(np.hypot(z - ZC, u).max())
            assert r < EXCURSION_FRAC * R0_MM, (
                f"ion {n} excursion {r:.2f} mm — not bounded inside the "
                f"trap (r0={R0_MM:.2f} mm, bound "
                f"{EXCURSION_FRAC * R0_MM:.2f} mm)")
    check("P1 q_z=0.45 vacuum: all trapped, excursion bounded", p1)

    # ------------------------------------------------------------- P2
    def p2():
        s = _spec(1.05, CollisionSpec(enabled=False), 100.0)
        res = _fly_all(s)
        gone = sum(1 for z, u, t, kind in res
                   if kind == 0 or t[-1] < 99.0)
        assert gone >= 3, (f"only {gone}/{len(res)} ions ejected above the "
                           f"q_z=0.908 stability boundary — Mathieu "
                           f"instability not reproduced")
    check("P2 q_z=1.05: ions ejected (Mathieu boundary)", p2)

    # ------------------------------------------------------------- P3
    def p3():
        s = load_any_spec(open(EXAMPLE).read())     # SHIPPED op point as-is
        s.source.n_ions = N_IONS
        s.integration.t_max_us = 200.0
        res = _fly_all(s)
        drops = []
        for n, (z, u, t, kind) in enumerate(res):
            assert t[-1] >= 199.0, (f"ion {n} lost during cooling "
                                    f"(kind {kind}) at the shipped point")
            e = (t < 40.0); l = (t > 160.0)
            a_e = float(np.sqrt(np.mean((z[e] - ZC) ** 2 + u[e] ** 2)))
            a_l = float(np.sqrt(np.mean((z[l] - ZC) ** 2 + u[l] ** 2)))
            drops.append(1.0 - a_l / a_e)
        mean_drop = float(np.mean(drops))
        assert mean_drop > 0.25, (f"mean rms-excursion drop {mean_drop:.0%} "
                                  f"<= 25% — He cooling not monotone/"
                                  f"effective at 20 mTorr")
    check("P3 shipped He point: rms excursion drops >25% early->late", p3)

    print("=" * 60)
    print(f"PASSED {3 - len(FAILED)}   FAILED {len(FAILED)}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
