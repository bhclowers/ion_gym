"""test_reflectron_timefocus.py — the reflectron TIME-FOCUS gate (G / S4).

The defining behaviour of a reflectron TOF: ions of one m/z with an
ENERGY SPREAD, which a pure drift smears out in arrival time, are brought
back to (nearly) ONE arrival time by the mirror — higher-energy ions
penetrate deeper, spend longer turning around, and are caught by the
slower ions at the focus. This gate proves the NATIVE reconstructed
reflectron (example_reflectron_rz) does that:

  T1  at the tuned drift (the focus condition), the return-TOF spread of
      a +/-5% energy ensemble is FAR tighter than a pure drift over the
      same path — the time focus, quantified.
  T2  the mechanism check: penetration depth scales with energy (higher
      KE goes deeper), the physical basis of the compensation, and grids
      are transparent (an is_grid entrance the ion flies through, not a
      wall — the r-z is_grid fix this example forced).

OPERATING POINT: native 97-ring reflectron (v_back=1000 V), 100 amu ions
accelerated to ~800 eV, +/-5% energy spread, drift tuned to the
first-order focus (~160 mm for this mirror). Vacuum, r-z, dt=0.2 ns.
"""
import _bootstrap  # noqa: F401
import numpy as np

from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                 ShapeSpec, SymmetrySpec, SourceSpec,
                                 CollisionSpec, IntegrationSpec)
from ion_gym.physics.build_rz import build_rz_run
# The gate's configuration (97 rings, v_back 1000 V) IS the
# shipped deck, so it loads the JSON rather than a package builder.
import pathlib as _pl
from ion_gym.io.paths import repo_root


def reflectron_rz_spec(n_rings=97, v_back=1000.0):
    sp = SimSpec.from_json(str(_pl.Path(repo_root()) / "tests" / "fixtures"
                               / "reflectron_rz_ring_stack.json"))
    _n = sum(1 for e in sp.geometry.electrodes if e.name.startswith("ring"))
    if int(n_rings) != _n or abs(float(v_back) - 1000.0) > 1e-9:
        raise ValueError(
            f"this gate is anchored to the SHIPPED deck (rings={_n}, "
            f"v_back=1000 V); asked for rings={n_rings}, v_back={v_back}. "
            "Sweeps regenerate the deck with the dev tooling.")
    return sp

FOCUS_DRIFT_MM = 160.0
V_BACK = 1000.0
KES = (760.0, 780.0, 800.0, 820.0, 840.0)   # +/-5% around 800 eV


def _assembly(drift):
    refl = reflectron_rz_spec(n_rings=97, v_back=V_BACK)
    els = []
    for e in refl.geometry.electrodes:
        p = dict(e.shapes[0].params)
        p["x_mm"] += drift
        els.append(ElectrodeSpec(name=f"m_{e.name}", dc=e.dc,
                                 is_grid=e.is_grid,
                                 shapes=[ShapeSpec("rect", p)]))
    return GeometrySpec(width_mm=drift + 103, height_mm=19.0, mm_per_gu=1.0,
                        symmetry=SymmetrySpec(coords="rz"), electrodes=els)


def _fly(geo, ke):
    s = SimSpec(geometry=geo, source=SourceSpec(seed=0, 
        distribution="point", n_ions=1, x0_mm=2.0, y0_mm=0.0, axis="x",
        direction=[1, 0, 0], ke_lo=ke, ke_hi=ke, mz_list=[100.0],
        tob_span_us=0.0), collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(dt_ns=0.2, t_max_us=300.0, rec_every=1),
        name="one")
    _, ff, cc, _ = build_rz_run(s)
    ti, xi = cc.index("t"), cc.index("x")
    traj, summ = ff(0)
    z, t = traj[:, xi], traj[:, ti]
    back = np.where((z[1:] < 2.0) & (z[:-1] >= 2.0))[0]   # return past launch
    rt = t[back[-1] + 1] if len(back) else np.nan
    return rt, z.max(), summ.get("kind")


def test_T1_time_focus():
    geo = _assembly(FOCUS_DRIFT_MM)
    rts = np.array([_fly(geo, ke)[0] for ke in KES])
    assert np.all(np.isfinite(rts)), ("some ions did not return — grid a "
                                      "wall? (is_grid)")
    refl_spread = (rts.max() - rts.min()) * 1000               # ns
    # pure drift over the SAME mean path, no mirror focusing
    ke = np.array(KES)
    v = np.sqrt(2 * ke * 1.602e-19 / (100 * 1.66e-27)) / 1e3   # mm/us
    path = rts.mean() * v.mean()                               # mm (mean)
    drift_only = path / v
    drift_spread = (drift_only.max() - drift_only.min()) * 1000
    ratio = drift_spread / refl_spread
    print(f"  time focus: reflectron spread {refl_spread:.1f} ns vs "
          f"pure-drift {drift_spread:.0f} ns over same path "
          f"-> {ratio:.0f}x tighter")
    assert ratio > 8.0, (f"reflectron did not time-focus: only {ratio:.1f}x "
                         f"tighter than drift (spread {refl_spread:.1f} ns)")
    print("  T1 reflectron brings a +/-5% energy spread to a tight time "
          "focus  OK")


def test_T2_mechanism_and_grid_transparency():
    geo = _assembly(FOCUS_DRIFT_MM)
    depths, kinds = [], []
    for ke in KES:
        _, zmax, kind = _fly(geo, ke)
        depths.append(zmax)
        kinds.append(kind)
    depths = np.array(depths)
    # higher energy penetrates deeper (the compensation mechanism)
    assert np.all(np.diff(depths) > 0), (f"penetration not monotonic in "
                                         f"energy: {depths}")
    # and the entrance GRID was transparent: ions reached the mirror
    # (depth >> the drift length), not splatted on the 0 V grid
    assert depths.min() > FOCUS_DRIFT_MM + 20, (
        f"ions stopped near the entrance grid (depth {depths.min():.0f} vs "
        f"drift {FOCUS_DRIFT_MM}) — is_grid transparency regressed")
    assert all(k == 1 for k in kinds), f"ions did not exit cleanly: {kinds}"
    print(f"  T2 penetration {depths.min():.0f}->{depths.max():.0f} mm rises "
          f"with energy; entrance grid transparent  OK")


if __name__ == "__main__":
    test_T1_time_focus()
    test_T2_mechanism_and_grid_transparency()
    print("REFLECTRON TIME-FOCUS: ALL PASS")
