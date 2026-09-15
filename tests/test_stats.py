"""
test_stats.py — gate the impact-statistics reducer.

Two layers:
  1. Synthetic summaries in the exact builder format (kind/tof/x_end/...)
     with known answers: fate counts, transmission, per-m/z grouping,
     R = t/(2*FWHM) against a hand-computed value.
  2. A real planar-einzel run (validate_einzel_planar.einzel_spec) reduced
     end-to-end: fate counts must sum to n, and the reducer must agree
     with a direct recount of r.summary — GUI and gates share this path.
"""
import math
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from ion_gym.physics.stats import (compute_stats, stats_markdown, auto_transmitted_fate,
                   mz_of_results, FWHM_PER_SIGMA)


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



class R:
    def __init__(self, s):
        self.summary = s


def test_synthetic():
    # two m/z groups, transmitted fate 3 (detector bound), plus losses
    res, mz = [], []
    # m/z 100: 4 ions, TOFs 10.0, 10.1, 9.9, 10.0 us  (mean 10.0, sd known)
    for t in (10.0, 10.1, 9.9, 10.0):
        res.append(R(dict(kind=3, tof=t, x_end=30.0, y_end=0.1, z_end=0.0,
                          r_end=0.1, n_col=0)))
        mz.append(100.0)
    # m/z 400: 2 ions, TOFs 20.0, 20.2
    for t in (20.0, 20.2):
        res.append(R(dict(kind=3, tof=t, x_end=30.0, y_end=-0.1, z_end=0.0,
                          r_end=0.1, n_col=0)))
        mz.append(400.0)
    # losses: 2 electrode impacts, 1 domain exit, 1 timeout
    for k in (0, 0, 1, 2):
        res.append(R(dict(kind=k, tof=1.0, x_end=5.0, y_end=2.0, z_end=0.0,
                          r_end=2.0, n_col=0)))
        mz.append(100.0)

    st = compute_stats(res, transmitted_fate=3, mz=mz)
    assert st.n_total == 10
    assert st.fate_counts == {3: 6, 0: 2, 1: 1, 2: 1}
    assert abs(st.transmission - 0.6) < 1e-12
    assert [g.mz for g in st.per_mz] == [100.0, 400.0]

    g = st.per_mz[0]
    xs = [10.0, 10.1, 9.9, 10.0]
    mu = sum(xs) / 4
    # POPULATION sd (ddof=0), matching stats._mean_std: the
    # flown ions ARE the population of the
    # simulation, and every banked R in the tree is ddof=0. This line used
    # (n - 1); at n = 4 the two conventions differ by sqrt(4/3) = 15.5%,
    # which is exactly the gap this test was reporting (30.028 vs 26.005).
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / 4)
    R_expect = mu / (2 * FWHM_PER_SIGMA * sd)
    assert abs(g.tof_mean - mu) < 1e-12
    assert abs(g.resolving_power - R_expect) / R_expect < 1e-12
    assert st.impact_spread["x"][0] == 5.0        # fate-0 locations tracked
    md = stats_markdown(st)
    assert "m/z" in md and "60.0%" in md
    print(f"  synthetic: fates {st.fate_counts}, T=60%, "
          f"R(mz100)={g.resolving_power:.1f} (expect {R_expect:.1f})  OK")


def test_auto_fate_and_mz_cycling():
    class B:
        x_min_on = x_max_on = y_min_on = y_max_on = z_min_on = z_max_on = False
    class Src:
        mz_list = [100.0, 400.0]
    class Spec:
        bounds = B()
        source = Src()
    s = Spec()
    assert auto_transmitted_fate(s) == 1          # no bounds -> domain exit
                                                  # (timeout is never
                                                  # transmission)
    s.bounds.x_max_on = True
    assert auto_transmitted_fate(s) == 3          # bound on -> plane fate
    assert mz_of_results(s, 5) == [100.0, 400.0, 100.0, 400.0, 100.0]
    print("  auto-fate heuristic + m/z cycling  OK")


def test_real_einzel():
    from ion_gym.physics.sim_build import build_run
    from ion_gym.physics.ensemble_driver import run

    spec = _einzel_spec(-100.0)
    model, fly, cols, births = build_run(spec)
    res = run(len(births), fly, check_every=5, decimate=1)

    fate = auto_transmitted_fate(spec)
    mz = mz_of_results(spec, len(res.results))
    st = compute_stats(res.results, transmitted_fate=fate, mz=mz)

    # reducer must agree with a direct recount
    assert st.n_total == len(res.results)
    assert sum(st.fate_counts.values()) == st.n_total
    direct = {}
    for r in res.results:
        k = r.summary["kind"]
        direct[k] = direct.get(k, 0) + 1
    assert st.fate_counts == direct
    n_ok = sum(1 for r in res.results if r.summary["kind"] == fate)
    assert st.n_transmitted == n_ok
    stats_markdown(st)                            # renders without error
    print(f"  real einzel: {st.n_total} ions, fates {st.fate_counts}, "
          f"auto fate={fate}, T={st.transmission*100:.0f}%  OK")


if __name__ == "__main__":
    test_synthetic()
    test_auto_fate_and_mz_cycling()
    test_real_einzel()
    print("\nSTATS GATE: ALL PASS")


def test_S_exit_plane_is_exact_and_decimation_independent():
    """M-STAT-X: the fate-3 exit state must lie ON the plane, and must not
    depend on rec_every.

    The bounding-plane crossing used to be found post-hoc by scanning the
    RECORDED trajectory, which is decimated: the reported exit state was the
    first recorded sample PAST the plane -- up to rec_every*dt of overshoot
    (~2 mm at rec_every=200, dt=8 ns, 1.3 mm/us). Exit-plane transverse spreads
    are 0.06-0.3 mm, so the statistic was dominated by the SAMPLING RATE rather
    than by the ion optics. It cannot be repaired by interpolating recorded
    points either: under RF they are several cycles apart.

    The kernel now detects the crossing per step and interpolates within the
    step. So: same ions, two very different rec_every -> identical exit stats.
    """
    import numpy as np
    from ion_gym.physics.scene3d import GeomScene, GridSpec, Electrode, Shape, Box3D
    from ion_gym.physics.build_scene3d import simspec_from_scene
    from ion_gym.physics.sim_build import build_run

    els = [Electrode(index=1, name="IN", voltage=500.0,
                     shapes=[Shape(within=[Box3D(0, 0, 0, 12, 12, 1)])]),
           Electrode(index=2, name="OUT", voltage=0.0,
                     shapes=[Shape(within=[Box3D(0, 0, 59, 12, 12, 60)])])]
    sc = GeomScene(grid=GridSpec(nx=2, ny=2, nz=2, mm_per_gu=1.0), electrodes=els,
               units="mm", name="exit_gate").at_resolution(1.0, margin_mm=1.0)
    sp = simspec_from_scene(sc)
    sp.source.n_ions = 8
    sp.source.distribution = "disc"
    sp.source.axis = "z"
    sp.source.r_mm = 1.0
    sp.source.x0_mm, sp.source.y0_mm, sp.source.z0_mm = 7.0, 7.0, 5.0
    sp.source.direction = [0, 0, 1]
    sp.source.mz_list = [115.0]
    sp.source.ke_lo = sp.source.ke_hi = 5.0
    # Seed so both rec_every values fly identical ions (random
    # is now the default; this test checks rec_every doesn't move
    # statistics, which requires identical ion ensemble both runs).
    sp.source.seed = 7
    sp.integration.dt_ns = 10.0
    sp.integration.t_max_us = 400.0
    sp.bounds.z_max_on = True
    sp.bounds.z_max = 50.0

    got = {}
    for rec in (5, 300):
        sp.integration.rec_every = rec
        _m, fly, _c, _b = build_run(sp, verbose=False)
        rows = [fly(i)[1] for i in range(sp.source.n_ions)]
        ex = [r for r in rows if r["kind"] == 3 and r["z_end"] > 25]
        assert ex, f"rec_every={rec}: nothing reached the plane"
        z = np.array([r["z_end"] for r in ex])
        # ON the plane, not past it
        assert np.allclose(z, 50.0, atol=1e-6), (
            f"rec_every={rec}: exit z = {z.mean():.4f} (plane is 50.0) — the "
            f"crossing is being read off the decimated trajectory")
        got[rec] = (len(ex),
                    float(np.mean([r["x_end"] for r in ex])),
                    float(np.mean([r["vx_end"] for r in ex])))

    # and the statistic must not depend on how often we wrote samples down
    assert got[5][0] == got[300][0], "rec_every changed the exit COUNT"
    assert abs(got[5][1] - got[300][1]) < 1e-9, "rec_every moved the centroid"
    assert abs(got[5][2] - got[300][2]) < 1e-9, "rec_every changed exit vx"


# ---------------------------------------------------------------------------
# stats_markdown was SHIPPED with two NameErrors that no test could see,
# because no test ever CALLED it on results that reach its later branches.
#
#   1. `FATE_NAMES.get(p.fate, ...)` — typo for FATE_NAME. Lives in the
#      "exit velocity" table, which only renders when a landing group has a
#      non-zero exit velocity. Every fixture up to now terminated ions
#      without one, so the line never executed.
#   2. `vrows` was bound INSIDE `if st.landing:` and read OUTSIDE it, so a
#      run in which nothing lands (all timeouts) raised NameError: vrows.
#
# Both were found by RUNNING the app, not by the suite. The fix is not
# the two characters; it is these tests. They render the markdown for every
# reachable combination of branches.
# ---------------------------------------------------------------------------
def _r(kind, **kw):
    """A result in the shape the stats layer actually consumes:
    an object with .summary = dict(kind=..., x_end=..., vx_end=..., ...).
    Keys verified against stats.py, not guessed."""
    d = dict(kind=kind, tof=100.0, x_end=0.0, y_end=0.0, z_end=0.0,
             r_end=0.0, n_col=30, ke_end=0.2,
             vx_end=0.0, vy_end=0.0, vz_end=0.0)
    d.update(kw)
    return R(d)


def test_stats_markdown_renders_exit_velocity_branch():
    """The branch that shipped broken: a bounding-plane fate WITH velocity."""
    import random
    rng = random.Random(3)
    res = [_r(3, z_end=150.7,                      # exactly on the exit plane
              x_end=rng.gauss(0, 0.15), y_end=rng.gauss(0, 0.15),
              vx_end=rng.gauss(0, 0.02), vy_end=rng.gauss(0, 0.02),
              vz_end=0.13,                          # non-zero -> vrows non-empty
              ke_end=0.17, n_col=129, tof=780.0)
           for _ in range(40)]
    st = compute_stats(res, transmitted_fate=3, mz=[180.0] * 40)
    md = stats_markdown(st)                # <-- raised NameError: FATE_NAMES
    assert "exit velocity" in md
    assert "divergence" in md
    assert "bounding plane" in md          # FATE_NAME actually resolved
    assert "FATE_NAMES" not in md


def test_stats_markdown_survives_no_landings():
    """st.landing is falsy ONLY when no ions came back at all (verified: an
    all-timeout run still produces a landing group for fate 2). That is the
    real reach of the bug -- a run that yields nothing, which happens. With
    `vrows` bound inside `if st.landing:` this raised NameError: vrows."""
    st = compute_stats([], transmitted_fate=3, mz=[])
    assert not st.landing and st.n_total == 0
    md = stats_markdown(st)                # <-- raised NameError: vrows
    assert "0 ions" in md
    assert "exit velocity" not in md       # correctly absent, not crashed


def test_stats_markdown_all_timeouts():
    """A run where nothing lands on the exit plane must still render."""
    res = [_r(2, tof=2000.0) for _ in range(5)]
    st = compute_stats(res, transmitted_fate=3, mz=[180.0] * 5)
    md = stats_markdown(st)
    assert "timeout" in md
    assert "exit velocity" not in md


def test_stats_markdown_every_fate_code():
    """Render with 0/1/2/3 all present — exercises every table at once."""
    from ion_gym.physics.stats import FATE_NAME
    res, mz = [], []
    for code in (0, 1, 2, 3):
        for i in range(6):
            res.append(_r(code, x_end=0.1 * i, y_end=-0.1 * i,
                          z_end=10.0 * code, vz_end=0.12,
                          vx_end=0.01, vy_end=0.01))
            mz.append(115.0)
    md = stats_markdown(compute_stats(res, transmitted_fate=3, mz=mz))
    for code, name in FATE_NAME.items():
        assert name in md, f"fate {code} ({name}) missing from the report"
