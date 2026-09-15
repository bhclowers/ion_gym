"""Gate: CMA-ES over a SimSpec recovers a better einzel focus than both the
unfocused lens and the hand-tuned -800 V reference (known-answer physics)."""
import tempfile
from ion_gym.physics.build_stl import einzel3d_spec
from ion_gym.physics.optimize import OptimizeSpec, optimize, _evaluate

# LOUD SKIP: the STL boolean path needs a trimesh boolean backend
# (manifold3d or blender) — an optional dependency, absent -> named
# skip, not an ImportError mid-test.
import pytest as _pytest
try:
    import manifold3d as _m3d  # noqa: F401
except ImportError:
    _pytest.skip("no trimesh boolean backend (pip install manifold3d) "
                 "— STL-boolean tests need it", allow_module_level=True)



def test_einzel_focus_recovery():
    base = einzel3d_spec(tempfile.mkdtemp(), lens_v=0.0)
    base.source.n_ions = 8
    osp = OptimizeSpec(params=["geometry.electrodes[1].dc"],
                       lo=[-2000.0], hi=[0.0], metric="exit_radius",
                       max_evals=30, sigma0=0.3, seed=2)
    res = optimize(base, osp)
    r0, _ = _evaluate(base, osp, [0.0], seed_base=1)
    r800, _ = _evaluate(base, osp, [-800.0], seed_base=1)
    assert res.best_value < r800 < r0, \
        f"optimizer did not beat references: {res.best_value} vs {r800}/{r0}"
    assert res.n_evals <= 36
    print(f"  einzel focus: {r0:.3f} (0V) / {r800:.3f} (-800V) -> "
          f"{res.best_value:.3f} mm at "
          f"{res.best_params['geometry.electrodes[1].dc']:.0f} V  OK")


def test_noisy_objective_crn_audit():
    """Gate: with He collisions ON (stochastic metric), rotating-CRN CMA-ES
    recovers an optimum that BEATS BOTH references under a PAIRED
    fresh-seed audit (same audit seeds for best and references), and the
    audit exposes noise (std > 0). This is the trust condition for every
    gas-phase optimization (reference stack, cooler, funnel).

    OPERATING POINT (re-derived; the gate was
    authored while the 3-D path silently flew in vacuum — same class as
    the stl3d closure): 0.6 Torr He gives 327 collisions/flight, not "a
    few", and every ion SPLATS mid-bore (transverse He random walk into
    the lens electrodes; t_max-independent, kind 0 at t~2-3 us,
    measured). At 0.002 Torr (~13 collisions/flight) the problem is
    well-posed and noisy: measured V-scan 0V 0.667+-0.093, -100V 0.422,
    -200V 0.176+-0.025 (the gas-era optimum: a WEAK lens collimates
    without wall scattering), -450V 1.43, -800V 3.88+-0.54. NOTE the
    reference ORDERING INVERTS under gas (0V beats -800V: the focused
    beam's slow passage scatters it into the walls), so the old
    vacuum-era side-claim r800 < r0 is retired — the trust condition is
    beating BOTH references, whichever order they fall in."""
    import math
    base = einzel3d_spec(tempfile.mkdtemp(), lens_v=0.0)
    base.source.n_ions = 8
    base.collisions.enabled = True
    base.collisions.gas = "He"
    base.collisions.T_k = 298.0
    base.collisions.P_torr = 0.002          # ~13 collisions per flight
    if hasattr(base.collisions, "P_pa"):
        base.collisions.P_pa = 0.002 * 133.322
    osp = OptimizeSpec(params=["geometry.electrodes[1].dc"],
                       lo=[-2000.0], hi=[0.0], metric="exit_radius",
                       max_evals=30, sigma0=0.3, seed=5,
                       n_repeats=1, crn="rotate", n_final=6)
    res = optimize(base, osp)
    assert math.isfinite(res.audit_value)
    assert res.audit_std > 0.0, "no noise detected — collisions not active?"
    audit_seed = 5 + 999_983                 # the SAME fresh seeds
    r0, _ = _evaluate(base, osp, [0.0], seed_base=audit_seed, n_repeats=6)
    r800, _ = _evaluate(base, osp, [-800.0], seed_base=audit_seed,
                        n_repeats=6)
    assert res.audit_value < min(r0, r800), \
        (f"noisy optimum not confirmed under paired audit: "
         f"{res.audit_value:.3f} vs 0V {r0:.3f} / -800V {r800:.3f}")
    print(f"  noisy CRN: audit {res.audit_value:.3f}±{res.audit_std:.3f} mm "
          f"beats 0V {r0:.3f} and -800V {r800:.3f} "
          f"(best V {res.best_params['geometry.electrodes[1].dc']:.0f})  OK")


if __name__ == "__main__":
    test_einzel_focus_recovery()
    test_noisy_objective_crn_audit()
    print("optimizer gates passed")
