"""
SDS transport honours charge, on every route.

The model (physics.sds): the cross section comes from the ion's mass and
the mobility scales with charge, K_z = abs(z) K1(m), with the drift along
the force (the sign of z); diffusion is tied to the Einstein relation,
D = K1 k T / e, the same for every z. These tests pin that on planar,
r-z and 3-D, for z = +1, +2, -2, +3:
  * unit: the per-ion numbers scale as stated, and the jump scale gives a
    mean squared 3-D jump of exactly 6 D dt for the kernels' own draw;
  * drift: in a uniform field the flown (viscous) velocity is K_z * E,
    with the sign of z, at every record once relaxed;
  * diffusion: in no field the flown spread is <r^2> = 6 D t on every
    route (the planar route's gas is 3-D too), whatever the charge.
"""
import math

import numpy as np
import pytest

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)
from ion_gym.physics import sds
from ion_gym.physics.constants import E_CHG, K_B

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 10.0)]
CHARGES = [1, 2, -2, 3]
MASS, T_K, P_TORR, N2 = 322.0, 300.0, 7.6, 28.0134


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(coords, depth_mm, charge, volts, n_ions=1, t_max_us=20.0):
    """Two end plates 18 mm apart (x = 1 and x = 19), the left one at
    `volts`: a uniform axial field mid-cell on every route. One species,
    born at rest mid-cell."""
    geo = GeometrySpec(
        width_mm=20.0, height_mm=10.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[ElectrodeSpec(name="left", dc=volts,
                                  shapes=[_rect(0, 0, 1, 10)]),
                    ElectrodeSpec(name="right", dc=0.0,
                                  shapes=[_rect(19, 0, 1, 10)])])
    src = SourceSpec(distribution="point", n_ions=n_ions, x0_mm=10.0,
                     y0_mm=2.0 if coords == "rz" else 5.0,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0],
                     ke_lo=0.0, ke_hi=0.0, mz_list=[MASS], charge=charge,
                     tob_span_us=0.0, seed=11)
    col = CollisionSpec(enabled=True, gas="N2", T_k=T_K, P_torr=P_TORR,
                        model="sds", gas_diam_nm=0.366)
    return SimSpec(geometry=geo, name="sds_transport", source=src,
                   collisions=col,
                   integration=IntegrationSpec(
                       dt_ns=4.0, t_max_us=t_max_us, rec_every=50,
                       record_channels=[_axial_field(coords)]))


def _axial_field(coords):
    """The recorded field along x: r-z records it as the axial component."""
    return "e_axial" if coords == "rz" else "e_x"


def _params(charge):
    return sds.transport_params(MASS, charge, N2, 0.366, T_K, P_TORR)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


# ------------------------------------------------------------------ unit
def test_mobility_scales_with_charge_and_nothing_else_does():
    one = _params(1)
    for z in (2, -2, 3):
        p = _params(z)
        assert p["K_mm2_per_V_us"] == pytest.approx(abs(z) * one["K_mm2_per_V_us"])
        for k in ("damping", "D_mm2_per_us", "mfp_mm", "V_mm_us", "K0"):
            assert p[k] == one[k], k


def test_einstein_relation_and_reduced_mobility():
    p = _params(1)
    K1 = p["K0"] * 1e-4 * (760.0 / P_TORR) * (T_K / 273.15)
    assert p["K_mm2_per_V_us"] == pytest.approx(K1)
    assert p["D_mm2_per_us"] == pytest.approx(K1 * K_B * T_K / E_CHG)
    # relaxation rate e / (m K1), in 1/us
    assert p["damping"] == pytest.approx(
        E_CHG / (MASS * 1.66053906660e-27 * K1) * 1e-6, rel=1e-6)


def test_jump_scale_gives_six_D_dt():
    """The kernels jump d * sqrt(V mfp dt / N) with d from icdf_jump_steps;
    its mean square must be 6 D dt (sampled here independently)."""
    p = _params(1)
    stats = sds.jump_statistics()
    rng = np.random.default_rng(0)
    d = np.array([sds._icdf_steps_at(stats, p["log_mr_ratio"], u)
                  for u in rng.random(200000)])
    dt = 4e-3
    msq = np.mean(d ** 2) * p["V_mm_us"] * p["mfp_mm"] * dt / sds.N_DIST_COLLISIONS
    assert msq == pytest.approx(6.0 * p["D_mm2_per_us"] * dt, rel=0.02)


def test_isotropic_vector_has_the_length_and_no_preferred_axis():
    v = np.array([sds.isotropic_vector(2.0) for _ in range(40000)])
    assert np.allclose(np.linalg.norm(v, axis=1), 2.0)
    assert np.allclose(v.mean(axis=0), 0.0, atol=0.04)
    assert np.allclose((v ** 2).mean(axis=0), 4.0 / 3.0, rtol=0.03)


def test_paper_estimators_are_consistent():
    # the two air fits are near-inverses over the fitted range, and the
    # air-to-gas ratio is 1 for air itself
    for d in (0.6, 0.8, 1.2, 2.0):
        back = sds.diameter_from_air_K0(sds.air_K0_from_diameter(d))
        assert back == pytest.approx(d, rel=0.05)
    assert sds.gas_over_air(322.0, 0.8, sds.AIR_MASS_AMU,
                            sds.AIR_DIAMETER_NM) == pytest.approx(1.0)


# -------------------------------------------------------------- flights
@pytest.mark.parametrize("charge", CHARGES)
@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_drift_is_K_z_times_E_with_the_sign_of_z(route, coords, depth,
                                                 charge):
    from ion_gym.physics.sim_build import build_run
    _model, fly, cols, _b = build_run(_deck(coords, depth, charge, 40.0))
    traj = np.asarray(fly(0)[0])
    vx = traj[:, cols.index("vx")]
    ex = traj[:, cols.index(_axial_field(coords))]
    relaxed = traj[:, cols.index("t")] > 0.5          # >> 1/damping
    assert relaxed.sum() > 10
    k = vx[relaxed] / ex[relaxed]
    want = math.copysign(_params(charge)["K_mm2_per_V_us"], charge)
    assert np.allclose(k, want, rtol=1e-3), (k.min(), k.max(), want)


@pytest.mark.parametrize("charge", [1, 2])
@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_diffusion_is_einstein_whatever_the_charge(route, coords, depth,
                                                   charge):
    from ion_gym.physics.sim_build import build_run
    n = 400
    _model, fly, cols, _b = build_run(
        _deck(coords, depth, charge, 0.0, n_ions=n, t_max_us=20.0))
    ix, iy, iz, it = (cols.index(c) for c in ("x", "y", "z", "t"))
    r2, t_end = [], None
    for i in range(n):
        tr = np.asarray(fly(i)[0])
        d = tr[-1, [ix, iy, iz]] - tr[0, [ix, iy, iz]]
        r2.append(d @ d)
        t_end = tr[-1, it] - tr[0, it]
    want = 6 * _params(charge)["D_mm2_per_us"] * t_end
    assert np.mean(r2) == pytest.approx(want, rel=0.15)
