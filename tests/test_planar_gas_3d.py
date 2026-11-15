"""
The planar route's buffer gas is three-dimensional, as on r-z and 3-D.

The planar kernel once passed vz = 0 to the shared hard-sphere step and
dropped the z result (a "2-D gas"), and SDS relaxed and diffused only in the
plane. Collisions shared energy over two axes, so light ions ran cold and
drifted fast (review D-4: m <vy^2>/kT 0.87 for 28 u in N2, 0.50 for 4 u in
Ar), and the gas never slowed the out-of-plane velocity (D-6: 8 eV along z
unchanged after 20 us at 1 Torr). A z gas flow was refused for the same
reason; it is now honoured.
"""
import math

import numpy as np
import pytest

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)
from ion_gym.physics.constants import KG_AMU, K_B

T_K = 300.0


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(coords, *, mass, gas, p_torr, model="hs", n_ions=100,
          direction=(1.0, 0.0, 0.0), ke=0.0, dt_ns=0.1, t_max_us=0.5,
          rec_every=50, flow=(0.0, 0.0, 0.0), depth_mm=0.0):
    geo = GeometrySpec(
        width_mm=10.0, height_mm=10.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[ElectrodeSpec(name="ground", dc=0.0,
                                  shapes=[_rect(0, 0, 10, 0.5)])])
    src = SourceSpec(distribution="point", n_ions=n_ions, x0_mm=5.0,
                     y0_mm=5.0 if coords == "xyz" else 3.0,
                     z0_mm=depth_mm / 2, direction=list(direction),
                     ke_lo=ke, ke_hi=ke, temperature_k=T_K, mz_list=[mass],
                     tob_span_us=0.0, seed=8)
    col = CollisionSpec(enabled=True, gas=gas, T_k=T_K, P_torr=p_torr,
                        model=model, sigma_model="fixed", sigma_m2=1.0e-18,
                        flow_mm_us=list(flow))
    return SimSpec(geometry=geo, name="planar_gas_3d", source=src,
                   collisions=col,
                   integration=IntegrationSpec(dt_ns=dt_ns, t_max_us=t_max_us,
                                               rec_every=rec_every))


def _fly_all(spec):
    from ion_gym.physics.sim_build import build_run
    _m, fly, cols, _b = build_run(spec)
    return [np.asarray(fly(i)[0]) for i in range(spec.source.n_ions)], cols


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


@pytest.mark.parametrize("mass,gas,p_torr", [(4.0, "Ar", 10.0),
                                              (28.0, "N2", 10.0)],
                         ids=["4u_Ar", "28u_N2"])
@pytest.mark.parametrize("coords", ["xyz", "rz"], ids=["planar", "rz"])
def test_light_ions_reach_the_gas_temperature_on_every_axis(coords, mass,
                                                            gas, p_torr):
    """Zero field: per-axis m <v_i^2> / kT over the equilibrated part of
    the flight is 1, on planar as on r-z."""
    trajs, cols = _fly_all(_deck(coords, mass=mass, gas=gas, p_torr=p_torr))
    it = cols.index("t")
    v = np.concatenate([tr[tr[:, it] > 0.1][:, [cols.index(c)
                                               for c in ("vx", "vy", "vz")]]
                        for tr in trajs])
    ratio = mass * KG_AMU * np.mean((v * 1e3) ** 2, axis=0) / (K_B * T_K)
    assert np.allclose(ratio, 1.0, atol=0.08), ratio


@pytest.mark.parametrize("model", ["hs", "sds"])
def test_the_gas_slows_the_out_of_plane_velocity(model):
    """8 eV along z at 1 Torr N2: the gas relaxes vz (1/gamma about 0.16
    us under SDS); it used to cross the gas ballistically along z."""
    spec = _deck("xyz", mass=100.0, gas="N2", p_torr=1.0, model=model,
                 n_ions=20, direction=(0.0, 0.0, 1.0), ke=8.0, dt_ns=1.0,
                 t_max_us=5.0, rec_every=100)
    trajs, cols = _fly_all(spec)
    ivz = cols.index("vz")
    v0 = math.sqrt(2 * 8.0 * 1.602176634e-19 / (100.0 * KG_AMU)) * 1e-3
    # born with the 8 eV axial speed (plus the 300 K thermal spread)
    assert np.mean([tr[0, ivz] for tr in trajs]) == pytest.approx(v0,
                                                                  rel=0.05)
    v_end = np.mean([tr[-1, ivz] for tr in trajs])
    assert abs(v_end) < 0.1 * v0, (v_end, v0)


@pytest.mark.parametrize("model", ["hs", "sds"])
@pytest.mark.parametrize("coords,depth", [("xyz", 0.0), ("xyz", 4.0)],
                         ids=["planar", "3d"])
def test_a_z_gas_flow_is_honoured(coords, depth, model):
    """A gas flowing along z carries the ions with it on planar as on 3-D
    (a planar deck declaring it was refused)."""
    spec = _deck(coords, mass=100.0, gas="N2", p_torr=5.0, model=model,
                 n_ions=40, dt_ns=0.5, t_max_us=3.0, rec_every=100,
                 flow=(0.0, 0.0, 0.05), depth_mm=depth)
    trajs, cols = _fly_all(spec)
    it, ivz = cols.index("t"), cols.index("vz")
    vz = np.concatenate([tr[tr[:, it] > 1.0][:, ivz] for tr in trajs])
    assert np.mean(vz) == pytest.approx(0.05, abs=0.015), np.mean(vz)


def test_sds_diffuses_along_z_on_planar():
    from ion_gym.physics import sds
    spec = _deck("xyz", mass=322.0, gas="N2", p_torr=7.6, model="sds",
                 n_ions=400, dt_ns=4.0, t_max_us=20.0, rec_every=1000)
    trajs, cols = _fly_all(spec)
    iz, it = cols.index("z"), cols.index("t")
    dz2 = np.mean([(tr[-1, iz] - tr[0, iz]) ** 2 for tr in trajs])
    t_end = trajs[0][-1, it] - trajs[0][0, it]
    D = sds.sds_ion_params(spec.collisions, 322.0, 1)["D_mm2_per_us"]
    assert dz2 == pytest.approx(2 * D * t_end, rel=0.2)
