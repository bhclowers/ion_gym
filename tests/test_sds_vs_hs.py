"""
SDS agrees with hard-sphere (HS) collisions at the same cross section.

HS resolves every collision, so its drift and diffusion come from first
principles, charge included. SDS is given the first-order Mason-Schamp
mobility for the same cross section (declared at charge 1), scales it with
charge and ties diffusion to the Einstein relation. On planar, r-z and 3-D,
the SDS drift mobility matches HS for z = 1 and 3, and the SDS diffusion
matches HS with no field, within max(5 %, 3 standard errors).

Slow (minutes): `pytest -m slow tests/test_sds_vs_hs.py`. The full table
for the documentation is printed by the study script beside the SDS model
write-up.
"""
import math

import numpy as np
import pytest

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)
from ion_gym.physics import ccs

pytestmark = pytest.mark.slow

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 10.0)]
MASS, T_K, P_TORR, VOLTS, GAP_MM = 322.0, 300.0, 2.0, 4.5, 18.0
N_IONS, T_MAX_US = 400, 50.0


def _sigma_and_K0():
    ref = CollisionSpec(enabled=True, gas="N2", T_k=T_K, P_torr=P_TORR,
                        model="hs", sigma_model="ccs_fit")
    sigma = ccs.ion_sigma_m2(ref, MASS, 1)
    fixed = CollisionSpec(enabled=True, gas="N2", T_k=T_K, P_torr=P_TORR,
                          model="hs", sigma_model="fixed", sigma_m2=sigma)
    K1 = ccs.low_field_mobility_m2Vs(fixed, MASS, 1)
    return sigma, K1 * 1e4 * (P_TORR / 760.0) * (273.15 / T_K)


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(coords, depth_mm, charge, model, volts):
    sigma, K0 = _sigma_and_K0()
    geo = GeometrySpec(
        width_mm=20.0, height_mm=10.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[ElectrodeSpec(name="left", dc=volts * np.sign(charge),
                                  shapes=[_rect(0, 0, 1, 10)]),
                    ElectrodeSpec(name="right", dc=0.0,
                                  shapes=[_rect(19, 0, 1, 10)])])
    src = SourceSpec(distribution="point", n_ions=N_IONS, x0_mm=6.0,
                     y0_mm=2.0 if coords == "rz" else 5.0,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0],
                     ke_lo=0.0, ke_hi=0.0, temperature_k=T_K,
                     mz_list=[MASS], charge=charge, tob_span_us=0.0, seed=21)
    if model == "hs":
        col = CollisionSpec(enabled=True, gas="N2", T_k=T_K, P_torr=P_TORR,
                            model="hs", sigma_model="fixed", sigma_m2=sigma)
        dt = 1.0
    else:
        col = CollisionSpec(enabled=True, gas="N2", T_k=T_K, P_torr=P_TORR,
                            model="sds", gas_diam_nm=0.366,
                            sds_species=[{"mass_amu": MASS, "gas": "N2",
                                          "charge": 1,
                                          "K0_cm2_per_Vs": K0}])
        dt = 5.0
    return SimSpec(geometry=geo, name=f"sds_vs_hs_{model}", source=src,
                   collisions=col,
                   integration=IntegrationSpec(dt_ns=dt, t_max_us=T_MAX_US,
                                               rec_every=1000))


def _displacements(spec):
    from ion_gym.physics.sim_build import build_run
    _m, fly, cols, _b = build_run(spec)
    ix, iy, iz, it = (cols.index(c) for c in ("x", "y", "z", "t"))
    d, dt = [], []
    for i in range(N_IONS):
        tr = np.asarray(fly(i)[0])
        k = int(np.searchsorted(tr[:, it], 1.0))
        d.append(tr[-1, [ix, iy, iz]] - tr[k, [ix, iy, iz]])
        dt.append(tr[-1, it] - tr[k, it])
    return np.asarray(d), np.asarray(dt)


def _mean_se(v):
    return float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v)))


def _agree(a, sa, b, sb):
    ratio = a / b
    se = ratio * math.hypot(sa / a, sb / b)
    assert abs(ratio - 1.0) <= max(0.05, 3 * se), (ratio, se)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


@pytest.mark.parametrize("charge", [1, 3])
@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_sds_drift_matches_hard_sphere(route, coords, depth, charge):
    e = VOLTS / GAP_MM
    out = []
    for model in ("hs", "sds"):
        d, dt = _displacements(_deck(coords, depth, charge, model, VOLTS))
        out.append(_mean_se(d[:, 0] / dt / e))
    _agree(out[1][0], out[1][1], out[0][0], out[0][1])


@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_sds_diffusion_matches_hard_sphere(route, coords, depth):
    dims = 3        # every route's gas is 3-D, the planar route's included
    out = []
    for model in ("hs", "sds"):
        d, dt = _displacements(_deck(coords, depth, 1, model, 0.0))
        out.append(_mean_se(np.sum(d[:, :dims] ** 2, axis=1)
                            / (2 * dims * dt)))
    _agree(out[1][0], out[1][1], out[0][0], out[0][1])
