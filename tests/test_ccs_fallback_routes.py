"""
Where the CCS fit does not apply (a gas other than N2), a hard-sphere deck
flies the static collisions.sigma_m2 for every ion, loudly. The fallback
must mean the same thing on every route: a deck on the fallback flies
exactly as the same deck stating sigma_model "fixed", on planar, r-z and
3-D. A multiply charged ion in N2 flies the fit at its MASS, the same on
every route. SDS reads no cross section and is unaffected.
"""
import dataclasses

import numpy as np
import pytest

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 2.0)]
CASES = [("He", 1), ("He", 2)]


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(coords, depth_mm, gas, charge, sigma_model, model="hs"):
    geo = GeometrySpec(
        width_mm=10.0, height_mm=6.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[ElectrodeSpec(name="top", dc=5.0,
                                  shapes=[_rect(0, 5, 10, 1)]),
                    ElectrodeSpec(name="bot", dc=0.0,
                                  shapes=[_rect(0, 0, 10, 1)])])
    src = SourceSpec(distribution="point", n_ions=2, x0_mm=2.0, y0_mm=3.0,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0],
                     ke_lo=1.0, ke_hi=1.0, mz_list=[322.0], charge=charge,
                     tob_span_us=0.0, seed=5)
    col = CollisionSpec(enabled=True, gas=gas, T_k=300.0, P_torr=1.0,
                        model=model, sigma_m2=1.25e-18,
                        sigma_model=sigma_model)
    return SimSpec(geometry=geo, name=f"fallback_{coords}", source=src,
                   collisions=col,
                   integration=IntegrationSpec(dt_ns=2.0, t_max_us=2.0,
                                               rec_every=5))


def _fly(spec):
    from ion_gym.physics.sim_build import build_run
    _model, fly, _cols, _births = build_run(spec)
    return [np.asarray(fly(i)[0]) for i in range(spec.source.n_ions)]


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


@pytest.mark.parametrize("gas,charge", CASES, ids=["He", "He_z2"])
@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_fallback_flies_as_fixed_on_every_route(route, coords, depth, gas,
                                                charge):
    fell_back = _deck(coords, depth, gas, charge, "ccs_fit")
    assert not [e for e in fell_back.validate() if "ccs_fit" in e]
    assert any("does not apply" in a for a in fell_back.advisories())
    stated = _deck(coords, depth, gas, charge, "fixed")
    for a, b in zip(_fly(fell_back), _fly(stated)):
        assert a.shape == b.shape and np.array_equal(a, b)


@pytest.mark.parametrize("charge", [2, -3])
@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_a_multiply_charged_ion_flies_the_fit_at_its_mass(route, coords,
                                                          depth, charge):
    """Under the CCS fit a z = 2 or -3 ion in N2 flies exactly as the same
    deck stating sigma_model 'fixed' at the fit's value for its mass."""
    from ion_gym.physics import ccs
    fit = _deck(coords, depth, "N2", charge, "ccs_fit")
    assert not any("does not apply" in a for a in fit.advisories())
    assert any("upper estimate" in a for a in fit.advisories())
    stated = _deck(coords, depth, "N2", charge, "fixed")
    stated.collisions.sigma_m2 = ccs.ion_sigma_m2(fit.collisions, 322.0, 1)
    for a, b in zip(_fly(fit), _fly(stated)):
        assert a.shape == b.shape and np.array_equal(a, b)


def test_app_readout_shows_the_fallback_sigma():
    """The app's sigma readout is display == solver input: under the
    fallback it names the fixed value and leaves the field editable."""
    from ion_gym.io.paths import repo_root
    from ion_gym.ui.sim_app import SimApp
    app = SimApp(SimSpec.from_json(
        str(repo_root() / "tests/fixtures/einzel_rz_gate_fixture.json")))
    app.w_gas_on.value = True
    app.w_col_model.value = "hs"
    app.w_sigma_model.value = "ccs_fit"
    app.w_sigma.value = 1.25e-18
    app.w_gas.value = "N2"
    app._update_sigma_honored()
    assert "the CCS fit" in app.w_sigma_honored.object
    assert app.w_sigma.disabled
    app.w_gas.value = "He"
    app._update_sigma_honored()
    assert "(fallback)" in app.w_sigma_honored.object
    assert "1.25e-18" in app.w_sigma_honored.object
    assert not app.w_sigma.disabled


@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_sds_in_helium_is_unaffected(route, coords, depth):
    """SDS reads no cross section: it flies, says nothing about a
    hard-sphere fallback, and the sigma model changes nothing. (Planar and
    r-z once refused it at flight while 3-D flew it.)"""
    import warnings
    spec = _deck(coords, depth, "He", 1, "ccs_fit", model="sds")
    assert not any("does not apply" in a for a in spec.advisories())
    stated = dataclasses.replace(
        spec, collisions=dataclasses.replace(spec.collisions,
                                             sigma_model="fixed"))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        for a, b in zip(_fly(spec), _fly(stated)):
            assert a.shape == b.shape and np.array_equal(a, b)
    assert not [w for w in seen if "does not apply" in str(w.message)]
