"""
SDS species values declared in the deck (collisions.sds_species): a
measured K0 and/or ion diameter per ion mass, each K0 tagged with the gas
it was measured in.

  * a K0 measured in the deck's gas flies as given, scaled by abs(z), on
    planar, r-z and 3-D;
  * a K0 measured in another gas is converted by the Mason-Schamp
    hard-sphere ratio at the ion's diameter, and a WARNING advisory names it;
  * a diameter alone gives K0 from the paper's air fit at that diameter;
  * undeclared masses are estimated from mass, as before;
  * malformed entries are refused by name at every door;
  * a deck without the field saves exactly as before.
"""
import math

import numpy as np
import pytest

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)
from ion_gym.physics import sds

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 10.0)]
MASS, T_K, P_TORR, N2, HE = 322.0, 300.0, 7.6, 28.0134, 4.002602
D_N2, D_HE = 0.366, 0.218


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(coords, depth_mm, charge, species, volts=40.0):
    geo = GeometrySpec(
        width_mm=20.0, height_mm=10.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[ElectrodeSpec(name="left", dc=volts,
                                  shapes=[_rect(0, 0, 1, 10)]),
                    ElectrodeSpec(name="right", dc=0.0,
                                  shapes=[_rect(19, 0, 1, 10)])])
    src = SourceSpec(distribution="point", n_ions=1, x0_mm=10.0,
                     y0_mm=2.0 if coords == "rz" else 5.0,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0],
                     ke_lo=0.0, ke_hi=0.0, mz_list=[MASS], charge=charge,
                     tob_span_us=0.0, seed=4)
    col = CollisionSpec(enabled=True, gas="N2", T_k=T_K, P_torr=P_TORR,
                        model="sds", gas_diam_nm=D_N2,
                        sds_species=list(species))
    field = "e_axial" if coords == "rz" else "e_x"
    return SimSpec(geometry=geo, name="sds_species", source=src,
                   collisions=col,
                   integration=IntegrationSpec(dt_ns=4.0, t_max_us=2.0,
                                               rec_every=50,
                                               record_channels=[field])), field


def _K1(K0):
    """mm^2/(V us) at the test's T and P for a K0 in cm^2/(V s)."""
    return K0 * 1e-4 * (760.0 / P_TORR) * (T_K / 273.15)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


@pytest.mark.parametrize("charge", [1, -2, 3])
@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_a_declared_K0_flies_as_given_on_every_route(route, coords, depth,
                                                     charge):
    from ion_gym.physics.sim_build import build_run
    spec, field = _deck(coords, depth, charge,
                        [{"mass_amu": MASS, "gas": "N2", "charge": 1,
                          "K0_cm2_per_Vs": 2.0}])
    _m, fly, cols, _b = build_run(spec)
    tr = np.asarray(fly(0)[0])
    relaxed = tr[:, cols.index("t")] > 0.5
    k = tr[relaxed, cols.index("vx")] / tr[relaxed, cols.index(field)]
    assert np.allclose(k, math.copysign(abs(charge) * _K1(2.0), charge),
                       rtol=1e-3), (k.min(), k.max())


@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_a_K0_measured_at_its_charge_flies_exactly_for_that_charge(
        route, coords, depth):
    """A K0 measured for the 2+ ion is that ion's mobility: a 2+ ion flies
    it as given, a 1+ ion of the same mass half of it (fixed cross
    section), and both diffuse with Einstein at the measured value."""
    from ion_gym.physics.sim_build import build_run
    entry = {"mass_amu": MASS, "gas": "N2", "charge": 2,
             "K0_cm2_per_Vs": 2.0}
    for z, want in ((2, _K1(2.0)), (-2, -_K1(2.0)), (1, 0.5 * _K1(2.0))):
        spec, field = _deck(coords, depth, z, [entry])
        _m, fly, cols, _b = build_run(spec)
        tr = np.asarray(fly(0)[0])
        relaxed = tr[:, cols.index("t")] > 0.5
        k = tr[relaxed, cols.index("vx")] / tr[relaxed, cols.index(field)]
        assert np.allclose(k, want, rtol=1e-3), (z, k.min(), k.max())
    p = sds.transport_params(MASS, 2, N2, D_N2, T_K, P_TORR, gas_name="N2",
                             species=entry)
    from ion_gym.physics.constants import E_CHG, K_B
    assert p["D_mm2_per_us"] == pytest.approx(
        _K1(2.0) * K_B * T_K / (2 * E_CHG))


def test_a_K0_from_another_gas_is_converted_and_warned():
    he = {"mass_amu": MASS, "gas": "He", "charge": 1, "K0_cm2_per_Vs": 6.0,
          "d_nm": 0.9, "gas_d_nm": D_HE}
    p = sds.transport_params(MASS, 1, N2, D_N2, T_K, P_TORR,
                             gas_name="N2", species=he)
    want = 6.0 * (sds.gas_over_air(MASS, 0.9, N2, D_N2)
                  / sds.gas_over_air(MASS, 0.9, HE, D_HE))
    assert p["K0"] == pytest.approx(want)
    assert "CONVERTED" in p["K0_source"]
    # the conversion is reversible: the N2 value declared back in He
    back = sds.resolve_K0(MASS, "He", HE, D_HE,
                          {"mass_amu": MASS, "gas": "N2", "charge": 1,
                           "K0_cm2_per_Vs": p["K0"], "d_nm": 0.9,
                           "gas_d_nm": D_N2})[0]
    assert back == pytest.approx(6.0)
    spec, _f = _deck("xyz", 0.0, 1, [he])
    warn = [a for a in spec.advisories() if a.startswith("WARNING: SDS")]
    assert len(warn) == 1 and "He" in warn[0] and "m 322" in warn[0]


def test_a_diameter_alone_and_an_undeclared_mass():
    p = sds.transport_params(MASS, 1, N2, D_N2, T_K, P_TORR, gas_name="N2",
                             species={"mass_amu": MASS, "d_nm": 1.0})
    assert p["K0"] == pytest.approx(sds.air_K0_from_diameter(1.0)
                                    * sds.gas_over_air(MASS, 1.0, N2, D_N2))
    est = sds.transport_params(MASS, 1, N2, D_N2, T_K, P_TORR, gas_name="N2")
    assert est["K0"] == pytest.approx(sds.estimated_K0(MASS, N2, D_N2)[0])
    assert est["K0_source"] == "estimated from mass"


def test_a_K0_without_a_diameter_finds_one_from_the_air_fit():
    p = sds.transport_params(MASS, 1, N2, D_N2, T_K, P_TORR, gas_name="N2",
                             species={"mass_amu": MASS, "gas": "N2",
                                      "charge": 1, "K0_cm2_per_Vs": 1.4})
    assert p["K0"] == 1.4
    d0 = sds.diameter_from_mass_nm(MASS)
    assert p["d_ion_nm"] == pytest.approx(sds.diameter_from_air_K0(
        1.4 / sds.gas_over_air(MASS, d0, N2, D_N2)))


def test_the_advisory_lists_what_each_species_flies():
    spec, _f = _deck("xyz", 0.0, 1, [{"mass_amu": MASS, "gas": "N2",
                                      "charge": 1, "K0_cm2_per_Vs": 2.0}])
    spec.source.mz_list = [MASS, 622.0]
    note = [a for a in spec.advisories()
            if a.startswith("collisions: SDS reduced mobilities")]
    assert len(note) == 1
    assert "m 322: K0 2 (declared 2 in N2 at charge +1)" in note[0]
    assert "m 622" in note[0] and "estimated from mass" in note[0]


def test_species_under_hard_sphere_are_named_as_unused():
    spec, _f = _deck("xyz", 0.0, 1, [{"mass_amu": MASS, "d_nm": 1.0}])
    spec.collisions.model = "hs"
    assert any("does not read them" in a for a in spec.advisories())


@pytest.mark.parametrize("entry,needle", [
    ({"mass_amu": MASS}, "neither"),
    ({"mass_amu": -1.0, "d_nm": 1.0}, "mass_amu"),
    ({"mass_amu": MASS, "d_nm": 0.0}, "d_nm"),
    ({"mass_amu": MASS, "charge": 1, "K0_cm2_per_Vs": 1.4}, "measured in"),
    ({"mass_amu": MASS, "gas": "Ne", "charge": 1, "K0_cm2_per_Vs": 1.4},
     "not one"),
    ({"mass_amu": MASS, "gas": "He", "charge": 1, "K0_cm2_per_Vs": 6.0},
     "gas_d_nm"),
    ({"mass_amu": MASS, "gas": "N2", "K0_cm2_per_Vs": 1.4}, "charge state"),
    ({"mass_amu": MASS, "gas": "N2", "charge": 0, "K0_cm2_per_Vs": 1.4},
     "charge state"),
    ({"mass_amu": MASS, "d_nm": 1.0, "Ko": 1.4}, "unknown key"),
])
def test_malformed_entries_are_refused_by_name(entry, needle):
    spec, _f = _deck("xyz", 0.0, 1, [entry])
    errs = [e for e in spec.validate() if "sds_species" in e]
    assert errs and any(needle in e for e in errs), errs
    from ion_gym.physics.collision3d import resolve_collisions
    with pytest.raises(ValueError, match="sds_species"):
        resolve_collisions(spec.collisions)


def test_a_mass_declared_twice_is_refused():
    spec, _f = _deck("xyz", 0.0, 1, [{"mass_amu": MASS, "d_nm": 1.0},
                                     {"mass_amu": MASS, "d_nm": 1.1}])
    assert any("declared twice" in e for e in spec.validate())


def test_saving_is_unchanged_without_the_field_and_round_trips_with_it():
    plain = CollisionSpec(enabled=True, model="sds").to_dict()
    assert "sds_species" not in plain
    entries = [{"mass_amu": MASS, "gas": "N2", "charge": 1,
                "K0_cm2_per_Vs": 2.0}]
    d = CollisionSpec(enabled=True, model="sds", sds_species=entries).to_dict()
    assert d["sds_species"] == entries
    assert CollisionSpec.from_dict(d).sds_species == entries
