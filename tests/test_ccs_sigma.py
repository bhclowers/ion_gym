"""Per-ion hard-sphere cross sections (physics.ccs) and the hard-sphere
step advisory (collision3d.hs_step_note). Requested by Brian 2026-10-17.

Fast: no field solve and no flight. Checks read the shipped coefficient
table and a SimSpec loaded from the shipped funnel deck (geometry only;
nothing is solved), so the file needs no fixture of its own.
"""
import copy
import math
from pathlib import Path

import pytest

from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import CollisionSpec, SimSpec
from ion_gym.physics import ccs
from ion_gym.physics.collision3d import (HS_DT_PER_COLLISION_TIME,
                                         gas_kernel_scalars, hs_step_note)


# A minimal gas-filled spec: N2, singly charged, three masses in range.
def _gas_spec(sigma_model="ccs_fit", gas="N2", charge=1, mz_list=None,
              P_torr=2.5, dt_ns=2.0):
    deck = Path(repo_root()) / "examples" / "ion_funnel_rz.json"
    spec = SimSpec.from_json(str(deck))
    spec.collisions = CollisionSpec(enabled=True, gas=gas, T_k=273.0,
                                    P_torr=P_torr, model="hs",
                                    sigma_model=sigma_model)
    spec.source.charge = charge
    spec.source.mz_list = list(mz_list or [118.0, 322.0, 622.0])
    spec.integration.dt_ns = dt_ns
    return spec


def test_fixed_model_is_the_default_and_returns_sigma_m2():
    col = CollisionSpec(enabled=True, sigma_m2=3.0e-18)
    assert col.sigma_model == "fixed"
    for mz in (50.0, 500.0, 5000.0):
        assert ccs.ion_sigma_m2(col, mz, 1) == 3.0e-18


def test_vacuum_returns_zero_like_the_old_scalars():
    assert ccs.ion_sigma_m2(None, 322.0, 1) == 0.0
    col = CollisionSpec(enabled=False, sigma_model="ccs_fit")
    assert ccs.ion_sigma_m2(col, 322.0, 1) == 0.0


def test_ccs_fit_matches_the_table_power_law():
    row = ccs.load_fit_table()["by_label"]["all"]
    col = CollisionSpec(enabled=True, gas="N2", sigma_model="ccs_fit")
    for mz in (118.0, 322.0, 622.0):
        want = row["A_A2"] * mz ** row["B"] * 1.0e-20
        assert ccs.ion_sigma_m2(col, mz, 1) == pytest.approx(want, rel=1e-12)
    # cross section grows with m/z (positive exponent)
    assert (ccs.ion_sigma_m2(col, 622.0, 1)
            > ccs.ion_sigma_m2(col, 118.0, 1))


def test_negative_ions_use_the_fit_too():
    col = CollisionSpec(enabled=True, gas="N2", sigma_model="ccs_fit")
    assert ccs.ion_sigma_m2(col, 322.0, -1) == ccs.ion_sigma_m2(col, 322.0, 1)


def test_fit_class_selects_its_own_row():
    col = CollisionSpec(enabled=True, gas="N2", sigma_model="ccs_fit",
                        ccs_fit_class="Lipids and lipid-like molecules")
    row = ccs.load_fit_table()["by_label"][col.ccs_fit_class]
    want = row["A_A2"] * 700.0 ** row["B"] * 1.0e-20
    assert ccs.ion_sigma_m2(col, 700.0, 1) == pytest.approx(want, rel=1e-12)


def test_extrapolation_warns_and_returns_a_value(capsys):
    ccs._EXTRAPOLATION_WARNED.clear()
    col = CollisionSpec(enabled=True, gas="N2", sigma_model="ccs_fit")
    sigma = ccs.ion_sigma_m2(col, 5000.0, 1)
    out = capsys.readouterr().out
    assert sigma > 0.0
    assert "[CCS] WARNING" in out and "EXTRAPOLATED" in out
    # the identical warning is printed once per m/z per session
    ccs.ion_sigma_m2(col, 5000.0, 1)
    assert "[CCS] WARNING" not in capsys.readouterr().out


def test_in_range_mz_prints_no_warning(capsys):
    ccs._EXTRAPOLATION_WARNED.clear()
    col = CollisionSpec(enabled=True, gas="N2", sigma_model="ccs_fit")
    ccs.ion_sigma_m2(col, 322.0, 1)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("gas,charge", [("He", 1), ("N2", 2), ("N2", -3)])
def test_fit_outside_its_data_is_refused(gas, charge):
    spec = _gas_spec(gas=gas, charge=charge)
    errs = spec.validate()
    assert any("ccs_fit" in e for e in errs), errs
    col = spec.collisions
    with pytest.raises(ccs.CCSModelRefused):
        ccs.ion_sigma_m2(col, 322.0, charge)


def test_unknown_model_and_class_are_refused_by_name():
    spec = _gas_spec(sigma_model="bogus")
    assert any("bogus" in e for e in spec.validate())
    spec = _gas_spec()
    spec.collisions.ccs_fit_class = "no such class"
    assert any("no such class" in e for e in spec.validate())


def test_shared_scalars_no_longer_carry_a_cross_section():
    g = gas_kernel_scalars(CollisionSpec(enabled=True))
    assert "sigma_m2" not in g


def test_save_omits_the_new_fields_at_their_defaults():
    d = CollisionSpec(enabled=True).to_dict()
    assert "sigma_model" not in d and "ccs_fit_class" not in d
    d = CollisionSpec(enabled=True, sigma_model="ccs_fit").to_dict()
    assert d["sigma_model"] == "ccs_fit"
    back = CollisionSpec.from_dict(d)
    assert back.sigma_model == "ccs_fit" and back.ccs_fit_class == "all"


def test_advisory_lists_the_flown_cross_sections():
    spec = _gas_spec(mz_list=[322.0, 5000.0])
    notes = [a for a in spec.advisories()
             if a.startswith("collisions: hard-sphere cross sections")]
    assert len(notes) == 1
    assert "m/z 322" in notes[0] and "EXTRAPOLATED" in notes[0]


def test_step_advisory_warns_on_a_coarse_step_and_never_raises():
    spec = _gas_spec(dt_ns=2.0)
    note = hs_step_note(spec, 2.0)
    assert note is not None and note.startswith("WARNING")
    # the requested step satisfies the rule for the worst ion at rest
    col = spec.collisions
    n_gas = col.P_pa / (1.380649e-23 * col.T_k)
    m_gas = 28.0134 * 1.66053906660e-27
    c_bar = math.sqrt(8 * 1.380649e-23 * col.T_k / (math.pi * m_gas))
    worst = max(n_gas * ccs.ion_sigma_m2(col, m, 1) * c_bar
                for m in spec.source.mz_list)
    dt_ok_ns = 1e9 / worst / HS_DT_PER_COLLISION_TIME
    assert hs_step_note(spec, 0.999 * dt_ok_ns) is None


def test_step_advisory_is_silent_for_sds_and_vacuum():
    spec = _gas_spec()
    spec.collisions.model = "sds"
    assert hs_step_note(spec, 2.0) is None
    spec = _gas_spec()
    spec.collisions.enabled = False
    assert hs_step_note(spec, 2.0) is None


def test_fixed_model_step_advisory_uses_the_fixed_sigma():
    spec = _gas_spec(sigma_model="fixed")
    a = hs_step_note(spec, 2.0)
    spec2 = copy.deepcopy(spec)
    spec2.collisions.sigma_m2 = spec.collisions.sigma_m2 / 100.0
    assert a is not None and hs_step_note(spec2, 2.0) is None


def test_mason_schamp_mobility_scalings_and_value():
    col = CollisionSpec(enabled=True, gas="N2", T_k=273.0, P_torr=2.5,
                        sigma_model="ccs_fit")
    k_322 = ccs.low_field_mobility_m2Vs(col, 322.0, 1)
    # hand evaluation of (3/16)(q/N) sqrt(2 pi / (mu k T)) / Omega
    kb = 1.380649e-23
    amu = 1.66053906660e-27
    n_gas = col.P_pa / (kb * col.T_k)
    mu = 322.0 * 28.0134 / (322.0 + 28.0134) * amu
    omega = ccs.ion_sigma_m2(col, 322.0, 1)
    want = (3 / 16) * (1.602176634e-19 / n_gas) * math.sqrt(
        2 * math.pi / (mu * kb * col.T_k)) / omega
    assert k_322 == pytest.approx(want, rel=1e-6)
    # mobility scales as 1/P at fixed T
    col4 = CollisionSpec(enabled=True, gas="N2", T_k=273.0, P_torr=10.0,
                         sigma_model="ccs_fit")
    assert ccs.low_field_mobility_m2Vs(col4, 322.0, 1) == pytest.approx(
        k_322 / 4.0, rel=1e-9)
    # heavier, larger ions are less mobile
    assert ccs.low_field_mobility_m2Vs(col, 3290.0, 1) < k_322
    with pytest.raises(ccs.CCSModelRefused):
        ccs.low_field_mobility_m2Vs(None, 322.0, 1)
