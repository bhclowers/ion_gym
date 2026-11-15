"""
set_flight_conditions(gas=...) applies every key it is given or refuses
it by name (review API-2).

It read only enabled / gas / p_torr / T_k / model and dropped every other
key silently -- including P_torr, the deck's own name for the pressure,
so a call written from a deck changed nothing and reported nothing.
"""
import pytest

from ion_gym.io.sim_spec import reflectron_vacuum_spec
from ion_gym.physics.flight_conditions import (FlightConditionError,
                                               set_flight_conditions)


def _spec():
    s = reflectron_vacuum_spec()
    s.collisions.enabled = True
    s.collisions.gas = "N2"
    return s


def test_the_deck_spelling_of_the_pressure_is_applied():
    s = _spec()
    lines = set_flight_conditions(s, gas={"P_torr": 3.5}, show=False)
    assert s.collisions.P_torr == pytest.approx(3.5)
    assert any("gas pressure" in ln for ln in lines)


def test_any_collision_field_is_applied_and_reported():
    s = _spec()
    lines = set_flight_conditions(
        s, gas={"p_torr": 2.0, "sigma_model": "fixed", "sigma_m2": 2e-18,
                "flow_mm_us": [0.0, 0.0, 0.1]}, show=False)
    c = s.collisions
    assert (c.P_torr, c.sigma_model, c.sigma_m2) == (2.0, "fixed", 2e-18)
    assert list(c.flow_mm_us) == [0.0, 0.0, 0.1]
    assert sum(ln.startswith("gas ") for ln in lines) == 4


@pytest.mark.parametrize("gas,match", [
    ({"pressure": 2.0}, r"no field\(s\) \['pressure'\]"),
    ({"P_pa": 266.0}, "P_pa is deprecated"),
    ({"p_torr": 1.0, "P_torr": 2.0}, "both p_torr and P_torr"),
], ids=["unknown", "pa", "both"])
def test_a_key_it_cannot_apply_is_refused(gas, match):
    s = _spec()
    before = s.collisions.P_torr
    with pytest.raises(FlightConditionError, match=match):
        set_flight_conditions(s, gas=gas, show=False)
    assert s.collisions.P_torr == before
