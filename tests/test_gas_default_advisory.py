"""
A gas deck that leaves out the pressure, temperature or gas flies the
default -- and now says so (review API-6).

The default itself is the R3 ruling (1 Torr, 273 K, N2); before, a deck
with collisions on and no pressure flew 1 Torr without a word.
"""
import json

from ion_gym.io.sim_spec import (CollisionSpec, SimSpec,
                                 reflectron_vacuum_spec)


def _deck(collisions):
    d = json.loads(reflectron_vacuum_spec().to_json())
    d["collisions"] = collisions
    return SimSpec.from_dict(d)


def _note(spec):
    return [a for a in spec.advisories()
            if a.startswith("collisions: the deck does not state")]


def test_the_defaulted_keys_are_named_with_their_values():
    note = _note(_deck({"enabled": True, "model": "hs",
                        "sigma_model": "fixed"}))
    assert len(note) == 1
    assert "P_torr (flying the default 1.0 Torr)" in note[0]
    assert "T_k (flying the default 273.0 K)" in note[0]
    assert "gas (flying the default 'N2')" in note[0]


def test_a_deck_that_states_them_gets_no_note():
    assert _note(_deck({"enabled": True, "gas": "N2", "T_k": 300.0,
                        "P_torr": 2.0, "model": "hs",
                        "sigma_model": "fixed"})) == []


def test_a_pressure_set_after_construction_is_not_a_default():
    s = reflectron_vacuum_spec()
    s.collisions = CollisionSpec(enabled=True, gas="N2", T_k=300.0,
                                 model="hs", sigma_model="fixed")
    s.collisions.set_pressure_torr(2.0)
    assert _note(s) == []


def test_vacuum_says_nothing():
    assert _note(reflectron_vacuum_spec()) == []
