"""
validate() refuses source values that cannot mean anything, by deck key
(review F6), and a staged assembly refuses a fractional beam charge.

Before, each was accepted and then truncated, padded or ignored where it
was used: charge 2.5 flew as charge 2 (int() everywhere), a two-entry
box_mm was padded with 0, temperature_k -300 gave the births of T = 0,
seed 2.5 and n_ions 2.5 reached the generators, negative energies and
spreads flew. Legitimate values stay accepted: a JSON 2.0 charge, a
neutral (0), a negative charge, a signed line length.
"""
import pytest

from ion_gym.io.sim_spec import reflectron_vacuum_spec

REFUSED = [("charge", 2.5), ("charge", True), ("charge", "2"),
           ("temperature_k", -300.0), ("seed", 2.5), ("seed", -1),
           ("n_ions", 0), ("n_ions", 2.5), ("ke_lo", -5.0),
           ("tob_span_us", -1.0), ("r_mm", -1.0)]
ACCEPTED = [("charge", 2.0), ("charge", 0), ("charge", -3),
            ("len_mm", -2.0), ("seed", None), ("temperature_k", 0.0)]


@pytest.mark.parametrize("key,value", REFUSED,
                         ids=[f"{k}={v!r}" for k, v in REFUSED])
def test_refused_by_key(key, value):
    s = reflectron_vacuum_spec()
    setattr(s.source, key, value)
    errs = s.validate()
    assert any(e.startswith(f"source.{key}") for e in errs), errs


@pytest.mark.parametrize("key,value", ACCEPTED,
                         ids=[f"{k}={v!r}" for k, v in ACCEPTED])
def test_still_accepted(key, value):
    s = reflectron_vacuum_spec()
    setattr(s.source, key, value)
    assert not [e for e in s.validate() if f"source.{key}" in e]


@pytest.mark.parametrize("box", [[1.0, 1.0], [-1.0, 1.0, 1.0],
                                 [1.0, True, 1.0]])
def test_a_box_needs_three_non_negative_extents(box):
    s = reflectron_vacuum_spec()
    s.source.distribution = "box"
    s.source.box_mm = box
    assert any(e.startswith("source.box_mm") for e in s.validate())


def test_ke_hi_below_ke_lo_is_refused():
    s = reflectron_vacuum_spec()
    s.source.ke_lo, s.source.ke_hi = 5.0, 1.0
    assert any(e.startswith("source.ke_lo..ke_hi") for e in s.validate())


@pytest.mark.parametrize("doc", [
    {"beam": {"charge": 2.5}, "stages": []},
    {"stages": [{"name": "a", "spec": {"source": {"charge": 1.5}}}]},
], ids=["beam", "stage"])
def test_an_assembly_refuses_a_fractional_charge(doc):
    from ion_gym.physics.staged_flight import _beam_charge
    with pytest.raises(ValueError, match="whole number"):
        _beam_charge(doc)


def test_an_assembly_takes_a_whole_charge():
    from ion_gym.physics.staged_flight import _beam_charge
    assert _beam_charge({"beam": {"charge": -2.0}}) == (-2, "beam.charge")
