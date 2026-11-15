"""
Pressure keeps one meaning however it is changed, and a legacy dt_us deck
is told how to convert.

Three edge cases:
  * c.P_torr = None left the old Pa in the kernel while a save wrote null,
    which reloads as 1 Torr: a silent 5 -> 1 Torr change. Now clearing it
    means what None means at construction: the default, marked defaulted
    (the advisory names it), and the kernel's Pa follows.
  * dataclasses.replace(c, P_pa=266) was overruled by the old P_torr with
    only a warning, and replace(c, P_torr=10) carried the stale Pa. P_pa
    is a field the constructor does not take, so replace re-derives it,
    and replace(c, P_pa=...) is refused.
  * a deck stating dt_us (the old microsecond key) was refused naming only
    dt_ns; the refusal now gives the conversion.
"""
import dataclasses
import json
import warnings

import pytest

from ion_gym.io.sim_spec import (CollisionSpec, IntegrationSpec,
                                 MissingPhysicsKey, SimSpec)
from ion_gym.physics.constants import TORR_PA


@pytest.fixture(autouse=True)
def _quiet_deprecations():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yield


def test_clearing_the_pressure_means_the_default_everywhere():
    c = CollisionSpec(P_torr=5.0)
    c.P_torr = None
    assert c.P_torr == CollisionSpec.P_TORR_DEFAULT
    assert c.P_pa == pytest.approx(c.P_torr * TORR_PA)      # the kernel's
    assert "P_torr" in c._defaulted                         # said so
    again = CollisionSpec.from_dict(json.loads(json.dumps(c.to_dict())))
    assert again.P_pa == pytest.approx(c.P_pa)              # save == flight


def test_replace_rederives_the_pa_and_refuses_setting_it():
    c = CollisionSpec(P_torr=5.0)
    assert dataclasses.replace(c, P_torr=10.0).P_pa == pytest.approx(
        10.0 * TORR_PA)
    assert dataclasses.replace(c, T_k=300.0).P_pa == pytest.approx(c.P_pa)
    with pytest.raises(ValueError, match="P_pa"):
        dataclasses.replace(c, P_pa=266.0)


def test_the_deprecated_keyword_still_converts():
    k = CollisionSpec(enabled=True, gas="N2", P_pa=50.0)
    assert k.P_pa == 50.0 and k.P_torr == pytest.approx(50.0 / TORR_PA)


def test_a_contradictory_constructor_call_is_refused():
    with pytest.raises(ValueError, match="disagree"):
        CollisionSpec(P_torr=5.0, P_pa=266.0)


def test_decks_keep_their_rules():
    # a legacy Pa-only deck keeps its exact Pa (bit-identity with old runs)
    legacy = CollisionSpec.from_dict({"enabled": True, "P_pa": 133.28})
    assert legacy.P_pa == 133.28
    assert legacy.P_torr == pytest.approx(133.28 / TORR_PA)
    assert "P_torr" not in legacy._defaulted
    # a deck stating both: Torr wins, with a warning naming the Pa
    with pytest.warns(UserWarning, match="disagree"):
        both = CollisionSpec.from_dict({"enabled": True, "P_torr": 5.0,
                                        "P_pa": 266.0})
    assert both.P_pa == pytest.approx(5.0 * TORR_PA)
    assert "P_pa" not in both.to_dict()


def test_a_dt_us_deck_is_told_the_conversion():
    with pytest.raises(MissingPhysicsKey,
                       match=r"dt_us = 0.002 -> dt_ns = 2"):
        IntegrationSpec.from_dict({"dt_us": 0.002, "t_max_us": 10.0})


def test_a_shipped_gas_deck_round_trips_its_pressure():
    from ion_gym.io.paths import repo_root
    spec = SimSpec.from_json(str(repo_root() / "examples"
                                 / "ion_funnel_rz.json"))
    again = SimSpec.from_dict(json.loads(spec.to_json()))
    assert again.collisions.P_pa == spec.collisions.P_pa
