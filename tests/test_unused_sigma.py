"""
A deck shows a hard-sphere cross section only when its flight uses one.

Every save wrote collisions.sigma_m2, so 18 shipped collision blocks
showed sigma_m2 = 2.27e-18 m^2 while flying in vacuum or on the per-ion
CCS fit, where nothing reads it. A reader took that number for the cross
section in use. Saves now leave the legacy value out unless the deck
states sigma_model "fixed"; a value someone stated is always written.
The loader restores the same default, so no flight changes.
"""
import glob
import json

import pytest

from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import CollisionSpec
from ion_gym.physics.ccs import LEGACY_FIXED_SIGMA_M2


def _collision_blocks(o):
    if isinstance(o, dict):
        if "sigma_m2" in o and "gas" in o:
            yield o
        for v in o.values():
            yield from _collision_blocks(v)
    elif isinstance(o, list):
        for v in o:
            yield from _collision_blocks(v)


def test_no_shipped_deck_shows_a_cross_section_it_does_not_fly():
    unused = []
    for p in sorted(glob.glob(str(repo_root() / "examples" / "**" / "*.json"),
                              recursive=True)):
        d = json.loads(open(p, encoding="utf-8").read())
        for c in _collision_blocks(d):
            if (c.get("sigma_model") != "fixed"
                    and c["sigma_m2"] == LEGACY_FIXED_SIGMA_M2):
                unused.append(p.split("examples/")[-1])
    assert not unused, f"unused legacy sigma_m2 in {unused}"


@pytest.mark.parametrize("kw, written", [
    (dict(enabled=False), False),
    (dict(enabled=True, sigma_model="ccs_fit"), False),
    (dict(enabled=True, sigma_model="ccs_fit", gas="He"), False),
    (dict(enabled=True, sigma_model="fixed"), True),
    (dict(enabled=True, sigma_model="ccs_fit", sigma_m2=1.3e-18), True),
    (dict(enabled=True, sigma_model="fixed", sigma_m2=1.3e-18), True),
])
def test_save_writes_sigma_only_when_flown_or_stated(kw, written):
    c = CollisionSpec(P_torr=1.0, **kw)
    d = c.to_dict()
    assert ("sigma_m2" in d) is written
    back = CollisionSpec.from_dict(d)
    assert back.sigma_m2 == c.sigma_m2
    assert back.sigma_model == c.sigma_model
