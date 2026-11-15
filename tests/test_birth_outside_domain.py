"""
An ion born outside the solved domain is reported, on every route.

It has no field there and the kernel ends it at its first step as having
left the domain -- recorded as fate 1 with no word of why (an x0 of 10 m
on a 20 mm deck looked like an ordinary loss). Now each route's birth
check raises BirthOutsideDomain, which the per-ion handler records as
fate 1 with born_outside_domain and a warning, while the other ions fly.
"""
import json
import warnings

import numpy as np
import pytest

from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.ion_envelope import summary_fate
from ion_gym.physics.birth_domain import (BirthOutsideDomain,
                                          refuse_birth_outside)
from ion_gym.physics.sim_build import build_run

EXAMPLES = repo_root() / "examples"


def _fly_one(deck, shift):
    d = json.loads((EXAMPLES / deck).read_text())
    d["source"].update(n_ions=1, seed=1, distribution="point")
    d["source"].update(shift)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = SimSpec.from_dict(d)
        model, fly, cols, births = build_run(spec)
    return fly(0)[1]


@pytest.mark.parametrize("deck,shift", [
    ("einzel_round_r-z.json", {"x0_mm": 1.0e4}),                    # r-z
    ("quadrupole_analytic_rods_2d.json", {"y0_mm": -500.0}),        # planar
    ("hexapole_native_3d.json", {"x0_mm": 1.0e4}),                  # 3-D
])
def test_a_birth_outside_the_domain_is_reported(deck, shift, capsys):
    s = _fly_one(deck, shift)
    assert summary_fate(s) == 1
    assert s.get("born_outside_domain") is True
    assert "outside the solved domain" in s["note"]
    assert "outside the solved domain" in capsys.readouterr().out


def test_the_edge_node_is_inside():
    refuse_birth_outside((11, 5), (10.0, 0.0), ("x", "y"), 0.1, "", "p")
    with pytest.raises(BirthOutsideDomain, match="y is at node -0.500"):
        refuse_birth_outside((11, 5), (3.0, -0.5), ("x", "y"), 0.1, "", "p")


def test_an_inside_birth_is_untouched():
    s = _fly_one("einzel_round_r-z.json", {})
    assert "born_outside_domain" not in s
    assert np.isfinite(s["tof"])
