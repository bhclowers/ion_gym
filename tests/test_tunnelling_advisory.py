"""
A step long enough to carry an ion through the thinnest wall is
announced after the build, with a step that would not tunnel.

The kernels test for metal where each step ends. The review flew 20 keV
ions of mass 1 at a 0.5 mm plate with the default 1 ns step: most landed
beyond it, and nothing said so. Now build_run raises an IonGymAdvisory
naming the wall and a step; flying at that step, the plate stops every
ion. No shipped deck is announced.
"""
import json
import re
import warnings

import numpy as np
import pytest

from ion_gym.advisories import IonGymAdvisory
from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.ion_envelope import summary_fate
from ion_gym.physics.sim_build import build_run
from ion_gym.physics.tunnelling import thinnest_wall

EXAMPLES = repo_root() / "examples"


def _plate_deck(dt_ns=None):
    """A 0.5 mm grounded plate across a planar domain; 20 keV, m = 1."""
    return SimSpec.from_dict({
        "name": "thin plate",
        "geometry": {
            "width_mm": 40.0, "height_mm": 10.0, "depth_mm": 0.0,
            "mm_per_gu": 0.1,
            "symmetry": {"coords": "xyz",
                         "planes": {"x": "none", "y": "none", "z": "none"}},
            "electrodes": [{"name": "plate", "dc": 0.0, "shapes": [
                {"type": "rect", "x_mm": 20.0, "y_mm": 0.0,
                 "width_mm": 0.5, "height_mm": 10.0}]}]},
        "source": {"n_ions": 40, "distribution": "point", "x0_mm": 5.0,
                   "y0_mm": 5.0, "z0_mm": 0.0, "direction": [1.0, 0.0, 0.0],
                   "ke_lo": 20000.0, "ke_hi": 20000.0, "mz_list": [1.0],
                   "charge": 1, "tob_span_us": 0.0, "seed": 1},
        "integration": {"dt_ns": dt_ns, "t_max_us": 1.0, "rec_every": 1,
                        "max_records": 100000}})


def _fly(spec):
    model, fly, cols, births = build_run(spec)
    return model, [summary_fate(fly(i)[1]) for i in range(len(births))]


def test_the_review_case_is_announced_with_a_safe_step():
    with pytest.warns(IonGymAdvisory, match="step THROUGH metal") as rec:
        model, fates = _fly(_plate_deck())
    (note,) = model.build_advisories
    assert "'plate'" in note
    assert any("step THROUGH metal" in str(w.message) for w in rec)
    assert fates.count(0) < len(fates)          # some ions did tunnel
    dt_ok = float(re.search(r"Use dt_ns <= ([0-9.e+-]+)", note).group(1))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        model, fates = _fly(_plate_deck(dt_ns=dt_ok))
    assert model.build_advisories == []
    assert not [w for w in seen if "THROUGH metal" in str(w.message)]
    assert fates == [0] * len(fates)            # the plate stops them all


def test_the_wall_is_measured_by_the_routes_hit_rule():
    lab = np.zeros((40, 9), np.int16)
    lab[20:25, :] = 1                           # a wall 5 nodes thick
    el = [type("E", (), {"is_grid": False})()]
    assert thinnest_wall(lab, 0.1, el, rz=False,
                         all_metal_cells=False)[0] == pytest.approx(0.5)
    lab3 = np.repeat(lab[:, :, None], 6, axis=2)
    assert thinnest_wall(lab3, 0.1, el, rz=False,
                         all_metal_cells=True)[0] == pytest.approx(0.4)
    el[0].is_grid = True                        # transparent: not a wall
    assert thinnest_wall(lab, 0.1, el, rz=False, all_metal_cells=False) \
        is None


@pytest.mark.parametrize("path", [p for p in sorted(EXAMPLES.glob("*.json"))
                                  if '"source"' in p.read_text()],
                         ids=lambda p: p.stem)
def test_no_shipped_deck_is_announced(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = SimSpec.from_json(str(path))
        spec.source.n_ions = 4
        spec.source.seed = 3
        if json.loads(path.read_text())["geometry"].get("depth_mm", 0) > 0:
            pytest.skip("3-D build; covered by the census, kept fast here")
        model = build_run(spec)[0]
    assert model.build_advisories == []
