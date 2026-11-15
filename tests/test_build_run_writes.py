"""
build_run writes derived values on the deck it is given, and only those.

Its contract (docstring, docs/PUBLIC_API.md): DC-ladder members' dc, a
derived step when dt_ns is null, e_x/e_y/e_z when a non-r-z deck records
e_axial/e_radial, and the drawn seed in random mode. Of these only the
ladder voltages reach a save. The added field channels used to: a save
after a build wrote e_x/e_y/e_z into the deck.
"""
import copy
import json
import warnings

import pytest

from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.sim_build import build_run

EXAMPLES = repo_root() / "examples"
# the shipped decks that build in under a second (r-z and planar)
FAST_DECKS = ["drift_tube_mason_schamp", "rf_drift_cell_allen_bush",
              "einzel_round_r-z", "kingdon_trap_rz",
              "quadrupole_analytic_rods_2d",
              "slim_tetramer_confinement_2-d_acrossxgap"]
_MISSING = object()


def _load(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = SimSpec.from_json(str(EXAMPLES / f"{name}.json"))
    spec.source.n_ions = 2
    return spec


def _build(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_run(spec)


def _diffs(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            yield from _diffs(a.get(k, _MISSING), b.get(k, _MISSING),
                              f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _diffs(x, y, f"{path}[{i}]")
    elif a != b:
        yield path, a, b


@pytest.mark.parametrize("name", FAST_DECKS)
@pytest.mark.parametrize("seed", [1, None])
def test_a_build_changes_no_saved_value_but_ladder_voltages(name, seed):
    spec = _load(name)
    spec.source.seed = seed
    before = copy.deepcopy(spec.to_dict())
    _build(spec)
    members = {i for i, e in enumerate(spec.geometry.electrodes)
               if e.dc_group}
    for path, a, b in _diffs(before, spec.to_dict()):
        i = int(path.split("[")[1].split("]")[0]) if path.startswith(
            ".geometry.electrodes[") else None
        assert path.endswith(".dc") and i in members, (path, a, b)


def _radial_planar():
    spec = _load("quadrupole_analytic_rods_2d")
    g = spec.geometry
    spec.optic_axis = {"axis": "z",
                       "through_mm": [g.width_mm / 2, g.height_mm / 2]}
    spec.integration.record_channels = ["speed", "e_axial"]
    return spec


def test_added_field_channels_fly_but_are_not_saved():
    spec = _radial_planar()
    _, _, cols, _ = _build(spec)
    assert {"e_x", "e_y", "e_z"} <= set(spec.integration.record_channels)
    assert "e_x" in cols
    saved = spec.to_dict()["integration"]["record_channels"]
    assert saved == ["speed", "e_axial"]
    # and a reload of that save builds the same columns
    again = SimSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    again.source.n_ions = 2
    assert _build(again)[2] == cols


def test_writing_the_same_channels_back_keeps_them_derived():
    spec = _radial_planar()
    _build(spec)
    spec.integration.record_channels = list(spec.integration.record_channels)
    assert spec.to_dict()["integration"]["record_channels"] == [
        "speed", "e_axial"]
    spec.integration.record_channels = ["speed", "e_axial", "e_x"]
    assert spec.to_dict()["integration"]["record_channels"] == [
        "speed", "e_axial", "e_x"]
