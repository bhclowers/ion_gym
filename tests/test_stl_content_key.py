"""
An STL electrode's geometry identity is its file's bytes.

The geometry key named the STL file only, so editing the mesh (moving a
rod) left a saved field matching the new geometry and load_field took it.
The key now carries a digest of each referenced STL, for decks that have
STL electrodes only.
"""
import json
import shutil
import warnings

import numpy as np
import pytest

from ion_gym.io import basis_cache, field_io
from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec

FIXTURE = repo_root() / "tests" / "fixtures" / "stl_quad"


@pytest.fixture
def deck(tmp_path):
    for p in FIXTURE.iterdir():
        shutil.copy(p, tmp_path / p.name)
    return tmp_path / "quad_stl_transport.json"


def _spec(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SimSpec.from_json(str(path))


def _key(path):
    return json.dumps(basis_cache.geometry_key_dict(_spec(path)),
                      sort_keys=True)


def test_editing_an_stl_changes_the_geometry_identity(deck):
    before = _key(deck)
    with open(deck.parent / "quad_rod_2.stl", "ab") as fh:
        fh.write(b"\n")                        # same name, other bytes
    assert _key(deck) != before


def test_a_missing_stl_is_keyed_as_missing(deck):
    (deck.parent / "quad_rod_3.stl").unlink()
    d = basis_cache.geometry_key_dict(_spec(deck))
    assert d["stl_content"]["quad_rod_3.stl"] == "missing"


def test_a_field_saved_before_the_edit_is_refused(deck, tmp_path):
    spec = _spec(deck)
    n = len(spec.geometry.electrodes)
    bases = {i + 1: np.zeros((6, 6)) for i in range(n)}
    solved = str(tmp_path / "solved")
    basis_cache.store(spec, bases, np.zeros((6, 6), np.int16), root=solved)
    p, _ = field_io.save_field(spec, directory=str(tmp_path / "fields"),
                               root=solved)
    with open(deck.parent / "quad_rod_1.stl", "ab") as fh:
        fh.write(b"\n")
    with pytest.raises(field_io.FieldMismatch, match="STL contents"):
        field_io.load_field(p, _spec(deck), root=str(tmp_path / "c"))


def test_decks_without_stl_are_keyed_as_before():
    spec = _spec(repo_root() / "examples" / "einzel_round_r-z.json")
    assert "stl_content" not in basis_cache.geometry_key_dict(spec)
