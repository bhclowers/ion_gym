"""
A banked STL flight, or a run's provenance record, can be rebuilt from
anywhere, and checks its meshes are the ones that flew.

The bank stored spec.to_dict(), which drops where the deck was loaded
from; its relative STL folder then resolved against the current
directory, so a banked STL flight could not be rebuilt from any other
directory. The bank and the provenance record now carry the deck's
location and an STL manifest, so a rebuild finds the meshes and the STL
preflight refuses ones that changed.
"""
import shutil
import warnings

import numpy as np
import pytest

from ion_gym.io import results_bank
from ion_gym.io.paths import repo_root
from ion_gym.io.run_provenance import replay_spec, run_provenance
from ion_gym.io.sim_spec import SimSpec
from ion_gym.io.stl_resolve import require_stls

FIXTURE = repo_root() / "tests" / "fixtures" / "stl_quad"


@pytest.fixture
def deck(tmp_path):
    d = tmp_path / "deck"
    d.mkdir()
    for p in FIXTURE.iterdir():
        shutil.copy(p, d / p.name)
    return d / "quad_stl_transport.json"


def _spec(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = SimSpec.from_json(str(path))
    spec.source.seed = 5
    return spec


def _bank(spec, where):
    cols = ["t", "x", "y"]
    return results_bank.bank_flight(
        where, spec=spec, n_flown=1, cols=cols,
        hits=[{"station": "det", "x": 0.0, "y": 0.0, "z": 0.0,
               "t_us": 1.0, "mz": 100.0}],
        trajs=[np.zeros((2, len(cols)))], operating_point="t")


def test_a_banked_stl_flight_rebuilds_from_another_directory(
        deck, tmp_path, monkeypatch):
    p = _bank(_spec(deck), tmp_path / "f.npz")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    spec = results_bank.load_flight_bank(p)[0]
    assert require_stls(spec) == deck.parent        # the meshes are found


def test_a_rebuild_refuses_meshes_changed_since_the_flight(deck, tmp_path):
    p = _bank(_spec(deck), tmp_path / "f.npz")
    with open(deck.parent / "quad_rod_2.stl", "ab") as fh:
        fh.write(b"\n")
    spec = results_bank.load_flight_bank(p)[0]
    # CHANGED (found, but not the bytes that flew) -- not merely missing
    with pytest.raises(FileNotFoundError, match=r"CHANGED[^\n]*quad_rod_2"):
        require_stls(spec)


def test_a_provenance_replay_finds_its_meshes(deck, tmp_path, monkeypatch):
    prov = run_provenance(_spec(deck))
    monkeypatch.chdir(tmp_path)
    assert require_stls(replay_spec(prov)) == deck.parent
