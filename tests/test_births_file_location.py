"""
A relative births_file is read beside the deck, never from the working
folder (review R-07).

It was looked up in the working folder first, so a same-named file there
silently replaced the deck's ions; the saved deck records only the
relative name, so nothing later revealed it.
"""
import numpy as np
import pytest

from ion_gym.io.sim_spec import SimSpec, reflectron_vacuum_spec

HEADER = "x,y,z,vx,vy,vz\n"


def _deck(tmp_path):
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    spec = reflectron_vacuum_spec()
    spec.source.distribution = "file"
    spec.source.births_file = "births.csv"
    spec.source.n_ions = 2
    spec.source.mz_list = [100.0]
    spec.to_json(str(deck_dir / "deck.json"))
    return deck_dir


def test_the_deck_folder_wins_over_the_working_folder(tmp_path,
                                                      monkeypatch):
    from ion_gym.physics.sim_build import generate_births
    deck_dir = _deck(tmp_path)
    (deck_dir / "births.csv").write_text(HEADER + "1,0.1,0,0,0,0\n"
                                         "1,0.2,0,0,0,0\n")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "births.csv").write_text(HEADER + "5,3,0,0,0,0\n5,3,0,0,0,0\n")
    monkeypatch.chdir(other)
    b = np.asarray(generate_births(SimSpec.from_json(
        str(deck_dir / "deck.json"))))
    assert b[0, 0] == 1.0 and b[0, 1] == pytest.approx(0.1)


def test_missing_beside_the_deck_is_refused_not_found_elsewhere(
        tmp_path, monkeypatch):
    from ion_gym.physics.sim_build import generate_births
    deck_dir = _deck(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "births.csv").write_text(HEADER + "5,3,0,0,0,0\n5,3,0,0,0,0\n")
    monkeypatch.chdir(other)
    with pytest.raises(FileNotFoundError, match="read beside the deck"):
        generate_births(SimSpec.from_json(str(deck_dir / "deck.json")))
