"""
A saved run carries what it needs to be flown again: the deck as flown,
the seed it used and the code versions.

A random-mode deck (source.seed null) draws its seed at build time and
kept it only on the live spec, so every saved run of such a deck stored
seed null and could not be repeated; no saved run recorded the code that
flew it beyond a release number. io.run_provenance captures the three
after the build, and replay_spec pins the seed so the same births are
drawn.
"""
import json
import re

import numpy as np
import pytest

from ion_gym.io import field_io, results_bank
from ion_gym.io.paths import repo_root
from ion_gym.io.run_provenance import (replay_spec, run_provenance,
                                       unrecorded)
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.sim_build import generate_births

DECK = repo_root() / "examples" / "ion_funnel_rz.json"


def _random_mode():
    spec = SimSpec.from_json(str(DECK))
    spec.source.seed = None
    return spec


def test_a_random_run_replays_to_the_same_births():
    spec = _random_mode()
    births = generate_births(spec)                  # the build's draw
    prov = run_provenance(spec)
    assert prov["seed_mode"] == "drawn"
    assert prov["spec"]["source"]["seed"] is None   # why the seed is kept
    again = replay_spec(json.loads(json.dumps(prov)))
    assert again.source.seed == prov["seed"]
    assert np.array_equal(generate_births(again), births)


def test_a_stated_seed_is_recorded_as_stated():
    spec = _random_mode()
    spec.source.seed = 1234
    prov = run_provenance(spec)
    assert (prov["seed"], prov["seed_mode"]) == (1234, "stated")


def test_an_undrawn_seed_is_recorded_and_refuses_replay():
    prov = run_provenance(_random_mode())
    assert (prov["seed"], prov["seed_mode"]) == (None, "not drawn")
    with pytest.raises(ValueError, match="not recorded"):
        replay_spec(prov)
    with pytest.raises(ValueError, match="not recorded"):
        replay_spec(unrecorded("loaded from an old file"))


def test_the_versions_name_the_code():
    v = run_provenance(_random_mode())["versions"]
    import ion_gym
    import numba
    assert v["ion_gym"] == ion_gym.__version__
    assert v["numba"] == numba.__version__ and v["numpy"] and v["python"]
    assert re.fullmatch(r"[0-9a-f]{40}", v["git"]["commit"])   # a checkout
    assert isinstance(v["git"]["dirty"], bool)


def test_saved_trajectories_carry_the_runs_provenance(tmp_path):
    spec = _random_mode()
    births = generate_births(spec)
    prov = run_provenance(spec)
    p = field_io.save_trajectories([(0, np.zeros((2, 3)), {"kind": 0})],
                                   spec, path=str(tmp_path / "t.traj.npz"),
                                   provenance=prov,
                                   columns=["t", "x", "y"])
    _, meta, _ = field_io.load_trajectories(p)
    assert meta["provenance"]["seed"] == prov["seed"]
    assert np.array_equal(generate_births(replay_spec(meta["provenance"])),
                          births)


def test_banks_record_versions_and_a_flight_its_seed(tmp_path):
    spec = _random_mode()
    generate_births(spec)
    seed = spec.source._drawn_seed
    cols = ["t", "x", "y"]
    p = results_bank.bank_flight(
        tmp_path / "f.npz", spec=spec, n_flown=1, cols=cols,
        hits=[{"station": "det", "x": 0.0, "y": 0.0, "z": 0.0,
               "t_us": 1.0, "mz": 100.0}],
        trajs=[np.zeros((2, len(cols)))], operating_point="test")
    meta, _ = results_bank.load_bank(p)
    assert (meta["seed"], meta["seed_mode"]) == (seed, "drawn")
    assert meta["versions"]["git"]["commit"]


def test_a_bank_given_the_spec_records_its_seed(tmp_path, capsys):
    from ion_gym.physics.ensemble_driver import IonResult
    spec = _random_mode()
    generate_births(spec)
    seed = spec.source._drawn_seed
    p = results_bank.bank_run(tmp_path / "a.npz", meta={
        "operating_point": "test"}, arrays={"tof": np.ones(3)}, spec=spec)
    meta, _ = results_bank.load_bank(p)
    assert (meta["seed"], meta["seed_mode"]) == (seed, "drawn")
    p = results_bank.bank_trajectories(
        tmp_path / "t.npz",
        results=[IonResult(index=0, traj=np.zeros((3, 2)),
                           summary={"tof": 1.0})],
        cols=["t", "x"], channels=("t", "x"),
        meta={"operating_point": "test"}, spec=spec)
    meta, _ = results_bank.load_bank(p)
    assert meta["seed"] == seed
    assert "records no seed" not in capsys.readouterr().out


def test_a_bank_without_the_spec_says_it_has_no_seed(tmp_path, capsys):
    p = results_bank.bank_run(tmp_path / "b.npz", meta={
        "operating_point": "test"}, arrays={"tof": np.ones(3)})
    meta, _ = results_bank.load_bank(p)
    assert (meta["seed"], meta["seed_mode"]) == (None, "not recorded")
    assert "b.npz records no seed" in capsys.readouterr().out
    # a seed the caller put in meta is kept, without the warning
    p = results_bank.bank_run(tmp_path / "c.npz", meta={
        "operating_point": "test", "seed": 5}, arrays={"tof": np.ones(3)})
    assert results_bank.load_bank(p)[0]["seed"] == 5
    assert "records no seed" not in capsys.readouterr().out


def test_the_downloaded_npz_carries_the_provenance():
    """Download NPZ is the same file Save trajectories writes
    (field_io.trajectories_npz), read back by the same reader."""
    from ion_gym.physics.ensemble_driver import IonResult
    spec = _random_mode()
    generate_births(spec)
    prov = run_provenance(spec)
    import io
    raw = field_io.trajectories_npz(
        [IonResult(index=0, traj=np.zeros((2, 3)), summary={"kind": 0})],
        spec, provenance=prov, columns=["t", "x", "y"])
    _, meta, _ = field_io.load_trajectories(io.BytesIO(raw))
    got = meta["provenance"]
    assert got["seed"] == prov["seed"] and got["seed_mode"] == "drawn"
