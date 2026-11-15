"""
A flight with an enormous number of integrator steps is announced, with
an estimate of how long it will take.

dt_ns 1e-6 over 400 us is 4e11 steps for ONE ion; with rec_every 1e9 the
recording fits, so every guard passed and the run simply took the better
part of a day without a word. The advisory names the step count and a
wall-time range; it warns rather than refuses (long trapping runs are
legitimate).
"""
import json
import warnings

import pytest

from ion_gym import cli
from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import (LONG_FLIGHT_ION_STEPS, SimSpec,
                                 long_flight_note)

EXAMPLES = repo_root() / "examples"
DECK = EXAMPLES / "kingdon_trap_rz.json"


def _long_deck(tmp_path):
    d = json.loads(DECK.read_text())
    d["integration"].update(dt_ns=1e-6, t_max_us=400.0, rec_every=10**9)
    d["source"]["n_ions"] = 1
    p = tmp_path / "long.json"
    p.write_text(json.dumps(d))
    return p


def _long_notes(spec):
    return [a for a in spec.advisories() if "a long flight" in a]


def test_the_review_case_is_announced_with_a_time(tmp_path):
    spec = SimSpec.from_json(str(_long_deck(tmp_path)))
    (note,) = _long_notes(spec)
    assert "4e+11 integrator steps" in note
    assert note.count(" h") == 2                   # hours, both ends


def test_validate_prints_it_and_still_passes_the_deck(tmp_path, capsys):
    assert cli.main(["validate", str(_long_deck(tmp_path))]) == 0
    assert "advisory: integration: a long flight" in capsys.readouterr().out


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.json")),
                         ids=lambda p: p.stem)
def test_no_shipped_deck_is_announced(path):
    raw = json.loads(path.read_text())
    if not (isinstance(raw, dict) and "source" in raw):
        pytest.skip("not a single-deck spec")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = SimSpec.from_json(str(path))
    assert not _long_notes(spec)


def test_the_threshold_is_the_ensemble_step_count():
    per_ion = LONG_FLIGHT_ION_STEPS / 10            # 10 ions = at threshold
    t_max_us = per_ion * 1.0 / 1e3                  # at dt_ns = 1
    assert long_flight_note(1.0, t_max_us, 10) is None
    assert long_flight_note(1.0, t_max_us, 11) is not None


@pytest.mark.parametrize("dt, t_max", [(0.0, 400.0), (-1.0, 400.0),
                                       (1.0, 0.0), (float("nan"), 400.0),
                                       (None, 400.0)])
def test_values_the_validator_refuses_are_left_to_it(dt, t_max):
    assert long_flight_note(dt, t_max, 1) is None
