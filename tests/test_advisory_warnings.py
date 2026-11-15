"""
Physics advisories are Python warnings (IonGymAdvisory), so the standard
controls reach them.

They were print()s deduplicated through a private module set: a second
identical build said nothing, and catch_warnings, -W error and pytest's
capture could not see, collect or escalate them.
"""
import warnings

import pytest

from ion_gym.advisories import IonGymAdvisory
from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.sim_build import build_run

DECK = repo_root() / "examples" / "ion_funnel_rz.json"


def _spec():
    spec = SimSpec.from_json(str(DECK))
    spec.source.n_ions = 2
    spec.source.seed = 1
    assert spec.advisories(), "the test deck must carry advisories"
    return spec


def _advisories(seen):
    return [str(w.message) for w in seen
            if issubclass(w.category, IonGymAdvisory)
            and str(w.message).startswith("[ADVISORY]")]


def test_every_build_raises_them_when_asked(capsys):
    n = len(_spec().advisories())
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        for _ in range(3):
            build_run(_spec())
    assert len(_advisories(seen)) == 3 * n
    assert "[ADVISORY]" not in capsys.readouterr().out     # not printed


def test_the_default_filter_states_a_sweep_once():
    n = len(_spec().advisories())
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("default")
        for _ in range(3):
            build_run(_spec())
    assert len(_advisories(seen)) == n


def test_they_can_be_made_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", IonGymAdvisory)
        with pytest.raises(IonGymAdvisory):
            build_run(_spec())
