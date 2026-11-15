"""
An ion flight leaves numba's random state as it found it.

The gas kernels seed numba's random generator per ion (np.random.seed in
@njit code). That generator is shared, per thread, with any @njit code a
user runs, so after one flight a user's seeded @njit sequence changed.
Every kernel call now goes through numba_rng.call_rng_preserved: the
user's sequence is unchanged, and a flight's own result does not depend
on what was drawn before it.
"""
import json
import warnings

import numpy as np
import pytest
from numba import njit

from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.sim_build import build_run

EXAMPLES = repo_root() / "examples"


@njit
def _seed(s):
    np.random.seed(s)


@njit
def _draws(n):
    out = np.empty(n)
    for i in range(n):
        out[i] = np.random.random()
    return out


def _gas_fly(deck, model=None):
    d = json.loads((EXAMPLES / deck).read_text())
    d["source"].update(n_ions=1, seed=3)
    if model is not None:
        d.setdefault("collisions", {}).update(
            enabled=True, gas="N2", P_torr=1.0, T_k=300.0, model=model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = SimSpec.from_dict(d)
        fly = build_run(spec)[1]
    return fly


CASES = [("slim_tetramer_confinement_2-d_acrossxgap.json", None),  # planar
         ("drift_tube_mason_schamp.json", None),                    # r-z
         ("hexapole_native_3d.json", "hs"),                         # 3-D HS
         ("hexapole_native_3d.json", "sds")]                        # 3-D SDS


@pytest.mark.parametrize("deck,model", CASES,
                         ids=["planar", "rz", "3d-hs", "3d-sds"])
def test_a_flight_leaves_the_users_numba_sequence_alone(deck, model):
    fly = _gas_fly(deck, model)
    _seed(7)
    want = _draws(5)
    _seed(7)
    _draws(2)
    fly(0)                                    # a gas flight in between
    assert np.array_equal(_draws(3), want[2:])


@pytest.mark.parametrize("deck,model", CASES[:2], ids=["planar", "rz"])
def test_a_flight_does_not_depend_on_earlier_draws(deck, model):
    fly = _gas_fly(deck, model)
    _seed(1)
    a = fly(0)
    _seed(99)
    _draws(17)
    b = fly(0)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
