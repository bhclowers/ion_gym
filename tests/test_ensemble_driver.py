"""
The ensemble driver's contract, independent of any kernel:

* fly_all returns every ion, in index order, with the same results at
  any worker count;
* an ion that raises stops the run and the error names that ion;
* a stop request ends the run early, cleanly, with what was flown.

These pin the driver while its two `while True` loops (the worker's
sentinel loop and the collector loop) are written as bounded for-loops;
they pass on the code before and after that change.
"""
import time

import numpy as np
import pytest

from ion_gym.physics.ensemble_driver import StopToken, fly_all, run


def _fly(i):
    rng = np.random.default_rng(i)
    time.sleep(0.001 * (i % 3))           # finish out of order
    traj = rng.random((5, 2))
    return traj, {"kind": int(i % 3), "x_end": float(traj[-1, 0])}


@pytest.mark.parametrize("n_workers", [1, 4])
def test_every_ion_in_order_at_any_worker_count(n_workers):
    ref = [_fly(i) for i in range(37)]
    got = fly_all(37, _fly, n_workers=n_workers, check_every=5)
    assert len(got) == 37
    for (t_ref, s_ref), (t, s) in zip(ref, got):
        assert np.array_equal(t, t_ref) and s == s_ref


@pytest.mark.parametrize("n_workers", [1, 4])
def test_a_failing_ion_is_named(n_workers):
    def fly(i):
        if i == 7:
            raise ValueError("boom on seven")
        return _fly(i)
    with pytest.raises((RuntimeError, ValueError), match="boom on seven"):
        fly_all(20, fly, n_workers=n_workers)
    if n_workers > 1:
        with pytest.raises(RuntimeError, match="failed on ion 7"):
            fly_all(20, fly, n_workers=n_workers)


@pytest.mark.parametrize("n_workers", [1, 4])
def test_a_stop_ends_the_run_cleanly(n_workers):
    token = StopToken()

    def fly(i):
        if i == 10:
            token.stop("enough")
        return _fly(i)
    prog = run(40, fly, n_workers=n_workers, stop_token=token,
               check_every=5)
    assert prog.stopped_early and prog.stop_reason == "enough"
    assert 10 <= len(prog.results) < 40
    idx = [r.index for r in prog.results]
    assert idx == sorted(idx)


def test_zero_ions():
    assert fly_all(0, _fly, n_workers=4) == []
