"""
The record-capacity arithmetic (sim_spec.min_rec_every and
check_recording_capacity) gives the smallest covering rec_every, and
refuses a flight it cannot count instead of looping.

Review F11: with t_max_us from 1e27 up, steps passed 2**53, `r -= 1` no
longer changed steps / r, and build_run hung inside the check meant to
refuse the deck. The hang case runs in a subprocess with a timeout, so a
regression fails here instead of hanging the suite.
"""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ion_gym.io.sim_spec import (MAX_FLIGHT_STEPS, _records_needed,
                                 check_recording_capacity, min_rec_every)


def _integ(dt_ns, t_max_us, cap=100_000, every=1):
    return SimpleNamespace(dt_ns=dt_ns, t_max_us=t_max_us, max_records=cap,
                           rec_every=every)


@pytest.mark.parametrize("t_max_us", [1.0, 400.0, 3000.0, 1e6, 1e9, 1e12])
@pytest.mark.parametrize("cap", [3, 1000, 100_000])
def test_the_smallest_covering_rec_every(t_max_us, cap):
    integ = _integ(1.0, t_max_us, cap)
    r = min_rec_every(integ)
    steps = t_max_us * 1e3
    assert _records_needed(steps, r) <= cap
    assert r == 1 or _records_needed(steps, r - 1) > cap


@pytest.mark.parametrize("t_max_us", [1e27, 1e40, float("inf"),
                                      float("nan")])
def test_an_uncountable_flight_is_refused_not_looped(t_max_us):
    code = (
        "from types import SimpleNamespace\n"
        "from ion_gym.io.sim_spec import min_rec_every, "
        "check_recording_capacity\n"
        f"i = SimpleNamespace(dt_ns=1.0, t_max_us=float({str(t_max_us)!r}), "
        "max_records=100000, rec_every=1)\n"
        "for f in (min_rec_every, check_recording_capacity):\n"
        "    try:\n"
        "        f(i)\n"
        "    except ValueError as e:\n"
        "        assert 'integrator steps' in str(e), e\n"
        "    else:\n"
        "        raise SystemExit(f'{f.__name__} did not refuse')\n"
        "print('refused')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "refused"


def test_just_below_the_limit_still_counts():
    steps = float(MAX_FLIGHT_STEPS) / 2
    integ = _integ(1.0, steps * 1e-3, cap=100_000, every=1)
    msg = check_recording_capacity(integ)
    assert msg and "rec_every=" in msg
