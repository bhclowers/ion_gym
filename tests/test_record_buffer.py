"""
collision3d.fly_gas_uniform / ensemble record every record_every steps
plus a final row. Their buffer was sized n_steps // record_every + 1, one
row short whenever record_every does not divide n_steps, and numba writes
past the end without a bounds check (the review measured a malloc abort,
exit 134, at dt 1e-3 us, t 1.003 us, record_every 10).

The end-to-end case runs in a subprocess, so a regression aborts that
process instead of corrupting the test run.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("n_steps,every,want", [
    (1003, 10, 102), (1000, 10, 101), (0, 10, 1), (5, 10, 2), (7, 1, 8)])
def test_buffer_holds_every_row_written(n_steps, every, want):
    from ion_gym.physics.collision3d import n_records
    assert n_records(n_steps, every) == want == math.ceil(n_steps / every) + 1


_CHILD = """
import json, sys
sys.path.insert(0, {root!r})
from ion_gym.physics.collision3d import ensemble
recs, cols = ensemble(2, 100.0, 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0), 300.0, 100.0, 1.0e-18, gas="N2",
                      dt_us=1e-3, t_us=1.003, record_every=10)
print(json.dumps([[float(v) for v in r[:, 0]] for r in recs]))
"""


def test_ensemble_survives_a_record_interval_that_does_not_divide():
    out = subprocess.run([sys.executable, "-c", _CHILD.format(root=str(ROOT))],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])
    times = json.loads(out.stdout.strip().splitlines()[-1])
    n_steps = int(1.003 / 1e-3)
    for t in times:
        assert len(t) == math.ceil(n_steps / 10) + 1
        assert all(b > a for a, b in zip(t, t[1:]))
        assert t[-1] == pytest.approx(n_steps * 1e-3)


def test_a_record_interval_below_one_is_refused():
    from ion_gym.physics.collision3d import ensemble
    with pytest.raises(ValueError, match="record_every"):
        ensemble(1, 100.0, 1, (0, 0, 0), (0, 0, 0), (1, 0, 0), 300.0, 100.0,
                 1e-18, gas="N2", dt_us=1e-3, t_us=0.01, record_every=0)
