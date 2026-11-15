"""
numba_rng.py -- an ion flight leaves numba's random state as it found it.

The collision and SDS kernels seed numba's random generator per ion
(np.random.seed inside @njit code, so each ion's draws depend only on its
own seed). That generator is per thread and shared with any @njit code the
user runs: after one flight a user's own seeded @njit random sequence came
out different. NumPy's own global generator was never touched.

call_rng_preserved(kernel, ...) saves the calling thread's numba random
state, calls the kernel, and restores the state, so a flight is invisible
to other numba code on that thread. The kernels are unchanged, so their
results are bit-identical. numba keeps this state behind a private
helper (numba._helperlib); a numba that removes it fails at import here,
loudly, and tests/test_numba_rng_isolation.py pins the behaviour.
"""
from numba import _helperlib


def call_rng_preserved(kernel, *args, **kwargs):
    """kernel(*args, **kwargs), with this thread's numba random state put
    back afterwards."""
    ptr = _helperlib.rnd_get_np_state_ptr()
    state = _helperlib.rnd_get_state(ptr)
    try:
        return kernel(*args, **kwargs)
    finally:
        _helperlib.rnd_set_state(ptr, state)
