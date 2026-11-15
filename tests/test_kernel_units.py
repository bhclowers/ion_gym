"""
Every argument of the flight and solve kernels states its unit.

The kernels mix units on purpose (pressure in Pa for the hard-sphere
kernels and in Torr for SDS, the step in ns, us or the caller's unit,
mass in Da) and take long positional argument lists, so a value in the
wrong unit flies without complaint: 1 Torr passed as P_pa is 1/133 of the
pressure meant. The signatures stay as they are (the kernels are
internal, docs/PUBLIC_API.md); instead each docstring carries an
"Arguments" block with one line per argument and its unit in brackets,
and this test holds every argument to it.
"""
import inspect
import re

import pytest

from ion_gym.physics import collision3d, flyer_plane, sds, solver2d, tracer3d

KERNELS = [
    collision3d.hs_collision_step, collision3d.fly_gas_uniform,
    collision3d.ensemble, tracer3d.fly3d, tracer3d.fly3d_sds,
    sds.sds_ion_params, flyer_plane.fly_checked, solver2d.solve_laplace,
]


def _params(fn):
    py = getattr(fn, "py_func", fn)          # numba dispatcher -> source
    return list(inspect.signature(py).parameters), inspect.getdoc(py) or ""


@pytest.mark.parametrize("fn", KERNELS, ids=lambda f: f.__name__)
def test_every_argument_has_a_unit_line(fn):
    names, doc = _params(fn)
    assert "Arguments (units in brackets" in doc, fn.__name__
    block = doc.split("Arguments (units in brackets", 1)[1]
    lines = block.splitlines()
    missing = []
    for n in names:
        if not any(re.match(rf"\s*(\w+,\s*)*{n}\b(,\s*\w+)*\s+\[[^\]]+\]", ln)
                   for ln in lines):
            missing.append(n)
    assert not missing, f"{fn.__name__}: no unit line for {missing}"


def test_the_pressure_units_differ_and_say_so():
    _, hs = _params(collision3d.hs_collision_step)
    _, sd = _params(sds.sds_ion_params)
    assert re.search(r"P_pa\s+\[Pa\]", hs)
    assert "P_torr [Torr]" in sd and "P_pa [Pa]" in sd
