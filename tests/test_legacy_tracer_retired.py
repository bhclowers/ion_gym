"""
The legacy 2-D numba tracer stays retired.

tracer_numba.fly never called the metal object it was given: it read
.surfaces and .bands from it and, when they were missing, flew a
hard-coded instrument (an 18 mm bore and three fixed bands). It was
retired with its kernel and field wrapper; the field build and sampler
that ionbench and tracer_tdep share remain.
"""
import ast

from ion_gym.io.paths import repo_root
from ion_gym.physics import tracer_numba

RETIRED = {"fly", "_fly_core", "FieldNumba"}
ROOT = repo_root()


def test_the_retired_names_are_gone():
    assert not RETIRED & set(vars(tracer_numba))
    assert {"build_field", "_sample"} <= set(vars(tracer_numba))


def test_nothing_imports_a_retired_name():
    hits = []
    for base in ("ion_gym", "tools", "tests"):
        for p in (ROOT / base).rglob("*.py"):
            for n in ast.walk(ast.parse(p.read_text())):
                if (isinstance(n, ast.ImportFrom) and n.module
                        and n.module.endswith("tracer_numba")):
                    hits += [(str(p), a.name) for a in n.names
                             if a.name in RETIRED]
    assert not hits, hits
