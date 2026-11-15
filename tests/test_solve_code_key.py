"""
Cached fields are keyed on the code that solves them (physics.solve_code).

The field caches keyed on geometry plus hand-set version markers, so a fix
to a solver or rasterizer reached cached geometries only when someone
bumped a marker. Now:

  * the solve-code key is part of every field cache key (planar and r-z
    through basis_cache, 3-D through build_stl3d), and not part of a
    field's geometry identity (field files, the editor);
  * a docstring or comment edit leaves the key alone; a code edit changes
    it; a listed function that no longer exists raises;
  * the listed builder functions reach no other function of their module,
    and no other ion_gym module, outside the solve path or the reasoned
    allowances below (so the list cannot quietly miss solve code);
  * a saved field is matched by its cache key, so one solved by other
    solve code is refused by name, and one from this code loads into the
    cache.
"""
import ast
import json
import os
import warnings

import numpy as np
import pytest

from ion_gym.advisories import IonGymAdvisory
from ion_gym.io import basis_cache, fa_cache, field_io
from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics import solve_code
from ion_gym.physics.solve_code import (PKG, SOLVE_FUNCTIONS, SOLVE_MODULES,
                                        key_of_sources, solve_code_key)

# called by a listed builder function, but nothing they compute is cached
NOT_CACHED_CALLS = {
    "_freeze": "the in-process model cache",
    "_geometry_key": "the cache key itself",
    "_grad2d": "E from the bases, recomputed at every build",
    "_electrode_bands": "hit bands from the bases, recomputed every build",
    "assemble_drive_groups": "voltage composition (bases are per volt)",
}
# imported by a listed builder function, but not solve code
NOT_SOLVE_MODULES = {
    "ion_gym.io": "basis_cache / fa_cache storage",
    "ion_gym.io.basis_cache": "storage and the key",
    "ion_gym.io.fa_cache": "storage",
    "ion_gym.physics.collision3d": "the r-z model's gas set-up",
    "ion_gym.physics.build_stl": "_require_trimesh, an install probe",
    "ion_gym.physics.build_stl3d": "functions already listed",
}

SRC = '''
"""module docstring"""
K = 2  # a constant


def f(x):
    """old wording"""
    # a comment
    return x * K
'''


def test_the_key_is_stable_and_names_its_code():
    assert solve_code_key() == solve_code_key()
    assert len(solve_code_key()) == 16


def test_wording_edits_keep_the_key_and_code_edits_change_it():
    base = key_of_sources({"m.py": SRC}, {"b.py": (SRC, ("f",))})
    worded = SRC.replace("old wording", "new").replace("a comment", "x")
    assert key_of_sources({"m.py": worded},
                          {"b.py": (worded, ("f",))}) == base
    coded = SRC.replace("x * K", "x * K + 1")
    assert key_of_sources({"m.py": SRC}, {"b.py": (coded, ("f",))}) != base
    # a constant the listed function reads is part of its code
    const = SRC.replace("K = 2", "K = 3")
    assert key_of_sources({"m.py": SRC}, {"b.py": (const, ("f",))}) != base


def test_a_listed_function_that_is_gone_raises():
    with pytest.raises(LookupError, match="no top-level function"):
        key_of_sources({}, {"b.py": (SRC, ("renamed",))})


@pytest.mark.parametrize("path", sorted(SOLVE_FUNCTIONS))
def test_the_listed_functions_reach_nothing_unlisted(path):
    names = SOLVE_FUNCTIONS[path]
    tree = ast.parse((PKG / path).read_text())
    top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    solve_mods = {"ion_gym." + p[:-3].replace("/", ".")
                  for p in SOLVE_MODULES}
    calls, mods = set(), set()
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name in names):
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in top and n.func.id not in names):
                calls.add(n.func.id)
            if (isinstance(n, ast.ImportFrom) and n.module
                    and n.module.startswith("ion_gym")
                    and n.module not in solve_mods):
                mods.add(n.module)
    assert not calls - set(NOT_CACHED_CALLS), calls - set(NOT_CACHED_CALLS)
    assert not mods - set(NOT_SOLVE_MODULES), mods - set(NOT_SOLVE_MODULES)


def _deck(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SimSpec.from_json(str(repo_root() / "examples" / name))


def test_the_cache_keys_carry_it_and_the_geometry_key_does_not(monkeypatch):
    from ion_gym.physics.build_stl3d import _bases_cache_key
    planar = _deck("einzel_round_r-z.json")
    shapes = _deck("hexapole_native_3d.json")
    from ion_gym.physics.build_shapes3d import _geom_bytes
    before = (basis_cache.key(planar),
              _bases_cache_key(shapes, geom_bytes=_geom_bytes, tag="t"),
              json.dumps(basis_cache.geometry_key_dict(planar),
                         sort_keys=True))
    monkeypatch.setattr(solve_code, "solve_code_key", lambda: "0" * 16)
    after = (basis_cache.key(planar),
             _bases_cache_key(shapes, geom_bytes=_geom_bytes, tag="t"),
             json.dumps(basis_cache.geometry_key_dict(planar),
                        sort_keys=True))
    assert after[0] != before[0] and after[1] != before[1]
    assert after[2] == before[2]           # geometry identity unchanged


def _saved_field(tmp_path):
    """A field saved the way the app saves it: solved bases in the cache
    (stood in for by storing synthetic ones), then save_field."""
    spec = _deck("einzel_round_r-z.json")
    shape = (8, 5)
    bases = {i + 1: np.full(shape, float(i))
             for i in range(len(spec.geometry.electrodes))}
    ele = np.zeros(shape, np.int16)
    solved = str(tmp_path / "solved")
    basis_cache.store(spec, bases, ele, root=solved)
    path, _ = field_io.save_field(spec, directory=str(tmp_path / "fields"),
                                  root=solved)
    return spec, path


def test_a_field_from_other_solve_code_is_refused(tmp_path, monkeypatch):
    spec, p = _saved_field(tmp_path)
    root = str(tmp_path / "cache")
    monkeypatch.setattr(solve_code, "solve_code_key", lambda: "f" * 16)
    with pytest.raises(field_io.FieldMismatch, match="solver code"):
        field_io.load_field(p, spec, root=root)
    assert not os.path.isdir(root) or not os.listdir(root)


def test_a_field_from_this_solve_code_is_cached(tmp_path):
    spec, p = _saved_field(tmp_path)
    root = str(tmp_path / "cache")
    with warnings.catch_warnings():
        warnings.simplefilter("error", IonGymAdvisory)
        field_io.load_field(p, spec, root=root)
    assert fa_cache.load(basis_cache.cache_key_dict(spec),
                         root=root)[0] is not None
