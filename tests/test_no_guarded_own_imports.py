"""
ion_gym's own cache modules are imported, never guarded, on the paths
that decide which solve runs and under which key.

`except ImportError` around `from ion_gym.io import basis_cache` (and
fa_cache) turned a broken install into a different answer, silently:
build_planar._geometry_key fell back to a hand-rolled second key,
planar_is_cached answered "cached", the planar solve banked nothing, and
build_needs_solve answered "will solve". A module of this package that
fails to import must fail the call.
"""
import sys

import pytest

import ion_gym.io

from ion_gym.io.sim_spec import (ElectrodeSpec, GeometrySpec, ShapeSpec,
                                 SimSpec, SymmetrySpec)


def _break(monkeypatch, name):
    """Make `from ion_gym.io import <name>` raise ImportError."""
    monkeypatch.delattr(ion_gym.io, name, raising=False)
    monkeypatch.setitem(sys.modules, f"ion_gym.io.{name}", None)


def _spec(depth_mm=0.0):
    ex = {"axis": "z", "lo_mm": 0.0, "hi_mm": depth_mm} if depth_mm else None
    params = {"x_mm": 0, "y_mm": 0, "width_mm": 10, "height_mm": 1}
    if ex:
        params["extrude"] = ex
    return SimSpec(geometry=GeometrySpec(
        width_mm=10.0, height_mm=6.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords="xyz"),
        electrodes=[ElectrodeSpec(name="p", dc=1.0,
                                  shapes=[ShapeSpec("rect", params)])]))


def test_planar_key_fails_instead_of_inventing_a_second_key(monkeypatch):
    from ion_gym.physics.build_planar import _geometry_key
    _break(monkeypatch, "basis_cache")
    with pytest.raises(ImportError):
        _geometry_key(_spec())


def test_planar_cache_question_fails_instead_of_guessing(monkeypatch):
    from ion_gym.physics import build_planar
    monkeypatch.setattr(build_planar, "_PLANAR_BASIS_CACHE", {})
    _break(monkeypatch, "basis_cache")
    with pytest.raises(ImportError):
        build_planar.planar_is_cached(_spec())


def test_3d_solve_question_fails_instead_of_guessing(monkeypatch):
    from ion_gym.physics.sim_build import build_needs_solve
    _break(monkeypatch, "fa_cache")
    with pytest.raises(ImportError):
        build_needs_solve(_spec(depth_mm=4.0))


# The STL routes solve by multigrid. An `except ImportError` around that
# import used to solve by SOR instead: a broken install changed the solver,
# and so the field. With multigrid unimportable the build must fail, and
# the SOR solver must never be reached.

def _break_multigrid(monkeypatch):
    import ion_gym.physics
    import ion_gym.physics.solver3d as solver3d
    monkeypatch.delattr(ion_gym.physics, "multigrid3d", raising=False)
    monkeypatch.setitem(sys.modules, "ion_gym.physics.multigrid3d", None)

    def _sor(*a, **k):
        raise AssertionError("solved by SOR in place of multigrid")
    monkeypatch.setattr(solver3d, "solve_bases", _sor)


def test_stl_2d_bases_fail_without_multigrid(monkeypatch):
    import numpy as np
    from ion_gym.physics import build_stl
    spec = _spec()
    nx, ny = build_stl._grid_counts_2d(spec.geometry)
    plate = np.zeros((nx, ny), bool)
    plate[nx // 2, 2:-2] = True
    monkeypatch.setattr(build_stl, "stl_masks_2d",
                        lambda spec, verbose=False: {1: plate})
    monkeypatch.setattr(build_stl, "_stl_geom_signature",
                        lambda spec, gb: ("no-multigrid test",))
    _break_multigrid(monkeypatch)
    with pytest.raises(ImportError, match="multigrid3d"):
        build_stl._stl_bases(spec, ground_border=False, verbose=False)


def test_stl_3d_bases_fail_without_multigrid(monkeypatch):
    import numpy as np
    from ion_gym.io import fa_cache
    from ion_gym.physics import build_stl3d
    spec = _spec(depth_mm=4.0)
    nx, ny, nz, _ = build_stl3d._grid_from_spec(spec)
    plate = np.zeros((nx, ny, nz), bool)
    plate[nx // 2, 2:-2, 2:-2] = True
    monkeypatch.setattr(fa_cache, "load", lambda key: (None, None))
    monkeypatch.setattr(build_stl3d, "solve_bases",
                        lambda *a, **k: pytest.fail("solved by SOR"),
                        raising=False)
    _break_multigrid(monkeypatch)
    with pytest.raises(ImportError, match="multigrid3d"):
        build_stl3d.build_stl3d_run(
            spec, masks_fn=lambda spec, verbose=False: {1: plate},
            geom_bytes=lambda spec: b"no-multigrid test")
