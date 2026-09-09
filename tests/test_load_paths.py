"""Load-path wiring tests — focused, purpose-built.

Every test here names ONE wiring seam and exercises it on the SMALLEST
input that fires the path. No test solves a real device; the multigrid
solve is stubbed wherever a test is about wiring (cache, dispatch), and
the only geometry used is a hand-written few-node fixture.

History: the first version of this file solved a
real SLIM board (216x56x21, minutes of multigrid) to check cache
and gas WIRING — a T1 violation. Rewritten per the approved plan. The
tiny mirror fixture immediately caught a real physics bug the SLIM board
had hidden: the retired import path's unfold concatenated the reflection on the wrong
side (mirror about the far end + duplicated node -> wrong board gap).
That is the argument for T1 in one sentence.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                 ShapeSpec, SourceSpec, IntegrationSpec,
                                 BoundsSpec, CollisionSpec)
from ion_gym.io.sim_spec import (SymmetrySpec)
from ion_gym.physics.sim_build import build_route

# ---------------------------------------------------------------- fixtures
def _tiny_planar_spec():
    geom = GeometrySpec(
        width_mm=6.0, height_mm=5.0, mm_per_gu=0.2,
        symmetry=SymmetrySpec(coords="xyz"),
        electrodes=[
            ElectrodeSpec(name="a", dc=0.0, shapes=[ShapeSpec(
                "rect", {"x_mm": 0.0, "y_mm": 0.0,
                         "width_mm": 6.0, "height_mm": 0.5})]),
            ElectrodeSpec(name="b", dc=5.0, shapes=[ShapeSpec(
                "rect", {"x_mm": 0.0, "y_mm": 4.5,
                         "width_mm": 6.0, "height_mm": 0.5})])])
    return SimSpec(name="tiny-planar", geometry=geom,
                   source=SourceSpec(seed=0, n_ions=1, x0_mm=3.0, y0_mm=2.5),
                   integration=IntegrationSpec(t_max_us=0.1),
                   bounds=BoundsSpec(),
                   collisions=CollisionSpec(enabled=False))


def _stub_solver(monkeypatch, counter=None):
    """Replace the multigrid bases solve with instant unit bases."""
    def _fake_solve(masks, *a, **k):
        if counter is not None:
            counter["n"] += 1
        shape = next(iter(masks.values())).shape
        return {i: np.ones(shape, np.float32) for i in masks}
    monkeypatch.setattr(
        "ion_gym.physics.multigrid3d.solve_bases_mg", _fake_solve)


def _isolate_cache(monkeypatch, tmp_path):
    """Redirect fa_cache.load/store to a per-test root. DEFAULT_ROOT is a
    DEFAULT ARGUMENT (bound at import), so patching the module constant
    does nothing — wrap the functions instead (looked up at call time)."""
    from ion_gym.io import fa_cache
    root = str(tmp_path / "cache")
    os.makedirs(root, exist_ok=True)
    real_load = fa_cache.load
    real_store = fa_cache.store

    def _load(spec_or_key, **kw):
        kw["root"] = root
        return real_load(spec_or_key, **kw)

    def _store(spec, arrays, **kw):
        kw["root"] = root
        return real_store(spec, arrays, **kw)

    monkeypatch.setattr(fa_cache, "load", _load)
    monkeypatch.setattr(fa_cache, "store", _store)


# ---------------------------------------------------- route classification
def test_route_planar_2d():
    r = build_route(_tiny_planar_spec())
    assert r.builder == "planar", r
    assert r.field_dims == 2, r


def test_route_matrix_all_builders():
    """build_route classifies every declared builder correctly — pure
    dispatch, tiny specs, no imported geometry, no solve. Pins the seam that misroutes
    silently."""
    def spec_with(builder="", coords="xyz", depth=0.0, stl=None):
        s = _tiny_planar_spec()
        s.builder = builder
        s.geometry.symmetry = SymmetrySpec(coords=coords)
        s.geometry.depth_mm = depth
        if stl:
            s.geometry.electrodes[0].stl = stl
        return s

    cases = [
        # The retired external-import builder is no longer a ROUTE; the
        # router refuses any builder it does not implement, and that
        # refusal is asserted separately below.
        (dict(coords="rz"), "rz"),
        (dict(depth=2.0, stl="rod.stl"), "stl3d"),
        (dict(depth=0.0, stl="rod.stl"), "stl2d"),
        (dict(depth=0.0), "planar"),
        (dict(builder="scene3d", depth=2.0), "scene3d"),
        # AMENDED: depth>0 native specs route to shapes3d
        # since the inline-extrusion wiring — 'unwired' predates
        # that feature.
        (dict(depth=2.0), "shapes3d"),
    ]
    for kw, expect in cases:
        r = build_route(spec_with(**kw))
        assert r.builder == expect, (kw, r)


def test_unknown_builder_is_refused_by_name():
    """A builder the router does not implement is REFUSED and quoted back --
    and no branch in the router knows any retired builder's name."""
    import pytest
    from ion_gym.physics.sim_build import build_route, KNOWN_BUILDERS
    for bogus in ("retired_importer", "device_builder", "wibble"):
        sp = _tiny_planar_spec()
        sp.builder = bogus
        with pytest.raises(ValueError) as ei:
            build_route(sp)
        msg = str(ei.value)
        assert bogus in msg, f"refusal did not quote {bogus!r}: {msg}"
        assert "Known builders" in msg
    # The router's contract is a LIST, not a set of named exclusions:
    # anything absent is refused, so no retired name needs to appear here.
    assert all(b.isidentifier() for b in KNOWN_BUILDERS)
