"""
Solved-field cache keys: two decks that rasterize different metal never
share a cached basis.

The basis cache is keyed on the geometry alone, so a geometry-affecting
field missing from the key makes the result depend on which deck was
built first. The planar origin_mm (the signed frame that places the node
lattice relative to the shapes) was such a field.
"""
import numpy as np
import pytest

from ion_gym.io import basis_cache
from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec)


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(origin):
    """The review's reproduction: the same two plates (absolute mm) in two
    signed frames."""
    geo = GeometrySpec(
        width_mm=20.0, height_mm=12.0, mm_per_gu=0.5, origin_mm=origin,
        electrodes=[ElectrodeSpec(name="top", dc=10.0,
                                  shapes=[_rect(-10, 2, 20, 1)]),
                    ElectrodeSpec(name="bot", dc=0.0,
                                  shapes=[_rect(-10, -2, 20, 1)])])
    src = SourceSpec(distribution="point", n_ions=1, x0_mm=-8.0, y0_mm=0.0,
                     z0_mm=0.0, direction=[1.0, 0.0, 0.0], ke_lo=1.0,
                     ke_hi=1.0, mz_list=[100.0], tob_span_us=0.0, seed=1)
    return SimSpec(geometry=geo, name="origin_key", source=src,
                   collisions=CollisionSpec(enabled=False),
                   integration=IntegrationSpec(dt_ns=2.0, t_max_us=0.5,
                                               rec_every=1))


A, B = [-10.0, -2.0], [-10.0, -6.0]


def test_origin_is_part_of_the_planar_key():
    assert basis_cache.key(_deck(A)) != basis_cache.key(_deck(B))


def test_zero_origin_keeps_the_legacy_key():
    assert basis_cache.key(_deck([0.0, 0.0])) == basis_cache.key(_deck(None))


def test_result_does_not_depend_on_cache_history(monkeypatch, tmp_path):
    from ion_gym.physics import build_planar
    from ion_gym.physics.sim_build import build_run

    def labels_after(sequence, cache):
        monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path / cache))
        build_planar.clear_memory_cache()
        model = None
        for origin in sequence:
            model = build_run(_deck(origin))[0]
        return np.array(model.ele)

    fresh = labels_after([B], "fresh")
    after_other = labels_after([A, B], "after_other")
    build_planar.clear_memory_cache()           # and from the disk tier
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path / "after_other"))
    from_disk = np.array(build_run(_deck(B))[0].ele)
    assert np.array_equal(fresh, after_other)
    assert np.array_equal(fresh, from_disk)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path / "default"))
