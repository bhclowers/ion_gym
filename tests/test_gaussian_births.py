"""
The truncated-Gaussian source places ions inside its truncation, from
the run's seed, and its rejection draw is bounded.

The draw used `while True`; gaussian_layout() keeps the truncation at
>= 1 sigma, so it always ended, but a layout that slipped past that rule
would have spun forever. It now gives up after GAUSSIAN_BIRTH_TRIES and
raises by name. (A seeded 500-ion birth set was checked bit-identical
before and after the bound.)
"""
import numpy as np
import pytest

from ion_gym.io.sim_spec import (ElectrodeSpec, GeometrySpec, ShapeSpec,
                                 SimSpec, SourceSpec, SymmetrySpec)
from ion_gym.physics.sim_build import generate_births


def _spec(seed=4):
    geo = GeometrySpec(width_mm=10.0, height_mm=10.0, mm_per_gu=0.5,
                       symmetry=SymmetrySpec(coords="xyz"),
                       electrodes=[ElectrodeSpec(name="g", shapes=[ShapeSpec(
                           "rect", {"x_mm": 0, "y_mm": 0, "width_mm": 10,
                                    "height_mm": 0.5})])])
    src = SourceSpec(distribution="gaussian", n_ions=500, x0_mm=5.0,
                     y0_mm=5.0, fwhm_mm=[1.0, 0.5, 0.0],
                     trunc_mm=[0.6, 0.3, 0.0], mz_list=[100.0], seed=seed)
    return SimSpec(geometry=geo, source=src)


def test_births_lie_inside_the_truncation_and_follow_the_seed():
    b = np.asarray(generate_births(_spec()))
    assert np.all(np.abs(b[:, 0] - 5.0) <= 0.6)
    assert np.all(np.abs(b[:, 1] - 5.0) <= 0.3)
    assert np.all(b[:, 2] == 0.0)
    assert np.std(b[:, 0]) > np.std(b[:, 1]) > 0.0
    assert np.array_equal(b, np.asarray(generate_births(_spec())))
    assert not np.array_equal(b, np.asarray(generate_births(_spec(5))))


def test_a_layout_no_draw_can_satisfy_is_refused(monkeypatch):
    monkeypatch.setattr(SourceSpec, "gaussian_layout",
                        lambda self: ([1.0, 0.0, 0.0], [1e-300, 0.0, 0.0]))
    with pytest.raises(RuntimeError, match="truncated Gaussian on axis x"):
        generate_births(_spec())
