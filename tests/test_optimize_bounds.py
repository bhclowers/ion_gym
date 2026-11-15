"""
optimize() refuses a max_evals that cannot bound the search.

The CMA-ES loop runs `while not es.stop()`, and max_evals (CMA's
maxfevals) is what bounds it. None, zero, a fraction or infinity used to
reach CMA unchecked; it is now refused by name before any evaluation
(and before the optional cma package is imported).
"""
import pytest

from ion_gym.physics.optimize import OptimizeSpec, optimize


@pytest.mark.parametrize("bad", [None, 0, -5, 2.5, float("inf"), True],
                         ids=["none", "zero", "negative", "fraction", "inf",
                              "bool"])
def test_an_unbounding_max_evals_is_refused(bad):
    osp = OptimizeSpec(params=["geometry.electrodes[0].dc"], lo=[-1.0],
                       hi=[1.0], max_evals=bad)
    with pytest.raises(ValueError, match="max_evals must be a positive"):
        optimize(None, osp)
