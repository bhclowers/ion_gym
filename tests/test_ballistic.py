"""
physics.ballistic: field-free motion along one axis, checked against
SI arithmetic done independently of ion_gym's constants.
"""
import numpy as np
import pytest

from ion_gym.physics.ballistic import (add_ballistic_axis, position_at,
                                       speed_from_energy, transverse_walk)

E_C = 1.602176634e-19          # C (exact, SI 2019)
AMU = 1.66053906660e-27        # kg (CODATA 2018)


@pytest.mark.parametrize("ev, da", [(1.0, 100.0), (25.0, 622.0),
                                    (2000.0, 18.0)])
def test_speed_matches_si_kinetic_energy(ev, da):
    v_m_s = np.sqrt(2.0 * ev * E_C / (da * AMU))
    assert speed_from_energy(ev, da) == pytest.approx(v_m_s * 1e-3,
                                                      rel=1e-12)


def test_negative_energy_is_refused():
    with pytest.raises(ValueError, match="negative energy"):
        speed_from_energy([1.0, -0.1], 100.0)


def test_position_is_linear_in_time():
    ext = add_ballistic_axis(np.zeros((3, 4)), 100.0, 5.0,
                             rng=np.random.default_rng(1), center_mm=2.0,
                             footprint_mm=1.0, energy_sigma_ev=0.5)
    assert np.all(np.abs(ext["pos0_mm"] - 2.0) <= 0.5)
    np.testing.assert_allclose(position_at(ext, 0.0), ext["pos0_mm"])
    np.testing.assert_allclose(
        position_at(ext, 3.0) - position_at(ext, 1.0),
        2.0 * ext["speed_mm_us"], rtol=1e-14)


def test_a_seeded_rng_reproduces_the_extension():
    kw = dict(rng=None, center_mm=0.0, footprint_mm=2.0, energy_sigma_ev=0.3)
    a = add_ballistic_axis(np.zeros((5, 4)), 200.0, 3.0,
                           **{**kw, "rng": np.random.default_rng(7)})
    b = add_ballistic_axis(np.zeros((5, 4)), 200.0, 3.0,
                           **{**kw, "rng": np.random.default_rng(7)})
    for k in ("pos0_mm", "energy_ev", "speed_mm_us"):
        np.testing.assert_array_equal(a[k], b[k])


def test_transverse_walk_is_the_speed_ratio_times_path():
    # dZ = (v_t / v_a) * L, and the mass cancels
    for da in (50.0, 500.0):
        ratio = speed_from_energy(0.04, da) / speed_from_energy(100.0, da)
        assert transverse_walk(0.04, 100.0, 250.0) == pytest.approx(
            ratio * 250.0, rel=1e-12)
