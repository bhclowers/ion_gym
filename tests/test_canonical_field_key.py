"""
A deck keys its solved 2-D field the same however its symmetry is spelled.

The planar/r-z cache key took the symmetry block as written, so the same
deck keyed apart with `planes: {}` (as a deck file usually has it) and
with every plane spelled out as 'none' (as the app's synced copy has it).
A field solved in a notebook or script from the deck as written was then
not found by the app, which re-solved it, and a saved field would not
match. The key now reads the normalised symmetry.
"""
import pytest

from ion_gym.io import basis_cache
from ion_gym.io.sim_spec import SimSpec


def _deck(coords="xyz", planes=None):
    return SimSpec.from_dict({
        "name": "key_spelling",
        "source": {"seed": 0, "distribution": "point", "n_ions": 1,
                   "x0_mm": 2.0, "y0_mm": 0.5, "z0_mm": 0.0, "ke_lo": 1.0,
                   "ke_hi": 1.0, "mz_list": [300.0], "tob_span_us": 0.0},
        "integration": {"dt_ns": 1.0, "t_max_us": 1.0},
        "geometry": {
            "width_mm": 4.0, "height_mm": 3.0, "depth_mm": 0.0,
            "mm_per_gu": 0.25,
            "symmetry": {"coords": coords, "planes": planes or {}},
            "electrodes": [{"name": "E1", "dc": 5.0, "shapes": [
                {"type": "rect", "x_mm": 0.0, "y_mm": 1.0, "width_mm": 0.5,
                 "height_mm": 2.0}]}]},
    })


@pytest.mark.parametrize("coords", ["xyz", "rz"])
def test_omitted_and_spelled_out_planes_key_alike(coords):
    bare = _deck(coords)
    spelled = _deck(coords, {"x": "none", "y": "none", "z": "none"})
    assert basis_cache.key(bare) == basis_cache.key(spelled)


def test_a_partial_spelling_keys_alike_too():
    assert (basis_cache.key(_deck(planes={"x": "none"}))
            == basis_cache.key(_deck()))


def test_a_real_plane_still_changes_the_key():
    assert (basis_cache.key(_deck(planes={"y": "mirror"}))
            != basis_cache.key(_deck()))


def test_the_app_finds_a_field_solved_from_the_deck_as_written(
        tmp_path, monkeypatch):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))
    from ion_gym.physics import build_planar
    from ion_gym.physics.sim_build import build_run
    from ion_gym.ui.sim_app import SimApp
    monkeypatch.setattr(build_planar, "_PLANAR_BASIS_CACHE", {})
    deck = _deck()
    build_run(deck)                          # solved as written
    app = SimApp(_deck())
    app._sync_spec()                         # the app's spelled-out copy
    assert basis_cache.load(app.spec)[0] is not None
