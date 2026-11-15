"""
A beam energy declared with a ZERO source.direction has nowhere to point.
It is ignored, loudly: ions are born at rest, an advisory and a births
warning say so, and the deck is not refused. This is what such decks have
always flown (births normalise the direction, and a zero vector left the
directed term at zero); the energy is still drawn, so the random stream
and every other birth quantity are unchanged.
"""
import dataclasses
import warnings

import numpy as np
import pytest

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 2.0)]
GASES = {"vacuum": dict(enabled=False),
         "hs": dict(enabled=True, gas="N2", T_k=300.0, P_torr=1.0,
                    model="hs"),
         "sds": dict(enabled=True, gas="N2", T_k=300.0, P_torr=1.0,
                     model="sds")}


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(coords="xyz", depth_mm=0.0, direction=(0.0, 0.0, 0.0),
          gas="vacuum"):
    geo = GeometrySpec(
        width_mm=10.0, height_mm=6.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[ElectrodeSpec(name="top", dc=5.0,
                                  shapes=[_rect(0, 5, 10, 1)]),
                    ElectrodeSpec(name="bot", dc=0.0,
                                  shapes=[_rect(0, 0, 10, 1)])])
    src = SourceSpec(distribution="disc", n_ions=4, x0_mm=5.0, y0_mm=3.0,
                     z0_mm=depth_mm / 2, r_mm=0.5, axis="x",
                     direction=list(direction), ke_lo=0.02, ke_hi=0.05,
                     mz_list=[120.0, 322.0], tob_span_us=0.0, seed=3)
    return SimSpec(geometry=geo, name="zero_direction", source=src,
                   collisions=CollisionSpec(**GASES[gas]),
                   integration=IntegrationSpec(dt_ns=2.0, t_max_us=0.2,
                                               rec_every=1))


def _births(spec):
    from ion_gym.physics.sim_build import generate_births
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b = np.asarray(generate_births(spec))
    return b, [str(w.message) for w in caught]


def test_validates_and_says_the_energy_is_ignored():
    spec = _deck()
    assert not [e for e in spec.validate() if "direction" in e]
    notes = [a for a in spec.advisories() if "nowhere to point" in a]
    assert len(notes) == 1 and notes[0].startswith("WARNING")
    assert "0.02..0.05 eV" in notes[0] and "IGNORED" in notes[0]
    births, said = _births(spec)
    assert any("nowhere to point" in m for m in said)
    assert np.array_equal(births[:, 3:6], np.zeros_like(births[:, 3:6]))


def test_the_random_stream_is_unchanged():
    """The energy is still drawn: births positions match the same deck with
    a real direction, so nothing but the directed velocity moves."""
    at_rest, _ = _births(_deck())
    aimed, said = _births(_deck(direction=(1.0, 0.0, 0.0)))
    assert not any("nowhere to point" in m for m in said)
    assert np.array_equal(at_rest[:, 0:3], aimed[:, 0:3])
    assert np.abs(aimed[:, 3]).min() > 0.0


def test_no_note_with_a_direction_or_without_an_energy():
    assert _deck(direction=(1.0, 0.0, 0.0)).source.zero_direction_note() is None
    spec = _deck()
    spec.source = dataclasses.replace(spec.source, ke_lo=0.0, ke_hi=0.0)
    assert spec.source.zero_direction_note() is None


@pytest.mark.parametrize("bad", [["a", 0.0, 0.0], [1.0, 0.0], 1.0,
                                 [float("nan"), 0.0, 0.0]],
                         ids=["string", "two", "scalar", "nan"])
def test_a_malformed_direction_is_refused_by_name_everywhere(bad):
    """validate() reports it (a string entry used to escape as a raw
    ValueError), and the note and the births generator raise it instead
    of returning no note (they used to return None silently)."""
    from ion_gym.physics.sim_build import generate_births
    spec = _deck(direction=(1.0, 0.0, 0.0))
    spec.source = dataclasses.replace(spec.source, direction=bad)
    errs = [e for e in spec.validate() if "source.direction" in e]
    # (NaN is also named by the deck-wide finite-number check)
    assert any("3-vector of finite floats" in e for e in errs), errs
    with pytest.raises(ValueError, match="source.direction must be"):
        spec.source.zero_direction_note()
    with pytest.raises(ValueError, match="source.direction must be"):
        generate_births(spec)


@pytest.fixture
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


@pytest.mark.parametrize("gas", list(GASES))
@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_born_at_rest_on_every_route(route, coords, depth, gas,
                                     _isolated_cache):
    from ion_gym.physics.sim_build import build_run
    _model, fly, cols, births = build_run(_deck(coords, depth, gas=gas))
    assert np.array_equal(np.asarray(births)[:, 3:6], np.zeros((8, 3)))
    traj = np.asarray(fly(0)[0])
    v0 = [traj[0, cols.index(c)] for c in ("vx", "vy", "vz") if c in cols]
    assert v0 and all(v == 0.0 for v in v0)
