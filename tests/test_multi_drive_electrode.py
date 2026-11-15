"""
One electrode may carry more than one drive group, on every route and
under every gas model. This is a required capability; stacked coincident
electrodes, by contrast, are refused.

Stacking coincident electrodes to sum drives is refused (C-03, a node
claimed by two electrodes), so listing the groups on one electrode is the
only way to put two drives on one conductor. These tests pin that it works:
  * amplitudes add: drives of 3 V and 2 V at one frequency and phase fly
    exactly as one 5 V drive -- planar, r-z and 3-D, in vacuum, HS and SDS;
  * distinct drives both act: a 1 MHz sin and a 3 MHz square on one
    electrode move a vacuum ion by the sum of what each moves it alone
    (the excursion is microns, so the field is linear over it).
"""
import numpy as np
import pytest

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, RFGroupSpec, ShapeSpec,
                                 SimSpec, SourceSpec, SymmetrySpec)

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 2.0)]
GASES = {"vacuum": dict(enabled=False),
         "hs": dict(enabled=True, gas="N2", T_k=300.0, P_torr=1.0,
                    model="hs"),
         "sds": dict(enabled=True, gas="N2", T_k=300.0, P_torr=1.0,
                     model="sds")}


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(coords, depth_mm, groups, gas="vacuum"):
    """A driven plate over a grounded one; the driven plate carries every
    group in `groups`. One ion born at rest between them."""
    geo = GeometrySpec(
        width_mm=10.0, height_mm=6.0, depth_mm=depth_mm, mm_per_gu=0.25,
        symmetry=SymmetrySpec(coords=coords), rf_groups=list(groups),
        electrodes=[ElectrodeSpec(name="driven", dc=0.0,
                                  rf_groups=[g.name for g in groups],
                                  shapes=[_rect(0, 5, 10, 1)]),
                    ElectrodeSpec(name="ground", dc=0.0,
                                  shapes=[_rect(0, 0, 10, 0.5)])])
    src = SourceSpec(distribution="point", n_ions=1, x0_mm=5.0, y0_mm=3.0,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0], ke_lo=0.0,
                     ke_hi=0.0, mz_list=[200.0], tob_span_us=0.0, seed=7)
    return SimSpec(geometry=geo, name=f"multidrive_{coords}_{gas}",
                   source=src, collisions=CollisionSpec(**GASES[gas]),
                   integration=IntegrationSpec(dt_ns=1.0, t_max_us=1.0,
                                               rec_every=1))


def _sin(name, amp, f_hz=1.0e6, phase=30.0):
    return RFGroupSpec(name=name, waveform="sin", frequency_hz=f_hz,
                       amplitude_v=amp, phase_deg=phase)


def _fly(spec):
    from ion_gym.physics.sim_build import build_run
    _model, fly, cols, _births = build_run(spec)
    traj, _summary = fly(0)
    return np.asarray(traj), cols


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


@pytest.mark.parametrize("gas", list(GASES))
@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_two_drives_on_one_electrode_add(route, coords, depth, gas):
    split, cols = _fly(_deck(coords, depth, [_sin("a", 3.0), _sin("b", 2.0)],
                             gas))
    whole, _ = _fly(_deck(coords, depth, [_sin("s", 5.0)], gas))
    assert split.shape == whole.shape
    y = cols.index("y")
    moved = np.abs(whole[:, y] - whole[0, y]).max()
    assert moved > 0.0                     # the drive acts at all
    if gas != "vacuum":                    # the gas model acts at all
        vac, _ = _fly(_deck(coords, depth, [_sin("s", 5.0)]))
        assert whole.shape != vac.shape or not np.allclose(whole, vac)
    assert np.allclose(split, whole, rtol=1e-6, atol=1e-9 * moved), (
        np.abs(split - whole).max())


@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_stacked_copies_refuse_and_say_how(route, coords, depth):
    """The other half of the ruling: a coincident copy of the plate, one
    drive each, is refused, and the refusal names the supported way."""
    from ion_gym.physics.raster2d import SHARED_NODE_HINT
    from ion_gym.physics.sim_build import build_run
    spec = _deck(coords, depth, [_sin("a", 3.0), _sin("b", 2.0)])
    plate = spec.geometry.electrodes[0]
    plate.rf_groups = ["a"]
    spec.geometry.electrodes.append(
        ElectrodeSpec(name="driven_copy", dc=0.0, rf_groups=["b"],
                      shapes=plate.shapes))
    with pytest.raises(ValueError, match="overlaps") as err:
        build_run(spec)
    assert SHARED_NODE_HINT in str(err.value)


def test_stl_overlap_refusal_says_how():
    from ion_gym.physics.raster2d import SHARED_NODE_HINT
    from ion_gym.physics.voxelize import ElectrodeOverlap
    assert SHARED_NODE_HINT in str(ElectrodeOverlap(1, 2, (0, 0, 0), 3))


@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_distinct_drives_on_one_electrode_both_act(route, coords, depth):
    lf = _sin("lf", 3.0, f_hz=1.0e6, phase=0.0)
    hf = RFGroupSpec(name="hf", waveform="square", frequency_hz=3.0e6,
                     amplitude_v=2.0, phase_deg=0.0)
    both, cols = _fly(_deck(coords, depth, [lf, hf]))
    only_lf, _ = _fly(_deck(coords, depth, [lf]))
    only_hf, _ = _fly(_deck(coords, depth, [hf]))
    y = cols.index("y")
    d_both = both[:, y] - both[0, y]
    d_sum = (only_lf[:, y] - only_lf[0, y]) + (only_hf[:, y] - only_hf[0, y])
    scale = np.abs(d_sum).max()
    assert np.abs(only_hf[:, y] - only_hf[0, y]).max() > 0.1 * scale
    assert np.abs(d_both - d_sum).max() < 1e-2 * scale, (
        np.abs(d_both - d_sum).max() / scale)
