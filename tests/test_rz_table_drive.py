"""
A table (gate) drive means the same thing on r-z as on planar: the
electrode sits at dc + amplitude_v * w(t) + offset_v, with w the table
value, before and after the step.

The r-z gate channel once dropped the table's first value and the group
offset, so before the gate a table [10, 30] with offset 5 held its
electrode at dc on r-z and at dc + 15 on planar.
"""
import numpy as np
import pytest

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, RFGroupSpec, ShapeSpec,
                                 SimSpec, SourceSpec, SymmetrySpec)

DC, AMP, V0, V1, OFFSET, TAU_US = 0.0, 1.0, 10.0, 30.0, 5.0, 0.5


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(coords, depth_mm=0.0):
    """The review's reproduction: a gated plate over a grounded one."""
    gate = RFGroupSpec(name="gt", amplitude_v=AMP, frequency_hz=0.0,
                       waveform="table", table_t_us=[0.0, TAU_US],
                       table_v=[V0, V1], interp="hold", offset_v=OFFSET)
    geo = GeometrySpec(
        width_mm=10.0, height_mm=6.0, depth_mm=depth_mm, mm_per_gu=0.25,
        symmetry=SymmetrySpec(coords=coords), rf_groups=[gate],
        electrodes=[ElectrodeSpec(name="gated", dc=DC, rf_groups=["gt"],
                                  shapes=[_rect(0, 5, 10, 1)]),
                    ElectrodeSpec(name="ground", dc=0.0,
                                  shapes=[_rect(0, 0, 10, 0.5)])])
    src = SourceSpec(distribution="point", n_ions=1, x0_mm=5.0, y0_mm=3.0,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0], ke_lo=0.0,
                     ke_hi=0.0, mz_list=[2000.0], tob_span_us=0.0, seed=1)
    return SimSpec(geometry=geo, name=f"gate_{coords}", source=src,
                   collisions=CollisionSpec(enabled=False),
                   integration=IntegrationSpec(dt_ns=1.0, t_max_us=1.0,
                                               rec_every=1))


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


def _gated_nodes(model):
    return np.asarray(model.ele) == 1


def test_pre_gate_potential_agrees_between_planar_and_rz():
    from ion_gym.physics.sim_build import build_run
    planar = build_run(_deck("xyz"))[0]
    rz = build_run(_deck("rz"))[0]
    want = DC + AMP * V0 + OFFSET
    v_planar = planar.phi_at(0.5 * TAU_US)[_gated_nodes(planar)]
    v_rz = np.asarray(rz.A)[_gated_nodes(rz)]
    assert np.allclose(v_planar, want), v_planar.max()
    assert np.allclose(v_rz, want), v_rz.max()


@pytest.mark.parametrize("coords,depth", [("rz", 0.0), ("xyz", 0.0),
                                          ("xyz", 2.0)],
                         ids=["rz", "planar", "3d"])
def test_ion_feels_the_pre_gate_field_on_every_route(coords, depth):
    from ion_gym.physics.sim_build import build_run
    _model, fly, cols, _births = build_run(_deck(coords, depth))
    traj, _summary = fly(0)
    traj = np.asarray(traj)
    row = np.searchsorted(traj[:, cols.index("t")], 0.4)
    assert traj[row, cols.index("vy")] != 0.0
