"""Test load_drive_template — the voltage/drive-only template loader.

Purpose-built: pins the exact declared contract — move both
rf and dc groups, wipe the target's groups, clear ALL electrode bindings,
touch nothing else, and don't alias the source. Self-contained fixtures
(K9).
"""

from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                  ShapeSpec, SourceSpec, IntegrationSpec,
                                  BoundsSpec, CollisionSpec, RFGroupSpec,
                                  DCGroupSpec, load_drive_template)
from ion_gym.io.sim_spec import (SymmetrySpec)


def _rect(name, x, dc, rf_groups, dc_group=None, dc_index=None):
    return ElectrodeSpec(
        name=name, dc=dc, rf_groups=list(rf_groups),
        dc_group=dc_group, dc_index=dc_index,
        shapes=[ShapeSpec("rect", {"x_mm": x, "y_mm": 0.0,
                                   "width_mm": 1.0, "height_mm": 1.0})])


def _spec(name, rf_names, dc_names, electrodes):
    return SimSpec(
        name=name,
        geometry=GeometrySpec(
            width_mm=10.0, height_mm=10.0, mm_per_gu=0.5,
            symmetry=SymmetrySpec(coords="xyz"),
            electrodes=electrodes,
            rf_groups=[RFGroupSpec(name=n, frequency_hz=1e5,
                                   amplitude_v=10.0) for n in rf_names],
            dc_groups=[DCGroupSpec(name=n, v_in=1.0, v_out=1.0)
                       for n in dc_names]),
        source=SourceSpec(seed=0, n_ions=5, x0_mm=5, y0_mm=5, mz_list=[100.0]),
        integration=IntegrationSpec(t_max_us=10.0, dt_ns=2.0),
        bounds=BoundsSpec(), collisions=CollisionSpec(enabled=False))


def _target():
    els = [_rect("a", 1.0, 3.0, ["OLD_RF"], "OLD_DC", 0),
           _rect("b", 2.0, 3.0, ["OLD_RF"], "OLD_DC", 1),
           _rect("c", 3.0, 0.0, [])]                # already unbound
    return _spec("target", ["OLD_RF"], ["OLD_DC"], els)


def _source():
    els = [_rect("s1", 0.0, 0.0, ["NEW_A", "NEW_B"], "NEW_DC", 0)]
    return _spec("template", ["NEW_A", "NEW_B", "NEW_C"],
                 ["NEW_DC"], els)


def test_groups_are_replaced():
    t = _target()
    rep = load_drive_template(t, _source())
    assert [g.name for g in t.geometry.rf_groups] == ["NEW_A", "NEW_B",
                                                      "NEW_C"]
    assert [g.name for g in t.geometry.dc_groups] == ["NEW_DC"]
    assert rep["rf_removed"] == ["OLD_RF"]
    assert rep["dc_loaded"] == ["NEW_DC"]


def test_all_electrode_bindings_cleared():
    t = _target()
    rep = load_drive_template(t, _source())
    for e in t.geometry.electrodes:
        assert e.rf_groups == []
        assert e.dc_group is None
        assert e.dc_index is None
    # 2 of 3 electrodes had bindings; the third was already unbound
    assert rep["electrodes_cleared"] == 2
    assert rep["electrodes_total"] == 3


def test_dc_values_and_geometry_untouched():
    t = _target()
    dc_before = [e.dc for e in t.geometry.electrodes]
    names_before = [e.name for e in t.geometry.electrodes]
    shapes_before = [len(e.shapes) for e in t.geometry.electrodes]
    load_drive_template(t, _source())
    assert [e.dc for e in t.geometry.electrodes] == dc_before
    assert [e.name for e in t.geometry.electrodes] == names_before
    assert [len(e.shapes) for e in t.geometry.electrodes] == shapes_before


def test_integration_physics_source_untouched():
    t = _target()
    before = (t.integration.dt_ns, t.integration.t_max_us,
              t.source.n_ions, t.source.mz_list[0],
              t.collisions.enabled)
    load_drive_template(t, _source())
    after = (t.integration.dt_ns, t.integration.t_max_us,
             t.source.n_ions, t.source.mz_list[0],
             t.collisions.enabled)
    assert before == after


def test_no_aliasing_to_source():
    t = _target()
    s = _source()
    load_drive_template(t, s)
    t.geometry.rf_groups[0].amplitude_v = 999.0
    # mutating the loaded group must NOT reach back into the source
    assert s.geometry.rf_groups[0].amplitude_v != 999.0
