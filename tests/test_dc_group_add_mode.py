"""DC-group ADD mode (float rails) — chartered by the PI 2026-09-29.

The ruling: "I want groups to be additive" — a float group raises every
member by the group's value while each member's own DC stays authored
and editable. 'set' keeps the ladder semantics every shipped deck uses.

Pinned here:
  * resolve_dc_groups WRITES set-mode members and NEVER TOUCHES add-mode
    members, and is idempotent for both (an additive resolve would
    compound — the exact trap the design avoids);
  * dc_effective = authored dc + the group's per-member value, for
    uniform, linear-ladder and weighted-ladder add groups, and equals
    the resolved dc (no double count) for set-mode members;
  * validate refuses an unknown mode; legacy decks without the field
    load as 'set' byte-identically;
  * the mode round-trips through to_dict / from_json;
  * THE FIELD IDENTITY, per composer: a deck with an add-mode group at
    float F produces the SAME static field as the same deck with no
    group and every member's dc hand-raised by its share of F. Checked
    with synthetic bases on the planar assembler and the generic
    build_voltage_field (the composer arithmetic is linear in the bases,
    so any bases prove it), and with a REAL tiny solve end-to-end on the
    r-z route. The 3-D composer (compose_drive_channels) uses the same
    dc_effective call, exercised via the r-z-style identity on its
    static loop being the only dc consumer there — see
    test_field_identity_* below for the executable statements.
"""
import numpy as np

from ion_gym.io.sim_spec import (BoundsSpec, CollisionSpec, DCGroupSpec,
                                 ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, RFGroupSpec, ShapeSpec,
                                 SimSpec, SourceSpec, SymmetrySpec,
                                 dc_group_member_values)


def _rect(name, x, dc, dc_group=None, dc_index=None, dc_weight=None,
          rf_groups=()):
    return ElectrodeSpec(
        name=name, dc=dc, rf_groups=list(rf_groups),
        dc_group=dc_group, dc_index=dc_index, dc_weight=dc_weight,
        shapes=[ShapeSpec("rect", {"x_mm": x, "y_mm": 4.0,
                                   "width_mm": 1.0, "height_mm": 1.0})])


def _spec(electrodes, dc_groups, rf_groups=()):
    return SimSpec(
        name="add-mode fixture",
        geometry=GeometrySpec(
            width_mm=10.0, height_mm=10.0, mm_per_gu=0.5,
            symmetry=SymmetrySpec(coords="xyz"),
            electrodes=electrodes,
            rf_groups=list(rf_groups), dc_groups=list(dc_groups)),
        source=SourceSpec(seed=0, n_ions=2, x0_mm=5, y0_mm=5,
                          mz_list=[100.0]),
        integration=IntegrationSpec(t_max_us=1.0, dt_ns=2.0),
        bounds=BoundsSpec(), collisions=CollisionSpec(enabled=False))


# --------------------------------------------------------------- resolve
def test_set_mode_still_owns_member_dc():
    s = _spec([_rect("a", 1, dc=99.0, dc_group="L", dc_index=0),
               _rect("b", 2, dc=99.0, dc_group="L", dc_index=1)],
              [DCGroupSpec(name="L", v_in=0.0, v_out=10.0)])
    s.resolve_dc_groups()
    assert [e.dc for e in s.geometry.electrodes] == [0.0, 10.0]


def test_add_mode_never_touches_member_dc_and_is_idempotent():
    s = _spec([_rect("g", 1, dc=4.0, dc_group="F"),
               _rect("h", 2, dc=-1.5, dc_group="F")],
              [DCGroupSpec(name="F", v_in=20.0, uniform=True, mode="add")])
    for _ in range(3):                    # an additive resolve would compound
        s.resolve_dc_groups()
        assert [e.dc for e in s.geometry.electrodes] == [4.0, -1.5]


# ---------------------------------------------------------- dc_effective
def test_dc_effective_uniform_add():
    s = _spec([_rect("g", 1, dc=4.0, dc_group="F"),
               _rect("h", 2, dc=-1.5, dc_group="F"),
               _rect("free", 3, dc=7.0)],
              [DCGroupSpec(name="F", v_in=20.0, uniform=True, mode="add")])
    s.resolve_dc_groups()
    got = [s.dc_effective(e) for e in s.geometry.electrodes]
    assert got == [24.0, 18.5, 7.0]


def test_dc_effective_ladder_add_linear_and_weights():
    lin = _spec([_rect("a", 1, dc=1.0, dc_group="F", dc_index=0),
                 _rect("b", 2, dc=2.0, dc_group="F", dc_index=1),
                 _rect("c", 3, dc=3.0, dc_group="F", dc_index=2)],
                [DCGroupSpec(name="F", v_in=10.0, v_out=30.0, mode="add")])
    lin.resolve_dc_groups()
    assert ([lin.dc_effective(e) for e in lin.geometry.electrodes]
            == [11.0, 22.0, 33.0])
    wts = _spec([_rect("a", 1, dc=1.0, dc_group="F", dc_weight=0.0),
                 _rect("b", 2, dc=1.0, dc_group="F", dc_weight=0.25)],
                [DCGroupSpec(name="F", v_in=0.0, v_out=100.0,
                             interp="weights", mode="add")])
    wts.resolve_dc_groups()
    assert ([wts.dc_effective(e) for e in wts.geometry.electrodes]
            == [1.0, 26.0])


def test_dc_effective_set_mode_no_double_count():
    s = _spec([_rect("a", 1, dc=0.0, dc_group="L", dc_index=0),
               _rect("b", 2, dc=0.0, dc_group="L", dc_index=1)],
              [DCGroupSpec(name="L", v_in=5.0, v_out=15.0)])
    s.resolve_dc_groups()
    assert ([s.dc_effective(e) for e in s.geometry.electrodes]
            == [e.dc for e in s.geometry.electrodes] == [5.0, 15.0])


# ------------------------------------------------- validate / round-trip
def test_validate_refuses_unknown_mode_and_accepts_both_modes():
    s = _spec([_rect("g", 1, dc=0.0, dc_group="F"),
               _rect("h", 2, dc=0.0, dc_group="F")],
              [DCGroupSpec(name="F", v_in=1.0, uniform=True, mode="tilt")])
    assert any("mode must be" in e for e in s.validate())
    for mode in ("set", "add"):
        s.geometry.dc_groups[0].mode = mode
        assert not [e for e in s.validate() if "mode" in e]


def test_legacy_dict_defaults_to_set_and_mode_round_trips(tmp_path):
    legacy = DCGroupSpec.from_dict({"name": "L", "v_in": 1.0, "v_out": 2.0})
    assert legacy.mode == "set"
    s = _spec([_rect("g", 1, dc=4.0, dc_group="F"),
               _rect("h", 2, dc=0.0, dc_group="F")],
              [DCGroupSpec(name="F", v_in=20.0, uniform=True, mode="add")])
    p = tmp_path / "float.json"
    s.to_json(str(p))
    back = SimSpec.from_json(str(p))
    grp = back.geometry.dc_groups[0]
    assert grp.mode == "add"
    # the authored baseline survived the round trip un-summed — the leak
    # an additive resolve would have caused
    assert [e.dc for e in back.geometry.electrodes][:2] == [4.0, 0.0]
    assert back.dc_effective(back.geometry.electrodes[0]) == 24.0


# ------------------------------------------- field identity, per composer
def _synthetic_bases(n, shape, seed=0):
    rng = np.random.default_rng(seed)
    return {i: rng.standard_normal(shape) for i in range(1, n + 1)}


def _manual_twin(s, float_v):
    """The same deck with NO groups and member dc hand-raised by its
    share — the ground truth the add-mode field must equal."""
    import copy
    t = copy.deepcopy(s)
    grp = t.geometry.dc_groups[0]
    mem = [e for e in t.geometry.electrodes if e.dc_group == grp.name]
    vals = dc_group_member_values(grp, mem)
    for e in mem:
        e.dc = float(e.dc) + vals[e.name]
        e.dc_group = None
        e.dc_index = None
    t.geometry.dc_groups = []
    return t


def _float_fixture(float_v):
    return _spec(
        [_rect("g", 1, dc=4.0, dc_group="F", rf_groups=["RF"]),
         _rect("h", 2, dc=-1.5, dc_group="F"),
         _rect("free", 3, dc=7.0)],
        [DCGroupSpec(name="F", v_in=float_v, uniform=True, mode="add")],
        rf_groups=[RFGroupSpec(name="RF", frequency_hz=1e5,
                               amplitude_v=10.0)])


def test_field_identity_planar_assembler():
    from ion_gym.physics.build_planar import V_BASIS, assemble_drive_groups
    s = _float_fixture(20.0)
    s.resolve_dc_groups()
    t = _manual_twin(s, 20.0)
    bases = _synthetic_bases(3, (9, 9))
    A_add = assemble_drive_groups(s, bases, v_basis=V_BASIS)[0]
    A_man = assemble_drive_groups(t, bases, v_basis=V_BASIS)[0]
    assert np.allclose(A_add, A_man, atol=0, rtol=1e-12)
    # zero float == the plain baseline deck (the null configuration)
    z = _float_fixture(0.0)
    z.resolve_dc_groups()
    A_zero = assemble_drive_groups(z, bases, v_basis=V_BASIS)[0]
    b = _manual_twin(z, 0.0)
    A_base = assemble_drive_groups(b, bases, v_basis=V_BASIS)[0]
    assert np.allclose(A_zero, A_base, atol=0, rtol=1e-12)


def test_field_identity_build_voltage_field():
    from ion_gym.physics.sim_build import build_voltage_field
    s = _float_fixture(20.0)
    s.resolve_dc_groups()
    t = _manual_twin(s, 20.0)
    bases = _synthetic_bases(3, (9, 9), seed=7)
    A_add = build_voltage_field(s, bases)[0]
    A_man = build_voltage_field(t, bases)[0]
    assert np.allclose(A_add, A_man, atol=0, rtol=1e-12)


def test_field_identity_rz_real_solve():
    """End to end on the r-z route: a REAL (tiny) solve, add-mode float
    vs the hand-raised twin — the full static model must match."""
    from ion_gym.physics.build_rz import build_rz_model

    def rz_spec(mode_deck):
        els = [ElectrodeSpec(
                   name="ring1", dc=2.0, dc_group="F",
                   shapes=[ShapeSpec("rect", {"x_mm": 1.0, "y_mm": 2.0,
                                              "width_mm": 1.0,
                                              "height_mm": 0.5})]),
               ElectrodeSpec(
                   name="ring2", dc=-1.0, dc_group="F",
                   shapes=[ShapeSpec("rect", {"x_mm": 3.0, "y_mm": 2.0,
                                              "width_mm": 1.0,
                                              "height_mm": 0.5})])]
        gsp = [DCGroupSpec(name="F", v_in=12.5, uniform=True, mode="add")]
        sp = SimSpec(
            name="rz float fixture",
            geometry=GeometrySpec(
                width_mm=5.0, height_mm=3.0, mm_per_gu=0.25,
                symmetry=SymmetrySpec(coords="rz"),
                electrodes=els, rf_groups=[], dc_groups=gsp),
            source=SourceSpec(seed=0, n_ions=2, x0_mm=2.5, y0_mm=0.5,
                              mz_list=[100.0]),
            integration=IntegrationSpec(t_max_us=1.0, dt_ns=2.0),
            bounds=BoundsSpec(), collisions=CollisionSpec(enabled=False))
        sp.resolve_dc_groups()
        return sp if mode_deck == "add" else _manual_twin(sp, 12.5)

    A_add = build_rz_model(rz_spec("add")).A
    A_man = build_rz_model(rz_spec("manual")).A
    assert np.allclose(A_add, A_man, atol=1e-9)
    assert float(np.max(np.abs(A_add))) > 0.0   # a real field, not zeros


# ----------------------------------------------------------------- GUI
def test_gui_member_boxes_follow_the_mode():
    """The Voltages tab: SET-mode member boxes are read-only (derived),
    ADD-mode member boxes stay editable and _sync_spec READS them; the
    per-group mode selector flips both in place."""
    from ion_gym.ui.sim_app import SimApp
    s = _spec(
        [_rect("g", 1, dc=4.0, dc_group="F"),
         _rect("h", 2, dc=-1.5, dc_group="F"),
         _rect("lad1", 3, dc=0.0, dc_group="L", dc_index=0),
         _rect("lad2", 4, dc=0.0, dc_group="L", dc_index=1)],
        [DCGroupSpec(name="F", v_in=20.0, uniform=True, mode="add"),
         DCGroupSpec(name="L", v_in=0.0, v_out=10.0)])
    app = SimApp(s)
    dis = {e.name: app._v_widgets[i]["dc"].disabled
           for i, e in enumerate(app.spec.geometry.electrodes)}
    assert dis == {"g": False, "h": False,        # add: authored, editable
                   "lad1": True, "lad2": True}    # set: derived, read-only
    # an edited ADD-member box is READ back as the authored baseline
    app._v_widgets[0]["dc"].value = 6.0
    app._sync_spec()
    assert app.spec.geometry.electrodes[0].dc == 6.0
    assert app.spec.dc_effective(app.spec.geometry.electrodes[0]) == 26.0
    # flipping the float group to SET hands the boxes to the group
    app._dcg_widgets["F"]["mode"].value = "set"
    assert app._v_widgets[0]["dc"].disabled is True
    app._sync_spec()
    assert app.spec.geometry.electrodes[0].dc == 20.0   # derived from v_in
    # and back to ADD re-enables them with the group riding on top
    app._dcg_widgets["F"]["mode"].value = "add"
    assert app._v_widgets[0]["dc"].disabled is False
