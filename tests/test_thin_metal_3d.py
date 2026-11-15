"""
3-D metal thinner than two nodes, and is_grid transparency on 3-D.

The 3-D hit test counts a point as in metal only inside a cell whose eight
corners are all metal, so a plate one node thick never stopped an ion: the
review flew one straight through a 225-node plate that planar and r-z stop
(C-05). Such an electrode is now refused by name; an electrode with more
than a few percent of its metal invisible to the hit test builds with a
warning. is_grid electrodes are transparent to the flight on every route
(they were opaque walls on 3-D only).
"""
import numpy as np
import pytest

from ion_gym.io.sim_spec import (ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(shapes, depth_mm=10.0, is_grid=False):
    """A grounded plate across the cell at x = 10 mm; an ion launched along
    +x from x = 3 mm in vacuum, through no field."""
    geo = GeometrySpec(
        width_mm=20.0, height_mm=10.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords="xyz"),
        electrodes=[ElectrodeSpec(name="plate", dc=0.0, is_grid=is_grid,
                                  shapes=list(shapes))])
    src = SourceSpec(distribution="point", n_ions=1, x0_mm=3.0, y0_mm=5.0,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0],
                     ke_lo=5.0, ke_hi=5.0, mz_list=[100.0], tob_span_us=0.0,
                     seed=1)
    return SimSpec(geometry=geo, name="thin_metal", source=src,
                   integration=IntegrationSpec(dt_ns=2.0, t_max_us=6.0,
                                               rec_every=10))


def _end_x(spec):
    from ion_gym.physics.sim_build import build_run
    _m, fly, _c, _b = build_run(spec)
    _traj, summary = fly(0)
    return float(summary["x_end"])


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


ONE_LAYER = [_rect(10.0, 0.0, 0.25, 10.0)]     # nodes at x = 10 only
TWO_LAYERS = [_rect(10.0, 0.0, 0.5, 10.0)]     # x = 10 and 10.5


def test_a_one_node_plate_is_refused_on_3d():
    from ion_gym.physics.sim_build import build_run
    with pytest.raises(ValueError, match="ions fly straight through it"):
        build_run(_deck(ONE_LAYER))


def test_a_two_node_plate_stops_the_ion_on_3d():
    assert _end_x(_deck(TWO_LAYERS)) == pytest.approx(10.0, abs=0.5)


@pytest.mark.parametrize("depth", [10.0, 0.0], ids=["3d", "planar"])
def test_an_is_grid_plate_is_transparent_on_every_route(depth):
    for shapes in (ONE_LAYER, TWO_LAYERS):
        assert _end_x(_deck(shapes, depth_mm=depth, is_grid=True)) > 12.0


def test_metal_partly_too_thin_builds_and_warns(capsys):
    spec = _deck([_rect(2.0, 0.0, 1.0, 10.0), _rect(15.0, 0.0, 0.25, 10.0)])
    spec.source.x0_mm = 8.0
    end = _end_x(spec)
    out = capsys.readouterr().out
    assert "lie in no all-metal cell" in out and "'plate'" in out
    assert end > 15.0                     # the one-node fin does not stop it


def test_the_rule_helper():
    from ion_gym.physics.tracer3d import nodes_the_hit_rule_sees
    m = np.zeros((6, 6, 6), bool)
    m[2, :, :] = True                     # one layer: nothing seen
    assert not nodes_the_hit_rule_sees(m).any()
    m[3, :, :] = True                     # two layers: all of it seen
    assert np.array_equal(nodes_the_hit_rule_sees(m), m)
