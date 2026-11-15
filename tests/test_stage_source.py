"""
A staged assembly takes its ions from the beam block, or from the one
stage named in beam.from_stage. No other stage's source is flown; only its
charge is read. So a SimSpec stage that is not the birth stage may omit its
source: it inherits the beam's m/z and charge, announced, on every route.
The birth stage still needs its own source, and a charge must be stated
somewhere (the beam or a source-declaring stage).
"""
import json

import pytest

from ion_gym.io.sim_spec import (ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)
from ion_gym.physics.staged_flight import load_assembly, stage_spec_dicts

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 2.0)]


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _stage(coords="xyz", depth_mm=0.0, charge=1):
    geo = GeometrySpec(
        width_mm=10.0, height_mm=6.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[ElectrodeSpec(name="top", dc=5.0,
                                  shapes=[_rect(0, 5, 10, 1)]),
                    ElectrodeSpec(name="bot", dc=0.0,
                                  shapes=[_rect(0, 0, 10, 1)])])
    src = SourceSpec(distribution="point", n_ions=1, x0_mm=1.0, y0_mm=3.0,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0],
                     ke_lo=1.0, ke_hi=1.0, mz_list=[322.0], charge=charge,
                     tob_span_us=0.0, seed=1)
    return SimSpec(geometry=geo, name=f"stage_{coords}", source=src,
                   integration=IntegrationSpec(dt_ns=2.0, t_max_us=1.0,
                                               rec_every=10)).to_dict()


def _doc(second, beam_charge=None, from_stage=None, first_source=True):
    first = _stage()
    if not first_source:
        first.pop("source")
    beam = {"mz": 322.0}
    if from_stage:
        beam["from_stage"] = from_stage
    else:
        beam.update({"p0_mm": [1.0, 3.0, 0.0], "v0_mm_us": [5.0, 0.0, 0.0],
                     "tob_us": 0.0})
    if beam_charge is not None:
        beam["charge"] = beam_charge
    return {"schema": "ion_gym.assembly/1", "name": "inherit",
            "stages": [
                {"name": "a", "spec": first,
                 "exit": {"axis": "x", "value_mm": 9.0, "sign": 1}},
                {"name": "b", "spec": second,
                 "pose": {"offset_mm": [20.0, 0.0, 0.0]}, "exit": None}],
            "beam": beam}


def _sourceless(coords="xyz", depth_mm=0.0):
    d = _stage(coords, depth_mm)
    d.pop("source")
    return d


def test_a_sourceless_stage_inherits_the_beam_and_says_so(capsys):
    pairs = stage_spec_dicts(_doc(_sourceless(), beam_charge=2))
    src = pairs[1][1]["source"]
    assert src["mz_list"] == [322.0] and src["charge"] == 2
    assert "source" not in _sourceless()          # the input is not edited
    out = capsys.readouterr().out
    assert "[ASSEMBLY] stage 'b' declares no source" in out
    assert "beam.charge" in out
    # the declaring stage is passed through as written
    assert pairs[0][1]["source"]["charge"] == 1


def test_the_charge_comes_from_a_declaring_stage_without_beam_charge(capsys):
    pairs = stage_spec_dicts(_doc(_sourceless()))
    assert pairs[1][1]["source"]["charge"] == 1
    assert "stage 'a'" in capsys.readouterr().out


def test_no_charge_anywhere_is_refused():
    with pytest.raises(ValueError, match="beam.*charge"):
        stage_spec_dicts(_doc(_sourceless(), first_source=False))


def test_the_birth_stage_must_have_a_source():
    with pytest.raises(ValueError, match="birth stage"):
        stage_spec_dicts(_doc(_sourceless(), beam_charge=1,
                              from_stage="a", first_source=False))


def test_a_non_simspec_stage_is_left_to_its_converter():
    scene = {"grid": {}, "electrodes": [], "units": "mm"}
    pairs = stage_spec_dicts(_doc(scene, beam_charge=1))
    assert pairs[1][1] is scene and "source" not in scene


@pytest.fixture
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path / "cache"))


@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_load_assembly_builds_a_sourceless_stage_on_every_route(
        route, coords, depth, tmp_path, _isolated_cache):
    path = tmp_path / "instrument.json"
    path.write_text(json.dumps(_doc(_sourceless(coords, depth))))
    regions, beam = load_assembly(str(path))
    assert [r.name for r in regions] == ["a", "b"]
    assert regions[1].charge == regions[0].charge == 1
    assert beam["mz"] == 322.0


def test_a_disagreeing_beam_charge_is_still_refused(tmp_path,
                                                    _isolated_cache):
    path = tmp_path / "instrument.json"
    path.write_text(json.dumps(_doc(_sourceless(), beam_charge=2)))
    with pytest.raises(ValueError, match="different source.charge"):
        load_assembly(str(path))
