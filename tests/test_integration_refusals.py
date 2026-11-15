"""
validate() refuses an integration that cannot describe a flight, by
field name, on every route.

Before: a t_max_us <= 0 flew a zero-length flight on r-z and 3-D (every
ion reported as a timeout) and failed planar with "negative dimensions
are not allowed"; a dt_ns <= 0 was refused only at sizing (record_volume)
without the deck key; a rec_every of 0 or 2.5 was silently read as 1 or
truncated.
"""
import pytest

from ion_gym.io.sim_spec import (ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 4.0)]
BAD = [("dt_ns", 0.0), ("dt_ns", -1.0), ("t_max_us", 0.0),
       ("t_max_us", -5.0), ("rec_every", 0), ("rec_every", 2.5),
       ("rec_every", True)]


def _deck(coords, depth, **integ):
    geo = GeometrySpec(width_mm=10.0, height_mm=6.0, depth_mm=depth,
                       mm_per_gu=0.5, symmetry=SymmetrySpec(coords=coords),
                       electrodes=[ElectrodeSpec(name="p", dc=5.0, shapes=[
                           ShapeSpec("rect", {"x_mm": 0, "y_mm": 5,
                                              "width_mm": 10,
                                              "height_mm": 1})])])
    src = SourceSpec(distribution="point", n_ions=1, x0_mm=5.0, y0_mm=3.0,
                     z0_mm=depth / 2, direction=[1.0, 0.0, 0.0], ke_lo=1.0,
                     ke_hi=1.0, mz_list=[100.0], tob_span_us=0.0, seed=1)
    args = dict(dt_ns=1.0, t_max_us=1.0, rec_every=1)
    args.update(integ)
    return SimSpec(geometry=geo, source=src,
                   integration=IntegrationSpec(**args))


@pytest.mark.parametrize("key,value", BAD,
                         ids=[f"{k}={v}" for k, v in BAD])
def test_validate_names_the_field(key, value):
    errs = _deck("xyz", 0.0, **{key: value}).validate()
    assert any(e.startswith(f"integration.{key} = ") for e in errs), errs


@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
@pytest.mark.parametrize("key,value", [("t_max_us", 0.0),
                                       ("t_max_us", -5.0),
                                       ("dt_ns", 0.0)],
                         ids=["t_max_0", "t_max_neg", "dt_0"])
def test_build_run_refuses_on_every_route(route, coords, depth, key, value,
                                          tmp_path, monkeypatch):
    from ion_gym.physics.sim_build import build_run
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))
    with pytest.raises(ValueError, match=f"integration.{key}"):
        build_run(_deck(coords, depth, **{key: value}))


def test_a_null_step_is_still_derived():
    errs = _deck("xyz", 0.0, dt_ns=None).validate()
    assert not [e for e in errs if "dt_ns" in e]
