"""
A random-mode run (source.seed null) is reproducible from the ONE seed it
prints, on every route and gas model.

On the 3-D route a random-mode run once drew two seeds -- births from one,
collision kernels from the other -- and printed both as the recipe, and
pinning either reproduced neither. Separately, a second build of the same
spec object repeated the first run (the drawn seed stayed cached), so
pooled runs were duplicates.
"""
import numpy as np
import pytest

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 4.0)]


def _rect(x, y, w, h):
    return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                              "height_mm": h})


def _deck(coords, depth_mm, model, seed=None):
    geo = GeometrySpec(
        width_mm=10.0, height_mm=6.0, depth_mm=depth_mm, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[ElectrodeSpec(name="top", dc=5.0,
                                  shapes=[_rect(0, 5, 10, 1)]),
                    ElectrodeSpec(name="bot", dc=0.0,
                                  shapes=[_rect(0, 0, 10, 1)])])
    src = SourceSpec(distribution="disc", n_ions=3, x0_mm=3.0,
                     y0_mm=2.0 if coords == "rz" else 3.0,
                     z0_mm=depth_mm / 2, r_mm=0.5, axis="x",
                     direction=[1.0, 0.0, 0.0], ke_lo=0.5, ke_hi=1.5,
                     temperature_k=300.0, mz_list=[200.0],
                     tob_span_us=0.0, seed=seed)
    col = CollisionSpec(enabled=True, gas="N2", T_k=300.0, P_torr=1.0,
                        model=model)
    return SimSpec(geometry=geo, name="one_seed", source=src, collisions=col,
                   integration=IntegrationSpec(dt_ns=2.0, t_max_us=1.0,
                                               rec_every=5))


def _run(spec):
    from ion_gym.physics.sim_build import build_run
    _m, fly, _c, births = build_run(spec)
    return np.asarray(births), [np.asarray(fly(i)[0])
                                for i in range(spec.source.n_ions)]


def _same(a, b):
    births_a, trajs_a = a
    births_b, trajs_b = b
    return (np.array_equal(births_a, births_b)
            and all(x.shape == y.shape and np.array_equal(x, y)
                    for x, y in zip(trajs_a, trajs_b)))


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))


@pytest.mark.parametrize("model", ["hs", "sds"])
@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_the_one_printed_seed_reproduces_the_run(route, coords, depth, model,
                                                 capsys):
    spec = _deck(coords, depth, model)
    first = _run(spec)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "random mode: drew" in ln]
    assert len(lines) == 1, out
    assert "policy RANDOM" not in out          # no second, independent draw
    drawn = spec.source._drawn_seed
    assert f"source.seed: {drawn}" in lines[0]
    pinned = _deck(coords, depth, model, seed=drawn)
    assert _same(first, _run(pinned))


@pytest.mark.parametrize("route,coords,depth", ROUTES,
                         ids=[r[0] for r in ROUTES])
def test_each_build_of_a_random_spec_is_a_fresh_reproducible_run(route,
                                                                 coords,
                                                                 depth):
    from ion_gym.physics.sim_build import build_run
    spec = _deck(coords, depth, "hs")
    _m1, fly1, _c1, births1 = build_run(spec)
    seed1 = spec.source._drawn_seed
    _m2, fly2, _c2, births2 = build_run(spec)    # built before fly1 is flown
    seed2 = spec.source._drawn_seed
    assert seed1 != seed2
    assert not np.array_equal(births1, births2)
    run1 = (np.asarray(births1), [np.asarray(fly1(i)[0]) for i in range(3)])
    assert _same(run1, _run(_deck(coords, depth, "hs", seed=seed1)))
    run2 = (np.asarray(births2), [np.asarray(fly2(i)[0]) for i in range(3)])
    assert _same(run2, _run(_deck(coords, depth, "hs", seed=seed2)))
