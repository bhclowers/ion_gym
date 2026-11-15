"""
Save solved field, Load field and Open saved field: one process on every
route (planar, r-z, 3-D).

Save solved field read only the planar/r-z cache (io.basis_cache). The
3-D routes keep their solved field in io.fa_cache under a key of their
own, so on a solved 3-D SLIM deck the button said "nothing to save". The
three field buttons also behaved differently by route: 2-D saves and
loads ran on the interface thread, and nothing stopped a second press
from starting a second job on the same file.

Now, on every route, each job runs in the background with a progress
line, the three buttons are disabled until it ends (a press while busy
is refused before anything is touched), and every field file is written
atomically. Every route saves the same format, the one notebooks use
(io.field_io.save_field): the cache entry as one file with its deck
beside it, listed in the picker and loaded back only by a deck with
exactly its cache key (geometry, STL contents, solver code). Older
array-level field files are listed with the reason and refused.

Fixtures are tiny decks (a few thousand nodes), one per route.
"""
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from ion_gym.io import basis_cache, fa_cache, field_io
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.sim_build import (build_route, build_run,
                                       solved_3d_cache_key)
from ion_gym.ui import sim_app

ROUTES = ("planar", "rz", "shapes3d")


def _deck(route, width=4.0):
    if route == "rz":
        coords, depth = "rz", 0.0
        shapes = [[{"type": "rect", "x_mm": 0.0, "y_mm": 1.0,
                    "width_mm": 0.5, "height_mm": 2.0}],
                  [{"type": "rect", "x_mm": width - 0.5, "y_mm": 1.0,
                    "width_mm": 0.5, "height_mm": 2.0}]]
    else:
        coords, depth = "xyz", (3.0 if route == "shapes3d" else 0.0)
        shapes = [[{"type": "rect", "x_mm": 0.0, "y_mm": 0.0,
                    "width_mm": 0.5, "height_mm": 3.0}],
                  [{"type": "rect", "x_mm": width - 0.5, "y_mm": 0.0,
                    "width_mm": 0.5, "height_mm": 3.0}]]
    spec = SimSpec.from_dict({
        "name": f"tiny_field_save_{route}",
        "source": {"seed": 0, "distribution": "point", "n_ions": 2,
                   "x0_mm": 2.0, "y0_mm": 0.5, "z0_mm": 1.5,
                   "ke_lo": 1.0, "ke_hi": 1.0, "mz_list": [300.0],
                   "tob_span_us": 0.0},
        "integration": {"dt_ns": 1.0, "t_max_us": 1.0},
        "geometry": {
            "width_mm": width, "height_mm": 3.0, "depth_mm": depth,
            "mm_per_gu": 0.25, "symmetry": {"coords": coords, "planes": {}},
            "electrodes": [
                {"name": "E1", "dc": 5.0, "shapes": shapes[0]},
                {"name": "E2", "dc": -5.0, "shapes": shapes[1]}]},
    })
    assert build_route(spec).builder == route
    return spec


def _cached(spec):
    """Is the deck's solved field in the disk cache (its own cache)?"""
    kd = solved_3d_cache_key(spec)
    if kd is None:
        return basis_cache.load(spec)[0] is not None
    return fa_cache.load(kd)[0] is not None


def _entry_dir(cache, spec):
    kd = solved_3d_cache_key(spec)
    return cache / (basis_cache.key(spec) if kd is None
                    else fa_cache.spec_key(kd))


def _forget_in_process(monkeypatch):
    """Planar and r-z also keep bases in process; a load must be judged
    by the disk cache alone."""
    from ion_gym.physics import build_planar, build_rz
    monkeypatch.setattr(build_planar, "_PLANAR_BASIS_CACHE", {})
    monkeypatch.setattr(build_rz, "_RZ_BASIS_CACHE", {})


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    cache, fields = tmp_path / "cache", tmp_path / "fields"
    monkeypatch.setenv("ION_GYM_CACHE", str(cache))
    monkeypatch.setenv("ION_GYM_FIELDS", str(fields))
    _forget_in_process(monkeypatch)
    return cache, fields


@pytest.fixture(params=ROUTES)
def solved(request, dirs):
    """A deck solved the way the app solves it: through the app's own
    copy (which writes out defaults such as every symmetry plane)."""
    app = sim_app.SimApp(_deck(request.param))
    app._sync_spec()             # the app syncs its deck before it solves
    build_run(app.spec)
    assert _cached(app.spec)
    return request.param, app.spec, app


def _save(app, fields):
    app._on_save_field()                   # headless: the worker is joined
    found = [p for p in sorted(fields.glob("*.npz"))
             if not p.name.startswith(".")]
    assert len(found) == 1, (app.w_runs_msg.object, list(fields.iterdir()))
    return found[0]


def _buttons_free(app):
    return not any(b.disabled for b in app._field_buttons())


# --------------------------------------------------- the same on every route
def test_a_solved_field_is_saved(dirs, solved):
    _, fields = dirs
    route, spec, app = solved
    path = _save(app, fields)
    assert "field saved" in app.w_runs_msg.object
    assert field_io.is_field_bundle(path), "one format on every route"
    assert path.name.endswith(".field.npz")
    assert list(fields.glob("*.spec.json")), "the deck is beside it"
    assert _buttons_free(app)
    bases, labels, meta = field_io.read_field_arrays(str(path))
    assert sorted(bases) == [1, 2]            # one basis per electrode
    assert {int(v) for v in set(labels.ravel().tolist())} <= {0, 1, 2}
    assert all(b.shape == labels.shape for b in bases.values())
    # per volt on every route: each basis is 1 on its own electrode's metal
    # (3-D caches float32, so 1 to float32 precision there)
    for k, b in bases.items():
        assert b.dtype == np.float64
        np.testing.assert_allclose(b[labels == k], 1.0, rtol=1e-6)
    # and the deck's DC values superpose to potentials between them
    phi = sum(el.dc * bases[i] for i, el in
              enumerate(spec.geometry.electrodes, start=1))
    assert -5.0 - 1e-6 <= phi.min() and phi.max() <= 5.0 + 1e-6


def test_an_older_format_file_is_listed_with_its_reason_and_refused(dirs):
    _, fields = dirs
    fields.mkdir(parents=True, exist_ok=True)
    old = fields / "older__0000.npz"
    field_io._atomic_savez(
        str(old), b1=np.zeros((4, 4)), ele=np.zeros((4, 4), np.int16),
        **{field_io._META: field_io._meta_to_member(
            {"kind": "ion_gym_field", "field_format": 1})})
    app = sim_app.SimApp(_deck("planar"))
    app._refresh_field_picker()
    assert any(lbl.startswith("⚠") for lbl in app.w_fieldpick.options)
    app.w_fieldpick.value = str(old)
    app._on_load_field()
    assert "field refused" in app.w_runs_msg.object
    assert "older field format" in app.w_runs_msg.object


def test_the_picker_lists_it_as_matching(dirs, solved):
    _, fields = dirs
    _, spec, app = solved
    _save(app, fields)
    app._refresh_field_picker()
    assert any(lbl.startswith("✓") for lbl in app.w_fieldpick.options), \
        app.w_fieldpick.options


def test_load_field_restores_it_after_the_cache_is_cleared(dirs, solved,
                                                           monkeypatch):
    cache, fields = dirs
    _, spec, app = solved
    path = _save(app, fields)
    shutil.rmtree(_entry_dir(cache, spec))
    _forget_in_process(monkeypatch)
    assert not _cached(spec)
    app.w_fieldpick.value = str(path)
    app._on_load_field()
    assert "field loaded" in app.w_runs_msg.object, app.w_runs_msg.object
    assert _cached(spec)
    assert _buttons_free(app)


def test_a_field_for_other_geometry_is_refused(dirs, solved):
    _, fields = dirs
    route, spec, app0 = solved
    path = _save(app0, fields)
    other = _deck(route, width=5.0)
    app = sim_app.SimApp(other)
    app._refresh_field_picker()
    app.w_fieldpick.value = str(path)
    app._on_load_field()
    assert "field refused" in app.w_runs_msg.object, app.w_runs_msg.object
    assert not _cached(other)
    assert _buttons_free(app)


def test_open_saved_field_brings_back_the_deck_and_its_field(
        dirs, solved, monkeypatch):
    cache, fields = dirs
    route, spec, app0 = solved
    path = _save(app0, fields)
    shutil.rmtree(_entry_dir(cache, spec))
    _forget_in_process(monkeypatch)
    app = sim_app.SimApp(_deck(route, width=5.0))     # another deck loaded
    app._refresh_field_picker()
    app.w_fieldpick.value = str(path)
    app._on_bootstrap_field()
    assert "session opened from file" in app.w_runs_msg.object, \
        (app.w_runs_msg.object, app.status.object)
    assert app.spec.geometry.width_mm == 4.0
    assert _cached(spec)


@pytest.mark.parametrize("route", ROUTES)
def test_an_unsolved_deck_says_to_solve_first(dirs, route):
    app = sim_app.SimApp(_deck(route))
    app._on_save_field()
    assert "nothing to save" in app.w_runs_msg.object
    assert "Solve it first" in app.w_runs_msg.object


# ----------------------------------------------------- one job at a time
def test_the_buttons_are_disabled_while_a_job_runs(dirs, solved,
                                                   monkeypatch):
    _, fields = dirs
    _, spec, app = solved
    during = []

    def fake_off_loop(label, work, expected, progress, done, say=None):
        during.append([b.disabled for b in app._field_buttons()])
        done(work(), None)

    monkeypatch.setattr(app, "_off_loop", fake_off_loop)
    app._on_save_field()
    assert during == [[True, True, True]]
    assert _buttons_free(app)


def test_a_press_while_busy_is_refused_before_anything_is_touched(
        dirs, solved):
    _, fields = dirs
    _, spec, app = solved
    app._field_busy = "save"
    editor = app.w_json.value
    app.w_fieldpick.value = None
    for press in (app._on_save_field, app._on_load_field,
                  app._on_bootstrap_field):
        app.w_runs_msg.object = ""
        press()
        assert "still running" in app.w_runs_msg.object, press.__name__
    assert app.w_json.value == editor
    assert not fields.exists() or not list(fields.glob("*.npz"))


def test_a_failed_write_leaves_no_file_and_frees_the_buttons(
        dirs, solved, monkeypatch):
    _, fields = dirs
    _, spec, app = solved

    def disk_full(fh, **members):
        fh.write(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(np, "savez_compressed", disk_full)
    app._on_save_field()
    assert "field save failed" in app.w_runs_msg.object
    assert "No space left" in app.w_runs_msg.object
    assert not list(fields.glob("*.npz")) and not list(fields.glob(".*"))
    assert _buttons_free(app)


def test_the_trajectory_button_greys_while_it_saves(dirs, monkeypatch):
    app = sim_app.SimApp(_deck("planar"))
    app._runs = {"r": SimpleNamespace(results=[], provenance=None)}
    app._active = "r"
    during = []

    def fake_off_loop(label, work, expected, progress, done, say=None):
        during.append(app.w_savetraj.disabled)
        done(None, OSError("disk full"))

    monkeypatch.setattr(app, "_off_loop", fake_off_loop)
    app._on_save_traj()
    assert during == [True]
    assert not app.w_savetraj.disabled
