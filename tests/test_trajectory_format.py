"""
One trajectory file format, for the dashboard and the notebooks alike.

Save trajectories wrote one file (float64, members t<i>, a _meta frame)
and Download NPZ built another (float32, members ion_<i>, no frame, the
mass from the run's summary). Load as run read only the first; notebook
05 carried its own reader for both. Now every writer (Save trajectories,
Download NPZ, notebooks) goes through field_io.save_trajectories /
trajectories_npz, which write the same members, and every reader
through field_io.load_trajectories. A precision option chooses float64
(default, exactly what was flown) or float32 (about half the size); a
float32 file keeps each ion's time column at float64, and the reader
puts it back.
"""
import io

import numpy as np
import pytest

from ion_gym.io import field_io
from ion_gym.io.sim_spec import SimSpec

COLS = ["t", "x", "y", "vx"]


def _spec():
    return SimSpec.from_dict({
        "name": "traj_format",
        "source": {"seed": 0, "distribution": "point", "n_ions": 3,
                   "x0_mm": 2.0, "y0_mm": 1.5, "z0_mm": 0.0, "ke_lo": 1.0,
                   "ke_hi": 1.0, "mz_list": [300.0], "tob_span_us": 0.0},
        "integration": {"dt_ns": 1.0, "t_max_us": 1.0},
        "geometry": {"width_mm": 4.0, "height_mm": 3.0, "depth_mm": 0.0,
                     "mm_per_gu": 0.25,
                     "symmetry": {"coords": "xyz", "planes": {}},
                     "electrodes": [{"name": "E1", "dc": 5.0, "shapes": [
                         {"type": "rect", "x_mm": 0.0, "y_mm": 0.0,
                          "width_mm": 0.5, "height_mm": 3.0}]}]},
    })


def _results():
    """Three ions; late times where float32 cannot resolve a 1 ns step."""
    rng = np.random.default_rng(1)
    out = []
    for i in range(3):
        n = 50
        t = 9000.0 + np.arange(n) * 1e-3          # us, 1 ns apart
        traj = np.column_stack([t, rng.random(n) * 10, rng.random(n) * 3,
                                rng.random(n)])
        out.append((i, traj, {"kind": 0, "tof": float(t[-1]),
                              "mz": 300.0 + i}))
    return out


def _load(raw):
    return field_io.load_trajectories(io.BytesIO(raw))


# --------------------------------------------- one file for every writer
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_save_and_download_write_the_same_file(tmp_path, dtype):
    spec, res = _spec(), _results()
    prov = {"seed": 7, "seed_mode": "declared"}
    p = field_io.save_trajectories(res, spec, path=str(tmp_path / "a.npz"),
                                   provenance=prov, dtype=dtype,
                                   columns=COLS)
    raw = field_io.trajectories_npz(res, spec, provenance=prov,
                                    dtype=dtype, columns=COLS)
    with np.load(p) as a, np.load(io.BytesIO(raw)) as b:
        assert sorted(a.files) == sorted(b.files)
        for k in a.files:
            nan_ok = a[k].dtype.kind == "f"       # absent values are NaN
            assert np.array_equal(a[k], b[k], equal_nan=nan_ok), k
    trajs, meta, _ = _load(raw)
    assert meta["precision"] == dtype and meta["columns"] == COLS
    assert meta["provenance"] == prov


def test_float64_is_exactly_what_was_flown(tmp_path):
    res = _results()
    raw = field_io.trajectories_npz(res, _spec(), columns=COLS)
    trajs, _, _ = _load(raw)
    for i, traj, _ in res:
        assert trajs[i].dtype == np.float64
        assert np.array_equal(trajs[i], traj)


def test_float32_is_smaller_and_keeps_time_at_float64():
    res = _results()
    raw64 = field_io.trajectories_npz(res, _spec(), columns=COLS)
    raw32 = field_io.trajectories_npz(res, _spec(), columns=COLS,
                                      dtype="float32")
    assert len(raw32) < len(raw64)
    with np.load(io.BytesIO(raw32)) as z:
        assert z["t0"].dtype == np.float32
        assert z["time0"].dtype == np.float64
    trajs, _, _ = _load(raw32)
    for i, traj, _ in res:
        got = trajs[i]
        assert got.dtype == np.float64
        assert np.array_equal(got[:, 0], traj[:, 0]), "time exact"
        # float32 cannot tell consecutive 1 ns stamps at 9000 us apart
        # within its own precision; the restored column does
        assert np.all(np.diff(got[:, 0]) > 0)
        np.testing.assert_allclose(got[:, 1:], traj[:, 1:], rtol=1e-6)


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="precision"):
        field_io.trajectories_npz(_results(), _spec(), columns=COLS,
                                  dtype="float16")


def test_columns_that_do_not_fit_the_trajectories_are_refused():
    with pytest.raises(ValueError, match="does not match the 3 columns"):
        field_io.trajectories_npz(_results(), _spec(),
                                  columns=["t", "x", "y"])


def test_the_mass_comes_from_the_run_first_then_the_deck():
    res = _results()
    res[2] = (2, res[2][1], {"kind": 0})            # no mass in the summary
    _, meta, _ = _load(field_io.trajectories_npz(res, _spec(),
                                                 columns=COLS))
    assert meta["mz_da"] == [300.0, 301.0, 300.0]
    assert meta["mz_source"] == ["summary", "summary", "spec"]


# ------------------------------------------------------------- reading
def test_an_older_download_is_refused_with_the_remedy():
    buf = io.BytesIO()
    np.savez_compressed(buf, ion_0000=np.zeros((2, 4), np.float32),
                        columns=np.array(COLS))
    with pytest.raises(field_io.TrajectoryFormatError,
                       match="Download the run again"):
        field_io.load_trajectories(io.BytesIO(buf.getvalue()))


@pytest.fixture
def flown_app(tmp_path, monkeypatch):
    """The dashboard with one real (tiny) run displayed."""
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ION_GYM_TRAJECTORIES", str(tmp_path / "traj"))
    from ion_gym.physics.ensemble_driver import (EnsembleProgress, IonResult,
                                                 fly_all)
    from ion_gym.physics.sim_build import build_run
    from ion_gym.ui.sim_app import SimApp
    app = SimApp(_spec())
    app._sync_spec()
    model, fly, cols, births = build_run(app.spec)
    results = [IonResult(index=i, traj=tr, summary=s) for i, (tr, s)
               in enumerate(fly_all(len(births), fly, n_workers=1))]
    app._model, app._cols = model, cols      # as the app's own build sets
    app._runs["r"] = EnsembleProgress(done=len(results), total=len(results),
                                      elapsed_s=0.0, n_terminated=0,
                                      results=results)
    app._active = "r"
    return app, tmp_path


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_app_saves_and_downloads_the_same_file(flown_app, dtype):
    app, tmp = flown_app
    app.w_traj_dtype.value = dtype
    app._on_prepare_download("npz")          # headless: built and joined
    raw = app._dl["npz"]["data"]
    assert raw is not None, app.w_runs_msg.object
    app._on_save_traj()
    saved = list((tmp / "traj").glob("*.traj.npz"))
    assert len(saved) == 1, app.w_runs_msg.object
    a, ma, _ = field_io.load_trajectories(str(saved[0]))
    b, mb, _ = _load(raw)
    assert ma["precision"] == mb["precision"] == dtype
    assert sorted(a) == sorted(b)
    for i in a:
        assert np.array_equal(a[i], b[i])


def test_load_as_run_opens_a_downloaded_file(flown_app):
    app, tmp = flown_app
    app._on_prepare_download("npz")
    dl = tmp / "traj" / "downloaded.traj.npz"
    dl.parent.mkdir(parents=True, exist_ok=True)
    dl.write_bytes(app._dl["npz"]["data"])
    app._refresh_field_picker()
    app.w_trajpick.value = str(dl)
    n_runs = len(app._runs)
    app._on_load_traj()
    assert len(app._runs) == n_runs + 1, app.status.object


def test_changing_the_precision_drops_a_prepared_npz(flown_app):
    app, _ = flown_app
    app._on_prepare_download("npz")
    assert app._dl["npz"]["data"] is not None
    app.w_traj_dtype.value = "float32"
    assert app._dl["npz"]["data"] is None
    app._on_prepare_download("npz")
    _, meta, _ = _load(app._dl["npz"]["data"])
    assert meta["precision"] == "float32"


def test_an_earlier_save_still_loads(tmp_path):
    """Format 1 (the earlier Save trajectories file) has the same layout
    at float64, so it keeps loading."""
    res = _results()
    p = field_io.save_trajectories(res, _spec(), path=str(tmp_path / "o.npz"),
                                   columns=COLS)
    with np.load(p, allow_pickle=False) as z:
        members = {k: z[k] for k in z.files}
    meta = field_io._member_to_meta(members[field_io._META])
    meta["traj_format"] = 1
    members[field_io._META] = field_io._meta_to_member(meta)
    old = str(tmp_path / "format1.npz")
    field_io._atomic_savez(old, **members)
    trajs, _, _ = field_io.load_trajectories(old)
    assert np.array_equal(trajs[0], res[0][1])
