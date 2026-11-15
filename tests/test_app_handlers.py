"""
The workbench's top-level handlers report failures with the exception's
type and stop the action they belong to (batch-9 handler audit).

Built headlessly, as test_ccs_fallback_routes builds the app.
"""
import pytest

from ion_gym.io.sim_spec import SimSpec

FIXTURE = "tests/fixtures/einzel_rz_gate_fixture.json"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))
    from ion_gym.io.paths import repo_root
    from ion_gym.ui.sim_app import SimApp
    return SimApp(SimSpec.from_json(str(repo_root() / FIXTURE)))


def test_apply_json_reports_the_type_and_says_it_failed(app):
    """It used to write "JSON error: <msg>" for every failure and return
    nothing, so a caller could not tell a refused load from a good one."""
    before = app.spec
    app.w_json.value = "{ this is not json"
    assert app._on_apply_json() is False
    assert "spec not loaded" in app.status.object
    assert "JSONDecodeError" in app.status.object
    assert app.spec is before


def test_apply_json_returns_true_on_a_good_document(app):
    from ion_gym.io.paths import repo_root
    app.w_json.value = (repo_root() / FIXTURE).read_text(encoding="utf-8")
    assert app._on_apply_json() is True
    assert "spec loaded" in app.status.object


def test_a_failing_delivery_is_not_swallowed_as_no_document(monkeypatch):
    """pe_view._run_off_doc: the try around add_periodic_callback also
    covered the first _poll(), whose _finish pops the callbacks before
    calling them, so an error in the delivery was taken for "no
    schedulable document" and the result silently never appeared."""
    import time

    import panel as pn

    from ion_gym.viz.pe_view import _run_off_doc

    class _Handle:
        def stop(self):
            pass

    def _schedule(cb, period):
        time.sleep(0.3)                    # let the worker finish first
        return _Handle()
    monkeypatch.setattr(type(pn.state), "curdoc",
                        property(lambda self: object()))
    monkeypatch.setattr(pn.state, "add_periodic_callback", _schedule)

    def _apply(res):
        raise ValueError("delivery failed")
    errors = []
    with pytest.raises(ValueError, match="delivery failed"):
        _run_off_doc(lambda: 42, _apply,
                     lambda e, tb: errors.append(e), join_timeout_s=5.0)
    assert errors == []


def test_bootstrap_stops_when_the_apply_fails(app, monkeypatch):
    """The bootstrap tested `self.spec is None`, which a failed apply never
    set, so it went on to load the field against the OLD spec and could
    announce "session opened from file"."""
    from ion_gym.io import field_io
    monkeypatch.setattr(field_io, "read_spec_from_field",
                        lambda p: (None, {"spec": {"broken": True}}))

    def _must_not_load(*a, **k):
        raise AssertionError("loaded a field after a failed apply")
    monkeypatch.setattr(field_io, "load_field", _must_not_load)
    monkeypatch.setattr(app.w_fieldpick, "value", "some_field.npz",
                        raising=False)
    app._on_bootstrap_field()
    assert "session opened" not in app.status.object
    assert "spec not loaded" in app.status.object


def test_a_failed_record_estimate_refuses_the_fly(app, monkeypatch):
    """An estimate that cannot be computed refuses the fly: flying on
    would skip the memory guard, and the warning was overwritten by the
    progress messages."""
    def _broken(spec):
        raise ValueError('no record width for this deck')
    monkeypatch.setattr(app, '_record_volume', _broken)
    assert app._record_guard(app.spec) is False
    assert 'REFUSED' in app.status.object
    assert 'ValueError' in app.status.object


def test_an_unreadable_mirror_frames_unmirrored_and_says_so(app,
                                                           monkeypatch):
    """The view is framed without mirrors and the status line says
    why. Said once, not per axis."""
    from ion_gym.physics import build_stl3d

    def _broken(spec):
        raise ValueError('mirror declaration unreadable')
    monkeypatch.setattr(build_stl3d, '_declared_mirror_axes', _broken)
    monkeypatch.setattr(app, '_scene', None, raising=False)
    app.status.object = 'ready'
    lo, hi = app._domain_range('x')
    app._domain_range('y')
    g = app.spec.geometry
    assert hi - lo == g.width_mm
    assert app.status.object.startswith('ready')
    assert app.status.object.count('framed without mirrors') == 1
    assert 'ValueError' in app.status.object


def test_a_run_without_provenance_is_saved_as_unrecorded(app):
    """A stored run with no captured provenance is described as unknown,
    never by the live spec (which may have changed since it flew)."""
    from ion_gym.physics.ensemble_driver import EnsembleProgress
    app._runs['old'] = EnsembleProgress(done=0, total=0, elapsed_s=0.0,
                                        n_terminated=0)
    prov = app._run_provenance('old')
    assert prov['seed_mode'] == 'not recorded' and prov['spec'] is None
    app._runs['new'] = EnsembleProgress(done=0, total=0, elapsed_s=0.0,
                                        n_terminated=0,
                                        provenance={'seed': 7})
    assert app._run_provenance('new') == {'seed': 7}
