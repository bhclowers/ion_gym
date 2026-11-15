"""
The command line reports a user's mistake in one line with exit status 2,
and has --version.

A missing, malformed or refused deck and a busy port each ended in a
traceback with status 1, the same as a crash; `--mem-holders deck.json`
read the deck path as the flag's number; and there was no --version.
"""
import errno
import json
import socket
import subprocess
import sys

import pytest

import ion_gym
from ion_gym import cli
from ion_gym.io.paths import repo_root

DECK = repo_root() / "examples" / "ion_funnel_rz.json"


def _error(capsys):
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"ion-gym {ion_gym.__version__}"


def test_a_missing_deck_is_a_usage_error(tmp_path, capsys):
    assert cli.main(["dashboard", str(tmp_path / "no.json"),
                     "--no-show"]) == cli.USAGE_ERROR
    assert "cannot open deck" in _error(capsys)


def test_malformed_json_is_a_usage_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    assert cli.main(["dashboard", str(bad), "--no-show"]) == cli.USAGE_ERROR
    assert "JSONDecodeError" in _error(capsys)


def test_a_deck_without_a_source_block_is_a_usage_error(tmp_path, capsys):
    d = json.loads(DECK.read_text())
    del d["source"]
    p = tmp_path / "nosource.json"
    p.write_text(json.dumps(d))
    assert cli.main(["dashboard", str(p), "--no-show"]) == cli.USAGE_ERROR
    assert "'source'" in _error(capsys)


def test_a_number_where_the_deck_goes_names_the_threshold_flag(capsys):
    assert cli.main(["dashboard", "8000", "--no-show"]) == cli.USAGE_ERROR
    assert "--mem-holders-above" in _error(capsys)


def test_referrers_without_holders_is_a_usage_error(capsys):
    assert cli.main(["dashboard", "--mem-referrers", "X",
                     "--no-show"]) == cli.USAGE_ERROR
    assert "requires --mem-holders" in _error(capsys)


@pytest.fixture
def no_server(monkeypatch):
    """Build nothing and serve nothing: record what would be served."""
    from ion_gym.ui import serve, sim_app, telemetry
    seen = {}
    monkeypatch.setattr(sim_app, "SimApp", lambda spec: spec)
    monkeypatch.setattr(serve, "serve_dashboard",
                        lambda app, **kw: seen.update(spec=app, **kw))
    # fresh switch lists, restored afterwards (the CLI writes item 0)
    monkeypatch.setattr(telemetry, "VERBOSE", [False])
    monkeypatch.setattr(telemetry, "HOLDERS_ABOVE_MB", [None])
    return seen


def test_the_holder_flag_no_longer_swallows_the_deck(no_server):
    from ion_gym.ui import telemetry
    assert cli.main(["dashboard", "--mem-holders", str(DECK),
                     "--no-show"]) == 0
    assert no_server["spec"].name == json.loads(DECK.read_text())["name"]
    assert telemetry.VERBOSE[0] is True


def test_the_threshold_flag_takes_megabytes(no_server):
    from ion_gym.ui import telemetry
    assert cli.main(["dashboard", "--mem-holders-above", "8000",
                     "--no-show"]) == 0
    assert telemetry.HOLDERS_ABOVE_MB[0] == 8000.0


def test_a_busy_port_is_a_usage_error_and_other_os_errors_are_not(
        monkeypatch, capsys):
    from ion_gym.ui import serve, sim_app
    monkeypatch.setattr(sim_app, "SimApp", lambda spec: spec)

    def _busy(app, **kw):
        raise OSError(errno.EADDRINUSE, "Address already in use")
    monkeypatch.setattr(serve, "serve_dashboard", _busy)
    assert cli.main(["dashboard", "--no-show", "--port", "5999"]) == 2
    assert "port 5999 is already in use" in _error(capsys)

    def _other(app, **kw):
        raise OSError(errno.EACCES, "Permission denied")
    monkeypatch.setattr(serve, "serve_dashboard", _other)
    with pytest.raises(OSError, match="Permission denied"):
        cli.main(["dashboard", "--no-show", "--port", "5999"])


def test_validate_passes_a_good_deck(capsys):
    assert cli.main(["validate", str(DECK)]) == 0
    assert "VALID" in capsys.readouterr().out


def test_validate_refuses_what_build_run_refuses(tmp_path, capsys):
    from ion_gym.io.sim_spec import SimSpec
    from ion_gym.physics.sim_build import build_run
    d = json.loads(DECK.read_text())
    d["integration"]["dt_ns"] = -1.0
    p = tmp_path / "negdt.json"
    p.write_text(json.dumps(d))
    assert cli.main(["validate", str(p)]) == 1
    out = capsys.readouterr().out
    assert "INVALID" in out and "REFUSED: integration.dt_ns" in out
    # the same deck is refused by build_run, before anything is built
    with pytest.raises(ValueError, match="integration.dt_ns"):
        build_run(SimSpec.from_json(str(p)))


def test_validate_on_an_unreadable_deck_is_a_usage_error(tmp_path, capsys):
    assert cli.main(["validate", str(tmp_path / "no.json")]) == 2
    assert "cannot open deck" in _error(capsys)


def test_the_dashboard_on_a_busy_port_exits_with_a_message():
    # held the way a running server holds it: every IPv4 interface (a
    # loopback-only holder does not stop a wildcard bind on macOS). The
    # dashboard, not `edit`: the editor is switched off and refuses first.
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        s.listen()
        port = s.getsockname()[1]
        run = subprocess.run(
            [sys.executable, "-m", "ion_gym", "dashboard", "--no-show",
             "--port", str(port)], cwd=repo_root(), capture_output=True,
            text=True, timeout=180)
    assert run.returncode == cli.USAGE_ERROR, run.stderr[-2000:]
    assert f"port {port} is already in use" in run.stderr
    assert "Traceback" not in run.stderr
