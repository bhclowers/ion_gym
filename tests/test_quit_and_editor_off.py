"""
The dashboard's Quit button, and the geometry editor switched off.

QUIT. `ion-gym dashboard` could only be stopped from its terminal. A Quit
button in the top bar now asks for confirmation, stops any running work,
replaces the page with a "shut down" notice (and asks the browser to
close the tab), and ends the process the way Ctrl-C does. Only the
launcher that owns the process arms it; served from a notebook it stays
disabled, since ending that process would end the kernel.

EDITOR OFF. While the editor is reworked (ui.serve.EDITOR_ENABLED) the
dashboard draws its link greyed out with no target, `/editor` serves a
notice, and `ion-gym edit` refuses in one line.
"""
import signal
import socket
import subprocess
import sys
import time

import pytest

from ion_gym import cli
from ion_gym.io.paths import repo_root
from ion_gym.ui import serve, sim_app


@pytest.fixture(scope="module")
def app():
    return sim_app.SimApp()


def _html_with(layout, text):
    import panel as pn
    return [p.object for p in layout.select(pn.pane.HTML)
            if text in str(p.object)]


# ------------------------------------------------------------ editor off
def test_the_editor_switch_is_off():
    assert serve.EDITOR_ENABLED is False


def test_the_dashboard_draws_the_editor_link_greyed_out(app):
    (box,) = _html_with(app.panel(), "Geometry editor")
    assert 'href="/editor"' not in box
    assert "not-allowed" in box and "opacity" in box
    assert serve.EDITOR_OFF_NOTE in box


def test_the_editor_route_serves_a_notice(app):
    routes = serve.dashboard_routes(app=app)
    assert routes["/editor"] is serve.editor_off_page
    import panel as pn
    md = serve.editor_off_page().select(pn.pane.Markdown)
    assert any("temporarily disabled" in m.object for m in md)


def test_ion_gym_edit_refuses_in_one_line(capsys):
    assert cli.main(["edit", "--no-show"]) == cli.USAGE_ERROR
    err = capsys.readouterr().err
    assert "temporarily disabled" in err and "Traceback" not in err


# ------------------------------------------------------------------ quit
def test_quit_is_disabled_until_a_launcher_arms_it():
    a = sim_app.SimApp()
    assert a.quit_btn.disabled
    assert not a._quit_confirm.visible
    with pytest.raises(TypeError, match="callable"):
        a.enable_quit("not a function")


def test_quit_asks_first_and_cancel_backs_out(app):
    app.enable_quit(lambda: None)
    assert not app.quit_btn.disabled
    app.quit_btn.clicks += 1
    assert app._quit_confirm.visible
    app._quit_no.clicks += 1
    assert not app._quit_confirm.visible


def test_confirming_stops_the_work_then_quits(monkeypatch):
    a = sim_app.SimApp()
    seen = []
    monkeypatch.setattr(sim_app, "QUIT_DELAY_S", 0.01)
    monkeypatch.setattr(a, "_on_stop", lambda *_: seen.append("stop"))
    a.enable_quit(lambda: seen.append("quit"))
    a.quit_btn.clicks += 1
    a._quit_yes.clicks += 1
    for _ in range(200):                    # bounded: 2 s at most
        if "quit" in seen:
            break
        time.sleep(0.01)
    assert seen == ["stop", "quit"]
    assert "quitting" in a.status.object


def test_the_page_notice_tries_to_close_the_tab():
    assert "window.close()" in sim_app.QUIT_PAGE_JS
    assert "ion_gym has shut down" in sim_app.QUIT_PAGE_JS


def test_serve_dashboard_arms_quit_only_when_given_an_action(monkeypatch):
    import panel as pn

    class Stub:
        armed = None

        def enable_quit(self, action):
            self.armed = action

        def start_telemetry(self):
            pass

        def panel(self):
            return pn.pane.Markdown("stub")

    monkeypatch.setattr(pn, "serve", lambda routes, **kw: routes)
    quiet, armed = Stub(), Stub()
    serve.serve_dashboard(app=quiet, show=False)
    routes = serve.serve_dashboard(app=armed, show=False, on_quit=print)
    assert quiet.armed is None and armed.armed is print
    assert "on_quit" not in routes and set(routes) == {"/", "/editor",
                                                       "/flight"}


def test_the_dashboard_command_arms_quit_with_its_own_exit(monkeypatch):
    seen = {}
    monkeypatch.setattr(sim_app, "SimApp", lambda spec: spec)
    monkeypatch.setattr(serve, "serve_dashboard",
                        lambda app, **kw: seen.update(kw))
    assert cli.main(["dashboard", "--no-show"]) == 0
    assert seen["on_quit"] is cli._quit_this_process


def test_the_exit_is_the_ctrl_c_signal_to_this_process(monkeypatch):
    sent = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: sent.append(
        (pid, sig)))
    cli._quit_this_process()
    assert sent == [(cli.os.getpid(), signal.SIGINT)]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_real_dashboard_process_ends_cleanly_on_quit():
    """End to end: the signal Quit sends stops a real `ion-gym dashboard`
    with exit status 0 and its 'stopped' line, with no thread keeping
    the process alive."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "ion_gym", "dashboard", "--no-show",
         "--port", str(port)], cwd=repo_root(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        up = False
        for _ in range(360):                # bounded: 180 s at most
            if proc.poll() is not None:
                break
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    up = True
                    break
            time.sleep(0.5)
        assert up, f"dashboard never listened: {proc.poll()!r}"
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, err[-2000:]
    assert "ion-gym: dashboard stopped" in out
    assert "Traceback" not in err
