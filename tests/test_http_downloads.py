"""
Run exports download over plain HTTP, at any size.

The Runs tab's Download buttons were Panel FileDownloads, which send the
whole file to the page as ONE base64 text message that the browser must
parse before saving. Downloading a 3-D SLIM run's trajectory NPZ froze
the tab and dropped its connection, and no file was saved. Prepared
exports are now published on the dashboard server's /export/<token>
route (ui.exports) and the Download control is an ordinary link, which
the browser streams to disk.
"""
import os
import socket
import time
import urllib.error
import urllib.request

import panel as pn
import pytest
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application

from ion_gym.ui import exports, serve, sim_app


@pytest.fixture
def fresh_exports(monkeypatch):
    monkeypatch.setattr(exports, "_FILES", {})
    monkeypatch.setattr(exports, "_SERVED", [False])


# ------------------------------------------------------------- the route
class TestExportHandler(AsyncHTTPTestCase):
    def get_app(self):
        return Application(exports.routes())

    def tearDown(self):
        exports._FILES.clear()
        super().tearDown()

    def test_serves_the_exact_bytes_as_an_attachment(self):
        blob = os.urandom(9 * 1024 * 1024 + 123)     # spans several chunks
        tok = exports.publish(blob, "trajectories.npz")
        r = self.fetch(exports.url(tok))
        assert r.code == 200 and r.body == blob
        assert r.headers["Content-Length"] == str(len(blob))
        assert r.headers["Content-Disposition"] == \
            'attachment; filename="trajectories.npz"'

    def test_withdrawn_or_replaced_files_answer_404(self):
        old = exports.publish(b"old", "a.csv")
        new = exports.publish(b"new", "a.csv", replaces=old)
        assert self.fetch(exports.url(old)).code == 404
        assert self.fetch(exports.url(new)).body == b"new"
        exports.withdraw(new)
        assert self.fetch(exports.url(new)).code == 404
        assert self.fetch("/export/not-a-token").code == 404


def test_publish_takes_bytes_only(fresh_exports):
    with pytest.raises(TypeError, match="bytes"):
        exports.publish("text", "a.csv")


# ------------------------------------------------------------ the server
def test_serve_dashboard_carries_the_export_route(monkeypatch,
                                                   fresh_exports):
    seen = {}

    class Stub:
        def start_telemetry(self):
            pass

        def panel(self):
            return pn.pane.Markdown("stub")

    monkeypatch.setattr(pn, "serve", lambda routes, **kw: seen.update(kw))
    serve.serve_dashboard(app=Stub(), show=False)
    assert exports.ROUTE in [p[0] for p in seen["extra_patterns"]]
    assert exports.served()


# --------------------------------------------------------------- the app
@pytest.fixture
def armed_app(monkeypatch, fresh_exports):
    a = sim_app.SimApp()
    d = a._dl["npz"]
    d.update(data=b"\x93NUMPY" + os.urandom(2000), secs=0.2, run="run 1",
             key=("run 1", 1))
    return a, d


def test_no_run_export_is_sent_through_the_page(armed_app):
    a, _ = armed_app
    for kind, d in a._dl.items():
        assert not isinstance(d["dl"], pn.widgets.FileDownload), kind
        assert isinstance(d["dl"], pn.pane.HTML), kind


def test_a_prepared_export_is_a_link_to_its_published_file(armed_app):
    a, d = armed_app
    exports.mark_served()
    a._arm_download("npz")
    assert d["token"] in exports._FILES
    assert f'href="/export/{d["token"]}"' in d["dl"].object
    assert 'download="trajectories.traj.npz"' in d["dl"].object
    assert "ordinary download" in a.w_runs_msg.object


def test_a_changed_run_withdraws_the_published_file(armed_app,
                                                     monkeypatch):
    a, d = armed_app
    exports.mark_served()
    a._arm_download("npz")
    tok = d["token"]
    monkeypatch.setattr(a, "_run_key", lambda: ("run 2", 2))
    a._disarm_stale_downloads()
    assert tok not in exports._FILES and d["token"] is None
    assert "href" not in d["dl"].object


def test_without_the_route_the_app_says_what_to_do_instead(armed_app):
    a, d = armed_app
    a._arm_download("npz")                   # served() is False here
    assert d["token"] is None and "href" not in d["dl"].object
    assert "Save trajectories" in a.w_runs_msg.object


# ------------------------------------------------------------ end to end
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_real_dashboard_server_streams_a_large_export(fresh_exports):
    port = _free_port()
    srv = serve.serve_dashboard(app=sim_app.SimApp(), port=port, show=False,
                                threaded=True)
    try:
        for _ in range(240):                 # bounded: 120 s at most
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.5)
        else:
            pytest.fail("the dashboard server never listened")
        blob = os.urandom(64 * 1024 * 1024)
        tok = exports.publish(blob, "trajectories.npz")
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{exports.url(tok)}",
                timeout=120) as r:
            got = r.read()
        assert got == blob
        exports.withdraw(tok)
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}{exports.url(tok)}", timeout=30)
        assert e.value.code == 404
    finally:
        srv.stop()
