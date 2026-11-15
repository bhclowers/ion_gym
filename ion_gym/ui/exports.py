"""Prepared files the dashboard hands to the browser, served over HTTP.

A Download button that carries its bytes (Panel's FileDownload) sends the
whole file to the page as ONE base64 text message, which the browser has
to receive and parse in JavaScript before it saves anything. A 3-D run's
trajectory NPZ of a few hundred MB froze the tab and dropped its
connection to the server, and nothing was saved. Prepared run exports are
published here instead and served at /export/<token> by an ordinary HTTP
handler on the dashboard's own server, so the browser streams them to
disk like any download, at any size.

A token is unguessable (secrets.token_urlsafe) and names one prepared
file; withdrawing it frees the bytes and the link answers 404.
"""
from __future__ import annotations

__all__ = ["ROUTE", "publish", "withdraw", "url", "routes", "served",
           "mark_served", "ExportHandler"]

import secrets
import threading

import tornado.web

ROUTE = r"/export/([A-Za-z0-9_-]+)"
CHUNK_BYTES = 4 * 1024 * 1024        # written and flushed per piece
_FILES = {}                          # token -> (filename, bytes)
_LOCK = threading.Lock()
_SERVED = [False]                    # a server carrying ROUTE was built


def publish(data, filename, replaces=None):
    """Publish `data` (bytes) for download as `filename`; returns its token.
    `replaces`: a token this file supersedes, withdrawn in the same step."""
    if not isinstance(data, bytes):
        raise TypeError(f"publish: data must be bytes, got "
                        f"{type(data).__name__}")
    token = secrets.token_urlsafe(16)
    with _LOCK:
        if replaces is not None:
            _FILES.pop(replaces, None)
        _FILES[token] = (str(filename), data)
    return token


def withdraw(token):
    """Forget a published file (a no-op for a token already withdrawn)."""
    with _LOCK:
        _FILES.pop(token, None)


def url(token):
    """The download path of a published file, relative to the server."""
    return f"/export/{token}"


def mark_served():
    """Record that the dashboard server carries the export route."""
    _SERVED[0] = True


def served():
    """True once a dashboard server with the export route has been built:
    only then can a page's download link be answered."""
    return _SERVED[0]


def routes():
    """The extra HTTP pattern for `pn.serve(extra_patterns=...)`."""
    return [(ROUTE, ExportHandler)]


class ExportHandler(tornado.web.RequestHandler):
    """GET /export/<token>: the published file, as an attachment."""

    async def get(self, token):
        with _LOCK:
            item = _FILES.get(token)
        if item is None:
            raise tornado.web.HTTPError(
                404, reason="No such export: it was replaced by a newer "
                            "one or withdrawn. Prepare it again.")
        name, data = item
        self.set_header("Content-Type", "application/octet-stream")
        self.set_header("Content-Disposition",
                        f'attachment; filename="{name}"')
        self.set_header("Content-Length", str(len(data)))
        view = memoryview(data)
        for start in range(0, len(data), CHUNK_BYTES):
            self.write(bytes(view[start:start + CHUNK_BYTES]))
            await self.flush()
