"""The desktop app: starting, one-at-a-time, the page's Quit button, saving.

The window itself needs a screen; `aria app --self-test --require-window` checks
it on the build machines. Everything behind it is tested here with the
terminal stand-in, against the shipped model.
"""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from aria import app
from aria.cli import main
from aria.serve import build_server, render_page

LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url):
    with LOCAL.open(url, timeout=60) as r:
        return r.status, r.read().decode()


def post(url, payload, headers=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with LOCAL.open(req, timeout=120) as r:
        return r.status, r.read().decode()


def test_only_the_app_page_has_a_quit_button():
    assert "const APP_MODE = true;" in render_page(app_mode=True)
    assert "const APP_MODE = false;" in render_page()


def test_one_aria_at_a_time(tmp_path):
    first, second = app.InstanceLock(tmp_path / "aria.lock"), app.InstanceLock(tmp_path / "aria.lock")
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_quit_and_hello_routes():
    server = build_server(port=0)
    url = f"http://127.0.0.1:{server.port}"
    threading.Thread(target=server.httpd.serve_forever, daemon=True).start()
    try:
        assert json.loads(get(url + "/api/hello")[1])["app"] == "aria"
        # Not the app: nothing to quit.
        with pytest.raises(urllib.error.HTTPError) as e:
            post(url + "/api/quit", {})
        assert e.value.code == 404
        asked = threading.Event()
        server.app.on_quit = asked.set
        # Another web site can't quit her.
        with pytest.raises(urllib.error.HTTPError) as e:
            post(url + "/api/quit", {}, {"Origin": "https://evil.example"})
        assert e.value.code == 403 and not asked.is_set()
        assert json.loads(post(url + "/api/quit", {})[1]) == {"ok": True}
        assert asked.wait(5)            # answered first, then asked to quit
    finally:
        server.httpd.shutdown()
        server.close()


def test_starting_needs_no_network_name_lookup(monkeypatch):
    # Python's web server looks up its address's network name as it starts,
    # which took ~35 s on GitHub's Macs. Aria's server doesn't.
    import socket

    def no_lookup(*a, **k):
        raise AssertionError("looked up a network name")
    monkeypatch.setattr(socket, "getfqdn", no_lookup)
    server = build_server(port=0)
    server.close()


def test_every_request_body_is_read(monkeypatch):
    """A reply sent with the request still unread can be lost on Windows: the
    connection is reset instead of closed. Seen there as WinError 10053."""
    import http.client
    import io
    from aria.serve import App, make_handler

    app_ = App({"default": {"label": "x"}}, "default")
    app_.on_quit = lambda: None
    handler = make_handler(app_)
    body = b'{"x": 1}'
    try:
        for route, ctype in (("/api/quit", "application/json"),
                             ("/api/jobs/" + "0" * 32 + "/cancel", "application/json"),
                             ("/api/nowhere", "application/json"),
                             ("/api/chat", "text/plain"),
                             ("/api/upload?name=song.mp3", "application/octet-stream")):
            h = handler.__new__(handler)
            h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
            h.path, h.command = route, "POST"
            h.request_version, h.requestline = "HTTP/1.1", f"POST {route} HTTP/1.1"
            h.client_address = ("127.0.0.1", 0)
            h.headers = http.client.parse_headers(io.BytesIO(
                f"Host: 127.0.0.1\r\nContent-Type: {ctype}\r\n"
                f"Content-Length: {len(body)}\r\n\r\n".encode()))
            h.do_POST()
            assert h.rfile.tell() == len(body), f"{route} left its body unread"
            assert h.wfile.getvalue().startswith(b"HTTP/1.0 ")
    finally:
        app_.jobs.stop()


def test_autosave_saves_only_what_changed():
    server = build_server(port=0)
    try:
        assert server.app.save_if_changed() == []
        server.session.learner.observe(["my cat is called Pepper", "what a lovely name"],
                                       force=True)
        assert server.app.save_if_changed() == ["pretrained"]
        assert (server.session.state_dir / "learned.pt").exists()
        assert server.app.save_if_changed() == []
    finally:
        server.close()


def test_the_app_runs_and_quits_from_its_page(isolated_aria_home):
    home = isolated_aria_home
    home.mkdir(parents=True)
    ui = app.Console(home)
    aria = app.AriaApp(ui, home, port=0, browser=False)
    aria.start()
    loop = threading.Thread(target=ui.mainloop, daemon=True)
    loop.start()
    deadline = time.monotonic() + 120
    while aria.url is None and loop.is_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert aria.url, "the app didn't start"

    # Opening the app again finds this one instead of starting a second.
    assert app.find_running(home) == aria.url
    other = app.InstanceLock(home / "aria.lock")
    assert app._hand_over(other, home, browser=False, wait=5) is True

    status, body = post(aria.url + "/api/chat", {"message": "hello there"})
    assert status == 200 and '"token"' in body
    page = get(aria.url + "/")[1]
    assert 'id="quit"' in page and "const APP_MODE = true;" in page

    post(aria.url + "/api/quit", {})
    loop.join(60)
    assert not loop.is_alive(), "the app didn't stop"
    assert not (home / "server.json").exists()
    assert (home / "online" / "learned.pt").exists()
    assert app.find_running(home) is None


def test_self_test_passes(monkeypatch, capsys):
    monkeypatch.setattr(app, "_log_to", lambda path: None)
    assert app.self_test() == 0
    out = capsys.readouterr().out
    assert "self-test passed" in out and "FAIL" not in out


def test_aria_app_is_a_command(capsys):
    with pytest.raises(SystemExit) as e:
        main(["app", "--help"])
    assert e.value.code == 0
    assert "--self-test" in capsys.readouterr().out
