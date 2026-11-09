"""The Raspberry Pi edition: reachable on the network by its own names only,
saving every few minutes and when stopped, one at a time.

raspberry-pi/install.sh and the service it sets up are checked for real on a
64-bit ARM machine by .github/workflows/raspberry-pi.yml.
"""

import http.client
import json
import socket

import pytest

from aria import pi


def request(port, path, host, method="GET", body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    headers = {"Host": host}
    if body is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(body).encode()
    conn.request(method, path, body=body, headers=headers)
    r = conn.getresponse()
    out = r.status, r.read().decode()
    conn.close()
    return out


def test_its_names_and_addresses(monkeypatch):
    monkeypatch.setattr(pi, "local_addresses", lambda: {"127.0.0.1", "192.168.1.40"})
    names = pi.Names(("aria.tail1234.ts.net",))
    here = names.current()
    host = socket.gethostname().lower().split(".")[0]
    assert {"localhost", host, f"{host}.local", "192.168.1.40",
            "aria.tail1234.ts.net"} <= here
    assert "evil.example" not in here
    assert f"http://{host}.local:8000" in names.addresses_to_visit(8000)
    assert "http://192.168.1.40:8000" in names.addresses_to_visit(8000)


def test_a_new_address_is_noticed(monkeypatch):
    monkeypatch.setattr(pi, "ADDRESS_REFRESH_SECONDS", 0)
    monkeypatch.setattr(pi, "local_addresses", lambda: {"10.0.0.5"})
    names = pi.Names()
    assert "10.0.0.5" in names.current()
    monkeypatch.setattr(pi, "local_addresses", lambda: {"10.0.0.9"})
    assert "10.0.0.9" in names.current() and "10.0.0.5" not in names.current()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(pi, "local_addresses", lambda: {"127.0.0.1", "192.168.1.40"})
    s = pi.PiServer(port=0, host="127.0.0.1")
    s.start()
    yield s
    s.stop()


def test_it_answers_its_own_names_and_refuses_others(server):
    host = socket.gethostname().lower().split(".")[0]
    for name in (f"{host}.local", "192.168.1.40", "localhost"):
        status, body = request(server.port, "/api/hello", f"{name}:{server.port}")
        assert status == 200 and json.loads(body)["app"] == "aria", name
    status, _ = request(server.port, "/api/hello", "evil.example")
    assert status == 403
    status, body = request(server.port, "/api/chat", f"192.168.1.40:{server.port}",
                           "POST", {"message": "hello from the phone"})
    assert status == 200 and '"token"' in body
    # No Quit button: the Pi is meant to keep running.
    page = request(server.port, "/", f"{host}.local:{server.port}")[1]
    assert "const APP_MODE = false;" in page


def test_it_saves_every_few_minutes_and_when_stopped(server):
    session = server.server.session
    assert server.autosave() == []                     # nothing new yet
    session.learner.observe(["my dog is called Biscuit", "a fine name"], force=True)
    assert server.autosave() == ["pretrained"]
    learned = session.state_dir / "learned.pt"
    learned.unlink()
    session.learner.observe(["I live by the sea", "how lovely"], force=True)
    server.stop()                                       # as on shutdown
    assert learned.exists()
    server.stop()                                       # twice is harmless


def test_only_one_runs_at_a_time(isolated_aria_home, capsys):
    from aria.app import InstanceLock
    isolated_aria_home.mkdir(parents=True)
    held = InstanceLock(isolated_aria_home / "aria.lock")
    assert held.acquire()
    try:
        assert pi.main(["--port", "0"]) == 1
    finally:
        held.release()
    assert "already running" in capsys.readouterr().err
