"""Tests for the local web UI.

The server is exercised over a real socket against a real (tiny) model, because
the failure modes worth catching here — a route that 500s, a stream that never
terminates, an exception that kills the handler thread — only show up end to end.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from aria.chat import ChatSession
from aria.data import prepare
from aria.cli import main
from aria.serve import PAGE, _Handler
from http.server import ThreadingHTTPServer

CORPUS = """The river ran past the old mill and turned east towards the sea.
She opened the window, and listened to the rain falling on the roof.
Every morning the baker lit the oven, long before the sky began to lighten.
They walked together along the quiet road, until the light began to fade.
The letter arrived on a Tuesday, and nobody knew who had sent it.
"""


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("serve")
    data = root / "data"
    (data / "raw").mkdir(parents=True)
    (data / "raw" / "c.txt").write_text(CORPUS * 150)
    prepare(data_dir=data, vocab_size=400, block_size=64,
            n_surrogate_dialogues=200, offline=True, verbose=False)

    out = root / "run"
    assert main(["pretrain", "--data-dir", str(data), "--out-dir", str(out),
                 "--preset", "tiny", "--block-size", "64", "--steps", "20",
                 "--batch-size", "4", "--warmup", "5", "--eval-interval", "20",
                 "--checkpoint-interval", "1000", "--fisher-batches", "2"]) == 0

    session = ChatSession(checkpoint=out / "base.pt", state_dir=root / "online",
                          data_dir=data, max_new_tokens=8)
    handler = type("H", (_Handler,), {"session": session,
                                      "lock": threading.Lock()})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, session
    httpd.shutdown()
    httpd.server_close()


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return r.status, r.read().decode()


def post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode()


def sse_events(body):
    out = []
    for part in body.split("\n\n"):
        if part.startswith("data: "):
            out.append(json.loads(part[6:]))
    return out


def test_page_is_self_contained():
    """A strict offline requirement: no CDN, no external fonts, no fetch to
    another host. The page must work with no internet at all."""
    for bad in ["http://", "https://", "//cdn", "src=\"//"]:
        assert bad not in PAGE, f"page references {bad}"


def test_index_serves_html(server):
    base, _ = server
    status, body = get(base, "/")
    assert status == 200
    assert "<title>Aria</title>" in body


def test_status_endpoint(server):
    base, _ = server
    status, body = get(base, "/api/status")
    assert status == 200
    d = json.loads(body)
    assert d["params_m"] > 0
    assert d["plasticity"] == "lora"
    assert "updates_applied" in d


def test_unknown_route_is_404_not_a_crash(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base, "/nope")
    assert e.value.code == 404


def test_chat_streams_tokens_then_a_learning_report(server):
    base, session = server
    before = session.learner.updates_applied
    status, body = post(base, "/api/chat", {"message": "hello there"})
    assert status == 200
    events = sse_events(body)
    assert any("token" in e for e in events), "no tokens streamed"
    assert "learn" in events[-1], "stream did not end with a learning report"
    assert session.learner.turns_seen > 0
    assert session.learner.updates_applied >= before


def test_chat_can_opt_out_of_learning(server):
    base, session = server
    import torch
    before = {n: p.clone() for n, p in session.model.named_parameters()}
    _, body = post(base, "/api/chat", {"message": "say something", "learn": False})
    assert "off for this message" in sse_events(body)[-1]["learn"]
    for n, p in session.model.named_parameters():
        assert torch.equal(before[n], p), n


def test_empty_message_is_rejected(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base, "/api/chat", {"message": "   "})
    assert e.value.code == 400


def test_slash_commands_run_over_http(server):
    base, _ = server
    _, body = post(base, "/api/command", {"command": "/status"})
    assert "replay_size" in json.loads(body)["output"]


def test_bad_command_does_not_500(server):
    base, _ = server
    _, body = post(base, "/api/command", {"command": "/definitely-not-a-command"})
    assert "unknown command" in json.loads(body)["output"]


def test_non_command_to_command_endpoint_is_rejected(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base, "/api/command", {"command": "hello"})
    assert e.value.code == 400


def test_server_survives_a_sequence_of_requests(server):
    """A handler exception used to be able to take the thread down silently;
    make sure the server is still answering after everything above."""
    base, _ = server
    for msg in ["one", "two", "three"]:
        post(base, "/api/chat", {"message": msg})
    assert get(base, "/api/status")[0] == 200
