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


# --- refusing other sites ------------------------------------------------
#
# Anything that reaches /api/* is written into the model's weights, so the
# server must not answer a page from another site. A cross-site form or
# text/plain fetch needs no CORS preflight; a DNS-rebinding page arrives with
# a foreign Host header.


def raw(base, path, body=b"", headers=None, method="POST"):
    req = urllib.request.Request(base + path, data=body if method == "POST" else None,
                                 headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_text_plain_post_is_refused(server):
    base, session = server
    before = len(session.learner.replay)
    code, _ = raw(base, "/api/command",
                  json.dumps({"command": "/teach injected text from another site"}).encode(),
                  {"Content-Type": "text/plain"})
    assert code == 415
    assert len(session.learner.replay) == before


def test_foreign_origin_is_refused(server):
    base, _ = server
    code, body = raw(base, "/api/command", json.dumps({"command": "/status"}).encode(),
                     {"Content-Type": "application/json",
                      "Origin": "https://evil.example"})
    assert code == 403 and "refused" in body
    code, _ = raw(base, "/api/command", json.dumps({"command": "/status"}).encode(),
                  {"Content-Type": "application/json", "Origin": "null"})
    assert code == 403


def test_foreign_host_is_refused_even_for_reads(server):
    """DNS rebinding: the attacker's name resolves to 127.0.0.1, so the browser
    lets their page read the response. The Host header gives it away."""
    base, _ = server
    code, _ = raw(base, "/api/command", json.dumps({"command": "/memory"}).encode(),
                  {"Content-Type": "application/json", "Host": "attacker.example:8000"})
    assert code == 403
    code, _ = raw(base, "/api/status", headers={"Host": "attacker.example"}, method="GET")
    assert code == 403


def test_own_page_is_accepted(server):
    base, _ = server
    port = base.rsplit(":", 1)[1]
    for host in (f"localhost:{port}", f"127.0.0.1:{port}"):
        code, _ = raw(base, "/api/command", json.dumps({"command": "/status"}).encode(),
                      {"Content-Type": "application/json", "Host": host,
                       "Origin": f"http://{host}"})
        assert code == 200, host


def test_hostname_parsing():
    from aria.serve import _hostname
    assert _hostname("localhost:8000") == "localhost"
    assert _hostname("LOCALHOST") == "localhost"
    assert _hostname("[::1]:8000") == "::1"
    assert _hostname("127.0.0.1") == "127.0.0.1"


def test_malformed_json_is_a_400_not_a_crash(server):
    base, _ = server
    code, _ = raw(base, "/api/command", b"[1, 2]", {"Content-Type": "application/json"})
    assert code == 400
    assert get(base, "/api/status")[0] == 200


# --- uploads -------------------------------------------------------------


def upload(base, name, data, **extra):
    import base64
    return post(base, "/api/upload",
                {"name": name, "data": base64.b64encode(data).decode(), **extra})


def test_upload_streams_progress_then_a_summary(server):
    base, session = server
    before = session.learner.updates_applied
    status, body = upload(base, "sample.txt", (CORPUS * 3).encode(), passes=1)
    assert status == 200
    events = sse_events(body)
    assert any("progress" in e for e in events)
    assert "learned sample.txt" in events[-1]["done"]
    assert session.learner.updates_applied > before


def test_upload_of_a_transcript_as_a_speaker(server):
    base, _ = server
    log = b"Sam: are you coming tonight\nJo: reckon I will, love\n" * 3
    _, body = upload(base, "chat.txt", log, speaker="Jo", passes=1)
    assert "replies by Jo" in sse_events(body)[-1]["done"]
    _, body = upload(base, "chat.txt", log, speaker="Nobody", passes=1)
    assert "speakers are" in sse_events(body)[-1]["error"]


def test_bad_uploads_are_rejected_cleanly(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base, "/api/upload", {"name": "x.txt", "data": "not base64!!"})
    assert e.value.code == 400
    _, body = upload(base, "song.mp3", b"\x00\x01\x02")
    assert "can't read" in sse_events(body)[-1]["error"]
    assert get(base, "/api/status")[0] == 200


def test_page_offers_a_file_upload():
    assert 'type="file"' in PAGE and ".docx" in PAGE and "/api/upload" in PAGE
