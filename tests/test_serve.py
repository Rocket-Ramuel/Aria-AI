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
from aria.serve import PAGE, App, FairLock, make_handler
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
    app = App.for_session(session)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
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
#
# A document is sent raw (application/octet-stream), lands on disk, and is
# learned by a background job; the page polls /api/jobs for progress.


def upload(base, name, data, **query):
    from urllib.parse import urlencode
    req = urllib.request.Request(
        f"{base}/api/upload?{urlencode(dict(name=name, **query))}", data=data,
        headers={"Content-Type": "application/octet-stream"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_for(base, job_id, timeout=120):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = {j["id"]: j for j in json.loads(get(base, "/api/jobs")[1])["jobs"]}
        if jobs[job_id]["state"] not in ("queued", "running"):
            return jobs[job_id]
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_upload_learns_in_the_background(server):
    base, session = server
    before = session.learner.updates_applied
    status, body = upload(base, "sample.txt", (CORPUS * 3).encode(), passes=1)
    assert status == 200 and body["job"]["state"] in ("queued", "running")
    job = wait_for(base, body["job"]["id"])
    assert job["state"] == "done", job
    assert "learned sample.txt" in job["summary"]
    assert job["total"] >= 1 and job["step"] == job["total"]
    assert session.learner.updates_applied > before
    # The uploaded copy is deleted once learned.
    assert not list((session.state_dir / "uploads").glob("*"))


def test_upload_of_a_transcript_as_a_speaker(server):
    base, _ = server
    log = b"Sam: are you coming tonight\nJo: reckon I will, love\n" * 3
    _, body = upload(base, "chat.txt", log, speaker="Jo", passes=1)
    assert "replies by Jo" in wait_for(base, body["job"]["id"])["summary"]
    _, body = upload(base, "chat.txt", log, speaker="Nobody", passes=1)
    job = wait_for(base, body["job"]["id"])
    assert job["state"] == "failed" and "speakers are" in job["error"]


def test_inspect_finds_the_speakers_of_a_transcript(server):
    base, _ = server
    req = urllib.request.Request(
        base + "/api/inspect?name=chat.txt", method="POST",
        data=b"Sam: hi\nJo: hiya\nSam: you well\nJo: not bad\nJo: you?\n",
        headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=30) as r:
        assert json.loads(r.read())["speakers"] == ["Jo", "Sam"]
    req = urllib.request.Request(
        base + "/api/inspect?name=essay.txt", method="POST",
        data=b"An essay. It has no speakers at all.\n",
        headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=30) as r:
        assert json.loads(r.read())["speakers"] is None


def test_a_running_upload_can_be_stopped(server):
    base, session = server
    status, body = upload(base, "long.txt", (CORPUS * 40).encode(), passes=50)
    job_id = body["job"]["id"]
    _, out = post(base, f"/api/jobs/{job_id}/cancel", {})
    assert json.loads(out)["ok"]
    job = wait_for(base, job_id)
    assert job["state"] == "stopped"
    assert job["step"] < job["total"] or job["total"] == 0


def test_chat_is_not_blocked_by_a_long_upload(server):
    """The point of learning in the background: a message waits for one
    gradient step, not for the whole document."""
    import time
    base, _ = server
    _, body = upload(base, "long.txt", (CORPUS * 40).encode(), passes=50)
    job_id = body["job"]["id"]
    deadline = time.time() + 60
    while json.loads(get(base, "/api/jobs")[1])["jobs"][-1]["phase"] != "learning":
        assert time.time() < deadline
        time.sleep(0.05)
    t0 = time.time()
    status, reply = post(base, "/api/chat", {"message": "hello while you read"})
    elapsed = time.time() - t0
    still = {j["id"]: j for j in json.loads(get(base, "/api/jobs")[1])["jobs"]}[job_id]
    post(base, f"/api/jobs/{job_id}/cancel", {})
    wait_for(base, job_id)
    assert status == 200 and "learn" in sse_events(reply)[-1]
    assert still["state"] == "running", "chat only answered after the upload finished"
    assert elapsed < 30


def test_bad_uploads_are_rejected_cleanly(server):
    base, _ = server
    code, body = upload(base, "song.mp3", b"\x00\x01\x02")
    assert code == 415 and "can't read" in body["error"]
    code, body = upload(base, "x.txt", b"hello", passes="0")
    assert code == 400
    req = urllib.request.Request(base + "/api/upload?name=x.txt", data=b"hello",
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 415
    assert get(base, "/api/status")[0] == 200


def test_upload_from_another_site_is_refused(server):
    base, _ = server
    req = urllib.request.Request(base + "/api/upload?name=x.txt", data=b"inject",
                                 headers={"Content-Type": "application/octet-stream",
                                          "Origin": "https://evil.example"},
                                 method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 403


def test_upload_command_points_to_the_paperclip(server):
    base, _ = server
    _, body = post(base, "/api/command", {"command": "/upload /etc/hosts"})
    assert "paperclip" in json.loads(body)["output"]


def test_page_offers_a_file_upload():
    assert 'type="file"' in PAGE and ".docx" in PAGE and "/api/upload" in PAGE
    assert 'id="attach"' in PAGE          # next to the message box, not hidden away


def test_fair_lock_serves_in_arrival_order():
    import time
    lock, order = FairLock(), []
    lock.acquire()

    def worker(i):
        lock.acquire()
        order.append(i)
        lock.release()

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
        time.sleep(0.05)          # make arrival order unambiguous
    lock.release()
    for t in threads:
        t.join(5)
    assert order == [0, 1, 2, 3, 4]


def test_switching_models_keeps_separate_memories(server, tmp_path):
    """Pretrained and blank sit side by side; each has its own memory."""
    from aria.pretrain import create_blank_checkpoint
    _, session = server
    blank = create_blank_checkpoint(tmp_path / "blank" / "base.pt", size="tiny",
                                    block_size=64)
    app = App({"pretrained": {"label": "Pretrained"},
               "blank": {"label": "Blank", "checkpoint": blank}},
              "pretrained", {"max_new_tokens": 4}, sessions={"pretrained": session})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        st = json.loads(get(base, "/api/status")[1])
        assert st["model"] == "pretrained"
        assert [m["key"] for m in st["models"]] == ["pretrained", "blank"]
        assert not st["models"][1]["loaded"]            # loaded on first use

        _, body = post(base, "/api/model", {"model": "blank"})
        st = json.loads(body)
        assert st["model"] == "blank" and st["plasticity"] == "full"
        _, reply = post(base, "/api/chat", {"message": "hello blank"})
        assert "learn" in sse_events(reply)[-1]
        assert app.sessions["blank"].history and \
            app.sessions["blank"].state_dir != session.state_dir

        with pytest.raises(urllib.error.HTTPError) as e:
            post(base, "/api/model", {"model": "nope"})
        assert e.value.code == 400
        post(base, "/api/model", {"model": "pretrained"})
        assert json.loads(get(base, "/api/status")[1])["model"] == "pretrained"
    finally:
        httpd.shutdown()
        httpd.server_close()
