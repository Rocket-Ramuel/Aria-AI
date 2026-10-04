"""A local web chat UI, built on the Python standard library only.

`python -m aria serve` starts a server on 127.0.0.1 and prints a URL. No web
framework, no build step, no CDN — the page is a single inline HTML string, so
it works with the network cable unplugged.

The server binds to localhost by default and holds a lock around the model:
generation and the online update both mutate it, so requests are serialised.
That is the right trade for a single-user local tool and the wrong one for
anything public — see `--host`.

Binding to localhost is not by itself a defence. Any web page open in the same
browser can send requests to 127.0.0.1, and a DNS-rebinding page can even read
the answers. Everything sent here is written into the model's weights, so the
handler refuses requests whose Host or Origin is not this server, and POSTs
that are not JSON (a cross-site form or `text/plain` fetch needs no CORS
preflight; `application/json` does, and is refused without one).
"""

from __future__ import annotations

import base64
import binascii
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .chat import ChatSession
from .documents import SUPPORTED_SUFFIXES
from .sample import build_chat_prompt, generate

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
MAX_UPLOAD_BYTES = 25_000_000

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aria</title>
<style>
  :root {
    --bg: #fbfbfa; --fg: #1a1a18; --muted: #6b6b66; --line: #e2e2dd;
    --user: #ecece7; --aria: #ffffff; --accent: #7a5c3e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #16161a; --fg: #e8e8e4; --muted: #90908a; --line: #2c2c32;
      --user: #23232a; --aria: #1c1c21; --accent: #c9a173;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; height: 100vh;
  }
  header {
    padding: 12px 18px; border-bottom: 1px solid var(--line);
    display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; letter-spacing: .01em; }
  header .meta { color: var(--muted); font-size: 12.5px; }
  header .spacer { flex: 1; }
  label.toggle { color: var(--muted); font-size: 12.5px; cursor: pointer; user-select: none; }
  #log { flex: 1; overflow-y: auto; padding: 18px; }
  .wrap { max-width: 720px; margin: 0 auto; }
  .msg {
    padding: 10px 13px; border-radius: 10px; margin-bottom: 10px;
    border: 1px solid var(--line); white-space: pre-wrap; word-wrap: break-word;
  }
  .msg.user { background: var(--user); }
  .msg.aria { background: var(--aria); }
  .msg .who {
    display: block; font-size: 11px; text-transform: uppercase;
    letter-spacing: .07em; color: var(--muted); margin-bottom: 4px;
  }
  .learn {
    font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--muted); margin: -4px 0 12px 2px; white-space: pre-wrap;
  }
  .learn.hidden { display: none; }
  footer { border-top: 1px solid var(--line); padding: 12px 18px; }
  form { max-width: 720px; margin: 0 auto; display: flex; gap: 8px; }
  input[type=text] {
    flex: 1; padding: 10px 12px; border-radius: 8px; font: inherit;
    border: 1px solid var(--line); background: var(--bg); color: var(--fg);
  }
  input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  button {
    padding: 10px 18px; border-radius: 8px; border: 1px solid var(--line);
    background: var(--accent); color: #fff; font: inherit; cursor: pointer;
  }
  button:disabled { opacity: .5; cursor: default; }
  .hint { max-width: 720px; margin: 8px auto 0; color: var(--muted); font-size: 12px; }
  details.upload { max-width: 720px; margin: 8px auto 0; font-size: 13px; color: var(--muted); }
  details.upload summary { cursor: pointer; user-select: none; }
  .uprow { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 8px; }
  .uprow input[type=text] { flex: 1 1 160px; padding: 7px 10px; font-size: 13px; }
  .uprow input[type=number] {
    width: 64px; padding: 7px 8px; border-radius: 8px; font: inherit; font-size: 13px;
    border: 1px solid var(--line); background: var(--bg); color: var(--fg);
  }
  .uprow button { padding: 7px 14px; font-size: 13px; }
  body.dragging #log { outline: 2px dashed var(--accent); outline-offset: -8px; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
</style>
</head>
<body>
<header>
  <h1>Aria</h1>
  <span class="meta" id="meta">loading…</span>
  <span class="spacer"></span>
  <label class="toggle"><input type="checkbox" id="showlearn" checked> show learning</label>
  <label class="toggle"><input type="checkbox" id="dolearn" checked> learn from this chat</label>
</header>

<div id="log"><div class="wrap" id="wrap"></div></div>

<footer>
  <form id="form">
    <input type="text" id="input" placeholder="Say something…" autocomplete="off" autofocus>
    <button id="send" type="submit">Send</button>
  </form>
  <div class="hint">
    Slash commands work here too — <code>/status</code>, <code>/memory</code>,
    <code>/correct &lt;what Aria should have said&gt;</code>, <code>/teach &lt;text&gt;</code>,
    <code>/consolidate</code>, <code>/save</code>.
  </div>
  <details class="upload" id="upload">
    <summary>Teach from a file — a writing sample, a chat log, a speech transcript</summary>
    <div class="uprow">
      <input type="file" id="file" accept="ACCEPT">
      <input type="text" id="speaker" placeholder="in a transcript, learn to answer like… (name, optional)">
      <label>passes <input type="number" id="passes" min="1" max="20" value="3"></label>
      <button id="learnfile" type="button">Learn</button>
    </div>
    <div class="hint">Prose is learned as Aria's own voice. A chat log or transcript
      with <code>Name: words</code> lines, given a name, teaches her to reply the
      way that person does. You can also drop a file anywhere on the page.</div>
  </details>
</footer>

<script>
const wrap = document.getElementById('wrap');
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const send = document.getElementById('send');
const showlearn = document.getElementById('showlearn');
const dolearn = document.getElementById('dolearn');

function scroll() { log.scrollTop = log.scrollHeight; }

async function readEvents(res, onEvent) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    const parts = buf.split('\\n\\n');
    buf = parts.pop();
    for (const p of parts) {
      if (p.startsWith('data: ')) onEvent(JSON.parse(p.slice(6)));
    }
  }
}

function bubble(who, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + who;
  const w = document.createElement('span');
  w.className = 'who';
  w.textContent = who === 'user' ? 'you' : 'aria';
  d.appendChild(w);
  d.appendChild(document.createTextNode(text));
  wrap.appendChild(d);
  scroll();
  return d;
}

function note(text) {
  const d = document.createElement('div');
  d.className = 'learn' + (showlearn.checked ? '' : ' hidden');
  d.dataset.learn = '1';
  d.textContent = text;
  wrap.appendChild(d);
  scroll();
}

showlearn.addEventListener('change', () => {
  document.querySelectorAll('[data-learn]').forEach(
    e => e.classList.toggle('hidden', !showlearn.checked));
});

async function refreshMeta() {
  const r = await fetch('./api/status');
  const s = await r.json();
  document.getElementById('meta').textContent =
    `${s.params_m.toFixed(2)}M params · ${s.plasticity} · `
    + `${s.updates_applied} updates applied · ${s.replay_size} memories`;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  send.disabled = true;

  if (text.startsWith('/')) {
    bubble('user', text);
    const r = await fetch('./api/command', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({command: text}),
    });
    const d = await r.json();
    note(d.output || '(no output)');
    await refreshMeta();
    send.disabled = false; input.focus();
    return;
  }

  bubble('user', text);
  const target = bubble('aria', '');
  const res = await fetch('./api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, learn: dolearn.checked}),
  });

  await readEvents(res, ev => {
    if (ev.token !== undefined) {
      target.appendChild(document.createTextNode(ev.token));
      scroll();
    } else if (ev.learn !== undefined) {
      note(ev.learn);
    } else if (ev.error !== undefined) {
      note('error: ' + ev.error);
    }
  });
  await refreshMeta();
  send.disabled = false; input.focus();
});

const fileInput = document.getElementById('file');
const learnBtn = document.getElementById('learnfile');

function toBase64(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

async function uploadFile(file) {
  if (!file) return;
  learnBtn.disabled = true; send.disabled = true;
  bubble('user', 'uploaded ' + file.name);
  const status = document.createElement('div');
  status.className = 'learn'; status.dataset.learn = '1';
  status.textContent = 'reading ' + file.name + ' …';
  wrap.appendChild(status); scroll();
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const res = await fetch('./api/upload', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        name: file.name, data: toBase64(bytes),
        speaker: document.getElementById('speaker').value.trim(),
        passes: parseInt(document.getElementById('passes').value, 10) || null,
      }),
    });
    if (!res.ok) {
      status.textContent = 'error: ' + ((await res.json()).error || res.status);
    } else {
      await readEvents(res, ev => {
        if (ev.progress) {
          status.textContent = `learning ${file.name} … step ${ev.progress[0]}/${ev.progress[1]}`;
        } else if (ev.done !== undefined) {
          status.textContent = ev.done + '\\n' + ev.learn;
        } else if (ev.error !== undefined) {
          status.textContent = 'error: ' + ev.error;
        }
        scroll();
      });
    }
  } catch (e) {
    status.textContent = 'error: ' + e;
  }
  fileInput.value = '';
  await refreshMeta();
  learnBtn.disabled = false; send.disabled = false;
}

learnBtn.addEventListener('click', () => uploadFile(fileInput.files[0]));
document.addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('dragging'); });
document.addEventListener('dragleave', e => { if (!e.relatedTarget) document.body.classList.remove('dragging'); });
document.addEventListener('drop', e => {
  e.preventDefault(); document.body.classList.remove('dragging');
  if (e.dataTransfer.files.length) {
    document.getElementById('upload').open = true;
    uploadFile(e.dataTransfer.files[0]);
  }
});

refreshMeta();
</script>
</body>
</html>
"""
PAGE = PAGE_TEMPLATE.replace("ACCEPT", ",".join(SUPPORTED_SUFFIXES))


def _hostname(host_header: str) -> str:
    """'localhost:8000' -> 'localhost', '[::1]:8000' -> '::1'."""
    h = host_header.strip().lower()
    if h.startswith("["):
        return h[1:].split("]", 1)[0]
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


class _Handler(BaseHTTPRequestHandler):
    session: ChatSession
    lock: threading.Lock
    allowed_hosts: frozenset = LOCAL_HOSTS
    server_version = "aria"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- helpers --------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _read_json(self, limit: int = 1_000_000) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError(f"request too large ({n} bytes)")
        if not n:
            return {}
        body = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")
        return body

    def _trusted(self) -> bool:
        """Is this request from the Aria page itself, not another site?"""
        if _hostname(self.headers.get("Host", "")) not in self.allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        if origin is not None:
            parts = urlsplit(origin)
            if parts.scheme not in ("http", "https") or \
                    (parts.hostname or "") not in self.allowed_hosts:
                return False
        return True

    def _refuse(self) -> None:
        self._json({"error": "refused: this server only answers its own page "
                             "(see --allow-host)"}, 403)

    # -- routes ---------------------------------------------------------

    def do_GET(self) -> None:
        if not self._trusted():
            self._refuse()
            return
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            with self.lock:
                st = self.session.learner.status()
                st["params_m"] = self.session.model.num_params() / 1e6
                st["learning"] = self.session.learning
            self._json(st)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if not self._trusted():
            self._refuse()
            return
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._json({"error": "expected Content-Type: application/json"}, 415)
            return
        if self.path == "/api/chat":
            self._chat()
        elif self.path == "/api/command":
            self._command()
        elif self.path == "/api/upload":
            self._upload()
        else:
            self._json({"error": "not found"}, 404)

    def _event(self, obj: Any) -> None:
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def _start_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _chat(self) -> None:
        try:
            body = self._read_json()
        except ValueError as e:
            self._json({"error": str(e)}, 400)
            return
        message = str(body.get("message") or "").strip()
        if not message:
            self._json({"error": "empty message"}, 400)
            return

        self._start_stream()
        s = self.session
        with self.lock:
            try:
                learn = bool(body.get("learn", True)) and s.learning
                prompt = build_chat_prompt(s.tok, s.history, message,
                                           s.model.cfg.block_size)
                pieces: list[int] = []
                decoder = s.tok.stream_decoder()
                for tid in generate(
                    s.model, prompt, max_new_tokens=s.max_new_tokens,
                    temperature=s.temperature, top_k=s.top_k, top_p=s.top_p,
                    stop_ids=(s.tok.eot_id, s.tok.user_id, s.tok.bos_id),
                    device=s.device,
                ):
                    pieces.append(tid)
                    piece = decoder.feed(tid)
                    if piece:
                        self._event({"token": piece})
                tail = decoder.flush()
                if tail:
                    self._event({"token": tail})

                reply = s.tok.decode(pieces, skip_special=True).strip()
                s.history.append(("user", message))
                s.history.append(("aria", reply))
                if learn:
                    self._event({"learn": s.learner.observe(s._recent_turns()).line()})
                else:
                    self._event({"learn": "[learn] off for this message"})
            except (BrokenPipeError, ConnectionResetError):
                return                      # browser navigated away mid-stream
            except Exception as e:          # keep the server alive
                self._event({"error": f"{type(e).__name__}: {e}"})

    def _command(self) -> None:
        import io
        from contextlib import redirect_stdout
        from .chat import _command as run_command

        try:
            cmd = str(self._read_json().get("command") or "").strip()
        except ValueError as e:
            self._json({"error": str(e)}, 400)
            return
        if not cmd.startswith("/"):
            self._json({"error": "not a command"}, 400)
            return
        buf = io.StringIO()
        with self.lock:
            try:
                with redirect_stdout(buf):
                    quit_ = run_command(self.session, cmd)
            except Exception as e:
                self._json({"output": f"error: {type(e).__name__}: {e}"})
                return
        out = buf.getvalue().rstrip()
        if quit_:
            self.session.save()
            out = (out + "\nsaved. (close the tab; the server is still running)").strip()
        self._json({"output": out})

    def _upload(self) -> None:
        try:
            body = self._read_json(limit=MAX_UPLOAD_BYTES * 4 // 3 + 4096)
            name = Path(str(body.get("name") or "upload.txt")).name
            data = base64.b64decode(body.get("data") or "", validate=True)
            passes = body.get("passes")
            passes = int(passes) if passes else None
            if passes is not None and not 1 <= passes <= 50:
                raise ValueError("passes must be between 1 and 50")
        except (ValueError, binascii.Error) as e:
            self._json({"error": str(e)}, 400)
            return
        if not data:
            self._json({"error": "empty file"}, 400)
            return
        speaker = str(body.get("speaker") or "").strip() or None

        self._start_stream()
        with self.lock:
            try:
                report, summary = self.session.upload(
                    name, data, speaker=speaker, passes=passes,
                    progress=lambda step, total, lr: self._event(
                        {"progress": [step, total]}),
                )
                self._event({"done": summary, "learn": report.line()})
            except (BrokenPipeError, ConnectionResetError):
                return
            except ValueError as e:
                self._event({"error": str(e)})
            except Exception as e:          # keep the server alive
                self._event({"error": f"{type(e).__name__}: {e}"})


def serve(
    checkpoint: str | Path | None = None,
    state_dir: str | Path | None = None,
    data_dir: str | Path = "data",
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    allow_hosts: tuple[str, ...] = (),
    **session_kwargs,
) -> None:
    session = ChatSession(checkpoint=checkpoint, state_dir=state_dir,
                          data_dir=data_dir, **session_kwargs)

    allowed = set(LOCAL_HOSTS) | {h.lower() for h in allow_hosts}
    if host not in ("0.0.0.0", "::", ""):
        allowed.add(host.lower())
    handler = type("Handler", (_Handler,), {
        "session": session, "lock": threading.Lock(),
        "allowed_hosts": frozenset(allowed),
    })
    httpd = ThreadingHTTPServer((host, port), handler)

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    print(f"Aria is running at {url}")
    print(f"  model      {session.model.num_params()/1e6:.2f}M parameters")
    print(f"  memory     {session.state_dir}")
    if host not in LOCAL_HOSTS:
        print("  warning: bound beyond localhost. Anything typed into this page "
              "is written into the model's weights — do not expose it.")
        if host in ("0.0.0.0", "::") and not allow_hosts:
            print("  note: requests are only accepted with Host: localhost. To "
                  "reach it by another name or address, pass --allow-host NAME.")
    print("  press Ctrl-C to stop (memory is saved on exit)")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        httpd.server_close()
        session.save()
        print(f"saved to {session.state_dir}")
