"""A local web chat UI, built on the Python standard library only.

`python -m aria serve` starts a server on 127.0.0.1 and prints a URL. No web
framework, no build step, no CDN — the page is a single inline HTML string, so
it works with the network cable unplugged.

Uploading a document is part of the conversation: the paperclip next to the
message box (or dropping a file on the page) sends the file to the server,
which streams it to disk — any size — and learns from it in the background.
Learning takes one gradient step at a time, and the model lock is handed out
first-come-first-served, so a chat message waits for at most one step, not for
the whole document. You can keep talking to Aria while she reads.

The page can switch between the pretrained model and a blank one, each with
its own memory, loaded on first use.

Binding to localhost is not by itself a defence. Any web page open in the same
browser can send requests to 127.0.0.1, and a DNS-rebinding page can even read
the answers. Everything sent here is written into the model's weights, so the
handler refuses requests whose Host or Origin is not this server, and POSTs
that are neither JSON nor a raw upload (a cross-site form or `text/plain`
fetch needs no CORS preflight; `application/json` and
`application/octet-stream` do, and are refused without one).
"""

from __future__ import annotations

import io
import itertools
import json
import queue
import re
import shutil
import sys
import threading
import uuid
import webbrowser
from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .chat import SNIFF_LINES, ChatSession, _command as run_command
from .documents import (SUBTITLE_SUFFIXES, SUPPORTED_SUFFIXES, TEXT_SUFFIXES,
                        iter_lines, parse_transcript, speakers, suffix_of)
from .pretrain import resolve_checkpoint
from .sample import build_chat_prompt, generate

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
INSPECT_MAX_BYTES = 1 << 20
_CANCEL_ROUTE = re.compile(r"^/api/jobs/([0-9a-f]{32})/cancel$")

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aria</title>
<style>
  :root {
    --bg: #fbfbfa; --fg: #1a1a18; --muted: #6b6b66; --line: #e2e2dd;
    --user: #ecece7; --aria: #ffffff; --accent: #7a5c3e; --card: #f3f1ec;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #16161a; --fg: #e8e8e4; --muted: #90908a; --line: #2c2c32;
      --user: #23232a; --aria: #1c1c21; --accent: #c9a173; --card: #1f1e1b;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; height: 100vh;
  }
  header {
    padding: 10px 16px; border-bottom: 1px solid var(--line);
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; letter-spacing: .01em; }
  header .meta { color: var(--muted); font-size: 12.5px; }
  header .spacer { flex: 1; }
  select {
    font: inherit; font-size: 13px; padding: 4px 8px; border-radius: 7px;
    border: 1px solid var(--line); background: var(--bg); color: var(--fg); max-width: 100%;
  }
  label.toggle { color: var(--muted); font-size: 12.5px; cursor: pointer; user-select: none; }
  #log { flex: 1; overflow-y: auto; padding: 18px 16px; }
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
    color: var(--muted); margin: -4px 0 12px 2px; white-space: pre-wrap; word-wrap: break-word;
  }
  .learn.hidden { display: none; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 10px 13px; margin-bottom: 12px;
  }
  .card-title { font-weight: 600; font-size: 14px; word-wrap: break-word; }
  .card-status {
    font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted);
    white-space: pre-wrap; word-wrap: break-word; margin-top: 4px;
  }
  .bar { height: 4px; background: var(--line); border-radius: 2px; margin-top: 8px; overflow: hidden; }
  .bar .fill { height: 100%; width: 0; background: var(--accent); transition: width .4s; }
  .choices { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  footer { border-top: 1px solid var(--line); padding: 10px 16px; }
  form { max-width: 720px; margin: 0 auto; display: flex; gap: 8px; }
  input[type=text] {
    flex: 1; min-width: 0; padding: 10px 12px; border-radius: 8px; font: inherit;
    border: 1px solid var(--line); background: var(--bg); color: var(--fg);
  }
  input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  button {
    padding: 10px 18px; border-radius: 8px; border: 1px solid var(--line);
    background: var(--accent); color: #fff; font: inherit; cursor: pointer;
  }
  button.icon {
    padding: 0 11px; background: transparent; color: var(--fg); display: flex; align-items: center;
  }
  button.icon:hover { border-color: var(--accent); color: var(--accent); }
  button.small { padding: 5px 11px; font-size: 13px; }
  button.quiet { background: transparent; color: var(--fg); }
  button.link {
    background: none; border: none; padding: 0; margin-top: 6px; color: var(--accent);
    font-size: 12.5px; text-decoration: underline;
  }
  button:disabled { opacity: .5; cursor: default; }
  .hint { max-width: 720px; margin: 7px auto 0; color: var(--muted); font-size: 12px; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  body.dragging #log { outline: 2px dashed var(--accent); outline-offset: -8px; }
</style>
</head>
<body>
<header>
  <h1>Aria</h1>
  <select id="model" aria-label="Model"></select>
  <span class="meta" id="meta">loading…</span>
  <span class="spacer"></span>
  <label class="toggle"><input type="checkbox" id="showlearn" checked> show learning</label>
  <label class="toggle"><input type="checkbox" id="dolearn" checked> learn from this chat</label>
</header>

<div id="log"><div class="wrap" id="wrap"></div></div>

<footer>
  <form id="form">
    <button type="button" id="attach" class="icon" title="Upload a document for Aria to read and learn from" aria-label="Upload a document">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
    </button>
    <input type="file" id="file" accept="ACCEPT" hidden>
    <input type="text" id="input" placeholder="Say something…" autocomplete="off" autofocus>
    <button id="send" type="submit">Send</button>
  </form>
  <div class="hint">
    The paperclip (or dropping a file on the page) gives Aria a document to read — a
    book, letters, a chat log, a speech transcript, any size. She learns its grammar
    and words in the background while you keep talking.
    Commands: <code>/status</code>, <code>/memory</code>,
    <code>/correct &lt;what Aria should have said&gt;</code>, <code>/teach &lt;text&gt;</code>,
    <code>/consolidate</code>, <code>/save</code>.
  </div>
</footer>

<script>
const TEXTY = TEXTY_JSON;
const wrap = document.getElementById('wrap');
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const send = document.getElementById('send');
const showlearn = document.getElementById('showlearn');
const dolearn = document.getElementById('dolearn');
const modelSel = document.getElementById('model');
const fileInput = document.getElementById('file');
const cards = new Map();
let polling = false;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function scroll() { log.scrollTop = log.scrollHeight; }
function enc(s) { return encodeURIComponent(s || ''); }
function size(n) {
  if (n < 1e3) return n + ' B';
  if (n < 1e6) return (n / 1e3).toFixed(0) + ' KB';
  if (n < 1e9) return (n / 1e6).toFixed(1) + ' MB';
  return (n / 1e9).toFixed(2) + ' GB';
}
async function postJSON(url, body) {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
                              body: JSON.stringify(body)});
  return [r, await r.json().catch(() => ({}))];
}

function bubble(who, text) {
  const d = el('div', 'msg ' + who);
  d.appendChild(el('span', 'who', who === 'user' ? 'you' : 'aria'));
  d.appendChild(document.createTextNode(text));
  wrap.appendChild(d);
  scroll();
  return d;
}

function note(text) {
  const d = el('div', 'learn' + (showlearn.checked ? '' : ' hidden'), text);
  d.dataset.learn = '1';
  wrap.appendChild(d);
  scroll();
}

showlearn.addEventListener('change', () => {
  document.querySelectorAll('[data-learn]').forEach(
    e => e.classList.toggle('hidden', !showlearn.checked));
});

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

let metaSeq = 0, switching = false;
async function refreshMeta() {
  // Several refreshes can be in flight (a finished upload, a reply, a model
  // switch); only the newest answer may update the page.
  const mine = ++metaSeq;
  const s = await (await fetch('./api/status')).json();
  if (mine !== metaSeq) return;
  let meta = `${s.params_m.toFixed(2)}M params · ${s.updates_applied.toLocaleString()} updates`
    + ` · ${s.replay_size.toLocaleString()} memories · ${s.disk_mb.toFixed(1)} MB on disk`;
  document.getElementById('meta').textContent = meta;
  if (modelSel.options.length !== s.models.length) {
    modelSel.innerHTML = '';
    for (const m of s.models) {
      const o = el('option', '', m.label);
      o.value = m.key;
      modelSel.appendChild(o);
    }
  }
  if (!switching) modelSel.value = s.model;
  modelSel.hidden = s.models.length < 2;
}

modelSel.addEventListener('change', async () => {
  modelSel.disabled = true;
  switching = true;
  const label = modelSel.options[modelSel.selectedIndex].textContent;
  note('switching to ' + label + ' …');
  const [r, d] = await postJSON('./api/model', {model: modelSel.value});
  note(r.ok ? 'now talking to ' + label + '. Each model keeps its own memory.'
            : 'error: ' + (d.error || r.status));
  switching = false;
  await refreshMeta();
  modelSel.disabled = false;
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  send.disabled = true;

  if (text.startsWith('/')) {
    bubble('user', text);
    const [, d] = await postJSON('./api/command', {command: text});
    note(d.output || d.error || '(no output)');
    await refreshMeta();
    send.disabled = false; input.focus();
    return;
  }

  bubble('user', text);
  const target = bubble('aria', '');
  let started = false;
  const res = await fetch('./api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, learn: dolearn.checked}),
  });
  await readEvents(res, ev => {
    if (ev.token !== undefined) {
      // Replies begin with the chat format's leading space; don't show it.
      const tok = started ? ev.token : ev.token.trimStart();
      if (tok) {
        started = true;
        target.appendChild(document.createTextNode(tok));
      }
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

// --- documents --------------------------------------------------------------

function makeCard(title) {
  const d = el('div', 'card');
  const st = el('div', 'card-status', 'sending…');
  const bar = el('div', 'bar');
  const fill = el('div', 'fill');
  bar.appendChild(fill);
  const stop = el('button', 'link', 'stop and keep what she has learned');
  stop.type = 'button'; stop.hidden = true;
  d.append(el('div', 'card-title', title), st, bar, stop);
  wrap.appendChild(d);
  scroll();
  return {d, st, bar, fill, stop};
}

function renderJob(card, j) {
  let text = '';
  if (j.state === 'queued') text = 'waiting for the document before it…';
  else if (j.state === 'running' && j.phase === 'learning')
    text = `learning — step ${j.step.toLocaleString()} of ${j.total.toLocaleString()}`
         + ` (${Math.floor(100 * j.step / Math.max(1, j.total))}%). Keep talking if you like.`;
  else if (j.state === 'running')
    text = `reading — ${j.examples.toLocaleString()} pieces so far`;
  else if (j.state === 'done') text = j.summary + '\\n' + j.line;
  else if (j.state === 'stopped')
    text = 'stopped; she keeps what she learned.' + (j.summary ? '\\n' + j.summary + '\\n' + j.line : '');
  else if (j.state === 'failed') text = 'error: ' + j.error;
  card.st.textContent = text;
  const pct = j.state === 'done' ? 100 : (j.total ? 100 * j.step / j.total : 0);
  card.fill.style.width = pct + '%';
  card.bar.hidden = j.state === 'failed' || (j.state === 'stopped' && !j.summary);
  const active = j.state === 'queued' || j.state === 'running';
  card.stop.hidden = !active;
  card.stop.onclick = async () => {
    card.stop.disabled = true;
    await postJSON('./api/jobs/' + j.id + '/cancel', {});
  };
}

async function poll() {
  if (polling) return;
  polling = true;
  try {
    while (true) {
      const d = await (await fetch('./api/jobs')).json();
      let active = false;
      for (const j of d.jobs) {
        if (!cards.has(j.id) && (j.state === 'queued' || j.state === 'running'))
          cards.set(j.id, makeCard('\\u{1F4C4} ' + j.name));   // e.g. after a page reload
        if (cards.has(j.id)) renderJob(cards.get(j.id), j);
        if (j.state === 'queued' || j.state === 'running') active = true;
      }
      if (!active) break;
      await new Promise(r => setTimeout(r, 1000));
    }
  } finally {
    polling = false;
    refreshMeta();
  }
}

// A chat log or transcript can be learned as one person's way of replying.
// Ask whose, before sending a potentially huge file.
async function chooseVoice(file) {
  const dot = file.name.lastIndexOf('.');
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : '';
  if (!TEXTY.includes(ext)) return '';
  let names = null;
  try {
    const r = await fetch('./api/inspect?name=' + enc(file.name), {
      method: 'POST', headers: {'Content-Type': 'application/octet-stream'},
      body: file.slice(0, 65536)});
    names = (await r.json()).speakers;
  } catch (e) { return ''; }
  if (!names || !names.length) return '';
  return new Promise(resolve => {
    const d = el('div', 'card');
    d.appendChild(el('div', 'card-status',
      'This looks like a conversation between ' + names.slice(0, 6).join(', ')
      + '. Learn to reply the way one of them does?'));
    const row = el('div', 'choices');
    const pick = v => { d.remove(); resolve(v); };
    for (const n of names.slice(0, 6)) {
      const b = el('button', 'small', 'like ' + n);
      b.type = 'button'; b.onclick = () => pick(n);
      row.appendChild(b);
    }
    const plain = el('button', 'small quiet', 'no — just learn the language');
    plain.type = 'button'; plain.onclick = () => pick('');
    row.appendChild(plain);
    d.appendChild(row);
    wrap.appendChild(d);
    scroll();
  });
}

async function startUpload(file) {
  bubble('user', '\\u{1F4C4} ' + file.name + ' (' + size(file.size) + ')');
  const speaker = await chooseVoice(file);
  const card = makeCard('\\u{1F4C4} ' + file.name
                        + (speaker ? ' — learning to reply like ' + speaker : ''));
  try {
    const r = await fetch('./api/upload?name=' + enc(file.name) + '&speaker=' + enc(speaker), {
      method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: file});
    const d = await r.json().catch(() => ({error: 'upload failed (' + r.status + ')'}));
    if (!r.ok) {
      card.st.textContent = 'error: ' + (d.error || r.status);
      card.bar.hidden = true;
      return;
    }
    cards.set(d.job.id, card);
    renderJob(card, d.job);
    poll();
  } catch (e) {
    card.st.textContent = 'error: ' + e;
    card.bar.hidden = true;
  }
}

document.getElementById('attach').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  for (const f of fileInput.files) startUpload(f);
  fileInput.value = '';
});
document.addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('dragging'); });
document.addEventListener('dragleave', e => { if (!e.relatedTarget) document.body.classList.remove('dragging'); });
document.addEventListener('drop', e => {
  e.preventDefault();
  document.body.classList.remove('dragging');
  for (const f of e.dataTransfer.files) startUpload(f);
});

refreshMeta();
poll();
</script>
</body>
</html>
"""
PAGE = (PAGE_TEMPLATE
        .replace("ACCEPT", ",".join(SUPPORTED_SUFFIXES))
        .replace("TEXTY_JSON", json.dumps(list(TEXT_SUFFIXES + SUBTITLE_SUFFIXES))))


# ---------------------------------------------------------------------------
# Sharing one model between a conversation and a long upload
# ---------------------------------------------------------------------------


class FairLock:
    """A lock handed out in the order it was asked for.

    A plain Lock lets the learning thread, which releases and re-acquires it
    after every step, win the race again and again; a chat request could wait
    for the whole document. A ticket lock guarantees the message goes next."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._next_ticket = 0
        self._serving = 0

    def acquire(self) -> None:
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving:
                self._cond.wait()

    def release(self) -> None:
        with self._cond:
            self._serving += 1
            self._cond.notify_all()

    def __enter__(self) -> "FairLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


@dataclass
class Job:
    """One uploaded document, from arrival on disk to learned."""

    model: str
    name: str
    path: Path
    size: int
    speaker: str | None = None
    passes: int | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "queued"          # queued, running, done, stopped, failed
    phase: str = ""
    step: int = 0
    total: int = 0
    examples: int = 0
    summary: str = ""
    line: str = ""
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)

    def view(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "id", "model", "name", "size", "speaker", "state", "phase", "step",
            "total", "examples", "summary", "line", "error")}


class JobRunner:
    """Learns uploaded documents one after another, on a background thread.

    Each gradient step takes the model lock on its own, so chat requests
    interleave with learning rather than waiting for it to finish."""

    KEEP = 50

    def __init__(self, app: "App") -> None:
        self.app = app
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.queue: queue.Queue[Job | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="aria-learning", daemon=True)
        self.thread.start()

    def submit(self, job: Job) -> None:
        self.jobs[job.id] = job
        while len(self.jobs) > self.KEEP:
            oldest = next(iter(self.jobs.values()))
            if oldest.state in ("queued", "running"):
                break
            self.jobs.popitem(last=False)
        self.queue.put(job)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        job.cancel.set()
        return True

    def views(self) -> list[dict[str, Any]]:
        return [j.view() for j in list(self.jobs.values())]

    def stop(self, timeout: float = 120.0) -> None:
        for job in list(self.jobs.values()):
            job.cancel.set()
        self.queue.put(None)
        self.thread.join(timeout)

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            if job is None:
                return
            try:
                self._learn(job)
            finally:
                job.path.unlink(missing_ok=True)

    def _learn(self, job: Job) -> None:
        if job.cancel.is_set():
            job.state = "stopped"
            return
        job.state = "running"
        lock = self.app.lock
        try:
            with lock:
                session = self.app.session(job.model)
            gen = session.iter_learn_file(job.path, job.name, speaker=job.speaker,
                                          passes=job.passes, should_stop=job.cancel.is_set)
            while True:
                with lock:
                    try:
                        p = next(gen)
                    except StopIteration as done:
                        report, summary = done.value
                        break
                job.phase, job.step, job.total, job.examples = p.phase, p.step, p.total, p.examples
            job.summary, job.line = summary, report.line()
            job.state = "stopped" if report.stopped else "done"
        except ValueError as e:
            job.state, job.error = "failed", str(e)
        except Exception as e:            # keep the worker alive for the next one
            job.state, job.error = "failed", f"{type(e).__name__}: {e}"
            print(f"aria: learning {job.name} failed: {job.error}", file=sys.stderr)


class App:
    """The models a server can talk to, which one is active, and the jobs.

    Sessions are created on first use, so offering the blank model costs
    nothing until someone switches to it."""

    def __init__(self, slots: dict[str, dict[str, Any]], active: str,
                 session_kwargs: dict[str, Any] | None = None,
                 sessions: dict[str, ChatSession] | None = None) -> None:
        self.slots = slots
        self.active = active
        self.session_kwargs = session_kwargs or {}
        self.sessions: dict[str, ChatSession] = dict(sessions or {})
        self.lock = FairLock()
        self.jobs = JobRunner(self)

    @classmethod
    def for_session(cls, session: ChatSession, label: str = "Aria") -> "App":
        return cls({"default": {"label": label}}, "default", sessions={"default": session})

    def session(self, key: str | None = None) -> ChatSession:
        """The session for `key` (default: active). Hold `self.lock`."""
        key = key or self.active
        if key not in self.sessions:
            slot = self.slots[key]
            s = ChatSession(checkpoint=slot.get("checkpoint"), state_dir=slot.get("state_dir"),
                            blank=slot.get("blank", False), **self.session_kwargs)
            # Leftovers from a server that was killed mid-upload.
            shutil.rmtree(s.state_dir / "uploads", ignore_errors=True)
            self.sessions[key] = s
        return self.sessions[key]

    def switch(self, key: str) -> None:
        if key not in self.slots:
            raise KeyError(key)
        self.session(key)
        self.active = key

    def models(self) -> list[dict[str, Any]]:
        return [{"key": k, "label": v["label"], "loaded": k in self.sessions}
                for k, v in self.slots.items()]

    def close(self) -> None:
        self.jobs.stop()
        with self.lock:
            for s in self.sessions.values():
                s.save()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _hostname(host_header: str) -> str:
    """'localhost:8000' -> 'localhost', '[::1]:8000' -> '::1'."""
    h = host_header.strip().lower()
    if h.startswith("["):
        return h[1:].split("]", 1)[0]
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def make_handler(app: App, allowed_hosts: frozenset = LOCAL_HOSTS) -> type:
    return type("Handler", (_Handler,), {"app": app, "allowed_hosts": allowed_hosts})


class _Handler(BaseHTTPRequestHandler):
    app: App
    allowed_hosts: frozenset = LOCAL_HOSTS
    server_version = "aria"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- helpers --------------------------------------------------------

    @property
    def route(self) -> str:
        return urlsplit(self.path).path

    def _query(self) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}

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
            raise ValueError(f"message too large ({n:,} bytes) — to give Aria a long "
                             f"text, upload it as a document instead")
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

    def _status(self) -> dict[str, Any]:
        app = self.app
        with app.lock:
            s = app.session()
            st = s.learner.status()
            st["params_m"] = s.model.num_params() / 1e6
            st["learning"] = s.learning
            st["model"] = app.active
            st["models"] = app.models()
        return st

    # -- routes ---------------------------------------------------------

    def do_GET(self) -> None:
        if not self._trusted():
            self._refuse()
            return
        if self.route in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.route == "/api/status":
            self._json(self._status())
        elif self.route == "/api/jobs":
            self._json({"jobs": self.app.jobs.views()})
        else:
            self._json({"error": "not found"}, 404)

    JSON_ROUTES = ("/api/chat", "/api/command", "/api/model")
    RAW_ROUTES = ("/api/upload", "/api/inspect")

    def do_POST(self) -> None:
        if not self._trusted():
            self._refuse()
            return
        route = self.route
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        cancel = _CANCEL_ROUTE.match(route)
        expected = "application/octet-stream" if route in self.RAW_ROUTES else "application/json"
        if ctype != expected:
            self._json({"error": f"expected Content-Type: {expected}"}, 415)
            return
        if route == "/api/chat":
            self._chat()
        elif route == "/api/command":
            self._command()
        elif route == "/api/model":
            self._model()
        elif route == "/api/upload":
            self._upload()
        elif route == "/api/inspect":
            self._inspect()
        elif cancel:
            ok = self.app.jobs.cancel(cancel.group(1))
            self._json({"ok": ok}, 200 if ok else 404)
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
        with self.app.lock:
            s = self.app.session()
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
        try:
            cmd = str(self._read_json().get("command") or "").strip()
        except ValueError as e:
            self._json({"error": str(e)}, 400)
            return
        if not cmd.startswith("/"):
            self._json({"error": "not a command"}, 400)
            return
        if cmd.split()[0].lower() == "/upload":
            # It would hold the model for the whole document; the paperclip
            # learns in the background instead.
            self._json({"output": "use the paperclip next to the message box, or drop "
                                  "a file on the page: she reads it in the background "
                                  "while you keep talking."})
            return
        buf = io.StringIO()
        with self.app.lock:
            session = self.app.session()
            try:
                with redirect_stdout(buf):
                    quit_ = run_command(session, cmd)
            except Exception as e:
                self._json({"output": f"error: {type(e).__name__}: {e}"})
                return
            if quit_:
                session.save()
        out = buf.getvalue().rstrip()
        if quit_:
            out = (out + "\nsaved. (close the tab; the server is still running)").strip()
        self._json({"output": out})

    def _model(self) -> None:
        try:
            key = str(self._read_json().get("model") or "")
            with self.app.lock:
                self.app.switch(key)
        except KeyError:
            self._json({"error": f"no model called {key!r}"}, 400)
            return
        except ValueError as e:
            self._json({"error": str(e)}, 400)
            return
        self._json(self._status())

    def _upload(self) -> None:
        """Receive a document of any size straight to disk, then queue it."""
        q = self._query()
        name = Path(q.get("name") or "upload.txt").name or "upload.txt"
        suffix = suffix_of(name)
        if suffix not in SUPPORTED_SUFFIXES:
            self._json({"error": f"can't read {suffix or 'extension-less'} files; use one "
                                 f"of {', '.join(SUPPORTED_SUFFIXES)}"}, 415)
            return
        try:
            passes = int(q["passes"]) if q.get("passes") else None
            if passes is not None and passes < 1:
                raise ValueError
        except ValueError:
            self._json({"error": "passes must be a positive whole number"}, 400)
            return
        if self.headers.get("Content-Length") is None:
            self._json({"error": "Content-Length required"}, 411)
            return
        length = int(self.headers["Content-Length"])
        if length <= 0:
            self._json({"error": "empty file"}, 400)
            return

        with self.app.lock:
            key = self.app.active
            folder = self.app.session(key).state_dir / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{uuid.uuid4().hex}{suffix}"
        try:
            with open(dest, "wb") as f:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ConnectionError("upload interrupted")
                    f.write(chunk)
                    remaining -= len(chunk)
        except ConnectionError:
            dest.unlink(missing_ok=True)
            return
        except OSError as e:
            dest.unlink(missing_ok=True)
            self._json({"error": f"couldn't store the upload: {e.strerror or e}"}, 507)
            return

        job = Job(model=key, name=name, path=dest, size=length,
                  speaker=(q.get("speaker") or "").strip() or None, passes=passes)
        self.app.jobs.submit(job)
        self._json({"job": job.view()})

    def _inspect(self) -> None:
        """Does the start of this file look like a transcript, and of whom?"""
        name = Path(self._query().get("name") or "upload.txt").name
        n = int(self.headers.get("Content-Length") or 0)
        if n > INSPECT_MAX_BYTES:
            self._json({"error": "send only the start of the file"}, 413)
            return
        head = self.rfile.read(n)
        result = None
        if suffix_of(name) in TEXT_SUFFIXES + SUBTITLE_SUFFIXES:
            try:
                turns = parse_transcript(list(itertools.islice(iter_lines(head, name),
                                                               SNIFF_LINES)))
                result = speakers(turns) if turns else None
            except ValueError:
                pass
        self._json({"speakers": result})


def serve(
    checkpoint: str | Path | None = None,
    state_dir: str | Path | None = None,
    data_dir: str | Path = "data",
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    allow_hosts: tuple[str, ...] = (),
    blank: bool = False,
    **session_kwargs,
) -> None:
    slots: dict[str, dict[str, Any]] = {}
    try:
        pretrained = resolve_checkpoint(checkpoint)
        label = "Pretrained — knows English" if not checkpoint else f"{pretrained.name}"
        slots["pretrained"] = {"label": label, "checkpoint": pretrained}
    except FileNotFoundError:
        if checkpoint:
            raise
    slots["blank"] = {"label": "Blank — learns everything from you", "blank": True}
    active = "blank" if blank or "pretrained" not in slots else "pretrained"
    slots[active]["state_dir"] = state_dir

    app = App(slots, active, dict(session_kwargs, data_dir=data_dir))
    with app.lock:
        session = app.session()

    allowed = set(LOCAL_HOSTS) | {h.lower() for h in allow_hosts}
    if host not in ("0.0.0.0", "::", ""):
        allowed.add(host.lower())
    httpd = ThreadingHTTPServer((host, port), make_handler(app, frozenset(allowed)))

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    print(f"Aria is running at {url}")
    print(f"  model      {slots[active]['label']}, "
          f"{session.model.num_params()/1e6:.2f}M parameters")
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
        print("\nstopping (finishing the current learning step) ...")
    finally:
        httpd.server_close()
        app.close()
        print(f"saved to {', '.join(str(s.state_dir) for s in app.sessions.values())}")
