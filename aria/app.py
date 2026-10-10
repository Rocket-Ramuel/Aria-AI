"""Aria as a desktop app.

Open it and a small window says Aria is starting; a moment later her chat
opens in your web browser. The window is how you can tell she's running and
how you stop her: closing it, Quit, or the "Quit Aria" button on the page
saves everything she has learned and shuts her down. Opening the app again
while she's running just opens the chat again.

The same code runs from source (`python -m aria app`, or the "Start Aria"
files in the download) and as the packaged app built from packaging/aria.spec.
Where Python has no Tk (some Linux setups) it runs in the terminal instead,
and Ctrl-C stops it.

`--self-test` starts Aria for real in a temporary folder — loads the model,
serves the page, chats, reads a short document, saves, quits, reloads — and
checks every step. It is how a freshly built app is checked before anyone
downloads it.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable

from .paths import data_dir, frozen

PREFERRED_PORT = 8765
LOG_NAME = "aria.log"
LOG_MAX_BYTES = 1 << 20
AUTOSAVE_SECONDS = 300
ICON = Path(__file__).with_name("icon.png")

# Requests to Aria herself must never go through a proxy.
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------------------
# Output, other programs, and messages for someone with no terminal open
# ---------------------------------------------------------------------------


class _Tee(io.TextIOBase):
    def __init__(self, *streams) -> None:
        self.streams = [s for s in streams if s is not None]

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        for stream in self.streams:
            try:
                stream.write(s)
                stream.flush()
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def _log_to(path: Path) -> None:
    """Copy everything printed to a log file in Aria's folder. An app opened
    by double-clicking has no terminal (on Windows, no output at all), so
    this file is where to look when something goes wrong."""
    try:
        if path.exists() and path.stat().st_size > LOG_MAX_BYTES:
            path.replace(path.with_name(path.name + ".1"))
        log = open(path, "a", encoding="utf-8", errors="replace")
    except OSError:
        return
    log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    sys.stdout = _Tee(log, sys.stdout)
    sys.stderr = _Tee(log, sys.stderr)


def _child_env() -> dict[str, str]:
    """The environment for programs Aria starts. A packaged app on Linux points
    LD_LIBRARY_PATH at its own libraries, and a browser started with those can
    crash; put back what it was."""
    env = dict(os.environ)
    if frozen() and sys.platform.startswith("linux"):
        orig = env.pop("LD_LIBRARY_PATH_ORIG", None)
        if orig is None:
            env.pop("LD_LIBRARY_PATH", None)
        else:
            env["LD_LIBRARY_PATH"] = orig
    return env


def open_in_browser(url: str) -> None:
    if frozen() and sys.platform.startswith("linux"):
        try:
            subprocess.Popen(["xdg-open", url], env=_child_env(), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            return
        except OSError:
            pass
    try:
        if not webbrowser.open(url):
            print(f"couldn't open a browser; go to {url}")
    except Exception as e:
        print(f"couldn't open a browser ({e}); go to {url}")


def _applescript(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def show_error(title: str, message: str) -> None:
    """An error message that reaches someone who has no terminal open."""
    print(f"{title}: {message}", file=sys.stderr)
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror(title, message, parent=root)
        root.destroy()
        return
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
        elif sys.platform == "darwin":
            subprocess.run(["osascript", "-e", f"display alert {_applescript(title)} "
                            f"message {_applescript(message)} as critical"],
                           capture_output=True, timeout=3600)
        else:
            subprocess.run(["zenity", "--error", "--title", title, "--text", message],
                           env=_child_env(), capture_output=True, timeout=3600)
    except Exception:
        pass


def _menu_entry() -> None:
    """On Linux, list the packaged app among the others once it has been
    opened, so it needn't be found in a folder again."""
    if not (frozen() and sys.platform.startswith("linux")):
        return
    exe = Path(sys.executable).resolve()
    icon = Path(getattr(sys, "_MEIPASS", exe.parent)) / "aria" / "icon.png"
    entry = ("[Desktop Entry]\nType=Application\nName=Aria\n"
             "Comment=A small AI that learns as you talk to her\n"
             f'Exec="{exe}"\nIcon={icon}\nTerminal=false\nCategories=Utility;\n'
             "StartupWMClass=Aria\n")
    apps = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") \
        / "applications"
    target = apps / "aria.desktop"
    try:
        if not target.exists() or target.read_text(encoding="utf-8") != entry:
            apps.mkdir(parents=True, exist_ok=True)
            target.write_text(entry, encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# One Aria at a time
# ---------------------------------------------------------------------------


class InstanceLock:
    """Only one Aria may use her memory at a time: two would each save over
    what the other learned. The system lets go of the lock if the app dies."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None

    def acquire(self) -> bool:
        if self._file is not None:
            return True
        f = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._file = f
        return True

    def release(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _info_path(home: Path) -> Path:
    return home / "server.json"


def find_running(home: Path, timeout: float = 2.0) -> str | None:
    """The address of an Aria that is already running, if she answers."""
    try:
        url = json.loads(_info_path(home).read_text(encoding="utf-8"))["url"]
        with _LOCAL.open(url + "/api/hello", timeout=timeout) as r:
            if json.load(r).get("app") == "aria":
                return url
    except Exception:
        pass
    return None


def _hand_over(lock: InstanceLock, home: Path, browser: bool, wait: float = 90.0) -> bool:
    """Another Aria holds the lock: open her chat instead. She may still be
    starting (wait for her) or on her way out (then take over). Returns False
    if this one should start after all."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        url = find_running(home)
        if url:
            print(f"Aria is already running at {url}")
            if browser:
                open_in_browser(url)
            return True
        if lock.acquire():
            return False
        time.sleep(0.5)
    show_error("Aria", "Aria seems to be running already but isn't answering. "
                       "If she's stuck, restart your computer and open her again.")
    return True


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


class Window:
    """The small window that shows Aria is running."""

    def __init__(self, home: Path) -> None:
        import tkinter as tk
        from tkinter import font, ttk

        self.on_open: Callable[[], None] = lambda: None
        self.on_quit: Callable[[], None] = lambda: None
        self._calls: queue.Queue = queue.Queue()
        self._alive = True
        self.root = root = tk.Tk(className="Aria")
        root.title("Aria")
        root.resizable(False, False)
        try:
            self._icon = tk.PhotoImage(file=str(ICON))
            root.iconphoto(True, self._icon)
        except Exception:
            pass

        title_font = font.nametofont("TkDefaultFont").copy()
        title_font.configure(size=20, weight="bold")
        frame = ttk.Frame(root, padding=(22, 16, 22, 18))
        frame.grid(sticky="nsew")
        ttk.Label(frame, text="Aria", font=title_font).grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.status = ttk.Label(frame, text="Starting Aria… The first time after "
                                "installing her can take a minute.", width=46,
                                wraplength=340, justify="left")
        self.status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 14))
        self.open_btn = ttk.Button(frame, text="Open chat", state="disabled",
                                   command=lambda: self.on_open())
        self.open_btn.grid(row=2, column=0, sticky="w")
        self.quit_btn = ttk.Button(frame, text="Quit", command=lambda: self.on_quit())
        self.quit_btn.grid(row=2, column=1, sticky="e")
        ttk.Label(frame, text=f"What she learns is kept in\n{home}", foreground="gray",
                  wraplength=340, justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(16, 0))

        root.protocol("WM_DELETE_WINDOW", lambda: self.on_quit())
        if sys.platform == "darwin":
            # Clicking the Dock icon, and Quit from the menu, Dock or Cmd-Q.
            root.createcommand("::tk::mac::ReopenApplication", lambda *a: self.on_open())
            root.createcommand("::tk::mac::Quit", lambda *a: self.on_quit())
        # Bring the window to the front once, without keeping it on top.
        root.lift()
        root.attributes("-topmost", True)
        root.after(400, lambda: root.attributes("-topmost", False))
        root.after(100, self._pump)

    def call(self, fn: Callable, *args) -> None:
        """Run `fn` on the window's thread; safe from any thread."""
        self._calls.put((fn, args))

    def _pump(self) -> None:
        while True:
            try:
                fn, args = self._calls.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args)
            except Exception:
                traceback.print_exc()
        if self._alive:
            self.root.after(100, self._pump)

    def every(self, ms: int, fn: Callable[[], None]) -> None:
        def tick():
            if self._alive:
                try:
                    fn()
                except Exception:
                    traceback.print_exc()
                self.root.after(ms, tick)
        self.root.after(ms, tick)

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def ready(self) -> None:
        self.open_btn.configure(state="normal")

    def busy(self, text: str) -> None:
        self.open_btn.configure(state="disabled")
        self.quit_btn.configure(state="disabled")
        self.set_status(text)

    def confirm(self, title: str, text: str) -> bool:
        from tkinter import messagebox
        return messagebox.askyesno(title, text, parent=self.root)

    def error(self, title: str, text: str) -> None:
        from tkinter import messagebox
        messagebox.showerror(title, text, parent=self.root)

    def mainloop(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self._alive = False
        self.root.destroy()


class Console:
    """The window's stand-in where there is no Tk: the terminal is the window,
    and Ctrl-C (or closing the terminal) quits."""

    def __init__(self, home: Path) -> None:
        self.on_open: Callable[[], None] = lambda: None
        self.on_quit: Callable[[], None] = lambda: None
        self._calls: queue.Queue = queue.Queue()
        self._alive = True
        self._timers: list[list] = []
        self._last = ""
        print(f"What she learns is kept in {home}")

    def call(self, fn: Callable, *args) -> None:
        self._calls.put((fn, args))

    def every(self, ms: int, fn: Callable[[], None]) -> None:
        self._timers.append([ms / 1000, time.monotonic() + ms / 1000, fn])

    def set_status(self, text: str) -> None:
        if text != self._last:
            self._last = text
            print(text)

    def ready(self) -> None:
        print("Press Ctrl-C to stop Aria; she saves what she has learned.")

    def busy(self, text: str) -> None:
        self.set_status(text)

    def confirm(self, title: str, text: str) -> bool:
        return True

    def error(self, title: str, text: str) -> None:
        print(f"{title}: {text}", file=sys.stderr)

    def mainloop(self) -> None:
        while self._alive:
            try:
                try:
                    fn, args = self._calls.get(timeout=0.2)
                    fn(*args)
                except queue.Empty:
                    pass
                now = time.monotonic()
                for t in self._timers:
                    if now >= t[1]:
                        t[1] = now + t[0]
                        t[2]()
            except KeyboardInterrupt:
                self.on_quit()
            except Exception:
                traceback.print_exc()

    def close(self) -> None:
        self._alive = False


def _describe(jobs: list[dict]) -> str:
    if not jobs:
        return ("Aria is running. Talk to her in your web browser — "
                "if you closed the page, click Open chat.")
    j = jobs[0]
    if j["state"] == "running" and j["phase"] == "learning" and j["total"]:
        text = f"Reading {j['name']} — {100 * j['step'] // max(1, j['total'])}%."
    else:
        text = f"Reading {j['name']}…"
    if len(jobs) > 1:
        text += f" ({len(jobs) - 1} more waiting.)"
    return text + " You can keep talking to her meanwhile."


# ---------------------------------------------------------------------------
# Running Aria behind the window
# ---------------------------------------------------------------------------


class AriaApp:
    """Starts the chat server, keeps the window up to date, and stops it all
    safely."""

    def __init__(self, ui, home: Path, port: int = PREFERRED_PORT,
                 browser: bool = True) -> None:
        self.ui, self.home, self.port, self.browser = ui, home, port, browser
        self.server = None
        self.url: str | None = None
        self.quitting = False
        self._last_save = time.monotonic()
        ui.on_open = self.open_chat
        ui.on_quit = self.quit

    def start(self) -> None:
        threading.Thread(target=self._start, name="aria-start", daemon=True).start()
        self.ui.every(2000, self._tick)

    def _start(self) -> None:
        try:
            # Loads PyTorch and the model: seconds, so not on the window's thread.
            from .device import describe
            from .serve import build_server
            server = build_server(host="127.0.0.1", port=(self.port, 0), app_mode=True)
        except BaseException as e:
            traceback.print_exc()
            self.ui.call(self._failed, e)
            return
        if self.quitting:
            server.close()
            return
        server.app.on_quit = lambda: self.ui.call(self.quit, True)
        self.server = server
        self.url = f"http://127.0.0.1:{server.port}"
        threading.Thread(target=server.httpd.serve_forever, name="aria-http",
                         daemon=True).start()
        _info_path(self.home).write_text(
            json.dumps({"url": self.url, "pid": os.getpid()}), encoding="utf-8")
        session = server.session
        print(f"Aria is running at {self.url}")
        print(f"  {server.label}, {session.model.num_params() / 1e6:.2f}M parameters, "
              f"on the {describe(session.device)}")
        print(f"  memory  {session.state_dir}")
        self.ui.call(self._ready)

    def _ready(self) -> None:
        self.ui.ready()
        self.ui.set_status(_describe([]))
        if self.browser:
            self.open_chat()

    def _failed(self, e: BaseException) -> None:
        self.ui.error("Aria couldn't start",
                      f"{type(e).__name__}: {e}\n\nThe details are in "
                      f"{self.home / LOG_NAME}")
        self.ui.close()

    def open_chat(self) -> None:
        if self.url and not self.quitting:
            open_in_browser(self.url)

    def _tick(self) -> None:
        if self.server is None or self.quitting:
            return
        self.ui.set_status(_describe(self.server.app.active_jobs()))
        if time.monotonic() - self._last_save > AUTOSAVE_SECONDS:
            self._last_save = time.monotonic()
            threading.Thread(target=self._autosave, name="aria-autosave",
                             daemon=True).start()

    def _autosave(self) -> None:
        try:
            if self.server.app.save_if_changed():
                print(f"{time.strftime('%H:%M')} saved what she has learned")
        except Exception:
            traceback.print_exc()

    def quit(self, asked: bool = False) -> None:
        """Save and stop. `asked`: the page's Quit button or the system asked,
        so don't ask again."""
        if self.quitting:
            return
        if self.server is not None and not asked:
            jobs = self.server.app.active_jobs()
            if jobs and not self.ui.confirm(
                    "Quit Aria?", f"She is still reading {jobs[0]['name']}. Quit anyway? "
                                  f"She keeps everything she has learned from it so far."):
                return
        self.quitting = True
        self.ui.busy("Saving what she has learned…")
        # Not a daemon: the save must finish even if the window goes first.
        threading.Thread(target=self._stop, name="aria-stop").start()

    def _stop(self) -> None:
        try:
            if self.server is not None:
                self.server.httpd.shutdown()
                self.server.close()
                print("saved; Aria has stopped")
        except Exception:
            traceback.print_exc()
        finally:
            try:
                _info_path(self.home).unlink(missing_ok=True)
            except OSError:
                pass
            self.ui.call(self.ui.close)


def _on_signals(ui, aria: AriaApp) -> None:
    """Ctrl-C, closing the terminal, or the system shutting down: save first."""
    def handler(signum, frame):
        if aria.quitting:
            print("still saving — one moment")
            return
        ui.call(aria.quit, True)
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def _make_ui(home: Path, window: bool):
    if window:
        try:
            return Window(home)
        except Exception as e:            # no Tk, or no screen
            print(f"(no window: {e}; running in this terminal instead)")
    return Console(home)


def run(port: int = PREFERRED_PORT, browser: bool = True, window: bool = True) -> int:
    home = data_dir()
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        show_error("Aria couldn't start", f"She has nowhere to keep her memory: {e}")
        return 1
    _log_to(home / LOG_NAME)
    lock = InstanceLock(home / "aria.lock")
    if not lock.acquire() and _hand_over(lock, home, browser):
        return 0
    try:
        _menu_entry()
        ui = _make_ui(home, window)
        aria = AriaApp(ui, home, port, browser)
        _on_signals(ui, aria)
        aria.start()
        ui.mainloop()
        return 0
    except Exception as e:
        traceback.print_exc()
        show_error("Aria couldn't start", f"{e}\n\nThe details are in {home / LOG_NAME}")
        return 1
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

SELF_TEST_TEXT = """\
The lighthouse keeper climbed the stairs every evening before the sun went down.
She polished the great lamp until it shone like a second moon.
On clear nights the beam reached the fishing boats far out in the bay.
The fishermen said they could steer home by it with their eyes closed.
In winter the storms came in from the west and shook the windows.
The keeper kept a kettle on the stove and a lantern by the door.
Her cat slept on the warm stones beside the chimney.
Every morning she wrote the weather in a blue notebook.
Some days the sea was calm and silver, and some days it was wild and grey.
When the supply boat came, she traded letters with her sister in the city.
Her sister wrote about trams and theatres and crowded markets.
The keeper wrote about gulls, tides and the colour of the sky.
Neither of them would have swapped places for anything.
"""


def _get(url: str, timeout: float = 60) -> bytes:
    with _LOCAL.open(url, timeout=timeout) as r:
        return r.read()


def _post(url: str, body: bytes, ctype: str = "application/json",
          timeout: float = 300) -> bytes:
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": ctype})
    with _LOCAL.open(req, timeout=timeout) as r:
        return r.read()


def self_test(require_window: bool = False) -> int:
    """Run Aria for real and check each step. Returns 0 if all pass."""
    temp = None
    if not os.environ.get("ARIA_HOME"):
        temp = tempfile.mkdtemp(prefix="aria-self-test-")
        os.environ["ARIA_HOME"] = temp
    home = data_dir()
    home.mkdir(parents=True, exist_ok=True)
    _log_to(home / LOG_NAME)
    print(f"Aria self-test: Python {sys.version.split()[0]} on {sys.platform}, "
          f"{'packaged app' if frozen() else 'from source'}; memory in {home}")
    failures: list[str] = []

    def check(what: str, fn: Callable[[], object]):
        t0 = time.monotonic()
        try:
            detail = fn()
        except BaseException as e:
            failures.append(what)
            print(f"FAIL  {what}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
        took = time.monotonic() - t0
        print(f"ok    {what}" + (f" — {detail}" if detail else "") + f" ({took:.1f}s)")
        return detail if detail is not None else True

    def window():
        try:
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            width = tkinter.PhotoImage(file=str(ICON)).width()
            root.update()
            root.destroy()
            return f"Tk {tkinter.TkVersion}, icon {width}px"
        except Exception as e:
            if require_window:
                raise
            return f"skipped ({type(e).__name__}: {e})"

    check("window", window)

    def pytorch():
        import torch
        return f"PyTorch {torch.__version__}"

    check("load PyTorch", pytorch)
    quit_asked = threading.Event()
    box: dict = {}

    def start():
        from .device import describe
        from .serve import build_server
        server = build_server(host="127.0.0.1", port=0, app_mode=True)
        server.app.on_quit = quit_asked.set
        threading.Thread(target=server.httpd.serve_forever, daemon=True).start()
        box["server"] = server
        box["url"] = f"http://127.0.0.1:{server.port}"
        s = server.session
        return (f"{server.label}, {s.model.num_params() / 1e6:.2f}M parameters "
                f"on the {describe(s.device)}")

    if not check("load the model and start the server", start):
        return _finish(failures, temp)
    url, server = box["url"], box["server"]

    def page():
        html = _get(url + "/").decode("utf-8")
        assert "<title>Aria</title>" in html and 'id="quit"' in html, "not the app's page"
        assert json.loads(_get(url + "/api/hello"))["app"] == "aria"
        return f"{len(html):,} bytes"

    def chat():
        raw = _post(url + "/api/chat",
                    json.dumps({"message": "Hello Aria! How are you today?"}).encode())
        events = [json.loads(line[6:]) for line in raw.decode("utf-8").split("\n\n")
                  if line.startswith("data: ")]
        errors = [e["error"] for e in events if "error" in e]
        assert not errors, errors
        reply = "".join(e.get("token", "") for e in events).strip()
        assert reply, "no reply"
        assert any("learn" in e for e in events), "no learning report"
        return repr(reply[:60])

    def document():
        job = json.loads(_post(url + "/api/upload?name=lighthouse.txt",
                               SELF_TEST_TEXT.encode(), "application/octet-stream"))["job"]
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            j = next(j for j in json.loads(_get(url + "/api/jobs"))["jobs"]
                     if j["id"] == job["id"])
            if j["state"] not in ("queued", "running"):
                assert j["state"] == "done", j.get("error") or j["state"]
                return j["summary"]
            time.sleep(0.5)
        raise TimeoutError("still reading after 15 minutes")

    def quit_():
        assert json.loads(_post(url + "/api/quit", b"{}"))["ok"]
        assert quit_asked.wait(10), "the Quit button did nothing"
        server.httpd.shutdown()
        server.close()

    state_dir = server.session.state_dir
    checkpoint = server.session.checkpoint
    updates = 0

    def saved():
        nonlocal updates
        assert (state_dir / "learned.pt").exists(), f"nothing saved in {state_dir}"
        from .chat import ChatSession
        s = ChatSession(checkpoint=checkpoint, state_dir=state_dir)
        assert s.restored, "the saved learning didn't load"
        updates = s.learner.updates_applied
        assert updates > 0, "no learning was saved"
        return f"{updates} updates reloaded from {state_dir}"

    for what, fn in (("the chat page", page), ("chat", chat),
                     ("read a document", document), ("quit from the page", quit_),
                     ("learning saved and reloaded", saved)):
        check(what, fn)
    return _finish(failures, temp)


def _finish(failures: list[str], temp: str | None) -> int:
    if failures:
        print(f"self-test FAILED: {', '.join(failures)}")
    else:
        print("self-test passed")
    if temp:
        shutil.rmtree(temp, ignore_errors=True)
    return 1 if failures else 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "aria app", description="Run Aria as an app: a small window, and her chat "
                                "in your web browser.")
    p.add_argument("--port", type=int, default=PREFERRED_PORT,
                   help=f"port to use if it's free (default {PREFERRED_PORT}; "
                        f"otherwise any free one)")
    p.add_argument("--no-browser", action="store_true",
                   help="don't open the chat in the browser")
    p.add_argument("--no-window", action="store_true",
                   help="run in this terminal instead of a window (Ctrl-C quits)")
    p.add_argument("--self-test", action="store_true",
                   help="start Aria in a temporary folder, check that chatting, "
                        "reading and saving all work, and quit")
    p.add_argument("--require-window", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    # macOS may pass a -psn_... argument to an app it launches; ignore extras.
    args, _ = build_parser().parse_known_args(argv)
    if args.self_test:
        return self_test(require_window=args.require_window)
    return run(port=args.port, browser=not args.no_browser, window=not args.no_window)


if __name__ == "__main__":
    sys.exit(main())
