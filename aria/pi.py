"""Aria on a Raspberry Pi: always on, for every device on your network.

    python -m aria.pi

is what raspberry-pi/install.sh sets up to start whenever the Pi turns on.
It serves her chat to the computers and phones on your network, at
http://<the Pi's name>.local:8000. It is separate from the desktop app: no
window and no Quit button, because it's meant to keep running, and unlike
`aria serve` it saves what she has learned every five minutes and whenever it
is stopped, including when the Pi shuts down or reboots.

It answers only requests addressed to this machine: its name, its name with
.local, one of its own network addresses, or a name given with --allow-host
(a Tailscale name, say). That, with the checks in aria.serve, is what stops
other web pages from talking to her. There is no password: anyone on your
network can talk to her, and what they say can end up in her memory, so
don't make it reachable from the internet.
"""

from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

from .paths import data_dir

DEFAULT_PORT = 8000
AUTOSAVE_SECONDS = 300
ADDRESS_REFRESH_SECONDS = 60


def local_addresses() -> set[str]:
    """This machine's IP addresses, on every network it's connected to."""
    found = {"127.0.0.1", "::1"}
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True,
                             timeout=5).stdout
        found.update(out.split())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        # The address used to reach other machines. Connecting a UDP socket
        # sends nothing; it only picks the route.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.add(s.getsockname()[0])
    except OSError:
        pass
    return {a.split("%")[0].lower() for a in found}


class Names:
    """The names requests may be addressed to. The addresses are looked up
    again every minute, because a Pi's address can change while it runs."""

    def __init__(self, extra: tuple[str, ...] = ()) -> None:
        host = socket.gethostname().lower()
        short = host.split(".")[0]
        self.fixed = {"localhost", host, short, f"{short}.local"} | \
            {h.strip().lower() for h in extra if h.strip()}
        self._addresses: set[str] = set()
        self._when = float("-inf")
        self._lock = threading.Lock()

    def current(self) -> frozenset[str]:
        with self._lock:
            if time.monotonic() - self._when >= ADDRESS_REFRESH_SECONDS:
                self._addresses = local_addresses()
                self._when = time.monotonic()
            return frozenset(self.fixed | self._addresses)

    def addresses_to_visit(self, port: int) -> list[str]:
        """What to type into a browser on another device."""
        here = self.current()
        names = sorted(n for n in self.fixed if n.endswith(".local"))
        ips = sorted(a for a in here - self.fixed
                     if a not in ("127.0.0.1", "::1") and ":" not in a)
        return [f"http://{n}:{port}" for n in names + ips]


class PiServer:
    """The chat server, plus saving every few minutes and on the way out."""

    def __init__(self, port: int = DEFAULT_PORT, allow_hosts: tuple[str, ...] = (),
                 host: str = "0.0.0.0", **session_kwargs) -> None:
        from .serve import build_server
        self.names = Names(allow_hosts)
        self.server = build_server(host=host, port=port, **session_kwargs)
        names = self.names

        class Handler(self.server.httpd.RequestHandlerClass):
            def _trusted(self) -> bool:
                self.allowed_hosts = names.current()
                return super()._trusted()

        self.server.httpd.RequestHandlerClass = Handler
        self._stop = threading.Event()
        self._serving = False

    @property
    def port(self) -> int:
        return self.server.port

    def start(self) -> None:
        self._serving = True
        threading.Thread(target=self.server.httpd.serve_forever, name="aria-http",
                         daemon=True).start()
        threading.Thread(target=self._autosave_every, name="aria-autosave",
                         daemon=True).start()

    def _autosave_every(self) -> None:
        while not self._stop.wait(AUTOSAVE_SECONDS):
            self.autosave()

    def autosave(self) -> list[str]:
        try:
            saved = self.server.app.save_if_changed()
        except Exception:
            traceback.print_exc()
            return []
        if saved:
            print(f"{time.strftime('%Y-%m-%d %H:%M')} saved what she has learned",
                  flush=True)
        return saved

    def stop(self) -> None:
        """Stop answering, let any learning step finish, and save."""
        if self._stop.is_set():
            return
        self._stop.set()
        if self._serving:
            self.server.httpd.shutdown()
        self.server.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        "python -m aria.pi",
        description="Run Aria all the time on a Raspberry Pi (or any always-on "
                    "Linux machine), for every device on your network.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"port to serve on (default {DEFAULT_PORT})")
    p.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                   help="another name she may be reached by, such as a Tailscale "
                        "name (repeatable)")
    args = p.parse_args(argv)

    home = data_dir()
    home.mkdir(parents=True, exist_ok=True)
    from .app import InstanceLock
    lock = InstanceLock(home / "aria.lock")
    if not lock.acquire():
        print("Aria is already running on this machine, using the same memory "
              "(as the service, perhaps: systemctl status aria).", file=sys.stderr)
        return 1

    try:
        pi = PiServer(args.port, tuple(args.allow_host))
    except OSError as e:
        lock.release()
        print(f"couldn't start on port {args.port}: {e}", file=sys.stderr)
        return 1

    stop = threading.Event()

    def on_signal(signum, frame):
        stop.set()

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        signal.signal(getattr(signal, name), on_signal)

    pi.start()
    from .device import describe
    session = pi.server.session
    print(f"Aria is running: {pi.server.label}, "
          f"{session.model.num_params() / 1e6:.2f}M parameters, on the "
          f"{describe(session.device)}", flush=True)
    print(f"  memory  {session.state_dir}")
    print("  on your other devices (same network), open:")
    for url in pi.names.addresses_to_visit(pi.port):
        print(f"    {url}")
    print(f"  she saves what she learns every {AUTOSAVE_SECONDS // 60} minutes "
          f"and when she's stopped", flush=True)
    try:
        while not stop.wait(1):
            pass
    finally:
        print("stopping: saving what she has learned ...", flush=True)
        pi.stop()
        lock.release()
        print("saved; Aria has stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
