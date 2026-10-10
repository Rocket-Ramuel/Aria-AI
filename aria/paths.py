"""Where Aria keeps things on disk.

The program and its memory are kept apart. The program — the code and the
model she ships with — can live anywhere: a downloaded folder, an installed
app (which on a Mac can't be written to), a fresh copy after an update. Her
memory lives in one per-user folder, the standard place on each system:

    macOS     ~/Library/Application Support/Aria
    Windows   %APPDATA%\\Aria
    Linux     ~/.local/share/aria   (or $XDG_DATA_HOME/aria)

so updating or re-downloading Aria never touches what she has learned. Set
ARIA_HOME to use another folder (a synced one, say).

Memory written by older versions next to the checkpoint (`checkpoints/online`,
`runs/blank`) is moved here the first time it is needed.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SHIPPED_NAME = Path("checkpoints") / "aria-small.pt"


def data_dir() -> Path:
    """Aria's per-user folder (not created here)."""
    env = os.environ.get("ARIA_HOME")
    if env:
        return Path(env).expanduser()
    return standard_data_dir(sys.platform, Path.home(), os.environ)


def standard_data_dir(platform: str, home: Path, env) -> Path:
    """The usual per-user data folder on `platform` (a `sys.platform` value)."""
    if platform == "darwin":
        return home / "Library" / "Application Support" / "Aria"
    if platform.startswith(("win", "cygwin")):
        return Path(env.get("APPDATA") or home / "AppData" / "Roaming") / "Aria"
    return Path(env.get("XDG_DATA_HOME") or home / ".local" / "share") / "aria"


def frozen() -> bool:
    """Running as a packaged app (PyInstaller) rather than from source."""
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Where the files Aria ships with are: the app bundle, or the repository."""
    if frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def shipped_checkpoint_candidates() -> list[Path]:
    # The current directory first (running from the repository), then the
    # app bundle or the repository the package was imported from, so the
    # shipped model is found wherever Aria is started.
    return [SHIPPED_NAME, resource_root() / SHIPPED_NAME]


def shipped_checkpoint() -> Path | None:
    for p in shipped_checkpoint_candidates():
        if p.exists():
            return p
    return None


def is_shipped(checkpoint: str | Path) -> bool:
    target = Path(checkpoint).resolve()
    return any(p.exists() and p.resolve() == target for p in shipped_checkpoint_candidates())


def blank_checkpoint() -> Path:
    return data_dir() / "blank" / "base.pt"


def memory_dir(checkpoint: str | Path) -> Path:
    """Where a checkpoint's learned weights and memories live.

    The shipped model's memory is in the per-user folder. A model you trained
    yourself keeps its memory next to it, so two different base models never
    share — and corrupt — one memory."""
    checkpoint = Path(checkpoint)
    beside = checkpoint.parent / "online"
    if is_shipped(checkpoint):
        new = data_dir() / "online"
        # If older memory can't be moved, keep using it where it is rather
        # than start over with an empty one.
        return new if _adopt(beside, new) else beside
    return beside


def adopt_old_blank() -> None:
    """Move a blank model made by an older version (./runs/blank) into the
    per-user folder."""
    old = Path("runs") / "blank"
    if (old / "base.pt").exists():
        _adopt(old, blank_checkpoint().parent)


def _has_memory(folder: Path) -> bool:
    return folder.is_dir() and any(folder.iterdir())


def _adopt(old: Path, new: Path) -> bool:
    """Move `old` to `new` if `new` has nothing in it yet.

    Returns False only when there is memory at `old` that couldn't be moved."""
    if not _has_memory(old) or _has_memory(new) or old.resolve() == new.resolve():
        return True
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            new.rmdir()
        shutil.move(str(old), str(new))
    except OSError as e:          # read-only location, permissions, a full disk
        print(f"aria: couldn't move Aria's memory from {old} ({e}); "
              f"using it where it is", file=sys.stderr)
        return False
    print(f"aria: moved Aria's memory from {old} to {new}, where updates "
          f"won't touch it", file=sys.stderr)
    return True
