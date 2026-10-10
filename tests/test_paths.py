"""Where Aria keeps her memory, and moving it there from older locations."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from aria import paths


def test_each_system_has_its_standard_folder(tmp_path):
    std = paths.standard_data_dir
    assert std("darwin", tmp_path, {}) == tmp_path / "Library" / "Application Support" / "Aria"
    assert std("linux", tmp_path, {}) == tmp_path / ".local" / "share" / "aria"
    assert std("linux", tmp_path, {"XDG_DATA_HOME": str(tmp_path / "xdg")}) == tmp_path / "xdg" / "aria"
    roaming = tmp_path / "Roaming"
    assert std("win32", tmp_path, {"APPDATA": str(roaming)}) == roaming / "Aria"
    assert std("win32", tmp_path, {}) == tmp_path / "AppData" / "Roaming" / "Aria"


def test_without_aria_home_the_standard_folder_is_used(monkeypatch):
    monkeypatch.delenv("ARIA_HOME")
    assert paths.data_dir() == paths.standard_data_dir(sys.platform, Path.home(), os.environ)


def test_aria_home_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("ARIA_HOME", str(tmp_path / "synced"))
    assert paths.data_dir() == tmp_path / "synced"


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    """A copy of the shipped checkpoint in a folder of its own."""
    root = tmp_path / "Aria-AI"
    ckpt = root / "checkpoints" / "aria-small.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"model")
    monkeypatch.chdir(root)
    return ckpt


def test_shipped_model_memory_lives_in_the_data_folder(shipped, isolated_aria_home):
    assert paths.memory_dir(shipped) == isolated_aria_home / "online"


def test_other_models_keep_memory_beside_them(tmp_path):
    mine = tmp_path / "runs" / "aria" / "base.pt"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"x")
    assert paths.memory_dir(mine) == mine.parent / "online"


def test_old_memory_is_moved_once(shipped, isolated_aria_home):
    old = shipped.parent / "online"
    old.mkdir()
    (old / "replay.json").write_text("{}", encoding="utf-8")
    new = paths.memory_dir(shipped)
    assert (new / "replay.json").exists() and not old.exists()
    # A second, newer copy appearing later doesn't overwrite what's there.
    old.mkdir()
    (old / "replay.json").write_text('{"old": 1}', encoding="utf-8")
    assert paths.memory_dir(shipped) == new
    assert (new / "replay.json").read_text(encoding="utf-8") == "{}"


def test_memory_that_cannot_be_moved_is_used_in_place(shipped, monkeypatch):
    old = shipped.parent / "online"
    old.mkdir()
    (old / "learned.pt").write_bytes(b"w")

    def refuse(*a, **k):
        raise PermissionError("read-only")
    monkeypatch.setattr(shutil, "move", refuse)
    assert paths.memory_dir(shipped) == old


def test_an_old_blank_model_is_adopted(tmp_path, monkeypatch, isolated_aria_home):
    monkeypatch.chdir(tmp_path)
    old = Path("runs") / "blank"
    (old / "online").mkdir(parents=True)
    (old / "base.pt").write_bytes(b"blank")
    paths.adopt_old_blank()
    assert paths.blank_checkpoint().read_bytes() == b"blank"
    assert (isolated_aria_home / "blank" / "online").is_dir()


def test_memory_files_are_utf8_on_every_system(tmp_path):
    """A memory written on Windows must read back on a Mac, and vice versa:
    the files are UTF-8 whatever the system's default encoding."""
    from aria.memory import Journal, ReplayBuffer
    buf = ReplayBuffer(capacity=4)
    buf.add(["café? \U0001f600 ñandú", "naïve façade"])
    buf.save(tmp_path / "replay.json")
    raw = (tmp_path / "replay.json").read_bytes()
    assert "café? \U0001f600".encode("utf-8") in raw
    j = Journal(tmp_path / "journal.jsonl")
    j.write(event="update", turns=["Zoë"])
    assert json.loads((tmp_path / "journal.jsonl").read_bytes().decode("utf-8"))["turns"] == ["Zoë"]
    assert ReplayBuffer.load(tmp_path / "replay.json").items[0]["turns"][1] == "naïve façade"
