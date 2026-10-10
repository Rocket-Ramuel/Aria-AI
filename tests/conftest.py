"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def isolated_aria_home(tmp_path, monkeypatch):
    """Every test gets its own Aria data folder, so nothing is ever written to
    the real per-user folder (~/Library/Application Support/Aria and co.)."""
    home = tmp_path / "aria-home"
    monkeypatch.setenv("ARIA_HOME", str(home))
    return home
