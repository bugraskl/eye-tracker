"""Tests for eye_tracker.paths (the --config-dir profile override)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from eye_tracker import paths
from eye_tracker.diagnostics import _profile_override
from eye_tracker.platform import autostart


@pytest.fixture(autouse=True)
def _reset_override() -> Iterator[None]:
    yield
    paths.set_base_override(None)


def test_base_override_is_none_for_the_default_profile() -> None:
    paths.set_base_override(None)
    assert paths.base_override() is None
    assert not _profile_override()
    assert autostart._active_profile() is None


def test_base_override_is_the_resolved_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    paths.set_base_override(Path("portable"))
    expected = (tmp_path / "portable").resolve()
    assert paths.base_override() == expected
    assert paths.config_dir() == expected
    # Autostart entries and the doctor report use the same public accessor.
    assert autostart._active_profile() == expected
    assert _profile_override()
