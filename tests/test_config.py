"""Tests for eye_tracker.config: migrations of settings written by earlier versions."""

from __future__ import annotations

import copy
import logging
import sys
from typing import Any

import pytest

from eye_tracker.config import LEGACY_HOTKEYS, HotkeySettings, Settings, describe_settings

LEGACY_TRIO = dict(
    zip(("toggle_tracking", "toggle_privacy", "recalibrate"), LEGACY_HOTKEYS, strict=True)
)


@pytest.mark.parametrize(("stored", "expected"), [("mediapipe", "facemesh"), ("opencv", "lite")])
def test_legacy_backend_names_are_renamed(
    stored: str, expected: str, caplog: pytest.LogCaptureFixture
) -> None:
    data: dict[str, Any] = {"general": {"backend": stored, "notifications": False}}
    before = copy.deepcopy(data)
    with caplog.at_level(logging.INFO, logger="eye_tracker.config"):
        settings = Settings.from_dict(data)
    assert settings.general.backend == expected
    assert settings.general.notifications is False  # the rest of the section is kept
    assert data == before  # the caller's dict is not modified
    assert any(
        r.levelno == logging.INFO and stored in r.getMessage() and expected in r.getMessage()
        for r in caplog.records
    )
    # Nothing is logged as a warning: the old name is valid, just renamed.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_current_and_unknown_backend_names_are_not_migrated() -> None:
    assert Settings.from_dict({"general": {"backend": "lite"}}).general.backend == "lite"
    assert Settings.from_dict({"general": {"backend": "tflite"}}).general.backend == "auto"
    assert Settings.from_dict({"general": {"backend": 3}}).general.backend == "auto"
    assert Settings.from_dict({"general": "broken"}).general.backend == "auto"


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_the_old_default_hotkeys_become_todays_defaults(
    platform: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    data = {"hotkeys": {"enabled": False, **LEGACY_TRIO}}
    before = copy.deepcopy(data)
    hotkeys = Settings.from_dict(data).hotkeys
    defaults = HotkeySettings()
    assert (hotkeys.toggle_tracking, hotkeys.toggle_privacy, hotkeys.recalibrate) == (
        defaults.toggle_tracking,
        defaults.toggle_privacy,
        defaults.recalibrate,
    )
    assert hotkeys.enabled is False  # other hotkey settings are kept
    assert data == before
    if platform == "win32":
        assert hotkeys.toggle_tracking == "ctrl+alt+meta+t"  # no AltGr collision


def test_old_defaults_are_recognised_regardless_of_case_and_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    data = {"hotkeys": {name: f" {combo.upper()} " for name, combo in LEGACY_TRIO.items()}}
    assert Settings.from_dict(data).hotkeys.toggle_tracking == "ctrl+alt+shift+t"


@pytest.mark.parametrize("changed", ["toggle_tracking", "toggle_privacy", "recalibrate"])
def test_hotkeys_the_user_changed_are_kept(changed: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the complete, untouched legacy trio is migrated."""
    monkeypatch.setattr(sys, "platform", "win32")
    stored = {**LEGACY_TRIO, changed: "ctrl+shift+f9"}
    hotkeys = Settings.from_dict({"hotkeys": stored}).hotkeys
    assert (hotkeys.toggle_tracking, hotkeys.toggle_privacy, hotkeys.recalibrate) == (
        stored["toggle_tracking"],
        stored["toggle_privacy"],
        stored["recalibrate"],
    )


def test_a_partial_legacy_trio_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    stored = {"toggle_tracking": "ctrl+alt+t", "toggle_privacy": "ctrl+alt+p"}
    hotkeys = Settings.from_dict({"hotkeys": stored}).hotkeys
    assert hotkeys.toggle_tracking == "ctrl+alt+t"
    assert hotkeys.toggle_privacy == "ctrl+alt+p"
    assert hotkeys.recalibrate == HotkeySettings().recalibrate  # missing: the default


def test_a_migrated_file_round_trips(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    path = tmp_path / "settings.json"
    Settings.from_dict({"general": {"backend": "opencv"}, "hotkeys": LEGACY_TRIO}).save(path)
    reloaded = Settings.load(path)
    assert reloaded.general.backend == "lite"
    assert reloaded.hotkeys == HotkeySettings()


def test_camera_device_documentation_mentions_stable_links_and_offline() -> None:
    doc = next(row["doc"] for row in describe_settings() if row["key"] == "camera.device")
    assert "/dev/v4l/by-id" in doc
    assert "URLs" in doc
    assert "offline" in doc


def test_old_defaults_equal_to_todays_are_left_alone_quietly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """On macOS Control+Option+T/P/C is still the default: no migration to report."""
    monkeypatch.setattr(sys, "platform", "darwin")
    with caplog.at_level(logging.INFO, logger="eye_tracker.config"):
        hotkeys = Settings.from_dict({"hotkeys": dict(LEGACY_TRIO)}).hotkeys
    assert hotkeys == HotkeySettings()
    assert "Hotkeys" not in caplog.text
