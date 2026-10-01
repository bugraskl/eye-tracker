"""Tests for eye_tracker.config: migrations of settings written by earlier versions."""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from eye_tracker.config import (
    CONFIG_VERSION,
    LEGACY_HOTKEYS,
    LEGACY_HOTKEYS_X11,
    HotkeySettings,
    Settings,
    describe_settings,
)

HOTKEY_FIELDS = ("toggle_tracking", "toggle_privacy", "recalibrate")
LEGACY_TRIO = dict(zip(HOTKEY_FIELDS, LEGACY_HOTKEYS, strict=True))
LEGACY_X11_TRIO = dict(zip(HOTKEY_FIELDS, LEGACY_HOTKEYS_X11, strict=True))
PLATFORMS = ["win32", "darwin", "linux", "freebsd14"]
#: What a missing, malformed or old "version" key looks like in a settings file.
OLD_VERSIONS: list[dict[str, Any]] = [
    {},
    {"version": 1},
    {"version": 0},
    {"version": "2"},
    {"version": True},
    {"version": None},
]


def trio(settings: Settings) -> tuple[str, str, str]:
    hotkeys = settings.hotkeys
    return (hotkeys.toggle_tracking, hotkeys.toggle_privacy, hotkeys.recalibrate)


def default_trio() -> tuple[str, str, str]:
    defaults = HotkeySettings()
    return (defaults.toggle_tracking, defaults.toggle_privacy, defaults.recalibrate)


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


def test_legacy_backend_names_are_renamed_in_files_of_every_version() -> None:
    for version in (1, CONFIG_VERSION, CONFIG_VERSION + 1):
        data = {"version": version, "general": {"backend": "opencv"}}
        assert Settings.from_dict(data).general.backend == "lite"


def test_current_and_unknown_backend_names_are_not_migrated() -> None:
    assert Settings.from_dict({"general": {"backend": "lite"}}).general.backend == "lite"
    assert Settings.from_dict({"general": {"backend": "tflite"}}).general.backend == "auto"
    assert Settings.from_dict({"general": {"backend": 3}}).general.backend == "auto"
    assert Settings.from_dict({"general": "broken"}).general.backend == "auto"


def test_the_file_format_version() -> None:
    assert CONFIG_VERSION == 2
    assert Settings().to_dict()["version"] == CONFIG_VERSION
    # Loading an old file stamps today's version, so the next save writes it.
    assert Settings.from_dict({"version": 1}).version == CONFIG_VERSION
    assert Settings.from_dict({}).version == CONFIG_VERSION


@pytest.mark.parametrize("platform", PLATFORMS)
def test_every_platform_defaults_to_ctrl_alt_meta(
    platform: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert default_trio() == ("ctrl+alt+meta+t", "ctrl+alt+meta+p", "ctrl+alt+meta+c")


@pytest.mark.parametrize("version", OLD_VERSIONS)
@pytest.mark.parametrize("platform", PLATFORMS)
def test_the_old_default_hotkeys_become_todays_defaults(
    platform: str, version: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    data = {**version, "hotkeys": {"enabled": False, **LEGACY_TRIO}}
    before = copy.deepcopy(data)
    settings = Settings.from_dict(data)
    assert trio(settings) == default_trio()
    assert settings.hotkeys.toggle_tracking == "ctrl+alt+meta+t"
    assert settings.hotkeys.enabled is False  # other hotkey settings are kept
    assert data == before


def test_old_defaults_are_recognised_regardless_of_case_and_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    data = {"hotkeys": {name: f" {combo.upper()} " for name, combo in LEGACY_TRIO.items()}}
    assert Settings.from_dict(data).hotkeys.toggle_tracking == "ctrl+alt+meta+t"


@pytest.mark.parametrize("platform", ["linux", "freebsd14"])
def test_the_first_x11_defaults_are_migrated(
    platform: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ctrl+Alt+Shift+T/P/C was the Linux default of later version 1 builds."""
    monkeypatch.setattr(sys, "platform", platform)
    with caplog.at_level(logging.INFO, logger="eye_tracker.config"):
        settings = Settings.from_dict({"version": 1, "hotkeys": dict(LEGACY_X11_TRIO)})
    assert trio(settings) == default_trio()
    assert "ctrl+alt+meta+t" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_the_x11_defaults_are_a_choice_elsewhere(
    platform: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+Alt+Shift+T/P/C never was a Windows or macOS default: the user picked it."""
    monkeypatch.setattr(sys, "platform", platform)
    settings = Settings.from_dict({"version": 1, "hotkeys": dict(LEGACY_X11_TRIO)})
    assert trio(settings) == LEGACY_HOTKEYS_X11


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("stored", [LEGACY_TRIO, LEGACY_X11_TRIO])
def test_old_defaults_chosen_in_a_current_file_are_kept(
    platform: str,
    stored: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A version 2 file holds what the user chose, even if it equals an old default."""
    monkeypatch.setattr(sys, "platform", platform)
    with caplog.at_level(logging.INFO, logger="eye_tracker.config"):
        settings = Settings.from_dict({"version": CONFIG_VERSION, "hotkeys": dict(stored)})
    assert trio(settings) == tuple(stored.values())
    assert "Hotkeys" not in caplog.text
    # A file from a newer version is not second-guessed either.
    newer = Settings.from_dict({"version": CONFIG_VERSION + 1, "hotkeys": dict(stored)})
    assert trio(newer) == tuple(stored.values())


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_chosen_old_default_survives_restarts(
    platform: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user re-records Ctrl+Alt+T/P/C in Settings: it must not be reset at login."""
    monkeypatch.setattr(sys, "platform", platform)
    path = tmp_path / "settings.json"
    settings = Settings()
    for name, combo in LEGACY_TRIO.items():
        setattr(settings.hotkeys, name, combo)
    settings.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == CONFIG_VERSION
    for _restart in range(3):
        reloaded = Settings.load(path)
        assert trio(reloaded) == LEGACY_HOTKEYS
        reloaded.save(path)


@pytest.mark.parametrize("changed", HOTKEY_FIELDS)
def test_hotkeys_the_user_changed_are_kept(changed: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the complete, untouched legacy trio is migrated."""
    monkeypatch.setattr(sys, "platform", "win32")
    stored = {**LEGACY_TRIO, changed: "ctrl+shift+f9"}
    hotkeys = Settings.from_dict({"version": 1, "hotkeys": stored}).hotkeys
    assert (hotkeys.toggle_tracking, hotkeys.toggle_privacy, hotkeys.recalibrate) == (
        stored["toggle_tracking"],
        stored["toggle_privacy"],
        stored["recalibrate"],
    )


def test_a_mixed_old_trio_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two defaults of one build and one of the other: somebody chose that."""
    monkeypatch.setattr(sys, "platform", "linux")
    stored = {**LEGACY_TRIO, "recalibrate": LEGACY_X11_TRIO["recalibrate"]}
    settings = Settings.from_dict({"version": 1, "hotkeys": stored})
    assert trio(settings) == tuple(stored.values())


def test_a_partial_legacy_trio_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    stored = {"toggle_tracking": "ctrl+alt+t", "toggle_privacy": "ctrl+alt+p"}
    hotkeys = Settings.from_dict({"version": 1, "hotkeys": stored}).hotkeys
    assert hotkeys.toggle_tracking == "ctrl+alt+t"
    assert hotkeys.toggle_privacy == "ctrl+alt+p"
    assert hotkeys.recalibrate == HotkeySettings().recalibrate  # missing: the default


def test_malformed_hotkeys_in_an_old_file_are_not_migrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert Settings.from_dict({"version": 1, "hotkeys": "ctrl+alt+t"}).hotkeys == HotkeySettings()
    stored = {**LEGACY_TRIO, "recalibrate": 3}
    hotkeys = Settings.from_dict({"version": 1, "hotkeys": stored}).hotkeys
    assert hotkeys.toggle_tracking == "ctrl+alt+t"  # not a complete trio: left alone
    assert hotkeys.recalibrate == HotkeySettings().recalibrate  # wrong type: the default


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_migrated_file_round_trips(
    platform: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"version": 1, "general": {"backend": "opencv"}, "hotkeys": LEGACY_TRIO}),
        encoding="utf-8",
    )
    migrated = Settings.load(path)
    assert migrated.general.backend == "lite"
    assert migrated.hotkeys == HotkeySettings()
    migrated.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == CONFIG_VERSION
    reloaded = Settings.load(path)
    assert reloaded.general.backend == "lite"
    assert reloaded.hotkeys == HotkeySettings()


def test_an_old_file_that_is_never_saved_migrates_on_every_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Until something is saved the file still says version 1 and still holds the
    old defaults, so every load reaches the same result: the migration is stable."""
    monkeypatch.setattr(sys, "platform", "darwin")
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"version": 1, "hotkeys": LEGACY_TRIO}), encoding="utf-8")
    first = Settings.load(path)
    second = Settings.load(path)
    assert first.hotkeys == second.hotkeys == HotkeySettings()
    assert trio(first) == ("ctrl+alt+meta+t", "ctrl+alt+meta+p", "ctrl+alt+meta+c")


def test_camera_device_documentation_mentions_stable_links_and_offline() -> None:
    doc = next(row["doc"] for row in describe_settings() if row["key"] == "camera.device")
    assert "/dev/v4l/by-id" in doc
    assert "URLs" in doc
    assert "offline" in doc
