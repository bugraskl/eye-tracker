"""Tests for releasing the camera to other apps (engine/camera_yield.py)."""

from __future__ import annotations

import pytest

from eye_tracker.engine.camera_yield import (
    YieldInputs,
    matching_app,
    normalize_process_name,
    should_yield,
)


def test_nothing_to_yield_for() -> None:
    inputs = YieldInputs(camera_in_use_by_other=False, running={"explorer.exe"}, pause_for_apps=[])
    assert should_yield(inputs, yield_camera=True) == (False, "")


def test_camera_in_use_yields_when_enabled() -> None:
    inputs = YieldInputs(camera_in_use_by_other=True, running=set(), pause_for_apps=[])
    ok, reason = should_yield(inputs, yield_camera=True)
    assert ok
    assert "camera" in reason.lower()


def test_camera_in_use_ignored_when_disabled() -> None:
    inputs = YieldInputs(camera_in_use_by_other=True, running=set(), pause_for_apps=[])
    assert should_yield(inputs, yield_camera=False) == (False, "")


def test_unknown_camera_usage_does_not_yield() -> None:
    inputs = YieldInputs(camera_in_use_by_other=None, running=set(), pause_for_apps=[])
    assert should_yield(inputs, yield_camera=True) == (False, "")


def test_listed_app_yields_even_when_camera_yield_is_off() -> None:
    inputs = YieldInputs(None, {"zoom.exe", "code.exe"}, ["zoom.exe"])
    assert should_yield(inputs, yield_camera=False) == (True, "zoom.exe is running")


def test_listed_app_reported_before_camera_usage() -> None:
    inputs = YieldInputs(True, {"obs64.exe"}, ["obs64.exe"])
    assert should_yield(inputs, yield_camera=True) == (True, "obs64.exe is running")


@pytest.mark.parametrize(
    ("configured", "running"),
    [
        ("zoom.exe", "zoom.exe"),
        ("Zoom.EXE", "zoom.exe"),
        ("zoom", "zoom.exe"),
        ("zoom.exe", "zoom"),
        ("Zoom.app", "zoom"),
        ("  obs64.exe ", "obs64.exe"),
        (r"C:\Program Files\obs-studio\bin\64bit\obs64.exe", "obs64.exe"),
        ("/Applications/OBS.app", "obs"),
    ],
)
def test_app_matching_is_forgiving(configured: str, running: str) -> None:
    assert matching_app({running}, [configured]) == configured.strip()


@pytest.mark.parametrize(
    ("configured", "running"),
    [("zoom", "zoomit.exe"), ("", "zoom.exe"), ("   ", "zoom.exe"), ("code", "vscode.exe")],
)
def test_app_matching_is_exact_on_the_name(configured: str, running: str) -> None:
    assert matching_app({running}, [configured]) is None


def test_first_listed_app_wins() -> None:
    running = {"a.exe", "b.exe"}
    assert matching_app(running, ["b.exe", "a.exe"]) == "b.exe"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Teams.exe", "teams"),
        ("FaceTime.app", "facetime"),
        (".exe", ".exe"),
        ("python3.12", "python3.12"),
        ("dir/sub\\Tool.EXE", "tool"),
    ],
)
def test_normalize_process_name(name: str, expected: str) -> None:
    assert normalize_process_name(name) == expected
