"""Tests for the default (unsupported-everywhere) PlatformServices contract."""

from __future__ import annotations

import pytest

from eye_tracker.platform.base import PlatformServices, process_basename
from eye_tracker.types import Rect, WindowRef


def test_defaults_are_safe_no_ops() -> None:
    services = PlatformServices()
    # Only macOS knows the Accessibility permission in detail.
    assert services.accessibility_status() == "unknown"
    # App Nap exists only on macOS: elsewhere nothing throttles us, so "in effect".
    assert services.set_background_activity(True) is True
    assert services.set_background_activity(False) is True
    assert services.camera_in_use_by_other_app() is None
    assert services.is_window_valid(WindowRef(handle=1, pid=2, rect=Rect(0, 0, 10, 10))) is False
    assert services.lock_screen() is False
    assert services.open_permission_settings("camera") is False


def test_contract_documents_the_desktop_and_camera_semantics() -> None:
    valid_doc = PlatformServices.is_window_valid.__doc__ or ""
    assert "virtual desktop" in valid_doc
    assert "Space" in valid_doc
    camera_doc = PlatformServices.camera_in_use_by_other_app.__doc__ or ""
    assert "refused" in camera_doc
    assert "cache" in camera_doc
    assert "unknown" in camera_doc
    status_doc = PlatformServices.accessibility_status.__doc__ or ""
    for value in ("granted", "missing", "stale", "unknown"):
        assert value in status_doc


def test_pane_queries_are_unsupported_by_default() -> None:
    services = PlatformServices()
    ref = WindowRef(handle=1, pid=2, rect=Rect(0, 0, 10, 10))
    assert services.window_app(ref) is None
    assert services.window_client_rect(ref) is None
    assert services.capabilities()["panes"] is False
    assert "title" in (PlatformServices.window_app.__doc__ or "")  # never read


@pytest.mark.parametrize(
    ("raw", "name"),
    [
        ("WindowsTerminal.exe", "windowsterminal"),
        ("C:\\Program Files\\WezTerm\\wezterm-gui.exe", "wezterm-gui"),
        ("/usr/bin/kitty", "kitty"),
        ("iTerm2", "iterm2"),
        ("  Code.EXE ", "code"),
    ],
)
def test_process_basename(raw: str, name: str) -> None:
    assert process_basename(raw) == name


def test_process_names_are_cached_per_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    calls: list[int] = []

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            calls.append(pid)
            if pid == 404:
                raise psutil.NoSuchProcess(pid)

        def name(self) -> str:
            return "WezTerm-GUI.exe"

    monkeypatch.setattr(psutil, "Process", FakeProcess)
    services = PlatformServices()
    assert services._app_process_name(7) == "wezterm-gui"
    assert services._app_process_name(7) == "wezterm-gui"
    assert calls == [7]
    assert services._app_process_name(404) is None
    assert services._app_process_name(None) is None
    assert services._app_process_name(0) is None
    assert calls == [7, 404]
