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


class FakeProcessTable:
    """``psutil.Process`` over a table ``pid -> (create_time, name)``; counts name reads."""

    def __init__(self, table: dict[int, tuple[float, str]]) -> None:
        self.table = table
        self.name_reads: list[int] = []
        owner = self

        class FakeProcess:
            def __init__(self, pid: int) -> None:
                import psutil

                if pid not in owner.table:
                    raise psutil.NoSuchProcess(pid)
                self.pid = pid

            def create_time(self) -> float:
                return owner.table[self.pid][0]

            def name(self) -> str:
                owner.name_reads.append(self.pid)
                return owner.table[self.pid][1]

        self.Process = FakeProcess


def test_process_names_are_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    procs = FakeProcessTable({7: (1000.0, "WezTerm-GUI.exe")})
    monkeypatch.setattr(psutil, "Process", procs.Process)
    services = PlatformServices()
    assert services._app_process_name(7) == "wezterm-gui"
    assert services._app_process_name(7) == "wezterm-gui"
    assert procs.name_reads == [7]  # the second answer came from the cache
    assert services._app_process_name(404) is None
    assert services._app_process_name(None) is None
    assert services._app_process_name(0) is None


def test_a_reused_pid_is_not_taken_for_the_process_that_quit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Claude app quit and its pid went to VS Code: VS Code must not pass as Claude."""
    import psutil

    procs = FakeProcessTable({4242: (1000.0, "claude.exe")})
    monkeypatch.setattr(psutil, "Process", procs.Process)
    services = PlatformServices()
    assert services._app_process_name(4242) == "claude"
    procs.table[4242] = (1500.0, "Code.exe")  # same pid, another process
    assert services._app_process_name(4242) == "code"
    del procs.table[4242]  # gone: nothing is answered from the cache
    assert services._app_process_name(4242) is None
    assert len(services._app_names) == 1  # the old entry was replaced, not kept
