"""Tests for the pane provider registry and its deny-list (panes/registry.py)."""

from __future__ import annotations

import pytest

from eye_tracker.config import PaneSettings
from eye_tracker.panes import registry as registry_module
from eye_tracker.panes.registry import (
    DENIED_APP_IDS,
    DENIED_PROCESSES,
    PaneRegistry,
    default_providers,
    is_denied,
)
from eye_tracker.panes.worker import PaneWorker
from eye_tracker.types import AppIdentity, Rect, WindowRef
from panes_fakes import FakeRunner, RecordingProvider, two_panes

ELECTRON_AND_CHROMIUM = [
    AppIdentity("code", "Chrome_WidgetWin_1"),
    AppIdentity("cursor", "Chrome_WidgetWin_1"),
    AppIdentity("claude", "Chrome_WidgetWin_1"),
    AppIdentity("chatgpt", ""),
    AppIdentity("chrome", "Chrome_WidgetWin_1"),
    AppIdentity("msedge", "Chrome_WidgetWin_1"),
    AppIdentity("firefox", "MozillaWindowClass"),
    AppIdentity("slack", "Chrome_WidgetWin_1"),
    AppIdentity("teams", ""),
    AppIdentity("ms-teams", "TeamsWebView"),
    AppIdentity("discord", "Chrome_WidgetWin_1"),
    AppIdentity("obsidian", "obsidian"),
    AppIdentity("electron", ""),
    # An unknown Electron app on Windows: caught by its window class alone.
    AppIdentity("someapp", "Chrome_WidgetWin_1"),
    # macOS bundle ids and X11 process names.
    AppIdentity("code helper", "com.microsoft.VSCode"),
    AppIdentity("google chrome", "com.google.Chrome"),
]


@pytest.mark.parametrize("app", ELECTRON_AND_CHROMIUM, ids=lambda a: f"{a.process}|{a.app_id}")
def test_denied_apps_never_reach_a_provider(app: AppIdentity) -> None:
    eager = RecordingProvider(snapshot=lambda ref: two_panes(ref.handle))
    registry = PaneRegistry([eager])
    assert is_denied(app)
    assert registry.providers_for(app) == []
    worker = PaneWorker(registry, synchronous=True)
    results: list[object] = []
    worker.detected.connect(results.append)
    worker.request_detect(WindowRef(handle=1, pid=2), app)
    assert eager.calls == []  # not even applies()
    [result] = results
    assert getattr(result, "snapshot", "missing") is None


def test_the_plan_lists_are_all_denied() -> None:
    required = {
        "code",
        "cursor",
        "claude",
        "chatgpt",
        "chrome",
        "msedge",
        "firefox",
        "slack",
        "teams",
        "ms-teams",
        "discord",
        "obsidian",
        "electron",
    }
    assert required <= DENIED_PROCESSES
    assert "Chrome_WidgetWin_1" in DENIED_APP_IDS


def test_terminals_are_not_denied_and_providers_are_asked_in_order() -> None:
    first = RecordingProvider("first", applies=False)
    second = RecordingProvider("second")
    third = RecordingProvider("third")
    registry = PaneRegistry([first, second, third])
    app = AppIdentity("windowsterminal", "CASCADIA_HOSTING_WINDOW_CLASS")
    assert not is_denied(app)
    assert [p.name for p in registry.providers_for(app)] == ["second", "third"]
    assert registry.provider("third") is third
    assert registry.provider("nope") is None
    assert registry.providers_for(None) == []


def test_a_provider_failing_in_applies_is_skipped() -> None:
    class Broken(RecordingProvider):
        def applies(self, app: AppIdentity) -> bool:
            raise RuntimeError("boom")

    good = RecordingProvider("good")
    registry = PaneRegistry([Broken("broken"), good])
    assert registry.providers_for(AppIdentity("kitty", "kitty")) == [good]


def _area(ref: WindowRef) -> Rect | None:
    return Rect(0, 0, 100, 100)


def test_default_providers_follow_the_settings() -> None:
    settings = PaneSettings()

    def names(system: str) -> list[str]:
        found = default_providers(settings, client_rect=_area, runner=FakeRunner(), system=system)
        return [p.name for p in found]

    # Windows Terminal is asked through UI Automation: on Windows only.
    assert names("win32") == ["wezterm", "windows_terminal", "tmux"]
    assert names("linux") == ["wezterm", "tmux"]
    assert names("darwin") == ["wezterm", "tmux"]
    settings.windows_terminal = False
    assert names("win32") == ["wezterm", "tmux"]
    settings.wezterm = False
    assert names("win32") == ["tmux"]
    settings.tmux = False
    assert names("win32") == []


def test_an_available_optional_provider_takes_its_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class WindowsTerminalProvider(RecordingProvider):
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)
            super().__init__("windows_terminal")

    def fake_optional(module: str, cls: str, client_rect: object, runner: object) -> object:
        assert (module, cls) == ("windows_terminal", "WindowsTerminalProvider")
        return WindowsTerminalProvider(client_rect=client_rect, runner=runner)

    monkeypatch.setattr(registry_module, "_optional_provider", fake_optional)
    settings = PaneSettings()
    names = [p.name for p in default_providers(settings, client_rect=_area, system="win32")]
    assert names == ["wezterm", "windows_terminal", "tmux"]
    settings.windows_terminal = False
    assert "windows_terminal" not in [
        p.name for p in default_providers(settings, client_rect=_area, system="win32")
    ]


def test_missing_optional_module_is_skipped() -> None:
    assert registry_module._optional_provider("does_not_exist", "Nope", _area, FakeRunner()) is None
