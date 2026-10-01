"""Tests for the Linux platform layer.

They run on every OS: desktop tools, ``/proc`` and the X server are faked, so
nothing here locks the screen, touches displays or moves the real cursor.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eye_tracker.platform import linux
from eye_tracker.platform.base import PlatformServices
from eye_tracker.types import Rect, WindowRef

_SESSION_VARS = (
    "XDG_SESSION_TYPE",
    "WAYLAND_DISPLAY",
    "DISPLAY",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_DESKTOP",
    "DESKTOP_SESSION",
    "KDE_FULL_SESSION",
    "SWAYSOCK",
    "HYPRLAND_INSTANCE_SIGNATURE",
    "XDG_SESSION_ID",
    "QT_QPA_PLATFORM",
    "QT_ENABLE_HIGHDPI_SCALING",
    "LD_LIBRARY_PATH",
    "LD_LIBRARY_PATH_ORIG",
    "YDOTOOL_SOCKET",
    "XDG_RUNTIME_DIR",
)

#: The real ``subprocess.run``: the autouse ``tools`` fixture replaces it (it is
#: the same function as ``linux.subprocess.run``), but the tests against the
#: real kernel need real child processes.
_real_subprocess_run = subprocess.run


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


Response = tuple[int, str] | Callable[[list[str]], tuple[int, str]] | BaseException


class FakeTools:
    """Stands in for ``shutil.which`` and ``subprocess.run``."""

    def __init__(self) -> None:
        self.available: set[str] = set()
        self.responses: dict[tuple[str, ...], Response] = {}
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def install(self, *names: str) -> None:
        self.available.update(names)

    def respond(self, prefix: Iterable[str], response: Response) -> None:
        self.responses[tuple(prefix)] = response

    def which(self, name: str, *args: Any, **kwargs: Any) -> str | None:
        return name if name in self.available else None

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        self.calls.append(argv)
        self.kwargs.append(kwargs)
        best: Response | None = None
        best_len = -1
        for prefix, response in self.responses.items():
            if tuple(argv[: len(prefix)]) == prefix and len(prefix) > best_len:
                best, best_len = response, len(prefix)
        if best is None:
            best = (1, "")
        if isinstance(best, BaseException):
            raise best
        rc, out = best(argv) if callable(best) else best
        if kwargs.get("check") and rc:  # like the real one: never fail silently
            raise subprocess.CalledProcessError(rc, argv, out, "")
        return subprocess.CompletedProcess(argv, rc, out, "")

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


@pytest.fixture(autouse=True)
def session_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank session; monkeypatch restores anything the code under test sets."""
    for key in _SESSION_VARS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def tools(monkeypatch: pytest.MonkeyPatch) -> FakeTools:
    fake = FakeTools()
    monkeypatch.setattr(linux.shutil, "which", fake.which)
    monkeypatch.setattr(linux.subprocess, "run", fake.run)
    monkeypatch.setattr(linux, "_xcb_plugin_usable", lambda: True)
    monkeypatch.setattr(linux, "_YDOTOOL_DEFAULT_SOCKETS", ())
    return fake


@pytest.fixture
def clock() -> Clock:
    return Clock()


class FakeXss:
    def __init__(self, idle: int | None = None) -> None:
        self.idle = idle

    def idle_ms(self) -> int | None:
        return self.idle

    def available(self) -> bool:
        return self.idle is not None


class FakeDBus:
    """Stands in for QtDBus: "unavailable" (``None``) unless a test scripts an answer."""

    def __init__(self) -> None:
        self.answers: dict[str, Any] = {}  # method -> reply, or callable(call) -> reply
        self.calls: list[dict[str, Any]] = []

    def call(self, **kwargs: Any) -> linux._DBusReply | None:
        self.calls.append(kwargs)
        answer = self.answers.get(kwargs["method"])
        return answer(kwargs) if callable(answer) else answer

    def methods(self) -> list[str]:
        return [call["method"] for call in self.calls]


def _no_x_server() -> Any:
    raise OSError("no X server in tests")


@pytest.fixture
def dbus() -> FakeDBus:
    return FakeDBus()


@pytest.fixture
def plat(clock: Clock, tmp_path: Path, dbus: FakeDBus) -> linux.LinuxPlatform:
    """Everything on the calling thread; no real X server, D-Bus or /proc is touched."""
    platform = linux.LinuxPlatform(
        proc_root=tmp_path / "proc",
        dev_root=tmp_path / "dev",
        sys_root=tmp_path / "sys",
        clock=clock,
        sleep=clock.advance,
        background_threads=False,
    )
    platform._xss = FakeXss()  # type: ignore[assignment]
    platform._ewmh = linux._EwmhClient(_no_x_server, clock=clock)
    platform._dbus = dbus  # type: ignore[assignment]
    return platform


def x11(monkeypatch: pytest.MonkeyPatch, desktop: str | None = None) -> None:
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    if desktop:
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)


def wayland(
    monkeypatch: pytest.MonkeyPatch, desktop: str | None = None, display: bool = True
) -> None:
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    if display:
        monkeypatch.setenv("DISPLAY", ":0")
    if desktop:
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)


# ------------------------------------------------------------------ session type
@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"XDG_SESSION_TYPE": "x11"}, False),
        ({"XDG_SESSION_TYPE": "wayland"}, True),
        ({"XDG_SESSION_TYPE": "Wayland"}, True),
        ({"WAYLAND_DISPLAY": "wayland-0"}, True),
        ({"XDG_SESSION_TYPE": "x11", "WAYLAND_DISPLAY": "wayland-1"}, True),
        ({"XDG_SESSION_TYPE": "tty"}, False),
    ],
)
def test_wayland_detection(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, env: dict[str, str], expected: bool
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert plat.is_wayland is expected


def test_desktop_detection() -> None:
    assert linux._is_gnome_env({"XDG_CURRENT_DESKTOP": "ubuntu:GNOME"})
    assert linux._is_gnome_env({"XDG_SESSION_DESKTOP": "gnome-classic"})
    assert not linux._is_gnome_env({"XDG_CURRENT_DESKTOP": "KDE"})
    assert linux._is_kde_env({"XDG_CURRENT_DESKTOP": "KDE"})
    assert linux._is_kde_env({"DESKTOP_SESSION": "plasmawayland"})
    assert linux._is_kde_env({"KDE_FULL_SESSION": "true"})
    assert not linux._is_kde_env({"XDG_CURRENT_DESKTOP": "XFCE"})


# ------------------------------------------------------------------ prepare_process
def test_prepare_forces_xcb_on_wayland_with_xwayland(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    wayland(monkeypatch, "GNOME")
    plat.prepare_process()
    # A list, so Qt falls back to native Wayland if the xcb plugin cannot load.
    assert os.environ["QT_QPA_PLATFORM"] == "xcb;wayland"
    assert os.environ["QT_ENABLE_HIGHDPI_SCALING"] == "0"


def test_prepare_keeps_native_wayland_without_xwayland(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    wayland(monkeypatch, "GNOME", display=False)
    plat.prepare_process()
    assert "QT_QPA_PLATFORM" not in os.environ
    assert os.environ["QT_ENABLE_HIGHDPI_SCALING"] == "0"


def test_prepare_keeps_native_wayland_without_xcb_cursor(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    wayland(monkeypatch, "KDE")
    monkeypatch.setattr(linux, "_xcb_plugin_usable", lambda: False)
    plat.prepare_process()
    assert "QT_QPA_PLATFORM" not in os.environ


def test_prepare_respects_explicit_platform_and_scaling(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    wayland(monkeypatch, "GNOME")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    monkeypatch.setenv("QT_ENABLE_HIGHDPI_SCALING", "1")
    plat.prepare_process()
    assert os.environ["QT_QPA_PLATFORM"] == "wayland"
    assert os.environ["QT_ENABLE_HIGHDPI_SCALING"] == "1"


def test_prepare_on_x11_does_not_touch_platform(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    x11(monkeypatch)
    plat.prepare_process()
    assert "QT_QPA_PLATFORM" not in os.environ
    assert os.environ["QT_ENABLE_HIGHDPI_SCALING"] == "0"


def test_child_env_undoes_our_overrides(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    wayland(monkeypatch, "KDE")
    plat.prepare_process()
    env = plat._child_env()
    assert "QT_QPA_PLATFORM" not in env
    assert "QT_ENABLE_HIGHDPI_SCALING" not in env
    assert os.environ["QT_QPA_PLATFORM"] == "xcb;wayland"  # our own process keeps them


def test_child_env_restores_library_path(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEI123")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/opt/lib")
    env = plat._child_env()
    assert env["LD_LIBRARY_PATH"] == "/opt/lib"
    assert "LD_LIBRARY_PATH_ORIG" not in env


def test_child_env_drops_bundle_library_path_when_frozen(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEI123")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert "LD_LIBRARY_PATH" not in plat._child_env()


def test_tools_run_with_child_env_and_timeout(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    wayland(monkeypatch, "KDE")
    plat.prepare_process()
    tools.install("kscreen-doctor")
    tools.respond(["kscreen-doctor"], (0, ""))
    assert plat.display_off()
    kwargs = tools.kwargs[0]
    assert "QT_QPA_PLATFORM" not in kwargs["env"]
    assert kwargs["timeout"] == pytest.approx(5.0)
    assert kwargs["stdin"] is subprocess.DEVNULL


# ------------------------------------------------------------------ lock screen
_ALL_LOCKERS = (
    "loginctl",
    "xdg-screensaver",
    "gnome-screensaver-command",
    "dm-tool",
    "xflock4",
    "qdbus",
    "qdbus6",
    "qdbus-qt6",
    "qdbus-qt5",
    "gdbus",
)


def test_lock_tries_every_locker_in_order(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "c2")
    tools.install(*_ALL_LOCKERS)
    assert plat.lock_screen() is False
    assert tools.names() == list(_ALL_LOCKERS)
    assert tools.calls[0] == ["loginctl", "lock-session", "c2"]
    assert tools.calls[-1][-1] == "org.freedesktop.ScreenSaver.Lock"


def test_lock_stops_at_first_success(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "7")
    tools.install(*_ALL_LOCKERS)
    tools.respond(["loginctl"], (1, ""))
    tools.respond(["xdg-screensaver"], (0, ""))
    assert plat.lock_screen() is True
    assert tools.calls == [["loginctl", "lock-session", "7"], ["xdg-screensaver", "lock"]]


def test_lock_skips_missing_tools_and_timeouts(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "7")
    tools.install("loginctl", "qdbus6")
    tools.respond(["loginctl"], subprocess.TimeoutExpired(["loginctl"], 5.0))
    tools.respond(["qdbus6"], (0, ""))
    assert plat.lock_screen() is True
    assert tools.calls == [
        ["loginctl", "lock-session", "7"],
        ["qdbus6", "org.freedesktop.ScreenSaver", "/ScreenSaver", "Lock"],
    ]


def test_lock_without_any_locker(plat: linux.LinuxPlatform, tools: FakeTools) -> None:
    assert plat.lock_screen() is False
    assert tools.calls == []


def test_lock_methods_lists_installed_tools_in_the_order_they_are_tried(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    """Regression (r2-docs-13): doctor showed "lock yes" whenever loginctl was installed."""
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    tools.install("gdbus", "dm-tool", "loginctl", "qdbus6")
    assert plat.lock_methods() == [
        "loginctl lock-session",
        "dm-tool lock",
        "qdbus6 org.freedesktop.ScreenSaver.Lock",
        "gdbus org.freedesktop.ScreenSaver.Lock",
    ]
    assert tools.calls == []  # listing runs nothing, not even a session lookup
    assert all(method.isascii() for method in plat.lock_methods())


@pytest.mark.parametrize(
    "env",
    [
        {"XDG_CURRENT_DESKTOP": "i3"},
        {"XDG_CURRENT_DESKTOP": "GNOME", "SWAYSOCK": "/run/user/1000/sway-ipc.sock"},
    ],
)
def test_lock_methods_flags_loginctl_where_nothing_may_listen(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    env: dict[str, str],
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    tools.install("loginctl", "xdg-screensaver")
    loginctl, xdg = plat.lock_methods()
    assert loginctl.startswith("loginctl lock-session (")
    assert "listens to logind" in loginctl
    assert xdg == "xdg-screensaver lock"


def test_lock_methods_without_any_tool(plat: linux.LinuxPlatform) -> None:
    assert plat.lock_methods() == []


def test_doctor_shows_the_linux_lock_methods(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    from eye_tracker import diagnostics

    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    tools.install("loginctl")
    section = diagnostics._platform_section(plat)
    assert section["lock_methods"] == plat.lock_methods()
    assert "known when it is tried" in section["notes"]["lock"]


def test_lock_resolves_display_session_without_session_id(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    tools.install("loginctl")
    tools.respond(["loginctl", "show-user"], (0, "3\n"))
    tools.respond(["loginctl", "lock-session"], (0, ""))
    tools.respond(["loginctl", "show-session"], (0, "yes\n"))
    assert plat.lock_screen() is True
    assert tools.calls == [
        ["loginctl", "show-user", "1000", "-p", "Display", "--value"],
        ["loginctl", "lock-session", "3"],
        ["loginctl", "show-session", "3", "-p", "LockedHint", "--value"],  # GNOME confirmed it
    ]


def test_lock_falls_back_to_callers_session(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    tools.install("loginctl")
    tools.respond(["loginctl", "show-user"], (0, "\n"))
    tools.respond(["loginctl", "lock-session"], (0, ""))
    assert plat.lock_screen() is True
    assert tools.calls[-1] == ["loginctl", "lock-session"]


@pytest.mark.parametrize("desktop", ["X-Cinnamon", "MATE", "XFCE"])
def test_lock_trusts_loginctl_on_desktops_that_handle_it(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, desktop: str
) -> None:
    # Their lock screens listen to logind but do not all report LockedHint.
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
    tools.install(*_ALL_LOCKERS)
    tools.respond(["loginctl", "lock-session"], (0, ""))
    assert plat.lock_screen() is True
    assert tools.calls == [["loginctl", "lock-session", "2"]]


_HINT_QUERY = ["loginctl", "show-session", "2", "-p", "LockedHint", "--value"]


@pytest.mark.parametrize("desktop", ["GNOME", "KDE", "Budgie:GNOME"])
def test_lock_on_gnome_and_kde_is_confirmed_by_locked_hint(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, desktop: str
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
    tools.install(*_ALL_LOCKERS)
    tools.respond(["loginctl", "lock-session"], (0, ""))
    hints = ["no\n"]
    tools.respond(["loginctl", "show-session"], lambda argv: (0, hints.pop(0) if hints else "yes"))
    assert plat.lock_screen() is True
    assert tools.calls == [["loginctl", "lock-session", "2"], _HINT_QUERY, _HINT_QUERY]


def test_lock_refused_by_gnome_policy_is_reported(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, clock: Clock
) -> None:
    """disable-lock-screen: gnome-shell ignores every request, yet the tools exit with 0."""
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")
    tools.install("loginctl", "xdg-screensaver", "gdbus")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    tools.respond(["xdg-screensaver"], (0, ""))
    tools.respond(["gdbus"], (0, "()"))
    tools.respond(["loginctl", "show-session"], (0, "no\n"))
    started = clock.now
    # False: the controller warns, and the shoulder guard falls back to its curtain.
    assert plat.lock_screen() is False
    locks = [call for call in tools.calls if call[:2] != ["loginctl", "show-session"]]
    assert [call[0] for call in locks] == ["loginctl", "xdg-screensaver", "gdbus"]
    # Each claimed success was checked for the bounded confirmation time.
    assert clock.now - started == pytest.approx(3 * linux._LOCK_CONFIRM_S)


def test_lock_on_gnome_confirmed_after_another_locker(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    tools.install("loginctl", "xdg-screensaver", "dm-tool")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    locked: list[bool] = []
    tools.respond(["xdg-screensaver"], lambda argv: (locked.append(True) or 0, ""))
    tools.respond(["loginctl", "show-session"], lambda argv: (0, "yes" if locked else "no"))
    assert plat.lock_screen() is True
    assert ["dm-tool", "lock"] not in tools.calls  # no second lock screen on top


def test_lock_on_gnome_seen_late_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    tools.install("loginctl")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    hint = ["no"]
    tools.respond(["loginctl", "show-session"], lambda argv: (0, hint[0]))

    def too_late(session: str | None, *, trust_unknown: bool) -> bool:
        hint[0] = "yes"  # the lock screen comes up just after the wait
        return False

    monkeypatch.setattr(plat, "_lock_confirmed", too_late)
    assert plat.lock_screen() is True


def test_lock_on_gnome_without_a_readable_hint_trusts_the_desktop(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    tools.install("loginctl", "xdg-screensaver")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    tools.respond(["loginctl", "show-session"], (1, "Failed to get session\n"))
    assert plat.lock_screen() is True
    assert ["xdg-screensaver", "lock"] not in tools.calls


@pytest.mark.parametrize(
    "socket_var", ["SWAYSOCK", "I3SOCK", "HYPRLAND_INSTANCE_SIGNATURE", "NIRI_SOCKET"]
)
def test_compositor_posing_as_gnome_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    proc_tree: ProcTree,
    socket_var: str,
) -> None:
    """sway with XDG_CURRENT_DESKTOP=GNOME (for portals) and no swayidle lock hook."""
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    wayland(monkeypatch, "GNOME")
    monkeypatch.setenv(socket_var, "/run/user/1000/wm.sock")
    assert not linux._desktop_handles_logind_lock(os.environ)
    assert not linux._locked_hint_authoritative(os.environ)
    tools.install("loginctl")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    tools.respond(["loginctl", "show-session"], (0, "no\n"))
    assert plat.lock_screen() is False  # nothing locked: say so
    # A locker started by hand is noticed, although the hint says "unlocked".
    proc_tree.process(200, [], comm="swaylock")
    plat._locked_cache = None
    assert plat.is_session_locked() is True


def test_lock_unconfirmed_loginctl_tries_other_lockers(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    clock: Clock,
    proc_tree: ProcTree,
) -> None:
    # i3 without xss-lock: logind emits Lock, nobody listens, loginctl still exits 0.
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    proc_tree.process(100, [], comm="i3")
    tools.install("loginctl", "dm-tool")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    tools.respond(["loginctl", "show-session"], (0, "no\n"))
    tools.respond(["dm-tool"], (0, ""))
    started = clock.now
    assert plat.lock_screen() is True
    assert tools.calls[0] == ["loginctl", "lock-session", "2"]
    assert tools.calls[-1] == ["dm-tool", "lock"]
    assert ["loginctl", "show-session", "2", "-p", "LockedHint", "--value"] in tools.calls
    assert clock.now - started == pytest.approx(linux._LOCK_CONFIRM_S)  # bounded wait


def test_lock_unconfirmed_loginctl_alone_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    proc_tree: ProcTree,
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "sway")
    tools.install("loginctl")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    tools.respond(["loginctl", "show-session"], (0, "no\n"))
    # False lets the controller warn, and the shoulder guard fall back to its curtain.
    assert plat.lock_screen() is False


def test_lock_confirmed_by_locker_process(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, tools: FakeTools, proc_tree: ProcTree
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    tools.install("loginctl", "dm-tool")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    tools.respond(["loginctl", "show-session"], (0, "no\n"))  # xss-lock + i3lock never set it
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)
        proc_tree.process(4321, [], comm="i3lock")  # xss-lock started the locker

    platform = linux.LinuxPlatform(
        proc_root=proc_tree.proc,
        dev_root=proc_tree.dev,
        sys_root=proc_tree.sys,
        clock=clock,
        sleep=sleep,
        background_threads=False,
    )
    platform._dbus = FakeDBus()  # type: ignore[assignment]
    assert platform.lock_screen() is True
    assert len(sleeps) == 1
    assert ["dm-tool", "lock"] not in tools.calls  # no second lock screen on top


def test_lock_slow_locker_is_not_stacked_with_another(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    proc_tree: ProcTree,
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    tools.install("loginctl", "xdg-screensaver", "dm-tool")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    waited: list[str | None] = []

    def nothing_in_time(session: str | None) -> bool:
        waited.append(session)
        proc_tree.process(4321, [], comm="i3lock")  # shows up just after the wait
        return False

    monkeypatch.setattr(plat, "_logind_lock_took_effect", nothing_in_time)
    assert plat.lock_screen() is True
    assert waited == ["2"]
    assert tools.calls == [["loginctl", "lock-session", "2"]]  # no second lock screen


def test_lock_confirmed_by_locked_hint(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "LXQt")
    tools.install("loginctl", "dm-tool")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    hints = ["no\n", "no\n", "yes\n"]
    tools.respond(["loginctl", "show-session"], lambda argv: (0, hints.pop(0) if hints else "yes"))
    assert plat.lock_screen() is True
    assert ["dm-tool", "lock"] not in tools.calls


# ------------------------------------------------------------------ display power
_XSET_Q_ENABLED = (
    "DPMS (Energy Star):\n  Standby: 600    Suspend: 600    Off: 600\n  DPMS is Enabled\n"
)


def test_display_off_x11_uses_xset_without_python_xlib(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    x11(monkeypatch, "XFCE")
    tools.install("xset", "kscreen-doctor", "busctl")
    tools.respond(["xset"], (0, ""))
    tools.respond(["xset", "q"], (0, _XSET_Q_ENABLED))
    assert plat.display_off()
    assert plat.wake_display()
    assert tools.calls == [
        ["xset", "q"],
        ["xset", "dpms", "force", "off"],
        ["xset", "q"],
        ["xset", "dpms", "force", "on"],
    ]


def test_xset_restores_disabled_dpms_after_waking(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    # "xset -dpms" (kiosk, presentation): "dpms force" would enable DPMS for good.
    x11(monkeypatch, "i3")
    tools.install("xset")
    tools.respond(["xset"], (0, ""))
    tools.respond(["xset", "q"], (0, _XSET_Q_ENABLED.replace("Enabled", "Disabled")))
    assert plat.display_off()
    assert tools.calls[-1] == ["xset", "dpms", "force", "off"]
    tools.respond(["xset", "q"], (0, _XSET_Q_ENABLED))  # "force" enabled it meanwhile
    assert plat.wake_display()
    assert tools.calls[-2:] == [["xset", "dpms", "force", "on"], ["xset", "-dpms"]]


@pytest.mark.parametrize(
    "output", ["Server does not have the DPMS Extension\n", "Display is not capable of DPMS\n"]
)
def test_xset_without_dpms_reports_failure(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, output: str
) -> None:
    # Xvnc, xrdp, many VMs: xset complains on stderr and still exits with 0.
    x11(monkeypatch, "XFCE")
    tools.install("xset")
    tools.respond(["xset"], (0, ""))
    tools.respond(["xset", "q"], (0, output))
    assert plat.display_off() is False
    assert tools.calls == [["xset", "q"]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_XSET_Q_ENABLED, (True, True)),
        (_XSET_Q_ENABLED.replace("Enabled", "Disabled"), (True, False)),
        ("Server does not have the DPMS Extension", (False, False)),
        ("Keyboard Control:\n  auto repeat: on", None),
        ("", None),
    ],
)
def test_parse_xset_dpms(text: str, expected: tuple[bool, bool] | None) -> None:
    assert linux._parse_xset_dpms(text) == expected


def test_x11_dpms_in_process_restores_disabled_state(
    x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay, tools: FakeTools
) -> None:
    tools.install("xset")
    xdisplay.dpms_enabled = False
    assert x11_plat.display_off() is True
    assert xdisplay.dpms_log == ["enable", "force 3"]
    assert x11_plat.wake_display() is True
    assert xdisplay.dpms_log == ["enable", "force 3", "force 0", "disable"]
    assert xdisplay.dpms_enabled is False
    assert tools.calls == []  # python-xlib did it all


def test_x11_dpms_in_process_keeps_enabled_state(
    x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay
) -> None:
    assert x11_plat.display_off() is True
    assert x11_plat.wake_display() is True
    assert xdisplay.dpms_log == ["force 3", "force 0"]
    assert xdisplay.dpms_enabled is True


def test_x11_without_dpms_reports_failure(
    x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay, tools: FakeTools
) -> None:
    tools.install("xset")
    tools.respond(["xset"], (0, ""))
    xdisplay.extensions.discard("DPMS")
    assert x11_plat.display_off() is False
    xdisplay.extensions.add("DPMS")
    xdisplay.dpms_capable_flag = False
    assert x11_plat.display_off() is False
    assert xdisplay.dpms_log == []
    assert tools.calls == []  # xset would only pretend
    assert x11_plat.capabilities()["display_off"] is False


def test_x11_dpms_disabled_again_when_monitors_wake_by_themselves(
    x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay
) -> None:
    # wake_on_return off: the user's input wakes the monitors, not wake_display().
    xdisplay.dpms_enabled = False
    assert x11_plat.display_off() is True
    assert x11_plat._restore_dpms_if_awake() is False  # still off: keep waiting
    xdisplay.power_level = 0
    assert x11_plat._restore_dpms_if_awake() is True
    assert xdisplay.dpms_enabled is False
    assert x11_plat._restore_dpms_if_awake() is True  # nothing left to do
    assert xdisplay.dpms_log == ["enable", "force 3", "disable"]


def test_display_off_kde_wayland_uses_kscreen_doctor(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    wayland(monkeypatch, "KDE")
    tools.install("xset", "kscreen-doctor")
    tools.respond(["xset"], (0, ""))
    tools.respond(["kscreen-doctor"], (0, ""))
    assert plat.display_off()
    assert plat.wake_display()
    assert tools.calls == [["kscreen-doctor", "--dpms", "off"], ["kscreen-doctor", "--dpms", "on"]]


def test_display_off_gnome_wayland_uses_mutter(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    wayland(monkeypatch, "ubuntu:GNOME")
    tools.install("xset", "busctl", "gdbus")
    tools.respond(["busctl"], (0, ""))
    assert plat.display_off()
    assert plat.wake_display()
    prefix = [
        "busctl",
        "--user",
        "set-property",
        "org.gnome.Mutter.DisplayConfig",
        "/org/gnome/Mutter/DisplayConfig",
        "org.gnome.Mutter.DisplayConfig",
        "PowerSaveMode",
        "i",
    ]
    assert tools.calls == [[*prefix, "1"], [*prefix, "0"]]


def test_display_off_gnome_falls_back_to_gdbus(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    wayland(monkeypatch, "GNOME")
    tools.install("gdbus")
    tools.respond(["gdbus"], (0, "()\n"))
    assert plat.display_off()
    call = tools.calls[0]
    assert call[:3] == ["gdbus", "call", "--session"]
    assert "org.freedesktop.DBus.Properties.Set" in call
    assert call[-2:] == ["PowerSaveMode", "<int32 1>"]


def test_display_off_sway_prefers_power_then_dpms(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    wayland(monkeypatch, "sway")
    monkeypatch.setenv("SWAYSOCK", "/run/user/1000/sway-ipc.sock")
    tools.install("swaymsg")
    tools.respond(["swaymsg", "output", "*", "dpms"], (0, ""))
    assert plat.display_off()
    assert tools.calls == [
        ["swaymsg", "output", "*", "power", "off"],
        ["swaymsg", "output", "*", "dpms", "off"],
    ]


def test_display_off_unknown_wayland_is_unsupported(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    wayland(monkeypatch, "Weston")
    tools.install("xset")
    tools.respond(["xset"], (0, ""))
    assert plat.display_off() is False
    assert plat.wake_display() is False
    assert tools.calls == []  # xset would only blank XWayland's virtual output


def test_wake_x11_falls_back_to_xtest(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    x11(monkeypatch)
    nudges: list[bool] = []

    def nudge() -> bool:
        nudges.append(True)
        return True

    monkeypatch.setattr(plat._ewmh, "nudge_pointer", nudge)
    assert plat.wake_display() is True
    assert nudges == [True]


def test_wake_wayland_never_uses_xtest(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    wayland(monkeypatch, "GNOME")
    monkeypatch.setattr(plat._ewmh, "nudge_pointer", lambda: pytest.fail("XTest used on Wayland"))
    assert plat.wake_display() is False


# ------------------------------------------------------------------ session lock state
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes\n", True),
        ("no\n", False),
        ("LockedHint=yes\n", True),
        ("LockedHint=no", False),
        ("  YES  ", True),
        ("", None),
        (None, None),
        ("maybe", None),
    ],
)
def test_parse_locked_hint(text: str | None, expected: bool | None) -> None:
    assert linux._parse_locked_hint(text) is expected


def test_session_locked_queries_logind_and_caches(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, clock: Clock
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    tools.install("loginctl")
    answers = iter([(0, "yes\n"), (0, "no\n")])
    tools.respond(["loginctl", "show-session"], lambda argv: next(answers))
    assert plat.is_session_locked() is True
    assert tools.calls == [["loginctl", "show-session", "5", "-p", "LockedHint", "--value"]]
    clock.advance(1.5)
    assert plat.is_session_locked() is True  # cached
    assert len(tools.calls) == 1
    clock.advance(1.0)
    assert plat.is_session_locked() is False
    assert len(tools.calls) == 2


def test_session_locked_unknown_without_loginctl(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    assert plat.is_session_locked() is None
    assert tools.calls == []


def test_session_locked_unknown_on_error(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    tools.install("loginctl")
    tools.respond(["loginctl"], (1, "Failed to get session: No session '5' known\n"))
    assert plat.is_session_locked() is None


def test_session_locked_rejects_odd_session_id(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "5 --help")
    tools.install("loginctl")
    assert plat.is_session_locked() is None
    assert tools.calls == []


def test_lock_invalidates_locked_cache(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    tools.install("loginctl")
    hints = ["no\n", "yes\n"]
    tools.respond(["loginctl", "show-session"], lambda argv: (0, hints.pop(0) if hints else "yes"))
    tools.respond(["loginctl", "lock-session"], (0, ""))
    assert plat.is_session_locked() is False
    assert plat.lock_screen() is True
    assert plat.is_session_locked() is True


def test_session_locked_by_locker_without_locked_hint(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    clock: Clock,
    proc_tree: ProcTree,
) -> None:
    # sway + swaylock: logind's LockedHint stays "no" the whole time.
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    wayland(monkeypatch, "sway")
    tools.install("loginctl")
    tools.respond(["loginctl", "show-session"], (0, "no\n"))
    proc_tree.process(100, [], comm="sway")
    assert plat.is_session_locked() is False
    proc_tree.process(200, [], comm="swaylock")
    clock.advance(2.5)
    assert plat.is_session_locked() is True
    proc_tree.remove(200)
    clock.advance(2.5)
    assert plat.is_session_locked() is False  # the unlock is seen too


def test_session_locked_locker_names_are_truncated_like_comm(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, proc_tree: ProcTree
) -> None:
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    proc_tree.process(300, [], comm="kscreenlocker_g")  # kscreenlocker_greet, 15 chars
    assert plat.is_session_locked() is True


def test_session_locked_ignores_other_users_lockers(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, proc_tree: ProcTree
) -> None:
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    proc_tree.login(1000)
    proc_tree.process(300, [], comm="i3lock", loginuid=1001)  # another user's login
    proc_tree.process(301, [], comm="slock", loginuid=linux._AUDIT_UID_UNSET)  # no login
    owner = os.stat(proc_tree.proc / "300").st_uid
    monkeypatch.setattr(linux, "_getuid", lambda: owner + 1)  # none of the entries is ours
    assert plat.is_session_locked() is False


def test_session_locked_by_own_locker_outside_a_login(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, proc_tree: ProcTree
) -> None:
    """A locker started by a service (no login UID) still counts when we own it."""
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    proc_tree.login(1000)
    proc_tree.process(300, [], comm="swaylock", loginuid=linux._AUDIT_UID_UNSET)
    monkeypatch.setattr(linux, "_getuid", lambda: os.stat(proc_tree.proc / "300").st_uid)
    assert plat.is_session_locked() is True


def test_session_locked_by_a_setuid_locker_when_we_have_no_login_uid(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, proc_tree: ProcTree
) -> None:
    """Started outside a login ourselves: the locker's login UID is compared with our UID."""
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    proc_tree.login(linux._AUDIT_UID_UNSET)
    proc_tree.process(400, [], comm="slock")
    owner = os.stat(proc_tree.proc / "400").st_uid
    monkeypatch.setattr(linux, "_getuid", lambda: owner + 1)  # root's entry, not ours
    (proc_tree.proc / "400" / "loginuid").write_text(str(owner + 2))  # another user's
    assert plat.is_session_locked() is False
    (proc_tree.proc / "400" / "loginuid").write_text(str(owner + 1))  # ours
    plat._locked_cache = None
    assert plat.is_session_locked() is True


def test_session_locked_by_a_setuid_locker(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, proc_tree: ProcTree
) -> None:
    """xss-lock -- slock: slock runs as 'nobody' and its /proc entry belongs to root."""
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    proc_tree.login(1000)
    proc_tree.process(100, [], comm="i3", loginuid=1000)
    proc_tree.process(400, [], comm="slock", loginuid=1000)
    owner = os.stat(proc_tree.proc / "400").st_uid
    monkeypatch.setattr(linux, "_getuid", lambda: owner + 1)  # the entry is not ours
    assert plat.is_session_locked() is True


def test_session_locked_without_login_uids_falls_back_to_the_owner(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, proc_tree: ProcTree
) -> None:
    # A kernel without audit support: no loginuid files at all.
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    proc_tree.process(300, [], comm="i3lock")
    owner = os.stat(proc_tree.proc / "300").st_uid
    monkeypatch.setattr(linux, "_getuid", lambda: owner + 1)
    assert plat.is_session_locked() is False
    monkeypatch.setattr(linux, "_getuid", lambda: owner)
    plat._locked_cache = None
    assert plat.is_session_locked() is True


def test_lock_confirmed_by_a_setuid_locker(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, tools: FakeTools, proc_tree: ProcTree
) -> None:
    """No second locker (dm-tool: a second unlock) on top of slock."""
    monkeypatch.setenv("XDG_SESSION_ID", "2")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    proc_tree.login(1000)
    tools.install("loginctl", "dm-tool")
    tools.respond(["loginctl", "lock-session"], (0, ""))
    tools.respond(["loginctl", "show-session"], (0, "no\n"))  # xss-lock never sets it

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        proc_tree.process(4321, [], comm="slock", loginuid=1000)

    platform = linux.LinuxPlatform(
        proc_root=proc_tree.proc,
        dev_root=proc_tree.dev,
        sys_root=proc_tree.sys,
        clock=clock,
        sleep=sleep,
        background_threads=False,
    )
    platform._dbus = FakeDBus()  # type: ignore[assignment]
    monkeypatch.setattr(linux, "_getuid", lambda: 12345)  # slock's entry belongs to root
    assert platform.lock_screen() is True
    assert ["dm-tool", "lock"] not in tools.calls


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1000\n", 1000), ("0", 0), ("4294967295", None), ("", None), ("junk", None)],
)
def test_login_uid_parsing(tmp_path: Path, text: str, expected: int | None) -> None:
    (tmp_path / "loginuid").write_text(text)
    assert linux._login_uid(str(tmp_path)) == expected
    assert linux._login_uid(str(tmp_path / "missing")) is None


def test_session_locked_without_logind_session_relies_on_lockers(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, proc_tree: ProcTree
) -> None:
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    monkeypatch.setattr(linux, "_getuid", lambda: None)  # no session can be looked up
    proc_tree.process(100, [], comm="i3")
    assert plat.is_session_locked() is False
    proc_tree.process(101, [], comm="slock")
    plat._locked_cache = None
    assert plat.is_session_locked() is True


def test_session_locked_failed_query_is_unknown_not_unlocked(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    proc_tree: ProcTree,
) -> None:
    # A lock screen that is up must not be mistaken for the user's return.
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    tools.install("loginctl")
    tools.respond(["loginctl", "show-session"], subprocess.TimeoutExpired(["loginctl"], 2.0))
    assert plat.is_session_locked() is None


def test_session_locked_gnome_trusts_locked_hint_without_scanning(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    proc_tree: ProcTree,
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")
    tools.install("loginctl")
    tools.respond(["loginctl", "show-session"], (0, "no\n"))
    monkeypatch.setattr(plat, "_locker_running", lambda: pytest.fail("scanned /proc"))
    assert plat.is_session_locked() is False


def test_session_lookup_failure_is_retried(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, clock: Clock
) -> None:
    # Autostarted by a systemd user unit (no XDG_SESSION_ID) while logind is slow.
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    tools.install("loginctl")
    answers: list[Any] = [subprocess.TimeoutExpired(["loginctl"], 2.0)]

    def show_user(argv: list[str]) -> tuple[int, str]:
        if answers:
            raise answers.pop(0)
        return 0, "3\n"

    tools.respond(["loginctl", "show-user"], show_user)
    tools.respond(["loginctl", "show-session"], (0, "yes\n"))
    assert plat.is_session_locked() is None
    clock.advance(5.0)
    assert plat.is_session_locked() is None  # not retried on every poll
    assert [call[1] for call in tools.calls] == ["show-user"]
    clock.advance(30.0)
    assert plat.is_session_locked() is True
    assert tools.calls[-1] == ["loginctl", "show-session", "3", "-p", "LockedHint", "--value"]


def test_locked_hint_through_dbus_in_process(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    dbus: FakeDBus,
    clock: Clock,
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    tools.install("loginctl")
    path = "/org/freedesktop/login1/session/_35"
    dbus.answers["GetSession"] = linux._DBusReply((path,))
    hints = [True, False]
    dbus.answers["Get"] = lambda call: linux._DBusReply((hints.pop(0),))
    assert plat.is_session_locked() is True
    clock.advance(2.5)
    assert plat.is_session_locked() is False
    assert tools.calls == []  # no loginctl process per poll
    assert dbus.methods() == ["GetSession", "Get", "Get"]  # the session path is cached
    get = dbus.calls[1]
    assert get["system"] is True
    assert get["path"] == path
    assert get["args"] == ("org.freedesktop.login1.Session", "LockedHint")


def test_locked_hint_dbus_error_forgets_session_path(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    dbus: FakeDBus,
    clock: Clock,
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    tools.install("loginctl")
    dbus.answers["GetSession"] = linux._DBusReply(("/org/freedesktop/login1/session/_35",))
    dbus.answers["Get"] = linux._DBusReply((), "org.freedesktop.DBus.Error.UnknownObject")
    assert plat.is_session_locked() is None
    clock.advance(2.5)
    assert plat.is_session_locked() is None
    assert dbus.methods() == ["GetSession", "Get", "GetSession", "Get"]
    assert tools.calls == []  # D-Bus answered: no pointless loginctl fallback


@pytest.mark.parametrize(
    ("session_reply", "hint_reply"),
    [
        (("/org/freedesktop/login1/session/_35",), ("yes",)),  # a variant left unconverted
        ((None,), (True,)),  # an object path QtDBus could not convert
    ],
)
def test_locked_hint_unexpected_dbus_reply_falls_back_to_loginctl(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    dbus: FakeDBus,
    session_reply: tuple[Any, ...],
    hint_reply: tuple[Any, ...],
) -> None:
    monkeypatch.setenv("XDG_SESSION_ID", "5")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    tools.install("loginctl")
    tools.respond(["loginctl", "show-session"], (0, "yes\n"))
    dbus.answers["GetSession"] = linux._DBusReply(session_reply)
    dbus.answers["Get"] = linux._DBusReply(hint_reply)
    assert plat.is_session_locked() is True
    assert tools.names() == ["loginctl"]


# ------------------------------------------------------------------ idle time
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(uint64 12345,)\n", 12345),
        ("(uint64 0,)", 0),
        ("(uint32 7,)", 7),
        ("(42,)", 42),
        ("Error: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_gdbus_uint(text: str | None, expected: int | None) -> None:
    assert linux._parse_gdbus_uint(text) == expected


def test_idle_x11_uses_screensaver_extension(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    x11(monkeypatch, "GNOME")
    tools.install("gdbus")
    plat._xss = FakeXss(1500)  # type: ignore[assignment]
    assert plat.seconds_since_input() == pytest.approx(1.5)
    assert tools.calls == []
    assert plat.seconds_since_key_input() is None


def test_idle_gnome_wayland_uses_mutter_with_extrapolated_cache(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, clock: Clock
) -> None:
    wayland(monkeypatch, "GNOME")
    plat._xss = FakeXss(999_999)  # type: ignore[assignment]  # XWayland idle must be ignored
    tools.install("gdbus")
    tools.respond(["gdbus"], (0, "(uint64 12345,)\n"))
    assert plat.seconds_since_input() == pytest.approx(12.345)
    assert tools.calls[0][-1] == "org.gnome.Mutter.IdleMonitor.GetIdletime"
    clock.advance(0.3)
    # Cached, and advanced with the clock so the implied last-input time stays put.
    assert plat.seconds_since_input() == pytest.approx(12.645)
    assert len(tools.calls) == 1
    clock.advance(0.3)
    assert plat.seconds_since_input() == pytest.approx(12.345)
    assert len(tools.calls) == 2


def test_idle_backs_off_after_failure(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, clock: Clock
) -> None:
    wayland(monkeypatch, "KDE")
    tools.install("gdbus")
    tools.respond(["gdbus"], (1, "Error: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown"))
    assert plat.seconds_since_input() is None
    clock.advance(10.0)
    assert plat.seconds_since_input() is None
    assert len(tools.calls) == 1
    clock.advance(55.0)
    assert plat.seconds_since_input() is None
    assert len(tools.calls) == 2


def test_idle_x11_without_libxss_falls_back_to_mutter(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    x11(monkeypatch, "GNOME")
    tools.install("gdbus")
    tools.respond(["gdbus"], (0, "(uint64 2000,)"))
    assert plat.seconds_since_input() == pytest.approx(2.0)


def test_idle_transient_failure_is_retried_soon(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, clock: Clock
) -> None:
    # gnome-shell stalls once (monitor hotplug, GPU hang): typing must not vanish for 60 s.
    wayland(monkeypatch, "GNOME")
    tools.install("gdbus")
    answers: list[Any] = [(0, "(uint64 100,)"), subprocess.TimeoutExpired(["gdbus"], 1.0)]

    def respond(argv: list[str]) -> tuple[int, str]:
        answer = answers.pop(0) if answers else (0, "(uint64 200,)")
        if isinstance(answer, BaseException):
            raise answer
        return answer

    tools.respond(["gdbus"], respond)
    assert plat.seconds_since_input() == pytest.approx(0.1)
    clock.advance(1.0)
    assert plat.seconds_since_input() is None
    clock.advance(0.5)
    assert plat.seconds_since_input() is None  # short back-off
    clock.advance(0.6)
    assert plat.seconds_since_input() == pytest.approx(0.2)
    assert len(tools.calls) == 3
    assert tools.kwargs[0]["timeout"] == pytest.approx(linux._IDLE_QUERY_TIMEOUT_S)


def test_idle_through_dbus_in_process(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    dbus: FakeDBus,
    clock: Clock,
) -> None:
    wayland(monkeypatch, "GNOME")
    tools.install("gdbus")
    dbus.answers["GetIdletime"] = linux._DBusReply((12345,))
    assert plat.seconds_since_input() == pytest.approx(12.345)
    clock.advance(0.6)
    assert plat.seconds_since_input() == pytest.approx(12.345)
    assert tools.calls == []  # no gdbus process twice a second
    call = dbus.calls[0]
    assert call["system"] is False
    assert call["service"] == "org.gnome.Mutter.IdleMonitor"
    assert call["path"] == "/org/gnome/Mutter/IdleMonitor/Core"


def test_idle_dbus_missing_service_backs_off(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, dbus: FakeDBus, clock: Clock
) -> None:
    wayland(monkeypatch, "KDE")
    dbus.answers["GetIdletime"] = linux._DBusReply((), "org.freedesktop.DBus.Error.ServiceUnknown")
    assert plat.seconds_since_input() is None
    clock.advance(30.0)
    assert plat.seconds_since_input() is None
    assert len(dbus.calls) == 1


def test_idle_dbus_hiccup_after_success_retries_soon(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, dbus: FakeDBus, clock: Clock
) -> None:
    wayland(monkeypatch, "GNOME")
    replies = [
        linux._DBusReply((500,)),
        linux._DBusReply((), "org.freedesktop.DBus.Error.NoReply"),
        linux._DBusReply((700,)),
    ]
    dbus.answers["GetIdletime"] = lambda call: replies.pop(0)
    assert plat.seconds_since_input() == pytest.approx(0.5)
    clock.advance(0.6)
    assert plat.seconds_since_input() is None
    clock.advance(1.1)
    assert plat.seconds_since_input() == pytest.approx(0.7)


def test_idle_unexpected_dbus_reply_falls_back_to_gdbus(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, dbus: FakeDBus
) -> None:
    wayland(monkeypatch, "GNOME")
    tools.install("gdbus")
    tools.respond(["gdbus"], (0, "(uint64 1500,)"))
    dbus.answers["GetIdletime"] = linux._DBusReply(("1500",))
    assert plat.seconds_since_input() == pytest.approx(1.5)
    assert tools.names() == ["gdbus"]


def test_idle_ignores_reset_caused_by_our_ydotool_move(
    monkeypatch: pytest.MonkeyPatch,
    plat: linux.LinuxPlatform,
    tools: FakeTools,
    clock: Clock,
    tmp_path: Path,
) -> None:
    wayland(monkeypatch, "GNOME")
    ydotool_ready(monkeypatch, tools, tmp_path)
    tools.install("gdbus")
    idle_ms = [5000, 600, 100]
    tools.respond(["gdbus"], lambda argv: (0, f"(uint64 {idle_ms.pop(0)},)"))
    assert plat.seconds_since_input() == pytest.approx(5.0)  # last input at 995
    assert plat.move_cursor(960, 540) is True  # uinput motion resets Mutter's idle timer
    clock.advance(0.6)
    # Mutter now says 0.6 s, but that reset was ours: the user has been idle for 5.6 s.
    assert plat.seconds_since_input() == pytest.approx(5.6)
    clock.advance(2.4)
    assert plat.seconds_since_input() == pytest.approx(0.1)  # real input later is seen


# ------------------------------------------------------------------ cursor
def test_cursor_x11_leaves_it_to_qt(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    x11(monkeypatch)
    tools.install("ydotool")
    assert plat.move_cursor(10, 20) is None
    assert tools.calls == []


_YDOTOOL_1_HELP = (
    "Usage: mousemove [OPTION]... [-x <xpos> -y <ypos>] [-- <xpos> <ypos>]\n"
    "  -a, --absolute             Use absolute position, not applied to wheel\n"
)


def ydotool_ready(monkeypatch: pytest.MonkeyPatch, tools: FakeTools, tmp_path: Path) -> None:
    """ydotool 1.x installed and ydotoold listening."""
    socket_path = tmp_path / "ydotool.sock"
    socket_path.write_text("")
    monkeypatch.setenv("YDOTOOL_SOCKET", str(socket_path))
    tools.install("ydotool")
    tools.respond(["ydotool", "mousemove", "--help"], (0, _YDOTOOL_1_HELP))
    tools.respond(["ydotool", "mousemove", "--absolute"], (0, ""))


def test_cursor_wayland_uses_ydotool(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, tmp_path: Path
) -> None:
    wayland(monkeypatch, "GNOME")
    ydotool_ready(monkeypatch, tools, tmp_path)
    assert plat.move_cursor(2560, -20) is True
    assert plat.move_cursor(10, 20) is True
    assert tools.calls == [
        ["ydotool", "mousemove", "--help"],  # probed once
        ["ydotool", "mousemove", "--absolute", "-x", "2560", "-y", "-20"],
        ["ydotool", "mousemove", "--absolute", "-x", "10", "-y", "20"],
    ]


def test_cursor_wayland_without_ydotool(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, tmp_path: Path
) -> None:
    wayland(monkeypatch, "GNOME")
    assert plat.move_cursor(1, 2) is False
    ydotool_ready(monkeypatch, tools, tmp_path)
    tools.respond(["ydotool", "mousemove", "--absolute"], (1, "failed to connect socket"))
    assert plat.move_cursor(1, 2) is False


def test_cursor_old_ydotool_is_unsupported(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, tmp_path: Path
) -> None:
    # Debian/Ubuntu still ship 0.1.x, whose CLI has no --absolute.
    wayland(monkeypatch, "GNOME")
    ydotool_ready(monkeypatch, tools, tmp_path)
    tools.respond(
        ["ydotool", "mousemove", "--help"], (1, "Usage: mousemove [--delay <ms>] <x> <y>")
    )
    assert plat.move_cursor(1, 2) is False
    assert plat.capabilities()["cursor"] is False
    assert tools.calls == [["ydotool", "mousemove", "--help"]]


def test_cursor_ydotool_needs_its_daemon(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, tmp_path: Path
) -> None:
    wayland(monkeypatch, "GNOME")
    ydotool_ready(monkeypatch, tools, tmp_path)
    monkeypatch.setenv("YDOTOOL_SOCKET", str(tmp_path / "missing.sock"))
    assert plat.move_cursor(1, 2) is False
    assert plat.capabilities()["cursor"] is False
    runtime = tmp_path / "run"
    runtime.mkdir()
    (runtime / ".ydotool_socket").write_text("")
    monkeypatch.delenv("YDOTOOL_SOCKET")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    assert plat.move_cursor(1, 2) is True


def test_cursor_sway_moves_through_its_ipc(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, tmp_path: Path
) -> None:
    wayland(monkeypatch, "sway")
    monkeypatch.setenv("SWAYSOCK", "/run/user/1000/sway-ipc.sock")
    ydotool_ready(monkeypatch, tools, tmp_path)
    tools.install("swaymsg")
    tools.respond(["swaymsg"], (0, ""))
    assert plat.move_cursor(2880, 540) is True
    assert tools.calls == [["swaymsg", "seat", "-", "cursor", "set", "2880", "540"]]
    # Older sway without the "-" seat alias: the default seat name.
    tools.calls.clear()
    tools.respond(["swaymsg", "seat", "-"], (2, "Error: seat - not found"))
    assert plat.move_cursor(1, 2) is True
    assert tools.calls[-1] == ["swaymsg", "seat", "seat0", "cursor", "set", "1", "2"]


def test_cursor_hyprland_moves_through_its_ipc(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    wayland(monkeypatch, "Hyprland")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc")
    tools.install("hyprctl")
    tools.respond(["hyprctl"], (0, "ok\n"))
    assert plat.move_cursor(100, 200) is True
    assert tools.calls == [["hyprctl", "dispatch", "movecursor", "100", "200"]]
    tools.respond(["hyprctl"], (0, "Invalid dispatcher\n"))  # hyprctl exits 0 regardless
    assert plat.move_cursor(100, 200) is False


def test_cursor_position_reliability(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform
) -> None:
    x11(monkeypatch)
    assert plat.cursor_position_reliable() is True
    wayland(monkeypatch, "GNOME")  # XWayland only sees the pointer over X11 windows
    assert plat.cursor_position_reliable() is False


# ------------------------------------------------------------------ camera in use
class ProcTree:
    """A fake ``/proc`` + ``/dev`` + ``/sys`` tree; fd "symlinks" are text files."""

    def __init__(self, root: Path) -> None:
        self.proc = root / "proc"
        self.dev = root / "dev"
        self.sys = root / "sys"
        self.proc.mkdir()
        self.dev.mkdir()
        (self.sys / "class" / "video4linux").mkdir(parents=True)
        (self.proc / "self").mkdir()  # non-numeric entries are ignored

    def device(self, name: str, physical: bool = True) -> None:
        (self.dev / name).write_text("")
        entry = self.sys / "class" / "video4linux" / name
        entry.mkdir()
        if physical:
            (entry / "device").mkdir()

    def process(
        self,
        pid: int,
        targets: list[str],
        comm: str = "app",
        maps: str = "",
        loginuid: int | None = None,
    ) -> None:
        base = self.proc / str(pid)
        (base / "fd").mkdir(parents=True, exist_ok=True)
        for fd, target in enumerate(targets):
            (base / "fd" / str(fd)).write_text(target)
        (base / "comm").write_text(comm + "\n")
        (base / "maps").write_text(maps)
        if loginuid is not None:
            (base / "loginuid").write_text(str(loginuid))

    def login(self, loginuid: int) -> None:
        """This process's audit login UID (``/proc/self/loginuid``)."""
        (self.proc / "self" / "loginuid").write_text(str(loginuid))

    def remove(self, pid: int) -> None:
        shutil.rmtree(self.proc / str(pid))


@pytest.fixture
def proc_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProcTree:
    tree = ProcTree(tmp_path)
    monkeypatch.setattr(linux, "_readlink", lambda path: Path(path).read_text())
    return tree


def test_camera_used_by_another_process(proc_tree: ProcTree, plat: linux.LinuxPlatform) -> None:
    proc_tree.device("video0")
    proc_tree.process(100, ["/dev/null", "socket:[123]"])
    proc_tree.process(200, ["/dev/null", "/dev/video0"])
    assert plat.camera_in_use_by_other_app() is True


def test_camera_held_only_by_us(proc_tree: ProcTree, plat: linux.LinuxPlatform) -> None:
    proc_tree.device("video0")
    proc_tree.process(os.getpid(), ["/dev/video0"])
    proc_tree.process(100, ["/dev/null"])
    assert plat.camera_in_use_by_other_app() is False


def test_camera_scan_ignores_unreadable_entries(
    proc_tree: ProcTree, plat: linux.LinuxPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_tree.device("video0")
    proc_tree.process(100, ["/dev/video0"])
    (proc_tree.proc / "300").mkdir()  # no fd directory: exited / not ours

    def readlink(path: str) -> str:
        if Path(path).parent.parent.name == "100":
            raise PermissionError(path)
        return Path(path).read_text()

    monkeypatch.setattr(linux, "_readlink", readlink)
    assert plat.camera_in_use_by_other_app() is False


def test_camera_deleted_suffix(proc_tree: ProcTree, plat: linux.LinuxPlatform) -> None:
    proc_tree.device("video0")
    proc_tree.process(100, ["/dev/video0 (deleted)"])
    assert plat.camera_in_use_by_other_app() is True


def test_camera_pipewire_monitoring_is_not_use(
    proc_tree: ProcTree, plat: linux.LinuxPlatform
) -> None:
    proc_tree.device("video0")
    proc_tree.process(100, ["/dev/video0"], comm="wireplumber")
    proc_tree.process(101, ["/dev/video0"], comm="pipewire")
    assert plat.camera_in_use_by_other_app() is False


def test_camera_pipewire_streaming_is_use(proc_tree: ProcTree, plat: linux.LinuxPlatform) -> None:
    proc_tree.device("video0")
    maps = "7f00-7f10 rw-s 00000000 00:05 42   /dev/video0\n"
    proc_tree.process(101, ["/dev/video0"], comm="pipewire", maps=maps)
    assert plat.camera_in_use_by_other_app() is True


def test_camera_virtual_loopback_is_ignored(proc_tree: ProcTree, plat: linux.LinuxPlatform) -> None:
    proc_tree.device("video0")
    proc_tree.device("video10", physical=False)  # OBS virtual camera
    proc_tree.process(100, ["/dev/video10"], comm="obs")
    assert plat.camera_in_use_by_other_app() is False


def test_camera_no_devices_means_not_in_use(proc_tree: ProcTree, plat: linux.LinuxPlatform) -> None:
    proc_tree.process(100, ["/dev/video0"])  # stale: no device node exists
    assert plat.camera_in_use_by_other_app() is False


def test_camera_without_dev_info_matches_any_video_node(
    tmp_path: Path, proc_tree: ProcTree, clock: Clock
) -> None:
    proc_tree.process(100, ["/dev/video3"])
    platform = linux.LinuxPlatform(
        proc_root=proc_tree.proc,
        dev_root=tmp_path / "missing",
        sys_root=proc_tree.sys,
        clock=clock,
        background_threads=False,
    )
    assert platform.camera_in_use_by_other_app() is True


def test_camera_unknown_without_proc(tmp_path: Path, clock: Clock) -> None:
    platform = linux.LinuxPlatform(
        proc_root=tmp_path / "nope", clock=clock, background_threads=False
    )
    assert platform.camera_in_use_by_other_app() is None
    assert platform.capabilities()["camera_in_use"] is False


def test_camera_scan_skips_other_users_processes(
    proc_tree: ProcTree, plat: linux.LinuxPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_tree.device("video0")
    proc_tree.process(100, ["/dev/video0"])
    owner = os.stat(proc_tree.proc / "100").st_uid
    monkeypatch.setattr(linux, "_readlink", lambda path: pytest.fail("read another user's fds"))
    monkeypatch.setattr(linux, "_getuid", lambda: owner + 1)
    assert plat.camera_in_use_by_other_app() is False


def test_camera_result_is_cached(
    proc_tree: ProcTree, plat: linux.LinuxPlatform, clock: Clock
) -> None:
    proc_tree.device("video0")
    assert plat.camera_in_use_by_other_app() is False
    proc_tree.process(100, ["/dev/video0"])
    clock.advance(2.0)
    assert plat.camera_in_use_by_other_app() is False
    clock.advance(1.5)
    assert plat.camera_in_use_by_other_app() is True


def test_camera_permission(proc_tree: ProcTree, plat: linux.LinuxPlatform) -> None:
    assert plat.permissions() == {"camera": None, "accessibility": None}
    proc_tree.device("video0")
    assert plat.permissions()["camera"] is True


# ------------------------------------------------------------------ camera: refused opens
def opened(foreign: bool | None) -> list[linux._CameraEvent]:
    """A capture attempt that failed at once: read-write open, then close (merged)."""
    return [linux._CameraEvent(opened=True, foreign=foreign, wrote=True)]


@pytest.fixture
def streaming(proc_tree: ProcTree) -> ProcTree:
    """A physical camera that this process streams from."""
    proc_tree.device("video0")
    proc_tree.process(os.getpid(), ["/dev/video0"], comm="eye-tracker")
    return proc_tree


def test_refused_open_releases_the_camera_for_a_grace_period(
    streaming: ProcTree, plat: linux.LinuxPlatform, clock: Clock
) -> None:
    # A browser joining a call opens the node, gets EBUSY and closes it again at once.
    assert plat.camera_in_use_by_other_app() is False
    plat._camera.handle_events(opened(foreign=True))
    assert plat.camera_in_use_by_other_app() is True  # no holder, but an attempt
    clock.advance(plat._camera.grace_s - 1.0)
    assert plat.camera_in_use_by_other_app() is True
    clock.advance(2.0)
    assert plat.camera_in_use_by_other_app() is False  # nobody took it: resume


def test_refused_open_then_app_takes_the_camera(
    streaming: ProcTree, plat: linux.LinuxPlatform, clock: Clock
) -> None:
    plat._camera.handle_events(opened(foreign=True))
    assert plat.camera_in_use_by_other_app() is True
    # We released the camera; the app retried and now streams.
    streaming.process(os.getpid(), [], comm="eye-tracker")
    (streaming.proc / str(os.getpid()) / "fd" / "0").unlink()
    streaming.process(700, ["/dev/video0"], comm="chrome")
    clock.advance(3.5)
    assert plat.camera_in_use_by_other_app() is True
    clock.advance(plat._camera.grace_s)
    assert plat.camera_in_use_by_other_app() is True  # held: stays released
    streaming.remove(700)
    clock.advance(3.5)
    assert plat.camera_in_use_by_other_app() is False
    assert plat._camera._false_alarms == 0


@pytest.mark.parametrize("foreign", [False, None])
def test_own_or_unattributed_opens_are_not_contention(
    streaming: ProcTree, plat: linux.LinuxPlatform, foreign: bool | None
) -> None:
    # Our own probe/reopen (foreign=False), or inotify that cannot tell (None).
    plat._camera.handle_events(opened(foreign=foreign))
    assert plat.camera_in_use_by_other_app() is False


def test_device_listing_is_not_contention(streaming: ProcTree, plat: linux.LinuxPlatform) -> None:
    # A browser enumerating cameras opens each node read-only and closes it.
    plat._camera.handle_events(
        [
            linux._CameraEvent(opened=True, foreign=True),
            linux._CameraEvent(opened=False, foreign=True),
        ]
    )
    assert plat.camera_in_use_by_other_app() is False


def test_app_keeping_the_node_open_is_found_by_the_rescan(
    streaming: ProcTree, plat: linux.LinuxPlatform, clock: Clock
) -> None:
    assert plat.camera_in_use_by_other_app() is False
    streaming.process(700, ["/dev/video0"], comm="zoom")  # opened, retrying, not closed yet
    plat._camera.handle_events([linux._CameraEvent(opened=True, foreign=True)])
    clock.advance(plat._camera.debounce_s)
    assert plat.camera_in_use_by_other_app() is True


def test_foreign_open_while_we_do_not_stream_is_not_contention(
    proc_tree: ProcTree, plat: linux.LinuxPlatform
) -> None:
    proc_tree.device("video0")
    plat._camera.handle_events(opened(foreign=True))  # it will succeed: no conflict
    assert plat.camera_in_use_by_other_app() is False


def test_repeated_false_alarms_back_off(
    streaming: ProcTree, plat: linux.LinuxPlatform, clock: Clock
) -> None:
    watcher = plat._camera

    def false_alarm() -> None:
        watcher.handle_events(opened(foreign=True))
        assert plat.camera_in_use_by_other_app() is True
        clock.advance(watcher.grace_s + 0.1)
        assert plat.camera_in_use_by_other_app() is False

    false_alarm()  # the first one is free: the user may simply have been slow to retry
    false_alarm()
    # An app keeps probing in the background: ignored for a while ...
    watcher.handle_events(opened(foreign=True))
    assert plat.camera_in_use_by_other_app() is False
    clock.advance(watcher.cooldown_s)
    false_alarm()  # ... then honoured again, with a longer cooldown afterwards
    clock.advance(watcher.cooldown_s)
    watcher.handle_events(opened(foreign=True))
    assert plat.camera_in_use_by_other_app() is False
    clock.advance(watcher.cooldown_s)
    watcher.handle_events(opened(foreign=True))
    assert plat.camera_in_use_by_other_app() is True


def _fan_event(mask: int, pid: int, length: int = 24) -> bytes:
    header = struct.pack("=IBBHQii", length, 3, 0, 24, mask, -1, pid)
    return header + b"\0" * (length - len(header))


def test_parse_fanotify_attributes_opens() -> None:
    own = 4242
    data = (
        _fan_event(0x20 | 0x10, own, length=56)  # our read-only open+close, merged, FID record
        + _fan_event(0x20, 0)  # another process (pid hidden from unprivileged listeners)
        + _fan_event(0x08, 0)  # ... closing a read-write descriptor
        + _fan_event(0x20 | 0x08, 0)  # a refused capture attempt, merged
        + _fan_event(0x4000, 0)  # queue overflow: unknown
        + _fan_event(0x01, 0)  # an access event: not asked for, ignored
    )
    assert linux._parse_fanotify(data, own) == [
        linux._CameraEvent(opened=True, foreign=False),
        linux._CameraEvent(opened=True, foreign=True),
        linux._CameraEvent(opened=False, foreign=True, wrote=True),
        linux._CameraEvent(opened=True, foreign=True, wrote=True),
        linux._CameraEvent(opened=False, foreign=None),
    ]
    assert linux._parse_fanotify(_fan_event(0x20, 0, length=0) * 3, own) == []  # malformed


def test_parse_inotify_has_no_attribution() -> None:
    data = (
        struct.pack("=iIII", 1, 0x20, 0, 0)
        + struct.pack("=iIII", 1, 0x10, 0, 4)
        + b"x\0\0\0"
        + struct.pack("=iIII", 1, 0x08, 0, 0)
    )
    assert linux._parse_inotify(data) == [
        linux._CameraEvent(opened=True, foreign=None),
        linux._CameraEvent(opened=False, foreign=None),
        linux._CameraEvent(opened=False, foreign=None, wrote=True),
    ]


def test_no_notifier_without_nodes() -> None:
    assert linux._open_camera_notifier([]) is None


def test_notifier_falls_back_when_the_kernel_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """No fanotify (kernel < 5.13, a sandbox): inotify; neither: ``None``, the watcher polls."""
    monkeypatch.setattr(sys, "platform", "linux")
    opened: list[tuple[str, list[str]]] = []

    def refused(error: Exception) -> type:
        class Refused:
            kind = "refused"

            def __init__(self, paths: list[str]) -> None:
                raise error

        return Refused

    class Inotify:
        kind = "inotify"

        def __init__(self, paths: list[str]) -> None:
            opened.append((self.kind, list(paths)))

    monkeypatch.setattr(linux, "_FanotifyNotifier", refused(OSError(1, "Operation not permitted")))
    monkeypatch.setattr(linux, "_InotifyNotifier", Inotify)
    assert isinstance(linux._open_camera_notifier(["/dev/video0"]), Inotify)
    assert opened == [("inotify", ["/dev/video0"])]
    # A C library without the wrapper is no reason to fail either.
    monkeypatch.setattr(linux, "_FanotifyNotifier", refused(AttributeError("fanotify_init")))
    assert isinstance(linux._open_camera_notifier(["/dev/video0"]), Inotify)
    monkeypatch.setattr(linux, "_InotifyNotifier", refused(OSError(24, "Too many open files")))
    assert linux._open_camera_notifier(["/dev/video0"]) is None


# ------------------------------------------------------------------ camera: background thread
class FakeNotifier:
    kind = "fake"

    def __init__(self) -> None:
        self.events: list[list[linux._CameraEvent]] = []
        self.ready = threading.Event()
        self.closed = False
        self.broken = False
        self.waits = 0

    @property
    def alive(self) -> bool:
        return not self.broken and not self.closed

    def push(self, events: list[linux._CameraEvent]) -> None:
        self.events.append(events)
        self.ready.set()

    def break_down(self) -> None:
        """Fail like a descriptor that cannot be read any more (see _FdNotifier.wait)."""
        self.broken = True
        self.push([linux._CameraEvent(opened=False, foreign=None)])

    def wait(self, timeout: float) -> list[linux._CameraEvent]:
        self.waits += 1
        if self.ready.wait(timeout):
            self.ready.clear()
            return self.events.pop(0) if self.events else []
        return []

    def close(self) -> None:
        self.closed = True


def _eventually(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def threaded(proc_tree: ProcTree) -> Iterable[tuple[linux.LinuxPlatform, list[FakeNotifier]]]:
    notifiers: list[FakeNotifier] = []

    def factory(paths: Any) -> FakeNotifier:
        notifiers.append(FakeNotifier())
        return notifiers[-1]

    platform = linux.LinuxPlatform(
        proc_root=proc_tree.proc,
        dev_root=proc_tree.dev,
        sys_root=proc_tree.sys,
        camera_notifier_factory=factory,
    )
    platform._camera.max_wait_s = 0.02
    try:
        yield platform, notifiers
    finally:
        platform._camera.stop()


def test_camera_scan_runs_off_the_calling_thread(
    proc_tree: ProcTree, threaded: tuple[linux.LinuxPlatform, list[FakeNotifier]]
) -> None:
    platform, notifiers = threaded
    proc_tree.device("video0")
    proc_tree.process(100, ["/dev/video0"])
    scanners: list[threading.Thread] = []
    scan = platform._camera._scan

    def recording_scan(devices: Any) -> bool | None:
        scanners.append(threading.current_thread())
        return scan(devices)

    platform._camera._scan = recording_scan  # type: ignore[method-assign]
    assert _eventually(lambda: platform.camera_in_use_by_other_app() is True)
    assert scanners
    assert threading.current_thread() not in scanners
    assert len(notifiers) == 1  # event-driven: the node is watched


def test_camera_thread_rescans_on_open_events(
    proc_tree: ProcTree, threaded: tuple[linux.LinuxPlatform, list[FakeNotifier]]
) -> None:
    platform, notifiers = threaded
    proc_tree.device("video0")
    assert _eventually(lambda: platform.camera_in_use_by_other_app() is False)
    assert _eventually(lambda: bool(notifiers))
    proc_tree.process(100, ["/dev/video0"])  # an app starts streaming ...
    notifiers[0].push([linux._CameraEvent(opened=True, foreign=None)])  # ... and is noticed
    assert _eventually(lambda: platform.camera_in_use_by_other_app() is True)


def test_camera_thread_stops_when_nobody_asks(
    proc_tree: ProcTree, threaded: tuple[linux.LinuxPlatform, list[FakeNotifier]]
) -> None:
    platform, notifiers = threaded
    proc_tree.device("video0")
    platform._camera.idle_stop_s = 0.1
    platform.camera_in_use_by_other_app()
    assert _eventually(lambda: platform._camera._thread is None)
    assert _eventually(lambda: bool(notifiers) and notifiers[0].closed)
    platform.camera_in_use_by_other_app()  # asking again restarts it
    assert _eventually(lambda: len(notifiers) == 2)


def test_camera_answer_after_an_idle_stop_is_fresh(
    proc_tree: ProcTree, threaded: tuple[linux.LinuxPlatform, list[FakeNotifier]]
) -> None:
    """Resuming after a pause: the forced check must not read the pre-pause answer."""
    platform, _notifiers = threaded
    watcher = platform._camera
    watcher.idle_stop_s = 0.1
    watcher.first_scan_wait_s = 10.0  # generous for slow CI; it returns once the scan is done
    proc_tree.device("video0")
    assert _eventually(lambda: platform.camera_in_use_by_other_app() is False)
    assert _eventually(lambda: watcher._thread is None)  # paused: nobody asked
    proc_tree.process(100, ["/dev/video0"])  # a video call started meanwhile
    assert platform.camera_in_use_by_other_app() is True  # at once, not one poll later
    proc_tree.remove(100)
    assert _eventually(lambda: watcher._thread is None)
    assert platform.camera_in_use_by_other_app() is False  # and the reverse


def test_camera_thread_that_dies_early_does_not_stall_the_query(
    proc_tree: ProcTree, threaded: tuple[linux.LinuxPlatform, list[FakeNotifier]]
) -> None:
    platform, _notifiers = threaded
    watcher = platform._camera
    watcher.first_scan_wait_s = 30.0

    def broken_devices() -> set[str] | None:
        raise RuntimeError("sysfs vanished")

    watcher._devices = broken_devices  # type: ignore[method-assign]
    started = time.monotonic()
    assert platform.camera_in_use_by_other_app() is None  # unknown, not stale
    assert time.monotonic() - started < 5.0


def test_camera_watcher_polls_once_its_notifier_broke(
    proc_tree: ProcTree, threaded: tuple[linux.LinuxPlatform, list[FakeNotifier]]
) -> None:
    """Not the 30 s safety rescan of a working notifier, and no new notifier per failure."""
    platform, notifiers = threaded
    watcher = platform._camera
    watcher.poll_s = 0.05
    watcher.safety_rescan_s = 3600.0
    proc_tree.device("video0")
    assert _eventually(lambda: platform.camera_in_use_by_other_app() is False)
    assert _eventually(lambda: bool(notifiers))
    notifiers[0].break_down()
    assert _eventually(lambda: notifiers[0].closed)
    waits = notifiers[0].waits
    proc_tree.process(100, ["/dev/video0"])  # no event will tell: only polling finds it
    assert _eventually(lambda: platform.camera_in_use_by_other_app() is True)
    assert len(notifiers) == 1
    assert notifiers[0].waits == waits  # the dead notifier is not waited on any more


def test_fd_notifier_reports_a_failing_descriptor_once(tmp_path: Path) -> None:
    class Notifier(linux._FdNotifier):
        def _parse(self, data: bytes) -> list[linux._CameraEvent]:
            raise AssertionError("nothing can be read")

    # A write-only file: waiting fails (Windows: select() wants sockets) or reading does.
    fd = os.open(tmp_path / "node", os.O_WRONLY | os.O_CREAT)
    notifier = Notifier(fd)
    assert notifier.alive
    assert notifier.wait(0.01) == [linux._CameraEvent(opened=False, foreign=None)]
    assert not notifier.alive  # closed: the watcher polls from now on
    assert notifier._fd == -1
    assert notifier.wait(0.0) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fanotify/inotify are Linux APIs")
@pytest.mark.parametrize("kind", ["fanotify", "inotify"])
def test_real_notifier_reports_opens(tmp_path: Path, kind: str) -> None:
    node = tmp_path / "video0"
    node.write_text("")
    cls = linux._FanotifyNotifier if kind == "fanotify" else linux._InotifyNotifier
    try:
        notifier = cls([str(node)])
    except OSError as exc:
        pytest.skip(f"{kind} unavailable here: {exc}")
    try:
        with open(node, "rb"):
            pass
        # The child opens read-write, like a capture attempt. (A real process: the
        # faked subprocess.run never started it, and only our own open was seen.)
        _real_subprocess_run(
            [sys.executable, "-c", f"open({str(node)!r}, 'r+b').close()"], check=True, timeout=60
        )
        events: list[linux._CameraEvent] = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any(e.wrote for e in events):
            events += notifier.wait(0.2)
    finally:
        notifier.close()
    opens = [event.foreign for event in events if event.opened]
    writes = [event.foreign for event in events if event.wrote]
    if kind == "fanotify":
        assert opens == [False, True]  # ours, then the child process's
        assert writes == [True]
    else:
        assert opens == [None, None]
        assert writes == [None]


# ------------------------------------------------------------------ EWMH windows
class FakeWindow:
    def __init__(self, display: FakeDisplay, wid: int) -> None:
        self.display = display
        self.id = wid
        self.props: dict[str, list[int]] = {}
        self.geometry = (0, 0, 100, 100)
        self.map_state = linux._X_IS_VIEWABLE
        self.sent: list[tuple[Any, int]] = []

    def get_full_property(self, atom: int, prop_type: int) -> Any:
        assert prop_type == linux._X_ANY_PROPERTY_TYPE
        values = self.props.get(self.display.atom_names[atom])
        return None if values is None else SimpleNamespace(value=list(values))

    def get_geometry(self) -> Any:
        return SimpleNamespace(width=self.geometry[2], height=self.geometry[3])

    def get_attributes(self) -> Any:
        return SimpleNamespace(map_state=self.map_state)

    def translate_coords(self, src: FakeWindow, x: int, y: int) -> Any:
        assert self.id == self.display.root.id  # only root-relative translations are used
        return SimpleNamespace(x=src.geometry[0] + x, y=src.geometry[1] + y)

    def send_event(self, event: Any, event_mask: int = 0) -> None:
        self.sent.append((event, event_mask))


class FakeDisplay:
    def __init__(self) -> None:
        self.atoms: dict[str, int] = {}
        self.atom_names: dict[int, str] = {}
        self.windows: dict[int, FakeWindow] = {}
        self.root = FakeWindow(self, 1)
        self.flushed = 0
        # DPMS extension state
        self.extensions = {"DPMS"}
        self.dpms_capable_flag = True
        self.dpms_enabled = True
        self.power_level = 0
        self.dpms_log: list[str] = []

    def has_extension(self, name: str) -> bool:
        return name in self.extensions

    def dpms_capable(self) -> Any:
        return SimpleNamespace(capable=self.dpms_capable_flag)

    def dpms_info(self) -> Any:
        return SimpleNamespace(power_level=self.power_level, state=self.dpms_enabled)

    def dpms_enable(self) -> None:
        self.dpms_log.append("enable")
        self.dpms_enabled = True

    def dpms_disable(self) -> None:
        self.dpms_log.append("disable")
        self.dpms_enabled = False
        self.power_level = 0  # disabling DPMS turns the monitors back on

    def dpms_force_level(self, level: int) -> None:
        assert self.dpms_enabled, "BadMatch: DPMS is disabled"
        self.dpms_log.append(f"force {level}")
        self.power_level = level

    def sync(self) -> None:
        pass

    def intern_atom(self, name: str) -> int:
        if name not in self.atoms:
            self.atoms[name] = len(self.atoms) + 100
            self.atom_names[self.atoms[name]] = name
        return self.atoms[name]

    def screen(self) -> Any:
        return SimpleNamespace(root=self.root)

    def create_resource_object(self, kind: str, wid: int) -> FakeWindow:
        assert kind == "window"
        if wid not in self.windows:
            raise LookupError(f"BadWindow {wid}")  # what python-xlib raises is an XError
        return self.windows[wid]

    def flush(self) -> None:
        self.flushed += 1

    def add(
        self, wid: int, rect: tuple[int, int, int, int], pid: int = 500, **props: list[Any]
    ) -> FakeWindow:
        win = FakeWindow(self, wid)
        win.geometry = rect
        win.props["_NET_WM_PID"] = [pid]
        for name, values in props.items():
            win.props[name] = [self.intern_atom(v) if isinstance(v, str) else v for v in values]
        self.windows[wid] = win
        for name in (*linux._SKIPPED_WINDOW_TYPES, "_NET_WM_STATE_HIDDEN"):
            self.intern_atom(name)
        return win

    def stack(self, *wids: int) -> None:
        self.root.props["_NET_CLIENT_LIST_STACKING"] = list(wids)


OWN_PID = 4242


@pytest.fixture
def xdisplay() -> FakeDisplay:
    return FakeDisplay()


@pytest.fixture
def x11_plat(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, xdisplay: FakeDisplay, clock: Clock
) -> linux.LinuxPlatform:
    x11(monkeypatch)
    client = linux._EwmhClient(lambda: xdisplay, clock=clock, own_pid=OWN_PID)

    def client_message(win: FakeWindow, atom: int, data: list[int]) -> Any:
        return ("ClientMessage", win.id, atom, data)

    monkeypatch.setattr(client, "_client_message", client_message)
    plat._ewmh = client
    return plat


def test_window_at_prefers_topmost_window(x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay):
    xdisplay.add(10, (0, 0, 1920, 1080))
    xdisplay.add(20, (100, 100, 800, 600), pid=501)
    xdisplay.add(30, (2000, 0, 800, 600), pid=502)  # other monitor
    xdisplay.stack(10, 20, 30)  # bottom → top
    ref = x11_plat.window_at(200, 200)
    assert ref is not None
    assert ref.handle == 20
    assert ref.pid == 501
    assert ref.rect == Rect(100, 100, 800, 600)
    assert x11_plat.window_at(2100, 50).handle == 30  # type: ignore[union-attr]
    assert x11_plat.window_at(5000, 5000) is None


def test_window_at_skips_hidden_unmapped_shell_and_own_windows(
    x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay
) -> None:
    xdisplay.add(5, (0, 0, 1920, 1080), _NET_WM_WINDOW_TYPE=["_NET_WM_WINDOW_TYPE_DESKTOP"])
    xdisplay.add(10, (0, 0, 1000, 1000), pid=501)
    xdisplay.add(20, (0, 0, 1000, 1000), _NET_WM_STATE=["_NET_WM_STATE_HIDDEN"])
    xdisplay.add(30, (0, 0, 1000, 1000)).map_state = 1  # IsUnviewable: other workspace
    xdisplay.add(40, (0, 0, 1000, 1000), pid=OWN_PID)  # our overlay
    xdisplay.add(50, (0, 0, 1920, 40), _NET_WM_WINDOW_TYPE=["_NET_WM_WINDOW_TYPE_DOCK"])
    xdisplay.stack(5, 10, 20, 30, 40, 50, 99)  # 99 vanished between listing and query
    ref = x11_plat.window_at(10, 10)
    assert ref is not None
    assert ref.handle == 10
    del xdisplay.windows[10]
    assert x11_plat.window_at(10, 10) is None  # only the desktop is left


def test_window_rect_includes_decorations_and_drops_csd_shadow(
    x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay
) -> None:
    xdisplay.add(10, (100, 130, 800, 600), _NET_FRAME_EXTENTS=[2, 2, 30, 2])
    xdisplay.add(20, (0, 0, 848, 648), _GTK_FRAME_EXTENTS=[24, 24, 24, 24])
    assert x11_plat.window_rect(WindowRef(10)) == Rect(98, 100, 804, 632)
    assert x11_plat.window_rect(WindowRef(20)) == Rect(24, 24, 800, 600)


def test_foreground_window(x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay) -> None:
    xdisplay.add(20, (100, 100, 800, 600), pid=501)
    xdisplay.add(40, (0, 0, 10, 10), pid=OWN_PID)
    xdisplay.root.props["_NET_ACTIVE_WINDOW"] = [20]
    ref = x11_plat.foreground_window()
    assert ref is not None
    assert (ref.handle, ref.pid, ref.rect) == (20, 501, Rect(100, 100, 800, 600))
    xdisplay.root.props["_NET_ACTIVE_WINDOW"] = [40]
    assert x11_plat.foreground_window() is None  # our own window
    xdisplay.root.props["_NET_ACTIVE_WINDOW"] = [0]
    assert x11_plat.foreground_window() is None


def test_activate_sends_net_active_window(
    x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay
) -> None:
    xdisplay.add(20, (100, 100, 800, 600))
    assert x11_plat.activate_window(WindowRef(20)) is True
    [(event, mask)] = xdisplay.root.sent
    assert event == ("ClientMessage", 20, xdisplay.atoms["_NET_ACTIVE_WINDOW"], [2, 0, 0, 0, 0])
    assert mask == (1 << 20) | (1 << 19)
    assert xdisplay.flushed == 1


def test_activate_refuses_minimised_or_missing_windows(
    x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay
) -> None:
    xdisplay.add(20, (0, 0, 10, 10), _NET_WM_STATE=["_NET_WM_STATE_HIDDEN"])
    assert x11_plat.activate_window(WindowRef(20)) is False
    assert x11_plat.activate_window(WindowRef(77)) is False
    assert x11_plat.activate_window(WindowRef("bogus")) is False
    assert xdisplay.root.sent == []
    assert x11_plat.is_window_valid(WindowRef(20)) is False


def test_window_functions_unsupported_on_wayland(
    monkeypatch: pytest.MonkeyPatch, x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay
) -> None:
    xdisplay.add(20, (0, 0, 800, 600))
    xdisplay.stack(20)
    xdisplay.root.props["_NET_ACTIVE_WINDOW"] = [20]
    wayland(monkeypatch, "GNOME")  # XWayland only sees X11 clients: never trust it
    assert x11_plat.foreground_window() is None
    assert x11_plat.window_at(10, 10) is None
    assert x11_plat.activate_window(WindowRef(20)) is False
    assert x11_plat.window_rect(WindowRef(20)) is None
    assert xdisplay.root.sent == []


def test_x11_connection_failure_backs_off(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, clock: Clock
) -> None:
    x11(monkeypatch)
    attempts: list[int] = []

    def factory() -> Any:
        attempts.append(1)
        raise OSError("Can't connect to display")

    plat._ewmh = linux._EwmhClient(factory, clock=clock)
    assert plat.foreground_window() is None
    assert plat.window_at(0, 0) is None
    assert len(attempts) == 1
    clock.advance(61.0)
    assert plat.foreground_window() is None
    assert len(attempts) == 2


def test_x11_connection_loss_reconnects(
    x11_plat: linux.LinuxPlatform, xdisplay: FakeDisplay, clock: Clock
) -> None:
    xdisplay.add(20, (0, 0, 800, 600))
    xdisplay.root.props["_NET_ACTIVE_WINDOW"] = [20]
    assert x11_plat.foreground_window() is not None

    def broken(*args: Any) -> Any:
        raise ConnectionResetError("X server went away")

    xdisplay.root.get_full_property = broken  # type: ignore[method-assign,assignment]
    assert x11_plat.foreground_window() is None
    assert x11_plat._ewmh._display is None
    del xdisplay.root.get_full_property
    clock.advance(2.0)
    assert x11_plat.foreground_window() is not None


# ------------------------------------------------------------------ capabilities
def _capability_keys() -> set[str]:
    return set(PlatformServices().capabilities())


def test_capabilities_x11(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    x11(monkeypatch)
    tools.install("loginctl", "xset")
    monkeypatch.setattr(linux, "_module_available", lambda name: True)
    plat._xss = FakeXss(0)  # type: ignore[assignment]
    caps = plat.capabilities()
    assert set(caps) == _capability_keys()
    assert caps["lock"]
    assert caps["display_off"]
    assert caps["input_idle"]
    assert caps["session_locked"]
    assert caps["focus"]
    assert caps["cursor"]
    assert caps["hotkeys"]
    assert not caps["key_idle"]


def test_capabilities_wayland(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools, tmp_path: Path
) -> None:
    wayland(monkeypatch, "GNOME", display=False)
    tools.install("busctl", "gdbus")
    monkeypatch.setattr(linux, "_module_available", lambda name: True)
    caps = plat.capabilities()
    assert set(caps) == _capability_keys()
    assert caps["display_off"]
    assert caps["input_idle"]
    assert not caps["focus"]
    assert not caps["cursor"]
    assert not caps["hotkeys"]
    ydotool_ready(monkeypatch, tools, tmp_path)
    assert plat.capabilities()["cursor"]


def test_module_imports_cleanly_everywhere() -> None:
    assert linux.LinuxPlatform.name == "linux"
    assert issubclass(linux.LinuxPlatform, PlatformServices)
