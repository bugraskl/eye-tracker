"""Tests for the Linux platform layer.

They run on every OS: desktop tools, ``/proc`` and the X server are faked, so
nothing here locks the screen, touches displays or moves the real cursor.
"""

from __future__ import annotations

import os
import subprocess
import sys
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
)


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


@pytest.fixture
def plat(clock: Clock, tmp_path: Path) -> linux.LinuxPlatform:
    platform = linux.LinuxPlatform(
        proc_root=tmp_path / "proc",
        dev_root=tmp_path / "dev",
        sys_root=tmp_path / "sys",
        clock=clock,
    )
    platform._xss = FakeXss()  # type: ignore[assignment]
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
    assert os.environ["QT_QPA_PLATFORM"] == "xcb"
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
    assert os.environ["QT_QPA_PLATFORM"] == "xcb"  # our own process keeps them


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


def test_lock_resolves_display_session_without_session_id(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    tools.install("loginctl")
    tools.respond(["loginctl", "show-user"], (0, "3\n"))
    tools.respond(["loginctl", "lock-session"], (0, ""))
    assert plat.lock_screen() is True
    assert tools.calls == [
        ["loginctl", "show-user", "1000", "-p", "Display", "--value"],
        ["loginctl", "lock-session", "3"],
    ]


def test_lock_falls_back_to_callers_session(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    tools.install("loginctl")
    tools.respond(["loginctl", "show-user"], (0, "\n"))
    tools.respond(["loginctl", "lock-session"], (0, ""))
    assert plat.lock_screen() is True
    assert tools.calls[-1] == ["loginctl", "lock-session"]


# ------------------------------------------------------------------ display power
def test_display_off_x11_uses_xset(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    x11(monkeypatch, "XFCE")
    tools.install("xset", "kscreen-doctor", "busctl")
    tools.respond(["xset"], (0, ""))
    assert plat.display_off()
    assert plat.wake_display()
    assert tools.calls == [["xset", "dpms", "force", "off"], ["xset", "dpms", "force", "on"]]


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
    monkeypatch.setattr(plat._ewmh, "nudge_pointer", lambda: nudges.append(True) or True)
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
    tools.install("loginctl")
    hints = iter([(0, "no\n"), (0, "yes\n")])
    tools.respond(["loginctl", "show-session"], lambda argv: next(hints))
    tools.respond(["loginctl", "lock-session"], (0, ""))
    assert plat.is_session_locked() is False
    assert plat.lock_screen() is True
    assert plat.is_session_locked() is True


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


# ------------------------------------------------------------------ cursor
def test_cursor_x11_leaves_it_to_qt(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    x11(monkeypatch)
    tools.install("ydotool")
    assert plat.move_cursor(10, 20) is None
    assert tools.calls == []


def test_cursor_wayland_uses_ydotool(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    wayland(monkeypatch, "GNOME")
    tools.install("ydotool")
    tools.respond(["ydotool"], (0, ""))
    assert plat.move_cursor(2560, -20) is True
    assert tools.calls == [["ydotool", "mousemove", "--absolute", "-x", "2560", "-y", "-20"]]


def test_cursor_wayland_without_ydotool(
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
) -> None:
    wayland(monkeypatch, "GNOME")
    assert plat.move_cursor(1, 2) is False
    tools.install("ydotool")
    tools.respond(["ydotool"], (1, "failed to connect socket"))
    assert plat.move_cursor(1, 2) is False


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

    def process(self, pid: int, targets: list[str], comm: str = "app", maps: str = "") -> None:
        base = self.proc / str(pid)
        (base / "fd").mkdir(parents=True)
        for fd, target in enumerate(targets):
            (base / "fd" / str(fd)).write_text(target)
        (base / "comm").write_text(comm + "\n")
        (base / "maps").write_text(maps)


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
        proc_root=proc_tree.proc, dev_root=tmp_path / "missing", sys_root=proc_tree.sys, clock=clock
    )
    assert platform.camera_in_use_by_other_app() is True


def test_camera_unknown_without_proc(tmp_path: Path, clock: Clock) -> None:
    platform = linux.LinuxPlatform(proc_root=tmp_path / "nope", clock=clock)
    assert platform.camera_in_use_by_other_app() is None
    assert platform.capabilities()["camera_in_use"] is False


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

    xdisplay.root.get_full_property = broken  # type: ignore[method-assign]
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
    monkeypatch: pytest.MonkeyPatch, plat: linux.LinuxPlatform, tools: FakeTools
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
    tools.install("ydotool")
    assert plat.capabilities()["cursor"]


def test_module_imports_cleanly_everywhere() -> None:
    assert linux.LinuxPlatform.name == "linux"
    assert issubclass(linux.LinuxPlatform, PlatformServices)
