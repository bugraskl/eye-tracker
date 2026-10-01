"""Tests for ``eye_tracker.platform`` (factory) and ``platform.windows``.

The policy code in :class:`WindowsPlatform` is exercised on every OS through a
fake Win32 binding and a fake registry. A handful of live tests run only on
Windows and only call read-only APIs: nothing here locks the screen, turns a
display off, synthesises input, moves the cursor or activates a window.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import eye_tracker.platform as platform_pkg
from eye_tracker import APP_ID
from eye_tracker.platform import windows
from eye_tracker.platform.base import PlatformServices
from eye_tracker.platform.windows import WindowsPlatform
from eye_tracker.types import AppIdentity, Rect, WindowRef

ON_WINDOWS = sys.platform == "win32"
windows_only = pytest.mark.skipif(not ON_WINDOWS, reason="needs the Win32 API")

OWN_PID = 4242
OUR_THREAD = 1


# --------------------------------------------------------------------- fakes
@dataclass
class FakeWindow:
    hwnd: int
    pid: int = 100
    tid: int = 10
    cls: str = "Notepad"
    rect: Rect | None = Rect(0, 0, 800, 600)  # noqa: RUF009 - Rect is immutable
    visible: bool = True
    iconic: bool = False
    cloaked: bool = False
    hung: bool = False
    ex_style: int = 0
    root: int | None = None  # top-level ancestor for child windows
    client: Rect | None = None  # client area (None: like the frame)


class FakeWin32:
    """Stand-in for ``windows._Win32`` with a scriptable foreground lock.

    ``unlock`` decides which strategy lets ``set_foreground`` succeed:
    ``"any"`` (no lock), ``"attach"``, ``"input"``, ``"alt"`` or ``"never"``.
    """

    def __init__(
        self,
        windows_: list[FakeWindow],
        *,
        foreground: int | None = None,
        unlock: str = "any",
        hit: int | None = None,
    ) -> None:
        self.windows = {w.hwnd: w for w in windows_}
        self.zorder = [w.hwnd for w in windows_]
        self.fg = foreground
        self.unlock = unlock
        self.hit = hit
        self.calls: list[tuple[Any, ...]] = []
        self.attached: set[tuple[int, int]] = set()
        self.injected = False
        self.alt = False
        self.idle: int | None = 1500
        self.desktop: str | None = "Default"
        self.cursor: tuple[int, int] = (10, 20)
        self.cursor_offsets: list[tuple[int, int]] = []
        self.cursor_ok = True
        self.image_path: str | None = r"C:\Python\python.exe"
        self.can_create_window = True
        self.own_package: str | None = None
        self.packages: dict[int, str] = {}  # pid -> package family name

    def _win(self, hwnd: int) -> FakeWindow | None:
        return self.windows.get(hwnd)

    # windows
    def foreground(self) -> int | None:
        return self.fg

    def root(self, hwnd: int) -> int:
        w = self._win(hwnd)
        return w.root if w is not None and w.root else hwnd

    def thread_process(self, hwnd: int) -> tuple[int, int]:
        w = self._win(hwnd)
        return (w.tid, w.pid) if w is not None else (0, 0)

    def class_name(self, hwnd: int) -> str:
        w = self._win(hwnd)
        return w.cls if w is not None else ""

    def frame_rect(self, hwnd: int) -> Rect | None:
        w = self._win(hwnd)
        return w.rect if w is not None else None

    def client_rect(self, hwnd: int) -> Rect | None:
        w = self._win(hwnd)
        if w is None:
            return None
        return w.client if w.client is not None else w.rect

    def is_window(self, hwnd: int) -> bool:
        return hwnd in self.windows

    def is_visible(self, hwnd: int) -> bool:
        return self.windows[hwnd].visible

    def is_iconic(self, hwnd: int) -> bool:
        return self.windows[hwnd].iconic

    def is_hung(self, hwnd: int) -> bool:
        return self.windows[hwnd].hung

    def is_cloaked(self, hwnd: int) -> bool:
        return self.windows[hwnd].cloaked

    def ex_style(self, hwnd: int) -> int:
        return self.windows[hwnd].ex_style

    def window_from_point(self, x: int, y: int) -> int | None:
        return self.hit

    def top_window(self) -> int | None:
        return self.zorder[0] if self.zorder else None

    def next_window(self, hwnd: int) -> int | None:
        index = self.zorder.index(hwnd) + 1
        return self.zorder[index] if index < len(self.zorder) else None

    def _allowed(self) -> bool:
        if self.unlock == "any":
            return True
        if self.unlock == "attach":
            fg_tid = self.windows[self.fg].tid if self.fg in self.windows else 0
            return (OUR_THREAD, fg_tid) in self.attached
        if self.unlock == "input":
            return self.injected
        if self.unlock == "alt":
            return self.alt
        return False

    def set_foreground(self, hwnd: int) -> bool:
        self.calls.append(("set_foreground", hwnd))
        if self._allowed():
            self.fg = hwnd
            return True
        return False

    def bring_to_top(self, hwnd: int) -> bool:
        self.calls.append(("bring_to_top", hwnd))
        return True

    def allow_set_foreground_any(self) -> None:
        self.calls.append(("allow_any",))

    def current_thread_id(self) -> int:
        return OUR_THREAD

    def attach_thread_input(self, thread: int, to_thread: int, attach: bool) -> bool:
        self.calls.append(("attach", thread, to_thread, attach))
        if attach:
            self.attached.add((thread, to_thread))
        else:
            self.attached.discard((thread, to_thread))
        return True

    def post_broadcast(self, msg: int, wparam: int, lparam: int) -> bool:
        self.calls.append(("post_broadcast", msg, wparam, lparam))
        return True

    def create_hidden_window(self) -> int | None:
        self.calls.append(("create_window",))
        return 0x5150 if self.can_create_window else None

    def send_message(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        self.calls.append(("send_message", hwnd, msg, wparam, lparam))
        return 0

    def destroy_window(self, hwnd: int) -> bool:
        self.calls.append(("destroy_window", hwnd))
        return True

    # input
    def send_empty_mouse_input(self) -> bool:
        self.calls.append(("empty_input",))
        self.injected = True
        return True

    def tap_alt(self) -> bool:
        self.calls.append(("tap_alt",))
        self.alt = True
        return True

    def nudge_mouse(self) -> bool:
        self.calls.append(("nudge",))
        self.cursor = (self.cursor[0] + 1, self.cursor[1])
        return True

    def idle_ms(self) -> int | None:
        return self.idle

    def set_cursor_pos(self, x: int, y: int) -> bool:
        self.calls.append(("set_cursor", x, y))
        if not self.cursor_ok:
            return False
        dx, dy = self.cursor_offsets.pop(0) if self.cursor_offsets else (0, 0)
        self.cursor = (x + dx, y + dy)
        return True

    def cursor_pos(self) -> tuple[int, int] | None:
        return self.cursor

    # session / process
    def lock_workstation(self) -> bool:
        self.calls.append(("lock",))
        return True

    def set_display_required(self) -> None:
        self.calls.append(("display_required",))

    def input_desktop_name(self) -> str | None:
        return self.desktop

    def enable_per_monitor_dpi(self) -> str:
        self.calls.append(("dpi",))
        return "per-monitor-v2"

    def set_app_user_model_id(self, app_id: str) -> bool:
        self.calls.append(("aumid", app_id))
        return True

    def process_image_path(self) -> str | None:
        return self.image_path

    def current_package_family_name(self) -> str | None:
        return self.own_package

    def process_package_family_name(self, pid: int) -> str | None:
        self.calls.append(("package_of", pid))
        return self.packages.get(pid)

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class ExplodingApi:
    """Every attribute access fails, like a broken or missing Win32 layer."""

    def __getattr__(self, name: str) -> Any:
        raise OSError(f"boom: {name}")


class FakeRegistry:
    """In-memory registry: ``{(hive, path): {value_name: data}}``."""

    def __init__(self, keys: dict[tuple[str, str], dict[str, Any]] | None = None) -> None:
        self.keys = keys or {}

    def subkeys(self, hive: str, path: str) -> list[str]:
        prefix = path + "\\"
        children: list[str] = []
        for key_hive, key_path in self.keys:
            if key_hive == hive and key_path.startswith(prefix):
                child = key_path[len(prefix) :].split("\\", 1)[0]
                if child not in children:
                    children.append(child)
        return children

    def values(self, hive: str, path: str) -> dict[str, Any]:
        return dict(self.keys.get((hive, path), {}))


class RaisingRegistry:
    def subkeys(self, hive: str, path: str) -> list[str]:
        raise OSError("registry unavailable")

    def values(self, hive: str, path: str) -> dict[str, Any]:
        raise OSError("registry unavailable")


def make_platform(api: Any, registry: Any | None = None, *, pid: int = OWN_PID) -> WindowsPlatform:
    plat = WindowsPlatform(api=api, registry=registry or FakeRegistry())
    plat._pid = pid
    return plat


WEBCAM = windows._WEBCAM_KEY
NON_PACKAGED = f"{WEBCAM}\\NonPackaged"
SESSION_START = 133_000_000_000_000_000  # FILETIME of the open camera session
BOOT = SESSION_START - 3600 * windows.FILETIME_TICKS_PER_S  # booted an hour earlier
IN_USE = {"LastUsedTimeStart": SESSION_START, "LastUsedTimeStop": 0}
STOPPED = {"LastUsedTimeStart": SESSION_START, "LastUsedTimeStop": 133_000_000_100_000}
ZOOM = r"C:\Program Files\Zoom\Zoom.exe"
ZOOM_KEY = "C:#Program Files#Zoom#Zoom.exe"
STORE_PYTHON = "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0"


@dataclass(frozen=True)
class Proc:
    """A fake running process: what ``_processes`` lists plus its creation time."""

    pid: int
    name: str
    created: int | None  # FILETIME ticks; None = unreadable


def proc(pid: int, name: str, *, created: int | None = SESSION_START - 600 * 10_000_000) -> Proc:
    """A running process, by default started ten minutes before the camera session."""
    return Proc(pid=pid, name=name, created=created)


def listed(processes: list[Proc] | None) -> list[windows._Process] | None:
    """``processes`` as ``WindowsPlatform._processes`` lists them."""
    if processes is None:
        return None
    return [windows._Process(pid=p.pid, name=p.name) for p in processes]


@pytest.fixture
def clean_qt_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Unset QT_ENABLE_HIGHDPI_SCALING for the test and restore the original afterwards.

    ``setenv`` first so monkeypatch remembers the original state even when the
    variable is created by the code under test.
    """
    monkeypatch.setenv("QT_ENABLE_HIGHDPI_SCALING", "placeholder")
    monkeypatch.delenv("QT_ENABLE_HIGHDPI_SCALING")
    return monkeypatch


# ------------------------------------------------------------ get_platform()
class TestGetPlatform:
    def test_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform_pkg, "_cache", {})
        first = platform_pkg.get_platform()
        assert platform_pkg.get_platform() is first
        assert isinstance(first, PlatformServices)

    @windows_only
    def test_windows_gets_windows_platform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform_pkg, "_cache", {})
        assert isinstance(platform_pkg.get_platform(), WindowsPlatform)

    def test_create_windows_does_not_touch_win32(self) -> None:
        # Construction is lazy, so this works on every OS.
        assert isinstance(platform_pkg._create("win32"), WindowsPlatform)

    def test_unknown_os_falls_back_to_base(self) -> None:
        assert type(platform_pkg._create("plan9")) is PlatformServices

    def test_import_error_falls_back_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(platform_pkg, "_IMPLEMENTATIONS", {"fake": ("does_not_exist", "Nope")})
        with caplog.at_level(logging.WARNING, logger="eye_tracker.platform"):
            result = platform_pkg._create("fakeos")
        assert type(result) is PlatformServices
        assert "unavailable" in caplog.text

    def test_non_platform_class_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Any importable class that is not a PlatformServices must be refused.
        monkeypatch.setattr(platform_pkg, "_IMPLEMENTATIONS", {"fake": ("windows", "_WinRegistry")})
        assert type(platform_pkg._create("fakeos")) is PlatformServices

    def test_reset_for_tests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform_pkg, "_cache", {})
        first = platform_pkg.get_platform()
        platform_pkg._reset_for_tests()
        assert platform_pkg.get_platform() is not first


# ------------------------------------------------------------ pure helpers
class TestHelpers:
    @pytest.mark.parametrize(
        ("now", "last", "expected"),
        [
            (10_000, 9_000, 1_000),
            (0, 0, 0),
            # 64-bit tick past the 32-bit wrap, stamp taken just before it
            (2**32 + 500, 2**32 - 500, 1_000),
            (5 * 2**32 + 42, 40, 2),
        ],
    )
    def test_elapsed_ms_handles_wrap(self, now: int, last: int, expected: int) -> None:
        assert windows._elapsed_ms(now, last) == expected

    def test_norm_path_is_windows_style_everywhere(self) -> None:
        assert windows._norm_path("C:/Program Files/App/APP.EXE") == r"c:\program files\app\app.exe"

    @pytest.mark.parametrize(
        ("handle", "expected"),
        [(123, 123), (0, None), (-5, None), (None, None), ("abc", None), (True, None)],
    )
    def test_hwnd_of(self, handle: Any, expected: int | None) -> None:
        assert windows._hwnd_of(WindowRef(handle=handle)) == expected

    def test_hwnd_of_none_ref(self) -> None:
        assert windows._hwnd_of(None) is None

    def test_camera_user_executable(self) -> None:
        user = windows._CameraUser("C:#Program Files#Zoom#Zoom.exe", packaged=False)
        assert user.executable == r"C:\Program Files\Zoom\Zoom.exe"
        packaged = windows._CameraUser("Microsoft.WindowsCamera_8wekyb3d8bbwe", packaged=True)
        assert packaged.executable == "Microsoft.WindowsCamera_8wekyb3d8bbwe"

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (IN_USE, True),
            (STOPPED, False),
            ({"LastUsedTimeStart": 0, "LastUsedTimeStop": 0}, False),
            ({"LastUsedTimeStart": 5}, False),
            ({}, False),
            ({"LastUsedTimeStart": "5", "LastUsedTimeStop": 0}, False),
        ],
    )
    def test_is_in_use(self, values: dict[str, Any], expected: bool) -> None:
        assert windows._is_in_use(values) is expected
        assert (windows._session_start(values) is not None) is expected

    def test_unix_to_filetime(self) -> None:
        assert windows._unix_to_filetime(0) == 116_444_736_000_000_000
        assert windows._unix_to_filetime(1.5) == 116_444_736_015_000_000

    def test_process_helpers_read_this_process(self) -> None:
        """psutil-backed helpers (read-only, any OS)."""
        processes = make_platform(FakeWin32([]))._processes()
        assert processes
        own = next(p for p in processes if p.pid == os.getpid())
        assert own.name == own.name.lower()
        created = WindowsPlatform._process_created(os.getpid())
        assert created is not None
        boot = WindowsPlatform._boot_filetime()
        assert boot is not None
        assert boot <= created
        exe = WindowsPlatform._process_exe(os.getpid())
        assert exe is not None
        assert Path(exe).is_absolute()
        if sys.platform == "win32":
            assert Path(exe).name.lower() == own.name  # Windows names a process after its image
        else:
            # The kernel's process name: a `#!` script's own name (pytest started
            # through its launcher script), else the executable's.
            assert own.name in {Path(exe).name.lower(), Path(sys.argv[0]).name.lower()}
        assert WindowsPlatform._process_exe(-1) is None
        assert WindowsPlatform._process_created(-1) is None

    def test_process_listing_asks_psutil_for_names_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Creation times of other accounts' processes cost ~6 ms each (GUI thread)."""
        import psutil

        requested: list[list[str]] = []
        real = psutil.process_iter

        def spy(attrs: list[str] | None = None, **kwargs: Any) -> Any:
            requested.append(list(attrs or []))
            return real(attrs, **kwargs)

        monkeypatch.setattr(psutil, "process_iter", spy)
        assert make_platform(FakeWin32([]))._processes()
        assert requested == [["name"]]


# ------------------------------------------------------------- lifecycle
class TestLifecycle:
    def test_capabilities_cover_base_keys(self) -> None:
        caps = make_platform(FakeWin32([])).capabilities()
        assert set(caps) == set(PlatformServices().capabilities())
        for key in ("lock", "display_off", "focus", "camera_in_use", "hotkeys", "input_idle"):
            assert caps[key] is True, key
        assert caps["key_idle"] is False

    @pytest.mark.usefixtures("clean_qt_env")
    def test_prepare_process_sets_env_and_calls_api(self) -> None:
        api = FakeWin32([])
        make_platform(api).prepare_process()
        assert windows.os.environ["QT_ENABLE_HIGHDPI_SCALING"] == "0"
        assert ("dpi",) in api.calls
        assert ("aumid", APP_ID) in api.calls

    def test_prepare_process_keeps_user_override_but_warns(
        self, clean_qt_env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clean_qt_env.setenv("QT_ENABLE_HIGHDPI_SCALING", "1")
        with caplog.at_level(logging.WARNING, logger="eye_tracker.platform.windows"):
            make_platform(FakeWin32([])).prepare_process()
        assert windows.os.environ["QT_ENABLE_HIGHDPI_SCALING"] == "1"
        assert "QT_ENABLE_HIGHDPI_SCALING" in caplog.text

    @pytest.mark.usefixtures("clean_qt_env")
    def test_prepare_process_survives_api_failure(self) -> None:
        make_platform(ExplodingApi()).prepare_process()
        assert windows.os.environ["QT_ENABLE_HIGHDPI_SCALING"] == "0"

    def test_missing_win32_api_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # What happens on macOS/Linux (or a stripped-down Windows): no DLLs load.
        loads: list[str] = []
        monkeypatch.setattr(windows, "_load_dll", loads.append)  # returns None
        with pytest.raises(OSError, match="only available on Windows"):
            windows._Win32()
        loads.clear()
        plat = WindowsPlatform(registry=FakeRegistry())
        assert plat.seconds_since_input() is None
        assert plat.foreground_window() is None
        assert plat.lock_screen() is False
        assert plat.is_session_locked() is None
        assert plat.activate_window(WindowRef(handle=1)) is False
        assert loads == ["user32"]  # the failure is remembered, not retried per call


# ---------------------------------------------------------- session/power
class TestSessionAndInput:
    def test_lock_display_off_and_wake_use_the_binding(self) -> None:
        api = FakeWin32([])
        plat = make_platform(api)
        assert plat.lock_screen() is True
        assert plat.display_off() is True
        api.cursor = (300, 400)
        assert plat.wake_display() is True
        assert api.names()[-3:] == ["display_required", "nudge", "set_cursor"]
        assert api.cursor == (300, 400)  # restored exactly

    def test_display_off_is_sent_to_a_window_of_our_own(self) -> None:
        """Nothing is queued in other apps' windows (a paused app would run it much later)."""
        api = FakeWin32([])
        assert make_platform(api).display_off() is True
        assert api.calls == [
            ("create_window",),
            (
                "send_message",
                0x5150,
                windows.WM_SYSCOMMAND,
                windows.SC_MONITORPOWER,
                windows.MONITOR_POWER_OFF,
            ),
            ("destroy_window", 0x5150),
        ]
        assert "post_broadcast" not in api.names()

    def test_display_off_falls_back_to_broadcast_without_a_window(self) -> None:
        api = FakeWin32([])
        api.can_create_window = False
        assert make_platform(api).display_off() is True
        assert api.calls[-1] == (
            "post_broadcast",
            windows.WM_SYSCOMMAND,
            windows.SC_MONITORPOWER,
            windows.MONITOR_POWER_OFF,
        )
        assert "send_message" not in api.names()

    def test_display_off_window_is_destroyed_when_sending_fails(self) -> None:
        api = FakeWin32([])

        def broken_send(*_args: Any) -> int:
            raise OSError("send failed")

        api.send_message = broken_send  # type: ignore[method-assign]
        assert make_platform(api).display_off() is False
        assert api.calls[-1] == ("destroy_window", 0x5150)

    def test_failures_are_reported_not_raised(self) -> None:
        plat = make_platform(ExplodingApi())
        assert plat.lock_screen() is False
        assert plat.display_off() is False
        assert plat.wake_display() is False
        assert plat.seconds_since_input() is None
        assert plat.is_session_locked() is None
        assert plat.move_cursor(1, 2) is None
        assert plat.foreground_window() is None
        assert plat.window_at(1, 2) is None
        assert plat.activate_window(WindowRef(handle=5)) is False
        assert plat.is_window_valid(WindowRef(handle=5)) is False
        assert plat.window_rect(WindowRef(handle=5)) is None

    def test_seconds_since_input(self) -> None:
        api = FakeWin32([])
        plat = make_platform(api)
        api.idle = 2500
        assert plat.seconds_since_input() == pytest.approx(2.5)
        api.idle = None
        assert plat.seconds_since_input() is None
        assert plat.seconds_since_key_input() is None

    @pytest.mark.parametrize(
        ("desktop", "expected"),
        [("Default", False), ("default", False), ("Winlogon", True), (None, True), ("", None)],
    )
    def test_is_session_locked(self, desktop: str | None, expected: bool | None) -> None:
        api = FakeWin32([])
        api.desktop = desktop
        assert make_platform(api).is_session_locked() is expected

    def test_move_cursor_retries_once_when_first_move_misses(self) -> None:
        api = FakeWin32([])
        api.cursor_offsets = [(7, 0)]
        assert make_platform(api).move_cursor(100, 200) is True
        assert [c for c in api.calls if c[0] == "set_cursor"] == [
            ("set_cursor", 100, 200),
            ("set_cursor", 100, 200),
        ]
        assert api.cursor == (100, 200)

    def test_move_cursor_single_call_when_on_target(self) -> None:
        api = FakeWin32([])
        assert make_platform(api).move_cursor(5, 6) is True
        assert api.names().count("set_cursor") == 1

    def test_move_cursor_failure(self) -> None:
        api = FakeWin32([])
        api.cursor_ok = False
        assert make_platform(api).move_cursor(5, 6) is False


# ------------------------------------------------------------------ windows
class TestWindows:
    def test_foreground_resolves_root_window(self) -> None:
        top = FakeWindow(1, pid=77, rect=Rect(1920, 0, 1920, 1040))
        child = FakeWindow(2, pid=77, cls="Edit", root=1)
        api = FakeWin32([top, child], foreground=2)
        ref = make_platform(api).foreground_window()
        assert ref is not None
        assert ref.handle == 1
        assert ref.pid == 77
        assert ref.rect == Rect(1920, 0, 1920, 1040)

    @pytest.mark.parametrize("cls", ["Progman", "WorkerW", "Shell_TrayWnd", "#32768"])
    def test_foreground_skips_shell_surfaces(self, cls: str) -> None:
        api = FakeWin32([FakeWindow(1, cls=cls)], foreground=1)
        assert make_platform(api).foreground_window() is None

    def test_foreground_skips_own_windows(self) -> None:
        api = FakeWin32([FakeWindow(1, pid=OWN_PID)], foreground=1)
        assert make_platform(api).foreground_window() is None

    def test_no_foreground(self) -> None:
        assert make_platform(FakeWin32([])).foreground_window() is None

    def test_window_at_returns_root(self) -> None:
        top = FakeWindow(1, pid=5)
        child = FakeWindow(2, pid=5, cls="Chrome_RenderWidgetHostHWND", root=1)
        api = FakeWin32([top, child], hit=2)
        ref = make_platform(api).window_at(10, 10)
        assert ref is not None
        assert ref.handle == 1

    def test_window_at_nothing_or_desktop(self) -> None:
        assert make_platform(FakeWin32([])).window_at(1, 1) is None
        api = FakeWin32([FakeWindow(1, cls="Progman")], hit=1)
        assert make_platform(api).window_at(1, 1) is None

    def test_window_at_looks_below_our_own_window(self) -> None:
        stack = [
            FakeWindow(1, pid=OWN_PID, rect=Rect(0, 0, 500, 500)),  # our overlay/dialog
            FakeWindow(2, visible=False),
            FakeWindow(3, ex_style=windows.WS_EX_TRANSPARENT),
            FakeWindow(4, cloaked=True),
            FakeWindow(5, iconic=True),
            FakeWindow(6, rect=Rect(900, 900, 50, 50)),  # does not contain the point
            FakeWindow(7, ex_style=windows.WS_EX_NOACTIVATE),
            FakeWindow(8, pid=9, rect=Rect(0, 0, 400, 400)),  # the user's window
            FakeWindow(9, cls="Progman", rect=Rect(0, 0, 4000, 4000)),
        ]
        api = FakeWin32(stack, hit=1)
        ref = make_platform(api).window_at(100, 100)
        assert ref is not None
        assert ref.handle == 8
        assert ref.pid == 9

    def test_window_at_below_transient_reaches_desktop(self) -> None:
        stack = [
            FakeWindow(1, cls="tooltips_class32", rect=Rect(0, 0, 50, 50)),
            FakeWindow(2, cls="Progman", rect=Rect(0, 0, 4000, 4000)),
            FakeWindow(3, rect=Rect(0, 0, 400, 400)),
        ]
        api = FakeWin32(stack, hit=1)
        assert make_platform(api).window_at(10, 10) is None

    def test_is_window_valid(self) -> None:
        api = FakeWin32(
            [
                FakeWindow(1, pid=3),
                FakeWindow(2, iconic=True),
                FakeWindow(3, visible=False),
                FakeWindow(4, cloaked=True),
            ]
        )
        plat = make_platform(api)
        assert plat.is_window_valid(WindowRef(handle=1, pid=3)) is True
        assert plat.is_window_valid(WindowRef(handle=1)) is True  # pid unknown
        assert plat.is_window_valid(WindowRef(handle=1, pid=99)) is False  # recycled HWND
        assert plat.is_window_valid(WindowRef(handle=2)) is False
        assert plat.is_window_valid(WindowRef(handle=3)) is False
        assert plat.is_window_valid(WindowRef(handle=4)) is False
        assert plat.is_window_valid(WindowRef(handle=5)) is False  # destroyed
        assert plat.is_window_valid(WindowRef(handle=None)) is False

    def test_window_rect(self) -> None:
        api = FakeWin32([FakeWindow(1, rect=Rect(-1920, 0, 1920, 1080))])
        plat = make_platform(api)
        assert plat.window_rect(WindowRef(handle=1)) == Rect(-1920, 0, 1920, 1080)
        assert plat.window_rect(WindowRef(handle=2)) is None

    def test_window_client_rect(self) -> None:
        window = FakeWindow(1, rect=Rect(0, 0, 1200, 800), client=Rect(8, 39, 1184, 753))
        plat = make_platform(FakeWin32([window]))
        assert plat.window_client_rect(WindowRef(handle=1)) == Rect(8, 39, 1184, 753)
        assert plat.window_client_rect(WindowRef(handle=2)) is None
        assert plat.window_client_rect(WindowRef(handle=None)) is None

    def test_window_app_names_process_and_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = FakeWin32(
            [
                FakeWindow(1, pid=100, cls="CASCADIA_HOSTING_WINDOW_CLASS"),
                FakeWindow(2, pid=200, cls="Chrome_WidgetWin_1"),
            ]
        )
        plat = make_platform(api)
        names = {100: "windowsterminal", 200: "code"}
        monkeypatch.setattr(plat, "_app_process_name", names.get)
        assert plat.window_app(WindowRef(handle=1, pid=100)) == AppIdentity(
            "windowsterminal", "CASCADIA_HOSTING_WINDOW_CLASS"
        )
        assert plat.window_app(WindowRef(handle=2)) == AppIdentity("code", "Chrome_WidgetWin_1")
        assert plat.window_app(WindowRef(handle=1, pid=999)) is None  # recycled HWND
        assert plat.window_app(WindowRef(handle=3)) is None  # gone
        names.clear()
        assert plat.window_app(WindowRef(handle=1)) is None  # process not readable

    def test_panes_capability(self) -> None:
        assert make_platform(FakeWin32([])).capabilities()["panes"] is True

    def test_same_window_compares_handles(self) -> None:
        plat = make_platform(FakeWin32([]))
        assert plat.same_window(WindowRef(handle=5), WindowRef(handle=5, pid=1))
        assert not plat.same_window(WindowRef(handle=5), WindowRef(handle=6))


# ------------------------------------------------------------- activation
def _activation_setup(
    unlock: str, *, hung: bool = False, fg_tid: int = 20, target_hung: bool = False
) -> FakeWin32:
    current = FakeWindow(1, pid=50, tid=fg_tid, hung=hung)
    target = FakeWindow(2, pid=60, tid=30, hung=target_hung)
    return FakeWin32([current, target], foreground=1, unlock=unlock)


def _attach_calls(api: FakeWin32) -> list[tuple[Any, ...]]:
    return [c for c in api.calls if c[0] == "attach"]


class TestActivation:
    def test_already_foreground(self) -> None:
        api = FakeWin32([FakeWindow(1)], foreground=1)
        assert make_platform(api).activate_window(WindowRef(handle=1)) is True
        assert "set_foreground" not in api.names()

    def test_minimised_or_invalid_windows_are_not_activated(self) -> None:
        api = FakeWin32([FakeWindow(1, iconic=True), FakeWindow(2)], foreground=2)
        plat = make_platform(api)
        assert plat.activate_window(WindowRef(handle=1)) is False
        assert plat.activate_window(WindowRef(handle=99)) is False
        assert plat.activate_window(WindowRef(handle="nope")) is False
        assert "set_foreground" not in api.names()

    def test_attach_thread_input_first_and_no_synthetic_input(self) -> None:
        api = _activation_setup("attach")
        assert make_platform(api).activate_window(WindowRef(handle=2)) is True
        assert api.fg == 2
        assert _attach_calls(api) == [
            ("attach", OUR_THREAD, 20, True),
            ("attach", OUR_THREAD, 20, False),
        ]
        assert "empty_input" not in api.names()
        assert "tap_alt" not in api.names()
        assert not api.attached

    def test_falls_back_to_empty_input(self) -> None:
        api = _activation_setup("input")
        assert make_platform(api).activate_window(WindowRef(handle=2)) is True
        assert "empty_input" in api.names()
        assert "tap_alt" not in api.names()
        assert not api.attached  # always detached again

    def test_falls_back_to_alt_tap(self) -> None:
        api = _activation_setup("alt")
        assert make_platform(api).activate_window(WindowRef(handle=2)) is True
        names = api.names()
        assert names.index("empty_input") < names.index("tap_alt")

    def test_gives_up_when_locked_out(self) -> None:
        api = _activation_setup("never")
        assert make_platform(api).activate_window(WindowRef(handle=2)) is False
        assert api.fg == 1
        assert not api.attached
        assert "bring_to_top" not in api.names()  # never raised without focus

    def test_raises_window_after_granted_switch(self) -> None:
        api = _activation_setup("any")
        assert make_platform(api).activate_window(WindowRef(handle=2)) is True
        names = api.names()
        assert names.index("set_foreground") < names.index("bring_to_top")

    def test_hung_foreground_is_never_attached(self) -> None:
        api = _activation_setup("input", hung=True)
        assert make_platform(api).activate_window(WindowRef(handle=2)) is True
        assert _attach_calls(api) == []

    def test_own_thread_in_foreground_needs_no_attach(self) -> None:
        api = _activation_setup("any", fg_tid=OUR_THREAD)
        assert make_platform(api).activate_window(WindowRef(handle=2)) is True
        assert _attach_calls(api) == []

    def test_hung_target_is_left_alone(self) -> None:
        """Raising a window whose thread does not pump would block our UI thread."""
        api = _activation_setup("any", target_hung=True)
        assert make_platform(api).activate_window(WindowRef(handle=2)) is False
        assert api.fg == 1
        for name in ("set_foreground", "bring_to_top", "attach", "empty_input", "tap_alt"):
            assert name not in api.names(), name


# ------------------------------------------------------------------ camera
class TestCamera:
    def test_active_camera_users(self) -> None:
        registry = FakeRegistry(
            {
                ("HKCU", WEBCAM): {"Value": "Allow"},
                ("HKCU", f"{WEBCAM}\\Microsoft.WindowsCamera_8wekyb3d8bbwe"): IN_USE,
                ("HKCU", f"{WEBCAM}\\Other.App_123"): STOPPED,
                ("HKCU", NON_PACKAGED): {"Value": "Allow"},
                ("HKCU", f"{NON_PACKAGED}\\C:#Zoom#Zoom.exe"): IN_USE,
                ("HKCU", f"{NON_PACKAGED}\\C:#Old#old.exe"): STOPPED,
                ("HKLM", f"{NON_PACKAGED}\\C:#Svc#svc.exe"): IN_USE,  # HKLM is ignored
            }
        )
        users = windows._active_camera_users(registry)
        assert sorted(users, key=lambda u: u.app) == [
            windows._CameraUser("C:#Zoom#Zoom.exe", packaged=False, start=SESSION_START),
            windows._CameraUser(
                "Microsoft.WindowsCamera_8wekyb3d8bbwe", packaged=True, start=SESSION_START
            ),
        ]

    def _platform(
        self,
        entries: dict[str, dict[str, Any]],
        processes: list[Proc] | None,
        *,
        exes: dict[int, str | None] | None = None,
        packaged: dict[str, dict[str, Any]] | None = None,
        api: FakeWin32 | None = None,
        boot: int | None = BOOT,
    ) -> WindowsPlatform:
        """Consent-store entries, running processes (``exes``: pid -> path) and boot time.

        Creation times are served on demand; ``self.created_reads`` records
        whose were read.
        """
        keys = {("HKCU", f"{NON_PACKAGED}\\{app}"): values for app, values in entries.items()}
        keys.update(
            {("HKCU", f"{WEBCAM}\\{app}"): values for app, values in (packaged or {}).items()}
        )
        plat = make_platform(api or FakeWin32([]), FakeRegistry(keys))
        plat._own_paths = frozenset({windows._norm_path(r"C:\Python\python.exe")})
        plat._processes = lambda: listed(processes)  # type: ignore[method-assign]
        created = {p.pid: p.created for p in processes or []}
        self.created_reads: list[int] = []

        def process_created(pid: int) -> int | None:
            self.created_reads.append(pid)
            return created.get(pid)

        plat._process_created = process_created  # type: ignore[method-assign]
        plat._process_exe = (exes or {}).get  # type: ignore[method-assign]
        plat._boot_filetime = lambda: boot  # type: ignore[method-assign]
        return plat

    def test_nobody_using_the_camera(self) -> None:
        plat = self._platform({}, [proc(1, "zoom.exe")], exes={1: ZOOM})
        assert plat.camera_in_use_by_other_app() is False

    def test_own_process_is_ignored(self) -> None:
        plat = self._platform({"C:#Python#python.exe": IN_USE}, [proc(1, "python.exe")])
        assert plat.camera_in_use_by_other_app() is False

    def test_other_running_app_counts(self) -> None:
        plat = self._platform({ZOOM_KEY: IN_USE}, [proc(1, "zoom.exe")], exes={1: ZOOM})
        assert plat.camera_in_use_by_other_app() is True

    def test_stale_entry_of_exited_app_is_ignored(self) -> None:
        plat = self._platform({ZOOM_KEY: IN_USE}, [proc(1, "explorer.exe")])
        assert plat.camera_in_use_by_other_app() is False

    def test_same_name_at_another_path_does_not_count(self) -> None:
        other = r"C:\Users\me\Downloads\Zoom.exe"
        plat = self._platform({ZOOM_KEY: IN_USE}, [proc(1, "zoom.exe")], exes={1: other})
        assert plat.camera_in_use_by_other_app() is False

    def test_app_started_after_the_session_does_not_count(self) -> None:
        """Chrome crashed mid-call; the Chrome running now never opened the camera."""
        later = proc(1, "zoom.exe", created=SESSION_START + 60 * windows.FILETIME_TICKS_PER_S)
        plat = self._platform({ZOOM_KEY: IN_USE}, [later], exes={1: ZOOM})
        assert plat.camera_in_use_by_other_app() is False

    def test_creation_time_rounding_is_tolerated(self) -> None:
        close = proc(1, "zoom.exe", created=SESSION_START + windows.FILETIME_TICKS_PER_S)
        plat = self._platform({ZOOM_KEY: IN_USE}, [close], exes={1: ZOOM})
        assert plat.camera_in_use_by_other_app() is True

    def test_unreadable_creation_time_or_path_is_trusted(self) -> None:
        plat = self._platform({ZOOM_KEY: IN_USE}, [proc(1, "zoom.exe", created=None)])
        assert plat.camera_in_use_by_other_app() is True  # path unreadable (elevated app)

    def test_session_from_before_the_last_boot_is_stale(self) -> None:
        """Power loss mid-call leaves LastUsedTimeStop at 0 forever."""
        processes_listed: list[int] = []

        def processes() -> list[windows._Process]:
            processes_listed.append(1)
            return [windows._Process(pid=1, name="zoom.exe")]

        next_day = SESSION_START + 24 * 3600 * windows.FILETIME_TICKS_PER_S
        plat = self._platform({ZOOM_KEY: IN_USE}, None, exes={1: ZOOM}, boot=next_day)
        plat._processes = processes  # type: ignore[method-assign]
        assert plat.camera_in_use_by_other_app() is False
        assert processes_listed == []  # no process scan needed

    def test_session_just_before_the_boot_time_goes_to_the_process_check(self) -> None:
        """The boot time comes from the current clock, which NTP may have moved."""
        boot = SESSION_START + 60 * windows.FILETIME_TICKS_PER_S
        plat = self._platform({ZOOM_KEY: IN_USE}, [proc(1, "zoom.exe")], exes={1: ZOOM}, boot=boot)
        assert plat.camera_in_use_by_other_app() is True

    def test_unknown_boot_time_skips_that_filter(self) -> None:
        plat = self._platform({ZOOM_KEY: IN_USE}, [proc(1, "zoom.exe")], exes={1: ZOOM}, boot=None)
        assert plat.camera_in_use_by_other_app() is True

    @pytest.mark.parametrize("processes", [None, []])
    def test_unknown_process_list_trusts_the_registry(self, processes: list[Proc] | None) -> None:
        plat = self._platform({ZOOM_KEY: IN_USE}, processes)
        assert plat.camera_in_use_by_other_app() is True

    def test_creation_time_is_read_only_for_matching_processes(self) -> None:
        """Reading it for every process stalled the GUI thread for about a second."""
        processes = [proc(pid, f"svchost{pid}.exe") for pid in range(10, 60)]
        processes.append(proc(7, "zoom.exe"))
        plat = self._platform({ZOOM_KEY: IN_USE}, processes, exes={7: ZOOM})
        assert plat.camera_in_use_by_other_app() is True
        assert self.created_reads == [7]

    def test_creation_time_is_read_only_for_the_matching_package(self) -> None:
        api = FakeWin32([])
        api.packages = {7: "MSTeams_8wekyb3d8bbwe", 8: "Some.Other_app"}
        plat = self._platform(
            {},
            [proc(5, "explorer.exe"), proc(7, "ms-teams.exe"), proc(8, "other.exe")],
            packaged={"MSTeams_8wekyb3d8bbwe": IN_USE},
            api=api,
        )
        assert plat.camera_in_use_by_other_app() is True
        assert self.created_reads == [7]

    def test_packaged_app_counts_while_its_package_runs(self) -> None:
        api = FakeWin32([])
        api.packages = {7: "Microsoft.WindowsCamera_8wekyb3d8bbwe"}
        plat = self._platform(
            {},
            [proc(5, "explorer.exe"), proc(7, "windowscamera.exe")],
            packaged={"microsoft.windowscamera_8wekyb3d8bbwe": IN_USE},
            api=api,
        )
        assert plat.camera_in_use_by_other_app() is True

    def test_stale_packaged_entry_is_ignored(self) -> None:
        """Teams lost power mid-call: its session stays open in the registry."""
        api = FakeWin32([])
        api.packages = {7: "Some.Other_app"}
        plat = self._platform(
            {},
            [proc(7, "ms-teams.exe")],
            packaged={"MSTeams_8wekyb3d8bbwe": IN_USE},
            api=api,
        )
        assert plat.camera_in_use_by_other_app() is False

    def test_packaged_process_started_after_the_session_does_not_count(self) -> None:
        api = FakeWin32([])
        api.packages = {7: "MSTeams_8wekyb3d8bbwe"}
        late = proc(7, "ms-teams.exe", created=SESSION_START + 60 * windows.FILETIME_TICKS_PER_S)
        plat = self._platform({}, [late], packaged={"MSTeams_8wekyb3d8bbwe": IN_USE}, api=api)
        assert plat.camera_in_use_by_other_app() is False

    def test_own_package_identity_is_ignored(self) -> None:
        """Under the Microsoft Store Python our own session is recorded by package name."""
        api = FakeWin32([])
        api.own_package = STORE_PYTHON
        api.packages = {OWN_PID: STORE_PYTHON}
        plat = self._platform(
            {},
            [proc(OWN_PID, "python.exe")],
            packaged={STORE_PYTHON.lower(): IN_USE},
            api=api,
        )
        assert plat.camera_in_use_by_other_app() is False
        assert plat._own_package() == STORE_PYTHON.lower()

    def test_other_package_is_not_mistaken_for_ours(self) -> None:
        api = FakeWin32([])
        api.own_package = STORE_PYTHON
        api.packages = {9: "Microsoft.WindowsCamera_8wekyb3d8bbwe"}
        plat = self._platform(
            {},
            [proc(9, "windowscamera.exe")],
            packaged={"Microsoft.WindowsCamera_8wekyb3d8bbwe": IN_USE},
            api=api,
        )
        assert plat.camera_in_use_by_other_app() is True

    def test_without_package_identity_packaged_entries_are_others(self) -> None:
        plat = self._platform({}, [proc(9, "x.exe")])
        assert plat._own_package() == ""
        assert not plat._is_own_camera_user(windows._CameraUser("", packaged=True, start=1))

    def test_package_lookup_failure_means_no_package(self) -> None:
        class NoPackageApi(FakeWin32):
            def current_package_family_name(self) -> str | None:
                raise OSError("no appmodel")

        plat = make_platform(NoPackageApi([]))
        assert plat._own_package() == ""

    def test_registry_failure_is_unknown(self) -> None:
        plat = make_platform(FakeWin32([]), RaisingRegistry())
        plat._own_paths = frozenset()
        assert plat.camera_in_use_by_other_app() is None

    def test_own_executables_include_process_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(windows.sys, "executable", r"D:\venv\Scripts\python.exe")
        api = FakeWin32([])
        api.image_path = r"C:\Base\Python312\python.exe"
        own = make_platform(api)._own_executables()
        assert windows._norm_path(r"D:\venv\Scripts\python.exe") in own
        assert windows._norm_path(r"C:\Base\Python312\python.exe") in own


class TestPermissions:
    @pytest.mark.parametrize(
        ("keys", "expected"),
        [
            ({("HKCU", WEBCAM): {"Value": "Allow"}}, True),
            ({("HKCU", WEBCAM): {"Value": "Deny"}}, False),
            (
                {("HKCU", WEBCAM): {"Value": "Allow"}, ("HKCU", NON_PACKAGED): {"Value": "Deny"}},
                False,
            ),
            ({("HKLM", WEBCAM): {"Value": "Deny"}, ("HKCU", WEBCAM): {"Value": "Allow"}}, False),
            ({}, None),
        ],
    )
    def test_camera_permission(
        self, keys: dict[tuple[str, str], dict[str, Any]], expected: bool | None
    ) -> None:
        perms = make_platform(FakeWin32([]), FakeRegistry(keys)).permissions()
        assert perms == {"camera": expected, "accessibility": None}

    def test_camera_permission_unknown_on_registry_error(self) -> None:
        perms = make_platform(FakeWin32([]), RaisingRegistry()).permissions()
        assert perms["camera"] is None

    def test_open_permission_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opened: list[str] = []
        monkeypatch.setattr(windows, "_open_uri", lambda uri: opened.append(uri) or True)
        plat = make_platform(FakeWin32([]))
        assert plat.open_permission_settings("camera") is True
        assert plat.open_permission_settings("accessibility") is False
        assert opened == ["ms-settings:privacy-webcam"]


# ------------------------------------------------- live, read-only (Windows)
@windows_only
class TestLiveReadOnly:
    """Real Win32 calls, restricted to ones that only *read* state."""

    def test_structure_sizes_match_the_sdk(self) -> None:
        pointer_size = ctypes.sizeof(ctypes.c_void_p)
        assert ctypes.sizeof(windows.INPUT) == (40 if pointer_size == 8 else 28)
        assert ctypes.sizeof(windows.LASTINPUTINFO) == 8

    def test_every_binding_has_a_prototype(self) -> None:
        api = windows._Win32()
        bound = {name: fn for name, fn in vars(api).items() if fn is not None}
        assert len(bound) > 30
        for name, fn in bound.items():
            assert fn.argtypes is not None, name
            assert fn.restype is not None, name

    def test_input_idle_and_session(self) -> None:
        plat = WindowsPlatform()
        idle = plat.seconds_since_input()
        assert isinstance(idle, float)
        assert idle >= 0.0
        # False on an interactive desktop; CI services may report True/None.
        assert plat.is_session_locked() in (False, True, None)

    def test_foreground_window_queries(self) -> None:
        plat = WindowsPlatform()
        ref = plat.foreground_window()
        if ref is None:
            pytest.skip("no user window in the foreground (headless session)")
        assert isinstance(ref.handle, int)
        assert isinstance(plat.is_window_valid(ref), bool)
        rect = plat.window_rect(ref)
        assert rect is None or (rect.w > 0 and rect.h > 0)
        client = plat.window_client_rect(ref)
        assert client is None or (client.w > 0 and client.h > 0)
        app = plat.window_app(ref)
        assert app is None or (app.process and app.process == app.process.lower())
        if rect is not None:
            cx, cy = rect.center
            hit = plat.window_at(int(cx), int(cy))
            assert hit is None or isinstance(hit.handle, int)

    def test_camera_and_permissions(self) -> None:
        plat = WindowsPlatform()
        assert plat.camera_in_use_by_other_app() in (True, False, None)
        perms = plat.permissions()
        assert set(perms) == {"camera", "accessibility"}
        assert perms["accessibility"] is None

    def test_registry_reader(self) -> None:
        reader = windows._WinRegistry()
        assert reader.subkeys("HKCU", "Software")
        assert reader.values("HKCU", r"Software\eye-tracker-tests\does-not-exist") == {}
        assert reader.subkeys("HKCU", r"Software\eye-tracker-tests\does-not-exist") == []

    def test_process_image_path(self) -> None:
        path = windows._Win32().process_image_path()
        assert path is not None
        assert Path(path).is_file()

    def test_package_identity_queries(self) -> None:
        api = windows._Win32()
        own = api.current_package_family_name()
        assert own is None or "_" in own  # "<name>_<publisher id>" under the Store Python
        assert api.process_package_family_name(os.getpid()) == own
        assert api.process_package_family_name(0) is None  # the idle process cannot be opened

    def test_hidden_window_is_created_and_destroyed(self) -> None:
        """Only our own invisible window is touched; no message is sent to it."""
        api = windows._Win32()
        hwnd = api.create_hidden_window()
        assert hwnd is not None
        try:
            assert api.is_window(hwnd)
            assert not api.is_visible(hwnd)
            assert api.thread_process(hwnd) == (api.current_thread_id(), os.getpid())
        finally:
            assert api.destroy_window(hwnd)
        assert not api.is_window(hwnd)

    def test_bring_to_top_does_not_wait_for_a_busy_thread(self) -> None:
        """A synchronous SetWindowPos would block until the owner pumps messages.

        The target is a hidden window of our own, owned by a helper thread that
        stops pumping for a while, like an app paused in a debugger.
        """
        api = windows._Win32()
        created = threading.Event()
        release = threading.Event()
        owned: list[int] = []

        def owner() -> None:
            hwnd = api.create_hidden_window()
            if hwnd is not None:
                owned.append(hwnd)
            created.set()
            release.wait(5.0)  # not pumping messages meanwhile
            if hwnd is not None:
                api.destroy_window(hwnd)

        thread = threading.Thread(target=owner, daemon=True)
        thread.start()
        try:
            assert created.wait(5.0)
            assert owned, "could not create the helper window"
            started = time.perf_counter()
            api.bring_to_top(owned[0])
            elapsed = time.perf_counter() - started
        finally:
            release.set()
            thread.join(5.0)
        assert elapsed < 0.5
