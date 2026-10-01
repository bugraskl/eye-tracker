"""Windows integration through the Win32 API (ctypes only, no pywin32).

Design
------
``_Win32`` is a thin binding layer. Every function it uses gets explicit
``argtypes``/``restype`` (so 64-bit handles are never truncated to ``int``) and
each method answers one question - "which window is in front?", "how long has
the user been idle?" - without any policy.

``WindowsPlatform`` holds the policy: which windows count as user windows, how
keyboard focus is acquired despite the foreground lock, what a failure means.
The split keeps the policy unit-testable with a fake binding on any OS.

The module imports on macOS and Linux too (the test suite imports every
module everywhere): no DLL is loaded until a ``WindowsPlatform`` method needs
one, and every public method degrades to ``None``/``False`` instead of raising.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import ntpath
import os
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, ParamSpec, Protocol, TypeVar, cast

from .. import APP_ID
from ..types import AppIdentity, Rect, WindowRef
from .base import PlatformServices

log = logging.getLogger(__name__)

__all__ = ["WindowsPlatform"]

# ---------------------------------------------------------------- constants
GA_ROOT = 2
GW_HWNDNEXT = 2
GWL_EXSTYLE = -20
WS_POPUP = 0x80000000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
HWND_TOP = 0
HWND_BROADCAST = 0xFFFF
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_ASYNCWINDOWPOS = 0x4000
WM_SYSCOMMAND = 0x0112
SC_MONITORPOWER = 0xF170
MONITOR_POWER_OFF = 2
ES_DISPLAY_REQUIRED = 0x00000002
ASFW_ANY = 0xFFFFFFFF
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
KEYEVENTF_KEYUP = 0x0002
VK_MENU = 0x12
#: An unassigned virtual key. Tapping it while Alt is held stops the Alt release
#: from opening the menu bar of the focused window (AutoHotkey's "menu mask key").
VK_MENU_MASK = 0xE8
UOI_NAME = 2
DESKTOP_SWITCHDESKTOP = 0x0100
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
PROCESS_PER_MONITOR_DPI_AWARE = 2
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_SUCCESS = 0
ERROR_ACCESS_DENIED = 5
ERROR_INSUFFICIENT_BUFFER = 122
APPMODEL_ERROR_NO_PACKAGE = 15700
E_ACCESSDENIED = -2147024891  # HRESULT 0x80070005 as a signed 32-bit value
#: Seconds from the FILETIME epoch (1601-01-01) to the Unix epoch (1970-01-01).
FILETIME_UNIX_OFFSET_S = 11_644_473_600
#: FILETIME counts 100 ns intervals.
FILETIME_TICKS_PER_S = 10_000_000

#: Desktop and taskbar: "no window here" rather than a window to focus.
_SHELL_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"})
#: Short-lived shell surfaces (menus, Alt+Tab, Start, tooltips...). Remembering
#: one of these as "the window on that monitor" and re-activating it later would
#: pop up the Start menu or a stale switcher, so they are skipped.
_TRANSIENT_CLASSES = frozenset(
    {
        "#32768",  # popup menus
        "tooltips_class32",
        "SysShadow",
        "ForegroundStaging",
        "MultitaskingViewFrame",
        "XamlExplorerHostIslandWindow",
        "TaskListThumbnailWnd",
        "NotifyIconOverflowWindow",
        "TopLevelWindowForOverflowXamlIsland",
        "Shell_InputSwitchTopLevelWindow",
        "Windows.UI.Core.CoreWindow",  # Start, Search, Action Center as top-level windows
    }
)
#: Upper bound for z-order walks (windows can be created/destroyed mid-walk).
_MAX_Z_WALK = 2048

_WEBCAM_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
)
_NON_PACKAGED = "NonPackaged"
_SETTINGS_URIS = {"camera": "ms-settings:privacy-webcam"}
#: Slack when checking that a process was created before a camera session
#: started. Both are FILETIMEs from the same clock; this only absorbs rounding.
_SESSION_START_SLACK = 2 * FILETIME_TICKS_PER_S
#: Slack when discarding camera sessions opened before the last boot.
_BOOT_TIME_SLACK = 600 * FILETIME_TICKS_PER_S


# --------------------------------------------------------------- structures
class LASTINPUTINFO(ctypes.Structure):
    _fields_ = (("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD))


ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))


# ---------------------------------------------------------------- helpers
P = ParamSpec("P")
R = TypeVar("R")


def _best_effort(default: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Turn any exception into ``default`` (logged at DEBUG), per the platform contract."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return fn(*args, **kwargs)
            except Exception:
                log.debug("%s failed", fn.__qualname__, exc_info=True)
                return cast(R, default)

        return wrapper

    return decorate


def _elapsed_ms(now_ms: int, last_ms: int) -> int:
    """Milliseconds from a 32-bit tick stamp to a 64-bit tick count.

    ``GetLastInputInfo`` reports a 32-bit tick that wraps every 49.7 days;
    subtracting modulo 2**32 keeps the result right across the wrap.
    """
    return ((now_ms & 0xFFFFFFFF) - (last_ms & 0xFFFFFFFF)) & 0xFFFFFFFF


def _norm_path(path: str) -> str:
    """Windows-style case/separator normalisation (works on any host OS)."""
    return ntpath.normcase(ntpath.normpath(path))


def _real_path(path: str) -> str | None:
    """``path`` with links and packaged-app file redirection resolved; ``None`` on error."""
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        return None
    return real.removeprefix("\\\\?\\")


def _unix_to_filetime(seconds: float) -> int:
    """Unix time (seconds, as psutil reports it) as a ``FILETIME`` tick count."""
    return int((seconds + FILETIME_UNIX_OFFSET_S) * FILETIME_TICKS_PER_S)


def _hwnd_of(ref: WindowRef | None) -> int | None:
    if ref is None:
        return None
    handle = ref.handle
    if isinstance(handle, int) and not isinstance(handle, bool) and handle > 0:
        return handle
    return None


def _open_uri(uri: str) -> bool:
    """Open a URI with the shell (``ms-settings:`` pages)."""
    if sys.platform != "win32":
        return False
    try:
        os.startfile(uri)
    except OSError as exc:
        log.warning("Could not open %s: %s", uri, exc)
        return False
    return True


# ------------------------------------------------------------ Win32 binding
def _load_dll(name: str) -> Any:
    if sys.platform != "win32":
        return None
    try:
        return ctypes.WinDLL(name, use_last_error=True)
    except OSError:
        log.debug("%s.dll not available", name)
        return None


def _last_error() -> int:
    """``GetLastError`` of the last call made through a ``use_last_error`` DLL."""
    if sys.platform == "win32":
        return ctypes.get_last_error()
    return 0


def _bind(dll: Any, name: str, restype: Any, *argtypes: Any, optional: bool = False) -> Any:
    """Look up ``name`` in ``dll`` and declare its prototype."""
    if dll is None:
        if optional:
            return None
        raise OSError(f"cannot bind {name}: library not loaded")
    try:
        fn = getattr(dll, name)
    except AttributeError:
        if optional:
            log.debug("%s is not available on this Windows version", name)
            return None
        raise
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


class _Win32:
    """Typed ctypes bindings. Created on first use; raises ``OSError`` off Windows."""

    def __init__(self) -> None:
        # Private WinDLL instances: prototypes live on the instance, so the
        # argtypes declared here cannot clash with other code using ctypes.windll.
        user32 = _load_dll("user32")
        if user32 is None:
            # Checked on the loaded DLL rather than sys.platform so type checkers
            # running for another OS still see (and check) the code below.
            raise OSError("the Win32 API is only available on Windows")
        kernel32 = _load_dll("kernel32")
        shell32 = _load_dll("shell32")
        dwmapi = _load_dll("dwmapi")
        shcore = _load_dll("shcore")

        w = wintypes
        hwnd, hdesk, hmodule = w.HWND, w.HANDLE, w.HMODULE
        hresult = ctypes.c_long  # not ctypes.HRESULT: that raises instead of returning

        # windows
        self._GetForegroundWindow = _bind(user32, "GetForegroundWindow", hwnd)
        self._GetAncestor = _bind(user32, "GetAncestor", hwnd, hwnd, w.UINT)
        self._GetWindowThreadProcessId = _bind(
            user32, "GetWindowThreadProcessId", w.DWORD, hwnd, ctypes.POINTER(w.DWORD)
        )
        self._GetClassNameW = _bind(
            user32, "GetClassNameW", ctypes.c_int, hwnd, w.LPWSTR, ctypes.c_int
        )
        self._GetWindowRect = _bind(user32, "GetWindowRect", w.BOOL, hwnd, ctypes.POINTER(w.RECT))
        self._GetClientRect = _bind(user32, "GetClientRect", w.BOOL, hwnd, ctypes.POINTER(w.RECT))
        self._ClientToScreen = _bind(
            user32, "ClientToScreen", w.BOOL, hwnd, ctypes.POINTER(w.POINT)
        )
        self._WindowFromPoint = _bind(user32, "WindowFromPoint", hwnd, w.POINT)
        self._GetTopWindow = _bind(user32, "GetTopWindow", hwnd, hwnd)
        self._GetWindow = _bind(user32, "GetWindow", hwnd, hwnd, w.UINT)
        self._GetWindowLongW = _bind(user32, "GetWindowLongW", w.LONG, hwnd, ctypes.c_int)
        self._IsWindow = _bind(user32, "IsWindow", w.BOOL, hwnd)
        self._IsWindowVisible = _bind(user32, "IsWindowVisible", w.BOOL, hwnd)
        self._IsIconic = _bind(user32, "IsIconic", w.BOOL, hwnd)
        self._IsHungAppWindow = _bind(user32, "IsHungAppWindow", w.BOOL, hwnd, optional=True)
        self._SetForegroundWindow = _bind(user32, "SetForegroundWindow", w.BOOL, hwnd)
        self._SetWindowPos = _bind(
            user32,
            "SetWindowPos",
            w.BOOL,
            hwnd,
            hwnd,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            w.UINT,
        )
        self._AllowSetForegroundWindow = _bind(user32, "AllowSetForegroundWindow", w.BOOL, w.DWORD)
        self._AttachThreadInput = _bind(
            user32, "AttachThreadInput", w.BOOL, w.DWORD, w.DWORD, w.BOOL
        )
        self._PostMessageW = _bind(user32, "PostMessageW", w.BOOL, hwnd, w.UINT, w.WPARAM, w.LPARAM)
        lresult = w.LPARAM  # LRESULT: pointer-sized signed integer
        self._SendMessageW = _bind(
            user32, "SendMessageW", lresult, hwnd, w.UINT, w.WPARAM, w.LPARAM
        )
        self._CreateWindowExW = _bind(
            user32,
            "CreateWindowExW",
            hwnd,
            w.DWORD,
            w.LPCWSTR,
            w.LPCWSTR,
            w.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            hwnd,
            w.HMENU,
            w.HINSTANCE,
            w.LPVOID,
        )
        self._DestroyWindow = _bind(user32, "DestroyWindow", w.BOOL, hwnd)
        # input
        self._SendInput = _bind(
            user32, "SendInput", w.UINT, w.UINT, ctypes.POINTER(INPUT), ctypes.c_int
        )
        self._GetLastInputInfo = _bind(
            user32, "GetLastInputInfo", w.BOOL, ctypes.POINTER(LASTINPUTINFO)
        )
        self._SetCursorPos = _bind(user32, "SetCursorPos", w.BOOL, ctypes.c_int, ctypes.c_int)
        self._GetCursorPos = _bind(user32, "GetCursorPos", w.BOOL, ctypes.POINTER(w.POINT))
        # session
        self._LockWorkStation = _bind(user32, "LockWorkStation", w.BOOL)
        self._OpenInputDesktop = _bind(user32, "OpenInputDesktop", hdesk, w.DWORD, w.BOOL, w.DWORD)
        self._CloseDesktop = _bind(user32, "CloseDesktop", w.BOOL, hdesk)
        self._GetUserObjectInformationW = _bind(
            user32,
            "GetUserObjectInformationW",
            w.BOOL,
            w.HANDLE,
            ctypes.c_int,
            w.LPVOID,
            w.DWORD,
            ctypes.POINTER(w.DWORD),
        )
        # process
        self._SetProcessDpiAwarenessContext = _bind(
            user32, "SetProcessDpiAwarenessContext", w.BOOL, ctypes.c_void_p, optional=True
        )
        self._SetProcessDPIAware = _bind(user32, "SetProcessDPIAware", w.BOOL, optional=True)
        self._SetProcessDpiAwareness = _bind(
            shcore, "SetProcessDpiAwareness", hresult, ctypes.c_int, optional=True
        )
        self._SetCurrentProcessExplicitAppUserModelID = _bind(
            shell32, "SetCurrentProcessExplicitAppUserModelID", hresult, w.LPCWSTR, optional=True
        )
        self._GetTickCount64 = _bind(kernel32, "GetTickCount64", ctypes.c_uint64)
        self._GetCurrentThreadId = _bind(kernel32, "GetCurrentThreadId", w.DWORD)
        self._SetThreadExecutionState = _bind(kernel32, "SetThreadExecutionState", w.DWORD, w.DWORD)
        self._GetModuleFileNameW = _bind(
            kernel32, "GetModuleFileNameW", w.DWORD, hmodule, w.LPWSTR, w.DWORD
        )
        self._GetModuleHandleW = _bind(kernel32, "GetModuleHandleW", hmodule, w.LPCWSTR)
        self._OpenProcess = _bind(kernel32, "OpenProcess", w.HANDLE, w.DWORD, w.BOOL, w.DWORD)
        self._CloseHandle = _bind(kernel32, "CloseHandle", w.BOOL, w.HANDLE)
        # Package identity (MSIX / Store apps): Windows 8 and later.
        self._GetCurrentPackageFamilyName = _bind(
            kernel32,
            "GetCurrentPackageFamilyName",
            w.LONG,
            ctypes.POINTER(ctypes.c_uint32),
            w.LPWSTR,
            optional=True,
        )
        self._GetPackageFamilyName = _bind(
            kernel32,
            "GetPackageFamilyName",
            w.LONG,
            w.HANDLE,
            ctypes.POINTER(ctypes.c_uint32),
            w.LPWSTR,
            optional=True,
        )
        self._DwmGetWindowAttribute = _bind(
            dwmapi,
            "DwmGetWindowAttribute",
            hresult,
            hwnd,
            w.DWORD,
            w.LPVOID,
            w.DWORD,
            optional=True,
        )

    # ------------------------------------------------------------- windows
    def foreground(self) -> int | None:
        return self._GetForegroundWindow() or None

    def root(self, hwnd: int) -> int:
        return self._GetAncestor(hwnd, GA_ROOT) or hwnd

    def thread_process(self, hwnd: int) -> tuple[int, int]:
        """``(thread id, process id)`` owning the window (``(0, 0)`` if it is gone)."""
        pid = wintypes.DWORD(0)
        tid = self._GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(tid), int(pid.value)

    def class_name(self, hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(257)
        return buf.value if self._GetClassNameW(hwnd, buf, len(buf)) > 0 else ""

    def frame_rect(self, hwnd: int) -> Rect | None:
        """Visible frame in physical pixels.

        DWM's extended frame bounds exclude the invisible resize borders that
        ``GetWindowRect`` includes on Windows 10/11 (which make maximised
        windows appear to spill onto the neighbouring monitor).
        """
        rect = wintypes.RECT()
        ok = False
        if self._DwmGetWindowAttribute is not None:
            hr = self._DwmGetWindowAttribute(
                hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
            )
            ok = hr == 0
        if not ok:
            ok = bool(self._GetWindowRect(hwnd, ctypes.byref(rect)))
        if not ok:
            return None
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return None
        return Rect(int(rect.left), int(rect.top), int(width), int(height))

    def client_rect(self, hwnd: int) -> Rect | None:
        """Client area in physical pixels (the process is per-monitor DPI aware)."""
        rect = wintypes.RECT()
        if not self._GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        origin = wintypes.POINT(0, 0)
        if not self._ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return None
        return Rect(int(origin.x), int(origin.y), int(width), int(height))

    def is_window(self, hwnd: int) -> bool:
        return bool(self._IsWindow(hwnd))

    def is_visible(self, hwnd: int) -> bool:
        return bool(self._IsWindowVisible(hwnd))

    def is_iconic(self, hwnd: int) -> bool:
        return bool(self._IsIconic(hwnd))

    def is_hung(self, hwnd: int) -> bool:
        return bool(self._IsHungAppWindow(hwnd)) if self._IsHungAppWindow is not None else False

    def is_cloaked(self, hwnd: int) -> bool:
        """Cloaked windows are "visible" but not shown (other virtual desktop, suspended UWP)."""
        if self._DwmGetWindowAttribute is None:
            return False
        value = wintypes.DWORD(0)
        hr = self._DwmGetWindowAttribute(
            hwnd, DWMWA_CLOAKED, ctypes.byref(value), ctypes.sizeof(value)
        )
        return hr == 0 and value.value != 0

    def ex_style(self, hwnd: int) -> int:
        return int(self._GetWindowLongW(hwnd, GWL_EXSTYLE)) & 0xFFFFFFFF

    def window_from_point(self, x: int, y: int) -> int | None:
        return self._WindowFromPoint(wintypes.POINT(int(x), int(y))) or None

    def top_window(self) -> int | None:
        return self._GetTopWindow(None) or None

    def next_window(self, hwnd: int) -> int | None:
        return self._GetWindow(hwnd, GW_HWNDNEXT) or None

    def set_foreground(self, hwnd: int) -> bool:
        return bool(self._SetForegroundWindow(hwnd))

    def bring_to_top(self, hwnd: int) -> bool:
        """Raise ``hwnd`` to the top of the z-order without waiting for its thread.

        ``BringWindowToTop`` (a synchronous ``SetWindowPos``) blocks the caller
        until the window's thread processes the request, which is forever for a
        busy or debugger-paused app. ``SWP_ASYNCWINDOWPOS`` posts it instead.
        """
        flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS
        return bool(self._SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, flags))

    def allow_set_foreground_any(self) -> None:
        self._AllowSetForegroundWindow(ASFW_ANY)

    def current_thread_id(self) -> int:
        return int(self._GetCurrentThreadId())

    def attach_thread_input(self, thread: int, to_thread: int, attach: bool) -> bool:
        return bool(self._AttachThreadInput(thread, to_thread, attach))

    def post_broadcast(self, msg: int, wparam: int, lparam: int) -> bool:
        return bool(self._PostMessageW(HWND_BROADCAST, msg, wparam, lparam))

    def create_hidden_window(self) -> int | None:
        """A hidden, never-activated top-level window owned by the calling thread.

        Uses the predefined ``STATIC`` class, whose window procedure hands system
        commands to ``DefWindowProc``, so no class has to be registered.
        """
        hwnd = self._CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            "STATIC",
            None,
            WS_POPUP,
            0,
            0,
            0,
            0,
            None,
            None,
            self._GetModuleHandleW(None),
            None,
        )
        return int(hwnd) if hwnd else None

    def send_message(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        """``SendMessageW``; only used on windows of the calling thread (a direct call)."""
        return int(self._SendMessageW(hwnd, msg, wparam, lparam))

    def destroy_window(self, hwnd: int) -> bool:
        return bool(self._DestroyWindow(hwnd))

    # --------------------------------------------------------------- input
    def _send(self, inputs: list[INPUT]) -> bool:
        array = (INPUT * len(inputs))(*inputs)
        sent = self._SendInput(len(inputs), array, ctypes.sizeof(INPUT))
        return int(sent) == len(inputs)

    def send_empty_mouse_input(self) -> bool:
        """Inject a no-op mouse event.

        Windows lets the process that injected the most recent input take the
        foreground (the same trick PowerToys uses), and an empty event has no
        visible effect.
        """
        return self._send([INPUT(type=INPUT_MOUSE)])

    def tap_alt(self) -> bool:
        """Press and release Alt with the menu-mask key in between.

        An Alt press lifts the foreground lock; the masked release keeps the
        focused window from entering menu mode.
        """
        events = []
        for vk, flags in (
            (VK_MENU, 0),
            (VK_MENU_MASK, 0),
            (VK_MENU_MASK, KEYEVENTF_KEYUP),
            (VK_MENU, KEYEVENTF_KEYUP),
        ):
            event = INPUT(type=INPUT_KEYBOARD)
            event.u.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
            events.append(event)
        return self._send(events)

    def nudge_mouse(self) -> bool:
        """Relative 1 px move there and back: real input that wakes sleeping displays."""
        events = []
        for dx in (1, -1):
            event = INPUT(type=INPUT_MOUSE)
            event.u.mi = MOUSEINPUT(
                dx=dx, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0
            )
            events.append(event)
        return self._send(events)

    def idle_ms(self) -> int | None:
        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not self._GetLastInputInfo(ctypes.byref(info)):
            return None
        return _elapsed_ms(int(self._GetTickCount64()), int(info.dwTime))

    def set_cursor_pos(self, x: int, y: int) -> bool:
        return bool(self._SetCursorPos(int(x), int(y)))

    def cursor_pos(self) -> tuple[int, int] | None:
        point = wintypes.POINT()
        if not self._GetCursorPos(ctypes.byref(point)):
            return None
        return int(point.x), int(point.y)

    # ------------------------------------------------------------- session
    def lock_workstation(self) -> bool:
        return bool(self._LockWorkStation())

    def set_display_required(self) -> None:
        # Without ES_CONTINUOUS this only resets the display idle timer once.
        self._SetThreadExecutionState(ES_DISPLAY_REQUIRED)

    def input_desktop_name(self) -> str | None:
        """Name of the desktop receiving input.

        ``None`` when it cannot be opened (the secure Winlogon desktop is not
        accessible to user processes), ``""`` when its name cannot be read.
        """
        desktop = self._OpenInputDesktop(0, False, DESKTOP_SWITCHDESKTOP)
        if not desktop:
            return None
        try:
            buf = ctypes.create_unicode_buffer(256)
            needed = wintypes.DWORD(0)
            ok = self._GetUserObjectInformationW(
                desktop, UOI_NAME, ctypes.byref(buf), ctypes.sizeof(buf), ctypes.byref(needed)
            )
            return buf.value if ok else ""
        finally:
            self._CloseDesktop(desktop)

    # ------------------------------------------------------------- process
    def enable_per_monitor_dpi(self) -> str:
        """Make the process Per-Monitor-V2 DPI aware; returns what was achieved."""
        if self._SetProcessDpiAwarenessContext is not None:
            ctx = ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            if self._SetProcessDpiAwarenessContext(ctx):
                return "per-monitor-v2"
            if _last_error() == ERROR_ACCESS_DENIED:
                return "already set"
        if self._SetProcessDpiAwareness is not None:
            hr = int(self._SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE))
            if hr == 0:
                return "per-monitor"
            if hr == E_ACCESSDENIED:
                return "already set"
        if self._SetProcessDPIAware is not None and self._SetProcessDPIAware():
            return "system"
        return "unaware"

    def set_app_user_model_id(self, app_id: str) -> bool:
        if self._SetCurrentProcessExplicitAppUserModelID is None:
            return False
        return int(self._SetCurrentProcessExplicitAppUserModelID(app_id)) >= 0

    def process_image_path(self) -> str | None:
        """Full path of this process's executable (the real interpreter behind a venv shim)."""
        buf = ctypes.create_unicode_buffer(32768)
        n = self._GetModuleFileNameW(None, buf, len(buf))
        return buf.value if 0 < n < len(buf) else None

    def current_package_family_name(self) -> str | None:
        """Package family name of this process, ``None`` without package identity.

        A process installed as an MSIX package (the Microsoft Store Python, for
        instance) is recorded under this name in the camera consent store.
        """
        if self._GetCurrentPackageFamilyName is None:
            return None
        return self._read_package_name(
            lambda length, buf: int(self._GetCurrentPackageFamilyName(length, buf))
        )

    def process_package_family_name(self, pid: int) -> str | None:
        """Package family name of process ``pid``; ``None`` when it has none or is inaccessible."""
        if self._GetPackageFamilyName is None:
            return None
        process = self._OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not process:
            return None
        try:
            return self._read_package_name(
                lambda length, buf: int(self._GetPackageFamilyName(process, length, buf))
            )
        finally:
            self._CloseHandle(process)

    @staticmethod
    def _read_package_name(call: Callable[[Any, Any], int]) -> str | None:
        """Run a ``Get*PackageFamilyName``-style call: ask for the size, then the name."""
        length = ctypes.c_uint32(0)
        status = call(ctypes.byref(length), None)
        # APPMODEL_ERROR_NO_PACKAGE: no package identity. Anything but "buffer
        # too small" (with the size filled in) means there is no name to read.
        if status != ERROR_INSUFFICIENT_BUFFER or length.value == 0:
            return None
        buf = ctypes.create_unicode_buffer(length.value)
        if call(ctypes.byref(length), buf) != ERROR_SUCCESS:
            return None
        return buf.value or None


# ------------------------------------------------------------------ registry
class _Registry(Protocol):
    """Read-only registry access (a fake replaces it in tests)."""

    def subkeys(self, hive: str, path: str) -> list[str]: ...

    def values(self, hive: str, path: str) -> dict[str, Any]: ...


class _WinRegistry:
    """``winreg``-backed reader. Missing keys read as empty."""

    _LIMIT = 4096

    def subkeys(self, hive: str, path: str) -> list[str]:
        if sys.platform != "win32":
            return []
        import winreg

        try:
            key = winreg.OpenKey(self._hive(hive), path, 0, winreg.KEY_READ)
        except OSError:
            return []
        names: list[str] = []
        with key:
            for index in range(self._LIMIT):
                try:
                    names.append(winreg.EnumKey(key, index))
                except OSError:
                    break
        return names

    def values(self, hive: str, path: str) -> dict[str, Any]:
        if sys.platform != "win32":
            return {}
        import winreg

        try:
            key = winreg.OpenKey(self._hive(hive), path, 0, winreg.KEY_READ)
        except OSError:
            return {}
        out: dict[str, Any] = {}
        with key:
            for index in range(self._LIMIT):
                try:
                    name, data, _kind = winreg.EnumValue(key, index)
                except OSError:
                    break
                out[name] = data
        return out

    @staticmethod
    def _hive(hive: str) -> Any:
        if sys.platform != "win32":
            raise OSError("no registry")
        import winreg

        return {"HKCU": winreg.HKEY_CURRENT_USER, "HKLM": winreg.HKEY_LOCAL_MACHINE}[hive]


@dataclass(frozen=True, slots=True)
class _CameraUser:
    """An app the camera consent store reports as currently streaming."""

    app: str  # package family name, or '#'-separated executable path for desktop apps
    packaged: bool
    #: ``LastUsedTimeStart`` of the open session (``FILETIME``, UTC); 0 if unknown.
    start: int = 0

    @property
    def executable(self) -> str:
        return self.app if self.packaged else self.app.replace("#", "\\")


@dataclass(frozen=True, slots=True)
class _Process:
    """A running process, as listed for the camera-session liveness check.

    Only what every process yields cheaply. The creation time is read later,
    for the few processes that match a session (see
    ``WindowsPlatform._process_created``): for processes of other accounts
    psutil can only get it from a full system snapshot, several milliseconds
    each, which for a whole listing stalled the GUI thread for about a second.
    """

    pid: int
    name: str  # lower-case executable name


def _session_start(values: dict[str, Any]) -> int | None:
    """``LastUsedTimeStart`` of a session that is still open, else ``None``."""
    start = values.get("LastUsedTimeStart")
    stop = values.get("LastUsedTimeStop")
    if isinstance(start, int) and start > 0 and isinstance(stop, int) and stop == 0:
        return start
    return None


def _is_in_use(values: dict[str, Any]) -> bool:
    return _session_start(values) is not None


def _active_camera_users(registry: _Registry) -> list[_CameraUser]:
    """Apps that started using the webcam and have not stopped yet.

    Windows records camera sessions per app under the ``CapabilityAccessManager``
    consent store (the same data behind the camera privacy indicator):
    ``LastUsedTimeStop`` is 0 while a session is open.
    """
    users: list[_CameraUser] = []
    for name in registry.subkeys("HKCU", _WEBCAM_KEY):
        path = f"{_WEBCAM_KEY}\\{name}"
        if name.lower() == _NON_PACKAGED.lower():
            for app in registry.subkeys("HKCU", path):
                start = _session_start(registry.values("HKCU", f"{path}\\{app}"))
                if start is not None:
                    users.append(_CameraUser(app, packaged=False, start=start))
        else:
            start = _session_start(registry.values("HKCU", path))
            if start is not None:
                users.append(_CameraUser(name, packaged=True, start=start))
    return users


def _consent_value(values: dict[str, Any]) -> str | None:
    value = values.get("Value")
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


# ------------------------------------------------------------------ platform
_KIND_OWN = "own"
_KIND_SHELL = "shell"
_KIND_TRANSIENT = "transient"
_KIND_NORMAL = "normal"


class WindowsPlatform(PlatformServices):
    """Windows 10/11 implementation of :class:`PlatformServices`.

    ``api`` and ``registry`` exist for tests; production code uses the defaults.
    """

    name = "windows"

    def __init__(self, api: _Win32 | None = None, registry: _Registry | None = None) -> None:
        self._api_obj = api
        self._api_error: str | None = None
        self._api_lock = threading.Lock()
        self._registry: _Registry = registry if registry is not None else _WinRegistry()
        self._pid = os.getpid()
        self._own_paths: frozenset[str] | None = None
        self._own_package_name: str | None = None

    @property
    def _api(self) -> _Win32:
        api = self._api_obj
        if api is None:
            with self._api_lock:
                if self._api_obj is None:
                    if self._api_error is not None:
                        raise OSError(self._api_error)
                    try:
                        self._api_obj = _Win32()
                    except OSError as exc:
                        # Remember the failure: callers poll several times a second.
                        self._api_error = str(exc)
                        log.warning("Windows integration unavailable: %s", exc)
                        raise
                api = self._api_obj
        return api

    # ---------------------------------------------------------------- lifecycle
    def prepare_process(self) -> None:
        """Per-Monitor-V2 DPI awareness + unscaled Qt, so Qt coordinates are physical pixels."""
        current = os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
        if current != "0":
            log.warning(
                "QT_ENABLE_HIGHDPI_SCALING=%s is set in the environment; cursor and window "
                "positions may be off on scaled displays",
                current,
            )
        try:
            api = self._api
            mode = api.enable_per_monitor_dpi()
            log.debug("DPI awareness: %s", mode)
            # Groups taskbar entries and notifications under the app, not python.exe.
            if not api.set_app_user_model_id(APP_ID):
                log.debug("Could not set the AppUserModelID")
        except Exception:
            log.debug("prepare_process failed", exc_info=True)

    def capabilities(self) -> dict[str, bool]:
        caps = super().capabilities()
        caps.update(
            lock=True,
            display_off=True,
            wake_display=True,
            input_idle=True,
            key_idle=False,
            session_locked=True,
            focus=True,
            cursor=True,
            camera_in_use=True,
            hotkeys=True,
            panes=True,
        )
        return caps

    # ------------------------------------------------------------ session/power
    @_best_effort(False)
    def lock_screen(self) -> bool:
        if self._api.lock_workstation():
            return True
        log.warning("LockWorkStation failed (locking may be disabled by policy)")
        return False

    @_best_effort(False)
    def display_off(self) -> bool:
        """Power the monitors down through ``DefWindowProc`` of a window of our own.

        Any top-level window's ``DefWindowProc`` handles ``SC_MONITORPOWER``, so
        a throw-away hidden window on the calling thread is enough, and sending
        to it is a plain function call. Broadcasting instead would queue the
        request in every window of the system: a suspended or debugger-paused
        app would run it whenever it next pumps messages, possibly long after
        the user came back, blacking the screens out again.
        """
        api = self._api
        hwnd = api.create_hidden_window()
        if hwnd is None:
            log.debug("Could not create a window for SC_MONITORPOWER; broadcasting instead")
            return api.post_broadcast(WM_SYSCOMMAND, SC_MONITORPOWER, MONITOR_POWER_OFF)
        try:
            api.send_message(hwnd, WM_SYSCOMMAND, SC_MONITORPOWER, MONITOR_POWER_OFF)
        finally:
            api.destroy_window(hwnd)
        return True

    @_best_effort(False)
    def wake_display(self) -> bool:
        api = self._api
        api.set_display_required()
        before = api.cursor_pos()
        ok = api.nudge_mouse()
        if before is not None:
            # Pointer acceleration can turn +1/-1 into an uneven pair; put it back exactly.
            api.set_cursor_pos(*before)
        return ok

    @_best_effort(None)
    def is_session_locked(self) -> bool | None:
        name = self._api.input_desktop_name()
        if name is None:
            return True  # the input desktop is the secure (Winlogon) desktop
        if not name:
            return None
        return name.lower() != "default"

    # -------------------------------------------------------------------- input
    @_best_effort(None)
    def seconds_since_input(self) -> float | None:
        ms = self._api.idle_ms()
        return None if ms is None else ms / 1000.0

    @_best_effort(None)
    def move_cursor(self, x: int, y: int) -> bool | None:
        api = self._api
        # A second SetCursorPos corrects the rare first move that lands off
        # target when crossing between monitors with different scale factors.
        for _attempt in range(2):
            if not api.set_cursor_pos(x, y):
                return False
            if api.cursor_pos() == (int(x), int(y)):
                return True
        return True

    # ------------------------------------------------------------------ windows
    def _classify(self, hwnd: int) -> tuple[str, int]:
        _tid, pid = self._api.thread_process(hwnd)
        if pid == self._pid:
            return _KIND_OWN, pid
        cls = self._api.class_name(hwnd)
        if cls in _SHELL_CLASSES:
            return _KIND_SHELL, pid
        if cls in _TRANSIENT_CLASSES:
            return _KIND_TRANSIENT, pid
        return _KIND_NORMAL, pid

    def _ref(self, hwnd: int, pid: int) -> WindowRef:
        return WindowRef(handle=int(hwnd), pid=pid or None, rect=self._api.frame_rect(hwnd))

    @_best_effort(None)
    def foreground_window(self) -> WindowRef | None:
        hwnd = self._api.foreground()
        if not hwnd:
            return None
        root = self._api.root(hwnd)
        kind, pid = self._classify(root)
        return self._ref(root, pid) if kind == _KIND_NORMAL else None

    @_best_effort(None)
    def window_at(self, x: int, y: int) -> WindowRef | None:
        api = self._api
        hit = api.window_from_point(x, y)
        if not hit:
            return None
        root = api.root(hit)
        kind, pid = self._classify(root)
        if kind == _KIND_NORMAL:
            return self._ref(root, pid)
        if kind == _KIND_SHELL:
            return None  # desktop or taskbar: no window there
        # One of our own windows or a transient shell surface is on top; the
        # user's window is the next suitable one below it.
        return self._window_below(root, x, y)

    def _window_below(self, start: int, x: int, y: int) -> WindowRef | None:
        api = self._api
        hwnd = api.next_window(start)
        for _step in range(_MAX_Z_WALK):
            if not hwnd:
                return None
            if self._covers_point(hwnd, x, y):
                kind, pid = self._classify(hwnd)
                if kind == _KIND_NORMAL:
                    return self._ref(hwnd, pid)
                if kind == _KIND_SHELL:
                    return None
            hwnd = api.next_window(hwnd)
        return None

    def _covers_point(self, hwnd: int, x: int, y: int) -> bool:
        api = self._api
        if not api.is_visible(hwnd) or api.is_iconic(hwnd) or api.is_cloaked(hwnd):
            return False
        if api.ex_style(hwnd) & (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE):
            return False  # click-through overlays and non-activatable tool windows
        rect = api.frame_rect(hwnd)
        return rect is not None and rect.contains(x, y)

    @_best_effort(False)
    def activate_window(self, ref: WindowRef) -> bool:
        """Give ``ref`` the foreground despite Windows' foreground lock.

        Strategies are tried from least to most intrusive; the first two
        inject no input at all, so the OS idle timer (which drives presence
        and the typing heuristic) is untouched in the common case.
        """
        api = self._api
        hwnd = _hwnd_of(ref)
        if hwnd is None or not api.is_window(hwnd) or api.is_iconic(hwnd):
            return False
        if api.foreground() == hwnd:
            return True
        if api.is_hung(hwnd):
            # An app that stopped pumping messages cannot take focus anyway, and
            # waiting on it would freeze our UI thread. (Windows only reports a
            # window as hung after ~5 s; bring_to_top never waits regardless.)
            log.debug("Not activating window %#x: it is not responding", hwnd)
            return False
        api.allow_set_foreground_any()

        if self._activate_attached(hwnd):
            return True
        if api.send_empty_mouse_input() and self._try_foreground(hwnd):
            return True
        if api.tap_alt() and self._try_foreground(hwnd):
            return True
        return api.foreground() == hwnd

    def _try_foreground(self, hwnd: int) -> bool:
        api = self._api
        # Raise only after a granted switch: raising a window that did not get
        # focus would show it on top while keystrokes still go elsewhere.
        if api.set_foreground(hwnd):
            api.bring_to_top(hwnd)
        return api.foreground() == hwnd

    def _activate_attached(self, hwnd: int) -> bool:
        """Share the foreground thread's input state while switching.

        While attached, our thread counts as part of the foreground queue, which
        is what ``SetForegroundWindow`` checks. Skipped for hung windows: sharing
        input state with a thread that does not pump messages can stall ours.
        """
        api = self._api
        current = api.current_thread_id()
        foreground = api.foreground()
        fg_thread = api.thread_process(foreground)[0] if foreground else 0
        if not fg_thread or fg_thread == current:
            return self._try_foreground(hwnd)
        if foreground and api.is_hung(foreground):
            return False
        if not api.attach_thread_input(current, fg_thread, True):
            return False
        try:
            return self._try_foreground(hwnd)
        finally:
            api.attach_thread_input(current, fg_thread, False)

    @_best_effort(False)
    def is_window_valid(self, ref: WindowRef) -> bool:
        api = self._api
        hwnd = _hwnd_of(ref)
        if hwnd is None or not api.is_window(hwnd):
            return False
        if ref.pid is not None and api.thread_process(hwnd)[1] != ref.pid:
            return False  # the handle was recycled for another process's window
        return api.is_visible(hwnd) and not api.is_iconic(hwnd) and not api.is_cloaked(hwnd)

    @_best_effort(None)
    def window_rect(self, ref: WindowRef) -> Rect | None:
        hwnd = _hwnd_of(ref)
        if hwnd is None or not self._api.is_window(hwnd):
            return None
        return self._api.frame_rect(hwnd)

    @_best_effort(None)
    def window_client_rect(self, ref: WindowRef) -> Rect | None:
        hwnd = _hwnd_of(ref)
        if hwnd is None or not self._api.is_window(hwnd):
            return None
        return self._api.client_rect(hwnd)

    @_best_effort(None)
    def window_app(self, ref: WindowRef) -> AppIdentity | None:
        api = self._api
        hwnd = _hwnd_of(ref)
        if hwnd is None or not api.is_window(hwnd):
            return None
        pid = api.thread_process(hwnd)[1]
        if not pid or (ref.pid is not None and pid != ref.pid):
            return None  # gone, or the handle was recycled for another process's window
        name = self._app_process_name(pid)
        if name is None:
            return None
        return AppIdentity(process=name, app_id=api.class_name(hwnd))

    # ------------------------------------------------------------------- camera
    def _own_executables(self) -> frozenset[str]:
        if self._own_paths is None:
            candidates: list[str | None] = [sys.executable, getattr(sys, "_base_executable", None)]
            try:
                # A venv's python.exe is only a launcher; the camera is opened by
                # the interpreter it starts, which is this process's image.
                candidates.append(self._api.process_image_path())
            except Exception:
                log.debug("Could not read the process image path", exc_info=True)
            # Started from a packaged (MSIX) app, such as a terminal inside
            # another app, this process sees redirected paths (``AppData\Roaming``)
            # while the consent store records the real file under
            # ``AppData\Local\Packages\<family>\LocalCache``. Resolving the path
            # through the file system gives that real location.
            candidates += [_real_path(p) for p in candidates if p]
            self._own_paths = frozenset(_norm_path(p) for p in candidates if p)
        return self._own_paths

    def _own_package(self) -> str:
        """Lower-cased package family name of this process; ``""`` without package identity."""
        if self._own_package_name is None:
            name: str | None = None
            try:
                name = self._api.current_package_family_name()
            except Exception:
                log.debug("Could not read the package identity", exc_info=True)
            self._own_package_name = (name or "").lower()
        return self._own_package_name

    def _is_own_camera_user(self, user: _CameraUser) -> bool:
        if user.packaged:
            # With package identity (e.g. the Microsoft Store Python) our own
            # session is recorded under the package family name, not a path.
            own = self._own_package()
            return bool(own) and user.app.lower() == own
        return _norm_path(user.executable) in self._own_executables()

    @_best_effort(None)
    def camera_in_use_by_other_app(self) -> bool | None:
        users = _active_camera_users(self._registry)
        others = [user for user in users if not self._is_own_camera_user(user)]
        if not others:
            return False
        # An app that crashed, or a machine that lost power, mid-session never
        # records the stop time, and the registry keeps that open session across
        # reboots. Sessions opened before this boot are certainly over.
        boot = self._boot_filetime()
        if boot is not None:
            # Generous slack: the boot time is derived from the current clock,
            # which NTP may have moved since. The process check below is exact.
            others = [u for u in others if u.start >= boot - _BOOT_TIME_SLACK]
            if not others:
                return False
        processes = self._processes()
        if not processes:
            return True  # cannot tell live sessions from stale ones: trust the registry
        return any(self._session_is_live(user, processes) for user in others)

    def _session_is_live(self, user: _CameraUser, processes: list[_Process]) -> bool:
        """Whether a process that can own ``user``'s session is still running.

        The process must match the entry (full executable path, or package
        family) and must have been started before the session was: a copy of
        the app launched after a crash never touched the camera. The cheap
        identity checks come first, so the creation time is only read for the
        handful of processes that can be the app at all.
        """
        api = self._api
        if user.packaged:
            family = user.app.lower()
            return any(
                (api.process_package_family_name(p.pid) or "").lower() == family
                and self._started_before(p.pid, user.start)
                for p in processes
            )
        target = _norm_path(user.executable)
        name = ntpath.basename(target)
        for process in processes:
            if process.name != name or not self._started_before(process.pid, user.start):
                continue
            exe = self._process_exe(process.pid)
            # A path we may not read (an elevated process) gets the benefit of the doubt.
            if exe is None or _norm_path(exe) == target:
                return True
        return False

    def _started_before(self, pid: int, session_start: int) -> bool:
        """Whether ``pid`` may have opened a camera session that began at ``session_start``.

        An unreadable creation time gets the benefit of the doubt.
        """
        created = self._process_created(pid)
        return created is None or created <= session_start + _SESSION_START_SLACK

    def _processes(self) -> list[_Process] | None:
        """Running processes (pid and name); ``None`` if they cannot be listed.

        Only the name is requested: it is cheap for every process, whereas
        asking psutil for the creation time as well costs a full system
        snapshot per process of another account (see :class:`_Process`).
        """
        try:
            import psutil
        except Exception:
            return None
        processes: list[_Process] = []
        try:
            for proc in psutil.process_iter(["name"]):
                name = proc.info.get("name")
                if name:
                    processes.append(_Process(pid=int(proc.pid), name=str(name).lower()))
        except Exception:
            log.debug("Could not list processes", exc_info=True)
            return None
        return processes

    @staticmethod
    def _process_created(pid: int) -> int | None:
        """Creation time of ``pid`` as a ``FILETIME`` tick count; ``None`` when unreadable."""
        try:
            import psutil

            created = psutil.Process(pid).create_time()
        except Exception:
            return None
        return _unix_to_filetime(created) if isinstance(created, (int, float)) else None

    @staticmethod
    def _process_exe(pid: int) -> str | None:
        """Full executable path of ``pid``; ``None`` when it is gone or not readable."""
        try:
            import psutil

            exe = psutil.Process(pid).exe()
        except Exception:
            return None
        return str(exe) if exe else None

    @staticmethod
    def _boot_filetime() -> int | None:
        """System boot time as a ``FILETIME`` tick count.

        With Fast Startup this is the last full boot (resuming from hibernation
        keeps the uptime), so it is early rather than late: safe as a filter.
        """
        try:
            import psutil

            return _unix_to_filetime(psutil.boot_time())
        except Exception:
            return None

    # -------------------------------------------------------------- permissions
    def permissions(self) -> dict[str, bool | None]:
        return {"camera": self._camera_permission(), "accessibility": None}

    def _camera_permission(self) -> bool | None:
        """``False`` when a Windows privacy switch blocks desktop apps from the camera."""
        try:
            device = _consent_value(self._registry.values("HKLM", _WEBCAM_KEY))
            user = _consent_value(self._registry.values("HKCU", _WEBCAM_KEY))
            desktop = _consent_value(
                self._registry.values("HKCU", f"{_WEBCAM_KEY}\\{_NON_PACKAGED}")
            )
        except Exception:
            log.debug("Could not read camera consent", exc_info=True)
            return None
        if "deny" in (device, user, desktop):
            return False
        if user == "allow":
            return True
        return None

    def open_permission_settings(self, name: str) -> bool:
        uri = _SETTINGS_URIS.get(name)
        return _open_uri(uri) if uri else False
