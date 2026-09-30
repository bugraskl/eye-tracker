"""macOS integration through pyobjc (Quartz, AppKit, ApplicationServices) and ctypes.

Design
------
* **Coordinates.** Quartz events, the Accessibility (AX) API and
  ``CGWindowListCopyWindowInfo`` all use global display points with a top-left
  origin - exactly Qt's global coordinates on macOS - so nothing is converted.
* **Permissions.** Idle time, lock state, cursor warping and the on-screen
  window list need no permission. Window-level focus needs Accessibility; without
  it the class still works at *application* level (``WindowRef.handle`` is
  ``(pid, None)``, rectangles come from the window list), so focus-follows-gaze
  degrades instead of failing.
* **Imports.** pyobjc is imported lazily (and cached) the first time a method
  needs it, so this module imports on every OS; every public method degrades to
  ``None``/``False`` when pyobjc or a framework symbol is missing.

``WindowRef.handle`` is a ``(pid, AXUIElement | None)`` tuple.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar

from ..types import Rect, WindowRef
from .base import PlatformServices

log = logging.getLogger(__name__)

__all__ = ["MacPlatform"]

#: Evaluated once; tests flip it to exercise macOS-only code paths elsewhere.
_IS_MACOS = sys.platform == "darwin"

LOGIN_FRAMEWORK = "/System/Library/PrivateFrameworks/login.framework/Versions/Current/login"
AVFOUNDATION_FRAMEWORK = "/System/Library/Frameworks/AVFoundation.framework"

_PERMISSION_URLS = {
    "accessibility": (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    ),
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
}

# Framework constants, used when a pyobjc build does not export the symbol.
_CG_EVENT_SOURCE_STATE_COMBINED = 0  # kCGEventSourceStateCombinedSessionState
_CG_ANY_INPUT_EVENT_TYPE = 0xFFFFFFFF  # kCGAnyInputEventType, i.e. (CGEventType)~0
_CG_EVENT_KEY_DOWN = 10  # kCGEventKeyDown
_CG_WINDOW_LIST_ON_SCREEN_ONLY = 1 << 0
_CG_WINDOW_LIST_EXCLUDE_DESKTOP = 1 << 4
_CG_NULL_WINDOW_ID = 0
_AX_VALUE_CGPOINT = 1  # kAXValueCGPointType
_AX_VALUE_CGSIZE = 2  # kAXValueCGSizeType
_AX_SUCCESS = 0
_NS_ACTIVATE_IGNORING_OTHER_APPS = 1 << 1
_NS_ACTIVATION_POLICY_ACCESSORY = 1
_AV_MEDIA_TYPE_VIDEO = "vide"  # value of AVMediaTypeVideo
_AV_STATUS = {1: False, 2: False, 3: True}  # restricted, denied, authorized (0 = not asked)

#: Default AX messaging timeout is 6 s; a hung app must not freeze our UI thread.
_AX_TIMEOUT_S = 0.5
_TRUST_TTL_S = 5.0
_TOOL_TIMEOUT_S = 5.0

_AX_POINT_RE = re.compile(r"x:\s*(-?[\d.]+)\s+y:\s*(-?[\d.]+)")
_AX_SIZE_RE = re.compile(r"w:\s*(-?[\d.]+)\s+h:\s*(-?[\d.]+)")


# ---------------------------------------------------------------------------
# indirections (tests replace these)
# ---------------------------------------------------------------------------
def _load_library(path: str) -> Any:
    return ctypes.CDLL(path)


def _tool(name: str) -> str | None:
    """Absolute path of a system tool (GUI apps may start with a minimal PATH)."""
    found = shutil.which(name)
    if found:
        return found
    for directory in ("/usr/bin", "/usr/sbin", "/bin"):
        candidate = f"{directory}/{name}"
        if os.path.exists(candidate):
            return candidate
    return None


def _run_ok(argv: Sequence[str], timeout: float = _TOOL_TIMEOUT_S) -> bool:
    exe = _tool(argv[0])
    if exe is None:
        return False
    try:
        result = subprocess.run(
            [exe, *argv[1:]],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        log.debug("%s failed: %s", argv[0], exc)
        return False
    if result.returncode != 0:
        log.debug("%s exited with %s", argv[0], result.returncode)
    return result.returncode == 0


def _reap(proc: Any) -> None:
    with contextlib.suppress(Exception):
        proc.wait(timeout=30)


# ---------------------------------------------------------------------------
# pure helpers (unit-tested with fakes on any OS)
# ---------------------------------------------------------------------------
def _err_value(result: Any) -> tuple[int, Any]:
    """Split pyobjc's ``(error, value)`` result of an out-parameter call."""
    if isinstance(result, tuple) and len(result) == 2:
        err, value = result
        try:
            return int(err or 0), value
        except (TypeError, ValueError):
            return -1, None
    return _AX_SUCCESS, result


def _split_handle(handle: Any) -> tuple[int | None, Any]:
    if isinstance(handle, tuple) and len(handle) == 2:
        pid, element = handle
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
            return pid, element
    return None, None


def _session_locked_from(info: Mapping[str, Any] | None) -> bool | None:
    """Interpret ``CGSessionCopyCurrentDictionary()``."""
    if info is None:
        return None  # no window-server session (e.g. ssh)
    try:
        if bool(info.get("CGSSessionScreenIsLocked", False)):
            return True
        # Fast user switching: another user owns the screen. Treated as locked so
        # the camera is released while nobody can see our session.
        on_console = info.get("kCGSSessionOnConsoleKey")
        return on_console is not None and not bool(on_console)
    except Exception:
        return None


def _pair_from_struct(value: Any, names: tuple[str, str]) -> tuple[float, float] | None:
    """(x, y) or (width, height) from a CGPoint/CGSize-like object or a 2-sequence."""
    try:
        if hasattr(value, names[0]) and hasattr(value, names[1]):
            return float(getattr(value, names[0])), float(getattr(value, names[1]))
        if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
            return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return None


def _pair_from_repr(text: str, size: bool) -> tuple[float, float] | None:
    """Parse an AXValue description such as ``{value = x:24.0 y:38.0 type = …}``."""
    match = (_AX_SIZE_RE if size else _AX_POINT_RE).search(text)
    if match is None:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None


def _rect_from(origin: tuple[float, float] | None, size: tuple[float, float] | None) -> Rect | None:
    if origin is None or size is None:
        return None
    values = (*origin, *size)
    if not all(math.isfinite(v) for v in values):
        return None
    x, y, w, h = (round(v) for v in values)
    if w <= 0 or h <= 0:
        return None
    return Rect(x, y, w, h)


def _cg_bounds(info: Mapping[str, Any]) -> Rect | None:
    bounds = info.get("kCGWindowBounds")
    if not bounds:
        return None
    try:
        origin = (float(bounds["X"]), float(bounds["Y"]))
        size = (float(bounds["Width"]), float(bounds["Height"]))
    except (KeyError, TypeError, ValueError):
        return None
    return _rect_from(origin, size)


def _cg_user_windows(
    windows: Iterable[Mapping[str, Any]], own_pid: int
) -> Iterator[tuple[int, Rect]]:
    """(pid, bounds) of ordinary app windows, front to back.

    Layer 0 is the normal window level; the menu bar, Dock, status items and our
    own always-on-top helper windows live on other layers or belong to us.
    """
    for info in windows:
        try:
            if int(info.get("kCGWindowLayer", 0)) != 0:
                continue
            if float(info.get("kCGWindowAlpha", 1.0)) <= 0.0:
                continue
            pid = int(info.get("kCGWindowOwnerPID", 0))
        except (TypeError, ValueError):
            continue
        if pid <= 0 or pid == own_pid:
            continue
        rect = _cg_bounds(info)
        if rect is None or rect.w < 2 or rect.h < 2:
            continue
        yield pid, rect


def _cg_window_at(
    windows: Iterable[Mapping[str, Any]], x: float, y: float, own_pid: int
) -> tuple[int, Rect] | None:
    for pid, rect in _cg_user_windows(windows, own_pid):
        if rect.contains(x, y):
            return pid, rect
    return None


def _cg_front_rect(windows: Iterable[Mapping[str, Any]], pid: int, own_pid: int) -> Rect | None:
    for owner, rect in _cg_user_windows(windows, own_pid):
        if owner == pid:
            return rect
    return None


def _ax_attr(ax: Any, element: Any, name: str) -> Any:
    err, value = _err_value(ax.AXUIElementCopyAttributeValue(element, name, None))
    return value if err == _AX_SUCCESS else None


def _ax_set(ax: Any, element: Any, name: str, value: Any) -> bool:
    try:
        return int(ax.AXUIElementSetAttributeValue(element, name, value) or 0) == _AX_SUCCESS
    except Exception as exc:
        log.debug("AX set %s failed: %s", name, exc)
        return False


def _ax_perform(ax: Any, element: Any, action: str) -> bool:
    try:
        return int(ax.AXUIElementPerformAction(element, action) or 0) == _AX_SUCCESS
    except Exception as exc:
        log.debug("AX action %s failed: %s", action, exc)
        return False


def _ax_pair(ax: Any, value: Any, size: bool) -> tuple[float, float] | None:
    """Unpack an AXValue holding a CGPoint (``size=False``) or a CGSize."""
    if value is None:
        return None
    names = ("width", "height") if size else ("x", "y")
    value_type = (
        getattr(ax, "kAXValueCGSizeType", _AX_VALUE_CGSIZE)
        if size
        else getattr(ax, "kAXValueCGPointType", _AX_VALUE_CGPOINT)
    )
    try:
        ok, struct = _ok_value(ax.AXValueGetValue(value, value_type, None))
        if ok and struct is not None:
            pair = _pair_from_struct(struct, names)
            if pair is not None:
                return pair
    except Exception as exc:
        log.debug("AXValueGetValue failed: %s", exc)
    # Some pyobjc versions bridge AXValue objects already, or cannot unpack them;
    # their description always carries the numbers.
    return _pair_from_struct(value, names) or _pair_from_repr(str(value), size)


def _ok_value(result: Any) -> tuple[bool, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), result[1]
    return True, result


def _ax_rect(ax: Any, element: Any) -> Rect | None:
    if ax is None or element is None:
        return None
    try:
        origin = _ax_pair(ax, _ax_attr(ax, element, "AXPosition"), size=False)
        size = _ax_pair(ax, _ax_attr(ax, element, "AXSize"), size=True)
    except Exception as exc:
        log.debug("AX geometry failed: %s", exc)
        return None
    return _rect_from(origin, size)


def _ax_window_of(ax: Any, element: Any) -> Any:
    """The window that contains an arbitrary UI element."""
    if _ax_attr(ax, element, "AXRole") == "AXWindow":
        return element
    window = _ax_attr(ax, element, "AXWindow")
    if window is not None:
        return window
    current = element
    for _ in range(32):  # bounded: a broken hierarchy must not loop forever
        current = _ax_attr(ax, current, "AXParent")
        if current is None:
            return None
        role = _ax_attr(ax, current, "AXRole")
        if role == "AXWindow":
            return current
        if role == "AXApplication":
            return None
    return None


# ---------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------
class MacPlatform(PlatformServices):
    """macOS 12+ implementation of :class:`PlatformServices`.

    ``clock`` exists for tests; production code uses the default.
    """

    name: ClassVar[str] = "macos"

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._modules: dict[str, Any] = {}
        self._trust_cache: tuple[float, bool] | None = None
        self._ax_ready = False
        self._ax_lock = threading.Lock()

    def _mod(self, name: str) -> Any:
        """Import a pyobjc module once; ``None`` when unavailable."""
        if name not in self._modules:
            try:
                module: Any = importlib.import_module(name)
            except Exception as exc:
                log.debug("%s unavailable: %s", name, exc)
                module = None
            self._modules.setdefault(name, module)
        return self._modules[name]

    def _ax(self) -> Any:
        """ApplicationServices, with a short global AX messaging timeout set once."""
        ax = self._mod("ApplicationServices")
        if ax is None:
            return None
        with self._ax_lock:
            if not self._ax_ready:
                self._ax_ready = True
                try:
                    # Setting the timeout on the system-wide element applies globally.
                    ax.AXUIElementSetMessagingTimeout(
                        ax.AXUIElementCreateSystemWide(), _AX_TIMEOUT_S
                    )
                except Exception as exc:
                    log.debug("AXUIElementSetMessagingTimeout failed: %s", exc)
        return ax

    def _ax_trusted(self) -> bool:
        now = self._clock()
        cached = self._trust_cache
        if cached is not None and now - cached[0] < _TRUST_TTL_S:
            return cached[1]
        trusted = bool(self._accessibility_permission())
        self._trust_cache = (now, trusted)
        return trusted

    # -------------------------------------------------------------- lifecycle
    def prepare_process(self) -> None:
        """Nothing to do: Qt already uses point coordinates, like Quartz and AX.

        The Dock icon is hidden by ``LSUIElement`` in the app bundle and, when
        running from source, by :meth:`set_accessory_app` after QApplication exists.
        """

    def set_accessory_app(self) -> bool:
        """Run as a menu-bar ("accessory") app: no Dock icon, no app menu.

        Must be called on the main thread after the QApplication was created.
        """
        appkit = self._mod("AppKit")
        if appkit is None:
            return False
        try:
            policy = getattr(
                appkit, "NSApplicationActivationPolicyAccessory", _NS_ACTIVATION_POLICY_ACCESSORY
            )
            return bool(appkit.NSApplication.sharedApplication().setActivationPolicy_(policy))
        except Exception as exc:
            log.debug("setActivationPolicy failed: %s", exc)
            return False

    def capabilities(self) -> dict[str, bool]:
        quartz = self._mod("Quartz") is not None
        return {
            "lock": _IS_MACOS,
            "display_off": _IS_MACOS and _tool("pmset") is not None,
            "wake_display": _IS_MACOS and _tool("caffeinate") is not None,
            "input_idle": quartz,
            "key_idle": quartz,
            "session_locked": quartz,
            "focus": self._mod("AppKit") is not None
            and self._mod("ApplicationServices") is not None,
            "cursor": True,
            "camera_in_use": False,
            "hotkeys": _IS_MACOS,
        }

    # ---------------------------------------------------------- session/power
    def lock_screen(self) -> bool:
        """Lock immediately (like Ctrl+Cmd+Q); fall back to display sleep."""
        if not _IS_MACOS:
            return False
        if self._sac_lock():
            log.info("Screen locked")
            return True
        if _run_ok(["pmset", "displaysleepnow"]):
            log.info(
                "Screen lock API unavailable; displays put to sleep instead (this locks "
                "when 'Require password immediately after sleep' is enabled)"
            )
            return True
        log.warning("Could not lock the screen")
        return False

    @staticmethod
    def _sac_lock() -> bool:
        """``SACLockScreenImmediate`` from the private login framework."""
        try:
            lib = _load_library(LOGIN_FRAMEWORK)
            func = lib.SACLockScreenImmediate
            func.restype = ctypes.c_int
            func.argtypes = []
            status = int(func())
        except (OSError, AttributeError, TypeError, ValueError) as exc:
            log.debug("SACLockScreenImmediate unavailable: %s", exc)
            return False
        if status != 0:
            log.debug("SACLockScreenImmediate returned %s", status)
        return status == 0

    def display_off(self) -> bool:
        return _IS_MACOS and _run_ok(["pmset", "displaysleepnow"])

    def wake_display(self) -> bool:
        """Declare user activity for 2 s (``caffeinate -u``), which wakes the displays."""
        if not _IS_MACOS:
            return False
        exe = _tool("caffeinate")
        if exe is None:
            return False
        try:
            proc = subprocess.Popen(
                [exe, "-u", "-t", "2"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, ValueError) as exc:
            log.debug("caffeinate failed: %s", exc)
            return False
        # Non-blocking for the caller, but the child must still be reaped.
        threading.Thread(target=_reap, args=(proc,), name="caffeinate-reaper", daemon=True).start()
        return True

    def is_session_locked(self) -> bool | None:
        quartz = self._mod("Quartz")
        if quartz is None:
            return None
        try:
            info = quartz.CGSessionCopyCurrentDictionary()
        except Exception as exc:
            log.debug("CGSessionCopyCurrentDictionary failed: %s", exc)
            return None
        return _session_locked_from(info)

    # ------------------------------------------------------------------ input
    def seconds_since_input(self) -> float | None:
        return self._seconds_since("kCGAnyInputEventType", _CG_ANY_INPUT_EVENT_TYPE)

    def seconds_since_key_input(self) -> float | None:
        """Time since the last key press; only a timestamp is read, never the key."""
        return self._seconds_since("kCGEventKeyDown", _CG_EVENT_KEY_DOWN)

    def _seconds_since(self, event_name: str, event_default: int) -> float | None:
        quartz = self._mod("Quartz")
        if quartz is None:
            return None
        try:
            state = getattr(
                quartz, "kCGEventSourceStateCombinedSessionState", _CG_EVENT_SOURCE_STATE_COMBINED
            )
            event_type = getattr(quartz, event_name, event_default)
            value = float(quartz.CGEventSourceSecondsSinceLastEventType(state, event_type))
        except Exception as exc:
            log.debug("CGEventSourceSecondsSinceLastEventType failed: %s", exc)
            return None
        return value if math.isfinite(value) and value >= 0.0 else None

    def move_cursor(self, x: int, y: int) -> bool | None:
        quartz = self._mod("Quartz")
        if quartz is None:
            return None  # let the caller fall back to QCursor
        try:
            err = quartz.CGWarpMouseCursorPosition((float(x), float(y)))
            # A warp suppresses local mouse events for ~0.25 s; re-associating the
            # mouse ends that, so the user can take over immediately.
            quartz.CGAssociateMouseAndMouseCursorPosition(True)
        except Exception as exc:
            log.debug("CGWarpMouseCursorPosition failed: %s", exc)
            return None
        return int(err or 0) == 0

    # ---------------------------------------------------------------- windows
    def foreground_window(self) -> WindowRef | None:
        appkit = self._mod("AppKit")
        if appkit is None:
            return None
        try:
            app = appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
            pid = int(app.processIdentifier()) if app is not None else 0
        except Exception as exc:
            log.debug("frontmostApplication failed: %s", exc)
            return None
        if pid <= 0 or pid == os.getpid():
            return None
        window = self._focused_ax_window(pid) if self._ax_trusted() else None
        rect = _ax_rect(self._ax(), window) if window is not None else None
        if rect is None:
            rect = self._cg_front_rect(pid)
        return WindowRef(handle=(pid, window), pid=pid, rect=rect)

    def _focused_ax_window(self, pid: int) -> Any:
        ax = self._ax()
        if ax is None:
            return None
        try:
            return _ax_attr(ax, ax.AXUIElementCreateApplication(pid), "AXFocusedWindow")
        except Exception as exc:
            log.debug("AXFocusedWindow failed: %s", exc)
            return None

    def window_at(self, x: int, y: int) -> WindowRef | None:
        """Frontmost ordinary window under a point (excluding our own).

        The on-screen window list is consulted first: it needs no permission,
        skips our click-through overlay and cannot hang on a busy app. The AX
        element is then looked up for window-level focus when permitted.
        """
        windows = self._cg_windows()
        if windows is None:
            return self._ax_window_at(x, y)
        hit = _cg_window_at(windows, x, y, os.getpid())
        if hit is None:
            return None
        pid, rect = hit
        window = self._ax_window_containing(pid, x, y, rect) if self._ax_trusted() else None
        if window is not None:
            rect = _ax_rect(self._ax(), window) or rect
        return WindowRef(handle=(pid, window), pid=pid, rect=rect)

    def _cg_windows(self) -> list[Mapping[str, Any]] | None:
        quartz = self._mod("Quartz")
        if quartz is None:
            return None
        try:
            options = getattr(
                quartz, "kCGWindowListOptionOnScreenOnly", _CG_WINDOW_LIST_ON_SCREEN_ONLY
            ) | getattr(
                quartz, "kCGWindowListExcludeDesktopElements", _CG_WINDOW_LIST_EXCLUDE_DESKTOP
            )
            info = quartz.CGWindowListCopyWindowInfo(
                options, getattr(quartz, "kCGNullWindowID", _CG_NULL_WINDOW_ID)
            )
        except Exception as exc:
            log.debug("CGWindowListCopyWindowInfo failed: %s", exc)
            return None
        return list(info or [])

    def _cg_front_rect(self, pid: int) -> Rect | None:
        windows = self._cg_windows()
        return _cg_front_rect(windows, pid, os.getpid()) if windows else None

    def _ax_window_containing(self, pid: int, x: int, y: int, hint: Rect) -> Any:
        """The app's AX window under the point, preferring the one matching ``hint``."""
        ax = self._ax()
        if ax is None:
            return None
        try:
            windows = _ax_attr(ax, ax.AXUIElementCreateApplication(pid), "AXWindows") or []
            fallback = None
            for window in windows:  # AXWindows is ordered front to back
                if _ax_attr(ax, window, "AXMinimized"):
                    continue
                rect = _ax_rect(ax, window)
                if rect is None or not rect.contains(x, y):
                    continue
                if rect == hint:
                    return window
                if fallback is None:
                    fallback = window
            return fallback
        except Exception as exc:
            log.debug("AXWindows lookup failed: %s", exc)
            return None

    def _ax_window_at(self, x: int, y: int) -> WindowRef | None:
        """Hit-test through Accessibility (used when the window list is unavailable)."""
        ax = self._ax()
        if ax is None or not self._ax_trusted():
            return None
        try:
            system = ax.AXUIElementCreateSystemWide()
            err, element = _err_value(
                ax.AXUIElementCopyElementAtPosition(system, float(x), float(y), None)
            )
            if err != _AX_SUCCESS or element is None:
                return None
            window = _ax_window_of(ax, element)
            if window is None:
                return None
            err, pid = _err_value(ax.AXUIElementGetPid(window, None))
        except Exception as exc:
            log.debug("AX hit test failed: %s", exc)
            return None
        if err != _AX_SUCCESS or not pid or int(pid) == os.getpid():
            return None
        return WindowRef(handle=(int(pid), window), pid=int(pid), rect=_ax_rect(ax, window))

    def activate_window(self, ref: WindowRef) -> bool:
        """Bring the app forward and raise the window (no synthetic click).

        ``activateWithOptions_`` alone is no longer enough on macOS 14+
        (cooperative activation ignores requests from inactive apps), so the
        Accessibility route (``AXFrontmost`` + ``AXRaise``) is used when permitted.
        """
        pid, window = _split_handle(ref.handle)
        if pid is None:
            return False
        ax = self._ax() if self._ax_trusted() else None
        if ax is not None and window is not None:
            try:
                minimized = bool(_ax_attr(ax, window, "AXMinimized"))
            except Exception:
                minimized = False
            if minimized:
                return False  # never un-minimise behind the user's back
            _ax_set(ax, window, "AXMain", True)
        activated = False
        appkit = self._mod("AppKit")
        if appkit is not None:
            try:
                app = appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
                if app is not None and not app.isTerminated():
                    options = getattr(
                        appkit,
                        "NSApplicationActivateIgnoringOtherApps",
                        _NS_ACTIVATE_IGNORING_OTHER_APPS,
                    )
                    activated = bool(app.activateWithOptions_(options))
            except Exception as exc:
                log.debug("activateWithOptions failed: %s", exc)
        if ax is not None:
            try:
                app_element = ax.AXUIElementCreateApplication(pid)
            except Exception as exc:
                log.debug("AXUIElementCreateApplication failed: %s", exc)
            else:
                activated = _ax_set(ax, app_element, "AXFrontmost", True) or activated
            if window is not None:
                _ax_perform(ax, window, "AXRaise")
        return activated

    def is_window_valid(self, ref: WindowRef) -> bool:
        pid, window = _split_handle(ref.handle)
        appkit = self._mod("AppKit")
        if pid is None or appkit is None:
            return False
        try:
            app = appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if app is None or app.isTerminated() or app.isHidden():
                return False
        except Exception as exc:
            log.debug("NSRunningApplication lookup failed: %s", exc)
            return False
        if window is None:
            return True
        ax = self._ax()
        if ax is None:
            return False
        try:
            if _ax_attr(ax, window, "AXMinimized"):
                return False
        except Exception:
            return False
        # A closed window's element answers every query with an error.
        return _ax_rect(ax, window) is not None

    def window_rect(self, ref: WindowRef) -> Rect | None:
        pid, window = _split_handle(ref.handle)
        if pid is None:
            return None
        if window is not None:
            return _ax_rect(self._ax(), window)
        return self._cg_front_rect(pid)

    def same_window(self, a: WindowRef | None, b: WindowRef | None) -> bool:
        if a is None or b is None:
            return False
        pid_a, win_a = _split_handle(a.handle)
        pid_b, win_b = _split_handle(b.handle)
        if pid_a is None or pid_a != pid_b:
            return False
        if win_a is None and win_b is None:
            return True  # application-level handles (no Accessibility permission)
        if win_a is None or win_b is None:
            return False
        try:
            cf = self._mod("CoreFoundation")
            if cf is not None:
                return bool(cf.CFEqual(win_a, win_b))
            return bool(win_a == win_b)
        except Exception:
            return False

    # ------------------------------------------------------------ permissions
    def permissions(self) -> dict[str, bool | None]:
        return {
            "camera": self._camera_permission(),
            "accessibility": self._accessibility_permission(),
        }

    def _accessibility_permission(self) -> bool | None:
        ax = self._ax()
        if ax is None:
            return None
        try:
            return bool(ax.AXIsProcessTrusted())
        except Exception as exc:
            log.debug("AXIsProcessTrusted failed: %s", exc)
            return None

    def _capture_device(self) -> tuple[Any, Any]:
        """``(AVCaptureDevice class, AVMediaTypeVideo)`` or ``(None, None)``.

        The AVFoundation pyobjc wrapper is optional; without it the class is
        looked up in the Objective-C runtime after loading the framework.
        """
        av = self._mod("AVFoundation")
        if av is not None:
            return av.AVCaptureDevice, getattr(av, "AVMediaTypeVideo", _AV_MEDIA_TYPE_VIDEO)
        objc = self._mod("objc")
        if objc is None:
            return None, None
        try:
            objc.loadBundle(
                "AVFoundation", {}, bundle_path=AVFOUNDATION_FRAMEWORK, scan_classes=False
            )
            return objc.lookUpClass("AVCaptureDevice"), _AV_MEDIA_TYPE_VIDEO
        except Exception as exc:
            log.debug("AVCaptureDevice unavailable: %s", exc)
            return None, None

    def _camera_permission(self) -> bool | None:
        device, media_type = self._capture_device()
        if device is None:
            return None
        try:
            status = int(device.authorizationStatusForMediaType_(media_type))
        except Exception as exc:
            log.debug("authorizationStatusForMediaType failed: %s", exc)
            return None
        return _AV_STATUS.get(status)

    def request_permission(self, name: str) -> None:
        try:
            if name == "accessibility":
                ax = self._ax()
                if ax is None:
                    return
                key = getattr(ax, "kAXTrustedCheckOptionPrompt", "AXTrustedCheckOptionPrompt")
                ax.AXIsProcessTrustedWithOptions({key: True})
                self._trust_cache = None
            elif name == "camera":
                # Needs block metadata, i.e. the AVFoundation wrapper. Without it the
                # prompt still appears the first time OpenCV opens the camera.
                av = self._mod("AVFoundation")
                if av is not None:
                    av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                        getattr(av, "AVMediaTypeVideo", _AV_MEDIA_TYPE_VIDEO),
                        lambda granted: log.info("Camera access granted: %s", bool(granted)),
                    )
        except Exception as exc:
            log.debug("request_permission(%s) failed: %s", name, exc)

    def open_permission_settings(self, name: str) -> bool:
        url = _PERMISSION_URLS.get(name)
        if url is None or not _IS_MACOS:
            return False
        return _run_ok(["open", url])
