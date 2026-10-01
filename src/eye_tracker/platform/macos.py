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
* **Permissions after an update.** Release builds are signed ad hoc, so every
  build has a new code identity. macOS keeps showing the old build as allowed
  under Privacy & Security › Accessibility, yet ``AXIsProcessTrusted()`` is
  ``False`` for the new one: the entry must be removed with "−" and the app
  added again. :meth:`MacPlatform.accessibility_status` reports this case as
  ``"stale"`` (it remembers which build was last trusted) so the UI can say so.
  ``eye-tracker-cli``, the bundle's second executable, identifies the build by
  the app's executable as well, but its own trust belongs to the terminal that
  started it: it never records a grant and reports a missing one as unknown.
* **Hung apps.** Accessibility calls block until the target app answers. An
  app that does not answer in time is left alone for a few seconds (window
  list and application-level handles only), so polling it cannot stall the UI.
* **App Nap.** A menu-bar app without visible windows is a candidate for App
  Nap, which throttles its timers by seconds. :meth:`MacPlatform.set_accessory_app`
  therefore also opts out of it (:meth:`MacPlatform.set_background_activity`).
* **Imports.** pyobjc is imported lazily (and cached) the first time a method
  needs it, so this module imports on every OS; every public method degrades to
  ``None``/``False`` when pyobjc or a framework symbol is missing.

``WindowRef.handle`` is a ``(pid, AXUIElement | None)`` tuple.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import json
import logging
import math
import os
import platform as _stdlib_platform
import plistlib
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from .. import __version__
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
_AX_ERROR_CANNOT_COMPLETE = -25204  # kAXErrorCannotComplete: the app did not answer in time
_NS_ACTIVATE_IGNORING_OTHER_APPS = 1 << 1
_NS_ACTIVATION_POLICY_ACCESSORY = 1
#: NSActivityUserInitiatedAllowingIdleSystemSleep: no App Nap and no timer
#: coalescing, while the system and the displays may still sleep when idle.
_NS_ACTIVITY_USER_INITIATED_ALLOWING_IDLE_SYSTEM_SLEEP = 0x00EFFFFF
_ACTIVITY_REASON = "Eye tracking follows the camera in real time"
_AV_MEDIA_TYPE_VIDEO = "vide"  # value of AVMediaTypeVideo
_AV_STATUS = {1: False, 2: False, 3: True}  # restricted, denied, authorized (0 = not asked)

#: Default AX messaging timeout is 6 s; a hung app must not freeze our UI thread.
_AX_TIMEOUT_S = 0.5
#: Tighter timeout for the frontmost app's focused window, which is polled.
_AX_APP_TIMEOUT_S = 0.25
#: How long an app that did not answer an AX request is left alone.
_AX_UNRESPONSIVE_S = 5.0
_TRUST_TTL_S = 5.0
_TOOL_TIMEOUT_S = 5.0
#: An AX frame and a window-list frame this close (points) are the same window.
_FRAME_TOLERANCE = 2
#: Remembers which build Accessibility was granted to (see ``accessibility_status``).
_ACCESSIBILITY_STATE_FILE = "macos-accessibility.json"
#: Name suffix of the bundle's command-line executable (``eye-tracker-cli``).
_CLI_SUFFIX = "-cli"

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


def _macos_major_version() -> int:
    """Major macOS version (e.g. 14); 0 when unknown or not on macOS."""
    try:
        return int((_stdlib_platform.mac_ver()[0] or "0").split(".")[0])
    except (ValueError, IndexError):
        return 0


def _bundle_main_executable(executable: Path) -> Path | None:
    """The main executable of the app bundle ``executable`` belongs to.

    The bundle holds two programs side by side in ``<App>.app/Contents/MacOS``
    (see ``eye-tracker.spec``): the app itself, named by ``CFBundleExecutable``
    in ``Contents/Info.plist``, and the ``eye-tracker-cli`` command. ``None``
    when ``executable`` is not inside a bundle or the plist cannot be read.
    """
    macos_dir = executable.parent
    contents = macos_dir.parent
    if macos_dir.name != "MacOS" or contents.name != "Contents":
        return None
    try:
        with open(contents / "Info.plist", "rb") as fh:
            info = plistlib.load(fh)
    except Exception as exc:  # OSError, InvalidFileException, an XML parser error ...
        log.debug("Cannot read the bundle's Info.plist: %s", exc)
        return None
    name = info.get("CFBundleExecutable") if isinstance(info, dict) else None
    if not isinstance(name, str) or not name or Path(name).name != name:
        return None
    return macos_dir / name


def _packaged_executables() -> tuple[Path, Path] | None:
    """``(this process's executable, the app's main executable)``; ``None`` from source."""
    if not getattr(sys, "frozen", False):
        return None
    try:
        executable = Path(sys.executable).resolve()
    except OSError:
        return None
    return executable, _bundle_main_executable(executable) or executable


def _build_fingerprint() -> str | None:
    """Identity of this packaged build; ``None`` when running from source.

    An ad-hoc signed app's Accessibility grant is tied to its code hash, which
    changes with every build - and so do the version, size and modification
    time of the bundle's main executable. That is the app's executable even
    in ``eye-tracker-cli``: both programs must name the same build.
    """
    executables = _packaged_executables()
    if executables is None:
        return None
    try:
        stat = executables[1].stat()
    except OSError:
        return None
    return f"{__version__}:{stat.st_size}:{stat.st_mtime_ns}"


def _runs_helper_executable() -> bool:
    """Whether this process is a packaged build's ``eye-tracker-cli``, not the app.

    macOS attributes the privacy permissions of such a process to whatever
    started it (the terminal, for a command typed there), so
    ``AXIsProcessTrusted()`` says nothing about the app's own grant.
    """
    executables = _packaged_executables()
    if executables is None:
        return False
    executable, main = executables
    return executable != main or executable.name.endswith(_CLI_SUFFIX)


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


def _frames_match(a: Rect, b: Rect, tolerance: int = _FRAME_TOLERANCE) -> bool:
    return (
        abs(a.x - b.x) <= tolerance
        and abs(a.y - b.y) <= tolerance
        and abs(a.w - b.w) <= tolerance
        and abs(a.h - b.h) <= tolerance
    )


def _ax_attr_err(ax: Any, element: Any, name: str) -> tuple[int, Any]:
    """``(AXError, value)`` of an attribute; the value is ``None`` on error."""
    err, value = _err_value(ax.AXUIElementCopyAttributeValue(element, name, None))
    return err, (value if err == _AX_SUCCESS else None)


def _ax_attr(ax: Any, element: Any, name: str) -> Any:
    return _ax_attr_err(ax, element, name)[1]


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


def _ax_rect_checked(ax: Any, element: Any) -> tuple[int, Rect | None]:
    """``(AXError, frame)`` of a window; the error lets callers spot hung apps."""
    if ax is None or element is None:
        return _AX_SUCCESS, None
    try:
        err, position = _ax_attr_err(ax, element, "AXPosition")
        if err != _AX_SUCCESS:
            # Closed, or its app is hung: asking for AXSize would wait all over again.
            return err, None
        err, size = _ax_attr_err(ax, element, "AXSize")
        if err != _AX_SUCCESS:
            return err, None
        origin = _ax_pair(ax, position, size=False)
        dimensions = _ax_pair(ax, size, size=True)
    except Exception as exc:
        log.debug("AX geometry failed: %s", exc)
        return _AX_SUCCESS, None
    return _AX_SUCCESS, _rect_from(origin, dimensions)


def _ax_rect(ax: Any, element: Any) -> Rect | None:
    return _ax_rect_checked(ax, element)[1]


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

    ``clock`` and ``state_path`` exist for tests; production code uses the
    defaults (``state_path``: the file remembering which build was trusted).
    """

    name: ClassVar[str] = "macos"

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        state_path: Path | None = None,
    ) -> None:
        self._clock = clock
        self._modules: dict[str, Any] = {}
        self._trust_cache: tuple[float, bool] | None = None
        self._ax_ready = False
        self._ax_lock = threading.Lock()
        #: pid -> clock time until which AX requests to that app are skipped.
        self._ax_unresponsive: dict[int, float] = {}
        self._activity: Any = None  # NSProcessInfo activity token (App Nap opt-out)
        self._activity_lock = threading.Lock()
        self._state_path = state_path
        self._trusted_build: str | None = None  # last build known to be trusted (cached)
        self._trusted_build_loaded = False
        self._build_id: str | None = None  # see _fingerprint
        self._fingerprint_known = False
        self._helper: bool | None = None  # see _is_helper
        self._warned_untrusted = False

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

    def _ax_responsive(self, pid: int) -> bool:
        """``False`` while an app that recently timed out an AX request is left alone."""
        until = self._ax_unresponsive.get(pid)
        if until is None:
            return True
        if self._clock() >= until:
            self._ax_unresponsive.pop(pid, None)
            return True
        return False

    def _note_ax_error(self, pid: int, err: int) -> None:
        if err == _AX_ERROR_CANNOT_COMPLETE:
            if pid not in self._ax_unresponsive:
                log.debug("App %s does not answer Accessibility requests; skipping it", pid)
            self._ax_unresponsive[pid] = self._clock() + _AX_UNRESPONSIVE_S

    # -------------------------------------------------------------- lifecycle
    def prepare_process(self) -> None:
        """Nothing to do: Qt already uses point coordinates, like Quartz and AX.

        The Dock icon is hidden by ``LSUIElement`` in the app bundle and, when
        running from source, by :meth:`set_accessory_app` after QApplication exists.
        """

    def set_accessory_app(self) -> bool:
        """Run as a menu-bar ("accessory") app: no Dock icon, no app menu, no App Nap.

        A menu-bar app without visible windows is exactly what App Nap throttles,
        so this also calls :meth:`set_background_activity`. Must be called on the
        main thread after the QApplication was created. Returns whether the
        activation policy was set.
        """
        self.set_background_activity(True)
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

    def set_background_activity(self, active: bool) -> bool:
        """Opt out of App Nap while ``active`` (``True``), or allow it again.

        Without this, macOS naps a window-less menu-bar app: its timers (the
        controller tick, the walk-away countdown) and the camera loop get
        delayed by up to seconds. The activity is "user initiated, allowing idle
        system sleep": the Mac and its displays still sleep when idle. The
        controller may end it while tracking is paused to save power. Returns
        whether the requested state is in effect.
        """
        with self._activity_lock:
            if active == (self._activity is not None):
                return True
            foundation = self._mod("Foundation")
            if foundation is None:
                return False
            try:
                info = foundation.NSProcessInfo.processInfo()
                if active:
                    options = getattr(
                        foundation,
                        "NSActivityUserInitiatedAllowingIdleSystemSleep",
                        _NS_ACTIVITY_USER_INITIATED_ALLOWING_IDLE_SYSTEM_SLEEP,
                    )
                    # The token must stay referenced: the activity ends with it.
                    self._activity = info.beginActivityWithOptions_reason_(
                        options, _ACTIVITY_REASON
                    )
                    log.debug("App Nap disabled")
                    return self._activity is not None
                info.endActivity_(self._activity)
                self._activity = None
                log.debug("App Nap allowed again")
                return True
            except Exception as exc:
                log.debug("NSProcessInfo activity change failed: %s", exc)
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

    def cursor_position_reliable(self) -> bool:
        """Quartz reports the pointer everywhere, over every app's windows."""
        return True

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
        window = None
        if self._ax_trusted() and self._ax_responsive(pid):
            window = self._focused_ax_window(pid)
        rect = None
        if window is not None:
            err, rect = _ax_rect_checked(self._ax(), window)
            self._note_ax_error(pid, err)
        if rect is None:
            rect = self._cg_front_rect(pid)
        return WindowRef(handle=(pid, window), pid=pid, rect=rect)

    def _focused_ax_window(self, pid: int) -> Any:
        ax = self._ax()
        if ax is None:
            return None
        try:
            app = ax.AXUIElementCreateApplication(pid)
            # Polled twice a second: a busy app must not hold the UI thread for
            # the full global timeout on every poll.
            with contextlib.suppress(Exception):
                ax.AXUIElementSetMessagingTimeout(app, _AX_APP_TIMEOUT_S)
            err, window = _ax_attr_err(ax, app, "AXFocusedWindow")
        except Exception as exc:
            log.debug("AXFocusedWindow failed: %s", exc)
            return None
        self._note_ax_error(pid, err)
        return window

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
        window = None
        if self._ax_trusted() and self._ax_responsive(pid):
            window = self._ax_window_containing(pid, x, y, rect)
        if window is not None:
            rect = _ax_rect(self._ax(), window) or rect
        return WindowRef(handle=(pid, window), pid=pid, rect=rect)

    def _cg_windows(self) -> list[Mapping[str, Any]] | None:
        """On-screen windows, front to back; ``None`` when the list is unavailable."""
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
        return None if info is None else list(info)

    def _cg_front_rect(self, pid: int) -> Rect | None:
        windows = self._cg_windows()
        return _cg_front_rect(windows, pid, os.getpid()) if windows else None

    def _ax_window_containing(self, pid: int, x: int, y: int, hint: Rect) -> Any:
        """The app's AX window under the point, preferring the one matching ``hint``."""
        ax = self._ax()
        if ax is None:
            return None
        try:
            err, windows = _ax_attr_err(ax, ax.AXUIElementCreateApplication(pid), "AXWindows")
            self._note_ax_error(pid, err)
            windows = windows or []
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
        (cooperative activation ignores requests from inactive apps while still
        reporting success), so the Accessibility route (``AXFrontmost`` +
        ``AXRaise``) is used when permitted. Without Accessibility on macOS 14+
        the request is still made but reported as failed: it has no effect for
        a menu-bar app.
        """
        pid, window = _split_handle(ref.handle)
        if pid is None:
            return False
        trusted = self._ax_trusted()
        if not trusted:
            self._warn_untrusted_once()
        # Every AX call to a hung app would block for the full timeout.
        ax = self._ax() if trusted and self._ax_responsive(pid) else None
        if ax is not None and window is not None:
            try:
                err, minimized = _ax_attr_err(ax, window, "AXMinimized")
            except Exception:
                err, minimized = _AX_SUCCESS, False
            self._note_ax_error(pid, err)
            if minimized:
                return False  # never un-minimise behind the user's back
            if err == _AX_ERROR_CANNOT_COMPLETE:
                ax = None
            else:
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
        if not trusted and _macos_major_version() >= 14:
            return False
        return activated

    def is_window_valid(self, ref: WindowRef) -> bool:
        """Running, not hidden or minimised, and on a Space that is shown right now.

        Accessibility keeps answering for windows on other Spaces, other Stage
        Manager stages or behind another app's full-screen Space, with frames
        inside the same display. Activating such a window would make macOS
        switch that display back to its Space, so the on-screen window list
        decides. Windows of an app that does not answer AX requests are not
        valid targets either, for now.
        """
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
            return self._on_screen(pid, None)
        if not self._ax_responsive(pid):
            return False
        ax = self._ax()
        if ax is None:
            return False
        try:
            err, minimized = _ax_attr_err(ax, window, "AXMinimized")
        except Exception:
            return False
        self._note_ax_error(pid, err)
        if minimized or err == _AX_ERROR_CANNOT_COMPLETE:
            return False
        # A closed window's element answers every query with an error.
        err, rect = _ax_rect_checked(ax, window)
        self._note_ax_error(pid, err)
        if rect is None:
            return False
        return self._on_screen(pid, rect)

    def _on_screen(self, pid: int, frame: Rect | None) -> bool:
        """Whether ``pid`` shows a window (with about this frame) on a visible Space."""
        windows = self._cg_windows()
        if windows is None:
            return True  # cannot tell: keep the previous behaviour
        return any(
            owner == pid and (frame is None or _frames_match(bounds, frame))
            for owner, bounds in _cg_user_windows(windows, os.getpid())
        )

    def window_rect(self, ref: WindowRef) -> Rect | None:
        pid, window = _split_handle(ref.handle)
        if pid is None:
            return None
        if window is not None and self._ax_responsive(pid):
            err, rect = _ax_rect_checked(self._ax(), window)
            self._note_ax_error(pid, err)
            if err != _AX_ERROR_CANNOT_COMPLETE:
                return rect
        # Application-level handle, or a hung app: the window list still knows
        # where its front window is.
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
        """Camera and Accessibility permission (``None``: unknown or not decided).

        In ``eye-tracker-cli`` Accessibility is ``None`` whatever macOS says: it
        judged whatever started the command (see :meth:`_is_helper`), usually the
        terminal, not the app, so ``doctor`` must neither call the app's grant
        missing nor report a trusted terminal's grant as the app's.
        """
        accessibility = None if self._is_helper() else self._accessibility_permission()
        return {"camera": self._camera_permission(), "accessibility": accessibility}

    def accessibility_status(self) -> str:
        """Accessibility permission, in more detail than :meth:`permissions`.

        * ``"granted"``: keyboard focus can follow the gaze.
        * ``"missing"``: never granted (or revoked); ``request_permission`` asks.
        * ``"stale"``: not trusted, but a *previous build* of the app was.
          Release builds are signed ad hoc, so each update has a new code
          identity; System Settings keeps showing the old entry as enabled while
          it no longer applies. The fix: remove Eye Tracker from Privacy &
          Security › Accessibility with "−", then add the app again.
        * ``"unknown"``: the Accessibility API is unavailable, or this is
          ``eye-tracker-cli``, whose answer from macOS belongs to whatever
          started it (usually the terminal), not to the app; see :meth:`_is_helper`.

        Only packaged builds can be told apart this way; a source checkout
        (whose permission belongs to the terminal or Python) reports "missing".
        """
        if self._is_helper():
            return "unknown"
        trusted = self._accessibility_permission()
        if trusted is None:
            return "unknown"
        self._trust_cache = (self._clock(), trusted)
        if trusted:
            return "granted"
        current = self._fingerprint()
        if current is None:
            return "missing"
        previous = self._load_trusted_build()
        return "stale" if previous is not None and previous != current else "missing"

    def _fingerprint(self) -> str | None:
        """This build's :func:`_build_fingerprint`, computed once per process."""
        if not self._fingerprint_known:
            self._fingerprint_known = True
            self._build_id = _build_fingerprint()
        return self._build_id

    def _is_helper(self) -> bool:
        """Whether this process is ``eye-tracker-cli`` (cached :func:`_runs_helper_executable`).

        macOS attributes its Accessibility trust to the process that started
        it - usually the terminal - so a trusted terminal must not be recorded
        as a trusted app build, and an untrusted one says nothing about the app.
        """
        if self._helper is None:
            self._helper = _runs_helper_executable()
        return self._helper

    def _accessibility_permission(self) -> bool | None:
        """Whether *this process* may use the Accessibility API (``AXIsProcessTrusted``)."""
        ax = self._ax()
        if ax is None:
            return None
        try:
            trusted = bool(ax.AXIsProcessTrusted())
        except Exception as exc:
            log.debug("AXIsProcessTrusted failed: %s", exc)
            return None
        if trusted and not self._is_helper():
            self._remember_trusted_build()
        return trusted

    def _warn_untrusted_once(self) -> None:
        if self._warned_untrusted:
            return
        self._warned_untrusted = True
        if self.accessibility_status() == "stale":
            log.warning(
                "Accessibility access was granted to an earlier version of Eye Tracker and "
                "does not apply to this one: keyboard focus cannot follow your gaze. In System "
                "Settings › Privacy & Security › Accessibility, remove Eye Tracker with '−' "
                "and add it again."
            )
        else:
            log.warning(
                "Accessibility permission is missing: keyboard focus cannot follow your gaze "
                "(System Settings › Privacy & Security › Accessibility)."
            )

    def _state_file(self) -> Path:
        if self._state_path is None:
            from .. import paths

            self._state_path = paths.data_dir() / _ACCESSIBILITY_STATE_FILE
        return self._state_path

    def _load_trusted_build(self) -> str | None:
        if not self._trusted_build_loaded:
            self._trusted_build_loaded = True
            try:
                data = json.loads(self._state_file().read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.debug("No record of an earlier Accessibility grant: %s", exc)
                data = None
            value = data.get("trusted_build") if isinstance(data, dict) else None
            self._trusted_build = value if isinstance(value, str) else None
        return self._trusted_build

    def _remember_trusted_build(self) -> None:
        """Record this build as trusted, so a later update can recognise a stale grant."""
        current = self._fingerprint()
        if current is None or self._load_trusted_build() == current:
            return
        path = self._state_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".tmp")
            partial.write_text(json.dumps({"trusted_build": current}), encoding="utf-8")
            os.replace(partial, path)
        except OSError as exc:
            log.debug("Could not record the Accessibility grant: %s", exc)
            return
        self._trusted_build = current

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
