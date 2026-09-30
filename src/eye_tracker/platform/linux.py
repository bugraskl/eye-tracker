"""Linux integration for X11 and Wayland sessions.

Design
------
Linux has no single desktop API, so every feature is a short list of
strategies tried in order, gated by what the session actually offers:

* **X11** gets the full feature set in-process: idle time from the
  MIT-SCREEN-SAVER extension (``libXss`` through ``ctypes``), window focus
  through EWMH (``python-xlib``) and cursor warping through Qt.
* **Wayland** forbids global window inspection and pointer warping by design.
  The app still runs (Qt is forced onto XWayland so calibration windows can
  be positioned), presence and screen locking work through ``loginctl`` and
  desktop-specific D-Bus tools, the cursor can move through ``ydotool`` when
  installed, and window focus is reported as unsupported.

Desktop tools (``loginctl``, ``xset``, ``kscreen-doctor``, ``gdbus`` …) are run
with ``subprocess`` - never through a shell - and a short timeout. Anything
that is polled (idle time, lock state, camera use) is cached so a busy tick
never turns into a stream of process launches.

The module imports on every OS (the test suite imports it on Windows and
macOS): ``python-xlib`` and the X libraries are loaded lazily, and every public
method degrades to ``None``/``False`` instead of raising.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from ..types import Rect, WindowRef
from .base import PlatformServices

log = logging.getLogger(__name__)

__all__ = ["DEV_ROOT", "PROC_ROOT", "SYS_ROOT", "LinuxPlatform"]

_T = TypeVar("_T")

#: Filesystem roots. Module-level so tests can point them at a fake tree.
PROC_ROOT = Path("/proc")
DEV_ROOT = Path("/dev")
SYS_ROOT = Path("/sys")

_ACTION_TIMEOUT_S = 5.0  # lock / display power: user-visible, may legitimately take a moment
_QUERY_TIMEOUT_S = 2.0  # polled queries must never stall a tick for long
_LOCKED_HINT_TTL_S = 2.0
_IDLE_TTL_S = 0.5
_CAMERA_TTL_S = 3.0
_RETRY_AFTER_FAILURE_S = 60.0

# X11 protocol constants (<X11/X.h>). Duplicated here so the EWMH logic below can
# be exercised with a fake display on machines without python-xlib.
_X_ANY_PROPERTY_TYPE = 0
_X_CURRENT_TIME = 0
_X_MOTION_NOTIFY = 6
_X_IS_VIEWABLE = 2
_X_SUBSTRUCTURE_NOTIFY_MASK = 1 << 19
_X_SUBSTRUCTURE_REDIRECT_MASK = 1 << 20

#: ``_NET_ACTIVE_WINDOW`` source indication "pager": window managers apply no
#: focus-stealing prevention to requests that come from pagers and taskbars.
_EWMH_SOURCE_PAGER = 2

#: Window types that are part of the desktop shell, not user windows.
_SKIPPED_WINDOW_TYPES = (
    "_NET_WM_WINDOW_TYPE_DESKTOP",
    "_NET_WM_WINDOW_TYPE_DOCK",
    "_NET_WM_WINDOW_TYPE_SPLASH",
    "_NET_WM_WINDOW_TYPE_NOTIFICATION",
)

_VIDEO_NODE_RE = re.compile(r"video\d+")
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
_GDBUS_UINT_RE = re.compile(r"\(\s*(?:u?int(?:16|32|64)\s+)?(\d+)\s*,?\s*\)")

#: Processes that keep camera nodes open to *monitor* them. They only count as
#: "using the camera" while they stream (see ``_maps_video_device``).
_CAMERA_BROKERS = ("pipewire", "wireplumber")

_MUTTER_IDLE = (
    "gdbus",
    "call",
    "--session",
    "--dest",
    "org.gnome.Mutter.IdleMonitor",
    "--object-path",
    "/org/gnome/Mutter/IdleMonitor/Core",
    "--method",
    "org.gnome.Mutter.IdleMonitor.GetIdletime",
)

#: Indirection so tests can fake ``/proc/<pid>/fd`` symlinks portably.
_readlink = os.readlink


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def _parse_gdbus_uint(text: str | None) -> int | None:
    """Parse a single unsigned integer from ``gdbus call`` output, e.g. ``(uint64 12345,)``."""
    match = _GDBUS_UINT_RE.search(text or "")
    return int(match.group(1)) if match else None


def _parse_locked_hint(text: str | None) -> bool | None:
    """Parse ``loginctl show-session -p LockedHint [--value]`` output."""
    lines = (text or "").strip().splitlines()
    if not lines:
        return None
    value = lines[0].strip()
    if "=" in value:
        value = value.split("=", 1)[1].strip()
    value = value.lower()
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def _desktop_tokens(env: Mapping[str, str]) -> set[str]:
    """Lower-cased desktop names from ``XDG_CURRENT_DESKTOP`` and friends."""
    raw = ":".join(
        env.get(key, "")
        for key in ("XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION")
    )
    return {token.strip().lower() for token in re.split(r"[:;]", raw) if token.strip()}


def _is_wayland_env(env: Mapping[str, str]) -> bool:
    return env.get("XDG_SESSION_TYPE", "").strip().lower() == "wayland" or bool(
        env.get("WAYLAND_DISPLAY")
    )


def _is_kde_env(env: Mapping[str, str]) -> bool:
    tokens = _desktop_tokens(env)
    return bool(env.get("KDE_FULL_SESSION")) or any(
        "kde" in token or "plasma" in token for token in tokens
    )


def _is_gnome_env(env: Mapping[str, str]) -> bool:
    # Covers "GNOME", "ubuntu:GNOME", "pop:GNOME", "GNOME-Classic", "Budgie:GNOME" …
    return any("gnome" in token for token in _desktop_tokens(env))


def _lock_commands(session_id: str | None) -> list[list[str]]:
    """Screen-lock strategies, most universal first."""
    commands: list[list[str]] = [
        # logind asks whichever locker the desktop registered (GNOME, KDE, Cinnamon,
        # xss-lock for tiling WMs …). Without an id it targets the caller's session.
        ["loginctl", "lock-session", session_id] if session_id else ["loginctl", "lock-session"],
        ["xdg-screensaver", "lock"],
        ["gnome-screensaver-command", "-l"],
        ["dm-tool", "lock"],
        ["xflock4"],
    ]
    # KDE's qdbus is packaged under several names depending on distro and Qt version.
    commands.extend(
        [tool, "org.freedesktop.ScreenSaver", "/ScreenSaver", "Lock"]
        for tool in ("qdbus", "qdbus6", "qdbus-qt6", "qdbus-qt5")
    )
    commands.append(
        [
            "gdbus",
            "call",
            "--session",
            "--dest",
            "org.freedesktop.ScreenSaver",
            "--object-path",
            "/ScreenSaver",
            "--method",
            "org.freedesktop.ScreenSaver.Lock",
        ]
    )
    return commands


def _display_power_commands(env: Mapping[str, str], on: bool) -> list[list[str]]:
    """Display power strategies for this session, in the order they are tried."""
    wayland = _is_wayland_env(env)
    commands: list[list[str]] = []
    if not wayland:
        # DPMS through the X server. Under XWayland it would only affect a virtual
        # output, which is why it is never tried on Wayland.
        commands.append(["xset", "dpms", "force", "on" if on else "off"])
    if wayland and _is_kde_env(env):
        commands.append(["kscreen-doctor", "--dpms", "on" if on else "off"])
    if _is_gnome_env(env):
        mode = "0" if on else "1"  # Mutter PowerSaveMode: 0 = on, 1 = standby
        commands.append(
            [
                "busctl",
                "--user",
                "set-property",
                "org.gnome.Mutter.DisplayConfig",
                "/org/gnome/Mutter/DisplayConfig",
                "org.gnome.Mutter.DisplayConfig",
                "PowerSaveMode",
                "i",
                mode,
            ]
        )
        commands.append(
            [
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.gnome.Mutter.DisplayConfig",
                "--object-path",
                "/org/gnome/Mutter/DisplayConfig",
                "--method",
                "org.freedesktop.DBus.Properties.Set",
                "org.gnome.Mutter.DisplayConfig",
                "PowerSaveMode",
                f"<int32 {mode}>",
            ]
        )
    if wayland and env.get("SWAYSOCK"):
        state = "on" if on else "off"
        # "power" replaced "dpms" in sway 1.8; older versions only know "dpms".
        commands.append(["swaymsg", "output", "*", "power", state])
        commands.append(["swaymsg", "output", "*", "dpms", state])
    if wayland and env.get("HYPRLAND_INSTANCE_SIGNATURE"):
        commands.append(["hyprctl", "dispatch", "dpms", "on" if on else "off"])
    return commands


def _xcb_plugin_usable() -> bool:
    """Whether Qt's xcb platform plugin can load.

    Qt >= 6.5 needs ``libxcb-cursor.so.0``, which many Wayland-first installs lack;
    forcing xcb without it would abort at start-up instead of falling back.
    """
    try:
        ctypes.CDLL("libxcb-cursor.so.0")
    except OSError:
        return False
    return True


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _is_camera_broker(proc_dir: str) -> bool:
    comm = _read_text(os.path.join(proc_dir, "comm")).strip()
    return comm.startswith(_CAMERA_BROKERS)


def _maps_video_device(proc_dir: str) -> bool:
    """Streaming V4L2 clients mmap the device's buffers; idle monitors do not."""
    return "/dev/video" in _read_text(os.path.join(proc_dir, "maps"))


def _video_target(target: str, devices: set[str] | None) -> bool:
    path = target.split(" (deleted)", 1)[0]
    if devices is None:
        return path.startswith("/dev/video")
    return path in devices


def _as_xid(handle: Any) -> int | None:
    if isinstance(handle, bool) or not isinstance(handle, int) or handle <= 0:
        return None
    return handle


def _is_connection_error(exc: BaseException) -> bool:
    return isinstance(exc, OSError) or type(exc).__name__ in {
        "ConnectionClosedError",
        "DisplayConnectionError",
    }


# ---------------------------------------------------------------------------
# X11 idle time (libXss via ctypes)
# ---------------------------------------------------------------------------
class _XScreenSaverInfo(ctypes.Structure):
    """``XScreenSaverInfo`` from ``<X11/extensions/scrnsaver.h>``."""

    _fields_ = (
        ("window", ctypes.c_ulong),
        ("state", ctypes.c_int),
        ("kind", ctypes.c_int),
        ("til_or_since", ctypes.c_ulong),
        ("idle", ctypes.c_ulong),
        ("eventMask", ctypes.c_ulong),
    )


def _load_first(sonames: Iterable[str], short_name: str) -> Any:
    for soname in sonames:
        try:
            return ctypes.CDLL(soname)
        except OSError:
            continue
    found = ctypes.util.find_library(short_name)
    if not found:
        raise OSError(f"lib{short_name} not found")
    return ctypes.CDLL(found)


class _XssIdle:
    """Milliseconds since the last input, from the MIT-SCREEN-SAVER extension.

    One private ``Display`` connection is opened lazily and reused; the lock
    serialises access because Xlib connections are not thread-safe unless
    ``XInitThreads`` ran before anything else touched Xlib.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._handles: tuple[Any, Any, int, int, Any] | None = None
        self._retry_at = 0.0

    def available(self) -> bool:
        with self._lock:
            return self._ensure()

    def idle_ms(self) -> int | None:
        with self._lock:
            if not self._ensure() or self._handles is None:
                return None
            _xlib, xss, display, root, info = self._handles
            try:
                if not xss.XScreenSaverQueryInfo(display, root, info):
                    return None
                return int(info.contents.idle)
            except Exception as exc:
                log.debug("XScreenSaverQueryInfo failed: %s", exc)
                return None

    def _ensure(self) -> bool:
        if self._handles is not None:
            return True
        now = self._clock()
        if now < self._retry_at:
            return False
        self._handles = self._open()
        if self._handles is None:
            self._retry_at = now + _RETRY_AFTER_FAILURE_S
            return False
        return True

    @staticmethod
    def _open() -> tuple[Any, Any, int, int, Any] | None:
        try:
            xlib = _load_first(("libX11.so.6", "libX11.so"), "X11")
            xss = _load_first(("libXss.so.1", "libXss.so"), "Xss")
            xlib.XOpenDisplay.restype = ctypes.c_void_p
            xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
            xlib.XDefaultRootWindow.restype = ctypes.c_ulong
            xlib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
            xlib.XCloseDisplay.restype = ctypes.c_int
            xlib.XCloseDisplay.argtypes = [ctypes.c_void_p]
            xss.XScreenSaverQueryExtension.restype = ctypes.c_int
            xss.XScreenSaverQueryExtension.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            xss.XScreenSaverAllocInfo.restype = ctypes.POINTER(_XScreenSaverInfo)
            xss.XScreenSaverAllocInfo.argtypes = []
            xss.XScreenSaverQueryInfo.restype = ctypes.c_int
            xss.XScreenSaverQueryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.POINTER(_XScreenSaverInfo),
            ]
        except (OSError, AttributeError) as exc:
            log.debug("libXss unavailable: %s", exc)
            return None
        display = xlib.XOpenDisplay(None)
        if not display:
            log.debug("XOpenDisplay failed")
            return None
        event_base, error_base = ctypes.c_int(), ctypes.c_int()
        # Querying info on a server without the extension would raise an X error,
        # and Xlib's default error handler terminates the process.
        if not xss.XScreenSaverQueryExtension(
            display, ctypes.byref(event_base), ctypes.byref(error_base)
        ):
            log.debug("X server lacks the MIT-SCREEN-SAVER extension")
            xlib.XCloseDisplay(display)
            return None
        info = xss.XScreenSaverAllocInfo()
        if not info:
            xlib.XCloseDisplay(display)
            return None
        root = int(xlib.XDefaultRootWindow(display))
        return (xlib, xss, display, root, info)


# ---------------------------------------------------------------------------
# X11 windows (EWMH via python-xlib)
# ---------------------------------------------------------------------------
def _default_display_factory() -> Any:
    from Xlib import display as xdisplay

    display = xdisplay.Display()
    # Errors of requests without a reply (e.g. an event sent to a window that
    # just closed) are printed to stderr by python-xlib unless handled.
    display.set_error_handler(_log_x_error)
    return display


def _log_x_error(error: Any, request: Any) -> None:
    log.debug("X11 error: %s", error)


class _EwmhClient:
    """Top-level windows through the EWMH hints of the running window manager.

    Uses its own ``Display`` connection (never Qt's) guarded by a lock, so any
    thread may call it. ``display_factory`` exists for tests.
    """

    def __init__(
        self,
        display_factory: Callable[[], Any] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        own_pid: int | None = None,
    ) -> None:
        self._factory = display_factory or _default_display_factory
        self._clock = clock
        self._own_pid = os.getpid() if own_pid is None else own_pid
        self._lock = threading.RLock()
        self._display: Any = None
        self._root: Any = None
        self._atoms: dict[str, int] = {}
        self._retry_at = 0.0

    # ------------------------------------------------------------- public API
    def foreground(self) -> WindowRef | None:
        def query() -> WindowRef | None:
            active = self._prop(self._root, "_NET_ACTIVE_WINDOW")
            wid = _as_xid(int(active[0])) if active else None
            if wid is None:
                return None
            win = self._window(wid)
            if not self._eligible(win):
                return None
            pid = self._pid(win)
            if pid == self._own_pid:
                return None
            return WindowRef(handle=wid, pid=pid, rect=self._frame_rect(win))

        return self._call(query, None)

    def window_at(self, x: int, y: int) -> WindowRef | None:
        def query() -> WindowRef | None:
            stack = self._prop(self._root, "_NET_CLIENT_LIST_STACKING")
            if not stack:
                stack = self._prop(self._root, "_NET_CLIENT_LIST") or []
            for raw in reversed(stack):  # the stacking list is bottom-to-top
                wid = _as_xid(int(raw))
                if wid is None:
                    continue
                try:
                    win = self._window(wid)
                    rect = self._frame_rect(win)
                    if rect is None or not rect.contains(x, y) or not self._eligible(win):
                        continue
                    pid = self._pid(win)
                except Exception as exc:
                    if _is_connection_error(exc):
                        raise
                    continue  # the window vanished between listing and querying it
                if pid == self._own_pid:
                    continue  # our overlay / dialogs: look at what is underneath
                return WindowRef(handle=wid, pid=pid, rect=rect)
            return None

        return self._call(query, None)

    def rect(self, wid: int) -> Rect | None:
        return self._call(lambda: self._frame_rect(self._window(wid)), None)

    def is_valid(self, wid: int) -> bool:
        return bool(self._call(lambda: self._eligible(self._window(wid)), False))

    def activate(self, wid: int) -> bool:
        def send() -> bool:
            win = self._window(wid)
            event = self._client_message(
                win,
                self._atom("_NET_ACTIVE_WINDOW"),
                [_EWMH_SOURCE_PAGER, _X_CURRENT_TIME, 0, 0, 0],
            )
            self._root.send_event(
                event, event_mask=_X_SUBSTRUCTURE_REDIRECT_MASK | _X_SUBSTRUCTURE_NOTIFY_MASK
            )
            self._display.flush()
            return True

        return bool(self._call(send, False))

    def nudge_pointer(self) -> bool:
        """Wake the displays with a 1 px XTest pointer move there and back."""

        def nudge() -> bool:
            if not self._display.has_extension("XTEST"):
                return False
            from Xlib.ext import xtest

            for dx in (1, -1):
                xtest.fake_input(self._display, _X_MOTION_NOTIFY, detail=1, x=dx, y=0)
            self._display.sync()
            return True

        return bool(self._call(nudge, False))

    # -------------------------------------------------------------- internals
    def _call(self, fn: Callable[[], _T], default: _T) -> _T:
        with self._lock:
            if not self._connect():
                return default
            try:
                return fn()
            except Exception as exc:
                log.debug("X11 request failed: %s", exc)
                if _is_connection_error(exc):
                    self._drop()
                return default

    def _connect(self) -> bool:
        if self._display is not None:
            return True
        now = self._clock()
        if now < self._retry_at:
            return False
        try:
            display = self._factory()
            root = display.screen().root
        except Exception as exc:
            log.debug("Cannot open an X11 connection: %s", exc)
            self._retry_at = now + _RETRY_AFTER_FAILURE_S
            return False
        self._display, self._root = display, root
        self._atoms.clear()
        return True

    def _drop(self) -> None:
        display, self._display, self._root = self._display, None, None
        self._retry_at = self._clock() + 1.0
        try:
            if display is not None:
                display.close()
        except Exception:
            pass

    def _atom(self, name: str) -> int:
        atom = self._atoms.get(name)
        if atom is None:
            atom = int(self._display.intern_atom(name))
            self._atoms[name] = atom
        return atom

    def _window(self, wid: int) -> Any:
        return self._display.create_resource_object("window", int(wid))

    def _prop(self, win: Any, name: str) -> list[int] | None:
        prop = win.get_full_property(self._atom(name), _X_ANY_PROPERTY_TYPE)
        value = getattr(prop, "value", None) if prop is not None else None
        if value is None:
            return None
        try:
            return [int(v) for v in value]
        except (TypeError, ValueError):
            return None

    def _pid(self, win: Any) -> int | None:
        values = self._prop(win, "_NET_WM_PID")
        return int(values[0]) if values else None

    def _eligible(self, win: Any) -> bool:
        """Mapped on the current desktop, not minimised, not part of the shell."""
        # Windows on other workspaces are unmapped by the WM (their client window
        # is then "unviewable"), so this also filters other desktops.
        if getattr(win.get_attributes(), "map_state", None) != _X_IS_VIEWABLE:
            return False
        state = self._prop(win, "_NET_WM_STATE") or []
        if self._atom("_NET_WM_STATE_HIDDEN") in state:
            return False
        types = self._prop(win, "_NET_WM_WINDOW_TYPE") or []
        skipped = {self._atom(name) for name in _SKIPPED_WINDOW_TYPES}
        return not any(t in skipped for t in types)

    def _frame_rect(self, win: Any) -> Rect | None:
        """Visible outer rectangle: client area + WM decorations - CSD shadows."""
        geometry = win.get_geometry()
        # Asking the root window to translate the client's origin yields root
        # (= global) coordinates, whatever the reparenting depth.
        origin = self._root.translate_coords(win, 0, 0)
        x, y = int(origin.x), int(origin.y)
        w, h = int(geometry.width), int(geometry.height)
        frame = self._prop(win, "_NET_FRAME_EXTENTS")  # left, right, top, bottom
        if frame and len(frame) >= 4:
            left, right, top, bottom = frame[:4]
            x, y, w, h = x - left, y - top, w + left + right, h + top + bottom
        shadow = self._prop(win, "_GTK_FRAME_EXTENTS")  # invisible CSD shadow margins
        if shadow and len(shadow) >= 4:
            left, right, top, bottom = shadow[:4]
            x, y, w, h = x + left, y + top, w - left - right, h - top - bottom
        if w <= 0 or h <= 0:
            return None
        return Rect(x, y, w, h)

    def _client_message(self, win: Any, atom: int, data: list[int]) -> Any:
        from Xlib.protocol import event

        return event.ClientMessage(window=win, client_type=atom, data=(32, data))


# ---------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------
class LinuxPlatform(PlatformServices):
    """Linux implementation of :class:`PlatformServices` (X11 and Wayland).

    The keyword arguments exist for tests; production code uses the defaults.
    """

    name: ClassVar[str] = "linux"

    def __init__(
        self,
        *,
        proc_root: str | os.PathLike[str] = PROC_ROOT,
        dev_root: str | os.PathLike[str] = DEV_ROOT,
        sys_root: str | os.PathLike[str] = SYS_ROOT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._proc_root = Path(proc_root)
        self._dev_root = Path(dev_root)
        self._sys_root = Path(sys_root)
        self._clock = clock
        #: Variables set by ``prepare_process`` -> their original value (None = unset),
        #: restored for child processes (see ``_child_env``).
        self._env_overrides: dict[str, str | None] = {}
        self._xss = _XssIdle(clock)
        self._ewmh = _EwmhClient(clock=clock)

        self._session_lock = threading.Lock()
        self._resolved_session: str | None = None  # "" = looked up, none found

        self._locked_lock = threading.Lock()
        self._locked_cache: tuple[float, bool | None] | None = None

        self._idle_lock = threading.Lock()
        self._idle_cache: tuple[float, float] | None = None
        self._idle_retry_at = 0.0

        self._camera_lock = threading.Lock()
        self._camera_cache: tuple[float, bool | None] | None = None

    # ---------------------------------------------------------------- session
    @property
    def is_wayland(self) -> bool:
        return _is_wayland_env(os.environ)

    def _x11_session(self) -> bool:
        """A real X11 session (XWayland does not count: it only sees X11 clients)."""
        return not self.is_wayland and bool(os.environ.get("DISPLAY"))

    # -------------------------------------------------------------- lifecycle
    def prepare_process(self) -> None:
        """Unscaled Qt, and XWayland instead of native Wayland when possible.

        Native Wayland clients cannot position windows or read global
        coordinates, which calibration windows and the gaze overlay rely on.
        """
        if "QT_ENABLE_HIGHDPI_SCALING" not in os.environ:
            self._set_env("QT_ENABLE_HIGHDPI_SCALING", "0")
        if not (self.is_wayland and os.environ.get("DISPLAY")):
            return
        if os.environ.get("QT_QPA_PLATFORM"):
            return  # the user chose a platform plugin explicitly
        if _xcb_plugin_usable():
            self._set_env("QT_QPA_PLATFORM", "xcb")
            log.info("Wayland session: using XWayland (QT_QPA_PLATFORM=xcb)")
        else:
            log.warning(
                "Wayland session without libxcb-cursor0: running as a native Wayland client; "
                "calibration windows may be misplaced. Install libxcb-cursor0 to fix this."
            )

    def _set_env(self, key: str, value: str) -> None:
        self._env_overrides.setdefault(key, os.environ.get(key))
        os.environ[key] = value

    def _child_env(self) -> dict[str, str]:
        """Environment for desktop tools: as the user's session, not as ours."""
        env = dict(os.environ)
        # kscreen-doctor & co. are Qt programs: an inherited QT_QPA_PLATFORM=xcb would
        # push them onto XWayland, where e.g. DPMS control does not exist.
        for key, original in self._env_overrides.items():
            if original is None:
                env.pop(key, None)
            else:
                env[key] = original
        # PyInstaller (and AppImage runtimes) point LD_LIBRARY_PATH at bundled
        # libraries, which can break system binaries linked against newer ones.
        original_path = env.pop("LD_LIBRARY_PATH_ORIG", None)
        if original_path is not None:
            if original_path:
                env["LD_LIBRARY_PATH"] = original_path
            else:
                env.pop("LD_LIBRARY_PATH", None)
        elif getattr(sys, "frozen", False):
            env.pop("LD_LIBRARY_PATH", None)
        return env

    def capabilities(self) -> dict[str, bool]:
        try:
            env = os.environ
            x11 = self._x11_session()
            has_display = bool(env.get("DISPLAY"))
            xlib = _module_available("Xlib")
            gnome_idle = _is_gnome_env(env) and shutil.which("gdbus") is not None
            return {
                "lock": any(_available(cmd) for cmd in _lock_commands(None)),
                "display_off": any(_available(c) for c in _display_power_commands(env, False)),
                "wake_display": any(_available(c) for c in _display_power_commands(env, True))
                or (x11 and xlib),
                "input_idle": (x11 and self._xss.available()) or gnome_idle,
                "key_idle": False,
                "session_locked": shutil.which("loginctl") is not None,
                "focus": x11 and xlib,
                "cursor": (not self.is_wayland) or shutil.which("ydotool") is not None,
                "camera_in_use": self._proc_root.is_dir(),
                # Same rule as platform.hotkeys: X11 key grabs (also through XWayland).
                "hotkeys": has_display and xlib,
            }
        except Exception:
            log.debug("capability probe failed", exc_info=True)
            return super().capabilities()

    # ----------------------------------------------------------- subprocesses
    def _run(self, argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
        """Run a desktop tool; ``None`` when it is missing, hangs or cannot start."""
        exe = shutil.which(argv[0])
        if exe is None:
            return None
        try:
            return subprocess.run(
                [exe, *argv[1:]],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=self._child_env(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.debug("%s timed out after %.1f s", argv[0], timeout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            log.debug("%s failed to run: %s", argv[0], exc)
        return None

    def _first_success(self, commands: Iterable[Sequence[str]], timeout: float) -> str | None:
        """Run commands in order until one exits with 0; return its name."""
        for argv in commands:
            result = self._run(argv, timeout)
            if result is None:
                continue
            if result.returncode == 0:
                return " ".join(argv[:2])
            log.debug(
                "%s exited with %s: %s",
                " ".join(argv[:2]),
                result.returncode,
                (result.stderr or "").strip()[:200],
            )
        return None

    # ---------------------------------------------------------- session/power
    def lock_screen(self) -> bool:
        try:
            used = self._first_success(_lock_commands(self._session_id()), _ACTION_TIMEOUT_S)
        except Exception:
            log.debug("lock_screen failed", exc_info=True)
            used = None
        if used is None:
            log.warning("Could not lock the screen: no screen locker responded")
            return False
        log.info("Screen locked via %s", used)
        with self._locked_lock:
            self._locked_cache = None  # the next poll must see the new state
        return True

    def display_off(self) -> bool:
        try:
            used = self._first_success(
                _display_power_commands(os.environ, on=False), _ACTION_TIMEOUT_S
            )
        except Exception:
            log.debug("display_off failed", exc_info=True)
            return False
        if used is None:
            log.debug("No display power control available for this session")
            return False
        log.debug("Displays turned off via %s", used)
        return True

    def wake_display(self) -> bool:
        try:
            used = self._first_success(
                _display_power_commands(os.environ, on=True), _ACTION_TIMEOUT_S
            )
            if used is not None:
                return True
            # Any input wakes DPMS-blanked X11 screens; XTest synthesises some.
            return self._x11_session() and self._ewmh.nudge_pointer()
        except Exception:
            log.debug("wake_display failed", exc_info=True)
            return False

    def is_session_locked(self) -> bool | None:
        now = self._clock()
        with self._locked_lock:
            cached = self._locked_cache
            if cached is not None and now - cached[0] < _LOCKED_HINT_TTL_S:
                return cached[1]
            try:
                value = self._query_locked_hint()
            except Exception:
                log.debug("LockedHint query failed", exc_info=True)
                value = None
            self._locked_cache = (now, value)
            return value

    def _query_locked_hint(self) -> bool | None:
        session = self._session_id()
        if session is None:
            return None
        result = self._run(
            ["loginctl", "show-session", session, "-p", "LockedHint", "--value"],
            _QUERY_TIMEOUT_S,
        )
        if result is None or result.returncode != 0:
            return None
        return _parse_locked_hint(result.stdout)

    def _session_id(self) -> str | None:
        """The logind session of this desktop.

        ``XDG_SESSION_ID`` is missing when the app is started by a systemd user
        unit; the user's graphical ("Display") session is used then.
        """
        env_id = os.environ.get("XDG_SESSION_ID", "").strip()
        if env_id:
            return env_id if _SESSION_ID_RE.fullmatch(env_id) else None
        with self._session_lock:
            if self._resolved_session is None:
                self._resolved_session = self._lookup_display_session()
            return self._resolved_session or None

    def _lookup_display_session(self) -> str:
        getuid = getattr(os, "getuid", None)
        if getuid is None:
            return ""
        result = self._run(
            ["loginctl", "show-user", str(getuid()), "-p", "Display", "--value"],
            _QUERY_TIMEOUT_S,
        )
        if result is None or result.returncode != 0:
            return ""
        lines = result.stdout.strip().splitlines()
        candidate = lines[0].strip() if lines else ""
        if "=" in candidate:
            candidate = candidate.split("=", 1)[1].strip()
        return candidate if _SESSION_ID_RE.fullmatch(candidate) else ""

    # ------------------------------------------------------------------ input
    def seconds_since_input(self) -> float | None:
        try:
            if self._x11_session():
                ms = self._xss.idle_ms()
                if ms is not None:
                    return ms / 1000.0
            return self._mutter_idle_s()
        except Exception:
            log.debug("seconds_since_input failed", exc_info=True)
            return None

    def _mutter_idle_s(self) -> float | None:
        """Idle time from GNOME's Mutter (the only source on GNOME Wayland).

        Each query launches ``gdbus``, so results are cached for a short while and
        extrapolated: returning a stale value unchanged would move the implied
        "last input" timestamp forward and look like fresh (keyboard) activity.
        """
        now = self._clock()
        with self._idle_lock:
            cached = self._idle_cache
            if cached is not None and now - cached[0] < _IDLE_TTL_S:
                return cached[1] + (now - cached[0])
            if now < self._idle_retry_at:
                return None
            result = self._run(_MUTTER_IDLE, _QUERY_TIMEOUT_S)
            ms = _parse_gdbus_uint(result.stdout) if result and result.returncode == 0 else None
            if ms is None:
                self._idle_cache = None
                self._idle_retry_at = now + _RETRY_AFTER_FAILURE_S
                log.debug(
                    "Mutter idle monitor unavailable; retrying in %.0f s", _RETRY_AFTER_FAILURE_S
                )
                return None
            seconds = ms / 1000.0
            self._idle_cache = (now, seconds)
            return seconds

    def move_cursor(self, x: int, y: int) -> bool | None:
        """X11: ``None`` (Qt warps the pointer). Wayland: ``ydotool`` if installed."""
        if not self.is_wayland:
            return None
        result = self._run(
            ["ydotool", "mousemove", "--absolute", "-x", str(int(x)), "-y", str(int(y))],
            _QUERY_TIMEOUT_S,
        )
        return result is not None and result.returncode == 0

    # ---------------------------------------------------------------- windows
    def foreground_window(self) -> WindowRef | None:
        if not self._x11_session():
            return None
        return self._ewmh.foreground()

    def window_at(self, x: int, y: int) -> WindowRef | None:
        if not self._x11_session():
            return None
        return self._ewmh.window_at(int(x), int(y))

    def activate_window(self, ref: WindowRef) -> bool:
        wid = _as_xid(ref.handle)
        if wid is None or not self._x11_session():
            return False
        if not self._ewmh.is_valid(wid):
            return False  # minimised or gone: never un-minimise behind the user's back
        return self._ewmh.activate(wid)

    def is_window_valid(self, ref: WindowRef) -> bool:
        wid = _as_xid(ref.handle)
        if wid is None or not self._x11_session():
            return False
        return self._ewmh.is_valid(wid)

    def window_rect(self, ref: WindowRef) -> Rect | None:
        wid = _as_xid(ref.handle)
        if wid is None or not self._x11_session():
            return None
        return self._ewmh.rect(wid)

    # ----------------------------------------------------------------- camera
    def camera_in_use_by_other_app(self) -> bool | None:
        """Whether another process holds a physical ``/dev/video*`` node open."""
        now = self._clock()
        with self._camera_lock:
            cached = self._camera_cache
            if cached is not None and now - cached[0] < _CAMERA_TTL_S:
                return cached[1]
            try:
                value = self._scan_camera_users()
            except Exception:
                log.debug("camera scan failed", exc_info=True)
                value = None
            self._camera_cache = (now, value)
            return value

    def _scan_camera_users(self) -> bool | None:
        if not self._proc_root.is_dir():
            return None
        devices = self._physical_video_devices()
        if devices is not None and not devices:
            return False  # no camera to compete for
        own_pid = os.getpid()
        with os.scandir(self._proc_root) as procs:
            for proc in procs:
                if not proc.name.isdigit() or int(proc.name) == own_pid:
                    continue
                if self._holds_video_device(proc.path, devices):
                    log.debug("Camera in use by pid %s", proc.name)
                    return True
        return False

    def _holds_video_device(self, proc_dir: str, devices: set[str] | None) -> bool:
        try:
            with os.scandir(os.path.join(proc_dir, "fd")) as fds:
                for fd in fds:
                    try:
                        target = _readlink(fd.path)
                    except OSError:
                        continue
                    if not _video_target(target, devices):
                        continue
                    # PipeWire keeps camera nodes open just to monitor them; it only
                    # competes for the camera while it streams to some client.
                    if _is_camera_broker(proc_dir):
                        return _maps_video_device(proc_dir)
                    return True
        except OSError:
            # Other users' processes (permission denied) or a process that exited.
            return False
        return False

    def _physical_video_devices(self) -> set[str] | None:
        """``/dev/videoN`` paths backed by hardware; ``None`` when unknown."""
        try:
            names = [p.name for p in self._dev_root.iterdir() if _VIDEO_NODE_RE.fullmatch(p.name)]
        except OSError:
            return None
        v4l = self._sys_root / "class" / "video4linux"
        if not v4l.is_dir():
            return {f"/dev/{name}" for name in names}
        # Loopback devices (OBS virtual camera, v4l2loopback) have no parent
        # "device" link; apps reading them never block our physical camera.
        return {
            f"/dev/{name}"
            for name in names
            if not (v4l / name).exists() or (v4l / name / "device").exists()
        }

    # ------------------------------------------------------------ permissions
    def permissions(self) -> dict[str, bool | None]:
        return {"camera": self._camera_access(), "accessibility": None}

    def _camera_access(self) -> bool | None:
        """Whether this user may open a camera node (``video`` group / logind ACL)."""
        try:
            nodes = [p for p in self._dev_root.iterdir() if _VIDEO_NODE_RE.fullmatch(p.name)]
        except OSError:
            return None
        if not nodes:
            return None
        return any(os.access(node, os.R_OK | os.W_OK) for node in nodes)


def _available(argv: Sequence[str]) -> bool:
    return shutil.which(argv[0]) is not None
