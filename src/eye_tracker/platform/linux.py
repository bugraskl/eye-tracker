"""Linux integration for X11 and Wayland sessions.

Design
------
Linux has no single desktop API, so every feature is a short list of
strategies tried in order, gated by what the session actually offers:

* **X11** gets the full feature set in-process: idle time from the
  MIT-SCREEN-SAVER extension (``libXss`` through ``ctypes``), window focus
  through EWMH and display power through the DPMS extension (both with
  ``python-xlib``), cursor warping through Qt.
* **Wayland** forbids global window inspection and pointer warping by design.
  The app still runs (Qt prefers XWayland so calibration windows can be
  positioned), presence and screen locking work through logind and
  desktop-specific tools, the cursor moves through the compositor's own IPC
  (sway, Hyprland) or ``ydotool``, and window focus is reported as unsupported.

Queries that are polled (idle time, lock state) go through D-Bus in-process
(QtDBus) once a Qt application exists, and fall back to ``gdbus``/``loginctl``.
Desktop tools are run with ``subprocess`` - never through a shell - and a short
timeout. Everything polled is cached so a busy tick never turns into a stream
of process launches, and the camera-in-use scan runs on a background thread.

Lock detection
--------------
logind's ``LockedHint`` is only set by lockers that report it (GNOME Shell,
KDE, light-locker ...). Tiling-WM lockers (swaylock, i3lock via xss-lock,
hyprlock, slock ...) never do, so a running locker process of this user's
login counts as "locked" as well (setuid lockers such as slock included, see
:meth:`LinuxPlatform._locker_running`), and ``loginctl lock-session`` only
counts as a successful lock when something reacted to it (see
:meth:`LinuxPlatform.lock_screen`). Tiling compositors that pose as GNOME or
KDE through ``XDG_CURRENT_DESKTOP`` are recognised by their IPC sockets.

Camera use by other apps
------------------------
A background thread looks for processes holding a physical ``/dev/video*``
node (``/proc/<pid>/fd``). With fanotify (unprivileged since Linux 5.13) it is
woken by every open and close of a camera node and can tell our own from other
processes'. V4L2 lets only one process stream, so another app that opens the
camera read-write while we stream and closes it again at once (a browser
joining a call gets EBUSY) was refused; the camera is then reported busy for a
grace period so that app can retry. With inotify only
(no process attribution) holders are still found as soon as they open the
device; without either, the scan is polled. Apps that capture through
PipeWire's camera portal never open the node themselves and cannot be seen;
``pause_for_apps`` covers those.

Cursor on Wayland
-----------------
``ydotool`` (1.x, with ``ydotoold`` running) moves the pointer through a
virtual uinput mouse. Its "absolute" move is emulated with relative motion,
so the compositor's pointer acceleration distorts it unless the acceleration
profile is flat. The injected motion also resets GNOME's idle timer; such
resets are recognised and not reported as user input.

The module imports on every OS (the test suite imports it on Windows and
macOS): ``python-xlib``, QtDBus and the X libraries are loaded lazily, and
every public method degrades to ``None``/``False`` instead of raising.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import importlib.util
import logging
import math
import os
import re
import select
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Protocol, TypeVar

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
_IDLE_QUERY_TIMEOUT_S = 1.0  # the idle time is read on every tick
_DBUS_TIMEOUT_MS = 500
_LOCKED_HINT_TTL_S = 2.0
_IDLE_TTL_S = 0.5
_RETRY_AFTER_FAILURE_S = 60.0
#: Retry delay after a failed idle query on a desktop where it worked before:
#: a busy gnome-shell must not hide the user's typing for a whole minute.
_IDLE_TRANSIENT_RETRY_S = 1.0
_SESSION_RETRY_S = 30.0
#: How long ``loginctl lock-session`` may take to make a locker appear.
_LOCK_CONFIRM_S = 2.0
_LOCK_CONFIRM_POLL_S = 0.2
#: An idle-timer reset this close to our own ``ydotool`` motion was caused by it.
_INJECTION_WINDOW_S = 0.5
#: While displays are off with DPMS enabled only for that, how often and how
#: long to watch for them waking up (so DPMS can be switched back off).
_DPMS_WATCH_INTERVAL_S = 1.0
_DPMS_WATCH_MAX_S = 12 * 3600.0

# X11 protocol constants (<X11/X.h>). Duplicated here so the EWMH logic below can
# be exercised with a fake display on machines without python-xlib.
_X_ANY_PROPERTY_TYPE = 0
_X_CURRENT_TIME = 0
_X_MOTION_NOTIFY = 6
_X_IS_VIEWABLE = 2
_X_SUBSTRUCTURE_NOTIFY_MASK = 1 << 19
_X_SUBSTRUCTURE_REDIRECT_MASK = 1 << 20
_DPMS_MODE_ON = 0  # Xlib.ext.dpms.DPMSModeOn
_DPMS_MODE_OFF = 3  # Xlib.ext.dpms.DPMSModeOff

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
_XSET_DPMS_RE = re.compile(r"dpms is (enabled|disabled)")

#: Processes that keep camera nodes open to *monitor* them. They only count as
#: "using the camera" while they stream (see ``_maps_video_device``).
_CAMERA_BROKERS = ("pipewire", "wireplumber")

#: Screen lockers that run exactly while the screen is locked and do not report
#: it to logind. ``/proc/<pid>/comm`` holds at most 15 characters.
_LOCKER_COMMS = frozenset(
    name[:15]
    for name in (
        "i3lock",
        "swaylock",
        "hyprlock",
        "xsecurelock",
        "slock",
        "gtklock",
        "waylock",
        "xlock",
        "alock",
        "sflock",
        "xtrlock",
        "kscreenlocker_greet",
        "xscreensaver-auth",
    )
)

#: Desktops whose own locker reacts to logind's ``Lock`` signal, so
#: ``loginctl lock-session`` is trusted there without further evidence.
_LOGIND_LOCK_DESKTOPS = (
    "gnome",
    "kde",
    "plasma",
    "cinnamon",
    "mate",
    "xfce",
    "budgie",
    "unity",
    "deepin",
)

#: Distributions that brand GNOME Shell sessions with their own name in front
#: (``ubuntu:GNOME``, ``pop:GNOME``). Other desktops that list GNOME second do so
#: for compatibility only: Budgie (``Budgie:GNOME``) locks with budgie-screensaver,
#: a gnome-screensaver 3.6 fork that never reports to logind's ``LockedHint``.
_GNOME_SHELL_VENDORS = ("ubuntu", "pop", "zorin", "endless")

#: Variables set by tiling window managers / compositors for their IPC (sway
#: also sets ``I3SOCK``). See ``_standalone_compositor``.
_COMPOSITOR_SOCKET_VARS = (
    "SWAYSOCK",
    "I3SOCK",
    "HYPRLAND_INSTANCE_SIGNATURE",
    "NIRI_SOCKET",
    "WAYFIRE_SOCKET",
)

#: D-Bus errors meaning "this service / method does not exist here" (as opposed
#: to a timeout or a busy service), in QtDBus error names and gdbus messages.
_DBUS_MISSING_MARKERS = (
    "ServiceUnknown",
    "UnknownMethod",
    "UnknownObject",
    "UnknownInterface",
    "NameHasNoOwner",
    "was not provided by any .service files",
)

_MUTTER_IDLE_SERVICE = "org.gnome.Mutter.IdleMonitor"
_MUTTER_IDLE_PATH = "/org/gnome/Mutter/IdleMonitor/Core"
_MUTTER_IDLE = (
    "gdbus",
    "call",
    "--session",
    "--dest",
    _MUTTER_IDLE_SERVICE,
    "--object-path",
    _MUTTER_IDLE_PATH,
    "--method",
    "org.gnome.Mutter.IdleMonitor.GetIdletime",
)
_LOGIN1 = "org.freedesktop.login1"
_LOGIN1_PATH = "/org/freedesktop/login1"
_LOGIN1_MANAGER = "org.freedesktop.login1.Manager"
_LOGIN1_SESSION = "org.freedesktop.login1.Session"
_DBUS_PROPERTIES = "org.freedesktop.DBus.Properties"

#: Where ``ydotoold`` listens when ``YDOTOOL_SOCKET`` is not set (besides
#: ``$XDG_RUNTIME_DIR/.ydotool_socket``). Module-level so tests can clear it.
_YDOTOOL_DEFAULT_SOCKETS: tuple[str, ...] = ("/tmp/.ydotool_socket",)

#: Indirection so tests can fake ``/proc/<pid>/fd`` symlinks portably.
_readlink = os.readlink


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def _getuid() -> int | None:
    """The real user id; ``None`` on systems without one (the tests run on Windows)."""
    getuid = getattr(os, "getuid", None)
    return int(getuid()) if getuid is not None else None


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


def _parse_xset_dpms(text: str | None) -> tuple[bool, bool] | None:
    """``(capable, enabled)`` from ``xset q`` output; ``None`` when it does not say."""
    lowered = (text or "").lower()
    if "does not have the dpms extension" in lowered or "not capable of dpms" in lowered:
        return False, False
    match = _XSET_DPMS_RE.search(lowered)
    if match is None:
        return None
    return True, match.group(1) == "enabled"


def _is_missing_service(text: str | None) -> bool:
    """Whether a D-Bus error says the service or method does not exist (vs. a hiccup)."""
    return any(marker in (text or "") for marker in _DBUS_MISSING_MARKERS)


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


def _standalone_compositor(env: Mapping[str, str]) -> bool:
    """Whether a tiling window manager / compositor runs the session.

    Such sessions often export ``XDG_CURRENT_DESKTOP=GNOME`` (or KDE) so that
    portals and settings apps work; their own sockets tell the truth, and then
    no GNOME or KDE lock screen exists to react to logind or report the lock.
    """
    return any(env.get(key) for key in _COMPOSITOR_SOCKET_VARS)


def _desktop_handles_logind_lock(env: Mapping[str, str]) -> bool:
    """Whether the desktop's own locker reacts to ``loginctl lock-session``."""
    if _standalone_compositor(env):
        return False
    if env.get("KDE_FULL_SESSION"):
        return True
    return any(name in token for token in _desktop_tokens(env) for name in _LOGIND_LOCK_DESKTOPS)


def _session_desktop(env: Mapping[str, str]) -> str:
    """The session's own desktop: the first ``XDG_CURRENT_DESKTOP`` entry (later
    ones are compatibility fallbacks), else the session name; lower-cased."""
    for key in ("XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION"):
        for token in re.split(r"[:;]", env.get(key, "")):
            if token.strip():
                return token.strip().lower()
    return ""


def _runs_gnome_shell(env: Mapping[str, str]) -> bool:
    """Whether GNOME Shell (or GNOME Flashback) runs the session, as opposed to a
    desktop that merely lists GNOME as a fallback (see :data:`_GNOME_SHELL_VENDORS`)."""
    own = _session_desktop(env)
    return "gnome" in own or own in _GNOME_SHELL_VENDORS


def _locked_hint_authoritative(env: Mapping[str, str]) -> bool:
    """Desktops whose lock screen always reports itself through logind's ``LockedHint``.

    Only these may have a locker's claimed success checked against the hint;
    elsewhere a lock that the hint never shows would be taken for a failure,
    and the next locker would lock the user a second time.
    """
    if _standalone_compositor(env):
        return False
    return _is_kde_env(env) or _runs_gnome_shell(env)


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


def _lock_method_label(argv: Sequence[str]) -> str:
    """How ``doctor`` names a :func:`_lock_commands` entry (ASCII, like the report).

    The D-Bus clients are named with the call they make; "gdbus call" alone
    would not say what it does.
    """
    if argv[0] == "gdbus" or argv[0].startswith("qdbus"):
        return f"{argv[0]} org.freedesktop.ScreenSaver.Lock"
    return " ".join(argv[:2])


def _display_power_commands(env: Mapping[str, str], on: bool) -> list[list[str]]:
    """Desktop-specific display power strategies, in the order they are tried.

    X11 DPMS is handled separately (:meth:`LinuxPlatform._x11_dpms`): under
    XWayland it would only affect a virtual output, and ``xset`` needs care.
    """
    wayland = _is_wayland_env(env)
    commands: list[list[str]] = []
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


def _ydotool_socket_present(env: Mapping[str, str]) -> bool:
    """Whether ``ydotoold`` appears to be running (its socket exists).

    ydotool 1.x cannot inject anything without the daemon; the client looks at
    ``$YDOTOOL_SOCKET``, else ``$XDG_RUNTIME_DIR/.ydotool_socket`` or
    ``/tmp/.ydotool_socket`` depending on its version.
    """
    explicit = env.get("YDOTOOL_SOCKET", "").strip()
    if explicit:
        candidates: list[str] = [explicit]
    else:
        runtime = env.get("XDG_RUNTIME_DIR", "").strip()
        candidates = [os.path.join(runtime, ".ydotool_socket")] if runtime else []
        candidates.extend(_YDOTOOL_DEFAULT_SOCKETS)
    return any(os.path.exists(path) for path in candidates)


def _pointer_command_ok(argv: Sequence[str], result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode != 0:
        return False
    if argv[0] == "hyprctl":
        # hyprctl exits with 0 even when the dispatcher failed; it prints "ok" on success.
        return (result.stdout or "").strip().lower() in ("", "ok")
    return True


def _xcb_plugin_usable() -> bool:
    """Whether Qt's xcb platform plugin can load.

    Qt >= 6.5 needs ``libxcb-cursor.so.0``, which many Wayland-first installs lack.
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


#: ``/proc/<pid>/loginuid`` of a process that did not come from a login: (uid_t)-1.
_AUDIT_UID_UNSET = 0xFFFF_FFFF


def _login_uid(proc_dir: str) -> int | None:
    """The audit login UID of a process; ``None`` when unset or unreadable."""
    text = _read_text(os.path.join(proc_dir, "loginuid")).strip()
    if not text.isdigit():
        return None
    value = int(text)
    return None if value == _AUDIT_UID_UNSET else value


def _belongs_to_user(entry: os.DirEntry[str], own_login: int | None, uid: int | None) -> bool:
    """Whether the process at ``entry`` belongs to this user (see ``_locker_running``).

    It does when this user owns its ``/proc`` entry, or when its audit login
    UID is ours (``own_login``, or ``uid`` when this process has none): the
    latter survives the setuid tricks of slock & co. A login UID that is unset
    on either side proves nothing, so then only the owner counts.
    """
    if uid is None:
        return True  # no users to tell apart (not a POSIX system: tests)
    login = _login_uid(entry.path)
    if login is not None and login == (own_login if own_login is not None else uid):
        return True
    try:
        return entry.stat(follow_symlinks=False).st_uid == uid
    except OSError:
        return False  # exited meanwhile


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
# X11 windows and DPMS (python-xlib)
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
    """Top-level windows (EWMH hints of the window manager) and DPMS of the X server.

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

    def dpms_status(self) -> tuple[bool, bool] | None:
        """``(capable, enabled)`` of the server's DPMS; ``None`` when X is unreachable."""

        def query() -> tuple[bool, bool]:
            if not self._display.has_extension("DPMS"):
                return False, False
            if not bool(getattr(self._display.dpms_capable(), "capable", False)):
                return False, False
            return True, bool(getattr(self._display.dpms_info(), "state", False))

        return self._call(query, None)

    def dpms_monitors_on(self) -> bool | None:
        """Whether the monitors are powered on (DPMS power level "On")."""

        def query() -> bool:
            level = getattr(self._display.dpms_info(), "power_level", _DPMS_MODE_ON)
            return int(level) == _DPMS_MODE_ON

        return self._call(query, None)

    def dpms_force(self, off: bool) -> bool:
        """Force the monitors off or on; DPMS must be enabled for this."""

        def force() -> bool:
            self._display.dpms_force_level(_DPMS_MODE_OFF if off else _DPMS_MODE_ON)
            self._display.sync()
            return True

        return bool(self._call(force, False))

    def dpms_set_enabled(self, enabled: bool) -> bool:
        def apply() -> bool:
            if enabled:
                self._display.dpms_enable()
            else:
                self._display.dpms_disable()
            self._display.sync()
            return True

        return bool(self._call(apply, False))

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
# D-Bus (QtDBus, in-process)
# ---------------------------------------------------------------------------
class _DBusReply(NamedTuple):
    """Result of a D-Bus method call."""

    values: tuple[Any, ...]
    #: D-Bus error name; empty on success.
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _dbus_unwrap(value: Any) -> Any:
    """Plain Python values from QtDBus wrappers (variants, object paths)."""
    variant = getattr(value, "variant", None)
    if callable(variant):
        return _dbus_unwrap(variant())
    path = getattr(value, "path", None)
    if callable(path) and type(value).__name__ == "QDBusObjectPath":
        return str(path())
    return value


class _QtDBus:
    """Blocking D-Bus method calls through QtDBus, without launching processes.

    Polled queries (idle time, lock state) would otherwise start ``gdbus`` or
    ``loginctl`` several times a second. :meth:`call` returns ``None`` when
    QtDBus, a Qt application or the bus is unavailable; callers then fall back
    to the command-line tools. Any thread may call it.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._modules: tuple[Any, Any, Any, Any] | None = None
        self._import_failed = False
        self._bus_retry_at: dict[bool, float] = {}

    def call(
        self,
        *,
        system: bool,
        service: str,
        path: str,
        interface: str,
        method: str,
        args: Sequence[Any] = (),
        timeout_ms: int = _DBUS_TIMEOUT_MS,
    ) -> _DBusReply | None:
        modules = self._load()
        if modules is None:
            return None
        app_class, qdbus, connection_class, message_class = modules
        try:
            # A connection made before the QCoreApplication exists "may misbehave"
            # (Qt's words); the command-line fallback covers that short window.
            if app_class.instance() is None:
                return None
            now = self._clock()
            with self._lock:
                if now < self._bus_retry_at.get(system, -math.inf):
                    return None
            bus = connection_class.systemBus() if system else connection_class.sessionBus()
            if not bus.isConnected():
                with self._lock:
                    self._bus_retry_at[system] = now + _RETRY_AFTER_FAILURE_S
                log.debug("D-Bus %s bus unavailable", "system" if system else "session")
                return None
            message = message_class.createMethodCall(service, path, interface, method)
            if args:
                message.setArguments(list(args))
            reply = bus.call(message, qdbus.CallMode.Block, int(timeout_ms))
            if reply.type() == message_class.MessageType.ReplyMessage:
                return _DBusReply(tuple(_dbus_unwrap(v) for v in reply.arguments()))
            return _DBusReply((), str(reply.errorName() or "org.freedesktop.DBus.Error.Failed"))
        except Exception as exc:
            log.debug("D-Bus call %s.%s failed: %s", interface, method, exc)
            return None

    def _load(self) -> tuple[Any, Any, Any, Any] | None:
        with self._lock:
            if self._modules is None and not self._import_failed:
                try:
                    from PySide6.QtCore import QCoreApplication
                    from PySide6.QtDBus import QDBus, QDBusConnection, QDBusMessage
                except Exception as exc:
                    log.debug("QtDBus unavailable: %s", exc)
                    self._import_failed = True
                else:
                    self._modules = (QCoreApplication, QDBus, QDBusConnection, QDBusMessage)
            return self._modules


# ---------------------------------------------------------------------------
# camera-node notifications (fanotify / inotify via ctypes)
# ---------------------------------------------------------------------------
class _CameraEvent(NamedTuple):
    """An open and/or close of a watched camera node."""

    opened: bool
    #: Whether another process did it; ``None`` when the notifier cannot tell.
    foreign: bool | None
    #: A descriptor opened for writing was closed. Capture opens the node
    #: read-write, device listings (browsers enumerating cameras, ``v4l2-ctl``
    #: queries) mostly read-only - so this marks a capture attempt ending.
    wrote: bool = False


class _CameraNotifier(Protocol):
    """Blocking source of :class:`_CameraEvent` for a set of device nodes."""

    kind: str

    @property
    def alive(self) -> bool:
        """False once the notifier failed for good (the watcher then polls)."""
        ...

    def wait(self, timeout: float) -> list[_CameraEvent]: ...

    def close(self) -> None: ...


# <linux/fanotify.h>, <sys/inotify.h>, <fcntl.h>
_FAN_CLOEXEC = 0x0000_0001
_FAN_NONBLOCK = 0x0000_0002
_FAN_REPORT_FID = 0x0000_0200
_FAN_MARK_ADD = 0x0000_0001
_FAN_CLOSE_WRITE = 0x0000_0008
_FAN_CLOSE_NOWRITE = 0x0000_0010
_FAN_OPEN = 0x0000_0020
_FAN_Q_OVERFLOW = 0x0000_4000
_FANOTIFY_METADATA_VERSION = 3
_IN_CLOSE_WRITE = 0x0000_0008
_IN_CLOSE_NOWRITE = 0x0000_0010
_IN_OPEN = 0x0000_0020
_AT_FDCWD = -100
_POLLIN = 0x0001  # <poll.h>; ``select.POLLIN`` does not exist on Windows, where tests also run
#: struct fanotify_event_metadata: event_len, vers, reserved, metadata_len, mask, fd, pid.
_FAN_EVENT = struct.Struct("=IBBHQii")
#: struct inotify_event without its trailing name: wd, mask, cookie, len.
_IN_EVENT = struct.Struct("=iIII")


def _parse_fanotify(data: bytes, own_pid: int) -> list[_CameraEvent]:
    """Events from a ``read`` of a fanotify descriptor.

    For an unprivileged listener the kernel reports the pid only for events
    caused by the listener's own process (0 otherwise), which is exactly the
    own-vs-foreign distinction needed here.
    """
    events: list[_CameraEvent] = []
    offset = 0
    while offset + _FAN_EVENT.size <= len(data):
        length, version, _reserved, _meta, mask, _fd, pid = _FAN_EVENT.unpack_from(data, offset)
        if length < _FAN_EVENT.size:
            break  # malformed: never loop forever
        offset += length
        # FAN_REPORT_FID groups never receive file descriptors (fd == FAN_NOFD).
        # Quick open/close pairs of one process may arrive merged in one event.
        if version != _FANOTIFY_METADATA_VERSION or mask & _FAN_Q_OVERFLOW:
            events.append(_CameraEvent(opened=False, foreign=None))
        elif mask & (_FAN_OPEN | _FAN_CLOSE_WRITE | _FAN_CLOSE_NOWRITE):
            events.append(
                _CameraEvent(
                    opened=bool(mask & _FAN_OPEN),
                    foreign=pid != own_pid,
                    wrote=bool(mask & _FAN_CLOSE_WRITE),
                )
            )
    return events


def _parse_inotify(data: bytes) -> list[_CameraEvent]:
    """Events from a ``read`` of an inotify descriptor (no process information)."""
    events: list[_CameraEvent] = []
    offset = 0
    while offset + _IN_EVENT.size <= len(data):
        _wd, mask, _cookie, name_len = _IN_EVENT.unpack_from(data, offset)
        offset += _IN_EVENT.size + name_len
        events.append(
            _CameraEvent(
                opened=bool(mask & _IN_OPEN), foreign=None, wrote=bool(mask & _IN_CLOSE_WRITE)
            )
        )
    return events


def _libc() -> Any:
    return ctypes.CDLL(None, use_errno=True)


def _os_error(what: str) -> OSError:
    err = ctypes.get_errno()
    return OSError(err, f"{what}: {os.strerror(err)}")


class _FdNotifier:
    """Reads events from a non-blocking notification descriptor."""

    kind = "notifier"

    def __init__(self, fd: int) -> None:
        self._fd = fd

    @property
    def alive(self) -> bool:
        """False once the descriptor failed and was closed (see :meth:`wait`)."""
        return self._fd >= 0

    def wait(self, timeout: float) -> list[_CameraEvent]:
        """Events that arrive within ``timeout`` seconds (possibly none).

        A failing descriptor is closed and reported once as a forced rescan;
        :attr:`alive` is then ``False`` so the watcher can fall back to polling.
        """
        if self._fd < 0:
            time.sleep(max(0.0, timeout))
            return []
        try:
            if not self._readable(timeout):
                return []
            data = os.read(self._fd, 16384)
        except BlockingIOError:
            return []
        except (OSError, ValueError) as exc:
            log.info("Camera %s stopped working (%s); polling the cameras instead", self.kind, exc)
            self.close()
            return [_CameraEvent(opened=False, foreign=None)]  # force a rescan
        return self._parse(data)

    def _readable(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the descriptor to have data.

        ``poll``, not ``select``: ``select`` cannot watch descriptors numbered
        ``FD_SETSIZE`` (1024) or higher, which a busy process may hand out.
        """
        make_poller = getattr(select, "poll", None)  # absent on Windows (tests)
        if make_poller is None:
            ready, _, _ = select.select([self._fd], [], [], max(0.0, timeout))
            return bool(ready)
        poller = make_poller()
        poller.register(self._fd, _POLLIN)
        return bool(poller.poll(max(0, math.ceil(timeout * 1000))))

    def close(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)

    def _parse(self, data: bytes) -> list[_CameraEvent]:
        raise NotImplementedError


class _FanotifyNotifier(_FdNotifier):
    """Unprivileged fanotify (Linux >= 5.13): opens with own-vs-foreign attribution."""

    kind = "fanotify"

    def __init__(self, paths: Sequence[str]) -> None:
        libc = _libc()
        init = libc.fanotify_init
        init.argtypes = [ctypes.c_uint, ctypes.c_uint]
        init.restype = ctypes.c_int
        mark = libc.fanotify_mark
        mark.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        mark.restype = ctypes.c_int
        # Unprivileged groups must report file handles instead of descriptors.
        fd = int(init(_FAN_CLOEXEC | _FAN_NONBLOCK | _FAN_REPORT_FID, os.O_RDONLY))
        if fd < 0:
            raise _os_error("fanotify_init")
        marked = 0
        for path in paths:
            mask = _FAN_OPEN | _FAN_CLOSE_WRITE | _FAN_CLOSE_NOWRITE
            if int(mark(fd, _FAN_MARK_ADD, mask, _AT_FDCWD, os.fsencode(path))) == 0:
                marked += 1
            else:
                log.debug("fanotify cannot watch %s: %s", path, _os_error("fanotify_mark"))
        if not marked:
            os.close(fd)
            raise OSError("fanotify could not watch any camera node")
        super().__init__(fd)
        self._own_pid = os.getpid()

    def _parse(self, data: bytes) -> list[_CameraEvent]:
        return _parse_fanotify(data, self._own_pid)


class _InotifyNotifier(_FdNotifier):
    """inotify: opens and closes of the nodes, without knowing who did them."""

    kind = "inotify"

    def __init__(self, paths: Sequence[str]) -> None:
        libc = _libc()
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add = libc.inotify_add_watch
        add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add.restype = ctypes.c_int
        flags = getattr(os, "O_NONBLOCK", 0o4000) | getattr(os, "O_CLOEXEC", 0o2000000)
        fd = int(init(flags))
        if fd < 0:
            raise _os_error("inotify_init1")
        watched = 0
        for path in paths:
            mask = _IN_OPEN | _IN_CLOSE_WRITE | _IN_CLOSE_NOWRITE
            if int(add(fd, os.fsencode(path), mask)) >= 0:
                watched += 1
        if not watched:
            os.close(fd)
            raise OSError("inotify could not watch any camera node")
        super().__init__(fd)

    def _parse(self, data: bytes) -> list[_CameraEvent]:
        return _parse_inotify(data)


def _open_camera_notifier(paths: Sequence[str]) -> _CameraNotifier | None:
    """The best available notifier for these device nodes, or ``None`` (poll)."""
    if not sys.platform.startswith("linux") or not paths:
        return None
    for cls in (_FanotifyNotifier, _InotifyNotifier):
        try:
            notifier = cls(paths)
        except (OSError, AttributeError, TypeError, ValueError) as exc:
            log.debug("%s unavailable for camera nodes: %s", cls.kind, exc)
            continue
        log.debug("Watching %s with %s", ", ".join(paths), cls.kind)
        return notifier
    return None


# ---------------------------------------------------------------------------
# camera watcher
# ---------------------------------------------------------------------------
class _CameraWatcher:
    """Whether another process uses - or just tried to open - a physical camera.

    :meth:`query` never runs the ``/proc`` scan on the caller's thread: a
    daemon thread keeps the answer fresh while somebody keeps asking and ends
    itself ``idle_stop_s`` after the last question. It rescans when notified of
    an open or close of a camera node (plus a slow safety rescan) or, without
    notifications (or once the notifier failed), every ``poll_s``. The first
    question after the thread (re)started briefly waits for its first scan.
    ``threaded=False`` scans inline at most every ``poll_s`` instead (tests).

    **Refused opens.** When a notifier attributes to another process the close
    of a read-write descriptor of a camera node while we hold a camera
    ourselves, that app just tried to start the camera and failed (V4L2 has one
    streamer; capture opens read-write, device listings read-only). The camera
    is then reported busy for ``grace_s`` so the controller releases it and the
    app can retry, and for as long as the app then holds it. Some apps also
    probe devices read-write, so when such a period ends without the app
    taking the camera it counts as a false alarm; from the second consecutive
    false alarm on, further attempts are ignored for a cooldown that doubles
    each time (``cooldown_s`` .. ``max_cooldown_s``), which bounds the cost of
    an app that keeps probing the cameras in the background.
    """

    poll_s = 3.0
    safety_rescan_s = 30.0
    idle_stop_s = 10.0
    debounce_s = 0.1
    grace_s = 30.0
    cooldown_s = 60.0
    max_cooldown_s = 1800.0
    #: Upper bound for one wait, so stop requests and idleness are noticed.
    max_wait_s = 1.0
    #: How long the first query after a (re)start waits for a fresh scan. A
    #: ``/proc`` scan takes milliseconds; this only bounds a pathological one.
    first_scan_wait_s = 0.5

    def __init__(
        self,
        *,
        devices: Callable[[], set[str] | None],
        scan: Callable[[set[str] | None], bool | None],
        own_use: Callable[[set[str] | None], bool],
        clock: Callable[[], float],
        threaded: bool,
        notifier_factory: Callable[[Sequence[str]], _CameraNotifier | None],
    ) -> None:
        self._devices = devices
        self._scan = scan
        self._own_use = own_use
        self._clock = clock
        self._threaded = threaded
        self._notifier_factory = notifier_factory
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        #: Set by every finished scan; replaced by a fresh one when the thread starts.
        self._scanned = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._last_query = -math.inf
        self._next_scan_at = -math.inf
        self._holders: bool | None = None
        self._contention_until: float | None = None
        self._contention_confirmed = False
        self._false_alarms = 0
        self._cooldown_until = -math.inf

    # ------------------------------------------------------------- public API
    def query(self) -> bool | None:
        """Latest answer; ``None`` when no scan could tell yet.

        The first question after the thread ended (nobody asked for
        ``idle_stop_s``, e.g. while tracking was paused or the screen locked)
        waits up to ``first_scan_wait_s`` for the new thread's first scan: the
        answer from before the pause may be hours old, and the controller asks
        exactly then to decide whether the camera may be reopened.
        """
        now = self._clock()
        started = False
        with self._lock:
            self._last_query = now
            due = now >= self._next_scan_at
            if self._threaded:
                started = self._ensure_thread_locked()
            scanned = self._scanned
        if started:
            scanned.wait(self.first_scan_wait_s)
        elif due and not self._threaded:
            self.rescan()
        with self._lock:
            return self._state_locked(self._clock())

    def rescan(self) -> None:
        """Scan ``/proc`` for other holders now (on the calling thread)."""
        self._rescan_with(self._devices())

    def handle_events(self, events: Sequence[_CameraEvent]) -> None:
        """React to opens/closes of camera nodes: rescan soon, spot refused opens."""
        if not events:
            return
        # A refused capture attempt: another process opened the node read-write
        # and closed it again (after EBUSY). An app that keeps it open is found by
        # the rescan instead; read-only opens are device listings.
        if any(event.foreign and event.wrote for event in events):
            try:
                ours = self._own_use(self._devices())
            except Exception:
                log.debug("own camera check failed", exc_info=True)
                ours = False
            if ours:
                with self._lock:
                    self._begin_contention_locked(self._clock())
        with self._lock:
            self._next_scan_at = min(self._next_scan_at, self._clock() + self.debounce_s)

    def stop(self, timeout: float = 2.0) -> None:
        """End the background thread (tests; production threads end when idle)."""
        with self._lock:
            self._stopping = True
            thread = self._thread
        self._wakeup.set()
        if thread is not None:
            thread.join(timeout)

    # -------------------------------------------------------------- internals
    def _ensure_thread_locked(self) -> bool:
        """Start the watcher thread unless it runs; ``True`` when it was started."""
        if self._thread is not None or self._stopping:
            return False
        # Nothing watched the nodes since the previous thread ended: the new one
        # must set up its notifier and rescan at once, not at the old deadline,
        # and the old answer no longer counts (``None`` until that scan is done).
        self._next_scan_at = -math.inf
        self._holders = None
        self._scanned = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(self._scanned,), name="camera-watch", daemon=True
        )
        self._thread.start()
        return True

    def _run(self, scanned: threading.Event) -> None:
        """The watcher thread; ``scanned`` is the event its first scan sets."""
        notifier: _CameraNotifier | None = None
        watched: frozenset[str] | None = None
        try:
            while True:
                now = self._clock()
                with self._lock:
                    if self._stopping or now - self._last_query > self.idle_stop_s:
                        # Cleared under the lock so a concurrent query starts a new thread.
                        if self._thread is threading.current_thread():
                            self._thread = None
                        return
                    due = self._next_scan_at
                if now >= due:
                    devices = self._devices()
                    key = frozenset(devices) if devices else None
                    if key != watched:  # first run, or cameras (un)plugged
                        if notifier is not None:
                            notifier.close()
                        watched = key
                        notifier = self._notifier_factory(sorted(key)) if key else None
                    self._rescan_with(devices)
                    interval = self.safety_rescan_s if notifier is not None else self.poll_s
                    with self._lock:
                        self._next_scan_at = self._clock() + interval
                    continue
                timeout = min(due - now, self.max_wait_s)
                if notifier is None:
                    self._wakeup.wait(timeout)
                    continue
                self.handle_events(notifier.wait(timeout))
                if not notifier.alive:
                    # Poll from now on (poll_s, not the slow safety rescan). The
                    # device set stays "watched", so a notifier that keeps failing
                    # is not re-created in a loop; the next device change or
                    # thread restart tries again.
                    notifier.close()
                    notifier = None
                    with self._lock:
                        self._next_scan_at = min(self._next_scan_at, self._clock() + self.poll_s)
        except Exception:
            log.debug("camera watcher stopped by an error", exc_info=True)
        finally:
            if notifier is not None:
                notifier.close()
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
            # A thread that died before its first scan must not keep a query
            # waiting for it. (Its own event: a successor has a new one.)
            scanned.set()

    def _rescan_with(self, devices: set[str] | None) -> None:
        try:
            holders = self._scan(devices)
        except Exception:
            log.debug("camera scan failed", exc_info=True)
            holders = None
        now = self._clock()
        with self._lock:
            self._holders = holders
            self._next_scan_at = max(self._next_scan_at, now + self.poll_s)
            if holders and self._contention_until is not None:
                self._contention_confirmed = True
            self._scanned.set()

    def _state_locked(self, now: float) -> bool | None:
        self._settle_contention_locked(now)
        if self._contention_until is not None:
            return True
        return self._holders

    def _begin_contention_locked(self, now: float) -> None:
        if now < self._cooldown_until:
            log.debug("Another app opened the camera; ignored (recent false alarms)")
            return
        if self._contention_until is None:
            log.info(
                "Another app tried to open the camera while tracking uses it; "
                "releasing it for %.0f s so that app can start",
                self.grace_s,
            )
            self._contention_confirmed = False
        self._contention_until = now + self.grace_s

    def _settle_contention_locked(self, now: float) -> None:
        until = self._contention_until
        if until is None or now < until:
            return
        self._contention_until = None
        if self._contention_confirmed or self._holders:
            self._false_alarms = 0
            return
        self._false_alarms += 1
        if self._false_alarms >= 2:
            cooldown = min(self.cooldown_s * 2 ** (self._false_alarms - 2), self.max_cooldown_s)
            self._cooldown_until = now + cooldown
            log.info(
                "No app took the camera after it was released; ignoring camera probes for %.0f s",
                cooldown,
            )
        else:
            log.info("No app took the camera after it was released; tracking resumes")


# ---------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------
class LinuxPlatform(PlatformServices):
    """Linux implementation of :class:`PlatformServices` (X11 and Wayland).

    The keyword arguments exist for tests; production code uses the defaults.
    ``background_threads=False`` keeps all work on the calling thread.
    """

    name: ClassVar[str] = "linux"

    def __init__(
        self,
        *,
        proc_root: str | os.PathLike[str] = PROC_ROOT,
        dev_root: str | os.PathLike[str] = DEV_ROOT,
        sys_root: str | os.PathLike[str] = SYS_ROOT,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        background_threads: bool = True,
        camera_notifier_factory: Callable[[Sequence[str]], _CameraNotifier | None] = (
            _open_camera_notifier
        ),
    ) -> None:
        self._proc_root = Path(proc_root)
        self._dev_root = Path(dev_root)
        self._sys_root = Path(sys_root)
        self._clock = clock
        self._sleep = sleep
        self._background = background_threads
        #: Variables set by ``prepare_process`` -> their original value (None = unset),
        #: restored for child processes (see ``_child_env``).
        self._env_overrides: dict[str, str | None] = {}
        self._xss = _XssIdle(clock)
        self._ewmh = _EwmhClient(clock=clock)
        self._dbus = _QtDBus(clock)

        self._session_lock = threading.Lock()
        self._resolved_session: str | None = None
        self._session_retry_at = -math.inf
        self._session_paths: dict[str, str] = {}  # logind session id -> D-Bus object path

        self._locked_lock = threading.Lock()
        self._locked_cache: tuple[float, bool | None] | None = None

        self._idle_lock = threading.Lock()
        self._last_input_at: float | None = None  # clock time of the last user input
        self._idle_checked_at = -math.inf
        self._idle_retry_at = -math.inf
        self._idle_ever_ok = False
        self._injected: tuple[float, float] | None = None  # our last ydotool move

        self._pointer_lock = threading.Lock()
        self._ydotool_cli: str | None = None  # "1" (usable CLI), "0" (0.1.x), "" (unknown)
        self._ydotool_probed_at = -math.inf
        self._ydotool_warned = False

        self._dpms_lock = threading.Lock()
        self._dpms_restore: str | None = None  # "xlib" / "xset": DPMS was off before
        self._dpms_watch: threading.Thread | None = None

        self._camera = _CameraWatcher(
            devices=self._physical_video_devices,
            scan=self._scan_camera_users,
            own_use=self._holds_camera_ourselves,
            clock=clock,
            threaded=background_threads,
            notifier_factory=camera_notifier_factory,
        )

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
            # A list, not just "xcb": if the xcb plugin still cannot load (another
            # xcb-util library missing) Qt falls back to native Wayland instead of
            # aborting at start-up.
            self._set_env("QT_QPA_PLATFORM", "xcb;wayland")
            log.info("Wayland session: preferring XWayland (QT_QPA_PLATFORM=xcb;wayland)")
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
            x11_dpms = False
            if x11:
                status = self._ewmh.dpms_status()  # None without python-xlib
                x11_dpms = status[0] if status is not None else shutil.which("xset") is not None
            return {
                "lock": any(_available(cmd) for cmd in _lock_commands(None)),
                "display_off": x11_dpms
                or any(_available(c) for c in _display_power_commands(env, False)),
                "wake_display": x11_dpms
                or any(_available(c) for c in _display_power_commands(env, True))
                or (x11 and xlib),
                "input_idle": (x11 and self._xss.available()) or gnome_idle,
                "key_idle": False,
                "session_locked": shutil.which("loginctl") is not None or self._proc_root.is_dir(),
                "focus": x11 and xlib,
                "cursor": (not self.is_wayland)
                or next(self._wayland_pointer_commands(0, 0), None) is not None,
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

    # -------------------------------------------------------------- lock screen
    def lock_screen(self) -> bool:
        """Lock the session with the first locker that demonstrably reacts.

        ``loginctl lock-session`` exits with 0 whether or not anything listens
        to logind's ``Lock`` signal (bare i3/sway without xss-lock or a swayidle
        ``lock`` hook). Outside desktops known to listen, it only counts once
        ``LockedHint`` is set or a locker process appeared; otherwise the other
        lockers are tried, and ``False`` tells the caller nothing locked.

        On GNOME and KDE, whose lock screens always set ``LockedHint``, every
        locker's success is checked that way: GNOME Shell ignores lock requests
        when locking is disabled by policy (``disable-lock-screen``), while
        ``loginctl`` and the D-Bus calls still report success. Only a hint that
        cannot be read at all falls back to trusting the desktop.
        """
        try:
            used = self._lock_with_tools()
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

    def lock_methods(self) -> list[str]:
        """The screen-lock tools installed here, in the order :meth:`lock_screen` tries them.

        Shown by ``doctor``. Being installed is not proof that a tool locks this
        session: ``loginctl lock-session`` exists on every systemd system, but on
        a bare i3 or sway session nothing listens to logind's ``Lock`` signal.
        Whether one works is only known when it is tried; :meth:`lock_screen`
        checks that and reports failure. Never raises.
        """
        try:
            handled = _desktop_handles_logind_lock(os.environ)
            methods: list[str] = []
            for argv in _lock_commands(None):
                if shutil.which(argv[0]) is None:
                    continue
                label = _lock_method_label(argv)
                if argv[0] == "loginctl" and not handled:
                    label += " (works only if a locker such as xss-lock listens to logind)"
                methods.append(label)
            return methods
        except Exception:
            log.debug("Listing the lock tools failed", exc_info=True)
            return []

    def _lock_with_tools(self) -> str | None:
        session = self._session_id()
        # Where LockedHint is authoritative, any locker's "success" can be checked.
        verify_all = session is not None and _locked_hint_authoritative(os.environ)
        # A desktop with a lock screen of its own: the display manager's lock
        # (dm-tool switches to the greeter) would come on top of it.
        own_locker = _desktop_handles_logind_lock(os.environ)
        unconfirmed: str | None = None  # a locker ran, but the screen did not lock in time
        for argv in _lock_commands(session):
            if shutil.which(argv[0]) is None:
                continue
            if unconfirmed is not None and self._lock_evident(session):
                return unconfirmed  # a slow locker reacted after all: do not stack another
            if unconfirmed is not None and own_locker and argv[0] == "dm-tool":
                # The desktop's locker said it locked but nothing shows it (or it
                # refused by policy): a greeter on top would make the user unlock
                # twice, or lock them again right after they unlocked.
                log.debug("Not stacking dm-tool lock on the desktop's own lock screen")
                continue
            result = self._run(argv, _ACTION_TIMEOUT_S)
            if result is None:
                continue
            name = " ".join(argv[:2])
            if result.returncode != 0:
                log.debug(
                    "%s exited with %s: %s",
                    name,
                    result.returncode,
                    (result.stderr or "").strip()[:200],
                )
                continue
            if argv[0] == "loginctl":
                confirmed = self._logind_lock_took_effect(session)
            else:
                confirmed = not verify_all or self._lock_confirmed(session, trust_unknown=True)
            if not confirmed:
                log.info(
                    "%s: the screen did not lock within %.0f s; trying other lockers",
                    name,
                    _LOCK_CONFIRM_S,
                )
                if unconfirmed is None:
                    unconfirmed = name
                continue
            return name
        # The last attempt may have been answered just after its wait.
        if unconfirmed is not None and self._lock_evident(session):
            return unconfirmed
        return None

    def _logind_lock_took_effect(self, session: str | None) -> bool:
        """Whether something handled logind's ``Lock`` signal (see :meth:`lock_screen`)."""
        env = os.environ
        if not _desktop_handles_logind_lock(env):
            return self._lock_confirmed(session, trust_unknown=False)
        if session is not None and _locked_hint_authoritative(env):
            return self._lock_confirmed(session, trust_unknown=True)
        # Cinnamon, MATE, Xfce ... listen to logind, but not all of their lock
        # screens report the hint: nothing reliable to check.
        return True

    def _lock_confirmed(self, session: str | None, *, trust_unknown: bool) -> bool:
        """Wait up to ``_LOCK_CONFIRM_S`` for the lock to show (``LockedHint`` or a locker).

        ``trust_unknown``: count it as locked when the hint could not be read
        at all (no evidence either way) and no locker process appeared.
        """
        deadline = self._clock() + _LOCK_CONFIRM_S
        hint_known = False
        while True:
            if session is not None:
                hint = self._logind_locked_hint(session)
                if hint is True:
                    return True
                hint_known = hint_known or hint is not None
            if self._locker_running():
                return True
            if self._clock() >= deadline:
                return trust_unknown and not hint_known
            self._sleep(_LOCK_CONFIRM_POLL_S)

    def _lock_evident(self, session: str | None) -> bool:
        """Whether the session shows as locked right now (locker process or hint).

        The process scan comes first: without QtDBus the hint costs a
        ``loginctl`` launch.
        """
        if self._locker_running():
            return True
        return session is not None and self._logind_locked_hint(session) is True

    # ------------------------------------------------------------- lock state
    def is_session_locked(self) -> bool | None:
        now = self._clock()
        with self._locked_lock:
            cached = self._locked_cache
            if cached is not None and now - cached[0] < _LOCKED_HINT_TTL_S:
                return cached[1]
            try:
                value = self._query_locked_state()
            except Exception:
                log.debug("lock state query failed", exc_info=True)
                value = None
            self._locked_cache = (now, value)
            return value

    def _query_locked_state(self) -> bool | None:
        """logind's ``LockedHint``, or a running locker that does not report to it."""
        session = self._session_id()
        hint = self._logind_locked_hint(session) if session is not None else None
        if hint is True:
            return True
        if hint is False and _locked_hint_authoritative(os.environ):
            return False  # GNOME / KDE always set the hint: no need to scan processes
        locker = self._locker_running()
        if locker:
            return True
        if hint is not None:
            return hint
        # logind did not answer. Without a session to ask, a locker process is the
        # only evidence there is; with one, a failed query means "unknown" rather
        # than "unlocked", so a real lock screen is never mistaken for a return.
        return locker if session is None else None

    def _logind_locked_hint(self, session: str) -> bool | None:
        answered, value = self._locked_hint_dbus(session)
        if answered:
            return value
        result = self._run(
            ["loginctl", "show-session", session, "-p", "LockedHint", "--value"],
            _QUERY_TIMEOUT_S,
        )
        if result is None or result.returncode != 0:
            return None
        return _parse_locked_hint(result.stdout)

    def _locked_hint_dbus(self, session: str) -> tuple[bool, bool | None]:
        """``(answered, LockedHint)`` from logind over D-Bus, without a process launch."""
        with self._session_lock:
            path = self._session_paths.get(session)
        if path is None:
            reply = self._dbus.call(
                system=True,
                service=_LOGIN1,
                path=_LOGIN1_PATH,
                interface=_LOGIN1_MANAGER,
                method="GetSession",
                args=(session,),
            )
            if reply is None:
                return False, None
            if not reply.ok:
                log.debug("logind GetSession(%s) failed: %s", session, reply.error)
                return True, None
            path = reply.values[0] if reply.values else None
            if not isinstance(path, str) or not path.startswith("/"):
                log.debug("Unexpected GetSession reply %r; using loginctl", reply.values)
                return False, None
            with self._session_lock:
                self._session_paths[session] = path
        reply = self._dbus.call(
            system=True,
            service=_LOGIN1,
            path=path,
            interface=_DBUS_PROPERTIES,
            method="Get",
            args=(_LOGIN1_SESSION, "LockedHint"),
        )
        if reply is None:
            return False, None
        if reply.ok:
            value = reply.values[0] if reply.values else None
            if isinstance(value, bool):
                return True, value
            log.debug("Unexpected LockedHint reply %r; using loginctl", reply.values)
            return False, None
        log.debug("LockedHint of session %s unavailable: %s", session, reply.error)
        with self._session_lock:
            self._session_paths.pop(session, None)  # the session may have ended
        return True, None

    def _locker_running(self) -> bool | None:
        """Whether a screen locker of this user's login runs; ``None`` without ``/proc``.

        Every process is looked at, not only those owned by this user: slock
        (and other lockers installed setuid root, e.g. swaylock built for
        shadow passwords, xlock) run as another user or have dropped privileges
        in a way that makes their ``/proc`` entry root's. ``comm`` is readable
        for every process, and so is ``loginuid``, which the audit subsystem
        sets at login and which no setuid changes: a locker counts as ours when
        we own it or when its login UID is ours (see :func:`_belongs_to_user`).
        (Not the audit *session* id: an app started by the systemd user manager
        has a different one than the graphical session its locker runs in.)
        Without login UIDs (no audit support) only the owner tells, as before.
        """
        root = str(self._proc_root)
        if not os.path.isdir(root):
            return None
        own_login = _login_uid(os.path.join(root, "self"))
        uid = _getuid()
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if not entry.name.isdigit():
                        continue
                    comm = _read_text(os.path.join(entry.path, "comm")).strip()
                    if comm in _LOCKER_COMMS and _belongs_to_user(entry, own_login, uid):
                        return True
        except OSError:
            return None
        return False

    def _session_id(self) -> str | None:
        """The logind session of this desktop.

        ``XDG_SESSION_ID`` is missing when the app is started by a systemd user
        unit; the user's graphical ("Display") session is used then. A failed
        lookup (logind still busy at login) is retried every 30 s.
        """
        env_id = os.environ.get("XDG_SESSION_ID", "").strip()
        if env_id:
            return env_id if _SESSION_ID_RE.fullmatch(env_id) else None
        with self._session_lock:
            if self._resolved_session:
                return self._resolved_session
            now = self._clock()
            if now < self._session_retry_at:
                return None
            found = self._lookup_display_session()
            if found:
                self._resolved_session = found
                return found
            self._session_retry_at = now + _SESSION_RETRY_S
            return None

    def _lookup_display_session(self) -> str:
        uid = _getuid()
        if uid is None:
            return ""
        result = self._run(
            ["loginctl", "show-user", str(uid), "-p", "Display", "--value"],
            _QUERY_TIMEOUT_S,
        )
        if result is None or result.returncode != 0:
            return ""
        lines = result.stdout.strip().splitlines()
        candidate = lines[0].strip() if lines else ""
        if "=" in candidate:
            candidate = candidate.split("=", 1)[1].strip()
        return candidate if _SESSION_ID_RE.fullmatch(candidate) else ""

    # ----------------------------------------------------------- display power
    def display_off(self) -> bool:
        try:
            if self._x11_session() and self._x11_dpms(off=True):
                log.debug("Displays turned off via X11 DPMS")
                return True
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
            if self._x11_session() and self._x11_dpms(off=False):
                return True
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

    def _x11_dpms(self, off: bool) -> bool:
        """Monitor power through the X server's DPMS extension.

        Forcing a power level requires DPMS to be enabled. A user who disabled it
        (``xset -dpms``: kiosks, presentations) gets it disabled again once the
        monitors are back on, and a server without DPMS (Xvnc, xrdp, many VMs)
        reports failure instead of pretending, so the caller can tell the user.
        """
        status = self._ewmh.dpms_status()
        if status is None:
            return self._xset_dpms(off)
        capable, enabled = status
        if not capable:
            log.debug("The X server has no usable DPMS")
            return False
        start_watch = False
        with self._dpms_lock:
            if not off:
                ok = self._ewmh.dpms_force(off=False)
                self._restore_dpms_locked()
                return ok
            if not enabled:
                if not self._ewmh.dpms_set_enabled(True):
                    return False
                self._dpms_restore = "xlib"
            ok = self._ewmh.dpms_force(off=True)
            if not ok:
                self._restore_dpms_locked()
            start_watch = ok and self._dpms_restore == "xlib"
        if start_watch and self._background:
            self._start_dpms_watch()
        return ok

    def _xset_dpms(self, off: bool) -> bool:
        """``xset`` fallback when python-xlib is unavailable."""
        if shutil.which("xset") is None:
            return False
        query = self._run(["xset", "q"], _QUERY_TIMEOUT_S)
        state = _parse_xset_dpms(query.stdout) if query and query.returncode == 0 else None
        if state is not None and not state[0]:
            # xset would print a complaint and still exit with 0.
            log.debug("The X server has no usable DPMS")
            return False
        with self._dpms_lock:
            if off and state is not None and not state[1]:
                self._dpms_restore = "xset"  # "xset dpms force" enables DPMS as a side effect
            result = self._run(["xset", "dpms", "force", "off" if off else "on"], _ACTION_TIMEOUT_S)
            ok = result is not None and result.returncode == 0
            if not off or not ok:
                self._restore_dpms_locked()
        return ok

    def _restore_dpms_locked(self) -> None:
        mode, self._dpms_restore = self._dpms_restore, None
        if mode == "xlib":
            self._ewmh.dpms_set_enabled(False)
        elif mode == "xset":
            self._run(["xset", "-dpms"], _ACTION_TIMEOUT_S)

    def _restore_dpms_if_awake(self) -> bool:
        """Disable DPMS again once the monitors are on; ``True`` when nothing is left to do."""
        with self._dpms_lock:
            if self._dpms_restore != "xlib":
                return True
            if self._ewmh.dpms_monitors_on() is not True:
                return False
            log.debug("Monitors woke up; disabling DPMS again as the user had it")
            self._restore_dpms_locked()
            return True

    def _start_dpms_watch(self) -> None:
        # Input wakes the monitors without us (e.g. wake_on_return is off), and
        # DPMS must not stay enabled with the server's blanking timeouts then.
        with self._dpms_lock:
            if self._dpms_watch is not None and self._dpms_watch.is_alive():
                return
            self._dpms_watch = threading.Thread(
                target=self._watch_dpms_wake, name="dpms-restore", daemon=True
            )
            self._dpms_watch.start()

    def _watch_dpms_wake(self) -> None:
        deadline = time.monotonic() + _DPMS_WATCH_MAX_S
        while time.monotonic() < deadline:
            time.sleep(_DPMS_WATCH_INTERVAL_S)
            try:
                if self._restore_dpms_if_awake():
                    return
            except Exception:
                log.debug("DPMS watch failed", exc_info=True)
                return

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

        Results are cached for a short while and extrapolated: returning a stale
        value unchanged would move the implied "last input" timestamp forward and
        look like fresh (keyboard) activity. An idle reset caused by our own
        ``ydotool`` motion is not user input and keeps the previous timestamp.
        """
        now = self._clock()
        with self._idle_lock:
            last = self._last_input_at
            if last is not None and now - self._idle_checked_at < _IDLE_TTL_S:
                return max(0.0, now - last)
            if now < self._idle_retry_at:
                return None
            ms, missing = self._query_mutter_idle_ms()
            if ms is None:
                self._last_input_at = None
                # Back off for long only where the service does not exist (KDE,
                # sway ...); a hiccup of a working gnome-shell is retried soon.
                permanent = missing or not self._idle_ever_ok
                delay = _RETRY_AFTER_FAILURE_S if permanent else _IDLE_TRANSIENT_RETRY_S
                self._idle_retry_at = now + delay
                log.debug("Mutter idle monitor unavailable; retrying in %.0f s", delay)
                return None
            self._idle_ever_ok = True
            self._idle_checked_at = now
            observed = now - ms / 1000.0
            current = last if last is not None and self._injected_input_at(observed) else observed
            self._last_input_at = current
            return max(0.0, now - current)

    def _injected_input_at(self, when: float) -> bool:
        injected = self._injected
        if injected is None:
            return False
        started, finished = injected
        return started - 0.05 <= when <= finished + _INJECTION_WINDOW_S

    def _query_mutter_idle_ms(self) -> tuple[int | None, bool]:
        """``(idle ms, service missing)``: in-process D-Bus first, ``gdbus`` otherwise."""
        reply = self._dbus.call(
            system=False,
            service=_MUTTER_IDLE_SERVICE,
            path=_MUTTER_IDLE_PATH,
            interface=_MUTTER_IDLE_SERVICE,
            method="GetIdletime",
        )
        if reply is not None and not reply.ok:
            return None, _is_missing_service(reply.error)
        if reply is not None:
            value = reply.values[0] if reply.values else None
            if isinstance(value, int) and not isinstance(value, bool):
                return max(0, value), False
            # An answer QtDBus did not convert as expected: never let that hide
            # the idle time for good - use gdbus instead.
            log.debug("Unexpected GetIdletime reply %r; using gdbus", reply.values)
        result = self._run(_MUTTER_IDLE, _IDLE_QUERY_TIMEOUT_S)
        if result is None:
            # Missing gdbus is permanent; a timeout is a busy gnome-shell.
            return None, shutil.which("gdbus") is None
        if result.returncode == 0:
            ms = _parse_gdbus_uint(result.stdout)
            if ms is not None:
                return ms, False
        return None, _is_missing_service(f"{result.stdout}\n{result.stderr}")

    # ----------------------------------------------------------------- cursor
    def cursor_position_reliable(self) -> bool:
        """``False`` on Wayland: XWayland only sees the pointer over X11 windows."""
        return not self.is_wayland

    def move_cursor(self, x: int, y: int) -> bool | None:
        """X11: ``None`` (Qt warps the pointer). Wayland: compositor IPC or ``ydotool``.

        sway and Hyprland place the pointer exactly through their IPC. ``ydotool``
        emulates absolute moves with relative motion, which pointer acceleration
        distorts unless its virtual device uses a flat profile; that cannot be
        checked from here.
        """
        if not self.is_wayland:
            return None
        x, y = int(x), int(y)
        for argv in self._wayland_pointer_commands(x, y):
            started = self._clock()
            result = self._run(argv, _QUERY_TIMEOUT_S)
            if result is None or not _pointer_command_ok(argv, result):
                log.debug("%s could not move the pointer", argv[0])
                continue
            if argv[0] == "ydotool":
                # uinput motion resets the compositor's idle timer like real input.
                with self._idle_lock:
                    self._injected = (started, self._clock())
            return True
        return False

    def _wayland_pointer_commands(self, x: int, y: int) -> Iterator[list[str]]:
        """Pointer-warp commands usable in this Wayland session, most exact first.

        A generator: ydotool is only probed when the compositor's IPC failed.
        """
        env = os.environ
        if env.get("SWAYSOCK") and shutil.which("swaymsg"):
            # "-" is the current seat (sway >= 1.5); older versions need a name.
            for seat in ("-", "seat0"):
                yield ["swaymsg", "seat", seat, "cursor", "set", str(x), str(y)]
        if env.get("HYPRLAND_INSTANCE_SIGNATURE") and shutil.which("hyprctl"):
            yield ["hyprctl", "dispatch", "movecursor", str(x), str(y)]
        if self._ydotool_usable():
            yield ["ydotool", "mousemove", "--absolute", "-x", str(x), "-y", str(y)]

    def _ydotool_usable(self) -> bool:
        """ydotool 1.x (``--absolute``) with its daemon running."""
        if shutil.which("ydotool") is None:
            return False
        with self._pointer_lock:
            now = self._clock()
            if not self._ydotool_cli and now >= self._ydotool_probed_at + _RETRY_AFTER_FAILURE_S:
                self._ydotool_probed_at = now
                result = self._run(["ydotool", "mousemove", "--help"], _QUERY_TIMEOUT_S)
                text = f"{result.stdout}\n{result.stderr}" if result is not None else ""
                if "--absolute" in text:
                    self._ydotool_cli = "1"
                elif result is not None:
                    self._ydotool_cli = "0"
            cli = self._ydotool_cli
            if cli == "0" and not self._ydotool_warned:
                self._ydotool_warned = True
                log.warning(
                    "ydotool is too old to move the pointer (0.1.x); install ydotool 1.x "
                    "and run ydotoold"
                )
        if cli != "1":
            return False
        if not _ydotool_socket_present(os.environ):
            log.debug("ydotool is installed but ydotoold is not running")
            return False
        return True

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
        """Whether another process holds a physical camera, or was refused one just now.

        Never scans on the calling thread (see :class:`_CameraWatcher`); ``None``
        until the first background scan finished.
        """
        return self._camera.query()

    def _user_processes(self) -> list[tuple[int, str]]:
        """``(pid, /proc/<pid>)`` of this user's processes.

        Other users' file descriptors are unreadable anyway; skipping them by
        owner is much cheaper than failing on each one.
        """
        uid = _getuid()
        found: list[tuple[int, str]] = []
        with os.scandir(self._proc_root) as entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                if uid is not None:
                    try:
                        if entry.stat(follow_symlinks=False).st_uid != uid:
                            continue
                    except OSError:
                        continue  # exited meanwhile
                found.append((int(entry.name), entry.path))
        return found

    def _scan_camera_users(self, devices: set[str] | None) -> bool | None:
        if not self._proc_root.is_dir():
            return None
        if devices is not None and not devices:
            return False  # no camera to compete for
        own_pid = os.getpid()
        for pid, path in self._user_processes():
            if pid != own_pid and self._holds_video_device(path, devices):
                log.debug("Camera in use by pid %s", pid)
                return True
        return False

    def _holds_camera_ourselves(self, devices: set[str] | None) -> bool:
        """Whether this process has a physical camera node open."""
        if devices is not None and not devices:
            return False
        fd_dir = self._proc_root / str(os.getpid()) / "fd"
        try:
            with os.scandir(fd_dir) as fds:
                for fd in fds:
                    try:
                        target = _readlink(fd.path)
                    except OSError:
                        continue
                    if _video_target(target, devices):
                        return True
        except OSError:
            return False
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
