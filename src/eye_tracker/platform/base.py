"""Operating-system integration contract.

Every method is *best effort*: implementations must never raise for an
unsupported operation. They return ``False``/``None`` instead and log at DEBUG
level, so the rest of the application can degrade gracefully (for example on a
Wayland session, where cursor warping is not allowed).

Thread-safety: all methods may be called from the Qt main thread. Methods
marked *any thread* may additionally be called from worker threads.
"""

from __future__ import annotations

import logging
import ntpath
from typing import ClassVar

from ..types import AppIdentity, Rect, WindowRef

log = logging.getLogger(__name__)

#: Process names remembered by :meth:`PlatformServices._app_process_name`.
_APP_NAME_CACHE_SIZE = 64
#: Executable extensions dropped from process names.
_EXECUTABLE_SUFFIXES = (".exe", ".app", ".bin")


def process_basename(name: str) -> str:
    """``"WindowsTerminal.exe"`` → ``"windowsterminal"``, ``"/usr/bin/kitty"`` →
    ``"kitty"``: the lower-cased file name without directory or executable extension."""
    base = ntpath.basename(name.replace("/", "\\")).strip().lower()
    for suffix in _EXECUTABLE_SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


class PlatformServices:
    """Default implementation: everything unsupported."""

    name: ClassVar[str] = "generic"

    # ---------------------------------------------------------------- lifecycle
    def prepare_process(self) -> None:
        """Called once, before the QApplication is created.

        Implementations set DPI awareness and Qt environment variables so that Qt
        global coordinates equal the native coordinates used by this class (see
        ``eye_tracker.types``).
        """

    def capabilities(self) -> dict[str, bool]:
        """Feature matrix shown by ``eye-tracker doctor`` and the settings UI.

        Keys: ``lock``, ``display_off``, ``wake_display``, ``input_idle``,
        ``key_idle``, ``session_locked``, ``focus``, ``cursor``, ``camera_in_use``,
        ``hotkeys``, ``panes`` (:meth:`window_app` and :meth:`window_client_rect`
        work, so split panes of supported terminals can be followed).
        """
        return {
            "lock": False,
            "display_off": False,
            "wake_display": False,
            "input_idle": False,
            "key_idle": False,
            "session_locked": False,
            "focus": False,
            "cursor": True,
            "camera_in_use": False,
            "hotkeys": False,
            "panes": False,
        }

    # ------------------------------------------------------------ session/power
    def lock_screen(self) -> bool:
        """Lock the session. *Any thread.*"""
        return False

    def display_off(self) -> bool:
        """Put all displays to sleep (without locking). *Any thread.*"""
        return False

    def wake_display(self) -> bool:
        """Wake the displays after ``display_off``. *Any thread.*"""
        return False

    def is_session_locked(self) -> bool | None:
        """Whether the session is locked / the lock screen is shown. *Any thread.*"""
        return None

    # -------------------------------------------------------------------- input
    def seconds_since_input(self) -> float | None:
        """Seconds since the last keyboard or mouse input system-wide. *Any thread.*"""
        return None

    def seconds_since_key_input(self) -> float | None:
        """Seconds since the last *keyboard* input, where the OS can tell. *Any thread.*

        Only a timestamp is ever read; key contents are never observed.
        """
        return None

    def cursor_position_reliable(self) -> bool:
        """Whether ``QCursor.pos()`` tracks the pointer everywhere on the desktop.

        ``False`` on Wayland: an XWayland client only sees pointer motion over X11
        windows, so the position it reports can be stale. The controller then
        stops deriving the current monitor, manual mouse use and learning labels
        from the cursor position.
        """
        return True

    def move_cursor(self, x: int, y: int) -> bool | None:
        """Move the pointer. Return ``None`` to let the caller use ``QCursor.setPos``."""
        return None

    # ------------------------------------------------------------------ windows
    def foreground_window(self) -> WindowRef | None:
        """The window that currently has keyboard focus (with its ``rect``)."""
        return None

    def window_at(self, x: int, y: int) -> WindowRef | None:
        """Top-level window under a screen point (excluding our own windows)."""
        return None

    def activate_window(self, ref: WindowRef) -> bool:
        """Bring a window to the front and give it keyboard focus (no synthetic click)."""
        return False

    def is_window_valid(self, ref: WindowRef) -> bool:
        """Window still exists, is visible, not minimised, and on the current
        virtual desktop / Space.

        Activating a window on another virtual desktop (Windows), workspace (X11)
        or Space (macOS) would make the system switch to it, so such windows are
        not valid switch targets.
        """
        return False

    def window_rect(self, ref: WindowRef) -> Rect | None:
        return None

    def window_client_rect(self, ref: WindowRef) -> Rect | None:
        """The window's content area (without frame and title bar), in the same
        global coordinates as :meth:`window_rect`. *Any thread.*

        Best effort: where the content area cannot be told apart it may be the
        frame rectangle. Split panes are mapped onto it.
        """
        return None

    def window_app(self, ref: WindowRef) -> AppIdentity | None:
        """The application the window belongs to (process name and class / bundle id).
        *Any thread.*

        Never reads the window title. ``None`` when unknown.
        """
        return None

    def _app_process_name(self, pid: int | None) -> str | None:
        """Lower-cased executable name of ``pid`` without its extension, cached per pid.

        Shared by the implementations of :meth:`window_app`. A pid whose process
        has gone is not cached; the cache is bounded.
        """
        if pid is None or pid <= 0:
            return None
        cache: dict[int, str] | None = getattr(self, "_app_names", None)
        if cache is None:
            cache = {}
            self._app_names = cache
        name = cache.get(pid)
        if name is not None:
            return name
        try:
            import psutil

            raw = str(psutil.Process(pid).name() or "")
        except Exception:
            return None
        name = process_basename(raw)
        if not name:
            return None
        if len(cache) >= _APP_NAME_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        cache[pid] = name
        return name

    def same_window(self, a: WindowRef | None, b: WindowRef | None) -> bool:
        if a is None or b is None:
            return False
        try:
            return bool(a.handle == b.handle)
        except Exception:
            return False

    # ------------------------------------------------------------------- camera
    def camera_in_use_by_other_app(self) -> bool | None:
        """Whether another process is streaming from a camera, or was just refused
        one (Linux). *Any thread.*

        Implementations may cache the answer for a few seconds (the check can
        cost milliseconds). ``None`` means unknown.
        """
        return None

    def running_process_names(self) -> set[str]:
        """Lower-cased executable names of running processes. *Any thread.*"""
        try:
            import psutil
        except Exception:
            return set()
        names: set[str] = set()
        for proc in psutil.process_iter(["name"]):
            name = proc.info.get("name")
            if name:
                names.add(str(name).lower())
        return names

    # -------------------------------------------------------------- permissions
    def permissions(self) -> dict[str, bool | None]:
        """Permission status, e.g. ``{"camera": True, "accessibility": None}``.

        ``None`` means "not applicable / unknown".
        """
        return {"camera": None, "accessibility": None}

    def request_permission(self, name: str) -> None:
        """Trigger the OS permission prompt, where one exists."""

    def open_permission_settings(self, name: str) -> bool:
        """Open the OS settings page for a permission."""
        return False

    def accessibility_status(self) -> str:
        """The Accessibility permission (macOS) in more detail than :meth:`permissions`.

        One of ``"granted"``, ``"missing"``, ``"stale"`` (granted to a previous
        build of the app, which no longer applies) or ``"unknown"`` (not
        applicable on this system, or cannot be determined).
        """
        return "unknown"

    def set_background_activity(self, active: bool) -> bool:
        """Keep timers unthrottled while tracking (macOS App Nap); a no-op elsewhere.

        ``True`` asks the system not to throttle the process, ``False`` allows it
        again. Returns whether the requested state is in effect.
        """
        return True

    # ------------------------------------------------------------------- misc
    @property
    def is_wayland(self) -> bool:
        return False
