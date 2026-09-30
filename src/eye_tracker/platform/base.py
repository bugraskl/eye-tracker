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
from typing import ClassVar

from ..types import Rect, WindowRef

log = logging.getLogger(__name__)


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
        ``hotkeys``.
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
        """Window still exists, is visible and not minimised."""
        return False

    def window_rect(self, ref: WindowRef) -> Rect | None:
        return None

    def same_window(self, a: WindowRef | None, b: WindowRef | None) -> bool:
        if a is None or b is None:
            return False
        try:
            return bool(a.handle == b.handle)
        except Exception:
            return False

    # ------------------------------------------------------------------- camera
    def camera_in_use_by_other_app(self) -> bool | None:
        """Whether another process is currently streaming from a camera. *Any thread.*"""
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

    # ------------------------------------------------------------------- misc
    @property
    def is_wayland(self) -> bool:
        return False
