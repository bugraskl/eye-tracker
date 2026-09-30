"""Per-monitor memory of the cursor position and the focused window.

When the user looks back at a monitor they usually want to continue where they
left off there: the cursor returns to where it was and keyboard focus goes back
to the window they were using, instead of both landing in the middle of the
screen.
"""

from __future__ import annotations

import logging
import math

from ..types import Monitor, Rect, WindowRef

log = logging.getLogger(__name__)

#: Accepted values of ``mode`` in :func:`choose_cursor_target`.
CURSOR_MODES = ("last", "center", "gaze")


class WindowMemory:
    """Last cursor position and last focused window, per monitor index."""

    def __init__(self) -> None:
        self._cursor: dict[int, tuple[int, int]] = {}
        self._window: dict[int, WindowRef] = {}

    def record_cursor(self, monitor_index: int, pos: tuple[int, int]) -> None:
        """Remember where the user left the cursor on a monitor."""
        self._cursor[monitor_index] = (int(pos[0]), int(pos[1]))

    def last_cursor(self, monitor_index: int) -> tuple[int, int] | None:
        return self._cursor.get(monitor_index)

    def record_window(self, monitor_index: int, ref: WindowRef) -> None:
        """Remember the window the user last worked in on a monitor."""
        self._window[monitor_index] = ref

    def last_window(self, monitor_index: int) -> WindowRef | None:
        return self._window.get(monitor_index)

    def forget_window(self, monitor_index: int) -> None:
        """Drop the remembered window (e.g. it was closed or moved elsewhere)."""
        self._window.pop(monitor_index, None)

    def clear(self) -> None:
        """Forget everything (e.g. after the monitor layout changed)."""
        self._cursor.clear()
        self._window.clear()


def choose_cursor_target(
    mode: str,
    monitor: Monitor,
    memory: WindowMemory,
    gaze: tuple[float, float] | None,
    window_rect: Rect | None,
) -> tuple[int, int]:
    """Where to put the cursor when switching to ``monitor``.

    * ``"last"`` - the remembered position on that monitor; else the centre of
      the part of ``window_rect`` (the window about to get focus) that lies on
      the monitor; else the monitor centre.
    * ``"center"`` - the monitor centre.
    * ``"gaze"`` - the gaze point; else the monitor centre.

    The result always lies inside ``monitor.rect``. Unknown modes behave like
    ``"center"``.
    """
    rect = monitor.rect
    if mode == "last":
        pos = memory.last_cursor(monitor.index)
        if pos is not None and rect.contains(pos[0], pos[1]):
            return rect.clamp(pos[0], pos[1])
        if window_rect is not None:
            visible = _intersection(window_rect, rect)
            if visible is not None:
                return rect.clamp(*visible.center)
    elif mode == "gaze":
        if gaze is not None and math.isfinite(gaze[0]) and math.isfinite(gaze[1]):
            return rect.clamp(gaze[0], gaze[1])
    elif mode != "center":
        log.warning("Unknown cursor target mode %r; using the monitor centre", mode)
    return rect.clamp(*rect.center)


def _intersection(a: Rect, b: Rect) -> Rect | None:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.right, b.right), min(a.bottom, b.bottom)
    if x1 <= x0 or y1 <= y0:
        return None
    return Rect(x0, y0, x1 - x0, y1 - y0)
