"""Mouse vs keyboard activity, derived without any input hooks.

Global keyboard hooks would need special permissions on every OS and look like
a key logger, so they are deliberately avoided. Instead:

* **Mouse** activity is detected by polling the cursor position. Moves the app
  makes itself (after a monitor switch) are announced through
  :meth:`InputTracker.note_programmatic_move` and are not user activity. Slow,
  precise motion (a few pixels per poll, e.g. dragging a slider) counts as soon
  as the pointer has drifted :data:`DRIFT_DISTANCE_PX` within
  :data:`DRIFT_WINDOW_S`; one-pixel jitter back and forth never does.
* **Keyboard** activity comes from the OS "seconds since last key press" where
  one exists (macOS). Elsewhere it is inferred: the OS idle timer was reset but
  the cursor did not move, so the input was most likely a key press. Only
  timestamps are ever read, never key contents.

Consequence of the inference: clicks and scrolling without moving the pointer
also count as "keyboard". Both block switching for a moment, which is the
desired effect anyway.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

log = logging.getLogger(__name__)

IdleProvider = Callable[[], float | None]

#: A cursor displacement larger than this between two polls is a mouse move.
MOVE_THRESHOLD_PX = 2.0
#: A move landing within this distance of a programmatic warp target is ours.
WARP_TOLERANCE_PX = 3.0
#: How long after a warp the arrival of the cursor is attributed to the warp.
WARP_WINDOW_S = 0.5
#: The OS idle timer must jump forward by more than this to count as new input
#: (absorbs the jitter between sampling ``now`` and sampling the idle time).
IDLE_RESET_EPS_S = 0.05
#: Pointer steps too small to be a move on their own count once the pointer has
#: drifted further than this from where the steps began...
DRIFT_DISTANCE_PX = 4.0
#: ...within this time. Measuring the net displacement (not the path length)
#: keeps jitter around one spot from ever adding up.
DRIFT_WINDOW_S = 1.0


class InputTracker:
    """Keeps timestamps of the user's last mouse, keyboard and any input.

    ``seconds_since_input`` / ``seconds_since_key_input`` are the platform's
    idle-time providers (they may return ``None`` when unsupported, and they
    must not return a stale cached value unchanged - see :meth:`poll`).
    All timestamps are in the caller's clock (``time.monotonic()``) and are
    ``-inf`` until the first activity is known.
    """

    def __init__(
        self,
        seconds_since_input: IdleProvider,
        seconds_since_key_input: IdleProvider,
    ) -> None:
        self._idle_fn = seconds_since_input
        self._key_idle_fn = seconds_since_key_input
        self._cursor: tuple[int, int] = (0, 0)
        self._has_cursor = False
        self._last_mouse = -math.inf
        self._last_key = -math.inf
        self._last_any = -math.inf
        self._manual_move = False
        self._warp: tuple[int, int] | None = None
        self._warp_time = -math.inf
        self._warp_polled = False
        # Where a run of small pointer steps began, and when (see _track_drift).
        self._drift_anchor: tuple[int, int] | None = None
        self._drift_since = -math.inf
        # Latest OS input time already attributed to some activity.
        self._os_seen: float | None = None
        self._prev_idle: float | None = None
        self._provider_errors: set[str] = set()

    # ------------------------------------------------------------ properties
    @property
    def last_mouse_activity(self) -> float:
        """Time of the last user-initiated cursor move."""
        return self._last_mouse

    @property
    def last_key_activity(self) -> float:
        """Time of the last (detected or inferred) keyboard input."""
        return self._last_key

    @property
    def last_any_activity(self) -> float:
        """Time of the last input of any kind."""
        return self._last_any

    @property
    def cursor(self) -> tuple[int, int]:
        """Cursor position seen by the latest poll (``(0, 0)`` before the first poll)."""
        return self._cursor

    @property
    def manual_move(self) -> bool:
        """``True`` if the most recent poll saw a user-initiated cursor move."""
        return self._manual_move

    def seconds_since_any(self, now: float) -> float | None:
        """Seconds since the last input of any kind, ``None`` if nothing is known yet."""
        if math.isinf(self._last_any):
            return None
        return max(0.0, now - self._last_any)

    # --------------------------------------------------------------- updates
    def note_programmatic_move(self, pos: tuple[int, int], now: float) -> None:
        """Announce that the app itself is warping the cursor to ``pos``."""
        self._warp = (int(pos[0]), int(pos[1]))
        self._warp_time = now
        self._warp_polled = False

    def poll(self, now: float, cursor: tuple[int, int]) -> None:
        """Sample the cursor position and the OS idle timers."""
        x, y = int(cursor[0]), int(cursor[1])
        had_cursor = self._has_cursor
        prev = self._cursor
        self._cursor = (x, y)
        self._has_cursor = True

        changed = had_cursor and (x, y) != prev
        self._manual_move = False
        if had_cursor and math.hypot(x - prev[0], y - prev[1]) > MOVE_THRESHOLD_PX:
            self._drift_anchor = None
            if not self._explained_by_warp(x, y, now):
                self._register_manual_move(now)
            self._warp = None
        elif changed:
            self._track_drift(prev, x, y, now)

        # A warp is given at least one poll to show up, even if polls are slow.
        if self._warp is not None:
            if self._warp_polled and now - self._warp_time > WARP_WINDOW_S:
                self._warp = None
            else:
                self._warp_polled = True

        key_idle = self._read("seconds_since_key_input", self._key_idle_fn)
        if key_idle is not None:
            self._last_key = max(self._last_key, now - key_idle)
            self._last_any = max(self._last_any, self._last_key)

        idle = self._read("seconds_since_input", self._idle_fn)
        if idle is None:
            return
        # An identical reading while time has passed is a cached/stale value; it
        # carries no information and would otherwise look like fresh input.
        stale = self._prev_idle is not None and idle == self._prev_idle
        self._prev_idle = idle
        if stale:
            if changed and self._os_seen is not None:
                # The pointer moved, so this poll did see input although the idle
                # timer repeats its value: it ticks in coarse steps (15.6 ms on
                # Windows), so two polls during continuous motion can both read 0.
                # Consume the reading, or once the pointer stops the next poll
                # would mistake this very input for a key press.
                self._os_seen = max(self._os_seen, now - idle)
            return
        input_time = now - idle
        if (
            key_idle is None
            and self._os_seen is not None
            and input_time > self._os_seen + IDLE_RESET_EPS_S
            and not changed
        ):
            # The idle timer was reset but the pointer did not move: a key press
            # (or a click/scroll in place).
            self._last_key = max(self._last_key, input_time)
        self._os_seen = input_time if self._os_seen is None else max(self._os_seen, input_time)
        self._last_any = max(self._last_any, input_time)

    # ------------------------------------------------------------- internals
    def _register_manual_move(self, now: float) -> None:
        self._manual_move = True
        self._last_mouse = now
        self._last_any = max(self._last_any, now)

    def _track_drift(self, prev: tuple[int, int], x: int, y: int, now: float) -> None:
        """Count slow pointer motion whose single steps are below the move threshold.

        Without this a slow drag (a slider, a precise selection) would register
        as neither mouse nor keyboard use and the cursor could be warped away
        in the middle of it.
        """
        warp = self._warp
        if warp is not None and math.hypot(x - warp[0], y - warp[1]) <= WARP_TOLERANCE_PX:
            self._drift_anchor = None  # our own warp settling onto its target
            return
        if self._drift_anchor is None or now - self._drift_since > DRIFT_WINDOW_S:
            self._drift_anchor = prev
            self._drift_since = now
        ax, ay = self._drift_anchor
        if math.hypot(x - ax, y - ay) > DRIFT_DISTANCE_PX:
            self._drift_anchor = None
            self._register_manual_move(now)

    def _explained_by_warp(self, x: int, y: int, now: float) -> bool:
        if self._warp is None:
            return False
        if math.hypot(x - self._warp[0], y - self._warp[1]) > WARP_TOLERANCE_PX:
            return False
        return (now - self._warp_time) <= WARP_WINDOW_S or not self._warp_polled

    def _read(self, name: str, fn: IdleProvider) -> float | None:
        try:
            value = fn()
        except Exception as exc:  # platform code must not break the tick
            if name not in self._provider_errors:
                self._provider_errors.add(name)
                log.debug("%s() failed: %s", name, exc)
            return None
        if value is None:
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(seconds) or seconds < 0.0:
            return None
        return seconds
