"""Decide *when* the cursor should jump to the monitor the user is looking at.

The gaze estimate is noisy, the user glances around constantly and a wrong
switch is far more annoying than a late one. :class:`SwitchDecider` therefore
only fires after the gaze has passed a chain of filters, evaluated in order on
every update:

1. **Gaze available** - no gaze (no face, blink, not calibrated) resets dwell.
   So does the caller's verdict that the user looks away from every monitor
   (``looking_away``), which is reported as *off screen*.
2. **Candidate monitor** - the monitor containing the gaze point. A gaze point
   slightly outside every monitor (estimation error near the outer edges) is
   attributed to the nearest monitor; one far outside (a glance at the phone or
   the keyboard) is ignored as *off screen*.
3. **Same monitor** - nothing to do.
4. **Hysteresis** - the gaze must clearly favour the other monitor: it must be
   nearer to it than to the current monitor by a margin. For a gaze point on
   the other monitor that margin is simply how far it is past the bezel; for a
   point outside every monitor (below the seam of two side-by-side monitors,
   where the keyboard usually is) it is the difference of the two distances,
   so jitter around the seam is held there just as on screen.
5. **Dwell** - the candidate must be stable for ``dwell_s`` across consecutive
   updates. A long gap between updates restarts the dwell because nothing is
   known about where the user looked in between.
6. **Guards** - recent mouse use, recent typing and a cooldown after the last
   switch block the switch. The dwell keeps accumulating while guarded, so the
   switch happens the moment the guard expires if the user is still looking.

Reading while typing
--------------------
Copying from a document on one monitor into an editor on the other means
looking at the source while the keystrokes go to the current monitor. With the
plain typing guard, the first reading pause longer than ``typing_grace_s``
would move the cursor - and the keyboard focus - to the source, and the next
keystrokes would land there. The decider therefore recognises a *reading
monitor*: another monitor the user kept typing while (or right after) looking
at. Evidence is a look at that monitor (two or more consecutive updates with
the gaze clearly on it) followed by typing that continues at least
:data:`READING_EVIDENCE_S` after the look began and resumes no later than
:data:`READING_RETURN_S` after it ended. A single noisy gaze sample never
counts, and neither does the usual "last keystroke while the eyes already move
on" of a user who is about to work on the other monitor.

For a reading monitor the typing guard lasts ``reading_grace_s`` after the last
keystroke instead of ``typing_grace_s``: reading pauses keep the focus where the
user types, glances back at the editor do not reset anything, and a user who
really wants to work on the reading monitor gets there after a longer look
(or at once by moving the mouse). The reading monitor is forgotten when the
cursor changes monitor, when typing has paused for ``reading_grace_s`` or when
the gaze has not been on it for :data:`READING_FORGET_S`. Switches to any other
monitor are unaffected. The rule itself is :class:`~.reading.ReadingTracker`,
which the split-pane decider shares.

The decider is pure logic: time is injected and it has no side effects, which
makes it cheap to test exhaustively.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..types import Monitor
from .reading import (
    READING_EVIDENCE_S,
    READING_FORGET_S,
    READING_MAX_GAP_S,
    READING_RETURN_S,
    ReadingTracker,
)

if TYPE_CHECKING:
    from ..config import SwitchingSettings

log = logging.getLogger(__name__)

#: A switch counts as ``pending`` (so the scheduler raises the frame rate) only
#: when it could fire within this many seconds. A switch that is blocked for a
#: long time - typically the user reads one monitor while typing on the other -
#: must not keep the camera at the fast rate.
PENDING_HORIZON_S = 0.5

#: Default of :attr:`SwitchConfig.reading_grace_s`.
DEFAULT_READING_GRACE_S = 6.0

# The reading rule's constants (READING_*) live in .reading and are re-exported
# here: they describe this decider's behaviour too.
__all__ = [
    "DEFAULT_READING_GRACE_S",
    "PENDING_HORIZON_S",
    "READING_EVIDENCE_S",
    "READING_FORGET_S",
    "READING_MAX_GAP_S",
    "READING_RETURN_S",
    "REASONS",
    "Decision",
    "SwitchConfig",
    "SwitchDecider",
]

#: Absorbs float rounding in time differences (``10.3 - 10.0 < 0.3`` is possible).
_EPS = 1e-6

#: Every value :attr:`Decision.reason` can take.
REASONS = (
    "switch",
    "same",
    "dwell",
    "no_gaze",
    "off_screen",
    "mouse",
    "typing",
    "cooldown",
    "hysteresis",
    "disabled",
)


@dataclass
class SwitchConfig:
    """Tuning of :class:`SwitchDecider`. Durations in seconds, fractions of monitor size."""

    #: How long the gaze must stay on another monitor before switching (0 = immediately).
    dwell_s: float = 0.3
    #: How much nearer the gaze must be to the other monitor than to the current
    #: one, as a fraction of the other monitor's smaller side. For a gaze point
    #: on the other monitor this is how far past the bezel it must be.
    hysteresis: float = 0.06
    #: Gaze further than this fraction of the nearest monitor's diagonal outside
    #: every monitor is treated as looking away from the screens.
    off_screen_margin: float = 0.35
    #: Minimum time between two switches.
    cooldown_s: float = 0.6
    #: No switching this long after the user moved the mouse.
    mouse_grace_s: float = 1.5
    #: No switching this long after the user typed.
    typing_grace_s: float = 2.0
    #: A longer pause between two updates restarts the dwell (0 disables the check).
    max_gap_s: float = 0.75
    #: Typing grace toward a *reading monitor* (see the module documentation).
    #: A value not above ``typing_grace_s`` turns the reading rule off.
    reading_grace_s: float = DEFAULT_READING_GRACE_S

    @classmethod
    def from_settings(cls, s: SwitchingSettings) -> SwitchConfig:
        """Build the configuration from the user-facing (millisecond based) settings."""
        # ``reading_grace_ms`` is read defensively so that older settings
        # objects without the field still produce the default behaviour.
        reading_ms = getattr(s, "reading_grace_ms", DEFAULT_READING_GRACE_S * 1000.0)
        return cls(
            dwell_s=s.dwell_ms / 1000.0,
            hysteresis=float(s.hysteresis),
            off_screen_margin=float(s.off_screen_margin),
            cooldown_s=s.cooldown_ms / 1000.0,
            mouse_grace_s=s.mouse_grace_ms / 1000.0,
            typing_grace_s=s.typing_grace_ms / 1000.0,
            reading_grace_s=float(reading_ms) / 1000.0,
        )


@dataclass
class Decision:
    """Outcome of one :meth:`SwitchDecider.update` call."""

    #: Monitor index to switch to *now*; ``None`` means do nothing.
    target: int | None
    #: Monitor the gaze currently favours (may equal the current monitor);
    #: ``None`` when there is no gaze or it is off screen.
    candidate: int | None
    #: Dwell progress toward ``candidate`` in ``[0, 1]`` (1 once the dwell is complete).
    progress: float
    #: Why this decision was taken; one of :data:`REASONS`.
    reason: str
    #: A switch may fire soon; the scheduler should sample faster.
    pending: bool

    @property
    def fired(self) -> bool:
        """``True`` when the caller should switch to :attr:`target` now."""
        return self.target is not None


class SwitchDecider:
    """Stateful filter from gaze points to monitor switches.

    Monitors are identified by :attr:`Monitor.index`. The caller reports the
    monitor the cursor is on (``current_monitor``); when the decider fires, the
    caller performs the switch and calls :meth:`notify_switched`.
    """

    def __init__(self, monitors: Sequence[Monitor], config: SwitchConfig) -> None:
        self._config = config
        self._monitors: tuple[Monitor, ...] = ()
        self._by_index: dict[int, Monitor] = {}
        self._dwell_candidate: int | None = None
        self._dwell_start = 0.0
        self._last_update: float | None = None
        self._last_switch = -math.inf
        self._last_target: int | None = None
        # Reading-while-typing state; it belongs to the monitor the cursor was
        # on (``_context``) and is dropped whenever that changes.
        self._context: int | None = None
        self._reading: ReadingTracker[int] = ReadingTracker()
        self.set_monitors(monitors)

    # ---------------------------------------------------------------- config
    @property
    def config(self) -> SwitchConfig:
        return self._config

    @property
    def monitors(self) -> tuple[Monitor, ...]:
        return self._monitors

    @property
    def last_switch_time(self) -> float:
        """Time of the most recent switch (``-inf`` if none yet)."""
        return self._last_switch

    @property
    def last_switch_target(self) -> int | None:
        """Monitor index of the most recent switch."""
        return self._last_target

    @property
    def reading_monitor(self) -> int | None:
        """The monitor currently treated as the user's reading monitor, if any."""
        return self._reading.reading

    def set_monitors(self, monitors: Sequence[Monitor]) -> None:
        """Replace the monitor layout. A dwell in progress is discarded."""
        self._monitors = tuple(monitors)
        self._by_index = {m.index: m for m in self._monitors}
        if len(self._by_index) != len(self._monitors):
            log.warning("Duplicate monitor indices in layout: %s", [m.index for m in monitors])
        self._clear_dwell()
        self._clear_reading()

    def set_config(self, config: SwitchConfig) -> None:
        """Change the tuning; takes effect on the next update."""
        self._config = config

    def reset(self) -> None:
        """Forget all transient state, including the cooldown."""
        self._clear_dwell()
        self._clear_reading()
        self._context = None
        self._last_update = None
        self._last_switch = -math.inf
        self._last_target = None

    # ---------------------------------------------------------------- update
    def update(
        self,
        now: float,
        gaze: tuple[float, float] | None,
        current_monitor: int | None,
        last_mouse_activity: float,
        last_key_activity: float,
        enabled: bool = True,
        *,
        looking_away: bool = False,
    ) -> Decision:
        """Feed one gaze sample and decide whether to switch now.

        ``current_monitor`` is the monitor the cursor is on (``None`` when it is
        on no monitor; then any candidate is a switch target). Activity times are
        ``-inf`` when there has never been any activity. ``looking_away`` is the
        caller's verdict that the user looks away from every monitor (e.g. the
        gaze model extrapolates far beyond its calibration); it overrides
        ``gaze`` and is reported as ``"off_screen"``.
        """
        cfg = self._config
        if self._last_update is not None:
            gap = now - self._last_update
            # A clock going backwards is treated like a gap: the dwell history is
            # no longer trustworthy.
            if gap < 0 or (cfg.max_gap_s > 0 and gap > cfg.max_gap_s + _EPS):
                self._clear_dwell()
            if gap < 0:
                self._clear_reading()
            elif gap > max(cfg.max_gap_s, READING_MAX_GAP_S) + _EPS:
                self._reading.end_look()
        self._last_update = now

        if not enabled or not self._monitors:
            self._clear_dwell()
            self._clear_reading()
            return Decision(None, None, 0.0, "disabled", False)

        if current_monitor != self._context:
            # The keyboard focus most likely moved with the cursor: whatever the
            # user was reading beside the old monitor says nothing about the new one.
            self._clear_reading()
            self._context = current_monitor

        current = self._by_index.get(current_monitor) if current_monitor is not None else None
        candidate, held = self._classify(gaze, current, looking_away)
        if candidate is not None and held is None:
            self._reading.observe(candidate.index, now)
        else:
            self._reading.end_look()
        self._reading.update(
            now,
            last_key_activity,
            typing_grace_s=cfg.typing_grace_s,
            reading_grace_s=cfg.reading_grace_s,
            context=self._context,
            active=self._context in self._by_index,
        )

        if held is not None or candidate is None:
            # (_classify always gives a reason when there is no candidate.)
            self._clear_dwell()
            index = candidate.index if candidate is not None else None
            return Decision(None, index, 0.0, held or "off_screen", False)

        if self._dwell_candidate != candidate.index:
            self._dwell_candidate = candidate.index
            self._dwell_start = now
        elapsed = now - self._dwell_start
        dwell_left = max(0.0, cfg.dwell_s - elapsed)
        if dwell_left <= _EPS:
            dwell_left = 0.0
        progress = 1.0 if cfg.dwell_s <= 0 else min(1.0, max(0.0, elapsed / cfg.dwell_s))

        guard_reason, guard_left = self._guard(
            now, last_mouse_activity, last_key_activity, candidate.index
        )

        if dwell_left > 0.0:
            # Keep sampling fast while dwelling, unless a guard will block the
            # switch well beyond the end of the dwell anyway.
            pending = guard_left <= dwell_left + PENDING_HORIZON_S
            return Decision(None, candidate.index, progress, "dwell", pending)

        if guard_reason is not None:
            pending = guard_left <= PENDING_HORIZON_S
            return Decision(None, candidate.index, 1.0, guard_reason, pending)

        # Fire. The dwell restarts and the cooldown is armed right here (not only
        # in notify_switched) so that a switch the caller could not perform, e.g.
        # cursor warping refused on Wayland, is not retried on every frame.
        self._clear_dwell()
        self._clear_reading()
        self._last_switch = now
        self._last_target = candidate.index
        if gaze is not None:  # always true here; narrows the type for the log call
            log.debug("Switch to monitor %d (gaze %.0f, %.0f)", candidate.index, *gaze)
        return Decision(candidate.index, candidate.index, 1.0, "switch", False)

    def notify_switched(self, now: float, monitor_index: int) -> None:
        """Record that the caller switched to ``monitor_index`` (arms the cooldown)."""
        self._last_switch = now
        self._last_target = monitor_index
        self._clear_dwell()
        self._clear_reading()

    # ------------------------------------------------------------- internals
    def _clear_dwell(self) -> None:
        self._dwell_candidate = None

    def _classify(
        self,
        gaze: tuple[float, float] | None,
        current: Monitor | None,
        looking_away: bool,
    ) -> tuple[Monitor | None, str | None]:
        """The candidate monitor and, unless the gaze clearly favours it, why not.

        Returns ``(candidate, None)`` when the gaze is clearly on another monitor
        and ``(candidate_or_None, reason)`` for ``"off_screen"``, ``"no_gaze"``,
        ``"same"`` and ``"hysteresis"``.
        """
        if looking_away:
            return None, "off_screen"
        if gaze is None or not (math.isfinite(gaze[0]) and math.isfinite(gaze[1])):
            return None, "no_gaze"
        gx, gy = float(gaze[0]), float(gaze[1])
        candidate = self._candidate(gx, gy, current)
        if candidate is None:
            return None, "off_screen"
        if current is None:
            return candidate, None
        if candidate.index == current.index:
            return candidate, "same"
        # How much nearer the gaze is to the candidate than to the current
        # monitor. On the candidate this is the distance past the bezel; off
        # screen it compares both distances, so a point just across the
        # bisector below or above a seam is held like one just past a bezel.
        margin = current.rect.distance_outside(gx, gy) - candidate.rect.distance_outside(gx, gy)
        needed = self._config.hysteresis * min(candidate.rect.w, candidate.rect.h)
        if margin + _EPS < needed:
            return candidate, "hysteresis"
        return candidate, None

    def _candidate(self, gx: float, gy: float, current: Monitor | None) -> Monitor | None:
        # Prefer the current monitor on ties (overlapping/mirrored displays, or a
        # point equidistant from two monitors) so ambiguity never causes a switch.
        if current is not None and current.rect.contains(gx, gy):
            return current
        for m in self._monitors:
            if m.rect.contains(gx, gy):
                return m
        best: Monitor | None = None
        best_d = math.inf
        for m in self._monitors:
            d = m.rect.distance_outside(gx, gy)
            if d < best_d or (d == best_d and current is not None and m.index == current.index):
                best, best_d = m, d
        if best is None or best_d > self._config.off_screen_margin * best.rect.diagonal:
            return None
        return best

    def _guard(
        self, now: float, last_mouse: float, last_key: float, candidate: int
    ) -> tuple[str | None, float]:
        """First blocking guard (spec order) and the time until *all* guards expire."""
        cfg = self._config
        typing_grace = cfg.typing_grace_s
        if candidate == self._reading.reading:
            typing_grace = max(typing_grace, cfg.reading_grace_s)
        checks = (
            ("mouse", cfg.mouse_grace_s - (now - last_mouse)),
            ("typing", typing_grace - (now - last_key)),
            ("cooldown", cfg.cooldown_s - (now - self._last_switch)),
        )
        reason: str | None = None
        longest = 0.0
        for name, left in checks:
            if left > _EPS:
                if reason is None:
                    reason = name
                longest = max(longest, left)
        return reason, longest

    # ------------------------------------------------------ reading monitor
    def _clear_reading(self) -> None:
        self._reading.clear()
