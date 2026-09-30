"""Decide *when* the cursor should jump to the monitor the user is looking at.

The gaze estimate is noisy, the user glances around constantly and a wrong
switch is far more annoying than a late one. :class:`SwitchDecider` therefore
only fires after the gaze has passed a chain of filters, evaluated in order on
every update:

1. **Gaze available** - no gaze (no face, blink, not calibrated) resets dwell.
2. **Candidate monitor** - the monitor containing the gaze point. A gaze point
   slightly outside every monitor (estimation error near the outer edges) is
   attributed to the nearest monitor; one far outside (a glance at the phone or
   the keyboard) is ignored as *off screen*.
3. **Same monitor** - nothing to do.
4. **Hysteresis** - the gaze must be clearly inside the other monitor, not just
   hovering over the bezel, so that jitter at the boundary cannot cause flapping.
5. **Dwell** - the candidate must be stable for ``dwell_s`` across consecutive
   updates. A long gap between updates restarts the dwell because nothing is
   known about where the user looked in between.
6. **Guards** - recent mouse use, recent typing and a cooldown after the last
   switch block the switch. The dwell keeps accumulating while guarded, so the
   switch happens the moment the guard expires if the user is still looking.

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

if TYPE_CHECKING:
    from ..config import SwitchingSettings

log = logging.getLogger(__name__)

#: A switch counts as ``pending`` (so the scheduler raises the frame rate) only
#: when it could fire within this many seconds. A switch that is blocked for a
#: long time - typically the user reads one monitor while typing on the other -
#: must not keep the camera at the fast rate.
PENDING_HORIZON_S = 0.5

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
    #: How far into the other monitor the gaze must be, as a fraction of that
    #: monitor's smaller side, measured from the edge of the current monitor.
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

    @classmethod
    def from_settings(cls, s: SwitchingSettings) -> SwitchConfig:
        """Build the configuration from the user-facing (millisecond based) settings."""
        return cls(
            dwell_s=s.dwell_ms / 1000.0,
            hysteresis=float(s.hysteresis),
            off_screen_margin=float(s.off_screen_margin),
            cooldown_s=s.cooldown_ms / 1000.0,
            mouse_grace_s=s.mouse_grace_ms / 1000.0,
            typing_grace_s=s.typing_grace_ms / 1000.0,
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

    def set_monitors(self, monitors: Sequence[Monitor]) -> None:
        """Replace the monitor layout. A dwell in progress is discarded."""
        self._monitors = tuple(monitors)
        self._by_index = {m.index: m for m in self._monitors}
        if len(self._by_index) != len(self._monitors):
            log.warning("Duplicate monitor indices in layout: %s", [m.index for m in monitors])
        self._clear_dwell()

    def set_config(self, config: SwitchConfig) -> None:
        """Change the tuning; takes effect on the next update."""
        self._config = config

    def reset(self) -> None:
        """Forget all transient state, including the cooldown."""
        self._clear_dwell()
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
    ) -> Decision:
        """Feed one gaze sample and decide whether to switch now.

        ``current_monitor`` is the monitor the cursor is on (``None`` when it is
        on no monitor; then any candidate is a switch target). Activity times are
        ``-inf`` when there has never been any activity.
        """
        cfg = self._config
        if self._last_update is not None:
            gap = now - self._last_update
            # A clock going backwards is treated like a gap: the dwell history is
            # no longer trustworthy.
            if gap < 0 or (cfg.max_gap_s > 0 and gap > cfg.max_gap_s + _EPS):
                self._clear_dwell()
        self._last_update = now

        if not enabled or not self._monitors:
            self._clear_dwell()
            return Decision(None, None, 0.0, "disabled", False)

        if gaze is None or not (math.isfinite(gaze[0]) and math.isfinite(gaze[1])):
            self._clear_dwell()
            return Decision(None, None, 0.0, "no_gaze", False)
        gx, gy = float(gaze[0]), float(gaze[1])

        current = self._by_index.get(current_monitor) if current_monitor is not None else None
        candidate = self._candidate(gx, gy, current)
        if candidate is None:
            self._clear_dwell()
            return Decision(None, None, 0.0, "off_screen", False)

        if current is not None and candidate.index == current.index:
            self._clear_dwell()
            return Decision(None, candidate.index, 0.0, "same", False)

        if current is not None:
            needed = cfg.hysteresis * min(candidate.rect.w, candidate.rect.h)
            if current.rect.distance_outside(gx, gy) + _EPS < needed:
                self._clear_dwell()
                return Decision(None, candidate.index, 0.0, "hysteresis", False)

        if self._dwell_candidate != candidate.index:
            self._dwell_candidate = candidate.index
            self._dwell_start = now
        elapsed = now - self._dwell_start
        dwell_left = max(0.0, cfg.dwell_s - elapsed)
        if dwell_left <= _EPS:
            dwell_left = 0.0
        progress = 1.0 if cfg.dwell_s <= 0 else min(1.0, max(0.0, elapsed / cfg.dwell_s))

        guard_reason, guard_left = self._guard(now, last_mouse_activity, last_key_activity)

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
        self._last_switch = now
        self._last_target = candidate.index
        log.debug("Switch to monitor %d (gaze %.0f, %.0f)", candidate.index, gx, gy)
        return Decision(candidate.index, candidate.index, 1.0, "switch", False)

    def notify_switched(self, now: float, monitor_index: int) -> None:
        """Record that the caller switched to ``monitor_index`` (arms the cooldown)."""
        self._last_switch = now
        self._last_target = monitor_index
        self._clear_dwell()

    # ------------------------------------------------------------- internals
    def _clear_dwell(self) -> None:
        self._dwell_candidate = None

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

    def _guard(self, now: float, last_mouse: float, last_key: float) -> tuple[str | None, float]:
        """First blocking guard (spec order) and the time until *all* guards expire."""
        cfg = self._config
        checks = (
            ("mouse", cfg.mouse_grace_s - (now - last_mouse)),
            ("typing", cfg.typing_grace_s - (now - last_key)),
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
