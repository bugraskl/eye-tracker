"""Decide *when* keyboard focus should move to the split pane the user looks at.

This runs only while the monitor-level decider is content (the gaze is on the
monitor the cursor is on), for the focused terminal window on that monitor.
Panes are much smaller than monitors and the gaze estimate is the same, so
:class:`PaneDecider` is stricter than :class:`~eye_tracker.engine.decision.SwitchDecider`
and measures everything in units of the calibration's gaze error ``σ`` on each
axis (:func:`eye_tracker.gaze.store.axis_error`). Filters, in order:

1. **Fresh snapshot** - the pane layout must be at most :attr:`PaneConfig.fresh_s`
   old (``"stale"``), and there must be one (``"no_panes"``).
2. **Inside the window** - gaze outside the terminal window does nothing.
3. **Candidate pane** - the pane under the gaze (``"no_pane"`` on a border or
   the tab bar).
4. **Same pane** - the focused pane: nothing to do.
5. **Size gate** - the candidate ``P`` and the focused pane ``C`` are separated
   along one axis: x for panes side by side, y for stacked ones, the dominant
   one for diagonal neighbours. ``P`` qualifies only if its extent on that axis
   is at least ``max(precision × σ_axis, min_pane_px)`` (``"small"``). A pane
   the gaze cannot reliably be placed in is never switched to.
6. **Hysteresis** - the gaze must be at least ``hysteresis × σ_axis`` past the
   divider between ``C`` and ``P``.
7. **Dwell, by majority** - at least :attr:`PaneConfig.majority` of the samples
   of the last ``dwell_s`` must have passed 5 and 6 for ``P``, and the history
   must span the whole dwell. A few stray samples do not restart it; a long gap
   between updates does.
8. **Guards** - recent mouse use, recent typing (longer toward a *reading
   pane*, see :mod:`eye_tracker.engine.reading`), the cooldown after the last
   pane switch, and a grace after the focused pane changed without us (the
   user switched panes by hand: their choice wins for a while).

The decider is pure logic: time is injected and it has no side effects.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..engine.reading import READING_MAX_GAP_S, ReadingTracker
from ..types import Rect
from .types import Pane, PaneSnapshot

if TYPE_CHECKING:
    from ..config import PaneSettings, SwitchingSettings

log = logging.getLogger(__name__)

#: A pane switch counts as ``pending`` (the scheduler raises the frame rate)
#: only when it could fire within this many seconds (as for monitor switches).
PENDING_HORIZON_S = 0.5

#: Gaze error assumed when the calibration says nothing about it (pixels).
FALLBACK_GAZE_ERROR_PX = 150.0

#: A focus change seen within this time after we asked for it is ours.
OWN_CHANGE_S = 5.0

#: Panes overlapping by at most this many pixels on an axis are not overlapping.
_TOUCH_PX = 2

_EPS = 1e-6

#: Every value :attr:`PaneDecision.reason` can take.
PANE_REASONS = (
    "switch",
    "same",
    "dwell",
    "no_gaze",
    "no_panes",
    "stale",
    "outside",
    "no_pane",
    "small",
    "hysteresis",
    "mouse",
    "typing",
    "cooldown",
    "manual",
    "disabled",
)


@dataclass
class PaneConfig:
    """Tuning of :class:`PaneDecider`. Durations in seconds, sizes in pixels."""

    dwell_s: float = 0.6
    typing_grace_s: float = 3.0
    reading_grace_s: float = 8.0
    cooldown_s: float = 1.0
    mouse_grace_s: float = 1.5
    #: Grace after the focused pane changed without us. Set from
    #: ``panes.typing_grace_ms`` (documented there): switching panes by hand is
    #: a keyboard shortcut or a click, so it gets the grace of typing, without a
    #: setting of its own.
    manual_grace_s: float = 3.0
    #: Minimum pane extent in units of the gaze error on the separating axis.
    precision: float = 2.5
    #: Distance past the divider, in units of the gaze error.
    hysteresis: float = 0.5
    min_pane_px: float = 240.0
    #: Oldest pane snapshot that is still trusted.
    fresh_s: float = 1.5
    #: A longer pause between two updates restarts the dwell.
    max_gap_s: float = 0.75
    #: Share of the dwell window's samples that must be on the candidate.
    majority: float = 0.8

    @classmethod
    def from_settings(cls, panes: PaneSettings, switching: SwitchingSettings) -> PaneConfig:
        """From the user-facing (millisecond based) settings; the mouse grace is shared."""
        return cls(
            dwell_s=panes.dwell_ms / 1000.0,
            typing_grace_s=panes.typing_grace_ms / 1000.0,
            reading_grace_s=panes.reading_grace_ms / 1000.0,
            cooldown_s=panes.cooldown_ms / 1000.0,
            mouse_grace_s=switching.mouse_grace_ms / 1000.0,
            manual_grace_s=panes.typing_grace_ms / 1000.0,
            precision=float(panes.precision),
            hysteresis=float(panes.hysteresis),
            min_pane_px=float(panes.min_pane_px),
        )


@dataclass
class PaneDecision:
    """Outcome of one :meth:`PaneDecider.update` call."""

    #: Pane to focus *now*; ``None`` means do nothing.
    target: Pane | None
    #: Id of the pane under the gaze (may be the focused one); ``None`` if none.
    candidate: Hashable | None
    #: Dwell progress toward ``candidate`` in ``[0, 1]``.
    progress: float
    #: Why this decision was taken; one of :data:`PANE_REASONS`.
    reason: str
    #: A pane switch may fire soon; the scheduler should sample faster.
    pending: bool

    @property
    def fired(self) -> bool:
        return self.target is not None


def separation_axis(a: Rect, b: Rect) -> int:
    """The axis that separates panes ``a`` and ``b``: 0 (x, side by side) or 1 (y,
    stacked). Diagonal neighbours get the axis of the larger centre offset."""
    overlap_x = min(a.right, b.right) - max(a.x, b.x)
    overlap_y = min(a.bottom, b.bottom) - max(a.y, b.y)
    apart_x = overlap_x <= _TOUCH_PX
    apart_y = overlap_y <= _TOUCH_PX
    if apart_x and not apart_y:
        return 0
    if apart_y and not apart_x:
        return 1
    (ax, ay), (bx, by) = a.center, b.center
    return 0 if abs(bx - ax) >= abs(by - ay) else 1


def _extent(rect: Rect, axis: int) -> int:
    return rect.w if axis == 0 else rect.h


def divider_margin(gaze: tuple[float, float], current: Rect, candidate: Rect, axis: int) -> float:
    """How far ``gaze`` is past the divider between ``current`` and ``candidate``
    on ``axis``, toward ``candidate`` (negative on the ``current`` side)."""
    g = gaze[axis]
    c0, c1 = (current.x, current.right) if axis == 0 else (current.y, current.bottom)
    p0, p1 = (candidate.x, candidate.right) if axis == 0 else (candidate.y, candidate.bottom)
    if (p0 + p1) >= (c0 + c1):  # candidate after the current pane
        divider = (c1 + p0) / 2.0 if c1 <= p0 else p0
        return g - divider
    divider = (p1 + c0) / 2.0 if p1 <= c0 else p1
    return divider - g


def pane_qualifies(
    candidate: Pane, current: Pane, sigma: tuple[float, float], config: PaneConfig
) -> bool:
    """The size gate: is ``candidate`` large enough to be told apart from ``current``?"""
    axis = separation_axis(current.rect, candidate.rect)
    needed = max(config.precision * sigma[axis], config.min_pane_px)
    return _extent(candidate.rect, axis) + _EPS >= needed


def eligible_panes(
    snapshot: PaneSnapshot | None, sigma: tuple[float, float], config: PaneConfig
) -> int:
    """How many panes besides the focused one pass the size gate."""
    if snapshot is None:
        return 0
    current = snapshot.focused
    if current is None:
        return 0
    return sum(
        1
        for p in snapshot.panes
        if p.id != current.id and pane_qualifies(p, current, sigma, config)
    )


class PaneDecider:
    """Stateful filter from gaze points to pane focus changes (see the module docs).

    The caller reports the focused window's rectangle and its latest pane
    snapshot with every update; when the decider fires, the caller asks the
    pane's provider to focus it and calls :meth:`notify_switched`.
    """

    def __init__(self, config: PaneConfig | None = None) -> None:
        self._config = config if config is not None else PaneConfig()
        self._samples: deque[tuple[float, Hashable | None]] = deque()
        self._last_update: float | None = None
        self._last_switch = -math.inf
        self._last_target: Hashable | None = None
        self._manual_at = -math.inf
        self._window: Any = None
        self._focused: Hashable | None = None
        self._reading: ReadingTracker[Hashable] = ReadingTracker()

    # ---------------------------------------------------------------- config
    @property
    def config(self) -> PaneConfig:
        return self._config

    def set_config(self, config: PaneConfig) -> None:
        self._config = config

    @property
    def reading_pane(self) -> Hashable | None:
        return self._reading.reading

    @property
    def last_switch_time(self) -> float:
        return self._last_switch

    def reset(self) -> None:
        """Forget everything: dwell, reading pane, the window seen and the cooldown."""
        self._samples.clear()
        self._reading.clear()
        self._last_update = None
        self._last_switch = -math.inf
        self._last_target = None
        self._manual_at = -math.inf
        self._window = None
        self._focused = None

    def notify_switched(self, now: float, pane_id: Hashable) -> None:
        """The caller focused ``pane_id`` (arms the cooldown)."""
        self._last_switch = now
        self._last_target = pane_id
        self._samples.clear()
        self._reading.clear()

    # ---------------------------------------------------------------- update
    def update(
        self,
        now: float,
        gaze: tuple[float, float] | None,
        window: Rect | None,
        snapshot: PaneSnapshot | None,
        sigma: tuple[float, float] | None,
        last_mouse_activity: float,
        last_key_activity: float,
        enabled: bool = True,
    ) -> PaneDecision:
        """Feed one gaze sample and decide whether to focus another pane now.

        ``window`` is the focused window's rectangle and ``snapshot`` its latest
        panes (``None`` when unknown); ``sigma`` is the gaze error ``(x, y)`` in
        pixels on this monitor (``None``: :data:`FALLBACK_GAZE_ERROR_PX`).
        Activity times are ``-inf`` when there has never been any.
        """
        cfg = self._config
        if self._last_update is not None:
            gap = now - self._last_update
            if gap < 0 or (cfg.max_gap_s > 0 and gap > cfg.max_gap_s + _EPS):
                self._samples.clear()
            if gap < 0:
                self._reading.clear()
            elif gap > max(cfg.max_gap_s, READING_MAX_GAP_S) + _EPS:
                self._reading.end_look()
        self._last_update = now

        if not enabled:
            self._samples.clear()
            self._reading.clear()
            return PaneDecision(None, None, 0.0, "disabled", False)
        if gaze is None or not (math.isfinite(gaze[0]) and math.isfinite(gaze[1])):
            return self._hold(now, last_key_activity, None, "no_gaze")
        if snapshot is None or not snapshot.panes:
            return self._hold(now, last_key_activity, None, "no_panes")
        if now - snapshot.taken_at > cfg.fresh_s + _EPS:
            return self._hold(now, last_key_activity, None, "stale")
        current = snapshot.focused
        if current is None:
            return self._hold(now, last_key_activity, None, "no_panes")
        self._observe_focus(now, snapshot.window_handle, current.id)
        gx, gy = float(gaze[0]), float(gaze[1])
        if window is not None and not window.contains(gx, gy):
            return self._hold(now, last_key_activity, None, "outside")
        candidate = next((p for p in snapshot.panes if p.rect.contains(gx, gy)), None)
        if candidate is None:
            return self._hold(now, last_key_activity, None, "no_pane")
        if candidate.id == current.id:
            return self._hold(now, last_key_activity, candidate.id, "same")

        sx, sy = sigma if sigma is not None else (FALLBACK_GAZE_ERROR_PX, FALLBACK_GAZE_ERROR_PX)
        axis = separation_axis(current.rect, candidate.rect)
        s_axis = (sx, sy)[axis]
        if not pane_qualifies(candidate, current, (sx, sy), cfg):
            return self._hold(now, last_key_activity, candidate.id, "small")
        margin = divider_margin((gx, gy), current.rect, candidate.rect, axis)
        if margin + _EPS < cfg.hysteresis * s_axis:
            return self._hold(now, last_key_activity, candidate.id, "hysteresis")

        # The gaze is clearly on the candidate.
        self._record(now, candidate.id)
        self._reading.observe(candidate.id, now)
        self._update_reading(now, last_key_activity, current.id)
        share, span = self._dwell(now, candidate.id)
        progress = (
            1.0
            if cfg.dwell_s <= 0
            else min(1.0, span / cfg.dwell_s) * min(1.0, share / max(cfg.majority, _EPS))
        )
        guard_reason, guard_left = self._guard(
            now, last_mouse_activity, last_key_activity, candidate.id
        )
        if span + _EPS < cfg.dwell_s or share + _EPS < cfg.majority:
            dwell_left = max(0.0, cfg.dwell_s * (1.0 - progress))
            pending = guard_left <= dwell_left + PENDING_HORIZON_S
            return PaneDecision(None, candidate.id, progress, "dwell", pending)
        if guard_reason is not None:
            pending = guard_left <= PENDING_HORIZON_S
            return PaneDecision(None, candidate.id, 1.0, guard_reason, pending)

        # Fire; the cooldown is armed here so that a focus request that fails is
        # not repeated on every frame.
        self.notify_switched(now, candidate.id)
        log.debug("Focus pane %s of %s", candidate.id, candidate.provider)
        return PaneDecision(candidate, candidate.id, 1.0, "switch", False)

    # ------------------------------------------------------------- internals
    def _hold(
        self, now: float, last_key: float, candidate: Hashable | None, reason: str
    ) -> PaneDecision:
        """No clear look at another pane on this update."""
        self._record(now, None)
        self._reading.end_look()
        self._update_reading(now, last_key, self._focused)
        return PaneDecision(None, candidate, 0.0, reason, False)

    def _record(self, now: float, pane_id: Hashable | None) -> None:
        samples = self._samples
        samples.append((now, pane_id))
        # Keep one sample at or before the start of the dwell window, so the
        # history can be seen to span it.
        start = now - self._config.dwell_s
        while len(samples) > 1 and samples[1][0] <= start + _EPS:
            samples.popleft()

    def _dwell(self, now: float, pane_id: Hashable) -> tuple[float, float]:
        """(share of samples on ``pane_id``, time spanned by the history)."""
        samples = self._samples
        hits = sum(1 for _t, pid in samples if pid == pane_id)
        return hits / len(samples), now - samples[0][0]

    def _observe_focus(self, now: float, window: Any, focused: Hashable) -> None:
        """Notice a new window, and focus changes we did not cause."""
        if not _same(window, self._window):
            self._window = window
            self._focused = focused
            self._samples.clear()
            self._reading.clear()
            return
        if focused == self._focused:
            return
        ours = focused == self._last_target and now - self._last_switch <= OWN_CHANGE_S
        if not ours:
            log.debug("Pane focus changed by the user (to %s)", focused)
            self._manual_at = now
        self._focused = focused
        self._samples.clear()
        self._reading.clear()  # the keyboard focus moved

    def _update_reading(self, now: float, last_key: float, context: Hashable | None) -> None:
        cfg = self._config
        self._reading.update(
            now,
            last_key,
            typing_grace_s=cfg.typing_grace_s,
            reading_grace_s=cfg.reading_grace_s,
            context=context,
            active=context is not None,
        )

    def _guard(
        self, now: float, last_mouse: float, last_key: float, candidate: Hashable
    ) -> tuple[str | None, float]:
        """First blocking guard and the time until *all* guards expire."""
        cfg = self._config
        typing_grace = cfg.typing_grace_s
        if candidate == self._reading.reading:
            typing_grace = max(typing_grace, cfg.reading_grace_s)
        checks: Sequence[tuple[str, float]] = (
            ("mouse", cfg.mouse_grace_s - (now - last_mouse)),
            ("typing", typing_grace - (now - last_key)),
            ("cooldown", cfg.cooldown_s - (now - self._last_switch)),
            ("manual", cfg.manual_grace_s - (now - self._manual_at)),
        )
        reason: str | None = None
        longest = 0.0
        for name, left in checks:
            if left > _EPS:
                if reason is None:
                    reason = name
                longest = max(longest, left)
        return reason, longest


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return bool(a == b)
    except Exception:
        return False
