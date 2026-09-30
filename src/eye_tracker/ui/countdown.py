"""Walk-away countdown toast.

When the controller sees nobody at the desk it announces the configured action
(lock, displays off, …) a few seconds in advance. The toast shows that
countdown with a shrinking ring and tells the user how to cancel it. It never
takes keyboard focus, so it cannot interrupt typing. It hides as soon as the
controller reports ``away_cancelled`` or the state leaves the tracking states.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QWidget

from ..config import Settings
from ..types import Rect, TrackingState, monitor_at
from . import util

log = logging.getLogger(__name__)

__all__ = ["CountdownToast"]

_TITLES = {
    "lock": "Locking in {n} s",
    "lock_and_display_off": "Locking in {n} s",
    "display_off": "Turning displays off in {n} s",
    "notify": "Marking you as away in {n} s",
}
_NOW_TITLES = {
    "lock": "Locking now…",
    "lock_and_display_off": "Locking now…",
    "display_off": "Turning displays off…",
    "notify": "Marking you as away…",
}
#: States in which a pending countdown can no longer complete.
_CANCEL_STATES = frozenset(
    {
        TrackingState.AWAY,
        TrackingState.PAUSED,
        TrackingState.PRIVACY,
        TrackingState.LOCKED,
        TrackingState.YIELDED,
        TrackingState.CALIBRATING,
    }
)

_TICK_MS = 250
_FADE_MS = 160


class CountdownToast(QWidget):
    """Small rounded always-on-top toast: "Locking in 8 s — move the mouse or …".

    Listens to ``away_warning(remaining_s)``, ``away_cancelled``,
    ``state_changed`` and ``settings_changed``. It keeps its own deadline, so the
    controller may send the warning once or repeat it. The text follows
    ``settings.presence.action``. For ``"none"`` nothing is shown, because
    nothing is going to happen. ``clock`` is injectable for tests.
    """

    def __init__(
        self,
        controller: Any,
        parent: QWidget | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        util.make_floating(self)
        self.setObjectName("countdownToast")
        self._controller = controller
        self._clock = clock
        self._settings: Settings = util.controller_settings(controller)
        self._deadline: float | None = None
        self._total = 1.0
        self._scale = 1.0
        self._title_font = QFont(self.font())
        self._hint_font = QFont(self.font())
        self._number_font = QFont(self.font())
        self._fonts_for_scale()

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(_FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)

        for name, slot in (
            ("away_warning", self.show_warning),
            ("away_cancelled", self.cancel),
            ("state_changed", self._on_state_changed),
            ("settings_changed", self._on_settings_changed),
        ):
            signal = getattr(controller, name, None)
            if signal is not None:
                signal.connect(slot)

    # ------------------------------------------------------------------ public API
    @property
    def action(self) -> str:
        """The presence action being counted down to (``settings.presence.action``)."""
        return self._settings.presence.action

    @property
    def active(self) -> bool:
        """Whether a countdown is running (the toast is shown)."""
        return self._deadline is not None

    def show_warning(self, remaining_s: float) -> None:
        """Start or update the countdown: the action happens in ``remaining_s``."""
        if self.action not in _TITLES:
            self.cancel()
            return
        try:
            remaining = float(remaining_s)
        except (TypeError, ValueError):
            remaining = 0.0
        if not math.isfinite(remaining):
            remaining = 0.0
        remaining = max(0.0, remaining)
        # A later deadline than the current one means a new countdown.
        if self._deadline is None or remaining > self.remaining() + 0.5:
            self._total = max(remaining, 1e-6)
        self._deadline = self._clock() + remaining
        self._refresh_accessible_text()
        if not self.isVisible():
            self._place()
            self._appear()
        self._timer.start()
        self.update()

    def cancel(self) -> None:
        """Stop the countdown and hide the toast."""
        self._deadline = None
        self._timer.stop()
        self._fade.stop()
        if self.isVisible():
            self.hide()

    def remaining(self) -> float:
        """Seconds left until the action (0 when not counting down)."""
        if self._deadline is None:
            return 0.0
        return max(0.0, self._deadline - self._clock())

    def seconds_left(self) -> int:
        """Remaining whole seconds as displayed (rounded up)."""
        return max(0, math.ceil(self.remaining() - 1e-6))

    def title_text(self) -> str:
        """E.g. ``"Locking in 8 s"`` (``"Locking now…"`` at zero)."""
        action = self.action if self.action in _TITLES else "lock"
        n = self.seconds_left()
        return _TITLES[action].format(n=n) if n > 0 else _NOW_TITLES[action]

    def hint_text(self) -> str:
        """How to cancel, which depends on whether input counts as presence."""
        if self._settings.presence.require_input_idle:
            return "Move the mouse or look at the camera to cancel"
        return "Look at the camera to cancel"

    def text(self) -> str:
        """The whole message on one line (also the accessible name)."""
        hint = self.hint_text()
        return f"{self.title_text()} — {hint[:1].lower()}{hint[1:]}"

    # ------------------------------------------------------------------ internals
    def _fonts_for_scale(self) -> None:
        base = QFont(self.font())
        size = base.pointSizeF() if base.pointSizeF() > 0 else 9.0
        self._title_font = QFont(base)
        self._title_font.setPointSizeF(size * 1.45)
        self._title_font.setWeight(QFont.Weight.DemiBold)
        self._hint_font = QFont(base)
        self._hint_font.setPointSizeF(size * 1.08)
        self._number_font = QFont(base)
        self._number_font.setPointSizeF(size * 1.6)
        self._number_font.setWeight(QFont.Weight.Bold)

    def _target_geometry(self) -> Rect | None:
        monitors = util.controller_monitors(self._controller)
        cursor = QCursor.pos()
        # Where the cursor is, is where the user last worked.
        monitor = monitor_at(monitors, cursor.x(), cursor.y()) or util.primary_monitor(monitors)
        if monitor is not None:
            return monitor.rect
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        g = screen.geometry()
        return Rect(g.x(), g.y(), g.width(), g.height())

    def _place(self) -> None:
        rect = self._target_geometry()
        screen = util.screen_for_point(*rect.center) if rect is not None else None
        if screen is not None:
            self.setScreen(screen)
        self._scale = s = util.ui_scale(screen)
        self._fonts_for_scale()
        title = QFontMetricsF(self._title_font, self)
        hint = QFontMetricsF(self._hint_font, self)
        action = self.action if self.action in _TITLES else "lock"
        widest_title = max(
            title.horizontalAdvance(_TITLES[action].format(n=max(10, math.ceil(self._total)))),
            title.horizontalAdvance(_NOW_TITLES[action]),
        )
        text_w = max(widest_title, hint.horizontalAdvance(self.hint_text()))
        width = math.ceil(22 * s + 52 * s + 18 * s + text_w + 26 * s)
        height = math.ceil(max(92 * s, title.height() + hint.height() + 44 * s))
        self.resize(width, height)
        if rect is not None:
            cx, cy = rect.center
            self.move(QPoint(round(cx - width / 2), round(cy - height / 2)))

    def _appear(self) -> None:
        self._fade.stop()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

    def _refresh_accessible_text(self) -> None:
        text = self.text()
        if self.accessibleName() != text:
            self.setAccessibleName(text)

    def _tick(self) -> None:
        if self._deadline is None or not self.isVisible():
            self._timer.stop()
            return
        self._refresh_accessible_text()
        self.update()

    def _on_state_changed(self, state: object) -> None:
        if isinstance(state, TrackingState) and state in _CANCEL_STATES:
            self.cancel()

    def _on_settings_changed(self, settings: object) -> None:
        if isinstance(settings, Settings):
            self._settings = settings
            if self.action not in _TITLES:
                self.cancel()
            elif self.isVisible():
                self._place()
                self.update()

    # --------------------------------------------------------------------- paint
    def paintEvent(self, event: QPaintEvent) -> None:
        s = self._scale
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        bounds = QRectF(self.rect())
        util.paint_surface(painter, bounds, 14 * s)

        # Countdown ring: a faint track and an accent arc that shrinks clockwise.
        d = 52 * s
        ring = QRectF(22 * s, (bounds.height() - d) / 2.0, d, d)
        pen_w = max(2.0, 4.0 * s)
        ring_inner = ring.adjusted(pen_w / 2, pen_w / 2, -pen_w / 2, -pen_w / 2)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 34), pen_w))
        painter.drawEllipse(ring_inner)
        fraction = min(1.0, self.remaining() / self._total) if self._total > 0 else 0.0
        if fraction > 0:
            arc = util.accent_gradient(ring.topLeft(), ring.bottomRight())
            pen = QPen(QBrush(arc), pen_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(ring_inner, 90 * 16, -round(fraction * 360 * 16))
        painter.setPen(util.qcolor(util.TEXT))
        painter.setFont(self._number_font)
        painter.drawText(ring, Qt.AlignmentFlag.AlignCenter, str(self.seconds_left()))

        left = ring.right() + 18 * s
        text_w = bounds.width() - left - 22 * s
        title_m = QFontMetricsF(self._title_font, self)
        hint_m = QFontMetricsF(self._hint_font, self)
        gap = 4 * s
        block = title_m.height() + gap + hint_m.height()
        top = (bounds.height() - block) / 2.0
        painter.setFont(self._title_font)
        painter.setPen(util.qcolor(util.TEXT))
        painter.drawText(
            QRectF(left, top, text_w, title_m.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self.title_text(),
        )
        painter.setFont(self._hint_font)
        painter.setPen(util.qcolor(util.TEXT_MUTED))
        painter.drawText(
            QRectF(left, top + title_m.height() + gap, text_w, hint_m.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self.hint_text(),
        )
        painter.end()
