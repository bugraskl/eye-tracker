"""Gaze dot: shows where the app thinks you are looking (a testing aid).

Each monitor gets its own small, click-through, always-on-top window that only
ever moves within that monitor. The dot is shown on the monitor that contains
the gaze point and hidden on all others. Small windows keep the cost
negligible, since a full-screen translucent overlay would recomposite a whole
display on every update. Keeping each window on its own monitor also avoids
DPI changes when a window crosses between displays.

A gaze point outside every monitor (looking at the desk or a phone) is shown as
a hollow ring clamped to the edge of the nearest monitor.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from PySide6.QtCore import QObject, QPointF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QPainter,
    QPaintEvent,
    QPen,
    QRadialGradient,
    QScreen,
)
from PySide6.QtWidgets import QWidget

from ..config import Settings
from ..types import GazePoint, Monitor, TrackingState, monitor_at, nearest_monitor
from . import util

log = logging.getLogger(__name__)

__all__ = ["DOT_DIAMETER", "GazeOverlay"]

#: Dot diameter in design pixels (scaled per monitor with ``ui_scale``).
DOT_DIAMETER = 28.0
_GLOW = 2.0  # window side as a multiple of the dot diameter (room for the glow)

#: States in which the controller produces gaze estimates.
_GAZE_STATES = frozenset({TrackingState.TRACKING})


class _DotWindow(QWidget):
    """A small click-through window that draws the dot for one monitor."""

    def __init__(self, monitor: Monitor) -> None:
        super().__init__(None)
        util.make_floating(self, click_through=True)
        self.setObjectName(f"gazeDot{monitor.index}")
        self.monitor = monitor
        screen = util.screen_for_monitor(monitor)
        self.scale = util.ui_scale(screen)
        self.radius = DOT_DIAMETER * self.scale / 2.0
        side = math.ceil(DOT_DIAMETER * _GLOW * self.scale)
        self.setFixedSize(side, side)
        if screen is not None:
            self.setScreen(screen)
        self.dot = QPointF(side / 2.0, side / 2.0)
        self.off_screen = False

    def place(self, x: float, y: float, *, off_screen: bool) -> None:
        """Centre the dot on (x, y), clamped into this window's monitor."""
        rect = self.monitor.rect
        cx, cy = rect.clamp(x, y)
        side = self.width()
        # The window never leaves its monitor; near an edge the dot moves within
        # the window instead (and may be partly cut off, like the real edge).
        left = min(max(cx - side // 2, rect.x), max(rect.x, rect.right - side))
        top = min(max(cy - side // 2, rect.y), max(rect.y, rect.bottom - side))
        if (left, top) != (self.x(), self.y()):
            self.move(left, top)
        dot = QPointF(cx - left, cy - top)
        if dot != self.dot or off_screen != self.off_screen:
            self.dot = dot
            self.off_screen = off_screen
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self.radius
        c = self.dot
        if self.off_screen:
            painter.setPen(QPen(QColor(255, 255, 255, 150), max(1.5, 2.0 * self.scale)))
            painter.setBrush(util.qcolor(util.ACCENT, 70))
            painter.drawEllipse(c, r * 0.8, r * 0.8)
            painter.end()
            return
        glow = QRadialGradient(c, r * _GLOW)
        glow.setColorAt(0.0, util.qcolor(util.ACCENT, 110))
        glow.setColorAt(0.55, util.qcolor(util.ACCENT, 40))
        glow.setColorAt(1.0, util.qcolor(util.ACCENT, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(c, r * _GLOW, r * _GLOW)
        painter.setBrush(util.accent_gradient(c - QPointF(r, r), c + QPointF(r, r)))
        painter.setPen(QPen(QColor(255, 255, 255, 230), max(1.5, 2.0 * self.scale)))
        painter.drawEllipse(c, r * 0.82, r * 0.82)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 235))
        painter.drawEllipse(c, r * 0.2, r * 0.2)
        painter.end()


class GazeOverlay(QObject):
    """Shows ``controller.gaze_changed`` as a soft dot on the monitor being looked at.

    By default it follows ``settings.ui.show_gaze_overlay`` (initially and on
    ``settings_changed``); :meth:`set_enabled` switches it directly. The dot
    hides when the gaze is ``None``, when tracking stops, or when no gaze update
    arrives for ``stale_ms`` (a frozen dot would be misleading).
    """

    def __init__(
        self,
        controller: Any,
        parent: QObject | None = None,
        *,
        follow_settings: bool = True,
        stale_ms: int = 2500,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._enabled = False
        self._monitors: list[Monitor] = []
        self._windows: dict[int, _DotWindow] = {}
        self._visible: int | None = None

        self._stale = QTimer(self)
        self._stale.setSingleShot(True)
        self._stale.setInterval(max(1, int(stale_ms)))
        self._stale.timeout.connect(self._on_stale)

        for name, slot in (
            ("gaze_changed", self.set_gaze),
            ("state_changed", self._on_state_changed),
        ):
            signal = getattr(controller, name, None)
            if signal is not None:
                signal.connect(slot)
        if follow_settings:
            signal = getattr(controller, "settings_changed", None)
            if signal is not None:
                signal.connect(self._on_settings_changed)

        app = QGuiApplication.instance()
        if isinstance(app, QGuiApplication):
            app.screenAdded.connect(self._on_screen_added)
            app.screenRemoved.connect(self._on_screens_changed)
            app.primaryScreenChanged.connect(self._on_screens_changed)
            for screen in QGuiApplication.screens():
                screen.geometryChanged.connect(self._on_screens_changed)

        if follow_settings:
            self.set_enabled(util.controller_settings(controller).ui.show_gaze_overlay)

    # ------------------------------------------------------------------ public API
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def visible_monitor(self) -> int | None:
        """Index of the monitor currently showing the dot, if any."""
        return self._visible

    @property
    def windows(self) -> dict[int, QWidget]:
        """The per-monitor dot windows created so far (by monitor index)."""
        return dict(self._windows)

    def set_enabled(self, enabled: bool) -> None:
        """Turn the overlay on or off (off also destroys its windows).

        Windows are created lazily by the first gaze update, so an enabled overlay
        costs nothing while tracking is paused.
        """
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if not enabled:
            self._stale.stop()
            self._destroy_windows()
        log.debug("Gaze overlay %s", "enabled" if enabled else "disabled")

    def set_gaze(self, point: GazePoint | None) -> None:
        """Move the dot to ``point`` (global coordinates); ``None`` hides it."""
        if not self._enabled:
            return
        if point is None or not (math.isfinite(point.x) and math.isfinite(point.y)):
            self._stale.stop()
            self._show_on(None)
            return
        # Re-read the layout on every update: it is a short list copy, and the
        # controller rebuilds it (debounced) some time after a screen change.
        self.refresh_monitors()
        if not self._monitors:
            return
        monitor = monitor_at(self._monitors, point.x, point.y)
        off_screen = monitor is None
        if monitor is None:
            monitor, _distance = nearest_monitor(self._monitors, point.x, point.y)
        window = self._window_for(monitor)
        window.place(point.x, point.y, off_screen=off_screen)
        self._show_on(monitor.index)
        self._stale.start()

    def refresh_monitors(self) -> None:
        """Re-read the monitor layout from the controller; windows are rebuilt on change."""
        monitors = util.controller_monitors(self._controller)
        if monitors == self._monitors:
            return
        self._destroy_windows()
        self._monitors = monitors

    def close(self) -> None:
        """Hide and destroy every window (the overlay can be re-enabled later)."""
        self._stale.stop()
        self._destroy_windows()

    # ------------------------------------------------------------------ internals
    def _window_for(self, monitor: Monitor) -> _DotWindow:
        window = self._windows.get(monitor.index)
        if window is None or window.monitor != monitor:
            if window is not None:
                window.hide()
                window.deleteLater()
            window = _DotWindow(monitor)
            self._windows[monitor.index] = window
        return window

    def _show_on(self, index: int | None) -> None:
        for monitor_index, window in self._windows.items():
            if monitor_index == index:
                if not window.isVisible():
                    window.show()
            elif window.isVisible():
                window.hide()
        self._visible = index

    def _destroy_windows(self) -> None:
        for window in self._windows.values():
            window.hide()
            window.deleteLater()
        self._windows.clear()
        self._visible = None

    def _on_stale(self) -> None:
        self._show_on(None)

    def _on_state_changed(self, state: object) -> None:
        if isinstance(state, TrackingState) and state not in _GAZE_STATES:
            self._stale.stop()
            self._show_on(None)

    def _on_settings_changed(self, settings: object) -> None:
        if isinstance(settings, Settings):
            self.set_enabled(settings.ui.show_gaze_overlay)

    def _on_screen_added(self, screen: QScreen) -> None:
        screen.geometryChanged.connect(self._on_screens_changed)
        self._on_screens_changed()

    def _on_screens_changed(self, *_args: object) -> None:
        # A window on a removed or resized screen would be moved somewhere by the
        # windowing system; drop them all. The next gaze update recreates them for
        # the new layout (with the right screen and DPI scale).
        self._stale.stop()
        self._destroy_windows()
