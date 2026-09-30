"""Privacy curtain: covers every monitor when someone looks over your shoulder.

Shown when the controller emits ``guard_changed(True)``: the shoulder guard
triggered with ``settings.privacy.guard_action == "curtain"``, or the ``"lock"``
action could not lock the screen. Hidden when the guard clears, when the camera
is switched off (the guard could never clear then), when the guard is turned
off in the settings, or when the user dismisses it with Esc or the button.

Esc only works in the window that has keyboard focus. The curtain appears in
response to the camera, not to input to this app, so Windows' foreground lock
may refuse to activate it: the curtain then asks the platform layer to activate
it (as gaze switching does for other apps' windows), and while none of its
windows has keyboard focus the text says to click Dismiss instead of promising
that Esc works; the Esc would go to the application hidden underneath.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from PySide6.QtCore import QObject, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCursor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QKeyEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QRadialGradient,
    QResizeEvent,
    QScreen,
)
from PySide6.QtWidgets import QPushButton, QWidget

from ..config import Settings
from ..platform.base import PlatformServices
from ..types import Monitor, Rect, TrackingState, WindowRef, monitor_at
from . import icons, util

log = logging.getLogger(__name__)

__all__ = ["HINT", "HINT_NO_KEYBOARD", "PrivacyCurtain"]

TITLE = "Someone is looking over your shoulder"
#: Shown while a curtain window has keyboard focus (Esc reaches it).
HINT = "Your screens stay covered until they leave. Press Esc to dismiss."
#: Shown while keyboard focus is elsewhere: Esc would go to the hidden window.
HINT_NO_KEYBOARD = "Your screens stay covered until they leave. Click Dismiss to close."
#: Activation completes asynchronously on some platforms, and a refused one
#: reports no focus change at all, so the hint is checked again after this.
FOCUS_RECHECK_MS = 300

#: States in which the camera is off, so the guard can no longer clear.
_CAMERA_OFF_STATES = frozenset(
    {TrackingState.PAUSED, TrackingState.PRIVACY, TrackingState.LOCKED, TrackingState.YIELDED}
)


class _CurtainWindow(QWidget):
    """Opaque full-monitor window with a lock, a message and a dismiss button."""

    dismiss_requested = Signal()

    def __init__(self, monitor: Monitor) -> None:
        super().__init__(None)
        util.make_floating(self, accept_focus=True, translucent=False)
        self.setObjectName(f"privacyCurtain{monitor.index}")
        self.setAccessibleName("Privacy curtain")
        #: What the window tells the user about dismissing it (see set_keyboard_ready).
        self.hint = HINT_NO_KEYBOARD
        self.setAccessibleDescription(f"{TITLE}. {self.hint}")
        self.monitor = monitor
        screen = util.screen_for_monitor(monitor)
        if screen is not None:
            self.setScreen(screen)
        self.scale = s = util.ui_scale(screen)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)

        base = QFont(self.font())
        size = base.pointSizeF() if base.pointSizeF() > 0 else 9.0
        self.title_font = QFont(base)
        self.title_font.setPointSizeF(size * 2.1)
        self.title_font.setWeight(QFont.Weight.DemiBold)
        self.hint_font = QFont(base)
        self.hint_font.setPointSizeF(size * 1.2)

        self.button = QPushButton("Dismiss  (Esc)", self)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Qt drops the rounding entirely when the radius exceeds half the height.
        radius = round(15 * s)
        self.button.setStyleSheet(
            "QPushButton {"
            f" color: {util.TEXT}; background: rgba(255,255,255,0.08);"
            f" border: 1px solid rgba(255,255,255,0.16); border-radius: {radius}px;"
            f" padding: {round(9 * s)}px {round(22 * s)}px; }}"
            "QPushButton:hover { background: rgba(255,255,255,0.14); }"
            "QPushButton:pressed { background: rgba(255,255,255,0.20); }"
        )
        self.button.clicked.connect(self.dismiss_requested.emit)
        r = monitor.rect
        self.setGeometry(QRect(r.x, r.y, r.w, r.h))

    def set_keyboard_ready(self, ready: bool) -> None:
        """Promise that Esc works only while a curtain window receives the keyboard."""
        hint = HINT if ready else HINT_NO_KEYBOARD
        if hint != self.hint:
            self.hint = hint
            self.setAccessibleDescription(f"{TITLE}. {hint}")
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            self.dismiss_requested.emit()
            return
        super().keyPressEvent(event)

    def _content_layout(self) -> tuple[QRectF, QRectF, QRectF, float]:
        """Lock badge, title and hint rects plus the button top, centred on the window."""
        s = self.scale
        w, h = float(self.width()), float(self.height())
        badge = 96 * s
        title_h = QFontMetricsF(self.title_font, self).height()
        hint_h = QFontMetricsF(self.hint_font, self).height()
        block = (
            badge + 28 * s + title_h + 10 * s + hint_h + 30 * s + self.button.sizeHint().height()
        )
        top = max(0.0, (h - block) / 2.0)
        badge_rect = QRectF((w - badge) / 2.0, top, badge, badge)
        title_rect = QRectF(0.0, badge_rect.bottom() + 28 * s, w, title_h)
        hint_rect = QRectF(0.0, title_rect.bottom() + 10 * s, w, hint_h)
        return badge_rect, title_rect, hint_rect, hint_rect.bottom() + 30 * s

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        _badge, _title, _hint, button_top = self._content_layout()
        hint = self.button.sizeHint()
        self.button.setGeometry(
            round((self.width() - hint.width()) / 2), round(button_top), hint.width(), hint.height()
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        w, h = float(self.width()), float(self.height())
        painter.fillRect(self.rect(), util.qcolor(util.CURTAIN))
        # A barely visible glow behind the message so the screen does not look dead.
        glow = QRadialGradient(QPointF(w / 2.0, h / 2.0), max(w, h) * 0.55)
        glow.setColorAt(0.0, util.qcolor(util.ACCENT, 26))
        glow.setColorAt(1.0, util.qcolor(util.ACCENT, 0))
        painter.fillRect(self.rect(), glow)

        badge, title, hint, _button_top = self._content_layout()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(util.qcolor("#FFFFFF", 14))
        painter.drawEllipse(badge)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        ring = util.accent_gradient(badge.topLeft(), badge.bottomRight())
        painter.setPen(QPen(ring, max(1.5, 2.0 * self.scale)))
        painter.drawEllipse(badge.adjusted(1, 1, -1, -1))
        lock = badge.adjusted(
            badge.width() * 0.3, badge.height() * 0.27, -badge.width() * 0.3, -badge.height() * 0.3
        )
        icons.paint_lock(painter, lock, util.qcolor(util.TEXT))

        painter.setFont(self.title_font)
        painter.setPen(util.qcolor(util.TEXT))
        painter.drawText(title, int(Qt.AlignmentFlag.AlignCenter), TITLE)
        painter.setFont(self.hint_font)
        painter.setPen(util.qcolor(util.TEXT_MUTED))
        painter.drawText(hint, int(Qt.AlignmentFlag.AlignCenter), self.hint)
        painter.end()


class PrivacyCurtain(QObject):
    """Covers every monitor while the shoulder guard is active.

    Listens to ``guard_changed(bool)``, ``state_changed`` and
    ``settings_changed``. :meth:`show_curtain` / :meth:`hide_curtain` can also
    be called directly. ``dismissed`` is emitted when the user closes it.
    """

    dismissed = Signal()

    def __init__(self, controller: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._settings: Settings = util.controller_settings(controller)
        self._windows: list[_CurtainWindow] = []
        #: The window that should get keyboard focus (until that is checked).
        self._focus_target: _CurtainWindow | None = None
        self._rebuild_pending = False
        for name, slot in (
            ("guard_changed", self._on_guard_changed),
            ("state_changed", self._on_state_changed),
            ("settings_changed", self._on_settings_changed),
        ):
            signal = getattr(controller, name, None)
            if signal is not None:
                signal.connect(slot)
        app = QGuiApplication.instance()
        if isinstance(app, QGuiApplication):
            app.screenAdded.connect(self._on_screen_added)
            app.screenRemoved.connect(self._on_screens_changed)
            app.focusWindowChanged.connect(self._update_hints)
            for screen in QGuiApplication.screens():
                screen.geometryChanged.connect(self._on_screens_changed)

    # ------------------------------------------------------------------ public API
    @property
    def is_showing(self) -> bool:
        return bool(self._windows)

    @property
    def windows(self) -> list[QWidget]:
        """The curtain windows currently shown (one per monitor)."""
        return list(self._windows)

    @property
    def has_keyboard(self) -> bool:
        """Whether a curtain window has keyboard focus (so Esc dismisses it)."""
        focus = QGuiApplication.focusWindow()
        return focus is not None and any(w.windowHandle() is focus for w in self._windows)

    def show_curtain(self) -> None:
        """Cover every monitor (idempotent)."""
        monitors = self._monitors()
        if self._windows and [w.monitor for w in self._windows] == monitors:
            for window in self._windows:
                window.raise_()
            return
        self._destroy_windows()
        for monitor in monitors:
            window = _CurtainWindow(monitor)
            window.dismiss_requested.connect(self.dismiss)
            self._windows.append(window)
            window.show()
            window.raise_()
        self._focus_window_under_cursor()
        log.info("Privacy curtain shown on %d monitor(s)", len(self._windows))

    def hide_curtain(self) -> None:
        """Remove the curtain (idempotent)."""
        if self._windows:
            log.info("Privacy curtain hidden")
        self._destroy_windows()

    def dismiss(self) -> None:
        """The user closed the curtain."""
        if not self._windows:
            return
        self.hide_curtain()
        self.dismissed.emit()

    def close(self) -> None:
        self._destroy_windows()

    # ------------------------------------------------------------------ internals
    def _monitors(self) -> list[Monitor]:
        """The controller's monitors plus any screen they do not cover yet.

        The controller rebuilds its layout a moment after a screen change; for
        privacy every screen Qt knows about right now must be covered.
        """
        monitors = util.controller_monitors(self._controller)
        covered = {m.rect for m in monitors}
        primary = QGuiApplication.primaryScreen()
        for screen in QGuiApplication.screens():
            g = screen.geometry()
            rect = Rect(g.x(), g.y(), g.width(), g.height())
            if rect in covered or rect.w <= 0 or rect.h <= 0:
                continue
            covered.add(rect)
            monitors.append(
                Monitor(
                    index=len(monitors), name=screen.name(), rect=rect, primary=screen == primary
                )
            )
        return monitors

    def _on_screens_changed(self, *_args: object) -> None:
        if self._windows and not self._rebuild_pending:
            # Deferred: Qt moves windows off a removed screen before this settles.
            self._rebuild_pending = True
            QTimer.singleShot(0, self, self._rebuild)

    def _on_screen_added(self, screen: QScreen) -> None:
        screen.geometryChanged.connect(self._on_screens_changed)
        self._on_screens_changed()

    def _rebuild(self) -> None:
        self._rebuild_pending = False
        if self._windows:
            self.show_curtain()

    def _focus_window_under_cursor(self) -> None:
        if not self._windows:
            return
        cursor = QCursor.pos()
        monitor = monitor_at([w.monitor for w in self._windows], cursor.x(), cursor.y())
        target = next((w for w in self._windows if w.monitor == monitor), self._windows[0])
        # Keyboard focus is what makes Esc work; the button works without it.
        target.activateWindow()
        target.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self._focus_target = target
        self._update_hints()
        # Activation is reported asynchronously; a refused one never is.
        QTimer.singleShot(FOCUS_RECHECK_MS, self, self._ensure_keyboard)

    def _ensure_keyboard(self) -> None:
        """If Qt's activation was refused, ask the platform layer to activate the curtain."""
        target, self._focus_target = self._focus_target, None
        if target is not None and target in self._windows and not self.has_keyboard:
            # Qt's request is a plain SetForegroundWindow on Windows, which the
            # foreground lock refuses while the user works in another app.
            self._activate_with_platform(target)
            QTimer.singleShot(FOCUS_RECHECK_MS, self, self._update_hints)
        self._update_hints()

    def _activate_with_platform(self, window: QWidget) -> None:
        """Activate ``window`` the way gaze switching activates other apps' windows."""
        platform = getattr(self._controller, "platform", None)
        if not isinstance(platform, PlatformServices):
            return
        try:
            if not platform.activate_window(WindowRef(int(window.winId()), os.getpid())):
                log.debug("The privacy curtain could not take keyboard focus")
        except Exception:
            log.debug("Activating the privacy curtain failed", exc_info=True)

    def _update_hints(self, *_args: object) -> None:
        ready = self.has_keyboard
        for window in self._windows:
            window.set_keyboard_ready(ready)

    def _destroy_windows(self) -> None:
        self._focus_target = None
        for window in self._windows:
            window.hide()
            window.deleteLater()
        self._windows.clear()

    def _on_guard_changed(self, active: bool) -> None:
        # The controller decides when a curtain is due: for the "curtain" action,
        # and as the fallback when the "lock" action could not lock the screen.
        if active:
            self.show_curtain()
        else:
            self.hide_curtain()

    def _on_state_changed(self, state: object) -> None:
        if isinstance(state, TrackingState) and state in _CAMERA_OFF_STATES:
            self.hide_curtain()

    def _on_settings_changed(self, settings: object) -> None:
        if isinstance(settings, Settings):
            self._settings = settings
            privacy = settings.privacy
            # Guard switched off, or switched to notifications only: nothing should
            # keep the screens covered any more.
            if not privacy.shoulder_guard or privacy.guard_action == "notify":
                self.hide_curtain()
