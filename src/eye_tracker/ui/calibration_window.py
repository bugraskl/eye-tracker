"""Full-screen calibration: one surface per monitor, an animated dot and a quality report.

Flow (keyboard driven, mouse optional)::

    intro ──Space──▶ running ──(all dots done)──▶ fitting ──▶ result ──Enter──▶ saved
      │                │  Space pauses                          │  R retries
      └──── Esc cancels at any point ◀──────────────────────────┘

Each monitor gets its own frameless, full-screen, stay-on-top surface. The
:class:`~eye_tracker.gaze.calibration.CalibrationCollector` decides which dot is
shown; the surface of that dot's monitor draws it while the other surfaces point
towards it. Observations arrive through ``Controller.observation``.

Time comes from an injectable clock and all progress is derived from the
collector, so the whole sequence can be driven deterministically in tests by
advancing a fake clock and calling :meth:`CalibrationWindow.tick`.

When the user's face disappears for more than :data:`NO_FACE_PAUSE_S` the
sequence pauses by itself (the collector runs on a clock that stands still
while paused) and resumes as soon as the face is back, so a user who glances at
the keyboard does not lose dots.

Fitting the model (:func:`~eye_tracker.gaze.calibration.evaluate`) takes about
0.1 s, longer on slow machines. It runs on a short-lived background thread whose
result the UI tick polls, so the windows stay responsive (Esc works, the spinner
turns) meanwhile. ``fit_in_thread=False`` fits synchronously on the tick after
the "Calculating" screen was shown, which keeps tests single-threaded.
"""

from __future__ import annotations

import contextlib
import logging
import math
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass

from PySide6.QtCore import QObject, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QCloseEvent,
    QColor,
    QConicalGradient,
    QFont,
    QGuiApplication,
    QKeyEvent,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QRadialGradient,
    QResizeEvent,
    QScreen,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..gaze.calibration import (
    EVENT_FINISHED,
    EVENT_RETRY,
    EVENT_TARGET,
    PHASE_SETTLE,
    CalibrationCollector,
    CalibrationReport,
    CalibrationSample,
    CalibrationTarget,
    evaluate,
    make_plan,
)
from ..gaze.model import GazeModel
from ..gaze.store import CalibrationData
from ..types import Monitor, Observation, layout_signature
from . import util
from .util import screen_for_monitor, ui_scale

log = logging.getLogger(__name__)

__all__ = [
    "NO_FACE_PAUSE_S",
    "STATE_CLOSED",
    "STATE_ERROR",
    "STATE_FITTING",
    "STATE_IDLE",
    "STATE_INTRO",
    "STATE_RESULT",
    "STATE_RUNNING",
    "CalibrationWindow",
]

# States of the calibration window.
STATE_IDLE = "idle"  # constructed, not started
STATE_INTRO = "intro"  # instructions shown, waiting for Space
STATE_RUNNING = "running"  # dots are being shown
STATE_FITTING = "fitting"  # computing the model
STATE_RESULT = "result"  # report shown, waiting for Save / Retry / Cancel
STATE_ERROR = "error"  # not enough data (or another problem); Retry / Cancel
STATE_CLOSED = "closed"  # finished (saved or cancelled)

#: Pause automatically when no face was seen for this long.
NO_FACE_PAUSE_S = 1.5
#: Without any observation for this long the camera is reported as missing.
NO_CAMERA_S = 2.0

TICK_MS = 30  # animation / collector tick while running
IDLE_TICK_MS = 250  # intro: only the face indicator needs refreshing

# Share of the settle phase used to glide from the previous dot (same monitor)
# or to pop in (new monitor). Moving targets are easier to follow than jumps.
GLIDE_SHARE = 0.45
POP_SHARE = 0.3

_ACCENT = QColor(util.ACCENT)
_ACCENT_2 = QColor(util.ACCENT_2)
_TEXT = QColor(util.TEXT)
_MUTED = QColor(util.TEXT_MUTED)
_SUCCESS = QColor(util.SUCCESS)
_WARNING = QColor(util.WARNING)
_DANGER = QColor(util.DANGER)
_BG_INNER = QColor(22, 26, 46, 235)  # 92 % opacity
_BG_OUTER = QColor(7, 9, 16, 235)

GRADE_COLORS: dict[str, QColor] = {
    "excellent": _SUCCESS,
    "good": _ACCENT_2,
    "fair": _WARNING,
    "poor": _DANGER,
}


# ============================================================ small helpers
def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - _clamp01(t)) ** 3


def _ease_in_out(t: float) -> float:
    t = _clamp01(t)
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _ease_out_back(t: float) -> float:
    t = _clamp01(t)
    c1 = 1.70158
    return 1 + (c1 + 1) * (t - 1) ** 3 + c1 * (t - 1) ** 2


def _rgba(color: QColor, alpha: float) -> QColor:
    out = QColor(color)
    out.setAlphaF(_clamp01(alpha) * color.alphaF())
    return out


def _css(color: QColor) -> str:
    return color.name(QColor.NameFormat.HexRgb)


def _grade_color(value: float) -> QColor:
    if value >= 0.97:
        return _SUCCESS
    if value >= 0.9:
        return _ACCENT_2
    if value >= 0.75:
        return _WARNING
    return _DANGER


def _spatial_order(monitors: Sequence[Monitor]) -> list[Monitor]:
    """Monitors left to right, then top to bottom (the order the dots visit them)."""
    return sorted(monitors, key=lambda m: (m.rect.x, m.rect.y, m.index))


def _monitor_labels(monitors: Sequence[Monitor]) -> dict[int, str]:
    """Human labels numbered left to right ("Screen 1 · main")."""
    labels: dict[int, str] = {}
    for n, monitor in enumerate(_spatial_order(monitors), start=1):
        text = f"Screen {n}"
        if monitor.primary:
            text += " · main"
        labels[monitor.index] = text
    return labels


FitResult = tuple[GazeModel, CalibrationReport]


def _start_fit(
    samples: Sequence[CalibrationSample], monitors: Sequence[Monitor]
) -> Future[FitResult]:
    """Run :func:`evaluate` on a daemon thread; the returned future is polled by the UI.

    Polling (rather than a cross-thread signal) means the thread never touches a
    Qt object, so closing the calibration while it runs is always safe: the
    result is simply dropped.
    """
    future: Future[FitResult] = Future()
    samples, monitors = list(samples), list(monitors)

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(evaluate(samples, monitors))
        except BaseException as exc:  # delivered to the UI thread, never lost
            future.set_exception(exc)

    threading.Thread(target=run, name="eye-tracker-calibration-fit", daemon=True).start()
    return future


@dataclass(frozen=True)
class _DotState:
    """What the active surface draws for the current target."""

    x: float  # global coordinates of the dot
    y: float
    target_x: float  # global coordinates of the target (dot's destination)
    target_y: float
    pop: float  # 0..1 appear animation
    ring: float | None  # 0..1 settle progress (shrinking ring)
    arc: float | None  # 0..1 collect progress (filling arc)
    retry: bool
    paused: bool


# ================================================================== card
class _Bar(QWidget):
    """A thin rounded progress bar used for per-monitor accuracy."""

    def __init__(self, value: float, color: QColor, scale: float, parent: QWidget | None = None):
        super().__init__(parent)
        self._value = _clamp01(value)
        self._color = color
        self.setFixedHeight(max(4, round(6 * scale)))
        self.setMinimumWidth(round(120 * scale))

    def paintEvent(self, _event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        rect = QRectF(self.rect())
        radius = rect.height() / 2
        p.setBrush(QColor(255, 255, 255, 28))
        p.drawRoundedRect(rect, radius, radius)
        if self._value > 0:
            fill = QRectF(
                rect.x(), rect.y(), max(rect.height(), rect.width() * self._value), rect.height()
            )
            p.setBrush(self._color)
            p.drawRoundedRect(fill, radius, radius)
        p.end()


class _Spinner(QWidget):
    """A rotating arc shown while the model is being fitted."""

    def __init__(self, scale: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        size = max(16, round(34 * scale))
        self.setFixedSize(size, size)
        self._scale = scale
        self._angle = 0.0

    def set_time(self, seconds: float) -> None:
        """Rotate to the position for ``seconds`` (one turn every 0.9 s)."""
        angle = (seconds / 0.9 * 360.0) % 360.0
        if abs(angle - self._angle) > 0.5:
            self._angle = angle
            self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = max(2.0, 3 * self._scale)
        box = QRectF(self.rect()).adjusted(width, width, -width, -width)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 36), width))
        p.drawEllipse(box)
        grad = QConicalGradient(box.center(), -self._angle)
        grad.setColorAt(0.0, _ACCENT_2)
        grad.setColorAt(0.3, _rgba(_ACCENT, 0.0))
        pen = QPen(QBrush(grad), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        # Qt angles are counter-clockwise in 1/16 degree; spin clockwise.
        p.drawArc(box, round(-self._angle * 16), 110 * 16)
        p.end()


class _Card(QFrame):
    """Centred content panel (instructions, results) built from simple parts."""

    def __init__(self, parent: QWidget, scale: float) -> None:
        super().__init__(parent)
        self.setObjectName("calibrationCard")
        self._scale = scale
        s = scale
        self.setStyleSheet(
            f"""
            QFrame#calibrationCard {{
                background: rgba(255, 255, 255, 0.045);
                border: 1px solid rgba(255, 255, 255, 0.09);
                border-radius: {round(22 * s)}px;
            }}
            QLabel {{ color: {_css(_TEXT)}; background: transparent; border: none; }}
            QLabel[muted="true"] {{ color: {_css(_MUTED)}; }}
            QLabel[keycap="true"] {{
                color: {_css(_TEXT)};
                border: 1px solid rgba(255, 255, 255, 0.28);
                border-bottom-width: 2px;
                border-radius: {round(6 * s)}px;
                padding: {round(2 * s)}px {round(9 * s)}px;
                font-weight: 600;
            }}
            QPushButton {{
                color: {_css(_TEXT)};
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: {round(10 * s)}px;
                padding: {round(9 * s)}px {round(20 * s)}px;
                font-size: 11pt;
            }}
            QPushButton:hover {{ background: rgba(255, 255, 255, 0.15); }}
            QPushButton:pressed {{ background: rgba(255, 255, 255, 0.05); }}
            QPushButton[primary="true"] {{
                color: white;
                border: none;
                font-weight: 600;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {_css(_ACCENT)}, stop:1 {_css(_ACCENT_2)});
            }}
            QPushButton[primary="true"]:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7C7FF5, stop:1 #4FDDF1);
            }}
            """
        )
        self._layout = QVBoxLayout(self)
        m = round(36 * s)
        self._layout.setContentsMargins(m, round(32 * s), m, round(30 * s))
        self._layout.setSpacing(round(12 * s))
        self.buttons: dict[str, QPushButton] = {}
        self.labels: dict[str, QLabel] = {}
        self.spinner: _Spinner | None = None

    # -------------------------------------------------------------- content
    def clear(self) -> None:
        self.buttons.clear()
        self.labels.clear()
        self.spinner = None
        _delete_layout(self._layout, keep=True)

    def add_label(
        self,
        text: str,
        *,
        name: str | None = None,
        size: float = 11.0,
        bold: bool = False,
        muted: bool = False,
        color: QColor | None = None,
        align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignHCenter,
    ) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(align)
        label.setTextFormat(Qt.TextFormat.PlainText)
        font = QFont(label.font())
        font.setPointSizeF(size)
        if bold:
            font.setWeight(QFont.Weight.DemiBold)
        label.setFont(font)
        if muted:
            label.setProperty("muted", True)
        if color is not None:
            label.setStyleSheet(f"color: {_css(color)};")
        self._layout.addWidget(label)
        if name:
            self.labels[name] = label
        return label

    def add_badge(self, text: str, color: QColor) -> QLabel:
        s = self._scale
        badge = QLabel(text.upper())
        font = QFont(badge.font())
        font.setPointSizeF(9.0)
        font.setWeight(QFont.Weight.Bold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        badge.setFont(font)
        badge.setStyleSheet(
            f"color: {_css(color)}; background: rgba({color.red()}, {color.green()},"
            f" {color.blue()}, 0.16); border: 1px solid rgba({color.red()}, {color.green()},"
            f" {color.blue()}, 0.45); border-radius: {round(11 * s)}px;"
            f" padding: {round(3 * s)}px {round(12 * s)}px;"
        )
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(badge)
        row.addStretch(1)
        self._layout.addLayout(row)
        self.labels["badge"] = badge
        return badge

    def add_keys(self, items: Sequence[tuple[str, str]]) -> None:
        row = QHBoxLayout()
        row.setSpacing(round(8 * self._scale))
        row.addStretch(1)
        for n, (key, action) in enumerate(items):
            if n:
                dot = QLabel("·")
                dot.setProperty("muted", True)
                row.addSpacing(round(6 * self._scale))
                row.addWidget(dot)
                row.addSpacing(round(6 * self._scale))
            cap = QLabel(key)
            cap.setProperty("keycap", True)
            text = QLabel(action)
            text.setProperty("muted", True)
            row.addWidget(cap)
            row.addWidget(text)
        row.addStretch(1)
        self._layout.addLayout(row)

    def add_bars(self, rows: Sequence[tuple[str, float]]) -> None:
        s = self._scale
        for label_text, value in rows:
            row = QHBoxLayout()
            row.setSpacing(round(12 * s))
            label = QLabel(label_text)
            label.setMinimumWidth(round(130 * s))
            label.setProperty("muted", True)
            pct = QLabel(f"{math.floor(value * 100 + 1e-9)}%")
            pct.setMinimumWidth(round(44 * s))
            pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(label)
            row.addWidget(_Bar(value, _grade_color(value), s), 1)
            row.addWidget(pct)
            self._layout.addLayout(row)

    def add_buttons(self, buttons: Sequence[tuple[str, str, Callable[[], object], bool]]) -> None:
        """``(name, text, callback, primary)`` for each button, left to right."""
        row = QHBoxLayout()
        row.setSpacing(round(10 * self._scale))
        row.addStretch(1)
        for name, text, callback, primary in buttons:
            button = QPushButton(text)
            # Keys are handled by the surface (Enter / R / Esc); focusable buttons
            # would swallow Space and Enter.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            if primary:
                button.setProperty("primary", True)
            button.clicked.connect(lambda _=False, cb=callback: cb())
            row.addWidget(button)
            self.buttons[name] = button
        row.addStretch(1)
        self._layout.addLayout(row)

    def add_spinner(self) -> _Spinner:
        spinner = _Spinner(self._scale)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(spinner)
        row.addStretch(1)
        self._layout.addLayout(row)
        self.spinner = spinner
        return spinner

    def add_spacing(self, points: float) -> None:
        self._layout.addSpacing(round(points * self._scale))

    def preferred_height(self, width: int) -> int:
        """Height the content needs at ``width`` (all current content counted)."""
        self._show_new_children()
        layout = self._layout
        layout.invalidate()
        if layout.hasHeightForWidth():
            return int(layout.totalHeightForWidth(width))
        return int(layout.totalSizeHint().height())

    def _show_new_children(self) -> None:
        # Widgets added to an already visible card stay hidden until a queued show
        # event, and layouts ignore hidden widgets, so the card would be measured
        # without them and end up too small. Show them now. Removed widgets were
        # hidden explicitly (and are awaiting deletion), so they stay hidden.
        for child in self.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            if child.isHidden() and not child.testAttribute(
                Qt.WidgetAttribute.WA_WState_ExplicitShowHide
            ):
                child.show()


def _delete_layout(layout: QLayout, *, keep: bool = False) -> None:
    """Remove (and schedule deletion of) everything in ``layout``; ``keep`` spares it."""
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        child = item.layout()
        if widget is not None:
            widget.hide()
            widget.deleteLater()
        elif child is not None:
            _delete_layout(child)
    if not keep:
        layout.deleteLater()


# ================================================================== surface
class _Surface(QWidget):
    """One monitor's full-screen calibration window."""

    MODE_CARD = "card"
    MODE_RUNNING = "running"

    def __init__(self, owner: CalibrationWindow, monitor: Monitor, screen: QScreen | None) -> None:
        super().__init__(
            None,
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._owner = owner
        self.monitor = monitor
        self.qscreen = screen
        self.scale = ui_scale(screen) if screen is not None else 1.0
        self.setWindowTitle("Eye Tracker — calibration")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(False)
        self.mode = self.MODE_CARD
        self.active = False
        self.dot: _DotState | None = None
        self.progress = 0.0
        self.progress_text = ""
        self.hint = ""
        self.hint_color = _WARNING
        self.keys_text = ""
        self.closing = False
        self.card = _Card(self, self.scale)
        self.card.hide()

    # -------------------------------------------------------------- showing
    def present(self) -> None:
        rect = self.monitor.rect
        if self.qscreen is not None:
            self.setScreen(self.qscreen)
        self.setGeometry(QRect(rect.x, rect.y, rect.w, rect.h))
        if sys.platform == "darwin":
            # A full-screen window gets its own Space on macOS (with an animation and
            # a black backdrop on the other displays); a borderless screen-sized
            # window stays on the current Space.
            self.show()
        else:
            self.showFullScreen()

    # -------------------------------------------------------------- card mode
    def show_card(self) -> _Card:
        self.mode = self.MODE_CARD
        self.card.show()
        self.place_card()
        self.update()
        return self.card

    def place_card(self) -> None:
        s = self.scale
        width = max(round(280 * s), min(round(600 * s), self.width() - round(48 * s)))
        height = self.card.preferred_height(width)
        x = (self.width() - width) // 2
        y = max(round(24 * s), int((self.height() - height) / 2 - self.height() * 0.04))
        self.card.setGeometry(x, y, width, height)

    # -------------------------------------------------------------- running mode
    def set_running(
        self,
        dot: _DotState | None,
        *,
        active: bool,
        progress: float,
        progress_text: str,
        hint: str,
        hint_color: QColor,
        keys_text: str,
    ) -> None:
        """Update the running display, repainting only what changed."""
        full = self.mode != self.MODE_RUNNING or active != self.active
        if not active and (self.dot is None) != (dot is None):
            full = True
        if (
            not active
            and dot is not None
            and self.dot is not None
            and (abs(dot.x - self.dot.x) > 0.5 or abs(dot.y - self.dot.y) > 0.5)
        ):
            full = True  # the pointer towards the dot moved
        old_dot_rect = self._dot_rect(self.dot) if self.active else None
        footer_changed = (
            abs(progress - self.progress) > 1e-4
            or progress_text != self.progress_text
            or hint != self.hint
            or keys_text != self.keys_text
        )
        self.mode = self.MODE_RUNNING
        self.card.hide()
        self.active, self.dot = active, dot
        self.progress, self.progress_text = progress, progress_text
        self.hint, self.hint_color, self.keys_text = hint, hint_color, keys_text
        if full:
            self.update()
            return
        if active:
            new_rect = self._dot_rect(dot)
            if old_dot_rect is not None and new_rect is not None:
                self.update(old_dot_rect.united(new_rect))
            elif new_rect is not None:
                self.update(new_rect)
            elif old_dot_rect is not None:
                self.update(old_dot_rect)
        if footer_changed:
            self.update(self._footer_rect())

    # -------------------------------------------------------------- events
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._owner.handle_key(int(event.key())):
            super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.closing:
            # Closed by the window manager (Alt+F4, …): treat as cancel.
            self._owner.cancel()
        event.accept()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.mode == self.MODE_CARD:
            self.place_card()

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        # Replace (not blend) the translucent background of the repainted region.
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        bg = QRadialGradient(QPointF(rect.center()), max(rect.width(), rect.height()) * 0.75)
        bg.setColorAt(0.0, _BG_INNER)
        bg.setColorAt(1.0, _BG_OUTER)
        p.fillRect(event.rect(), QBrush(bg))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if self.mode == self.MODE_RUNNING:
            if self.active and self.dot is not None:
                self._paint_dot(p, self.dot)
            elif self.dot is not None:
                self._paint_pointer(p, self.dot)
            self._paint_footer(p)
        p.end()

    # -------------------------------------------------------------- geometry
    def _local(self, x: float, y: float) -> QPointF:
        return self.mapFromGlobal(QPointF(x, y))

    def _dot_extent(self) -> float:
        s = self.scale
        return max(66 * s, 11 * s * 3.4) + 6 * s

    def _dot_rect(self, dot: _DotState | None) -> QRect | None:
        if dot is None:
            return None
        c = self._local(dot.x, dot.y)
        e = self._dot_extent()
        return QRectF(c.x() - e, c.y() - e, 2 * e, 2 * e).toAlignedRect()

    def _footer_geometry(self) -> tuple[float, float, float, float]:
        s = self.scale
        width = min(self.width() * 0.36, 520 * s)
        height = max(3.0, 4 * s)
        x0 = (self.width() - width) / 2
        y0 = self.height() - 72 * s
        return x0, y0, width, height

    def _footer_rect(self) -> QRect:
        s = self.scale
        top = int(self.height() - 72 * s - 96 * s)
        return QRect(0, top, self.width(), self.height() - top)

    # -------------------------------------------------------------- painting
    def _paint_dot(self, p: QPainter, dot: _DotState) -> None:
        s = self.scale
        c = self._local(dot.x, dot.y)
        r = 11 * s * max(0.0, dot.pop)
        base_a = QColor(107, 114, 128) if dot.paused else _ACCENT
        base_b = QColor(156, 163, 175) if dot.paused else _ACCENT_2

        # Soft glow.
        if r > 0:
            glow = QRadialGradient(c, r * 3.4)
            glow.setColorAt(0.0, _rgba(base_a, 0.45))
            glow.setColorAt(0.45, _rgba(base_b, 0.14))
            glow.setColorAt(1.0, _rgba(base_b, 0.0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(c, r * 3.4, r * 3.4)

        ring_end = 11 * s + 8 * s
        if dot.ring is not None:
            # Shrinking ring: tells the eyes where to go and when collection starts.
            t = _ease_out_cubic(dot.ring)
            radius = 64 * s + (ring_end - 64 * s) * t
            color = _WARNING if dot.retry else _TEXT
            pen = QPen(_rgba(color, 0.25 + 0.6 * min(1.0, dot.ring / 0.15)), max(1.5, 2 * s))
            if dot.paused:
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, radius, radius)
        if dot.arc is not None:
            # Collection progress arc, clockwise from 12 o'clock.
            box = QRectF(c.x() - ring_end, c.y() - ring_end, 2 * ring_end, 2 * ring_end)
            width = max(2.0, 3 * s)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 38), width))
            p.drawEllipse(box)
            if dot.arc > 0:
                if dot.retry:
                    brush = QBrush(_WARNING)
                else:
                    grad = QConicalGradient(c, 90.0)
                    grad.setColorAt(0.0, base_b)
                    grad.setColorAt(1.0, base_a)
                    brush = QBrush(grad)
                pen = QPen(brush, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawArc(box, 90 * 16, -round(360 * 16 * _clamp01(dot.arc)))

        if r > 0:
            disc = QLinearGradient(c.x() - r, c.y() - r, c.x() + r, c.y() + r)
            disc.setColorAt(0.0, base_a)
            disc.setColorAt(1.0, base_b)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(disc))
            p.drawEllipse(c, r, r)
            # A crisp centre point gives the eyes something precise to fixate.
            p.setBrush(QColor(255, 255, 255, 240))
            core = max(1.5, 2.6 * s * dot.pop)
            p.drawEllipse(c, core, core)

    def _paint_pointer(self, p: QPainter, dot: _DotState) -> None:
        s = self.scale
        rect = self.rect()
        center = QPointF(rect.center())
        target = self._local(dot.target_x, dot.target_y)
        dx, dy = target.x() - center.x(), target.y() - center.y()
        dist = math.hypot(dx, dy)
        if dist >= 1.0:
            ux, uy = dx / dist, dy / dist
            reach = min(rect.width(), rect.height()) * 0.28
            tip = QPointF(center.x() + ux * reach, center.y() + uy * reach)
            angle = math.atan2(uy, ux)
            size = 26 * s
            path = QPainterPath()
            for n, sign in enumerate((-1, 1)):
                a = angle + math.pi + sign * math.radians(40)
                end = QPointF(tip.x() + size * math.cos(a), tip.y() + size * math.sin(a))
                if n == 0:
                    path.moveTo(end)
                    path.lineTo(tip)
                else:
                    path.lineTo(end)
            pen = QPen(_rgba(_ACCENT_2, 0.85), max(3.0, 5 * s))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        font = QFont(self.font())
        font.setPointSizeF(13.0)
        p.setFont(font)
        p.setPen(_MUTED)
        text_rect = QRectF(0, center.y() - 20 * s, rect.width(), 40 * s)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "Follow the dot on the other screen")

    def _paint_footer(self, p: QPainter) -> None:
        s = self.scale
        x0, y0, width, height = self._footer_geometry()
        radius = height / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 30))
        p.drawRoundedRect(QRectF(x0, y0, width, height), radius, radius)
        fill = width * _clamp01(self.progress)
        if fill > 0:
            grad = QLinearGradient(x0, y0, x0 + width, y0)
            grad.setColorAt(0.0, _ACCENT)
            grad.setColorAt(1.0, _ACCENT_2)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(QRectF(x0, y0, max(height, fill), height), radius, radius)

        font = QFont(self.font())
        font.setPointSizeF(10.0)
        p.setFont(font)
        p.setPen(_MUTED)
        full_w = float(self.width())
        if self.progress_text:
            p.drawText(
                QRectF(0, y0 - 30 * s, full_w, 22 * s),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                self.progress_text,
            )
        if self.keys_text:
            font.setPointSizeF(9.0)
            p.setFont(font)
            p.drawText(
                QRectF(0, y0 + 14 * s, full_w, 22 * s),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                self.keys_text,
            )
        if self.hint and self.active:
            font.setPointSizeF(14.0)
            font.setWeight(QFont.Weight.DemiBold)
            p.setFont(font)
            p.setPen(self.hint_color)
            p.drawText(
                QRectF(0, y0 - 90 * s, full_w, 44 * s),
                Qt.AlignmentFlag.AlignCenter,
                self.hint,
            )


# ================================================================== controller
class CalibrationWindow(QObject):
    """Runs a calibration on every monitor and hands the result to the controller.

    Args:
        controller: The app controller (``monitors()``, ``backend_info()``,
            ``begin_calibration()``, ``finish_calibration(data)`` and the
            ``observation`` signal).
        parent: Optional QObject parent.
        clock: Monotonic clock in seconds (inject a fake one in tests).
        points_per_monitor: Dots per monitor (1, 5 or a square number, see
            :func:`~eye_tracker.gaze.calibration.make_plan`).
        settle_s, collect_s, min_samples, max_retries: Collector timing, see
            :class:`~eye_tracker.gaze.calibration.CalibrationCollector`.
        margin: Distance of the outer dots from the screen edges (fraction).
        fit_in_thread: Fit the model on a background thread (default) instead
            of synchronously on the next tick.

    Signals:
        finished(bool): ``True`` after a calibration was saved, ``False`` when cancelled.
        state_changed(str): One of the ``STATE_*`` constants.

    Keys: Space starts / pauses, Enter saves, R retries, Esc cancels.
    """

    finished = Signal(bool)
    state_changed = Signal(str)

    def __init__(
        self,
        controller: object,
        parent: QObject | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        points_per_monitor: int = 9,
        settle_s: float = 0.8,
        collect_s: float = 1.0,
        min_samples: int = 5,
        max_retries: int = 1,
        margin: float = 0.1,
        fit_in_thread: bool = True,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._clock = clock
        self._points_per_monitor = points_per_monitor
        self._timing = (settle_s, collect_s, min_samples, max_retries)
        self._margin = margin
        self._fit_in_thread = fit_in_thread
        self._fit_future: Future[FitResult] | None = None
        self._fit_started = 0.0

        self._state = STATE_IDLE
        self._monitors: list[Monitor] = []
        self._labels: dict[int, str] = {}
        self._surfaces: list[_Surface] = []
        self._collector: CalibrationCollector | None = None
        self._plan: list[CalibrationTarget] = []
        self._model: GazeModel | None = None
        self._report: CalibrationReport | None = None
        self._data: CalibrationData | None = None
        self._error = ""

        self._prev_target: CalibrationTarget | None = None
        self._last_target: CalibrationTarget | None = None
        self._retrying = False
        self._manual_pause = False
        self._auto_pause = False
        self._pause_started: float | None = None
        self._paused_total = 0.0
        self._last_obs_at: float | None = None
        self._last_face_at: float | None = None
        self._face_grace_until = -math.inf
        self._connected = False

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.tick)

    # ============================================================== properties
    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        """True from :meth:`start` until the calibration is saved or cancelled."""
        return self._state not in (STATE_IDLE, STATE_CLOSED)

    @property
    def report(self) -> CalibrationReport | None:
        return self._report

    @property
    def model(self) -> GazeModel | None:
        return self._model

    @property
    def data(self) -> CalibrationData | None:
        """The saved calibration (after :meth:`save`)."""
        return self._data

    @property
    def error(self) -> str:
        return self._error

    @property
    def paused(self) -> bool:
        return self._manual_pause or self._auto_pause

    @property
    def current_target(self) -> CalibrationTarget | None:
        return self._collector.current if self._collector is not None else None

    @property
    def progress(self) -> float:
        return self._collector.progress if self._collector is not None else 0.0

    @property
    def plan(self) -> list[CalibrationTarget]:
        return list(self._plan)

    @property
    def monitors(self) -> list[Monitor]:
        return list(self._monitors)

    def surfaces(self) -> list[QWidget]:
        """The per-monitor windows (primary first)."""
        return list(self._surfaces)

    def hint(self) -> str:
        """The hint currently shown under the dot ("" when none)."""
        return self._hint(self._clock())[0]

    # ============================================================== lifecycle
    def start(self) -> bool:
        """Open the surfaces on every monitor and show the instructions.

        Returns ``False`` (and emits ``finished(False)``) when there is no monitor.
        Calling it while a calibration is open just raises the windows.
        """
        if self.is_active:
            self._focus_primary()
            return True
        try:
            monitors = list(self._controller.monitors())  # type: ignore[attr-defined]
        except Exception:
            log.exception("Could not list monitors for calibration")
            monitors = []
        if not monitors:
            log.warning("Calibration needs at least one monitor")
            self._state = STATE_CLOSED
            self.finished.emit(False)
            return False
        self._monitors = monitors
        self._labels = _monitor_labels(monitors)
        primary = next((m for m in monitors if m.primary), monitors[0])
        ordered = [primary, *(m for m in monitors if m is not primary)]
        self._surfaces = [_Surface(self, m, self._screen_for(m)) for m in ordered]

        signal = getattr(self._controller, "observation", None)
        if signal is not None:
            signal.connect(self._on_observation)
            self._connected = True
        app = QGuiApplication.instance()
        if isinstance(app, QGuiApplication):
            app.screenAdded.connect(self._on_screens_changed)
            app.screenRemoved.connect(self._on_screens_changed)
        try:
            self._controller.begin_calibration()  # type: ignore[attr-defined]
        except Exception:
            log.exception("controller.begin_calibration failed")

        log.info("Calibration opened on %d monitor(s)", len(monitors))
        self._set_state(STATE_INTRO)
        self._show_intro()
        for surface in self._surfaces:
            surface.present()
        self._focus_primary()
        self._timer.start(IDLE_TICK_MS)
        return True

    def show(self) -> bool:
        """Alias of :meth:`start`."""
        return self.start()

    def begin(self) -> None:
        """Start (or restart) showing the dots."""
        if self._state in (STATE_IDLE, STATE_CLOSED):
            return
        settle_s, collect_s, min_samples, max_retries = self._timing
        self._plan = make_plan(self._monitors, self._points_per_monitor, self._margin)
        self._collector = CalibrationCollector(
            self._plan,
            settle_s=settle_s,
            collect_s=collect_s,
            min_samples=min_samples,
            max_retries=max_retries,
        )
        now = self._clock()
        self._prev_target = self._last_target = None
        self._retrying = False
        self._manual_pause = self._auto_pause = False
        self._pause_started = None
        self._paused_total = 0.0
        self._face_grace_until = now + NO_FACE_PAUSE_S
        self._model = self._report = None
        self._error = ""
        self._drop_fit()
        self._collector.start(now)
        self._set_state(STATE_RUNNING)
        for surface in self._surfaces:
            surface.setCursor(Qt.CursorShape.BlankCursor)
        self._timer.start(TICK_MS)
        self.tick()

    def retry(self) -> None:
        """Discard the collected dots and run the sequence again."""
        log.info("Calibration restarted")
        self.begin()

    def toggle_pause(self) -> None:
        """Pause or resume the dots (Space while running)."""
        if self._state != STATE_RUNNING:
            return
        self._manual_pause = not self._manual_pause
        self._update_pause(self._clock())
        self._render()

    def save(self) -> bool:
        """Hand the calibration to the controller (Enter on the result screen)."""
        if self._state != STATE_RESULT or self._model is None or self._report is None:
            return False
        collector = self._collector
        assert collector is not None
        try:
            now_monitors = list(self._controller.monitors())  # type: ignore[attr-defined]
        except Exception:
            now_monitors = self._monitors
        if layout_signature(now_monitors) != layout_signature(self._monitors):
            self._show_error("The monitor layout changed during calibration. Please try again.")
            return False
        try:
            backend, feature_version = self._controller.backend_info()  # type: ignore[attr-defined]
        except Exception:
            log.exception("controller.backend_info failed")
            backend = feature_version = ""
        if not backend or not feature_version:
            # Without them the calibration could never be matched to the backend later.
            self._show_error("The vision backend is not ready. Please try again.")
            return False
        data = CalibrationData(
            backend=str(backend),
            feature_version=str(feature_version),
            layout_signature=layout_signature(self._monitors),
            monitors=list(self._monitors),
            samples=collector.samples,
            implicit_samples=[],
            model=self._model,
            report=self._report.to_dict(),
        )
        self._data = data
        log.info("Calibration saved: %s", self._report.summary())
        self._finish(data)
        return True

    def cancel(self) -> None:
        """Close without saving (Esc)."""
        if self._state in (STATE_IDLE, STATE_CLOSED):
            return
        log.info("Calibration cancelled")
        self._finish(None)

    close = cancel

    # ============================================================== input
    def handle_key(self, key: int) -> bool:
        """Keyboard control shared by every surface. Returns True if handled."""
        K = Qt.Key
        enter = key in (K.Key_Return.value, K.Key_Enter.value)
        if key == K.Key_Escape.value:
            self.cancel()
            return True
        if self._state == STATE_INTRO and (enter or key == K.Key_Space.value):
            self.begin()
            return True
        if self._state == STATE_RUNNING:
            if key == K.Key_Space.value:
                self.toggle_pause()
                return True
            if key == K.Key_R.value:
                self.retry()
                return True
        if self._state == STATE_RESULT:
            if enter:
                self.save()
                return True
            if key == K.Key_R.value:
                self.retry()
                return True
        if self._state == STATE_ERROR and (enter or key in (K.Key_R.value, K.Key_Space.value)):
            self.retry()
            return True
        return False

    def _on_observation(self, obs: object) -> None:
        if not isinstance(obs, Observation) or self._state == STATE_CLOSED:
            return
        now = self._clock()
        # Any observation proves the camera works; a motion-gate copy (skipped)
        # repeats the last analysed frame, whose face is still in an unchanged image.
        self._last_obs_at = now
        if obs.face_count > 0:
            self._last_face_at = now
        if self._state == STATE_RUNNING:
            # The controller turns the motion gate off while calibrating, but a
            # skipped copy may still predate the current dot: never record one.
            if not obs.skipped and not self.paused and self._collector is not None:
                self._collector.add(obs)
        elif self._state == STATE_INTRO:
            self._refresh_face_chip()

    def _on_screens_changed(self, *_args: object) -> None:
        if not self.is_active:
            return
        log.info("Monitor layout changed; calibration cancelled")
        notify = getattr(self._controller, "notify", None)
        if notify is not None:
            try:
                notify.emit("Calibration cancelled", "The monitor layout changed.")
            except Exception:
                log.debug("Could not emit notify", exc_info=True)
        self.cancel()

    # ============================================================== ticking
    def tick(self) -> None:
        """Advance the sequence to the current clock time and redraw.

        Called by a timer every 30 ms while dots are shown; tests call it directly
        after advancing a fake clock.
        """
        if self._state == STATE_INTRO:
            self._refresh_face_chip()
        elif self._state == STATE_RUNNING:
            self._tick_running()
        elif self._state == STATE_FITTING:
            self._tick_fitting()

    def _tick_running(self) -> None:
        collector = self._collector
        if collector is None:
            return
        now = self._clock()
        face_missing = now >= self._face_grace_until and (
            self._last_face_at is None or now - self._last_face_at > NO_FACE_PAUSE_S
        )
        if face_missing != self._auto_pause:
            self._auto_pause = face_missing
            log.debug("Calibration %s", "paused: no face" if face_missing else "resumed")
            self._update_pause(now)
        for event in collector.update(self._collector_time(now)):
            if event == EVENT_TARGET:
                self._prev_target = self._last_target
                self._last_target = collector.current
                self._retrying = False
            elif event == EVENT_RETRY:
                self._retrying = True
            elif event == EVENT_FINISHED:
                self._enter_fitting()
                return
        self._render()

    def _collector_time(self, now: float) -> float:
        """Wall time minus paused time: the collector's clock stands still while paused."""
        paused = self._paused_total
        if self._pause_started is not None:
            paused += now - self._pause_started
        return now - paused

    def _update_pause(self, now: float) -> None:
        if self.paused and self._pause_started is None:
            self._pause_started = now
        elif not self.paused and self._pause_started is not None:
            self._paused_total += now - self._pause_started
            self._pause_started = None

    def _hint(self, now: float) -> tuple[str, QColor]:
        if self._state != STATE_RUNNING:
            return "", _MUTED
        if self._manual_pause:
            return "Paused — press Space to continue", _TEXT
        if self._auto_pause:
            if self._last_obs_at is None or now - self._last_obs_at > NO_CAMERA_S:
                return "Waiting for the camera…", _WARNING
            return "Can't see your face — check the camera", _WARNING
        if self._retrying:
            return "Keep looking at the dot…", _WARNING
        return "", _MUTED

    def _dot_state(self) -> _DotState | None:
        collector = self._collector
        target = collector.current if collector is not None else None
        if collector is None or target is None:
            return None
        t = collector.phase_progress
        x, y, pop = target.x, target.y, 1.0
        ring: float | None = None
        arc: float | None = None
        if collector.phase == PHASE_SETTLE:
            prev = self._prev_target
            if prev is not None and prev.monitor_index == target.monitor_index:
                g = _ease_in_out(t / GLIDE_SHARE)
                x = prev.x + (target.x - prev.x) * g
                y = prev.y + (target.y - prev.y) * g
            else:
                pop = _ease_out_back(t / POP_SHARE)
            ring = t
        else:
            arc = t
        return _DotState(x, y, target.x, target.y, pop, ring, arc, self._retrying, self.paused)

    def _render(self) -> None:
        if self._state != STATE_RUNNING or self._collector is None:
            return
        collector = self._collector
        dot = self._dot_state()
        target = collector.current
        hint, color = self._hint(self._clock())
        total = len(self._plan)
        index = min(collector.current_index + 1, total)
        text = f"Dot {index} of {total}"
        keys = "Space  pause    ·    R  restart    ·    Esc  cancel"
        for surface in self._surfaces:
            active = target is not None and surface.monitor.index == target.monitor_index
            surface.set_running(
                dot,
                active=active,
                progress=collector.progress,
                progress_text=text,
                hint=hint,
                hint_color=color,
                keys_text=keys,
            )

    # ============================================================== fitting
    def _enter_fitting(self) -> None:
        self._set_state(STATE_FITTING)
        self._fit_started = self._clock()
        for surface in self._surfaces:
            surface.unsetCursor()
            card = surface.show_card()
            card.clear()
            if surface is self._surfaces[0]:
                card.add_spinner()
                card.add_label("Calculating your calibration…", size=16.0, bold=True)
                card.add_label("This takes a moment.", muted=True)
            else:
                card.add_label("Almost done…", muted=True)
            surface.place_card()
        collector = self._collector
        if self._fit_in_thread:
            samples = collector.samples if collector is not None else []
            self._fit_future = _start_fit(samples, self._monitors)
        # Without a thread the fit runs on the next tick, so the "Calculating"
        # screen is painted first; with one, ticks animate the spinner and poll.
        self._timer.start(TICK_MS)

    def _tick_fitting(self) -> None:
        now = self._clock()
        for surface in self._surfaces:
            if surface.card.spinner is not None:
                surface.card.spinner.set_time(now - self._fit_started)
        future = self._fit_future
        if future is None:
            collector = self._collector
            samples = collector.samples if collector is not None else []
            try:
                result = evaluate(samples, self._monitors)
            except Exception as exc:
                self._fit_failed(exc)
            else:
                self._fit_succeeded(*result)
            return
        if not future.done():
            return
        self._fit_future = None
        error = future.exception()
        if error is not None:
            self._fit_failed(error)
        else:
            self._fit_succeeded(*future.result())

    def _fit_succeeded(self, model: GazeModel, report: CalibrationReport) -> None:
        self._timer.stop()
        self._model, self._report = model, report
        self._set_state(STATE_RESULT)
        self._show_result(report)

    def _fit_failed(self, error: BaseException) -> None:
        collector = self._collector
        if isinstance(error, ValueError):
            # evaluate() raises ValueError when there is not enough data.
            log.info("Calibration could not be fitted: %s", error)
            skipped = len(collector.skipped_points) if collector is not None else 0
            self._show_error(
                f"Your face was not clearly visible for {skipped} of {len(self._plan)} dots."
                if skipped
                else "Not enough usable data was collected."
            )
            return
        # A numerical failure must not leave full-screen windows without a way out.
        log.error("Calibration fitting failed", exc_info=error)
        self._show_error(
            f"Something went wrong while fitting the calibration: {error}",
            badge="Calibration failed",
        )

    def _drop_fit(self) -> None:
        """Forget a running fit; its thread finishes on its own and the result is ignored."""
        if self._fit_future is not None:
            self._fit_future.cancel()
            self._fit_future = None

    # ============================================================== screens
    def _show_intro(self) -> None:
        n_points = self._points_per_monitor * len(self._monitors)
        settle_s, collect_s, _, _ = self._timing
        seconds = max(5, round(n_points * (settle_s + collect_s) / 5) * 5)
        screens = "screen" if len(self._monitors) == 1 else f"{len(self._monitors)} screens"
        for surface in self._surfaces:
            card = surface.show_card()
            card.clear()
            card.add_label("Let's calibrate", size=22.0, bold=True)
            card.add_label(
                "Look at each dot as it appears. Turn your head naturally — don't hold it still.",
                size=13.0,
            )
            card.add_label(
                f"{n_points} dots on {'one' if len(self._monitors) == 1 else 'your'} {screens}"
                f" · about {seconds} seconds",
                muted=True,
            )
            card.add_spacing(6)
            card.add_label("", name="face", size=10.5)
            card.add_spacing(4)
            card.add_keys([("Space", "start"), ("Esc", "cancel")])
            surface.place_card()
        self._refresh_face_chip()

    def _refresh_face_chip(self) -> None:
        now = self._clock()
        if self._last_obs_at is None or now - self._last_obs_at > NO_CAMERA_S:
            text, color = "●  Waiting for the camera…", _MUTED
        elif self._last_face_at is not None and now - self._last_face_at <= 1.0:
            text, color = "●  Face detected — ready", _SUCCESS
        else:
            text, color = "●  Looking for your face…", _WARNING
        for surface in self._surfaces:
            label = surface.card.labels.get("face")
            if label is not None and label.text() != text:
                label.setText(text)
                label.setStyleSheet(f"color: {_css(color)};")

    def _show_result(self, report: CalibrationReport) -> None:
        collector = self._collector
        skipped = len(collector.skipped_points) if collector is not None else 0
        multi = len(self._monitors) > 1
        for surface in self._surfaces:
            surface.unsetCursor()
            card = surface.show_card()
            card.clear()
            if surface is not self._surfaces[0]:
                card.add_label("Your results are on the main screen.", muted=True, size=12.0)
                surface.place_card()
                continue
            card.add_badge(report.grade, GRADE_COLORS.get(report.grade, _TEXT))
            if multi:
                pct = math.floor(report.monitor_accuracy * 100 + 1e-9)
                card.add_label(f"{pct}%", name="headline", size=40.0, bold=True)
                card.add_label("of glances landed on the right screen", muted=True)
            else:
                error = report.mean_error_px
                value = f"{error:.0f} px" if math.isfinite(error) else "—"
                card.add_label(value, name="headline", size=40.0, bold=True)
                card.add_label("average distance from the dots", muted=True)
            card.add_spacing(6)
            if multi and report.per_monitor_accuracy:
                # Left to right, like the dots and the "Screen n" labels.
                order = {m.index: n for n, m in enumerate(_spatial_order(self._monitors))}
                rows = [
                    (self._labels.get(index, f"Screen {index + 1}"), value)
                    for index, value in sorted(
                        report.per_monitor_accuracy.items(),
                        key=lambda kv: (order.get(kv[0], len(order)), kv[0]),
                    )
                ]
                card.add_bars(rows)
                card.add_spacing(4)
            details = f"{report.n_samples} samples from {report.n_points} dots"
            if math.isfinite(report.mean_error_px) and multi:
                details = f"Mean error {report.mean_error_px:.0f} px · " + details
            if skipped:
                details += f" · {skipped} skipped"
            card.add_label(details, name="details", muted=True, size=10.0)
            if report.grade in ("fair", "poor"):
                tip = (
                    "Tip: sit where you usually sit, make sure your face is evenly lit and "
                    "turn your head a little towards each dot. Press R to try again."
                )
                card.add_label(tip, name="tip", size=10.5, color=_WARNING)
            else:
                card.add_label(
                    "Stored only on this computer. It keeps improving as you use the mouse.",
                    name="tip",
                    size=10.0,
                    muted=True,
                )
            card.add_spacing(8)
            card.add_buttons(
                [
                    ("save", "Save   ⏎", self.save, True),
                    ("retry", "Retry   R", self.retry, False),
                    ("cancel", "Cancel   Esc", self.cancel, False),
                ]
            )
            surface.place_card()
        self._focus_primary()

    def _show_error(self, message: str, *, badge: str = "Not enough data") -> None:
        self._timer.stop()
        self._error = message
        self._set_state(STATE_ERROR)
        for surface in self._surfaces:
            surface.unsetCursor()
            card = surface.show_card()
            card.clear()
            if surface is not self._surfaces[0]:
                card.add_label("See the main screen.", muted=True, size=12.0)
                surface.place_card()
                continue
            card.add_badge(badge, _DANGER)
            card.add_label("Let's try that again", size=20.0, bold=True)
            card.add_label(message, name="message", size=12.0)
            card.add_label(
                "Check that the camera sees your whole face, that the room is not too dark "
                "and that nothing else is using the camera.",
                muted=True,
                size=10.5,
            )
            card.add_spacing(8)
            card.add_buttons(
                [
                    ("retry", "Try again   R", self.retry, True),
                    ("cancel", "Cancel   Esc", self.cancel, False),
                ]
            )
            surface.place_card()
        self._focus_primary()

    # ============================================================== internals
    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    def _finish(self, data: CalibrationData | None) -> None:
        if self._state == STATE_CLOSED:
            return
        self._set_state(STATE_CLOSED)
        self._timer.stop()
        self._drop_fit()
        if self._connected:
            with contextlib.suppress(RuntimeError, TypeError, AttributeError):
                self._controller.observation.disconnect(self._on_observation)  # type: ignore[attr-defined]
            self._connected = False
        app = QGuiApplication.instance()
        if isinstance(app, QGuiApplication):
            for sig in (app.screenAdded, app.screenRemoved):
                with contextlib.suppress(RuntimeError, TypeError):
                    sig.disconnect(self._on_screens_changed)
        try:
            self._controller.finish_calibration(data)  # type: ignore[attr-defined]
        except Exception:
            log.exception("controller.finish_calibration failed")
        for surface in self._surfaces:
            surface.closing = True
            surface.close()
            surface.deleteLater()
        self._surfaces = []
        self.finished.emit(data is not None)

    def _focus_primary(self) -> None:
        if not self._surfaces:
            return
        primary = self._surfaces[0]
        primary.raise_()
        primary.activateWindow()
        primary.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    @staticmethod
    def _screen_for(monitor: Monitor) -> QScreen | None:
        try:
            screen = screen_for_monitor(monitor)
        except Exception:
            log.debug("screen_for_monitor failed", exc_info=True)
            screen = None
        if screen is None:
            screen = (
                QGuiApplication.screenAt(
                    QRect(monitor.rect.x, monitor.rect.y, monitor.rect.w, monitor.rect.h).center()
                )
                or QGuiApplication.primaryScreen()
            )
        return screen
