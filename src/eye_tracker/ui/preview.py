"""Live camera preview with what the tracker sees.

Opening the window asks the controller for annotated preview frames
(``set_preview(True)``); hiding, minimising or closing it stops them again, so
the extra drawing work only happens while someone is watching. Frames are shown
here only: they are converted to a ``QImage`` in memory and never saved.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import (
    QCloseEvent,
    QHideEvent,
    QImage,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPixmap,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME
from ..types import GazePoint, Monitor, Observation, TrackingState, monitor_at
from . import icons, util

log = logging.getLogger(__name__)

__all__ = ["FrameView", "PreviewWindow", "bgr_to_qimage"]

NOTE = "Frames are shown only here and are never saved."
_DASH = "—"


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert an OpenCV-style ``uint8`` frame to a ``QImage`` that owns its pixels.

    Accepts BGR ``(h, w, 3)``, BGRA ``(h, w, 4)`` and greyscale ``(h, w)`` or
    ``(h, w, 1)`` arrays, contiguous or not. Raises ``ValueError`` for any other
    shape or dtype.
    """
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {array.dtype}")
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 2:
        fmt = QImage.Format.Format_Grayscale8
    elif array.ndim == 3 and array.shape[2] == 3:
        fmt = QImage.Format.Format_BGR888
    elif array.ndim == 3 and array.shape[2] == 4:
        # Byte-ordered RGBA is endian-independent, unlike ARGB32.
        array = array[:, :, [2, 1, 0, 3]]
        fmt = QImage.Format.Format_RGBA8888
    else:
        raise ValueError(f"unsupported image shape {array.shape}")
    height, width = array.shape[:2]
    if width == 0 or height == 0:
        raise ValueError("empty image")
    array = np.ascontiguousarray(array)
    channels = 1 if array.ndim == 2 else int(array.shape[2])
    # Rows of a C-contiguous array are packed, so a row is width * channels bytes.
    # (``strides[0]`` is not usable: numpy may report any stride for an axis of
    # length 1, e.g. 1 for a one-row image, which makes Qt return a null image.)
    image = QImage(array.data, width, height, width * channels, fmt)
    if image.isNull():
        raise ValueError(f"cannot convert image of shape {array.shape}")
    # QImage only borrows the numpy buffer; copy so the image outlives the array.
    return image.copy()


class FrameView(QWidget):
    """Paints the latest frame scaled to fit (aspect kept) or a placeholder text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("previewFrame")
        self.setMinimumSize(QSize(320, 240))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self._image: QImage | None = None
        self._placeholder = "Waiting for the camera…"

    @property
    def image(self) -> QImage | None:
        return self._image

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_placeholder(self, text: str) -> None:
        if text != self._placeholder:
            self._placeholder = text
            if self._image is None:
                self.update()

    def sizeHint(self) -> QSize:
        return QSize(640, 480)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        bounds = QRectF(self.rect())
        radius = 12.0 * util.ui_scale(self.screen())
        clip = QPainterPath()
        clip.addRoundedRect(bounds, radius, radius)
        painter.fillPath(clip, util.qcolor("#0B0D12"))
        if self._image is not None and not self._image.isNull():
            size = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF(
                (bounds.width() - size.width()) / 2.0,
                (bounds.height() - size.height()) / 2.0,
                size.width(),
                size.height(),
            )
            painter.setClipPath(clip)
            painter.drawImage(target, self._image)
        else:
            painter.setPen(util.qcolor(util.TEXT_MUTED))
            painter.drawText(bounds, int(Qt.AlignmentFlag.AlignCenter), self._placeholder)
        painter.end()


class PreviewWindow(QWidget):
    """Camera preview plus live details: backend, head pose, faces, rates, gaze monitor.

    Listens to ``preview_frame``, ``observation``, ``stats_changed``,
    ``gaze_changed`` and ``state_changed``; calls ``controller.set_preview`` when
    shown and hidden. Closing only hides it, so the app can keep one instance.
    """

    def __init__(self, controller: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._controller = controller
        self._preview_on = False
        self._frame_errors = 0
        self._monitors: list[Monitor] = []
        self._state = util.controller_state(controller)
        self.setWindowTitle(f"Camera preview — {APP_NAME}")
        self.setWindowIcon(icons.app_icon())
        self.setObjectName("previewWindow")
        s = util.ui_scale(self.screen())

        root = QVBoxLayout(self)
        root.setContentsMargins(round(16 * s), round(12 * s), round(16 * s), round(16 * s))
        root.setSpacing(round(10 * s))

        note_row = QHBoxLayout()
        note_row.setSpacing(round(8 * s))
        lock = QLabel(self)
        lock_px = round(16 * s)
        lock.setPixmap(_lock_pixmap(lock_px))
        lock.setFixedSize(lock_px, lock_px)
        self.note_label = QLabel(NOTE, self)
        self.note_label.setObjectName("previewNote")
        note_row.addWidget(lock)
        note_row.addWidget(self.note_label, 1)
        root.addLayout(note_row)

        self.view = FrameView(self)
        root.addWidget(self.view, 1)

        panel = QFrame(self)
        panel.setObjectName("previewInfo")
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(panel)
        grid.setContentsMargins(round(12 * s), round(10 * s), round(12 * s), round(10 * s))
        grid.setHorizontalSpacing(round(18 * s))
        grid.setVerticalSpacing(round(6 * s))
        self.values: dict[str, QLabel] = {}
        fields = (
            ("state", "State"),
            ("backend", "Backend"),
            ("faces", "Faces"),
            ("pose", "Head yaw / pitch"),
            ("monitor", "Looking at"),
            ("fps", "Camera rate"),
            ("inference", "Analysis time"),
            ("skipped", "Frames skipped"),
            ("cpu", "CPU"),
        )
        rows = math.ceil(len(fields) / 2)
        for i, (key, caption) in enumerate(fields):
            column = (i // rows) * 2
            row = i % rows
            label = QLabel(caption, panel)
            util.set_muted(label)
            value = QLabel(_DASH, panel)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setObjectName(f"previewValue_{key}")
            grid.addWidget(label, row, column)
            grid.addWidget(value, row, column + 1)
            self.values[key] = value
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        root.addWidget(panel)

        for name, slot in (
            ("preview_frame", self.set_frame),
            ("observation", self.set_observation),
            ("stats_changed", self.set_stats),
            ("gaze_changed", self.set_gaze),
            ("state_changed", self.set_state),
        ):
            signal = getattr(controller, name, None)
            if signal is not None:
                signal.connect(slot)

        self._show_backend()
        self.set_state(self._state)
        self.resize(round(680 * s), round(660 * s))

    # ------------------------------------------------------------------ public API
    @property
    def preview_active(self) -> bool:
        """Whether this window has asked the controller for preview frames."""
        return self._preview_on

    def set_frame(self, frame: object) -> None:
        """Show an annotated BGR frame (``np.ndarray``)."""
        if not self.isVisible() or not isinstance(frame, np.ndarray):
            return
        try:
            image = bgr_to_qimage(frame)
        except ValueError as exc:
            self._frame_errors += 1
            if self._frame_errors == 1:  # a broken stream would flood the log
                log.warning("Cannot show preview frame: %s", exc)
            return
        self.view.set_image(image)

    def set_observation(self, obs: object) -> None:
        if not isinstance(obs, Observation):
            return
        self._set("faces", str(obs.face_count))
        if obs.head_yaw is not None and obs.head_pitch is not None:
            self._set("pose", f"{obs.head_yaw:+.0f}° / {obs.head_pitch:+.0f}°")
        elif not obs.face_present:
            self._set("pose", _DASH)

    def set_stats(self, stats: object) -> None:
        if not isinstance(stats, Mapping):
            return
        fps = _number(stats.get("fps"))
        target = _number(stats.get("target_fps"))
        if fps is not None:
            text = util.format_fps(fps)
            if target is not None and target > 0:
                text += f" (target {util.format_fps(target)})"
            self._set("fps", text)
        inference = _number(stats.get("inference_ms"))
        if inference is not None:
            self._set("inference", f"{inference:.1f} ms")
        skipped = _number(stats.get("skip_ratio"))
        if skipped is not None:
            self._set("skipped", f"{skipped * 100:.0f} %")
        cpu = _number(stats.get("cpu_percent"))
        if cpu is not None:
            self._set("cpu", f"{cpu:.1f} %")
        backend = stats.get("backend")
        if isinstance(backend, str) and backend:
            self._show_backend(backend)

    def set_gaze(self, point: object) -> None:
        if not isinstance(point, GazePoint):
            self._set("monitor", _DASH)
            return
        if not self._monitors:
            self._monitors = util.controller_monitors(self._controller)
        monitor = monitor_at(self._monitors, point.x, point.y)
        self._set("monitor", util.monitor_label(monitor) if monitor else "Off-screen")

    def set_state(self, state: object) -> None:
        if not isinstance(state, TrackingState):
            return
        self._state = state
        self._set("state", state.label)
        if state.camera_active:
            self.view.set_placeholder("Waiting for the camera…")
        else:
            self.view.set_placeholder(f"Camera is off: {state.label.lower()}")
            self.view.set_image(None)

    # ------------------------------------------------------------------ internals
    def _set(self, key: str, text: str) -> None:
        label = self.values[key]
        if label.text() != text:
            label.setText(text)

    def _show_backend(self, name: str | None = None) -> None:
        info = getattr(self._controller, "backend_info", None)
        version = ""
        if callable(info):
            try:
                backend, version = info()
                name = name or backend
            except Exception:
                log.debug("backend_info() failed", exc_info=True)
        if name:
            self._set("backend", f"{name} ({version})" if version else name)

    def _set_preview(self, enabled: bool) -> None:
        if enabled == self._preview_on:
            return
        self._preview_on = enabled
        setter = getattr(self._controller, "set_preview", None)
        if callable(setter):
            try:
                setter(enabled)
            except Exception:
                log.exception("controller.set_preview(%s) failed", enabled)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._monitors = util.controller_monitors(self._controller)
        self._show_backend()
        self._set_preview(True)

    def hideEvent(self, event: QHideEvent) -> None:
        # Also sent when the window is minimised: no one is watching then.
        super().hideEvent(event)
        self._set_preview(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._set_preview(False)
        self.view.set_image(None)
        super().closeEvent(event)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _lock_pixmap(size: int) -> QPixmap:
    image = QImage(size * 2, size * 2, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        icons.paint_lock(painter, QRectF(0, 0, size * 2, size * 2), util.qcolor(util.ACCENT))
    finally:
        painter.end()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(2.0)
    return pixmap
