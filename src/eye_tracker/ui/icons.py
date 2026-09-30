"""Icons drawn with ``QPainter``: no image files, crisp at every size.

* :func:`app_icon` - the application icon: a rounded square with the brand's
  indigo→cyan gradient and a white eye.
* :func:`tray_icon` - a monochrome eye glyph per :class:`TrackingState`
  (tracking: accent pupil; paused: pause bars; privacy: slashed eye; away:
  closed eye; camera error: red badge; needs calibration: amber badge).
* :func:`render_app_icon_png` / :func:`render_tray_image` - the same drawings as
  ``QImage`` (used by ``scripts/make_icons.py`` and the tests).
* :func:`paint_lock` / :func:`status_dot_icon` - small glyphs for other widgets.

Every size is painted from vector paths rather than scaled from one bitmap, so
16 px tray icons stay sharp. Tray glyphs are drawn on a 16-unit grid; on macOS
they are template images (``QIcon.setIsMask``) that the menu bar tints itself.
Elsewhere the glyph colour follows the taskbar (:func:`util.system_tray_is_dark`)
and a faint halo in the opposite colour keeps it readable if that guess is wrong.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QRadialGradient,
)

from ..types import TrackingState
from . import util

log = logging.getLogger(__name__)

__all__ = [
    "APP_ICON_SIZES",
    "TRAY_SIZES",
    "app_icon",
    "clear_cache",
    "paint_lock",
    "render_app_icon_png",
    "render_tray_image",
    "status_dot_icon",
    "tray_icon",
]

#: Pixel sizes rendered into the tray icon: 16-24 px at 100-150 %, the macOS
#: menu bar (18/22 pt at 1x and 2x) and larger sizes for high-DPI taskbars.
TRAY_SIZES: tuple[int, ...] = (16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 64)
#: Pixel sizes rendered into the application icon.
APP_ICON_SIZES: tuple[int, ...] = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256, 512)

_GRID = 16.0  # design grid of the tray glyphs
_STROKE = 1.5  # lid stroke on that grid

# (base glyph, overlay) per state. Overlays are drawn over a knocked-out gap so
# they stay legible at 16 px and in single-colour (template) mode.
_TRAY_STYLES: dict[TrackingState, tuple[str, str | None]] = {
    TrackingState.STARTING: ("starting", None),
    TrackingState.NEEDS_CALIBRATION: ("open", "warning"),
    TrackingState.CALIBRATING: ("calibrating", None),
    TrackingState.TRACKING: ("tracking", None),
    TrackingState.PAUSED: ("pause", None),
    TrackingState.PRIVACY: ("open", "slash"),
    TrackingState.AWAY: ("closed", None),
    TrackingState.LOCKED: ("closed", None),
    TrackingState.YIELDED: ("pause", None),
    TrackingState.CAMERA_ERROR: ("open", "error"),
}

_icon_cache: dict[tuple[object, ...], QIcon] = {}


def clear_cache() -> None:
    """Forget cached ``QIcon`` objects (e.g. after a theme change)."""
    _icon_cache.clear()


# ----------------------------------------------------------------------- primitives
def _new_image(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    return image


def _almond(rect: QRectF, pinch: float = 0.22) -> QPainterPath:
    """Closed eye outline with pointed corners, filling ``rect``'s height."""
    left, right = rect.left(), rect.right()
    cy = rect.center().y()
    # A cubic whose two control points sit at +-k peaks at 0.75 k.
    k = rect.height() / 2.0 / 0.75
    dx = rect.width() * pinch
    path = QPainterPath(QPointF(left, cy))
    path.cubicTo(left + dx, cy - k, right - dx, cy - k, right, cy)
    path.cubicTo(right - dx, cy + k, left + dx, cy + k, left, cy)
    path.closeSubpath()
    return path


def _stroke(path: QPainterPath, width: float) -> QPainterPath:
    """Outline of ``path`` stroked with round caps and joins, as a fillable path."""
    stroker = QPainterPathStroker()
    stroker.setWidth(width)
    stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
    stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return stroker.createStroke(path)


def _circle(cx: float, cy: float, r: float) -> QPainterPath:
    path = QPainterPath()
    path.addEllipse(QPointF(cx, cy), r, r)
    return path


def _line(x0: float, y0: float, x1: float, y1: float) -> QPainterPath:
    path = QPainterPath(QPointF(x0, y0))
    path.lineTo(x1, y1)
    return path


def _rounded(x: float, y: float, w: float, h: float, r: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(QRectF(x, y, w, h), r, r)
    return path


# ------------------------------------------------------------------------ tray glyph
@dataclass
class _Glyph:
    mono: list[QPainterPath] = field(default_factory=list)
    accent: list[QPainterPath] = field(default_factory=list)
    overlay: list[QPainterPath] = field(default_factory=list)
    knockout: list[QPainterPath] = field(default_factory=list)
    overlay_kind: str | None = None


def _tray_glyph(base: str, overlay: str | None) -> _Glyph:
    glyph = _Glyph(overlay_kind=overlay)
    if base == "closed":
        lid = QPainterPath(QPointF(1.6, 6.6))
        lid.cubicTo(4.4, 11.4, 11.6, 11.4, 14.4, 6.6)
        glyph.mono.append(_stroke(lid, _STROKE))
        for t, dx, dy in ((0.2, -1.0, 1.7), (0.5, 0.0, 2.1), (0.8, 1.0, 1.7)):
            point = lid.pointAtPercent(t)
            glyph.mono.append(
                _stroke(_line(point.x(), point.y(), point.x() + dx, point.y() + dy), 1.3)
            )
    else:
        # Apex 3.4..12.6 leaves a clear gap between lids and a 2.4-unit pupil.
        glyph.mono.append(_stroke(_almond(QRectF(1.25, 3.4, 13.5, 9.2)), _STROKE))
        if base == "open":
            glyph.mono.append(_circle(8.0, 8.0, 2.4))
        elif base == "starting":
            glyph.mono.append(_stroke(_circle(8.0, 8.0, 2.0), 1.2))
        elif base == "tracking":
            glyph.accent.append(_circle(8.0, 8.0, 2.6))
        elif base == "calibrating":
            glyph.accent.append(_stroke(_circle(8.0, 8.0, 2.35), 1.1))
            glyph.accent.append(_circle(8.0, 8.0, 0.95))
        elif base == "pause":
            # Whole-unit x edges keep the bars sharp at 16 px.
            glyph.mono.append(_rounded(5.0, 5.5, 2.0, 5.0, 0.6))
            glyph.mono.append(_rounded(9.0, 5.5, 2.0, 5.0, 0.6))
        else:
            raise ValueError(f"unknown tray glyph {base!r}")

    if overlay == "slash":
        line = _line(2.4, 2.4, 13.6, 13.6)
        glyph.overlay.append(_stroke(line, _STROKE))
        glyph.knockout.append(_stroke(line, _STROKE + 2.6))
    elif overlay in {"error", "warning"}:
        glyph.overlay.append(_circle(12.55, 12.55, 2.85))
        glyph.knockout.append(_circle(12.55, 12.55, 2.85 + 1.25))
    elif overlay is not None:
        raise ValueError(f"unknown tray overlay {overlay!r}")
    return glyph


def _paint_tray_glyph(
    painter: QPainter, state: TrackingState, size: int, *, dark: bool, mask: bool
) -> None:
    base, overlay = _TRAY_STYLES.get(state, ("open", None))
    glyph = _tray_glyph(base, overlay)

    if mask:
        # Template image: only alpha matters; the menu bar picks the colour.
        fg = QColor(0, 0, 0)
        accent_brush = QBrush(fg)
        halo: QColor | None = None
        badge = {"error": fg, "warning": fg}
    else:
        fg = util.qcolor("#F3F4F6" if dark else "#1F2937")
        gradient = QLinearGradient(QPointF(5.2, 5.2), QPointF(10.8, 10.8))
        gradient.setColorAt(0.0, util.qcolor(util.ACCENT_LIGHT if dark else util.ACCENT_DEEP))
        gradient.setColorAt(1.0, util.qcolor(util.ACCENT_2 if dark else util.ACCENT_2_DEEP))
        accent_brush = QBrush(gradient)
        halo = QColor(0, 0, 0, 110) if dark else QColor(255, 255, 255, 150)
        badge = {"error": util.qcolor(util.DANGER), "warning": util.qcolor(util.WARNING)}

    units_per_px = _GRID / float(size)
    halo_pen = (
        QPen(
            halo,
            2.0 * units_per_px,  # one device pixel on each side of the edge
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        if halo is not None
        else None
    )

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.scale(size / _GRID, size / _GRID)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if halo_pen is not None:
        painter.setPen(halo_pen)
        for path in glyph.mono + glyph.accent:
            painter.drawPath(path)
    painter.setPen(Qt.PenStyle.NoPen)
    for path in glyph.mono:
        painter.fillPath(path, fg)
    for path in glyph.accent:
        painter.fillPath(path, accent_brush)

    if glyph.overlay:
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        for path in glyph.knockout:
            painter.fillPath(path, QColor(0, 0, 0))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        color = badge.get(glyph.overlay_kind or "", fg)
        if halo_pen is not None:
            painter.setPen(halo_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for path in glyph.overlay:
                painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen)
        for path in glyph.overlay:
            painter.fillPath(path, color)
    painter.restore()


def render_tray_image(
    state: TrackingState, size: int, *, dark: bool = True, mask: bool = False
) -> QImage:
    """The tray glyph for ``state`` as a ``size``×``size`` ARGB image.

    ``dark`` means a dark taskbar (light glyph); ``mask`` renders a black
    template image for the macOS menu bar.
    """
    if size < 1:
        raise ValueError("size must be positive")
    image = _new_image(size)
    painter = QPainter(image)
    try:
        _paint_tray_glyph(painter, state, size, dark=dark, mask=mask)
    finally:
        painter.end()
    return image


def tray_icon(state: TrackingState, *, dark: bool | None = None, mask: bool | None = None) -> QIcon:
    """Monochrome tray icon for ``state`` with a pixmap for every tray size.

    ``mask=None`` uses template images on macOS only; ``dark=None`` asks
    :func:`util.system_tray_is_dark`. Icons are cached per combination.
    """
    if mask is None:
        mask = sys.platform == "darwin"
    if dark is None:
        dark = True if mask else util.system_tray_is_dark()
    key = ("tray", state, bool(dark), bool(mask))
    icon = _icon_cache.get(key)
    if icon is None:
        icon = QIcon()
        for size in TRAY_SIZES:
            image = render_tray_image(state, size, dark=bool(dark), mask=bool(mask))
            icon.addPixmap(QPixmap.fromImage(image))
        icon.setIsMask(bool(mask))
        _icon_cache[key] = icon
    return icon


# ------------------------------------------------------------------------- app icon
def _soft_shadow(painter: QPainter, rect: QRectF, radius: float, spread: float) -> None:
    """Cheap blurred drop shadow: stacked, slightly offset translucent rounded rects."""
    steps = 10
    for i in range(steps):
        grow = spread * (i + 1) / steps
        alpha = int(38 * (1.0 - i / steps) / 2.2)
        r = rect.adjusted(-grow, -grow + spread * 0.35, grow, grow + spread * 0.35)
        path = QPainterPath()
        path.addRoundedRect(r, radius + grow, radius + grow)
        painter.fillPath(path, QColor(8, 10, 30, alpha))


def _paint_app_icon(painter: QPainter, size: int, *, padded: bool) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = float(size)
    if padded:
        # macOS icon grid: an 824/1024 rounded square with room for its shadow.
        side = s * 824.0 / 1024.0
        body = QRectF((s - side) / 2.0, (s - side) / 2.0 - s * 0.01, side, side)
    else:
        body = QRectF(0.0, 0.0, s, s)
    w = body.width()
    radius = w * 0.225
    if padded:
        _soft_shadow(painter, body, radius, s * 0.03)

    tile = QPainterPath()
    tile.addRoundedRect(body, radius, radius)
    painter.fillPath(tile, util.accent_gradient(body.topLeft(), body.bottomRight()))
    # Gentle top light gives the flat tile some depth without looking glossy.
    light = QLinearGradient(body.topLeft(), body.bottomLeft())
    light.setColorAt(0.0, QColor(255, 255, 255, 56))
    light.setColorAt(0.55, QColor(255, 255, 255, 0))
    painter.fillPath(tile, light)

    cx, cy = body.center().x(), body.center().y()
    eye = _almond(QRectF(body.left() + w * 0.15, cy - w * 0.215, w * 0.70, w * 0.43), 0.23)
    painter.fillPath(eye.translated(0.0, w * 0.018), QColor(20, 18, 70, 60))
    painter.fillPath(eye, QColor(255, 255, 255))

    painter.save()
    painter.setClipPath(eye)
    iris_r = w * 0.155
    iris = QRadialGradient(QPointF(cx, cy), iris_r)
    iris.setColorAt(0.0, util.qcolor("#3B3A98"))
    iris.setColorAt(0.75, util.qcolor("#26236E"))
    iris.setColorAt(1.0, util.qcolor("#1B1850"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(iris))
    painter.drawEllipse(QPointF(cx, cy), iris_r, iris_r)
    ring = QPen(util.qcolor(util.ACCENT_2, 170), max(1.0, w * 0.012))
    painter.setPen(ring)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QPointF(cx, cy), iris_r * 0.82, iris_r * 0.82)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(util.qcolor("#0B0B24"))
    painter.drawEllipse(QPointF(cx, cy), w * 0.068, w * 0.068)
    painter.setBrush(QColor(255, 255, 255, 235))
    painter.drawEllipse(QPointF(cx + w * 0.052, cy - w * 0.055), w * 0.03, w * 0.03)
    painter.restore()
    painter.restore()


def render_app_icon_png(size: int, *, padded: bool = False) -> QImage:
    """The application icon as a ``size``×``size`` ARGB image.

    ``padded`` follows the macOS icon grid (smaller tile with a soft shadow);
    use it for ``.icns``. Windows and Linux icons use the full square.
    """
    if size < 1:
        raise ValueError("size must be positive")
    image = _new_image(size)
    painter = QPainter(image)
    try:
        _paint_app_icon(painter, size, padded=padded)
    finally:
        painter.end()
    return image


def app_icon() -> QIcon:
    """The application icon with a pixmap for every common size (cached)."""
    key = ("app",)
    icon = _icon_cache.get(key)
    if icon is None:
        icon = QIcon()
        for size in APP_ICON_SIZES:
            icon.addPixmap(QPixmap.fromImage(render_app_icon_png(size)))
        _icon_cache[key] = icon
    return icon


# --------------------------------------------------------------------- small glyphs
def paint_lock(painter: QPainter, rect: QRectF, color: QColor | QBrush) -> None:
    """Draw a padlock filling ``rect`` (keyhole cut out, so any background shows)."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    w, h = rect.width(), rect.height()
    body = QRectF(rect.left() + w * 0.14, rect.top() + h * 0.44, w * 0.72, h * 0.54)
    body_path = QPainterPath()
    body_path.addRoundedRect(body, w * 0.12, w * 0.12)
    hole_y = body.top() + body.height() * 0.42
    slot = _rounded(body.center().x() - w * 0.03, hole_y, w * 0.06, h * 0.17, w * 0.03)
    # A boolean union: with odd-even filling the circle/slot overlap would stay solid.
    hole = _circle(body.center().x(), hole_y, w * 0.075).united(slot)
    body_path = body_path.subtracted(hole)

    shackle = QPainterPath(QPointF(rect.left() + w * 0.3, body.top() + h * 0.02))
    shackle.lineTo(rect.left() + w * 0.3, rect.top() + h * 0.28)
    shackle.arcTo(
        QRectF(rect.left() + w * 0.3, rect.top() + h * 0.06, w * 0.4, h * 0.44), 180, -180
    )
    shackle.lineTo(rect.left() + w * 0.7, body.top() + h * 0.02)
    brush = color if isinstance(color, QBrush) else QBrush(color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.fillPath(_stroke(shackle, w * 0.1), brush)
    painter.fillPath(body_path, brush)
    painter.restore()


def status_dot_icon(color: str) -> QIcon:
    """A small filled circle (menu status indicator) in ``color`` (``#RRGGBB``)."""
    key = ("dot", color)
    icon = _icon_cache.get(key)
    if icon is None:
        icon = QIcon()
        for size in (16, 20, 24, 32, 48):
            image = _new_image(size)
            painter = QPainter(image)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(util.qcolor(color))
                r = size * 0.22
                painter.drawEllipse(QPointF(size / 2.0, size / 2.0), r, r)
            finally:
                painter.end()
            icon.addPixmap(QPixmap.fromImage(image))
        _icon_cache[key] = icon
    return icon
