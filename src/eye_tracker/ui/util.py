"""Shared helpers for the Qt user interface.

* DPI: :func:`ui_scale` turns a design size in pixels into device pixels for a
  screen. On Windows and Linux/X11 the app disables Qt's own high-DPI scaling (so
  Qt coordinates equal native pixels, see :mod:`eye_tracker.types`); hand-painted
  sizes must therefore be scaled explicitly. Font sizes in *points* are already
  converted with the screen's DPI by Qt and need no scaling.
* Screens: :func:`screen_for_monitor` maps the controller's :class:`Monitor`
  objects back to ``QScreen`` instances.
* Palette: the colours of the app's calm dark surfaces and its indigo→cyan accent.
* Small, UI-only formatting helpers and defensive accessors for the controller.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from collections.abc import Sequence

from PySide6.QtCore import QKeyCombination, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QScreen,
)
from PySide6.QtWidgets import QWidget

from ..config import Settings
from ..platform.hotkeys import Hotkey, parse_hotkey
from ..types import Monitor, TrackingState

log = logging.getLogger(__name__)

__all__ = [
    "ACCENT",
    "ACCENT_2",
    "ACCENT_2_DEEP",
    "ACCENT_DEEP",
    "ACCENT_LIGHT",
    "CURTAIN",
    "DANGER",
    "SUCCESS",
    "SURFACE",
    "SURFACE_ALPHA",
    "TEXT",
    "TEXT_MUTED",
    "WARNING",
    "accent_gradient",
    "controller_monitors",
    "controller_settings",
    "controller_state",
    "elide",
    "format_cpu",
    "format_fps",
    "hotkey_sequence",
    "make_floating",
    "monitor_label",
    "paint_surface",
    "parse_hotkey_text",
    "primary_monitor",
    "qcolor",
    "scaled",
    "screen_for_monitor",
    "screen_for_point",
    "set_muted",
    "system_tray_is_dark",
    "ui_scale",
]

# --------------------------------------------------------------------------- palette
#: Accent gradient start (indigo 500) and end (cyan 400).
ACCENT = "#6366F1"
ACCENT_2 = "#22D3EE"
#: A lighter indigo for accents drawn on dark backgrounds.
ACCENT_LIGHT = "#818CF8"
#: Deeper accent pair with enough contrast on light backgrounds.
ACCENT_DEEP = "#4F46E5"
ACCENT_2_DEEP = "#0891B2"

#: Dark translucent surface used by toasts and floating panels.
SURFACE = "#12141C"
SURFACE_ALPHA = 238
#: Near-black, fully opaque background of the privacy curtain.
CURTAIN = "#07080C"

TEXT = "#F5F7FA"
TEXT_MUTED = "#A0A8B8"

SUCCESS = "#34D399"
WARNING = "#F59E0B"
DANGER = "#EF4444"


def qcolor(value: str, alpha: int | None = None) -> QColor:
    """``QColor`` from a ``#RRGGBB`` string with an optional 0..255 alpha."""
    color = QColor(value)
    if alpha is not None:
        color.setAlpha(max(0, min(255, int(alpha))))
    return color


def accent_gradient(start: QPointF, end: QPointF, *, deep: bool = False) -> QLinearGradient:
    """The indigo→cyan brand gradient between two points.

    ``deep`` selects the darker pair, which keeps its contrast on light surfaces.
    """
    gradient = QLinearGradient(start, end)
    gradient.setColorAt(0.0, qcolor(ACCENT_DEEP if deep else ACCENT))
    gradient.setColorAt(1.0, qcolor(ACCENT_2_DEEP if deep else ACCENT_2))
    return gradient


# ------------------------------------------------------------------------------- DPI
def ui_scale(screen: QScreen | None = None) -> float:
    """Factor from design pixels (96 DPI) to device pixels on ``screen``.

    macOS always returns 1.0: Qt coordinates are points there and Qt handles
    Retina scaling itself. Elsewhere the factor is ``logicalDotsPerInch / 96``
    (1.5 on a 150 % display). When Qt's own high-DPI scaling is active (for
    example a native Wayland session) the logical DPI stays at 96, so the result
    is 1.0 and nothing is scaled twice. ``None`` means the primary screen.
    """
    if sys.platform == "darwin":
        return 1.0
    if screen is None:
        if QGuiApplication.instance() is None:
            return 1.0
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return 1.0
    try:
        dpi = float(screen.logicalDotsPerInch())
    except (RuntimeError, TypeError, ValueError):  # screen already destroyed
        return 1.0
    if not math.isfinite(dpi) or dpi <= 0:
        return 1.0
    return min(max(dpi / 96.0, 0.5), 4.0)


def scaled(pixels: float, scale: float) -> int:
    """Round a design size to whole device pixels (never less than 1)."""
    return max(1, round(pixels * scale))


# --------------------------------------------------------------------------- screens
def _geometry_tuple(screen: QScreen) -> tuple[int, int, int, int]:
    g = screen.geometry()
    return (g.x(), g.y(), g.width(), g.height())


def screen_for_monitor(monitor: Monitor) -> QScreen | None:
    """The ``QScreen`` that corresponds to a controller :class:`Monitor`.

    Matching is by exact geometry first (monitor rects are built from
    ``QScreen.geometry()``), then by name, then by the screen containing the
    monitor's centre, then by index. ``None`` without a GUI application.
    """
    if QGuiApplication.instance() is None:
        return None
    screens = QGuiApplication.screens()
    wanted = tuple(monitor.rect.to_list())
    for screen in screens:
        if _geometry_tuple(screen) == wanted:
            return screen
    for screen in screens:
        if monitor.name and screen.name() == monitor.name:
            return screen
    cx, cy = monitor.rect.center
    under_centre = QGuiApplication.screenAt(QPoint(int(cx), int(cy)))
    if under_centre is not None:
        return under_centre
    if 0 <= monitor.index < len(screens):
        return screens[monitor.index]
    return None


def screen_for_point(x: float, y: float) -> QScreen | None:
    """Screen containing a global point, else the primary screen."""
    if QGuiApplication.instance() is None:
        return None
    return QGuiApplication.screenAt(QPoint(int(x), int(y))) or QGuiApplication.primaryScreen()


def primary_monitor(monitors: Sequence[Monitor]) -> Monitor | None:
    """The primary monitor, else the first one, else ``None``."""
    for monitor in monitors:
        if monitor.primary:
            return monitor
    return monitors[0] if monitors else None


def monitor_label(monitor: Monitor) -> str:
    """Human label such as ``"Monitor 2 · DELL U2720Q"`` (indices shown from 1)."""
    label = f"Monitor {monitor.index + 1}"
    name = monitor.name.strip()
    if name:
        label += f" · {name}"
    return label


# ------------------------------------------------------------------------ formatting
def format_fps(fps: float) -> str:
    """``"12 fps"``, ``"0.5 fps"`` (one decimal only below 1 fps)."""
    if not math.isfinite(fps) or fps <= 0:
        return "0 fps"
    if fps < 0.95:
        return f"{fps:.1f} fps"
    return f"{round(fps)} fps"


def format_cpu(percent: float) -> str:
    """``"CPU 0.6 %"`` (share of the whole machine)."""
    if not math.isfinite(percent) or percent < 0:
        percent = 0.0
    return f"CPU {percent:.1f} %"


def elide(text: str, limit: int) -> str:
    """Shorten ``text`` to at most ``limit`` characters with a trailing ellipsis."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


# ------------------------------------------------------------------------ controller
def controller_settings(controller: object) -> Settings:
    """The controller's current :class:`Settings` (defaults when it has none).

    The widgets only read settings. They accept a ``settings`` attribute,
    property or zero-argument method so they do not depend on how the controller
    exposes it, and fall back to defaults for a partial fake in tests.
    """
    value = getattr(controller, "settings", None)
    if callable(value):
        try:
            value = value()
        except Exception:
            log.debug("controller.settings() failed", exc_info=True)
            value = None
    return value if isinstance(value, Settings) else Settings()


def controller_state(controller: object) -> TrackingState:
    """The controller's current :class:`TrackingState` (``STARTING`` if unknown)."""
    value = getattr(controller, "state", None)
    if callable(value):
        try:
            value = value()
        except Exception:
            value = None
    return value if isinstance(value, TrackingState) else TrackingState.STARTING


def controller_monitors(controller: object) -> list[Monitor]:
    """``controller.monitors()`` that never raises (empty list on failure)."""
    monitors = getattr(controller, "monitors", None)
    if not callable(monitors):
        return []
    try:
        return [m for m in monitors() if isinstance(m, Monitor)]
    except Exception:
        log.debug("controller.monitors() failed", exc_info=True)
        return []


# --------------------------------------------------------------------------- theming
def system_tray_is_dark() -> bool:
    """Best guess whether the taskbar / panel behind the tray icon is dark.

    The tray icon is monochrome, so it has to pick a glyph colour. (macOS does
    not need this: its menu-bar icon is a template image the system tints.)

    * Windows: the taskbar follows ``SystemUsesLightTheme``, which is separate
      from the app theme Qt reports.
    * GNOME: the top bar is dark whatever the theme.
    * otherwise: Qt's colour scheme, then the window palette.

    The icons also draw a faint contrasting halo, so a wrong guess still
    leaves a readable icon.
    """
    if sys.platform == "win32":
        value = _windows_personalize_value("SystemUsesLightTheme")
        return value != 1  # missing value: Windows 10's default dark taskbar
    if sys.platform.startswith("linux"):
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        if "gnome" in desktop or "unity" in desktop:
            return True
    return _app_prefers_dark()


def _windows_personalize_value(name: str) -> int | None:
    try:
        import winreg

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            value, _kind = winreg.QueryValueEx(key, name)
        return int(value)
    except (OSError, ImportError, TypeError, ValueError):
        return None


def _app_prefers_dark() -> bool:
    if QGuiApplication.instance() is None:
        return True
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except AttributeError:  # Qt < 6.5
        pass
    window = QGuiApplication.palette().color(QPalette.ColorRole.Window)
    return window.lightness() < 128


# --------------------------------------------------------------------------- hotkeys
_QT_KEYS: dict[str, Qt.Key] = {
    "space": Qt.Key.Key_Space,
    "enter": Qt.Key.Key_Return,
    "tab": Qt.Key.Key_Tab,
    "escape": Qt.Key.Key_Escape,
    "backspace": Qt.Key.Key_Backspace,
    "insert": Qt.Key.Key_Insert,
    "delete": Qt.Key.Key_Delete,
    "home": Qt.Key.Key_Home,
    "end": Qt.Key.Key_End,
    "pageup": Qt.Key.Key_PageUp,
    "pagedown": Qt.Key.Key_PageDown,
    "left": Qt.Key.Key_Left,
    "right": Qt.Key.Key_Right,
    "up": Qt.Key.Key_Up,
    "down": Qt.Key.Key_Down,
    "minus": Qt.Key.Key_Minus,
    "equal": Qt.Key.Key_Equal,
    "bracketleft": Qt.Key.Key_BracketLeft,
    "bracketright": Qt.Key.Key_BracketRight,
    "backslash": Qt.Key.Key_Backslash,
    "semicolon": Qt.Key.Key_Semicolon,
    "quote": Qt.Key.Key_Apostrophe,
    "grave": Qt.Key.Key_QuoteLeft,
    "comma": Qt.Key.Key_Comma,
    "period": Qt.Key.Key_Period,
    "slash": Qt.Key.Key_Slash,
}


def _qt_key(name: str) -> Qt.Key | None:
    if len(name) == 1 and "a" <= name <= "z":
        return Qt.Key(Qt.Key.Key_A.value + ord(name) - ord("a"))
    if len(name) == 1 and name.isdigit():
        return Qt.Key(Qt.Key.Key_0.value + int(name))
    if len(name) > 1 and name[0] == "f" and name[1:].isdigit():
        number = int(name[1:])
        if 1 <= number <= 35:
            return Qt.Key(Qt.Key.Key_F1.value + number - 1)
        return None
    return _QT_KEYS.get(name)


def parse_hotkey_text(text: str | None) -> Hotkey | None:
    """``parse_hotkey`` that returns ``None`` instead of raising (empty or invalid text)."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return parse_hotkey(text)
    except (TypeError, ValueError):
        return None


def hotkey_sequence(
    value: str | Hotkey | None, *, macos: bool | None = None
) -> QKeySequence | None:
    """A ``QKeySequence`` for a hotkey setting such as ``"ctrl+alt+t"`` (or a parsed ``Hotkey``).

    Used to *display* global hotkeys next to menu items, where Qt renders them
    natively (``Ctrl+Alt+T`` or ``⌃⌥T``, and as the DBusMenu ``shortcut``
    property on Linux). Returns ``None`` for empty or invalid text. On macOS
    Qt's ``ControlModifier`` is the Command key, so our ``ctrl`` (the physical
    Control key) maps to ``MetaModifier`` and ``meta`` (Command) to
    ``ControlModifier``.
    """
    hotkey = value if isinstance(value, Hotkey) else parse_hotkey_text(value)
    if hotkey is None:
        return None
    key = _qt_key(hotkey.key)
    if key is None:
        return None
    mac = sys.platform == "darwin" if macos is None else macos
    table = {
        "ctrl": Qt.KeyboardModifier.MetaModifier if mac else Qt.KeyboardModifier.ControlModifier,
        "alt": Qt.KeyboardModifier.AltModifier,
        "shift": Qt.KeyboardModifier.ShiftModifier,
        "meta": Qt.KeyboardModifier.ControlModifier if mac else Qt.KeyboardModifier.MetaModifier,
    }
    modifiers = Qt.KeyboardModifier.NoModifier
    for name in hotkey.modifiers:
        modifiers |= table[name]
    return QKeySequence(QKeyCombination(modifiers, key))


# ------------------------------------------------------------------ floating windows
def make_floating(
    widget: QWidget,
    *,
    click_through: bool = False,
    accept_focus: bool = False,
    translucent: bool = True,
) -> None:
    """Configure a top-level widget as a frameless, always-on-top helper window.

    * ``click_through``: mouse input passes to whatever is below (gaze dot).
    * ``accept_focus``: the window may take keyboard focus (privacy curtain);
      otherwise it never activates, so it cannot steal focus from the user's app.
    * ``translucent``: per-pixel alpha background (rounded, see-through corners).

    ``Tool`` keeps the window out of the taskbar and Alt+Tab. On macOS tool
    windows hide whenever the app is inactive, which a menu-bar app always is,
    hence ``WA_MacAlwaysShowToolWindow``.
    """
    flags = (
        Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.Tool
        | Qt.WindowType.NoDropShadowWindowHint
    )
    if click_through:
        flags |= Qt.WindowType.WindowTransparentForInput
    if not accept_focus:
        flags |= Qt.WindowType.WindowDoesNotAcceptFocus
    widget.setWindowFlags(flags)
    widget.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
    if translucent:
        widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        widget.setAutoFillBackground(False)
    if not accept_focus:
        widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
    if click_through:
        widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)


def set_muted(widget: QWidget, opacity: float = 0.62) -> None:
    """Show a widget's text in a softer shade of the current theme's text colour.

    Unlike ``setEnabled(False)`` this stays readable on light and dark themes and
    keeps the text selectable.
    """
    palette = widget.palette()
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text):
        color = palette.color(role)
        color.setAlphaF(max(0.0, min(1.0, opacity)))
        palette.setColor(role, color)
    widget.setPalette(palette)


def paint_surface(
    painter: QPainter,
    rect: QRectF,
    radius: float,
    *,
    color: str = SURFACE,
    alpha: int = SURFACE_ALPHA,
    border_alpha: int = 26,
) -> None:
    """Paint the app's dark translucent rounded surface with a hairline border."""
    path = QPainterPath()
    inner = rect.adjusted(0.5, 0.5, -0.5, -0.5)
    path.addRoundedRect(inner, radius, radius)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.fillPath(path, qcolor(color, alpha))
    if border_alpha > 0:
        painter.setPen(QPen(QColor(255, 255, 255, border_alpha), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
    painter.restore()
