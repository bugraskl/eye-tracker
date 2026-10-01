"""First-run assistant: what the app does, camera check, permissions, walk-away action.

Pages::

    Welcome → Camera → Permissions (macOS only) → Walk away → Finish

The camera choice is applied immediately so the live preview and the "face
detected" indicator reflect the selected device (and reverted on Cancel). All
other choices are applied when the user presses Finish, together with
``general.first_run_done = True``. If "Calibrate now" is ticked,
:attr:`FirstRunWizard.calibration_requested` is emitted after the wizard closed.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from .. import APP_NAME
from ..config import Settings, describe_settings
from ..platform import autostart
from ..platform.base import PlatformServices
from ..platform.hotkeys import format_hotkey, parse_hotkey
from ..types import Observation, TrackingState
from .icons import app_icon
from .preview import bgr_to_qimage
from .settings_dialog import (
    CameraProbe,
    camera_in_use,
    platform_from_controller,
    settings_from_controller,
)
from .util import ACCENT, ACCENT_2, DANGER, SUCCESS, WARNING, controller_state, ui_scale

log = logging.getLogger(__name__)

__all__ = [
    "CAMERA_PRIVACY_HELP",
    "PAGE_CAMERA",
    "PAGE_FINISH",
    "PAGE_PERMISSIONS",
    "PAGE_PRESENCE",
    "PAGE_WELCOME",
    "PRESENCE_CHOICES",
    "FirstRunWizard",
    "frame_to_image",
]

PAGE_WELCOME = 0
PAGE_CAMERA = 1
PAGE_PERMISSIONS = 2
PAGE_PRESENCE = 3
PAGE_FINISH = 4

#: Walk-away choices: (value, title, description). ``"off"`` disables presence detection.
PRESENCE_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("lock", "Lock the computer", "Recommended. Unlock as usual when you are back."),
    (
        "lock_and_display_off",
        "Lock and turn the displays off",
        "Also saves power while you are away.",
    ),
    ("display_off", "Turn the displays off", "They come back when you return. Does not lock."),
    ("notify", "Only show a notification", "Nothing else happens."),
    ("off", "Do nothing", "Walk-away detection stays off. You can turn it on later."),
)

_CAPABILITY_FOR = {
    "lock": ("lock",),
    "lock_and_display_off": ("lock", "display_off"),
    "display_off": ("display_off",),
}

# How long a face counts as "detected" after the last observation that had one.
_FACE_FRESH_S = 1.0
_CAMERA_STALE_S = 2.5

#: Where to allow camera access, by ``sys.platform`` (Linux: every other Unix).
CAMERA_PRIVACY_HELP: dict[str, str] = {
    "win32": (
        "If Windows blocks the camera: Settings › Privacy & security › Camera, turn on "
        "“Camera access” and “Let desktop apps access your camera”."
    ),
    "darwin": (
        f"If macOS blocks the camera: System Settings › Privacy & Security › Camera, "
        f"allow {APP_NAME}."
    ),
    "linux": (
        "If your account may not use the camera: add it to the “video” group "
        "(sudo usermod -aG video $USER), then log out and back in."
    ),
}


def _meta(key: str) -> dict[str, Any]:
    return next((row for row in describe_settings() if row["key"] == key), {})


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
    return label


def frame_to_image(frame: object) -> QImage | None:
    """A QImage that owns a copy of a ``uint8`` camera frame; ``None`` if not an image."""
    if not isinstance(frame, np.ndarray):
        return None
    try:
        return bgr_to_qimage(frame)
    except ValueError:
        return None


# ================================================================ small widgets
class _Illustration(QWidget):
    """Two monitors and a cursor hopping to the one being looked at."""

    def __init__(self, scale: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scale = scale
        self.setFixedHeight(round(120 * scale))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event: QPaintEvent) -> None:
        s = self._scale
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = self.palette().color(QPalette.ColorRole.WindowText)
        w = self.width()
        mon_w, mon_h, gap = 150 * s, 88 * s, 22 * s
        x0 = (w - 2 * mon_w - gap) / 2
        y0 = 6 * s
        for n in range(2):
            rect = QRectF(x0 + n * (mon_w + gap), y0, mon_w, mon_h)
            p.setPen(QPen(_alpha(text, 0.55), max(1.5, 2 * s)))
            p.setBrush(_alpha(text, 0.05) if n == 0 else _alpha(QColor(ACCENT), 0.12))
            p.drawRoundedRect(rect, 8 * s, 8 * s)
            stand = QRectF(rect.center().x() - 16 * s, rect.bottom() + 5 * s, 32 * s, 4 * s)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_alpha(text, 0.35))
            p.drawRoundedRect(stand, 2 * s, 2 * s)
        # Gaze arc from the left monitor to the right one.
        start = QPointF(x0 + mon_w * 0.55, y0 + mon_h * 0.45)
        end = QPointF(x0 + mon_w + gap + mon_w * 0.45, y0 + mon_h * 0.45)
        path = QPainterPath(start)
        path.quadTo(QPointF((start.x() + end.x()) / 2, y0 - 2 * s), end)
        grad = QLinearGradient(start, end)
        grad.setColorAt(0, QColor(ACCENT))
        grad.setColorAt(1, QColor(ACCENT_2))
        pen = QPen(QBrush(grad), max(2.0, 3 * s), Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        # Cursor arrow at the destination.
        cx, cy = end.x() + 4 * s, end.y() + 4 * s
        arrow = QPainterPath(QPointF(cx, cy))
        for dx, dy in ((0, 22), (6, 16.5), (10.5, 25), (14, 23.5), (9.5, 15), (17, 15)):
            arrow.lineTo(QPointF(cx + dx * s, cy + dy * s))
        arrow.closeSubpath()
        p.setPen(QPen(QColor(20, 20, 30), max(1.0, 1.2 * s)))
        p.setBrush(QColor(255, 255, 255))
        p.drawPath(arrow)
        p.end()


def _alpha(color: QColor, alpha: float) -> QColor:
    out = QColor(color)
    out.setAlphaF(alpha)
    return out


class _PreviewView(QWidget):
    """Camera preview with rounded corners; frames are kept in memory only."""

    def __init__(self, scale: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scale = scale
        self._image: QImage | None = None
        self._placeholder = "Camera preview"
        self.setMinimumSize(round(320 * scale), round(200 * scale))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, frame: object) -> None:
        image = frame_to_image(frame)
        if image is not None:
            self._image = image
            self.update()

    def clear(self, placeholder: str = "Camera preview") -> None:
        self._image = None
        self._placeholder = placeholder
        self.update()

    @property
    def has_image(self) -> bool:
        return self._image is not None

    def paintEvent(self, _event: QPaintEvent) -> None:
        s = self._scale
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        radius = 12 * s
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        p.setClipPath(clip)
        p.fillRect(rect, QColor(12, 14, 22))
        if self._image is not None and not self._image.isNull():
            size = self._image.size().scaled(
                rect.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio
            )
            target = QRectF(
                rect.x() + (rect.width() - size.width()) / 2,
                rect.y() + (rect.height() - size.height()) / 2,
                size.width(),
                size.height(),
            )
            p.drawImage(target, self._image)
        else:
            p.setPen(QColor(156, 163, 175))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder)
        p.setClipping(False)
        p.setPen(QPen(QColor(255, 255, 255, 30), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, radius, radius)
        p.end()


# ======================================================================= pages
class _WelcomePage(QWizardPage):
    def __init__(self, wizard: FirstRunWizard) -> None:
        super().__init__()
        s = wizard.scale
        self.setTitle(f"Welcome to {APP_NAME}")
        self.setSubTitle("Look at a monitor and your cursor — and keyboard focus — follow.")
        layout = QVBoxLayout(self)
        layout.setSpacing(round(12 * s))
        layout.addWidget(_Illustration(s))
        intro = QLabel(
            f"{APP_NAME} uses your webcam to notice which screen you are looking at and moves "
            "the mouse cursor there, so you can read on one monitor and start typing on the "
            "other without reaching for the mouse. It can also lock the computer when you "
            "walk away."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        promises = QFrame()
        promises.setObjectName("promises")
        promises.setStyleSheet(
            "QFrame#promises { background: rgba(99, 102, 241, 0.08);"
            " border: 1px solid rgba(99, 102, 241, 0.30);"
            f" border-radius: {round(10 * s)}px; }}"
        )
        grid = QVBoxLayout(promises)
        grid.setContentsMargins(round(16 * s), round(12 * s), round(16 * s), round(12 * s))
        grid.setSpacing(round(6 * s))
        for text in (
            "Runs entirely on this computer — no account, no internet connection.",
            "Camera frames are analysed in memory and never saved or sent anywhere.",
            "Privacy mode switches the camera off completely, from the tray or a hotkey.",
        ):
            row = QLabel(f"<span style='color:{SUCCESS}'>✓</span>&nbsp;&nbsp;{text}")
            row.setTextFormat(Qt.TextFormat.RichText)
            row.setWordWrap(True)
            grid.addWidget(row)
        layout.addWidget(promises)
        layout.addStretch(1)


class _CameraPage(QWizardPage):
    def __init__(self, wizard: FirstRunWizard) -> None:
        super().__init__()
        self._wizard = wizard
        s = wizard.scale
        self.setTitle("Camera")
        self.setSubTitle("Sit as you usually do and check that your face is detected.")
        layout = QVBoxLayout(self)
        layout.setSpacing(round(10 * s))

        row = QHBoxLayout()
        row.addWidget(QLabel("Camera:"))
        self.device = QComboBox()
        self.device.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self.device, 1)
        self.detect = QPushButton("Detect cameras")
        self.detect.setToolTip("Look for connected cameras (each may briefly turn on).")
        row.addWidget(self.detect)
        layout.addLayout(row)

        self.preview = _PreviewView(s)
        layout.addWidget(self.preview, 1)

        self.status = QLabel()
        font = QFont(self.status.font())
        font.setPointSizeF(font.pointSizeF() * 1.15)
        font.setWeight(QFont.Weight.DemiBold)
        self.status.setFont(font)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        # Shown only when the camera cannot be opened: the OS privacy switch is
        # the usual culprit, and not something "try another camera" would fix.
        self.help = _muted(CAMERA_PRIVACY_HELP.get(sys.platform, CAMERA_PRIVACY_HELP["linux"]))
        self.help.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.help.setVisible(False)
        layout.addWidget(self.help)
        row = QHBoxLayout()
        row.addStretch(1)
        self.privacy_settings = QPushButton("Open camera privacy settings")
        self.privacy_settings.clicked.connect(wizard.open_camera_privacy_settings)
        self.privacy_settings.setVisible(False)
        row.addWidget(self.privacy_settings)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(
            _muted(
                "The preview is shown only here and is never saved. The camera light may "
                "stay on while Eye Tracker is running; privacy mode turns it off."
            )
        )


class _PermissionsPage(QWizardPage):
    def __init__(self, wizard: FirstRunWizard) -> None:
        super().__init__()
        self._wizard = wizard
        s = wizard.scale
        self.setTitle("Permissions")
        self.setSubTitle("macOS asks you to allow a few things once.")
        layout = QVBoxLayout(self)
        layout.setSpacing(round(14 * s))
        grid = QGridLayout()
        grid.setHorizontalSpacing(round(12 * s))
        grid.setVerticalSpacing(round(6 * s))
        self.status: dict[str, QLabel] = {}
        self.allow: dict[str, QPushButton] = {}
        rows = (
            (
                "camera",
                "Camera",
                "Needed to see where you are looking.",
            ),
            (
                "accessibility",
                "Accessibility",
                "Lets Eye Tracker give keyboard focus to the window on the screen you look "
                "at. Without it only the cursor moves.",
            ),
        )
        for n, (name, title, text) in enumerate(rows):
            heading = QLabel(f"<b>{title}</b>")
            grid.addWidget(heading, n * 2, 0)
            status = QLabel()
            status.setWordWrap(True)
            self.status[name] = status
            grid.addWidget(status, n * 2, 1)
            allow = QPushButton("Allow…")
            allow.clicked.connect(lambda _=False, p=name: wizard.request_permission(p))
            self.allow[name] = allow
            grid.addWidget(allow, n * 2, 2)
            settings = QPushButton("Open Settings")
            settings.clicked.connect(lambda _=False, p=name: wizard.open_permission_settings(p))
            grid.addWidget(settings, n * 2, 3)
            grid.addWidget(_muted(text), n * 2 + 1, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addWidget(
            _muted("You can change these later in System Settings → Privacy & Security.")
        )
        layout.addStretch(1)


class _PresencePage(QWizardPage):
    def __init__(self, wizard: FirstRunWizard) -> None:
        super().__init__()
        s = wizard.scale
        self.setTitle("When you walk away")
        self.setSubTitle("Eye Tracker notices when you leave. What should happen?")
        layout = QVBoxLayout(self)
        layout.setSpacing(round(4 * s))
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QRadioButton] = {}
        caps = wizard.capabilities
        for value, title, description in PRESENCE_CHOICES:
            button = QRadioButton(title)
            if any(not caps.get(c, True) for c in _CAPABILITY_FOR.get(value, ())):
                button.setText(f"{title} (may not work on this system)")
            self.group.addButton(button)
            self.buttons[value] = button
            layout.addWidget(button)
            if description:
                hint = _muted(description)
                hint.setContentsMargins(round(26 * s), 0, 0, round(6 * s))
                layout.addWidget(hint)
        layout.addSpacing(round(10 * s))
        row = QHBoxLayout()
        row.addWidget(QLabel("After"))
        self.timeout = QSpinBox()
        meta = _meta("presence.away_timeout_s")
        self.timeout.setRange(int(meta.get("lo") or 5), int(meta.get("hi") or 3600))
        self.timeout.setSingleStep(5)
        self.timeout.setSuffix(" seconds")
        self.timeout.setToolTip(str(meta.get("doc") or ""))
        row.addWidget(self.timeout)
        row.addWidget(QLabel("without a face or any input"))
        row.addStretch(1)
        layout.addLayout(row)
        self.countdown = _muted("")
        layout.addWidget(self.countdown)
        layout.addStretch(1)
        for button in self.buttons.values():
            button.toggled.connect(self._update_enabled)

    def _update_enabled(self) -> None:
        off = self.buttons["off"].isChecked()
        self.timeout.setEnabled(not off)


class _FinishPage(QWizardPage):
    def __init__(self, wizard: FirstRunWizard) -> None:
        super().__init__()
        s = wizard.scale
        self.setTitle("You're all set")
        self.setSubTitle(_finish_subtitle())
        layout = QVBoxLayout(self)
        layout.setSpacing(round(10 * s))
        self.calibrate = QCheckBox("Calibrate now (recommended, about 30 seconds)")
        self.calibrate.setChecked(True)
        self.calibrate.setToolTip(
            "Switching needs a short calibration: you look at a few dots on each screen."
        )
        layout.addWidget(self.calibrate)
        #: Shown instead of a recommendation when there is nothing to switch between.
        self.one_monitor = _muted(
            "With one monitor there is nothing to switch between, so no calibration is "
            "needed: walk-away detection, privacy mode and the shoulder guard work as they "
            "are. Connect a second monitor and calibrate from the tray menu."
        )
        self.one_monitor.setVisible(False)
        layout.addWidget(self.one_monitor)
        self.autostart = QCheckBox(f"Start {APP_NAME} when I log in")
        layout.addWidget(self.autostart)
        layout.addSpacing(round(10 * s))
        self.shortcuts = QLabel()
        self.shortcuts.setTextFormat(Qt.TextFormat.RichText)
        self.shortcuts.setWordWrap(True)
        layout.addWidget(self.shortcuts)
        layout.addStretch(1)
        layout.addWidget(_muted("Everything can be changed later under Settings."))


def _finish_subtitle() -> str:
    """Where the app lives from now on, and how its icon is used there."""
    if sys.platform == "darwin":
        return f"{APP_NAME} keeps running in the menu bar: click the eye icon for its menu."
    if sys.platform == "win32":
        # Windows 11 puts new tray icons in the hidden overflow area.
        return (
            f"{APP_NAME} keeps running in the system tray: click the eye icon for its menu. "
            "If you don't see it, click ^ on the taskbar."
        )
    return (
        f"{APP_NAME} keeps running in the system tray: right-click the eye icon for its "
        "menu, double-click it for the settings."
    )


# ====================================================================== wizard
class FirstRunWizard(QWizard):
    """First-run assistant.

    Args:
        controller: The app controller (``settings``, ``apply_settings``,
            ``set_preview``, ``observation``/``preview_frame``/``state_changed``
            signals, ``platform``).
        parent: Optional parent widget.
        show_permissions: Show the macOS permissions page. ``None`` (default)
            shows it on macOS only.
        clock: Monotonic clock used for the face indicator (tests inject one).

    Signals:
        calibration_requested: Emitted after Finish when "Calibrate now" was ticked.
    """

    calibration_requested = Signal()

    def __init__(
        self,
        controller: object,
        parent: QWidget | None = None,
        *,
        show_permissions: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._clock = clock
        self._platform: PlatformServices = platform_from_controller(controller)
        try:
            self.capabilities: dict[str, bool] = dict(self._platform.capabilities())
        except Exception:
            self.capabilities = {}
        self._show_permissions = (
            sys.platform == "darwin" if show_permissions is None else bool(show_permissions)
        )
        self.scale = ui_scale(self.screen() or QGuiApplication.primaryScreen())
        settings = settings_from_controller(controller)
        self._original_device = settings.camera.device
        self._device = settings.camera.device
        self._preview_on = False
        self._last_obs_at: float | None = None
        self._last_face_at: float | None = None
        self._tracking_state = controller_state(controller)
        self._wants_calibration = False
        self._autostart_error: str | None = None

        self.setWindowTitle(f"{APP_NAME} setup")
        self.setWindowIcon(app_icon())
        self.setWizardStyle(
            QWizard.WizardStyle.MacStyle
            if sys.platform == "darwin"
            else QWizard.WizardStyle.ModernStyle
        )
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setPixmap(QWizard.WizardPixmap.LogoPixmap, app_icon().pixmap(round(48 * self.scale)))
        self.setButtonText(QWizard.WizardButton.FinishButton, "Finish")
        self.setMinimumSize(round(640 * self.scale), round(560 * self.scale))

        self.welcome_page = _WelcomePage(self)
        self.camera_page = _CameraPage(self)
        self.permissions_page = _PermissionsPage(self)
        self.presence_page = _PresencePage(self)
        self.finish_page = _FinishPage(self)
        self.setPage(PAGE_WELCOME, self.welcome_page)
        self.setPage(PAGE_CAMERA, self.camera_page)
        self.setPage(PAGE_PERMISSIONS, self.permissions_page)
        self.setPage(PAGE_PRESENCE, self.presence_page)
        self.setPage(PAGE_FINISH, self.finish_page)
        self.setStartId(PAGE_WELCOME)

        self._probe = CameraProbe(self)
        self._probe.finished.connect(self._on_cameras_detected)
        self.camera_page.detect.clicked.connect(self.detect_cameras)
        self.camera_page.device.activated.connect(self._on_device_chosen)
        self._fill_devices([])
        self._load_presence(settings)
        self._load_finish(settings)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._refresh_status)
        self.currentIdChanged.connect(self._on_page_changed)
        for name in ("observation", "preview_frame", "state_changed"):
            signal = getattr(controller, name, None)
            if signal is not None:
                slot = {
                    "observation": self._on_observation,
                    "preview_frame": self._on_preview_frame,
                    "state_changed": self._on_state_changed,
                }[name]
                signal.connect(slot)
        self._refresh_status()
        self._refresh_permissions()

    # ============================================================== public API
    def nextId(self) -> int:
        """Skip the permissions page where it does not apply."""
        if self.currentId() == PAGE_CAMERA and not self._show_permissions:
            return PAGE_PRESENCE
        return super().nextId()

    @property
    def wants_calibration(self) -> bool:
        """Whether the user asked to calibrate right after the wizard (set on Finish)."""
        return self._wants_calibration

    @property
    def autostart_error(self) -> str | None:
        return self._autostart_error

    def chosen_presence(self) -> str:
        """The selected walk-away choice (a ``PRESENCE_CHOICES`` value)."""
        for value, button in self.presence_page.buttons.items():
            if button.isChecked():
                return value
        return "lock"

    def set_presence_choice(self, value: str) -> None:
        button = self.presence_page.buttons.get(value)
        if button is not None:
            button.setChecked(True)

    def final_settings(self) -> Settings:
        """The settings Finish applies (current controller settings plus the choices)."""
        settings = settings_from_controller(self._controller)
        settings.camera.device = self._device
        choice = self.chosen_presence()
        if choice == "off":
            settings.presence.enabled = False
        else:
            settings.presence.enabled = True
            settings.presence.action = choice
            settings.presence.away_timeout_s = int(self.presence_page.timeout.value())
        settings.general.first_run_done = True
        return settings

    def detect_cameras(self) -> None:
        """Probe for cameras in the background and refresh the device list."""
        settings = settings_from_controller(self._controller)
        # The camera behind the live preview is not probed (see camera_in_use).
        if self._probe.start(settings.camera.api, skip=camera_in_use(self._controller)):
            self.camera_page.detect.setEnabled(False)
            self.camera_page.detect.setText("Detecting…")

    def select_device(self, device: str) -> None:
        """Switch to another camera (applied immediately)."""
        device = device.strip()
        if not device or device == self._device:
            return
        self._device = device
        self._fill_devices(self._known_cameras())
        self._apply_device(device)

    def request_permission(self, name: str) -> None:
        try:
            self._platform.request_permission(name)
        except Exception:
            log.warning("Requesting the %s permission failed", name, exc_info=True)
        self._refresh_permissions()

    def open_permission_settings(self, name: str) -> bool:
        """Open the OS settings page for a permission. Returns whether one opened."""
        try:
            return bool(self._platform.open_permission_settings(name))
        except Exception:
            log.warning("Opening the %s settings failed", name, exc_info=True)
            return False

    def open_camera_privacy_settings(self) -> None:
        """The camera page's button: the OS page, or the steps where there is none."""
        if not self.open_permission_settings("camera"):
            # No settings page to open (Linux): the help text says what to do.
            self.camera_page.help.setVisible(True)
            self.camera_page.help.setStyleSheet(f"color: {WARNING};")

    # ============================================================== QWizard
    def accept(self) -> None:
        settings = self.final_settings()
        try:
            self._controller.apply_settings(settings)  # type: ignore[attr-defined]
        except Exception:
            log.exception("Applying the first-run settings failed")
        self._apply_autostart(self.finish_page.autostart.isChecked())
        self._wants_calibration = self.finish_page.calibrate.isChecked()
        self._stop_preview()
        super().accept()
        log.info("First-run setup finished (calibrate now: %s)", self._wants_calibration)
        if self._wants_calibration:
            self.calibration_requested.emit()

    def reject(self) -> None:
        self._stop_preview()
        if self._device != self._original_device:
            # The camera choice was applied live; Cancel restores the previous one.
            self._device = self._original_device
            self._apply_device(self._original_device)
        super().reject()

    # ============================================================== pages
    def _monitor_count(self) -> int | None:
        """How many monitors the controller sees (``None`` if it cannot say)."""
        monitors = getattr(self._controller, "monitors", None)
        try:
            return len(monitors()) if callable(monitors) else None
        except Exception:
            log.debug("monitors() failed", exc_info=True)
            return None

    def _load_presence(self, settings: Settings) -> None:
        page = self.presence_page
        choice = settings.presence.action if settings.presence.enabled else "off"
        if choice == "none":
            choice = "off"
        (page.buttons.get(choice) or page.buttons["lock"]).setChecked(True)
        page.timeout.setValue(int(settings.presence.away_timeout_s))
        warning = int(settings.presence.warning_s)
        # Like CountdownToast.hint_text: input cancels the countdown only where the
        # settings count it and the system reports it (not on Wayland outside GNOME).
        if settings.presence.require_input_idle and self.capabilities.get("input_idle", True):
            cancel = "move the mouse or look at the camera"
        else:
            cancel = "look at the camera"
        page.countdown.setText(
            f"A {warning}-second countdown comes first: {cancel} to cancel it."
            if warning
            else "The action happens without a countdown."
        )

    def _load_finish(self, settings: Settings) -> None:
        page = self.finish_page
        try:
            supported = bool(autostart.is_supported())
            enabled = bool(autostart.is_enabled()) if supported else False
        except Exception:
            supported, enabled = False, False
        page.autostart.setChecked(enabled)
        page.autostart.setEnabled(supported)
        # A calibration only serves switching: with one monitor it would change
        # nothing, so it is neither ticked nor recommended there.
        single = self._monitor_count() == 1
        page.calibrate.setChecked(not single and settings.switching.enabled)
        page.one_monitor.setVisible(single)
        rows = []
        labels = {
            "toggle_tracking": "Pause / resume",
            "toggle_privacy": "Privacy mode",
            "recalibrate": "Recalibrate",
        }
        for field, label in labels.items():
            text = getattr(settings.hotkeys, field, "")
            try:
                shown = format_hotkey(parse_hotkey(text))
            except ValueError:
                continue
            rows.append(f"{label}: <b>{shown}</b>")
        if settings.hotkeys.enabled and rows and self.capabilities.get("hotkeys", True):
            page.shortcuts.setText("Shortcuts — " + " · ".join(rows))
        else:
            page.shortcuts.setText("")

    def _on_page_changed(self, page_id: int) -> None:
        if page_id == PAGE_CAMERA:
            self._start_preview()
            self._status_timer.start()
        else:
            self._stop_preview()
            self._status_timer.stop()
        if page_id == PAGE_PERMISSIONS:
            self._refresh_permissions()
            self._status_timer.start()

    # ============================================================== camera
    def _start_preview(self) -> None:
        if self._preview_on:
            return
        self._preview_on = True
        # One owner among others (the preview window): see Controller.set_preview.
        self._call_controller("set_preview", True, self)

    def _stop_preview(self) -> None:
        if not self._preview_on:
            return
        self._preview_on = False
        self._call_controller("set_preview", False, self)
        self.camera_page.preview.clear()

    def _on_preview_frame(self, frame: object) -> None:
        if self._preview_on:
            self.camera_page.preview.set_frame(frame)

    def _on_observation(self, obs: object) -> None:
        if not isinstance(obs, Observation):
            return
        now = self._clock()
        self._last_obs_at = now
        if self._tracking_state == TrackingState.CAMERA_ERROR:
            # Frames arrive again: the controller clears the error with its next
            # state change, but the indicator need not wait for it.
            self._tracking_state = TrackingState.STARTING
        if obs.face_count > 0:
            self._last_face_at = now
        if self.currentId() == PAGE_CAMERA:
            self._refresh_status()

    def _on_state_changed(self, state: object) -> None:
        if isinstance(state, TrackingState):
            self._tracking_state = state
        self._refresh_status()

    def face_status(self) -> str:
        """What the camera indicator shows.

        ``"face"`` (a face was seen within the last second), ``"no_face"``,
        ``"waiting"`` (no frames yet), ``"error"`` (the camera cannot be opened),
        ``"blocked"`` (it cannot be opened and the OS reports that camera access
        is denied), ``"off"`` (paused or privacy mode) or ``"busy"`` (another app
        has the camera).
        """
        state = self._tracking_state
        if state == TrackingState.CAMERA_ERROR:
            return "blocked" if self._camera_permission() is False else "error"
        if state == TrackingState.YIELDED:
            return "busy"
        if not state.camera_active:
            return "off"
        now = self._clock()
        if self._last_obs_at is None or now - self._last_obs_at > _CAMERA_STALE_S:
            return "waiting"
        if self._last_face_at is not None and now - self._last_face_at <= _FACE_FRESH_S:
            return "face"
        return "no_face"

    def _refresh_status(self) -> None:
        status = self.face_status()
        text, color = {
            "face": ("✓  Face detected", SUCCESS),
            "no_face": ("✗  No face detected — sit in front of the camera", WARNING),
            "waiting": ("Starting the camera…", None),
            "error": (
                "✗  The camera could not be opened — another app may be using it, or the "
                "system's privacy settings may block it",
                DANGER,
            ),
            "blocked": ("✗  Camera access is blocked in the system's privacy settings", DANGER),
            "off": ("The camera is off — tracking is paused or in privacy mode", WARNING),
            "busy": ("The camera is in use by another app — close it to continue", WARNING),
        }[status]
        page = self.camera_page
        label = page.status
        if label.text() != text:
            label.setText(text)
            label.setStyleSheet(f"color: {color};" if color else "")
        failed = status in ("error", "blocked")
        page.help.setVisible(failed)
        page.privacy_settings.setVisible(failed)
        if self.currentId() == PAGE_PERMISSIONS:
            self._refresh_permissions()

    def _camera_permission(self) -> bool | None:
        """The OS camera permission (``None``: unknown or not applicable)."""
        try:
            value = dict(self._platform.permissions()).get("camera")
        except Exception:
            log.debug("permissions() failed", exc_info=True)
            return None
        return value if isinstance(value, bool) else None

    def _known_cameras(self) -> list[str]:
        combo = self.camera_page.device
        return [str(combo.itemData(i)) for i in range(combo.count())]

    def _fill_devices(self, devices: list[Any]) -> None:
        combo = self.camera_page.device
        combo.blockSignals(True)
        try:
            combo.clear()
            seen: set[str] = set()
            for item in devices:
                if isinstance(item, str):
                    value, label = item, (f"Camera {item}" if item.isdigit() else item)
                else:
                    index = getattr(item, "index", None)
                    if index is None:
                        continue
                    value = str(index)
                    w, h = getattr(item, "width", 0), getattr(item, "height", 0)
                    label = getattr(item, "name", f"Camera {index}")
                    if w and h:
                        label += f" — {w}×{h}"
                if value in seen:
                    continue
                seen.add(value)
                combo.addItem(label, value)
            if self._device not in seen:
                label = f"Camera {self._device}" if self._device.isdigit() else self._device
                combo.addItem(label, self._device)
            combo.setCurrentIndex(combo.findData(self._device))
        finally:
            combo.blockSignals(False)

    def _on_cameras_detected(self, cameras: object) -> None:
        page = self.camera_page
        page.detect.setEnabled(True)
        page.detect.setText("Detect cameras")
        self._fill_devices(list(cameras) if isinstance(cameras, list) else [])

    def _on_device_chosen(self, index: int) -> None:
        value = self.camera_page.device.itemData(index)
        if isinstance(value, str):
            self.select_device(value)

    def _apply_device(self, device: str) -> None:
        settings = settings_from_controller(self._controller)
        if settings.camera.device == device:
            return
        settings.camera.device = device
        self._last_obs_at = self._last_face_at = None
        self.camera_page.preview.clear("Switching camera…")
        try:
            self._controller.apply_settings(settings)  # type: ignore[attr-defined]
        except Exception:
            log.exception("Switching the camera failed")
        self._refresh_status()

    # ============================================================== permissions
    def _refresh_permissions(self) -> None:
        try:
            perms = dict(self._platform.permissions())
        except Exception:
            perms = {}
        stale = self._accessibility_status() == "stale"
        for name, label in self.permissions_page.status.items():
            value = perms.get(name)
            if value is True:
                text, color = "✓ Allowed", SUCCESS
            elif name == "accessibility" and stale:
                # Granted to an earlier build: the toggle looks on but no longer applies.
                text = "✗ Granted to an earlier version — remove it with “−” and add it again"
                color = DANGER
            elif value is False:
                text, color = "✗ Not allowed yet", DANGER
            else:
                text, color = "Not decided yet", None
            if label.text() != text:
                label.setText(text)
                label.setStyleSheet(f"color: {color};" if color else "")
            # Nothing left to ask for once granted.
            self.permissions_page.allow[name].setEnabled(value is not True)

    def _accessibility_status(self) -> str:
        """``platform.accessibility_status()`` ("granted", "missing", "stale", "unknown").

        Only macOS implements it; elsewhere, or when it fails, ``"unknown"``.
        """
        try:
            return str(self._platform.accessibility_status())
        except Exception:
            log.debug("accessibility_status() failed", exc_info=True)
            return "unknown"

    # ============================================================== misc
    def _apply_autostart(self, wanted: bool) -> None:
        if not self.finish_page.autostart.isEnabled():
            return  # unsupported here: nothing to change
        try:
            if wanted == bool(autostart.is_enabled()):
                return
            if wanted:
                autostart.enable()
            else:
                autostart.disable()
        except autostart.AutostartError as exc:
            log.warning("Changing start at login failed: %s", exc)
            self._autostart_error = str(exc)
            notify = getattr(self._controller, "notify", None)
            if notify is not None:
                try:
                    notify.emit("Start at login", str(exc))
                except Exception:
                    log.debug("Could not emit notify", exc_info=True)

    def _call_controller(self, name: str, *args: object) -> None:
        method = getattr(self._controller, name, None)
        if method is None:
            return
        try:
            method(*args)
        except Exception:
            log.warning("controller.%s failed", name, exc_info=True)
