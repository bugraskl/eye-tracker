"""About dialog: version, links, licence, privacy statement and third-party notices."""

from __future__ import annotations

import html
import logging
import platform
import sys

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import Qt, qVersion
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, REPO_URL, __version__
from . import icons, util

log = logging.getLogger(__name__)

__all__ = ["PRIVACY_STATEMENT", "THIRD_PARTY", "AboutDialog", "system_info"]

TAGLINE = "Look at a monitor. Your cursor and keyboard focus follow."
COPYRIGHT = "© 2026 Buğra Şıkel and contributors · MIT License"

PRIVACY_STATEMENT = (
    "Everything runs on this computer. Camera frames are analysed in memory and are "
    "never saved, uploaded or shared. Eye Tracker contains no network code at all. "
    "Only numbers are stored: your settings and the calibration (head-pose and eye "
    "measurements with the positions of the calibration dots). The camera is released "
    "completely in privacy mode, while paused, and while the screen is locked."
)

#: (component, licence, what it is used for, homepage)
THIRD_PARTY: tuple[tuple[str, str, str, str], ...] = (
    ("Qt 6 / PySide6", "LGPL-3.0", "User interface", "https://www.qt.io/qt-for-python"),
    ("OpenCV", "Apache-2.0", "Camera capture and image processing", "https://opencv.org"),
    (
        "MediaPipe Face Landmarker",
        "Apache-2.0",
        "Face landmarks, head pose and iris position (model included)",
        "https://ai.google.dev/edge/mediapipe",
    ),
    (
        "YuNet face detector (OpenCV Zoo)",
        "MIT",
        "Lightweight face detection (model included)",
        "https://github.com/opencv/opencv_zoo",
    ),
    ("NumPy", "BSD-3-Clause", "Numerical computing", "https://numpy.org"),
    ("platformdirs", "MIT", "Per-user folders", "https://github.com/tox-dev/platformdirs"),
    (
        "psutil",
        "BSD-3-Clause",
        "Process and CPU information",
        "https://github.com/giampaolo/psutil",
    ),
    (
        "python-xlib (Linux)",
        "LGPL-2.1-or-later",
        "X11 window focus and hotkeys",
        "https://github.com/python-xlib/python-xlib",
    ),
    (
        "PyObjC (macOS)",
        "MIT",
        "macOS system integration",
        "https://github.com/ronaldoussoren/pyobjc",
    ),
)


def system_info() -> str:
    """One line with the versions that matter for bug reports."""
    parts = [
        f"{APP_NAME} {__version__}",
        f"Python {platform.python_version()}",
        f"Qt {qVersion()} (PySide6 {pyside_version})",
    ]
    cv2 = sys.modules.get("cv2")  # reported only if already loaded; never imported here
    if cv2 is not None:
        parts.append(f"OpenCV {getattr(cv2, '__version__', '?')}")
    app = QGuiApplication.instance()
    if app is not None:
        parts.append(f"{QGuiApplication.platformName()} platform")
    parts.append(f"{platform.system()} {platform.release()} ({platform.machine()})")
    return " · ".join(parts)


def _third_party_html() -> str:
    cell = 'style="padding:3px 12px 3px 0"'
    nowrap = 'style="padding:3px 12px 3px 0; white-space:nowrap"'
    rows = "".join(
        "<tr>"
        f'<td {cell}><a href="{html.escape(url)}">{html.escape(name)}</a></td>'
        f"<td {nowrap}>{html.escape(licence)}</td>"
        f'<td style="padding:3px 0">{html.escape(use)}</td>'
        "</tr>"
        for name, licence, use, url in THIRD_PARTY
    )
    return (
        '<table cellspacing="0" cellpadding="0">'
        '<tr><th align="left">Component</th><th align="left">Licence</th>'
        '<th align="left">Used for</th></tr>'
        f"{rows}</table>"
        "<p>Each component is used under its own licence; the licence texts ship with "
        "the packages. Qt is linked dynamically and can be replaced, as the LGPL requires. "
        "The trained models are redistributed unmodified.</p>"
    )


class AboutDialog(QDialog):
    """Modal "About Eye Tracker" dialog (no controller needed)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setWindowIcon(icons.app_icon())
        self.setObjectName("aboutDialog")
        s = util.ui_scale(self.screen())

        root = QVBoxLayout(self)
        root.setContentsMargins(round(22 * s), round(20 * s), round(22 * s), round(16 * s))
        root.setSpacing(round(14 * s))

        # Header: icon, name, version, tagline, links.
        header = QHBoxLayout()
        header.setSpacing(round(16 * s))
        icon_label = QLabel(self)
        icon_px = round(72 * s)
        icon_label.setPixmap(icons.app_icon().pixmap(icon_px, icon_px))
        icon_label.setFixedSize(icon_px, icon_px)
        icon_label.setAccessibleName(f"{APP_NAME} icon")
        header.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)

        titles = QVBoxLayout()
        titles.setSpacing(round(3 * s))
        name = QLabel(APP_NAME, self)
        font = QFont(name.font())
        font.setPointSizeF(max(font.pointSizeF(), 9.0) * 1.9)
        font.setWeight(QFont.Weight.DemiBold)
        name.setFont(font)
        self.version_label = QLabel(f"Version {__version__}", self)
        self.version_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        util.set_muted(self.version_label)
        tagline = QLabel(TAGLINE, self)
        tagline.setWordWrap(True)
        links = QLabel(
            f'<a href="{REPO_URL}">GitHub</a> &nbsp;·&nbsp; '
            f'<a href="{REPO_URL}/issues">Report an issue</a> &nbsp;·&nbsp; '
            f'<a href="{REPO_URL}/blob/main/LICENSE">MIT License</a>',
            self,
        )
        links.setTextFormat(Qt.TextFormat.RichText)
        links.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        links.setOpenExternalLinks(True)
        for widget in (name, self.version_label, tagline, links):
            titles.addWidget(widget)
        header.addLayout(titles, 1)
        root.addLayout(header)

        # Privacy statement.
        privacy_title = QLabel("Privacy", self)
        bold = QFont(privacy_title.font())
        bold.setWeight(QFont.Weight.DemiBold)
        privacy_title.setFont(bold)
        self.privacy_label = QLabel(PRIVACY_STATEMENT, self)
        self.privacy_label.setWordWrap(True)
        self.privacy_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(privacy_title)
        root.addWidget(self.privacy_label)

        # Third-party notices.
        notices_title = QLabel("Third-party software", self)
        notices_title.setFont(bold)
        self.notices = QTextBrowser(self)
        self.notices.setOpenExternalLinks(True)
        self.notices.setFrameShape(QFrame.Shape.StyledPanel)
        self.notices.setHtml(_third_party_html())
        self.notices.setMinimumHeight(round(170 * s))
        root.addWidget(notices_title)
        root.addWidget(self.notices, 1)

        footer = QLabel(f"{COPYRIGHT}\n{system_info()}", self)
        footer.setWordWrap(True)
        util.set_muted(footer)
        footer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(footer)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.copy_button = QPushButton("Copy system info", self)
        self.copy_button.setToolTip("Copy version details for a bug report")
        buttons.addButton(self.copy_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.copy_button.clicked.connect(self.copy_system_info)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.resize(round(560 * s), round(600 * s))

    def copy_system_info(self) -> None:
        """Put :func:`system_info` on the clipboard."""
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(system_info())
            self.copy_button.setText("Copied ✓")
