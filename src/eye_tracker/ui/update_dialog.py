"""The update window: checking, what is new, downloading, installing.

It only shows the state of the :class:`~eye_tracker.update.service.UpdateService`
and sends the user's choices back to it; closing it never stops a running check
(a download has its own Cancel button).
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__
from ..update.release import RELEASES_PAGE
from ..update.service import Phase, UpdateService, UpdateState
from . import icons


class UpdateDialog(QDialog):
    """One window for every phase of an update (see the module docstring)."""

    def __init__(self, service: UpdateService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = service
        self.setWindowTitle(f"{APP_NAME} updates")
        self.setWindowIcon(icons.app_icon())
        self.setMinimumWidth(440)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        self._title = QLabel()
        font = self._title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self._title.setFont(font)
        self._title.setWordWrap(True)
        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setOpenExternalLinks(True)
        self._detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._bar = QProgressBar()
        self._bar.setTextVisible(True)
        layout.addWidget(self._title)
        layout.addWidget(self._detail)
        layout.addWidget(self._bar)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._primary = QPushButton()
        self._primary.setDefault(True)
        self._primary.clicked.connect(self._on_primary)
        self._secondary = QPushButton()
        self._secondary.clicked.connect(self._on_secondary)
        buttons.addWidget(self._secondary)
        buttons.addWidget(self._primary)
        layout.addLayout(buttons)

        self._primary_action: Callable[[], object] = self.close
        self._secondary_action: Callable[[], object] = self.close
        service.state_changed.connect(self._show_state)
        self._show_state(service.state)

    # ------------------------------------------------------------------ public
    def present(self) -> None:
        """Show the window; look for an update unless one is known or being fetched."""
        state = self._service.state
        known = state.phase is Phase.FAILED and state.release is not None
        if state.phase in (Phase.IDLE, Phase.UP_TO_DATE, Phase.FAILED) and not known:
            self._service.check_now()
        self.show()
        self.raise_()
        self.activateWindow()

    # ----------------------------------------------------------------- display
    def _show_state(self, state: object) -> None:
        if not isinstance(state, UpdateState):
            return
        release = state.release
        self._bar.setVisible(False)
        self._secondary.setVisible(False)
        self._primary.setVisible(True)
        self._primary.setEnabled(True)
        self._primary.setText("Close")
        self._primary_action = self.close

        if state.phase is Phase.CHECKING:
            self._text("Checking for updates…", "Asking GitHub for the latest release.")
            self._busy_bar()
        elif state.phase is Phase.UP_TO_DATE:
            self._text("You have the latest version", f"{APP_NAME} {__version__} is up to date.")
        elif state.phase is Phase.AVAILABLE and release is not None:
            notes = f'<a href="{release.page_url}">What is new in {release.version}</a>'
            if self._service.can_install:
                self._text(
                    f"Version {release.version} is available",
                    f"You have {__version__}. {APP_NAME} will download the new setup, check it "
                    f"and restart itself.<br>{notes}",
                )
                self._primary.setText("Install and restart")
                self._primary_action = self._service.install
                self._secondary.setText("Later")
                self._secondary.setVisible(True)
                self._secondary_action = self.close
            else:
                self._text(
                    f"Version {release.version} is available",
                    f"You have {__version__}. This copy cannot update itself; download the new "
                    f"version from the releases page.<br>{notes}",
                )
                self._primary.setText("Open the download page")
                self._primary_action = self._open_releases
                self._secondary.setText("Close")
                self._secondary.setVisible(True)
                self._secondary_action = self.close
        elif state.phase is Phase.DOWNLOADING:
            self._text(f"Downloading {release.version if release else ''}…", "")
            self._progress(state)
            self._primary.setText("Cancel")
            self._primary_action = self._service.cancel
        elif state.phase is Phase.INSTALLING:
            self._text("Installing…", f"{APP_NAME} will close and start again in a moment.")
            self._busy_bar()
            self._primary.setVisible(False)
        elif state.phase is Phase.FAILED:
            self._text("The update did not work", state.message)
            retry = release is not None and self._service.can_install
            self._primary.setText("Try again")
            self._primary_action = self._service.install if retry else self._service.check_now
            self._secondary.setText("Close")
            self._secondary.setVisible(True)
            self._secondary_action = self.close
        else:
            self._text("", "")

    def _text(self, title: str, detail: str) -> None:
        self._title.setText(title)
        self._detail.setText(detail)
        self._detail.setVisible(bool(detail))

    def _busy_bar(self) -> None:
        self._bar.setRange(0, 0)
        self._bar.setVisible(True)

    def _progress(self, state: UpdateState) -> None:
        if state.total > 0:
            self._bar.setRange(0, state.total)
            self._bar.setValue(min(state.done, state.total))
            self._bar.setFormat(f"{state.done / 1e6:.1f} / {state.total / 1e6:.1f} MB")
        else:
            self._bar.setRange(0, 0)
        self._bar.setVisible(True)

    # ------------------------------------------------------------------ events
    def _on_primary(self) -> None:
        self._primary_action()

    def _on_secondary(self) -> None:
        self._secondary_action()

    def _open_releases(self) -> None:
        release = self._service.state.release
        QDesktopServices.openUrl(QUrl(release.page_url if release else RELEASES_PAGE))
