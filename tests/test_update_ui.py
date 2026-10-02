"""Tests for the update window, the tray entry, the settings switch and the app wiring."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from eye_tracker import REPO_URL
from eye_tracker import app as app_module
from eye_tracker.ui import tray as tray_module
from eye_tracker.ui.update_dialog import UpdateDialog
from eye_tracker.update import installer
from eye_tracker.update import service as svc
from eye_tracker.update.fetch import Fetcher
from eye_tracker.update.release import Asset, ReleaseInfo, download_url
from eye_tracker.update.service import Phase, UpdateService, UpdateState
from test_app import TWO_MONITORS, Harness, _tray_messages, build, settle  # noqa: F401
from test_ui_basic import FakeController, _tray, cleanup, controller  # noqa: F401
from test_ui_dialogs import dialog  # noqa: F401
from test_update_fetch import FakeTransport

TAG = "v0.2.2"
NAME = "EyeTracker-0.2.2-windows-x64-setup.exe"


def release(*, installable: bool = True) -> ReleaseInfo:
    return ReleaseInfo(
        version="0.2.2",
        tag=TAG,
        page_url=f"{REPO_URL}/releases/tag/{TAG}",
        installer=Asset(NAME, download_url(TAG, NAME), 100) if installable else None,
        checksums=Asset("SHA256SUMS.txt", download_url(TAG, "SHA256SUMS.txt"), 50)
        if installable
        else None,
    )


@pytest.fixture
def updates(qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setattr(svc, "supported", lambda: True)
    monkeypatch.setattr(installer, "updates_dir", lambda: tmp_path / "updates")
    made: list[UpdateService] = []

    def make(kind: str = installer.KIND_SETUP) -> UpdateService:
        service = UpdateService(
            fetcher_factory=lambda: Fetcher(FakeTransport()),
            memory_path=tmp_path / "updates.json",
            kind=kind,
        )
        made.append(service)
        return service

    yield make
    for service in made:
        service.shutdown()
        service.deleteLater()


def shown(dialog_: UpdateDialog, name: str) -> bool:
    widget = {"primary": dialog_._primary, "secondary": dialog_._secondary, "bar": dialog_._bar}[
        name
    ]
    return widget.isVisibleTo(dialog_)


# ------------------------------------------------------------------------ the window
def test_an_installable_update_offers_install_and_later(updates: Callable[..., Any]) -> None:
    service = updates()
    service._state = UpdateState(Phase.AVAILABLE, release())
    dialog_ = UpdateDialog(service)
    assert "0.2.2" in dialog_._title.text()
    assert dialog_._primary.text() == "Install and restart"
    assert dialog_._secondary.text() == "Later"
    assert shown(dialog_, "primary")
    assert shown(dialog_, "secondary")
    assert REPO_URL in dialog_._detail.text()  # the link to what is new


def test_a_copy_that_cannot_update_itself_points_to_the_page(updates: Callable[..., Any]) -> None:
    service = updates(kind=installer.KIND_PORTABLE)
    service._state = UpdateState(Phase.AVAILABLE, release())
    dialog_ = UpdateDialog(service)
    assert dialog_._primary.text() == "Open the download page"
    assert "cannot update itself" in dialog_._detail.text()


def test_a_release_without_a_verifiable_installer_is_not_installed(
    updates: Callable[..., Any],
) -> None:
    service = updates()
    service._state = UpdateState(Phase.AVAILABLE, release(installable=False))
    dialog_ = UpdateDialog(service)
    assert dialog_._primary.text() == "Open the download page"


def test_the_download_shows_progress_and_can_be_cancelled(updates: Callable[..., Any]) -> None:
    service = updates()
    service._state = UpdateState(Phase.AVAILABLE, release())
    dialog_ = UpdateDialog(service)
    dialog_.show()
    service._set(UpdateState(Phase.DOWNLOADING, release(), done=5_000_000, total=20_000_000))
    assert shown(dialog_, "bar")
    assert dialog_._bar.maximum() == 20_000_000
    assert dialog_._bar.value() == 5_000_000
    assert "5.0 / 20.0 MB" in dialog_._bar.format()
    assert dialog_._primary.text() == "Cancel"
    service.cancel()
    assert service._cancel.is_set()


def test_installing_has_no_buttons_and_failing_offers_another_try(
    updates: Callable[..., Any],
) -> None:
    service = updates()
    service._state = UpdateState(Phase.AVAILABLE, release())
    dialog_ = UpdateDialog(service)
    dialog_.show()
    service._set(UpdateState(Phase.INSTALLING, release()))
    assert not shown(dialog_, "primary")
    service._set(UpdateState(Phase.FAILED, release(), "The download does not match its checksum."))
    assert "checksum" in dialog_._detail.text()
    assert dialog_._primary.text() == "Try again"
    assert shown(dialog_, "secondary")


def test_checking_and_up_to_date(updates: Callable[..., Any]) -> None:
    service = updates()
    dialog_ = UpdateDialog(service)
    dialog_.show()
    service._set(UpdateState(Phase.CHECKING))
    assert shown(dialog_, "bar")
    assert dialog_._bar.maximum() == 0  # busy indicator
    service._set(UpdateState(Phase.UP_TO_DATE, release()))
    assert not shown(dialog_, "bar")
    assert "up to date" in dialog_._detail.text()
    assert dialog_._primary.text() == "Close"


def test_presenting_looks_for_an_update_only_when_none_is_known(
    updates: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    service = updates()
    asked: list[bool] = []
    monkeypatch.setattr(service, "check_now", lambda: asked.append(True) or True)
    dialog_ = UpdateDialog(service)
    dialog_.present()
    assert asked == [True]  # nothing known yet
    service._state = UpdateState(Phase.AVAILABLE, release())
    dialog_.present()
    assert asked == [True]  # one is known: shown, not asked again
    service._state = UpdateState(Phase.FAILED, release(), "boom")
    dialog_.present()
    assert asked == [True]  # a failed install keeps its release for another try
    service._state = UpdateState(Phase.FAILED, None, "offline")
    dialog_.present()
    assert asked == [True, True]


# --------------------------------------------------------------------------- the tray
def test_the_tray_names_the_new_version(controller: FakeController, cleanup: list[Any]) -> None:  # noqa: F811
    tray = _tray(controller, cleanup)
    assert tray.action_update.text() == "Check for updates…"
    tray.set_update_available("0.2.2")
    assert tray.action_update.text() == "Update to 0.2.2…"
    tray.set_update_available(None)
    assert tray.action_update.text() == "Check for updates…"


def test_the_tray_entry_opens_the_update_window(
    controller: FakeController,  # noqa: F811
    cleanup: list[Any],  # noqa: F811
) -> None:
    tray = _tray(controller, cleanup)
    opened: list[bool] = []
    tray.open_update.connect(lambda: opened.append(True))
    tray.action_update.trigger()
    assert opened == [True]


def test_the_tray_entry_is_hidden_where_updates_cannot_be_checked(
    controller: FakeController,  # noqa: F811
    cleanup: list[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tray_module, "update_supported", lambda: True)
    assert _tray(controller, cleanup).action_update.isVisible()
    monkeypatch.setattr(tray_module, "update_supported", lambda: False)
    assert not _tray(controller, cleanup).action_update.isVisible()


# ----------------------------------------------------------------------- the settings
def test_the_settings_offer_the_daily_check(
    dialog: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    box = dialog.widget_for("updates.check")
    assert not box.isChecked()  # off by default
    assert dialog.settings().updates.check is False
    box.setChecked(True)
    assert dialog.settings().updates.check is True


# ---------------------------------------------------------------------- the whole app
def test_the_app_starts_with_the_check_off(
    build: Callable[..., Harness],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc, "supported", lambda: True)
    h = build(monitors=TWO_MONITORS)
    service = h.app._updates
    assert service is not None
    assert not service._timer.isActive()
    assert service.state.phase is Phase.IDLE


def test_turning_the_setting_on_schedules_the_check(
    build: Callable[..., Harness],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc, "supported", lambda: True)
    h = build(monitors=TWO_MONITORS)
    service = h.app._updates
    assert service is not None
    settings = h.controller.settings.copy()
    settings.updates.check = True
    h.controller.apply_settings(settings)
    assert service._timer.isActive()
    settings = settings.copy()
    settings.updates.check = False
    h.controller.apply_settings(settings)
    assert not service._timer.isActive()


def test_a_new_version_is_announced_and_the_notification_opens_the_window(
    build: Callable[..., Harness],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    monkeypatch.setattr(svc, "supported", lambda: True)
    h = build(monitors=TWO_MONITORS, background=False)
    titles = _tray_messages(h, monkeypatch)
    service = h.app._updates
    assert service is not None
    service._set(UpdateState(Phase.AVAILABLE, release()))
    assert h.tray.action_update.text() == "Update to 0.2.2…"
    service.announce.emit(release())
    assert titles == ["Update available"]
    opened: list[bool] = []
    monkeypatch.setattr(h.app, "open_update", lambda: opened.append(True))
    h.app._on_message_clicked()
    assert opened == [True]


def test_being_up_to_date_resets_the_tray_entry(
    build: Callable[..., Harness],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc, "supported", lambda: True)
    h = build(monitors=TWO_MONITORS)
    service = h.app._updates
    assert service is not None
    service._set(UpdateState(Phase.AVAILABLE, release()))
    service._set(UpdateState(Phase.UP_TO_DATE, release()))
    assert h.tray.action_update.text() == "Check for updates…"


def test_the_window_opens_once_and_is_closed_with_the_app(
    build: Callable[..., Harness],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    monkeypatch.setattr(svc, "supported", lambda: True)
    h = build(monitors=TWO_MONITORS)
    service = h.app._updates
    assert service is not None
    monkeypatch.setattr(service, "check_now", lambda: True)
    h.app.open_update()
    first = h.app._update_dialog
    assert isinstance(first, UpdateDialog)
    h.app.open_update()
    assert h.app._update_dialog is first
    assert first.isVisible()
    h.ctx.shutdown()
    settle(qapp)
    assert h.app._update_dialog is None
    assert app_module.PROMPT_UPDATE == "update"
