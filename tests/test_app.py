"""Smoke tests for eye_tracker.app: the whole tray app built offscreen.

``build_app`` creates the real controller and UI, but the vision worker, the
cursor, the hotkey manager and the platform services are fakes: no camera is
opened, no thread started, the pointer never moves, no global hotkey is
registered and nothing is locked. Autostart is faked as well.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from eye_tracker import app as app_module
from eye_tracker import ipc, paths
from eye_tracker.app import AppContext, build_app, run_app
from eye_tracker.config import Settings
from eye_tracker.logging_setup import shutdown_logging
from eye_tracker.platform import autostart
from eye_tracker.platform.base import PlatformServices
from eye_tracker.platform.hotkeys import HotkeyManager
from eye_tracker.types import Monitor, Rect, TrackingState, WorkerStats
from eye_tracker.ui.tray import TrayIcon

TWO_MONITORS = [
    Monitor(0, "left", Rect(0, 0, 800, 600), primary=True),
    Monitor(1, "right", Rect(800, 0, 800, 600)),
]


# -------------------------------------------------------------------------- fakes
class FakeWorker:
    """Stands in for VisionWorker: no thread, no camera, calls are recorded."""

    def __init__(self, *callbacks: Any) -> None:
        self.callbacks = callbacks
        self.started = False
        self.stopped = False
        self.active: list[bool] = []
        self.previews: list[bool] = []
        self.backend_info: tuple[str, str] | None = ("fake", "fake-1")
        self.stats = WorkerStats(fps=4.0, camera_open=True)

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 3.0) -> None:
        self.stopped = True

    def set_interval(self, seconds: float) -> None:
        pass

    def set_active(self, active: bool) -> None:
        self.active.append(active)

    def set_max_faces(self, n: int) -> None:
        pass

    def set_motion_gate(self, enabled: bool, threshold: float) -> None:
        pass

    def set_preview(self, enabled: bool) -> None:
        self.previews.append(enabled)

    def reconfigure(self, source_factory: Any = None, backend_factory: Any = None) -> None:
        pass


class FakeCursor:
    def __init__(self) -> None:
        self.moves: list[tuple[int, int]] = []

    def pos(self) -> tuple[int, int]:
        return (100, 100)

    def set_pos(self, x: int, y: int) -> bool:
        self.moves.append((x, y))
        return True


@dataclass
class Harness:
    ctx: AppContext
    workers: list[FakeWorker] = field(default_factory=list)
    cursor: FakeCursor = field(default_factory=FakeCursor)

    @property
    def app(self) -> app_module.EyeTrackerApp:
        assert self.ctx.app is not None
        return self.ctx.app

    @property
    def controller(self) -> Any:
        assert self.ctx.controller is not None
        return self.ctx.controller

    @property
    def tray(self) -> TrayIcon:
        assert self.app.tray is not None
        return self.app.tray

    @property
    def worker(self) -> FakeWorker:
        (worker,) = self.workers
        return worker


@pytest.fixture(autouse=True)
def fake_autostart(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The app, tray, wizard and settings window read (and could change) the login item."""
    calls: list[str] = []
    monkeypatch.setattr(autostart, "is_supported", lambda: True)
    monkeypatch.setattr(autostart, "is_enabled", lambda config_dir=None: False)
    monkeypatch.setattr(autostart, "status", lambda config_dir=None: autostart.Status.DISABLED)
    monkeypatch.setattr(
        autostart, "enable", lambda background=True, config_dir=None: calls.append("enable")
    )
    monkeypatch.setattr(autostart, "disable", lambda config_dir=None: calls.append("disable"))

    def refresh(config_dir: Path | None = None) -> bool:
        calls.append("refresh")
        return False

    monkeypatch.setattr(autostart, "refresh", refresh)
    return calls


@pytest.fixture
def build(qapp: QApplication, app_dirs: Path) -> Iterator[Callable[..., Harness]]:
    harnesses: list[Harness] = []

    def make(
        *,
        settings: Settings | None = None,
        monitors: list[Monitor] | None = None,
        controller_factory: Any = None,
        platform: PlatformServices | None = None,
        **options: Any,
    ) -> Harness:
        if settings is None:
            settings = Settings()
            settings.general.first_run_done = True
        settings.save(paths.settings_file())
        workers: list[FakeWorker] = []
        cursor = FakeCursor()

        def worker_factory(*callbacks: Any) -> FakeWorker:
            worker = FakeWorker(*callbacks)
            workers.append(worker)
            return worker

        kwargs: dict[str, Any] = {
            "worker_factory": worker_factory,
            "cursor": cursor,
            "hotkey_manager_factory": lambda: HotkeyManager(note="tests"),
        }
        if monitors is not None:
            kwargs["monitors_provider"] = lambda: list(monitors)
        args = argparse.Namespace(**{"background": True, "calibrate": False, **options})
        ctx = build_app(
            args,
            platform=platform if platform is not None else PlatformServices(),
            controller_factory=controller_factory,
            controller_kwargs=None if controller_factory else kwargs,
        )
        harness = Harness(ctx, workers, cursor)
        harnesses.append(harness)
        return harness

    yield make
    for harness in harnesses:
        harness.ctx.shutdown()
    shutdown_logging()
    qapp.processEvents()


def settle(qapp: QApplication, rounds: int = 3) -> None:
    """Run pending zero-timeout timers and deferred deletions."""
    for _ in range(rounds):
        qapp.processEvents()


# ----------------------------------------------------------------------- building
def test_build_app_creates_and_wires_everything(
    build: Callable[..., Harness], qapp: QApplication
) -> None:
    h = build(monitors=TWO_MONITORS)
    ctx = h.ctx
    assert ctx.exit_code is None
    assert ctx.server is not None
    assert ctx.server.is_listening
    assert isinstance(h.app.tray, TrayIcon)
    assert h.app.overlay is not None
    assert h.app.countdown is not None
    assert h.app.curtain is not None
    assert h.worker.started
    assert not h.worker.stopped
    assert qapp.applicationName() == "Eye Tracker"
    assert not qapp.quitOnLastWindowClosed()

    # The command socket reaches the controller.
    reply = ipc.send_command("status")
    assert reply is not None
    status = json.loads(reply)
    assert status["state"] == TrackingState.NEEDS_CALIBRATION.value
    assert status["calibrated"] is False
    assert ipc.send_command("pause") == "ok"
    assert h.controller.state is TrackingState.PAUSED
    assert h.tray.state is TrackingState.PAUSED

    settle(qapp)
    # --background: no first-run wizard, no startup windows.
    assert h.app.wizard is None
    assert h.app.calibration_window is None
    assert h.app.settings_dialog is None
    assert h.cursor.moves == []


def test_ui_requests_open_single_windows(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build()
    h.controller.ui_requested.emit("settings")
    dialog = h.app.settings_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert ipc.send_command("settings") == "ok"
    assert h.app.settings_dialog is dialog

    # "show" raises an open window instead of opening another one.
    h.controller.ui_requested.emit("show")
    assert h.app.settings_dialog is dialog
    dialog.reject()
    settle(qapp)
    assert h.app.settings_dialog is None

    # Offscreen there is no system tray: "show" falls back to the settings.
    h.controller.ui_requested.emit("show")
    assert h.app.settings_dialog is not None
    h.app.settings_dialog.reject()
    settle(qapp)

    # With a tray, "show" pops up the tray menu instead.
    monkeypatch.setattr(TrayIcon, "available", property(lambda self: True))
    monkeypatch.setattr(h.tray.tray, "isVisible", lambda: True)
    h.controller.ui_requested.emit("show")
    assert h.tray.menu.isVisible()
    assert h.app.settings_dialog is None
    h.tray.menu.hide()

    h.tray.open_about.emit()
    assert h.app._about is not None
    assert h.app._about.isVisible()
    h.tray.open_preview.emit()
    preview = h.app.preview
    assert preview is not None
    assert preview.isVisible()
    assert h.worker.previews[-1] is True
    h.tray.open_preview.emit()
    assert h.app.preview is preview
    preview.close()
    assert h.worker.previews[-1] is False


def test_calibration_opens_once(build: Callable[..., Harness], qapp: QApplication) -> None:
    h = build(monitors=TWO_MONITORS)
    h.controller.calibration_required.emit("hotkey")
    window = h.app.calibration_window
    assert window is not None
    assert window.is_active
    assert h.controller.state is TrackingState.CALIBRATING

    h.controller.calibration_required.emit("ipc")
    h.tray.open_calibration.emit()
    assert ipc.send_command("calibrate") == "ok"
    assert h.app.calibration_window is window

    window.cancel()
    settle(qapp)
    assert h.app.calibration_window is None
    assert h.controller.state is TrackingState.NEEDS_CALIBRATION

    h.tray.open_calibration.emit()
    second = h.app.calibration_window
    assert second is not None
    assert second is not window
    assert second.is_active


def test_automatic_calibration_reasons_only_notify(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build(monitors=TWO_MONITORS)
    shown: list[tuple[str, str]] = []

    def notify(title: str, message: str, **kwargs: Any) -> bool:
        shown.append((title, message))
        return True

    monkeypatch.setattr(h.tray, "notify", notify)
    h.controller.calibration_required.emit("the monitor layout changed")
    h.controller.calibration_required.emit("the monitor layout changed")
    assert h.app.calibration_window is None
    assert len(shown) == 1
    assert shown[0][0] == "Calibration needed"
    assert "the monitor layout changed" in shown[0][1]

    # Another notification replaced ours: clicking it must not calibrate.
    h.controller.calibration_required.emit("another reason")
    h.controller.notify.emit("Camera", "Camera unavailable")
    h.tray.tray.messageClicked.emit()
    assert h.app.calibration_window is None

    h.controller.calibration_required.emit("a third reason")
    h.tray.tray.messageClicked.emit()
    window = h.app.calibration_window
    assert window is not None
    assert window.is_active


def test_single_monitor_gets_no_calibration_nag(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build(monitors=TWO_MONITORS[:1])
    shown: list[str] = []

    def notify(title: str, message: str, **kwargs: Any) -> bool:
        shown.append(title)
        return True

    monkeypatch.setattr(h.tray, "notify", notify)
    h.controller.calibration_required.emit("the monitor layout changed")
    assert shown == []


def test_helpers_follow_the_controller(build: Callable[..., Harness]) -> None:
    h = build()
    curtain, countdown, overlay = h.app.curtain, h.app.countdown, h.app.overlay
    assert curtain is not None
    assert countdown is not None
    assert overlay is not None

    h.controller.guard_changed.emit(True)
    assert curtain.is_showing
    h.controller.guard_changed.emit(False)
    assert not curtain.is_showing

    h.controller.away_warning.emit(8.0)
    assert countdown.active
    assert countdown.isVisible()
    h.controller.away_cancelled.emit()
    assert not countdown.active

    assert not overlay.enabled
    updated = h.controller.settings.copy()
    updated.ui.show_gaze_overlay = True
    h.controller.apply_settings(updated)
    assert overlay.enabled
    assert Settings.load(paths.settings_file()).ui.show_gaze_overlay


def test_first_run_wizard_then_calibration(
    build: Callable[..., Harness], qapp: QApplication, fake_autostart: list[str]
) -> None:
    h = build(settings=Settings(), monitors=TWO_MONITORS, background=False)
    settle(qapp)
    wizard = h.app.wizard
    assert wizard is not None
    assert wizard.isVisible()

    # While the wizard is open, "settings" raises the wizard.
    h.controller.ui_requested.emit("settings")
    assert h.app.settings_dialog is None

    wizard.accept()  # "Calibrate now" is ticked by default
    settle(qapp)
    assert h.app.wizard is None
    assert h.controller.settings.general.first_run_done
    assert Settings.load(paths.settings_file()).general.first_run_done
    window = h.app.calibration_window
    assert window is not None
    assert window.is_active
    assert fake_autostart == ["refresh"]  # "Start at login" was left unticked


def test_putting_off_the_offered_calibration_says_what_that_means(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r3-ux-docs-01: Esc on the calibration the setup assistant opened left no hint."""
    h = build(settings=Settings(), monitors=TWO_MONITORS, background=False)
    titles = _tray_messages(h, monkeypatch)
    settle(qapp)
    wizard = h.app.wizard
    assert wizard is not None
    wizard.accept()  # "Calibrate now" is ticked with two monitors
    settle(qapp)
    window = h.app.calibration_window
    assert window is not None
    window.cancel()
    settle(qapp)
    assert titles == ["Calibration needed"]
    # One the user asked for and cancelled needs no such hint (even unannounced).
    h.app._announced.clear()
    h.tray.open_calibration.emit()
    window = h.app.calibration_window
    assert window is not None
    window.cancel()
    settle(qapp)
    assert titles == ["Calibration needed"]


def test_background_start_skips_the_wizard(
    build: Callable[..., Harness], qapp: QApplication
) -> None:
    h = build(settings=Settings(), background=True)
    settle(qapp)
    assert h.app.wizard is None


def test_calibrate_option_opens_the_calibration(
    build: Callable[..., Harness], qapp: QApplication
) -> None:
    h = build(calibrate=True)
    assert h.app.calibration_window is None  # opens once the event loop runs
    settle(qapp)
    window = h.app.calibration_window
    assert window is not None
    assert window.is_active


def test_second_instance_hands_over_and_exits(
    build: Callable[..., Harness], qapp: QApplication
) -> None:
    first = build()
    second = build(background=False)
    assert second.ctx.exit_code == 0
    assert second.ctx.app is None
    assert second.workers == []
    # The first instance was asked to show itself (no tray offscreen: settings).
    assert first.app.settings_dialog is not None

    third = build(calibrate=True)
    assert third.ctx.exit_code == 0
    assert first.app.calibration_window is not None
    assert third.ctx.exec() == 0

    # run_app returns the hand-over result without entering the event loop.
    assert run_app(argparse.Namespace(background=True)) == 0


def test_a_login_start_does_not_pop_up_the_running_instance(
    build: Callable[..., Harness], qapp: QApplication
) -> None:
    """Regression (r2-ui-app-04): a --background launch asked the instance to show
    itself, which popped up its tray menu, the setup assistant or the settings."""
    first = build()
    second = build(background=True)
    assert second.ctx.exit_code == 0
    assert second.ctx.app is None
    settle(qapp)
    assert first.app.settings_dialog is None
    assert first.app.wizard is None


def test_a_launch_waits_for_an_instance_that_is_shutting_down(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (r2-ui-app-02): quitting and starting again right away left
    nothing running: the quitting instance still held the lock while it released
    the camera, and the new launch gave up at once."""
    monkeypatch.setattr(ipc, "STARTUP_WAIT_MS", 0)
    holder = ipc.InstanceLock()  # the quitting instance
    assert holder.acquire()
    waits: list[int] = []

    def wait(ms: int) -> None:
        waits.append(ms)
        if len(waits) == 2:
            holder.release()  # it has stopped the camera and exits

    monkeypatch.setattr(app_module, "_wait_ms", wait)
    try:
        h = build(background=False)
    finally:
        holder.release()
    # This launch took over as the instance.
    assert h.ctx.exit_code is None
    assert h.ctx.app is not None
    assert h.ctx.server is not None
    assert h.ctx.server.lock is not None
    assert h.ctx.server.lock.is_held
    assert waits == [app_module._HANDOVER_RETRY_MS] * 2


def test_a_launch_hands_over_once_a_busy_instance_answers(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "STARTUP_WAIT_MS", 0)
    busy = "error: Eye Tracker is starting or shutting down; try again"
    replies = iter([busy, busy, "ok"])
    sent: list[str] = []

    def send(command: str, timeout_ms: int = 0, *, wait_ms: int = 0) -> str:
        sent.append(command)
        return next(replies)

    monkeypatch.setattr(ipc, "send_command", send)
    monkeypatch.setattr(app_module, "_wait_ms", lambda ms: None)
    holder = ipc.InstanceLock()  # an instance that is still starting up
    assert holder.acquire()
    try:
        h = build(background=False)
    finally:
        holder.release()
    assert h.ctx.exit_code == 0
    assert h.ctx.app is None
    assert sent == ["show", "show", "show"]


def test_options_for_a_new_instance_are_reported_as_not_applied(
    build: Callable[..., Harness], caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Regression (r2-docs-15): --trace (and --camera, --backend) were silently
    dropped when the app was already running, and no trace file appeared."""
    build()
    trace = tmp_path / "trace.jsonl"
    with caplog.at_level(logging.WARNING, logger="eye_tracker.app"):
        second = build(trace=str(trace), camera="clip.mp4")
    assert second.ctx.exit_code == 0
    assert not trace.exists()
    (message,) = [r.getMessage() for r in caplog.records if "not applied" in r.getMessage()]
    assert "--trace, --camera were not applied" in message
    assert "--backend" not in message
    assert app_module.cli_command_text("ctl", "quit") in message

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="eye_tracker.app"):
        assert build(calibrate=True).ctx.exit_code == 0  # nothing ignored
    assert not any("not applied" in r.getMessage() for r in caplog.records)


def test_the_setup_assistant_from_the_settings_keeps_the_walk_away_lock(
    build: Callable[..., Harness], qapp: QApplication
) -> None:
    """Regression (r2-ui-app-01): asking for the assistant "at next start" stored
    first_run_done=False, which replaced the walk-away lock by a notification at
    once and for good when the assistant was cancelled."""
    h = build()
    h.app.open_settings()
    dialog = h.app.settings_dialog
    assert dialog is not None
    dialog.setup_requested.emit()  # "Run setup assistant…"
    wizard = h.app.wizard
    assert wizard is not None
    assert h.controller.settings.general.first_run_done is True
    wizard.reject()
    settle(qapp)
    assert h.app.wizard is None
    assert h.controller.settings.general.first_run_done is True
    assert Settings.load(paths.settings_file()).general.first_run_done is True


def test_session_overrides_stay_out_of_the_settings_file(
    build: Callable[..., Harness],
) -> None:
    h = build(camera="clip.mp4", backend="lite")
    assert h.controller.settings.camera.device == "clip.mp4"
    assert h.controller.settings.general.backend == "lite"
    updated = h.controller.settings.copy()
    updated.general.notifications = False
    h.controller.apply_settings(updated)
    on_disk = Settings.load(paths.settings_file())
    assert on_disk.general.notifications is False
    assert on_disk.camera.device == "0"
    assert on_disk.general.backend == "auto"

    # A value the user picks explicitly is saved.
    updated = h.controller.settings.copy()
    updated.camera.device = "1"
    h.controller.apply_settings(updated)
    assert Settings.load(paths.settings_file()).camera.device == "1"


def test_log_level_follows_settings_unless_given(build: Callable[..., Harness]) -> None:
    settings = Settings()
    settings.general.first_run_done = True
    settings.general.log_level = "DEBUG"
    h = build(settings=settings)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    updated = h.controller.settings.copy()
    updated.general.log_level = "WARNING"
    h.controller.apply_settings(updated)
    assert root.level == logging.WARNING
    h.ctx.shutdown()

    locked = build(settings=settings, log_level="ERROR")
    assert root.level == logging.ERROR
    updated = locked.controller.settings.copy()
    updated.general.log_level = "INFO"
    locked.controller.apply_settings(updated)
    assert root.level == logging.ERROR
    assert paths.log_file().is_file()


def test_shutdown_is_clean_and_idempotent(
    build: Callable[..., Harness], qapp: QApplication
) -> None:
    h = build()
    h.app.open_settings()
    h.app.open_about()
    h.app.open_preview()
    h.app.open_calibration()
    h.controller.guard_changed.emit(True)
    h.controller.away_warning.emit(5.0)
    assert h.controller.state is TrackingState.CALIBRATING

    h.ctx.shutdown()
    assert h.app.closed
    assert h.worker.stopped
    assert h.ctx.server is not None
    assert not h.ctx.server.is_listening
    assert not ipc.is_running()
    assert h.app.settings_dialog is None
    assert h.app.calibration_window is None
    assert h.app.preview is None
    assert h.app.curtain is not None
    assert not h.app.curtain.is_showing
    assert not h.tray.tray.isVisible()
    # The open calibration was cancelled before the controller stopped.
    assert h.controller.state is not TrackingState.CALIBRATING
    assert h.app.handle_command("status").startswith("error:")

    h.ctx.shutdown()  # idempotent
    h.app.open_settings()  # ignored after shutdown
    assert h.app.settings_dialog is None
    settle(qapp)


def test_startup_failure_returns_an_error(
    build: Callable[..., Harness], qapp: QApplication
) -> None:
    def broken(settings: Settings, services: PlatformServices) -> Any:
        raise RuntimeError("no controller today")

    h = build(controller_factory=broken)
    assert h.ctx.exit_code == 1
    assert h.ctx.app is None
    assert h.ctx.exec() == 1
    assert not ipc.is_running()
    # The instance lock was released too: the next launch may become the instance.
    lock = ipc.InstanceLock()
    assert lock.acquire()
    assert lock.is_held
    lock.release()


# -------------------------------------------------------------------- event loop
def _run_with_safety_net(qapp: QApplication, ctx: AppContext, action: Callable[[], None]) -> int:
    safety = QTimer()
    safety.setSingleShot(True)
    safety.timeout.connect(lambda: qapp.exit(99))
    safety.start(10_000)
    QTimer.singleShot(20, action)
    try:
        return ctx.exec()
    finally:
        safety.stop()


def test_exec_quits_on_ipc_quit(build: Callable[..., Harness], qapp: QApplication) -> None:
    h = build()
    replies: list[str | None] = []
    code = _run_with_safety_net(qapp, h.ctx, lambda: replies.append(ipc.send_command("quit")))
    assert code == 0
    assert replies == ["ok"]
    assert h.app.closed
    assert h.worker.stopped
    assert not ipc.is_running()


def test_exec_quits_on_tray_quit(build: Callable[..., Harness], qapp: QApplication) -> None:
    h = build()
    code = _run_with_safety_net(qapp, h.ctx, h.tray.quit_requested.emit)
    assert code == 0
    assert h.app.closed


sigint_ignored = pytest.mark.skipif(
    signal.getsignal(signal.SIGINT) is signal.SIG_IGN,
    reason="SIGINT is ignored in this process (started in the background)",
)


@sigint_ignored
def test_exec_quits_on_ctrl_c(build: Callable[..., Harness], qapp: QApplication) -> None:
    h = build()
    before = signal.getsignal(signal.SIGINT)
    code = _run_with_safety_net(qapp, h.ctx, lambda: signal.raise_signal(signal.SIGINT))
    assert code == 0
    assert h.app.closed
    assert signal.getsignal(signal.SIGINT) is before


@sigint_ignored
def test_signal_handler_is_installed_only_inside_the_block(qapp: QApplication) -> None:
    before = signal.getsignal(signal.SIGINT)
    requested: list[bool] = []
    with app_module._quit_on_signals(lambda: requested.append(True)):
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not before
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert requested == []  # deferred to the event loop
        qapp.processEvents()
        assert requested == [True]
    assert signal.getsignal(signal.SIGINT) is before


def test_ignored_signals_stay_ignored(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    installed: list[Any] = []
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_IGN)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.append(handler))
    with app_module._quit_on_signals(lambda: None):
        pass
    assert installed == []


# -------------------------------------------------------------------- review fixes
def _tray_messages(h: Harness, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Titles the tray shows, with its real settings and mute checks (no tray offscreen)."""
    titles: list[str] = []
    monkeypatch.setattr(h.tray, "_messages_available", lambda: True)
    monkeypatch.setattr(h.tray, "_show_message", lambda title, *_args: titles.append(title))
    return titles


def _wait_for(qapp: QApplication, condition: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    return condition()


def test_calibration_prompt_repeats_after_the_calibration_was_usable(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (ui_app-08, journeys-19): each reason was announced once per process."""
    h = build(monitors=TWO_MONITORS, background=False)
    titles = _tray_messages(h, monkeypatch)
    h.controller.calibration_required.emit("the monitor layout changed")
    h.controller.calibration_required.emit("the monitor layout changed")
    assert titles == ["Calibration needed"]  # while still uncalibrated: once
    # The user recalibrates (or returns to a known desk): usable again.
    controller_type = type(h.controller)
    monkeypatch.setattr(controller_type, "is_calibrated", property(lambda self: True))
    h.controller.state_changed.emit(TrackingState.TRACKING)
    monkeypatch.setattr(controller_type, "is_calibrated", property(lambda self: False))
    # Days later, another dock: the same reason text deserves a new notification.
    h.controller.calibration_required.emit("the monitor layout changed")
    assert titles == ["Calibration needed", "Calibration needed"]


def test_a_calibration_prompt_that_was_not_shown_is_not_counted(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build(monitors=TWO_MONITORS, background=False)
    titles = _tray_messages(h, monkeypatch)
    quiet = h.controller.settings.copy()
    quiet.general.notifications = False
    h.controller.apply_settings(quiet)
    h.controller.calibration_required.emit("the monitor layout changed")
    assert titles == []
    loud = h.controller.settings.copy()
    loud.general.notifications = True
    h.controller.apply_settings(loud)
    h.controller.calibration_required.emit("the monitor layout changed")
    assert titles == ["Calibration needed"]


def test_a_prompt_muted_after_a_login_start_is_shown_later(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "STARTUP_QUIET_S", 0.5)
    h = build(monitors=TWO_MONITORS)  # --background: notifications muted at first
    titles = _tray_messages(h, monkeypatch)
    assert h.tray.muted_for > 0
    # A dock brings a monitor up while the login start is still quiet.
    h.controller.calibration_required.emit("the monitor layout changed")
    assert titles == []
    assert _wait_for(qapp, lambda: bool(titles))
    assert titles == ["Calibration needed"]
    h.tray.tray.messageClicked.emit()
    window = h.app.calibration_window
    assert window is not None
    assert window.is_active


def test_an_uncalibrated_login_start_still_says_why_nothing_switches(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r3-ux-docs-01: a --background start dropped the calibration prompt for good."""
    monkeypatch.setattr(app_module, "STARTUP_QUIET_S", 0.3)
    h = build(monitors=TWO_MONITORS)  # --background, set up, never calibrated
    titles = _tray_messages(h, monkeypatch)
    settle(qapp)
    assert titles == []  # a login start stays quiet at first
    assert _wait_for(qapp, lambda: bool(titles))
    assert titles == ["Calibration needed"]


def test_no_calibration_prompt_where_it_changes_nothing(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r3-ux-docs-03: with switching turned off a calibration achieves nothing."""
    settings = Settings()
    settings.general.first_run_done = True
    settings.switching.enabled = False
    h = build(settings=settings, monitors=TWO_MONITORS, background=False)
    titles = _tray_messages(h, monkeypatch)
    settle(qapp)
    h.controller.calibration_required.emit("not calibrated yet")
    assert titles == []
    assert h.controller.state is TrackingState.TRACKING


def test_privacy_mode_kept_from_the_last_run_is_announced(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r3-ux-docs-02: privacy mode survives a restart; the user is told once."""
    monkeypatch.setattr(app_module, "STARTUP_QUIET_S", 0.3)
    paths.state_file().write_text('{"privacy_mode": true}\n', encoding="utf-8")
    h = build(monitors=TWO_MONITORS)  # a login start, e.g. after an overnight update
    titles = _tray_messages(h, monkeypatch)
    assert h.controller.privacy
    assert h.controller.state is TrackingState.PRIVACY
    assert True not in h.worker.active  # the camera never came on
    assert _wait_for(qapp, lambda: bool(titles))
    # No calibration prompt while the camera is off on purpose.
    assert titles == ["Privacy mode is still on"]


def test_window_requests_during_a_calibration_raise_it_instead(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (ui_app-12): a dialog hidden under the calibration took its keyboard."""
    h = build()
    h.app.open_calibration()
    window = h.app.calibration_window
    assert window is not None
    raised: list[bool] = []

    def raise_windows() -> bool:
        raised.append(True)
        return True

    monkeypatch.setattr(window, "start", raise_windows)
    h.controller.ui_requested.emit("settings")  # e.g. 'eye-tracker ctl settings'
    h.tray.open_preview.emit()
    h.tray.open_about.emit()
    h.app.open_wizard()
    assert raised == [True, True, True, True]
    assert h.app.settings_dialog is None
    assert h.app.preview is None
    assert h.app._about is None
    assert h.app.wizard is None
    assert window.is_active


def test_unfinished_setup_is_offered_after_a_login_start(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (journeys-05): starting at login skipped the setup assistant forever."""
    monkeypatch.setattr(app_module, "STARTUP_QUIET_S", 0.3)
    h = build(settings=Settings(), background=True)  # first_run_done is False
    titles = _tray_messages(h, monkeypatch)
    settle(qapp)
    assert h.app.wizard is None  # a login start stays quiet
    assert _wait_for(qapp, lambda: bool(titles))
    assert titles == ["Finish setting up Eye Tracker"]
    h.tray.tray.messageClicked.emit()
    wizard = h.app.wizard
    assert wizard is not None
    wizard.reject()
    settle(qapp)
    assert h.app.wizard is None


def test_a_second_launch_opens_the_unfinished_setup(
    build: Callable[..., Harness], qapp: QApplication
) -> None:
    h = build(settings=Settings(), background=True)
    settle(qapp)
    assert h.app.wizard is None
    h.controller.ui_requested.emit("show")  # launched again from the Start menu
    assert h.app.wizard is not None
    assert h.app.settings_dialog is None


def test_startup_points_the_login_item_at_this_copy(
    build: Callable[..., Harness], qapp: QApplication, fake_autostart: list[str]
) -> None:
    build()
    assert fake_autostart == []  # once the event loop runs
    settle(qapp)
    assert fake_autostart == ["refresh"]


def test_a_lock_holder_that_does_not_answer_keeps_other_launches_out(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """windows-09: the instance lock decides who runs, not a racy socket probe."""
    monkeypatch.setattr(ipc, "STARTUP_WAIT_MS", 300)
    monkeypatch.setattr(app_module, "HANDOVER_WAIT_MS", 300)  # then it gives up
    holder = ipc.InstanceLock()  # e.g. an instance that is still starting up, or hangs
    assert holder.acquire()
    try:
        blocked = build()
        assert blocked.ctx.exit_code == 1
        assert blocked.ctx.app is None
        assert blocked.workers == []
    finally:
        holder.release()
    h = build()
    assert h.ctx.exit_code is None
    server = h.ctx.server
    assert server is not None
    assert server.lock is not None
    assert server.lock.is_held
    h.ctx.shutdown()
    assert not server.lock.is_held  # released for the next launch


def test_a_calibration_reminder_after_being_away_is_shown_again(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The controller reminds the user on their return, with the same reason."""
    h = build(monitors=TWO_MONITORS, background=False)
    titles = _tray_messages(h, monkeypatch)
    reason = "the monitor layout changed"
    h.controller.calibration_required.emit(reason)
    assert titles == ["Calibration needed"]
    for state in (TrackingState.AWAY, TrackingState.LOCKED, TrackingState.PRIVACY):
        h.controller.state_changed.emit(state)  # gone for a while ...
        h.controller.state_changed.emit(TrackingState.NEEDS_CALIBRATION)  # ... and back
        h.controller.calibration_required.emit(reason)  # the controller's reminder
    assert titles == ["Calibration needed"] * 4
    # Other states do not repeat it.
    h.controller.state_changed.emit(TrackingState.PAUSED)
    h.controller.calibration_required.emit(reason)
    assert len(titles) == 4


def test_dismissing_the_curtain_tells_the_controller(build: Callable[..., Harness]) -> None:
    h = build()
    curtain = h.app.curtain
    assert curtain is not None
    h.controller._show_curtain(True)  # what the shoulder guard does
    assert curtain.is_showing
    curtain.dismiss()  # Esc or the button
    assert not curtain.is_showing
    assert h.controller._curtain is False  # switching resumes


class PermissionPlatform(PlatformServices):
    """Records which privacy settings pages would be opened (none is)."""

    def __init__(self) -> None:
        self.opened: list[str] = []

    def open_permission_settings(self, name: str) -> bool:
        self.opened.append(name)
        return True


def test_clicking_a_permission_notification_opens_the_privacy_settings(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    platform = PermissionPlatform()
    h = build(platform=platform, background=False)
    titles = _tray_messages(h, monkeypatch)
    h.controller.notify.emit("Camera access blocked", "Allow camera access …")
    h.controller.permission_needed.emit("camera")
    assert titles == ["Camera access blocked"]
    h.tray.tray.messageClicked.emit()
    assert platform.opened == ["camera"]
    h.tray.tray.messageClicked.emit()  # a click is used once
    assert platform.opened == ["camera"]

    # Another notification replaced it: its click does not open the settings.
    h.controller.notify.emit("Accessibility access needed", "…")
    h.controller.permission_needed.emit("accessibility")
    h.controller.notify.emit("Tracking paused", "…")
    h.tray.tray.messageClicked.emit()
    assert platform.opened == ["camera"]


def test_a_permission_without_a_shown_notification_is_not_clickable(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    platform = PermissionPlatform()
    h = build(monitors=TWO_MONITORS, platform=platform, background=False)
    _tray_messages(h, monkeypatch)
    h.controller.calibration_required.emit("the monitor layout changed")
    # Notifications off: the controller emits permission_needed alone.
    h.controller.permission_needed.emit("camera")
    h.tray.tray.messageClicked.emit()  # the calibration notification was clicked
    assert platform.opened == []
    window = h.app.calibration_window
    assert window is not None


def test_permission_notifications_muted_after_a_login_start_are_not_clickable(
    build: Callable[..., Harness],
) -> None:
    platform = PermissionPlatform()
    h = build(platform=platform)  # --background: notifications muted at first
    assert h.tray.muted_for > 0
    h.controller.notify.emit("Camera access blocked", "…")
    h.controller.permission_needed.emit("camera")
    h.tray.tray.messageClicked.emit()
    assert platform.opened == []


class AccessibilityPlatform(PermissionPlatform):
    """macOS-like: Accessibility belongs to an earlier build until the test grants it."""

    def __init__(self) -> None:
        super().__init__()
        self.accessibility = "stale"

    def accessibility_status(self) -> str:
        return self.accessibility


@pytest.mark.parametrize("granted_meanwhile", [False, True])
def test_a_permission_notice_muted_after_a_login_start_is_shown_later(
    build: Callable[..., Harness],
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    granted_meanwhile: bool,
) -> None:
    """journeys-15: the controller says it once, at startup, inside the quiet period."""
    monkeypatch.setattr(app_module, "STARTUP_QUIET_S", 0.3)
    platform = AccessibilityPlatform()
    h = build(platform=platform)  # --background: notifications muted at first
    titles = _tray_messages(h, monkeypatch)
    assert h.tray.muted_for > 0
    h.controller.notify.emit("Accessibility access needed", "Remove it and add it again.")
    h.controller.permission_needed.emit("accessibility")
    assert titles == []
    if granted_meanwhile:
        platform.accessibility = "granted"  # fixed before the quiet period ended
    assert _wait_for(qapp, lambda: h.app._deferred_permission is None)
    if granted_meanwhile:
        assert titles == []
        return
    assert titles == ["Accessibility access needed"]
    h.tray.tray.messageClicked.emit()
    assert platform.opened == ["accessibility"]


def test_a_muted_notice_is_not_repeated_when_notifications_were_off(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a notify just before it, permission_needed has nothing to repeat."""
    monkeypatch.setattr(app_module, "STARTUP_QUIET_S", 0.3)
    platform = AccessibilityPlatform()
    h = build(platform=platform)
    titles = _tray_messages(h, monkeypatch)
    h.controller.permission_needed.emit("accessibility")
    assert h.app._deferred_permission is None
    assert not _wait_for(qapp, lambda: bool(titles), timeout=0.8)


def test_the_login_item_is_left_alone_where_unsupported(
    build: Callable[..., Harness],
    qapp: QApplication,
    fake_autostart: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(autostart, "is_supported", lambda: False)
    build()
    settle(qapp)
    assert fake_autostart == []


def test_a_failing_login_item_check_does_not_stop_the_app(
    build: Callable[..., Harness], qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(config_dir: Path | None = None) -> bool:
        raise autostart.AutostartError("read-only home")

    monkeypatch.setattr(autostart, "refresh", broken)
    h = build()
    settle(qapp)
    assert h.ctx.exit_code is None
    assert not h.app.closed
    assert ipc.send_command("status") is not None


@pytest.mark.parametrize("failing", ["send_command", "create_app", "server"])
def test_the_instance_lock_is_released_when_startup_raises(
    build: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    """windows-09: an unexpected error must not keep every later launch out."""

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{failing} failed")

    if failing == "send_command":
        monkeypatch.setattr(ipc, "send_command", boom)
    elif failing == "create_app":
        monkeypatch.setattr(app_module, "_create_app", boom)
    else:
        monkeypatch.setattr(ipc, "InstanceServer", boom)
    with pytest.raises(RuntimeError, match=failing):
        build()
    lock = ipc.InstanceLock()
    assert lock.acquire()
    assert lock.is_held
    lock.release()
