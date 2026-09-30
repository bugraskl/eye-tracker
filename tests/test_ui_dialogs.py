"""Tests for the settings dialog, the calibration window and the first-run wizard.

Everything runs on Qt's offscreen platform against a fake controller that has
the real controller's signals and the methods these windows call. Start at
login, camera probing, the log-folder opener and diagnostics are replaced by
fakes, so nothing here touches the registry, a camera or the desktop.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from eye_tracker.config import Settings, describe_settings
from eye_tracker.gaze.calibration import CalibrationTarget
from eye_tracker.gaze.store import CalibrationData
from eye_tracker.platform import autostart
from eye_tracker.platform.base import PlatformServices
from eye_tracker.types import Monitor, Observation, Rect, TrackingState, layout_signature
from eye_tracker.ui import calibration_window as cw
from eye_tracker.ui import settings_dialog as sd
from eye_tracker.ui import wizard as wz
from eye_tracker.ui.calibration_window import CalibrationWindow
from eye_tracker.ui.settings_dialog import SettingsDialog, format_stats, hotkey_text_from_event
from eye_tracker.ui.wizard import FirstRunWizard
from eye_tracker.vision.camera import CameraInfo

pytestmark = pytest.mark.usefixtures("qapp")

LEFT = Monitor(0, "Left", Rect(0, 0, 1920, 1080), primary=True)
RIGHT = Monitor(1, "Right", Rect(1920, 0, 1920, 1080))
BACKEND = ("mediapipe", "mp-pose-iris-1")
MACOS = sys.platform == "darwin"


# =========================================================================== fakes
class FakePlatform(PlatformServices):
    """Records permission requests; claims every capability."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.perms: dict[str, bool | None] = {"camera": True, "accessibility": False}

    def capabilities(self) -> dict[str, bool]:
        return dict.fromkeys(super().capabilities(), True)

    def permissions(self) -> dict[str, bool | None]:
        return dict(self.perms)

    def request_permission(self, name: str) -> None:
        self.calls.append(("request", name))

    def open_permission_settings(self, name: str) -> bool:
        self.calls.append(("open", name))
        return True


class FakeController(QObject):
    """The Controller signals and methods used by these three windows."""

    state_changed = Signal(object)
    observation = Signal(object)
    stats_changed = Signal(object)
    notify = Signal(str, str)
    preview_frame = Signal(object)
    calibration_required = Signal(str)
    settings_changed = Signal(object)

    def __init__(
        self,
        settings: Settings | None = None,
        monitors: tuple[Monitor, ...] = (LEFT, RIGHT),
    ) -> None:
        super().__init__()
        self._settings = (settings or Settings()).copy()
        self._monitors = list(monitors)
        self.platform = FakePlatform()
        self.state = TrackingState.NEEDS_CALIBRATION
        self.backend = BACKEND
        self.applied: list[Settings] = []
        self.finished: list[CalibrationData | None] = []
        self.begun = 0
        self.previews: list[bool] = []
        self.notifications: list[tuple[str, str]] = []
        self.is_calibrated = False
        self.calibration_reason = "no calibration yet"
        self.hotkey_manager: Any = None
        self.notify.connect(lambda title, text: self.notifications.append((title, text)))

    @property
    def settings(self) -> Settings:
        return self._settings

    def apply_settings(self, settings: Settings) -> None:
        self.applied.append(settings)
        self._settings = settings.copy()
        self.settings_changed.emit(self._settings)

    def monitors(self) -> list[Monitor]:
        return list(self._monitors)

    def backend_info(self) -> tuple[str, str]:
        return self.backend

    def calibration(self) -> CalibrationData | None:
        return None

    def begin_calibration(self) -> None:
        self.begun += 1
        self.state = TrackingState.CALIBRATING

    def finish_calibration(self, data: CalibrationData | None) -> None:
        self.finished.append(data)
        self.state = TrackingState.TRACKING if data is not None else TrackingState.NEEDS_CALIBRATION

    def set_preview(self, enabled: bool) -> None:
        self.previews.append(bool(enabled))


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeAutostart:
    """Replaces the start-at-login functions (never touches the real OS entry)."""

    def __init__(self) -> None:
        self.enabled = False
        self.calls: list[str] = []
        self.fail: str | None = None

    def enable(self, background: bool = True) -> None:
        self.calls.append("enable" if background else "enable-foreground")
        if self.fail:
            raise autostart.AutostartError(self.fail)
        self.enabled = True

    def disable(self) -> None:
        self.calls.append("disable")
        if self.fail:
            raise autostart.AutostartError(self.fail)
        self.enabled = False


@pytest.fixture(autouse=True)
def fake_autostart(monkeypatch: pytest.MonkeyPatch) -> FakeAutostart:
    fake = FakeAutostart()
    monkeypatch.setattr(autostart, "is_supported", lambda: True)
    monkeypatch.setattr(autostart, "is_enabled", lambda: fake.enabled)
    monkeypatch.setattr(autostart, "enable", fake.enable)
    monkeypatch.setattr(autostart, "disable", fake.disable)
    return fake


@pytest.fixture(autouse=True)
def no_real_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Camera probing and the diagnostics report are faked everywhere."""

    def probe(max_index: int = 4, api: str = "auto") -> list[CameraInfo]:
        return [CameraInfo(0, "Integrated Camera", 640, 480), CameraInfo(2, "USB Cam", 1280, 720)]

    monkeypatch.setattr(sd, "probe_cameras", probe)
    monkeypatch.setattr(sd, "load_diagnostics_text", lambda: "FAKE DIAGNOSTICS REPORT")


def _dispose(widget: QWidget | QObject) -> None:
    if isinstance(widget, QWidget):
        widget.close()
    widget.deleteLater()
    QApplication.processEvents()


def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Process events until ``condition`` holds (for work done on helper threads)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    QApplication.processEvents()
    return condition()


def _button(parent: QWidget, text: str) -> QPushButton:
    for button in parent.findChildren(QPushButton):
        if button.text().replace("&", "") == text:
            return button
    raise AssertionError(f"no button {text!r}")


def _non_default_settings() -> Settings:
    """Every field changed from its default, with floats a spin box cannot show exactly."""
    s = Settings()
    s.general.backend = "opencv"
    s.general.start_paused = True
    s.general.first_run_done = True
    s.general.notifications = False
    s.general.log_level = "DEBUG"
    s.camera.device = "2"
    s.camera.width = 1280
    s.camera.height = 720
    s.camera.api = "dshow"
    s.performance.profile = "eco"
    s.performance.motion_gate = False
    s.performance.motion_threshold = 3.33
    s.switching.enabled = False
    s.switching.dwell_ms = 450
    s.switching.hysteresis = 0.073
    s.switching.off_screen_margin = 0.421
    s.switching.cooldown_ms = 900
    s.switching.mouse_grace_ms = 1234
    s.switching.typing_grace_ms = 2500
    s.switching.cursor_target = "gaze"
    s.switching.focus_window = False
    s.switching.smoothing = 0.377
    s.presence.enabled = False
    s.presence.action = "display_off"
    s.presence.away_timeout_s = 120
    s.presence.warning_s = 15
    s.presence.require_input_idle = False
    s.presence.wake_on_return = False
    s.privacy.pause_when_locked = False
    s.privacy.yield_camera = False
    s.privacy.pause_for_apps = ["zoom.exe", "obs64.exe"]
    s.privacy.shoulder_guard = True
    s.privacy.guard_action = "lock"
    s.privacy.guard_delay_s = 3.25
    s.hotkeys.enabled = False
    s.hotkeys.toggle_tracking = "Ctrl+Shift+F9"  # not canonical: must survive untouched
    s.hotkeys.toggle_privacy = "alt+meta+p"
    s.hotkeys.recalibrate = "ctrl+alt+k"
    s.learning.adaptive = False
    s.learning.max_samples = 800
    s.learning.drift_alerts = False
    s.ui.show_gaze_overlay = True
    return s


# ================================================================== settings dialog
@pytest.fixture
def controller() -> FakeController:
    return FakeController()


@pytest.fixture
def dialog(controller: FakeController) -> Iterator[SettingsDialog]:
    dlg = SettingsDialog(controller)
    yield dlg
    _dispose(dlg)


def test_every_setting_has_a_widget(dialog: SettingsDialog) -> None:
    keys = {row["key"] for row in describe_settings()}
    assert set(dialog.bound_keys()) == keys


def test_tooltips_come_from_the_settings_documentation(dialog: SettingsDialog) -> None:
    for row in describe_settings():
        if not row["doc"] or row["key"] == "general.first_run_done":
            continue  # the wizard checkbox explains itself in UI terms
        assert row["doc"] in dialog.widget_for(row["key"]).toolTip(), row["key"]


def test_pages_cover_the_six_sections(dialog: SettingsDialog) -> None:
    keys = [key for key, _title, _sub in SettingsDialog.PAGES]
    assert keys == ["general", "switching", "presence", "camera", "hotkeys", "diagnostics"]
    for key in keys:
        dialog.show_page(key)
        assert dialog.current_page() == key


def test_round_trip_leaves_every_value_untouched() -> None:
    original = _non_default_settings()
    controller = FakeController(original)
    dlg = SettingsDialog(controller)
    try:
        shown = dlg.settings()
        assert shown == original
        assert shown is not controller.settings
        assert not dlg.is_modified()
        apply_button = dlg._buttons.button(QDialogButtonBox.StandardButton.Apply)
        assert apply_button is not None
        assert not apply_button.isEnabled()
        # OK without edits closes without reconfiguring anything.
        dlg.accept()
        assert controller.applied == []
        assert dlg.result() == QDialog.DialogCode.Accepted
    finally:
        _dispose(dlg)


def test_apply_hands_an_edited_copy_to_the_controller(
    controller: FakeController, dialog: SettingsDialog
) -> None:
    before = controller.settings
    dwell = dialog.widget_for("switching.dwell_ms")
    assert isinstance(dwell, QSpinBox)
    dwell.setValue(700)
    action = dialog.widget_for("presence.action")
    assert isinstance(action, QComboBox)
    action.setCurrentIndex(action.findData("display_off"))
    apps = dialog.widget_for("privacy.pause_for_apps")
    apps.setPlainText("zoom.exe\n\n  teams.exe  \n")  # type: ignore[attr-defined]

    assert dialog.is_modified()
    apply_button = dialog._buttons.button(QDialogButtonBox.StandardButton.Apply)
    assert apply_button is not None
    assert apply_button.isEnabled()
    applied: list[object] = []
    dialog.applied.connect(applied.append)
    QTest.mouseClick(apply_button, Qt.MouseButton.LeftButton)

    assert len(controller.applied) == 1
    new = controller.applied[0]
    assert new is not before
    assert new.switching.dwell_ms == 700
    assert new.presence.action == "display_off"
    assert new.privacy.pause_for_apps == ["zoom.exe", "teams.exe"]
    expected = before.copy()
    expected.switching.dwell_ms = 700
    expected.presence.action = "display_off"
    expected.privacy.pause_for_apps = ["zoom.exe", "teams.exe"]
    assert new == expected
    assert applied == [new]
    # The applied state is the new baseline.
    assert not dialog.is_modified()
    assert not apply_button.isEnabled()


def test_float_edits_are_converted_back_from_display_units(
    controller: FakeController, dialog: SettingsDialog
) -> None:
    dialog.widget_for("switching.hysteresis").setValue(12)  # type: ignore[attr-defined]
    dialog.widget_for("switching.smoothing").setValue(80)  # type: ignore[attr-defined]
    assert dialog.apply()
    new = controller.applied[-1]
    assert new.switching.hysteresis == pytest.approx(0.12)
    assert new.switching.smoothing == pytest.approx(0.8)


def test_dependent_widgets_follow_their_master_checkbox(dialog: SettingsDialog) -> None:
    enabled = dialog.widget_for("switching.enabled")
    dwell = dialog.widget_for("switching.dwell_ms")
    assert isinstance(enabled, QCheckBox)
    enabled.setChecked(True)
    assert dwell.isEnabled()
    enabled.setChecked(False)
    assert not dwell.isEnabled()


def test_restore_defaults_keeps_the_first_run_flag() -> None:
    controller = FakeController(_non_default_settings())
    dlg = SettingsDialog(controller)
    try:
        dlg.restore_defaults()
        expected = Settings()
        expected.general.first_run_done = True
        assert dlg.settings() == expected
        assert dlg.is_modified()
    finally:
        _dispose(dlg)


def test_external_settings_change_refreshes_untouched_widgets_only(
    controller: FakeController, dialog: SettingsDialog
) -> None:
    dialog.widget_for("switching.dwell_ms").setValue(1500)  # type: ignore[attr-defined]
    other = controller.settings.copy()
    other.general.notifications = False  # e.g. changed from the tray
    other.switching.dwell_ms = 100
    controller.settings_changed.emit(other)
    shown = dialog.settings()
    assert shown.general.notifications is False
    assert shown.switching.dwell_ms == 1500  # the user's pending edit wins


def test_presence_page_promises_that_nothing_leaves_the_computer(dialog: SettingsDialog) -> None:
    texts = [label.text() for label in dialog.findChildren(QLabel)]
    assert any("Everything stays on this computer" in t for t in texts)


# ------------------------------------------------------------------------ hotkeys
def test_hotkey_edit_canonicalises_valid_text(dialog: SettingsDialog) -> None:
    edit = dialog.hotkey_edit("toggle_tracking")
    edit.set_hotkey("Ctrl + Alt + X")
    assert edit.hotkey() == "ctrl+alt+x"
    assert dialog.is_valid()


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("x", "modifier"),
        ("ctrl+alt", "needs a key"),
        ("ctrl+alt+nosuchkey", "unsupported key"),
        ("shift+a", "Shift alone"),
    ],
)
def test_invalid_hotkeys_block_apply(
    controller: FakeController, dialog: SettingsDialog, text: str, fragment: str
) -> None:
    dialog.hotkey_edit("toggle_privacy").set_hotkey(text)
    assert not dialog.is_valid()
    error = dialog.hotkey_errors()["toggle_privacy"]
    assert fragment in error
    ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok is not None
    assert not ok.isEnabled()
    assert dialog.apply() is False
    assert controller.applied == []
    # Fixing it clears the error.
    dialog.hotkey_edit("toggle_privacy").set_hotkey("ctrl+alt+j")
    assert dialog.is_valid()
    assert ok.isEnabled()


def test_duplicate_hotkeys_are_rejected(dialog: SettingsDialog) -> None:
    dialog.hotkey_edit("toggle_tracking").set_hotkey("ctrl+alt+c")  # recalibrate's default
    errors = dialog.hotkey_errors()
    assert len(errors) == 1
    assert "same shortcut" in next(iter(errors.values()))


def test_recorded_hotkey_is_applied(controller: FakeController, dialog: SettingsDialog) -> None:
    edit = dialog.hotkey_edit("recalibrate")
    mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
    QTest.keyClick(edit, Qt.Key.Key_K, mods)
    # On macOS Qt reports the Command key as Control; our "meta" is Command.
    expected = "alt+meta+k" if MACOS else "ctrl+alt+k"
    assert edit.hotkey() == expected
    QTest.keyClick(edit, Qt.Key.Key_Backspace)
    assert edit.hotkey() == ""
    edit.set_hotkey("ctrl+alt+k")
    assert dialog.apply()
    assert controller.applied[-1].hotkeys.recalibrate == "ctrl+alt+k"


def test_hotkey_text_from_key_events() -> None:
    KM = Qt.KeyboardModifier

    def event(key: Qt.Key, mods: Qt.KeyboardModifier, macos: bool) -> str | None:
        return hotkey_text_from_event(QKeyEvent(QKeyEvent.Type.KeyPress, key, mods), macos=macos)

    assert event(Qt.Key.Key_Control, KM.ControlModifier, False) is None
    assert event(Qt.Key.Key_P, KM.ControlModifier | KM.AltModifier, False) == "ctrl+alt+p"
    assert event(Qt.Key.Key_F5, KM.ShiftModifier, False) == "shift+f5"
    # Shift+1 arrives as "!"; hotkeys bind the physical key.
    assert event(Qt.Key.Key_Exclam, KM.ControlModifier | KM.ShiftModifier, False) == (
        "ctrl+shift+1"
    )
    # macOS: Command is reported as Control, Control as Meta.
    assert event(Qt.Key.Key_G, KM.ControlModifier | KM.ShiftModifier, True) == "shift+meta+g"
    assert event(Qt.Key.Key_G, KM.MetaModifier, True) == "ctrl+g"


# ------------------------------------------------------------------ camera & stats
def test_detect_cameras_fills_the_device_list(
    controller: FakeController, dialog: SettingsDialog
) -> None:
    dialog.detect_cameras()
    assert _wait_until(dialog._detect_button.isEnabled)
    combo = dialog.widget_for("camera.device")
    assert isinstance(combo, QComboBox)
    values = [combo.itemData(i) for i in range(combo.count())]
    assert values == ["0", "2"]
    assert combo.currentData() == "0"  # the configured device stays selected
    assert "Integrated Camera" in combo.currentText()
    combo.setCurrentIndex(combo.findData("2"))
    assert dialog.apply()
    assert controller.applied[-1].camera.device == "2"


def test_live_stats_are_shown(controller: FakeController, dialog: SettingsDialog) -> None:
    controller.stats_changed.emit(
        {
            "fps": 11.94,
            "target_fps": 12.0,
            "inference_ms": 9.51,
            "skip_ratio": 0.4,
            "cpu_percent": 0.64,
            "state": "tracking",
            "state_label": "Tracking",
            "backend": "mediapipe",
        }
    )
    text = dialog.stats_text()
    for part in ("Tracking", "11.9 fps", "target 12", "9.5 ms", "40 % skipped", "CPU 0.6 %"):
        assert part in text


def test_format_stats_tolerates_junk() -> None:
    assert format_stats(None) == ""
    assert format_stats({"fps": float("nan"), "state": "nonsense"}) == "nonsense"
    assert format_stats({"state": TrackingState.PAUSED, "cpu_percent": 1}) == "Paused · CPU 1.0 %"


# --------------------------------------------------------------- start at login
def test_start_at_login_checkbox_uses_autostart(
    fake_autostart: FakeAutostart, dialog: SettingsDialog
) -> None:
    box = next(
        b for b in dialog.findChildren(QCheckBox) if b.text() == "Start Eye Tracker when I log in"
    )
    assert box.isEnabled()
    assert not box.isChecked()
    box.setChecked(True)
    assert dialog.is_modified()
    assert dialog.apply()
    assert fake_autostart.calls == ["enable"]
    fake_autostart.fail = "access denied"
    box.setChecked(False)
    assert dialog.apply() is False
    assert "access denied" in dialog._status.text()


# ------------------------------------------------------------------ diagnostics
def test_diagnostics_page_copy_and_log_folder(
    monkeypatch: pytest.MonkeyPatch, app_dirs: Any, dialog: SettingsDialog
) -> None:
    dialog.show_page("diagnostics")
    assert dialog.diagnostics_text() == "FAKE DIAGNOSTICS REPORT"

    QTest.mouseClick(_button(dialog, "Copy report"), Qt.MouseButton.LeftButton)
    clipboard = QGuiApplication.clipboard()
    assert clipboard is not None
    assert clipboard.text() == "FAKE DIAGNOSTICS REPORT"

    opened: list[QUrl] = []

    class Desktop:
        @staticmethod
        def openUrl(url: QUrl) -> bool:
            opened.append(url)
            return True

    monkeypatch.setattr(sd, "QDesktopServices", Desktop)
    QTest.mouseClick(_button(dialog, "Open log folder"), Qt.MouseButton.LeftButton)
    from eye_tracker import paths

    assert [u.toLocalFile() for u in opened] == [
        QUrl.fromLocalFile(str(paths.log_dir())).toLocalFile()
    ]


def test_diagnostics_failure_is_shown_not_raised(
    monkeypatch: pytest.MonkeyPatch, dialog: SettingsDialog
) -> None:
    def broken() -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(sd, "load_diagnostics_text", broken)
    dialog.refresh_diagnostics()
    assert "boom" in dialog.diagnostics_text()


def test_diagnostics_module_absence_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    from eye_tracker import diagnostics

    monkeypatch.undo()  # use the real loader
    monkeypatch.delattr(diagnostics, "collect_report")
    assert "not available" in sd.load_diagnostics_text()


def test_recalibrate_button_requests_calibration(dialog: SettingsDialog) -> None:
    requested: list[bool] = []
    dialog.calibration_requested.connect(lambda: requested.append(True))
    QTest.mouseClick(_button(dialog, "Recalibrate…"), Qt.MouseButton.LeftButton)
    assert requested == [True]


def test_unsupported_hotkeys_explain_the_ctl_alternative(controller: FakeController) -> None:
    from eye_tracker.platform.hotkeys import HotkeyManager

    controller.hotkey_manager = HotkeyManager(note="Wayland session")
    dlg = SettingsDialog(controller)
    try:
        texts = " ".join(label.text() for label in dlg.findChildren(QLabel))
        assert "Wayland session" in texts
        assert "eye-tracker ctl toggle" in texts
    finally:
        _dispose(dlg)


# =============================================================== calibration window
def _observation(
    target: CalibrationTarget | None,
    now: float,
    rng: np.random.Generator,
    *,
    face: bool = True,
    usable: bool = True,
    skipped: bool = False,
) -> Observation:
    """A synthetic observation whose features encode the dot being looked at."""
    features = None
    if face and usable and target is not None:
        x, y = target.x / 1000.0, target.y / 1000.0
        features = np.array([x, y, 0.3 * x - 0.2 * y, 1.0]) + rng.normal(0, 0.004, 4)
    return Observation(
        timestamp=now,
        face_count=1 if face else 0,
        features=features,
        quality=1.0 if face else 0.0,
        skipped=skipped,
    )


def _make_window(
    controller: FakeController, clock: FakeClock, *, fit_in_thread: bool = False
) -> CalibrationWindow:
    return CalibrationWindow(
        controller,
        clock=clock,
        points_per_monitor=5,
        settle_s=0.2,
        collect_s=0.3,
        min_samples=3,
        max_retries=1,
        fit_in_thread=fit_in_thread,
    )


def _run_dots(
    window: CalibrationWindow,
    controller: FakeController,
    clock: FakeClock,
    make: Callable[[CalibrationTarget | None], Observation | None],
    *,
    step: float = 0.05,
    max_steps: int = 2000,
) -> None:
    """Advance the fake clock tick by tick, feeding one observation per tick."""
    for _ in range(max_steps):
        if window.state != cw.STATE_RUNNING:
            return
        obs = make(window.current_target)
        if obs is not None:
            controller.observation.emit(obs)
        clock.advance(step)
        window.tick()
    raise AssertionError("calibration did not finish")


@pytest.fixture
def calib() -> Iterator[tuple[CalibrationWindow, FakeController, FakeClock]]:
    controller = FakeController()
    clock = FakeClock()
    window = _make_window(controller, clock)
    yield window, controller, clock
    window.cancel()
    _dispose(window)


def test_calibration_opens_one_frameless_topmost_surface_per_monitor(
    calib: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, _clock = calib
    states: list[str] = []
    window.state_changed.connect(states.append)
    assert window.start()
    assert controller.begun == 1
    assert window.state == cw.STATE_INTRO
    assert states == [cw.STATE_INTRO]
    surfaces = window.surfaces()
    assert [s.monitor for s in surfaces] == [LEFT, RIGHT]  # type: ignore[attr-defined]
    for surface in surfaces:
        flags = surface.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint
        assert flags & Qt.WindowType.WindowStaysOnTopHint
        assert surface.isVisible()
    # Calling start() again only raises the windows.
    assert window.start()
    assert controller.begun == 1


def test_full_calibration_produces_calibration_data(
    calib: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = calib
    rng = np.random.default_rng(7)
    saved: list[bool] = []
    window.finished.connect(saved.append)
    window.start()
    QApplication.processEvents()  # let the surfaces settle at full-screen size, as in the app
    # The intro shows whether the camera sees a face.
    controller.observation.emit(_observation(None, clock(), rng))
    window.tick()
    chips = [s.card.labels["face"].text() for s in window.surfaces()]  # type: ignore[attr-defined]
    assert all("Face detected" in text for text in chips)

    QTest.keyClick(window.surfaces()[0], Qt.Key.Key_Space)
    assert window.state == cw.STATE_RUNNING
    assert len(window.plan) == 10
    seen_monitors: set[int] = set()

    def make(target: CalibrationTarget | None) -> Observation:
        if target is not None:
            seen_monitors.add(target.monitor_index)
        return _observation(target, clock(), rng)

    progress_before = window.progress
    _run_dots(window, controller, clock, make)
    assert seen_monitors == {0, 1}
    assert progress_before < 0.1
    assert window.state == cw.STATE_FITTING
    window.tick()  # the fit runs on the tick after "Calculating…" was shown
    assert window.state == cw.STATE_RESULT
    report = window.report
    assert report is not None
    assert report.grade == "excellent"
    assert report.per_monitor_accuracy == {0: 1.0, 1: 1.0}
    primary = window.surfaces()[0]
    card = primary.card  # type: ignore[attr-defined]
    assert set(card.buttons) == {"save", "retry", "cancel"}
    assert card.labels["badge"].text() == "EXCELLENT"
    # The card was rebuilt while visible: it must be sized for all of its content.
    QApplication.processEvents()
    for label in card.labels.values():
        assert label.isVisible()
        assert label.height() > 0
    assert card.height() >= card.layout().totalHeightForWidth(card.width()) - 1

    QTest.keyClick(primary, Qt.Key.Key_Return)
    assert window.state == cw.STATE_CLOSED
    assert saved == [True]
    assert window.surfaces() == []
    assert len(controller.finished) == 1
    data = controller.finished[0]
    assert isinstance(data, CalibrationData)
    assert data is window.data
    assert (data.backend, data.feature_version) == BACKEND
    assert data.monitors == [LEFT, RIGHT]
    assert data.layout_signature == layout_signature([LEFT, RIGHT])
    assert data.implicit_samples == []
    assert len(data.samples) == report.n_samples >= 30
    assert {s.point_id for s in data.samples} == {t.point_id for t in window.plan}
    assert data.model.is_fitted
    assert data.report["grade"] == "excellent"
    assert data.grade == "excellent"
    created = datetime.fromisoformat(data.created_at)
    assert created.utcoffset() == timedelta(0)
    ok, reason = data.is_compatible(*BACKEND, [LEFT, RIGHT])
    assert ok, reason


def test_calibration_fits_on_a_background_thread() -> None:
    controller = FakeController()
    clock = FakeClock()
    window = _make_window(controller, clock, fit_in_thread=True)
    rng = np.random.default_rng(3)
    try:
        window.start()
        window.begin()
        _run_dots(window, controller, clock, lambda t: _observation(t, clock(), rng))
        assert window.state == cw.STATE_FITTING
        spinner = window.surfaces()[0].card.spinner  # type: ignore[attr-defined]
        assert spinner is not None

        def done() -> bool:
            clock.advance(0.03)
            window.tick()
            return window.state != cw.STATE_FITTING

        assert _wait_until(done)
        assert window.state == cw.STATE_RESULT
        assert window.save()
        assert isinstance(controller.finished[-1], CalibrationData)
    finally:
        window.cancel()
        _dispose(window)


def test_escape_cancels_the_calibration(
    calib: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = calib
    results: list[bool] = []
    window.finished.connect(results.append)
    window.start()
    window.begin()
    rng = np.random.default_rng(1)
    for _ in range(5):
        controller.observation.emit(_observation(window.current_target, clock(), rng))
        clock.advance(0.05)
        window.tick()
    QTest.keyClick(window.surfaces()[1], Qt.Key.Key_Escape)  # any screen works
    assert window.state == cw.STATE_CLOSED
    assert controller.finished == [None]
    assert results == [False]
    assert window.surfaces() == []
    window.cancel()  # idempotent
    assert controller.finished == [None]


def test_closing_a_surface_cancels(
    calib: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, _clock = calib
    window.start()
    window.surfaces()[0].close()  # e.g. Alt+F4
    assert window.state == cw.STATE_CLOSED
    assert controller.finished == [None]


def test_missing_face_pauses_and_shows_a_hint(
    calib: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = calib
    rng = np.random.default_rng(2)
    window.start()
    window.begin()
    for _ in range(5):
        controller.observation.emit(_observation(window.current_target, clock(), rng))
        clock.advance(0.05)
        window.tick()
    for _ in range(40):  # 2 s of frames without a face (the grace period is 1.5 s)
        controller.observation.emit(_observation(None, clock(), rng, face=False))
        clock.advance(0.05)
        window.tick()
    assert window.paused
    assert window.hint() == "Can't see your face — check the camera"
    # While paused the dots stand still.
    progress = window.progress
    for _ in range(40):
        controller.observation.emit(_observation(None, clock(), rng, face=False))
        clock.advance(0.05)
        window.tick()
    assert window.progress == progress
    # No frames at all: the camera itself is the problem.
    clock.advance(3.0)
    window.tick()
    assert window.hint() == "Waiting for the camera…"
    # The face returns: the dots continue.
    controller.observation.emit(_observation(window.current_target, clock(), rng))
    clock.advance(0.05)
    window.tick()
    assert not window.paused
    assert window.hint() == ""


def test_space_pauses_manually(
    calib: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = calib
    rng = np.random.default_rng(4)
    window.start()
    window.begin()
    QTest.keyClick(window.surfaces()[0], Qt.Key.Key_Space)
    assert window.paused
    progress = window.progress
    for _ in range(20):
        controller.observation.emit(_observation(window.current_target, clock(), rng))
        clock.advance(0.05)
        window.tick()
    assert window.progress == progress
    assert "Paused" in window.hint()
    QTest.keyClick(window.surfaces()[0], Qt.Key.Key_Space)
    assert not window.paused


def test_skipped_observations_are_never_recorded_and_retry_works(
    calib: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = calib
    rng = np.random.default_rng(5)
    window.start()
    window.begin()
    # A face is visible (no pause) but every frame is a motion-gate copy.
    _run_dots(window, controller, clock, lambda t: _observation(t, clock(), rng, skipped=True))
    assert window.state == cw.STATE_FITTING
    window.tick()
    assert window.state == cw.STATE_ERROR
    assert "10 of 10 dots" in window.error
    assert controller.finished == []  # still open: the user may retry
    # R starts over.
    QTest.keyClick(window.surfaces()[0], Qt.Key.Key_R)
    assert window.state == cw.STATE_RUNNING
    assert window.error == ""
    _run_dots(window, controller, clock, lambda t: _observation(t, clock(), rng))
    window.tick()
    assert window.state == cw.STATE_RESULT


def test_retry_from_the_result_screen(
    calib: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = calib
    rng = np.random.default_rng(6)
    window.start()
    window.begin()
    _run_dots(window, controller, clock, lambda t: _observation(t, clock(), rng))
    window.tick()
    assert window.state == cw.STATE_RESULT
    QTest.keyClick(window.surfaces()[0], Qt.Key.Key_R)
    assert window.state == cw.STATE_RUNNING
    assert window.report is None
    assert window.progress < 0.1
    assert controller.finished == []


def test_save_refuses_without_backend_info(
    calib: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = calib
    rng = np.random.default_rng(8)
    window.start()
    window.begin()
    _run_dots(window, controller, clock, lambda t: _observation(t, clock(), rng))
    window.tick()
    controller.backend = ("", "")
    assert window.save() is False
    assert window.state == cw.STATE_ERROR
    assert controller.finished == []


def test_calibration_without_monitors_finishes_immediately() -> None:
    controller = FakeController(monitors=())
    window = CalibrationWindow(controller, clock=FakeClock())
    results: list[bool] = []
    window.finished.connect(results.append)
    try:
        assert window.start() is False
        assert results == [False]
        assert controller.begun == 0
        assert not window.is_active
    finally:
        _dispose(window)


# ==================================================================== first-run wizard
@pytest.fixture
def wizard_env() -> Iterator[tuple[FakeController, FakeClock, list[FirstRunWizard]]]:
    controller = FakeController()
    clock = FakeClock()
    created: list[FirstRunWizard] = []
    yield controller, clock, created
    for wizard in created:
        _dispose(wizard)


def _wizard(
    env: tuple[FakeController, FakeClock, list[FirstRunWizard]], **kwargs: Any
) -> FirstRunWizard:
    controller, clock, created = env
    wizard = FirstRunWizard(controller, clock=clock, **kwargs)
    created.append(wizard)
    wizard.show()
    return wizard


def test_wizard_flow_applies_choices(
    wizard_env: tuple[FakeController, FakeClock, list[FirstRunWizard]],
    fake_autostart: FakeAutostart,
) -> None:
    controller, clock, _ = wizard_env
    original = controller.settings.copy()
    wizard = _wizard(wizard_env, show_permissions=False)
    requested: list[bool] = []
    wizard.calibration_requested.connect(lambda: requested.append(True))
    assert wizard.currentId() == wz.PAGE_WELCOME

    wizard.next()
    assert wizard.currentId() == wz.PAGE_CAMERA
    assert controller.previews == [True]
    assert wizard.face_status() == "waiting"
    controller.observation.emit(Observation(timestamp=clock(), face_count=1, quality=1.0))
    assert wizard.face_status() == "face"
    assert wizard.camera_page.status.text().startswith("✓")
    clock.advance(1.5)
    controller.observation.emit(Observation(timestamp=clock(), face_count=0))
    assert wizard.face_status() == "no_face"
    assert wizard.camera_page.status.text().startswith("✗")
    controller.preview_frame.emit(np.zeros((48, 64, 3), dtype=np.uint8))
    assert wizard.camera_page.preview.has_image

    wizard.next()  # the permissions page is skipped off macOS
    assert wizard.currentId() == wz.PAGE_PRESENCE
    assert controller.previews == [True, False]
    wizard.set_presence_choice("display_off")
    wizard.presence_page.timeout.setValue(90)

    wizard.next()
    assert wizard.currentId() == wz.PAGE_FINISH
    assert wizard.finish_page.calibrate.isChecked()  # default on
    wizard.finish_page.autostart.setChecked(True)
    wizard.accept()

    assert wizard.result() == QDialog.DialogCode.Accepted
    assert len(controller.applied) == 1
    final = controller.applied[0]
    expected = original.copy()
    expected.general.first_run_done = True
    expected.presence.enabled = True
    expected.presence.action = "display_off"
    expected.presence.away_timeout_s = 90
    assert final == expected
    assert fake_autostart.calls == ["enable"]
    assert wizard.wants_calibration
    assert requested == [True]


def test_wizard_walk_away_off_and_no_calibration(
    wizard_env: tuple[FakeController, FakeClock, list[FirstRunWizard]],
    fake_autostart: FakeAutostart,
) -> None:
    controller, _clock, _ = wizard_env
    wizard = _wizard(wizard_env, show_permissions=False)
    requested: list[bool] = []
    wizard.calibration_requested.connect(lambda: requested.append(True))
    wizard.set_presence_choice("off")
    assert not wizard.presence_page.timeout.isEnabled()
    wizard.finish_page.calibrate.setChecked(False)
    wizard.accept()
    final = controller.applied[-1]
    assert final.presence.enabled is False
    assert final.general.first_run_done is True
    assert fake_autostart.calls == []  # unchanged: nothing to do
    assert requested == []
    assert not wizard.wants_calibration


def test_wizard_autostart_failure_is_reported_not_raised(
    wizard_env: tuple[FakeController, FakeClock, list[FirstRunWizard]],
    fake_autostart: FakeAutostart,
) -> None:
    controller, _clock, _ = wizard_env
    fake_autostart.fail = "no permission"
    wizard = _wizard(wizard_env, show_permissions=False)
    wizard.finish_page.autostart.setChecked(True)
    wizard.accept()
    assert controller.applied[-1].general.first_run_done
    assert wizard.autostart_error is not None
    assert "no permission" in wizard.autostart_error
    assert controller.notifications
    assert "no permission" in controller.notifications[-1][1]


def test_wizard_permissions_page_on_macos(
    wizard_env: tuple[FakeController, FakeClock, list[FirstRunWizard]],
) -> None:
    controller, _clock, _ = wizard_env
    wizard = _wizard(wizard_env, show_permissions=True)
    wizard.next()
    wizard.next()
    assert wizard.currentId() == wz.PAGE_PERMISSIONS
    status = wizard.permissions_page.status
    assert status["camera"].text() == "✓ Allowed"
    assert status["accessibility"].text() == "✗ Not allowed yet"
    assert not wizard.permissions_page.allow["camera"].isEnabled()
    assert wizard.permissions_page.allow["accessibility"].isEnabled()
    wizard.request_permission("accessibility")
    wizard.open_permission_settings("camera")
    assert controller.platform.calls == [("request", "accessibility"), ("open", "camera")]
    wizard.next()
    assert wizard.currentId() == wz.PAGE_PRESENCE


def test_wizard_camera_choice_is_live_and_reverted_on_cancel(
    wizard_env: tuple[FakeController, FakeClock, list[FirstRunWizard]],
) -> None:
    controller, _clock, _ = wizard_env
    wizard = _wizard(wizard_env, show_permissions=False)
    wizard.next()
    wizard.detect_cameras()
    combo = wizard.camera_page.device
    assert _wait_until(lambda: combo.count() == 2)
    assert [combo.itemData(i) for i in range(combo.count())] == ["0", "2"]
    wizard.select_device("2")
    assert controller.applied[-1].camera.device == "2"
    assert combo.currentData() == "2"
    wizard.reject()
    assert controller.applied[-1].camera.device == "0"
    assert controller.settings.general.first_run_done is False
    assert controller.previews[-1] is False


def test_wizard_reports_a_camera_that_is_off(
    wizard_env: tuple[FakeController, FakeClock, list[FirstRunWizard]],
) -> None:
    controller, clock, _ = wizard_env
    controller.state = TrackingState.PAUSED
    wizard = _wizard(wizard_env, show_permissions=False)
    wizard.next()
    assert wizard.face_status() == "off"
    controller.state_changed.emit(TrackingState.CAMERA_ERROR)
    assert wizard.face_status() == "error"
    controller.state_changed.emit(TrackingState.NEEDS_CALIBRATION)
    controller.observation.emit(Observation(timestamp=clock(), face_count=1, quality=1.0))
    assert wizard.face_status() == "face"


def test_wizard_frame_conversion_rejects_junk() -> None:
    assert wz.frame_to_image(None) is None
    assert wz.frame_to_image(np.zeros((4, 4), dtype=np.float32)) is None
    image = wz.frame_to_image(np.zeros((4, 6, 3), dtype=np.uint8))
    assert image is not None
    assert (image.width(), image.height()) == (6, 4)


def test_hotkeys_are_suspended_while_a_shortcut_is_recorded(
    controller: FakeController, dialog: SettingsDialog
) -> None:
    """Registered combinations never reach the dialog, so they are released while recording."""
    calls: list[bool] = []
    controller.suspend_hotkeys = calls.append  # type: ignore[attr-defined]
    edit = next(iter(dialog._hotkey_edits.values()))
    edit._set_recording(True)
    edit._set_recording(True)  # no duplicate notification
    edit._set_recording(False)
    assert calls == [True, False]
    edit._set_recording(True)
    edit.hide()  # closing the dialog mid-recording must restore the hotkeys
    assert calls == [True, False, True, False]
