"""Tests for the tray, icons, gaze overlay, countdown, curtain, preview and about UI.

Every widget is driven through a small fake controller that has the real
controller's signals and properties, so these tests exercise exactly what the
app wires up; one test also runs the widgets against the real ``Controller``
with a fake camera worker. Nothing here locks the screen, moves the real
cursor, registers hotkeys or writes login items.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QKeySequence,
    QPainter,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QWidget

from eye_tracker import __version__
from eye_tracker.config import Settings
from eye_tracker.platform.hotkeys import Hotkey, HotkeyManager, format_hotkey, parse_hotkey
from eye_tracker.types import (
    GazePoint,
    Monitor,
    Observation,
    Rect,
    TrackingState,
    WorkerStats,
)
from eye_tracker.ui import icons, util
from eye_tracker.ui.about import THIRD_PARTY, AboutDialog, system_info
from eye_tracker.ui.countdown import CountdownToast
from eye_tracker.ui.curtain import PrivacyCurtain
from eye_tracker.ui.overlay import GazeOverlay
from eye_tracker.ui.preview import PreviewWindow, bgr_to_qimage
from eye_tracker.ui.tray import TrayIcon

pytestmark = pytest.mark.usefixtures("qapp")

LEFT = Monitor(0, "Left", Rect(0, 0, 1920, 1080), primary=True)
RIGHT = Monitor(1, "Right", Rect(1920, 0, 1920, 1080))
S = TrackingState


# --------------------------------------------------------------------------- fakes
class FakeHotkeyManager:
    """What the tray reads from ``controller.hotkey_manager``."""

    supported = True
    note = None

    def __init__(self, registered: dict[str, str] | None = None) -> None:
        self.bindings = {name: parse_hotkey(text) for name, text in (registered or {}).items()}

    @property
    def registered(self) -> dict[str, Hotkey]:
        return dict(self.bindings)


class FakeController(QObject):
    """The Controller's public signals, properties and the methods the UI calls.

    The state is derived from the ``paused``/``privacy`` flags with the real
    controller's priority (privacy > paused).
    """

    state_changed = Signal(object)
    gaze_changed = Signal(object)
    observation = Signal(object)
    stats_changed = Signal(object)
    notify = Signal(str, str)
    switched = Signal(int)
    away_warning = Signal(float)
    away_cancelled = Signal()
    guard_changed = Signal(bool)
    preview_frame = Signal(object)
    calibration_required = Signal(str)
    settings_changed = Signal(object)
    ui_requested = Signal(str)

    def __init__(self, monitors: tuple[Monitor, ...] = (LEFT, RIGHT)) -> None:
        super().__init__()
        self.settings = Settings()
        self.state = S.TRACKING
        self.paused = False
        self.privacy = False
        self.preview_enabled = False
        self.hotkey_manager: FakeHotkeyManager | None = None
        self.yield_reason = ""
        self.calibration_reason = ""
        self._monitors = list(monitors)
        self.calls: list[tuple[str, Any]] = []

    def monitors(self) -> list[Monitor]:
        return list(self._monitors)

    def backend_info(self) -> tuple[str, str]:
        return ("mediapipe", "mp-pose-iris-1")

    def set_state(self, state: TrackingState) -> None:
        """Test helper: jump to ``state`` (flags follow)."""
        self.paused = state is S.PAUSED
        self.privacy = state is S.PRIVACY
        self._emit(state)

    def _emit(self, state: TrackingState) -> None:
        if state is not self.state:
            self.state = state
            self.state_changed.emit(state)

    def _derive(self) -> None:
        self._emit(S.PRIVACY if self.privacy else S.PAUSED if self.paused else S.TRACKING)

    def set_preview(self, enabled: bool) -> None:
        self.calls.append(("set_preview", enabled))
        self.preview_enabled = enabled

    def set_privacy(self, enabled: bool) -> None:
        self.calls.append(("set_privacy", enabled))
        self.privacy = enabled
        self._derive()

    def toggle_pause(self) -> None:
        self.calls.append(("toggle_pause", None))
        self.paused = not self.paused
        self._derive()

    def apply_settings(self, settings: Settings) -> None:
        self.calls.append(("apply_settings", settings))
        self.settings = settings.copy()
        self.settings_changed.emit(self.settings)


class FakeAutostart:
    def __init__(self, *, supported: bool = True, fail: bool = False) -> None:
        self.supported = supported
        self.fail = fail
        self.enabled = False
        self.background: bool | None = None

    def is_supported(self) -> bool:
        return self.supported

    def is_enabled(self) -> bool:
        return self.enabled

    def enable(self, background: bool = True) -> None:
        if self.fail:
            raise OSError("Could not enable start at login: access denied")
        self.enabled = True
        self.background = background

    def disable(self) -> None:
        self.enabled = False


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def controller() -> FakeController:
    return FakeController()


@pytest.fixture
def cleanup() -> Iterator[list[Any]]:
    """Objects to close and delete after the test (keeps windows from piling up)."""
    items: list[Any] = []
    yield items
    for item in reversed(items):
        for method in ("dispose", "close"):
            func = getattr(item, method, None)
            if callable(func):
                func()
                break
        if isinstance(item, QObject):
            item.deleteLater()
    QApplication.processEvents()


def _opaque_pixels(image: QImage, threshold: int = 40) -> list[QColor]:
    return [
        image.pixelColor(x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > threshold
    ]


def _rect_tuple(widget: QWidget) -> tuple[int, int, int, int]:
    g = widget.geometry()
    return (g.x(), g.y(), g.width(), g.height())


def _screen_monitors() -> tuple[Monitor, Monitor]:
    """Two monitors whose first one is exactly the offscreen primary screen."""
    g = QGuiApplication.primaryScreen().geometry()
    first = Monitor(0, "A", Rect(g.x(), g.y(), g.width(), g.height()), primary=True)
    second = Monitor(1, "B", Rect(g.x() + g.width(), g.y(), 1280, 720))
    return first, second


# --------------------------------------------------------------------------- icons
@pytest.mark.parametrize("state", list(TrackingState))
def test_tray_images_render_for_every_state(state: TrackingState) -> None:
    for dark in (True, False):
        for mask in (False, True):
            for size in (16, 22, 32, 64):
                image = icons.render_tray_image(state, size, dark=dark, mask=mask)
                assert not image.isNull()
                assert (image.width(), image.height()) == (size, size)
                assert len(_opaque_pixels(image)) > size  # a glyph was drawn
                assert image.pixelColor(0, 0).alpha() < 60  # corners stay see-through
    icon = icons.tray_icon(state, dark=True, mask=False)
    assert not icon.isNull()
    for size in (16, 24, 48):
        assert not icon.pixmap(size, size).isNull()
    assert not icon.isMask()


def test_tray_template_icons_are_monochrome_masks() -> None:
    for state in TrackingState:
        icon = icons.tray_icon(state, mask=True)
        assert icon.isMask()
        image = icons.render_tray_image(state, 32, mask=True)
        colours = _opaque_pixels(image, threshold=200)
        assert colours
        assert all(c.red() == c.green() == c.blue() == 0 for c in colours)


def test_tray_glyph_colour_follows_taskbar_theme() -> None:
    def mean_lightness(image: QImage) -> float:
        colours = _opaque_pixels(image, threshold=220)
        return sum(c.lightness() for c in colours) / len(colours)

    on_dark = icons.render_tray_image(S.PAUSED, 32, dark=True)
    on_light = icons.render_tray_image(S.PAUSED, 32, dark=False)
    assert mean_lightness(on_dark) > 180
    assert mean_lightness(on_light) < 90


def test_tray_glyph_has_a_contrasting_halo() -> None:
    # A light glyph (dark taskbar guess) still has a dark rim, so it stays
    # readable if the guess was wrong and the taskbar is light.
    image = icons.render_tray_image(S.TRACKING, 32, dark=True)
    rim = [c for c in _opaque_pixels(image, threshold=20) if c.lightness() < 60]
    assert rim


def test_tray_states_are_visually_distinct() -> None:
    states = [
        S.TRACKING,
        S.PAUSED,
        S.PRIVACY,
        S.AWAY,
        S.CAMERA_ERROR,
        S.NEEDS_CALIBRATION,
        S.STARTING,
        S.CALIBRATING,
    ]
    images = {s: icons.render_tray_image(s, 32, dark=True) for s in states}
    for i, a in enumerate(states):
        for b in states[i + 1 :]:
            assert images[a] != images[b], (a, b)


def test_error_badge_is_red_and_calibration_badge_amber() -> None:
    error = icons.render_tray_image(S.CAMERA_ERROR, 32, dark=True)
    badge = error.pixelColor(25, 25)  # centre of the badge on the 16-unit grid
    assert badge.red() > 200
    assert badge.green() < 110
    assert badge.alpha() > 240
    amber = icons.render_tray_image(S.NEEDS_CALIBRATION, 32, dark=True).pixelColor(25, 25)
    assert amber.red() > 200
    assert 120 < amber.green() < 200
    assert amber.blue() < 80


def test_tray_icon_is_cached() -> None:
    first = icons.tray_icon(S.PAUSED, dark=False, mask=False)
    assert first.cacheKey() == icons.tray_icon(S.PAUSED, dark=False, mask=False).cacheKey()
    icons.clear_cache()
    assert not icons.tray_icon(S.PAUSED, dark=False, mask=False).isNull()


def test_app_icon_renders_at_every_size() -> None:
    icon = icons.app_icon()
    assert not icon.isNull()
    for size in (16, 32, 256):
        assert not icon.pixmap(size, size).isNull()
    image = icons.render_app_icon_png(256)
    assert image.size().width() == 256
    assert image.pixelColor(0, 0).alpha() == 0  # rounded corner
    assert image.pixelColor(128, 20).alpha() == 255  # gradient tile
    assert image.pixelColor(70, 128).lightness() > 240  # white of the eye
    padded = icons.render_app_icon_png(256, padded=True)
    assert padded.pixelColor(12, 128).alpha() < image.pixelColor(12, 128).alpha()
    with pytest.raises(ValueError, match="positive"):
        icons.render_app_icon_png(0)
    with pytest.raises(ValueError, match="positive"):
        icons.render_tray_image(S.TRACKING, 0)


def test_small_glyphs() -> None:
    assert not icons.status_dot_icon(util.SUCCESS).pixmap(16, 16).isNull()
    image = QImage(64, 64, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    icons.paint_lock(painter, QRectF(0, 0, 64, 64), QColor("white"))
    painter.end()
    assert len(_opaque_pixels(image)) > 400
    assert image.pixelColor(32, 42).alpha() == 0  # the keyhole is cut out


# ---------------------------------------------------------------------------- util
def test_ui_scale_and_formatting() -> None:
    assert util.ui_scale(None) == pytest.approx(1.0)  # offscreen screens are 96 DPI
    assert util.scaled(28, 1.5) == 42
    assert util.scaled(0.2, 1.0) == 1
    assert util.format_fps(4.2) == "4 fps"
    assert util.format_fps(12.0) == "12 fps"
    assert util.format_fps(0.5) == "0.5 fps"
    assert util.format_fps(float("nan")) == "0 fps"
    assert util.format_cpu(0.64) == "CPU 0.6 %"
    assert util.format_cpu(float("inf")) == "CPU 0.0 %"
    assert util.elide("a  b\nc", 10) == "a b c"
    assert util.elide("abcdefghijkl", 6) == "abcde…"
    assert util.monitor_label(RIGHT) == "Monitor 2 · Right"


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS UI is measured in points")
def test_ui_scale_follows_logical_dpi() -> None:
    class Screen:
        def __init__(self, dpi: float) -> None:
            self.dpi = dpi

        def logicalDotsPerInch(self) -> float:
            return self.dpi

    assert util.ui_scale(Screen(144.0)) == pytest.approx(1.5)  # type: ignore[arg-type]
    assert util.ui_scale(Screen(192.0)) == pytest.approx(2.0)  # type: ignore[arg-type]
    assert util.ui_scale(Screen(float("nan"))) == 1.0  # type: ignore[arg-type]
    assert util.ui_scale(Screen(0.0)) == 1.0  # type: ignore[arg-type]
    assert util.ui_scale(Screen(10_000.0)) == 4.0  # type: ignore[arg-type]


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS shows ⌃⌥ glyphs instead")
def test_hotkey_sequence_display() -> None:
    portable = QKeySequence.SequenceFormat.PortableText
    sequence = util.hotkey_sequence("ctrl+alt+t")
    assert sequence is not None
    assert sequence.toString(portable) == "Ctrl+Alt+T"
    shift_f5 = util.hotkey_sequence("Shift+F5")
    assert shift_f5 is not None
    assert shift_f5.toString(portable) == "Shift+F5"
    parsed = util.hotkey_sequence(parse_hotkey("ctrl+shift+space"))
    assert parsed is not None
    assert parsed.toString(portable) == "Ctrl+Shift+Space"
    for bad in ("", "   ", "p", "ctrl+", "ctrl+f99", None):
        assert util.hotkey_sequence(bad) is None
    assert util.parse_hotkey_text("nonsense") is None
    assert util.parse_hotkey_text("ctrl+alt+p") == parse_hotkey("ctrl+alt+p")


def test_hotkey_sequence_swaps_control_and_command_on_macos() -> None:
    sequence = util.hotkey_sequence("ctrl+meta+k", macos=True)
    assert sequence is not None
    combo = sequence[0]  # type: ignore[index]  # QKeySequence is indexable at runtime
    # Qt's ControlModifier is Command on macOS; the physical Control key is Meta.
    assert combo.keyboardModifiers() == (
        Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ControlModifier
    )
    alt_only = util.hotkey_sequence("ctrl+alt+k", macos=True)
    assert alt_only is not None
    modifiers = alt_only[0].keyboardModifiers()  # type: ignore[index]
    assert modifiers & Qt.KeyboardModifier.MetaModifier
    assert not modifiers & Qt.KeyboardModifier.ControlModifier


def test_screen_for_monitor_matches_geometry() -> None:
    screen = QGuiApplication.primaryScreen()
    first, _second = _screen_monitors()
    assert util.screen_for_monitor(Monitor(0, "whatever", first.rect)) == screen
    assert util.screen_for_monitor(Monitor(0, screen.name(), Rect(1, 2, 3, 4))) == screen
    # Unknown geometry and name: fall back to the screen under the centre / by index.
    assert util.screen_for_monitor(Monitor(0, "", Rect(10, 10, 50, 50))) is not None
    assert util.screen_for_point(-99999, -99999) == screen
    assert util.primary_monitor([RIGHT, LEFT]) == LEFT
    assert util.primary_monitor([RIGHT]) == RIGHT
    assert util.primary_monitor([]) is None


def test_controller_accessors_are_defensive() -> None:
    class Broken:
        state = "nonsense"

        def monitors(self) -> list[Monitor]:
            raise RuntimeError("boom")

        def settings(self) -> Settings:
            raise RuntimeError("boom")

    broken = Broken()
    assert isinstance(util.controller_settings(broken), Settings)
    assert util.controller_state(broken) is S.STARTING
    assert util.controller_monitors(broken) == []
    assert util.controller_monitors(object()) == []

    class Good:
        def __init__(self) -> None:
            self.custom = Settings()
            self.custom.general.notifications = False

        def settings(self) -> Settings:
            return self.custom

        @property
        def state(self) -> TrackingState:
            return S.PAUSED

    good = Good()
    assert util.controller_settings(good) is good.custom
    assert util.controller_state(good) is S.PAUSED
    assert isinstance(util.system_tray_is_dark(), bool)


# ---------------------------------------------------------------------------- tray
def _tray(controller: Any, cleanup: list[Any], **kwargs: Any) -> TrayIcon:
    kwargs.setdefault("autostart", FakeAutostart())
    tray = TrayIcon(controller, **kwargs)
    cleanup.append(tray)
    return tray


def _capture_messages(
    tray: TrayIcon, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, str, bool]]:
    shown: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(tray, "_messages_available", lambda: True)
    monkeypatch.setattr(tray, "_show_message", lambda t, m, c: shown.append((t, m, c)))
    return shown


@pytest.mark.parametrize("state", list(TrackingState))
def test_tray_menu_reflects_state(
    state: TrackingState, controller: FakeController, cleanup: list[Any]
) -> None:
    tray = _tray(controller, cleanup)
    controller.set_state(state)
    assert tray.state is state
    expected_pause = "Resume tracking" if state is S.PAUSED else "Pause tracking"
    assert tray.action_pause.text() == expected_pause
    assert tray.action_privacy.isCheckable()
    assert tray.action_privacy.isChecked() == (state is S.PRIVACY)
    assert tray.action_status.text().startswith(state.label)
    assert not tray.action_status.isEnabled()
    assert not tray.tray.icon().isNull()
    assert tray.tray.toolTip().startswith(f"Eye Tracker — {state.label}")
    needs = state is S.NEEDS_CALIBRATION
    assert tray.action_calibrate.font().bold() == needs
    assert tray.action_calibrate.text() == ("Calibrate now…" if needs else "Calibrate…")
    busy = state is S.CALIBRATING
    assert tray.action_pause.isEnabled() == (not busy)
    assert tray.action_privacy.isEnabled() == (not busy)
    assert tray.action_calibrate.isEnabled() == (not busy)


def test_tray_menu_layout(controller: FakeController, cleanup: list[Any]) -> None:
    tray = _tray(controller, cleanup)
    actions = [a for a in tray.menu.actions() if not a.isSeparator()]
    assert [a.text() for a in actions][1:] == [
        "Pause tracking",
        "Privacy mode",
        "Calibrate…",
        "Show gaze dot",
        "Camera preview…",
        "Settings…",
        "Start at login",
        "About Eye Tracker",
        "Quit Eye Tracker",
    ]
    # macOS must not move "About"/"Quit"/"Settings" into an application menu.
    assert all(a.menuRole() == a.MenuRole.NoRole for a in actions)
    checkable = {a.text() for a in actions if a.isCheckable()}
    assert checkable == {"Privacy mode", "Show gaze dot", "Start at login"}


def test_tray_pause_text_follows_the_paused_flag(
    controller: FakeController, cleanup: list[Any]
) -> None:
    # Paused *and* in privacy mode: the state shows PRIVACY, but leaving privacy
    # mode would still leave tracking paused, so the item must offer "Resume".
    tray = _tray(controller, cleanup)
    tray.action_pause.trigger()
    assert controller.state is S.PAUSED
    tray.action_privacy.trigger()
    assert controller.state is S.PRIVACY
    assert tray.action_privacy.isChecked()
    assert tray.action_pause.text() == "Resume tracking"
    tray.action_pause.trigger()  # resumes while privacy mode stays on
    assert controller.state is S.PRIVACY  # no state change, but the flag changed
    assert tray.action_pause.text() == "Pause tracking"
    tray.action_privacy.trigger()
    assert controller.state is S.TRACKING
    assert not tray.action_privacy.isChecked()


def test_tray_tooltip_format(controller: FakeController, cleanup: list[Any]) -> None:
    tray = _tray(controller, cleanup)
    controller.stats_changed.emit({"fps": 4.2, "cpu_percent": 0.64, "state": "tracking"})
    assert tray.tooltip_text() == "Eye Tracker — Tracking · 4 fps · CPU 0.6 %"
    assert tray.tray.toolTip() == tray.tooltip_text()
    controller.set_state(S.PAUSED)  # camera off: no frame rate
    assert tray.tray.toolTip() == "Eye Tracker — Paused · CPU 0.6 %"
    controller.set_state(S.CAMERA_ERROR)
    controller.stats_changed.emit({"last_error": "Camera 0 could not be opened"})
    assert tray.tray.toolTip() == "Eye Tracker — Camera unavailable"
    assert tray.status_text() == "Camera unavailable — Camera 0 could not be opened"
    tray.update_stats({"fps": float("nan"), "cpu_percent": "x"})  # garbage is ignored
    tray.update_stats("not a mapping")  # type: ignore[arg-type]
    controller.set_state(S.TRACKING)
    assert tray.tooltip_text() == "Eye Tracker — Tracking"
    tray.update_stats({"fps": 0.5, "cpu_percent": True})
    assert tray.tooltip_text() == "Eye Tracker — Tracking · 0.5 fps"
    assert len(tray.tooltip_text()) <= 127


def test_tray_status_line_explains_yield_and_calibration(
    controller: FakeController, cleanup: list[Any]
) -> None:
    tray = _tray(controller, cleanup)
    controller.yield_reason = "zoom.exe is running"
    controller.set_state(S.YIELDED)
    assert tray.action_status.text() == "Paused — zoom.exe is running"
    assert tray.tray.toolTip().startswith("Eye Tracker — Camera in use by another app")
    controller.yield_reason = ""
    controller.set_state(S.TRACKING)
    controller.set_state(S.YIELDED)
    assert tray.action_status.text() == S.YIELDED.label

    controller.calibration_reason = "the monitor layout changed"
    controller.set_state(S.NEEDS_CALIBRATION)
    assert "the monitor layout changed" in tray.action_calibrate.toolTip()


def test_tray_actions_drive_controller(controller: FakeController, cleanup: list[Any]) -> None:
    tray = _tray(controller, cleanup)
    emitted: list[str] = []
    for name in ("open_settings", "open_calibration", "open_preview", "open_about"):
        getattr(tray, name).connect(lambda name=name: emitted.append(name))
    tray.quit_requested.connect(lambda: emitted.append("quit"))

    tray.action_privacy.trigger()
    assert ("set_privacy", True) in controller.calls
    assert tray.action_privacy.isChecked()
    tray.action_privacy.trigger()
    assert ("set_privacy", False) in controller.calls
    assert not tray.action_privacy.isChecked()

    tray.action_pause.trigger()
    assert tray.action_pause.text() == "Resume tracking"
    tray.action_pause.trigger()
    assert tray.action_pause.text() == "Pause tracking"
    tray._on_activated(QSystemTrayIcon.ActivationReason.MiddleClick)
    assert controller.paused

    tray.action_overlay.trigger()
    applied = [c[1] for c in controller.calls if c[0] == "apply_settings"]
    assert applied
    assert applied[-1].ui.show_gaze_overlay is True
    assert controller.settings.ui.show_gaze_overlay is True
    assert tray.action_overlay.isChecked()
    tray.action_overlay.trigger()
    assert controller.settings.ui.show_gaze_overlay is False
    assert not tray.action_overlay.isChecked()

    for action in (
        tray.action_settings,
        tray.action_calibrate,
        tray.action_preview,
        tray.action_about,
        tray.action_quit,
    ):
        action.trigger()
    tray._on_activated(QSystemTrayIcon.ActivationReason.DoubleClick)
    tray._on_activated(QSystemTrayIcon.ActivationReason.Trigger)  # single click: nothing
    assert emitted == [
        "open_settings",
        "open_calibration",
        "open_preview",
        "open_about",
        "quit",
        "open_settings",
    ]


def test_tray_gaze_dot_keeps_other_settings(controller: FakeController, cleanup: list[Any]) -> None:
    tray = _tray(controller, cleanup)
    # Changed behind the tray's back (e.g. through IPC): must not be undone.
    controller.settings.presence.action = "display_off"
    tray.action_overlay.trigger()
    assert controller.settings.ui.show_gaze_overlay is True
    assert controller.settings.presence.action == "display_off"


def test_tray_survives_failing_controller(cleanup: list[Any]) -> None:
    class Failing(FakeController):
        def apply_settings(self, settings: Settings) -> None:
            raise RuntimeError("disk full")

        def toggle_pause(self) -> None:
            raise RuntimeError("boom")

    controller = Failing()
    tray = _tray(controller, cleanup)
    tray.action_overlay.trigger()
    assert not tray.action_overlay.isChecked()  # reverted: the setting did not change
    tray.action_pause.trigger()  # logged, not raised
    assert tray.action_pause.text() == "Pause tracking"


def test_tray_autostart_toggle(controller: FakeController, cleanup: list[Any]) -> None:
    backend = FakeAutostart()
    tray = _tray(controller, cleanup, autostart=backend)
    assert tray.action_autostart.isVisible()
    assert not tray.action_autostart.isChecked()
    tray.action_autostart.trigger()
    assert backend.enabled
    assert backend.background is True  # a login start runs in the background
    assert tray.action_autostart.isChecked()
    tray.action_autostart.trigger()
    assert not backend.enabled
    assert not tray.action_autostart.isChecked()
    backend.enabled = True  # changed elsewhere, e.g. in Task Manager
    tray._on_menu_about_to_show()
    assert tray.action_autostart.isChecked()


def test_tray_autostart_failure_is_reported(
    controller: FakeController, cleanup: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    tray = _tray(controller, cleanup, autostart=FakeAutostart(fail=True))
    shown = _capture_messages(tray, monkeypatch)
    tray.action_autostart.trigger()
    assert not tray.action_autostart.isChecked()
    assert shown == [("Start at login", "Could not enable start at login: access denied", True)]


def test_tray_uses_the_platform_autostart_module(
    controller: FakeController, cleanup: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from eye_tracker.platform import autostart

    state = {"enabled": False}
    calls: list[tuple[str, object]] = []

    def enable(background: bool = True) -> None:
        calls.append(("enable", background))
        raise autostart.AutostartError("Could not enable start at login: registry is read-only")

    def disable() -> None:
        calls.append(("disable", None))
        state["enabled"] = False

    monkeypatch.setattr(autostart, "is_supported", lambda: True)
    monkeypatch.setattr(autostart, "is_enabled", lambda: state["enabled"])
    monkeypatch.setattr(autostart, "enable", enable)
    monkeypatch.setattr(autostart, "disable", disable)

    tray = TrayIcon(controller)  # default backend: eye_tracker.platform.autostart
    cleanup.append(tray)
    shown = _capture_messages(tray, monkeypatch)
    # Errors from a user action are shown even with notifications off and muted.
    quiet = controller.settings.copy()
    quiet.general.notifications = False
    controller.apply_settings(quiet)
    tray.mute_notifications(60.0)

    tray.action_autostart.trigger()
    assert calls == [("enable", True)]
    assert not tray.action_autostart.isChecked()
    assert shown == [
        ("Start at login", "Could not enable start at login: registry is read-only", True)
    ]

    state["enabled"] = True
    tray._on_menu_about_to_show()
    assert tray.action_autostart.isChecked()
    tray.action_autostart.trigger()
    assert calls[-1] == ("disable", None)
    assert not tray.action_autostart.isChecked()

    def explode(background: bool = True) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(autostart, "enable", explode)
    tray.action_autostart.trigger()
    assert shown[-1][1] == "The login item could not be changed; see the log for details."


def test_tray_hides_autostart_when_unsupported(
    controller: FakeController, cleanup: list[Any]
) -> None:
    tray = _tray(controller, cleanup, autostart=FakeAutostart(supported=False))
    assert not tray.action_autostart.isVisible()


def test_tray_notifications_respect_settings(
    controller: FakeController, cleanup: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    tray = _tray(controller, cleanup, clock=clock)
    shown = _capture_messages(tray, monkeypatch)

    controller.notify.emit("Switched", "hello")
    assert tray.notify("t", "own message")
    tray.mute_notifications(5.0)
    assert not tray.notify("t", "muted")
    controller.notify.emit("t", "muted controller message")  # a login start stays quiet
    assert tray.notify("t", "forced while muted", force=True, critical=True)
    clock.t += 6.0
    assert tray.notify("t", "unmuted")

    quiet = controller.settings.copy()
    quiet.general.notifications = False
    controller.apply_settings(quiet)
    assert not tray.notify("t", "suppressed")  # the app's own informational message
    assert tray.notify("t", "forced", force=True)
    assert [m for _t, m, _c in shown] == [
        "hello",
        "own message",
        "forced while muted",
        "unmuted",
        "forced",
    ]
    assert shown[2][2] is True  # critical → warning icon


def test_tray_does_not_filter_controller_notifications_twice(
    controller: FakeController, cleanup: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # With notifications off the controller only sends what the user asked for
    # (walk-away "notify", shoulder guard "notify") and failures: always shown.
    quiet = controller.settings.copy()
    quiet.general.notifications = False
    controller.apply_settings(quiet)
    tray = _tray(controller, cleanup)
    shown = _capture_messages(tray, monkeypatch)
    controller.notify.emit("Could not lock the screen", "This system does not allow it.")
    assert shown == [("Could not lock the screen", "This system does not allow it.", False)]


def test_tray_notify_without_tray_is_quiet(controller: FakeController, cleanup: list[Any]) -> None:
    tray = _tray(controller, cleanup)
    # The offscreen platform has no system tray: nothing is shown, nothing raises.
    tray.show()
    assert tray.notify("title", "message") is False
    tray.hide()


def test_tray_dispose_is_final_and_idempotent(
    controller: FakeController, cleanup: list[Any]
) -> None:
    tray = _tray(controller, cleanup)
    tray.dispose()
    tray.dispose()
    QApplication.processEvents()  # the menu is deleted now
    controller.set_state(S.PAUSED)  # late signals are ignored, not crashing
    controller.stats_changed.emit({"fps": 3.0})
    controller.settings_changed.emit(Settings())
    controller.notify.emit("late", "message")
    tray.show()
    assert tray.notify("t", "m", force=True) is False
    assert tray.state is S.TRACKING


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS shows ⌃⌥ glyphs instead")
def test_tray_shows_hotkeys_from_settings(controller: FakeController, cleanup: list[Any]) -> None:
    tray = _tray(controller, cleanup)
    portable = QKeySequence.SequenceFormat.PortableText
    assert tray.action_pause.shortcut().toString(portable) == "Ctrl+Alt+T"
    assert tray.action_privacy.shortcut().toString(portable) == "Ctrl+Alt+P"
    assert tray.action_calibrate.shortcut().toString(portable) == "Ctrl+Alt+C"
    assert tray.action_pause.shortcutContext() == Qt.ShortcutContext.WidgetShortcut
    assert tray.action_pause.toolTip().endswith("· Ctrl+Alt+T")
    changed = controller.settings.copy()
    changed.hotkeys.enabled = False
    controller.apply_settings(changed)
    assert tray.action_pause.shortcut().isEmpty()
    assert "Ctrl" not in tray.action_pause.toolTip()


def test_tray_shows_only_registered_hotkeys(controller: FakeController, cleanup: list[Any]) -> None:
    tray = _tray(controller, cleanup)
    # After controller.start(): only what the OS accepted is shown. Here the
    # privacy combination is owned by another app and recalibrate was remapped.
    controller.hotkey_manager = FakeHotkeyManager(
        {"toggle_tracking": "ctrl+alt+t", "recalibrate": "ctrl+shift+k"}
    )
    tray._on_menu_about_to_show()
    expected = util.hotkey_sequence("ctrl+shift+k")
    assert expected is not None
    assert tray.action_calibrate.shortcut() == expected
    assert tray.action_privacy.shortcut().isEmpty()
    label = format_hotkey(parse_hotkey("ctrl+shift+k"))
    assert tray.action_calibrate.toolTip().endswith(f"· {label}")

    controller.hotkey_manager = FakeHotkeyManager({})  # e.g. unsupported on Wayland
    tray._on_menu_about_to_show()
    assert tray.action_pause.shortcut().isEmpty()
    assert tray.action_calibrate.shortcut().isEmpty()


# ------------------------------------------------------------------------ countdown
def test_countdown_text_and_ticks(controller: FakeController, cleanup: list[Any]) -> None:
    clock = FakeClock()
    toast = CountdownToast(controller, clock=clock)
    cleanup.append(toast)
    controller.away_warning.emit(8.0)
    assert toast.isVisible()
    assert toast.active
    assert toast.text() == "Locking in 8 s — move the mouse or look at the camera to cancel"
    assert toast.accessibleName() == toast.text()
    clock.t += 1.2
    assert toast.title_text() == "Locking in 7 s"
    assert toast.remaining() == pytest.approx(6.8)
    toast._tick()
    assert toast.accessibleName().startswith("Locking in 7 s")
    assert not toast.grab().isNull()  # paints the ring and the text
    clock.t += 7.0
    assert toast.title_text() == "Locking now…"
    assert toast.seconds_left() == 0
    assert not toast.grab().isNull()  # empty ring
    controller.away_cancelled.emit()
    assert not toast.isVisible()
    assert not toast.active
    assert toast.remaining() == 0.0


@pytest.mark.parametrize(
    ("action", "title"),
    [
        ("lock", "Locking in 8 s"),
        ("lock_and_display_off", "Locking in 8 s"),
        ("display_off", "Turning displays off in 8 s"),
        ("notify", "Marking you as away in 8 s"),
    ],
)
def test_countdown_text_per_presence_action(
    action: str, title: str, controller: FakeController, cleanup: list[Any]
) -> None:
    settings = controller.settings.copy()
    settings.presence.action = action
    controller.apply_settings(settings)
    toast = CountdownToast(controller, clock=FakeClock())
    cleanup.append(toast)
    controller.away_warning.emit(7.5)
    assert toast.isVisible()
    assert toast.title_text() == title
    assert toast.text().startswith(f"{title} — ")


def test_countdown_is_centred_on_a_monitor(controller: FakeController, cleanup: list[Any]) -> None:
    toast = CountdownToast(controller, clock=FakeClock())
    cleanup.append(toast)
    toast.show_warning(10.0)
    centre = toast.geometry().center()
    cursor = QPoint(QGuiApplication.primaryScreen().geometry().center())
    expected = RIGHT if RIGHT.rect.contains(cursor.x(), cursor.y()) else LEFT
    assert expected.rect.contains(centre.x(), centre.y())
    assert abs(centre.x() - expected.rect.center[0]) <= 1
    assert toast.width() < expected.rect.w


def test_countdown_without_monitors_uses_the_primary_screen(cleanup: list[Any]) -> None:
    toast = CountdownToast(FakeController(monitors=()), clock=FakeClock())
    cleanup.append(toast)
    toast.show_warning(3.0)
    screen = QGuiApplication.primaryScreen().geometry()
    assert screen.contains(toast.geometry().center())


def test_countdown_follows_presence_action(controller: FakeController, cleanup: list[Any]) -> None:
    clock = FakeClock()
    toast = CountdownToast(controller, clock=clock)
    cleanup.append(toast)
    settings = controller.settings.copy()
    settings.presence.action = "display_off"
    controller.apply_settings(settings)
    toast.show_warning(4.5)
    assert toast.title_text() == "Turning displays off in 5 s"

    settings = settings.copy()
    settings.presence.action = "notify"
    settings.presence.require_input_idle = False
    controller.apply_settings(settings)
    assert toast.isVisible()
    assert toast.text() == "Marking you as away in 5 s — look at the camera to cancel"

    settings = settings.copy()
    settings.presence.action = "none"  # nothing will happen, so nothing to announce
    controller.apply_settings(settings)
    assert not toast.isVisible()
    toast.show_warning(5.0)
    assert not toast.isVisible()


def test_countdown_repeated_warnings_keep_the_ring_total(
    controller: FakeController, cleanup: list[Any]
) -> None:
    clock = FakeClock()
    toast = CountdownToast(controller, clock=clock)
    cleanup.append(toast)
    toast.show_warning(10.0)
    clock.t += 3.0
    toast.show_warning(7.0)  # the controller repeating its warning
    assert toast._total == pytest.approx(10.0)
    toast.show_warning(20.0)  # a later deadline: a new countdown
    assert toast._total == pytest.approx(20.0)
    toast.show_warning(float("nan"))
    assert toast.remaining() == 0.0


@pytest.mark.parametrize("state", [S.AWAY, S.PAUSED, S.PRIVACY, S.LOCKED, S.YIELDED])
def test_countdown_hides_when_state_changes(
    state: TrackingState, controller: FakeController, cleanup: list[Any]
) -> None:
    toast = CountdownToast(controller, clock=FakeClock())
    cleanup.append(toast)
    toast.show_warning(5.0)
    controller.set_state(state)
    assert not toast.isVisible()


def test_countdown_never_takes_focus(controller: FakeController, cleanup: list[Any]) -> None:
    toast = CountdownToast(controller, clock=FakeClock())
    cleanup.append(toast)
    flags = toast.windowFlags()
    assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.FramelessWindowHint
    assert toast.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)


# ------------------------------------------------------------------------- curtain
def _guarded(controller: FakeController, action: str = "curtain") -> None:
    settings = controller.settings.copy()
    settings.privacy.shoulder_guard = True
    settings.privacy.guard_action = action
    controller.apply_settings(settings)


def test_curtain_covers_every_screen(cleanup: list[Any]) -> None:
    monitors = _screen_monitors()
    controller = FakeController(monitors=monitors)
    curtain = PrivacyCurtain(controller)
    cleanup.append(curtain)
    _guarded(controller)
    controller.guard_changed.emit(True)
    assert curtain.is_showing
    windows = curtain.windows
    assert [_rect_tuple(w) for w in windows] == [tuple(m.rect.to_list()) for m in monitors]
    for window in windows:
        assert window.isVisible()
        assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert not windows[0].grab().isNull()
    controller.guard_changed.emit(True)  # repeated trigger: same windows
    assert curtain.windows == windows
    controller.guard_changed.emit(False)
    assert not curtain.is_showing
    assert not any(w.isVisible() for w in windows)


def test_curtain_also_covers_screens_the_controller_has_not_seen(cleanup: list[Any]) -> None:
    # Right after a hot-plug the controller's layout is still the old one.
    far = Monitor(0, "far", Rect(50_000, 0, 1280, 720), primary=True)
    controller = FakeController(monitors=(far,))
    curtain = PrivacyCurtain(controller)
    cleanup.append(curtain)
    curtain.show_curtain()
    rects = {_rect_tuple(w) for w in curtain.windows}
    expected = {tuple(far.rect.to_list())}
    for screen in QGuiApplication.screens():
        g = screen.geometry()
        expected.add((g.x(), g.y(), g.width(), g.height()))
    assert rects == expected


def test_curtain_rebuilds_after_a_screen_change(cleanup: list[Any]) -> None:
    first, second = _screen_monitors()
    controller = FakeController(monitors=(first,))
    curtain = PrivacyCurtain(controller)
    cleanup.append(curtain)
    curtain.show_curtain()
    assert len(curtain.windows) == 1
    controller._monitors = [first, second]  # a monitor was plugged in
    curtain._on_screens_changed()
    QApplication.processEvents()
    assert [w.monitor for w in curtain.windows] == [first, second]  # type: ignore[attr-defined]
    curtain.hide_curtain()
    curtain._on_screens_changed()  # nothing shown: nothing rebuilt
    QApplication.processEvents()
    assert not curtain.is_showing


def test_curtain_escape_and_button_dismiss(controller: FakeController, cleanup: list[Any]) -> None:
    curtain = PrivacyCurtain(controller)
    cleanup.append(curtain)
    dismissed: list[bool] = []
    curtain.dismissed.connect(lambda: dismissed.append(True))
    _guarded(controller)
    controller.guard_changed.emit(True)
    window = curtain.windows[1]
    QApplication.sendEvent(
        window, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
    )
    assert curtain.is_showing  # other keys do nothing
    QApplication.sendEvent(
        window,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier),
    )
    assert not curtain.is_showing
    assert dismissed == [True]
    curtain.dismiss()  # nothing shown: no second signal
    assert dismissed == [True]

    controller.guard_changed.emit(True)
    button = curtain.windows[0].button  # type: ignore[attr-defined]
    assert button.isVisible()
    button.click()
    assert not curtain.is_showing
    assert dismissed == [True, True]


def test_curtain_shows_whatever_the_controller_asks_for(
    controller: FakeController, cleanup: list[Any]
) -> None:
    curtain = PrivacyCurtain(controller)
    cleanup.append(curtain)
    # "lock" that could not lock: the controller falls back to the curtain.
    _guarded(controller, action="lock")
    controller.guard_changed.emit(True)
    assert curtain.is_showing

    controller.set_state(S.PRIVACY)  # the guard can never clear now
    assert not curtain.is_showing

    controller.guard_changed.emit(True)
    assert curtain.is_showing
    _guarded(controller, action="notify")  # switched to notifications only
    assert not curtain.is_showing

    _guarded(controller, action="curtain")
    controller.guard_changed.emit(True)
    assert curtain.is_showing
    settings = controller.settings.copy()
    settings.privacy.shoulder_guard = False
    controller.apply_settings(settings)
    assert not curtain.is_showing


def test_curtain_falls_back_to_qt_screens(cleanup: list[Any]) -> None:
    controller = FakeController(monitors=())
    curtain = PrivacyCurtain(controller)
    cleanup.append(curtain)
    curtain.show_curtain()
    assert len(curtain.windows) == len(QGuiApplication.screens())
    curtain.hide_curtain()
    assert not curtain.is_showing


# ------------------------------------------------------------------------- overlay
def _overlay(controller: FakeController, cleanup: list[Any], **kwargs: Any) -> GazeOverlay:
    overlay = GazeOverlay(controller, **kwargs)
    cleanup.append(overlay)
    return overlay


def _visible(overlay: GazeOverlay) -> dict[int, bool]:
    return {index: window.isVisible() for index, window in overlay.windows.items()}


def test_overlay_shows_dot_only_on_gaze_monitor(
    controller: FakeController, cleanup: list[Any]
) -> None:
    overlay = _overlay(controller, cleanup, follow_settings=False)
    overlay.set_enabled(True)
    assert overlay.windows == {}  # created lazily
    controller.gaze_changed.emit(GazePoint(2500.0, 400.0, 1.0))
    assert overlay.visible_monitor == 1
    assert _visible(overlay) == {1: True}
    right = overlay.windows[1]
    g = right.geometry()
    assert g.contains(2500, 400)
    assert RIGHT.rect.contains(g.left(), g.top())
    assert RIGHT.rect.contains(g.right(), g.bottom())
    assert right.windowFlags() & Qt.WindowType.WindowTransparentForInput  # click-through
    assert not right.grab().isNull()

    controller.gaze_changed.emit(GazePoint(300.0, 300.0, 2.0))
    assert overlay.visible_monitor == 0
    assert _visible(overlay) == {0: True, 1: False}

    controller.gaze_changed.emit(None)
    assert overlay.visible_monitor is None
    assert not any(_visible(overlay).values())


def test_overlay_dot_is_scaled_for_the_screen(
    controller: FakeController, cleanup: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(util, "ui_scale", lambda screen=None: 2.0)
    overlay = _overlay(controller, cleanup, follow_settings=False)
    overlay.set_enabled(True)
    overlay.set_gaze(GazePoint(500.0, 500.0, 0.0))
    window = overlay.windows[0]
    assert window.radius == pytest.approx(28.0)  # type: ignore[attr-defined]
    assert window.width() >= 2 * 28 * 2


def test_overlay_keeps_windows_inside_their_monitor(
    controller: FakeController, cleanup: list[Any]
) -> None:
    overlay = _overlay(controller, cleanup, follow_settings=False)
    overlay.set_enabled(True)
    overlay.set_gaze(GazePoint(1919.0, 1079.0, 0.0))  # bottom-right corner of LEFT
    window = overlay.windows[0]
    g = window.geometry()
    assert LEFT.rect.contains(g.right(), g.bottom())
    dot = window.dot  # type: ignore[attr-defined]
    assert (g.x() + dot.x(), g.y() + dot.y()) == (1919.0, 1079.0)

    overlay.set_gaze(GazePoint(-400.0, 500.0, 1.0))  # left of every monitor
    assert overlay.visible_monitor == 0
    assert window.off_screen  # type: ignore[attr-defined]
    assert window.geometry().left() == 0
    assert not window.grab().isNull()
    overlay.set_gaze(GazePoint(float("nan"), 1.0, 2.0))
    assert overlay.visible_monitor is None


def test_overlay_follows_settings_and_state(controller: FakeController, cleanup: list[Any]) -> None:
    overlay = _overlay(controller, cleanup)
    assert not overlay.enabled  # default setting is off
    controller.gaze_changed.emit(GazePoint(100.0, 100.0, 0.0))
    assert overlay.windows == {}
    settings = controller.settings.copy()
    settings.ui.show_gaze_overlay = True
    controller.apply_settings(settings)
    assert overlay.enabled
    controller.gaze_changed.emit(GazePoint(100.0, 100.0, 0.0))
    assert overlay.visible_monitor == 0
    controller.set_state(S.PAUSED)
    assert overlay.visible_monitor is None
    settings = settings.copy()
    settings.ui.show_gaze_overlay = False
    controller.apply_settings(settings)
    assert not overlay.enabled
    assert overlay.windows == {}


def test_overlay_starts_enabled_from_settings(cleanup: list[Any]) -> None:
    controller = FakeController()
    controller.settings.ui.show_gaze_overlay = True
    overlay = _overlay(controller, cleanup)
    assert overlay.enabled


def test_overlay_hides_stale_dot(controller: FakeController, cleanup: list[Any]) -> None:
    overlay = _overlay(controller, cleanup, follow_settings=False, stale_ms=10)
    overlay.set_enabled(True)
    overlay.set_gaze(GazePoint(100.0, 100.0, 0.0))
    assert overlay.visible_monitor == 0
    QTest.qWait(60)
    assert overlay.visible_monitor is None


def test_overlay_follows_layout_changes(controller: FakeController, cleanup: list[Any]) -> None:
    overlay = _overlay(controller, cleanup, follow_settings=False)
    overlay.set_enabled(True)
    overlay.set_gaze(GazePoint(2500.0, 100.0, 0.0))
    assert overlay.visible_monitor == 1
    # The right monitor was unplugged; the controller's layout changes a moment
    # later. The next gaze update picks it up without any extra call.
    controller._monitors = [LEFT]
    overlay.set_gaze(GazePoint(2500.0, 100.0, 1.0))
    assert overlay.visible_monitor == 0  # off-screen marker on the nearest monitor
    assert set(overlay.windows) == {0}

    overlay._on_screens_changed()  # Qt reported a screen change: windows dropped
    assert overlay.windows == {}
    assert overlay.visible_monitor is None
    overlay.set_gaze(GazePoint(10.0, 10.0, 2.0))
    assert overlay.visible_monitor == 0


# ------------------------------------------------------------------------- preview
def test_bgr_to_qimage_converts_colours() -> None:
    frame = np.zeros((2, 3, 3), np.uint8)
    frame[0, 0] = (255, 0, 0)  # blue in BGR
    frame[0, 1] = (0, 255, 0)
    frame[1, 2] = (0, 0, 255)  # red
    image = bgr_to_qimage(frame)
    assert (image.width(), image.height()) == (3, 2)
    assert image.pixelColor(0, 0) == QColor(0, 0, 255)
    assert image.pixelColor(1, 0) == QColor(0, 255, 0)
    assert image.pixelColor(2, 1) == QColor(255, 0, 0)
    frame[:] = 0  # the image owns a copy of the pixels
    assert image.pixelColor(0, 0) == QColor(0, 0, 255)


def test_bgr_to_qimage_other_layouts() -> None:
    grey = np.array([[0, 128], [255, 64]], np.uint8)
    image = bgr_to_qimage(grey)
    assert image.pixelColor(1, 0) == QColor(128, 128, 128)
    assert bgr_to_qimage(grey[:, :, None]).pixelColor(0, 1) == QColor(255, 255, 255)

    bgra = np.zeros((1, 1, 4), np.uint8)
    bgra[0, 0] = (10, 20, 30, 255)
    assert bgr_to_qimage(bgra).pixelColor(0, 0) == QColor(30, 20, 10)

    wide = np.zeros((4, 8, 3), np.uint8)
    wide[:, ::2] = (0, 0, 200)
    strided = wide[:, ::2]  # non-contiguous view
    image = bgr_to_qimage(strided)
    assert image.width() == 4
    assert image.pixelColor(3, 3) == QColor(200, 0, 0)

    odd = np.zeros((3, 5, 3), np.uint8)  # 15-byte rows need no padding either
    odd[2, 4] = (1, 2, 3)
    assert bgr_to_qimage(odd).pixelColor(4, 2) == QColor(3, 2, 1)

    rgb_view = np.zeros((2, 2, 3), np.uint8)[:, :, ::-1]  # negative strides
    assert not bgr_to_qimage(rgb_view).isNull()


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 7, 3), (5, 1, 3), (1, 1), (1, 1, 4)])
def test_bgr_to_qimage_single_row_or_column(shape: tuple[int, ...]) -> None:
    # numpy reports arbitrary strides for length-1 axes; the row size must not
    # come from them (a one-row frame used to give a null image).
    frame = np.full(shape, 77, np.uint8)
    image = bgr_to_qimage(frame)
    assert not image.isNull()
    assert (image.width(), image.height()) == (shape[1], shape[0])
    assert image.pixelColor(shape[1] - 1, shape[0] - 1).red() == 77


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 2, 3), np.float32),
        np.zeros((2, 2, 2), np.uint8),
        np.zeros((0, 4, 3), np.uint8),
        np.zeros((2, 2, 2, 3), np.uint8),
    ],
)
def test_bgr_to_qimage_rejects_bad_frames(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"image|shape"):
        bgr_to_qimage(frame)


def test_preview_window_lifecycle(controller: FakeController, cleanup: list[Any]) -> None:
    window = PreviewWindow(controller)
    cleanup.append(window)
    assert not window.preview_active
    window.show()
    assert controller.calls[-1] == ("set_preview", True)
    assert window.preview_active
    assert controller.preview_enabled
    assert window.values["backend"].text() == "mediapipe (mp-pose-iris-1)"
    assert "never saved" in window.note_label.text()

    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, :, 2] = 255
    controller.preview_frame.emit(frame)
    image = window.view.image
    assert image is not None
    assert (image.width(), image.height()) == (640, 480)
    assert image.pixelColor(10, 10) == QColor(255, 0, 0)
    controller.preview_frame.emit(np.zeros((2, 2), np.float64))  # ignored, logged once
    controller.preview_frame.emit("not a frame")
    assert window.view.image is image

    controller.observation.emit(
        Observation(
            timestamp=1.0,
            face_count=2,
            features=np.zeros(8),
            quality=1.0,
            head_yaw=-12.3,
            head_pitch=4.0,
        )
    )
    assert window.values["faces"].text() == "2"
    assert window.values["pose"].text() == "-12° / +4°"
    controller.observation.emit(Observation(timestamp=2.0, face_count=0))
    assert window.values["faces"].text() == "0"
    assert window.values["pose"].text() == "—"
    controller.stats_changed.emit(
        {
            "fps": 11.8,
            "target_fps": 12.0,
            "inference_ms": 9.44,
            "skip_ratio": 0.35,
            "cpu_percent": 0.84,
            "backend": "opencv",
        }
    )
    assert window.values["fps"].text() == "12 fps (target 12 fps)"
    assert window.values["inference"].text() == "9.4 ms"
    assert window.values["skipped"].text() == "35 %"
    assert window.values["cpu"].text() == "0.8 %"
    assert window.values["backend"].text().startswith("opencv")
    controller.gaze_changed.emit(GazePoint(2500.0, 10.0, 1.0))
    assert window.values["monitor"].text() == "Monitor 2 · Right"
    controller.gaze_changed.emit(GazePoint(-5000.0, 10.0, 1.0))
    assert window.values["monitor"].text() == "Off-screen"
    controller.gaze_changed.emit(None)
    assert window.values["monitor"].text() == "—"
    assert not window.grab().isNull()

    window.close()
    assert controller.calls[-1] == ("set_preview", False)
    assert not window.preview_active
    assert not controller.preview_enabled
    assert window.view.image is None
    window.show()  # can be reopened
    assert controller.calls[-1] == ("set_preview", True)
    window.hide()
    assert controller.calls[-1] == ("set_preview", False)
    assert [c for c in controller.calls if c[0] == "set_preview"] == [
        ("set_preview", True),
        ("set_preview", False),
        ("set_preview", True),
        ("set_preview", False),
    ]


def test_preview_ignores_frames_while_hidden_and_camera_off(
    controller: FakeController, cleanup: list[Any]
) -> None:
    window = PreviewWindow(controller)
    cleanup.append(window)
    controller.preview_frame.emit(np.zeros((4, 4, 3), np.uint8))
    assert window.view.image is None  # hidden: nothing converted
    window.show()
    controller.preview_frame.emit(np.zeros((4, 4, 3), np.uint8))
    assert window.view.image is not None
    controller.set_state(S.PRIVACY)
    assert window.view.image is None
    assert window.values["state"].text() == S.PRIVACY.label
    assert not window.view.grab().isNull()  # placeholder text path


# --------------------------------------------------------------------------- about
def test_about_dialog(cleanup: list[Any]) -> None:
    dialog = AboutDialog()
    cleanup.append(dialog)
    assert __version__ in dialog.version_label.text()
    assert "never saved" in dialog.privacy_label.text()
    notices = dialog.notices.toPlainText()
    for needle in ("MediaPipe", "YuNet", "OpenCV", "Qt 6", "LGPL-3.0", "Apache-2.0", "MIT"):
        assert needle in notices
    assert len(THIRD_PARTY) >= 6
    info = system_info()
    for needle in (__version__, "Python", "Qt"):
        assert needle in info
    dialog.copy_system_info()
    clipboard = QGuiApplication.clipboard()
    assert clipboard is not None
    assert clipboard.text() == info
    assert not dialog.grab().isNull()


def test_every_widget_uses_the_system_font(controller: FakeController, cleanup: list[Any]) -> None:
    # Custom-painted widgets derive their fonts from the application font so the
    # platform's UI font (Segoe UI, SF Pro, Cantarell…) is used everywhere.
    toast = CountdownToast(controller, clock=FakeClock())
    cleanup.append(toast)
    app_font = QFont(QApplication.font())
    assert toast._title_font.family() == app_font.family()
    assert isinstance(toast, QWidget)


# --------------------------------------------------------------- real controller
class _NoCursor:
    """Reads a fixed position; moving the real pointer would be a test bug."""

    def pos(self) -> tuple[int, int]:
        return (100, 100)

    def set_pos(self, x: int, y: int) -> bool:
        raise AssertionError("the UI tests must never move the cursor")


class _Worker:
    """The VisionWorker surface the controller drives (no camera, no thread)."""

    def __init__(self, *_factories_and_callbacks: Callable[..., Any]) -> None:
        self.stats = WorkerStats(fps=4.0, camera_open=True)
        self.backend_info: tuple[str, str] | None = ("fake", "fake-1")
        self.active: list[bool] = []

    def start(self) -> None:
        pass

    def stop(self, timeout: float = 3.0) -> None:
        pass

    def set_interval(self, seconds: float) -> None:
        pass

    def set_active(self, active: bool) -> None:
        self.active.append(active)

    def set_max_faces(self, n: int) -> None:
        pass

    def set_motion_gate(self, enabled: bool, threshold: float) -> None:
        pass

    def set_preview(self, enabled: bool) -> None:
        pass

    def reconfigure(self, source_factory: Any = None, backend_factory: Any = None) -> None:
        pass


class _Hotkeys(HotkeyManager):
    """Accepts every combination without touching the OS."""

    supported: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self.bindings: dict[str, Hotkey] = {}

    def register(self, name: str, hotkey: Hotkey | str, callback: Callable[[], None]) -> bool:
        self.bindings[name] = hotkey if isinstance(hotkey, Hotkey) else parse_hotkey(hotkey)
        return True

    def unregister_all(self) -> None:
        self.bindings.clear()

    @property
    def registered(self) -> dict[str, Hotkey]:
        return dict(self.bindings)

    def stop(self) -> None:
        self.bindings.clear()


def test_widgets_follow_the_real_controller(app_dirs: Any, cleanup: list[Any]) -> None:
    from eye_tracker import paths
    from eye_tracker.engine.controller import Controller
    from eye_tracker.platform.base import PlatformServices

    controller = Controller(
        Settings(),
        PlatformServices(),  # every OS action unsupported: nothing real happens
        worker_factory=_Worker,
        monitors_provider=lambda: [LEFT, RIGHT],
        cursor=_NoCursor(),
        hotkey_manager_factory=_Hotkeys,
    )
    tray = _tray(controller, cleanup)
    overlay = _overlay(controller, cleanup)  # type: ignore[arg-type]
    toast = CountdownToast(controller, clock=FakeClock())
    cleanup.append(toast)
    curtain = PrivacyCurtain(controller)
    cleanup.append(curtain)
    controller.start()
    try:
        # No calibration file in the temporary config dir.
        assert controller.state is S.NEEDS_CALIBRATION
        assert tray.state is S.NEEDS_CALIBRATION
        assert tray.action_calibrate.text() == "Calibrate now…"
        assert "not calibrated yet" in tray.action_calibrate.toolTip()

        tray._on_menu_about_to_show()  # hotkeys are registered by start()
        assert set(tray._hotkey_labels) == {"toggle_tracking", "toggle_privacy", "recalibrate"}

        tray.action_pause.trigger()
        assert controller.paused
        assert tray.state is S.PAUSED
        assert tray.action_pause.text() == "Resume tracking"
        tray.action_privacy.trigger()
        assert controller.privacy
        assert tray.state is S.PRIVACY
        assert tray.action_privacy.isChecked()
        assert tray.action_pause.text() == "Resume tracking"
        tray.action_privacy.trigger()
        tray.action_pause.trigger()
        assert tray.state is S.NEEDS_CALIBRATION
        assert not tray.action_privacy.isChecked()

        tray.action_overlay.trigger()
        assert controller.settings.ui.show_gaze_overlay
        assert overlay.enabled
        assert Settings.load(paths.settings_file()).ui.show_gaze_overlay
        assert not curtain.is_showing
        assert not toast.isVisible()
    finally:
        controller.shutdown()
