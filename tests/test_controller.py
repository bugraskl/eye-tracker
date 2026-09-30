"""Tests for eye_tracker.engine.controller.Controller.

Everything the controller touches is faked: the vision worker (observations are
pushed synchronously), the platform (lock/display/window calls are recorded,
never performed), the cursor, the clock, the monitor layout and the hotkey
manager. Qt runs offscreen; config and calibration files live in a temp dir.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from eye_tracker import paths
from eye_tracker.config import Settings
from eye_tracker.engine import controller as controller_module
from eye_tracker.engine.controller import COMMANDS, UI_COMMANDS, Controller, QtCursor, qt_monitors
from eye_tracker.gaze.calibration import CalibrationSample
from eye_tracker.gaze.model import GazeModel
from eye_tracker.gaze.store import CalibrationData, load_calibration, save_calibration
from eye_tracker.platform.base import PlatformServices
from eye_tracker.types import (
    Monitor,
    Observation,
    Rect,
    TrackingState,
    WindowRef,
    WorkerStats,
    layout_signature,
    virtual_bounds,
)

LEFT = Monitor(0, "left", Rect(0, 0, 1920, 1080), primary=True)
RIGHT = Monitor(1, "right", Rect(1920, 0, 1920, 1080))
MONITORS = [LEFT, RIGHT]
BACKEND = ("fake", "fake-1")
RIGHT_CENTRE = (2880, 540)
# Synthetic features: the gaze point in kilo-pixels, so a linear model is exact.
FEATURE_SCALE = 1000.0

S = TrackingState


# ----------------------------------------------------------------------------- fakes
class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


class FakeCursor:
    def __init__(self, pos: tuple[int, int] = (500, 500)) -> None:
        self.position = pos
        self.moves: list[tuple[int, int]] = []
        self.allow = True

    def pos(self) -> tuple[int, int]:
        return self.position

    def set_pos(self, x: int, y: int) -> bool:
        self.moves.append((x, y))
        if self.allow:
            self.position = (x, y)
        return self.allow


class FakeWorker:
    """Records every control call; observations/stats/previews are pushed by the test."""

    def __init__(
        self,
        source_factory: Callable[[], Any],
        backend_factory: Callable[[], Any],
        on_observation: Callable[[Observation], None],
        on_stats: Callable[[WorkerStats], None],
        on_preview: Callable[[np.ndarray], None],
    ) -> None:
        self.source_factory = source_factory
        self.backend_factory = backend_factory
        self.on_observation = on_observation
        self.on_stats = on_stats
        self.on_preview = on_preview
        self.intervals: list[float] = []
        self.active: list[bool] = []
        self.max_faces: list[int] = []
        self.gates: list[tuple[bool, float]] = []
        self.previews: list[bool] = []
        self.reconfigures: list[tuple[Any, Any]] = []
        self.started = False
        self.stopped = False
        self.backend_info: tuple[str, str] | None = BACKEND
        self.stats = WorkerStats(fps=4.0, camera_open=True)

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 3.0) -> None:
        self.stopped = True

    def set_interval(self, seconds: float) -> None:
        self.intervals.append(seconds)

    def set_active(self, active: bool) -> None:
        self.active.append(active)

    def set_max_faces(self, n: int) -> None:
        self.max_faces.append(n)

    def set_motion_gate(self, enabled: bool, threshold: float) -> None:
        self.gates.append((enabled, threshold))

    def set_preview(self, enabled: bool) -> None:
        self.previews.append(enabled)

    def reconfigure(self, source_factory: Any = None, backend_factory: Any = None) -> None:
        self.reconfigures.append((source_factory, backend_factory))

    @property
    def is_active(self) -> bool:
        return bool(self.active) and self.active[-1]

    def push(self, obs: Observation) -> None:
        self.on_observation(obs)


class FakePlatform(PlatformServices):
    """Records OS actions instead of performing them (never locks this machine)."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.locked: bool | None = False
        self.camera_in_use: bool | None = False
        self.processes: set[str] = set()
        self.idle: float | None = None
        self.key_idle: float | None = None
        self.foreground: WindowRef | None = None
        self.window_under: WindowRef | None = None
        self.lock_ok = True

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def lock_screen(self) -> bool:
        self.calls.append(("lock_screen",))
        return self.lock_ok

    def display_off(self) -> bool:
        self.calls.append(("display_off",))
        return True

    def wake_display(self) -> bool:
        self.calls.append(("wake_display",))
        return True

    def is_session_locked(self) -> bool | None:
        return self.locked

    def seconds_since_input(self) -> float | None:
        return self.idle

    def seconds_since_key_input(self) -> float | None:
        return self.key_idle

    def move_cursor(self, x: int, y: int) -> bool | None:
        raise AssertionError("tests move the cursor through FakeCursor only")

    def foreground_window(self) -> WindowRef | None:
        return self.foreground

    def window_at(self, x: int, y: int) -> WindowRef | None:
        self.calls.append(("window_at", x, y))
        return self.window_under

    def activate_window(self, ref: WindowRef) -> bool:
        self.calls.append(("activate_window", ref.handle))
        return True

    def is_window_valid(self, ref: WindowRef) -> bool:
        return True

    def window_rect(self, ref: WindowRef) -> Rect | None:
        return ref.rect

    def camera_in_use_by_other_app(self) -> bool | None:
        return self.camera_in_use

    def running_process_names(self) -> set[str]:
        return set(self.processes)


class FakeHotkeys:
    supported = True
    note = None

    def __init__(self) -> None:
        self.bindings: dict[str, tuple[str, Callable[[], None]]] = {}
        self.fail: set[str] = set()
        self.stopped = False

    def register(self, name: str, hotkey: Any, callback: Callable[[], None]) -> bool:
        if str(hotkey) in self.fail:
            return False
        self.bindings[name] = (str(hotkey), callback)
        return True

    def unregister_all(self) -> None:
        self.bindings.clear()

    def stop(self) -> None:
        self.stopped = True
        self.bindings.clear()


# --------------------------------------------------------------------------- helpers
def features_for(x: float, y: float) -> np.ndarray:
    return np.array([x / FEATURE_SCALE, y / FEATURE_SCALE])


def gaze_obs(point: tuple[float, float], faces: int = 1) -> Observation:
    return Observation(timestamp=0.0, face_count=faces, features=features_for(*point), quality=1.0)


def no_face() -> Observation:
    return Observation(timestamp=0.0, face_count=0)


def make_calibration(monitors: list[Monitor] = MONITORS) -> CalibrationData:
    samples: list[CalibrationSample] = []
    point = 0
    for m in monitors:
        for nx in (0.1, 0.5, 0.9):
            for ny in (0.1, 0.5, 0.9):
                x, y = m.rect.denormalize(nx, ny)
                samples.extend(
                    CalibrationSample(features_for(x, y), x, y, m.index, point) for _ in range(2)
                )
                point += 1
    X = np.vstack([s.features for s in samples])
    Y = np.array([(s.x, s.y) for s in samples])
    model = GazeModel(degree=1, alpha=0.0).fit(X, Y, bounds=virtual_bounds(monitors))
    return CalibrationData(
        backend=BACKEND[0],
        feature_version=BACKEND[1],
        layout_signature=layout_signature(monitors),
        monitors=list(monitors),
        samples=samples,
        implicit_samples=[],
        model=model,
        report={"grade": "excellent"},
    )


def make_settings() -> Settings:
    s = Settings()
    s.switching.smoothing = 0.0  # gaze == model output, exactly
    s.presence.away_timeout_s = 10
    s.presence.warning_s = 5
    return s


@dataclass
class Harness:
    controller: Controller
    platform: FakePlatform
    cursor: FakeCursor
    clock: FakeClock
    hotkeys: FakeHotkeys
    monitors: list[Monitor]
    workers: list[FakeWorker]
    events: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def worker(self) -> FakeWorker:
        return self.workers[-1]

    @property
    def state(self) -> TrackingState:
        return self.controller.state

    def push(self, obs: Observation, dt: float = 0.1) -> None:
        self.clock.advance(dt)
        self.worker.push(obs)

    def tick(self, dt: float = 0.1) -> None:
        self.clock.advance(dt)
        self.controller.tick()

    def feed(self, obs: Observation, seconds: float, step: float = 0.1) -> None:
        """Push ``obs`` every ``step`` seconds (with a housekeeping tick each time)."""
        for _ in range(round(seconds / step)):
            self.push(obs, step)
            self.controller.tick()


_SIGNALS = (
    "state_changed",
    "gaze_changed",
    "observation",
    "stats_changed",
    "switched",
    "away_warning",
    "guard_changed",
    "preview_frame",
    "calibration_required",
    "settings_changed",
    "ui_requested",
)


@pytest.fixture
def make_controller(qapp: Any, app_dirs: Any) -> Iterator[Callable[..., Harness]]:
    created: list[Controller] = []

    def factory(
        settings: Settings | None = None,
        *,
        calibrated: bool = True,
        platform: FakePlatform | None = None,
        hotkeys: FakeHotkeys | None = None,
        start: bool = True,
    ) -> Harness:
        if calibrated:
            save_calibration(paths.calibration_file(), make_calibration())
        platform = platform or FakePlatform()
        hotkeys = hotkeys or FakeHotkeys()
        cursor = FakeCursor()
        clock = FakeClock()
        monitors = list(MONITORS)
        workers: list[FakeWorker] = []

        def worker_factory(*args: Any) -> FakeWorker:
            worker = FakeWorker(*args)
            workers.append(worker)
            return worker

        controller = Controller(
            settings or make_settings(),
            platform,
            worker_factory=worker_factory,
            monitors_provider=lambda: list(monitors),
            cursor=cursor,
            clock=clock,
            hotkey_manager_factory=lambda: hotkeys,
        )
        created.append(controller)
        h = Harness(controller, platform, cursor, clock, hotkeys, monitors, workers)
        for name in _SIGNALS:
            bucket: list[Any] = []
            h.events[name] = bucket
            getattr(controller, name).connect(bucket.append)
        h.events["notify"] = []
        controller.notify.connect(lambda title, msg: h.events["notify"].append((title, msg)))
        h.events["away_cancelled"] = []
        controller.away_cancelled.connect(lambda: h.events["away_cancelled"].append(True))
        if start:
            controller.start()
        return h

    yield factory
    for controller in created:
        controller.shutdown()
        controller.deleteLater()


def settle_mouse(h: Harness) -> None:
    """Let every guard expire (mouse grace, typing grace, cooldown)."""
    h.tick(3.0)


# ----------------------------------------------------------------------------- start
def test_start_configures_worker_and_hotkeys(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    assert h.state is S.TRACKING
    assert h.worker.started
    assert h.worker.active[0] is False  # camera kept closed until the state is known
    assert h.worker.is_active
    assert h.worker.max_faces == [1]
    assert h.worker.gates[-1] == (True, 2.0)
    assert h.worker.intervals[-1] == pytest.approx(1 / 4)  # balanced "idle"
    assert h.controller.backend_info() == BACKEND
    assert h.controller.is_calibrated
    assert set(h.hotkeys.bindings) == {"toggle_tracking", "toggle_privacy", "recalibrate"}
    assert h.events["state_changed"] == [S.TRACKING]


def test_needs_calibration_without_calibration(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(calibrated=False)
    assert h.state is S.NEEDS_CALIBRATION
    assert h.worker.is_active  # presence and the guard still use the camera
    assert h.events["calibration_required"] == []
    assert h.controller.calibration_reason == "not calibrated yet"


def test_start_paused_never_opens_camera(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.general.start_paused = True
    h = make_controller(s)
    assert h.state is S.PAUSED
    assert True not in h.worker.active


def test_locked_session_at_start_keeps_camera_closed(
    make_controller: Callable[..., Harness],
) -> None:
    platform = FakePlatform()
    platform.locked = True
    h = make_controller(platform=platform)
    assert h.state is S.LOCKED
    assert True not in h.worker.active


# ------------------------------------------------------------------------- switching
def test_switch_after_dwell_restores_cursor_and_window(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    h.platform.foreground = WindowRef(handle=7, pid=42, rect=Rect(2000, 100, 1200, 800))
    h.tick(0.6)  # first cursor poll; the focused window is remembered for monitor 1
    h.platform.foreground = None
    h.cursor.position = (2600, 400)  # the user works on the right monitor ...
    h.tick()
    h.cursor.position = (500, 500)  # ... and comes back to the left one
    h.tick()
    settle_mouse(h)

    for _ in range(3):  # 0.0, 0.1, 0.2 s of dwell
        h.push(gaze_obs(RIGHT_CENTRE))
    assert h.events["switched"] == []
    assert h.cursor.moves == []

    h.push(gaze_obs(RIGHT_CENTRE))  # 0.3 s: dwell complete
    assert h.events["switched"] == [1]
    assert h.cursor.moves == [(2600, 400)]  # the remembered position, not the centre
    assert ("activate_window", 7) in h.platform.calls
    assert "window_at" not in h.platform.names()
    assert h.events["gaze_changed"][-1].x == pytest.approx(RIGHT_CENTRE[0])

    # The warp is not user activity, and the cursor now matches the gaze: no more switches.
    h.tick()
    h.feed(gaze_obs(RIGHT_CENTRE), 1.0)
    assert h.events["switched"] == [1]


def test_switch_to_centre_focuses_window_under_cursor(
    make_controller: Callable[..., Harness],
) -> None:
    s = make_settings()
    s.switching.cursor_target = "center"
    h = make_controller(s)
    h.platform.window_under = WindowRef(handle=9, pid=1, rect=Rect(1920, 0, 1920, 1080))
    h.feed(gaze_obs((2500, 300)), 0.5)
    assert h.events["switched"] == [1]
    assert h.cursor.moves == [RIGHT_CENTRE]
    assert ("window_at", *RIGHT_CENTRE) in h.platform.calls
    assert ("activate_window", 9) in h.platform.calls


def test_switch_can_be_disabled(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.switching.enabled = False
    h = make_controller(s)
    h.feed(gaze_obs(RIGHT_CENTRE), 1.0)
    assert h.events["switched"] == []
    assert h.events["gaze_changed"]  # the gaze is still estimated (overlay, preview)


def test_typing_guard_suppresses_switch_until_it_expires(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    h.platform.key_idle = 0.1  # the user keeps typing
    h.feed(gaze_obs(RIGHT_CENTRE), 1.5)
    assert h.events["switched"] == []

    h.platform.key_idle = 1000.0  # typing stopped; the guard runs out after 2 s
    h.feed(gaze_obs(RIGHT_CENTRE), 1.0)
    assert h.events["switched"] == []
    h.feed(gaze_obs(RIGHT_CENTRE), 1.2)
    assert h.events["switched"] == [1]


def test_mouse_guard_suppresses_switch(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.tick()
    for i in range(10):  # the user moves the mouse on the left monitor
        h.cursor.position = (500 + 20 * i, 500)
        h.push(gaze_obs(RIGHT_CENTRE))
        h.controller.tick()
    assert h.events["switched"] == []
    h.feed(gaze_obs(RIGHT_CENTRE), 1.6)  # mouse grace (1.5 s) over
    assert h.events["switched"] == [1]


def test_refused_cursor_warp_notifies_once(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.cursor.allow = False
    h.feed(gaze_obs(RIGHT_CENTRE), 0.5)
    h.feed(gaze_obs(RIGHT_CENTRE), 2.0)
    assert len(h.cursor.moves) >= 2
    titles = [title for title, _ in h.events["notify"]]
    assert titles.count("Cannot move the cursor") == 1


def test_wrong_switch_is_recorded_as_drift(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.switching.cursor_target = "center"
    h = make_controller(s)
    h.tick()
    h.feed(gaze_obs(RIGHT_CENTRE), 0.5)
    assert h.events["switched"] == [1]
    h.tick()
    h.cursor.position = (400, 400)  # dragged straight back to the left monitor
    h.tick()
    assert h.controller._drift.event_count == 1


# ------------------------------------------------------------------------- presence
def test_presence_warning_then_lock_then_return(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()  # timeout 10 s, 5 s countdown, action "lock"
    h.feed(no_face(), 4.5, step=0.5)
    assert h.events["away_warning"] == []
    h.feed(no_face(), 0.5, step=0.5)
    assert h.events["away_warning"] == [pytest.approx(5.0)]
    assert h.worker.intervals[-1] == pytest.approx(1 / 12)  # sample fast to cancel quickly
    h.feed(no_face(), 5.0, step=0.5)
    assert h.platform.names().count("lock_screen") == 1
    assert h.state is S.AWAY
    assert h.events["away_cancelled"] == [True]  # the countdown toast goes away
    assert h.worker.intervals[-1] == pytest.approx(1.0)  # "away" rate

    h.push(gaze_obs((500, 500)))
    assert h.state is S.TRACKING
    assert "wake_display" not in h.platform.names()


def test_display_off_and_wake_on_return(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.presence.action = "display_off"
    h = make_controller(s)
    h.feed(no_face(), 10.0, step=0.5)
    assert h.platform.names() == ["display_off"]
    assert h.state is S.AWAY
    h.push(gaze_obs((500, 500)))
    assert h.platform.names() == ["display_off", "wake_display"]
    assert h.state is S.TRACKING


def test_input_cancels_countdown(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.tick()
    h.feed(no_face(), 5.0, step=0.5)
    assert len(h.events["away_warning"]) == 1
    h.cursor.position = (800, 600)  # the user touches the mouse
    h.tick()
    assert h.events["away_cancelled"] == [True]
    h.feed(no_face(), 5.0, step=0.5)
    assert "lock_screen" not in h.platform.names()


def test_privacy_mode_releases_camera_and_freezes_presence(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    h.controller.set_privacy(True)
    assert h.state is S.PRIVACY
    assert not h.worker.is_active
    for _ in range(120):  # a minute without a camera is never absence
        h.push(no_face(), 0.5)  # late frames are ignored
        h.controller.tick()
    assert h.events["away_warning"] == []
    assert h.platform.calls == []

    h.controller.set_privacy(False)
    assert h.state is S.TRACKING
    assert h.worker.is_active
    h.feed(no_face(), 4.0, step=0.5)
    assert h.events["away_warning"] == []  # timers restarted, not resumed


def test_session_lock_pauses_and_unlock_resumes(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.platform.locked = True
    h.tick(2.1)
    assert h.state is S.LOCKED
    assert not h.worker.is_active
    h.platform.locked = False
    h.tick(2.1)
    assert h.state is S.TRACKING
    assert h.worker.is_active


def test_session_lock_ignored_when_disabled(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.privacy.pause_when_locked = False
    h = make_controller(s)
    h.platform.locked = True
    h.tick(2.1)
    assert h.state is S.TRACKING


# --------------------------------------------------------------------- camera yield
def test_yield_camera_to_other_app(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.platform.camera_in_use = True
    h.tick(3.1)
    assert h.state is S.YIELDED
    assert h.controller.yield_reason == "Another app is using the camera"
    assert not h.worker.is_active
    assert h.events["notify"][-1][0] == "Tracking paused"
    h.platform.camera_in_use = False
    h.tick(3.1)
    assert h.state is S.TRACKING
    assert h.worker.is_active


def test_pause_for_listed_app(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.privacy.yield_camera = False
    s.privacy.pause_for_apps = ["Zoom.exe"]
    h = make_controller(s)
    h.platform.camera_in_use = True  # ignored: yield_camera is off
    h.tick(3.1)
    assert h.state is S.TRACKING
    h.platform.processes = {"zoom.exe", "explorer.exe"}
    h.tick(3.1)
    assert h.state is S.YIELDED
    assert h.controller.yield_reason == "Zoom.exe is running"


# ----------------------------------------------------------------------------- guard
def test_shoulder_guard_curtain(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.privacy.shoulder_guard = True
    s.privacy.guard_action = "curtain"
    h = make_controller(s)
    assert h.worker.max_faces[-1] == 2
    h.feed(gaze_obs((500, 500), faces=2), 1.8)
    assert h.events["guard_changed"] == []
    h.feed(gaze_obs((500, 500), faces=2), 0.4)
    assert h.events["guard_changed"] == [True]
    assert h.controller.guard_active
    h.feed(gaze_obs((500, 500), faces=1), 1.7)
    assert h.events["guard_changed"] == [True, False]
    assert "lock_screen" not in h.platform.names()


def test_shoulder_guard_lock_action(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.privacy.shoulder_guard = True
    s.privacy.guard_action = "lock"
    h = make_controller(s)
    h.feed(gaze_obs((500, 500), faces=2), 2.5)
    assert h.platform.names() == ["lock_screen"]
    assert h.events["guard_changed"] == []


def test_privacy_mode_takes_the_curtain_down(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.privacy.shoulder_guard = True
    h = make_controller(s)
    h.feed(gaze_obs((500, 500), faces=2), 2.5)
    assert h.events["guard_changed"] == [True]
    h.controller.set_privacy(True)
    assert h.events["guard_changed"] == [True, False]


# --------------------------------------------------------------- calibration/layout
def test_layout_change_invalidates_calibration(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.monitors.append(Monitor(2, "third", Rect(3840, 0, 1920, 1080)))
    h.controller.refresh_monitors()
    assert h.state is S.NEEDS_CALIBRATION
    assert h.events["calibration_required"] == ["the monitor layout changed"]
    assert len(h.controller.monitors()) == 3

    del h.monitors[2]  # plugged back as it was: the calibration is valid again
    h.controller.refresh_monitors()
    assert h.state is S.TRACKING
    assert h.events["calibration_required"] == ["the monitor layout changed"]


def test_begin_and_finish_calibration(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(calibrated=False)
    h.controller.begin_calibration()
    assert h.state is S.CALIBRATING
    assert h.worker.gates[-1] == (False, 2.0)  # every frame is a sample
    assert h.worker.intervals[-1] == pytest.approx(1 / 24)
    obs = no_face()
    for _ in range(80):  # 40 s without a face: presence is frozen while calibrating
        h.push(obs, 0.5)
        h.controller.tick()
    assert h.events["observation"][-1] is obs
    assert h.events["away_warning"] == []

    h.controller.finish_calibration(None)  # cancelled
    assert h.state is S.NEEDS_CALIBRATION
    assert h.worker.gates[-1] == (True, 2.0)

    h.controller.begin_calibration()
    h.controller.finish_calibration(make_calibration())
    assert h.state is S.TRACKING
    stored = load_calibration(paths.calibration_file())
    assert stored is not None
    assert stored.backend == BACKEND[0]
    assert h.controller.calibration() is not None


def test_begin_calibration_turns_privacy_off(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.controller.set_privacy(True)
    h.controller.begin_calibration()
    assert not h.controller.privacy
    assert h.state is S.CALIBRATING
    assert h.worker.is_active


def test_backend_change_revalidates_calibration(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    new = h.controller.settings.copy()
    new.general.backend = "opencv"
    h.controller.apply_settings(new)
    source, backend = h.worker.reconfigures[-1]
    assert source is None
    assert backend is not None
    # The worker still reports the old backend; the settings decide meanwhile.
    assert h.controller.backend_info()[0] == "opencv"
    assert h.state is S.NEEDS_CALIBRATION
    assert "fake" in h.events["calibration_required"][-1]


def test_implicit_learning_refits_and_saves(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.tick()
    for i in range(25):  # 25 "move the mouse, let it rest" moments
        point = (300 + 40 * i, 400)
        h.cursor.position = point
        h.tick()
        h.push(gaze_obs(point), dt=0.4)
    stored = load_calibration(paths.calibration_file())
    assert stored is not None
    assert len(stored.implicit_samples) == 25
    assert all(s.point_id < 0 for s in stored.implicit_samples)
    assert h.events["notify"] == []  # predictions agreed with the cursor: no drift alert


# ---------------------------------------------------------------------- settings/IPC
def test_apply_settings_persists_and_reconfigures(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    new = h.controller.settings.copy()
    new.switching.dwell_ms = 800
    new.privacy.shoulder_guard = True
    new.camera.device = "1"
    new.performance.motion_threshold = 4.0
    new.hotkeys.toggle_privacy = "ctrl+alt+x"
    h.controller.apply_settings(new)

    saved = Settings.load(paths.settings_file())
    assert saved.to_dict() == new.to_dict()
    assert h.events["settings_changed"][-1].switching.dwell_ms == 800
    assert h.controller.settings is not new  # a private copy
    assert h.worker.max_faces[-1] == 2
    assert h.worker.gates[-1] == (True, 4.0)
    source, backend = h.worker.reconfigures[-1]
    assert source is not None
    assert backend is None
    assert h.hotkeys.bindings["toggle_privacy"][0] == "ctrl+alt+x"

    h.feed(gaze_obs(RIGHT_CENTRE), 0.7)  # the longer dwell is in effect
    assert h.events["switched"] == []
    h.feed(gaze_obs(RIGHT_CENTRE), 0.2)
    assert h.events["switched"] == [1]


def test_handle_command_covers_every_ipc_command(
    make_controller: Callable[..., Harness],
) -> None:
    from eye_tracker.ipc import COMMANDS as IPC_COMMANDS

    assert set(COMMANDS) == set(IPC_COMMANDS)
    h = make_controller()
    c = h.controller

    assert c.handle_command("pause") == "ok"
    assert h.state is S.PAUSED
    assert c.handle_command("resume") == "ok"
    assert h.state is S.TRACKING
    assert c.handle_command(" TOGGLE\n") == "ok"
    assert h.state is S.PAUSED
    assert c.handle_command("toggle") == "ok"
    assert h.state is S.TRACKING
    assert c.handle_command("privacy-on") == "ok"
    assert h.state is S.PRIVACY
    assert c.handle_command("privacy-off") == "ok"
    assert h.state is S.TRACKING
    assert c.handle_command("privacy-toggle") == "ok"
    assert c.privacy
    assert c.handle_command("privacy-toggle") == "ok"
    assert not c.privacy
    assert c.handle_command("calibrate") == "ok"
    assert h.events["calibration_required"] == ["ipc"]

    status = json.loads(c.handle_command("status"))
    assert status["state"] == "tracking"
    assert status["calibrated"] is True
    assert status["backend"] == "fake"
    assert status["monitors"] == 2

    for cmd in UI_COMMANDS:
        assert c.handle_command(cmd) == "ok"
    assert h.events["ui_requested"] == list(UI_COMMANDS)

    assert c.handle_command("explode").startswith("error:")
    assert c.handle_command("").startswith("error:")


# --------------------------------------------------------------------- rate/stats/etc
def test_rate_follows_activity(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.switching.cursor_target = "center"
    h = make_controller(s)
    assert h.worker.intervals[-1] == pytest.approx(1 / 4)  # idle
    h.push(gaze_obs(RIGHT_CENTRE))  # a switch is being considered
    assert h.worker.intervals[-1] == pytest.approx(1 / 12)
    h.feed(gaze_obs(RIGHT_CENTRE), 1.5)
    assert h.events["switched"] == [1]
    assert h.worker.intervals[-1] == pytest.approx(1 / 4)  # settled on the right monitor
    h.platform.key_idle = 0.1
    h.tick()
    assert h.worker.intervals[-1] == pytest.approx(1 / 2)  # typing
    h.push(no_face())
    assert h.worker.intervals[-1] == pytest.approx(1 / 2)  # no face: "noface" rate
    h.controller.begin_calibration()
    assert h.worker.intervals[-1] == pytest.approx(1 / 24)


def test_profile_change_updates_rate(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    new = h.controller.settings.copy()
    new.performance.profile = "eco"
    h.controller.apply_settings(new)
    assert h.worker.intervals[-1] == pytest.approx(1 / 2)  # eco idle


def test_camera_error_from_worker_stats(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.worker.on_stats(WorkerStats(camera_open=False, last_error="Camera 0 could not be opened"))
    assert h.state is S.CAMERA_ERROR
    assert h.events["notify"][-1] == ("Camera unavailable", "Camera 0 could not be opened")
    h.worker.on_stats(WorkerStats(camera_open=True))
    assert h.state is S.TRACKING


def test_stats_changed(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.tick(2.1)
    assert h.events["stats_changed"]
    stats = h.events["stats_changed"][-1]
    for key in ("fps", "target_fps", "inference_ms", "skip_ratio", "cpu_percent", "state"):
        assert key in stats
    assert stats["backend"] == "fake"
    assert stats["state"] == "tracking"
    assert stats["target_fps"] == pytest.approx(4.0)
    assert h.controller.stats_snapshot() == stats


def test_preview_frames_are_forwarded(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.controller.set_preview(True)
    assert h.worker.previews == [True]
    assert h.worker.intervals[-1] == pytest.approx(1 / 24)
    frame = np.zeros((4, 4, 3), np.uint8)
    h.worker.on_preview(frame)
    assert h.events["preview_frame"] == [frame]
    h.controller.set_preview(False)
    h.worker.on_preview(frame)
    assert len(h.events["preview_frame"]) == 1


def test_hotkeys_are_marshalled_to_the_main_thread(
    qapp: Any, make_controller: Callable[..., Harness]
) -> None:
    h = make_controller()
    toggle = h.hotkeys.bindings["toggle_tracking"][1]
    thread = threading.Thread(target=toggle)
    thread.start()
    thread.join()
    assert h.state is S.TRACKING  # not handled on the hotkey thread
    qapp.processEvents()
    assert h.state is S.PAUSED

    h.hotkeys.bindings["toggle_privacy"][1]()  # on the main thread: handled directly
    assert h.state is S.PRIVACY
    h.hotkeys.bindings["recalibrate"][1]()
    assert h.events["calibration_required"] == ["hotkey"]


def test_observations_from_worker_thread(
    qapp: Any, make_controller: Callable[..., Harness]
) -> None:
    h = make_controller()
    obs = gaze_obs((500, 500))
    thread = threading.Thread(target=h.worker.push, args=(obs,))
    thread.start()
    thread.join()
    assert h.events["observation"] == []
    qapp.processEvents()
    assert h.events["observation"] == [obs]


def test_hotkey_registration_failure_notifies(make_controller: Callable[..., Harness]) -> None:
    hotkeys = FakeHotkeys()
    hotkeys.fail = {"ctrl+alt+c"}
    h = make_controller(hotkeys=hotkeys)
    assert "recalibrate" not in hotkeys.bindings
    assert h.events["notify"][-1][0] == "Hotkey unavailable"


def test_hotkeys_disabled(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.hotkeys.enabled = False
    h = make_controller(s)
    assert h.hotkeys.bindings == {}
    assert h.hotkeys.stopped


def test_shutdown_stops_everything(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.controller.shutdown()
    h.controller.shutdown()  # idempotent
    assert h.worker.stopped
    assert h.hotkeys.stopped
    h.push(gaze_obs(RIGHT_CENTRE))  # late callbacks are ignored
    assert h.events["observation"] == []


def test_preview_requested_before_start(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(start=False)
    h.controller.set_preview(True)
    h.controller.start()
    assert h.worker.previews == [True]
    assert h.worker.intervals[-1] == pytest.approx(1 / 24)


def test_blink_holds_the_gaze(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.push(gaze_obs(RIGHT_CENTRE))
    h.push(gaze_obs(RIGHT_CENTRE))
    blink = gaze_obs(RIGHT_CENTRE)
    blink.blink = True
    h.push(blink)  # a blink must not restart the dwell
    h.push(gaze_obs(RIGHT_CENTRE))
    assert h.events["switched"] == [1]


# ------------------------------------------------------------------ default seams
def test_qt_monitors_offscreen(qapp: Any) -> None:
    monitors = qt_monitors()
    assert monitors
    assert [m.index for m in monitors] == list(range(len(monitors)))
    assert all(m.rect.w > 0 and m.rect.h > 0 for m in monitors)


def test_qt_cursor_reads_position(qapp: Any) -> None:
    x, y = QtCursor(FakePlatform()).pos()  # reading only; never warps the real pointer
    assert isinstance(x, int)
    assert isinstance(y, int)


def test_default_factories_build_but_do_not_open(qapp: Any) -> None:
    from eye_tracker.vision.camera import Camera
    from eye_tracker.vision.worker import VisionWorker

    s = Settings()
    source = controller_module._source_factory(s)()  # created, not opened
    assert isinstance(source, Camera)
    assert source.fps == 15
    assert not source.is_open
    s.performance.profile = "responsive"
    assert controller_module._source_factory(s)().fps == 30

    worker = controller_module._make_vision_worker(
        lambda: source, lambda: None, lambda _o: None, lambda _s: None, lambda _f: None
    )
    assert isinstance(worker, VisionWorker)
    assert not worker.is_running


def test_platform_property_and_hotkey_suspension(qapp, app_dirs) -> None:
    """suspend_hotkeys releases the OS registrations and restores them afterwards."""
    from eye_tracker.config import Settings
    from eye_tracker.engine.controller import Controller
    from eye_tracker.platform.base import PlatformServices

    class _Manager:
        supported = True
        note = ""

        def __init__(self) -> None:
            self.registered: list[str] = []
            self.unregister_calls = 0

        def register(self, name, hotkey, callback) -> bool:
            self.registered.append(name)
            return True

        def unregister_all(self) -> None:
            self.unregister_calls += 1
            self.registered.clear()

        def start(self) -> None: ...

        def stop(self) -> None: ...

    class _Worker:
        stats = None
        backend_info = None

        def __init__(self, *args, **kwargs) -> None: ...

        def __getattr__(self, name):
            return lambda *a, **k: None

    manager = _Manager()
    platform = PlatformServices()
    controller = Controller(
        Settings(),
        platform,
        worker_factory=_Worker,
        monitors_provider=lambda: [],
        hotkey_manager_factory=lambda: manager,
    )
    try:
        controller.start()
        assert controller.platform is platform
        assert sorted(manager.registered) == ["recalibrate", "toggle_privacy", "toggle_tracking"]

        controller.suspend_hotkeys(True)
        assert manager.registered == []
        fired: list[str] = []
        controller.calibration_required.connect(fired.append)
        controller._handle_hotkey("recalibrate")  # a press racing the release is ignored
        assert fired == []

        controller.suspend_hotkeys(False)
        assert sorted(manager.registered) == ["recalibrate", "toggle_privacy", "toggle_tracking"]
        controller._handle_hotkey("recalibrate")
        assert fired == ["hotkey"]
    finally:
        controller.shutdown()
