"""Tests for eye_tracker.engine.controller.Controller.

Everything the controller touches is faked: the vision worker (observations are
pushed synchronously), the platform (lock/display/window calls are recorded,
never performed), the cursor, the clock, the monitor layout and the hotkey
manager. Qt runs offscreen; config and calibration files live in a temp dir.
"""

from __future__ import annotations

import dataclasses
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
from eye_tracker.gaze.learning import refit_model
from eye_tracker.gaze.model import GazeModel
from eye_tracker.gaze.store import (
    CalibrationData,
    CalibrationLibrary,
    load_calibration,
    save_calibration,
)
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
LEFT_CENTRE = (960, 540)
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
        # False: moves succeed but pos() keeps reporting the old position, like
        # an XWayland client whose pointer went over a native Wayland window.
        self.track = True

    def pos(self) -> tuple[int, int]:
        return self.position

    def set_pos(self, x: int, y: int) -> bool:
        self.moves.append((x, y))
        if self.allow and self.track:
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
        self.activate_ok = True
        self.cursor_reliable = True
        self.accessibility: str | None = None  # None: not applicable (not macOS)
        self.camera_permission: bool | None = None
        self.background: list[bool] = []

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
        return self.activate_ok

    def is_window_valid(self, ref: WindowRef) -> bool:
        return True

    def window_rect(self, ref: WindowRef) -> Rect | None:
        return ref.rect

    def camera_in_use_by_other_app(self) -> bool | None:
        return self.camera_in_use

    def running_process_names(self) -> set[str]:
        return set(self.processes)

    def cursor_position_reliable(self) -> bool:
        return self.cursor_reliable

    def permissions(self) -> dict[str, bool | None]:
        return {"camera": self.camera_permission, "accessibility": None}

    def accessibility_status(self) -> str | None:
        return self.accessibility

    def set_background_activity(self, active: bool) -> bool:
        self.background.append(active)
        return True


class FakeHotkeys:
    supported = True
    note = None

    def __init__(self) -> None:
        self.bindings: dict[str, tuple[str, Callable[[], None]]] = {}
        self.fail: set[str] = set()
        self.errors: dict[str, str] = {}
        self.stopped = False

    def register(self, name: str, hotkey: Any, callback: Callable[[], None]) -> bool:
        if str(hotkey) in self.fail:
            return False
        self.bindings[name] = (str(hotkey), callback)
        return True

    @property
    def registered(self) -> dict[str, str]:
        return {name: combo for name, (combo, _callback) in self.bindings.items()}

    def last_error(self, name: str) -> str | None:
        return self.errors.get(name)

    def unregister_all(self) -> None:
        self.bindings.clear()

    def stop(self) -> None:
        self.stopped = True
        self.bindings.clear()


# --------------------------------------------------------------------------- helpers
def run_now(job: Callable[[], None]) -> None:
    """Synchronous ``lock_runner``."""
    job()


def features_for(x: float, y: float) -> np.ndarray:
    return np.array([x / FEATURE_SCALE, y / FEATURE_SCALE])


def gaze_obs(point: tuple[float, float], faces: int = 1) -> Observation:
    return Observation(timestamp=0.0, face_count=faces, features=features_for(*point), quality=1.0)


def no_face() -> Observation:
    return Observation(timestamp=0.0, face_count=0)


def make_calibration(
    monitors: list[Monitor] = MONITORS, nonlinear: tuple[int, ...] | None = None
) -> CalibrationData:
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
    model = GazeModel(degree=1, alpha=0.0, nonlinear=nonlinear).fit(
        X, Y, bounds=virtual_bounds(monitors)
    )
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
    s.general.first_run_done = True  # walk-away actions only notify before that
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
        threaded_locks: bool = False,
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
            # Screen locks run on a thread in the app; synchronously here unless
            # a test is about that thread.
            lock_runner=None if threaded_locks else run_now,
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
    h.platform.key_idle = 0.1  # the user keeps typing on the left monitor
    h.feed(gaze_obs(LEFT_CENTRE), 1.5)
    h.platform.key_idle = 1000.0  # typing stopped as the user looks right
    h.feed(gaze_obs(RIGHT_CENTRE), 1.0)
    assert h.events["switched"] == []
    h.feed(gaze_obs(RIGHT_CENTRE), 1.2)  # the typing grace (2 s) is over
    assert h.events["switched"] == [1]


def test_reading_pause_does_not_move_focus_to_the_reading_monitor(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    h.platform.key_idle = 0.1  # typing on the left while reading the right monitor
    h.feed(gaze_obs(RIGHT_CENTRE), 1.5)
    h.platform.key_idle = 1000.0  # a reading pause
    h.feed(gaze_obs(RIGHT_CENTRE), 4.0)
    assert h.events["switched"] == []
    h.feed(gaze_obs(RIGHT_CENTRE), 2.5)  # reading grace (6 s) over
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
    new.general.backend = "lite"
    h.controller.apply_settings(new)
    source, backend = h.worker.reconfigures[-1]
    assert source is None
    assert backend is not None
    # The worker still reports the old backend; the settings decide meanwhile.
    assert h.controller.backend_info()[0] == "lite"
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


def test_privacy_mode_survives_a_restart(make_controller: Callable[..., Harness]) -> None:
    """r3-ux-docs-02: after a reboot or a silent upgrade the camera came back on."""
    first = make_controller()
    first.controller.set_privacy(True)
    first.controller.shutdown()
    second = make_controller()
    assert second.controller.privacy
    assert second.controller.privacy_restored
    assert second.state is S.PRIVACY
    assert True not in second.worker.active  # the camera never opened
    assert json.loads(second.controller.handle_command("status"))["privacy"] is True
    second.controller.set_privacy(False)
    assert not second.controller.privacy_restored
    second.controller.shutdown()
    third = make_controller()
    assert not third.controller.privacy
    assert third.state is S.TRACKING


def test_privacy_mode_is_forgotten_when_not_remembered(
    make_controller: Callable[..., Harness],
) -> None:
    first = make_controller()
    first.controller.set_privacy(True)
    first.controller.shutdown()
    forgetful = make_settings()
    forgetful.privacy.remember_privacy_mode = False
    second = make_controller(forgetful)
    assert not second.controller.privacy
    assert second.state is S.TRACKING
    second.controller.shutdown()
    # Remembering turned on later does not bring back the choice of a run that ignored it.
    third = make_controller()
    assert not third.controller.privacy


def test_a_damaged_privacy_state_file_counts_as_off(
    make_controller: Callable[..., Harness],
) -> None:
    paths.state_file().write_text("{not json", encoding="utf-8")
    h = make_controller()
    assert not h.controller.privacy
    paths.state_file().write_text('{"privacy_mode": "yes"}', encoding="utf-8")
    assert not make_controller().controller.privacy


def test_pause_and_privacy_hotkeys_say_what_they_did(
    make_controller: Callable[..., Harness],
) -> None:
    """r3-ux-docs-05: both are toggles, and pausing also stops walk-away detection."""
    h = make_controller()
    toggle = h.hotkeys.bindings["toggle_tracking"][1]
    privacy = h.hotkeys.bindings["toggle_privacy"][1]
    toggle()
    title, message = h.events["notify"][-1]
    assert title == "Tracking paused"
    assert "walk-away detection" in message
    toggle()
    assert h.events["notify"][-1] == ("Tracking resumed", "The camera is on again.")
    privacy()
    title, message = h.events["notify"][-1]
    assert title == "Privacy mode on"
    assert "The camera is off" in message
    toggle()  # paused as well while private
    toggle()
    assert h.events["notify"][-1] == (
        "Tracking resumed",
        "Privacy mode is still on: the camera stays off.",
    )
    privacy()
    assert h.events["notify"][-1] == ("Privacy mode off", "The camera is on again.")
    # The same over the command socket (ctl); a command that changes nothing is silent.
    count = len(h.events["notify"])
    assert h.controller.handle_command("resume") == "ok"
    assert len(h.events["notify"]) == count
    assert h.controller.handle_command("privacy-on") == "ok"
    assert h.events["notify"][-1][0] == "Privacy mode on"
    # Notifications turned off: none of these.
    quiet = h.controller.settings.copy()
    quiet.general.notifications = False
    h.controller.apply_settings(quiet)
    count = len(h.events["notify"])
    privacy()
    assert len(h.events["notify"]) == count


def test_mode_confirmation_names_the_registered_hotkey(
    make_controller: Callable[..., Harness],
) -> None:
    from eye_tracker.platform.hotkeys import format_hotkey, parse_hotkey

    class RealHotkeys(FakeHotkeys):
        @property
        def registered(self) -> dict[str, Any]:  # type: ignore[override]
            return {name: parse_hotkey(combo) for name, (combo, _cb) in self.bindings.items()}

    hotkeys = RealHotkeys()
    h = make_controller(hotkeys=hotkeys)
    assert h.controller.handle_command("pause") == "ok"
    label = format_hotkey(parse_hotkey(Settings().hotkeys.toggle_tracking))
    assert f"until you resume ({label})." in h.events["notify"][-1][1]


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
    hotkeys.fail = {Settings().hotkeys.recalibrate}  # this platform's default
    h = make_controller(hotkeys=hotkeys)
    assert "recalibrate" not in hotkeys.bindings
    title, message = h.events["notify"][-1]
    assert title == "Hotkey unavailable"
    assert "used by another app" in message


def test_hotkey_failure_notification_gives_the_managers_reason(
    make_controller: Callable[..., Harness],
) -> None:
    hotkeys = FakeHotkeys()
    s = make_settings()
    s.hotkeys.recalibrate = "ctrl+alt+t"
    hotkeys.fail = {"ctrl+alt+t"}
    hotkeys.errors["recalibrate"] = (
        "Ctrl+Alt+T is AltGr+T, which types '\u20ba' on the Turkish Q keyboard layout."
    )
    h = make_controller(s, hotkeys=hotkeys)
    assert h.events["notify"][-1] == (
        "Hotkey unavailable",
        "Ctrl+Alt+T is AltGr+T, which types '\u20ba' on the Turkish Q keyboard layout. "
        "Choose another in Settings \u2192 Hotkeys.",
    )


def test_hotkeys_disabled(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.hotkeys.enabled = False
    h = make_controller(s)
    assert h.hotkeys.bindings == {}
    assert h.hotkeys.stopped
    assert h.controller.status()["hotkeys"] == {"registered": [], "errors": {}}


IN_USE = "⌃⌥⌘C is already in use by another application"


def _hotkey_notices(h: Harness) -> list[str]:
    return [message for title, message in h.events["notify"] if title == "Hotkey unavailable"]


def test_status_says_which_hotkeys_the_system_took(
    make_controller: Callable[..., Harness],
) -> None:
    """Regression (r2-docs-12): `doctor` can only learn from the running app why a
    hotkey failed, e.g. because another application owns the combination."""
    hotkeys = FakeHotkeys()
    hotkeys.fail = {Settings().hotkeys.recalibrate}
    hotkeys.errors["recalibrate"] = IN_USE + "."
    h = make_controller(hotkeys=hotkeys)
    expected = {
        "registered": ["toggle_privacy", "toggle_tracking"],
        "errors": {"recalibrate": IN_USE},
    }
    assert h.controller.status()["hotkeys"] == expected
    assert json.loads(h.controller.handle_command("status"))["hotkeys"] == expected

    # Released while Settings → Hotkeys records a shortcut: not a failure.
    h.controller.suspend_hotkeys(True)
    suspended = h.controller.status()["hotkeys"]
    assert suspended["registered"] == []
    assert set(suspended["errors"]) == {"toggle_tracking", "toggle_privacy", "recalibrate"}
    assert all("Settings → Hotkeys" in reason for reason in suspended["errors"].values())
    h.controller.suspend_hotkeys(False)
    assert h.controller.status()["hotkeys"] == expected

    # A cleared hotkey is neither registered nor an error.
    s = h.controller.settings.copy()
    s.hotkeys.recalibrate = ""
    h.controller.apply_settings(s)
    assert h.controller.status()["hotkeys"] == {
        "registered": ["toggle_privacy", "toggle_tracking"],
        "errors": {},
    }


def test_doctor_reads_the_hotkeys_from_the_running_app(
    make_controller: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    from eye_tracker import diagnostics

    hotkeys = FakeHotkeys()
    hotkeys.fail = {Settings().hotkeys.recalibrate}
    hotkeys.errors["recalibrate"] = IN_USE
    h = make_controller(hotkeys=hotkeys)
    monkeypatch.setattr(
        diagnostics, "_instance_status", lambda: json.loads(h.controller.handle_command("status"))
    )
    names = ["toggle_tracking", "toggle_privacy", "recalibrate"]
    assert diagnostics._hotkey_registration(names, None, True) == {
        "toggle_tracking": "registered",
        "toggle_privacy": "registered",
        "recalibrate": f"not registered: {IN_USE}",
    }


def test_leaving_the_hotkey_fields_does_not_repeat_the_failure(
    make_controller: Callable[..., Harness],
) -> None:
    """Regression (r2-ui-app-05): the settings dialog releases the hotkeys while a
    shortcut field records and gives them back afterwards; an unchanged failure
    was announced again every time."""
    hotkeys = FakeHotkeys()
    default = Settings().hotkeys.recalibrate
    hotkeys.fail = {default}
    hotkeys.errors["recalibrate"] = IN_USE
    h = make_controller(hotkeys=hotkeys)
    assert _hotkey_notices(h) == [f"{IN_USE}. Choose another in Settings → Hotkeys."]

    for _ in range(3):
        h.controller.suspend_hotkeys(True)
        h.controller.suspend_hotkeys(False)
    assert len(_hotkey_notices(h)) == 1

    # Changing the hotkeys is a deliberate act: every failure is reported again.
    s = h.controller.settings.copy()
    s.hotkeys.toggle_tracking = "ctrl+alt+meta+y"
    hotkeys.fail.add("ctrl+alt+meta+y")
    hotkeys.errors["toggle_tracking"] = "⌃⌥⌘Y is already in use by another application"
    h.controller.apply_settings(s)
    assert len(_hotkey_notices(h)) == 2
    assert IN_USE in _hotkey_notices(h)[-1]
    assert "⌃⌥⌘Y" in _hotkey_notices(h)[-1]

    # Once a combination works, a later failure of it is news again.
    hotkeys.fail.discard(default)
    h.controller.suspend_hotkeys(True)
    h.controller.suspend_hotkeys(False)
    assert "recalibrate" in hotkeys.bindings
    assert len(_hotkey_notices(h)) == 2  # ⌃⌥⌘Y: unchanged, not repeated
    hotkeys.fail.add(default)
    h.controller.suspend_hotkeys(True)
    h.controller.suspend_hotkeys(False)
    assert len(_hotkey_notices(h)) == 3
    assert _hotkey_notices(h)[-1].startswith(IN_USE)

    # Switching the hotkeys off and on again announces the failures afresh.
    s = h.controller.settings.copy()
    s.hotkeys.enabled = False
    h.controller.apply_settings(s)
    s = s.copy()
    s.hotkeys.enabled = True
    h.controller.apply_settings(s)
    assert len(_hotkey_notices(h)) == 4


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


# ======================================================== review regressions
THIRD = Monitor(2, "third", Rect(3840, 0, 1920, 1080))
USER_BOX = (0.35, 0.25, 0.3, 0.4)
ONLOOKER_BOX = (0.8, 0.1, 0.1, 0.13)


def blind_obs() -> Observation:
    """A frame too dark or uniform to judge (lens covered, shutter closed)."""
    return Observation(timestamp=0.0, face_count=0, blind=True)


def with_size(obs: Observation, size: tuple[int, int]) -> Observation:
    return dataclasses.replace(obs, frame_size=size)


def boxed_obs(faces: int, box: tuple[float, float, float, float]) -> Observation:
    return dataclasses.replace(gaze_obs((500, 500), faces=faces), face_box=box)


def guard_settings(action: str) -> Settings:
    s = make_settings()
    s.privacy.shoulder_guard = True
    s.privacy.guard_action = action
    return s


def lock_session(h: Harness) -> None:
    h.platform.locked = True
    h.tick(2.1)
    assert h.state is S.LOCKED


def unlock_session(h: Harness) -> None:
    h.platform.locked = False
    h.tick(2.1)


def ticks(h: Harness, seconds: float, step: float = 0.5) -> None:
    for _ in range(round(seconds / step)):
        h.tick(step)


def titles(h: Harness) -> list[str]:
    return [title for title, _ in h.events["notify"]]


class FakeBackend:
    """What the worker builds through the controller's backend factory."""

    name = BACKEND[0]
    feature_version = BACKEND[1]
    feature_names = ("gaze_x", "gaze_y")
    gaze_features = ("gaze_x", "gaze_y")


@pytest.fixture
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Backend factories build a :class:`FakeBackend` whatever the setting says
    (like ``auto`` falling back); returns the backend names requested."""
    requested: list[str] = []

    def backend_factory(settings: Settings, max_faces: int) -> Callable[[], FakeBackend]:
        name = settings.general.backend

        def create() -> FakeBackend:
            requested.append(name)
            return FakeBackend()

        return create

    monkeypatch.setattr(controller_module, "_backend_factory", backend_factory)
    return requested


# ------------------------------------------ unreliable pointer / refused warps
def test_unreliable_pointer_position_follows_the_last_switch(
    make_controller: Callable[..., Harness],
) -> None:
    platform = FakePlatform()
    platform.cursor_reliable = False
    h = make_controller(platform=platform)
    h.cursor.track = False  # pos() keeps saying "left" (XWayland over a Wayland window)
    h.feed(gaze_obs(RIGHT_CENTRE), 0.5)
    assert h.events["switched"] == [1]
    h.feed(gaze_obs(RIGHT_CENTRE), 10.0)
    # No re-warp every cooldown although the reported position never changed.
    assert h.events["switched"] == [1]
    assert len(h.cursor.moves) == 1
    h.feed(gaze_obs(LEFT_CENTRE), 1.0)
    assert h.events["switched"] == [1, 0]


def test_unreliable_pointer_position_is_not_learned_from(
    make_controller: Callable[..., Harness],
) -> None:
    platform = FakePlatform()
    platform.cursor_reliable = False
    h = make_controller(platform=platform)
    h.cursor.track = False
    h.feed(gaze_obs(RIGHT_CENTRE), 0.5)
    assert h.events["switched"] == [1]
    h.cursor.position = (400, 400)  # a stale position shows up
    h.tick()
    h.push(gaze_obs((400, 400)), dt=0.4)
    h.tick(3.0)
    assert h.controller._learner.samples == []  # not a learning label
    assert h.controller._drift.event_count == 0  # not an undone (or kept) switch


def test_refused_pointer_moves_are_not_switches_and_back_off(
    make_controller: Callable[..., Harness],
) -> None:
    limit = controller_module.WARP_REFUSAL_LIMIT
    h = make_controller()
    h.platform.window_under = WindowRef(handle=9, pid=1, rect=Rect(1920, 0, 1920, 1080))
    h.cursor.allow = False
    h.feed(gaze_obs(RIGHT_CENTRE), 3.0)
    assert len(h.cursor.moves) == limit
    assert h.events["switched"] == []
    assert h.controller.status()["switches"] == 0
    assert "activate_window" not in h.platform.names()  # nothing moved, nothing focused
    assert titles(h) == ["Cannot move the cursor"]

    h.feed(gaze_obs(RIGHT_CENTRE), 55.0, step=0.5)
    assert len(h.cursor.moves) == limit  # switching is paused
    h.feed(gaze_obs(RIGHT_CENTRE), 5.0, step=0.5)  # the pause is over: one more try
    assert len(h.cursor.moves) == limit + 1
    assert titles(h) == ["Cannot move the cursor"]  # told once

    h.cursor.allow = True  # e.g. ydotoold was started
    h.feed(gaze_obs(RIGHT_CENTRE), 2 * controller_module.WARP_BACKOFF_S, step=0.5)
    assert h.events["switched"] == [1]
    assert ("activate_window", 9) in h.platform.calls


def test_pointer_moves_refused_by_a_new_lock_screen_are_not_reported(
    make_controller: Callable[..., Harness],
) -> None:
    """r3-logic-04: a lock screen or UAC prompt that takes the input right after
    a lock poll refused enough moves to suspend switching and warn the user."""
    h = make_controller()
    h.tick(2.0)  # the lock poll has just run
    h.platform.locked = True  # e.g. a UAC prompt raised by a background updater
    h.cursor.allow = False
    h.feed(gaze_obs(RIGHT_CENTRE), 1.9, step=1 / 12)
    assert len(h.cursor.moves) == 1  # the first refusal revealed the lock
    assert h.controller._warp_refusals == 0  # and did not count towards the backoff
    assert titles(h) == []
    assert h.state is S.LOCKED
    unlock_session(h)
    h.cursor.allow = True
    h.feed(gaze_obs(RIGHT_CENTRE), 1.0)
    assert h.events["switched"] == [1]


# --------------------------------------------------------------- shoulder guard
def test_guard_does_not_lock_again_right_after_the_user_unlocked(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller(guard_settings("lock"))
    two = gaze_obs((500, 500), faces=2)
    h.feed(two, 2.5)
    assert h.platform.names().count("lock_screen") == 1
    lock_session(h)
    unlock_session(h)
    assert h.state is S.TRACKING
    h.feed(two, 5.0)  # the colleague is still there
    assert h.platform.names().count("lock_screen") == 1
    assert h.events["guard_changed"] == [True]  # the curtain covers the screens instead
    h.feed(gaze_obs((500, 500)), 2.0)  # out of view for a moment
    assert h.events["guard_changed"] == [True, False]
    # A second face soon after may be the same colleague: covered, not locked.
    h.feed(two, 2.5)
    assert h.platform.names().count("lock_screen") == 1
    assert h.events["guard_changed"] == [True, False, True]
    # Once nobody has looked over the shoulder for the whole grace, it locks again.
    h.feed(gaze_obs((500, 500)), 2.0)
    h.feed(gaze_obs((500, 500)), controller_module.GUARD_RELOCK_GRACE_S, step=1.0)
    h.feed(two, 2.5)
    assert h.platform.names().count("lock_screen") == 2


def test_guard_grace_survives_a_colleague_who_drops_out_of_view(
    make_controller: Callable[..., Harness],
) -> None:
    """r3-logic-01: every 1.5 s dropout (the colleague turns to the user or looks
    down at notes) ended the relock grace, so the next look locked the user out."""
    h = make_controller(guard_settings("lock"))
    two = gaze_obs((500, 500), faces=2)
    h.feed(two, 2.5)
    lock_session(h)
    unlock_session(h)
    h.feed(two, 3.0)
    assert h.events["guard_changed"] == [True]
    h.controller.dismiss_curtain()  # Esc: the user carries on working
    for _ in range(20):  # a long visit, well beyond GUARD_RELOCK_GRACE_S in total
        h.feed(gaze_obs((500, 500)), 2.0)  # profile view: one face detected
        h.feed(two, 30.0, step=0.5)  # looking at the screen again
    assert h.platform.names().count("lock_screen") == 1
    assert h.state is S.TRACKING


def test_no_switching_under_the_privacy_curtain(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(guard_settings("curtain"))
    h.platform.window_under = WindowRef(handle=42, pid=1, rect=Rect(1920, 0, 1920, 1080))
    h.feed(gaze_obs((500, 500), faces=2), 2.5)
    assert h.events["guard_changed"] == [True]
    h.feed(gaze_obs(RIGHT_CENTRE, faces=2), 2.0)  # is the other screen covered too?
    assert h.events["switched"] == []
    assert h.cursor.moves == []
    assert "window_at" not in h.platform.names()
    assert "activate_window" not in h.platform.names()

    h.controller.dismiss_curtain()  # Esc: the user carries on
    assert h.events["guard_changed"] == [True, False]
    assert h.controller.guard_active
    h.feed(gaze_obs(RIGHT_CENTRE, faces=2), 1.0)
    assert h.events["switched"] == [1]


def test_guard_run_does_not_span_a_calibration(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(guard_settings("lock"))
    two = gaze_obs((500, 500), faces=2)
    h.feed(two, 1.8)
    h.controller.begin_calibration()
    h.feed(two, 1.0)
    h.controller.finish_calibration(None)
    h.push(two)
    assert "lock_screen" not in h.platform.names()
    h.feed(two, 2.5)  # a fresh, continuous run
    assert h.platform.names().count("lock_screen") == 1


def test_guard_run_does_not_span_a_camera_error(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(guard_settings("lock"))
    two = gaze_obs((500, 500), faces=2)
    h.feed(two, 1.8)
    h.worker.on_stats(WorkerStats(camera_open=False, last_error="Camera stopped"))
    assert h.state is S.CAMERA_ERROR
    h.tick(1.0)
    h.worker.on_stats(WorkerStats(camera_open=True))
    h.push(two)
    assert "lock_screen" not in h.platform.names()


def test_onlooker_left_alone_keeps_the_curtain_but_counts_as_a_face(
    make_controller: Callable[..., Harness],
) -> None:
    # Faces are not recognised, so walk-away detection cannot tell the onlooker
    # from the user: any face in view counts as someone at the computer (the
    # guard's judgement "not the user" must never lock a user out, r2-controller-01).
    h = make_controller(guard_settings("curtain"))
    h.feed(boxed_obs(2, USER_BOX), 2.5)
    assert h.events["guard_changed"] == [True]
    h.feed(boxed_obs(1, ONLOOKER_BOX), 30.0, step=0.5)  # the user left, the onlooker stayed
    assert h.events["guard_changed"] == [True]  # the curtain stays up in front of them
    assert h.controller.guard_active
    assert h.events["away_warning"] == []
    assert "lock_screen" not in h.platform.names()
    h.feed(no_face(), 3.0, step=0.5)  # missed for a moment: still covered
    assert h.events["guard_changed"] == [True]


# r2-controller-01: the user ~65 cm from the camera; a colleague leans in beside
# them to ~45 cm, so the colleague's face is the larger (primary) one.
SEATED_USER_BOX = (0.40, 0.30, 0.20, 0.27)
LEANING_COLLEAGUE_BOX = (0.05, 0.15, 0.30, 0.40)


def test_colleague_leaning_in_closer_than_the_user_never_locks_the_user(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller(guard_settings("curtain"))  # walk-away action "lock"
    h.feed(boxed_obs(1, SEATED_USER_BOX), 2.0)
    h.feed(boxed_obs(2, LEANING_COLLEAGUE_BOX), 2.5)
    assert h.events["guard_changed"] == [True]
    # The colleague leaves; the user reads without touching anything.
    h.feed(boxed_obs(1, SEATED_USER_BOX), 2.0)
    assert h.events["guard_changed"] == [True, False]  # recognised as the user
    assert not h.controller.guard_active
    h.feed(boxed_obs(1, SEATED_USER_BOX), 30.0, step=0.5)
    assert h.events["away_warning"] == []
    assert "lock_screen" not in h.platform.names()


def test_user_misjudged_as_the_onlooker_is_never_locked_and_input_lifts_the_curtain(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller(guard_settings("curtain"))
    # The guard starts with both faces in view and the colleague's is the larger
    # one: it takes the colleague for the user and the user for the onlooker.
    h.feed(boxed_obs(2, LEANING_COLLEAGUE_BOX), 2.5)
    assert h.events["guard_changed"] == [True]
    h.feed(boxed_obs(1, SEATED_USER_BOX), 30.0, step=0.5)
    assert h.events["guard_changed"] == [True]
    assert h.events["away_warning"] == []  # the visible face keeps the user present
    assert "lock_screen" not in h.platform.names()
    h.cursor.position = (800, 600)  # the user touches the mouse
    h.tick()
    h.feed(boxed_obs(1, SEATED_USER_BOX), 2.0)
    assert h.events["guard_changed"] == [True, False]
    assert not h.controller.guard_active


def test_new_onlooker_after_a_misjudged_departure_is_reported(
    make_controller: Callable[..., Harness],
) -> None:
    """r3-logic-03: the guard stayed silent for every later onlooker until input."""
    h = make_controller(guard_settings("notify"))
    h.feed(boxed_obs(1, SEATED_USER_BOX), 3.0)
    h.feed(boxed_obs(2, LEANING_COLLEAGUE_BOX), 3.0)
    assert titles(h) == ["Someone is looking at your screen"]
    leaned_back = (0.55, 0.36, 0.13, 0.18)  # the user made room and stays there
    h.feed(boxed_obs(1, leaned_back), 120.0, step=0.5)
    assert h.controller._guard.owner_missing
    h.feed(boxed_obs(2, leaned_back), 10.0, step=0.5)  # a stranger looks on
    assert titles(h) == ["Someone is looking at your screen"] * 2
    assert "lock_screen" not in h.platform.names()


def test_user_left_alone_clears_the_guard(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(guard_settings("curtain"))
    h.feed(boxed_obs(2, USER_BOX), 2.5)
    h.feed(boxed_obs(1, USER_BOX), 5.5, step=0.5)  # the onlooker left
    assert h.events["guard_changed"] == [True, False]
    assert h.events["away_warning"] == []


# ----------------------------------------------------------------- camera yield
@pytest.mark.parametrize("leave", ["resume", "privacy-off", "unlock"])
def test_camera_in_use_is_checked_before_the_camera_reopens(
    make_controller: Callable[..., Harness], leave: str
) -> None:
    h = make_controller()
    c = h.controller
    if leave == "resume":
        c.pause()
    elif leave == "privacy-off":
        c.set_privacy(True)
    else:
        lock_session(h)
    assert not h.worker.is_active
    h.platform.camera_in_use = True  # a video call starts meanwhile
    h.tick(5.0)
    opened = h.worker.active.count(True)
    if leave == "resume":
        c.resume()
    elif leave == "privacy-off":
        c.set_privacy(False)
    else:
        unlock_session(h)
    assert h.state is S.YIELDED
    assert h.worker.active.count(True) == opened  # the call's camera was never grabbed


# --------------------------------------------------------------- camera errors
def test_camera_failing_again_after_a_pause_is_noticed(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    failing = WorkerStats(camera_open=False, last_error="Camera 0 could not be opened")
    h.worker.stats = failing
    h.worker.on_stats(failing)
    assert h.state is S.CAMERA_ERROR
    h.controller.pause()
    h.controller.resume()
    assert h.state is S.TRACKING  # judged afresh
    # The worker fails with the same message again, so it publishes nothing new.
    h.tick(controller_module.CAMERA_ERROR_RECHECK_S - 1.0)
    assert h.state is S.TRACKING  # a slow camera gets time to open
    h.tick(1.5)
    assert h.state is S.CAMERA_ERROR
    assert titles(h).count("Camera unavailable") == 2


def test_camera_working_after_a_pause_is_no_error(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.worker.on_stats(WorkerStats(camera_open=False, last_error="Camera 0 could not be opened"))
    h.controller.pause()
    h.controller.resume()
    h.worker.stats = WorkerStats(camera_open=True)
    h.tick(controller_module.CAMERA_ERROR_RECHECK_S + 1.0)
    assert h.state is S.TRACKING


def test_failing_backend_is_a_camera_error(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.worker.on_stats(
        WorkerStats(
            camera_open=True,
            last_error="Vision backend keeps failing: boom",
            extra={"backend_failing": True},
        )
    )
    assert h.state is S.CAMERA_ERROR
    h.worker.on_stats(WorkerStats(camera_open=True, extra={"backend_failing": False}))
    assert h.state is S.TRACKING


def test_blocked_camera_permission_is_explained(make_controller: Callable[..., Harness]) -> None:
    platform = FakePlatform()
    platform.camera_permission = False
    h = make_controller(platform=platform)
    needed: list[str] = []
    h.controller.permission_needed.connect(needed.append)
    h.worker.on_stats(WorkerStats(camera_open=False, last_error="Camera 0 could not be opened"))
    assert h.state is S.CAMERA_ERROR
    assert titles(h) == ["Camera access blocked"]
    assert needed == ["camera"]


def test_input_ends_away_when_the_camera_failed_meanwhile(
    make_controller: Callable[..., Harness],
) -> None:
    s = make_settings()
    s.presence.action = "display_off"
    h = make_controller(s)
    h.feed(no_face(), 10.0, step=0.5)
    assert h.state is S.AWAY
    h.worker.on_stats(WorkerStats(camera_open=False, last_error="Camera unplugged"))
    ticks(h, 10.0)
    assert h.state is S.AWAY  # nobody there
    h.platform.idle = 0.2  # the user is back at the keyboard
    h.tick()
    assert h.state is S.CAMERA_ERROR
    assert h.platform.names() == ["display_off", "wake_display"]


# ------------------------------------------------------------------ blind camera
def test_covered_camera_pauses_walk_away_detection(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    h.feed(gaze_obs((500, 500)), 1.0)
    h.feed(blind_obs(), 60.0, step=0.5)  # shutter closed; the user keeps reading
    assert h.events["away_warning"] == []
    assert "lock_screen" not in h.platform.names()
    assert titles(h) == ["Camera appears covered"]


def test_darkness_after_the_user_left_still_locks(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.feed(no_face(), 4.0, step=0.5)  # the user left, then the lights went out
    h.feed(blind_obs(), 7.0, step=0.5)
    assert h.platform.names().count("lock_screen") == 1
    assert "Camera appears covered" not in titles(h)


def test_covered_camera_locks_after_a_long_time_without_input(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    h.feed(gaze_obs((500, 500)), 1.0)
    h.feed(blind_obs(), controller_module.BLIND_FREEZE_MAX_S - 1.0, step=0.5)
    assert h.events["away_warning"] == []
    h.feed(blind_obs(), 12.0, step=0.5)  # desk lamp off and gone: lock eventually
    assert h.platform.names().count("lock_screen") == 1


def test_covered_camera_keeps_the_curtain_up(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(guard_settings("curtain"))
    h.feed(gaze_obs((500, 500), faces=2), 2.5)
    assert h.events["guard_changed"] == [True]
    h.feed(blind_obs(), 5.0, step=0.5)  # covering the camera does not uncover the screens
    assert h.events["guard_changed"] == [True]
    h.feed(no_face(), 2.0)
    assert h.events["guard_changed"] == [True, False]


# ---------------------------------------------------------------- look-away
def test_glance_below_the_screens_is_off_screen(make_controller: Callable[..., Harness]) -> None:
    save_calibration(paths.calibration_file(), make_calibration(nonlinear=(0, 1)))
    h = make_controller(calibrated=False)
    assert h.state is S.TRACKING
    h.feed(gaze_obs(LEFT_CENTRE), 0.5)
    h.feed(gaze_obs((2500, 4000)), 2.0)  # the phone on the desk, below the right monitor
    assert h.events["switched"] == []
    assert h.controller._last_decision is not None
    assert h.controller._last_decision.reason == "off_screen"
    assert h.events["gaze_changed"][-1] is None
    assert h.controller._learner.samples == []


def test_glance_below_the_screens_without_gaze_features_is_not_detected(
    make_controller: Callable[..., Harness],
) -> None:
    # The control: a model that does not know its gaze features, with a backend
    # that declares none, clips the estimate to just below the right monitor.
    h = make_controller()
    h.feed(gaze_obs((2500, 4000)), 2.0)
    assert h.events["switched"] == [1]


def test_backend_gaze_features_detect_looking_away_with_an_old_model(
    make_controller: Callable[..., Harness], fake_backends: list[str]
) -> None:
    h = make_controller()  # its model predates ``nonlinear``
    h.worker.backend_factory()  # the worker builds the backend
    h.tick()
    assert h.controller.gaze_feature_indices() == (0, 1)
    h.feed(gaze_obs((2500, 4000)), 2.0)
    assert h.events["switched"] == []


def test_phone_below_the_monitors_does_not_switch_with_realistic_features(
    make_controller: Callable[..., Harness],
) -> None:
    from gaze_synth import GAZE, TWO, below_points, calibration_samples, synth_features

    rng = np.random.default_rng(7)
    samples = calibration_samples(TWO, rng, noise=0.5)
    X = np.vstack([s.features for s in samples])
    Y = np.array([(s.x, s.y) for s in samples])
    # Degree 3 is what calibration selects for such data; clipped and bent, its
    # estimates for a glance at the phone land just below the right monitor,
    # well within the decider's off-screen margin.
    model = GazeModel(degree=3, alpha=1.0, nonlinear=GAZE).fit(X, Y, bounds=virtual_bounds(TWO))
    cal = dataclasses.replace(make_calibration(TWO), samples=samples, model=model)
    save_calibration(paths.calibration_file(), cal)
    h = make_controller(calibrated=False)
    assert h.state is S.TRACKING
    for cm in (60.0, 80.0):
        phone = below_points([RIGHT], rng, 40, cm)  # the cursor is on the left monitor
        for features in synth_features(phone, rng, noise=0.5):
            h.push(Observation(timestamp=0.0, face_count=1, features=features, quality=1.0))
            h.controller.tick()
    assert h.events["switched"] == []


def test_gaze_estimated_just_below_a_monitor_still_switches(
    make_controller: Callable[..., Harness],
) -> None:
    """r2-gaze-engine-03: looking away is judged from the combined gaze direction
    on the calibrated monitors, not from each feature's calibrated range."""
    save_calibration(paths.calibration_file(), make_calibration(nonlinear=(0, 1)))
    h = make_controller(calibrated=False)
    assert h.state is S.TRACKING
    model = h.controller._model
    assert model is not None
    assert model.has_linear_estimate
    taskbar = (2880, 1200)  # the estimate lands a little below the right monitor
    assert model.looks_away(features_for(*taskbar))  # what the per-feature test said
    h.feed(gaze_obs(taskbar), 1.0)
    assert h.events["switched"] == [1]
    h.feed(gaze_obs((960, 2600)), 2.0)  # papers on the desk below the left monitor
    assert h.events["switched"] == [1]


# --------------------------------------------------------- backend identity
def test_new_backend_with_the_old_identity_is_accepted(
    make_controller: Callable[..., Harness], fake_backends: list[str]
) -> None:
    h = make_controller()
    old_factory = h.worker.backend_factory
    old_factory()
    h.tick()
    assert h.controller.backend_info() == BACKEND
    new = h.controller.settings.copy()
    new.general.backend = "lite"
    h.controller.apply_settings(new)
    assert h.controller.backend_info()[0] == "lite"  # the settings decide meanwhile
    assert h.state is S.NEEDS_CALIBRATION
    old_factory()  # the old backend recreated before the swap: not the new one
    h.tick()
    assert h.controller.backend_info()[0] == "lite"
    _source, new_factory = h.worker.reconfigures[-1]
    new_factory()  # "lite" was unavailable and the same backend came back
    h.tick()
    assert fake_backends == ["auto", "auto", "lite"]
    assert h.controller.backend_info() == BACKEND
    assert h.state is S.TRACKING


def test_frames_of_the_previous_backend_do_not_invalidate_the_calibration(
    make_controller: Callable[..., Harness],
) -> None:
    limit = controller_module.FEATURE_MISMATCH_LIMIT
    h = make_controller()
    stale = Observation(timestamp=0.0, face_count=1, features=np.full(3, 0.5), quality=1.0)
    for _ in range(limit - 1):
        h.push(stale)
    h.push(gaze_obs((500, 500)))  # the new backend's features fit
    for _ in range(limit - 1):
        h.push(stale)
    assert h.state is S.TRACKING
    assert h.events["calibration_required"] == []
    h.push(stale)  # consistently wrong: the calibration really does not fit
    assert h.state is S.NEEDS_CALIBRATION
    assert h.events["calibration_required"] == [
        "the camera features no longer match the calibration"
    ]


# --------------------------------------------------- calibration announcements
def test_layout_change_while_locked_is_announced_after_unlock(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    lock_session(h)
    h.monitors.append(THIRD)
    h.controller.refresh_monitors()  # docked while locked
    assert h.events["calibration_required"] == []
    unlock_session(h)
    assert h.state is S.NEEDS_CALIBRATION
    h.tick(1.0)
    assert h.events["calibration_required"] == []  # the layout gets time to settle
    ticks(h, 2.0)
    assert h.events["calibration_required"] == ["the monitor layout changed"]
    ticks(h, 5.0)
    assert h.events["calibration_required"] == ["the monitor layout changed"]


def test_layout_restored_while_locked_is_not_announced(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    lock_session(h)
    h.monitors.append(THIRD)
    h.controller.refresh_monitors()
    del h.monitors[2]
    h.controller.refresh_monitors()
    unlock_session(h)
    ticks(h, 5.0)
    assert h.state is S.TRACKING
    assert h.events["calibration_required"] == []


def test_layout_change_while_away_is_announced_on_return(
    make_controller: Callable[..., Harness],
) -> None:
    s = make_settings()
    s.presence.action = "display_off"
    h = make_controller(s)
    h.feed(no_face(), 10.0, step=0.5)
    assert h.state is S.AWAY
    h.monitors.append(THIRD)
    h.controller.refresh_monitors()
    assert h.events["calibration_required"] == []
    h.feed(gaze_obs((500, 500)), 3.0, step=0.5)
    assert h.state is S.NEEDS_CALIBRATION
    assert h.events["calibration_required"] == ["the monitor layout changed"]


def test_unusable_calibration_is_recalled_when_the_user_comes_back(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    h.monitors.append(THIRD)
    h.controller.refresh_monitors()
    assert h.events["calibration_required"] == ["the monitor layout changed"]
    lock_session(h)
    unlock_session(h)
    ticks(h, 3.0)
    assert len(h.events["calibration_required"]) == 1  # said a moment ago
    lock_session(h)
    h.tick(controller_module.CALIBRATION_REMINDER_S)  # the night passes
    unlock_session(h)
    ticks(h, 3.0)
    assert h.events["calibration_required"] == ["the monitor layout changed"] * 2


def test_never_calibrated_is_recalled_when_the_user_comes_back(
    make_controller: Callable[..., Harness],
) -> None:
    """r3-ux-docs-01: a setup that was never calibrated (and starts at login) was
    never told again why nothing switches."""
    h = make_controller(calibrated=False)
    lock_session(h)
    unlock_session(h)
    ticks(h, 3.0)
    assert h.events["calibration_required"] == []  # the app offered it at start
    lock_session(h)
    h.tick(controller_module.CALIBRATION_REMINDER_S)  # the night passes
    unlock_session(h)
    ticks(h, 3.0)
    assert h.events["calibration_required"] == ["not calibrated yet"]


def test_no_calibration_is_asked_for_where_it_changes_nothing(
    make_controller: Callable[..., Harness],
) -> None:
    """r3-ux-docs-03: one monitor, or switching turned off, needs no calibration:
    no warning state, no prompts."""
    s = make_settings()
    s.switching.enabled = False
    h = make_controller(s, calibrated=False)
    assert h.state is S.TRACKING
    lock_session(h)
    h.tick(controller_module.CALIBRATION_REMINDER_S)
    unlock_session(h)
    ticks(h, 3.0)
    assert h.events["calibration_required"] == []
    # Turning switching on: now a calibration is what is missing, and the user
    # back from the night is told so.
    on = h.controller.settings.copy()
    on.switching.enabled = True
    h.controller.apply_settings(on)
    assert h.state is S.NEEDS_CALIBRATION
    ticks(h, 3.0)
    assert h.events["calibration_required"] == ["not calibrated yet"]
    # One monitor left (a laptop undocked): nothing to calibrate for.
    del h.monitors[1]
    h.controller.refresh_monitors()
    assert h.state is S.TRACKING
    assert h.controller.status()["calibrated"] is False


# ------------------------------------------------------- calibration profiles
def test_calibration_profiles_follow_the_desk(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    office = h.controller.calibration()
    h.monitors.append(THIRD)
    h.controller.refresh_monitors()
    assert h.state is S.NEEDS_CALIBRATION
    h.controller.begin_calibration()
    h.controller.finish_calibration(make_calibration(h.monitors))
    home = h.controller.calibration()
    assert h.state is S.TRACKING
    assert len(h.controller.calibrations()) == 2

    del h.monitors[2]  # back at the office
    h.controller.refresh_monitors()
    assert h.state is S.TRACKING
    assert h.controller.calibration() is office
    h.monitors.append(THIRD)  # and home again
    h.controller.refresh_monitors()
    assert h.state is S.TRACKING
    assert h.controller.calibration() is home
    assert h.events["calibration_required"] == ["the monitor layout changed"]  # once, ever

    stored = CalibrationLibrary.load(paths.calibration_file())
    assert len(stored) == 2
    assert stored.latest is not None
    assert stored.latest.layout_signature == home.layout_signature  # most recently used
    assert h.controller.status()["calibration"]["profiles"] == 2


def test_learned_samples_stay_with_their_profile(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.tick()
    for i in range(3):  # three "move the mouse, let it rest" moments at the office
        point = (300 + 200 * i, 400)
        h.cursor.position = point
        h.tick()
        h.push(gaze_obs(point), dt=0.4)
    assert len(h.controller._learner.samples) == 3

    h.monitors.append(THIRD)
    h.controller.refresh_monitors()
    h.controller.begin_calibration()
    h.controller.finish_calibration(make_calibration(h.monitors))
    assert h.controller._learner.samples == []  # a new desk learns afresh

    del h.monitors[2]
    h.controller.refresh_monitors()
    assert len(h.controller._learner.samples) == 3
    office = [p for p in CalibrationLibrary.load(paths.calibration_file()) if len(p.monitors) == 2]
    assert len(office[0].implicit_samples) == 3  # saved with the office profile


def test_start_uses_the_profile_for_the_current_layout(
    make_controller: Callable[..., Harness],
) -> None:
    save_calibration(paths.calibration_file(), make_calibration())
    save_calibration(paths.calibration_file(), make_calibration([*MONITORS, THIRD]))
    h = make_controller(calibrated=False)  # started at the two-monitor desk
    assert h.state is S.TRACKING
    cal = h.controller.calibration()
    assert cal is not None
    assert cal.layout_signature == layout_signature(MONITORS)
    stored = CalibrationLibrary.load(paths.calibration_file()).latest
    assert stored is not None
    assert stored.layout_signature == layout_signature(MONITORS)


# ------------------------------------------------------------------- camera
def test_calibration_records_the_camera(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(calibrated=False)
    h.controller.begin_calibration()
    for _ in range(3):
        h.push(with_size(gaze_obs((500, 500)), (640, 480)))
    assert h.controller.camera_identity() == ("0", (640, 480))
    h.controller.finish_calibration(make_calibration())
    stored = load_calibration(paths.calibration_file())
    assert stored is not None
    assert stored.camera == "0"
    assert stored.frame_size == (640, 480)
    assert h.state is S.TRACKING


def test_another_camera_needs_another_calibration(
    make_controller: Callable[..., Harness],
) -> None:
    save_calibration(paths.calibration_file(), dataclasses.replace(make_calibration(), camera="0"))
    h = make_controller(calibrated=False)
    assert h.state is S.TRACKING
    new = h.controller.settings.copy()
    new.camera.device = "1"
    h.controller.apply_settings(new)
    assert h.state is S.NEEDS_CALIBRATION
    assert h.controller.calibration_reason == "the camera changed"
    assert h.events["calibration_required"] == ["the camera changed"]
    back = h.controller.settings.copy()
    back.camera.device = "0"
    h.controller.apply_settings(back)
    assert h.state is S.TRACKING


def test_each_camera_has_its_own_profile(make_controller: Callable[..., Harness]) -> None:
    for device in ("0", "1"):
        save_calibration(
            paths.calibration_file(), dataclasses.replace(make_calibration(), camera=device)
        )
    h = make_controller(calibrated=False)
    cal = h.controller.calibration()
    assert h.state is S.TRACKING
    assert cal is not None
    assert cal.camera == "0"
    new = h.controller.settings.copy()
    new.camera.device = "1"
    h.controller.apply_settings(new)
    cal = h.controller.calibration()
    assert h.state is S.TRACKING
    assert cal is not None
    assert cal.camera == "1"


def test_camera_aspect_change_needs_another_calibration(
    make_controller: Callable[..., Harness],
) -> None:
    save_calibration(
        paths.calibration_file(),
        dataclasses.replace(make_calibration(), camera="0", frame_size=(640, 480)),
    )
    h = make_controller(calibrated=False)
    h.push(with_size(gaze_obs((500, 500)), (1280, 960)))  # same shape, more pixels
    assert h.state is S.TRACKING
    h.push(with_size(gaze_obs((500, 500)), (1280, 720)))  # 16:9: another field of view
    assert h.state is S.NEEDS_CALIBRATION
    assert h.controller.calibration_reason.startswith("the camera's aspect ratio changed")
    assert len(h.events["calibration_required"]) == 1
    h.push(with_size(gaze_obs((500, 500)), (640, 480)))
    assert h.state is S.TRACKING


# ------------------------------------------------------ calibration + privacy
@pytest.mark.parametrize("saved", [False, True])
def test_calibration_restores_privacy_mode(
    make_controller: Callable[..., Harness], saved: bool
) -> None:
    h = make_controller()
    h.controller.set_privacy(True)
    h.controller.begin_calibration()
    assert h.worker.is_active
    h.controller.finish_calibration(make_calibration() if saved else None)
    assert h.controller.privacy
    assert h.state is S.PRIVACY
    assert not h.worker.is_active


def test_privacy_changed_during_calibration_wins(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.controller.set_privacy(True)
    h.controller.begin_calibration()
    h.controller.set_privacy(False)  # e.g. `eye-tracker ctl privacy-off`
    h.controller.finish_calibration(None)
    assert not h.controller.privacy
    assert h.state is S.TRACKING


# ------------------------------------------------------------------ learning
def test_implausible_learning_labels_are_ignored(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    h.tick()
    h.cursor.position = (3700, 1000)  # the pointer parked on the right ...
    h.tick()
    h.push(gaze_obs((200, 100)), dt=0.4)  # ... while the user reads on the left
    assert h.controller._learner.samples == []
    assert h.controller._drift.event_count == 1  # still evidence about accuracy
    h.cursor.position = (3000, 600)
    h.tick()
    h.push(gaze_obs((3000, 600)), dt=0.4)
    assert len(h.controller._learner.samples) == 1


def test_kept_switch_counts_as_correct(make_controller: Callable[..., Harness]) -> None:
    s = make_settings()
    s.switching.cursor_target = "center"
    h = make_controller(s)
    h.feed(gaze_obs(RIGHT_CENTRE), 0.5)
    assert h.events["switched"] == [1]
    h.feed(gaze_obs(RIGHT_CENTRE), 2.5)
    assert h.controller._drift.event_count == 1
    assert h.controller._drift.error_rate == 0.0


def test_smaller_learning_capacity_refits_the_model(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    h.tick()
    for i in range(25):  # enough settles for one refit
        point = (300 + 40 * i, 400)
        h.cursor.position = point
        h.tick()
        h.push(gaze_obs(point), dt=0.4)
    cal = h.controller.calibration()
    assert cal is not None
    refined = cal.model
    assert len(cal.implicit_samples) == 25
    new = h.controller.settings.copy()
    new.learning.max_samples = 0
    h.controller.apply_settings(new)
    assert cal.implicit_samples == []
    assert cal.model is not refined
    assert h.controller._model is cal.model
    stored = load_calibration(paths.calibration_file())
    assert stored is not None
    assert stored.implicit_samples == []


# -------------------------------------------------------------------- preview
def test_preview_consumers_are_counted(make_controller: Callable[..., Harness]) -> None:
    h = make_controller()
    window, wizard = object(), object()
    h.controller.set_preview(True, window)
    h.controller.set_preview(True, wizard)
    h.controller.set_preview(False, window)  # the preview window is hidden
    assert h.worker.previews == [True]
    assert h.controller.preview_enabled
    frame = np.zeros((4, 4, 3), np.uint8)
    h.worker.on_preview(frame)
    assert h.events["preview_frame"] == [frame]  # the wizard still gets frames
    h.controller.set_preview(False, wizard)
    assert h.worker.previews == [True, False]
    assert not h.controller.preview_enabled


# ------------------------------------------------------------- macOS specifics
@pytest.mark.parametrize("status", ["stale", "missing"])
def test_missing_accessibility_is_reported_at_start(
    make_controller: Callable[..., Harness], status: str
) -> None:
    platform = FakePlatform()
    platform.accessibility = status
    h = make_controller(platform=platform, start=False)
    needed: list[str] = []
    h.controller.permission_needed.connect(needed.append)
    h.controller.start()
    assert titles(h) == ["Accessibility access needed"]
    assert needed == ["accessibility"]
    assert ("remove Eye Tracker" in h.events["notify"][0][1]) is (status == "stale")


def test_accessibility_is_not_needed_without_window_focus(
    make_controller: Callable[..., Harness],
) -> None:
    platform = FakePlatform()
    platform.accessibility = "missing"
    s = make_settings()
    s.switching.focus_window = False
    h = make_controller(s, platform=platform)
    assert h.events["notify"] == []


def test_failed_activations_report_missing_accessibility(
    make_controller: Callable[..., Harness],
) -> None:
    platform = FakePlatform()
    platform.accessibility = "granted"
    s = make_settings()
    s.switching.cursor_target = "center"
    h = make_controller(s, platform=platform)
    platform.window_under = WindowRef(handle=9, pid=1, rect=Rect(0, 0, 3840, 1080))
    platform.activate_ok = False
    platform.accessibility = "missing"  # revoked while running
    for point in (RIGHT_CENTRE, LEFT_CENTRE, RIGHT_CENTRE):
        h.feed(gaze_obs(point), 1.0)
    assert h.events["switched"] == [1, 0, 1]
    assert titles(h) == ["Accessibility access needed"]


def test_app_nap_is_allowed_only_while_paused_or_private(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller()
    assert h.platform.background == [True]
    h.controller.pause()
    assert h.platform.background == [True, False]
    h.controller.resume()
    assert h.platform.background == [True, False, True]
    lock_session(h)  # unlocking must still be noticed promptly
    unlock_session(h)
    h.controller.set_privacy(True)
    assert h.platform.background == [True, False, True, False]


# ------------------------------------------------------------- screen lock thread
class SlowLockPlatform(FakePlatform):
    """``lock_screen`` blocks until the test releases it (a hung screensaver)."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.lock_threads: list[int] = []

    def lock_screen(self) -> bool:
        self.lock_threads.append(threading.get_ident())
        self.entered.set()
        assert self.release.wait(5.0), "the test never released the lock call"
        return super().lock_screen()


def finish_lock(h: Harness, qapp: Any) -> None:
    """Let a slow lock call return and deliver its result to the main thread."""
    platform = h.platform
    assert isinstance(platform, SlowLockPlatform)
    platform.release.set()
    thread = h.controller._lock_thread
    assert thread is not None
    thread.join(5.0)
    assert not thread.is_alive()
    qapp.processEvents()  # the queued result


def test_walk_away_lock_does_not_block_the_gui_thread(
    make_controller: Callable[..., Harness], qapp: Any
) -> None:
    platform = SlowLockPlatform()
    h = make_controller(platform=platform, threaded_locks=True)
    h.feed(no_face(), 10.0, step=0.5)  # returns although lock_screen is still blocked
    assert platform.entered.wait(5.0)
    assert platform.lock_threads != [threading.get_ident()]
    assert h.state is S.AWAY
    h.feed(no_face(), 5.0, step=0.5)  # still away, still one lock request in flight
    assert len(platform.lock_threads) == 1
    finish_lock(h, qapp)
    assert h.platform.names() == ["lock_screen"]
    assert titles(h) == []
    # The lock is noticed soon, so the camera is released promptly.
    h.platform.locked = True
    h.tick(0.6)
    assert h.state is S.LOCKED


def test_failed_walk_away_lock_notifies_from_the_thread_result(
    make_controller: Callable[..., Harness], qapp: Any
) -> None:
    platform = SlowLockPlatform()
    platform.lock_ok = False
    h = make_controller(platform=platform, threaded_locks=True)
    h.feed(no_face(), 10.0, step=0.5)
    assert platform.entered.wait(5.0)
    assert titles(h) == []  # nothing known yet
    finish_lock(h, qapp)
    assert titles(h) == ["Could not lock the screen"]


def test_guard_lock_runs_on_a_thread_and_falls_back_to_the_curtain(
    make_controller: Callable[..., Harness], qapp: Any
) -> None:
    platform = SlowLockPlatform()
    platform.lock_ok = False
    h = make_controller(guard_settings("lock"), platform=platform, threaded_locks=True)
    h.feed(gaze_obs((500, 500), faces=2), 2.5)
    assert platform.entered.wait(5.0)
    assert h.events["guard_changed"] == []
    finish_lock(h, qapp)
    assert h.events["guard_changed"] == [True]  # the onlooker is still there


def test_failed_guard_lock_after_the_onlooker_left_shows_no_curtain(
    make_controller: Callable[..., Harness], qapp: Any
) -> None:
    platform = SlowLockPlatform()
    platform.lock_ok = False
    h = make_controller(guard_settings("lock"), platform=platform, threaded_locks=True)
    h.feed(gaze_obs((500, 500), faces=2), 2.5)
    assert platform.entered.wait(5.0)
    h.feed(gaze_obs((500, 500)), 2.0)  # gone before the lock call returned
    assert not h.controller.guard_active
    finish_lock(h, qapp)
    assert h.events["guard_changed"] == []


def test_successful_guard_lock_from_the_thread_arms_the_relock_grace(
    make_controller: Callable[..., Harness], qapp: Any
) -> None:
    platform = SlowLockPlatform()
    h = make_controller(guard_settings("lock"), platform=platform, threaded_locks=True)
    two = gaze_obs((500, 500), faces=2)
    h.feed(two, 2.5)
    assert platform.entered.wait(5.0)
    finish_lock(h, qapp)
    lock_session(h)
    unlock_session(h)
    h.feed(two, 5.0)  # the colleague is still there: the curtain, not another lock
    assert h.platform.names().count("lock_screen") == 1
    assert h.events["guard_changed"] == [True]


def test_shutdown_waits_briefly_for_a_lock_in_progress(
    make_controller: Callable[..., Harness],
) -> None:
    platform = SlowLockPlatform()
    h = make_controller(platform=platform, threaded_locks=True)
    h.feed(no_face(), 10.0, step=0.5)
    assert platform.entered.wait(5.0)
    thread = h.controller._lock_thread
    assert thread is not None
    threading.Timer(0.1, platform.release.set).start()
    h.controller.shutdown()
    assert not thread.is_alive()
    assert titles(h) == []  # no result is handled after shutdown


# ------------------------------------------------------------ first-run safety
@pytest.mark.parametrize("action", ["lock", "lock_and_display_off", "display_off"])
def test_walk_away_only_notifies_before_the_setup_is_finished(
    make_controller: Callable[..., Harness], action: str
) -> None:
    """journeys-05: never lock (or blank) a PC whose owner has not finished setup."""
    s = make_settings()
    s.general.first_run_done = False
    s.presence.action = action
    h = make_controller(s)
    h.feed(no_face(), 10.0, step=0.5)
    assert h.state is S.AWAY
    assert h.platform.names() == []  # neither locked nor displays off
    assert titles(h) == ["Are you still there?"]
    assert "Finish the setup" in h.events["notify"][0][1]

    finished = h.controller.settings.copy()
    finished.general.first_run_done = True
    h.controller.apply_settings(finished)
    h.push(gaze_obs((500, 500)))  # back ...
    h.feed(no_face(), 10.0, step=0.5)  # ... and away again, after the setup
    assert h.platform.names()[0] == ("display_off" if "display" in action else "lock_screen")


def test_no_action_stays_silent_before_the_setup_is_finished(
    make_controller: Callable[..., Harness],
) -> None:
    s = make_settings()
    s.general.first_run_done = False
    s.presence.action = "none"
    h = make_controller(s)
    h.feed(no_face(), 10.0, step=0.5)
    assert h.state is S.AWAY
    assert h.events["notify"] == []
    assert h.platform.names() == []


# ================================================== second review regressions
# ------------------------------------------------------------ r2-controller-02
def test_preview_frame_captured_before_privacy_mode_is_not_shown(
    make_controller: Callable[..., Harness], qapp: Any
) -> None:
    h = make_controller()
    h.controller.set_preview(True, "window")
    frame = np.zeros((4, 4, 3), np.uint8)

    def from_worker() -> None:
        thread = threading.Thread(target=h.worker.on_preview, args=(frame,))
        thread.start()
        thread.join()

    from_worker()  # analysed just before the privacy hotkey was handled
    h.controller.set_privacy(True)
    qapp.processEvents()
    assert h.events["preview_frame"] == []
    from_worker()  # finished by the worker after the camera was switched off
    qapp.processEvents()
    assert h.events["preview_frame"] == []

    h.controller.set_privacy(False)
    from_worker()
    qapp.processEvents()
    assert h.events["preview_frame"] == [frame]


# ------------------------------------------------------------ r2-controller-03
def pause_when_locked_off() -> Settings:
    s = make_settings()
    s.privacy.pause_when_locked = False
    return s


def lock_session_keep_tracking(h: Harness) -> None:
    h.platform.locked = True
    h.tick(2.1)
    assert h.state is S.TRACKING  # pause_when_locked is off


def test_no_switching_on_the_lock_screen(make_controller: Callable[..., Harness]) -> None:
    h = make_controller(pause_when_locked_off())
    lock_session_keep_tracking(h)
    h.cursor.allow = False  # the lock screen owns the input desktop
    for _ in range(4):  # the user at the desk looks at both screens
        h.feed(gaze_obs(RIGHT_CENTRE), 2.0)
        h.feed(gaze_obs(LEFT_CENTRE), 2.0)
    assert h.cursor.moves == []
    assert "activate_window" not in h.platform.names()
    assert titles(h) == []  # no "Cannot move the cursor" (with a Wayland hint) on Windows

    h.platform.locked = False
    h.cursor.allow = True
    h.tick(2.1)
    h.feed(gaze_obs(RIGHT_CENTRE), 1.0)
    assert h.events["switched"] == [1]


def test_refused_moves_around_a_lock_do_not_suspend_switching_after_unlock(
    make_controller: Callable[..., Harness],
) -> None:
    h = make_controller(pause_when_locked_off())
    # The session is locked but that is not noticed yet (polled every
    # LOCK_POLL_S): the system refuses the moves and switching is suspended.
    h.cursor.allow = False
    h.feed(gaze_obs(RIGHT_CENTRE), 3.0)
    assert len(h.cursor.moves) == controller_module.WARP_REFUSAL_LIMIT
    lock_session_keep_tracking(h)
    h.platform.locked = False
    h.cursor.allow = True
    h.tick(2.1)
    h.feed(gaze_obs(RIGHT_CENTRE), 1.0)
    assert h.events["switched"] == [1]  # at once, not WARP_BACKOFF_S later


# ------------------------------------------------------------ r2-controller-04
def learned_office_profile() -> CalibrationData:
    """The two-monitor profile after learning that the gaze lands 300 px further
    right than the calibration says (40 samples)."""
    cal = make_calibration()
    learned = []
    for i, x in enumerate(np.linspace(200.0, 3400.0, 40)):
        target = float(x) + 300.0
        monitor = 0 if target < 1920 else 1
        learned.append(CalibrationSample(features_for(x, 540), target, 540, monitor, -1 - i, 0.5))
    model = refit_model(cal.samples, learned, cal.model)
    return dataclasses.replace(cal, implicit_samples=learned, model=model)


def predicted_x(h: Harness, x: float) -> float:
    model = h.controller._model
    assert model is not None
    return float(model.predict(features_for(x, 540))[0])


def stored_office() -> CalibrationData:
    office = [p for p in CalibrationLibrary.load(paths.calibration_file()) if len(p.monitors) == 2]
    assert len(office) == 1
    return office[0]


def test_lower_learning_capacity_applies_to_a_profile_used_later(
    make_controller: Callable[..., Harness],
) -> None:
    save_calibration(paths.calibration_file(), learned_office_profile())
    save_calibration(paths.calibration_file(), make_calibration([*MONITORS, THIRD]))
    h = make_controller(calibrated=False, start=False)
    h.monitors.append(THIRD)  # at the home desk
    h.controller.start()
    assert h.state is S.TRACKING
    new = h.controller.settings.copy()
    new.learning.max_samples = 0  # "forget what was learned"
    h.controller.apply_settings(new)

    del h.monitors[2]  # docked at the office
    h.controller.refresh_monitors()
    assert h.state is S.TRACKING
    cal = h.controller.calibration()
    assert cal is not None
    assert len(cal.monitors) == 2
    assert h.controller._learner.samples == []
    assert cal.implicit_samples == []
    assert predicted_x(h, 960) == pytest.approx(960, abs=1)  # no learned shift left
    assert stored_office().implicit_samples == []


def test_lower_learning_capacity_in_the_settings_file_applies_at_start(
    make_controller: Callable[..., Harness],
) -> None:
    office = learned_office_profile()
    save_calibration(paths.calibration_file(), office)
    s = make_settings()
    s.learning.max_samples = 10
    h = make_controller(s, calibrated=False)
    assert h.state is S.TRACKING
    cal = h.controller.calibration()
    assert cal is not None
    kept = h.controller._learner.samples
    assert len(kept) == 10
    assert cal.implicit_samples == kept
    expected = refit_model(office.samples, kept, office.model)
    assert predicted_x(h, 960) == pytest.approx(float(expected.predict(features_for(960, 540))[0]))
    assert len(stored_office().implicit_samples) == 10


# ------------------------------------------------------------ r2-controller-05
def test_unreliable_pointer_follows_its_monitor_when_screens_are_renumbered(
    make_controller: Callable[..., Harness],
) -> None:
    b_alone = Monitor(0, "right", RIGHT.rect, primary=True)
    c_alone = Monitor(1, "third", THIRD.rect)
    save_calibration(paths.calibration_file(), make_calibration([*MONITORS, THIRD]))
    save_calibration(paths.calibration_file(), make_calibration([b_alone, c_alone]))
    platform = FakePlatform()
    platform.cursor_reliable = False
    h = make_controller(calibrated=False, platform=platform, start=False)
    h.monitors.append(THIRD)
    h.controller.start()
    h.cursor.track = False  # the reported position stays on the left monitor
    h.feed(gaze_obs(RIGHT_CENTRE), 0.5)
    assert h.events["switched"] == [1]  # the pointer is on B, monitor 1

    h.monitors[:] = [b_alone, c_alone]  # A unplugged: B is now 0, C is now 1
    h.controller.refresh_monitors()
    assert h.state is S.TRACKING
    assert h.controller._assumed_monitor == 0  # still B
    h.feed(gaze_obs((4800, 540)), 1.0)  # look at C
    assert h.events["switched"] == [1, 1]
    assert h.cursor.moves[-1][0] >= THIRD.rect.x


def test_unreliable_pointer_follows_its_monitor_by_name_when_the_desktop_moves(
    make_controller: Callable[..., Harness],
) -> None:
    # A unplugged and the desktop laid out anew from x = 0: B takes A's place and
    # C takes B's old rectangle, so only the screen names tell them apart.
    b_moved = Monitor(0, "right", LEFT.rect, primary=True)
    c_moved = Monitor(1, "third", RIGHT.rect)
    save_calibration(paths.calibration_file(), make_calibration([*MONITORS, THIRD]))
    save_calibration(paths.calibration_file(), make_calibration([b_moved, c_moved]))
    platform = FakePlatform()
    platform.cursor_reliable = False
    h = make_controller(calibrated=False, platform=platform, start=False)
    h.monitors.append(THIRD)
    h.controller.start()
    h.cursor.track = False
    h.feed(gaze_obs(RIGHT_CENTRE), 0.5)
    assert h.events["switched"] == [1]  # the pointer is on B

    h.monitors[:] = [b_moved, c_moved]
    h.controller.refresh_monitors()
    assert h.state is S.TRACKING
    assert h.controller._assumed_monitor == 0  # still B, now on the left
    h.feed(gaze_obs(RIGHT_CENTRE), 1.0)  # look at C, where B used to be
    assert h.events["switched"] == [1, 1]


def test_same_monitor_prefers_names_then_places() -> None:
    same = controller_module._same_monitor
    a = Monitor(0, "DP-1", LEFT.rect)
    b = Monitor(1, "DP-2", RIGHT.rect)
    c = Monitor(2, "HDMI-1", THIRD.rect)
    # Unchanged, and renumbered with the coordinates kept.
    assert same(b, [a, b, c], [a, b, c]) == b
    renumbered_b = Monitor(0, "DP-2", RIGHT.rect)
    assert same(b, [a, b, c], [renumbered_b, Monitor(1, "HDMI-1", THIRD.rect)]) == renumbered_b
    # Laid out anew: the name follows the monitor.
    moved = [Monitor(0, "DP-2", LEFT.rect), Monitor(1, "HDMI-1", RIGHT.rect)]
    assert same(b, [a, b, c], moved) == moved[0]
    # Names that cannot tell (duplicates, positional fallbacks): the same place.
    twins = [Monitor(0, "U2419H", LEFT.rect), Monitor(1, "U2419H", RIGHT.rect)]
    assert same(twins[1], twins, [twins[0]]) is None  # the right one was unplugged
    fallback = [Monitor(i, f"Screen {i + 1}", m.rect) for i, m in enumerate([LEFT, RIGHT, THIRD])]
    renumbered = [Monitor(0, "Screen 1", RIGHT.rect), Monitor(1, "Screen 2", THIRD.rect)]
    assert same(fallback[1], fallback, renumbered) == renumbered[0]
    # Gone: whatever covers its centre now, if anything.
    assert same(c, [a, b, c], [a, b]) is None
    wider = Monitor(0, "DP-9", Rect(0, 0, 5760, 1080))
    assert same(c, [a, b, c], [wider]) == wider


# ------------------------------------------------------------ r2-controller-06
def test_failed_lock_after_the_displays_went_off_wakes_them_on_return(
    make_controller: Callable[..., Harness],
) -> None:
    s = make_settings()
    s.presence.action = "lock_and_display_off"
    platform = FakePlatform()
    platform.lock_ok = False
    h = make_controller(s, platform=platform)
    h.feed(no_face(), 10.0, step=0.5)
    assert h.platform.names() == ["display_off", "lock_screen"]
    assert titles(h) == ["Could not lock the screen"]
    h.push(gaze_obs((500, 500)))  # back, face only (no input)
    assert h.platform.names() == ["display_off", "lock_screen", "wake_display"]
    assert h.state is S.TRACKING


def test_failed_lock_from_the_thread_after_the_displays_went_off_wakes_them(
    make_controller: Callable[..., Harness], qapp: Any
) -> None:
    s = make_settings()
    s.presence.action = "lock_and_display_off"
    platform = SlowLockPlatform()
    platform.lock_ok = False
    h = make_controller(s, platform=platform, threaded_locks=True)
    h.feed(no_face(), 10.0, step=0.5)
    assert platform.entered.wait(5.0)
    finish_lock(h, qapp)
    h.push(gaze_obs((500, 500)))
    assert h.platform.names() == ["display_off", "lock_screen", "wake_display"]


def test_successful_lock_and_display_off_does_not_wake_the_displays(
    make_controller: Callable[..., Harness],
) -> None:
    s = make_settings()
    s.presence.action = "lock_and_display_off"
    h = make_controller(s)
    h.feed(no_face(), 10.0, step=0.5)
    assert h.platform.names() == ["display_off", "lock_screen"]
    h.push(gaze_obs((500, 500)))  # back before the lock was noticed
    assert "wake_display" not in h.platform.names()
