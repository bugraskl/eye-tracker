"""The orchestrator: camera, gaze model, decision engine and OS actions wired together.

:class:`Controller` lives on the Qt main thread and owns every moving part:

* a vision worker (camera + face analysis on a background thread),
* the gaze model from the calibration, smoothed by a One Euro filter,
* the pure decision engine (:mod:`.decision`, :mod:`.presence`, :mod:`.guard`,
  :mod:`.scheduler`, :mod:`.input_state`, :mod:`.window_memory`,
  :mod:`.camera_yield`),
* the platform services that move the cursor, focus windows and lock the screen,
* global hotkeys.

Threading
---------
Worker and hotkey callbacks arrive on foreign threads. They are marshalled to
the main thread through private queued signals; a callback that already runs on
the main thread is handled directly. Everything else runs on the main thread,
so no locking is needed beyond the hand-off of preview frames.

State
-----
The public :class:`~eye_tracker.types.TrackingState` is *derived* from a small
set of independent flags (user pause, privacy mode, session locked, camera
yielded, user away, camera error, calibrating, calibration valid), in this
priority order::

    PRIVACY > LOCKED > CALIBRATING > PAUSED > YIELDED > AWAY > CAMERA_ERROR
            > NEEDS_CALIBRATION > TRACKING

Deriving instead of transitioning means overlapping causes (paused *and* the
session locked, say) resolve naturally when one of them ends. Whether the camera
is open follows directly from the state (``TrackingState.camera_active``).

Housekeeping
------------
A ``QTimer`` ticks every 100 ms while tracking (cursor polling resolution for the
mouse/typing guards) and every 500 ms otherwise. Each tick polls input, records
the focused window per monitor, checks for a locked session and for other apps
wanting the camera (on slower schedules), advances the walk-away timers, adapts
the camera frame rate and publishes statistics.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QGuiApplication

from .. import __version__, paths
from ..config import Settings
from ..gaze.filters import PointFilter
from ..gaze.learning import DriftMonitor, ImplicitLearner, refit_model
from ..gaze.store import CalibrationData, load_calibration, save_calibration
from ..platform.base import PlatformServices
from ..trace import TraceWriter
from ..types import (
    GazePoint,
    Monitor,
    Observation,
    Rect,
    TrackingState,
    WindowRef,
    WorkerStats,
    layout_signature,
    monitor_at,
    nearest_monitor,
)
from .camera_yield import YieldInputs, should_yield
from .decision import Decision, SwitchConfig, SwitchDecider
from .guard import GuardConfig, ShoulderGuard
from .input_state import InputTracker
from .presence import PresenceConfig, PresenceEvent, PresenceMonitor, PresenceState
from .scheduler import PROFILES, RateContext, RatePolicy
from .window_memory import WindowMemory, choose_cursor_target

if TYPE_CHECKING:
    from ..gaze.model import GazeModel
    from ..platform.hotkeys import HotkeyManager
    from ..vision.backends.base import VisionBackend
    from ..vision.camera import FrameSource

    SourceFactory = Callable[[], FrameSource]
    BackendFactory = Callable[[], VisionBackend]
    ObservationCallback = Callable[[Observation], None]
    StatsCallback = Callable[[WorkerStats], None]
    PreviewCallback = Callable[[np.ndarray], None]
    WorkerFactory = Callable[
        [SourceFactory, BackendFactory, ObservationCallback, StatsCallback, PreviewCallback],
        "WorkerLike",
    ]
    MonitorsProvider = Callable[[], list[Monitor]]
    HotkeyManagerFactory = Callable[[], HotkeyManager]

log = logging.getLogger(__name__)

#: Housekeeping tick while tracking: the resolution of mouse/typing detection.
TICK_ACTIVE_MS = 100
#: Housekeeping tick in every other state.
TICK_IDLE_MS = 500
#: How often the focused window is sampled (remembered per monitor).
WINDOW_POLL_S = 0.5
#: How often the session lock state is checked.
LOCK_POLL_S = 2.0
#: How often other apps' camera use and the pause-for-apps list are checked.
YIELD_POLL_S = 3.0
#: How often ``stats_changed`` is emitted.
STATS_PERIOD_S = 2.0
#: Screen hot-plug events come in bursts (and displays drop out during sleep);
#: the layout is re-read once they have settled.
SCREEN_DEBOUNCE_MS = 1000
#: The worker interval is only updated when it changes by more than this fraction.
RATE_TOLERANCE = 0.05
#: Moving the mouse back to the previous monitor this soon after an automatic
#: switch marks that switch as wrong (drift evidence).
WRONG_SWITCH_S = 2.0
#: During a blink the last gaze point is held for this long, so a blink does not
#: restart the dwell toward another monitor.
BLINK_HOLD_S = 0.5
#: A longer gap without a usable gaze restarts the smoothing filter, so the next
#: estimate does not glide in from a stale position.
FILTER_RESET_S = 1.0
#: A filtered gaze jump larger than this fraction of the smallest monitor side
#: counts as "gaze moving" (the scheduler samples faster for a moment).
GAZE_MOVE_FRACTION = 0.1
GAZE_MOVING_HOLD_S = 0.5
#: Without an observation for this long (or 2.5 frame intervals, if longer) the
#: camera verdict is unknown and the walk-away timers freeze.
STALE_OBSERVATION_S = 2.5
#: Keyboard activity this recent selects the scheduler's "typing" rate.
TYPING_RATE_WINDOW_S = 2.0
#: Minimum time between two "accuracy dropped" notifications.
DRIFT_ALERT_COOLDOWN_S = 1800.0

#: IPC commands handled by :meth:`Controller.handle_command`.
COMMANDS = (
    "show",
    "settings",
    "pause",
    "resume",
    "toggle",
    "privacy-on",
    "privacy-off",
    "privacy-toggle",
    "calibrate",
    "status",
    "quit",
)
#: Commands that only the UI can carry out; forwarded through ``ui_requested``.
UI_COMMANDS = ("show", "settings", "quit")

#: Hotkey setting names (``Settings.hotkeys.<name>``) in registration order.
HOTKEY_ACTIONS = ("toggle_tracking", "toggle_privacy", "recalibrate")

_CAMERA_OFF = frozenset(
    {TrackingState.PAUSED, TrackingState.PRIVACY, TrackingState.LOCKED, TrackingState.YIELDED}
)


# ---------------------------------------------------------------------------- seams
class CursorLike(Protocol):
    """Mouse pointer access (injectable for tests)."""

    def pos(self) -> tuple[int, int]:
        """Current pointer position in global coordinates."""
        ...

    def set_pos(self, x: int, y: int) -> bool | None:
        """Warp the pointer. ``False`` means the system refused."""
        ...


class WorkerLike(Protocol):
    """The part of :class:`~eye_tracker.vision.worker.VisionWorker` the controller uses."""

    def start(self) -> None: ...
    def stop(self, timeout: float = 3.0) -> None: ...
    def set_interval(self, seconds: float) -> None: ...
    def set_active(self, active: bool) -> None: ...
    def set_max_faces(self, n: int) -> None: ...
    def set_motion_gate(self, enabled: bool, threshold: float) -> None: ...
    def set_preview(self, enabled: bool) -> None: ...
    def reconfigure(
        self,
        source_factory: SourceFactory | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None: ...

    @property
    def stats(self) -> WorkerStats: ...

    @property
    def backend_info(self) -> tuple[str, str] | None: ...


class QtCursor:
    """Default :class:`CursorLike`: the platform layer moves the pointer, Qt is the fallback.

    ``PlatformServices.move_cursor`` returns ``None`` where Qt's ``QCursor.setPos``
    is the right tool (X11), ``True``/``False`` where it did the work itself.
    """

    def __init__(self, platform: PlatformServices) -> None:
        self._platform = platform

    def pos(self) -> tuple[int, int]:
        point = QCursor.pos()
        return (point.x(), point.y())

    def set_pos(self, x: int, y: int) -> bool | None:
        try:
            result = self._platform.move_cursor(int(x), int(y))
        except Exception:
            log.debug("Platform move_cursor failed; using Qt", exc_info=True)
            result = None
        if result is None:
            QCursor.setPos(int(x), int(y))
            return True
        return bool(result)


def qt_monitors() -> list[Monitor]:
    """The monitors Qt knows about, indexed in ``QGuiApplication.screens()`` order.

    Returns an empty list when no ``QGuiApplication`` exists.
    """
    app = QGuiApplication.instance()
    if not isinstance(app, QGuiApplication):
        return []
    primary = QGuiApplication.primaryScreen()
    monitors: list[Monitor] = []
    for screen in QGuiApplication.screens():
        geometry = screen.geometry()
        if geometry.width() <= 0 or geometry.height() <= 0:
            continue
        index = len(monitors)
        monitors.append(
            Monitor(
                index=index,
                name=screen.name() or f"Screen {index + 1}",
                rect=Rect(geometry.x(), geometry.y(), geometry.width(), geometry.height()),
                primary=primary is not None and screen == primary,
                scale=float(screen.devicePixelRatio()),
            )
        )
    return monitors


def _make_vision_worker(
    source_factory: SourceFactory,
    backend_factory: BackendFactory,
    on_observation: ObservationCallback,
    on_stats: StatsCallback,
    on_preview: PreviewCallback,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> WorkerLike:
    # Imported lazily: the vision stack pulls in OpenCV, which tests with a fake
    # worker (and `eye-tracker ctl`) never need.
    from ..vision.worker import VisionWorker

    return VisionWorker(
        source_factory, backend_factory, on_observation, on_stats, on_preview, clock
    )


def _camera_fps(settings: Settings) -> int:
    """Capture rate to request: 15 fps unless the profile analyses faster than that.

    A lower capture rate lets the camera expose longer in dim light (less noise,
    steadier landmarks), and 15 fps covers every mode of eco and balanced except
    calibration, which works fine with 15 samples per second.
    """
    rates = PROFILES.get(settings.performance.profile, PROFILES["balanced"])
    return 30 if rates["active"] > 15 else 15


def _source_factory(settings: Settings) -> SourceFactory:
    cam = settings.camera
    device, width, height, api = cam.device, cam.width, cam.height, cam.api
    fps = _camera_fps(settings)

    def factory() -> FrameSource:
        from ..vision.camera import open_source

        return open_source(device, width, height, api, fps=fps)

    return factory


def _backend_factory(settings: Settings, max_faces: int) -> BackendFactory:
    name = settings.general.backend

    def factory() -> VisionBackend:
        from ..vision.backends import create_backend

        return create_backend(name, max_faces)

    return factory


def _layout_key(monitors: list[Monitor]) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple((m.index, m.rect.x, m.rect.y, m.rect.w, m.rect.h) for m in monitors)


def _finite(value: float, digits: int = 2) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None


# ------------------------------------------------------------------------ controller
class Controller(QObject):
    """Main-thread orchestrator of tracking, switching, presence and privacy features.

    Args:
        settings: Initial settings (copied; read them back through :attr:`settings`).
        platform: OS integration (``eye_tracker.platform.get_platform()``).
        worker_factory: ``(source_factory, backend_factory, on_observation, on_stats,
            on_preview) -> worker``; defaults to a :class:`VisionWorker`.
        monitors_provider: ``() -> list[Monitor]``; defaults to :func:`qt_monitors`.
        cursor: Pointer access; defaults to :class:`QtCursor`.
        clock: Monotonic time source in seconds.
        hotkey_manager_factory: ``() -> HotkeyManager``; defaults to
            ``platform.hotkeys.create_hotkey_manager``.
        parent: Qt parent.

    Call :meth:`start` once signals are connected and :meth:`shutdown` before exit.
    """

    #: The tracking state changed (:class:`TrackingState`).
    state_changed = Signal(object)
    #: New smoothed gaze estimate (:class:`GazePoint`), or ``None`` when lost.
    gaze_changed = Signal(object)
    #: Every observation from the camera while it is on (calibration UI, preview, wizard).
    observation = Signal(object)
    #: Periodic statistics dict (see :meth:`stats_snapshot`).
    stats_changed = Signal(object)
    #: A notification for the tray: ``(title, message)``.
    notify = Signal(str, str)
    #: The cursor was moved to the monitor with this index.
    switched = Signal(int)
    #: Walk-away countdown started: seconds until the away action.
    away_warning = Signal(float)
    #: The countdown is over (the user came back, or it elapsed); hide the toast.
    away_cancelled = Signal()
    #: Shoulder-guard curtain on/off (only emitted when ``guard_action == "curtain"``).
    guard_changed = Signal(bool)
    #: Annotated camera frame (BGR ``np.ndarray``) while the preview is enabled.
    preview_frame = Signal(object)
    #: A (re)calibration is needed or was requested; the argument is the reason.
    calibration_required = Signal(str)
    #: Settings were applied (the new :class:`Settings`).
    settings_changed = Signal(object)
    #: An IPC command only the UI can perform: ``"show"``, ``"settings"`` or ``"quit"``.
    ui_requested = Signal(str)

    # Hand-off from foreign threads to the main thread (queued connections).
    _observation_received = Signal(object)
    _stats_received = Signal(object)
    _preview_ready = Signal()
    _hotkey_pressed = Signal(str)

    def __init__(
        self,
        settings: Settings,
        platform: PlatformServices,
        *,
        worker_factory: WorkerFactory | None = None,
        monitors_provider: MonitorsProvider | None = None,
        cursor: CursorLike | None = None,
        clock: Callable[[], float] = time.monotonic,
        hotkey_manager_factory: HotkeyManagerFactory | None = None,
        trace_path: str | Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._trace = TraceWriter(trace_path) if trace_path else None
        self._last_decision: Decision | None = None
        self._settings = settings.copy()
        self._platform = platform
        self._worker_factory: WorkerFactory = worker_factory or functools.partial(
            _make_vision_worker, clock=clock
        )
        self._monitors_provider: MonitorsProvider = monitors_provider or qt_monitors
        self._cursor: CursorLike = cursor if cursor is not None else QtCursor(platform)
        self._clock = clock
        self._hotkey_factory = hotkey_manager_factory
        self._main_thread = threading.get_ident()

        s = self._settings
        now = clock()
        self._monitors: list[Monitor] = []
        self._min_side = 1000.0
        self._state = TrackingState.STARTING
        self._started = False
        self._closed = False

        # Independent causes from which the state is derived (see _derive_state).
        self._paused = False
        self._privacy = False
        self._calibrating = False
        self._session_locked = False
        self._yield_reason = ""
        self._away = False
        self._camera_error = False
        self._preview = False

        # Decision engine.
        self._decider = SwitchDecider([], SwitchConfig.from_settings(s.switching))
        self._presence = PresenceMonitor(PresenceConfig.from_settings(s.presence), now)
        self._guard = ShoulderGuard(GuardConfig.from_settings(s.privacy))
        self._policy = RatePolicy(s.performance.profile)
        self._input = InputTracker(platform.seconds_since_input, platform.seconds_since_key_input)
        self._memory = WindowMemory()
        self._filter = PointFilter(s.switching.smoothing)
        self._learner = ImplicitLearner(max_samples=s.learning.max_samples)
        self._drift = DriftMonitor()

        # Calibration.
        self._calibration: CalibrationData | None = None
        self._model: GazeModel | None = None  # set only while the calibration is usable
        self._calibration_reason = "not calibrated yet"
        self._implicit_dirty = False

        # What we know about the vision backend (for calibration compatibility).
        self._reported_backend: tuple[str, str] | None = None
        self._stale_backend: tuple[str, str] | None = None
        self._expected_cache: tuple[str, tuple[str, str] | None] | None = None

        # Worker.
        self._worker: WorkerLike | None = None
        self._interval: float | None = None
        self._mode = ""
        self._worker_stats = WorkerStats()
        self._camera_error_message: str | None = None
        self._preview_lock = threading.Lock()
        self._preview_pending: np.ndarray | None = None

        # Tracking transients.
        self._last_obs_time: float | None = None
        self._last_face: bool | None = None
        self._last_gaze: tuple[float, float] | None = None
        self._last_gaze_time: float | None = None
        self._gaze_none_sent = True
        self._gaze_moving_until = -math.inf
        self._pending = False
        self._last_switch: tuple[float, int | None, int] | None = None
        self._switch_count = 0
        self._cursor_warning_shown = False

        # Presence / guard side effects.
        self._warning_shown = False
        self._displays_off_only = False
        self._curtain = False

        # Housekeeping schedule (set in start()).
        self._next_window_poll = math.inf
        self._next_lock_check = math.inf
        self._next_yield_check = math.inf
        self._next_stats = math.inf
        self._process: Any = None
        self._cpu_count = 1
        self._cpu_percent: float | None = None
        self._last_stats: dict[str, Any] = {}

        self._hotkeys: HotkeyManager | None = None
        self._hotkeys_suspended = False
        self._screens_connected = False

        self._timer = QTimer(self)
        self._timer.setInterval(TICK_IDLE_MS)
        self._timer.timeout.connect(self.tick)
        self._screen_timer = QTimer(self)
        self._screen_timer.setSingleShot(True)
        self._screen_timer.setInterval(SCREEN_DEBOUNCE_MS)
        self._screen_timer.timeout.connect(self.refresh_monitors)

        queued = Qt.ConnectionType.QueuedConnection
        self._observation_received.connect(self._handle_observation, queued)
        self._stats_received.connect(self._handle_worker_stats, queued)
        self._preview_ready.connect(self._deliver_preview, queued)
        self._hotkey_pressed.connect(self._handle_hotkey, queued)

    # ================================================================== lifecycle
    def start(self) -> None:
        """Load the calibration, start the camera worker, hotkeys and housekeeping."""
        if self._started or self._closed:
            return
        now = self._clock()
        s = self._settings
        self._set_monitors(self._read_monitors())
        self._presence.reset(now)
        self._paused = self._paused or bool(s.general.start_paused)
        self._load_calibration()

        worker = self._worker_factory(
            _source_factory(s),
            _backend_factory(s, self._max_faces()),
            self._on_worker_observation,
            self._on_worker_stats,
            self._on_worker_preview,
        )
        self._worker = worker
        # Keep the camera closed until the state says otherwise (start paused,
        # locked session, privacy mode set before start()).
        worker.set_active(False)
        worker.set_max_faces(self._max_faces())
        if self._preview:
            worker.set_preview(True)
        self._started = True
        self._apply_motion_gate()
        self._sync_backend_info()
        self._validate_calibration(announce=False)

        # Environment checks run before the camera may open.
        self._check_session_lock(now)
        self._check_camera_yield(now)
        self._next_window_poll = now + WINDOW_POLL_S
        self._next_lock_check = now + LOCK_POLL_S
        self._next_yield_check = now + YIELD_POLL_S
        self._next_stats = now + STATS_PERIOD_S
        self._measure_cpu()  # baseline for the first reading

        self._update_state()  # STARTING -> real state: sets active flag and interval
        worker.start()
        self._setup_hotkeys()
        self._connect_screens()
        self._timer.start(self._tick_interval())
        log.info(
            "Controller started: %s, %d monitor(s), calibration %s",
            self._state.value,
            len(self._monitors),
            "usable" if self._model is not None else f"unusable ({self._calibration_reason})",
        )

    def shutdown(self) -> None:
        """Stop the worker (releasing the camera), hotkeys and timers. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._screen_timer.stop()
        self._disconnect_screens()
        if self._hotkeys is not None:
            try:
                self._hotkeys.stop()
            except Exception:
                log.debug("Stopping hotkeys failed", exc_info=True)
        if self._worker is not None:
            try:
                self._worker.stop()
            except Exception:
                log.warning("Stopping the vision worker failed", exc_info=True)
        if self._implicit_dirty and self._calibration is not None:
            self._calibration.implicit_samples = self._learner.samples
            self._save_calibration(self._calibration, quiet=True)
        if self._trace is not None:
            self._trace.close()
        log.info("Controller stopped")

    # ================================================================ properties
    @property
    def state(self) -> TrackingState:
        """Current tracking state."""
        return self._state

    @property
    def settings(self) -> Settings:
        """Settings in effect. Treat as read-only; change them with :meth:`apply_settings`."""
        return self._settings

    @property
    def paused(self) -> bool:
        """The user paused tracking."""
        return self._paused

    @property
    def privacy(self) -> bool:
        """Privacy mode (camera fully off) is on."""
        return self._privacy

    @property
    def preview_enabled(self) -> bool:
        return self._preview

    @property
    def is_calibrated(self) -> bool:
        """A calibration usable with the current camera backend and monitors exists."""
        return self._model is not None

    @property
    def calibration_reason(self) -> str:
        """Why the calibration is unusable (``""`` when it is usable)."""
        return "" if self._model is not None else self._calibration_reason

    @property
    def guard_active(self) -> bool:
        """The shoulder guard currently sees a second face."""
        return self._guard.active

    @property
    def presence_state(self) -> PresenceState:
        return self._presence.state

    @property
    def yield_reason(self) -> str:
        """Why the camera was released for another app (``""`` if it was not)."""
        return self._yield_reason

    @property
    def hotkey_manager(self) -> HotkeyManager | None:
        """The global hotkey manager (``None`` before :meth:`start`)."""
        return self._hotkeys

    @property
    def platform(self) -> PlatformServices:
        """The OS integration layer this controller was built with."""
        return self._platform

    def suspend_hotkeys(self, suspended: bool) -> None:
        """Release (``True``) or restore (``False``) the global hotkeys.

        The settings dialog suspends them while it records a new shortcut: the OS
        delivers a registered combination to us instead of the focused window, so
        without this the user could not re-record a shortcut that is in use.
        """
        suspended = bool(suspended)
        if suspended == self._hotkeys_suspended:
            return
        self._hotkeys_suspended = suspended
        if self._hotkeys is None or self._closed:
            return
        if suspended:
            try:
                self._hotkeys.unregister_all()
            except Exception:
                log.debug("Releasing hotkeys failed", exc_info=True)
        else:
            self._register_hotkeys()

    def monitors(self) -> list[Monitor]:
        """The current monitor layout."""
        if not self._monitors and not self._started:
            return self._read_monitors()
        return list(self._monitors)

    def calibration(self) -> CalibrationData | None:
        """The loaded calibration (also when it is not usable with the current layout)."""
        return self._calibration

    def backend_info(self) -> tuple[str, str]:
        """``(name, feature_version)`` of the vision backend in use (or expected).

        ``("", "")`` when no backend is available.
        """
        info = self._current_backend()
        return info if info is not None else ("", "")

    def away_remaining(self) -> float:
        """Seconds until the walk-away action (meaningful during a countdown)."""
        return self._presence.remaining(self._clock())

    def stats_snapshot(self) -> dict[str, Any]:
        """The most recent ``stats_changed`` payload (empty before the first one)."""
        return dict(self._last_stats)

    # ============================================================ user commands
    def pause(self) -> None:
        """Pause tracking; the camera is released."""
        if not self._paused:
            self._paused = True
            log.info("Tracking paused")
            self._update_state()

    def resume(self) -> None:
        """Resume after :meth:`pause` (privacy mode, if on, stays on)."""
        if self._paused:
            self._paused = False
            log.info("Tracking resumed")
            self._update_state()

    def toggle_pause(self) -> None:
        if self._paused:
            self.resume()
        else:
            self.pause()

    def set_privacy(self, enabled: bool) -> None:
        """Privacy mode: the camera is fully released (its light goes off)."""
        enabled = bool(enabled)
        if enabled == self._privacy:
            return
        self._privacy = enabled
        log.info("Privacy mode %s", "on" if enabled else "off")
        self._update_state()

    def toggle_privacy(self) -> None:
        self.set_privacy(not self._privacy)

    def set_preview(self, enabled: bool) -> None:
        """Deliver annotated camera frames through ``preview_frame`` (and sample faster)."""
        enabled = bool(enabled)
        if enabled == self._preview:
            return
        self._preview = enabled
        if self._worker is not None:
            self._worker.set_preview(enabled)
        if not enabled:
            with self._preview_lock:
                self._preview_pending = None
        self._update_rate(self._clock())

    def begin_calibration(self) -> None:
        """Enter ``CALIBRATING``: the camera runs at the calibration rate and every
        observation is forwarded through ``observation``.

        An explicit calibration request needs the camera, so privacy mode is turned
        off; a user pause is kept and resumes after the calibration.
        """
        if self._calibrating:
            return
        if self._privacy:
            log.info("Privacy mode turned off to calibrate")
            self._privacy = False
        self._calibrating = True
        log.info("Calibration started")
        self._update_state()

    def finish_calibration(self, data: CalibrationData | None) -> None:
        """Leave ``CALIBRATING``. ``data`` is saved and used; ``None`` means cancelled."""
        if data is not None:
            self._save_calibration(data, quiet=False)
            self._calibration = data
            self._learner.clear()
            if data.implicit_samples:
                self._learner.load(data.implicit_samples)
            self._implicit_dirty = False
            self._drift.reset()
            self._model = None  # force a fresh validation below
            self._validate_calibration(announce=False)
            self._reset_tracking()
            if self._model is None:
                log.warning("The new calibration is not usable: %s", self._calibration_reason)
            else:
                log.info("Calibration saved (%s)", data.grade or "ungraded")
        elif self._calibrating:
            log.info("Calibration cancelled")
        self._calibrating = False
        self._update_state()
        self._apply_motion_gate()

    def apply_settings(self, settings: Settings) -> None:
        """Adopt new settings: persist them and reconfigure every component."""
        new = settings.copy()
        old = self._settings
        self._settings = new
        try:
            new.save(paths.settings_file())
        except OSError as exc:
            log.error("Could not save settings: %s", exc)
            self._notify("Settings not saved", str(exc), force=True)

        self._decider.set_config(SwitchConfig.from_settings(new.switching))
        self._filter.set_smoothing(new.switching.smoothing)
        self._presence.set_config(PresenceConfig.from_settings(new.presence))
        self._guard.set_config(GuardConfig.from_settings(new.privacy))
        self._policy.set_profile(new.performance.profile)
        self._learner.set_max_samples(new.learning.max_samples)

        backend_changed = old.general.backend != new.general.backend
        if backend_changed:
            self._expected_cache = None
        worker = self._worker
        if worker is not None:
            worker.set_max_faces(self._max_faces())
            self._apply_motion_gate()
            source_changed = old.camera != new.camera or _camera_fps(old) != _camera_fps(new)
            if backend_changed:
                # The worker keeps reporting the old backend until the new one
                # exists; ignore those reports meanwhile.
                self._stale_backend = self._safe_backend_info()
                self._reported_backend = None
            if source_changed or backend_changed:
                worker.reconfigure(
                    source_factory=_source_factory(new) if source_changed else None,
                    backend_factory=(
                        _backend_factory(new, self._max_faces()) if backend_changed else None
                    ),
                )
        if old.hotkeys != new.hotkeys and self._hotkeys is not None:
            self._register_hotkeys()

        if self._started and not self._closed:
            now = self._clock()
            if not new.privacy.shoulder_guard:
                self._update_guard(now, None)  # a disabled guard clears at once
            if old.privacy != new.privacy:
                self._next_yield_check = now
            if backend_changed:
                self._validate_calibration(announce=True)
            self._update_presence(now)
            self._update_state()
            self._update_rate(now, force=True)
            self._update_timer()
        log.info("Settings applied")
        self.settings_changed.emit(new)

    def refresh_monitors(self) -> None:
        """Re-read the monitor layout (called automatically after screen changes)."""
        if self._closed:
            return
        monitors = self._read_monitors()
        if not monitors:
            log.debug("No monitors reported; keeping the previous layout")
            return
        if _layout_key(monitors) == _layout_key(self._monitors):
            self._monitors = monitors  # names or scale factors may have changed
            return
        log.info(
            "Monitor layout changed: %s",
            ", ".join(f"{m.index}:{m.rect.w}x{m.rect.h}@{m.rect.x},{m.rect.y}" for m in monitors),
        )
        self._set_monitors(monitors)
        self._memory.clear()
        self._reset_tracking()
        if self._started:
            self._validate_calibration(announce=True)
            self._update_state()

    def handle_command(self, command: str) -> str:
        """Execute an IPC command (see :data:`COMMANDS`) and return the reply line.

        ``status`` returns a one-line JSON object; UI commands are forwarded
        through ``ui_requested``; everything else returns ``"ok"`` or
        ``"error: …"``.
        """
        cmd = (command or "").strip().lower()
        if cmd == "pause":
            self.pause()
        elif cmd == "resume":
            self.resume()
        elif cmd == "toggle":
            self.toggle_pause()
        elif cmd == "privacy-on":
            self.set_privacy(True)
        elif cmd == "privacy-off":
            self.set_privacy(False)
        elif cmd == "privacy-toggle":
            self.toggle_privacy()
        elif cmd == "calibrate":
            self.calibration_required.emit("ipc")
        elif cmd == "status":
            return json.dumps(self.status(), ensure_ascii=False, separators=(",", ":"))
        elif cmd in UI_COMMANDS:
            self.ui_requested.emit(cmd)
        else:
            return f"error: unknown command {(command or '').strip()!r}"
        return "ok"

    def status(self) -> dict[str, Any]:
        """JSON-safe summary of the running instance (``eye-tracker ctl status``)."""
        cal = self._calibration
        stats = self._last_stats
        return {
            "version": __version__,
            "state": self._state.value,
            "label": self._state.label,
            "paused": self._paused,
            "privacy": self._privacy,
            "calibrated": self._model is not None,
            "calibration": None
            if cal is None
            else {
                "grade": cal.grade,
                "created_at": cal.created_at,
                "backend": cal.backend,
                "usable": self._model is not None,
                "reason": self.calibration_reason,
            },
            "backend": self.backend_info()[0],
            "monitors": len(self._monitors),
            "presence": self._presence.state.value,
            "guard_active": self._guard.active,
            "yield_reason": self._yield_reason or None,
            "fps": stats.get("fps"),
            "target_fps": stats.get("target_fps"),
            "cpu_percent": stats.get("cpu_percent"),
            "switches": self._switch_count,
        }

    # ================================================================ housekeeping
    def tick(self) -> None:
        """One housekeeping step (normally driven by the internal timer)."""
        if not self._started or self._closed:
            return
        now = self._clock()
        self._poll_input(now)
        if now >= self._next_window_poll:
            self._next_window_poll = now + WINDOW_POLL_S
            self._record_foreground_window()
        if now >= self._next_lock_check:
            self._next_lock_check = now + LOCK_POLL_S
            self._check_session_lock(now)
        if now >= self._next_yield_check:
            self._next_yield_check = now + YIELD_POLL_S
            self._check_camera_yield(now)
        self._sync_backend_info()
        self._update_presence(now)
        self._update_state()
        self._update_rate(now)
        self._update_timer()
        if now >= self._next_stats:
            self._next_stats = now + STATS_PERIOD_S
            self._emit_stats()

    def _poll_input(self, now: float) -> None:
        pos = self._cursor_pos()
        if pos is None:
            return
        self._input.poll(now, pos)
        if self._input.manual_move:
            self._on_manual_move(pos, now)

    def _on_manual_move(self, pos: tuple[int, int], now: float) -> None:
        monitor = monitor_at(self._monitors, pos[0], pos[1])
        if monitor is not None:
            self._memory.record_cursor(monitor.index, pos)
        if self._model is not None and self._settings.learning.adaptive:
            self._learner.on_manual_cursor(pos[0], pos[1], now)
        last = self._last_switch
        if last is None:
            return
        switched_at, source, _target = last
        if now - switched_at > WRONG_SWITCH_S:
            self._last_switch = None
        elif monitor is not None and source is not None and monitor.index == source:
            # The user dragged the pointer straight back: that switch was wrong.
            self._last_switch = None
            self._drift.record_wrong_switch()
            log.debug("Automatic switch undone by the user")
            self._check_drift(now)

    def _record_foreground_window(self) -> None:
        if self._state is not TrackingState.TRACKING or not self._settings.switching.focus_window:
            return
        ref = self._platform_call("foreground_window")
        if not isinstance(ref, WindowRef) or ref.rect is None or not self._monitors:
            return
        cx, cy = ref.rect.center
        monitor = monitor_at(self._monitors, cx, cy) or nearest_monitor(self._monitors, cx, cy)[0]
        self._memory.record_window(monitor.index, ref)

    def _check_session_lock(self, now: float) -> None:
        locked = self._platform_call("is_session_locked")
        if locked is None:
            return
        locked = bool(locked)
        if locked == self._session_locked:
            return
        self._session_locked = locked
        if locked:
            log.info("Session locked")
            return
        log.info("Session unlocked")
        # Whoever unlocked is present; start the walk-away timers afresh.
        self._presence.reset(now)
        self._away = False
        self._displays_off_only = False
        self._hide_countdown()

    def _check_camera_yield(self, now: float) -> None:
        if self._state in (
            TrackingState.PRIVACY,
            TrackingState.LOCKED,
            TrackingState.PAUSED,
            TrackingState.CALIBRATING,
        ):
            return  # the camera is off anyway, or explicitly wanted
        p = self._settings.privacy
        apps = [a for a in p.pause_for_apps if a.strip()]
        reason = ""
        if apps or p.yield_camera:
            # Listing processes costs a few milliseconds; only when it matters.
            running = self._platform_call("running_process_names", default=set()) if apps else set()
            in_use = self._platform_call("camera_in_use_by_other_app") if p.yield_camera else None
            yes, why = should_yield(
                YieldInputs(in_use if isinstance(in_use, bool) else None, set(running or ()), apps),
                p.yield_camera,
            )
            reason = why if yes else ""
        if reason == self._yield_reason:
            return
        previous, self._yield_reason = self._yield_reason, reason
        if reason:
            log.info("Releasing the camera: %s", reason)
            if not previous:
                self._notify("Tracking paused", f"{reason} — the camera was released.")
        else:
            log.info("The camera is free again; tracking resumes")

    def _emit_stats(self) -> None:
        stats = self._safe_worker_stats()
        active = self._state.camera_active
        data: dict[str, Any] = {
            "fps": _finite(stats.fps) if active else 0.0,
            "target_fps": _finite(1.0 / self._interval)
            if active and self._interval and self._interval > 0
            else 0.0,
            "inference_ms": _finite(stats.inference_ms),
            "skip_ratio": _finite(stats.skip_ratio, 3),
            "cpu_percent": self._measure_cpu(),
            "state": self._state.value,
            "state_label": self._state.label,
            "backend": self.backend_info()[0],
            "camera_open": bool(stats.camera_open),
            "last_error": stats.last_error,
            "mode": self._mode,
        }
        self._last_stats = data
        self.stats_changed.emit(data)

    def _measure_cpu(self) -> float | None:
        """This process's CPU use as a percentage of the whole machine."""
        try:
            if self._process is None:
                import psutil

                self._process = psutil.Process()
                self._cpu_count = psutil.cpu_count() or 1
            value = float(self._process.cpu_percent(None)) / self._cpu_count
        except Exception:
            log.debug("CPU measurement failed", exc_info=True)
            return None
        self._cpu_percent = round(value, 2)
        return self._cpu_percent

    # ========================================================== worker callbacks
    def _on_main_thread(self) -> bool:
        return threading.get_ident() == self._main_thread

    def _on_worker_observation(self, obs: Observation) -> None:  # any thread
        if self._closed:
            return
        if self._on_main_thread():
            self._handle_observation(obs)
        else:
            self._observation_received.emit(obs)

    def _on_worker_stats(self, stats: WorkerStats) -> None:  # any thread
        if self._closed:
            return
        if self._on_main_thread():
            self._handle_worker_stats(stats)
        else:
            self._stats_received.emit(stats)

    def _on_worker_preview(self, frame: np.ndarray) -> None:  # any thread
        if self._closed:
            return
        # Only the newest frame matters; never let frames pile up in the queue.
        with self._preview_lock:
            already_queued = self._preview_pending is not None
            self._preview_pending = frame
        if already_queued:
            return
        if self._on_main_thread():
            self._deliver_preview()
        else:
            self._preview_ready.emit()

    def _deliver_preview(self) -> None:
        with self._preview_lock:
            frame, self._preview_pending = self._preview_pending, None
        if frame is not None and self._preview and not self._closed:
            self.preview_frame.emit(frame)

    def _handle_worker_stats(self, stats: WorkerStats) -> None:
        if self._closed:
            return
        self._worker_stats = stats
        self._sync_backend_info()
        if self._state.camera_active:
            error = bool(stats.last_error) and not stats.camera_open
            self._set_camera_error(error, stats.last_error)
        self._update_state()

    def _set_camera_error(self, error: bool, message: str | None) -> None:
        if error == self._camera_error:
            if error:
                self._camera_error_message = message
            return
        self._camera_error = error
        self._camera_error_message = message if error else None
        if error:
            log.warning("Camera problem: %s", message)
            self._notify("Camera unavailable", message or "The camera could not be opened.")
        else:
            log.info("Camera working again")

    # ============================================================== observations
    def _handle_observation(self, obs: Observation) -> None:
        if self._closed or not self._started or not self._state.camera_active:
            return  # a frame analysed just before the camera was switched off
        now = self._clock()
        self._last_obs_time = now
        self._last_face = bool(obs.face_present)
        if self._camera_error:
            self._set_camera_error(False, None)  # frames arrive, so the camera works
            self._update_state()
        self.observation.emit(obs)
        if self._state is TrackingState.CALIBRATING:
            self._update_rate(now)
            return
        self._update_guard(now, int(obs.face_count))
        self._update_presence(now)
        self._update_state()
        self._last_decision = None
        if self._state is TrackingState.TRACKING:
            self._track(obs, now)
            self._update_state()
        self._update_rate(now)
        self._update_timer()
        if self._trace is not None:
            self._write_trace(obs, now)

    def _write_trace(self, obs: Observation, now: float) -> None:
        trace = self._trace
        if trace is None:
            return
        cursor = self._cursor_pos()
        decision = self._last_decision
        gaze = self._last_gaze if self._last_gaze_time == now else None
        trace.write(
            {
                "t": trace.number(now, 3),
                "state": self._state.value,
                "faces": int(obs.face_count),
                "usable": bool(obs.usable),
                "skipped": bool(obs.skipped),
                "blink": bool(obs.blink),
                "q": trace.number(obs.quality, 3),
                "yaw": trace.number(obs.head_yaw, 2),
                "pitch": trace.number(obs.head_pitch, 2),
                "ms": trace.number(obs.inference_ms, 2),
                "f": trace.vector(obs.features),
                "gaze": trace.vector(gaze, 1),
                "cursor": list(cursor) if cursor else None,
                "cand": decision.candidate if decision else None,
                "reason": decision.reason if decision else None,
                "prog": trace.number(decision.progress, 2) if decision else None,
                "target": decision.target if decision else None,
            }
        )

    def _track(self, obs: Observation, now: float) -> None:
        gaze = self._estimate_gaze(obs, now)
        if self._model is None:
            return  # the calibration turned out to be unusable
        self._publish_gaze(gaze, now)
        if self._settings.learning.adaptive:
            self._learn(obs, now)
        cursor = self._cursor_pos()
        current = monitor_at(self._monitors, cursor[0], cursor[1]) if cursor else None
        decision = self._decider.update(
            now,
            gaze,
            current.index if current is not None else None,
            self._input.last_mouse_activity,
            self._input.last_key_activity,
            enabled=self._settings.switching.enabled,
        )
        self._pending = decision.pending
        self._last_decision = decision
        if decision.target is not None:
            self._switch_to(decision.target, gaze, cursor, now)

    def _estimate_gaze(self, obs: Observation, now: float) -> tuple[float, float] | None:
        model = self._model
        if model is None:
            return None
        if obs.usable and obs.features is not None:
            try:
                raw = model.predict(obs.features)
            except (ValueError, RuntimeError) as exc:
                log.warning("Gaze model rejected the camera features: %s", exc)
                self._invalidate_calibration("the camera features no longer match the calibration")
                return None
            x, y = float(raw[0]), float(raw[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            if self._last_gaze_time is None or now - self._last_gaze_time > FILTER_RESET_S:
                self._filter.reset()
            fx, fy = self._filter.update(x, y, now)
            last = self._last_gaze
            if last is not None and math.hypot(fx - last[0], fy - last[1]) > (
                GAZE_MOVE_FRACTION * self._min_side
            ):
                self._gaze_moving_until = now + GAZE_MOVING_HOLD_S
            self._last_gaze = (fx, fy)
            self._last_gaze_time = now
            return (fx, fy)
        if (
            obs.blink
            and self._last_gaze is not None
            and self._last_gaze_time is not None
            and now - self._last_gaze_time <= BLINK_HOLD_S
        ):
            return self._last_gaze
        return None

    def _publish_gaze(self, gaze: tuple[float, float] | None, now: float) -> None:
        if gaze is None:
            if not self._gaze_none_sent:
                self._gaze_none_sent = True
                self.gaze_changed.emit(None)
            return
        self._gaze_none_sent = False
        self.gaze_changed.emit(GazePoint(gaze[0], gaze[1], now))

    def _switch_to(
        self,
        target: int,
        gaze: tuple[float, float] | None,
        cursor: tuple[int, int] | None,
        now: float,
    ) -> None:
        monitor = next((m for m in self._monitors if m.index == target), None)
        if monitor is None:
            return
        sw = self._settings.switching
        previous = monitor_at(self._monitors, cursor[0], cursor[1]) if cursor else None
        if previous is not None and cursor is not None:
            self._memory.record_cursor(previous.index, cursor)

        window = self._remembered_window(monitor) if sw.focus_window else None
        x, y = choose_cursor_target(
            sw.cursor_target,
            monitor,
            self._memory,
            gaze,
            window.rect if window is not None else None,
        )
        # Warp first, then activate: input injected by activation fallbacks must
        # not be mistaken for the user typing (see InputTracker).
        self._input.note_programmatic_move((x, y), now)
        self._move_cursor(x, y)
        if sw.focus_window:
            if window is None:
                found = self._platform_call("window_at", x, y)
                window = found if isinstance(found, WindowRef) else None
            if window is not None:
                if self._platform_call("activate_window", window, default=False):
                    self._memory.record_window(monitor.index, window)
                else:
                    log.debug("Could not activate the window on monitor %d", monitor.index)

        self._decider.notify_switched(now, target)
        self._last_switch = (now, previous.index if previous is not None else None, target)
        self._switch_count += 1
        log.debug("Switched to monitor %d (cursor %d, %d)", target, x, y)
        self.switched.emit(target)

    def _remembered_window(self, monitor: Monitor) -> WindowRef | None:
        """The last window used on ``monitor`` if it still exists and is still there."""
        ref = self._memory.last_window(monitor.index)
        if ref is None:
            return None
        if not self._platform_call("is_window_valid", ref, default=False):
            self._memory.forget_window(monitor.index)
            return None
        rect = self._platform_call("window_rect", ref)
        if not isinstance(rect, Rect):
            rect = ref.rect
        if rect is None:
            return ref
        if not monitor.rect.contains(*rect.center):
            self._memory.forget_window(monitor.index)  # moved to another monitor
            return None
        return ref if rect == ref.rect else dataclasses.replace(ref, rect=rect)

    def _move_cursor(self, x: int, y: int) -> bool:
        try:
            result = self._cursor.set_pos(int(x), int(y))
        except Exception:
            log.warning("Moving the cursor failed", exc_info=True)
            result = False
        ok = result is None or bool(result)
        if not ok and not self._cursor_warning_shown:
            self._cursor_warning_shown = True
            self._notify(
                "Cannot move the cursor",
                "The system refused to move the mouse pointer. On Wayland, install "
                "ydotool (see the documentation).",
            )
        return ok

    # ------------------------------------------------------------------ learning
    def _learn(self, obs: Observation, now: float) -> None:
        cal, model = self._calibration, self._model
        if cal is None or model is None or self._learner.max_samples <= 0:
            return
        sample = self._learner.on_observation(obs, now, self._monitor_index_at)
        if sample is None:
            return
        self._implicit_dirty = True
        self._drift.record(self._predicted_monitor(model, sample.features), sample.monitor_index)
        self._check_drift(now)
        if self._learner.should_refit():
            self._refit(cal, model)

    def _refit(self, cal: CalibrationData, model: GazeModel) -> None:
        self._learner.mark_refit()
        learned = self._learner.samples
        try:
            refined = refit_model(cal.samples, learned, model)
        except (ValueError, np.linalg.LinAlgError) as exc:
            log.warning("Could not refine the gaze model: %s", exc)
            return
        cal.model = refined
        cal.implicit_samples = learned
        self._model = refined
        self._save_calibration(cal, quiet=True)
        log.info("Gaze model refined with %d learned samples", len(learned))

    def _check_drift(self, now: float) -> None:
        if self._settings.learning.drift_alerts and self._drift.should_alert(
            now, DRIFT_ALERT_COOLDOWN_S
        ):
            self._notify(
                "Accuracy dropped",
                "Accuracy dropped — recalibrate? Choose Calibrate… in the tray menu.",
            )

    def _predicted_monitor(self, model: GazeModel, features: np.ndarray) -> int | None:
        if not self._monitors:
            return None
        try:
            px, py = (float(v) for v in model.predict(features))
        except (ValueError, RuntimeError):
            return None
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        return nearest_monitor(self._monitors, px, py)[0].index

    def _monitor_index_at(self, x: float, y: float) -> int | None:
        monitor = monitor_at(self._monitors, x, y)
        return monitor.index if monitor is not None else None

    # ================================================================== presence
    def _update_presence(self, now: float) -> None:
        events = self._presence.update(
            now, self._presence_face(now), self._input.seconds_since_any(now)
        )
        self._handle_presence_events(events)

    def _presence_face(self, now: float) -> bool | None:
        """The camera's verdict for presence, ``None`` when it cannot tell."""
        state = self._state
        if (
            not state.camera_active
            or state in (TrackingState.CALIBRATING, TrackingState.CAMERA_ERROR)
            or self._last_obs_time is None
            or self._last_face is None
        ):
            return None
        stale = max(STALE_OBSERVATION_S, 2.5 * (self._interval or 0.0))
        if now - self._last_obs_time > stale:
            return None
        return self._last_face

    def _handle_presence_events(self, events: list[PresenceEvent]) -> None:
        for event in events:
            if event.kind == "warn":
                self._warning_shown = True
                self.away_warning.emit(float(event.remaining_s))
            elif event.kind == "cancel":
                self._hide_countdown()
            elif event.kind == "away":
                self._hide_countdown()
                self._away = True
                self._perform_away_action()
            elif event.kind == "return":
                self._away = False
                self._hide_countdown()
                if self._displays_off_only and self._settings.presence.wake_on_return:
                    log.info("User returned; waking the displays")
                    self._platform_call("wake_display")
                self._displays_off_only = False

    def _perform_away_action(self) -> None:
        action = self._settings.presence.action
        log.info("User away; action: %s", action)
        if action == "none":
            return
        if action == "notify":
            self._notify(
                "Are you still there?",
                "Nobody has been at the computer for a while.",
                force=True,
            )
            return
        lock = action in ("lock", "lock_and_display_off") and not self._session_locked
        display_off = action in ("display_off", "lock_and_display_off")
        displays_ok = False
        if display_off:
            # Displays first: a lock screen could otherwise swallow the request.
            displays_ok = bool(self._platform_call("display_off", default=False))
            if not displays_ok:
                log.warning("Turning the displays off is not supported here")
                if not lock:
                    self._notify(
                        "Could not turn off the displays",
                        "This system does not allow it; choose another walk-away action.",
                        force=True,
                    )
        if lock:
            if self._platform_call("lock_screen", default=False):
                # Notice the lock soon so the camera is released promptly.
                self._next_lock_check = min(self._next_lock_check, self._clock() + 0.5)
            else:
                log.warning("Locking the screen is not supported here")
                self._notify(
                    "Could not lock the screen",
                    "This system does not allow it; choose another walk-away action.",
                    force=True,
                )
        self._displays_off_only = display_off and displays_ok and not lock

    def _hide_countdown(self) -> None:
        if self._warning_shown:
            self._warning_shown = False
            self.away_cancelled.emit()

    # ===================================================================== guard
    def _update_guard(self, now: float, face_count: int | None) -> None:
        result = self._guard.update(now, face_count)
        if result == "trigger":
            self._on_guard_trigger()
        elif result == "clear":
            self._show_curtain(False)

    def _on_guard_trigger(self) -> None:
        action = self._settings.privacy.guard_action
        log.info("Shoulder guard: second face detected; action %s", action)
        if action == "lock":
            if self._session_locked or self._platform_call("lock_screen", default=False):
                return
            log.warning("Could not lock the screen; showing the privacy curtain instead")
            self._show_curtain(True)
        elif action == "curtain":
            self._show_curtain(True)
        else:
            self._notify(
                "Someone is looking at your screen",
                "A second face has been in view for a few seconds.",
                force=True,
            )

    def _show_curtain(self, visible: bool) -> None:
        if visible == self._curtain:
            return
        self._curtain = visible
        self.guard_changed.emit(visible)

    def _clear_guard(self) -> None:
        self._guard.reset()
        self._show_curtain(False)

    # ===================================================================== state
    def _derive_state(self) -> TrackingState:
        if not self._started:
            return TrackingState.STARTING
        if self._privacy:
            return TrackingState.PRIVACY
        if self._session_locked and self._settings.privacy.pause_when_locked:
            return TrackingState.LOCKED
        if self._calibrating:
            return TrackingState.CALIBRATING
        if self._paused:
            return TrackingState.PAUSED
        if self._yield_reason:
            return TrackingState.YIELDED
        if self._away:
            return TrackingState.AWAY
        if self._camera_error:
            return TrackingState.CAMERA_ERROR
        if self._model is None:
            return TrackingState.NEEDS_CALIBRATION
        return TrackingState.TRACKING

    def _update_state(self) -> None:
        new = self._derive_state()
        if new is self._state:
            return
        old, self._state = self._state, new
        log.info("State: %s -> %s", old.value, new.value)
        self._enter_state(old, new, self._clock())
        self.state_changed.emit(new)

    def _enter_state(self, old: TrackingState, new: TrackingState, now: float) -> None:
        if self._worker is not None:
            self._worker.set_active(new.camera_active)
        if not new.camera_active:
            # Judged afresh once the camera is back on.
            self._camera_error = False
            self._last_obs_time = None
            self._last_face = None
            self._clear_guard()
        if not new.camera_active or new is TrackingState.CALIBRATING:
            # Time without a camera verdict never counts as absence.
            self._handle_presence_events(self._presence.update(now, None, None))
        if new is not TrackingState.TRACKING:
            self._reset_tracking()
        if TrackingState.CALIBRATING in (old, new):
            self._apply_motion_gate()
        if old in _CAMERA_OFF or old is TrackingState.CALIBRATING:
            self._next_yield_check = min(self._next_yield_check, now)
        self._update_rate(now)
        self._update_timer()

    def _reset_tracking(self) -> None:
        self._decider.reset()
        self._filter.reset()
        self._pending = False
        self._last_gaze = None
        self._last_gaze_time = None
        self._gaze_moving_until = -math.inf
        self._publish_gaze(None, 0.0)

    def _update_rate(self, now: float, *, force: bool = False) -> None:
        worker = self._worker
        if worker is None:
            return
        sw = self._settings.switching
        ctx = RateContext(
            state=self._state,
            pending_switch=self._pending,
            gaze_moving=now < self._gaze_moving_until,
            typing=now - self._input.last_key_activity
            < max(TYPING_RATE_WINDOW_S, sw.typing_grace_ms / 1000.0),
            face_present=self._last_face is not False,
            presence_warning=self._presence.state is PresenceState.WARNING,
            preview=self._preview,
        )
        interval = self._policy.interval(ctx)
        self._mode = self._policy.mode(ctx)
        current = self._interval
        if (
            not force
            and current is not None
            and abs(interval - current) <= RATE_TOLERANCE * current
        ):
            return
        self._interval = interval
        worker.set_interval(interval)

    def _tick_interval(self) -> int:
        fast = (
            self._state is TrackingState.TRACKING or self._presence.state is PresenceState.WARNING
        )
        return TICK_ACTIVE_MS if fast else TICK_IDLE_MS

    def _update_timer(self) -> None:
        interval = self._tick_interval()
        if self._timer.interval() != interval:
            self._timer.setInterval(interval)

    def _apply_motion_gate(self) -> None:
        if self._worker is None:
            return
        perf = self._settings.performance
        # During calibration every frame is a sample; skipped copies would not be.
        enabled = perf.motion_gate and self._state is not TrackingState.CALIBRATING
        self._worker.set_motion_gate(enabled, perf.motion_threshold)

    def _max_faces(self) -> int:
        return 2 if self._settings.privacy.shoulder_guard else 1

    # =============================================================== calibration
    def _load_calibration(self) -> None:
        data = load_calibration(paths.calibration_file())
        self._calibration = data
        self._model = None
        if data is None:
            self._calibration_reason = "not calibrated yet"
            return
        self._learner.clear()
        if data.implicit_samples:
            self._learner.load(data.implicit_samples)

    def _save_calibration(self, data: CalibrationData, *, quiet: bool) -> None:
        try:
            save_calibration(paths.calibration_file(), data)
        except (OSError, ValueError) as exc:
            log.error("Could not save the calibration: %s", exc)
            if not quiet:
                self._notify("Calibration not saved", str(exc), force=True)
            return
        self._implicit_dirty = False

    def _validate_calibration(self, *, announce: bool) -> None:
        """Decide whether the calibration fits the backend and monitors in use."""
        data = self._calibration
        was_usable = self._model is not None
        if data is None:
            self._model = None
            self._calibration_reason = "not calibrated yet"
            return
        if not self._monitors:
            return  # nothing to compare with; keep the current verdict
        info = self._current_backend()
        if info is not None:
            ok, reason = data.is_compatible(info[0], info[1], self._monitors)
        elif not data.model.is_fitted:
            ok, reason = False, "the calibration has no fitted model"
        elif layout_signature(self._monitors) != data.layout_signature:
            ok, reason = False, "the monitor layout changed"
        else:
            # Backend unknown (none installed?): the layout is all that can be checked.
            ok, reason = True, ""
        if ok:
            if not was_usable:
                log.info("Calibration is usable")
                self._reset_tracking()
            self._model = data.model
            self._calibration_reason = ""
            return
        self._model = None
        self._calibration_reason = reason
        if was_usable:
            log.warning("Calibration no longer usable: %s", reason)
            if announce and self._may_interrupt():
                self.calibration_required.emit(reason)

    def _invalidate_calibration(self, reason: str) -> None:
        was_usable = self._model is not None
        self._model = None
        self._calibration_reason = reason
        if was_usable and self._may_interrupt():
            self.calibration_required.emit(reason)

    def _may_interrupt(self) -> bool:
        """Whether a calibration prompt is appropriate right now.

        Not while the user is away or displays are off (monitors drop out during
        display sleep and come back unchanged), and not with a single monitor,
        where switching has nothing to do.
        """
        return (
            len(self._monitors) >= 2
            and not self._calibrating
            and not self._displays_off_only
            and self._state not in (TrackingState.AWAY, TrackingState.LOCKED, TrackingState.PRIVACY)
        )

    # =================================================================== backend
    def _safe_backend_info(self) -> tuple[str, str] | None:
        worker = self._worker
        if worker is None:
            return None
        try:
            info = worker.backend_info
        except Exception:
            return None
        if not info:
            return None
        return (str(info[0]), str(info[1]))

    def _sync_backend_info(self) -> None:
        info = self._safe_backend_info()
        if info is None or info == self._reported_backend:
            return
        if self._stale_backend is not None and info == self._stale_backend:
            return  # still the backend from before a reconfigure
        self._stale_backend = None
        self._reported_backend = info
        log.debug("Vision backend in use: %s (%s)", *info)
        self._validate_calibration(announce=True)

    def _current_backend(self) -> tuple[str, str] | None:
        return self._reported_backend or self._expected_backend()

    def _expected_backend(self) -> tuple[str, str] | None:
        """The backend the settings select, before the worker has created it."""
        name = self._settings.general.backend
        cached = self._expected_cache
        if cached is not None and cached[0] == name:
            return cached[1]
        info: tuple[str, str] | None
        try:
            from ..vision.backends import backend_class

            cls = backend_class(name)
            info = (cls.name, cls.feature_version)
        except Exception as exc:
            log.debug("Cannot determine the vision backend: %s", exc)
            info = None
        self._expected_cache = (name, info)
        return info

    def _safe_worker_stats(self) -> WorkerStats:
        worker = self._worker
        if worker is not None:
            try:
                return worker.stats
            except Exception:
                log.debug("Reading worker stats failed", exc_info=True)
        return self._worker_stats

    # =================================================================== hotkeys
    def _setup_hotkeys(self) -> None:
        if self._hotkeys is None:
            factory = self._hotkey_factory
            if factory is None:
                from ..platform.hotkeys import create_hotkey_manager

                factory = create_hotkey_manager
            try:
                self._hotkeys = factory()
            except Exception:
                log.warning("Global hotkeys unavailable", exc_info=True)
                return
        self._register_hotkeys()

    def _register_hotkeys(self) -> None:
        manager = self._hotkeys
        if manager is None or self._hotkeys_suspended:
            return
        hk = self._settings.hotkeys
        try:
            manager.unregister_all()
        except Exception:
            log.debug("Releasing hotkeys failed", exc_info=True)
        if not hk.enabled:
            with contextlib.suppress(Exception):
                manager.stop()
            return
        if not manager.supported:
            if manager.note:
                log.info("Global hotkeys unavailable: %s", manager.note)
            return
        failed: list[str] = []
        for name in HOTKEY_ACTIONS:
            combo = str(getattr(hk, name, "") or "").strip()
            if not combo:
                continue
            try:
                ok = manager.register(name, combo, functools.partial(self._on_hotkey, name))
            except Exception:
                log.warning("Registering hotkey %s failed", combo, exc_info=True)
                ok = False
            if not ok:
                failed.append(combo)
        if failed:
            self._notify(
                "Hotkey unavailable",
                f"{', '.join(failed)} could not be registered (invalid, or used by another "
                "app). Choose another in Settings → Hotkeys.",
            )

    def _on_hotkey(self, name: str) -> None:  # hotkey thread (or the main thread on macOS)
        if self._closed:
            return
        if self._on_main_thread():
            self._handle_hotkey(name)
        else:
            self._hotkey_pressed.emit(name)

    def _handle_hotkey(self, name: str) -> None:
        if self._closed:
            return
        if self._hotkeys_suspended:
            log.debug("Hotkey %s ignored while hotkeys are suspended", name)
            return
        log.info("Hotkey: %s", name)
        if name == "toggle_tracking":
            self.toggle_pause()
        elif name == "toggle_privacy":
            self.toggle_privacy()
        elif name == "recalibrate":
            self.calibration_required.emit("hotkey")

    # =================================================================== screens
    def _connect_screens(self) -> None:
        app = QGuiApplication.instance()
        if not isinstance(app, QGuiApplication) or self._screens_connected:
            return
        app.screenAdded.connect(self._on_screen_added)
        app.screenRemoved.connect(self._schedule_monitor_refresh)
        app.primaryScreenChanged.connect(self._schedule_monitor_refresh)
        for screen in QGuiApplication.screens():
            screen.geometryChanged.connect(self._schedule_monitor_refresh)
        self._screens_connected = True

    def _disconnect_screens(self) -> None:
        app = QGuiApplication.instance()
        if not self._screens_connected or not isinstance(app, QGuiApplication):
            return
        self._screens_connected = False
        with contextlib.suppress(RuntimeError, TypeError):
            app.screenAdded.disconnect(self._on_screen_added)
            app.screenRemoved.disconnect(self._schedule_monitor_refresh)
            app.primaryScreenChanged.disconnect(self._schedule_monitor_refresh)

    def _on_screen_added(self, screen: Any) -> None:
        with contextlib.suppress(RuntimeError, AttributeError):
            screen.geometryChanged.connect(self._schedule_monitor_refresh)
        self._schedule_monitor_refresh()

    def _schedule_monitor_refresh(self, *_args: object) -> None:
        if not self._closed:
            self._screen_timer.start()

    def _read_monitors(self) -> list[Monitor]:
        try:
            return list(self._monitors_provider())
        except Exception:
            log.warning("Reading the monitor layout failed", exc_info=True)
            return []

    def _set_monitors(self, monitors: list[Monitor]) -> None:
        self._monitors = list(monitors)
        self._decider.set_monitors(self._monitors)
        sides = [min(m.rect.w, m.rect.h) for m in self._monitors if m.rect.w > 0 and m.rect.h > 0]
        self._min_side = float(min(sides)) if sides else 1000.0

    # ==================================================================== helpers
    def _cursor_pos(self) -> tuple[int, int] | None:
        try:
            x, y = self._cursor.pos()
        except Exception:
            log.debug("Reading the cursor position failed", exc_info=True)
            return None
        return (int(x), int(y))

    def _platform_call(self, name: str, *args: Any, default: Any = None) -> Any:
        """Call a platform method; the contract says they never raise, but be safe."""
        try:
            return getattr(self._platform, name)(*args)
        except Exception:
            log.debug("Platform call %s failed", name, exc_info=True)
            return default

    def _notify(self, title: str, message: str, *, force: bool = False) -> None:
        """Emit ``notify``. Informational messages respect the notifications setting;
        ``force`` is for actions the user chose to be notified about."""
        log.info("Notification: %s — %s", title, message)
        if force or self._settings.general.notifications:
            self.notify.emit(title, message)
