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
the main thread is handled directly. Locking the screen can block for seconds,
so it runs on a short-lived thread whose result comes back the same way; so do
the split-pane providers, on the pane worker's thread (:mod:`eye_tracker.panes`).
Everything else runs on the main thread,
so no locking is needed beyond the hand-off of preview frames and the record of
the backend the worker built.

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
Before the camera reopens after a pause, privacy mode or a locked session, the
"another app uses the camera" check runs first, so a call that started
meanwhile never has its camera grabbed.

Calibrations
------------
The calibration file holds one profile per setup (monitor layout, vision
backend, camera; see :class:`~eye_tracker.gaze.store.CalibrationLibrary`). The
controller uses the profile that fits the current setup and switches profiles
by itself when the setup changes (a laptop moving between docks), so a
recalibration is only needed for a setup that was never calibrated. When the
calibration becomes unusable ``calibration_required`` is emitted; if the user
cannot be interrupted at that moment (away, locked, privacy mode, displays off)
the announcement is delivered once they are back and the monitor layout has
settled.

Split panes (experimental)
--------------------------
With ``panes.enabled`` the focused window is followed as well: its application
is identified, its panes are asked for (once when it gets focus, then about
every :data:`PANE_REFRESH_S` while the gaze is inside it) and, while the monitor
decider finds the gaze on the cursor's monitor, :class:`~eye_tracker.panes.decider.PaneDecider`
decides when the keyboard focus moves to another pane. That only asks the
terminal to focus the pane: the cursor stays and no window is activated. Pane
focus waits ``panes.after_monitor_switch_ms`` after a monitor switch.

Unreliable pointer position
---------------------------
On Wayland the pointer position Qt reports can be stale (an XWayland client
only sees the pointer over X11 windows). There the monitor the pointer is on is
taken to be the target of the last successful switch, and nothing is learned
from the pointer position (manual moves, implicit labels, undone switches).

Housekeeping
------------
A ``QTimer`` ticks every 100 ms while tracking (cursor polling resolution for the
mouse/typing guards) and every 500 ms otherwise. Each tick polls input, records
the focused window per monitor, checks for a locked session and for other apps
wanting the camera (on slower schedules), advances the walk-away timers, adapts
the camera frame rate, delivers deferred calibration announcements and
publishes statistics.
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
from ..config import Settings, atomic_write_text
from ..gaze.filters import PointFilter
from ..gaze.learning import DriftMonitor, ImplicitLearner, plausible_label, refit_model
from ..gaze.model import gaze_feature_indices
from ..gaze.store import CalibrationData, CalibrationLibrary, axis_error
from ..panes.decider import (
    FALLBACK_GAZE_ERROR_PX,
    PaneConfig,
    PaneDecider,
    PaneDecision,
    eligible_panes,
)
from ..panes.registry import PaneRegistry, default_providers, is_denied
from ..panes.types import PaneSnapshot
from ..panes.worker import DetectResult, FocusResult, PaneWorker
from ..platform.base import PlatformServices
from ..trace import TraceWriter
from ..types import (
    AppIdentity,
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
    #: Runs a job off the GUI thread (see ``Controller(lock_runner=...)``).
    JobRunner = Callable[[Callable[[], None]], None]
    #: Builds the pane providers for settings (see ``Controller(pane_registry_factory=...)``).
    PaneRegistryFactory = Callable[[Settings], PaneRegistry]
    #: Builds the pane worker for a registry.
    PaneWorkerFactory = Callable[[PaneRegistry], PaneWorker]

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
#: switch marks that switch as wrong (drift evidence); a switch left alone this
#: long counts as correct.
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
#: Consecutive observations the gaze model must reject (wrong feature vector)
#: before the calibration is declared unusable. After a backend change a few
#: frames analysed by the old backend are still on their way; they must not
#: invalidate a calibration that fits the new one.
FEATURE_MISMATCH_LIMIT = 5
#: After the user unlocks a session the shoulder guard locked, a second face in
#: view shows the privacy curtain instead of locking again: otherwise a colleague
#: sitting down next to the user would lock the screen again two seconds after
#: every unlock. The grace lasts this long after the unlock and after every later
#: trigger it absorbs, so it holds while the colleague keeps coming back into
#: view (turning to the user, looking down at notes) and ends once nobody has
#: looked over the user's shoulder for this long.
GUARD_RELOCK_GRACE_S = 300.0
#: Consecutive refused pointer moves after which switching pauses (backing off
#: from ``WARP_BACKOFF_S``, doubling up to ``WARP_BACKOFF_MAX_S``) instead of
#: retrying after every cooldown.
WARP_REFUSAL_LIMIT = 3
WARP_BACKOFF_S = 60.0
WARP_BACKOFF_MAX_S = 1800.0
#: A deferred "calibration needed" announcement is delivered once the user could
#: be interrupted for this long and no monitor change is pending: displays that
#: drop out during sleep come back unchanged and must not prompt.
ANNOUNCE_SETTLE_S = 2 * SCREEN_DEBOUNCE_MS / 1000.0
#: When the user comes back (from away, a locked session or privacy mode) to a
#: calibration that is still unusable, they are reminded, at most this often.
CALIBRATION_REMINDER_S = 1800.0
#: Blindness (lens covered, shutter closed, dark room) that begins within this
#: many seconds of seeing the user's face or their input means the user covered
#: the camera: walk-away detection then pauses instead of counting "no face".
BLIND_ONSET_S = 3.0
#: ...but for at most this long without any keyboard or mouse input: someone who
#: switched the desk lamp off and left must still be locked out eventually.
BLIND_FREEZE_MAX_S = 300.0
#: Minimum time between two "camera appears covered" notifications.
BLIND_NOTICE_COOLDOWN_S = 600.0
#: The worker reports a camera failure only when its message changes, so a camera
#: that still fails the same way after a pause would go unnoticed. With no frame
#: this long after the camera was switched on, the worker's error is read back.
CAMERA_ERROR_RECHECK_S = 8.0
#: Consecutive failed window activations after which a missing Accessibility
#: permission (macOS) is reported.
ACTIVATION_FAILURE_LIMIT = 3
#: How long :meth:`Controller.shutdown` waits for a screen lock still in progress.
LOCK_JOIN_TIMEOUT_S = 2.0
#: While the gaze is inside the focused terminal window its panes are asked for
#: again this often (at the next window poll), so the layout the pane decider
#: sees is never much older than its freshness limit (1.5 s).
PANE_REFRESH_S = 1.0
#: How long :meth:`Controller.shutdown` waits for a pane provider call in progress.
PANE_JOIN_TIMEOUT_S = 2.0
#: Walk-away actions that lock the session or blank the displays. Until the
#: first-run setup is finished they only notify: the owner of a PC who never
#: saw the setup assistant must not be locked out by an app they just installed.
_DISRUPTIVE_AWAY_ACTIONS = frozenset({"lock", "lock_and_display_off", "display_off"})


def effective_away_action(settings: Settings) -> str:
    """The walk-away action that really runs: ``settings.presence.action``, except
    that a lock or blanked displays only notify while the first-run setup is not
    finished (see ``_DISRUPTIVE_AWAY_ACTIONS``). The countdown toast announces this
    one, so it never says "Locking" before a mere notification."""
    action = settings.presence.action
    if not settings.general.first_run_done and action in _DISRUPTIVE_AWAY_ACTIONS:
        return "notify"
    return action


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
#: Camera-off states in which the camera-in-use check is skipped; leaving one
#: re-runs it before the camera opens.
_UNCHECKED_CAMERA_OFF = frozenset(
    {TrackingState.PAUSED, TrackingState.PRIVACY, TrackingState.LOCKED}
)
#: States the user "comes back" from; a still unusable calibration is then
#: announced again (see CALIBRATION_REMINDER_S).
_ABSENT_STATES = frozenset({TrackingState.AWAY, TrackingState.LOCKED, TrackingState.PRIVACY})
#: States in which the app may be napped by macOS: the user stopped tracking,
#: and only their own action (hotkey, tray, IPC) resumes it. A locked session is
#: not one: its end must be noticed promptly.
_NAPPABLE_STATES = frozenset({TrackingState.PAUSED, TrackingState.PRIVACY})


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
                name=screen.name() or _fallback_screen_name(index),
                rect=Rect(geometry.x(), geometry.y(), geometry.width(), geometry.height()),
                primary=primary is not None and screen == primary,
                scale=float(screen.devicePixelRatio()),
            )
        )
    return monitors


def _fallback_screen_name(index: int) -> str:
    """The name :func:`qt_monitors` gives a screen that has none (by position)."""
    return f"Screen {index + 1}"


def _same_monitor(old: Monitor, before: list[Monitor], after: list[Monitor]) -> Monitor | None:
    """The monitor of the layout ``after`` that is most likely the physical
    monitor ``old`` of the layout ``before``.

    Indices follow Qt's screen order and say nothing once a screen was added or
    removed. A screen's name (its connector or device name) follows the monitor
    even when the desktop is laid out anew, so a name that is unique in both
    layouts decides first - unless it is merely a positional fallback name.
    Then the same rectangle, then whichever monitor now covers the old centre.
    """
    exact = next((m for m in after if m.name == old.name and m.rect == old.rect), None)
    if exact is not None:
        return exact
    named = [m for m in after if m.name == old.name]
    if (
        len(named) == 1
        and sum(m.name == old.name for m in before) == 1
        and old.name != _fallback_screen_name(old.index)
    ):
        return named[0]
    same_rect = next((m for m in after if m.rect == old.rect), None)
    if same_rect is not None:
        return same_rect
    return monitor_at(after, *old.rect.center)


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


def _backend_gaze_indices(backend: object) -> tuple[int, ...] | None:
    """Indices of a backend's (class's or instance's) gaze-direction features."""
    try:
        return gaze_feature_indices(
            tuple(getattr(backend, "feature_names", ())),
            tuple(getattr(backend, "gaze_features", ())),
        )
    except Exception:
        log.debug("Cannot read the gaze features of %r", backend, exc_info=True)
        return None


@functools.cache
def _named_backend_gaze_indices(name: str) -> tuple[int, ...] | None:
    """:func:`_backend_gaze_indices` for a backend known only by name (``None`` if unknown)."""
    try:
        from ..vision.backends import backend_class

        cls = backend_class(name)
    except Exception:
        return None
    return _backend_gaze_indices(cls)


@dataclasses.dataclass(frozen=True, slots=True)
class _BuiltBackend:
    """A backend the worker built through one of our factories."""

    #: Which factory built it (see ``Controller._make_backend_factory``).
    generation: int
    #: ``(name, feature_version)``.
    info: tuple[str, str]
    #: Indices of its gaze-direction features (``None``: not declared).
    gaze_indices: tuple[int, ...] | None


def _saved_privacy_mode() -> bool:
    """Whether privacy mode was on when the app last ran (``paths.state_file()``).

    An unreadable file counts as "off": it is only ever written by
    :func:`_save_privacy_mode`, so damage means it never held a choice.
    """
    try:
        data = json.loads(paths.state_file().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        log.warning("Could not read the remembered privacy mode: %s", exc)
        return False
    return isinstance(data, dict) and data.get("privacy_mode") is True


def _save_privacy_mode(enabled: bool) -> None:
    """Remember privacy mode for the next start (never raises)."""
    try:
        atomic_write_text(paths.state_file(), json.dumps({"privacy_mode": enabled}) + "\n")
    except OSError as exc:
        log.warning("Could not remember privacy mode for the next start: %s", exc)


def _layout_key(monitors: list[Monitor]) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple((m.index, m.rect.x, m.rect.y, m.rect.w, m.rect.h) for m in monitors)


def _pane_providers(settings: Settings) -> tuple[bool, bool, bool]:
    """The provider switches of the split-pane settings."""
    p = settings.panes
    return (p.tmux, p.wezterm, p.windows_terminal)


def _trace_id(value: Any) -> str | int | None:
    """A pane id as the trace stores it."""
    if value is None or isinstance(value, int):
        return value
    return str(value)


def _same_handle(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


def _finite(value: float, digits: int = 2) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None


def _known_size(size: tuple[int, int] | None) -> tuple[int, int] | None:
    """``size`` as ``(w, h)`` ints if both are positive, else ``None`` (unknown)."""
    if size is None:
        return None
    try:
        w, h = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (w, h) if w > 0 and h > 0 else None


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
        trace_path: Write a per-frame trace (JSON lines) to this file.
        lock_runner: ``(job) -> None`` that runs ``job`` off the GUI thread, where
            the screen is locked (``lock_screen`` can block for seconds); defaults
            to a short-lived thread. Tests pass a synchronous runner.
        pane_registry_factory: ``(settings) -> PaneRegistry`` with the split-pane
            providers; defaults to ``panes.registry.default_providers``.
        pane_worker_factory: ``(registry) -> PaneWorker``; defaults to a worker
            with its own thread. Tests pass a synchronous one.
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
    #: Shoulder-guard curtain on/off (for the "curtain" action, and as the fallback
    #: of the "lock" action).
    guard_changed = Signal(bool)
    #: Annotated camera frame (BGR ``np.ndarray``) while the preview is enabled.
    preview_frame = Signal(object)
    #: A (re)calibration is needed or was requested; the argument is the reason.
    #:
    #: Emitted for explicit requests (``"hotkey"``, ``"ipc"``), when a usable
    #: calibration becomes unusable (deferred until the user can be interrupted),
    #: and as a reminder, at most every :data:`CALIBRATION_REMINDER_S`, when the
    #: user comes back from away, a locked session or privacy mode to a
    #: calibration that is still unusable. The same reason can therefore arrive
    #: more than once; each emission is meant to be shown.
    calibration_required = Signal(str)
    #: Settings were applied (the new :class:`Settings`).
    settings_changed = Signal(object)
    #: An IPC command only the UI can perform: ``"show"``, ``"settings"`` or ``"quit"``.
    ui_requested = Signal(str)
    #: A feature is blocked by an OS permission (``"camera"`` or
    #: ``"accessibility"``). Emitted right after the ``notify`` that explains it
    #: (also when informational notifications are turned off and that ``notify``
    #: was not emitted), so the UI can open
    #: ``platform.open_permission_settings(name)`` when the notification is clicked.
    permission_needed = Signal(str)

    # Hand-off from foreign threads to the main thread (queued connections).
    _observation_received = Signal(object)
    _stats_received = Signal(object)
    _preview_ready = Signal()
    _hotkey_pressed = Signal(str)
    _lock_done = Signal(bool)

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
        lock_runner: JobRunner | None = None,
        pane_registry_factory: PaneRegistryFactory | None = None,
        pane_worker_factory: PaneWorkerFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._trace = TraceWriter(trace_path) if trace_path else None
        self._last_decision: Decision | None = None
        self._last_pane_decision: PaneDecision | None = None
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
        # Privacy mode was switched back on at start (remember_privacy_mode).
        self._privacy_restored = False
        self._calibrating = False
        self._session_locked = False
        self._yield_reason = ""
        self._away = False
        self._camera_error = False
        # Privacy mode as it was when a calibration began (restored afterwards).
        self._privacy_before_calibration = False
        # Consumers of preview frames (id of each owner, see set_preview).
        self._preview_owners: set[int] = set()

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

        # Split-pane focus (experimental, see eye_tracker.panes). The worker and
        # its thread exist only while the feature is on.
        self._pane_decider = PaneDecider(PaneConfig.from_settings(s.panes, s.switching))
        self._pane_registry_factory: PaneRegistryFactory = (
            pane_registry_factory or self._default_pane_registry
        )
        self._pane_worker_factory: PaneWorkerFactory = pane_worker_factory or functools.partial(
            PaneWorker, clock=clock
        )
        self._pane_worker: PaneWorker | None = None
        self._pane_capable: bool | None = None  # the platform's "panes" capability
        self._pane_window: WindowRef | None = None  # the focused window being followed
        self._pane_app: AppIdentity | None = None
        self._pane_snapshot: PaneSnapshot | None = None
        self._pane_requested_at = -math.inf
        self._pane_sigma_cache: tuple[CalibrationData, int, Rect, tuple[float, float] | None] | (
            None
        ) = None
        self._pane_sigma_used: tuple[float, float] | None = None
        self._pane_switches = 0

        # Calibration: every saved profile, and the one in use.
        self._library = CalibrationLibrary()
        self._calibration: CalibrationData | None = None
        self._model: GazeModel | None = None  # set only while the calibration is usable
        self._calibration_reason = "not calibrated yet"
        self._implicit_dirty = False
        self._feature_mismatches = 0
        # "Calibration needed" announcements (see _announce_calibration).
        self._announce_pending: str | None = None
        self._interruptible_since: float | None = None
        self._last_announce = -math.inf

        # What we know about the vision backend (for calibration compatibility).
        self._reported_backend: tuple[str, str] | None = None
        self._expected_cache: tuple[str, tuple[str, str] | None] | None = None
        # Every backend factory handed to the worker gets a generation number; the
        # factory records what it built (on the worker thread, hence the lock).
        self._backend_lock = threading.Lock()
        self._backend_generation = 0
        self._built_backend: _BuiltBackend | None = None

        # Worker.
        self._worker: WorkerLike | None = None
        self._interval: float | None = None
        self._mode = ""
        self._worker_stats = WorkerStats()
        self._camera_error_message: str | None = None
        self._camera_on_since: float | None = None
        self._frame_size: tuple[int, int] | None = None  # of the latest analysed frame
        self._preview_lock = threading.Lock()
        self._preview_pending: np.ndarray | None = None

        # Tracking transients.
        self._last_obs_time: float | None = None
        self._last_face: bool | None = None
        self._face_seen_at: float | None = None
        self._last_gaze: tuple[float, float] | None = None
        self._last_gaze_time: float | None = None
        self._looking_away = False
        self._gaze_none_sent = True
        self._gaze_moving_until = -math.inf
        self._pending = False
        self._last_switch: tuple[float, int | None, int] | None = None
        self._switch_count = 0

        # Pointer: whether its reported position can be trusted (not on Wayland),
        # and otherwise the monitor we last moved it to.
        self._cursor_reliable = True
        self._assumed_monitor: int | None = None
        # Refused pointer moves (see _move_cursor).
        self._warp_refusals = 0
        self._warp_suspensions = 0
        self._switching_suspended_until = -math.inf
        self._cursor_warning_shown = False
        # Window activation failures (macOS Accessibility, see _check_accessibility).
        self._activation_failures = 0
        self._accessibility_notice_shown = False
        self._background_active: bool | None = None

        # Blind camera (see _face_verdict).
        self._blind_since: float | None = None
        self._blind_freeze = False
        self._blind_notice_at = -math.inf

        # Presence / guard side effects.
        self._warning_shown = False
        # The walk-away action left the displays off and the session unlocked, so
        # the displays are to be woken when the user returns (wake_on_return).
        self._displays_off_only = False
        # The walk-away action turned the displays off (whether or not it also
        # locked); a lock that fails afterwards leaves them "off only".
        self._away_displays_off = False
        self._curtain = False
        self._guard_locked = False
        self._guard_relock_until = -math.inf
        # Screen lock requests in flight ("away", "guard"; see _request_lock).
        self._lock_runner: JobRunner = lock_runner or self._run_lock_thread
        self._lock_purposes: set[str] = set()
        self._lock_thread: threading.Thread | None = None

        # Housekeeping schedule (set in start()).
        self._next_window_poll = math.inf
        self._next_lock_check = math.inf
        self._next_yield_check = math.inf
        self._yield_checked_at = -math.inf
        self._next_stats = math.inf
        self._process: Any = None
        self._cpu_count = 1
        self._cpu_percent: float | None = None
        self._last_stats: dict[str, Any] = {}

        self._hotkeys: HotkeyManager | None = None
        self._hotkeys_suspended = False
        #: (hotkey name, combination) pairs whose failure was announced. The
        #: hotkeys are registered again after every stay in Settings → Hotkeys
        #: (see suspend_hotkeys); a failure that has not changed is not repeated.
        self._announced_hotkey_failures: set[tuple[str, str]] = set()
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
        self._lock_done.connect(self._finish_lock, queued)

    # ================================================================== lifecycle
    def start(self) -> None:
        """Load the calibration, start the camera worker, hotkeys and housekeeping."""
        if self._started or self._closed:
            return
        now = self._clock()
        s = self._settings
        self._cursor_reliable = bool(self._platform_call("cursor_position_reliable", default=True))
        if not self._cursor_reliable:
            log.info(
                "The pointer position cannot be read reliably here (Wayland); the "
                "current monitor is taken from the last switch"
            )
        self._set_monitors(self._read_monitors())
        self._presence.reset(now)
        self._paused = self._paused or bool(s.general.start_paused)
        if _saved_privacy_mode() and not self._privacy:
            if s.privacy.remember_privacy_mode:
                # Turned on before the last quit, reboot or update: the camera must
                # not come back on by itself. Applied before the first state, so
                # the worker never opens it.
                self._privacy = True
                self._privacy_restored = True
                log.info("Privacy mode is still on from the previous run")
            else:
                # Not remembered: turning remembering on later must not bring back
                # a choice from a run that did not apply it.
                _save_privacy_mode(False)
        self._load_calibration()

        worker = self._worker_factory(
            _source_factory(s),
            self._make_backend_factory(s),
            self._on_worker_observation,
            self._on_worker_stats,
            self._on_worker_preview,
        )
        self._worker = worker
        # Keep the camera closed until the state says otherwise (start paused,
        # locked session, privacy mode set before start()).
        worker.set_active(False)
        worker.set_max_faces(self._max_faces())
        if self._preview_owners:
            worker.set_preview(True)
        self._started = True
        self._apply_motion_gate()
        self._sync_backend_info()
        self._validate_calibration(announce=False)
        if self._model is None and self._calibration_useful() and not self._privacy:
            # The app offers the calibration at start (also after a login start,
            # once its quiet period is over); the reminders on the user's return
            # (_remind_calibration) follow no sooner than CALIBRATION_REMINDER_S.
            # In privacy mode it does not, and leaving it brings the reminder.
            self._last_announce = now

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
        self._check_accessibility()
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
        self._stop_pane_worker()
        lock_thread = self._lock_thread
        if lock_thread is not None and lock_thread.is_alive():
            # Let a lock in progress finish before this object may be deleted.
            lock_thread.join(LOCK_JOIN_TIMEOUT_S)
        if self._implicit_dirty and self._calibration is not None:
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
    def privacy_restored(self) -> bool:
        """Privacy mode is on because it was on when the app last ran
        (``privacy.remember_privacy_mode``) and has not been changed since."""
        return self._privacy_restored and self._privacy

    @property
    def preview_enabled(self) -> bool:
        """At least one consumer wants preview frames (see :meth:`set_preview`)."""
        return bool(self._preview_owners)

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
        """The calibration in use, or the most recent one when none fits the current
        setup (then it is not usable; see :attr:`calibration_reason`)."""
        return self._calibration

    def calibrations(self) -> list[CalibrationData]:
        """Every saved calibration profile, most recently used first."""
        return self._library.profiles

    def backend_info(self) -> tuple[str, str]:
        """``(name, feature_version)`` of the vision backend in use (or expected).

        ``("", "")`` when no backend is available.
        """
        info = self._current_backend()
        return info if info is not None else ("", "")

    def gaze_feature_indices(self) -> tuple[int, ...] | None:
        """Indices of the gaze-direction features of the backend in use (or expected).

        What ``calibration.evaluate(..., nonlinear=...)`` takes, so that a new
        calibration applies nonlinear terms only to where the user looks, not to
        where their head is. ``None`` when the backend declares none (or is unknown).
        """
        with self._backend_lock:
            built = self._built_backend
        if built is not None and built.generation == self._backend_generation:
            return built.gaze_indices
        info = self._current_backend()
        return _named_backend_gaze_indices(info[0]) if info is not None else None

    def camera_identity(self) -> tuple[str, tuple[int, int]]:
        """``(device, (width, height))`` of the camera in use.

        ``device`` is the camera device setting and the size is that of the frames
        analysed most recently (``(0, 0)`` before the first). A new calibration
        records both (``CalibrationData.camera`` and ``frame_size``) so that it is
        only used with this camera; :meth:`finish_calibration` fills them in when
        they are missing.
        """
        return (str(self._settings.camera.device), self._frame_size or (0, 0))

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
        """Privacy mode: the camera is fully released (its light goes off).

        The choice is remembered for the next start (``privacy.remember_privacy_mode``).
        """
        enabled = bool(enabled)
        # An explicit choice made during a calibration is what applies afterwards.
        self._privacy_before_calibration = False
        if enabled == self._privacy:
            return
        self._privacy = enabled
        self._privacy_restored = False
        log.info("Privacy mode %s", "on" if enabled else "off")
        _save_privacy_mode(enabled)
        self._update_state()

    def toggle_privacy(self) -> None:
        self.set_privacy(not self._privacy)

    def set_preview(self, enabled: bool, owner: object | None = None) -> None:
        """Deliver annotated camera frames through ``preview_frame`` (and sample faster).

        Several consumers (the preview window, the first-run wizard) can want
        frames at once; each passes itself as ``owner`` and frames flow while at
        least one of them has enabled the preview. Calls without an owner share
        one anonymous owner.
        """
        before = bool(self._preview_owners)
        if enabled:
            self._preview_owners.add(id(owner))
        else:
            self._preview_owners.discard(id(owner))
        active = bool(self._preview_owners)
        if active == before:
            return
        if self._worker is not None:
            self._worker.set_preview(active)
        if not active:
            with self._preview_lock:
                self._preview_pending = None
        self._update_rate(self._clock())

    def dismiss_curtain(self) -> None:
        """The user dismissed the privacy curtain: monitor switching resumes.

        The shoulder guard stays active; it shows the curtain again only after the
        second face has left and come back.
        """
        if self._curtain:
            log.info("Privacy curtain dismissed by the user")
            self._show_curtain(False)

    def begin_calibration(self) -> None:
        """Enter ``CALIBRATING``: the camera runs at the calibration rate and every
        observation is forwarded through ``observation``.

        An explicit calibration request needs the camera, so privacy mode is turned
        off meanwhile; like a user pause, it applies again once the calibration is
        finished or cancelled (unless it was changed during the calibration).
        """
        if self._calibrating:
            return
        self._privacy_before_calibration = self._privacy
        if self._privacy:
            log.info("Privacy mode turned off to calibrate")
            self._privacy = False
        self._calibrating = True
        log.info("Calibration started")
        self._update_state()

    def finish_calibration(self, data: CalibrationData | None) -> None:
        """Leave ``CALIBRATING``. ``data`` is saved and used; ``None`` means cancelled.

        ``data.camera`` and ``data.frame_size`` are filled in from
        :meth:`camera_identity` when they are unknown. The calibration is stored
        as the profile of the current setup; profiles of other setups are kept.
        """
        if data is not None:
            data = self._with_camera_identity(data)
            self._use_profile(data, save=False)
            self._save_calibration(data, quiet=False)
            self._model = None  # force a fresh validation below
            self._validate_calibration(announce=False)
            self._reset_tracking()
            if self._model is None:
                log.warning("The new calibration is not usable: %s", self._calibration_reason)
            else:
                log.info("Calibration saved (%s)", data.grade or "ungraded")
        elif self._calibrating:
            log.info("Calibration cancelled")
            # The camera may have changed its frame size meanwhile (not judged
            # during the calibration).
            self._validate_calibration(announce=True)
        if self._privacy_before_calibration and not self._privacy:
            log.info("Privacy mode on again after the calibration")
            self._privacy = True
        self._privacy_before_calibration = False
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
        self._pane_decider.set_config(PaneConfig.from_settings(new.panes, new.switching))
        if old.panes != new.panes or old.switching.enabled != new.switching.enabled:
            self._reconfigure_panes(providers_changed=_pane_providers(old) != _pane_providers(new))
        self._filter.set_smoothing(new.switching.smoothing)
        self._presence.set_config(PresenceConfig.from_settings(new.presence))
        self._guard.set_config(GuardConfig.from_settings(new.privacy))
        self._policy.set_profile(new.performance.profile)
        if self._learner.set_max_samples(new.learning.max_samples):
            # The model still carries the influence of the dropped samples.
            self._refit_after_trim()

        backend_changed = old.general.backend != new.general.backend
        source_changed = old.camera != new.camera or _camera_fps(old) != _camera_fps(new)
        if backend_changed:
            self._expected_cache = None
            # Unknown until the worker has built the new backend (which may even
            # have the same identity as the old one); the settings decide meanwhile.
            self._reported_backend = None
        if source_changed:
            self._frame_size = None  # known again with the first frame of the new source
        worker = self._worker
        if worker is not None:
            worker.set_max_faces(self._max_faces())
            self._apply_motion_gate()
            if source_changed or backend_changed:
                worker.reconfigure(
                    source_factory=_source_factory(new) if source_changed else None,
                    backend_factory=self._make_backend_factory(new) if backend_changed else None,
                )
        if old.hotkeys != new.hotkeys and self._hotkeys is not None:
            # A deliberate change: report every hotkey that does not work now,
            # also one that was announced before.
            self._announced_hotkey_failures.clear()
            self._register_hotkeys()

        if self._started and not self._closed:
            now = self._clock()
            if not new.privacy.shoulder_guard:
                self._update_guard(now, None)  # a disabled guard clears at once
            if old.privacy != new.privacy:
                self._next_yield_check = now
            if backend_changed or source_changed:
                # Another backend or camera may need another calibration profile.
                self._validate_calibration(announce=True)
            self._update_presence(now)
            self._update_state()
            self._update_rate(now, force=True)
            self._update_timer()
            self._check_accessibility()
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
        mode_commands: dict[str, Callable[[], None]] = {
            "pause": self.pause,
            "resume": self.resume,
            "toggle": self.toggle_pause,
            "privacy-on": functools.partial(self.set_privacy, True),
            "privacy-off": functools.partial(self.set_privacy, False),
            "privacy-toggle": self.toggle_privacy,
        }
        change = mode_commands.get(cmd)
        if change is not None:
            before = (self._paused, self._privacy)
            change()
            self._confirm_mode_change(before)
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
                "profiles": len(self._library),
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
            "hotkeys": self._hotkey_status(),
            "panes": self._pane_status(),
        }

    def _pane_status(self) -> dict[str, Any]:
        """Split-pane focus at a glance: never titles, commands or paths."""
        snapshot = self._pane_snapshot
        window = self._pane_window
        sigma = self._pane_sigma_used
        if sigma is None and window is not None and window.rect is not None:
            sigma = self._pane_sigma(window.rect)
        worker = self._pane_worker
        fallback = (FALLBACK_GAZE_ERROR_PX, FALLBACK_GAZE_ERROR_PX)
        return {
            "enabled": bool(self._settings.panes.enabled),
            "supported": self._panes_supported(),
            "provider": snapshot.provider if snapshot is not None else None,
            "panes": len(snapshot.panes) if snapshot is not None else 0,
            "eligible": eligible_panes(snapshot, sigma or fallback, self._pane_decider.config),
            "switches": self._pane_switches,
            "sigma_px": None if sigma is None else [round(sigma[0], 1), round(sigma[1], 1)],
            "failed_providers": sorted(worker.disabled_providers) if worker is not None else [],
        }

    def _hotkey_status(self) -> dict[str, Any]:
        """The global hotkeys as the OS took them: ``{"registered": [names],
        "errors": {name: why it is not registered}}``.

        Only this process knows: ``eye-tracker doctor`` reads it through
        ``status`` (it cannot register anything itself without taking the
        combinations away from the running app). Empty while hotkeys are off or
        unsupported here, where nothing is registered on purpose.
        """
        manager = self._hotkeys
        hk = self._settings.hotkeys
        if manager is None or not hk.enabled or not manager.supported:
            return {"registered": [], "errors": {}}
        configured = [name for name in HOTKEY_ACTIONS if str(getattr(hk, name, "") or "").strip()]
        if self._hotkeys_suspended:
            # Released on purpose; they come back when the settings dialog is done.
            released = "released while a shortcut is being recorded in Settings → Hotkeys"
            return {"registered": [], "errors": dict.fromkeys(configured, released)}
        try:
            registered = sorted(str(name) for name in manager.registered)
        except Exception:
            log.debug("Reading the registered hotkeys failed", exc_info=True)
            registered = []
        errors: dict[str, str] = {}
        for name in configured:
            if name in registered:
                continue
            reason = self._hotkey_error(manager, name)
            if reason:
                errors[name] = reason
        status: dict[str, Any] = {"registered": registered, "errors": errors}
        note = getattr(manager, "note", None)
        if isinstance(note, str) and note.strip():
            # A limitation the running manager found (another program grabbing
            # the Super key on its own): doctor shows it.
            status["note"] = note.strip()
        return status

    # ================================================================ housekeeping
    def tick(self) -> None:
        """One housekeeping step (normally driven by the internal timer)."""
        if not self._started or self._closed:
            return
        now = self._clock()
        self._poll_input(now)
        self._settle_last_switch(now)
        if now >= self._next_window_poll:
            self._next_window_poll = now + WINDOW_POLL_S
            self._record_foreground_window(now)
        if now >= self._next_lock_check:
            self._next_lock_check = now + LOCK_POLL_S
            self._check_session_lock(now)
        if now >= self._next_yield_check:
            self._next_yield_check = now + YIELD_POLL_S
            self._check_camera_yield(now)
        self._sync_backend_info()
        self._recheck_camera_error(now)
        self._update_presence(now)
        self._update_state()
        self._deliver_calibration_notice(now)
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
        # A stale position (Wayland) says nothing about where the user put the
        # pointer: no remembered positions, learning labels or undone switches.
        if self._input.manual_move and self._cursor_reliable:
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
            self._drift.record_switch(True)
        elif monitor is not None and source is not None and monitor.index == source:
            # The user dragged the pointer straight back: that switch was wrong.
            self._last_switch = None
            self._drift.record_wrong_switch()
            log.debug("Automatic switch undone by the user")
            self._check_drift(now)

    def _settle_last_switch(self, now: float) -> None:
        """A switch the user left alone for WRONG_SWITCH_S was a correct one.

        Recording those as well keeps the drift statistics honest: with adaptive
        learning off, undone switches would otherwise be the only events.
        """
        last = self._last_switch
        if last is not None and now - last[0] > WRONG_SWITCH_S:
            self._last_switch = None
            self._drift.record_switch(True)

    def _record_foreground_window(self, now: float) -> None:
        if self._state is not TrackingState.TRACKING:
            return
        remember = self._settings.switching.focus_window
        panes = self._panes_wanted()
        if not (remember or panes):
            return
        ref = self._platform_call("foreground_window")
        if not isinstance(ref, WindowRef) or ref.rect is None or not self._monitors:
            if panes:
                self._forget_pane_window()
            return
        if remember:
            cx, cy = ref.rect.center
            monitor = (
                monitor_at(self._monitors, cx, cy) or nearest_monitor(self._monitors, cx, cy)[0]
            )
            self._memory.record_window(monitor.index, ref)
        if panes:
            self._follow_pane_window(ref, now)

    # ---------------------------------------------------------- split panes
    def _panes_supported(self) -> bool:
        """The platform's ``panes`` capability (asked once)."""
        if self._pane_capable is None:
            caps = self._platform_call("capabilities", default={})
            self._pane_capable = bool(isinstance(caps, dict) and caps.get("panes", False))
        return self._pane_capable

    def _panes_wanted(self) -> bool:
        s = self._settings
        return bool(s.panes.enabled and s.switching.enabled and self._panes_supported())

    def _default_pane_registry(self, settings: Settings) -> PaneRegistry:
        return PaneRegistry(default_providers(settings.panes, client_rect=self._pane_client_rect))

    def _pane_client_rect(self, ref: WindowRef) -> Rect | None:  # pane worker thread
        rect = self._platform_call("window_client_rect", ref)
        return rect if isinstance(rect, Rect) else None

    def _ensure_pane_worker(self) -> PaneWorker | None:
        if self._pane_worker is None and self._panes_wanted() and not self._closed:
            try:
                worker = self._pane_worker_factory(self._pane_registry_factory(self._settings))
                worker.detected.connect(self._on_pane_detected)
                worker.focused.connect(self._on_pane_focused)
                worker.start()
            except Exception:
                log.warning("Split-pane focus could not start", exc_info=True)
                return None
            self._pane_worker = worker
            log.info("Split-pane focus is on (experimental)")
        return self._pane_worker

    def _stop_pane_worker(self) -> None:
        worker, self._pane_worker = self._pane_worker, None
        if worker is None:
            return
        try:
            worker.stop(PANE_JOIN_TIMEOUT_S)
        except Exception:
            log.debug("Stopping the pane worker failed", exc_info=True)

    def _reconfigure_panes(self, *, providers_changed: bool) -> None:
        """Split-pane settings changed: start, stop or re-provision the worker."""
        if not self._panes_wanted():
            if self._pane_worker is not None:
                log.info("Split-pane focus is off")
            self._stop_pane_worker()
            self._forget_pane_window()
            self._pane_decider.reset()
            return
        worker = self._pane_worker
        if worker is not None and providers_changed:
            worker.set_registry(self._pane_registry_factory(self._settings))
            self._pane_snapshot = None
            self._pane_requested_at = -math.inf

    def _forget_pane_window(self) -> None:
        self._pane_window = None
        self._pane_app = None
        self._pane_snapshot = None
        self._pane_requested_at = -math.inf

    def _follow_pane_window(self, ref: WindowRef, now: float) -> None:
        """Track the focused window and ask for its panes when useful.

        A new window is asked about at once; the window being followed again
        every :data:`PANE_REFRESH_S` while the gaze is inside it. Windows of
        denied applications (see ``panes.registry``) are never asked about.
        """
        current = self._pane_window
        if current is None or not self._platform_call("same_window", current, ref, default=False):
            self._pane_window = ref
            self._pane_snapshot = None
            self._pane_requested_at = -math.inf
            app = self._platform_call("window_app", ref)
            self._pane_app = app if isinstance(app, AppIdentity) else None
            fresh = True
        else:
            # Keep the handle object the snapshot belongs to; only the frame moves.
            if ref.rect != current.rect:
                self._pane_window = dataclasses.replace(current, rect=ref.rect)
            fresh = False
        app = self._pane_app
        if app is None or is_denied(app):
            return
        if not fresh:
            gaze = self._last_gaze
            rect = ref.rect
            inside = gaze is not None and rect is not None and rect.contains(*gaze)
            if not inside or now - self._pane_requested_at < PANE_REFRESH_S:
                return
        worker = self._ensure_pane_worker()
        window = self._pane_window
        if worker is None or window is None:
            return
        self._pane_requested_at = now
        worker.request_detect(window, app)

    def _on_pane_detected(self, result: DetectResult) -> None:
        window = self._pane_window
        if self._closed or window is None or not _same_handle(result.window_handle, window.handle):
            return  # the user moved on to another window meanwhile
        self._pane_snapshot = result.snapshot

    def _on_pane_focused(self, result: FocusResult) -> None:
        if self._closed:
            return
        if not result.ok:
            log.debug("Could not focus pane %s", result.pane_id)
            return
        self._pane_switches += 1
        snapshot = self._pane_snapshot
        if snapshot is not None and _same_handle(snapshot.window_handle, result.window_handle):
            self._pane_snapshot = snapshot.with_focus(result.pane_id)
        self._pane_requested_at = -math.inf  # confirm the new layout at the next poll

    def _pane_sigma(self, rect: Rect) -> tuple[float, float] | None:
        """The calibration's gaze error on the monitor showing ``rect`` (cached)."""
        cal = self._calibration
        if cal is None or not self._monitors:
            return None
        cx, cy = rect.center
        monitor = monitor_at(self._monitors, cx, cy) or nearest_monitor(self._monitors, cx, cy)[0]
        cached = self._pane_sigma_cache
        if (
            cached is not None
            and cached[0] is cal
            and cached[1] == monitor.index
            and cached[2] == monitor.rect
        ):
            return cached[3]
        sigma = axis_error(cal, monitor)
        self._pane_sigma_cache = (cal, monitor.index, monitor.rect, sigma)
        return sigma

    def _pane_step(
        self,
        now: float,
        gaze: tuple[float, float] | None,
        decision: Decision,
        enabled: bool,
    ) -> PaneDecision:
        """Move the keyboard focus between split panes (only while the monitor
        decider is content). Never moves the cursor nor activates a window."""
        s = self._settings.panes
        worker = self._pane_worker
        window = self._pane_window
        active = (
            enabled
            and s.enabled
            and worker is not None
            and window is not None
            and decision.target is None
            and decision.reason == "same"
            and now - self._decider.last_switch_time >= s.after_monitor_switch_ms / 1000.0
        )
        rect = window.rect if window is not None else None
        sigma = self._pane_sigma(rect) if active and rect is not None else None
        if active:
            self._pane_sigma_used = sigma
        pane = self._pane_decider.update(
            now,
            gaze,
            rect,
            self._pane_snapshot,
            sigma,
            self._input.last_mouse_activity,
            self._input.last_key_activity,
            enabled=active,
        )
        if pane.target is not None and worker is not None and window is not None:
            worker.request_focus(window, pane.target)
        return pane

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
        self._away_displays_off = False
        self._hide_countdown()
        # Pointer moves refused around the lock (the lock screen owns the input
        # desktop until it is noticed, at most LOCK_POLL_S) say nothing about the
        # unlocked desktop: a backoff armed then must not outlive the lock.
        self._warp_refusals = 0
        self._warp_suspensions = 0
        self._switching_suspended_until = -math.inf
        if self._guard_locked:
            # The user just unlocked a lock the shoulder guard caused. If the
            # onlooker is still there, cover the screens rather than lock again.
            self._guard_locked = False
            self._guard_relock_until = now + GUARD_RELOCK_GRACE_S

    def _check_camera_yield(self, now: float, *, force: bool = False) -> None:
        """Decide whether another app needs the camera (``_yield_reason``).

        Skipped while the camera is off anyway or explicitly wanted, unless
        ``force`` (the camera is about to be switched back on).
        """
        if not force and self._state in (
            TrackingState.PRIVACY,
            TrackingState.LOCKED,
            TrackingState.PAUSED,
            TrackingState.CALIBRATING,
        ):
            return
        self._yield_checked_at = now
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
        # A frame the worker captured just before the camera was switched off
        # (privacy mode, pause, lock) arrives after the state change; shown then,
        # it would stay on screen under "camera off". Dropped, like the late
        # observations in _handle_observation.
        if (
            frame is not None
            and self._preview_owners
            and not self._closed
            and self._state.camera_active
        ):
            self.preview_frame.emit(frame)

    def _handle_worker_stats(self, stats: WorkerStats) -> None:
        if self._closed:
            return
        self._worker_stats = stats
        self._sync_backend_info()
        if self._state.camera_active:
            self._judge_camera(stats)
        self._update_state()

    def _judge_camera(self, stats: WorkerStats) -> None:
        """Set the camera error from the worker's statistics.

        A backend that keeps failing counts too: the camera would look healthy
        while nothing is analysed.
        """
        error = bool(stats.last_error) and (
            not stats.camera_open or bool(stats.extra.get("backend_failing"))
        )
        self._set_camera_error(error, stats.last_error)

    def _recheck_camera_error(self, now: float) -> None:
        """Read the worker's error back when the camera stays silent after switching on.

        The worker publishes an error only when its message changes, so a camera
        that fails the same way as before a pause, lock or yield would otherwise
        leave the state at TRACKING with no frames and walk-away detection frozen.
        """
        since = self._camera_on_since
        if (
            since is None
            or self._camera_error
            or self._last_obs_time is not None
            or not self._state.camera_active
            or now - since < CAMERA_ERROR_RECHECK_S
        ):
            return
        self._judge_camera(self._safe_worker_stats())

    def _set_camera_error(self, error: bool, message: str | None) -> None:
        if error == self._camera_error:
            if error:
                self._camera_error_message = message
            return
        self._camera_error = error
        self._camera_error_message = message if error else None
        if not error:
            log.info("Camera working again")
            return
        log.warning("Camera problem: %s", message)
        permissions = self._platform_call("permissions", default=None)
        if isinstance(permissions, dict) and permissions.get("camera") is False:
            self._notify(
                "Camera access blocked",
                "The system privacy settings do not allow Eye Tracker to use the camera. "
                "Allow camera access there; tracking starts by itself afterwards.",
            )
            self.permission_needed.emit("camera")
        else:
            self._notify("Camera unavailable", message or "The camera could not be opened.")

    # ============================================================== observations
    def _handle_observation(self, obs: Observation) -> None:
        if self._closed or not self._started or not self._state.camera_active:
            return  # a frame analysed just before the camera was switched off
        now = self._clock()
        previous = self._last_obs_time
        self._last_obs_time = now
        if self._camera_error:
            self._set_camera_error(False, None)  # frames arrive, so the camera works
            self._update_state()
        face = self._face_verdict(obs, now)
        self._note_frame_size(obs)
        self.observation.emit(obs)
        if self._state is TrackingState.CALIBRATING:
            self._last_face = face
            # The guard sees nothing meanwhile: no multi-face run may span the
            # calibration and trigger on the first frame after it.
            self._update_guard(now, None)
            self._update_rate(now)
            return
        if previous is None or now - previous > max(
            STALE_OBSERVATION_S, 2.5 * (self._interval or 0.0)
        ):
            self._update_guard(now, None)  # the stream was interrupted: nothing is "continuous"
        self._update_guard(now, None if obs.blind else int(obs.face_count), obs.face_box)
        # Every detected face counts as the user for walk-away detection, also
        # one the shoulder guard judges to be the onlooker's: faces are compared
        # by their boxes, not recognised, and a wrong "that is not the user"
        # would lock out a user sitting at their desk (see engine/guard.py).
        self._last_face = face
        self._update_presence(now)
        self._update_state()
        self._last_decision = None
        self._last_pane_decision = None
        if self._state is TrackingState.TRACKING:
            self._track(obs, now)
            self._update_state()
        self._update_rate(now)
        self._update_timer()
        if self._trace is not None:
            self._write_trace(obs, now)

    def _face_verdict(self, obs: Observation, now: float) -> bool | None:
        """The observation's verdict for walk-away detection: face, no face, or
        ``None`` when the camera cannot tell.

        A blind frame (covered lens, closed shutter, dark room) cannot tell. If the
        blindness began while the user was demonstrably there (face just seen, or
        recent input), they covered the camera: presence pauses, for at most
        BLIND_FREEZE_MAX_S without input. Blindness that begins after the face was
        already gone (the user left, then the lights went out) keeps counting as
        "nobody here", so the walk-away lock still happens.
        """
        if not obs.blind:
            if self._blind_since is not None:
                log.info("The camera sees again")
                self._end_blind_episode()
            present = bool(obs.face_present)
            if present:
                self._face_seen_at = now
            return present
        if self._blind_since is None:
            self._blind_since = now
            seen = self._face_seen_at
            face_recent = seen is not None and now - seen <= BLIND_ONSET_S
            input_recent = now - self._input.last_any_activity <= BLIND_ONSET_S
            self._blind_freeze = face_recent or input_recent
            log.info(
                "The camera sees nothing (covered or dark)%s",
                "; walk-away detection paused" if self._blind_freeze else "",
            )
            if self._blind_freeze and now - self._blind_notice_at >= BLIND_NOTICE_COOLDOWN_S:
                self._blind_notice_at = now
                self._notify(
                    "Camera appears covered",
                    "The camera sees nothing, so walk-away detection is paused. Uncover it, "
                    "or turn on privacy mode to switch the camera off.",
                )
        if not self._blind_freeze:
            return False
        evidence = max(self._blind_since, self._input.last_any_activity)
        if now - evidence > BLIND_FREEZE_MAX_S:
            return False
        return None

    def _end_blind_episode(self) -> None:
        self._blind_since = None
        self._blind_freeze = False

    def _note_frame_size(self, obs: Observation) -> None:
        """Remember the camera's frame size; a new one can need another calibration."""
        size = _known_size(obs.frame_size)
        if size is None or size == self._frame_size:
            return
        self._frame_size = size
        log.debug("Camera frames are %dx%d", *size)
        # During a calibration the new profile is judged when it is finished.
        if self._calibration is not None and self._state is not TrackingState.CALIBRATING:
            self._validate_calibration(announce=True)

    def _write_trace(self, obs: Observation, now: float) -> None:
        trace = self._trace
        if trace is None:
            return
        cursor = self._cursor_pos()
        decision = self._last_decision
        pane = self._last_pane_decision
        snapshot = self._pane_snapshot
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
                # Split panes: ids only (tmux "%3", WezTerm numbers), never titles.
                "pane_cand": _trace_id(pane.candidate) if pane else None,
                "pane_reason": pane.reason if pane else None,
                "pane_prog": trace.number(pane.progress, 2) if pane else None,
                "pane_target": _trace_id(pane.target.id) if pane and pane.target else None,
                "n_panes": len(snapshot.panes) if snapshot is not None else 0,
            }
        )

    def _track(self, obs: Observation, now: float) -> None:
        gaze = self._estimate_gaze(obs, now)
        if self._model is None:
            return  # the calibration turned out to be unusable
        self._publish_gaze(gaze, now)
        if self._settings.learning.adaptive and not self._looking_away:
            self._learn(obs, now)
        cursor = self._cursor_pos()
        current = self._current_monitor(cursor)
        # No switching (and so no focus change) under the privacy curtain: focus
        # would move to a hidden window, and Esc would no longer reach the curtain.
        # Nor on the lock screen (tracking goes on there when pause_when_locked is
        # off): the desktop is not what the user sees, and the system refuses the
        # pointer moves, which would count towards the refused-warp backoff.
        enabled = (
            self._settings.switching.enabled
            and not self._curtain
            and not self._session_locked
            and now >= self._switching_suspended_until
        )
        decision = self._decider.update(
            now,
            gaze,
            current,
            self._input.last_mouse_activity,
            self._input.last_key_activity,
            enabled=enabled,
            looking_away=self._looking_away,
        )
        self._last_decision = decision
        if decision.target is not None:
            self._switch_to(decision.target, gaze, cursor, current, now)
        pane = self._pane_step(now, gaze, decision, enabled)
        self._last_pane_decision = pane
        self._pending = decision.pending or pane.pending

    def _current_monitor(self, cursor: tuple[int, int] | None) -> int | None:
        """Index of the monitor the pointer is on (``None`` if unknown)."""
        if not self._cursor_reliable:
            return self._assumed_monitor
        monitor = monitor_at(self._monitors, cursor[0], cursor[1]) if cursor else None
        return monitor.index if monitor is not None else None

    def _estimate_gaze(self, obs: Observation, now: float) -> tuple[float, float] | None:
        self._looking_away = False
        model = self._model
        if model is None:
            return None
        if obs.usable and obs.features is not None:
            try:
                away = model.looks_away(
                    obs.features, self.gaze_feature_indices(), monitors=self._monitors
                )
                raw = None if away else model.predict(obs.features)
            except (ValueError, RuntimeError) as exc:
                self._feature_mismatches += 1
                if self._feature_mismatches < FEATURE_MISMATCH_LIMIT:
                    # Most likely a frame the previous backend analysed.
                    log.debug("Gaze model rejected the camera features: %s", exc)
                    return None
                log.warning("Gaze model rejected the camera features: %s", exc)
                self._invalidate_calibration("the camera features no longer match the calibration")
                return None
            self._feature_mismatches = 0
            if raw is None:
                # The gaze direction points far off every monitor (phone, desk, a
                # person beside the screens). Forget the last point so a blink
                # right after does not bring it back, and restart the filter.
                self._looking_away = True
                self._last_gaze = None
                self._last_gaze_time = None
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
        current: int | None,
        now: float,
    ) -> None:
        monitor = next((m for m in self._monitors if m.index == target), None)
        if monitor is None:
            return
        sw = self._settings.switching
        if self._cursor_reliable and current is not None and cursor is not None:
            self._memory.record_cursor(current, cursor)

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
        if not self._move_cursor(x, y, now):
            # Nothing moved, so this was no switch: no focus change, no count. The
            # decider armed its cooldown when it fired, so it retries after that.
            return
        if sw.focus_window:
            if window is None:
                found = self._platform_call("window_at", x, y)
                window = found if isinstance(found, WindowRef) else None
            if window is not None:
                self._activate(monitor, window)

        self._decider.notify_switched(now, target)
        self._pane_decider.reset()  # another monitor: another pane context
        self._assumed_monitor = target
        if self._last_switch is not None:
            self._drift.record_switch(True)  # the previous switch was kept until now
        # Without a reliable pointer position an undone switch cannot be seen.
        self._last_switch = (now, current, target) if self._cursor_reliable else None
        self._switch_count += 1
        log.debug("Switched to monitor %d (cursor %d, %d)", target, x, y)
        self.switched.emit(target)

    def _activate(self, monitor: Monitor, window: WindowRef) -> None:
        if self._platform_call("activate_window", window, default=False):
            self._memory.record_window(monitor.index, window)
            self._activation_failures = 0
            return
        log.debug("Could not activate the window on monitor %d", monitor.index)
        self._activation_failures += 1
        if self._activation_failures >= ACTIVATION_FAILURE_LIMIT:
            # Activation also fails for ordinary reasons (Windows' foreground
            # lock, a window that closed); only a missing permission is reported.
            self._activation_failures = 0
            self._check_accessibility()

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

    def _move_cursor(self, x: int, y: int, now: float) -> bool:
        """Warp the pointer; False if the system refused.

        Refusals in a row pause switching (backing off) instead of retrying after
        every cooldown, and the user is told once.
        """
        try:
            result = self._cursor.set_pos(int(x), int(y))
        except Exception:
            log.warning("Moving the cursor failed", exc_info=True)
            result = False
        if result is None or bool(result):
            self._warp_refusals = 0
            self._warp_suspensions = 0
            self._switching_suspended_until = -math.inf
            return True
        # Windows refuses pointer moves as soon as the secure desktop (lock screen,
        # UAC prompt, Ctrl+Alt+Del) has the input, up to LOCK_POLL_S before the
        # regular poll notices. Ask now: a refusal caused by a lock says nothing
        # about the desktop, so it neither counts towards the backoff nor tells
        # the user that the cursor cannot be moved.
        self._next_lock_check = now + LOCK_POLL_S
        self._check_session_lock(now)
        if self._session_locked:
            log.info("The system refused to move the pointer: the session is locked")
            return False
        self._warp_refusals += 1
        log.info("The system refused to move the pointer (%d in a row)", self._warp_refusals)
        if self._warp_refusals >= WARP_REFUSAL_LIMIT:
            self._warp_suspensions += 1
            delay = min(
                WARP_BACKOFF_MAX_S, WARP_BACKOFF_S * 2 ** min(self._warp_suspensions - 1, 16)
            )
            self._switching_suspended_until = now + delay
            log.warning("Pointer moves keep failing; switching paused for %.0f s", delay)
            if not self._cursor_warning_shown:
                self._cursor_warning_shown = True
                self._notify(
                    "Cannot move the cursor",
                    "The system keeps refusing to move the mouse pointer, so switching "
                    "monitors is paused for now. On Wayland, install ydotool (see the "
                    "documentation).",
                )
        return False

    # ------------------------------------------------------------------ learning
    def _learn(self, obs: Observation, now: float) -> None:
        cal, model = self._calibration, self._model
        if cal is None or model is None or self._learner.max_samples <= 0:
            return
        sample = self._learner.on_observation(obs, now, self._monitor_index_at)
        if sample is None:
            return
        predicted = self._predict_point(model, sample.features)
        predicted_monitor = (
            nearest_monitor(self._monitors, *predicted)[0].index
            if predicted is not None and self._monitors
            else None
        )
        # The comparison is drift evidence either way, also for a rejected label.
        self._drift.record(predicted_monitor, sample.monitor_index)
        self._check_drift(now)
        if predicted is not None and not plausible_label(predicted, sample, self._monitors):
            # The pointer was parked while the user read elsewhere: not a label.
            self._learner.discard_last()
            log.debug("Ignored an implausible learning sample at %.0f, %.0f", sample.x, sample.y)
            return
        self._implicit_dirty = True
        if self._learner.should_refit():
            self._refit(cal, model)

    def _refit(self, cal: CalibrationData, model: GazeModel) -> None:
        self._learner.mark_refit()
        learned = self._learner.samples
        try:
            refined = refit_model(cal.samples, learned, model, monitors=cal.monitors)
        except (ValueError, np.linalg.LinAlgError) as exc:
            log.warning("Could not refine the gaze model: %s", exc)
            return
        cal.model = refined
        cal.implicit_samples = learned
        self._model = refined
        self._save_calibration(cal, quiet=True)
        log.info("Gaze model refined with %d learned samples", len(learned))

    def _refit_after_trim(self) -> None:
        """Refit the calibration after learned samples were dropped (smaller capacity)."""
        cal = self._calibration
        if cal is None or not cal.model.is_fitted:
            return
        learned = self._learner.samples
        try:
            refined = refit_model(cal.samples, learned, cal.model, monitors=cal.monitors)
        except (ValueError, np.linalg.LinAlgError) as exc:
            log.warning("Could not refit the gaze model: %s", exc)
            return
        cal.model = refined
        cal.implicit_samples = learned
        if self._model is not None:
            self._model = refined
        self._save_calibration(cal, quiet=True)
        log.info("Gaze model refitted with the %d learned samples kept", len(learned))

    def _check_drift(self, now: float) -> None:
        if self._settings.learning.drift_alerts and self._drift.should_alert(
            now, DRIFT_ALERT_COOLDOWN_S
        ):
            self._notify(
                "Accuracy dropped",
                "Accuracy dropped — recalibrate? Choose Calibrate… in the tray menu.",
            )

    @staticmethod
    def _predict_point(model: GazeModel, features: np.ndarray) -> tuple[float, float] | None:
        try:
            px, py = (float(v) for v in model.predict(features))
        except (ValueError, RuntimeError):
            return None
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        return (px, py)

    def _monitor_index_at(self, x: float, y: float) -> int | None:
        monitor = monitor_at(self._monitors, x, y)
        return monitor.index if monitor is not None else None

    # ================================================================== presence
    def _update_presence(self, now: float) -> None:
        face = self._presence_face(now)
        if (
            face is None
            and self._presence.state is PresenceState.AWAY
            and self._state.camera_active
            and self._state is not TrackingState.CALIBRATING
        ):
            # The camera should be watching but cannot tell (it failed or stalled
            # while the user was away). Input must still be able to prove their
            # return; without input AWAY simply stays, as with "no face".
            face = False
        events = self._presence.update(now, face, self._input.seconds_since_any(now))
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
                self._away_displays_off = False

    def _perform_away_action(self) -> None:
        action = self._settings.presence.action
        log.info("User away; action: %s", action)
        if action == "none":
            return
        setup_pending = not self._settings.general.first_run_done
        effective = effective_away_action(self._settings)
        if effective != action:
            log.info("First-run setup not finished: notifying instead of %s", action)
            action = effective
        if action == "notify":
            message = "Nobody has been at the computer for a while."
            if setup_pending:
                message += " Finish the setup to choose what happens when you walk away."
            self._notify("Are you still there?", message, force=True)
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
        # Set before the lock is requested: _finish_lock may run synchronously
        # inside _request_lock, and a failed lock corrects _displays_off_only.
        self._away_displays_off = display_off and displays_ok
        self._displays_off_only = self._away_displays_off and not lock
        if lock:
            self._request_lock("away")  # the result arrives in _finish_lock

    def _hide_countdown(self) -> None:
        if self._warning_shown:
            self._warning_shown = False
            self.away_cancelled.emit()

    # ===================================================================== guard
    def _update_guard(
        self,
        now: float,
        face_count: int | None,
        face_box: tuple[float, float, float, float] | None = None,
    ) -> None:
        # Input tells the guard that someone works at the computer: while it
        # judges the user gone, that person's face becomes the user's again.
        result = self._guard.update(now, face_count, face_box, self._input.last_any_activity)
        if result == "trigger":
            self._on_guard_trigger(now)
        elif result == "clear":
            # Only the curtain comes down. "Clear" means fewer than two faces for
            # a moment (GuardConfig.clear_s), which a colleague who turns to the
            # user or looks down at notes already causes; ending the relock grace
            # here would lock the user out again as soon as they look back.
            self._show_curtain(False)

    def _on_guard_trigger(self, now: float) -> None:
        action = self._settings.privacy.guard_action
        log.info("Shoulder guard: second face detected; action %s", action)
        if action == "lock":
            if self._session_locked:
                return
            if now < self._guard_relock_until:
                log.info("Shoulder guard: showing the curtain instead of locking again")
                # Most likely the same visit that followed the unlock: keep the
                # grace going while the second face keeps coming back.
                self._guard_relock_until = now + GUARD_RELOCK_GRACE_S
                self._show_curtain(True)
                return
            # If locking fails, _finish_lock shows the privacy curtain instead.
            self._request_lock("guard")
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

    # ============================================================== screen lock
    def _request_lock(self, purpose: str) -> None:
        """Lock the session without blocking the GUI thread.

        ``lock_screen`` may take seconds (a D-Bus call to a busy screensaver,
        ``loginctl``, AppleScript) and is safe on any thread, so it runs through
        ``lock_runner`` and the result comes back to :meth:`_finish_lock` on the
        main thread. ``purpose`` (``"away"`` or ``"guard"``) decides what a
        failure does; a request while another is in flight joins it.
        """
        first = not self._lock_purposes
        self._lock_purposes.add(purpose)
        if not first:
            return
        log.info("Locking the screen (%s)", purpose)
        try:
            self._lock_runner(self._lock_job)
        except Exception:
            log.warning("Could not start locking the screen", exc_info=True)
            self._finish_lock(False)

    def _lock_job(self) -> None:  # lock thread (or the main thread, see lock_runner)
        try:
            ok = bool(self._platform.lock_screen())
        except Exception:
            log.debug("lock_screen failed", exc_info=True)
            ok = False
        if self._on_main_thread():
            self._finish_lock(ok)
            return
        if self._closed:
            return
        with contextlib.suppress(RuntimeError):  # deleted after a shutdown timed out
            self._lock_done.emit(ok)

    def _run_lock_thread(self, job: Callable[[], None]) -> None:
        """Default ``lock_runner``: a short-lived daemon thread."""
        thread = threading.Thread(target=job, name="eye-tracker-lock", daemon=True)
        self._lock_thread = thread
        thread.start()

    def _finish_lock(self, ok: bool) -> None:
        """The screen lock requested by :meth:`_request_lock` has finished."""
        purposes, self._lock_purposes = self._lock_purposes, set()
        if self._closed or not purposes:
            return
        if ok:
            if "guard" in purposes:
                self._guard_locked = True
            # Notice the lock soon so the camera is released promptly.
            self._next_lock_check = min(self._next_lock_check, self._clock() + 0.5)
            return
        log.warning("Locking the screen failed or is not supported here")
        if "away" in purposes and self._away:
            # The displays may already be off; with the session unlocked they are
            # "off only" and are woken when the user comes back.
            self._displays_off_only = self._away_displays_off
        if "away" in purposes:
            self._notify(
                "Could not lock the screen",
                "This system does not allow it; choose another walk-away action.",
                force=True,
            )
        if "guard" in purposes and self._guard.active:
            # The onlooker is still there: cover the screens instead.
            log.warning("Showing the privacy curtain instead of the lock")
            self._show_curtain(True)

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
        if self._model is None and self._calibration_useful():
            return TrackingState.NEEDS_CALIBRATION
        return TrackingState.TRACKING

    def _calibration_useful(self) -> bool:
        """Whether a calibration would change anything: it only serves monitor
        switching, so not with one monitor or with switching turned off. Then
        presence, privacy mode and the shoulder guard are all there is, and the
        state must not ask for a calibration (warning badge, prompts)."""
        return len(self._monitors) >= 2 and self._settings.switching.enabled

    def _update_state(self) -> None:
        new = self._derive_state()
        if new is self._state:
            return
        if (
            self._started
            and self._state in _UNCHECKED_CAMERA_OFF
            and new.camera_active
            and new is not TrackingState.CALIBRATING
        ):
            # The camera is about to reopen, but whether another app uses it was
            # not checked while it was off: a call may have started meanwhile.
            now = self._clock()
            self._check_camera_yield(now, force=True)
            self._next_yield_check = now + YIELD_POLL_S
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
        if new.camera_active and (old is TrackingState.STARTING or not old.camera_active):
            self._camera_on_since = now
        if not new.camera_active:
            # Judged afresh once the camera is back on.
            self._camera_on_since = None
            self._camera_error = False
            self._last_obs_time = None
            self._last_face = None
            self._end_blind_episode()
            self._clear_guard()
            with self._preview_lock:
                self._preview_pending = None  # captured while the camera was on
        elif new in (TrackingState.CALIBRATING, TrackingState.CAMERA_ERROR):
            # No observations reach the guard in these states; a run in progress
            # must not survive them.
            self._update_guard(now, None)
        if not new.camera_active or new is TrackingState.CALIBRATING:
            # Time without a camera verdict never counts as absence.
            self._handle_presence_events(self._presence.update(now, None, None))
        if new is not TrackingState.TRACKING:
            self._reset_tracking()
        if TrackingState.CALIBRATING in (old, new):
            self._apply_motion_gate()
        if (old in _CAMERA_OFF or old is TrackingState.CALIBRATING) and (
            self._yield_checked_at != now
        ):
            self._next_yield_check = min(self._next_yield_check, now)
        if (
            old in _ABSENT_STATES
            and new not in _ABSENT_STATES
            and new is not TrackingState.CALIBRATING
        ):
            self._remind_calibration(now)
        self._set_background_activity(new not in _NAPPABLE_STATES)
        self._update_rate(now)
        self._update_timer()

    def _reset_tracking(self) -> None:
        self._decider.reset()
        self._pane_decider.reset()
        self._filter.reset()
        self._pending = False
        self._last_gaze = None
        self._last_gaze_time = None
        self._looking_away = False
        self._feature_mismatches = 0
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
            preview=bool(self._preview_owners),
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

    def _set_background_activity(self, active: bool) -> None:
        """Keep timers unthrottled while tracking (macOS App Nap); a no-op elsewhere."""
        if active == self._background_active:
            return
        self._background_active = active
        self._platform_call("set_background_activity", active)

    # =============================================================== calibration
    def _load_calibration(self) -> None:
        self._library = CalibrationLibrary.load(paths.calibration_file())
        self._use_profile(self._library.latest, save=False)
        if self._calibration is None:
            self._calibration_reason = "not calibrated yet"

    def _use_profile(self, data: CalibrationData | None, *, save: bool) -> None:
        """Make ``data`` the calibration in use, with its own learned samples.

        The samples learned for the previous profile are kept in that profile.
        ``save`` also marks ``data`` as the most recently used and writes the file.
        """
        if data is self._calibration:
            return
        self._flush_learned()
        self._calibration = data
        self._model = None
        self._learner.clear()
        if data is not None and data.implicit_samples:
            self._learner.load(data.implicit_samples)
        self._implicit_dirty = False
        self._drift.reset()
        self._feature_mismatches = 0
        changed = False
        if data is not None and len(self._learner.samples) < len(data.implicit_samples):
            # The learning capacity was lowered after this profile learned (while
            # another profile was in use, or in the settings file): load() kept
            # only what fits, but the stored model was fitted with all of them.
            changed = self._refit_trimmed_profile(data)
        if save and data is not None and self._library.mark_used(data):
            changed = True
        if changed:
            self._write_library(quiet=True)

    def _refit_trimmed_profile(self, data: CalibrationData) -> bool:
        """Refit ``data`` with the learned samples the learner kept.

        Otherwise the dropped samples would keep steering the gaze (with a
        capacity of 0, "no learned influence" would not hold) and stay in the
        file. Returns whether ``data`` changed (and should be saved); on failure
        the profile is left as it was.
        """
        kept = self._learner.samples
        if data.model.is_fitted:
            try:
                data.model = refit_model(data.samples, kept, data.model, monitors=data.monitors)
            except (ValueError, np.linalg.LinAlgError) as exc:
                log.warning(
                    "Could not refit the gaze model after trimming learned samples: %s", exc
                )
                return False
        data.implicit_samples = kept
        log.info(
            "Learned samples trimmed to the capacity of %d; gaze model refitted with %d",
            self._learner.max_samples,
            len(kept),
        )
        return True

    def _flush_learned(self) -> None:
        """Copy unsaved learned samples into the calibration they belong to."""
        if self._implicit_dirty and self._calibration is not None:
            self._calibration.implicit_samples = self._learner.samples

    def _with_camera_identity(self, data: CalibrationData) -> CalibrationData:
        """``data`` with the camera device and frame size filled in where unknown."""
        changes: dict[str, Any] = {}
        if not data.camera.strip():
            changes["camera"] = str(self._settings.camera.device)
        if _known_size(data.frame_size) is None and self._frame_size is not None:
            changes["frame_size"] = self._frame_size
        return dataclasses.replace(data, **changes) if changes else data

    def _save_calibration(self, data: CalibrationData, *, quiet: bool) -> bool:
        """Store ``data`` as the most recently used profile and write the file."""
        self._flush_learned()
        self._library.put(data)
        return self._write_library(quiet=quiet)

    def _write_library(self, *, quiet: bool) -> bool:
        try:
            self._library.save(paths.calibration_file())
        except (OSError, ValueError) as exc:
            log.error("Could not save the calibration: %s", exc)
            if not quiet:
                self._notify("Calibration not saved", str(exc), force=True)
            return False
        self._implicit_dirty = False
        return True

    def _validate_calibration(self, *, announce: bool) -> None:
        """Decide which calibration fits the backend, monitors and camera in use.

        The profile in use is kept while it fits; otherwise the best saved profile
        for the current setup is taken (and learned samples follow it). Without
        one the calibration is unusable and, if it was usable and ``announce``,
        ``calibration_required`` is emitted (now or once the user is back).
        """
        was_usable = self._model is not None
        if not self._monitors and self._calibration is not None:
            return  # nothing to compare with; keep the current verdict
        data, reason = self._match_profile()
        if data is not None:
            switched = data is not self._calibration
            if switched:
                log.info(
                    "Using the calibration of %s for this setup",
                    data.created_at or "an earlier session",
                )
                self._use_profile(data, save=True)
            if not was_usable or switched:
                log.info("Calibration is usable")
                self._reset_tracking()
            self._model = data.model
            self._calibration_reason = ""
            self._announce_pending = None
            self._interruptible_since = None
            return
        self._model = None
        self._calibration_reason = reason
        if was_usable:
            log.warning("Calibration no longer usable: %s", reason)
            if announce:
                self._announce_calibration(reason)

    def _match_profile(self) -> tuple[CalibrationData | None, str]:
        """The profile to use for the current setup, or ``(None, reason)``."""
        current = self._calibration
        info = self._current_backend()
        camera = str(self._settings.camera.device)
        frame_size = self._frame_size
        if info is not None:
            name, version = info
            reason = "not calibrated yet"
            if current is not None:
                ok, reason = current.is_compatible(
                    name, version, self._monitors, camera=camera, frame_size=frame_size
                )
                if ok:
                    return current, ""
            data, why = self._library.match(
                name, version, self._monitors, camera=camera, frame_size=frame_size
            )
            if data is not None:
                return data, ""
            return None, reason if current is not None else why
        # Backend unknown (none installed?): the layout is all that can be checked.
        signature = layout_signature(self._monitors)

        def fits(p: CalibrationData) -> bool:
            return p.model.is_fitted and p.layout_signature == signature

        if current is not None and fits(current):
            return current, ""
        other = next((p for p in self._library if fits(p)), None)
        if other is not None:
            return other, ""
        if current is None:
            return None, "not calibrated yet"
        if not current.model.is_fitted:
            return None, "the calibration has no fitted model"
        return None, "the monitor layout changed"

    def _invalidate_calibration(self, reason: str) -> None:
        was_usable = self._model is not None
        self._model = None
        self._calibration_reason = reason
        if was_usable:
            self._announce_calibration(reason)

    def _announce_calibration(self, reason: str) -> None:
        """Emit ``calibration_required`` now, or once the user can be interrupted."""
        if self._may_interrupt():
            self._emit_calibration_required(reason)
        else:
            log.debug("Calibration announcement deferred: %s", reason)
            self._announce_pending = reason
            self._interruptible_since = None

    def _remind_calibration(self, now: float) -> None:
        """The user is back (from away, a lock or privacy mode): remind them of a
        calibration that is still unusable, unless that was said recently.

        Also when there never was one: otherwise a setup that starts at login and
        was never calibrated would never be told why nothing switches.
        """
        if (
            self._model is not None
            or self._announce_pending is not None
            or now - self._last_announce < CALIBRATION_REMINDER_S
        ):
            return
        self._announce_pending = self._calibration_reason
        self._interruptible_since = None

    def _deliver_calibration_notice(self, now: float) -> None:
        """Deliver a deferred announcement once the user has been interruptible for
        ANNOUNCE_SETTLE_S with no monitor change pending."""
        if self._announce_pending is None:
            return
        if self._model is not None:
            self._announce_pending = None  # usable again: nothing to say
            self._interruptible_since = None
            return
        if not self._may_interrupt() or self._screen_timer.isActive():
            self._interruptible_since = None
            return
        if self._interruptible_since is None:
            self._interruptible_since = now
            return
        if now - self._interruptible_since >= ANNOUNCE_SETTLE_S:
            self._emit_calibration_required(self._calibration_reason or self._announce_pending)

    def _emit_calibration_required(self, reason: str) -> None:
        self._announce_pending = None
        self._interruptible_since = None
        self._last_announce = self._clock()
        self.calibration_required.emit(reason)

    def _may_interrupt(self) -> bool:
        """Whether a calibration prompt is appropriate right now.

        Not while the user is away or displays are off (monitors drop out during
        display sleep and come back unchanged), and not where a calibration would
        not change anything (one monitor, or switching turned off; a pending
        announcement is delivered once that changes).
        """
        return (
            self._calibration_useful()
            and not self._calibrating
            and not self._displays_off_only
            and self._state not in (TrackingState.AWAY, TrackingState.LOCKED, TrackingState.PRIVACY)
        )

    # =================================================================== backend
    def _make_backend_factory(self, settings: Settings) -> BackendFactory:
        """A backend factory for the worker that records what it built.

        Every factory gets a new generation number. The worker keeps reporting its
        previous backend until it has built the new one, and the new backend can
        even have the same identity (``auto`` falling back to ``lite`` again), so
        identities alone cannot tell a stale report from a fresh one.
        """
        self._backend_generation += 1
        generation = self._backend_generation
        create = _backend_factory(settings, self._max_faces())

        def factory() -> VisionBackend:  # worker thread
            backend = create()
            built = _BuiltBackend(
                generation,
                (str(backend.name), str(backend.feature_version)),
                _backend_gaze_indices(backend),
            )
            with self._backend_lock:
                self._built_backend = built
            return backend

        return factory

    def _safe_backend_info(self) -> tuple[str, str] | None:
        worker = self._worker
        if worker is None:
            return None
        with self._backend_lock:
            built = self._built_backend
        if built is not None:
            # Only the backend built by the latest factory counts.
            return built.info if built.generation == self._backend_generation else None
        if self._backend_generation > 1:
            return None  # a new backend was requested and is not built yet
        # The worker never used our factory (a test double): trust what it says.
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
        self._reported_backend = info
        self._feature_mismatches = 0
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

    # ============================================================ permissions
    def _check_accessibility(self) -> None:
        """Tell the user (once) when focus cannot follow the gaze for lack of the
        macOS Accessibility permission, or because it belongs to an older build."""
        sw = self._settings.switching
        if (
            self._accessibility_notice_shown
            or not (sw.enabled and sw.focus_window)
            or len(self._monitors) < 2
        ):
            return
        status = self._platform_call("accessibility_status")
        if status not in ("missing", "stale"):
            return
        self._accessibility_notice_shown = True
        if status == "stale":
            message = (
                "Keyboard focus cannot follow your gaze: the Accessibility permission "
                "belongs to a previous version of Eye Tracker. In System Settings › Privacy "
                "& Security › Accessibility, remove Eye Tracker with “−” and add it again."
            )
        else:
            message = (
                "Keyboard focus cannot follow your gaze until Eye Tracker may control "
                "your computer: System Settings › Privacy & Security › Accessibility."
            )
        self._notify("Accessibility access needed", message, force=True)
        self.permission_needed.emit("accessibility")

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
            # Turning them on again announces every failure afresh.
            self._announced_hotkey_failures.clear()
            with contextlib.suppress(Exception):
                manager.stop()
            return
        if not manager.supported:
            if manager.note:
                log.info("Global hotkeys unavailable: %s", manager.note)
            return
        failed: list[str] = []
        failing: set[tuple[str, str]] = set()
        for name in HOTKEY_ACTIONS:
            combo = str(getattr(hk, name, "") or "").strip()
            if not combo:
                continue
            try:
                ok = manager.register(name, combo, functools.partial(self._on_hotkey, name))
            except Exception:
                log.warning("Registering hotkey %s failed", combo, exc_info=True)
                ok = False
            if ok:
                continue
            key = (name, combo.lower())
            failing.add(key)
            if key in self._announced_hotkey_failures:
                log.debug("Hotkey %s for %s is still unavailable (already announced)", combo, name)
                continue
            reason = self._hotkey_error(manager, name)
            failed.append(
                reason or f"{combo} could not be registered (invalid, or used by another app)"
            )
        # Pairs that register now (or are no longer configured) are forgotten, so
        # a combination that fails again later is announced again.
        self._announced_hotkey_failures = failing
        if failed:
            self._notify(
                "Hotkey unavailable",
                "; ".join(failed) + ". Choose another in Settings → Hotkeys.",
            )

    @staticmethod
    def _hotkey_error(manager: HotkeyManager, name: str) -> str | None:
        """The manager's explanation of why ``name`` is not registered, if it has one."""
        try:
            reason = manager.last_error(name)
        except Exception:
            log.debug("Reading the hotkey error failed", exc_info=True)
            return None
        if not reason:
            return None
        # Joined into one sentence with the others; the caller adds the full stop.
        return str(reason).strip().rstrip(".") or None

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
        before = (self._paused, self._privacy)
        if name == "toggle_tracking":
            self.toggle_pause()
            self._confirm_mode_change(before)
        elif name == "toggle_privacy":
            self.toggle_privacy()
            self._confirm_mode_change(before)
        elif name == "recalibrate":
            self.calibration_required.emit("hotkey")

    def _confirm_mode_change(self, before: tuple[bool, bool]) -> None:
        """Confirm a pause or privacy change made by a hotkey or ``ctl`` command.

        Both are toggles, and the tray icon often sits in a hidden overflow area,
        so without a word the user cannot tell which way a press went, and
        pausing also stops walk-away detection. ``before`` is ``(paused,
        privacy)`` before the change. The tray menu needs no confirmation: its
        check marks show the result.
        """
        paused, privacy = self._paused, self._privacy
        was_paused, was_private = before
        stopped = (
            "Switching monitors and walk-away detection pause"
            if self._settings.presence.enabled
            else "Switching monitors pauses"
        )
        if self._state.camera_active:
            back_on = "The camera is on again."
        else:  # e.g. the session is locked, or another app has the camera
            back_on = f"The camera stays off for now ({self._state.label})."
        if privacy != was_private:
            if privacy:
                title = "Privacy mode on"
                message = (
                    f"The camera is off. {stopped} until you turn privacy mode "
                    f"off{self._hotkey_hint('toggle_privacy')}."
                )
            else:
                title = "Privacy mode off"
                message = "Tracking is still paused: the camera stays off." if paused else back_on
        elif paused != was_paused:
            if paused:
                title = "Tracking paused"
                message = (
                    f"The camera is released. {stopped} until you "
                    f"resume{self._hotkey_hint('toggle_tracking')}."
                )
            else:
                title = "Tracking resumed"
                message = "Privacy mode is still on: the camera stays off." if privacy else back_on
        else:
            return
        self._notify(title, message)

    def _hotkey_hint(self, name: str) -> str:
        """``" (Ctrl+Alt+Win+T)"`` for a registered hotkey ``name``, else ``""``."""
        from ..platform.hotkeys import Hotkey, format_hotkey

        manager = self._hotkeys
        try:
            hotkey = manager.registered.get(name) if manager is not None else None
        except Exception:
            log.debug("Reading the registered hotkeys failed", exc_info=True)
            return ""
        return f" ({format_hotkey(hotkey)})" if isinstance(hotkey, Hotkey) else ""

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
        before = self._monitors
        assumed = next((m for m in before if m.index == self._assumed_monitor), None)
        self._monitors = list(monitors)
        self._decider.set_monitors(self._monitors)
        self._pane_decider.reset()
        self._pane_sigma_cache = None
        sides = [min(m.rect.w, m.rect.h) for m in self._monitors if m.rect.w > 0 and m.rect.h > 0]
        self._min_side = float(min(sides)) if sides else 1000.0
        if self._cursor_reliable:
            return
        if assumed is not None:
            # Monitors are numbered in the order Qt lists the screens, so removing
            # a screen (or changing the primary) renumbers the others: follow the
            # physical monitor the pointer was moved to, not its old number.
            match = _same_monitor(assumed, before, self._monitors)
            self._assumed_monitor = match.index if match is not None else None
        if not any(m.index == self._assumed_monitor for m in self._monitors):
            # A stale position is still the best guess until the first switch.
            cursor = self._cursor_pos()
            monitor = monitor_at(self._monitors, *cursor) if cursor else None
            self._assumed_monitor = monitor.index if monitor is not None else None

    # ==================================================================== helpers
    def _cursor_pos(self) -> tuple[int, int] | None:
        try:
            x, y = self._cursor.pos()
        except Exception:
            log.debug("Reading the cursor position failed", exc_info=True)
            return None
        return (int(x), int(y))

    def _platform_call(self, name: str, *args: Any, default: Any = None) -> Any:
        """Call a platform method; the contract says they never raise, but be safe.

        A method the object does not have (a partial test double) returns
        ``default``.
        """
        method = getattr(self._platform, name, None)
        if method is None:
            return default
        try:
            return method(*args)
        except Exception:
            log.debug("Platform call %s failed", name, exc_info=True)
            return default

    def _notify(self, title: str, message: str, *, force: bool = False) -> None:
        """Emit ``notify``. Informational messages respect the notifications setting;
        ``force`` is for actions the user chose to be notified about."""
        log.info("Notification: %s — %s", title, message)
        if force or self._settings.general.notifications:
            self.notify.emit(title, message)
