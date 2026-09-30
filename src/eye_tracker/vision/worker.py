"""Background thread that captures frames and turns them into observations.

The worker owns the camera and the vision backend. Both are created, used and
released on the worker thread (backends need not be thread-safe, and camera
drivers are happiest when one thread does all the talking). The rest of the
application steers it through thread-safe setters and receives results through
callbacks, which run on the worker thread; the Qt controller marshals them to
the main thread.

Low CPU use comes from four mechanisms:

* the loop sleeps between frames for the interval chosen by the rate scheduler,
* a :class:`~eye_tracker.vision.motion.MotionGate` skips inference while the
  picture is unchanged (a copy of the last observation is emitted instead),
* OpenCV's thread pool is capped (see :mod:`eye_tracker.vision.threads`), and
* the camera is released entirely whenever the worker is inactive.

Analysed frames without a face are also checked for being *blind* (lens
covered, shutter closed, unlit room); such observations carry
``Observation.blind`` so that presence detection can tell "cannot see" from
"nobody here".
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..types import Observation, WorkerStats
from .backends.base import BackendUnavailable, VisionBackend
from .camera import FrameSource
from .motion import MotionGate, is_blind, thumbnail
from .threads import limit_opencv_threads

log = logging.getLogger(__name__)

ObservationCallback = Callable[[Observation], None]
StatsCallback = Callable[[WorkerStats], None]
PreviewCallback = Callable[[np.ndarray], None]
SourceFactory = Callable[[], FrameSource]
BackendFactory = Callable[[], VisionBackend]

#: Interval used until the controller sets one (4 fps).
DEFAULT_INTERVAL_S = 0.25
_MAX_INTERVAL_S = 60.0
_EWMA_ALPHA = 0.2
_SKIP_ALPHA = 0.1
# Repeated identical problems (a missing camera, a buggy callback) are logged at
# most this often at WARNING level; repeats in between go to DEBUG.
_LOG_REPEAT_S = 60.0
# Timestamps handed to the backend must strictly increase even with a coarse
# clock (time.monotonic() ticks every ~16 ms on Windows before Python 3.13).
_MIN_TIMESTAMP_STEP = 1e-4


@dataclasses.dataclass(frozen=True, slots=True)
class _Config:
    """Snapshot of the settings the worker thread acts on."""

    interval: float
    active: bool
    max_faces: int
    gate_enabled: bool
    gate_threshold: float
    preview: bool


def _skipped_copy(obs: Observation, now: float) -> Observation:
    features = None if obs.features is None else obs.features.copy()
    return dataclasses.replace(
        obs, timestamp=now, skipped=True, inference_ms=0.0, features=features
    )


class VisionWorker:
    """Runs capture and analysis on a background thread.

    Args:
        source_factory: Creates the frame source. Called on the worker thread
            each time the camera has to be (re)opened; may raise.
        backend_factory: Creates the vision backend on the worker thread; may
            raise (e.g. ``BackendUnavailable``).
        on_observation: Receives every observation, including ``skipped`` copies
            emitted while the motion gate holds.
        on_stats: Receives a :class:`WorkerStats` snapshot at most once per second,
            plus immediately when the camera opens or closes or an error appears or
            clears. Besides the counters, ``extra`` carries ``"backend"``,
            ``"feature_version"``, ``"device"`` (the frame source's settings
            string, once opened), ``"frame_size"`` (``(width, height)`` of the
            analysed frames) and ``"backend_failing"`` (the backend keeps
            raising; ``last_error`` says why).
        on_preview: Receives an annotated copy of each analysed frame while the
            preview is enabled.
        clock: Time source for observation timestamps and frame pacing.

    All callbacks run on the worker thread. Exceptions they raise are logged and
    otherwise ignored.
    """

    #: Delays between attempts to open the camera or create the backend.
    BACKOFF_S: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    #: Consecutive ``process()`` failures after which the backend is recreated.
    BACKEND_FAILURE_LIMIT = 3
    #: Minimum seconds between routine ``on_stats`` calls.
    STATS_PERIOD_S = 1.0
    #: The backend is closed (freeing its memory) after this long inactive.
    IDLE_BACKEND_CLOSE_S = 300.0
    THREAD_NAME = "eye-tracker-vision"

    def __init__(
        self,
        source_factory: SourceFactory,
        backend_factory: BackendFactory,
        on_observation: ObservationCallback,
        on_stats: StatsCallback | None = None,
        on_preview: PreviewCallback | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_observation = on_observation
        self._on_stats = on_stats
        self._on_preview = on_preview
        self._clock = clock

        # Shared with the controlling thread; guarded by _cond.
        self._cond = threading.Condition()
        self._interval = DEFAULT_INTERVAL_S
        self._active = True
        self._max_faces = 1
        self._gate_enabled = True
        self._gate_threshold = 2.0
        self._preview = False
        self._pending_source: SourceFactory | None = None
        self._pending_backend: BackendFactory | None = None
        self._stop_requested = False
        # Bumped by changes that must cut a back-off wait short.
        self._control_gen = 0
        self._thread: threading.Thread | None = None

        # Worker-thread state.
        self._source_factory = source_factory
        self._backend_factory = backend_factory
        self._source: FrameSource | None = None
        self._backend: VisionBackend | None = None
        self._backend_info: tuple[str, str] | None = None
        self._applied_max_faces: int | None = None
        self._gate = MotionGate(threshold=self._gate_threshold)
        self._gate_in_use = False
        self._last_obs: Observation | None = None
        self._last_ts = -math.inf
        self._slot: float | None = None  # scheduled start of the current frame
        self._backoff_step = 0
        self._backend_failures = 0
        self._backend_recreations = 0
        self._log_times: dict[str, float] = {}

        # Statistics; guarded by _stats_lock.
        self._stats_lock = threading.Lock()
        self._stats = WorkerStats(
            target_fps=1.0 / DEFAULT_INTERVAL_S, extra={"backend_failing": False}
        )
        self._dt_ewma: float | None = None
        self._last_frame_t: float | None = None
        self._counters = {
            "frames": 0,
            "processed": 0,
            "skipped": 0,
            "camera_opens": 0,
            "backend_errors": 0,
        }
        self._last_publish = -math.inf

    # ================================================================ control API
    def start(self) -> None:
        """Start the worker thread (no-op if it is already running)."""
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                if self._stop_requested:
                    log.warning("Vision worker is still stopping; start() ignored")
                return
            self._stop_requested = False
            self._thread = threading.Thread(target=self._run, name=self.THREAD_NAME, daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the thread and wait up to ``timeout`` seconds for it to finish.

        The camera is released and the backend closed on the worker thread before
        it exits. Safe to call repeatedly, before ``start()`` and from callbacks
        (then it does not wait).
        """
        with self._cond:
            self._stop_requested = True
            self._control_gen += 1
            self._cond.notify_all()
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout)
        if thread.is_alive():
            # Typically a camera driver blocking in open(); the daemon thread
            # releases everything as soon as that call returns.
            log.warning("Vision worker did not stop within %.1f s", timeout)
        else:
            with self._cond:
                if self._thread is thread:
                    self._thread = None

    @property
    def is_running(self) -> bool:
        with self._cond:
            return self._thread is not None and self._thread.is_alive() and not self._stop_requested

    def set_interval(self, seconds: float) -> None:
        """Target period between analysed frames (0 = as fast as possible).

        A shorter interval takes effect immediately, even mid-wait.
        """
        value = float(seconds)
        if math.isnan(value):
            raise ValueError("interval must be a number")
        value = min(max(value, 0.0), _MAX_INTERVAL_S)
        with self._cond:
            if value == self._interval:
                return
            self._interval = value
            self._cond.notify_all()
        with self._stats_lock:
            self._stats.target_fps = 1.0 / value if value > 0 else 0.0

    def set_active(self, active: bool) -> None:
        """``False`` releases the camera and idles; ``True`` (re)opens it."""
        with self._cond:
            if self._active == bool(active):
                return
            self._active = bool(active)
            self._control_gen += 1
            self._cond.notify_all()

    def set_max_faces(self, n: int) -> None:
        """Faces the backend should look for (2 for the shoulder guard)."""
        with self._cond:
            self._max_faces = max(1, int(n))

    def set_motion_gate(self, enabled: bool, threshold: float) -> None:
        """Enable/disable the motion gate and set its threshold (grey levels)."""
        with self._cond:
            self._gate_enabled = bool(enabled)
            self._gate_threshold = max(0.0, float(threshold))

    def set_preview(self, enabled: bool) -> None:
        """Deliver annotated frames to ``on_preview``.

        While the preview is on the motion gate is bypassed so the picture stays
        live.
        """
        with self._cond:
            self._preview = bool(enabled)
            self._cond.notify_all()

    def reconfigure(
        self,
        source_factory: SourceFactory | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        """Swap the camera and/or backend; applied on the worker thread.

        The old source is released and the old backend closed before the new ones
        are created. Any pending back-off is cancelled so the change is tried at
        once.
        """
        if source_factory is None and backend_factory is None:
            return
        with self._cond:
            if source_factory is not None:
                self._pending_source = source_factory
            if backend_factory is not None:
                self._pending_backend = backend_factory
            self._control_gen += 1
            self._cond.notify_all()

    @property
    def stats(self) -> WorkerStats:
        """A snapshot of the current statistics (safe to keep and mutate)."""
        with self._stats_lock:
            # The counters go in every snapshot, also those published before any
            # frame was analysed (a camera that opens but whose frames all fail).
            extra = {**self._stats.extra, **self._counters}
            return dataclasses.replace(self._stats, extra=extra)

    @property
    def backend_info(self) -> tuple[str, str] | None:
        """``(name, feature_version)`` of the most recently created backend."""
        return self._backend_info

    # ================================================================ thread body
    def _run(self) -> None:
        log.debug("Vision worker started")
        # Every frame (skipped ones too) goes through OpenCV colour conversions
        # and resizes; an uncapped pool multiplies their CPU cost.
        limit_opencv_threads()
        self._slot = None
        self._backoff_step = 0
        self._backend_failures = 0
        self._backend_recreations = 0
        with self._stats_lock:
            self._stats.extra["backend_failing"] = False
        try:
            while True:
                with self._cond:
                    if self._stop_requested:
                        break
                    cfg = self._snapshot_locked()
                    new_source, self._pending_source = self._pending_source, None
                    new_backend, self._pending_backend = self._pending_backend, None
                try:
                    if new_source is not None or new_backend is not None:
                        self._apply_reconfigure(new_source, new_backend)
                    self._iteration(cfg)
                except Exception:
                    # A bug must not kill the thread (the camera would stay on with
                    # nobody reading it); report it and retry after a pause.
                    self._log_throttled("loop", "Unexpected error in the vision worker", exc=True)
                    self._set_error("Internal error in the vision worker (see the log)")
                    self._release_source()
                    self._backoff_wait()
        finally:
            self._release_source()
            self._close_backend()
            self._reset_tracking()
            self._publish_stats(force=True)
            log.debug("Vision worker stopped")

    def _snapshot_locked(self) -> _Config:
        return _Config(
            interval=self._interval,
            active=self._active,
            max_faces=self._max_faces,
            gate_enabled=self._gate_enabled,
            gate_threshold=self._gate_threshold,
            preview=self._preview,
        )

    def _has_pending_locked(self) -> bool:
        return self._pending_source is not None or self._pending_backend is not None

    def _iteration(self, cfg: _Config) -> None:
        if not cfg.active:
            self._go_idle()
            return
        self._apply_config(cfg)
        if self._backend is None and not self._create_backend(cfg):
            self._backoff_wait()
            return
        with self._cond:
            # Creating a backend can take a second; privacy mode, a lock or stop()
            # may have arrived meanwhile, and the camera must then stay off.
            if self._stop_requested or not self._active or self._has_pending_locked():
                return
        if not self._ensure_source():
            self._backoff_wait()
            return
        with self._cond:
            # Opening a camera can take seconds; if the worker was deactivated,
            # stopped or reconfigured meanwhile, act on that before using it.
            if self._stop_requested or not self._active or self._has_pending_locked():
                return
        # Frames are scheduled on a grid (slot, slot + interval, ...) so that a timer
        # overshoot in one wait (~16 ms on Windows) is recovered in the next and the
        # average rate matches the target. Falling more than one interval off the
        # grid (a slow frame, a changed interval) restarts it from now.
        now = self._clock()
        if self._slot is None or abs(now - self._slot) > cfg.interval:
            self._slot = now
        if self._cycle(cfg):
            self._slot = self._wait_next(self._slot)
        else:
            self._slot = None
            self._backoff_wait()

    def _cycle(self, cfg: _Config) -> bool:
        """Read and analyse one frame. Returns ``False`` to back off before the next."""
        source = self._source
        backend = self._backend
        assert source is not None
        assert backend is not None

        read_started = time.perf_counter()
        try:
            frame = source.read()
        except Exception as exc:
            log.debug("Frame source read raised", exc_info=True)
            frame = None
            if source.is_open:
                self._set_error(f"Camera read failed: {exc}")
        read_ms = (time.perf_counter() - read_started) * 1e3
        if frame is None:
            if source.is_open:
                return True  # transient; the source gives up by itself if it persists
            message = source.last_error or "Camera stopped delivering frames"
            self._log_throttled("read", message)
            self._set_error(message)
            self._release_source()
            self._reset_tracking()
            return False

        now = self._clock()
        if now <= self._last_ts:
            now = self._last_ts + _MIN_TIMESTAMP_STEP
        self._last_ts = now
        self._backoff_step = 0
        if self._backend_recreations == 0:
            # A failing backend's error stays until a frame is analysed again.
            self._clear_error()

        use_gate = cfg.gate_enabled and not cfg.preview
        if use_gate != self._gate_in_use:
            self._gate.reset()
            self._gate_in_use = use_gate
        last = self._last_obs
        # Never hold a closed-eyes observation: the reopening can be too subtle
        # for the gate, and a repeated blink would suppress gaze for seconds.
        if (
            use_gate
            and last is not None
            and not last.blink
            and not self._gate.should_process(frame, now)
        ):
            self._record_frame(now, read_ms, skipped=True)
            self._safe_call(self._on_observation, _skipped_copy(last, now), "observation")
            return True

        process_started = time.perf_counter()
        try:
            obs = backend.process(frame, now)
        except Exception as exc:
            self._backend_failures += 1
            with self._stats_lock:
                self._counters["backend_errors"] += 1
            self._log_throttled("process", "Vision backend failed to process a frame", exc=True)
            if self._backend_failures < self.BACKEND_FAILURE_LIMIT:
                return True
            self._backend_recreations += 1
            self._log_throttled(
                "recreate",
                f"Vision backend failed {self._backend_failures} times in a row; recreating it",
            )
            # Make the failure visible (the camera would otherwise look healthy
            # while nothing is analysed) and keep the camera off while backing off.
            with self._stats_lock:
                self._stats.extra["backend_failing"] = True
            self._set_error(f"Vision backend keeps failing: {str(exc) or type(exc).__name__}")
            self._close_backend()
            self._release_source()
            # Back off before recreating, longer each time it keeps failing, so a
            # persistently broken backend cannot burn CPU being rebuilt.
            self._backoff_step = self._backend_recreations - 1
            return False
        self._backend_failures = 0
        if self._backend_recreations:
            self._backend_recreations = 0
            with self._stats_lock:
                self._stats.extra["backend_failing"] = False
            self._clear_error()
        elapsed_ms = (time.perf_counter() - process_started) * 1e3
        if use_gate:
            self._gate.mark_processed(frame, now, obs.face_box)
        if obs.face_count == 0 and not obs.blind:
            reference = self._gate.reference if use_gate else None
            if is_blind(reference if reference is not None else thumbnail(frame)):
                obs = dataclasses.replace(obs, blind=True)
        with self._stats_lock:
            if self._stats.extra.get("frame_size") != obs.frame_size:
                self._stats.extra["frame_size"] = obs.frame_size
        self._last_obs = obs
        self._record_frame(now, read_ms, skipped=False, inference_ms=obs.inference_ms or elapsed_ms)
        self._safe_call(self._on_observation, obs, "observation")

        if cfg.preview and self._on_preview is not None:
            try:
                annotated = backend.annotate(frame, obs)
            except Exception:
                self._log_throttled("annotate", "Could not annotate the preview frame", exc=True)
            else:
                self._safe_call(self._on_preview, annotated, "preview")
        return True

    # ----------------------------------------------------------- resources
    def _create_backend(self, cfg: _Config) -> bool:
        try:
            backend = self._backend_factory()
        except Exception as exc:
            message = f"Vision backend unavailable: {exc}"
            # A missing runtime or model is expected on some systems; anything else
            # is a bug worth a traceback.
            self._log_throttled("backend", message, exc=not isinstance(exc, BackendUnavailable))
            self._set_error(message)
            # Nothing can be analysed, so keep the camera (and its light) off.
            self._release_source()
            return False
        self._backend = backend
        self._backend_failures = 0
        self._backend_info = (backend.name, backend.feature_version)
        self._applied_max_faces = None
        self._apply_config(cfg)
        with self._stats_lock:
            self._stats.extra["backend"] = backend.name
            self._stats.extra["feature_version"] = backend.feature_version
        log.debug("Vision backend %s created", backend.name)
        return True

    def _close_backend(self) -> None:
        backend, self._backend = self._backend, None
        self._applied_max_faces = None
        if backend is None:
            return
        try:
            backend.close()
        except Exception:
            log.debug("Closing the vision backend failed", exc_info=True)
        # A new backend may produce different features; never reuse old results.
        self._reset_tracking()

    def _ensure_source(self) -> bool:
        if self._source is not None and self._source.is_open:
            return True
        self._release_source()
        try:
            source = self._source_factory()
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            self._log_throttled("open", f"Camera unavailable: {message}")
            self._set_error(message)
            return False
        self._source = source
        try:
            opened = source.open()
            error = source.last_error
        except Exception as exc:
            log.debug("Frame source open() raised", exc_info=True)
            opened = False
            error = f"Camera could not be opened: {exc}"
        if not opened:
            message = error or "Camera could not be opened"
            self._log_throttled("open", message)
            self._set_error(message)
            self._release_source()
            return False
        device = getattr(source, "device", None)
        with self._stats_lock:
            self._counters["camera_opens"] += 1
            self._stats.camera_open = True
            self._stats.extra["device"] = None if device is None else str(device)
        if self._backend_recreations == 0:
            self._clear_error()
        self._publish_stats(force=True)
        return True

    def _release_source(self) -> None:
        source, self._source = self._source, None
        if source is None:
            return
        try:
            source.release()
        except Exception:
            log.debug("Releasing the frame source failed", exc_info=True)
        with self._stats_lock:
            was_open = self._stats.camera_open
            self._stats.camera_open = False
            self._stats.fps = 0.0
            self._dt_ewma = None
            self._last_frame_t = None
        if was_open:
            self._publish_stats(force=True)

    def _apply_reconfigure(
        self, source_factory: SourceFactory | None, backend_factory: BackendFactory | None
    ) -> None:
        if source_factory is not None:
            self._release_source()
            self._source_factory = source_factory
        if backend_factory is not None:
            self._close_backend()
            self._backend_factory = backend_factory
            # A different backend gets a clean slate.
            self._backend_recreations = 0
            with self._stats_lock:
                self._stats.extra["backend_failing"] = False
        self._reset_tracking()
        self._backoff_step = 0
        self._log_times.clear()
        self._clear_error()

    def _apply_config(self, cfg: _Config) -> None:
        self._gate.threshold = cfg.gate_threshold
        if self._backend is not None and cfg.max_faces != self._applied_max_faces:
            try:
                self._backend.set_max_faces(cfg.max_faces)
            except Exception:
                log.warning("Vision backend rejected max_faces=%d", cfg.max_faces, exc_info=True)
            self._applied_max_faces = cfg.max_faces

    def _reset_tracking(self) -> None:
        self._gate.reset()
        self._last_obs = None

    # -------------------------------------------------------------- waiting
    def _go_idle(self) -> None:
        """Release the camera and block until activated, reconfigured or stopped."""
        self._release_source()
        self._reset_tracking()
        self._backoff_step = 0
        self._slot = None
        close_at = time.monotonic() + self.IDLE_BACKEND_CLOSE_S
        with self._cond:
            while not self._stop_requested and not self._active and not self._has_pending_locked():
                if self._backend is None:
                    self._cond.wait()
                    continue
                remaining = close_at - time.monotonic()
                if remaining > 0:
                    self._cond.wait(remaining)
                    continue
                # Inactive for a long time (privacy mode, locked session): free the
                # backend's memory. Closing it under the lock is fine, it is fast.
                log.debug("Vision worker idle; closing the backend to free memory")
                self._close_backend()

    def _wait_next(self, slot: float) -> float:
        """Sleep until ``slot + interval`` and return that time.

        Setters wake the wait early so a shortened interval applies at once;
        ``stop()``, ``set_active(False)`` and ``reconfigure()`` end it.
        """
        with self._cond:
            due = slot + self._interval
            while not self._stop_requested and self._active and not self._has_pending_locked():
                due = slot + self._interval
                remaining = due - self._clock()
                if remaining <= 0:
                    break
                if not self._cond.wait(min(remaining, _MAX_INTERVAL_S)):
                    # Timed out. Not re-checking the clock keeps the loop moving
                    # even with a clock that does not advance.
                    break
            return due

    def _backoff_wait(self) -> None:
        delay = self.BACKOFF_S[min(self._backoff_step, len(self.BACKOFF_S) - 1)]
        self._backoff_step += 1
        deadline = time.monotonic() + delay
        with self._cond:
            generation = self._control_gen
            while not self._stop_requested and self._control_gen == generation:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._cond.wait(remaining)

    # ----------------------------------------------------------- statistics
    def _record_frame(
        self, now: float, read_ms: float, *, skipped: bool, inference_ms: float = 0.0
    ) -> None:
        with self._stats_lock:
            stats = self._stats
            if self._last_frame_t is not None:
                dt = now - self._last_frame_t
                if dt > 0:
                    if self._dt_ewma is None:
                        self._dt_ewma = dt
                    else:
                        self._dt_ewma += _EWMA_ALPHA * (dt - self._dt_ewma)
            self._last_frame_t = now
            stats.fps = 1.0 / self._dt_ewma if self._dt_ewma else 0.0
            stats.skip_ratio += _SKIP_ALPHA * ((1.0 if skipped else 0.0) - stats.skip_ratio)
            self._counters["frames"] += 1
            if skipped:
                self._counters["skipped"] += 1
            else:
                self._counters["processed"] += 1
                if stats.inference_ms <= 0.0:
                    stats.inference_ms = inference_ms
                else:
                    stats.inference_ms += _EWMA_ALPHA * (inference_ms - stats.inference_ms)
            previous_read = float(stats.extra.get("read_ms", read_ms))
            stats.extra["read_ms"] = previous_read + _EWMA_ALPHA * (read_ms - previous_read)
            stats.extra["motion"] = self._gate.last_motion
        self._publish_stats()

    def _set_error(self, message: str) -> None:
        with self._stats_lock:
            changed = self._stats.last_error != message
            self._stats.last_error = message
        if changed:
            self._publish_stats(force=True)

    def _clear_error(self) -> None:
        with self._stats_lock:
            changed = self._stats.last_error is not None
            self._stats.last_error = None
        if changed:
            self._publish_stats(force=True)

    def _publish_stats(self, *, force: bool = False) -> None:
        if self._on_stats is None:
            return
        now = time.monotonic()
        if not force and now - self._last_publish < self.STATS_PERIOD_S:
            return
        self._last_publish = now
        self._safe_call(self._on_stats, self.stats, "stats")

    # -------------------------------------------------------------- helpers
    def _safe_call(self, callback: Callable[[Any], None], value: Any, what: str) -> None:
        try:
            callback(value)
        except Exception:
            self._log_throttled(f"cb-{what}", f"Error in the {what} callback", exc=True)

    def _log_throttled(self, key: str, message: str, *, exc: bool = False) -> None:
        now = time.monotonic()
        if now - self._log_times.get(key, -math.inf) >= _LOG_REPEAT_S:
            self._log_times[key] = now
            log.warning(message, exc_info=exc)
        else:
            log.debug(message, exc_info=exc)
