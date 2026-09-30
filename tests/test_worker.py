"""Tests for eye_tracker.vision.worker.VisionWorker using fake sources and backends."""

from __future__ import annotations

import dataclasses
import itertools
import threading
import time
from collections.abc import Callable, Iterator
from typing import ClassVar

import cv2
import numpy as np
import pytest

from eye_tracker.types import Observation, WorkerStats
from eye_tracker.vision import worker as worker_mod
from eye_tracker.vision.backends.base import BackendUnavailable, VisionBackend
from eye_tracker.vision.camera import CameraError
from eye_tracker.vision.worker import VisionWorker

FAST_BACKOFF = (0.01, 0.02, 0.04)
TIMEOUT = 3.0


# --------------------------------------------------------------------------- fakes
class FakeSource:
    """In-memory frame source with scriptable behaviour."""

    def __init__(
        self,
        *,
        can_open: Callable[[], bool] = lambda: True,
        frame_fn: Callable[[int], np.ndarray] | None = None,
        die_after: int | None = None,
    ) -> None:
        self.can_open = can_open
        self.frame_fn = frame_fn or (lambda _n: np.full((48, 64, 3), 100, np.uint8))
        self.die_after = die_after
        self.open_calls = 0
        self.release_calls = 0
        self.reads = 0
        self.threads: set[int] = set()
        self._open = False
        self._error: str | None = None

    def open(self) -> bool:
        self.threads.add(threading.get_ident())
        self.open_calls += 1
        if not self.can_open():
            self._error = "fake camera is busy"
            return False
        self._open = True
        self._error = None
        return True

    def read(self) -> np.ndarray | None:
        if not self._open:
            return None
        self.reads += 1
        if self.die_after is not None and self.reads > self.die_after:
            self._open = False
            self._error = "fake camera unplugged"
            return None
        return self.frame_fn(self.reads)

    def release(self) -> None:
        self.release_calls += 1
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def last_error(self) -> str | None:
        return self._error


class FakeBackend(VisionBackend):
    name: ClassVar[str] = "fake"
    feature_names: ClassVar[tuple[str, ...]] = ("a", "b")
    feature_version: ClassVar[str] = "fake-1"

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.process_calls = 0
        self.close_calls = 0
        self.max_faces_calls: list[int] = []
        self.timestamps: list[float] = []
        self.threads: set[int] = set()

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Observation:
        self.threads.add(threading.get_ident())
        self.process_calls += 1
        self.timestamps.append(timestamp)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated inference failure")
        h, w = frame_bgr.shape[:2]
        return Observation(
            timestamp=timestamp,
            face_count=1,
            features=np.array([1.0, float(self.process_calls)]),
            quality=1.0,
            face_box=(0.25, 0.2, 0.5, 0.6),
            inference_ms=1.5,
            frame_size=(w, h),
        )

    def set_max_faces(self, n: int) -> None:
        self.threads.add(threading.get_ident())
        self.max_faces_calls.append(n)

    def annotate(self, frame_bgr: np.ndarray, observation: Observation) -> np.ndarray:
        out = frame_bgr.copy()
        out[0, 0] = (1, 2, 3)
        return out

    def close(self) -> None:
        self.threads.add(threading.get_ident())
        self.close_calls += 1


class Recorder:
    """Thread-safe sink for worker callbacks."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.observations: list[Observation] = []
        self.stats: list[WorkerStats] = []
        self.previews: list[np.ndarray] = []

    def on_observation(self, obs: Observation) -> None:
        with self.cond:
            self.observations.append(obs)
            self.cond.notify_all()

    def on_stats(self, stats: WorkerStats) -> None:
        with self.cond:
            self.stats.append(stats)
            self.cond.notify_all()

    def on_preview(self, frame: np.ndarray) -> None:
        with self.cond:
            self.previews.append(frame)
            self.cond.notify_all()

    def wait_for(self, predicate: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
        with self.cond:
            return self.cond.wait_for(predicate, timeout)

    def count(self) -> int:
        with self.cond:
            return len(self.observations)


def strictly_increasing(values: list[float]) -> bool:
    return all(b > a for a, b in itertools.pairwise(values))


def wait_until(predicate: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


class Harness:
    """A worker wired to fakes; every source/backend it creates is recorded."""

    def __init__(self, **source_kwargs: object) -> None:
        self.recorder = Recorder()
        self.sources: list[FakeSource] = []
        self.backends: list[FakeBackend] = []
        self.source_kwargs = source_kwargs
        self.backend_fail_times = 0
        self.factory_threads: set[int] = set()
        self.worker = VisionWorker(
            self.make_source,
            self.make_backend,
            self.recorder.on_observation,
            on_stats=self.recorder.on_stats,
            on_preview=self.recorder.on_preview,
        )
        self.worker.BACKOFF_S = FAST_BACKOFF
        self.worker.set_interval(0.005)

    def make_source(self) -> FakeSource:
        self.factory_threads.add(threading.get_ident())
        source = FakeSource(**self.source_kwargs)  # type: ignore[arg-type]
        self.sources.append(source)
        return source

    def make_backend(self) -> FakeBackend:
        self.factory_threads.add(threading.get_ident())
        backend = FakeBackend(fail_times=self.backend_fail_times)
        self.backend_fail_times = 0  # only the next backend is faulty
        self.backends.append(backend)
        return backend

    @property
    def obs(self) -> list[Observation]:
        with self.recorder.cond:
            return list(self.recorder.observations)


@pytest.fixture
def harness() -> Iterator[Harness]:
    h = Harness()
    try:
        yield h
    finally:
        h.worker.stop()


def changing_frames(n: int) -> np.ndarray:
    return np.full((48, 64, 3), 40 if n % 2 else 200, np.uint8)


# ------------------------------------------------------------------ lifecycle
def test_emits_observations_on_worker_thread(harness: Harness) -> None:
    harness.worker.set_motion_gate(False, 2.0)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 5)
    main = threading.get_ident()
    backend = harness.backends[0]
    assert len(harness.backends) == 1
    assert main not in backend.threads
    assert main not in harness.factory_threads
    assert harness.factory_threads == backend.threads
    assert harness.sources[0].threads == backend.threads
    stamps = [o.timestamp for o in harness.obs]
    assert strictly_increasing(stamps)
    assert harness.worker.backend_info == ("fake", "fake-1")
    assert harness.worker.is_running


def test_stop_joins_releases_and_closes(harness: Harness) -> None:
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 2)
    harness.worker.stop(timeout=TIMEOUT)
    assert not harness.worker.is_running
    backend = harness.backends[0]
    assert backend.close_calls == 1
    assert threading.get_ident() not in backend.threads  # closed on the worker thread
    assert not harness.sources[0].is_open
    assert harness.sources[0].release_calls >= 1
    count = harness.recorder.count()
    harness.worker.stop()  # idempotent
    time.sleep(0.02)
    assert harness.recorder.count() == count
    assert harness.worker.stats.camera_open is False


def test_stop_before_start_is_harmless() -> None:
    h = Harness()
    h.worker.stop()
    assert not h.worker.is_running
    assert h.sources == []


def test_restart_after_stop(harness: Harness) -> None:
    harness.worker.start()
    harness.worker.start()  # second start is a no-op
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 1)
    harness.worker.stop()
    harness.worker.start()
    n = harness.recorder.count()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) > n)
    assert len(harness.backends) == 2


def test_stop_from_callback_does_not_deadlock() -> None:
    h = Harness()

    def stop_on_first(obs: Observation) -> None:
        h.worker.stop()

    h.worker._on_observation = stop_on_first
    h.worker.start()
    assert wait_until(lambda: not h.worker.is_running)
    assert wait_until(lambda: h.backends and h.backends[0].close_calls == 1)


# -------------------------------------------------------------------- pacing
def test_interval_limits_frame_rate(harness: Harness) -> None:
    harness.worker.set_motion_gate(False, 2.0)
    harness.worker.set_interval(0.04)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 4)
    stamps = [o.timestamp for o in harness.obs[:4]]
    gaps = np.diff(stamps)
    # time.monotonic() may tick in ~16 ms steps on Windows, so allow some slack.
    assert gaps.min() >= 0.02
    assert harness.worker.stats.target_fps == pytest.approx(25.0)


def test_shorter_interval_wakes_the_loop(harness: Harness) -> None:
    harness.worker.set_interval(30.0)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 1)
    started = time.monotonic()
    harness.worker.set_interval(0.005)
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 2, 2.0)
    assert time.monotonic() - started < 2.0


def test_invalid_interval() -> None:
    worker = VisionWorker(FakeSource, FakeBackend, lambda _o: None)
    with pytest.raises(ValueError, match="interval"):
        worker.set_interval(float("nan"))
    worker.set_interval(-1.0)
    assert worker.stats.target_fps == 0.0  # clamped to 0 = unlimited


def test_frozen_clock_still_progresses() -> None:
    recorder = Recorder()
    backend = FakeBackend()
    worker = VisionWorker(FakeSource, lambda: backend, recorder.on_observation, clock=lambda: 42.0)
    worker.set_interval(0.005)
    worker.set_motion_gate(False, 2.0)
    worker.start()
    try:
        assert recorder.wait_for(lambda: len(recorder.observations) >= 4)
    finally:
        worker.stop()
    stamps = backend.timestamps
    assert stamps[0] == 42.0
    assert strictly_increasing(stamps)


# ----------------------------------------------------------------- activity
def test_set_active_false_releases_camera(harness: Harness) -> None:
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 2)
    harness.worker.set_active(False)
    first = harness.sources[0]
    assert wait_until(lambda: not first.is_open and not harness.worker.stats.camera_open)
    assert first.release_calls >= 1
    count = harness.recorder.count()
    time.sleep(0.03)
    assert harness.recorder.count() == count  # idle: nothing emitted
    assert harness.recorder.wait_for(lambda: any(not s.camera_open for s in harness.recorder.stats))

    harness.worker.set_active(True)
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) > count)
    assert len(harness.sources) == 2  # a fresh source was opened
    assert harness.sources[1].is_open
    assert len(harness.backends) == 1  # backend survives short pauses


def test_inactive_before_start_never_opens_camera(harness: Harness) -> None:
    harness.worker.set_active(False)
    harness.worker.start()
    time.sleep(0.03)
    assert harness.sources == []
    assert harness.recorder.count() == 0


def test_long_idle_closes_backend(harness: Harness) -> None:
    harness.worker.IDLE_BACKEND_CLOSE_S = 0.02
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 1)
    harness.worker.set_active(False)
    assert wait_until(lambda: harness.backends[0].close_calls == 1)
    harness.worker.set_active(True)
    assert wait_until(lambda: len(harness.backends) == 2)
    n = harness.recorder.count()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) > n)


# --------------------------------------------------------------- motion gate
def test_motion_gate_emits_skipped_copies(harness: Harness) -> None:
    harness.worker.set_motion_gate(True, 2.0)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 6)
    obs = harness.obs
    backend = harness.backends[0]
    assert backend.process_calls == 1  # identical frames: analysed once (max_skip_s = 2 s)
    assert not obs[0].skipped
    for copy in obs[1:6]:
        assert copy.skipped
        assert copy.timestamp > obs[0].timestamp
        assert copy.inference_ms == 0.0
        assert copy.face_count == obs[0].face_count
        assert copy.features is not None
        assert obs[0].features is not None
        np.testing.assert_array_equal(copy.features, obs[0].features)
        assert copy.features is not obs[0].features
    stamps = [o.timestamp for o in obs]
    assert strictly_increasing(stamps)
    assert harness.worker.stats.skip_ratio > 0.0


def test_motion_gate_processes_changing_frames() -> None:
    h = Harness(frame_fn=changing_frames)
    h.worker.set_motion_gate(True, 2.0)
    h.worker.start()
    try:
        assert h.recorder.wait_for(lambda: len(h.recorder.observations) >= 5)
    finally:
        h.worker.stop()
    assert not any(o.skipped for o in h.obs)


def test_motion_gate_disabled_processes_everything(harness: Harness) -> None:
    harness.worker.set_motion_gate(False, 2.0)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 5)
    harness.worker.stop()
    assert not any(o.skipped for o in harness.obs)
    assert harness.backends[0].process_calls == len(harness.obs)


def test_preview_bypasses_gate_and_delivers_frames(harness: Harness) -> None:
    harness.worker.set_motion_gate(True, 2.0)
    harness.worker.set_preview(True)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.previews) >= 3)
    harness.worker.stop()
    assert not any(o.skipped for o in harness.obs)
    preview = harness.recorder.previews[0]
    assert preview.shape == (48, 64, 3)
    assert tuple(preview[0, 0]) == (1, 2, 3)  # annotated by the backend


def test_preview_off_stops_frames(harness: Harness) -> None:
    harness.worker.set_preview(True)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.previews) >= 1)
    harness.worker.set_preview(False)
    n = harness.recorder.count()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= n + 3)
    with harness.recorder.cond:
        previews = len(harness.recorder.previews)
    time.sleep(0.02)
    with harness.recorder.cond:
        assert len(harness.recorder.previews) == previews


# ------------------------------------------------------------ failure handling
def test_open_failure_backs_off_and_recovers() -> None:
    allowed = threading.Event()
    attempts: list[float] = []

    def can_open() -> bool:
        attempts.append(time.monotonic())
        return allowed.is_set()

    h = Harness(can_open=can_open)
    h.worker.start()
    try:
        assert wait_until(lambda: len(attempts) >= 4)
        gaps = np.diff(attempts[:4])
        assert gaps[0] >= FAST_BACKOFF[0] * 0.8
        assert gaps[2] >= FAST_BACKOFF[2] * 0.8  # delays grow instead of spinning
        assert h.recorder.count() == 0
        assert h.recorder.wait_for(
            lambda: any(s.last_error == "fake camera is busy" for s in h.recorder.stats)
        )
        stats = h.worker.stats
        assert not stats.camera_open
        assert stats.last_error == "fake camera is busy"
        assert all(not s.is_open for s in h.sources)

        allowed.set()
        h.worker.reconfigure(source_factory=h.make_source)  # cancels the back-off
        assert h.recorder.wait_for(lambda: len(h.recorder.observations) >= 1)
        assert wait_until(lambda: h.worker.stats.last_error is None)
        assert h.worker.stats.camera_open
    finally:
        h.worker.stop()


def test_keeps_retrying_at_the_longest_backoff(harness: Harness) -> None:
    harness.source_kwargs["can_open"] = lambda: False
    harness.worker.BACKOFF_S = (0.001, 0.002, 0.004)
    harness.worker.start()
    # More attempts than back-off steps: the last delay repeats forever.
    assert wait_until(lambda: len(harness.sources) >= 8)
    assert harness.worker.is_running
    assert harness.worker._backoff_step >= 7


def test_source_factory_exception_is_reported() -> None:
    recorder = Recorder()

    def factory() -> FakeSource:
        raise CameraError("Video source not found: missing.mp4")

    worker = VisionWorker(factory, FakeBackend, recorder.on_observation, recorder.on_stats)
    worker.BACKOFF_S = FAST_BACKOFF
    worker.start()
    try:
        assert wait_until(lambda: worker.stats.last_error is not None)
        assert "missing.mp4" in (worker.stats.last_error or "")
        assert worker.is_running
    finally:
        worker.stop()


def test_backend_unavailable_keeps_camera_closed() -> None:
    recorder = Recorder()
    sources: list[FakeSource] = []
    attempts: list[int] = []

    def backend_factory() -> VisionBackend:
        attempts.append(1)
        raise BackendUnavailable("no model")

    def source_factory() -> FakeSource:
        sources.append(FakeSource())
        return sources[-1]

    worker = VisionWorker(source_factory, backend_factory, recorder.on_observation)
    worker.BACKOFF_S = FAST_BACKOFF
    worker.start()
    try:
        assert wait_until(lambda: len(attempts) >= 3)
        assert "no model" in (worker.stats.last_error or "")
        assert sources == []  # never opened the camera without a backend
        assert recorder.count() == 0
    finally:
        worker.stop()


def test_backend_recreated_after_repeated_failures(harness: Harness) -> None:
    harness.backend_fail_times = 3  # the first backend fails three times in a row
    harness.worker.set_motion_gate(False, 2.0)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 2)
    assert len(harness.backends) == 2
    assert harness.backends[0].close_calls == 1
    assert harness.backends[0].process_calls == 3
    assert harness.worker.stats.extra["backend_errors"] == 3


def test_persistently_failing_backend_backs_off() -> None:
    backends: list[FakeBackend] = []

    def factory() -> FakeBackend:
        backends.append(FakeBackend(fail_times=10**6))
        return backends[-1]

    worker = VisionWorker(FakeSource, factory, lambda _o: None)
    worker.BACKOFF_S = (0.03, 0.06, 0.12)
    worker.set_interval(0.0)
    worker.set_motion_gate(False, 2.0)
    worker.start()
    try:
        time.sleep(0.15)
    finally:
        worker.stop()
    # Without back-off this would recreate the backend hundreds of times.
    assert 1 <= len(backends) <= 5
    assert all(b.close_calls == 1 for b in backends)


def test_single_backend_failure_does_not_recreate(harness: Harness) -> None:
    harness.backend_fail_times = 1
    harness.worker.set_motion_gate(False, 2.0)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 3)
    assert len(harness.backends) == 1


def test_source_death_triggers_reopen() -> None:
    h = Harness(die_after=3)
    h.worker.set_motion_gate(False, 2.0)
    h.worker.start()
    try:
        assert wait_until(lambda: len(h.sources) >= 2)
        assert h.recorder.wait_for(
            lambda: any(s.last_error == "fake camera unplugged" for s in h.recorder.stats)
        )
        assert h.sources[0].release_calls >= 1
        assert h.recorder.wait_for(lambda: len(h.recorder.observations) >= 5)
    finally:
        h.worker.stop()


def test_callback_exceptions_do_not_kill_the_thread() -> None:
    backend = FakeBackend()
    calls = {"obs": 0, "stats": 0}

    def bad_observation(_obs: Observation) -> None:
        calls["obs"] += 1
        raise ValueError("consumer bug")

    def bad_stats(_stats: WorkerStats) -> None:
        calls["stats"] += 1
        raise ValueError("consumer bug")

    worker = VisionWorker(FakeSource, lambda: backend, bad_observation, bad_stats)
    worker.set_interval(0.002)
    worker.set_motion_gate(False, 2.0)
    worker.start()
    try:
        assert wait_until(lambda: calls["obs"] >= 10)
        assert calls["stats"] >= 1
        assert worker.is_running
        assert backend.process_calls >= 10
    finally:
        worker.stop()


def test_preview_callback_exception_is_contained(harness: Harness) -> None:
    def bad_preview(_frame: np.ndarray) -> None:
        raise RuntimeError("paint failed")

    harness.worker._on_preview = bad_preview
    harness.worker.set_preview(True)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 5)
    assert harness.worker.is_running


# ----------------------------------------------------------- configuration
def test_set_max_faces_applies_on_worker_thread(harness: Harness) -> None:
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 1)
    backend = harness.backends[0]
    assert backend.max_faces_calls == [1]
    harness.worker.set_max_faces(2)
    assert wait_until(lambda: backend.max_faces_calls == [1, 2])
    assert threading.get_ident() not in backend.threads


def test_new_backend_gets_current_max_faces(harness: Harness) -> None:
    harness.worker.set_max_faces(2)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 1)
    assert harness.backends[0].max_faces_calls == [2]


def test_reconfigure_swaps_source_and_backend(harness: Harness) -> None:
    harness.worker.set_motion_gate(False, 2.0)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 2)
    old_source, old_backend = harness.sources[0], harness.backends[0]

    new_source = FakeSource(frame_fn=lambda _n: np.zeros((24, 32, 3), np.uint8))
    new_backend = FakeBackend()
    harness.worker.reconfigure(source_factory=lambda: new_source)
    assert wait_until(lambda: new_source.reads > 0)
    assert not old_source.is_open
    assert old_backend.close_calls == 0  # only the camera changed

    harness.worker.reconfigure(backend_factory=lambda: new_backend)
    assert wait_until(lambda: new_backend.process_calls > 0)
    assert old_backend.close_calls == 1
    assert harness.obs[-1].frame_size == (32, 24)
    harness.worker.reconfigure()  # nothing to do


def test_reconfigure_while_inactive(harness: Harness) -> None:
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 1)
    harness.worker.set_active(False)
    new_backend = FakeBackend()
    harness.worker.reconfigure(backend_factory=lambda: new_backend)
    assert wait_until(lambda: harness.backends[0].close_calls == 1)
    assert new_backend.process_calls == 0
    harness.worker.set_active(True)
    assert wait_until(lambda: new_backend.process_calls > 0)


def test_stats_snapshot(harness: Harness) -> None:
    harness.worker.set_interval(0.01)
    harness.worker.set_motion_gate(False, 2.0)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 6)
    stats = harness.worker.stats
    assert stats.camera_open
    assert stats.last_error is None
    assert stats.fps > 0.0
    assert stats.target_fps == pytest.approx(100.0)
    assert stats.inference_ms == pytest.approx(1.5)
    assert stats.extra["backend"] == "fake"
    assert stats.extra["feature_version"] == "fake-1"
    assert stats.extra["frames"] >= 6
    assert stats.extra["camera_opens"] == 1
    stats.extra["frames"] = -1  # snapshots are independent copies
    assert harness.worker.stats.extra["frames"] != -1
    # The camera opening is published immediately.
    assert any(s.camera_open for s in harness.recorder.stats)


def test_stats_report_device_and_frame_size() -> None:
    class NamedSource(FakeSource):
        device = "/dev/v4l/by-id/usb-cam-video-index0"

    recorder = Recorder()
    worker = VisionWorker(NamedSource, FakeBackend, recorder.on_observation, recorder.on_stats)
    worker.set_interval(0.005)
    worker.set_motion_gate(False, 2.0)
    worker.start()
    try:
        assert recorder.wait_for(lambda: len(recorder.observations) >= 2)
        extra = worker.stats.extra
    finally:
        worker.stop()
    assert extra["device"] == "/dev/v4l/by-id/usb-cam-video-index0"
    assert extra["frame_size"] == (64, 48)
    assert extra["backend_failing"] is False


def test_worker_caps_opencv_threads(monkeypatch: pytest.MonkeyPatch, harness: Harness) -> None:
    calls: list[int] = []
    monkeypatch.setattr(worker_mod, "limit_opencv_threads", lambda: calls.append(1) or 1)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 1)
    assert calls == [1]


# ------------------------------------------------- deactivation while starting
def test_camera_stays_off_when_deactivated_during_backend_creation() -> None:
    """Privacy mode pressed while the backend loads: the camera must never open."""
    creating = threading.Event()
    release = threading.Event()
    sources: list[FakeSource] = []

    def slow_backend() -> FakeBackend:
        creating.set()
        release.wait(TIMEOUT)
        return FakeBackend()

    def source_factory() -> FakeSource:
        sources.append(FakeSource())
        return sources[-1]

    worker = VisionWorker(source_factory, slow_backend, lambda _o: None)
    worker.start()
    try:
        assert creating.wait(TIMEOUT)
        worker.set_active(False)
        release.set()
        time.sleep(0.05)
        assert sources == []
        worker.set_active(True)
        assert wait_until(lambda: bool(sources) and sources[0].is_open)
    finally:
        release.set()
        worker.stop()


def test_stop_during_backend_creation_never_opens_the_camera() -> None:
    creating = threading.Event()
    release = threading.Event()
    sources: list[FakeSource] = []

    def slow_backend() -> FakeBackend:
        creating.set()
        release.wait(TIMEOUT)
        return FakeBackend()

    worker = VisionWorker(lambda: sources.append(FakeSource()) or sources[-1], slow_backend, print)
    worker.start()
    assert creating.wait(TIMEOUT)
    stopper = threading.Thread(target=worker.stop, args=(TIMEOUT,))
    stopper.start()
    time.sleep(0.02)
    release.set()
    stopper.join(TIMEOUT)
    assert not worker.is_running
    assert sources == []


# ------------------------------------------------------- failing backend
def test_backend_that_keeps_failing_is_reported_and_releases_the_camera() -> None:
    """A backend that raises on every frame must not look like healthy tracking."""
    h = Harness()

    def always_failing() -> FakeBackend:
        h.backends.append(FakeBackend(fail_times=10**6))
        return h.backends[-1]

    h.worker._backend_factory = always_failing
    h.worker.BACKOFF_S = (0.05,)
    h.worker.set_motion_gate(False, 2.0)
    h.worker.start()
    try:
        assert wait_until(lambda: bool(h.worker.stats.extra.get("backend_failing")))
        assert h.recorder.wait_for(
            lambda: any(
                (s.last_error or "").startswith("Vision backend keeps failing")
                and "simulated inference failure" in (s.last_error or "")
                for s in h.recorder.stats
            )
        )
        # The camera is released while the worker backs off.
        assert h.recorder.wait_for(
            lambda: any(
                not s.camera_open and s.extra.get("backend_failing") for s in h.recorder.stats
            )
        )
        assert h.sources[0].release_calls >= 1
        # The error is sticky: reopening the camera does not clear it.
        assert wait_until(lambda: len(h.sources) >= 2)
        with h.recorder.cond:
            reopened = [
                s for s in h.recorder.stats if s.camera_open and s.extra.get("camera_opens", 0) > 1
            ]
        assert reopened
        assert all(s.last_error for s in reopened)
        assert h.recorder.count() == 0
    finally:
        h.worker.stop()


def test_backend_failure_clears_once_frames_are_analysed_again() -> None:
    h = Harness()
    h.backend_fail_times = 3  # the first backend fails three times in a row
    h.worker.BACKOFF_S = (0.01,)
    h.worker.set_motion_gate(False, 2.0)
    h.worker.start()
    try:
        assert h.recorder.wait_for(lambda: len(h.recorder.observations) >= 2)
        assert wait_until(lambda: h.worker.stats.last_error is None)
        assert h.worker.stats.extra["backend_failing"] is False
        assert any(s.extra.get("backend_failing") for s in h.recorder.stats)
    finally:
        h.worker.stop()


# ------------------------------------------------------------- blinks and gate
class ScriptedBackend(FakeBackend):
    """Reports a blink for the first ``blinks`` analysed frames."""

    def __init__(self, blinks: int = 0, face_count: int = 1) -> None:
        super().__init__()
        self.blinks = blinks
        self.face_count = face_count

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Observation:
        obs = super().process(frame_bgr, timestamp)
        blink = self.process_calls <= self.blinks
        if self.face_count == 0:
            return Observation(timestamp=timestamp, face_count=0, frame_size=obs.frame_size)
        return dataclasses.replace(obs, blink=blink, face_count=self.face_count)


def _run_worker(
    backend: VisionBackend, frame_fn: Callable[[int], np.ndarray], n: int, *, gate: bool = True
) -> list[Observation]:
    recorder = Recorder()
    worker = VisionWorker(
        lambda: FakeSource(frame_fn=frame_fn), lambda: backend, recorder.on_observation
    )
    worker.set_interval(0.002)
    worker.set_motion_gate(gate, 2.0)
    worker.start()
    try:
        assert recorder.wait_for(lambda: len(recorder.observations) >= n)
    finally:
        worker.stop()
    with recorder.cond:
        return list(recorder.observations)


def test_gate_never_holds_a_blink() -> None:
    """After an analysed blink the next frame is analysed even if it looks the same."""
    backend = ScriptedBackend(blinks=3)
    obs = _run_worker(backend, lambda _n: np.full((48, 64, 3), 100, np.uint8), 8)
    assert [o.skipped for o in obs[:4]] == [False, False, False, False]
    assert not obs[3].blink
    assert all(o.skipped for o in obs[4:8])  # eyes open: the gate holds again


# ------------------------------------------------------------------ blindness
def _dark(_n: int) -> np.ndarray:
    return np.full((48, 64, 3), 3, np.uint8)


def _textured(_n: int) -> np.ndarray:
    rng = np.random.default_rng(5)
    return cv2.resize(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), (64, 48))


@pytest.mark.parametrize("gate", [True, False])
def test_dark_featureless_frames_without_a_face_are_blind(gate: bool) -> None:
    obs = _run_worker(ScriptedBackend(face_count=0), _dark, 4, gate=gate)
    assert all(o.blind for o in obs)
    assert not any(o.face_present for o in obs)
    if gate:
        assert any(o.skipped for o in obs)  # skipped copies keep the flag


def test_ordinary_empty_scene_is_not_blind() -> None:
    obs = _run_worker(ScriptedBackend(face_count=0), _textured, 3)
    assert not any(o.blind for o in obs)


def test_a_face_in_the_dark_is_not_blind() -> None:
    obs = _run_worker(ScriptedBackend(face_count=1), _dark, 3)
    assert not any(o.blind for o in obs)
    assert all(o.face_present for o in obs)


def test_stats_published_at_most_once_per_second(harness: Harness) -> None:
    harness.worker.set_interval(0.002)
    harness.worker.set_motion_gate(False, 2.0)
    harness.worker.start()
    assert harness.recorder.wait_for(lambda: len(harness.recorder.observations) >= 12)
    # One forced publish when the camera opened; routine ones are 1 s apart.
    assert len(harness.recorder.stats) <= 2


# ------------------------------------------------------------ review hardening
def test_switching_on_forgets_the_last_error_without_publishing() -> None:
    """A camera that fails the same way after a pause is reported again."""
    recorder = Recorder()
    calls: list[int] = []
    release = threading.Event()
    message = "Camera 0 could not be opened"

    def factory() -> FakeSource:
        calls.append(1)
        if len(calls) > 1:
            release.wait(TIMEOUT)  # hold the next attempt until the test is ready
        raise CameraError(message)

    worker = VisionWorker(factory, FakeBackend, recorder.on_observation, recorder.on_stats)
    worker.BACKOFF_S = FAST_BACKOFF
    worker.start()
    try:
        assert wait_until(lambda: worker.stats.last_error == message)
        worker.set_active(False)
        with recorder.cond:
            published = len(recorder.stats)
        worker.set_active(True)
        assert worker.stats.last_error is None  # forgotten at once ...
        with recorder.cond:
            assert len(recorder.stats) == published  # ... without a "healthy" report
        release.set()
        # The same failure again is news again.
        assert recorder.wait_for(lambda: sum(s.last_error == message for s in recorder.stats) >= 2)
    finally:
        release.set()
        worker.stop()


def test_reconfigured_backend_identity_is_unknown_until_it_exists(harness: Harness) -> None:
    harness.worker.start()
    assert wait_until(lambda: harness.worker.backend_info == ("fake", "fake-1"))
    release = threading.Event()

    class NewBackend(FakeBackend):
        name: ClassVar[str] = "new"
        feature_version: ClassVar[str] = "new-1"

    def factory() -> FakeBackend:
        release.wait(TIMEOUT)
        return NewBackend()

    harness.worker.reconfigure(backend_factory=factory)
    # The old backend's identity must not pass for the new one's meanwhile.
    assert harness.worker.backend_info is None
    release.set()
    assert wait_until(lambda: harness.worker.backend_info == ("new", "new-1"))
    harness.worker.reconfigure(source_factory=harness.make_source)  # camera only
    assert harness.worker.backend_info == ("new", "new-1")


class _DelayedLock:
    """A lock whose acquisition by the thread that created it can be delayed.

    Stands in for the worker's statistics lock to widen a race window on purpose.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner = threading.get_ident()
        self.delay = 0.0

    def __enter__(self) -> bool:
        if self.delay and threading.get_ident() == self._owner:
            time.sleep(self.delay)
        return self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


def test_the_worker_wakes_with_the_last_error_already_forgotten() -> None:
    """Regression: set_active(True) forgot the error only after waking the thread.

    A camera failing at once with the old message was then compared with the old
    error, found not to be news (nothing published), and erased right after: the
    failure was lost until the next retry.
    """
    recorder = Recorder()
    message = "Camera 0 could not be opened"
    seen: list[str | None] = []  # the error the worker compares with, per attempt
    holder: list[VisionWorker] = []

    def factory() -> FakeSource:
        seen.append(holder[0]._stats.last_error)
        raise CameraError(message)

    worker = VisionWorker(factory, FakeBackend, recorder.on_observation, recorder.on_stats)
    holder.append(worker)
    slow = _DelayedLock()
    worker._stats_lock = slow  # type: ignore[assignment]
    worker.BACKOFF_S = (TIMEOUT * 10,)  # one attempt per activation
    worker.start()
    try:
        assert wait_until(lambda: worker.stats.last_error == message)
        worker.set_active(False)
        slow.delay = 0.3  # time enough for the thread to wake and fail again
        worker.set_active(True)
        slow.delay = 0.0
        assert wait_until(lambda: len(seen) >= 2)
        assert seen[1] is None
        assert recorder.wait_for(lambda: sum(s.last_error == message for s in recorder.stats) >= 2)
    finally:
        worker.stop()


def test_a_backend_built_by_a_replaced_factory_does_not_report_its_identity(
    harness: Harness,
) -> None:
    """A creation still running when reconfigure() replaced its factory."""

    class OldBackend(FakeBackend):
        name: ClassVar[str] = "old"
        feature_version: ClassVar[str] = "old-1"

    class NewBackend(FakeBackend):
        name: ClassVar[str] = "new"
        feature_version: ClassVar[str] = "new-1"

    old_entered, old_release = threading.Event(), threading.Event()
    new_entered, new_release = threading.Event(), threading.Event()

    def old_factory() -> FakeBackend:
        old_entered.set()
        old_release.wait(TIMEOUT)
        return OldBackend()

    def new_factory() -> FakeBackend:
        new_entered.set()
        new_release.wait(TIMEOUT)
        return NewBackend()

    harness.worker.start()
    assert wait_until(lambda: harness.worker.backend_info == ("fake", "fake-1"))
    try:
        harness.worker.reconfigure(backend_factory=old_factory)
        assert old_entered.wait(TIMEOUT)
        harness.worker.reconfigure(backend_factory=new_factory)  # while "old" is built
        old_release.set()
        assert new_entered.wait(TIMEOUT)
        # "old" was built after the second reconfigure: it is not what comes next.
        assert harness.worker.backend_info is None
        new_release.set()
        assert wait_until(lambda: harness.worker.backend_info == ("new", "new-1"))
    finally:
        old_release.set()
        new_release.set()
