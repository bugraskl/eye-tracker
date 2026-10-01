"""Tests for the pane worker thread (panes/worker.py). Providers are fakes."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from PySide6.QtCore import QCoreApplication

from eye_tracker.panes.registry import PaneRegistry
from eye_tracker.panes.types import Pane, PaneError
from eye_tracker.panes.worker import FAILURE_LIMIT, DetectResult, FocusResult, PaneWorker
from eye_tracker.types import AppIdentity, Rect, WindowRef
from panes_fakes import RecordingProvider, two_panes

APP = AppIdentity("kitty", "kitty")
REF = WindowRef(handle=7, pid=70)
PANE = Pane("%1", Rect(960, 0, 960, 1080), False, "fake")


class Clock:
    def __init__(self, t: float = 50.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def sync_worker(
    *providers: RecordingProvider, clock: Clock | None = None
) -> tuple[PaneWorker, list[Any], list[Any]]:
    worker = PaneWorker(PaneRegistry(providers), clock=clock or Clock(), synchronous=True)
    detected: list[Any] = []
    focused: list[Any] = []
    worker.detected.connect(detected.append)
    worker.focused.connect(focused.append)
    return worker, detected, focused


def test_detect_stamps_the_snapshot_with_the_worker_clock(qapp: Any) -> None:
    provider = RecordingProvider(snapshot=lambda ref: two_panes(ref.handle, taken_at=-1.0))
    worker, detected, _ = sync_worker(provider, clock=Clock(123.0))
    request = worker.request_detect(REF, APP)
    [result] = detected
    assert isinstance(result, DetectResult)
    assert result.request_id == request
    assert result.window_handle == 7
    assert result.snapshot is not None
    assert result.snapshot.taken_at == 123.0
    assert [p.id for p in result.snapshot.panes] == ["%0", "%1"]


def test_first_provider_with_panes_wins(qapp: Any) -> None:
    empty = RecordingProvider("empty")
    found = RecordingProvider("found", snapshot=lambda ref: two_panes(ref.handle, provider="found"))
    later = RecordingProvider("later", snapshot=lambda ref: two_panes(ref.handle))
    worker, detected, _ = sync_worker(empty, found, later)
    worker.request_detect(REF, APP)
    assert detected[0].snapshot.provider == "found"
    assert ("detect", 7) in empty.calls
    assert ("detect", 7) not in later.calls


def test_focus_goes_to_the_panes_provider(qapp: Any) -> None:
    fake = RecordingProvider("fake")
    other = RecordingProvider("other")
    worker, _, focused = sync_worker(other, fake)
    worker.request_focus(REF, PANE)
    assert focused == [FocusResult(7, "%1", True)]
    assert fake.calls == [("focus", "%1")]
    assert other.calls == []
    worker.request_focus(REF, Pane("%2", PANE.rect, False, "gone"))
    assert focused[-1].ok is False


def test_a_provider_failing_three_times_in_a_row_is_switched_off(qapp: Any) -> None:
    flaky = RecordingProvider("flaky", fail=PaneError("tmux did not answer within 1 s"))
    backup = RecordingProvider("backup", snapshot=lambda ref: two_panes(ref.handle))
    worker, detected, _ = sync_worker(flaky, backup)
    for _ in range(FAILURE_LIMIT + 2):
        worker.request_detect(REF, APP)
    assert sum(1 for c in flaky.calls if c[0] == "detect") == FAILURE_LIMIT
    assert worker.disabled_providers == frozenset({"flaky"})
    assert all(r.snapshot is not None for r in detected)  # the next provider answered
    worker.request_focus(REF, Pane("%1", PANE.rect, False, "flaky"))
    assert ("focus", "%1") not in flaky.calls


def test_a_success_resets_the_failure_count(qapp: Any) -> None:
    provider = RecordingProvider("p", snapshot=lambda ref: two_panes(ref.handle))
    worker, _, _ = sync_worker(provider)
    for _ in range(5):
        provider.fail = PaneError("busy")
        worker.request_detect(REF, APP)
        worker.request_detect(REF, APP)
        provider.fail = None
        worker.request_detect(REF, APP)
    assert worker.disabled_providers == frozenset()


def test_slow_calls_count_as_failures(qapp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = RecordingProvider("slow", snapshot=lambda ref: two_panes(ref.handle))
    worker = PaneWorker(PaneRegistry([provider]), synchronous=True, call_timeout_s=0.5)
    results: list[Any] = []
    worker.detected.connect(results.append)
    ticks = iter([0.0, 1.0] * FAILURE_LIMIT)
    monkeypatch.setattr("eye_tracker.panes.worker.time.monotonic", lambda: next(ticks))
    for _ in range(FAILURE_LIMIT):
        worker.request_detect(REF, APP)
    assert all(r.snapshot is None for r in results)  # late answers are dropped
    assert worker.disabled_providers == frozenset({"slow"})


def test_denied_app_gets_an_empty_answer(qapp: Any) -> None:
    provider = RecordingProvider(snapshot=lambda ref: two_panes(ref.handle))
    worker, detected, _ = sync_worker(provider)
    worker.request_detect(REF, AppIdentity("code", "Chrome_WidgetWin_1"))
    assert detected[0].snapshot is None
    assert provider.calls == []


# ----------------------------------------------------------------- the thread
def _wait(predicate: Any, timeout: float = 3.0) -> bool:
    """Run the Qt event loop until ``predicate`` holds (results arrive as queued calls)."""
    app = QCoreApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_thread_answers_and_stops_cleanly(qapp: Any) -> None:
    release = threading.Event()
    threads: list[str] = []

    def blocking(ref: WindowRef) -> Any:
        threads.append(threading.current_thread().name)
        release.wait(2.0)
        return two_panes(ref.handle)

    provider = RecordingProvider(snapshot=blocking)
    worker = PaneWorker(PaneRegistry([provider]))
    lock = threading.Lock()
    detected: list[Any] = []
    focused: list[Any] = []

    delivered: list[str] = []

    def on_detect(result: Any) -> None:
        with lock:
            delivered.append(threading.current_thread().name)
            detected.append(result)

    def on_focus(result: Any) -> None:
        with lock:
            focused.append(result)

    worker.detected.connect(on_detect)
    worker.focused.connect(on_focus)
    worker.start()
    try:
        first = worker.request_detect(REF, APP)
        assert _wait(lambda: len(threads) == 1)  # running, blocked in the provider
        worker.request_detect(REF, APP)  # replaced by the next one before it starts
        last = worker.request_detect(REF, APP)
        worker.request_focus(REF, PANE)  # goes before the waiting detection
        release.set()
        assert _wait(lambda: len(detected) == 2 and len(focused) == 1)
        assert [r.request_id for r in detected] == [first, last]
        assert provider.calls.index(("focus", "%1")) < len(provider.calls) - 1
        assert threads[0] == "eye-tracker-panes"
        assert set(delivered) == {threading.main_thread().name}
    finally:
        worker.stop()
    assert not worker.running
    worker.request_detect(REF, APP)  # after stop: dropped, no thread
    _wait(lambda: False, timeout=0.05)
    assert len(detected) == 2


def test_stop_does_not_wait_forever_for_a_hung_provider(qapp: Any) -> None:
    hang = threading.Event()
    provider = RecordingProvider(snapshot=lambda ref: hang.wait(5.0) and None)
    worker = PaneWorker(PaneRegistry([provider]))
    worker.start()
    worker.request_detect(REF, APP)
    assert _wait(lambda: bool(provider.calls))
    started = time.monotonic()
    worker.stop(timeout=0.2)
    assert time.monotonic() - started < 1.0
    hang.set()


# ------------------------------------------------------------ closing providers
def test_replaced_and_final_providers_are_closed_on_the_worker_thread(qapp: Any) -> None:
    old = RecordingProvider("old", snapshot=lambda ref: two_panes(ref.handle))
    new = RecordingProvider("new", snapshot=lambda ref: two_panes(ref.handle))
    worker = PaneWorker(PaneRegistry([old]))
    worker.start()
    try:
        worker.request_detect(REF, APP)
        assert _wait(lambda: bool(old.calls))
        worker.set_registry(PaneRegistry([new]))
        assert _wait(lambda: old.closed_on == ["eye-tracker-panes"])
        assert new.closed_on == []
    finally:
        worker.stop()
    assert new.closed_on == ["eye-tracker-panes"]  # before the thread ended
    assert old.closed_on == ["eye-tracker-panes"]  # once


def test_request_stop_returns_at_once_and_the_thread_closes_up(qapp: Any) -> None:
    release = threading.Event()
    provider = RecordingProvider(snapshot=lambda ref: release.wait(5.0) and None)
    worker = PaneWorker(PaneRegistry([provider]))
    results: list[Any] = []
    worker.detected.connect(results.append)
    worker.start()
    worker.request_detect(REF, APP)
    assert _wait(lambda: bool(provider.calls))  # busy in the provider
    started = time.monotonic()
    worker.request_stop()
    assert time.monotonic() - started < 0.1  # no join
    assert provider.closed_on == []  # still in use on the thread
    release.set()
    assert _wait(lambda: provider.closed_on == ["eye-tracker-panes"])
    assert _wait(lambda: not worker.running)
    assert results == []  # nothing is delivered after a stop


def test_synchronous_worker_closes_on_the_calling_thread(qapp: Any) -> None:
    old = RecordingProvider("old")
    new = RecordingProvider("new")
    worker = PaneWorker(PaneRegistry([old]), synchronous=True)
    worker.set_registry(PaneRegistry([new]))
    assert old.closed_on == [threading.current_thread().name]
    worker.stop()
    assert new.closed_on == [threading.current_thread().name]


def test_a_failing_close_is_contained(qapp: Any) -> None:
    class Broken(RecordingProvider):
        def close(self) -> None:
            raise OSError("CoUninitialize said no")

    after = RecordingProvider("after")
    registry = PaneRegistry([Broken("broken"), after])
    registry.close()
    assert after.closed_on == [threading.current_thread().name]
