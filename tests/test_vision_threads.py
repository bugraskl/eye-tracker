"""Tests for eye_tracker.vision.threads (OpenCV thread-pool cap)."""

from __future__ import annotations

import cv2
import pytest

from eye_tracker.vision import threads


class FakePool:
    def __init__(self, size: int) -> None:
        self.size = size
        self.calls: list[int] = []

    def get(self) -> int:
        return self.size

    def set(self, n: int) -> None:
        self.calls.append(n)
        self.size = n


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch) -> FakePool:
    fake = FakePool(16)
    monkeypatch.setattr(threads.cv2, "getNumThreads", fake.get)
    monkeypatch.setattr(threads.cv2, "setNumThreads", fake.set)
    return fake


def test_caps_a_large_pool(pool: FakePool) -> None:
    assert threads.limit_opencv_threads() == threads.MAX_OPENCV_THREADS == 1
    assert pool.calls == [1]
    assert threads.limit_opencv_threads() == 1  # already capped: nothing to do
    assert pool.calls == [1]


def test_never_raises_the_limit(pool: FakePool) -> None:
    pool.size = 2
    assert threads.limit_opencv_threads(4) == 2
    assert pool.calls == []


def test_invalid_limit_means_one(pool: FakePool) -> None:
    assert threads.limit_opencv_threads(0) == 1


def test_builds_without_a_pool_are_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> int:
        raise cv2.error("no parallel framework")

    monkeypatch.setattr(threads.cv2, "getNumThreads", broken)
    assert threads.limit_opencv_threads() == 1


def test_backends_cap_the_pool(pool: FakePool) -> None:
    from eye_tracker.vision.backends.lite_backend import LiteBackend

    LiteBackend().close()
    assert pool.size == 1
