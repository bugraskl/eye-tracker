"""Tests for eye_tracker.vision.motion.MotionGate."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from eye_tracker.vision.motion import MotionGate

FACE_BOX = (0.3, 0.2, 0.4, 0.5)


@pytest.fixture
def base() -> np.ndarray:
    """A textured 640x480 BGR frame (smooth, so small shifts matter little)."""
    rng = np.random.default_rng(1234)
    noise = rng.integers(0, 255, size=(24, 32, 3), dtype=np.uint8)
    return cv2.resize(noise, (640, 480), interpolation=cv2.INTER_CUBIC)


def with_patches(frame: np.ndarray, value: int, size: int = 16) -> np.ndarray:
    """Paint two small squares where the eyes of FACE_BOX would be (like moving irises)."""
    out = frame.copy()
    # Eye band of FACE_BOX spans x 205..435, y 132..240 in a 640x480 frame.
    for cx in (270, 370):
        out[180 : 180 + size, cx : cx + size] = value
    return out


def test_first_frame_is_always_processed(base: np.ndarray) -> None:
    gate = MotionGate()
    assert gate.should_process(base, 0.0)


def test_identical_frame_is_skipped(base: np.ndarray) -> None:
    gate = MotionGate()
    gate.mark_processed(base, 0.0)
    assert not gate.should_process(base.copy(), 0.1)
    assert gate.last_motion == 0.0


def test_global_change_is_processed(base: np.ndarray) -> None:
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(base, 0.0)
    brighter = cv2.add(base, np.full_like(base, 10))
    assert gate.should_process(brighter, 0.1)
    assert gate.last_motion >= 2.0


def test_sensor_noise_is_ignored(base: np.ndarray) -> None:
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(base, 0.0)
    rng = np.random.default_rng(7)
    noise = rng.integers(-4, 5, size=base.shape)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert not gate.should_process(noisy, 0.1)
    assert gate.last_motion < 1.0


def test_max_skip_forces_refresh(base: np.ndarray) -> None:
    gate = MotionGate(max_skip_s=2.0)
    gate.mark_processed(base, 10.0)
    assert not gate.should_process(base, 11.9)
    assert gate.should_process(base, 12.0)


def test_reference_is_last_processed_frame(base: np.ndarray) -> None:
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(base, 0.0)
    # Slow drift: each step is small, but it accumulates against the reference.
    step = [cv2.add(base, np.full_like(base, k)) for k in (1, 2, 3)]
    assert not gate.should_process(step[0], 0.1)
    assert gate.should_process(step[2], 0.2)
    gate.mark_processed(step[2], 0.2)
    assert not gate.should_process(step[2], 0.3)


def test_reset_forgets_reference(base: np.ndarray) -> None:
    gate = MotionGate()
    gate.mark_processed(base, 0.0)
    gate.reset()
    assert gate.should_process(base, 0.1)
    assert gate.last_motion == 0.0


def test_threshold_is_adjustable(base: np.ndarray) -> None:
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(base, 0.0)
    brighter = cv2.add(base, np.full_like(base, 10))
    gate.threshold = 50.0
    assert not gate.should_process(brighter, 0.1)
    gate.threshold = 5.0
    assert gate.should_process(brighter, 0.1)


def test_eye_band_detects_small_eye_movement(base: np.ndarray) -> None:
    before = with_patches(base, 40)
    after = with_patches(base, 220)

    without_face = MotionGate(threshold=2.0)
    without_face.mark_processed(before, 0.0)
    assert not without_face.should_process(after, 0.1)
    global_motion = without_face.last_motion

    with_face = MotionGate(threshold=2.0)
    with_face.mark_processed(before, 0.0, FACE_BOX)
    assert with_face.should_process(after, 0.1)
    assert with_face.last_motion > global_motion


def test_change_outside_eye_band_uses_global_threshold(base: np.ndarray) -> None:
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(base, 0.0, FACE_BOX)
    changed = base.copy()
    changed[400:416, 20:36] = 255  # small change far from the face
    assert not gate.should_process(changed, 0.1)


def test_face_box_at_the_edge_is_tolerated(base: np.ndarray) -> None:
    gate = MotionGate()
    gate.mark_processed(base, 0.0, (0.95, 0.95, 0.2, 0.2))  # band mostly outside
    assert not gate.should_process(base, 0.1)
    gate.mark_processed(base, 0.2, (0.5, 0.5, 0.0, 0.0))  # degenerate box
    assert not gate.should_process(base, 0.3)


def test_face_box_is_forgotten_when_face_disappears(base: np.ndarray) -> None:
    before = with_patches(base, 40)
    after = with_patches(base, 220)
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(before, 0.0, FACE_BOX)
    gate.mark_processed(before, 0.1, None)
    assert not gate.should_process(after, 0.2)


def test_mark_processed_after_should_process_same_frame(base: np.ndarray) -> None:
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(base, 0.0)
    brighter = cv2.add(base, np.full_like(base, 20))
    assert gate.should_process(brighter, 0.1)
    gate.mark_processed(brighter, 0.1, FACE_BOX)
    assert not gate.should_process(brighter.copy(), 0.2)


@pytest.mark.parametrize("kind", ["gray", "bgra", "single-channel"])
def test_other_pixel_formats(base: np.ndarray, kind: str) -> None:
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    frame = {
        "gray": gray,
        "bgra": cv2.cvtColor(base, cv2.COLOR_BGR2BGRA),
        "single-channel": gray[:, :, None],
    }[kind]
    gate = MotionGate()
    assert gate.should_process(frame, 0.0)
    gate.mark_processed(frame, 0.0, FACE_BOX)
    assert not gate.should_process(frame.copy(), 0.1)


def test_frame_size_change_is_handled(base: np.ndarray) -> None:
    gate = MotionGate()
    gate.mark_processed(base, 0.0, FACE_BOX)
    smaller = cv2.resize(base, (320, 240), interpolation=cv2.INTER_AREA)
    assert not gate.should_process(smaller, 0.1)


def test_cost_is_small(base: np.ndarray) -> None:
    gate = MotionGate()
    gate.mark_processed(base, 0.0, FACE_BOX)
    frames = [cv2.add(base, np.full_like(base, k % 2)) for k in range(4)]
    for frame in frames:  # warm-up
        gate.should_process(frame, 0.1)
    runs = 200
    started = time.perf_counter()
    for i in range(runs):
        gate.should_process(frames[i % 4], 0.1)
    per_call_ms = (time.perf_counter() - started) * 1e3 / runs
    # ~0.2 ms on a desktop CPU; generous bound for slow CI runners.
    assert per_call_ms < 2.0
