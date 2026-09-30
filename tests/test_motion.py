"""Tests for eye_tracker.vision.motion.MotionGate."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from eye_tracker.vision import motion
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


def eyes_frame(
    base: np.ndarray,
    *,
    iris_dx: float = 0.0,
    closed: bool = False,
    noise: float = 1.0,
    face_w: int = 140,
    seed: int = 0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """A face with two eyes (sclera, dark iris) on ``base``, plus its face box.

    The eyes cover only a few percent of the eye band, like in a webcam frame.
    """
    frame = base.copy()
    face_h = round(face_w * 1.25)
    x0, y0 = 320 - face_w // 2, 240 - face_h // 2
    cv2.ellipse(frame, (320, 240), (face_w // 2, face_h // 2), 0, 0, 360, (150, 160, 190), -1)
    eye_w = 0.22 * face_w
    eye_h = 0.30 * eye_w
    cy = y0 + 0.40 * face_h
    for cx in (320 - 0.22 * face_w, 320 + 0.22 * face_w):
        if closed:
            p0 = (round(cx - eye_w / 2), round(cy))
            p1 = (round(cx + eye_w / 2), round(cy))
            cv2.line(frame, p0, p1, (90, 90, 110), 2, cv2.LINE_AA)
            continue
        axes = (round(eye_w / 2), round(eye_h / 2))
        cv2.ellipse(frame, (round(cx), round(cy)), axes, 0, 0, 360, (190, 190, 190), -1)
        # Sub-pixel iris position (4 fractional bits).
        centre = (round((cx + iris_dx) * 16), round(cy * 16))
        cv2.circle(frame, centre, round(0.2 * eye_w * 16), (60, 50, 40), -1, cv2.LINE_AA, 4)
    rng = np.random.default_rng(seed)
    noisy = cv2.GaussianBlur(
        frame.astype(np.float32) + rng.normal(0, noise, frame.shape), (0, 0), 0.8
    )
    box = (x0 / 640, y0 / 480, face_w / 640, face_h / 480)
    return np.clip(noisy, 0, 255).astype(np.uint8), box


@pytest.mark.parametrize("face_w", [90, 140, 220])
def test_eye_only_glance_passes_the_gate(base: np.ndarray, face_w: int) -> None:
    """A 4 px iris shift (an eye-only glance at another monitor) must be analysed."""
    reference, box = eyes_frame(base, face_w=face_w, seed=1)
    glance, _ = eyes_frame(base, face_w=face_w, iris_dx=4.0, seed=2)
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(reference, 0.0, box)
    assert gate.should_process(glance, 0.1)
    # The same change is invisible to the whole-frame thumbnail alone.
    blind_to_eyes = MotionGate(threshold=2.0)
    blind_to_eyes.mark_processed(reference, 0.0)
    assert not blind_to_eyes.should_process(glance, 0.1)


@pytest.mark.parametrize("noise", [1.0, 3.0, 6.0])
def test_noisy_still_face_is_skipped(base: np.ndarray, noise: float) -> None:
    """Camera noise (up to a dim-light sigma of 6) must not defeat the gate."""
    reference, box = eyes_frame(base, face_w=90, noise=noise, seed=1)
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(reference, 0.0, box)
    for seed in range(2, 6):
        still, _ = eyes_frame(base, face_w=90, noise=noise, seed=seed)
        assert not gate.should_process(still, 0.1), seed
        assert gate.last_motion < 2.0


def test_eyes_reopening_passes_the_gate(base: np.ndarray) -> None:
    closed, box = eyes_frame(base, closed=True, seed=1)
    reopened, _ = eyes_frame(base, seed=2)
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(closed, 0.0, box)
    assert gate.should_process(reopened, 0.1)


def test_band_score_is_reported_on_the_threshold_scale(base: np.ndarray) -> None:
    reference, box = eyes_frame(base, seed=1)
    glance, _ = eyes_frame(base, iris_dx=4.0, seed=1)
    gate = MotionGate(threshold=2.0)
    gate.mark_processed(reference, 0.0, box)
    gate.should_process(glance, 0.1)
    roi_ref = motion._roi_thumb(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), motion._eye_band(box))
    roi_new = motion._roi_thumb(cv2.cvtColor(glance, cv2.COLOR_BGR2GRAY), motion._eye_band(box))
    assert roi_ref is not None
    assert roi_new is not None
    band = motion._top_abs_diff(roi_new, roi_ref)
    assert gate.last_motion == pytest.approx(band / motion.BAND_THRESHOLD_FACTOR)


def test_top_abs_diff_is_robust_to_where_the_change_is() -> None:
    a = np.zeros((16, 32), np.float32)
    b = a.copy()
    b[2:4, 3:5] = 100.0  # 4 of 512 cells change
    small = motion._top_abs_diff(a, b)
    assert small == pytest.approx(100.0 * 4 / 51)  # top 10 % = 51 cells
    assert motion._mean_abs_diff(a, b) == pytest.approx(100.0 * 4 / 512)
    moved = np.roll(b, 10, axis=1)
    assert motion._top_abs_diff(a, moved) == pytest.approx(small)


# -------------------------------------------------------------------- blindness
def test_thumbnail_formats(base: np.ndarray) -> None:
    for frame in (
        base,
        cv2.cvtColor(base, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(base, cv2.COLOR_BGR2BGRA),
    ):
        thumb = motion.thumbnail(frame)
        assert thumb.shape == (24, 32)
        assert thumb.dtype == np.float32


def test_covered_lens_is_blind() -> None:
    black = np.zeros((480, 640, 3), np.uint8)
    assert motion.is_blind(motion.thumbnail(black))
    # A closed shutter at full sensor gain: dark, noisy, but featureless.
    rng = np.random.default_rng(0)
    shutter = np.clip(rng.normal(18, 8, (480, 640, 3)), 0, 255).astype(np.uint8)
    assert motion.is_blind(motion.thumbnail(shutter))


def test_a_hand_over_the_lens_is_blind() -> None:
    yy = np.linspace(0, 1, 480, dtype=np.float32)[:, None, None]
    glow = (np.ones((480, 640, 3), np.float32) * (8 + 14 * yy) * (0.6, 0.8, 1.0)).astype(np.uint8)
    assert motion.is_blind(motion.thumbnail(glow))


def test_dim_scene_with_structure_is_not_blind(base: np.ndarray) -> None:
    dim = (base.astype(np.float32) * 0.2).astype(np.uint8)  # mean ~25, but textured
    assert not motion.is_blind(motion.thumbnail(dim))


def test_bright_flat_scene_is_not_blind() -> None:
    """A white wall behind an empty chair means 'nobody here', not 'cannot tell'."""
    wall = np.full((480, 640, 3), 200, np.uint8)
    assert not motion.is_blind(motion.thumbnail(wall))


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
