"""Synthetic webcam features for the gaze tests (not a test module itself).

A user ~65 cm from 24" 1080p panels (53 cm wide). The head turns part of the way
towards the target (a random share, as people do) and the eyes do the rest; the
webcam reports head pose, head position and iris offsets like the face-mesh
backend. Flat screens make pixels ~ tan(angle): the mapping is non-linear.
"""

from __future__ import annotations

import numpy as np

from eye_tracker.gaze.calibration import CalibrationSample, make_plan
from eye_tracker.types import Monitor, Rect

PX_CM = 53.0 / 1920
FEATURE_NAMES = ("yaw", "pitch", "roll", "tx", "ty", "tz", "iris_h", "iris_v")
#: The gaze-direction features (head rotation and iris offsets).
GAZE = (0, 1, 6, 7)
NOISE_SCALE = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 0.01, 0.006])

TWO = [
    Monitor(0, "left", Rect(0, 0, 1920, 1080), primary=True),
    Monitor(1, "right", Rect(1920, 0, 1920, 1080)),
]
THREE = [Monitor(i, f"m{i}", Rect(1920 * (i - 1), 0, 1920, 1080)) for i in range(3)]
STACKED = [
    Monitor(0, "top", Rect(0, -1080, 1920, 1080)),
    Monitor(1, "bottom", Rect(0, 0, 1920, 1080)),
]


def synth_features(
    points: np.ndarray,
    rng: np.random.Generator,
    noise: float = 0.0,
    camera_x: float = 1920.0,
    head_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    head_share: float = 0.55,
    pitch_share: float = 0.8,
) -> np.ndarray:
    """8-D features (yaw, pitch, roll, tx, ty, tz, iris_h, iris_v) for gaze ``points``.

    ``head_offset`` (cm; right, down, back) moves the user's head away from where
    it was while calibrating: ``(0, 8, 0)`` sits 8 cm lower, ``(0, 0, 12)`` leans
    back. ``head_share`` is the mean share of a gaze shift the head makes (the
    eyes do the rest), ``pitch_share`` scales it for vertical shifts: changing
    them models a user who moves the head more or less than while calibrating.
    None of these change the random stream, so equal seeds stay comparable.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    target_x = (pts[:, 0] - camera_x) * PX_CM
    target_y = pts[:, 1] * PX_CM + 2.0
    dx, dy, dz = head_offset
    hx = rng.normal(0, 2.5, n) + dx
    hy = rng.normal(0, 1.5, n) + dy
    hz = 65.0 + rng.normal(0, 3.0, n) + dz
    gaze_yaw = np.degrees(np.arctan2(target_x - hx, hz))
    gaze_pitch = np.degrees(np.arctan2(target_y - hy, hz))
    share = np.clip(head_share + rng.normal(0, 0.12, n), 0.1, 0.95)
    yaw = share * gaze_yaw + rng.normal(0, 2.0, n)
    pitch = pitch_share * share * gaze_pitch + rng.normal(0, 1.5, n)
    iris_h = 0.5 + 0.42 * np.sin(np.radians(gaze_yaw - yaw))
    iris_v = 0.05 + 0.20 * np.sin(np.radians(gaze_pitch - pitch))
    roll = rng.normal(0, 1.5, n)
    feats = np.column_stack([yaw, pitch, roll, hx, hy, -hz, iris_h, iris_v])
    return feats + rng.normal(size=feats.shape) * NOISE_SCALE * noise


def grid_dataset(
    monitors: list[Monitor],
    rng: np.random.Generator,
    noise: float = 0.0,
    per_point: int = 20,
    camera_x: float = 1920.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibration-like data: a 3x3 grid per monitor, ``per_point`` samples each."""
    points: list[tuple[float, float]] = []
    groups: list[int] = []
    for m in monitors:
        for ny in (0.1, 0.5, 0.9):
            for nx in (0.1, 0.5, 0.9):
                points += [m.rect.denormalize(nx, ny)] * per_point
                groups += [len(groups) // per_point] * per_point
    P = np.array(points)
    return synth_features(P, rng, noise, camera_x), P, np.array(groups)


def calibration_samples(
    monitors: list[Monitor],
    rng: np.random.Generator,
    noise: float = 0.0,
    *,
    per_point: int = 20,
    camera_x: float = 1920.0,
    points_per_monitor: int = 9,
) -> list[CalibrationSample]:
    """Samples as the calibration collector would record them for ``make_plan(monitors)``."""
    samples = []
    for t in make_plan(monitors, points_per_monitor):
        feats = synth_features(np.tile((t.x, t.y), (per_point, 1)), rng, noise, camera_x)
        samples += [CalibrationSample(f, t.x, t.y, t.monitor_index, t.point_id) for f in feats]
    return samples


def random_points(
    monitors: list[Monitor], rng: np.random.Generator, n: int, margin: float = 0.05
) -> np.ndarray:
    """Uniform points on the monitors, avoiding the outer ``margin`` (5 %: like real use)."""
    idx = rng.integers(0, len(monitors), n)
    nx, ny = rng.uniform(margin, 1 - margin, n), rng.uniform(margin, 1 - margin, n)
    return np.array(
        [monitors[i].rect.denormalize(a, b) for i, a, b in zip(idx, nx, ny, strict=True)]
    )


def below_points(
    monitors: list[Monitor], rng: np.random.Generator, n: int, cm: float
) -> np.ndarray:
    """Points ``cm`` below the bottom edge of the monitors (a phone or papers on the desk)."""
    left = min(m.rect.x for m in monitors)
    right = max(m.rect.right for m in monitors)
    bottom = max(m.rect.bottom for m in monitors)
    return np.column_stack([rng.uniform(left, right, n), np.full(n, bottom + cm / PX_CM)])


def pixel_errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.hypot(pred[:, 0] - truth[:, 0], pred[:, 1] - truth[:, 1])
