"""Tests for the vision backends and backend selection.

Tests that need a real face use the ``face_image`` fixture and are skipped unless
``EYE_TRACKER_TEST_FACE`` points at a photo with one frontal face.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from eye_tracker.vision import backends
from eye_tracker.vision.backends import (
    BackendUnavailable,
    VisionBackend,
    available_backends,
    backend_class,
    create_backend,
)
from eye_tracker.vision.backends import mediapipe_backend as mpb
from eye_tracker.vision.backends.mediapipe_backend import (
    EYE_A,
    EYE_B,
    EyeMetrics,
    eye_metrics,
    pose_from_matrix,
)
from eye_tracker.vision.backends.opencv_backend import OpenCVBackend, geometry_features

AVAILABLE = available_backends()
needs_mediapipe = pytest.mark.skipif(
    "mediapipe" not in AVAILABLE, reason="MediaPipe is not installed"
)
needs_opencv = pytest.mark.skipif("opencv" not in AVAILABLE, reason="YuNet is unavailable")

log = logging.getLogger(__name__)


@pytest.fixture(params=AVAILABLE or [pytest.param("none", marks=pytest.mark.skip)])
def backend(request: pytest.FixtureRequest) -> Iterator[VisionBackend]:
    instance = create_backend(request.param)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def mp_backend() -> Iterator[VisionBackend]:
    if "mediapipe" not in AVAILABLE:
        pytest.skip("MediaPipe is not installed")
    instance = create_backend("mediapipe")
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def cv_backend() -> Iterator[VisionBackend]:
    if "opencv" not in AVAILABLE:
        pytest.skip("YuNet is unavailable")
    instance = create_backend("opencv")
    try:
        yield instance
    finally:
        instance.close()


# ------------------------------------------------------------------ selection
def test_available_backends_order() -> None:
    assert set(AVAILABLE) <= {"mediapipe", "opencv"}
    assert [name for name in backends.BACKEND_NAMES if name in AVAILABLE] == AVAILABLE


def test_unknown_backend_raises() -> None:
    with pytest.raises(BackendUnavailable, match="Unknown vision backend"):
        create_backend("tensorflow")
    with pytest.raises(BackendUnavailable):
        backend_class("tensorflow")


@pytest.mark.skipif(not AVAILABLE, reason="no backend available")
def test_auto_prefers_first_available() -> None:
    instance = create_backend("auto")
    try:
        assert instance.name == AVAILABLE[0]
        assert backend_class("auto") is type(instance)
        assert backend_class(" AUTO ").feature_version == instance.feature_version
    finally:
        instance.close()


@needs_opencv
def test_auto_falls_back_when_mediapipe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: object, *args: object, **kwargs: object) -> None:
        raise BackendUnavailable("simulated MediaPipe failure")

    monkeypatch.setattr(mpb.MediaPipeBackend, "__init__", broken)
    instance = create_backend("auto")
    try:
        assert isinstance(instance, OpenCVBackend)
    finally:
        instance.close()


def test_auto_reports_all_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: object, *args: object, **kwargs: object) -> None:
        raise BackendUnavailable(f"{type(self).__name__} broken")

    monkeypatch.setattr(mpb.MediaPipeBackend, "__init__", broken)
    monkeypatch.setattr(OpenCVBackend, "__init__", broken)
    with pytest.raises(BackendUnavailable) as info:
        create_backend("auto")
    assert "MediaPipeBackend broken" in str(info.value)
    assert "OpenCVBackend broken" in str(info.value)


@needs_mediapipe
def test_mediapipe_missing_model(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailable, match="model not found"):
        mpb.MediaPipeBackend(model_path=tmp_path / "missing.task")


def test_opencv_missing_model(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailable, match="model not found"):
        OpenCVBackend(model_path=tmp_path / "missing.onnx")


def test_backend_metadata() -> None:
    assert mpb.MediaPipeBackend.name == "mediapipe"
    assert mpb.MediaPipeBackend.feature_version == "mp-pose-iris-1"
    assert mpb.MediaPipeBackend.feature_names == (
        "yaw",
        "pitch",
        "roll",
        "tx",
        "ty",
        "tz",
        "iris_h",
        "iris_v",
    )
    assert OpenCVBackend.name == "opencv"
    assert OpenCVBackend.feature_version == "yunet-geom-1"
    assert OpenCVBackend.feature_names == (
        "nose_dx",
        "nose_dy",
        "roll",
        "face_cx",
        "face_cy",
        "scale",
    )


# ---------------------------------------------------------- no-face behaviour
def test_blank_frame_has_no_face(backend: VisionBackend) -> None:
    frame = np.full((480, 640, 3), 90, np.uint8)
    obs = backend.process(frame, 1.0)
    assert obs.timestamp == 1.0
    assert obs.face_count == 0
    assert obs.features is None
    assert obs.quality == 0.0
    assert not obs.usable
    assert not obs.face_present
    assert obs.frame_size == (640, 480)
    assert obs.inference_ms > 0.0


def test_repeated_timestamps_are_tolerated(backend: VisionBackend) -> None:
    frame = np.zeros((240, 320, 3), np.uint8)
    for ts in (5.0, 5.0, 4.0, 5.0005):
        assert backend.process(frame, ts).face_count == 0


@pytest.mark.parametrize("kind", ["gray", "bgra", "large"])
def test_other_frame_formats(backend: VisionBackend, kind: str) -> None:
    frame = {
        "gray": np.zeros((240, 320), np.uint8),
        "bgra": np.zeros((240, 320, 4), np.uint8),
        "large": np.zeros((1080, 1920, 3), np.uint8),
    }[kind]
    obs = backend.process(frame, 1.0)
    assert obs.face_count == 0
    assert obs.frame_size == (frame.shape[1], frame.shape[0])


def test_annotate_returns_a_copy(backend: VisionBackend) -> None:
    frame = np.full((240, 320, 3), 60, np.uint8)
    obs = backend.process(frame, 1.0)
    out = backend.annotate(frame, obs)
    assert out.shape == frame.shape
    assert out is not frame
    assert np.all(frame == 60)  # input untouched
    assert np.any(out != 60)  # "no face" label drawn


def test_close_is_idempotent_and_final(backend: VisionBackend) -> None:
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError):
        backend.process(np.zeros((10, 10, 3), np.uint8), 1.0)


@needs_mediapipe
def test_mediapipe_set_max_faces_is_lazy(mp_backend: VisionBackend) -> None:
    assert isinstance(mp_backend, mpb.MediaPipeBackend)
    mp_backend.set_max_faces(2)
    assert mp_backend.max_faces == 2
    assert mp_backend._rebuild
    mp_backend.process(np.zeros((120, 160, 3), np.uint8), 1.0)
    assert not mp_backend._rebuild
    mp_backend.set_max_faces(2)  # unchanged: no rebuild
    assert not mp_backend._rebuild
    mp_backend.set_max_faces(0)
    assert mp_backend.max_faces == 1


# ------------------------------------------------------------- pure geometry
def _rot(axis: str, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def _matrix(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = translation
    return m


def test_pose_from_identity() -> None:
    yaw, pitch, roll, tx, ty, tz = pose_from_matrix(_matrix(np.eye(3), (1.0, 2.0, -60.0)))
    assert (yaw, pitch, roll) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert (tx, ty, tz) == (1.0, 2.0, -60.0)


@pytest.mark.parametrize(("axis", "index"), [("y", 0), ("x", 1), ("z", 2)])
def test_pose_single_axis(axis: str, index: int) -> None:
    for angle in (-25.0, 12.5):
        pose = pose_from_matrix(_matrix(_rot(axis, angle), (0.0, 0.0, -50.0)))
        assert pose[index] == pytest.approx(angle, abs=1e-6)
        others = [pose[i] for i in range(3) if i != index]
        assert others == pytest.approx([0.0, 0.0], abs=1e-6)


def test_pose_accepts_flat_lists() -> None:
    flat = _matrix(_rot("y", 10.0), (0.0, 0.0, -50.0)).ravel().tolist()
    assert pose_from_matrix(flat)[0] == pytest.approx(10.0)


def _eye_landmarks(iris_dx: float = 0.0, iris_dy: float = 0.0, lid_gap: float = 12.0) -> np.ndarray:
    pts = np.zeros((478, 2))
    for (c0, c1, iris, top, bottom), x0 in ((EYE_A, 100.0), (EYE_B, 200.0)):
        pts[c0] = (x0, 100.0)
        pts[c1] = (x0 + 40.0, 100.0)
        pts[iris] = (x0 + 20.0 + iris_dx, 100.0 + iris_dy)
        pts[top] = (x0 + 20.0, 100.0 - lid_gap / 2)
        pts[bottom] = (x0 + 20.0, 100.0 + lid_gap / 2)
    return pts


def test_eye_metrics_centred() -> None:
    metrics = eye_metrics(_eye_landmarks())
    assert metrics == EyeMetrics(iris_h=0.5, iris_v=0.0, openness=0.3)


def test_eye_metrics_iris_moves_same_way_in_both_eyes() -> None:
    right = eye_metrics(_eye_landmarks(iris_dx=8.0))
    down = eye_metrics(_eye_landmarks(iris_dy=4.0))
    assert right is not None
    assert down is not None
    assert right.iris_h == pytest.approx(0.7)
    assert down.iris_v == pytest.approx(0.1)
    assert down.iris_h == pytest.approx(0.5)


def test_eye_metrics_blink() -> None:
    metrics = eye_metrics(_eye_landmarks(lid_gap=2.0))
    assert metrics is not None
    assert metrics.openness < mpb.BLINK_OPENNESS


def test_eye_metrics_is_rotation_invariant() -> None:
    pts = _eye_landmarks(iris_dx=6.0, iris_dy=-3.0)
    c, s = math.cos(0.4), math.sin(0.4)
    rotated = pts @ np.array([[c, -s], [s, c]]).T + (37.0, -12.0)
    a, b = eye_metrics(pts), eye_metrics(rotated)
    assert a is not None
    assert b is not None
    assert (b.iris_h, b.iris_v, b.openness) == pytest.approx((a.iris_h, a.iris_v, a.openness))


def test_eye_metrics_degenerate() -> None:
    pts = _eye_landmarks()
    pts[EYE_B[1]] = pts[EYE_B[0]]  # zero-width eye
    assert eye_metrics(pts) is None
    assert eye_metrics(np.zeros((100, 2))) is None
    nan = _eye_landmarks()
    nan[EYE_A[0]] = (np.nan, np.nan)
    assert eye_metrics(nan) is None


def _yunet_row(
    *,
    re: tuple[float, float] = (100.0, 100.0),
    le: tuple[float, float] = (140.0, 100.0),
    nose: tuple[float, float] = (120.0, 122.0),
    rm: tuple[float, float] = (105.0, 140.0),
    lm: tuple[float, float] = (135.0, 140.0),
    box: tuple[float, float, float, float] = (80.0, 60.0, 80.0, 100.0),
) -> np.ndarray:
    return np.array([*box, *re, *le, *nose, *rm, *lm, 0.9], dtype=np.float32)


def test_geometry_frontal() -> None:
    result = geometry_features(_yunet_row(), 320.0, 240.0)
    assert result is not None
    features, yaw, pitch = result
    nose_dx, nose_dy, roll, cx, cy, scale = features
    assert nose_dx == pytest.approx(0.0)
    assert nose_dy == pytest.approx(0.0)
    assert roll == pytest.approx(0.0)
    assert (cx, cy) == pytest.approx((120 / 320, 110 / 240))
    assert scale == pytest.approx(40 / 320)
    assert (yaw, pitch) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_geometry_signs() -> None:
    turned = geometry_features(_yunet_row(nose=(128.0, 122.0)), 320, 240)
    lowered = geometry_features(_yunet_row(nose=(120.0, 128.0)), 320, 240)
    tilted = geometry_features(_yunet_row(le=(140.0, 90.0)), 320, 240)
    assert turned is not None
    assert lowered is not None
    assert tilted is not None
    assert turned[0][0] > 0
    assert turned[1] > 0  # yaw
    assert lowered[0][1] > 0
    assert lowered[2] > 0  # pitch
    assert tilted[0][2] > 0  # counter-clockwise roll


def test_geometry_mirror_flips_horizontal_features() -> None:
    row = _yunet_row(nose=(127.0, 120.0), le=(140.0, 95.0))
    width = 320.0
    mirrored = row.copy()
    # Mirror every x coordinate; YuNet labels eyes/mouth corners by image side.
    mirrored[0] = width - (row[0] + row[2])
    mirrored[4:6] = (width - row[6], row[7])
    mirrored[6:8] = (width - row[4], row[5])
    mirrored[8] = width - row[8]
    mirrored[10:12] = (width - row[12], row[13])
    mirrored[12:14] = (width - row[10], row[11])
    a = geometry_features(row, width, 240)
    b = geometry_features(mirrored, width, 240)
    assert a is not None
    assert b is not None
    assert b[0][0] == pytest.approx(-a[0][0], abs=1e-6)  # nose_dx
    assert b[0][2] == pytest.approx(-a[0][2], abs=1e-6)  # roll
    assert b[0][1] == pytest.approx(a[0][1], abs=1e-6)  # nose_dy


def test_geometry_degenerate() -> None:
    assert geometry_features(_yunet_row(le=(100.0, 100.0)), 320, 240) is None
    assert geometry_features(_yunet_row(rm=(105.0, 100.0), lm=(135.0, 100.0)), 320, 240) is None


# ------------------------------------------------------------------ real faces
def _load(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None, f"could not decode {path}"
    return image


def _camera_like(image: np.ndarray, longest: int = 640) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, longest / max(h, w))
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _run(backend: VisionBackend, frame: np.ndarray, start: float, n: int = 3):
    """Process a frame a few times (VIDEO-mode trackers settle) and return the last result."""
    obs = None
    for i in range(n):
        obs = backend.process(frame, start + 0.05 * i)
    assert obs is not None
    return obs


def _face_crop(image: np.ndarray) -> np.ndarray:
    """Crop generously around the face found by YuNet (for composites)."""
    detector = OpenCVBackend()
    try:
        obs = detector.process(image, 1.0)
    finally:
        detector.close()
    assert obs.face_count >= 1
    assert obs.face_box is not None
    h, w = image.shape[:2]
    x, y, bw, bh = obs.face_box
    cx, cy = (x + bw / 2) * w, (y + bh / 2) * h
    half = max(bw * w, bh * h) * 0.9
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
    return image[y0:y1, x0:x1]


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


@needs_mediapipe
def test_mediapipe_real_face(face_image: Path, mp_backend: VisionBackend) -> None:
    frame = _camera_like(_load(face_image))
    obs = _run(mp_backend, frame, 1.0, n=10)
    assert obs.face_count == 1
    assert obs.features is not None
    assert obs.features.shape == (len(mp_backend.feature_names),)
    assert np.all(np.isfinite(obs.features))
    assert obs.quality >= 0.5
    assert not obs.blink
    assert obs.usable
    assert obs.face_box is not None
    assert obs.head_yaw == pytest.approx(obs.features[0])
    iris_h = obs.features[6]
    assert 0.2 < iris_h < 0.8
    log.info("MediaPipe: %.1f ms/frame at %dx%d", obs.inference_ms, *obs.frame_size)
    assert obs.inference_ms < 250.0


@needs_mediapipe
def test_mediapipe_mirror_flips_yaw_and_iris(face_image: Path, mp_backend: VisionBackend) -> None:
    frame = _camera_like(_load(face_image))
    a = _run(mp_backend, frame, 1.0)
    b = _run(mp_backend, cv2.flip(frame, 1), 10.0)
    assert a.features is not None
    assert b.features is not None
    yaw_a, yaw_b = a.features[0], b.features[0]
    assert abs(yaw_a + yaw_b) < max(3.0, 0.5 * abs(yaw_a))
    if abs(yaw_a) >= 2.0:
        assert np.sign(yaw_a) == -np.sign(yaw_b)
    # Mirroring swaps the eyes and reverses the corner axes: iris_h -> 1 - iris_h.
    assert a.features[6] + b.features[6] == pytest.approx(1.0, abs=0.08)
    assert a.features[1] == pytest.approx(b.features[1], abs=5.0)  # pitch unchanged


@needs_mediapipe
def test_mediapipe_iris_is_position_and_scale_invariant(
    face_image: Path, mp_backend: VisionBackend
) -> None:
    image = _load(face_image)
    base = _run(mp_backend, _camera_like(image), 1.0)
    h, w = image.shape[:2]
    shifted = _camera_like(image[: int(h * 0.92), int(w * 0.06) :])
    smaller = _camera_like(image, longest=420)
    for i, frame in enumerate((shifted, smaller)):
        obs = _run(mp_backend, frame, 20.0 + 10 * i)
        assert base.features is not None
        assert obs.features is not None
        assert obs.features[6] == pytest.approx(base.features[6], abs=0.05)  # iris_h
        assert obs.features[7] == pytest.approx(base.features[7], abs=0.05)  # iris_v
        assert obs.features[0] == pytest.approx(base.features[0], abs=4.0)  # yaw


@needs_opencv
def test_opencv_real_face(face_image: Path, cv_backend: VisionBackend) -> None:
    frame = _camera_like(_load(face_image))
    obs = cv_backend.process(frame, 1.0)
    assert obs.face_count == 1
    assert obs.features is not None
    assert obs.features.shape == (len(cv_backend.feature_names),)
    assert np.all(np.isfinite(obs.features))
    assert obs.quality >= 0.6
    assert obs.usable
    log.info("OpenCV: %.1f ms/frame at %dx%d", obs.inference_ms, *obs.frame_size)
    assert obs.inference_ms < 250.0
    mirrored = cv_backend.process(cv2.flip(frame, 1), 2.0)
    assert mirrored.features is not None
    if abs(obs.features[0]) >= 0.03:
        assert np.sign(obs.features[0]) == -np.sign(mirrored.features[0])


@needs_mediapipe
@needs_opencv
def test_backends_agree_on_face_box(
    face_image: Path, mp_backend: VisionBackend, cv_backend: VisionBackend
) -> None:
    frame = _camera_like(_load(face_image))
    a = _run(mp_backend, frame, 1.0)
    b = cv_backend.process(frame, 1.0)
    assert a.face_box is not None
    assert b.face_box is not None
    assert _iou(a.face_box, b.face_box) > 0.4
    assert a.head_yaw is not None
    assert b.head_yaw is not None
    assert np.sign(a.head_yaw) == np.sign(b.head_yaw) or abs(a.head_yaw) < 3.0


def _two_faces(image: np.ndarray, scales: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    face = _face_crop(image)
    size = 260
    canvas = np.full((480, 640, 3), 235, np.uint8)
    for slot, scale in enumerate(scales):
        side = round(size * scale)
        tile = cv2.resize(face, (side, side), interpolation=cv2.INTER_AREA)
        x0 = 20 if slot == 0 else 640 - 20 - side
        y0 = (480 - side) // 2
        canvas[y0 : y0 + side, x0 : x0 + side] = tile
    return canvas


@needs_opencv
def test_opencv_counts_two_faces(face_image: Path, cv_backend: VisionBackend) -> None:
    obs = cv_backend.process(_two_faces(_load(face_image)), 1.0)
    assert obs.face_count == 2


@needs_mediapipe
@needs_opencv
def test_mediapipe_max_faces(face_image: Path, mp_backend: VisionBackend) -> None:
    canvas = _two_faces(_load(face_image))
    assert _run(mp_backend, canvas, 1.0).face_count == 1
    mp_backend.set_max_faces(2)
    assert _run(mp_backend, canvas, 10.0).face_count == 2
    out = mp_backend.annotate(canvas, _run(mp_backend, canvas, 20.0))
    assert out.shape == canvas.shape


@needs_mediapipe
@needs_opencv
@pytest.mark.parametrize("name", ["mediapipe", "opencv"])
def test_primary_face_is_the_largest(face_image: Path, name: str) -> None:
    canvas = _two_faces(_load(face_image), scales=(0.7, 1.2))
    instance = create_backend(name, max_faces=2)
    try:
        obs = _run(instance, canvas, 1.0)
    finally:
        instance.close()
    assert obs.face_count == 2
    assert obs.face_box is not None
    x, _, w, _ = obs.face_box
    assert x + w / 2 > 0.5  # the larger copy is on the right


@needs_mediapipe
def test_mediapipe_annotate_draws_landmarks(face_image: Path, mp_backend: VisionBackend) -> None:
    frame = _camera_like(_load(face_image))
    obs = _run(mp_backend, frame, 1.0)
    out = mp_backend.annotate(frame, obs)
    assert out.shape == frame.shape
    changed = np.any(out != frame, axis=2)
    assert changed.sum() > 200
