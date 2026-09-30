"""Tests for the face-mesh backend (MediaPipe's landmark network run with OpenCV DNN).

Pure helpers are tested with synthetic landmarks; the backend itself runs on
drawn cartoon faces (``face_drawing.py``), which the landmark network accepts.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from eye_tracker import paths
from eye_tracker.types import Observation
from eye_tracker.vision.backends import BackendUnavailable
from eye_tracker.vision.backends import facemesh_backend as fm
from eye_tracker.vision.backends.face_geometry import FaceGeometry, load_face_geometry
from eye_tracker.vision.backends.facemesh_backend import (
    EYE_A,
    EYE_B,
    BlinkDetector,
    EyeMetrics,
    FaceMeshBackend,
    PoseEstimator,
    Roi,
    camera_matrix,
    eye_metrics,
    load_landmark_net,
    pose_from_rotation,
    roi_from_detection,
    roi_from_landmarks,
)
from face_drawing import draw_face, rotate, two_faces

YAW, PITCH, ROLL, TX, TY, TZ, IRIS_H, IRIS_V = range(8)


@pytest.fixture
def backend() -> Iterator[FaceMeshBackend]:
    instance = FaceMeshBackend()
    try:
        yield instance
    finally:
        instance.close()


def _run(
    backend: FaceMeshBackend, frame: np.ndarray, start: float = 1.0, n: int = 3
) -> Observation:
    obs = None
    for i in range(n):
        obs = backend.process(frame, start + 0.05 * i)
    assert obs is not None
    return obs


def _features(frame: np.ndarray, **kwargs: int) -> np.ndarray:
    instance = FaceMeshBackend(**kwargs)
    try:
        obs = _run(instance, frame)
    finally:
        instance.close()
    assert obs.features is not None
    return obs.features


# ------------------------------------------------------------------ eye metrics
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
    assert eye_metrics(_eye_landmarks()) == EyeMetrics(iris_h=0.5, iris_v=0.0, openness=0.3)


def test_eye_metrics_iris_moves_same_way_in_both_eyes() -> None:
    right = eye_metrics(_eye_landmarks(iris_dx=8.0))
    down = eye_metrics(_eye_landmarks(iris_dy=4.0))
    assert right is not None
    assert down is not None
    assert right.iris_h == pytest.approx(0.7)
    assert down.iris_v == pytest.approx(0.1)
    assert down.iris_h == pytest.approx(0.5)


def test_eye_metrics_closed_eyes() -> None:
    metrics = eye_metrics(_eye_landmarks(lid_gap=2.0))
    assert metrics is not None
    assert metrics.openness < fm.BLINK_OPENNESS


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


# ---------------------------------------------------------------------- blinks
def test_blink_uses_the_absolute_threshold_until_a_baseline_exists() -> None:
    detector = BlinkDetector(min_samples=5)
    assert detector.threshold == fm.BLINK_OPENNESS
    assert detector.update(0.10)
    assert not detector.update(0.13)


def test_blink_threshold_adapts_to_narrow_eyes() -> None:
    """A user whose open eyes read 0.18 must not blink at 0.11 (e.g. looking down)."""
    detector = BlinkDetector(min_samples=5)
    for _ in range(5):
        assert not detector.update(0.18)
    assert detector.threshold == pytest.approx(0.09)
    assert not detector.update(0.11)  # 0.11 < 0.12 would be a blink with a fixed threshold
    assert detector.update(0.05)


def test_blink_threshold_never_exceeds_the_absolute_or_drops_below_the_minimum() -> None:
    wide = BlinkDetector(min_samples=3)
    for _ in range(3):
        wide.update(0.40)
    assert wide.threshold == fm.BLINK_OPENNESS
    narrow = BlinkDetector(min_samples=3)
    for _ in range(3):
        narrow.update(0.13)
    assert narrow.threshold == fm.BLINK_MIN_OPENNESS


def test_blink_baseline_ignores_closed_eyes_and_nan() -> None:
    detector = BlinkDetector(min_samples=3)
    for _ in range(10):
        assert detector.update(0.02)  # closed: never part of the baseline
    assert detector.threshold == fm.BLINK_OPENNESS
    assert not detector.update(float("nan"))
    for _ in range(3):
        detector.update(0.2)
    assert detector.threshold == pytest.approx(0.1)
    detector.reset()
    assert detector.threshold == fm.BLINK_OPENNESS


# ------------------------------------------------------------------------ crops
def test_roi_transform_centres_scales_and_levels() -> None:
    roi = Roi(cx=300.0, cy=200.0, side=128.0, angle=20.0)
    m = roi.transform()
    centre = m @ (300.0, 200.0, 1.0)
    assert centre == pytest.approx((128.0, 128.0))
    # Two points on a line descending at 20° towards image-right end up level,
    # twice as far apart (256 / 128).
    d = (math.cos(math.radians(20.0)) * 30.0, math.sin(math.radians(20.0)) * 30.0)
    a = m @ (300.0 - d[0], 200.0 - d[1], 1.0)
    b = m @ (300.0 + d[0], 200.0 + d[1], 1.0)
    assert a[1] == pytest.approx(b[1])
    assert b[0] - a[0] == pytest.approx(120.0)


def test_roi_validity() -> None:
    assert Roi(10.0, 10.0, 100.0, 0.0).valid(640, 480)
    assert not Roi(10.0, 10.0, fm.MIN_ROI_SIDE / 2, 0.0).valid(640, 480)
    assert not Roi(-1.0, 10.0, 100.0, 0.0).valid(640, 480)
    assert not Roi(10.0, 480.0, 100.0, 0.0).valid(640, 480)
    assert not Roi(math.nan, 10.0, 100.0, 0.0).valid(640, 480)


def test_roi_from_detection() -> None:
    # YuNet row: box, right eye (image left), left eye, nose, mouth corners, score.
    row = np.array([100, 50, 80, 100, 120, 90, 160, 110, 140, 110, 125, 130, 155, 130, 0.9])
    roi = roi_from_detection(row)
    assert (roi.cx, roi.cy) == pytest.approx((140.0, 100.0))
    assert roi.side == pytest.approx(150.0)
    assert roi.angle == pytest.approx(math.degrees(math.atan2(20.0, 40.0)))


def test_roi_from_landmarks() -> None:
    pts = np.zeros((478, 2))
    pts[:468] = (200.0, 150.0)
    pts[0] = (150.0, 100.0)
    pts[1] = (250.0, 220.0)
    pts[33] = (170.0, 140.0)
    pts[263] = (230.0, 140.0)
    pts[470] = (999.0, 999.0)  # irises do not widen the crop
    roi = roi_from_landmarks(pts)
    assert (roi.cx, roi.cy) == pytest.approx((200.0, 160.0))
    assert roi.side == pytest.approx(120.0 * fm.ROI_SCALE)
    assert roi.angle == pytest.approx(0.0)


# ------------------------------------------------------------------------- pose
def _ry(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_pose_from_identity() -> None:
    pose = pose_from_rotation(np.eye(3), (1.0, 2.0, 60.0))
    assert (pose.yaw, pose.pitch, pose.roll) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert (pose.tx, pose.ty, pose.tz) == (1.0, 2.0, 60.0)


def test_pose_sign_convention() -> None:
    # Camera frame: x right, y down, z away; a frontal face's nose points along -z.
    to_right = pose_from_rotation(_ry(-20.0), (0, 0, 60))  # nose towards image-right
    lowered = pose_from_rotation(_rx(15.0), (0, 0, 60))  # nose towards image-bottom
    ccw = pose_from_rotation(_rz(-10.0), (0, 0, 60))  # counter-clockwise on screen
    assert (to_right.yaw, to_right.pitch, to_right.roll) == pytest.approx((20, 0, 0), abs=1e-6)
    assert (lowered.yaw, lowered.pitch, lowered.roll) == pytest.approx((0, 15, 0), abs=1e-6)
    assert (ccw.yaw, ccw.pitch, ccw.roll) == pytest.approx((0, 0, 10), abs=1e-6)


def test_pose_mirror_flips_yaw_and_roll() -> None:
    mirror = np.diag([-1.0, 1.0, 1.0])
    rotation = _ry(-18.0) @ _rx(7.0) @ _rz(12.0)
    a = pose_from_rotation(rotation, (5.0, 1.0, 60.0))
    b = pose_from_rotation(mirror @ rotation @ mirror, (-5.0, 1.0, 60.0))
    assert b.yaw == pytest.approx(-a.yaw)
    assert b.roll == pytest.approx(-a.roll)
    assert b.pitch == pytest.approx(a.pitch)


def test_camera_matrix() -> None:
    k = camera_matrix(640, 480)
    assert k[0, 0] == k[1, 1] == 640.0
    assert (k[0, 2], k[1, 2]) == (320.0, 240.0)


@pytest.fixture(scope="module")
def geometry() -> FaceGeometry:
    return load_face_geometry(paths.model_path(fm.GEOMETRY_FILE))


def _project(
    geometry: FaceGeometry, rotation: np.ndarray, tvec: tuple[float, float, float]
) -> np.ndarray:
    obj = geometry.vertices * (1.0, -1.0, -1.0)
    rvec, _ = cv2.Rodrigues(rotation)
    img, _ = cv2.projectPoints(obj, rvec, np.array(tvec, float), camera_matrix(640, 480), None)
    points = np.zeros((478, 2))
    points[:468] = img.reshape(-1, 2)
    return points


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (np.eye(3), (0.0, 0.0, 0.0)),
        (_ry(-20.0), (20.0, 0.0, 0.0)),
        (_rx(12.0), (0.0, 12.0, 0.0)),
        (_rz(-8.0), (0.0, 0.0, 8.0)),
    ],
)
def test_pose_estimator_recovers_a_projected_pose(
    geometry: FaceGeometry, rotation: np.ndarray, expected: tuple[float, float, float]
) -> None:
    estimator = PoseEstimator(geometry)
    points = _project(geometry, rotation, (3.0, -2.0, 55.0))
    pose = estimator.estimate(points, 640, 480)
    assert pose is not None
    assert (pose.yaw, pose.pitch, pose.roll) == pytest.approx(expected, abs=0.2)
    assert (pose.tx, pose.ty, pose.tz) == pytest.approx((3.0, -2.0, 55.0), abs=0.2)
    # Tracking from the previous solution gives the same answer.
    again = estimator.estimate(points, 640, 480)
    assert again is not None
    assert again.yaw == pytest.approx(pose.yaw, abs=1e-3)


def test_pose_estimator_rejects_bad_points(geometry: FaceGeometry) -> None:
    estimator = PoseEstimator(geometry)
    points = _project(geometry, np.eye(3), (0.0, 0.0, 50.0))
    assert estimator.estimate(points, 640, 480) is not None
    points[geometry.basis_ids[0]] = np.nan
    assert estimator.estimate(points, 640, 480) is None
    assert estimator.rvec is None


def test_pose_estimator_resets_on_a_new_frame_size(geometry: FaceGeometry) -> None:
    estimator = PoseEstimator(geometry)
    estimator.estimate(_project(geometry, np.eye(3), (0.0, 0.0, 50.0)), 640, 480)
    assert estimator.camera[0, 0] == 640.0
    small = _project(geometry, np.eye(3), (0.0, 0.0, 50.0)) / 2.0
    pose = estimator.estimate(small, 320, 240)
    assert estimator.camera[0, 0] == 320.0
    assert pose is not None
    assert pose.tz == pytest.approx(50.0, abs=0.5)


# ------------------------------------------------------------------ model load
def test_landmark_model_missing(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailable, match="not found"):
        load_landmark_net(tmp_path / "missing.tflite")


def test_landmark_model_garbage(tmp_path: Path) -> None:
    path = tmp_path / "garbage.tflite"
    path.write_bytes(b"\x00not a model" * 100)
    with pytest.raises(BackendUnavailable, match="cannot run the face landmark model"):
        load_landmark_net(path)


def test_landmark_model_wrong_network(tmp_path: Path) -> None:
    path = tmp_path / "yunet.tflite"
    shutil.copyfile(paths.model_path("face_detection_yunet_2023mar.onnx"), path)
    with pytest.raises(BackendUnavailable):
        load_landmark_net(path)


def test_landmark_self_test_rejects_wrong_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runtime that computes different numbers must not be trusted."""
    monkeypatch.setattr(fm, "_SELF_TEST_MEAN", (40.0, 200.0))
    with pytest.raises(BackendUnavailable, match="wrong values"):
        load_landmark_net(paths.model_path(fm.MODEL_FILE))


def test_landmark_self_test_passes() -> None:
    net = load_landmark_net(paths.model_path(fm.MODEL_FILE))
    fm._check_landmark_net(net)


def test_landmark_model_from_a_non_ascii_path(tmp_path: Path) -> None:
    folder = tmp_path / "Kullanıcı Buğra şçö"
    folder.mkdir()
    path = folder / fm.MODEL_FILE
    shutil.copyfile(paths.model_path(fm.MODEL_FILE), path)
    if sys.platform == "win32" and fm._windows_short_path(path) is None:
        pytest.skip("8.3 short names are disabled on this volume")
    assert load_landmark_net(path) is not None


def test_missing_geometry(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailable, match="geometry"):
        FaceMeshBackend(geometry_path=tmp_path / "missing.binarypb")


def test_missing_detector(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailable, match="YuNet model not found"):
        FaceMeshBackend(detector_path=tmp_path / "missing.onnx")


# ------------------------------------------------------------- the backend
def test_features_of_a_frontal_face(backend: FaceMeshBackend) -> None:
    obs = _run(backend, draw_face())
    assert obs.usable
    assert obs.features is not None
    f = obs.features
    assert abs(f[YAW]) < 5.0
    assert abs(f[PITCH]) < 5.0
    assert abs(f[ROLL]) < 3.0
    assert abs(f[TX]) < 2.0
    assert 40.0 < f[TZ] < 90.0  # ~60 cm for a 160 px face at 640 px
    assert f[IRIS_H] == pytest.approx(0.5, abs=0.05)
    assert abs(f[IRIS_V]) < 0.1
    assert obs.head_yaw == pytest.approx(f[YAW])
    assert obs.head_pitch == pytest.approx(f[PITCH])
    landmarks = backend.landmarks
    assert landmarks is not None
    assert landmarks.shape == (478, 2)
    assert np.all((landmarks >= 0.0) & (landmarks <= 1.0))


def test_mirrored_image_flips_yaw_roll_and_iris() -> None:
    frame = rotate(draw_face(cx=220, iris_dx=3.0), 8.0)
    a = _features(frame)
    b = _features(cv2.flip(frame, 1))
    assert a[YAW] > 5.0  # a face left of centre is turned towards image-right
    assert b[YAW] == pytest.approx(-a[YAW], abs=2.5)
    assert a[ROLL] > 5.0
    assert b[ROLL] == pytest.approx(-a[ROLL], abs=2.5)
    assert b[PITCH] == pytest.approx(a[PITCH], abs=2.5)
    assert b[TX] == pytest.approx(-a[TX], abs=1.0)
    assert a[IRIS_H] + b[IRIS_H] == pytest.approx(1.0, abs=0.05)


def test_roll_follows_image_rotation() -> None:
    assert _features(rotate(draw_face(), 15.0))[ROLL] == pytest.approx(15.0, abs=2.0)
    assert _features(rotate(draw_face(), -15.0))[ROLL] == pytest.approx(-15.0, abs=2.0)


def test_iris_follows_the_eyes() -> None:
    base = _features(draw_face())[IRIS_H]
    assert _features(draw_face(iris_dx=4.0))[IRIS_H] > base + 0.05
    assert _features(draw_face(iris_dx=-4.0))[IRIS_H] < base - 0.05


def test_position_and_distance() -> None:
    high = _features(draw_face(cy=180))
    low = _features(draw_face(cy=300))
    assert high[TY] < low[TY]
    assert high[PITCH] > low[PITCH]  # above the lens, the face looks down at it
    near = _features(draw_face())
    far = _features(draw_face(scale=0.5))
    assert far[TZ] == pytest.approx(2.0 * near[TZ], rel=0.15)


def test_tracking_costs_one_inference_per_frame(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"infer": 0, "detect": 0}
    infer, detect = backend._infer, backend._detect

    def counting_infer(*args: object, **kwargs: object) -> object:
        calls["infer"] += 1
        return infer(*args, **kwargs)

    def counting_detect(*args: object, **kwargs: object) -> object:
        calls["detect"] += 1
        return detect(*args, **kwargs)

    monkeypatch.setattr(backend, "_infer", counting_infer)
    monkeypatch.setattr(backend, "_detect", counting_detect)
    frame = draw_face()
    backend.process(frame, 1.0)
    assert calls == {"infer": 1, "detect": 1}  # found by YuNet, then landmarks
    for i in range(5):
        assert backend.process(draw_face(cx=320 + 2 * i), 1.1 + 0.1 * i).usable
    assert calls == {"infer": 6, "detect": 1}  # tracked: no more detection

    blank = np.full((480, 640, 3), 120, np.uint8)
    for i in range(3):
        assert backend.process(blank, 2.0 + 0.1 * i).face_count == 0
    assert calls == {"infer": 7, "detect": 4}  # lost once, then only (cheap) detection


def test_face_is_reacquired_after_it_was_lost(backend: FaceMeshBackend) -> None:
    assert _run(backend, draw_face(cx=250)).usable
    assert backend.process(np.full((480, 640, 3), 120, np.uint8), 2.0).face_count == 0
    obs = backend.process(draw_face(cx=400), 2.1)
    assert obs.usable
    assert obs.face_box is not None
    assert obs.face_box[0] + obs.face_box[2] / 2 == pytest.approx(400 / 640, abs=0.03)


def test_frame_size_change_restarts_tracking(backend: FaceMeshBackend) -> None:
    assert _run(backend, draw_face()).usable
    small = cv2.resize(draw_face(), (320, 240), interpolation=cv2.INTER_AREA)
    obs = backend.process(small, 5.0)
    assert obs.frame_size == (320, 240)
    assert obs.face_count == 1


def test_face_at_the_border_has_lower_quality(backend: FaceMeshBackend) -> None:
    obs = _run(backend, draw_face(cx=70))
    assert obs.face_count == 1
    if obs.features is not None:
        assert obs.quality == 0.5


def test_set_max_faces_clamps(backend: FaceMeshBackend) -> None:
    assert backend.max_faces == 1
    backend.set_max_faces(0)
    assert backend.max_faces == 1
    backend.set_max_faces(99)
    assert backend.max_faces == fm.MAX_FACES_LIMIT


def test_shoulder_guard_counts_the_onlooker(backend: FaceMeshBackend) -> None:
    canvas = two_faces()
    assert _run(backend, canvas).face_count == 1
    backend.set_max_faces(2)
    obs = _run(backend, canvas, start=10.0)
    assert obs.face_count == 2
    assert obs.usable
    out = backend.annotate(canvas, obs)
    assert np.any(out != canvas)


def test_shoulder_guard_counts_at_most_every_period(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    widths: list[int] = []
    detect = backend._detect

    def spy(frame: np.ndarray, input_width: int) -> list[np.ndarray]:
        widths.append(input_width)
        return detect(frame, input_width)

    monkeypatch.setattr(backend, "_detect", spy)
    backend.set_max_faces(2)
    canvas = two_faces()
    counts = [backend.process(canvas, 1.0 + 0.1 * i).face_count for i in range(11)]
    assert counts == [2] * 11  # the count is reused between detections
    assert widths == [fm.GUARD_DETECT_WIDTH] * 3  # t = 1.0, 1.5 and 2.0


def test_shoulder_guard_does_not_count_the_user_twice_after_moving(
    backend: FaceMeshBackend,
) -> None:
    """Between counts the user may move; old detections must not become 'others'."""
    backend.set_max_faces(2)
    first = backend.process(draw_face(cx=200), 1.0)
    assert first.face_count == 1
    assert first.face_box is not None
    old_centre = first.face_box[0] + first.face_box[2] / 2
    for i in range(1, 7):  # 150 px to the right within 0.42 s
        obs = backend.process(draw_face(cx=200 + 25 * i), 1.0 + 0.07 * i)
        assert obs.face_count == 1
    assert backend._guard_ts == 1.0  # tracked all along: no fresh count
    landmarks = backend.landmarks
    assert landmarks is not None
    assert landmarks[:468, 0].min() > old_centre  # the old detection is elsewhere now


def test_a_much_larger_face_takes_over_with_the_guard_on(backend: FaceMeshBackend) -> None:
    backend.set_max_faces(2)
    small_only = draw_face(cx=150, cy=200, scale=0.5)
    first = _run(backend, small_only)
    assert first.face_box is not None
    assert first.face_box[0] < 0.4
    both = draw_face(small_only, cx=440, scale=1.0)
    obs = _run(backend, both, start=5.0)
    assert obs.face_count == 2
    assert obs.face_box is not None
    assert obs.face_box[0] > 0.5  # the user (large, near the camera) is primary


def test_blink_threshold_is_exposed(backend: FaceMeshBackend) -> None:
    assert backend.blink_threshold == fm.BLINK_OPENNESS
    _run(backend, draw_face(), n=fm.BLINK_BASELINE_MIN_SAMPLES + 1)
    assert fm.BLINK_MIN_OPENNESS <= backend.blink_threshold <= fm.BLINK_OPENNESS


def test_annotate_ignores_a_stale_observation(backend: FaceMeshBackend) -> None:
    frame = draw_face()
    obs = _run(backend, frame)
    fresh = backend.annotate(frame, obs)
    stale = backend.annotate(frame, Observation(timestamp=obs.timestamp - 1.0, face_count=1))
    assert np.count_nonzero(np.any(fresh != frame, axis=2)) > np.count_nonzero(
        np.any(stale != frame, axis=2)
    )


def test_real_face_matches_mediapipe(face_image: Path) -> None:
    """Landmarks agree with MediaPipe's own runtime on the maintainers' reference photo.

    Only meaningful for that photo (``EYE_TRACKER_TEST_FACE_REFERENCE=1``); for
    any other face the landmarks just have to be plausible.
    """
    image = cv2.imdecode(np.fromfile(str(face_image), np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.resize(image, (640, round(image.shape[0] * 640 / image.shape[1])))
    instance = FaceMeshBackend()
    try:
        obs = _run(instance, frame, n=4)
        landmarks = instance.landmarks
    finally:
        instance.close()
    assert obs.usable
    assert landmarks is not None
    if os.environ.get("EYE_TRACKER_TEST_FACE_REFERENCE") != "1":
        return
    # MediaPipe 1.0.1 (Face Landmarker, IMAGE mode) on the same resized photo.
    expected = {468: (0.464, 0.196), 473: (0.586, 0.191)}
    expected_x = {33: 0.440, 133: 0.496, 362: 0.558, 263: 0.610}
    for index, (x, y) in expected.items():
        assert landmarks[index] == pytest.approx((x, y), abs=0.005)
    for index, x in expected_x.items():
        assert landmarks[index][0] == pytest.approx(x, abs=0.005)
