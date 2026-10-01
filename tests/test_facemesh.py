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
    crop_caught_up,
    crop_settled,
    eye_metrics,
    load_landmark_net,
    pose_from_rotation,
    primary_score,
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


def _features(frame: np.ndarray, max_faces: int = 1) -> np.ndarray:
    instance = FaceMeshBackend(max_faces=max_faces)
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


def test_blink_baseline_ignores_closed_eyes_once_learnt_and_nan() -> None:
    detector = BlinkDetector(min_samples=3)
    assert not detector.update(float("nan"))
    assert detector.threshold == fm.BLINK_OPENNESS  # NaN is not learnt
    for _ in range(3):
        detector.update(0.2)
    assert detector.threshold == pytest.approx(0.1)
    for _ in range(10):
        assert detector.update(0.02)  # closed: not part of an existing baseline
    assert detector.threshold == pytest.approx(0.1)
    detector.reset()
    assert detector.threshold == fm.BLINK_OPENNESS


def test_blink_baseline_is_learnt_from_eyes_below_the_absolute_threshold() -> None:
    """r2-vision-04: narrow eyes that never open past 0.12 must not blink forever."""
    detector = BlinkDetector()
    readings = [detector.update(0.09) for _ in range(fm.BLINK_BASELINE_MIN_SAMPLES)]
    assert all(readings)  # the absolute threshold applies until the baseline exists ...
    assert not detector.update(0.09)  # ... and then the user's own eyes count as open
    assert detector.threshold == pytest.approx(fm.BLINK_MIN_OPENNESS)
    assert detector.update(0.03)  # a real blink still is one


def test_blink_baseline_bootstrap_shrugs_off_blinks() -> None:
    """Blinks among the first readings do not move the median baseline much."""
    detector = BlinkDetector(min_samples=10)
    for i in range(10):
        detector.update(0.02 if i in (3, 7) else 0.24)
    assert detector.threshold == pytest.approx(fm.BLINK_OPENNESS)  # 0.5 * 0.24 > 0.12
    assert detector.update(0.05)


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


def test_crop_settled() -> None:
    used = Roi(300.0, 200.0, 200.0, 0.0)
    assert crop_settled(used, used)
    assert crop_settled(used, Roi(306.0, 204.0, 205.0, 3.0))  # jitter (the angle is ignored)
    assert not crop_settled(used, Roi(300.0, 190.0, 200.0, 0.0))  # 5 % shift
    assert not crop_settled(used, Roi(300.0, 200.0, 190.0, 0.0))  # 5 % smaller
    assert crop_settled(used, Roi(300.0, 190.0, 200.0, 0.0), shift=fm.ROI_CATCH_UP_SHIFT)
    assert not crop_settled(used, Roi(math.nan, 200.0, 200.0, 0.0))
    assert not crop_settled(Roi(300.0, 200.0, 0.0, 0.0), used)


def test_crop_caught_up_needs_a_confident_face_while_catching_up() -> None:
    used = Roi(300.0, 200.0, 200.0, 0.0)
    jitter = Roi(306.0, 204.0, 205.0, 3.0)
    catching_up = Roi(300.0, 186.0, 200.0, 0.0)  # 7 % shift
    far = Roi(300.0, 180.0, 200.0, 0.0)  # 10 % shift
    confident, doubtful = fm.CATCH_UP_MIN_LOGIT, fm.CATCH_UP_MIN_LOGIT - 0.1
    # A settled crop ends the passes whatever the logit (a blurred, still face).
    assert crop_caught_up(used, jitter, doubtful)
    # A crop catching up with a moving face only with a confident logit: far
    # behind a jump the network asks for small shifts while still off the face.
    assert crop_caught_up(used, catching_up, confident)
    assert not crop_caught_up(used, catching_up, doubtful)
    assert not crop_caught_up(used, catching_up, math.nan)
    assert not crop_caught_up(used, far, 30.0)


def test_primary_score_prefers_large_then_central_faces() -> None:
    centre = primary_score(100.0, 320.0, 240.0, 640, 480)
    corner = primary_score(100.0, 0.0, 0.0, 640, 480)
    assert centre == pytest.approx(100.0)
    assert corner == pytest.approx(100.0 * (1.0 - fm.PRIMARY_CENTRE_WEIGHT))
    assert primary_score(150.0, 0.0, 0.0, 640, 480) > centre  # size dominates
    left = primary_score(100.0, 180.0, 240.0, 640, 480)
    right = primary_score(100.0, 460.0, 240.0, 640, 480)
    assert left == pytest.approx(right)


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
    folder = tmp_path / "KullanÄ±cÄ± BuÄŸra ÅŸÃ§Ã¶"
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
    calls = _counting(backend, monkeypatch)
    frame = draw_face()
    backend.process(frame, 1.0)
    # Found by YuNet, landmarks on its crop, and once more on the landmark rule's
    # crop (YuNet's box is larger than the landmark box).
    assert calls == {"infer": 2, "detect": 1}
    assert backend.settled
    for i in range(4):
        assert backend.process(draw_face(cx=320 + 2 * i), 1.1 + 0.1 * i).usable
        assert backend.settled
    assert calls == {"infer": 6, "detect": 1}  # tracked: one inference, no detection
    # The primary-face check after an acquisition (PRIMARY_RECHECK_S), then
    # every PRIMARY_CHECK_S: one detection, still one inference.
    backend.process(draw_face(cx=328), 1.0 + fm.PRIMARY_RECHECK_S)
    assert calls == {"infer": 7, "detect": 2}
    backend.process(draw_face(cx=328), 1.0 + fm.PRIMARY_RECHECK_S + 0.1)
    assert calls == {"infer": 8, "detect": 2}

    blank = np.full((480, 640, 3), 120, np.uint8)
    for i in range(3):
        assert backend.process(blank, 2.0 + 0.1 * i).face_count == 0
        assert backend.settled
    assert calls == {"infer": 9, "detect": 5}  # lost once, then only (cheap) detection


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


# ------------------------------------------- tracking after a jump (r2-vision-01)
def _counting(backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count the backend's landmark inferences and YuNet detections."""
    calls = {"infer": 0, "detect": 0}
    infer, detect = backend._infer, backend._detect

    def counting_infer(*args: object, **kwargs: object) -> object:
        calls["infer"] += 1
        return infer(*args, **kwargs)  # type: ignore[arg-type]

    def counting_detect(*args: object, **kwargs: object) -> object:
        calls["detect"] += 1
        return detect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(backend, "_infer", counting_infer)
    monkeypatch.setattr(backend, "_detect", counting_detect)
    return calls


def _converged_features(frame: np.ndarray) -> np.ndarray:
    instance = FaceMeshBackend()
    try:
        obs = _run(instance, frame, n=6)
    finally:
        instance.close()
    assert obs.features is not None
    return obs.features


@pytest.mark.parametrize(
    ("frac_x", "frac_y", "nudge", "noise_seed"),
    [
        (0.0, -0.25, (0, 0), None),
        (0.0, -0.2, (0, 0), None),
        (0.25, 0.0, (0, 0), None),
        # The same upward jump one pixel further, or with a little sensor noise.
        # Such details (and the CPU's floating-point rounding: Linux CI) decided
        # whether a pass far behind the face that asked for only a small shift, or
        # a rejected second pass, ended the frame with 12-20° of pitch off.
        (0.0, -0.25, (0, 1), None),
        (0.0, -0.25, (1, 0), None),
        (0.0, -0.25, (0, 0), 0),
        (0.0, -0.25, (1, 0), 1),
    ],
)
def test_first_observation_after_a_jump_is_not_biased(
    backend: FaceMeshBackend,
    frac_x: float,
    frac_y: float,
    nudge: tuple[int, int],
    noise_seed: int | None,
) -> None:
    """A posture shift between two analysed frames (idle frame rates).

    One pass on the old crop gave confident landmarks pulled towards the old
    position: 11-14° of pitch or yaw off for these jumps.
    """
    _run(backend, draw_face(cy=260), n=6)
    assert backend._roi is not None
    side = backend._roi.side
    moved = draw_face(
        cx=320 + round(frac_x * side) + nudge[0], cy=260 + round(frac_y * side) + nudge[1]
    )
    if noise_seed is not None:  # about one grey level, like a quiet webcam sensor
        noise = np.random.default_rng(noise_seed).normal(0.0, 1.0, moved.shape)
        moved = np.clip(moved + noise, 0, 255).astype(np.uint8)
    obs = backend.process(moved, 1.4)  # before the primary-face check is due
    assert obs.features is not None
    assert obs.quality == 1.0
    truth = _converged_features(moved)
    assert obs.features[YAW] == pytest.approx(truth[YAW], abs=2.0)
    assert obs.features[PITCH] == pytest.approx(truth[PITCH], abs=2.0)
    assert obs.features[IRIS_H] == pytest.approx(truth[IRIS_H], abs=0.02)


def test_a_moderate_move_costs_no_extra_inference_but_is_not_settled(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(backend, draw_face(), n=4)
    assert backend._roi is not None
    shift = round(0.06 * backend._roi.side)
    calls = _counting(backend, monkeypatch)
    moved = draw_face(cx=320 + shift)
    assert backend.process(moved, 1.3).usable  # before the primary-face check is due
    assert calls == {"infer": 1, "detect": 0}
    assert not backend.settled  # the motion gate must not repeat this one ...
    assert backend.process(moved, 1.4).usable
    assert backend.settled  # ... the next frame has caught up
    assert calls == {"infer": 2, "detect": 0}


def test_a_large_jump_is_followed_within_the_frame(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(backend, draw_face(cy=260), n=4)
    assert backend._roi is not None
    old = backend._roi
    calls = _counting(backend, monkeypatch)
    jump = round(0.25 * old.side)
    backend.process(draw_face(cy=260 - jump), 1.4)
    assert 2 <= calls["infer"] <= 4
    assert calls["detect"] <= 1
    assert backend._roi is not None
    assert old.cy - backend._roi.cy == pytest.approx(jump, abs=0.05 * old.side)


def test_unsettled_results_count_as_settled_after_a_streak(backend: FaceMeshBackend) -> None:
    """A crop that keeps shifting on a still picture must not disable the motion gate."""
    for _ in range(fm.MAX_UNSETTLED_FRAMES - 1):
        backend._note_settled(False)
        assert not backend.settled
    backend._note_settled(False)
    assert backend.settled
    backend._note_settled(False)
    assert backend.settled
    backend._note_settled(True)
    backend._note_settled(False)
    assert not backend.settled


def test_reset_forgets_the_tracked_face(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a pause the picture may differ; an old crop must not pick its face."""
    small_only = draw_face(cx=150, cy=200, scale=0.5)
    both = draw_face(small_only, cx=440, scale=1.0)
    first = _run(backend, small_only, n=2)
    assert first.face_box is not None
    assert first.face_box[0] < 0.4
    threshold = backend.blink_threshold
    backend.reset()
    assert backend._roi is None
    assert backend.landmarks is None
    assert backend.settled
    assert backend.blink_threshold == threshold  # the user's eyes did not change
    calls = _counting(backend, monkeypatch)
    obs = backend.process(both, 1.15)  # well before the next primary-face check
    assert calls["detect"] == 1  # searched afresh ...
    assert obs.face_box is not None
    assert obs.face_box[0] > 0.5  # ... and found the user, not the old crop's face


# ------------------------------------------------ the primary face (r2-vision-02/03)
def test_the_user_takes_over_from_a_background_face_with_the_guard_off(
    backend: FaceMeshBackend,
) -> None:
    """A face acquired while the user was away must not keep the primary role."""
    assert backend.max_faces == 1
    small_only = draw_face(cx=150, cy=200, scale=0.5)
    for i in range(5):  # the user is away; a poster's face is all there is
        first = backend.process(small_only, 1.0 + 0.25 * i)
        assert first.face_box is not None
        assert first.face_box[0] < 0.4
    both = draw_face(small_only, cx=440, scale=1.0)
    boxes = []
    for i in range(10):
        obs = backend.process(both, 2.25 + 0.25 * i)
        assert obs.face_box is not None
        boxes.append(obs.face_box[0])
    switched = [x > 0.5 for x in boxes]
    assert switched[-1]  # the user (large, near the camera) is primary ...
    assert switched.index(True) <= math.ceil(fm.PRIMARY_CHECK_S / 0.25)  # ... soon ...
    assert all(switched[switched.index(True) :])  # ... and stays so


def test_a_stronger_face_the_network_cannot_read_does_not_steal_tracking(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = draw_face(cx=180, scale=0.6)
    assert _run(backend, user).usable
    infer = backend._infer

    def blind_to_the_right(frame: np.ndarray, roi: Roi) -> tuple[np.ndarray, float] | None:
        return None if roi.cx > 320 else infer(frame, roi)

    monkeypatch.setattr(backend, "_infer", blind_to_the_right)
    both = draw_face(user, cx=460, scale=1.0)  # e.g. someone close in profile
    for i in range(12):
        obs = backend.process(both, 2.0 + 0.25 * i)
        assert obs.usable
        assert obs.face_box is not None
        assert obs.face_box[0] < 0.5


@pytest.mark.parametrize(("ratio", "switches"), [(1.2, False), (1.5, True)])
def test_primary_switch_compares_like_with_like(
    backend: FaceMeshBackend, ratio: float, switches: bool
) -> None:
    """r2-vision-03: YuNet's box is ~13 % larger than the landmark box of the same face.

    Comparing one with the other let a face only ~1.15x the user's take over.
    Both faces sit symmetrically about the image centre, so only size counts.
    """
    backend.set_max_faces(2)
    left = draw_face(cx=180, scale=0.6)
    first = _run(backend, left)
    assert first.face_box is not None
    assert first.face_box[0] < 0.5
    both = draw_face(left, cx=460, scale=0.6 * ratio)
    for i in range(8):
        obs = backend.process(both, 2.0 + 0.2 * i)  # spans several guard counts
        assert obs.face_count == 2
    assert obs.face_box is not None
    assert (obs.face_box[0] > 0.5) == switches


def test_a_comparable_face_is_tried_when_the_best_cannot_be_read(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = draw_face(draw_face(cx=180, scale=0.8), cx=440, scale=1.0)
    infer = backend._infer

    def blind_to_the_right(frame: np.ndarray, roi: Roi) -> tuple[np.ndarray, float] | None:
        return None if roi.cx > 320 else infer(frame, roi)

    monkeypatch.setattr(backend, "_infer", blind_to_the_right)
    obs = backend.process(pair, 1.0)
    assert obs.usable
    assert obs.face_box is not None
    assert obs.face_box[0] < 0.5


def test_a_background_face_is_not_tried_when_the_user_cannot_be_read(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user turning away must not hand the gaze control to a poster behind them."""
    scene = draw_face(draw_face(cx=110, cy=130, scale=0.4), cx=420, scale=1.0)
    assert len(backend._detect(scene, fm.DETECT_WIDTH)) == 2
    infer = backend._infer

    def blind_to_the_user(frame: np.ndarray, roi: Roi) -> tuple[np.ndarray, float] | None:
        return None if roi.cx > 320 else infer(frame, roi)

    monkeypatch.setattr(backend, "_infer", blind_to_the_user)
    for i in range(3):
        obs = backend.process(scene, 1.0 + 0.25 * i)
        assert obs.features is None
        assert obs.face_count == 1  # someone is there, nothing usable for gaze
        assert obs.face_box is not None
        assert obs.face_box[0] > 0.4  # the user's box, not the poster's


# ------------------------------------------------------- head roll (r2-vision-05)
def test_a_strongly_rolled_face_is_acquired(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YuNet's eye points barely follow head roll; a seed far off is retried rotated."""
    frame = draw_face()

    def misjudged(face: np.ndarray) -> Roi:
        seed = roi_from_detection(face)
        return Roi(seed.cx, seed.cy, seed.side, seed.angle + 90.0)

    monkeypatch.setattr(fm, "roi_from_detection", misjudged)
    assert backend._infer(frame, misjudged(backend._detect(frame, fm.DETECT_WIDTH)[0])) is None
    got = [backend.process(frame, 1.0 + 0.25 * i).usable for i in range(3)]
    assert got[0]
    assert all(got)


def test_rotated_retries_are_rate_limited(
    backend: FaceMeshBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A face the network never reads (a profile) must not cost three inferences a frame."""
    frame = draw_face()
    calls = {"infer": 0}

    def unreadable(_frame: np.ndarray, _roi: Roi) -> None:
        calls["infer"] += 1

    monkeypatch.setattr(backend, "_infer", unreadable)
    per_frame = []
    t = 1.0
    while t < 1.0 + fm.ROLL_RETRY_PERIOD_S + 0.6:
        before = calls["infer"]
        assert backend.process(frame, t).face_count == 1
        per_frame.append(calls["infer"] - before)
        t += 0.25
    assert per_frame[0] == 3  # the seed, then rotated either way
    retries = [i for i, n in enumerate(per_frame) if n == 3]
    assert retries == [0, round(fm.ROLL_RETRY_PERIOD_S / 0.25)]
    assert all(n == 1 for i, n in enumerate(per_frame) if i not in retries)


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
    frame = _photo_frame(face_image)
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


def _photo_frame(face_image: Path) -> np.ndarray:
    """The face photo scaled to a webcam-like 640 px width (decoded from bytes: any path)."""
    image = cv2.imdecode(np.fromfile(str(face_image), np.uint8), cv2.IMREAD_COLOR)
    assert image is not None, f"cannot decode {face_image}"
    return cv2.resize(image, (640, round(image.shape[0] * 640 / image.shape[1])))


def _moved(frame: np.ndarray, centre: tuple[float, float], degrees: float, dy: float) -> np.ndarray:
    """``frame`` rotated about ``centre`` and shifted down by ``dy`` pixels."""
    m = cv2.getRotationMatrix2D(centre, degrees, 1.0)
    m[1, 2] += dy
    size = (frame.shape[1], frame.shape[0])
    return cv2.warpAffine(frame, m, size, borderMode=cv2.BORDER_REPLICATE)


def test_real_face_after_a_posture_shift(face_image: Path) -> None:
    """r2-vision-01 on a photo: a shift by a fifth of the crop between two frames."""
    frame = _photo_frame(face_image)
    instance = FaceMeshBackend()
    try:
        assert _run(instance, frame, n=4).usable
        roi = instance._roi
        assert roi is not None
        # Upwards (the hard direction) unless the face is too close to the top.
        dy = -0.2 * roi.side if roi.cy - roi.side * 0.7 > 0 else 0.2 * roi.side
        moved = _moved(frame, (roi.cx, roi.cy), 0.0, dy)
        obs = instance.process(moved, 1.4)
    finally:
        instance.close()
    assert obs.features is not None
    truth = _converged_features(moved)
    assert obs.features[YAW] == pytest.approx(truth[YAW], abs=2.0)
    assert obs.features[PITCH] == pytest.approx(truth[PITCH], abs=2.0)
    assert obs.features[IRIS_H] == pytest.approx(truth[IRIS_H], abs=0.02)


@pytest.mark.parametrize("degrees", [55.0, 60.0, -60.0])
def test_real_face_with_strong_head_roll_is_acquired(face_image: Path, degrees: float) -> None:
    """r2-vision-05: YuNet's eye points barely follow a roll of 45° or more."""
    frame = _photo_frame(face_image)
    instance = FaceMeshBackend()
    try:
        faces = instance._detect(frame, fm.DETECT_WIDTH)
        assert faces
        x, y, w, h = (float(v) for v in max(faces, key=fm._det_side)[:4])
        rolled = _moved(frame, (x + w / 2.0, y + h / 2.0), degrees, 0.0)
        if not instance._detect(rolled, fm.DETECT_WIDTH):
            pytest.skip("YuNet does not find this photo's face rolled that far")
        got = [instance.process(rolled, 1.0 + 0.25 * i) for i in range(3)]
    finally:
        instance.close()
    usable = [o for o in got if o.features is not None]
    assert usable
    assert usable[-1].features is not None
    assert usable[-1].features[ROLL] == pytest.approx(degrees, abs=8.0)
