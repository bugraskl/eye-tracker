"""Lite backend: YuNet face detection and five-point head geometry.

This backend needs nothing beyond OpenCV and a 230 kB model and runs in a few
milliseconds at 320 px. It has no iris landmarks, so it relies on head movement
alone: the features describe where the nose points relative to the eyes and
mouth, plus where the face is in the frame. Blinks cannot be detected.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from ... import paths
from ...types import Observation
from ..threads import limit_opencv_threads
from .base import BackendUnavailable, VisionBackend
from .overlay import (
    COLOR_BOX,
    COLOR_OTHER,
    COLOR_POINT,
    as_bgr,
    draw_box,
    draw_label,
    line_thickness,
    observation_label,
    to_bgr,
)

log = logging.getLogger(__name__)

MODEL_FILE = "face_detection_yunet_2023mar.onnx"
#: Frames are downscaled to at most this width before detection.
MAX_INPUT_WIDTH = 320
#: Detections below this confidence are ignored (and not counted as faces).
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 50
#: For a frontal face the nose tip sits about this far from the eye line
#: towards the mouth line (fraction of the eye-to-mouth distance).
NOSE_BASELINE = 0.55
#: Gains turning the nose offsets into rough angles for display. Only the raw
#: offsets are used for gaze estimation, so these need not be exact.
YAW_GAIN = 1.6
PITCH_GAIN = 2.5
BORDER_MARGIN = 0.01


def create_yunet(
    model_path: Path | None = None, input_width: int = MAX_INPUT_WIDTH
) -> cv2.FaceDetectorYN:
    """Create a YuNet detector for frames about ``input_width`` pixels wide.

    Raises:
        BackendUnavailable: The OpenCV build lacks ``FaceDetectorYN`` or the model
            cannot be loaded.
    """
    if not hasattr(cv2, "FaceDetectorYN"):
        raise BackendUnavailable(
            f"OpenCV {cv2.__version__} has no FaceDetectorYN (4.5.4 or newer is required)"
        )
    path = Path(model_path) if model_path else paths.model_path(MODEL_FILE)
    try:
        # Read by Python and passed as a buffer: OpenCV cannot open non-ASCII
        # paths on Windows.
        model = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise BackendUnavailable(
            f"YuNet model not found at {path} (run scripts/fetch_models.py)"
        ) from exc
    try:
        return cv2.FaceDetectorYN.create(
            "onnx",
            model,
            np.empty(0, dtype=np.uint8),
            (int(input_width), max(1, round(input_width * 3 / 4))),
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            TOP_K,
        )
    except cv2.error as exc:
        raise BackendUnavailable(f"OpenCV could not load the YuNet model: {exc}") from exc


def geometry_features(
    face: np.ndarray, frame_width: float, frame_height: float
) -> tuple[np.ndarray, float, float] | None:
    """Features and rough ``(yaw, pitch)`` in degrees for one YuNet detection row.

    ``face`` is ``[x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, rm_x, rm_y,
    lm_x, lm_y, score]`` in pixels of a ``frame_width`` x ``frame_height`` image.
    YuNet's "right eye" is the one on the image left, so the eye axis points to
    image-right for any upright face. Returns ``None`` for degenerate geometry.
    """
    x, y, w, h = (float(v) for v in face[:4])
    right_eye = face[4:6].astype(np.float64)
    left_eye = face[6:8].astype(np.float64)
    nose = face[8:10].astype(np.float64)
    mouth_mid = (face[10:12].astype(np.float64) + face[12:14].astype(np.float64)) / 2.0

    eye_mid = (right_eye + left_eye) / 2.0
    axis = left_eye - right_eye
    ipd = float(math.hypot(axis[0], axis[1]))
    if not ipd >= 1.0:  # also rejects NaN
        return None
    u = axis / ipd
    down = np.array([-u[1], u[0]])  # perpendicular pointing to the chin
    mouth_depth = float((mouth_mid - eye_mid) @ down)
    if not mouth_depth >= 1.0:
        return None

    nose_rel = nose - eye_mid
    # Horizontal offset in eye distances: grows as the head turns to image-right.
    nose_dx = float(nose_rel @ u) / ipd
    # Vertical position between the eye line (0) and mouth line (1), relative to a
    # frontal face. Normalising by the eye-to-mouth distance keeps it independent
    # of how far away the user sits and of head yaw (which shrinks the eye distance).
    nose_dy = float(nose_rel @ down) / mouth_depth - NOSE_BASELINE
    # Counter-clockwise positive, matching the sign of the facemesh backend's roll.
    roll = math.degrees(math.atan2(-axis[1], axis[0]))
    face_cx = (x + w / 2.0) / frame_width
    face_cy = (y + h / 2.0) / frame_height
    scale = ipd / frame_width

    features = np.array([nose_dx, nose_dy, roll, face_cx, face_cy, scale], dtype=np.float64)
    if not np.all(np.isfinite(features)):
        return None
    yaw = math.degrees(math.asin(_clamp(nose_dx * YAW_GAIN)))
    pitch = math.degrees(math.asin(_clamp(nose_dy * PITCH_GAIN)))
    return features, yaw, pitch


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return min(max(v, lo), hi)


def _clipped_box(
    face: np.ndarray, width: float, height: float
) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in face[:4])
    x0, y0 = max(x / width, 0.0), max(y / height, 0.0)
    x1, y1 = min((x + w) / width, 1.0), min((y + h) / height, 1.0)
    return (x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0))


@dataclass(slots=True)
class _Detections:
    """Landmarks of the last processed frame, for ``annotate``."""

    timestamp: float
    points: np.ndarray  # (5, 2) normalised: eyes, nose, mouth corners
    others: list[tuple[float, float, float, float]]  # boxes of the other faces


class LiteBackend(VisionBackend):
    """Head-geometry features from OpenCV's YuNet face detector.

    Creating an instance caps OpenCV's process-wide thread pool (see
    :mod:`eye_tracker.vision.threads`).

    Args:
        model_path: YuNet ONNX model; defaults to the bundled model.
        max_faces: Accepted for interface compatibility; YuNet always finds every
            face, and ``face_count`` reports all of them.

    Raises:
        BackendUnavailable: The OpenCV build lacks ``FaceDetectorYN`` or the model
            cannot be loaded.
    """

    name: ClassVar[str] = "lite"
    feature_names: ClassVar[tuple[str, ...]] = (
        "nose_dx",
        "nose_dy",
        "roll",
        "face_cx",
        "face_cy",
        "scale",
    )
    feature_version: ClassVar[str] = "yunet-geom-1"
    gaze_features: ClassVar[tuple[str, ...]] = ("nose_dx", "nose_dy")

    def __init__(self, model_path: Path | None = None, max_faces: int = 1) -> None:
        limit_opencv_threads()
        self._detector: cv2.FaceDetectorYN | None = create_yunet(model_path, MAX_INPUT_WIDTH)
        self._input_size: tuple[int, int] | None = None
        self._max_faces = max(1, int(max_faces))
        self._last: _Detections | None = None

    def set_max_faces(self, n: int) -> None:
        self._max_faces = max(1, int(n))

    def close(self) -> None:
        self._detector = None
        self._last = None

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Observation:
        detector = self._detector
        if detector is None:
            raise RuntimeError("LiteBackend is closed")
        started = time.perf_counter()
        height, width = frame_bgr.shape[:2]
        image = as_bgr(frame_bgr)
        if width > MAX_INPUT_WIDTH:
            scale = MAX_INPUT_WIDTH / width
            size = (MAX_INPUT_WIDTH, max(1, round(height * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        small_h, small_w = image.shape[:2]
        if (small_w, small_h) != self._input_size:
            self._input_size = (small_w, small_h)
            detector.setInputSize(self._input_size)

        _, detections = detector.detect(image)
        faces: list[np.ndarray] = []
        if detections is not None:
            faces = [row for row in detections if float(row[14]) >= SCORE_THRESHOLD]
        if not faces:
            self._last = None
            return Observation(
                timestamp=timestamp,
                face_count=0,
                inference_ms=(time.perf_counter() - started) * 1e3,
                frame_size=(width, height),
            )

        primary = max(range(len(faces)), key=lambda i: float(faces[i][2] * faces[i][3]))
        face = faces[primary]
        score = float(face[14])
        x, y, w, h = (float(v) for v in face[:4])
        x0, y0 = x / small_w, y / small_h
        x1, y1 = (x + w) / small_w, (y + h) / small_h
        face_box = _clipped_box(face, small_w, small_h)

        geometry = geometry_features(face, small_w, small_h)
        features: np.ndarray | None = None
        head_yaw: float | None = None
        head_pitch: float | None = None
        quality = 0.0
        if geometry is not None:
            features, head_yaw, head_pitch = geometry
            touches_border = (
                x0 <= BORDER_MARGIN
                or y0 <= BORDER_MARGIN
                or x1 >= 1.0 - BORDER_MARGIN
                or y1 >= 1.0 - BORDER_MARGIN
            )
            quality = min(score, 0.5) if touches_border else score

        self._last = _Detections(
            timestamp=timestamp,
            points=face[4:14].reshape(5, 2).astype(np.float64) / (small_w, small_h),
            others=[_clipped_box(f, small_w, small_h) for i, f in enumerate(faces) if i != primary],
        )
        return Observation(
            timestamp=timestamp,
            face_count=len(faces),
            features=features,
            quality=quality,
            blink=False,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            face_box=face_box,
            inference_ms=(time.perf_counter() - started) * 1e3,
            frame_size=(width, height),
        )

    def annotate(self, frame_bgr: np.ndarray, observation: Observation) -> np.ndarray:
        out = to_bgr(frame_bgr)
        h, w = out.shape[:2]
        thickness = line_thickness(out)
        last = self._last
        if last is not None and last.timestamp == observation.timestamp:
            for px, py in last.points * (w, h):
                center = (round(float(px)), round(float(py)))
                cv2.circle(out, center, thickness + 2, COLOR_POINT, -1, cv2.LINE_AA)
            for box in last.others:
                draw_box(out, box, COLOR_OTHER, thickness)
        if observation.face_box is not None:
            draw_box(out, observation.face_box, COLOR_BOX, thickness)
        draw_label(out, observation_label(observation))
        return out
