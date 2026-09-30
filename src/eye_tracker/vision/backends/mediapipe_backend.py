"""MediaPipe Face Landmarker backend: head pose plus iris position.

This is the accurate default. The Face Landmarker returns 478 landmarks
(including both irises) and a facial transformation matrix per face. The
feature vector combines head pose (rotation and translation from the matrix)
with the iris position inside each eye, so the gaze model can use head turns
and eye movements alike.

MediaPipe objects are bound to the thread that created them; the vision worker
creates, uses and closes this backend on its own thread.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from ... import paths
from ...types import Observation
from .._mp_shim import import_mediapipe
from .base import BackendUnavailable, VisionBackend

log = logging.getLogger(__name__)

MODEL_FILE = "face_landmarker.task"
#: Frames are downscaled so that their longest side is at most this many pixels.
MAX_INPUT_SIDE = 640
#: Mean eye openness (lid gap / eye width) below which the eyes count as closed.
#: Open eyes measure roughly 0.25-0.35.
BLINK_OPENNESS = 0.12
#: A face whose landmark box comes this close (normalised) to the image edge is
#: probably cut off, so its landmarks are less trustworthy.
BORDER_MARGIN = 0.01
MAX_FACES_LIMIT = 5

# Per eye: (first corner, second corner, iris centre, upper lid, lower lid).
# MediaPipe numbers landmarks anatomically, and in a camera image the subject's
# right eye (33/133) appears on the left. Taking 33 -> 133 for one eye and
# 362 -> 263 for the other makes both corner axes point towards image-right, so
# the iris coordinate moves in the same direction in both eyes and can be
# averaged.
EYE_A = (33, 133, 468, 159, 145)
EYE_B = (362, 263, 473, 386, 374)

_EYE_A_CONTOUR = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_EYE_B_CONTOUR = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
_IRIS_A = (468, 469, 470, 471, 472)
_IRIS_B = (473, 474, 475, 476, 477)
_MIN_LANDMARKS = 478
_MIN_EYE_WIDTH_PX = 1.5

# BGR colours for the preview overlay.
_COLOR_EYE = (140, 230, 120)
_COLOR_IRIS = (255, 220, 60)
_COLOR_BOX = (240, 170, 70)
_COLOR_OTHER = (60, 150, 255)
_COLOR_TEXT = (255, 255, 255)


#: Frames after (re)creating the landmarker during which native stderr stays muted.
_QUIET_FRAMES = 2


@contextlib.contextmanager
def _quiet_native_stderr() -> Iterator[None]:
    """Silence MediaPipe's native (absl/TFLite) start-up chatter on stderr.

    MediaPipe prints "Logging before InitGoogle()", XNNPACK and feedback-tensor
    notices straight to file descriptor 2, ignoring GLOG_minloglevel. They are
    harmless but alarming in a terminal. File descriptor 2 is pointed at the null
    device only around landmarker creation and the first frames, and never when
    DEBUG logging is on (useful when diagnosing MediaPipe itself).
    """
    if log.isEnabledFor(logging.DEBUG):
        yield
        return
    try:
        sys.stderr.flush()
        saved = os.dup(2)
    except (OSError, ValueError, AttributeError):
        yield
        return
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
        finally:
            os.close(devnull)
        yield
    finally:
        with contextlib.suppress(OSError):
            os.dup2(saved, 2)
        os.close(saved)


@dataclass(frozen=True, slots=True)
class EyeMetrics:
    """Iris position and lid opening, averaged over both eyes.

    ``iris_h`` is the iris centre projected onto the corner-to-corner axis
    (0 at the image-left corner, 1 at the image-right corner, ~0.5 looking
    straight). ``iris_v`` is the signed distance from that axis in eye widths
    (positive = towards the bottom of the image). ``openness`` is the lid gap in
    eye widths.
    """

    iris_h: float
    iris_v: float
    openness: float


def pose_from_matrix(matrix: Any) -> tuple[float, float, float, float, float, float]:
    """``(yaw, pitch, roll, tx, ty, tz)`` from a 4x4 facial transformation matrix.

    Angles are in degrees; the translation is in MediaPipe's canonical face
    units (roughly centimetres, ``tz`` negative in front of the camera).
    Mirroring the image flips the sign of yaw and roll.
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    r = m[:3, :3]
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    tx, ty, tz = (float(v) for v in m[:3, 3])
    return yaw, pitch, roll, tx, ty, tz


def eye_metrics(points_px: np.ndarray) -> EyeMetrics | None:
    """Compute :class:`EyeMetrics` from landmark positions in pixels.

    ``points_px`` has shape ``(478, 2)`` (or more rows / columns). Pixel units
    matter: normalised landmark coordinates are anisotropic for non-square
    images. Returns ``None`` when the eyes are degenerate (too small).
    """
    if points_px.shape[0] < _MIN_LANDMARKS:
        return None
    h_values: list[float] = []
    v_values: list[float] = []
    openness: list[float] = []
    for c0, c1, iris, lid_top, lid_bottom in (EYE_A, EYE_B):
        a = points_px[c0, :2]
        axis = points_px[c1, :2] - a
        width = float(math.hypot(axis[0], axis[1]))
        if not width >= _MIN_EYE_WIDTH_PX:  # also rejects NaN
            return None
        u = axis / width
        normal = np.array([-u[1], u[0]])  # image-down for an upright face
        rel = points_px[iris, :2] - a
        h_values.append(float(rel @ u) / width)
        v_values.append(float(rel @ normal) / width)
        lid = points_px[lid_top, :2] - points_px[lid_bottom, :2]
        openness.append(float(math.hypot(lid[0], lid[1])) / width)
    return EyeMetrics(
        iris_h=sum(h_values) / 2.0,
        iris_v=sum(v_values) / 2.0,
        openness=sum(openness) / 2.0,
    )


def _downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _box(points: np.ndarray) -> tuple[float, float, float, float]:
    """Normalised ``(x0, y0, x1, y1)`` of landmark points (may exceed 0..1)."""
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    x1, y1 = points[:, 0].max(), points[:, 1].max()
    return float(x0), float(y0), float(x1), float(y1)


def _clipped_xywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in box)
    return (x0, y0, x1 - x0, y1 - y0)


# ----------------------------------------------------------------------------
# Preview drawing helpers, shared with the OpenCV backend. Frames drawn on here
# are only ever shown in the preview window.


def to_bgr(frame: np.ndarray) -> np.ndarray:
    """A new 3-channel BGR copy of a BGR, BGRA or greyscale frame."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    return frame.copy()


def draw_label(image: np.ndarray, lines: list[str]) -> None:
    """Draw a few lines of white text with a dark outline in the top-left corner."""
    w = image.shape[1]
    scale = max(0.4, min(1.2, w / 1000))
    thickness = max(1, round(scale * 2))
    y = int(24 * scale) + 4
    for text in lines:
        cv2.putText(
            image,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            _COLOR_TEXT,
            thickness,
            cv2.LINE_AA,
        )
        y += int(26 * scale) + 4


def observation_label(obs: Observation) -> list[str]:
    """Text lines describing an observation for preview overlays."""
    if obs.face_count == 0:
        return ["no face"]
    lines: list[str] = []
    if obs.head_yaw is not None and obs.head_pitch is not None:
        lines.append(f"yaw {obs.head_yaw:+.0f}  pitch {obs.head_pitch:+.0f}")
    extras = []
    if obs.face_count > 1:
        extras.append(f"{obs.face_count} faces")
    if obs.blink:
        extras.append("eyes closed")
    if obs.features is None:
        extras.append("unusable")
    if extras:
        lines.append(", ".join(extras))
    return lines


@dataclass(slots=True)
class _Landmarks:
    """Primary-face landmarks of the last processed frame, for ``annotate``."""

    timestamp: float
    points: np.ndarray  # (478, 2) normalised
    others: list[tuple[float, float, float, float]]  # clipped (x, y, w, h) of other faces


class MediaPipeBackend(VisionBackend):
    """Head pose and iris features from the MediaPipe Face Landmarker.

    Args:
        model_path: ``face_landmarker.task``; defaults to the bundled model.
        max_faces: Faces to look for (2 enables the shoulder guard; costs CPU).

    Raises:
        BackendUnavailable: MediaPipe is not installed or the model cannot be loaded.
    """

    name: ClassVar[str] = "mediapipe"
    feature_names: ClassVar[tuple[str, ...]] = (
        "yaw",
        "pitch",
        "roll",
        "tx",
        "ty",
        "tz",
        "iris_h",
        "iris_v",
    )
    feature_version: ClassVar[str] = "mp-pose-iris-1"

    def __init__(self, model_path: Path | None = None, max_faces: int = 1) -> None:
        path = Path(model_path) if model_path else paths.model_path(MODEL_FILE)
        try:
            self._mp = import_mediapipe()
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.vision import face_landmarker
        except Exception as exc:
            # Not only ImportError: a broken install (missing DLL, protobuf clash)
            # fails in other ways, and "auto" must still fall back to OpenCV.
            raise BackendUnavailable(
                f"MediaPipe is not available ({exc}); use the 'opencv' backend"
            ) from exc
        self._base_options = base_options
        self._face_landmarker = face_landmarker
        try:
            # Loaded into memory rather than passed as a path: MediaPipe's native
            # file loading cannot handle non-ASCII install paths on Windows.
            self._model = path.read_bytes()
        except OSError as exc:
            raise BackendUnavailable(
                f"MediaPipe model not found at {path} (run scripts/fetch_models.py)"
            ) from exc
        self._num_faces = _clamp_faces(max_faces)
        self._landmarker: Any = None
        self._quiet_frames = 0
        self._rebuild = False
        self._closed = False
        self._last_ts_ms = -1
        self._last: _Landmarks | None = None
        try:
            self._create_landmarker()
        except Exception as exc:
            raise BackendUnavailable(f"MediaPipe could not load its model: {exc}") from exc

    # ----------------------------------------------------------- lifecycle
    def _create_landmarker(self) -> None:
        self._close_landmarker()
        options = self._face_landmarker.FaceLandmarkerOptions(
            base_options=self._base_options.BaseOptions(model_asset_buffer=self._model),
            running_mode=self._mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self._num_faces,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=False,
        )
        with _quiet_native_stderr():
            self._landmarker = self._face_landmarker.FaceLandmarker.create_from_options(options)
        self._quiet_frames = _QUIET_FRAMES
        self._rebuild = False
        log.debug("MediaPipe face landmarker created (num_faces=%d)", self._num_faces)

    def _close_landmarker(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            # Closing explicitly matters: MediaPipe's finaliser raises when it
            # runs during interpreter shutdown.
            try:
                landmarker.close()
            except Exception:
                log.debug("Closing the MediaPipe landmarker failed", exc_info=True)

    def set_max_faces(self, n: int) -> None:
        n = _clamp_faces(n)
        if n != self._num_faces:
            self._num_faces = n
            # Rebuilt lazily on the next frame, i.e. on the worker thread.
            self._rebuild = True

    @property
    def max_faces(self) -> int:
        """Number of faces the landmarker looks for."""
        return self._num_faces

    def close(self) -> None:
        self._closed = True
        self._close_landmarker()
        self._last = None

    # ------------------------------------------------------------ analysis
    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Observation:
        if self._closed:
            raise RuntimeError("MediaPipeBackend is closed")
        started = time.perf_counter()
        if self._rebuild or self._landmarker is None:
            self._create_landmarker()

        height, width = frame_bgr.shape[:2]
        small = _downscale(frame_bgr, MAX_INPUT_SIDE)
        rgb = np.ascontiguousarray(_to_rgb(small))
        # VIDEO mode requires strictly increasing integer millisecond timestamps.
        ts_ms = max(int(timestamp * 1000.0), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._quiet_frames > 0:
            self._quiet_frames -= 1
            with _quiet_native_stderr():
                result = self._landmarker.detect_for_video(image, ts_ms)
        else:
            result = self._landmarker.detect_for_video(image, ts_ms)

        faces = list(result.face_landmarks or [])
        matrices = list(result.facial_transformation_matrixes or [])
        if not faces:
            self._last = None
            return Observation(
                timestamp=timestamp,
                face_count=0,
                inference_ms=(time.perf_counter() - started) * 1e3,
                frame_size=(width, height),
            )

        all_points = [
            np.array([(p.x, p.y) for p in landmarks], dtype=np.float64) for landmarks in faces
        ]
        boxes = [_box(pts) for pts in all_points]
        areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes]
        primary = int(np.argmax(areas))
        points = all_points[primary]
        x0, y0, x1, y1 = boxes[primary]
        face_box = _clipped_xywh(boxes[primary])

        small_h, small_w = small.shape[:2]
        eyes = eye_metrics(points * (small_w, small_h))
        pose = pose_from_matrix(matrices[primary]) if primary < len(matrices) else None

        features: np.ndarray | None = None
        quality = 0.0
        blink = False
        if eyes is not None and pose is not None:
            vector = np.array([*pose, eyes.iris_h, eyes.iris_v], dtype=np.float64)
            if np.all(np.isfinite(vector)):
                features = vector
                touches_border = (
                    x0 <= BORDER_MARGIN
                    or y0 <= BORDER_MARGIN
                    or x1 >= 1.0 - BORDER_MARGIN
                    or y1 >= 1.0 - BORDER_MARGIN
                )
                quality = 0.5 if touches_border else 1.0
                blink = eyes.openness < BLINK_OPENNESS

        self._last = _Landmarks(
            timestamp=timestamp,
            points=points,
            others=[_clipped_xywh(b) for i, b in enumerate(boxes) if i != primary],
        )
        return Observation(
            timestamp=timestamp,
            face_count=len(faces),
            features=features,
            quality=quality,
            blink=blink,
            head_yaw=pose[0] if pose is not None else None,
            head_pitch=pose[1] if pose is not None else None,
            face_box=face_box,
            inference_ms=(time.perf_counter() - started) * 1e3,
            frame_size=(width, height),
        )

    # ------------------------------------------------------------- preview
    def annotate(self, frame_bgr: np.ndarray, observation: Observation) -> np.ndarray:
        out = to_bgr(frame_bgr)
        h, w = out.shape[:2]
        thickness = max(1, round(max(h, w) / 640))
        last = self._last
        if last is not None and last.timestamp == observation.timestamp:
            pts = np.round(last.points * (w, h)).astype(np.int32)
            for contour in (_EYE_A_CONTOUR, _EYE_B_CONTOUR):
                cv2.polylines(out, [pts[list(contour)]], True, _COLOR_EYE, thickness, cv2.LINE_AA)
            for iris in (_IRIS_A, _IRIS_B):
                centre = last.points[iris[0]] * (w, h)
                ring = last.points[list(iris[1:])] * (w, h)
                radius = float(np.mean(np.hypot(*(ring - centre).T)))
                cx, cy = (round(float(v)) for v in centre)
                cv2.circle(
                    out, (cx, cy), max(1, round(radius)), _COLOR_IRIS, thickness, cv2.LINE_AA
                )
                cv2.circle(out, (cx, cy), max(1, thickness), _COLOR_IRIS, -1, cv2.LINE_AA)
            for bx, by, bw, bh in last.others:
                draw_box(out, (bx, by, bw, bh), _COLOR_OTHER, thickness)
        if observation.face_box is not None:
            draw_box(out, observation.face_box, _COLOR_BOX, thickness)
        draw_label(out, observation_label(observation))
        return out


def draw_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a normalised ``(x, y, w, h)`` box onto ``image`` in place."""
    h, w = image.shape[:2]
    x, y, bw, bh = box
    p0 = (round(x * w), round(y * h))
    p1 = (round((x + bw) * w), round((y + bh) * h))
    cv2.rectangle(image, p0, p1, color, thickness, cv2.LINE_AA)


def _clamp_faces(n: int) -> int:
    return max(1, min(int(n), MAX_FACES_LIMIT))
