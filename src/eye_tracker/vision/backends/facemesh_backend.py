"""Face mesh backend: 478 face landmarks (with irises) and head pose via OpenCV DNN.

This is the accurate default. It runs MediaPipe's face landmark network (the
``face_landmarks_detector.tflite`` member of the Face Landmarker bundle,
Apache-2.0, see ``models/NOTICE.md``) with OpenCV's DNN module. The MediaPipe
runtime itself is deliberately not used: it contains a usage-logging client
that uploads to Google servers, and this app never touches the network.

Each analysed frame costs about one landmark inference (about 12 ms on one core
of a current desktop CPU, 10 ms of it in the network; a frame without a face
costs one YuNet detection, about 7 ms):

1. **Region of interest.** A square crop, rotated so that the eye line is
   horizontal, is tracked from the previous frame's landmarks (MediaPipe's rule:
   landmark bounding box, 1.5x its longer side). When there is no previous face,
   when the network's face-presence score drops below 0.5 or when the crop
   degenerates, YuNet (the lite backend's detector) finds faces and the best
   one (largest, then most central) seeds a new crop. When the network rejects
   that seed or accepts it only half-heartedly, it is also tried rotated
   (YuNet misjudges strong head roll) and the most face-like crop wins; when
   nothing works, comparable runners-up are tried.
2. **Landmarks.** The 256x256 crop goes through the network; its 478 points
   (468 mesh points plus five per iris) are mapped back into the frame. When
   the face moved noticeably since the crop was chosen, the landmarks are
   biased towards the old position, so the network runs again on the crop
   they ask for (and after a large jump on a crop from a fresh YuNet detection)
   until the crop sits on the face: a still or slowly moving face costs one
   inference, a posture shift between two frames two to four (see
   ``FaceMeshBackend._settle``).
3. **Head pose.** ``cv2.solvePnP`` fits MediaPipe's canonical face mesh to the
   landmarks MediaPipe itself uses for rigid fitting (its Procrustes basis),
   with approximate intrinsics (focal length = longer frame side, principal
   point = frame centre) and the previous pose as the initial guess.
4. **Eyes.** Iris position inside each eye and lid opening (blinks).

With the shoulder guard on (``max_faces >= 2``) YuNet additionally counts faces
at :data:`GUARD_DETECT_WIDTH` pixels at most every :data:`GUARD_PERIOD_S`
seconds, which finds onlookers several metres away. With the guard off it still
looks every :data:`PRIMARY_CHECK_S` seconds (one 320 px detection, about 0.5 %
of one core): either way, a clearly stronger face elsewhere in the picture
takes over as the primary face, so that a poster or a colleague acquired while
the user was away does not keep the gaze control once the user is back.

Feature vector (:attr:`FaceMeshBackend.feature_names`):

* ``yaw``, ``pitch``, ``roll`` in degrees. Yaw is positive when the nose points
  towards the right edge of the camera image, pitch when it points towards the
  bottom edge (head lowered), roll when the head is tilted counter-clockwise in
  the image. Mirroring the image flips the sign of yaw and roll.
* ``tx``, ``ty``, ``tz``: position of the head centre relative to the camera in
  centimetres (x right, y down, z away from the camera), only as accurate as the
  assumed focal length.
* ``iris_h``, ``iris_v``: see :class:`EyeMetrics`.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from ... import paths
from ...types import Observation
from ..threads import limit_opencv_threads
from .base import BackendUnavailable, VisionBackend
from .face_geometry import FaceGeometry, GeometryError, load_face_geometry
from .lite_backend import create_yunet
from .overlay import (
    COLOR_BOX,
    COLOR_EYE,
    COLOR_IRIS,
    COLOR_OTHER,
    COLOR_POSE,
    as_bgr,
    draw_box,
    draw_label,
    line_thickness,
    observation_label,
    to_bgr,
)

log = logging.getLogger(__name__)

MODEL_FILE = "face_landmarks_detector.tflite"
GEOMETRY_FILE = "geometry_pipeline_metadata_landmarks.binarypb"

#: Side of the square network input in pixels.
INPUT_SIZE = 256
#: Crop side relative to the longer side of the face (MediaPipe uses 1.5).
ROI_SCALE = 1.5
#: Face-presence probability below which the tracked face counts as lost.
PRESENCE_THRESHOLD = 0.5
#: A crop smaller than this (frame pixels) is too coarse to track.
MIN_ROI_SIDE = 24.0
#: YuNet input width for (re-)acquiring the face.
DETECT_WIDTH = 320
#: YuNet input width while the shoulder guard counts faces. Measured with a face
#: photo scaled into a 640x480 frame, YuNet finds faces from about 20 px wide at
#: 320 px input, 14 px at 480 and 12 px at 640. With a typical 65-78° webcam a
#: face is roughly 60-75 px divided by its distance in metres wide at 640 px, so
#: 480 px reaches onlookers about 4 m away (320 px: about 2.5-3 m) for half the
#: cost of 640 px (about 13 ms against 26 ms on one core).
GUARD_DETECT_WIDTH = 480
#: Minimum seconds between shoulder-guard face counts; the count is reused in
#: between (the guard itself waits seconds before reacting).
GUARD_PERIOD_S = 0.5
#: YuNet detections below this confidence are ignored (and not counted as faces).
DETECTION_SCORE = 0.6
#: A face elsewhere in the image whose primary score (see :func:`primary_score`)
#: is this much higher than the tracked face's takes over as the primary face.
PRIMARY_SWITCH_RATIO = 1.3
#: How much less a face at the image corner counts than the same face at the
#: centre when choosing the primary face (the user usually sits in front of the
#: camera; posters, TVs and passers-by are usually off to the side). Mild on
#: purpose: size dominates, centrality decides between similar faces.
PRIMARY_CENTRE_WEIGHT = 0.3
#: With the shoulder guard off, YuNet looks for a better primary face this often
#: (seconds) while a face is tracked, so a face acquired while the user was away
#: (a poster, a colleague) does not keep the role once the user is back. One
#: 320 px detection costs about 7 ms, i.e. about 0.5 % of one core at this period.
PRIMARY_CHECK_S = 1.5
#: ... and this soon after a face was (re-)acquired, when the user may still be
#: on the way into the picture.
PRIMARY_RECHECK_S = 0.5
#: YuNet's box side divided by the side of the 468-point landmark box, for the
#: same face (measured 1.10-1.18 at 320 and 480 px input). Converts a tracked
#: face's size to YuNet's scale when YuNet itself did not report it.
YUNET_TO_MESH_SIDE = 1.13
#: When the best detection fails the landmark network (a profile, a hand over the
#: face), up to this many detections are tried in all ...
MAX_SEED_CANDIDATES = 3
#: ... but only those scoring at least this share of the best one. A face much
#: smaller than the user's is a background face, and the user turning away must
#: not hand it the gaze control.
SEED_CANDIDATE_MIN_RATIO = 0.5
#: YuNet's eye points hardly follow a rolled head at 320 px (a 55° roll reads as
#: about 5°), and the landmark network rejects crops more than about 40° off, or
#: worse, fits them with a partly rolled face. A seed that fails, or passes with
#: a presence logit below ROLL_RETRY_BELOW_LOGIT, is therefore also tried rotated
#: by this angle either way, and the most face-like crop wins ...
ROLL_RETRY_DEG = 45.0
#: ... (aligned faces score a logit of about 12-22, 9 when heavily blurred; such
#: partial fits about 2-8) ...
ROLL_RETRY_BELOW_LOGIT = 10.0
#: ... at most this often (seconds) when it does not help, so a face the network
#: can never read (a profile) does not cost two extra inferences on every frame.
ROLL_RETRY_PERIOD_S = 2.0
#: A crop counts as settled on the face when the landmarks it produced ask for a
#: crop whose centre is at most this share of its side away ...
ROI_SETTLE_SHIFT = 0.04
#: ... and whose side differs by at most this share (still-face jitter stays
#: below about 0.01 for both).
ROI_SETTLE_SCALE = 0.035
#: A crop whose centre is further off than this share of its side (or whose side
#: is off by more than ROI_SETTLE_SCALE) is re-run within the same frame. It
#: pulls the landmarks towards where the face was: by up to 10° of pitch and 0.1
#: of iris travel after a 5-8 cm posture shift between two frames, and upward
#: jumps show up mostly as a changed side, because the network follows them only
#: a fraction of the way. Between the two shift limits the bias stays below
#: about 1.5° and the next frame removes it (the motion gate then does not hold
#: the observation, see ``VisionBackend.settled``), so a steadily moving head
#: costs no extra inference.
ROI_CATCH_UP_SHIFT = 0.08
#: ... provided that the network is sure it sees a face in the crop (a presence
#: logit of at least this). Far behind a large jump, an upward one especially,
#: the network follows the face only a fraction of the way per pass, so a crop
#: can ask for a shift below ROI_CATCH_UP_SHIFT while its landmarks are still
#: 35-45 px (12-20° of pitch) off. Measured on drawn faces after such jumps,
#: those crops scored a logit of 4-6, crops in the same band that were on the
#: face (landmarks within 1.5 px) 19-23; for photos see ROLL_RETRY_BELOW_LOGIT.
#: Without this, which of the two a jump ended on depended on pixel-level details
#: (a 1 px shift of the face, sensor noise, the CPU's floating-point rounding).
CATCH_UP_MIN_LOGIT = 10.0
#: After this many analysed frames in a row whose crop did not settle, the
#: result counts as settled anyway. A crop catching up with a jump settles
#: within a frame or two; one that keeps shifting slightly on a still picture is
#: jitter, and the motion gate must be able to hold it to keep the CPU use low.
MAX_UNSETTLED_FRAMES = 3
#: Mean eye openness (lid gap / eye width) below which the eyes always count as
#: closed. Open eyes measure roughly 0.2-0.35.
BLINK_OPENNESS = 0.12
#: Once the user's own open-eye baseline is known, the blink threshold becomes
#: ``BLINK_BASELINE_RATIO * baseline``, clamped to [BLINK_MIN_OPENNESS,
#: BLINK_OPENNESS]: narrow eyes or a downward gaze never read as a blink.
BLINK_MIN_OPENNESS = 0.08
BLINK_BASELINE_RATIO = 0.5
#: Open-eye samples needed before the baseline is trusted, and how many are kept.
BLINK_BASELINE_MIN_SAMPLES = 30
BLINK_BASELINE_WINDOW = 90
#: A face whose landmark box comes this close (normalised) to the image edge is
#: probably cut off, so its landmarks are less trustworthy.
BORDER_MARGIN = 0.01
MAX_FACES_LIMIT = 5

_OUTPUT_NAMES = ["Identity", "Identity_1"]  # landmarks, face-presence logit
_NUM_LANDMARKS = 478
_MESH_POINTS = 468
_MIN_EYE_WIDTH_PX = 1.5

# Per eye: (first corner, second corner, iris centre, upper lid, lower lid).
# Landmarks are numbered anatomically, and in a camera image the subject's
# right eye (33/133) appears on the left. Taking 33 -> 133 for one eye and
# 362 -> 263 for the other makes both corner axes point towards image-right, so
# the iris coordinate moves in the same direction in both eyes and can be
# averaged.
EYE_A = (33, 133, 468, 159, 145)
EYE_B = (362, 263, 473, 386, 374)
#: Outer eye corners on the image-left and image-right side: the eye line.
_EYE_LINE = (33, 263)
_NOSE_TIP = 4
#: Preview arrow showing where the face points: 5 cm out of the face (the pose
#: model frame's -z axis).
_POSE_ARROW_CM = np.array((0.0, 0.0, -5.0))

_EYE_A_CONTOUR = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_EYE_B_CONTOUR = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
_IRIS_A = (468, 469, 470, 471, 472)
_IRIS_B = (473, 474, 475, 476, 477)

Box = tuple[float, float, float, float]


# ----------------------------------------------------------------------------
# Pure helpers (unit-tested without the network)


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


def eye_metrics(points_px: np.ndarray) -> EyeMetrics | None:
    """Compute :class:`EyeMetrics` from landmark positions in pixels.

    ``points_px`` has shape ``(478, 2)`` (or more rows / columns). Pixel units
    matter: normalised landmark coordinates are anisotropic for non-square
    images. Returns ``None`` when the eyes are degenerate (too small).
    """
    if points_px.shape[0] < _NUM_LANDMARKS:
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


class BlinkDetector:
    """Decides whether the eyes are closed, relative to the user's own eyes.

    A fixed openness threshold misfires for narrow eyes and for a downward gaze
    (the lids close in when looking at a monitor below the camera). The detector
    keeps a rolling median of open-eye readings and, once it has enough of them,
    flags a blink below half of that baseline. The threshold never rises above
    the absolute :data:`BLINK_OPENNESS`, so wide eyes behave as before.
    """

    def __init__(
        self,
        absolute: float = BLINK_OPENNESS,
        minimum: float = BLINK_MIN_OPENNESS,
        ratio: float = BLINK_BASELINE_RATIO,
        min_samples: int = BLINK_BASELINE_MIN_SAMPLES,
        window: int = BLINK_BASELINE_WINDOW,
    ) -> None:
        self.absolute = float(absolute)
        self.minimum = float(minimum)
        self.ratio = float(ratio)
        self.min_samples = max(1, int(min_samples))
        self._history: deque[float] = deque(maxlen=max(self.min_samples, int(window)))

    @property
    def threshold(self) -> float:
        """Current openness threshold below which the eyes count as closed."""
        if len(self._history) < self.min_samples:
            return self.absolute
        baseline = float(np.median(np.fromiter(self._history, dtype=np.float64)))
        return min(self.absolute, max(self.minimum, self.ratio * baseline))

    def update(self, openness: float) -> bool:
        """Classify one reading (``True`` = closed) and learn from open eyes.

        Until the baseline exists every finite reading is learnt, not only those
        above the absolute threshold: eyes that stay below it (narrow eyes, a
        monitor below the camera) would otherwise never teach the detector
        anything and read as closed forever. The median shrugs off the few blinks
        among them. Afterwards only open eyes are learnt, so a long stretch of
        closed eyes cannot drag the baseline down.
        """
        if not math.isfinite(openness):
            return False
        closed = openness < self.threshold
        if not closed or len(self._history) < self.min_samples:
            self._history.append(openness)
        return closed

    def reset(self) -> None:
        self._history.clear()


@dataclass(frozen=True, slots=True)
class Roi:
    """A square, rotated crop of the frame that the landmark network sees.

    ``(cx, cy)`` is the centre and ``side`` the side length in frame pixels;
    ``angle`` is the direction of the eye line in degrees (image coordinates,
    y down), which the crop rotates to horizontal.
    """

    cx: float
    cy: float
    side: float
    angle: float

    def transform(self) -> np.ndarray:
        """2x3 affine matrix mapping frame pixels to network-input pixels."""
        # Positive angles rotate counter-clockwise on screen, which levels an eye
        # line that descends towards image-right by that angle.
        m = cv2.getRotationMatrix2D((self.cx, self.cy), self.angle, INPUT_SIZE / self.side)
        m[0, 2] += INPUT_SIZE / 2.0 - self.cx
        m[1, 2] += INPUT_SIZE / 2.0 - self.cy
        return m

    def valid(self, width: int, height: int) -> bool:
        """True when the crop is finite, large enough and centred inside the frame."""
        return (
            all(math.isfinite(v) for v in (self.cx, self.cy, self.side, self.angle))
            and self.side >= MIN_ROI_SIDE
            and 0.0 <= self.cx < width
            and 0.0 <= self.cy < height
        )


def _eye_line_angle(left: np.ndarray, right: np.ndarray) -> float:
    return math.degrees(math.atan2(float(right[1] - left[1]), float(right[0] - left[0])))


def roi_from_detection(face: np.ndarray) -> Roi:
    """Crop for a YuNet detection row (pixels: box, right eye, left eye, ...)."""
    x, y, w, h = (float(v) for v in face[:4])
    # YuNet's "right eye" is the one on the image left.
    angle = _eye_line_angle(face[4:6], face[6:8])
    return Roi(x + w / 2.0, y + h / 2.0, max(w, h) * ROI_SCALE, angle)


def roi_from_landmarks(points_px: np.ndarray) -> Roi:
    """Crop for tracking a face into the next frame (MediaPipe's rule)."""
    mesh = points_px[:_MESH_POINTS, :2]
    x0, y0 = mesh.min(axis=0)
    x1, y1 = mesh.max(axis=0)
    angle = _eye_line_angle(points_px[_EYE_LINE[0]], points_px[_EYE_LINE[1]])
    side = max(float(x1 - x0), float(y1 - y0)) * ROI_SCALE
    return Roi(float(x0 + x1) / 2.0, float(y0 + y1) / 2.0, side, angle)


def crop_settled(
    used: Roi, wanted: Roi, shift: float = ROI_SETTLE_SHIFT, scale: float = ROI_SETTLE_SCALE
) -> bool:
    """True when ``wanted`` (the crop the landmarks ask for) matches ``used``.

    Landmarks from a crop that is centred on the face are trustworthy; when the
    face sits off-centre in the crop (it moved since the crop was chosen) they
    are pulled towards the crop centre, and the crop they ask for is off by a
    noticeable share of its side. ``shift`` and ``scale`` are the tolerated
    centre offset and side change, as shares of ``used.side`` (see
    :data:`ROI_SETTLE_SHIFT`).
    """
    if not (used.side > 0.0 and all(math.isfinite(v) for v in (wanted.cx, wanted.cy, wanted.side))):
        return False
    offset = math.hypot(wanted.cx - used.cx, wanted.cy - used.cy) / used.side
    return offset <= shift and abs(wanted.side / used.side - 1.0) <= scale


def crop_caught_up(used: Roi, wanted: Roi, logit: float) -> bool:
    """True when a frame's landmark passes may stop at the crop ``used``.

    That is when it is settled on the face (:func:`crop_settled`), or when it is
    catching up with a moving face (``wanted`` within :data:`ROI_CATCH_UP_SHIFT`)
    and the network is confident that it sees a face in it (``logit``, the
    presence logit of the pass on ``used``, of at least
    :data:`CATCH_UP_MIN_LOGIT`). See ``FaceMeshBackend._settle``.
    """
    if crop_settled(used, wanted):
        return True
    return logit >= CATCH_UP_MIN_LOGIT and crop_settled(used, wanted, shift=ROI_CATCH_UP_SHIFT)


def primary_score(side: float, cx: float, cy: float, width: int, height: int) -> float:
    """How strongly a face of ``side`` pixels centred at ``(cx, cy)`` claims to be the user.

    The face side, reduced by up to :data:`PRIMARY_CENTRE_WEIGHT` the further the
    face is from the image centre (0 at the centre, the full weight at a corner).
    """
    half_w, half_h = width / 2.0, height / 2.0
    offset = math.hypot((cx - half_w) / half_w, (cy - half_h) / half_h) / math.sqrt(2.0)
    return side * (1.0 - PRIMARY_CENTRE_WEIGHT * min(offset, 1.0))


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Head rotation (degrees) and position (centimetres), see the module docs."""

    yaw: float
    pitch: float
    roll: float
    tx: float
    ty: float
    tz: float


def pose_from_rotation(rotation: np.ndarray, translation: np.ndarray | Sequence[float]) -> HeadPose:
    """:class:`HeadPose` from a model-to-camera rotation matrix and translation.

    The model frame is the canonical face flipped to OpenCV's camera convention
    (x right, y down, z away from the camera), so a face looking straight into
    the camera has the identity rotation.
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    nose = -r[:, 2]  # the face's forward direction, towards the camera when frontal
    yaw = math.degrees(math.atan2(nose[0], -nose[2]))
    pitch = math.degrees(math.atan2(nose[1], math.hypot(nose[0], nose[2])))
    # Direction of the model x axis in the image; image y points down, so a
    # counter-clockwise tilt lifts its right end (negative y).
    roll = math.degrees(math.atan2(-r[1, 0], r[0, 0]))
    tx, ty, tz = (float(v) for v in np.asarray(translation, dtype=np.float64).ravel()[:3])
    return HeadPose(yaw, pitch, roll, tx, ty, tz)


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Approximate pinhole intrinsics: focal length = longer side, centred."""
    f = float(max(width, height))
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


class PoseEstimator:
    """Head pose by fitting the canonical face mesh to the landmarks."""

    def __init__(self, geometry: FaceGeometry) -> None:
        ids = np.array(geometry.basis_ids, dtype=np.intp)
        self._ids = ids
        # Canonical frame (x right, y up, z towards the camera) -> OpenCV-style
        # object frame (x right, y down, z away from the camera).
        self._object = np.ascontiguousarray(geometry.vertices[ids] * (1.0, -1.0, -1.0))
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None
        self._size: tuple[int, int] | None = None
        self.camera: np.ndarray = camera_matrix(1, 1)

    @property
    def rvec(self) -> np.ndarray | None:
        return self._rvec

    @property
    def tvec(self) -> np.ndarray | None:
        return self._tvec

    def reset(self) -> None:
        """Forget the previous pose (a new face or a new frame size)."""
        self._rvec = None
        self._tvec = None

    def estimate(self, points_px: np.ndarray, width: int, height: int) -> HeadPose | None:
        """Pose for ``(478, 2)`` landmark pixels in a ``width`` x ``height`` frame."""
        if (width, height) != self._size:
            self._size = (width, height)
            self.camera = camera_matrix(width, height)
            self.reset()
        image = np.ascontiguousarray(points_px[self._ids, :2], dtype=np.float64)
        if not np.all(np.isfinite(image)):
            self.reset()
            return None
        solution = None
        if self._rvec is not None and self._tvec is not None:
            solution = self._solve(image, cv2.SOLVEPNP_ITERATIVE, self._rvec, self._tvec)
        if solution is None:
            # A global solver for a fresh face (or when the tracked solution
            # drifted into an implausible one).
            solution = self._solve(image, cv2.SOLVEPNP_SQPNP)
        if solution is None:
            self.reset()
            return None
        self._rvec, self._tvec = solution
        rotation, _ = cv2.Rodrigues(self._rvec)
        return pose_from_rotation(rotation, self._tvec)

    def _solve(
        self,
        image: np.ndarray,
        method: int,
        rvec: np.ndarray | None = None,
        tvec: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        guess = rvec is not None and tvec is not None
        try:
            ok, r, t = cv2.solvePnP(
                self._object,
                image,
                self.camera,
                None,
                rvec.copy() if rvec is not None else None,
                tvec.copy() if tvec is not None else None,
                useExtrinsicGuess=guess,
                flags=method,
            )
        except cv2.error:
            log.debug("solvePnP failed", exc_info=True)
            return None
        if not ok or not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
            return None
        rotation, _ = cv2.Rodrigues(r)
        # In front of the camera and facing it (nose towards the lens).
        if float(t.ravel()[2]) <= 0.0 or float(rotation[2, 2]) <= 0.0:
            return None
        return r.reshape(3, 1), t.reshape(3, 1)


# ----------------------------------------------------------------------------
# Model loading


def _read_tflite(model: str | np.ndarray) -> cv2.dnn.Net:
    # OpenCV 5 defaults to its new graph engine, which cannot deliver this
    # network's named outputs correctly; the classic engine is exact. OpenCV 4.x
    # only has the classic engine (and no `engine` argument).
    if hasattr(cv2.dnn, "ENGINE_CLASSIC"):
        return cv2.dnn.readNetFromTFLite(model, engine=cv2.dnn.ENGINE_CLASSIC)
    return cv2.dnn.readNetFromTFLite(model)


def _windows_short_path(path: Path) -> str | None:
    """The 8.3 alias of ``path`` if Windows provides a pure-ASCII one."""
    if sys.platform != "win32":
        return None
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    get_short = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
    length = int(get_short(str(path), buffer, len(buffer)))
    if 0 < length < len(buffer) and buffer.value.isascii():
        return str(buffer.value)
    return None


def self_test_blob() -> np.ndarray:
    """The fixed input of the load-time self-test: a different gradient per channel.

    Asymmetric on purpose, so that a runtime that mixes up the tensor layout or
    the channel order produces visibly different landmarks.
    """
    ramp = np.linspace(0.0, 1.0, INPUT_SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(ramp, ramp)
    return np.stack([xx, yy, (xx + yy) / 2.0])[None]


#: Reference outputs for :func:`self_test_blob`, recorded with OpenCV 5.0's
#: classic engine (MediaPipe's own runtime agrees to within 0.005 of the frame
#: on real faces): presence logit, mean landmark (x, y) and iris 468 (x, y) in
#: network-input pixels.
_SELF_TEST_FLAG = -8.30
_SELF_TEST_MEAN = (127.64, 116.29)
_SELF_TEST_IRIS = (95.10, 85.45)
_SELF_TEST_PX_TOLERANCE = 2.0
_SELF_TEST_FLAG_TOLERANCE = 1.5


def _check_landmark_net(net: cv2.dnn.Net) -> None:
    """Make sure the network runs and computes what it should.

    Some OpenCV engines load the model but return the wrong tensors (OpenCV
    5.0's new engine, for example); comparing one fixed input against reference
    values turns such a runtime into a clean "unavailable" instead of garbage
    landmarks.

    Raises:
        ValueError: The outputs have the wrong shape or values.
    """
    net.setInput(self_test_blob())
    landmarks, flag = net.forward(_OUTPUT_NAMES)
    if landmarks.size != _NUM_LANDMARKS * 3 or flag.size != 1:
        raise ValueError(
            f"unexpected output shapes {landmarks.shape} and {flag.shape} (not the landmark model)"
        )
    points = landmarks.reshape(-1, 3)[:, :2].astype(np.float64)
    mean = points.mean(axis=0)
    iris = points[EYE_A[2]]
    logit = float(flag.ravel()[0])
    worst = max(
        float(np.abs(mean - _SELF_TEST_MEAN).max()), float(np.abs(iris - _SELF_TEST_IRIS).max())
    )
    if not (
        worst <= _SELF_TEST_PX_TOLERANCE
        and abs(logit - _SELF_TEST_FLAG) <= _SELF_TEST_FLAG_TOLERANCE
    ):
        raise ValueError(
            f"the model computes wrong values (landmarks off by {worst:.1f} px, "
            f"presence logit {logit:.2f} instead of {_SELF_TEST_FLAG:.2f})"
        )


def load_landmark_net(path: Path) -> cv2.dnn.Net:
    """Load and self-test the landmark network.

    OpenCV opens file paths with the ANSI code page on Windows, so non-ASCII
    paths fail there; in-memory loading avoids that, but OpenCV 5.0 ignores the
    engine choice for in-memory models (and its default engine cannot run this
    network). The loaders are therefore tried in order: the path itself (unless
    it is non-ASCII on Windows), the in-memory model, and the Windows 8.3 short
    path. Each result must pass :func:`_check_landmark_net`.

    Raises:
        BackendUnavailable: The model is missing or cannot be run correctly.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BackendUnavailable(
            f"Face landmark model not found at {path} (run scripts/fetch_models.py)"
        ) from exc
    loaders: list[tuple[str, Callable[[], cv2.dnn.Net]]] = []
    text = str(path)
    windows_unsafe = sys.platform == "win32" and not text.isascii()
    if not windows_unsafe:
        loaders.append(("path", lambda: _read_tflite(text)))
    loaders.append(("memory", lambda: _read_tflite(np.frombuffer(data, dtype=np.uint8))))
    if windows_unsafe:
        short = _windows_short_path(path)
        if short is not None:
            loaders.append(("short path", lambda: _read_tflite(short)))
    errors: list[str] = []
    for label, load in loaders:
        try:
            net = load()
            _check_landmark_net(net)
        except (cv2.error, ValueError, TypeError) as exc:
            detail = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            errors.append(f"{label}: {detail}")
            log.debug("Loading the landmark model via %s failed", label, exc_info=True)
            continue
        return net
    hint = (
        " (moving the app to a folder with an ASCII-only path may help)" if windows_unsafe else ""
    )
    raise BackendUnavailable(
        f"OpenCV {cv2.__version__} cannot run the face landmark model{hint}: " + "; ".join(errors)
    )


# ----------------------------------------------------------------------------
# The backend


@dataclass(slots=True)
class _Drawing:
    """What ``annotate`` needs from the last processed frame."""

    timestamp: float
    points: np.ndarray | None  # (478, 2) normalised, primary face
    others: list[Box]  # normalised boxes of the other faces
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    camera: np.ndarray | None


def _det_box(face: np.ndarray, width: int, height: int) -> Box:
    """Clipped, normalised ``(x, y, w, h)`` of a YuNet row in frame pixels."""
    x, y, w, h = (float(v) for v in face[:4])
    x0, y0 = max(x / width, 0.0), max(y / height, 0.0)
    x1, y1 = min((x + w) / width, 1.0), min((y + h) / height, 1.0)
    return (x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0))


def _det_side(face: np.ndarray) -> float:
    return max(float(face[2]), float(face[3]))


def _det_centre(face: np.ndarray) -> tuple[float, float]:
    return float(face[0] + face[2] / 2.0), float(face[1] + face[3] / 2.0)


def _det_score(face: np.ndarray, width: int, height: int) -> float:
    return primary_score(_det_side(face), *_det_centre(face), width, height)


def _in_crop(face: np.ndarray, roi: Roi) -> bool:
    """True when the detection's centre lies inside the (unrotated) crop square."""
    cx, cy = _det_centre(face)
    half = roi.side / 2.0
    return abs(cx - roi.cx) < half and abs(cy - roi.cy) < half


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class FaceMeshBackend(VisionBackend):
    """Head pose and iris features from MediaPipe's face landmark network.

    Creating an instance caps OpenCV's process-wide thread pool (see
    :mod:`eye_tracker.vision.threads`).

    Args:
        model_path: Landmark network (``face_landmarks_detector.tflite``).
        max_faces: Faces to report; 2 or more enables shoulder-guard counting.
        geometry_path: Canonical face geometry (``.binarypb``).
        detector_path: YuNet face detector (``.onnx``).

    Raises:
        BackendUnavailable: OpenCV lacks the needed modules or a model cannot be
            loaded.
    """

    name: ClassVar[str] = "facemesh"
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
    feature_version: ClassVar[str] = "facemesh-pose-iris-1"
    gaze_features: ClassVar[tuple[str, ...]] = ("yaw", "pitch", "iris_h", "iris_v")

    def __init__(
        self,
        model_path: Path | None = None,
        max_faces: int = 1,
        *,
        geometry_path: Path | None = None,
        detector_path: Path | None = None,
    ) -> None:
        if not hasattr(cv2, "dnn") or not hasattr(cv2.dnn, "readNetFromTFLite"):
            raise BackendUnavailable(
                f"OpenCV {cv2.__version__} cannot read TFLite models (4.10 or newer is required)"
            )
        limit_opencv_threads()
        try:
            geometry = load_face_geometry(
                Path(geometry_path) if geometry_path else paths.model_path(GEOMETRY_FILE)
            )
        except GeometryError as exc:
            raise BackendUnavailable(
                f"Face geometry unavailable: {exc} (run scripts/fetch_models.py)"
            ) from exc
        self._net: cv2.dnn.Net | None = load_landmark_net(
            Path(model_path) if model_path else paths.model_path(MODEL_FILE)
        )
        self._detector_model = Path(detector_path) if detector_path else None
        self._detector: cv2.FaceDetectorYN | None = create_yunet(self._detector_model, DETECT_WIDTH)
        self._guard_detector: cv2.FaceDetectorYN | None = None
        self._detector_sizes: dict[int, tuple[int, int]] = {}
        self._pose = PoseEstimator(geometry)
        # Nose tip in the pose model frame, for the preview's direction line.
        self._nose_tip = geometry.vertices[_NOSE_TIP] * (1.0, -1.0, -1.0)
        self._blink = BlinkDetector()
        self._max_faces = _clamp_faces(max_faces)
        self._roi: Roi | None = None
        self._settled = True
        self._unsettled_frames = 0
        self._frame_size: tuple[int, int] | None = None
        # Other faces (normalised boxes) from the latest shoulder-guard count,
        # reused until the next count is due.
        self._guard_others: list[Box] = []
        self._guard_ts = -math.inf
        # Primary-face check with the guard off: time of the last one and the
        # period until the next.
        self._primary_ts = -math.inf
        self._primary_period = PRIMARY_CHECK_S
        # Last time rotated seeds were tried in vain (see ROLL_RETRY_PERIOD_S).
        self._roll_retry_ts = -math.inf
        self._last: _Drawing | None = None
        self._landmarks: np.ndarray | None = None
        self._closed = False

    # ----------------------------------------------------------- lifecycle
    def set_max_faces(self, n: int) -> None:
        n = _clamp_faces(n)
        if n == self._max_faces:
            return
        self._max_faces = n
        # Count again on the next frame with the new setting.
        self._guard_others = []
        self._guard_ts = -math.inf

    @property
    def max_faces(self) -> int:
        """Most faces reported in ``face_count``."""
        return self._max_faces

    @property
    def settled(self) -> bool:
        """``False`` while the tracking crop is still catching up with a moved face.

        A frame re-runs the network within itself only after a large jump (see
        ``_settle``); after a moderate move, or a jump that a few passes could
        not quite follow, the crop the landmarks ask for is still off by more
        than :data:`ROI_SETTLE_SHIFT` / :data:`ROI_SETTLE_SCALE` and the next
        frame continues from it. The remaining bias is small, but the motion
        gate must not repeat it for seconds. ``True`` again after
        :data:`MAX_UNSETTLED_FRAMES` unsettled frames in a row.
        """
        return self._settled

    def reset(self) -> None:
        """Forget the tracked face; the next frame looks for faces afresh.

        The vision worker calls this when the camera was released (pause,
        privacy mode, a lock) or replaced: the picture may have changed
        completely meanwhile, and a crop left over from before could latch onto
        whatever face is there now. The blink baseline is kept, it belongs to
        the user rather than to a camera session.
        """
        self._forget_tracking()
        self._last = None
        self._landmarks = None

    @property
    def landmarks(self) -> np.ndarray | None:
        """Normalised ``(478, 2)`` landmarks of the last frame's primary face."""
        return None if self._landmarks is None else self._landmarks.copy()

    @property
    def blink_threshold(self) -> float:
        """Current eye-openness threshold for blinks (diagnostics)."""
        return self._blink.threshold

    def close(self) -> None:
        self._closed = True
        self._net = None
        self._detector = None
        self._guard_detector = None
        self._forget_tracking()
        self._last = None
        self._landmarks = None
        self._blink.reset()

    def _forget_tracking(self) -> None:
        """Drop everything carried from frame to frame except the blink baseline."""
        self._roi = None
        self._settled = True
        self._unsettled_frames = 0
        self._guard_others = []
        self._guard_ts = -math.inf
        self._primary_ts = -math.inf
        self._primary_period = PRIMARY_CHECK_S
        self._roll_retry_ts = -math.inf
        self._pose.reset()

    # ------------------------------------------------------------ analysis
    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Observation:
        if self._closed or self._net is None:
            raise RuntimeError("FaceMeshBackend is closed")
        started = time.perf_counter()
        frame = as_bgr(frame_bgr)
        height, width = frame.shape[:2]
        if (width, height) != self._frame_size:
            self._frame_size = (width, height)
            self._forget_tracking()
        guard = self._max_faces >= 2

        # YuNet rows found in this very frame, if any search ran (reused below so
        # that no frame runs the detector twice at the same resolution).
        detections: list[np.ndarray] | None = None
        # Shoulder guard: a fresh face count on this frame, if one is due.
        counted: list[np.ndarray] | None = None
        if guard and not 0.0 <= timestamp - self._guard_ts < GUARD_PERIOD_S:
            counted = detections = self._guard_count(frame, timestamp)

        # Landmarks, the crop they came from and the pass's face-presence logit.
        found: tuple[np.ndarray, Roi, float] | None = None
        roi = self._roi
        tracking = roi is not None
        if roi is not None:
            if detections is None and not guard and self._primary_check_due(timestamp):
                detections = self._detect(frame, DETECT_WIDTH)
                self._schedule_primary_check(timestamp, PRIMARY_CHECK_S)
            candidate = None
            if detections is not None:
                candidate = self._primary_candidate(roi, detections, width, height)
            if candidate is not None:
                # A clearly stronger face elsewhere (the user came back): it takes
                # over if the network can read it; otherwise (a profile, say)
                # the tracked face stays rather than being lost to it.
                result = self._infer(frame, candidate)
                if result is not None:
                    self._pose.reset()
                    found = (result[0], candidate, result[1])
            if found is None:
                result = self._infer(frame, roi)
                if result is not None:
                    found = (result[0], roi, result[1])
        if found is None:
            # Lost (or never had) the face: find one. With the guard on, the
            # search doubles as a fresh face count.
            tracking = False
            self._pose.reset()
            if detections is None:
                if guard:
                    detections = counted = self._guard_count(frame, timestamp)
                else:
                    detections = self._detect(frame, DETECT_WIDTH)
            if not guard:
                # The search chose the primary face; look again soon, the user
                # may be on the way into the picture.
                self._schedule_primary_check(timestamp, PRIMARY_RECHECK_S)
            found = self._acquire(frame, detections, width, height, timestamp)

        if found is None:
            self._note_settled(True)
            return self._without_landmarks(detections or [], timestamp, width, height, started)
        points, settled = self._settle(frame, *found, detections, reseed=tracking)
        self._note_settled(settled)
        roi = roi_from_landmarks(points)
        self._roi = roi if roi.valid(width, height) else None
        return self._with_landmarks(points, counted, timestamp, width, height, started)

    def _note_settled(self, settled: bool) -> None:
        self._unsettled_frames = 0 if settled else self._unsettled_frames + 1
        self._settled = settled or self._unsettled_frames >= MAX_UNSETTLED_FRAMES

    def _primary_check_due(self, timestamp: float) -> bool:
        # Written like the guard's check so that a clock jumping backwards (a
        # new timestamp source) counts as due rather than as "checked recently".
        return not 0.0 <= timestamp - self._primary_ts < self._primary_period

    def _schedule_primary_check(self, timestamp: float, period: float) -> None:
        self._primary_ts = timestamp
        self._primary_period = period

    def _acquire(
        self,
        frame: np.ndarray,
        detections: list[np.ndarray],
        width: int,
        height: int,
        timestamp: float,
    ) -> tuple[np.ndarray, Roi, float] | None:
        """Seed the landmark network from YuNet detections; the first that works wins.

        Candidates go best first (largest, then most central: see
        :func:`primary_score`). When the network rejects the best one or
        accepts it only with a low presence logit (:data:`ROLL_RETRY_BELOW_LOGIT`),
        it is also tried rotated by :data:`ROLL_RETRY_DEG` either way (YuNet
        misjudges strong head roll) and the most face-like crop wins. When it
        still fails, comparable runners-up are tried (see
        :data:`SEED_CANDIDATE_MIN_RATIO`). A frame where the first seed works
        well costs one inference, as before. Returns the landmarks, their crop
        and its presence logit.
        """
        if not detections:
            return None
        ranked = sorted(detections, key=lambda d: _det_score(d, width, height), reverse=True)
        best_score = _det_score(ranked[0], width, height)
        for index, face in enumerate(ranked[:MAX_SEED_CANDIDATES]):
            if index and _det_score(face, width, height) < SEED_CANDIDATE_MIN_RATIO * best_score:
                break
            seed = roi_from_detection(face)
            if not seed.valid(width, height):
                continue
            result = self._infer(frame, seed)
            doubtful = result is None or result[1] < ROLL_RETRY_BELOW_LOGIT
            if (
                index == 0
                and doubtful
                and not 0.0 <= timestamp - self._roll_retry_ts < ROLL_RETRY_PERIOD_S
            ):
                rolled = self._better_rolled(
                    frame, seed, -math.inf if result is None else result[1]
                )
                if rolled is not None:
                    return rolled
                # Turning the crop did not help: the face is not merely rolled.
                self._roll_retry_ts = timestamp
            if result is not None:
                return result[0], seed, result[1]
        return None

    def _better_rolled(
        self, frame: np.ndarray, seed: Roi, logit_to_beat: float
    ) -> tuple[np.ndarray, Roi, float] | None:
        """``seed`` rolled by :data:`ROLL_RETRY_DEG` either way, if that looks more like a face.

        Both rotations are run; the one with the higher presence logit is
        returned (landmarks, crop, logit) when it beats ``logit_to_beat`` (the
        unrotated seed's). Passing the presence threshold is not enough: a crop
        50-100° off can pass with a confident, wrongly rolled fit, but measured
        on rolled photos it scores a logit of about 2-14 against 18-28 for the
        right orientation.
        """
        best: tuple[np.ndarray, Roi, float] | None = None
        for turn in (ROLL_RETRY_DEG, -ROLL_RETRY_DEG):
            crop = Roi(seed.cx, seed.cy, seed.side, seed.angle + turn)
            result = self._infer(frame, crop)
            if result is not None and result[1] > logit_to_beat:
                best, logit_to_beat = (result[0], crop, result[1]), result[1]
        return best

    def _settle(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        used: Roi,
        logit: float,
        detections: list[np.ndarray] | None,
        *,
        reseed: bool,
    ) -> tuple[np.ndarray, bool]:
        """Re-run the network until its crop sits on the face; returns landmarks and settledness.

        ``points`` came from the crop ``used`` with the presence logit ``logit``.
        One pass on a crop chosen from the previous frame follows a face that
        moved by up to about 10 % of the crop almost completely, but after a
        larger jump (a posture shift between two frames at the idle frame
        rates) the landmarks stay biased towards the old position, upward jumps
        especially, while the network may still report a face. So:

        1. If the crop the landmarks ask for is close to ``used`` (within
           :data:`ROI_SETTLE_SHIFT`, or within :data:`ROI_CATCH_UP_SHIFT` with a
           confident presence logit, see :func:`crop_caught_up`), they are final
           for this frame. This is the steady state, also for a steadily moving
           head: no extra cost.
        2. Otherwise one more pass on that crop, which suffices for moderate
           jumps (about 10 ms).
        3. Still not caught up, or that crop was rejected: a large jump, which
           the network follows only a fraction of the way per pass. With
           ``reseed`` (tracking), YuNet finds the face independently of the old
           crop (the detection nearest the wanted crop, about 7 ms unless this
           frame already ran one) and the network runs on its crop and once more
           on the crop its landmarks ask for. Without (the face was just found
           by YuNet), one more plain pass.

        Each step keeps the previous landmarks when the network rejects the new
        crop, and a rejected crop is not tried twice. The result counts as
        settled when the last pass's crop matched within the stricter
        :data:`ROI_SETTLE_SHIFT`.
        """
        height, width = frame.shape[:2]
        wanted = roi_from_landmarks(points)
        if crop_caught_up(used, wanted, logit) or not wanted.valid(width, height):
            return points, crop_settled(used, wanted)
        second = self._infer(frame, wanted)
        if second is not None:
            points, used = second[0], wanted
            wanted = roi_from_landmarks(points)
            if crop_caught_up(used, wanted, second[1]) or not wanted.valid(width, height):
                return points, crop_settled(used, wanted)
        elif not reseed:
            return points, False
        # Whether `wanted` still deserves a pass (the network just rejected it if not).
        last_pass = second is not None
        if reseed:
            if detections is None:
                detections = self._detect(frame, DETECT_WIDTH)
            nearby = [face for face in detections if _in_crop(face, wanted)]
            if nearby:
                face = min(
                    nearby,
                    key=lambda d: math.hypot(
                        _det_centre(d)[0] - wanted.cx, _det_centre(d)[1] - wanted.cy
                    ),
                )
                seed = roi_from_detection(face)
                third = self._infer(frame, seed) if seed.valid(width, height) else None
                if third is not None:
                    points, used = third[0], seed
                    wanted = roi_from_landmarks(points)
                    last_pass = True
        if last_pass and wanted.valid(width, height):
            last = self._infer(frame, wanted)
            if last is not None:
                points, used = last[0], wanted
        return points, crop_settled(used, roi_from_landmarks(points))

    def _infer(self, frame: np.ndarray, roi: Roi) -> tuple[np.ndarray, float] | None:
        """Landmarks (frame pixels) and face-presence logit for a crop, or ``None``.

        ``None`` when the presence probability is below
        :data:`PRESENCE_THRESHOLD`. The logit is returned rather than the
        probability because it keeps ranking crops where the probability has
        long saturated at 1.0 (see ``_acquire``).
        """
        assert self._net is not None
        m = roi.transform()
        crop = cv2.warpAffine(
            frame,
            m,
            (INPUT_SIZE, INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
        self._net.setInput(blob)
        raw, flag = self._net.forward(_OUTPUT_NAMES)
        logit = float(flag.ravel()[0])
        if not _sigmoid(logit) >= PRESENCE_THRESHOLD:  # also rejects NaN
            return None
        crop_points = raw.reshape(-1, 3)[:_NUM_LANDMARKS, :2].astype(np.float64)
        inverse = cv2.invertAffineTransform(m)
        points = crop_points @ inverse[:, :2].T + inverse[:, 2]
        if not np.all(np.isfinite(points)):
            return None
        return points, logit

    def _detect(self, frame: np.ndarray, input_width: int) -> list[np.ndarray]:
        """YuNet detections (rows in frame pixels) with score >= DETECTION_SCORE."""
        height, width = frame.shape[:2]
        detector: cv2.FaceDetectorYN | None = self._detector
        if input_width == GUARD_DETECT_WIDTH:
            if self._guard_detector is None:
                self._guard_detector = create_yunet(self._detector_model, GUARD_DETECT_WIDTH)
            detector = self._guard_detector
        assert detector is not None
        scale = min(1.0, input_width / width)
        image = frame
        if scale < 1.0:
            size = (input_width, max(1, round(height * scale)))
            image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        size = (image.shape[1], image.shape[0])
        if self._detector_sizes.get(input_width) != size:
            detector.setInputSize(size)
            self._detector_sizes[input_width] = size
        _, rows = detector.detect(image)
        if rows is None:
            return []
        faces = []
        for row in rows:
            if float(row[14]) >= DETECTION_SCORE:
                face = row.astype(np.float64)
                face[:14] /= scale
                faces.append(face)
        return faces

    def _guard_count(self, frame: np.ndarray, timestamp: float) -> list[np.ndarray]:
        """Detect every face at the guard's resolution and note when."""
        faces = self._detect(frame, GUARD_DETECT_WIDTH)
        self._guard_ts = timestamp
        return faces

    @staticmethod
    def _primary_candidate(
        roi: Roi, detections: list[np.ndarray], width: int, height: int
    ) -> Roi | None:
        """Crop of a clearly stronger face elsewhere than the tracked ``roi``, if any.

        Faces are compared by :func:`primary_score` and must beat the tracked
        face by :data:`PRIMARY_SWITCH_RATIO`. Like is compared with like: the
        tracked face's size is YuNet's box for it when this frame's detections
        include it, and otherwise its landmark box converted to YuNet's scale
        (:data:`YUNET_TO_MESH_SIDE`). Comparing YuNet's box (about 13 % larger)
        with the landmark box would let a face only 1.15x the user's take over.
        """
        others = [face for face in detections if not _in_crop(face, roi)]
        if not others:
            return None
        tracked = [face for face in detections if _in_crop(face, roi)]
        if tracked:
            tracked_score = max(_det_score(face, width, height) for face in tracked)
        else:
            mesh_side = roi.side / ROI_SCALE
            tracked_score = primary_score(
                mesh_side * YUNET_TO_MESH_SIDE, roi.cx, roi.cy, width, height
            )
        best = max(others, key=lambda d: _det_score(d, width, height))
        if _det_score(best, width, height) <= PRIMARY_SWITCH_RATIO * tracked_score:
            return None
        seed = roi_from_detection(best)
        return seed if seed.valid(width, height) else None

    def _other_faces(
        self, counted: list[np.ndarray] | None, primary_px: np.ndarray, width: int, height: int
    ) -> list[Box]:
        """Faces besides the primary one, as normalised boxes.

        A fresh guard count (``counted``, from this very frame) is split into the
        primary face and the others, and the others are kept until the next
        count. They are matched against the primary face only here, where both
        come from the same frame: matching stale detections against a later
        face position would count the user twice after a quick head movement.
        """
        if self._max_faces < 2:
            return []
        if counted is None:
            return self._guard_others
        x0, y0 = primary_px.min(axis=0)
        x1, y1 = primary_px.max(axis=0)
        others = []
        for face in counted:
            cx = float(face[0] + face[2] / 2.0)
            cy = float(face[1] + face[3] / 2.0)
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                others.append(_det_box(face, width, height))
        self._guard_others = others
        return others

    def _without_landmarks(
        self,
        detections: list[np.ndarray],
        timestamp: float,
        width: int,
        height: int,
        started: float,
    ) -> Observation:
        self._roi = None
        self._landmarks = None
        # Best first, in the order the faces were tried as the primary face.
        faces = sorted(detections, key=lambda d: _det_score(d, width, height), reverse=True)
        boxes = [_det_box(f, width, height) for f in faces]
        self._last = _Drawing(timestamp, None, boxes[1:], None, None, None)
        # YuNet sees a face the landmark network rejects (a profile, heavy
        # blur): someone is there, but nothing usable for gaze.
        return Observation(
            timestamp=timestamp,
            face_count=min(len(faces), self._max_faces),
            face_box=boxes[0] if boxes else None,
            inference_ms=(time.perf_counter() - started) * 1e3,
            frame_size=(width, height),
        )

    def _with_landmarks(
        self,
        points: np.ndarray,
        counted: list[np.ndarray] | None,
        timestamp: float,
        width: int,
        height: int,
        started: float,
    ) -> Observation:
        mesh = points[:_MESH_POINTS]
        x0, y0 = mesh.min(axis=0) / (width, height)
        x1, y1 = mesh.max(axis=0) / (width, height)
        face_box = (
            max(float(x0), 0.0),
            max(float(y0), 0.0),
            max(min(float(x1), 1.0) - max(float(x0), 0.0), 0.0),
            max(min(float(y1), 1.0) - max(float(y0), 0.0), 0.0),
        )
        others = self._other_faces(counted, mesh, width, height)
        eyes = eye_metrics(points)
        pose = self._pose.estimate(points, width, height)

        features: np.ndarray | None = None
        quality = 0.0
        blink = False
        if eyes is not None and pose is not None:
            vector = np.array(
                [
                    pose.yaw,
                    pose.pitch,
                    pose.roll,
                    pose.tx,
                    pose.ty,
                    pose.tz,
                    eyes.iris_h,
                    eyes.iris_v,
                ],
                dtype=np.float64,
            )
            if np.all(np.isfinite(vector)):
                features = vector
                touches_border = (
                    x0 <= BORDER_MARGIN
                    or y0 <= BORDER_MARGIN
                    or x1 >= 1.0 - BORDER_MARGIN
                    or y1 >= 1.0 - BORDER_MARGIN
                )
                quality = 0.5 if touches_border else 1.0
                blink = self._blink.update(eyes.openness)

        normalised = points / (width, height)
        self._landmarks = normalised
        self._last = _Drawing(
            timestamp,
            normalised,
            others,
            None if self._pose.rvec is None else self._pose.rvec.copy(),
            None if self._pose.tvec is None else self._pose.tvec.copy(),
            self._pose.camera.copy(),
        )
        return Observation(
            timestamp=timestamp,
            face_count=min(1 + len(others), self._max_faces),
            features=features,
            quality=quality,
            blink=blink,
            head_yaw=pose.yaw if pose is not None else None,
            head_pitch=pose.pitch if pose is not None else None,
            face_box=face_box,
            inference_ms=(time.perf_counter() - started) * 1e3,
            frame_size=(width, height),
        )

    # ------------------------------------------------------------- preview
    def annotate(self, frame_bgr: np.ndarray, observation: Observation) -> np.ndarray:
        out = to_bgr(frame_bgr)
        thickness = line_thickness(out)
        last = self._last
        if last is not None and last.timestamp == observation.timestamp:
            if last.points is not None:
                self._draw_face(out, last, thickness)
            for box in last.others:
                draw_box(out, box, COLOR_OTHER, thickness)
        if observation.face_box is not None:
            draw_box(out, observation.face_box, COLOR_BOX, thickness)
        draw_label(out, observation_label(observation))
        return out

    def _draw_face(self, out: np.ndarray, last: _Drawing, thickness: int) -> None:
        assert last.points is not None
        h, w = out.shape[:2]
        pixels = last.points * (w, h)
        pts = np.round(pixels).astype(np.int32)
        for contour in (_EYE_A_CONTOUR, _EYE_B_CONTOUR):
            cv2.polylines(out, [pts[list(contour)]], True, COLOR_EYE, thickness, cv2.LINE_AA)
        for iris in (_IRIS_A, _IRIS_B):
            centre = pixels[iris[0]]
            ring = pixels[list(iris[1:])]
            radius = float(np.mean(np.hypot(*(ring - centre).T)))
            cx, cy = (round(float(v)) for v in centre)
            cv2.circle(out, (cx, cy), max(1, round(radius)), COLOR_IRIS, thickness, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), max(1, thickness), COLOR_IRIS, -1, cv2.LINE_AA)
        if last.rvec is not None and last.tvec is not None and last.camera is not None:
            # Where the nose points: from the tip, 5 cm out of the face.
            tip = self._nose_tip
            ends, _ = cv2.projectPoints(
                np.array([tip, tip + _POSE_ARROW_CM]), last.rvec, last.tvec, last.camera, None
            )
            ax, ay, bx, by = (float(v) for v in ends.ravel()[:4])
            if all(math.isfinite(v) for v in (ax, ay, bx, by)):
                cv2.line(
                    out,
                    (round(ax), round(ay)),
                    (round(bx), round(by)),
                    COLOR_POSE,
                    thickness + 1,
                    cv2.LINE_AA,
                )


def _clamp_faces(n: int) -> int:
    return max(1, min(int(n), MAX_FACES_LIMIT))
