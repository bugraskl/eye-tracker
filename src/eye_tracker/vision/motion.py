"""Motion gate: skip face analysis while the camera picture is unchanged.

Someone reading or typing sits almost perfectly still for long stretches. The
gate compares a tiny greyscale thumbnail of each new frame with the thumbnail of
the last *analysed* frame and only lets the frame through when enough has
changed, which removes most inference work at a cost of ~0.2 ms per frame.

A whole-frame thumbnail cannot see a pure eye movement: at 32x24 pixels the
irises are a fraction of a pixel. When the caller passes the face box of the
analysed frame, the gate therefore also watches a small, higher-resolution
thumbnail of the eye band, so a glance at another monitor is noticed even when
the head does not move. The eyes cover only a few percent of that band, so the
band is scored by its most-changed cells (the top decile), not by its mean,
which would dilute an iris movement below the camera noise.

The module also decides whether a frame is *blind*: so dark and featureless
(lens covered, privacy shutter closed, unlit room) that "no face" means "cannot
tell" rather than "nobody here".
"""

from __future__ import annotations

import cv2
import numpy as np

NormBox = tuple[float, float, float, float]

# Eye band inside a face box (fractions of the box). Generous enough for the
# face-mesh landmark box (eyes at ~33 % of its height) and the YuNet detection
# box (eyes at ~41 %).
_BAND_X0, _BAND_X1 = 0.05, 0.95
_BAND_Y0, _BAND_Y1 = 0.15, 0.60
_ROI_SIZE = (32, 16)
_MIN_ROI_PX = 8
#: Share of eye-band cells whose mean change scores the band.
BAND_TOP_FRACTION = 0.10
#: The eye band's score must exceed this multiple of the whole-frame threshold.
#: Its top-decile statistic reads a few grey levels even for a still picture
#: with camera noise, while an eye movement scores well above ten.
BAND_THRESHOLD_FACTOR = 3.0

#: Whole-frame thumbnail size used by the gate and the blindness test.
THUMB_SIZE = (32, 24)
#: A thumbnail darker than this mean grey level (0-255) ...
BLIND_MAX_MEAN = 32.0
#: ... and flatter than this standard deviation shows nothing recognisable.
BLIND_MAX_STD = 5.0


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    channels = frame.shape[2]
    if channels == 1:
        return frame[:, :, 0]
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _thumb(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def thumbnail(frame: np.ndarray, size: tuple[int, int] = THUMB_SIZE) -> np.ndarray:
    """Greyscale ``float32`` thumbnail of a BGR, BGRA or greyscale frame."""
    return _thumb(_gray(frame), size)


def is_blind(thumb: np.ndarray) -> bool:
    """True when a thumbnail is too dark and uniform to show a face.

    Covered lenses and closed privacy shutters give near-black, featureless
    pictures; any real scene with a person in it has far more contrast.
    """
    mean, std = cv2.meanStdDev(thumb)
    return float(mean[0, 0]) < BLIND_MAX_MEAN and float(std[0, 0]) < BLIND_MAX_STD


def _eye_band(face_box: NormBox) -> NormBox:
    x, y, w, h = face_box
    return (
        x + _BAND_X0 * w,
        y + _BAND_Y0 * h,
        (_BAND_X1 - _BAND_X0) * w,
        (_BAND_Y1 - _BAND_Y0) * h,
    )


def _roi_thumb(gray: np.ndarray, band: NormBox) -> np.ndarray | None:
    height, width = gray.shape[:2]
    bx, by, bw, bh = band
    x0 = max(0, int(bx * width))
    y0 = max(0, int(by * height))
    x1 = min(width, round((bx + bw) * width))
    y1 = min(height, round((by + bh) * height))
    if x1 - x0 < _MIN_ROI_PX or y1 - y0 < _MIN_ROI_PX // 2:
        return None
    return _thumb(gray[y0:y1, x0:x1], _ROI_SIZE)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.norm(a, b, cv2.NORM_L1)) / a.size


def _top_abs_diff(a: np.ndarray, b: np.ndarray, fraction: float = BAND_TOP_FRACTION) -> float:
    """Mean of the largest ``fraction`` of absolute differences.

    Robust to where in the band the change happens and insensitive to the size
    of the band, unlike the plain mean.
    """
    diff = cv2.absdiff(a, b).ravel()
    k = max(1, int(diff.size * fraction))
    return float(np.partition(diff, diff.size - k)[diff.size - k :].mean())


class MotionGate:
    """Decides whether a frame differs enough from the last analysed one.

    Args:
        threshold: Mean absolute grey-level difference (0-255 scale) of the
            whole-frame thumbnail that counts as motion. Camera noise after
            downscaling is well below 1. The eye band uses
            :data:`BAND_THRESHOLD_FACTOR` times this value for its top-decile
            score.
        size: Whole-frame thumbnail size ``(width, height)``.
        max_skip_s: Always analyse at least this often, so that presence, blink
            and face-count information never goes stale.

    Not thread-safe; used by the vision worker thread only.
    """

    def __init__(
        self,
        threshold: float = 2.0,
        size: tuple[int, int] = THUMB_SIZE,
        max_skip_s: float = 2.0,
    ) -> None:
        self.threshold = float(threshold)
        self.size = (int(size[0]), int(size[1]))
        self.max_skip_s = float(max_skip_s)
        self._ref: np.ndarray | None = None
        self._ref_roi: np.ndarray | None = None
        self._band: NormBox | None = None
        self._last_processed: float | None = None
        self._last_motion = 0.0
        # (frame, gray, thumbnail) from the latest should_process(), reused by
        # mark_processed() for the same frame object to avoid a second resize.
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def last_motion(self) -> float:
        """Motion measured by the latest :meth:`should_process` call (diagnostics).

        The larger of the whole-frame difference and the eye-band score divided
        by :data:`BAND_THRESHOLD_FACTOR`, so it compares directly with
        ``threshold``.
        """
        return self._last_motion

    @property
    def reference(self) -> np.ndarray | None:
        """Thumbnail of the last analysed frame (read-only view), if any."""
        if self._ref is None:
            return None
        view = self._ref.view()
        view.flags.writeable = False
        return view

    def should_process(self, frame_bgr: np.ndarray, now: float) -> bool:
        """True if ``frame_bgr`` should be analysed.

        True when nothing has been analysed yet, when ``max_skip_s`` has passed
        since the last analysed frame, or when the whole frame or the eye band
        changed enough (see ``threshold``).
        """
        self._cache = None
        if self._ref is None or self._last_processed is None:
            return True
        if now - self._last_processed >= self.max_skip_s:
            return True
        gray = _gray(frame_bgr)
        thumb = _thumb(gray, self.size)
        motion = _mean_abs_diff(thumb, self._ref)
        if self._band is not None and self._ref_roi is not None:
            roi = _roi_thumb(gray, self._band)
            if roi is not None:
                band = _top_abs_diff(roi, self._ref_roi)
                motion = max(motion, band / BAND_THRESHOLD_FACTOR)
        self._last_motion = motion
        if motion < self.threshold:
            return False
        self._cache = (frame_bgr, gray, thumb)
        return True

    def mark_processed(
        self, frame_bgr: np.ndarray, now: float, face_box: NormBox | None = None
    ) -> None:
        """Remember ``frame_bgr`` as the last analysed frame.

        ``face_box`` is the normalised ``(x, y, w, h)`` face box found in this
        frame (``Observation.face_box``); it enables eye-band motion detection.
        """
        cache, self._cache = self._cache, None
        if cache is not None and cache[0] is frame_bgr:
            gray, thumb = cache[1], cache[2]
        else:
            gray = _gray(frame_bgr)
            thumb = _thumb(gray, self.size)
        self._ref = thumb
        self._last_processed = now
        self._band = None
        self._ref_roi = None
        if face_box is not None:
            band = _eye_band(face_box)
            roi = _roi_thumb(gray, band)
            if roi is not None:
                self._band = band
                self._ref_roi = roi

    def reset(self) -> None:
        """Forget the reference frame; the next frame is always analysed."""
        self._ref = None
        self._ref_roi = None
        self._band = None
        self._last_processed = None
        self._last_motion = 0.0
        self._cache = None
