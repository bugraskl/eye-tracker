"""Motion gate: skip face analysis while the camera picture is unchanged.

Someone reading or typing sits almost perfectly still for long stretches. The
gate compares a tiny greyscale thumbnail of each new frame with the thumbnail of
the last *analysed* frame and only lets the frame through when enough has
changed, which removes most inference work at a cost of ~0.2 ms per frame.

A whole-frame thumbnail cannot see a pure eye movement: at 32x24 pixels the
irises are a fraction of a pixel. When the caller passes the face box of the
analysed frame, the gate therefore also watches a small, higher-resolution
thumbnail of the eye band, so a glance at another monitor is noticed even when
the head does not move.
"""

from __future__ import annotations

import cv2
import numpy as np

NormBox = tuple[float, float, float, float]

# Eye band inside a face box (fractions of the box). Generous enough for the
# MediaPipe landmark box (eyes at ~33 % of its height) and the YuNet detection
# box (eyes at ~41 %).
_BAND_X0, _BAND_X1 = 0.05, 0.95
_BAND_Y0, _BAND_Y1 = 0.15, 0.60
_ROI_SIZE = (32, 16)
_MIN_ROI_PX = 8


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


class MotionGate:
    """Decides whether a frame differs enough from the last analysed one.

    Args:
        threshold: Mean absolute grey-level difference (0-255 scale) that counts
            as motion. Camera noise after downscaling is well below 1.
        size: Whole-frame thumbnail size ``(width, height)``.
        max_skip_s: Always analyse at least this often, so that presence, blink
            and face-count information never goes stale.

    Not thread-safe; used by the vision worker thread only.
    """

    def __init__(
        self,
        threshold: float = 2.0,
        size: tuple[int, int] = (32, 24),
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
        """Motion measured by the latest :meth:`should_process` call (diagnostics)."""
        return self._last_motion

    def should_process(self, frame_bgr: np.ndarray, now: float) -> bool:
        """True if ``frame_bgr`` should be analysed.

        True when nothing has been analysed yet, when ``max_skip_s`` has passed
        since the last analysed frame, or when the whole frame or the eye band
        changed by at least ``threshold``.
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
                motion = max(motion, _mean_abs_diff(roi, self._ref_roi))
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
