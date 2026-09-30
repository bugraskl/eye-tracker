"""Drawing helpers for the camera preview, shared by the vision backends.

Frames drawn on here are only ever shown in the preview window; nothing is
written anywhere.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...types import Observation

Color = tuple[int, int, int]

# BGR colours for the preview overlay.
COLOR_EYE: Color = (140, 230, 120)
COLOR_IRIS: Color = (255, 220, 60)
COLOR_BOX: Color = (240, 170, 70)
COLOR_OTHER: Color = (60, 150, 255)
COLOR_POINT: Color = (140, 230, 120)
COLOR_POSE: Color = (80, 80, 255)
_COLOR_TEXT: Color = (255, 255, 255)


def to_bgr(frame: np.ndarray) -> np.ndarray:
    """A new 3-channel BGR copy of a BGR, BGRA or greyscale frame."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    return frame.copy()


def as_bgr(frame: np.ndarray) -> np.ndarray:
    """``frame`` as a contiguous 3-channel BGR image, copying only when needed."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        return np.ascontiguousarray(frame)
    return to_bgr(frame)


def draw_label(image: np.ndarray, lines: list[str]) -> None:
    """Draw a few lines of white text with a dark outline in the top-left corner."""
    w = image.shape[1]
    scale = max(0.4, min(1.2, w / 1000))
    thickness = max(1, round(scale * 2))
    y = int(24 * scale) + 4
    for text in lines:
        for color, width in (((0, 0, 0), thickness + 2), (_COLOR_TEXT, thickness)):
            cv2.putText(
                image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, width, cv2.LINE_AA
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


def draw_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    color: Color,
    thickness: int,
) -> None:
    """Draw a normalised ``(x, y, w, h)`` box onto ``image`` in place."""
    h, w = image.shape[:2]
    x, y, bw, bh = box
    p0 = (round(x * w), round(y * h))
    p1 = (round((x + bw) * w), round((y + bh) * h))
    cv2.rectangle(image, p0, p1, color, thickness, cv2.LINE_AA)


def line_thickness(image: np.ndarray) -> int:
    """Stroke width that stays visible at any preview resolution."""
    h, w = image.shape[:2]
    return max(1, round(max(h, w) / 640))
