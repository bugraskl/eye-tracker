"""Cartoon faces for the vision tests (not a test module itself).

Face photos are not committed to the repository, but a simple drawn face (hair,
skin-coloured oval, eyes with irises, brows, nose and mouth) is recognised by
both YuNet and the MediaPipe landmark network. That lets the vision backends be
tested end to end on every CI runner. Drawing is deterministic, so the frames
are identical everywhere.
"""

from __future__ import annotations

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
BACKGROUND = (200, 205, 210)


def draw_face(
    image: np.ndarray | None = None,
    *,
    cx: int = WIDTH // 2,
    cy: int = HEIGHT // 2,
    scale: float = 1.0,
    iris_dx: float = 0.0,
    blur: float = 1.2,
) -> np.ndarray:
    """Draw a frontal cartoon face centred at ``(cx, cy)``.

    At ``scale=1`` the face is 160 px wide, like a user ~60 cm from a 640x480
    webcam. ``iris_dx`` shifts both irises horizontally (pixels at scale 1).
    Draws onto a copy of ``image`` (default: a plain 640x480 background).
    """
    img = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8) if image is None else image.copy()
    height = img.shape[0]
    s = scale

    def p(dx: float, dy: float) -> tuple[int, int]:
        return round(cx + dx * s), round(cy + dy * s)

    def r(v: float) -> int:
        return max(1, round(v * s))

    cv2.ellipse(img, p(0, -40), (r(95), r(95)), 0, 180, 360, (40, 45, 60), -1, cv2.LINE_AA)
    cv2.ellipse(img, p(0, 0), (r(80), r(105)), 0, 0, 360, (150, 175, 215), -1, cv2.LINE_AA)
    x0, y0 = p(-35, 95)
    x1, _ = p(35, 0)
    cv2.rectangle(img, (x0, y0), (x1, height), (140, 165, 205), -1)
    for side in (-1, 1):
        ex, ey = 32 * side, -15
        cv2.ellipse(
            img, p(ex, ey - 20), (r(22), r(6)), 0, 180, 360, (40, 50, 70), r(4), cv2.LINE_AA
        )
        cv2.ellipse(img, p(ex, ey), (r(18), r(8)), 0, 0, 360, (235, 235, 235), -1, cv2.LINE_AA)
        cv2.circle(img, p(ex + iris_dx, ey), r(7), (60, 80, 110), -1, cv2.LINE_AA)
        cv2.circle(img, p(ex + iris_dx, ey), r(3), (10, 10, 10), -1, cv2.LINE_AA)
        cv2.ellipse(img, p(ex, ey), (r(18), r(8)), 0, 0, 360, (60, 70, 90), 1, cv2.LINE_AA)
    nose = np.array([p(0, -10), p(-12, 25), p(12, 25)], np.int32)
    cv2.polylines(img, [nose], False, (110, 130, 170), r(2), cv2.LINE_AA)
    cv2.circle(img, p(-7, 25), r(3), (90, 105, 150), -1)
    cv2.circle(img, p(7, 25), r(3), (90, 105, 150), -1)
    cv2.ellipse(img, p(0, 55), (r(28), r(10)), 0, 0, 180, (70, 70, 160), r(4), cv2.LINE_AA)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def two_faces(*, onlooker_scale: float = 0.22) -> np.ndarray:
    """The user (left of centre) and a small onlooker's face in the top right."""
    canvas = draw_face(cx=260)
    onlooker = draw_face(cx=560, cy=150, scale=onlooker_scale)
    mask = np.zeros(canvas.shape[:2], np.uint8)
    radius = round(200 * onlooker_scale)
    cv2.ellipse(mask, (560, 150), (radius, radius), 0, 0, 360, 255, -1)
    canvas[mask > 0] = onlooker[mask > 0]
    return canvas


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    """``image`` rotated counter-clockwise (as seen on screen) about its centre."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
