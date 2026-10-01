"""Shared value types used by the vision, gaze, engine, platform and UI layers.

Coordinate convention
---------------------
Every screen coordinate in this code base is a *Qt global coordinate*:

* Windows and Linux/X11: the app disables Qt's high-DPI scaling and is
  per-monitor DPI aware (see ``PlatformServices.prepare_process``), so Qt
  coordinates are native physical pixels, identical to Win32 / X11 coordinates.
* macOS: Qt coordinates are Cocoa points with a top-left origin, identical to
  the coordinates used by Quartz events and the Accessibility API.

Because both sides agree, positions can be passed between Qt and the platform
layer without conversion.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle in global screen coordinates (``right``/``bottom`` exclusive)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def diagonal(self) -> float:
        return math.hypot(self.w, self.h)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.right and self.y <= py < self.bottom

    def distance_outside(self, px: float, py: float) -> float:
        """Euclidean distance from the point to the rectangle (0 when inside)."""
        dx = max(self.x - px, 0.0, px - self.right)
        dy = max(self.y - py, 0.0, py - self.bottom)
        return math.hypot(dx, dy)

    def clamp(self, px: float, py: float) -> tuple[int, int]:
        """Nearest integer point that lies inside the rectangle."""
        cx = min(max(round(px), self.x), self.right - 1)
        cy = min(max(round(py), self.y), self.bottom - 1)
        return cx, cy

    def normalize(self, px: float, py: float) -> tuple[float, float]:
        return ((px - self.x) / self.w, (py - self.y) / self.h)

    def denormalize(self, nx: float, ny: float) -> tuple[float, float]:
        return (self.x + nx * self.w, self.y + ny * self.h)

    def union(self, other: Rect) -> Rect:
        x0 = min(self.x, other.x)
        y0 = min(self.y, other.y)
        x1 = max(self.right, other.right)
        y1 = max(self.bottom, other.bottom)
        return Rect(x0, y0, x1 - x0, y1 - y0)

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    @classmethod
    def from_list(cls, values: Sequence[int]) -> Rect:
        x, y, w, h = (int(v) for v in values)
        return cls(x, y, w, h)


@dataclass(frozen=True, slots=True)
class Monitor:
    """A physical display as seen by Qt."""

    index: int
    name: str
    rect: Rect
    primary: bool = False
    scale: float = 1.0


def virtual_bounds(monitors: Iterable[Monitor]) -> Rect:
    """Bounding rectangle of the whole virtual desktop."""
    rects = [m.rect for m in monitors]
    if not rects:
        raise ValueError("no monitors")
    out = rects[0]
    for r in rects[1:]:
        out = out.union(r)
    return out


def layout_signature(monitors: Iterable[Monitor]) -> str:
    """Stable fingerprint of a monitor arrangement (order independent).

    A calibration is only valid for the layout it was recorded on.
    """
    parts = sorted(tuple(m.rect.to_list()) for m in monitors)
    digest = hashlib.sha1(repr(parts).encode("ascii")).hexdigest()
    return digest[:16]


def monitor_at(monitors: Iterable[Monitor], px: float, py: float) -> Monitor | None:
    for m in monitors:
        if m.rect.contains(px, py):
            return m
    return None


def nearest_monitor(monitors: Iterable[Monitor], px: float, py: float) -> tuple[Monitor, float]:
    """Monitor closest to the point and the distance to it (0 when inside)."""
    best: tuple[Monitor, float] | None = None
    for m in monitors:
        d = m.rect.distance_outside(px, py)
        if best is None or d < best[1]:
            best = (m, d)
    if best is None:
        raise ValueError("no monitors")
    return best


@dataclass(slots=True)
class Observation:
    """Result of analysing one camera frame.

    ``features`` is a backend-specific 1-D float vector (see
    ``VisionBackend.feature_names``) describing head pose and eye state of the
    primary (largest) face. It is ``None`` when no usable face was found.
    """

    timestamp: float
    face_count: int
    features: np.ndarray | None = None
    quality: float = 0.0
    blink: bool = False
    head_yaw: float | None = None
    head_pitch: float | None = None
    face_box: tuple[float, float, float, float] | None = None
    skipped: bool = False
    inference_ms: float = 0.0
    frame_size: tuple[int, int] = (0, 0)
    #: The frame is too dark or uniform to judge (lens covered, shutter closed,
    #: unlit room). ``face_count`` is then meaningless: presence and the
    #: shoulder guard must treat the frame as "cannot tell", not "nobody here".
    blind: bool = False

    @property
    def face_present(self) -> bool:
        return self.face_count > 0 and not self.blind

    @property
    def usable(self) -> bool:
        """True when the observation can drive gaze estimation."""
        return self.features is not None and not self.blink and self.quality >= 0.3


@dataclass(frozen=True, slots=True)
class GazePoint:
    """Estimated gaze position on the virtual desktop."""

    x: float
    y: float
    timestamp: float


@dataclass(frozen=True, eq=False)
class WindowRef:
    """Opaque handle to a top-level window, owned by the platform layer.

    ``handle`` is an HWND (Windows), an X11 window id (Linux) or an
    ``(pid, AXUIElement)`` pair (macOS). Compare with
    ``PlatformServices.same_window`` rather than ``==``.
    """

    handle: Any
    pid: int | None = None
    rect: Rect | None = None


@dataclass(frozen=True, slots=True)
class AppIdentity:
    """Which application a window belongs to (``PlatformServices.window_app``).

    ``process`` is the lower-cased executable name without its extension
    (``"windowsterminal"``, ``"wezterm-gui"``). ``app_id`` is the window class name
    on Windows (``"CASCADIA_HOSTING_WINDOW_CLASS"``), the ``WM_CLASS`` class on X11
    and the bundle identifier on macOS; ``""`` when unknown. Titles are never part
    of it.
    """

    process: str
    app_id: str = ""


class TrackingState(enum.Enum):
    """High-level state shown in the tray and used by the rate scheduler."""

    STARTING = "starting"
    NEEDS_CALIBRATION = "needs_calibration"
    CALIBRATING = "calibrating"
    TRACKING = "tracking"
    PAUSED = "paused"
    PRIVACY = "privacy"
    AWAY = "away"
    LOCKED = "locked"
    YIELDED = "yielded"
    CAMERA_ERROR = "camera_error"

    @property
    def label(self) -> str:
        return _STATE_LABELS[self]

    @property
    def camera_active(self) -> bool:
        """Whether the camera should be open in this state."""
        return self in {
            TrackingState.STARTING,
            TrackingState.NEEDS_CALIBRATION,
            TrackingState.CALIBRATING,
            TrackingState.TRACKING,
            TrackingState.AWAY,
            TrackingState.CAMERA_ERROR,
        }


_STATE_LABELS = {
    TrackingState.STARTING: "Starting…",
    TrackingState.NEEDS_CALIBRATION: "Needs calibration",
    TrackingState.CALIBRATING: "Calibrating",
    TrackingState.TRACKING: "Tracking",
    TrackingState.PAUSED: "Paused",
    TrackingState.PRIVACY: "Privacy mode (camera off)",
    TrackingState.AWAY: "Away",
    TrackingState.LOCKED: "Screen locked (camera off)",
    TrackingState.YIELDED: "Camera in use by another app",
    TrackingState.CAMERA_ERROR: "Camera unavailable",
}


@dataclass(slots=True)
class WorkerStats:
    """Rolling statistics published by the vision worker."""

    fps: float = 0.0
    target_fps: float = 0.0
    inference_ms: float = 0.0
    skip_ratio: float = 0.0
    camera_open: bool = False
    last_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
