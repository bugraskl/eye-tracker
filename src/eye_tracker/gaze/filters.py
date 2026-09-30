"""Smoothing for the gaze point.

Gaze estimates jitter by tens of pixels from frame to frame. A fixed low-pass
filter would either leave that jitter visible or make monitor switches feel
sluggish. The One Euro filter (Casiez, Roussel & Vogel, CHI 2012) adapts its cut-off
to the speed of the signal: heavy smoothing while the gaze rests, almost none
while it jumps to another monitor. It is time-aware, which matters here because
the frame rate changes all the time (adaptive rate, motion gate).
"""

from __future__ import annotations

import math

# PointFilter tuning; both parameters are interpolated by `smoothing` (0..1).
# beta is in pixel units: at 0.004 a 2000 px/s gaze jump raises the cut-off by
# 8 Hz. Frame-to-frame jitter of a webcam gaze estimate also "moves" at roughly
# 1000 px/s, so a large beta would keep the cut-off high while the gaze rests and
# heavy smoothing would barely smooth. Heavy smoothing therefore uses a small
# beta; a monitor jump (~20000 px/s at 12 fps) still opens the filter within a frame.
_MIN_CUTOFF_RAW = 8.0
_MIN_CUTOFF_SMOOTH = 0.3
_BETA_RAW = 0.004
_BETA_SMOOTH = 0.0005
_D_CUTOFF = 1.0


class OneEuroFilter:
    """One-dimensional One Euro filter.

    ``min_cutoff`` (Hz) sets the smoothing of a slow signal, ``beta`` how quickly
    the cut-off rises with speed and ``d_cutoff`` (Hz) the smoothing of the speed
    estimate itself. Timestamps are seconds; a sample whose timestamp does not
    advance is ignored and the previous output is returned.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("min_cutoff and d_cutoff must be > 0 and beta >= 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @property
    def value(self) -> float | None:
        """Last filtered value (``None`` before the first sample or after a reset)."""
        return self._x

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = None

    def __call__(self, x: float, t: float) -> float:
        x = float(x)
        if not math.isfinite(x):
            # A NaN would poison the state forever; hold the last value instead.
            return x if self._x is None else self._x
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        dt = t - self._t
        if dt <= 0.0:
            return self._x
        dx = (x - self._x) / dt
        edx = self._dx + _smoothing_factor(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        filtered = self._x + _smoothing_factor(cutoff, dt) * (x - self._x)
        self._x, self._dx, self._t = filtered, edx, t
        return filtered


class PointFilter:
    """Two-dimensional One Euro filter for gaze points in pixels.

    ``smoothing`` 0..1 maps linearly to ``min_cutoff`` from 8 Hz (light) to 0.3 Hz
    (heavy) and to ``beta`` from 0.004 to 0.0005. 0 returns the raw input
    unchanged. The filters keep running at smoothing 0 so that turning smoothing
    back on continues from the current gaze position.
    """

    def __init__(self, smoothing: float) -> None:
        self._fx = OneEuroFilter(_MIN_CUTOFF_RAW, _BETA_RAW, _D_CUTOFF)
        self._fy = OneEuroFilter(_MIN_CUTOFF_RAW, _BETA_RAW, _D_CUTOFF)
        self._smoothing = 0.0
        self.set_smoothing(smoothing)

    @property
    def smoothing(self) -> float:
        return self._smoothing

    def set_smoothing(self, smoothing: float) -> None:
        """Change the strength (clamped to 0..1) without resetting the state."""
        s = float(smoothing)
        s = 0.0 if not math.isfinite(s) else min(max(s, 0.0), 1.0)
        self._smoothing = s
        min_cutoff = _MIN_CUTOFF_RAW + (_MIN_CUTOFF_SMOOTH - _MIN_CUTOFF_RAW) * s
        beta = _BETA_RAW + (_BETA_SMOOTH - _BETA_RAW) * s
        for f in (self._fx, self._fy):
            f.min_cutoff = min_cutoff
            f.beta = beta

    def update(self, x: float, y: float, t: float) -> tuple[float, float]:
        fx, fy = self._fx(x, t), self._fy(y, t)
        if self._smoothing <= 0.0:
            return float(x), float(y)
        return fx, fy

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()


def _smoothing_factor(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)
