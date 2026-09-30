"""Adaptive camera frame rate - the main lever for low CPU use.

Face analysis dominates the app's CPU cost, so the rate at which frames are
analysed is chosen from what is going on: fast while a switch is being
considered or during calibration, slow while the user reads or types on one
monitor, slower still when nobody is there. Each profile maps these *modes* to
frames per second.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..types import TrackingState

log = logging.getLogger(__name__)

#: Modes a :class:`RatePolicy` distinguishes (the keys of every profile).
MODES = ("calibrating", "away", "active", "noface", "typing", "idle")

#: Mode reported for states in which the worker does not use the camera.
CAMERA_OFF_MODE = "off"

#: Frames per second for each profile and mode.
PROFILES: dict[str, dict[str, float]] = {
    "eco": {
        "active": 8.0,
        "idle": 2.0,
        "typing": 1.0,
        "noface": 1.0,
        "away": 0.5,
        "calibrating": 15.0,
    },
    "balanced": {
        "active": 12.0,
        "idle": 4.0,
        "typing": 2.0,
        "noface": 2.0,
        "away": 1.0,
        "calibrating": 24.0,
    },
    "responsive": {
        "active": 20.0,
        "idle": 8.0,
        "typing": 4.0,
        "noface": 3.0,
        "away": 1.0,
        "calibrating": 30.0,
    },
}

DEFAULT_PROFILE = "balanced"

#: States in which the camera is released; the rate is irrelevant there.
CAMERA_OFF_STATES = frozenset(
    {TrackingState.PAUSED, TrackingState.PRIVACY, TrackingState.LOCKED, TrackingState.YIELDED}
)

#: Value returned by :meth:`RatePolicy.interval` / :meth:`RatePolicy.fps` for
#: camera-off states (the worker is inactive anyway).
CAMERA_OFF_VALUE = 1.0


@dataclass
class RateContext:
    """What the controller knows right now that matters for the frame rate."""

    state: TrackingState
    #: The switch decider is considering a switch (:attr:`Decision.pending`).
    pending_switch: bool = False
    #: The gaze point is moving quickly (a switch may be coming).
    gaze_moving: bool = False
    #: The user typed recently.
    typing: bool = False
    #: A face was seen in the latest observation.
    face_present: bool = True
    #: The walk-away countdown is running (sample fast so it cancels quickly).
    presence_warning: bool = False
    #: The camera preview window is open.
    preview: bool = False


class RatePolicy:
    """Chooses the interval between analysed frames from a :class:`RateContext`."""

    def __init__(self, profile: str = DEFAULT_PROFILE) -> None:
        self._profile = DEFAULT_PROFILE
        self._rates = PROFILES[DEFAULT_PROFILE]
        self.set_profile(profile)

    @property
    def profile(self) -> str:
        return self._profile

    def set_profile(self, profile: str) -> None:
        """Switch profile; unknown names fall back to ``"balanced"`` with a warning."""
        if profile not in PROFILES:
            log.warning("Unknown performance profile %r; using %r", profile, DEFAULT_PROFILE)
            profile = DEFAULT_PROFILE
        self._profile = profile
        self._rates = PROFILES[profile]

    def mode(self, ctx: RateContext) -> str:
        """Name of the mode selected for ``ctx`` (one of :data:`MODES` or ``"off"``)."""
        state = ctx.state
        if state in CAMERA_OFF_STATES:
            return CAMERA_OFF_MODE
        if state is TrackingState.CALIBRATING or ctx.preview:
            return "calibrating"
        if state is TrackingState.AWAY:
            return "away"
        if ctx.presence_warning:
            # Fast so the countdown is cancelled as soon as the user is back.
            return "active"
        if not ctx.face_present:
            return "noface"
        if ctx.pending_switch or ctx.gaze_moving:
            return "active"
        if ctx.typing:
            return "typing"
        return "idle"

    def fps(self, ctx: RateContext) -> float:
        """Target analysed frames per second for ``ctx``."""
        mode = self.mode(ctx)
        if mode == CAMERA_OFF_MODE:
            return CAMERA_OFF_VALUE
        return self._rates[mode]

    def interval(self, ctx: RateContext) -> float:
        """Seconds between analysed frames for ``ctx``."""
        mode = self.mode(ctx)
        if mode == CAMERA_OFF_MODE:
            return CAMERA_OFF_VALUE
        return 1.0 / self._rates[mode]
