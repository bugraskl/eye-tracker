"""Shoulder-surfer guard: react when a second face keeps looking at the screen.

A second face must be visible for ``delay_s`` before the guard triggers, so
people walking past are ignored. A shoulder surfer is far away and partly
hidden behind the user, so the detector misses their face in some frames:
dropouts shorter than ``gap_s`` (or than a couple of frame intervals at slow
frame rates) do not restart the delay. Once triggered, fewer than two faces
must be seen continuously for ``clear_s`` before the guard clears, so a face
that briefly drops out of detection does not flicker the privacy curtain.

When only one face is left, the guard checks that it is the user's. The user
sits closest to the camera, so while two faces are visible the primary
(largest) face box is the user's. If the one remaining face does not match it
- the user walked away and the onlooker stayed - the guard stays active and
reports :attr:`ShoulderGuard.owner_missing`, so that the curtain is not lifted
for the onlooker and the caller can treat the user as absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import PrivacySettings

log = logging.getLogger(__name__)

#: Normalised face box ``(x, y, w, h)`` as in ``Observation.face_box``.
FaceBox = tuple[float, float, float, float]

#: At slow frame rates a single missed frame already spans ``2 x interval``;
#: a run survives gaps of up to this many frame intervals.
GAP_FRAMES = 2.5

#: ... but never gaps longer than this, however slowly frames arrive.
MAX_GAP_S = 3.0

#: The remaining face is the user's when its area is at least this fraction of
#: the user's last known face area (an onlooker behind the user is much smaller)...
OWNER_MIN_AREA_RATIO = 0.5

#: ...or when it overlaps the user's last known face box at least this much (IoU).
OWNER_MIN_IOU = 0.3


@dataclass
class GuardConfig:
    """Tuning of :class:`ShoulderGuard` (seconds)."""

    enabled: bool = False
    #: A second face must be visible this long before the guard triggers.
    delay_s: float = 2.0
    #: Once triggered, fewer than two faces must be seen this long to clear.
    clear_s: float = 1.5
    #: Missed detections of the second face shorter than this do not restart ``delay_s``.
    gap_s: float = 1.0

    @classmethod
    def from_settings(cls, s: PrivacySettings) -> GuardConfig:
        """Build from user settings (``s.guard_action`` is handled by the controller)."""
        return cls(enabled=bool(s.shoulder_guard), delay_s=float(s.guard_delay_s))


class ShoulderGuard:
    """Debounced detector for "more than one face in front of the camera"."""

    def __init__(self, config: GuardConfig | None = None) -> None:
        self._config = config if config is not None else GuardConfig()
        self._active = False
        self._multi_since: float | None = None
        self._multi_last = 0.0
        self._single_since: float | None = None
        self._last_update: float | None = None
        self._owner_box: FaceBox | None = None
        self._owner_missing = False

    @property
    def active(self) -> bool:
        """``True`` between a ``"trigger"`` and the following ``"clear"``."""
        return self._active

    @property
    def owner_missing(self) -> bool:
        """The guard is active and the only face in view is not the user's.

        The user has most likely walked away while the onlooker stayed; the
        caller should not count that face as the user being present.
        """
        return self._owner_missing

    @property
    def config(self) -> GuardConfig:
        return self._config

    def set_config(self, config: GuardConfig) -> None:
        """Change the tuning. Disabling an active guard clears it on the next update."""
        self._config = config

    def reset(self) -> None:
        """Forget everything (inactive, no pending run). Emits nothing."""
        self._active = False
        self._multi_since = None
        self._single_since = None
        self._last_update = None
        self._owner_box = None
        self._owner_missing = False

    def update(
        self, now: float, face_count: int | None, face_box: FaceBox | None = None
    ) -> str | None:
        """Feed one observation; return ``"trigger"``, ``"clear"`` or ``None``.

        ``face_count`` is ``None`` when the camera cannot tell (camera off): the
        guard keeps its state, but an interrupted observation is not
        "continuous", so any run in progress restarts. ``face_box`` is the
        primary (largest) face of the observation (``Observation.face_box``);
        without it the guard cannot tell whose face remains and clears as soon
        as fewer than two faces are seen for ``clear_s``.
        """
        cfg = self._config
        if not cfg.enabled:
            was_active = self._active
            self.reset()
            if was_active:
                log.info("Shoulder guard disabled while active; clearing")
                return "clear"
            return None

        if face_count is None:
            self._multi_since = None
            self._single_since = None
            self._last_update = None
            self._owner_missing = False
            return None

        interval = 0.0 if self._last_update is None else max(0.0, now - self._last_update)
        self._last_update = now
        tolerance = max(cfg.gap_s, min(GAP_FRAMES * interval, MAX_GAP_S))

        if face_count >= 2:
            self._single_since = None
            self._owner_missing = False
            if face_box is not None:
                self._owner_box = face_box  # the largest face is the user's
            if self._active:
                return None
            if self._multi_since is None or now - self._multi_last > tolerance:
                self._multi_since = now
            self._multi_last = now
            if now - self._multi_since >= cfg.delay_s:
                self._active = True
                self._multi_since = None
                log.info("Shoulder guard triggered (%d faces)", face_count)
                return "trigger"
            return None

        if self._multi_since is not None and now - self._multi_last > tolerance:
            self._multi_since = None
        if not self._active:
            return None

        if face_count == 1 and face_box is not None and self._owner_box is not None:
            if not self._is_owner(face_box):
                # The user left and the onlooker stayed: keep the curtain up.
                if not self._owner_missing:
                    log.info("Shoulder guard: only the onlooker's face is left")
                self._owner_missing = True
                self._single_since = None
                return None
            self._owner_box = face_box  # follow the user's movements
        self._owner_missing = False
        if self._single_since is None:
            self._single_since = now
        if now - self._single_since >= cfg.clear_s:
            self._active = False
            self._single_since = None
            log.info("Shoulder guard cleared")
            return "clear"
        return None

    # ------------------------------------------------------------- internals
    def _is_owner(self, box: FaceBox) -> bool:
        owner = self._owner_box
        if owner is None:
            return True
        owner_area = _area(owner)
        if owner_area <= 0.0:
            return True
        return _area(box) >= OWNER_MIN_AREA_RATIO * owner_area or _iou(box, owner) >= OWNER_MIN_IOU


def _area(box: FaceBox) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def _iou(a: FaceBox, b: FaceBox) -> float:
    """Intersection over union of two ``(x, y, w, h)`` boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0
