"""Shoulder-surfer guard: react when a second face keeps looking at the screen.

A second face must be visible for ``delay_s`` before the guard triggers, so
people walking past are ignored. A shoulder surfer is far away and partly
hidden behind the user, so the detector misses their face in some frames:
dropouts shorter than ``gap_s`` (or than a couple of frame intervals at slow
frame rates) do not restart the delay. Once triggered, fewer than two faces
must be seen continuously for ``clear_s`` before the guard clears, so a face
that briefly drops out of detection does not flicker the privacy curtain.

**The guard never decides whether the user is present.** It cannot recognise
faces; it can only compare face boxes, and any rule that turned "this face is
not the user's" into "nobody is at the computer" would sooner or later lock
out a user who merely moved in their chair while a colleague was in view.
Walk-away detection therefore counts every detected face as the user. The
guard only decides whether its privacy reaction (curtain, notification, lock)
is still needed.

Whose face is whose is judged by continuity, not by size:

* While the guard is idle and exactly one face is in view (and no second face
  was seen moments ago), that face is the user's (the *owner*); its box is
  remembered and followed.
* While two or more faces are in view only the primary (largest) face box is
  known. It updates the owner's box only when it overlaps it, so a colleague
  who leans in closer than the user never becomes the owner. Only when no
  owner is known yet (the guard was switched on with two faces in view) is
  the largest face taken as the owner.
* Once triggered, a lone face that does not match the owner (much smaller and
  not overlapping) means the user left and the onlooker stayed. The guard then
  stays active (:attr:`ShoulderGuard.owner_missing`), also through frames
  without any face, so an onlooker the detector misses for a moment does not
  get the curtain lifted in front of them.
* It clears once the owner is seen alone again, or once someone uses the
  keyboard or mouse while the owner is judged missing (``last_input``):
  whoever works at the computer with a single face in view is the user from
  then on. That also heals a wrong judgement - the guard never stays stuck
  on a user sitting at their own desk.
"""

from __future__ import annotations

import logging
import math
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
#: In frames with several faces, overlap alone decides whether the primary face
#: is still the user's: the area test would accept a colleague leaning in.
OWNER_MIN_IOU = 0.3

#: Keyboard or mouse input counts as "someone works at the computer" only when
#: it happened at least this long after the owner was judged missing. The user
#: who gets up and leaves often touches the mouse once more after their face has
#: left the picture; that must not make the onlooker the owner.
INPUT_ADOPT_DELAY_S = 2.0


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
        # The user left while the guard was active (a lone stranger's face was
        # seen) and has not been seen since; set at ``_owner_left_at``.
        self._owner_left = False
        self._owner_left_at = -math.inf

    @property
    def active(self) -> bool:
        """``True`` between a ``"trigger"`` and the following ``"clear"``."""
        return self._active

    @property
    def owner_missing(self) -> bool:
        """The guard is active and the user has not been seen since only another
        person's face was left in view.

        The user has most likely walked away while the onlooker stayed, so the
        guard stays active (the curtain stays up) until the user is back. This is
        a judgement from face boxes, not a recognition: it must never be used to
        decide that the user is absent (see the module docstring).
        """
        return self._owner_left

    @property
    def config(self) -> GuardConfig:
        return self._config

    def set_config(self, config: GuardConfig) -> None:
        """Change the tuning. Disabling an active guard clears it on the next update."""
        self._config = config

    def reset(self) -> None:
        """Forget everything (inactive, no pending run, no owner). Emits nothing."""
        self._active = False
        self._multi_since = None
        self._single_since = None
        self._last_update = None
        self._owner_box = None
        self._owner_left = False
        self._owner_left_at = -math.inf

    def update(
        self,
        now: float,
        face_count: int | None,
        face_box: FaceBox | None = None,
        last_input: float = -math.inf,
    ) -> str | None:
        """Feed one observation; return ``"trigger"``, ``"clear"`` or ``None``.

        ``face_count`` is ``None`` when the camera cannot tell (camera off): the
        guard keeps its state, but an interrupted observation is not
        "continuous", so any run in progress restarts. ``face_box`` is the
        primary (largest) face of the observation (``Observation.face_box``);
        without it the guard cannot tell whose face remains and clears as soon
        as fewer than two faces are seen for ``clear_s``. ``last_input`` is the
        time of the latest keyboard or mouse input (same clock as ``now``;
        ``-inf`` if unknown): input while the owner is judged missing means
        someone works at the computer, see the module docstring.
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
            # Nothing is known about this moment: runs restart, but whether the
            # owner left is kept (a camera hiccup does not bring them back).
            self._multi_since = None
            self._single_since = None
            self._last_update = None
            return None

        interval = 0.0 if self._last_update is None else max(0.0, now - self._last_update)
        self._last_update = now
        tolerance = max(cfg.gap_s, min(GAP_FRAMES * interval, MAX_GAP_S))

        if face_count >= 2:
            self._single_since = None
            self._follow_owner_among_faces(face_box)
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
            if face_count == 1 and face_box is not None and self._multi_since is None:
                # Alone at the desk: the user. Not while a second face was seen
                # moments ago - the lone face of such a dropout may be either one.
                self._owner_box = face_box
            return None

        someone_working = (
            self._owner_left and last_input >= self._owner_left_at + INPUT_ADOPT_DELAY_S
        )
        if face_count == 1 and face_box is not None and self._owner_box is not None:
            if someone_working or self._is_owner(face_box):
                if self._owner_left:
                    log.info(
                        "Shoulder guard: the user is back%s",
                        " (keyboard or mouse input)" if someone_working else "",
                    )
                self._owner_left = False
                self._owner_box = face_box  # follow the user's movements
            else:
                # The user left and the onlooker stayed: keep the curtain up.
                if not self._owner_left:
                    log.info("Shoulder guard: only the onlooker's face is left")
                    self._owner_left = True
                    self._owner_left_at = now
                self._single_since = None
                return None
        elif self._owner_left:
            # No face (or no box) after the user left: most likely the onlooker
            # was missed for a moment, so the curtain stays up - unless someone
            # is using the computer.
            if not someone_working:
                self._single_since = None
                return None
            log.info("Shoulder guard: keyboard or mouse input while no face is in view")
            self._owner_left = False
        if self._single_since is None:
            self._single_since = now
        if now - self._single_since >= cfg.clear_s:
            self._active = False
            self._single_since = None
            log.info("Shoulder guard cleared")
            return "clear"
        return None

    # ------------------------------------------------------------- internals
    def _follow_owner_among_faces(self, face_box: FaceBox | None) -> None:
        """Update the owner from the primary face of a frame with several faces.

        The primary face is the largest one, which is the user's only if nobody
        comes closer to the camera than they are. So it is taken as the user's
        only where it overlaps their last known box (continuity); the largest
        face is merely the fallback while no owner is known at all.
        """
        if face_box is None:
            return
        owner = self._owner_box
        if owner is None or _iou(face_box, owner) >= OWNER_MIN_IOU:
            self._owner_box = face_box
            if self._owner_left:
                log.info("Shoulder guard: the user is back next to the onlooker")
            self._owner_left = False

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
