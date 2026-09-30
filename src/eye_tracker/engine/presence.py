"""Walk-away detection with a cancellable countdown.

The monitor is fed a verdict about the camera image (face or no face) and the
time since the last keyboard/mouse input. When neither shows the user for long
enough it moves through ``PRESENT -> WARNING -> AWAY`` and emits events that the
controller turns into UI (countdown toast) and actions (lock, displays off).
Any sign of the user cancels the countdown or signals their return.

Keyboard/mouse input counts as presence (by default) because the camera can
miss a face - a user leaning back, bad light, a turned head - and locking the
screen under someone who is typing is the worst possible failure.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import PresenceSettings

log = logging.getLogger(__name__)

#: Input more recent than this counts as proof that someone is at the computer.
INPUT_EVIDENCE_S = 1.5

#: The countdown never starts before the face has been missing this long, even
#: when ``warning_s >= away_timeout_s``; otherwise a single frame without a
#: detected face would flash the countdown.
MIN_ABSENCE_BEFORE_WARNING_S = 1.0


class PresenceState(enum.Enum):
    """Where the user is, as far as the camera and input can tell."""

    PRESENT = "present"
    WARNING = "warning"
    AWAY = "away"


@dataclass
class PresenceEvent:
    """A state transition reported by :meth:`PresenceMonitor.update`.

    ``kind`` is one of:

    * ``"warn"`` - countdown started; ``remaining_s`` seconds until *away*.
    * ``"cancel"`` - countdown cancelled (the user is back, or presence
      detection was paused/disabled).
    * ``"away"`` - the user is gone; perform the configured action.
    * ``"return"`` - the user came back after being *away*.
    """

    kind: str
    remaining_s: float = 0.0


@dataclass
class PresenceConfig:
    """Tuning of :class:`PresenceMonitor` (seconds)."""

    enabled: bool = True
    #: Total time without presence evidence before the user counts as away.
    away_timeout_s: float = 45.0
    #: Length of the cancellable countdown at the end of ``away_timeout_s``
    #: (0 = no countdown).
    warning_s: float = 10.0
    #: Recent keyboard/mouse input counts as presence.
    require_input_idle: bool = True

    @classmethod
    def from_settings(cls, s: PresenceSettings) -> PresenceConfig:
        """Build from user settings (``s.action`` is handled by the controller)."""
        return cls(
            enabled=bool(s.enabled),
            away_timeout_s=float(s.away_timeout_s),
            warning_s=float(s.warning_s),
            require_input_idle=bool(s.require_input_idle),
        )


class PresenceMonitor:
    """State machine ``PRESENT -> WARNING -> AWAY`` driven by face and input evidence."""

    def __init__(self, config: PresenceConfig, now: float) -> None:
        self._config = config
        self._state = PresenceState.PRESENT
        self._last_seen = now
        # Set when timing must restart on the next update (re-enabled after a
        # period in which nothing was observed).
        self._restart = False

    @property
    def state(self) -> PresenceState:
        return self._state

    @property
    def config(self) -> PresenceConfig:
        return self._config

    @property
    def last_seen(self) -> float:
        """Time of the most recent presence evidence (or end of the last unknown period)."""
        return self._last_seen

    def set_config(self, config: PresenceConfig) -> None:
        """Change the tuning; the state is re-evaluated on the next update.

        Disabling while *WARNING*/*AWAY* makes the next update emit ``cancel`` /
        ``return`` so that UI driven by earlier events is torn down.
        """
        if not self._config.enabled and config.enabled:
            # Nothing was observed while disabled; that time is not absence.
            self._restart = True
        self._config = config

    def reset(self, now: float) -> None:
        """Back to *PRESENT* without emitting events (e.g. after the session was unlocked)."""
        self._state = PresenceState.PRESENT
        self._last_seen = now
        self._restart = False

    def remaining(self, now: float) -> float:
        """Seconds until *AWAY* (meaningful while *WARNING*; 0 when away, inf when disabled)."""
        if not self._config.enabled:
            return math.inf
        if self._state is PresenceState.AWAY:
            return 0.0
        absent = 0.0 if self._restart else max(0.0, now - self._last_seen)
        return max(0.0, self._config.away_timeout_s - absent)

    def update(
        self, now: float, face_present: bool | None, seconds_since_input: float | None
    ) -> list[PresenceEvent]:
        """Advance the state machine and return the transitions that happened.

        ``face_present`` is ``None`` when the camera cannot tell (camera off,
        privacy mode, paused): timers are frozen so that time without a camera is
        never counted as absence. ``seconds_since_input`` is ``None`` when the OS
        cannot report input idle time.
        """
        cfg = self._config
        if not cfg.enabled or self._restart:
            self._restart = False
            self._last_seen = now
            return self._to_present("cancel", "return")

        if face_present is None:
            self._last_seen = now
            # A countdown cannot continue without a camera; take it down rather
            # than leave the toast frozen on screen. AWAY stays AWAY: the
            # controller resets presence when the session is unlocked.
            if self._state is PresenceState.WARNING:
                return self._to_present("cancel", None)
            return []

        evidence = face_present is True or (
            cfg.require_input_idle
            and seconds_since_input is not None
            and math.isfinite(seconds_since_input)
            and seconds_since_input < INPUT_EVIDENCE_S
        )
        if evidence:
            self._last_seen = now
            return self._to_present("cancel", "return")

        if self._state is PresenceState.AWAY:
            return []

        timeout = max(0.0, cfg.away_timeout_s)
        warning = max(0.0, cfg.warning_s)
        if warning > 0.0:
            threshold = max(timeout - warning, min(MIN_ABSENCE_BEFORE_WARNING_S, timeout))
        else:
            threshold = timeout
        absent = max(0.0, now - self._last_seen)

        if absent < threshold:
            if self._state is PresenceState.WARNING:
                # Only reachable after a config change lengthened the timeout.
                return self._to_present("cancel", None)
            return []

        if self._state is PresenceState.PRESENT and warning > 0.0:
            if absent >= timeout:
                # Updates were sparse (process suspended, very low frame rate) and
                # the whole countdown elapsed unseen. Never act without warning:
                # restart the countdown so that the user always gets ``warning_s``.
                self._last_seen = now - threshold
                absent = threshold
            self._state = PresenceState.WARNING
            remaining = max(0.0, timeout - absent)
            log.info("No presence for %.0f s; away in %.0f s", absent, remaining)
            return [PresenceEvent("warn", remaining)]

        if absent >= timeout:
            self._state = PresenceState.AWAY
            log.info("User away (no presence for %.0f s)", absent)
            return [PresenceEvent("away")]
        return []

    # ------------------------------------------------------------- internals
    def _to_present(self, from_warning: str | None, from_away: str | None) -> list[PresenceEvent]:
        previous = self._state
        self._state = PresenceState.PRESENT
        if previous is PresenceState.WARNING and from_warning:
            log.info("Presence countdown cancelled")
            return [PresenceEvent(from_warning)]
        if previous is PresenceState.AWAY and from_away:
            log.info("User returned")
            return [PresenceEvent(from_away)]
        return []
