"""Shoulder-surfer guard: react when a second face keeps looking at the screen.

A second face must be visible *continuously* for ``delay_s`` before the guard
triggers (people walking past are ignored), and must be gone continuously for
``clear_s`` before it clears (a face that briefly drops out of detection does
not flicker the privacy curtain).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import PrivacySettings

log = logging.getLogger(__name__)


@dataclass
class GuardConfig:
    """Tuning of :class:`ShoulderGuard` (seconds)."""

    enabled: bool = False
    #: A second face must be visible this long before the guard triggers.
    delay_s: float = 2.0
    #: Once triggered, fewer than two faces must be seen this long to clear.
    clear_s: float = 1.5

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
        self._single_since: float | None = None

    @property
    def active(self) -> bool:
        """``True`` between a ``"trigger"`` and the following ``"clear"``."""
        return self._active

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

    def update(self, now: float, face_count: int | None) -> str | None:
        """Feed one observation; return ``"trigger"``, ``"clear"`` or ``None``.

        ``face_count`` is ``None`` when the camera cannot tell (camera off): the
        guard keeps its state, but an interrupted observation is not
        "continuous", so any run in progress restarts.
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
            return None

        if face_count >= 2:
            self._single_since = None
            if self._active:
                return None
            if self._multi_since is None:
                self._multi_since = now
            if now - self._multi_since >= cfg.delay_s:
                self._active = True
                self._multi_since = None
                log.info("Shoulder guard triggered (%d faces)", face_count)
                return "trigger"
            return None

        self._multi_since = None
        if not self._active:
            return None
        if self._single_since is None:
            self._single_since = now
        if now - self._single_since >= cfg.clear_s:
            self._active = False
            self._single_since = None
            log.info("Shoulder guard cleared")
            return "clear"
        return None
