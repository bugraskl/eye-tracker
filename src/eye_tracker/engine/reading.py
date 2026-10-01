"""Recognise what the user *reads* while typing somewhere else.

Copying from a document into an editor means looking at the source while the
keystrokes go elsewhere. A decider that moves the keyboard focus to where the
user looks must not take those looks for a wish to work there.
:class:`ReadingTracker` finds such a *reading target*: another target (a
monitor, a split pane) the user kept typing while, or right after, looking at.
Evidence is a look at the target (two or more consecutive updates with the gaze
clearly on it) followed by typing that continues at least
:data:`READING_EVIDENCE_S` after the look began and resumes no later than
:data:`READING_RETURN_S` after it ended. A single noisy gaze sample never counts,
and neither does the usual "last keystroke while the eyes already move on" of a
user who is about to work on the other target.

The reading target is forgotten when typing has paused for the reading grace
or when the gaze has not been on it for :data:`READING_FORGET_S`. The owner
forgets it (:meth:`ReadingTracker.clear`) when the keyboard focus moves, since
what the user read beside the old focus says nothing about the new one.

Targets are identified by any hashable key; the tracker is pure logic driven by
injected times, like the deciders that use it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

log = logging.getLogger(__name__)

#: Typing must go on at least this long after a look at another target began
#: for that target to become a reading target. Shorter overlaps are a user
#: finishing a word while the eyes already move to where they want to work.
READING_EVIDENCE_S = 1.0

#: Typing that resumes within this time after a look at another target ended
#: also makes it a reading target: the user glanced over, did not want to
#: switch and carried on typing where they were.
READING_RETURN_S = 2.0

#: A reading target the gaze has not rested on for this long is forgotten.
READING_FORGET_S = 60.0

#: Updates further apart than this end a look. Longer than a decider's dwell gap
#: on purpose: while the user types the camera runs at 1-2 fps, and reading must
#: be recognised at exactly that rate.
READING_MAX_GAP_S = 1.5

#: Absorbs float rounding in time differences.
_EPS = 1e-6

K = TypeVar("K", bound=Hashable)


@dataclass(slots=True)
class _Look(Generic[K]):
    """A run of consecutive updates with the gaze clearly on one other target."""

    key: K
    start: float
    last: float

    @property
    def repeated(self) -> bool:
        """Seen on more than one update (a single noisy sample is no look)."""
        return self.last > self.start


class ReadingTracker(Generic[K]):
    """The reading target beside the current keyboard focus (see the module docs).

    Per update the owner reports a clear look at another target with
    :meth:`observe` or the absence of one with :meth:`end_look`, then calls
    :meth:`update`. A long gap between updates ends the look (:meth:`end_look`);
    a clock going backwards or a focus change clears everything (:meth:`clear`).
    """

    def __init__(self) -> None:
        self._look: _Look[K] | None = None
        self._prev_look: _Look[K] | None = None
        self._reading: K | None = None
        self._reading_seen = -math.inf

    @property
    def reading(self) -> K | None:
        """The current reading target, if any."""
        return self._reading

    def clear(self) -> None:
        """Forget the looks and the reading target."""
        self._look = None
        self._prev_look = None
        self._reading = None
        self._reading_seen = -math.inf

    def observe(self, key: K, now: float) -> None:
        """The gaze is clearly on ``key`` (not the focused target) at ``now``."""
        look = self._look
        if look is None or look.key != key:
            self.end_look()
            self._look = _Look(key, now, now)
        else:
            look.last = now
        if key == self._reading:
            self._reading_seen = now

    def end_look(self) -> None:
        """The gaze is not clearly on another target (or the updates had a gap)."""
        look = self._look
        self._look = None
        # A single-sample "look" is kept out so that one noisy gaze sample cannot
        # replace the evidence of a real look just before it.
        if look is not None and look.repeated:
            self._prev_look = look

    def update(
        self,
        now: float,
        last_key: float,
        *,
        typing_grace_s: float,
        reading_grace_s: float,
        context: object = None,
        active: bool = True,
    ) -> None:
        """Recognise and expire the reading target.

        ``last_key`` is the time of the last keystroke (``-inf`` for never).
        A ``reading_grace_s`` not above ``typing_grace_s`` turns the rule off.
        ``active`` is False while there is no focused target to read beside, so
        no new reading target is recognised; ``context`` (the focused target)
        only appears in the debug log.
        """
        if reading_grace_s <= typing_grace_s:
            self._reading = None
            return
        typing_recent = now - last_key < reading_grace_s - _EPS
        if typing_recent and active:
            for look in (self._look, self._prev_look):
                if (
                    look is not None
                    and look.repeated
                    and look.start + READING_EVIDENCE_S <= last_key + _EPS
                    and last_key <= look.last + READING_RETURN_S + _EPS
                ):
                    if self._reading != look.key:
                        log.debug("Typing at %s while reading %s", context, look.key)
                        self._reading = look.key
                    self._reading_seen = max(self._reading_seen, look.last)
                    break
        if self._reading is not None and (
            not typing_recent or now - self._reading_seen > READING_FORGET_S
        ):
            self._reading = None
