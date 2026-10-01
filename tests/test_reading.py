"""Tests for the shared reading rule (engine/reading.py) with non-monitor keys.

The monitor decider's use of it is covered by tests/test_decision.py.
"""

from __future__ import annotations

import math

from eye_tracker.engine import decision
from eye_tracker.engine.reading import (
    READING_EVIDENCE_S,
    READING_FORGET_S,
    READING_RETURN_S,
    ReadingTracker,
)

GRACES = {"typing_grace_s": 3.0, "reading_grace_s": 8.0}


def look(
    tracker: ReadingTracker[str], key: str, start: float, stop: float, last_key: float
) -> None:
    t = start
    while t <= stop + 1e-9:
        tracker.observe(key, t)
        tracker.update(t, last_key, **GRACES)
        t = round(t + 0.5, 6)


def test_typing_through_a_look_at_another_pane_makes_it_the_reading_pane() -> None:
    tracker: ReadingTracker[str] = ReadingTracker()
    look(tracker, "%2", 0.0, 2.0, last_key=READING_EVIDENCE_S + 0.5)
    assert tracker.reading == "%2"
    tracker.end_look()
    tracker.update(3.0, 2.5, **GRACES)
    assert tracker.reading == "%2"


def test_a_single_sample_or_a_short_overlap_is_no_evidence() -> None:
    tracker: ReadingTracker[str] = ReadingTracker()
    tracker.observe("%2", 0.0)
    tracker.update(0.0, 1.5, **GRACES)
    assert tracker.reading is None
    tracker.end_look()
    look(tracker, "%3", 10.0, 11.0, last_key=10.2)  # typing stopped as the look began
    tracker.end_look()
    tracker.update(11.0 + READING_RETURN_S + 1.0, 10.2, **GRACES)
    assert tracker.reading is None


def test_reading_ends_with_a_typing_pause_or_when_forgotten() -> None:
    tracker: ReadingTracker[int] = ReadingTracker()
    for t in (0.0, 0.5, 1.0, 1.5):
        tracker.observe(7, t)
        tracker.update(t, 1.5, **GRACES)
    assert tracker.reading == 7
    tracker.end_look()
    tracker.update(1.5 + 8.0, 1.5, **GRACES)  # typing paused for the reading grace
    assert tracker.reading is None

    for t in (20.0, 20.5, 21.0, 21.5):
        tracker.observe(7, t)
        tracker.update(t, t, **GRACES)
    tracker.end_look()
    late = 21.5 + READING_FORGET_S + 1.0
    tracker.update(late, late, **GRACES)  # typing on, but never looked at again
    assert tracker.reading is None


def test_rule_is_off_when_the_reading_grace_is_not_longer() -> None:
    tracker: ReadingTracker[str] = ReadingTracker()
    for t in (0.0, 0.5, 1.0, 1.5):
        tracker.observe("a", t)
        tracker.update(t, 1.5, typing_grace_s=3.0, reading_grace_s=3.0)
    assert tracker.reading is None


def test_inactive_context_recognises_nothing_and_clear_forgets() -> None:
    tracker: ReadingTracker[str] = ReadingTracker()
    for t in (0.0, 0.5, 1.0, 1.5):
        tracker.observe("a", t)
        tracker.update(t, 1.5, active=False, **GRACES)
    assert tracker.reading is None
    tracker.update(1.5, 1.5, **GRACES)
    assert tracker.reading == "a"
    tracker.clear()
    assert tracker.reading is None
    tracker.update(2.0, -math.inf, **GRACES)
    assert tracker.reading is None


def test_monitor_decider_re_exports_the_constants() -> None:
    assert decision.READING_FORGET_S == READING_FORGET_S
    assert decision.READING_EVIDENCE_S == READING_EVIDENCE_S
