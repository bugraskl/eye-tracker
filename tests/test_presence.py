"""Tests for walk-away detection (engine/presence.py)."""

from __future__ import annotations

import math

import pytest

from eye_tracker.config import PresenceSettings
from eye_tracker.engine.presence import (
    INPUT_EVIDENCE_S,
    PresenceConfig,
    PresenceEvent,
    PresenceMonitor,
    PresenceState,
)

IDLE = 999.0  # seconds since input: nobody touched anything


def kinds(events: list[PresenceEvent]) -> list[str]:
    return [e.kind for e in events]


def monitor(**overrides: object) -> PresenceMonitor:
    cfg = PresenceConfig(away_timeout_s=45.0, warning_s=10.0)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return PresenceMonitor(cfg, now=0.0)


def feed(
    pm: PresenceMonitor,
    start: float,
    stop: float,
    *,
    face: bool | None,
    idle: float | None = IDLE,
    dt: float = 0.5,
) -> list[tuple[float, PresenceEvent]]:
    """Update every ``dt`` seconds in ``[start, stop]`` and collect (time, event)."""
    out: list[tuple[float, PresenceEvent]] = []
    n = round((stop - start) / dt)
    for i in range(n + 1):
        t = start + i * dt
        out.extend((t, e) for e in pm.update(t, face, idle))
    return out


def test_face_keeps_user_present() -> None:
    pm = monitor()
    assert feed(pm, 0, 120, face=True) == []
    assert pm.state is PresenceState.PRESENT


def test_full_cycle_warn_away_return() -> None:
    pm = monitor()
    events = feed(pm, 0.5, 60, face=False)
    assert [(t, e.kind) for t, e in events] == [(35.0, "warn"), (45.0, "away")]
    assert events[0][1].remaining_s == pytest.approx(10.0)
    assert pm.state is PresenceState.AWAY
    assert pm.remaining(60.0) == 0.0
    assert kinds(pm.update(61.0, True, IDLE)) == ["return"]
    assert pm.state is PresenceState.PRESENT


def test_face_during_countdown_cancels() -> None:
    pm = monitor()
    feed(pm, 0.5, 40, face=False)
    assert pm.state is PresenceState.WARNING
    assert pm.remaining(40.0) == pytest.approx(5.0)
    assert kinds(pm.update(40.5, True, IDLE)) == ["cancel"]
    assert pm.state is PresenceState.PRESENT
    # The absence timer restarts from the moment the face was seen.
    assert feed(pm, 41, 75, face=False) == []
    assert kinds(pm.update(75.5, False, IDLE)) == ["warn"]


def test_recent_input_counts_as_presence() -> None:
    pm = monitor()
    assert feed(pm, 0.5, 120, face=False, idle=0.2) == []
    assert pm.state is PresenceState.PRESENT


def test_input_cancels_countdown() -> None:
    pm = monitor()
    feed(pm, 0.5, 36, face=False)
    assert pm.state is PresenceState.WARNING
    assert kinds(pm.update(36.5, False, INPUT_EVIDENCE_S - 0.1)) == ["cancel"]


def test_old_input_is_not_presence() -> None:
    pm = monitor()
    events = feed(pm, 0.5, 50, face=False, idle=INPUT_EVIDENCE_S + 0.1)
    assert [e.kind for _, e in events] == ["warn", "away"]


def test_input_ignored_when_not_required() -> None:
    pm = monitor(require_input_idle=False)
    events = feed(pm, 0.5, 50, face=False, idle=0.0)
    assert [e.kind for _, e in events] == ["warn", "away"]


def test_unknown_input_idle_relies_on_face_only() -> None:
    pm = monitor()
    events = feed(pm, 0.5, 50, face=False, idle=None)
    assert [e.kind for _, e in events] == ["warn", "away"]


def test_unknown_face_freezes_timers() -> None:
    pm = monitor()
    feed(pm, 0.5, 20, face=False)
    # Camera off (privacy mode) for a long time: never counts as absence.
    assert feed(pm, 20.5, 500, face=None) == []
    assert pm.state is PresenceState.PRESENT
    # Absence counts again from the end of the unknown period.
    events = feed(pm, 500.5, 560, face=False)
    assert [(t, e.kind) for t, e in events] == [(535.0, "warn"), (545.0, "away")]


def test_unknown_face_during_countdown_cancels_it() -> None:
    pm = monitor()
    feed(pm, 0.5, 38, face=False)
    assert pm.state is PresenceState.WARNING
    assert kinds(pm.update(38.5, None, IDLE)) == ["cancel"]
    assert pm.state is PresenceState.PRESENT


def test_unknown_face_while_away_stays_away() -> None:
    pm = monitor()
    feed(pm, 0.5, 50, face=False)
    assert feed(pm, 50.5, 100, face=None) == []
    assert pm.state is PresenceState.AWAY


def test_no_countdown_goes_straight_to_away() -> None:
    pm = monitor(warning_s=0.0)
    events = feed(pm, 0.5, 60, face=False)
    assert [(t, e.kind) for t, e in events] == [(45.0, "away")]


def test_sparse_updates_never_skip_the_countdown() -> None:
    pm = monitor()
    pm.update(1.0, True, IDLE)
    # Next update only after the whole timeout (e.g. the process was suspended).
    events = pm.update(100.0, False, IDLE)
    assert kinds(events) == ["warn"]
    assert events[0].remaining_s == pytest.approx(10.0)
    assert pm.update(105.0, False, IDLE) == []
    assert kinds(pm.update(110.0, False, IDLE)) == ["away"]


def test_warning_longer_than_timeout_still_waits_briefly() -> None:
    pm = monitor(away_timeout_s=5.0, warning_s=30.0)
    # A single missed detection must not flash the countdown.
    assert pm.update(0.5, False, IDLE) == []
    events = pm.update(1.0, False, IDLE)
    assert kinds(events) == ["warn"]
    assert events[0].remaining_s == pytest.approx(4.0)
    assert kinds(pm.update(5.0, False, IDLE)) == ["away"]


def test_disabled_is_always_present() -> None:
    pm = monitor(enabled=False)
    assert feed(pm, 0.5, 200, face=False) == []
    assert pm.state is PresenceState.PRESENT
    assert math.isinf(pm.remaining(200.0))


def test_disabling_during_countdown_cancels_it() -> None:
    pm = monitor()
    feed(pm, 0.5, 36, face=False)
    assert pm.state is PresenceState.WARNING
    pm.set_config(PresenceConfig(enabled=False))
    assert kinds(pm.update(36.5, False, IDLE)) == ["cancel"]
    assert pm.update(37.0, False, IDLE) == []


def test_disabling_while_away_reports_return() -> None:
    pm = monitor()
    feed(pm, 0.5, 50, face=False)
    pm.set_config(PresenceConfig(enabled=False))
    assert kinds(pm.update(50.5, False, IDLE)) == ["return"]
    assert pm.state is PresenceState.PRESENT


def test_reenabling_does_not_count_disabled_time_as_absence() -> None:
    pm = monitor(enabled=False)
    pm.update(1.0, False, IDLE)
    pm.set_config(PresenceConfig(enabled=True))
    assert pm.remaining(500.0) == pytest.approx(45.0)
    assert pm.update(500.0, False, IDLE) == []
    events = feed(pm, 500.5, 546, face=False)
    assert [(t, e.kind) for t, e in events] == [(535.0, "warn"), (545.0, "away")]


def test_longer_timeout_during_countdown_cancels_it() -> None:
    pm = monitor()
    feed(pm, 0.5, 36, face=False)
    pm.set_config(PresenceConfig(away_timeout_s=300.0, warning_s=10.0))
    assert kinds(pm.update(36.5, False, IDLE)) == ["cancel"]
    assert pm.state is PresenceState.PRESENT


def test_reset_returns_to_present_silently() -> None:
    pm = monitor()
    feed(pm, 0.5, 50, face=False)
    assert pm.state is PresenceState.AWAY
    pm.reset(60.0)
    assert pm.state is PresenceState.PRESENT
    assert pm.last_seen == 60.0
    assert pm.update(60.5, False, IDLE) == []
    assert pm.remaining(60.5) == pytest.approx(44.5)


def test_config_from_settings() -> None:
    s = PresenceSettings(enabled=False, away_timeout_s=90, warning_s=5, require_input_idle=False)
    assert PresenceConfig.from_settings(s) == PresenceConfig(
        enabled=False, away_timeout_s=90.0, warning_s=5.0, require_input_idle=False
    )
    assert PresenceConfig.from_settings(PresenceSettings()) == PresenceConfig()
