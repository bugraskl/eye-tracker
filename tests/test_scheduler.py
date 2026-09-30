"""Tests for the adaptive frame-rate policy (engine/scheduler.py)."""

from __future__ import annotations

import logging

import pytest

from eye_tracker.config import PerformanceSettings
from eye_tracker.engine.scheduler import (
    CAMERA_OFF_STATES,
    MODES,
    PROFILES,
    RateContext,
    RatePolicy,
)
from eye_tracker.types import TrackingState

T = TrackingState.TRACKING

EXPECTED = {
    #              active idle typing noface away calibrating
    "eco": (8.0, 2.0, 1.0, 1.0, 0.5, 15.0),
    "balanced": (12.0, 4.0, 2.0, 2.0, 1.0, 24.0),
    "responsive": (20.0, 8.0, 4.0, 3.0, 1.0, 30.0),
}

# One context per mode.
CONTEXTS = {
    "active": RateContext(T, pending_switch=True),
    "idle": RateContext(T),
    "typing": RateContext(T, typing=True),
    "noface": RateContext(T, face_present=False),
    "away": RateContext(TrackingState.AWAY, face_present=False),
    "calibrating": RateContext(TrackingState.CALIBRATING),
}


@pytest.mark.parametrize("profile", sorted(EXPECTED))
def test_profile_rates(profile: str) -> None:
    policy = RatePolicy(profile)
    names = ("active", "idle", "typing", "noface", "away", "calibrating")
    for name, fps in zip(names, EXPECTED[profile], strict=True):
        ctx = CONTEXTS[name]
        assert policy.mode(ctx) == name
        assert policy.fps(ctx) == fps
        assert policy.interval(ctx) == pytest.approx(1.0 / fps)


def test_every_profile_defines_every_mode() -> None:
    for rates in PROFILES.values():
        assert set(rates) == set(MODES)


def test_profiles_match_settings_choices() -> None:
    from dataclasses import fields

    meta = next(f for f in fields(PerformanceSettings) if f.name == "profile").metadata
    assert set(meta["choices"]) == set(PROFILES)


@pytest.mark.parametrize("state", sorted(CAMERA_OFF_STATES, key=lambda s: s.value))
def test_camera_off_states_return_one_second(state: TrackingState) -> None:
    policy = RatePolicy("responsive")
    ctx = RateContext(state, pending_switch=True, preview=True)
    assert policy.interval(ctx) == 1.0
    assert policy.fps(ctx) == 1.0
    assert policy.mode(ctx) == "off"


def test_preview_runs_at_calibration_rate() -> None:
    policy = RatePolicy()
    assert policy.fps(RateContext(T, preview=True, typing=True)) == 24.0


def test_calibration_beats_everything() -> None:
    policy = RatePolicy()
    ctx = RateContext(TrackingState.CALIBRATING, face_present=False, typing=True)
    assert policy.mode(ctx) == "calibrating"


def test_away_beats_presence_warning() -> None:
    policy = RatePolicy()
    ctx = RateContext(TrackingState.AWAY, presence_warning=True, face_present=False)
    assert policy.mode(ctx) == "away"


def test_presence_warning_samples_fast_without_a_face() -> None:
    policy = RatePolicy()
    ctx = RateContext(T, presence_warning=True, face_present=False)
    assert policy.mode(ctx) == "active"


def test_no_face_beats_pending_switch() -> None:
    policy = RatePolicy()
    assert policy.mode(RateContext(T, face_present=False, pending_switch=True)) == "noface"


def test_pending_switch_or_gaze_motion_beats_typing() -> None:
    policy = RatePolicy()
    assert policy.mode(RateContext(T, pending_switch=True, typing=True)) == "active"
    assert policy.mode(RateContext(T, gaze_moving=True, typing=True)) == "active"


@pytest.mark.parametrize(
    "state",
    [TrackingState.STARTING, TrackingState.NEEDS_CALIBRATION, TrackingState.CAMERA_ERROR],
)
def test_other_camera_states_use_normal_modes(state: TrackingState) -> None:
    policy = RatePolicy()
    assert policy.mode(RateContext(state)) == "idle"
    assert policy.mode(RateContext(state, face_present=False)) == "noface"


def test_unknown_profile_falls_back_to_balanced(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        policy = RatePolicy("turbo")
    assert policy.profile == "balanced"
    assert "turbo" in caplog.text


def test_set_profile_changes_rates() -> None:
    policy = RatePolicy("eco")
    idle = RateContext(T)
    assert policy.fps(idle) == 2.0
    policy.set_profile("responsive")
    assert policy.profile == "responsive"
    assert policy.fps(idle) == 8.0
