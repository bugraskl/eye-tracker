"""Scenario tests for the monitor switching decision (engine/decision.py)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from eye_tracker.config import SwitchingSettings
from eye_tracker.engine.decision import (
    PENDING_HORIZON_S,
    REASONS,
    Decision,
    SwitchConfig,
    SwitchDecider,
)
from eye_tracker.types import Monitor, Rect

NEVER = -math.inf


def mon(index: int, x: int, y: int, w: int = 1920, h: int = 1080) -> Monitor:
    return Monitor(index=index, name=f"M{index}", rect=Rect(x, y, w, h), primary=index == 0)


# Two 1080p monitors side by side: 0 = left, 1 = right.
DUAL = [mon(0, 0, 0), mon(1, 1920, 0)]
# Three in a row with the primary in the middle.
TRIPLE = [mon(0, -1920, 0), mon(1, 0, 0), mon(2, 1920, 0)]
# 1 on top of 0.
STACKED = [mon(0, 0, 0), mon(1, 0, -1080)]
# A 1440p external monitor with a 1080p laptop panel centred below it.
LAPTOP_BELOW = [mon(0, 0, 0, 2560, 1440), mon(1, 320, 1440, 1920, 1080)]

LEFT = (900.0, 500.0)
RIGHT = (2900.0, 500.0)


def decider(monitors: Sequence[Monitor] = DUAL, **overrides: float) -> SwitchDecider:
    return SwitchDecider(monitors, SwitchConfig(**overrides))


def step(
    d: SwitchDecider,
    now: float,
    gaze: tuple[float, float] | None,
    *,
    current: int | None,
    mouse: float = NEVER,
    key: float = NEVER,
    enabled: bool = True,
) -> Decision:
    return d.update(now, gaze, current, mouse, key, enabled=enabled)


def run(
    d: SwitchDecider,
    times: Sequence[float],
    gaze: tuple[float, float] | None,
    *,
    current: int | None,
    mouse: float = NEVER,
    key: float = NEVER,
) -> list[Decision]:
    return [step(d, t, gaze, current=current, mouse=mouse, key=key) for t in times]


def ticks(start: float, stop: float, dt: float = 0.1) -> list[float]:
    n = round((stop - start) / dt)
    return [round(start + i * dt, 6) for i in range(n + 1)]


def first_switch(decisions: Sequence[Decision], times: Sequence[float]) -> float | None:
    for t, dec in zip(times, decisions, strict=True):
        if dec.fired:
            return t
    return None


# --------------------------------------------------------------------- basics
def test_looking_at_current_monitor_does_nothing() -> None:
    d = decider()
    for dec in run(d, ticks(0, 2), LEFT, current=0):
        assert dec.reason == "same"
        assert dec.target is None
        assert dec.candidate == 0
        assert not dec.pending


def test_switch_fires_after_dwell_with_rising_progress() -> None:
    d = decider()
    times = [0.0, 0.1, 0.2, 0.3]
    out = run(d, times, RIGHT, current=0)
    assert [o.reason for o in out] == ["dwell", "dwell", "dwell", "switch"]
    assert [round(o.progress, 3) for o in out] == [0.0, 0.333, 0.667, 1.0]
    assert all(o.pending for o in out[:3])
    assert out[3].target == 1
    assert out[3].candidate == 1
    assert not out[3].pending


def test_dwell_zero_switches_immediately() -> None:
    d = decider(dwell_s=0.0)
    dec = step(d, 5.0, RIGHT, current=0)
    assert dec.reason == "switch"
    assert dec.target == 1
    assert dec.progress == 1.0


def test_all_reasons_are_documented() -> None:
    assert set(REASONS) == {
        "switch",
        "same",
        "dwell",
        "no_gaze",
        "off_screen",
        "mouse",
        "typing",
        "cooldown",
        "hysteresis",
        "disabled",
    }


# ------------------------------------------------------------ dwell resetting
def test_missing_gaze_resets_dwell() -> None:
    d = decider()
    run(d, [0.0, 0.1, 0.2], RIGHT, current=0)
    assert step(d, 0.25, None, current=0).reason == "no_gaze"
    # Dwell starts from scratch: 0.3 s after 0.3, not after 0.0.
    out = run(d, [0.3, 0.4, 0.5, 0.6], RIGHT, current=0)
    assert [o.reason for o in out] == ["dwell", "dwell", "dwell", "switch"]


def test_nan_gaze_is_treated_as_no_gaze() -> None:
    d = decider()
    dec = step(d, 0.0, (math.nan, 10.0), current=0)
    assert dec.reason == "no_gaze"
    assert dec.candidate is None


def test_gap_between_updates_restarts_dwell() -> None:
    d = decider(max_gap_s=0.75)
    step(d, 0.0, RIGHT, current=0)
    step(d, 0.2, RIGHT, current=0)
    # 1 s without any update: nothing is known about that time.
    late = step(d, 1.2, RIGHT, current=0)
    assert late.reason == "dwell"
    assert late.progress == 0.0
    assert step(d, 1.4, RIGHT, current=0).reason == "dwell"
    assert step(d, 1.5, RIGHT, current=0).reason == "switch"


def test_gap_below_limit_keeps_dwell() -> None:
    d = decider(dwell_s=1.0, max_gap_s=0.75)
    out = run(d, [0.0, 0.7, 1.0], RIGHT, current=0)
    assert [o.reason for o in out] == ["dwell", "dwell", "switch"]


def test_same_monitor_glance_resets_dwell() -> None:
    d = decider()
    run(d, [0.0, 0.1, 0.2], RIGHT, current=0)
    assert step(d, 0.25, LEFT, current=0).reason == "same"
    assert step(d, 0.3, RIGHT, current=0).reason == "dwell"
    assert step(d, 0.5, RIGHT, current=0).reason == "dwell"
    assert step(d, 0.6, RIGHT, current=0).fired


# ------------------------------------------------------------------ hysteresis
def test_gaze_just_across_the_bezel_is_held_by_hysteresis() -> None:
    d = decider()
    # 20 px into the right monitor; threshold is 0.06 * 1080 = 64.8 px.
    near_bezel = (1940.0, 500.0)
    out = run(d, ticks(0, 3), near_bezel, current=0)
    assert all(o.reason == "hysteresis" for o in out)
    assert all(o.candidate == 1 and not o.fired and not o.pending for o in out)


def test_gaze_clearly_across_the_bezel_passes_hysteresis() -> None:
    d = decider()
    past = (1920.0 + 70.0, 500.0)
    out = run(d, [0.0, 0.3], past, current=0)
    assert [o.reason for o in out] == ["dwell", "switch"]


def test_hysteresis_flicker_at_bezel_resets_dwell() -> None:
    d = decider()
    step(d, 0.0, RIGHT, current=0)
    step(d, 0.1, RIGHT, current=0)
    assert step(d, 0.2, (1930.0, 500.0), current=0).reason == "hysteresis"
    assert step(d, 0.3, RIGHT, current=0).reason == "dwell"
    assert step(d, 0.6, RIGHT, current=0).fired


def test_hysteresis_is_symmetric_for_switching_back() -> None:
    d = decider()
    near_bezel_left = (1900.0, 500.0)  # 20 px into the left monitor
    assert step(d, 0.0, near_bezel_left, current=1).reason == "hysteresis"
    assert step(d, 0.1, (1800.0, 500.0), current=1).reason == "dwell"


def test_hysteresis_zero_switches_right_at_the_edge() -> None:
    d = decider(hysteresis=0.0)
    out = run(d, [0.0, 0.3], (1921.0, 500.0), current=0)
    assert out[-1].fired


def test_unknown_current_monitor_skips_hysteresis() -> None:
    d = decider()
    near_bezel = (1925.0, 500.0)
    out = run(d, [0.0, 0.3], near_bezel, current=None)
    assert [o.reason for o in out] == ["dwell", "switch"]
    assert out[-1].target == 1


def test_stale_current_index_is_treated_as_unknown() -> None:
    d = decider()
    out = run(d, [0.0, 0.3], LEFT, current=7)
    assert out[-1].reason == "switch"
    assert out[-1].target == 0


# ------------------------------------------------------------ off-screen gaze
def test_glance_at_phone_below_monitors_is_ignored() -> None:
    d = decider()
    # Far below the right monitor: > 0.35 * 2203 px (its diagonal) outside.
    phone = (2900.0, 1080.0 + 900.0)
    out = run(d, ticks(0, 2), phone, current=0)
    assert all(o.reason == "off_screen" and o.candidate is None for o in out)


def test_glance_at_phone_mid_dwell_resets_dwell() -> None:
    d = decider()
    run(d, [0.0, 0.1, 0.2], RIGHT, current=0)
    assert step(d, 0.25, (2900.0, 3000.0), current=0).reason == "off_screen"
    back = run(d, [0.3, 0.4, 0.5, 0.6], RIGHT, current=0)
    assert [o.reason for o in back] == ["dwell", "dwell", "dwell", "switch"]


def test_gaze_slightly_outside_is_attributed_to_nearest_monitor() -> None:
    d = decider()
    beyond_right_edge = (3840.0 + 150.0, 500.0)
    out = run(d, [0.0, 0.3], beyond_right_edge, current=0)
    assert out[0].candidate == 1
    assert out[-1].target == 1


def test_off_screen_near_current_monitor_is_same() -> None:
    d = decider()
    above_left = (900.0, -200.0)
    assert step(d, 0.0, above_left, current=0).reason == "same"


def test_off_screen_margin_scales_with_diagonal() -> None:
    d = decider(off_screen_margin=0.1)
    # 300 px below: beyond 0.1 * 2203 = 220 px, so off screen.
    assert step(d, 0.0, (2900.0, 1380.0), current=0).reason == "off_screen"
    d.set_config(SwitchConfig(off_screen_margin=0.35))
    assert step(d, 0.1, (2900.0, 1380.0), current=0).reason == "dwell"


# ------------------------------------------------------------------- layouts
def test_three_monitor_row_switch_to_far_right() -> None:
    d = decider(TRIPLE)
    out = run(d, [0.0, 0.1, 0.2, 0.3], (2900.0, 500.0), current=0)
    assert out[-1].target == 2


def test_three_monitor_sweep_restarts_dwell_per_candidate() -> None:
    d = decider(TRIPLE)
    # The eyes sweep from the left monitor over the middle one to the right one.
    assert step(d, 0.0, (900.0, 500.0), current=0).candidate == 1
    assert step(d, 0.1, (900.0, 500.0), current=0).reason == "dwell"
    assert step(d, 0.2, (2900.0, 500.0), current=0).progress == 0.0
    assert step(d, 0.3, (2900.0, 500.0), current=0).reason == "dwell"
    assert step(d, 0.4, (2900.0, 500.0), current=0).reason == "dwell"
    final = step(d, 0.5, (2900.0, 500.0), current=0)
    assert final.reason == "switch"
    assert final.target == 2


def test_three_monitor_middle_to_left_and_back() -> None:
    d = decider(TRIPLE)
    out = run(d, [0.0, 0.3], (-1000.0, 500.0), current=1)
    assert out[-1].target == 0
    d.notify_switched(0.3, 0)
    out = run(d, [1.0, 1.3], (1000.0, 500.0), current=0)
    assert out[-1].target == 1


def test_stacked_monitors_switch_up_and_down() -> None:
    d = decider(STACKED)
    up = run(d, [0.0, 0.3], (960.0, -500.0), current=0)
    assert up[-1].target == 1
    d.notify_switched(0.3, 1)
    down = run(d, [1.0, 1.3], (960.0, 600.0), current=1)
    assert down[-1].target == 0


def test_stacked_monitors_vertical_hysteresis() -> None:
    d = decider(STACKED)
    # 30 px above the shared edge: less than 0.06 * 1080 = 64.8 px.
    assert step(d, 0.0, (960.0, -30.0), current=0).reason == "hysteresis"
    assert step(d, 0.1, (960.0, -70.0), current=0).reason == "dwell"


def test_laptop_below_layout() -> None:
    d = decider(LAPTOP_BELOW)
    # Looking down at the laptop panel.
    assert step(d, 0.0, (1200.0, 1500.0), current=0).reason == "hysteresis"  # 60 px < 64.8
    out = run(d, [0.1, 0.4], (1200.0, 1900.0), current=0)
    assert out[-1].target == 1
    d.notify_switched(0.4, 1)
    # Back up to the external monitor.
    out = run(d, [2.0, 2.3], (1200.0, 700.0), current=1)
    assert out[-1].target == 0


def test_laptop_below_empty_corner_maps_to_nearest_monitor() -> None:
    d = decider(LAPTOP_BELOW)
    # Below the external monitor but left of the laptop panel: nearer the external one.
    assert step(d, 0.0, (100.0, 1500.0), current=0).reason == "same"
    # Same point while the cursor is on the laptop: gaze favours the external monitor.
    dec = step(d, 0.1, (100.0, 1520.0), current=1)
    assert dec.candidate == 0


def test_overlapping_monitors_prefer_current() -> None:
    mirrored = [mon(0, 0, 0), mon(1, 0, 0)]
    d = decider(mirrored)
    assert step(d, 0.0, (500.0, 500.0), current=1).reason == "same"
    assert step(d, 0.1, (500.0, 500.0), current=0).reason == "same"


# ---------------------------------------------------------------------- guards
def test_mouse_guard_blocks_then_fires_as_soon_as_it_expires() -> None:
    d = decider()
    mouse = 0.1  # user moved the mouse; grace 1.5 s -> blocked until 1.6
    times = ticks(0.0, 2.0)
    out = run(d, times, RIGHT, current=0, mouse=mouse)
    reasons = [o.reason for o in out]
    assert reasons[:3] == ["dwell"] * 3
    assert set(reasons[3:16]) == {"mouse"}
    assert first_switch(out, times) == pytest.approx(1.6)
    # The dwell kept accumulating while guarded: progress stays at 1.
    assert all(o.progress == 1.0 for o in out[3:16])


def test_typing_guard_blocks_then_fires_as_soon_as_it_expires() -> None:
    d = decider()
    times = ticks(0.0, 3.0)
    out = run(d, times, RIGHT, current=0, key=0.5)
    assert {o.reason for o in out[3:25]} == {"typing"}
    assert first_switch(out, times) == pytest.approx(2.5)


def test_guard_expiry_after_user_looked_away_does_not_fire() -> None:
    d = decider()
    run(d, ticks(0.0, 1.0), RIGHT, current=0, key=0.5)
    # The user looked back at their own monitor before the typing grace ended...
    run(d, ticks(1.1, 2.4), LEFT, current=0, key=0.5)
    # ...so a glance at the other monitor right at expiry needs a fresh dwell.
    out = run(d, [2.5, 2.6, 2.8], RIGHT, current=0, key=0.5)
    assert [o.reason for o in out] == ["dwell", "dwell", "switch"]
    assert out[0].progress == 0.0


def test_mouse_guard_reported_before_typing_guard() -> None:
    d = decider(dwell_s=0.0)
    assert step(d, 1.0, RIGHT, current=0, mouse=0.9, key=0.9).reason == "mouse"
    assert step(d, 2.5, RIGHT, current=0, mouse=0.9, key=0.9).reason == "typing"
    assert step(d, 3.0, RIGHT, current=0, mouse=0.9, key=0.9).fired


def test_pending_only_when_switch_could_fire_soon() -> None:
    d = decider()
    # Typing just now: blocked for 2 s, far beyond the dwell -> keep the slow rate.
    typing = step(d, 10.0, RIGHT, current=0, key=10.0)
    assert typing.reason == "dwell"
    assert not typing.pending
    blocked = run(d, [10.1, 10.2, 10.3], RIGHT, current=0, key=10.0)[-1]
    assert blocked.reason == "typing"
    assert not blocked.pending
    # Close to expiry the scheduler should speed up.
    later = run(d, ticks(10.4, 11.4), RIGHT, current=0, key=10.0)
    assert not any(o.pending for o in later)
    soon = step(d, 12.0 - PENDING_HORIZON_S + 0.1, RIGHT, current=0, key=10.0)
    assert soon.reason == "typing"
    assert soon.pending


def test_cooldown_prevents_ping_pong() -> None:
    d = decider(cooldown_s=1.0)
    out = run(d, [0.0, 0.3], RIGHT, current=0)
    assert out[-1].target == 1
    d.notify_switched(0.3, 1)
    # Noisy gaze snaps back to the left monitor right after the switch.
    back = run(d, ticks(0.4, 1.2), LEFT, current=1)
    reasons = [o.reason for o in back]
    assert not any(o.fired for o in back)
    assert reasons[:3] == ["dwell"] * 3
    assert set(reasons[3:]) == {"cooldown"}
    # Sampling speeds up only once the cooldown is about to end.
    assert not back[3].pending
    assert back[-1].pending
    assert step(d, 1.3, LEFT, current=1).fired


def test_single_frame_blip_never_switches() -> None:
    d = decider()
    for i in range(20):
        t = i * 0.1
        gaze = RIGHT if i % 3 == 0 else LEFT
        assert not step(d, t, gaze, current=0).fired


def test_failed_switch_is_not_retried_every_frame() -> None:
    d = decider()
    out = run(d, [0.0, 0.3], RIGHT, current=0)
    assert out[-1].fired
    # The caller could not move the cursor (no notify_switched, cursor still on 0).
    retry = run(d, ticks(0.4, 0.8), RIGHT, current=0)
    assert not any(o.fired for o in retry)
    assert retry[-1].reason == "cooldown"
    assert step(d, 0.9, RIGHT, current=0).fired


def test_switch_from_no_monitor_after_dwell() -> None:
    d = decider()
    out = run(d, [0.0, 0.1, 0.2, 0.3], LEFT, current=None)
    assert [o.reason for o in out] == ["dwell", "dwell", "dwell", "switch"]
    assert out[-1].target == 0


# ------------------------------------------------------------ configuration
def test_disabled_resets_dwell() -> None:
    d = decider()
    run(d, [0.0, 0.1, 0.2], RIGHT, current=0)
    off = step(d, 0.25, RIGHT, current=0, enabled=False)
    assert off.reason == "disabled"
    assert off.target is None
    assert not off.pending
    assert step(d, 0.3, RIGHT, current=0).progress == 0.0


def test_no_monitors_is_disabled() -> None:
    d = decider([])
    assert step(d, 0.0, RIGHT, current=None).reason == "disabled"


def test_set_monitors_discards_dwell() -> None:
    d = decider()
    run(d, [0.0, 0.1, 0.2], RIGHT, current=0)
    d.set_monitors(DUAL)
    assert step(d, 0.3, RIGHT, current=0).reason == "dwell"


def test_reset_clears_cooldown() -> None:
    d = decider(cooldown_s=5.0)
    run(d, [0.0, 0.3], RIGHT, current=0)
    d.notify_switched(0.3, 1)
    assert d.last_switch_target == 1
    d.reset()
    assert d.last_switch_time == -math.inf
    out = run(d, [0.4, 0.7], LEFT, current=1)
    assert out[-1].fired


def test_set_config_takes_effect_immediately() -> None:
    d = decider(dwell_s=2.0)
    step(d, 0.0, RIGHT, current=0)
    assert step(d, 0.5, RIGHT, current=0).reason == "dwell"
    d.set_config(SwitchConfig(dwell_s=0.5))
    assert step(d, 0.6, RIGHT, current=0).fired


def test_config_from_settings_converts_milliseconds() -> None:
    s = SwitchingSettings(
        dwell_ms=450,
        hysteresis=0.1,
        off_screen_margin=0.5,
        cooldown_ms=800,
        mouse_grace_ms=1000,
        typing_grace_ms=2500,
    )
    cfg = SwitchConfig.from_settings(s)
    assert cfg == SwitchConfig(
        dwell_s=0.45,
        hysteresis=0.1,
        off_screen_margin=0.5,
        cooldown_s=0.8,
        mouse_grace_s=1.0,
        typing_grace_s=2.5,
        max_gap_s=SwitchConfig().max_gap_s,
    )


def test_defaults_match_settings_defaults() -> None:
    assert SwitchConfig.from_settings(SwitchingSettings()) == SwitchConfig()
