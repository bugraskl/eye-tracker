"""Scenario tests for the split-pane focus decision (panes/decider.py)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from eye_tracker.config import PaneSettings, SwitchingSettings
from eye_tracker.panes.decider import (
    FALLBACK_GAZE_ERROR_PX,
    PANE_REASONS,
    PaneConfig,
    PaneDecider,
    PaneDecision,
    divider_margin,
    eligible_panes,
    separation_axis,
)
from eye_tracker.panes.types import Pane, PaneSnapshot
from eye_tracker.types import Rect

NEVER = -math.inf
WINDOW = Rect(0, 0, 1920, 1080)
LEFT = Rect(0, 0, 960, 1080)
RIGHT = Rect(960, 0, 960, 1080)
SIGMA = (100.0, 80.0)
ON_LEFT = (400.0, 500.0)
ON_RIGHT = (1500.0, 500.0)


def snap(
    now: float,
    *rects: Rect,
    focused: int = 0,
    handle: object = 7,
) -> PaneSnapshot:
    rects = rects or (LEFT, RIGHT)
    panes = tuple(Pane(f"%{i}", r, i == focused, "tmux") for i, r in enumerate(rects))
    return PaneSnapshot(handle, panes, now)


def ticks(start: float, stop: float, dt: float = 0.1) -> list[float]:
    n = round((stop - start) / dt)
    return [round(start + i * dt, 6) for i in range(n + 1)]


def run(
    d: PaneDecider,
    times: Sequence[float],
    gaze: tuple[float, float] | None,
    *,
    rects: Sequence[Rect] = (LEFT, RIGHT),
    focused: int = 0,
    sigma: tuple[float, float] | None = SIGMA,
    mouse: float = NEVER,
    key: float = NEVER,
    window: Rect | None = WINDOW,
    snapshot_age: float = 0.0,
) -> list[PaneDecision]:
    out = []
    for t in times:
        s = snap(t - snapshot_age, *rects, focused=focused)
        out.append(d.update(t, gaze, window, s, sigma, mouse, key))
    return out


def first_fire(decisions: Sequence[PaneDecision], times: Sequence[float]) -> float | None:
    for t, dec in zip(times, decisions, strict=True):
        if dec.fired:
            return t
    return None


# ----------------------------------------------------------------- geometry
def test_separation_axis() -> None:
    assert separation_axis(LEFT, RIGHT) == 0
    top, bottom = Rect(0, 0, 1920, 540), Rect(0, 540, 1920, 540)
    assert separation_axis(top, bottom) == 1
    # A one-cell border between panes is still "side by side".
    assert separation_axis(Rect(0, 0, 952, 1080), Rect(960, 0, 960, 1080)) == 0
    # Diagonal neighbours: the larger centre offset decides.
    assert separation_axis(Rect(0, 0, 400, 400), Rect(1400, 500, 400, 400)) == 0
    assert separation_axis(Rect(0, 0, 400, 400), Rect(500, 1400, 400, 400)) == 1


def test_divider_margin() -> None:
    assert divider_margin((1010.0, 0.0), LEFT, RIGHT, 0) == pytest.approx(50.0)
    assert divider_margin((900.0, 0.0), RIGHT, LEFT, 0) == pytest.approx(60.0)
    gap_left, gap_right = Rect(0, 0, 952, 100), Rect(960, 0, 960, 100)
    assert divider_margin((1006.0, 0.0), gap_left, gap_right, 0) == pytest.approx(50.0)


# ------------------------------------------------------------------- switching
def test_switch_after_the_dwell() -> None:
    d = PaneDecider()
    times = ticks(0.0, 1.0)
    decisions = run(d, times, ON_RIGHT)
    assert first_fire(decisions, times) == pytest.approx(0.6)
    fired = next(dec for dec in decisions if dec.fired)
    assert fired.target is not None
    assert fired.target.id == "%1"
    assert fired.reason == "switch"
    assert decisions[0].reason == "dwell"
    assert decisions[0].pending
    assert all(dec.reason in PANE_REASONS for dec in decisions)


def test_dwell_is_a_majority_vote() -> None:
    d = PaneDecider()
    run(d, ticks(0.0, 2.0), ON_LEFT)  # working in the focused pane
    times = ticks(2.1, 3.0)
    gazes = [ON_RIGHT] * len(times)
    gazes[3] = ON_LEFT  # one stray sample
    decisions = [
        d.update(t, g, WINDOW, snap(t), SIGMA, NEVER, NEVER)
        for t, g in zip(times, gazes, strict=True)
    ]
    fired_at = first_fire(decisions, times)
    assert fired_at is not None
    assert 2.6 <= fired_at <= 2.9  # not restarted by the stray sample


def test_scattered_gaze_never_switches() -> None:
    d = PaneDecider()
    times = ticks(0.0, 5.0)
    decisions = [
        d.update(t, ON_RIGHT if i % 2 else ON_LEFT, WINDOW, snap(t), SIGMA, NEVER, NEVER)
        for i, t in enumerate(times)
    ]
    assert first_fire(decisions, times) is None


def test_small_pane_is_never_a_target() -> None:
    narrow = (Rect(0, 0, 1680, 1080), Rect(1680, 0, 240, 1080))
    d = PaneDecider()
    decisions = run(d, ticks(0.0, 3.0), (1800.0, 500.0), rects=narrow)
    assert {dec.reason for dec in decisions} == {"small"}
    # A worse calibration makes even half a screen too small.
    d = PaneDecider()
    decisions = run(d, ticks(0.0, 3.0), ON_RIGHT, sigma=(400.0, 80.0))
    assert {dec.reason for dec in decisions} == {"small"}
    # min_pane_px applies with a perfect calibration too.
    d = PaneDecider(PaneConfig(min_pane_px=1000.0))
    assert run(d, [0.0], ON_RIGHT, sigma=(1.0, 1.0))[0].reason == "small"


def test_stacked_panes_use_the_vertical_error() -> None:
    stacked = (Rect(0, 0, 1920, 540), Rect(0, 540, 1920, 540))
    below = (900.0, 800.0)
    d = PaneDecider()
    assert run(d, [0.0], below, rects=stacked, sigma=(10.0, 300.0))[0].reason == "small"
    d = PaneDecider()
    times = ticks(0.0, 1.0)
    assert first_fire(run(d, times, below, rects=stacked, sigma=(900.0, 80.0)), times)


def test_gaze_near_the_divider_is_held() -> None:
    d = PaneDecider()
    times = ticks(0.0, 2.0)
    decisions = run(d, times, (1000.0, 500.0))  # 40 px past it; σx 100 needs 50
    assert {dec.reason for dec in decisions} == {"hysteresis"}
    assert first_fire(run(PaneDecider(), times, (1015.0, 500.0)), times) is not None


def test_nothing_happens_without_a_fresh_snapshot_or_outside_the_window() -> None:
    d = PaneDecider()
    assert run(d, ticks(0.0, 1.0), ON_RIGHT, snapshot_age=2.0)[-1].reason == "stale"
    assert d.update(0.0, ON_RIGHT, WINDOW, None, SIGMA, NEVER, NEVER).reason == "no_panes"
    d = PaneDecider()
    assert run(d, [0.0], (2500.0, 500.0), window=WINDOW)[0].reason == "outside"
    gap = (Rect(0, 0, 950, 1080), Rect(970, 0, 950, 1080))
    assert run(d, [0.1], (960.0, 500.0), rects=gap)[0].reason == "no_pane"
    assert run(d, [0.2], ON_LEFT)[0].reason == "same"
    assert run(d, [0.3], None)[0].reason == "no_gaze"


def test_disabled_clears_the_dwell() -> None:
    d = PaneDecider()
    run(d, ticks(0.0, 0.5), ON_RIGHT)
    off = d.update(0.6, ON_RIGHT, WINDOW, snap(0.6), SIGMA, NEVER, NEVER, enabled=False)
    assert off.reason == "disabled"
    times = ticks(0.7, 1.4)
    assert first_fire(run(d, times, ON_RIGHT), times) == pytest.approx(1.3)


def test_a_long_gap_restarts_the_dwell() -> None:
    d = PaneDecider()
    run(d, ticks(0.0, 0.5), ON_RIGHT)
    times = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
    assert first_fire(run(d, times, ON_RIGHT), times) == pytest.approx(2.1)


def test_fallback_error_without_calibration_accuracy() -> None:
    d = PaneDecider()
    times = ticks(0.0, 1.0)
    assert first_fire(run(d, times, ON_RIGHT, sigma=None), times) is not None
    assert FALLBACK_GAZE_ERROR_PX * PaneConfig().precision < 960


# ----------------------------------------------------------------------- guards
def test_typing_mouse_and_cooldown_guards() -> None:
    d = PaneDecider()
    times = ticks(0.0, 4.0)
    decisions = run(d, times, ON_RIGHT, key=0.0)  # typed at 0: 3 s of grace
    assert first_fire(decisions, times) == pytest.approx(3.0)
    assert decisions[10].reason == "typing"
    assert not decisions[10].pending  # far from firing: no fast sampling
    assert decisions[26].pending  # within PENDING_HORIZON_S of the end

    d = PaneDecider()
    decisions = run(d, times, ON_RIGHT, mouse=0.5)
    assert first_fire(decisions, times) == pytest.approx(2.0)
    assert decisions[10].reason == "mouse"

    # After a switch, the next one waits for the cooldown.
    d = PaneDecider()
    run(d, ticks(0.0, 0.6), ON_RIGHT)  # fires at 0.6: focus moves to %1
    times = ticks(0.7, 2.0)
    decisions = run(d, times, ON_LEFT, focused=1)
    assert first_fire(decisions, times) == pytest.approx(1.6)
    assert "manual" not in {dec.reason for dec in decisions}  # that change was ours


def test_focus_changed_by_the_user_holds_switching() -> None:
    d = PaneDecider()
    run(d, ticks(0.0, 1.0), ON_LEFT)  # focus on %0
    times = ticks(1.1, 5.0)
    # The user focused %1 themselves while looking at %0.
    decisions = run(d, times, ON_LEFT, focused=1)
    fired_at = first_fire(decisions, times)
    assert fired_at is not None
    assert fired_at >= 1.1 + PaneConfig().manual_grace_s - 1e-6
    assert decisions[10].reason == "manual"


def test_another_window_starts_from_scratch() -> None:
    d = PaneDecider()
    run(d, ticks(0.0, 0.5), ON_RIGHT)
    other = snap(0.6, handle=8)
    assert d.update(0.6, ON_RIGHT, WINDOW, other, SIGMA, NEVER, NEVER).reason == "dwell"
    times = ticks(0.7, 1.3)
    decisions = [
        d.update(t, ON_RIGHT, WINDOW, snap(t, handle=8), SIGMA, NEVER, NEVER) for t in times
    ]
    assert first_fire(decisions, times) == pytest.approx(1.2)


def test_reading_another_pane_while_typing_keeps_the_focus() -> None:
    """Typing in %0 while reading the output in %1: reading pauses do not move focus."""
    d = PaneDecider()
    for t in ticks(0.0, 3.0, 0.5):
        d.update(t, ON_RIGHT, WINDOW, snap(t), SIGMA, NEVER, t)  # typing all along
    assert d.reading_pane == "%1"
    # A reading pause: no keystroke since 3.0 s. The typing grace (3 s) would
    # end at 6.0; the reading grace (8 s) holds until 11.0.
    times = ticks(3.1, 12.0)
    decisions = run(d, times, ON_RIGHT, key=3.0)
    fired_at = first_fire(decisions, times)
    assert fired_at == pytest.approx(11.0)


def test_eligible_panes_and_config_from_settings() -> None:
    three = snap(0.0, LEFT, Rect(960, 0, 960, 540), Rect(960, 540, 960, 540))
    cfg = PaneConfig()
    assert eligible_panes(three, SIGMA, cfg) == 2
    assert eligible_panes(three, (100.0, 300.0), cfg) == 2  # side by side: x decides
    assert eligible_panes(three, (500.0, 80.0), cfg) == 0
    assert eligible_panes(None, SIGMA, cfg) == 0

    panes = PaneSettings(dwell_ms=900, precision=3.0, cooldown_ms=200, typing_grace_ms=1000)
    switching = SwitchingSettings(mouse_grace_ms=700)
    cfg = PaneConfig.from_settings(panes, switching)
    assert (cfg.dwell_s, cfg.precision, cfg.cooldown_s) == (0.9, 3.0, 0.2)
    assert (cfg.mouse_grace_s, cfg.typing_grace_s, cfg.manual_grace_s) == (0.7, 1.0, 1.0)
    assert (cfg.reading_grace_s, cfg.hysteresis, cfg.min_pane_px) == (8.0, 0.5, 240.0)


def test_reset_forgets_the_cooldown_and_the_window() -> None:
    d = PaneDecider()
    run(d, ticks(0.0, 0.6), ON_RIGHT)
    assert d.last_switch_time == pytest.approx(0.6)
    d.reset()
    assert d.last_switch_time == -math.inf
    assert d.reading_pane is None
