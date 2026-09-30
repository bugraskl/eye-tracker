"""Tests for hook-free mouse/keyboard activity tracking (engine/input_state.py)."""

from __future__ import annotations

import math

import pytest

from eye_tracker.engine.input_state import InputTracker


class FakeIdle:
    """OS idle-time provider driven by explicit input timestamps."""

    def __init__(self, last_input: float | None = 0.0) -> None:
        self.last_input = last_input
        self.now = 0.0
        self.fail = False

    def __call__(self) -> float | None:
        if self.fail:
            raise OSError("boom")
        if self.last_input is None:
            return None
        return max(0.0, self.now - self.last_input)


def make(
    idle: FakeIdle | None = None, key_idle: FakeIdle | None = None
) -> tuple[InputTracker, FakeIdle, FakeIdle]:
    idle = idle or FakeIdle()
    key_idle = key_idle or FakeIdle(last_input=None)
    return InputTracker(idle, key_idle), idle, key_idle


def poll(tr: InputTracker, idle: FakeIdle, key: FakeIdle, now: float, pos: tuple[int, int]):
    idle.now = key.now = now
    tr.poll(now, pos)


# ----------------------------------------------------------------------- mouse
def test_initial_state_is_never() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    assert tr.last_mouse_activity == -math.inf
    assert tr.last_key_activity == -math.inf
    assert tr.last_any_activity == -math.inf
    assert tr.seconds_since_any(10.0) is None
    assert not tr.manual_move


def test_first_poll_only_sets_the_baseline() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    tr.poll(1.0, (500, 500))
    assert tr.cursor == (500, 500)
    assert tr.last_mouse_activity == -math.inf
    assert not tr.manual_move


def test_cursor_move_is_mouse_activity() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    tr.poll(1.0, (500, 500))
    tr.poll(1.1, (520, 505))
    assert tr.manual_move
    assert tr.last_mouse_activity == 1.1
    assert tr.last_any_activity == 1.1
    assert tr.seconds_since_any(1.6) == pytest.approx(0.5)
    tr.poll(1.2, (520, 505))
    assert not tr.manual_move
    assert tr.last_mouse_activity == 1.1


def test_tiny_jitter_is_not_mouse_activity() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    tr.poll(1.0, (500, 500))
    tr.poll(1.1, (502, 500))
    tr.poll(1.2, (503, 501))
    assert tr.last_mouse_activity == -math.inf


def test_programmatic_move_is_not_user_activity() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    tr.poll(1.0, (500, 500))
    tr.note_programmatic_move((2880, 540), 1.05)
    tr.poll(1.1, (2881, 541))
    assert not tr.manual_move
    assert tr.last_mouse_activity == -math.inf
    assert tr.cursor == (2881, 541)


def test_user_move_after_warp_is_detected() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    tr.poll(1.0, (500, 500))
    tr.note_programmatic_move((2880, 540), 1.05)
    tr.poll(1.1, (2880, 540))
    tr.poll(1.2, (2870, 540))
    assert tr.manual_move
    assert tr.last_mouse_activity == 1.2


def test_warp_landing_elsewhere_is_user_activity() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    tr.poll(1.0, (500, 500))
    tr.note_programmatic_move((2880, 540), 1.05)
    tr.poll(1.1, (2900, 540))  # the user moved the mouse during the warp
    assert tr.manual_move


def test_warp_explains_only_one_jump() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    tr.poll(1.0, (500, 500))
    tr.note_programmatic_move((2880, 540), 1.0)
    tr.poll(1.1, (2880, 540))
    tr.poll(1.2, (500, 500))  # user drags it back
    tr.poll(1.3, (2880, 540))  # ...and to the warp target again
    assert tr.manual_move
    assert tr.last_mouse_activity == 1.3


def test_warp_seen_late_by_a_slow_poll_is_still_explained() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    tr.poll(1.0, (500, 500))
    tr.note_programmatic_move((2880, 540), 1.0)
    tr.poll(2.0, (2880, 540))  # first poll after the warp, 1 s later
    assert not tr.manual_move


def test_warp_expires() -> None:
    tr = InputTracker(lambda: None, lambda: None)
    tr.poll(1.0, (500, 500))
    tr.note_programmatic_move((2880, 540), 1.0)
    tr.poll(1.1, (500, 500))  # warp not visible yet (asynchronous warp)
    tr.poll(1.4, (2880, 540))  # arrives within the window
    assert not tr.manual_move
    tr.note_programmatic_move((100, 100), 2.0)
    tr.poll(2.1, (2880, 540))
    tr.poll(3.0, (100, 100))  # far too late: a user move to the same place
    assert tr.manual_move


# -------------------------------------------------------------------- keyboard
def test_key_idle_provider_sets_key_activity() -> None:
    tr, idle, key = make(FakeIdle(last_input=4.0), FakeIdle(last_input=4.0))
    poll(tr, idle, key, 5.0, (0, 0))
    assert tr.last_key_activity == pytest.approx(4.0)
    assert tr.last_any_activity == pytest.approx(4.0)
    key.last_input = 5.5
    idle.last_input = 5.5
    poll(tr, idle, key, 6.0, (0, 0))
    assert tr.last_key_activity == pytest.approx(5.5)


def test_idle_reset_without_cursor_motion_is_typing() -> None:
    tr, idle, key = make(FakeIdle(last_input=0.0))
    poll(tr, idle, key, 1.0, (100, 100))
    assert tr.last_key_activity == -math.inf  # first poll: baseline only
    idle.last_input = 1.07  # a key press between polls
    poll(tr, idle, key, 1.1, (100, 100))
    assert tr.last_key_activity == pytest.approx(1.07)
    assert tr.last_mouse_activity == -math.inf
    assert tr.seconds_since_any(1.1) == pytest.approx(0.03)


def test_idle_reset_with_cursor_motion_is_mouse_not_typing() -> None:
    tr, idle, key = make(FakeIdle(last_input=0.0))
    poll(tr, idle, key, 1.0, (100, 100))
    idle.last_input = 1.08
    poll(tr, idle, key, 1.1, (300, 100))
    assert tr.last_mouse_activity == 1.1
    assert tr.last_key_activity == -math.inf
    # The mouse stopped; the OS input time does not move, so still no typing.
    poll(tr, idle, key, 1.2, (300, 100))
    poll(tr, idle, key, 1.3, (300, 100))
    assert tr.last_key_activity == -math.inf


def test_small_cursor_motion_blocks_typing_inference() -> None:
    tr, idle, key = make(FakeIdle(last_input=0.0))
    poll(tr, idle, key, 1.0, (100, 100))
    idle.last_input = 1.08
    poll(tr, idle, key, 1.1, (101, 100))  # slow, precise mouse motion
    assert tr.last_key_activity == -math.inf
    assert tr.last_mouse_activity == -math.inf
    assert tr.last_any_activity == pytest.approx(1.08)


def test_idle_jitter_is_ignored() -> None:
    tr, idle, key = make(FakeIdle(last_input=0.0))
    poll(tr, idle, key, 1.0, (100, 100))
    idle.last_input = 0.03  # clock skew between our clock and the OS idle timer
    poll(tr, idle, key, 1.1, (100, 100))
    assert tr.last_key_activity == -math.inf


def test_stale_cached_idle_value_is_ignored() -> None:
    values = iter([30.0, 30.0, 30.0])
    tr = InputTracker(lambda: next(values), lambda: None)
    tr.poll(1.0, (0, 0))
    tr.poll(1.5, (0, 0))
    tr.poll(2.0, (0, 0))
    assert tr.last_key_activity == -math.inf


def test_failing_providers_are_tolerated() -> None:
    tr, idle, key = make(FakeIdle(last_input=0.0))
    idle.fail = True
    key.fail = True
    poll(tr, idle, key, 1.0, (100, 100))
    poll(tr, idle, key, 1.1, (200, 100))
    assert tr.last_mouse_activity == 1.1
    assert tr.last_key_activity == -math.inf


def test_invalid_provider_values_are_ignored() -> None:
    values = iter([float("nan"), -1.0, "garbage"])
    tr = InputTracker(lambda: next(values), lambda: None)  # type: ignore[arg-type,return-value]
    for i in range(3):
        tr.poll(1.0 + i, (0, 0))
    assert tr.last_any_activity == -math.inf
    assert tr.seconds_since_any(5.0) is None


def test_os_idle_populates_any_activity() -> None:
    tr, idle, key = make(FakeIdle(last_input=2.0))
    poll(tr, idle, key, 10.0, (0, 0))
    assert tr.seconds_since_any(10.0) == pytest.approx(8.0)
