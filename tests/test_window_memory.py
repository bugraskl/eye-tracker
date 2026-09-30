"""Tests for per-monitor cursor/window memory (engine/window_memory.py)."""

from __future__ import annotations

import logging

import pytest

from eye_tracker.engine.window_memory import WindowMemory, choose_cursor_target
from eye_tracker.types import Monitor, Rect, WindowRef

LEFT = Monitor(0, "left", Rect(0, 0, 1920, 1080), primary=True)
RIGHT = Monitor(1, "right", Rect(1920, 0, 2560, 1440))


def test_cursor_memory_per_monitor() -> None:
    mem = WindowMemory()
    assert mem.last_cursor(0) is None
    mem.record_cursor(0, (100, 200))
    mem.record_cursor(1, (2000, 300))
    mem.record_cursor(0, (150, 250))
    assert mem.last_cursor(0) == (150, 250)
    assert mem.last_cursor(1) == (2000, 300)


def test_window_memory_forget_and_clear() -> None:
    mem = WindowMemory()
    a, b = WindowRef(handle=1), WindowRef(handle=2)
    mem.record_window(0, a)
    mem.record_window(1, b)
    assert mem.last_window(0) is a
    mem.forget_window(0)
    mem.forget_window(5)  # unknown index is fine
    assert mem.last_window(0) is None
    assert mem.last_window(1) is b
    mem.record_cursor(1, (2000, 10))
    mem.clear()
    assert mem.last_window(1) is None
    assert mem.last_cursor(1) is None


def test_last_mode_returns_remembered_position() -> None:
    mem = WindowMemory()
    mem.record_cursor(1, (3000, 700))
    assert choose_cursor_target("last", RIGHT, mem, None, None) == (3000, 700)


def test_last_mode_ignores_position_outside_monitor() -> None:
    mem = WindowMemory()
    mem.record_cursor(1, (100, 100))  # stale: layout changed
    assert choose_cursor_target("last", RIGHT, mem, None, None) == (3200, 720)


def test_last_mode_falls_back_to_window_centre() -> None:
    mem = WindowMemory()
    window = Rect(2000, 100, 800, 600)
    assert choose_cursor_target("last", RIGHT, mem, None, window) == (2400, 400)


def test_last_mode_uses_visible_part_of_straddling_window() -> None:
    mem = WindowMemory()
    window = Rect(1520, 100, 800, 400)  # half on the left monitor
    target = choose_cursor_target("last", RIGHT, mem, None, window)
    assert target == (2120, 300)
    assert RIGHT.rect.contains(*target)


def test_last_mode_ignores_window_on_another_monitor() -> None:
    mem = WindowMemory()
    window = Rect(100, 100, 500, 500)
    assert choose_cursor_target("last", RIGHT, mem, None, window) == (3200, 720)


def test_center_mode() -> None:
    mem = WindowMemory()
    mem.record_cursor(0, (5, 5))
    assert choose_cursor_target("center", LEFT, mem, (10.0, 10.0), None) == (960, 540)


def test_gaze_mode_uses_gaze_inside_monitor() -> None:
    mem = WindowMemory()
    assert choose_cursor_target("gaze", RIGHT, mem, (2500.4, 900.6), None) == (2500, 901)


def test_gaze_mode_clamps_into_monitor_away_from_the_edges() -> None:
    # Never onto the outermost row, column or corner (auto-hide taskbars, docks
    # and hot corners react to it): 2 % of 2560 x 1440 is a 51 x 29 px inset.
    mem = WindowMemory()
    target = choose_cursor_target("gaze", RIGHT, mem, (5000.0, -40.0), None)
    assert target == (RIGHT.rect.right - 1 - 51, 29)
    assert RIGHT.rect.contains(*target)


@pytest.mark.parametrize(
    ("gaze", "expected"),
    [
        ((-150.0, -100.0), (38, 22)),  # hot corner at the top left
        ((960.0, 1200.0), (960, 1079 - 22)),  # auto-hidden taskbar at the bottom
        ((1925.0, 500.0), (1919 - 38, 500)),  # the edge next to the other monitor
    ],
)
def test_gaze_mode_keeps_the_minimum_inset_on_1080p(
    gaze: tuple[float, float], expected: tuple[int, int]
) -> None:
    assert choose_cursor_target("gaze", LEFT, WindowMemory(), gaze, None) == expected


def test_gaze_mode_inset_on_a_small_monitor_uses_the_minimum() -> None:
    small = Monitor(3, "small", Rect(0, 0, 400, 300))
    assert choose_cursor_target("gaze", small, WindowMemory(), (-5.0, 999.0), None) == (16, 283)


@pytest.mark.parametrize("gaze", [None, (float("nan"), 3.0)])
def test_gaze_mode_without_gaze_uses_centre(gaze: tuple[float, float] | None) -> None:
    assert choose_cursor_target("gaze", LEFT, WindowMemory(), gaze, None) == (960, 540)


def test_unknown_mode_uses_centre(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        target = choose_cursor_target("teleport", LEFT, WindowMemory(), (5.0, 5.0), None)
    assert target == (960, 540)
    assert "teleport" in caplog.text


def test_target_is_always_inside_monitor() -> None:
    odd = Monitor(2, "tiny", Rect(-7, -3, 1, 1))
    mem = WindowMemory()
    for mode in ("last", "center", "gaze"):
        assert choose_cursor_target(mode, odd, mem, (1e6, -1e6), Rect(-100, -100, 50, 50)) == (
            -7,
            -3,
        )
