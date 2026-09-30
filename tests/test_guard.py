"""Tests for the shoulder-surfer guard (engine/guard.py)."""

from __future__ import annotations

from eye_tracker.config import PrivacySettings
from eye_tracker.engine.guard import GuardConfig, ShoulderGuard


def guard(delay_s: float = 2.0, clear_s: float = 1.5) -> ShoulderGuard:
    return ShoulderGuard(GuardConfig(enabled=True, delay_s=delay_s, clear_s=clear_s))


def feed(
    g: ShoulderGuard, start: float, stop: float, faces: int | None, dt: float = 0.25
) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    n = round((stop - start) / dt)
    for i in range(n + 1):
        t = start + i * dt
        result = g.update(t, faces)
        if result is not None:
            out.append((t, result))
    return out


def test_default_config_is_disabled() -> None:
    g = ShoulderGuard()
    assert feed(g, 0, 10, faces=3) == []
    assert not g.active


def test_single_face_never_triggers() -> None:
    g = guard()
    assert feed(g, 0, 30, faces=1) == []
    assert feed(g, 30.25, 40, faces=0) == []


def test_second_face_triggers_after_delay() -> None:
    g = guard(delay_s=2.0)
    assert feed(g, 0, 10, faces=2) == [(2.0, "trigger")]
    assert g.active


def test_passer_by_is_ignored() -> None:
    g = guard(delay_s=2.0)
    assert feed(g, 0, 1.5, faces=2) == []
    # Walked out of view: the run of two faces restarts.
    assert feed(g, 1.75, 2.5, faces=1) == []
    assert feed(g, 2.75, 4.5, faces=2) == []
    assert feed(g, 4.75, 4.75, faces=2) == [(4.75, "trigger")]


def test_clears_after_second_face_is_gone() -> None:
    g = guard(delay_s=1.0, clear_s=1.5)
    feed(g, 0, 1, faces=2)
    assert g.active
    assert feed(g, 1.25, 5, faces=1) == [(2.75, "clear")]
    assert not g.active


def test_brief_dropout_does_not_clear() -> None:
    g = guard(delay_s=1.0, clear_s=1.5)
    feed(g, 0, 1, faces=2)
    assert feed(g, 1.25, 2.25, faces=1) == []
    assert feed(g, 2.5, 3.0, faces=2) == []
    assert g.active
    # The clear timer restarted when the second face reappeared.
    assert feed(g, 3.25, 5.0, faces=1) == [(4.75, "clear")]


def test_unknown_face_count_keeps_state() -> None:
    g = guard(delay_s=1.0)
    feed(g, 0, 1, faces=2)
    assert feed(g, 1.25, 20, faces=None) == []
    assert g.active


def test_unknown_face_count_breaks_a_pending_run() -> None:
    g = guard(delay_s=2.0)
    feed(g, 0, 1.5, faces=2)
    g.update(1.75, None)
    assert g.update(2.0, 2) is None  # run restarted at 2.0
    assert g.update(3.75, 2) is None
    assert g.update(4.0, 2) == "trigger"


def test_zero_delay_triggers_immediately() -> None:
    g = guard(delay_s=0.0)
    assert g.update(5.0, 2) == "trigger"


def test_disabling_while_active_clears_once() -> None:
    g = guard(delay_s=0.0)
    g.update(0.0, 2)
    g.set_config(GuardConfig(enabled=False))
    assert g.update(0.5, 2) == "clear"
    assert g.update(1.0, 2) is None
    assert not g.active


def test_reset_is_silent() -> None:
    g = guard(delay_s=0.0)
    g.update(0.0, 3)
    g.reset()
    assert not g.active
    assert g.update(0.5, 1) is None


def test_config_from_settings() -> None:
    s = PrivacySettings(shoulder_guard=True, guard_delay_s=4.0)
    cfg = GuardConfig.from_settings(s)
    assert cfg.enabled
    assert cfg.delay_s == 4.0
    assert cfg.clear_s == GuardConfig().clear_s
    assert not GuardConfig.from_settings(PrivacySettings()).enabled
