"""Tests for the shoulder-surfer guard (engine/guard.py)."""

from __future__ import annotations

import math

from eye_tracker.config import PrivacySettings
from eye_tracker.engine.guard import INPUT_ADOPT_DELAY_S, GuardConfig, ShoulderGuard


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


def test_brief_dropouts_of_the_second_face_do_not_restart_the_delay() -> None:
    # A distant, partly hidden onlooker is missed in some frames (4 fps idle rate).
    g = guard(delay_s=2.0)
    out: list[tuple[float, str]] = []
    for i in range(20):
        t = i * 0.25
        faces = 1 if i % 4 == 3 else 2  # every fourth detection misses the onlooker
        result = g.update(t, faces)
        if result is not None:
            out.append((t, result))
    assert out == [(2.0, "trigger")]


def test_single_missed_frame_is_tolerated_at_slow_frame_rates() -> None:
    # 1 fps (eco profile while typing): one miss is a 2 s gap between detections.
    g = guard(delay_s=2.0)
    assert g.update(0.0, 2) is None
    assert g.update(1.0, 1) is None
    assert g.update(2.0, 2) == "trigger"


def test_long_dropout_restarts_the_delay() -> None:
    g = guard(delay_s=2.0)
    assert feed(g, 0, 1.5, faces=2) == []
    assert feed(g, 1.75, 3.0, faces=1) == []  # gone for 1.5 s
    assert feed(g, 3.25, 5.0, faces=2) == []
    assert feed(g, 5.25, 5.25, faces=2) == [(5.25, "trigger")]


# The user close to the camera, and a small onlooker behind them.
OWNER = (0.35, 0.25, 0.3, 0.4)
OWNER_MOVED = (0.40, 0.28, 0.28, 0.38)
ONLOOKER = (0.75, 0.10, 0.08, 0.1)
# r2-gaze-engine-01: a user about 65 cm from the camera, and a colleague leaning
# in beside them to about 45 cm (a face side ~1.5x the user's), not overlapping.
SEATED_USER = (0.40, 0.30, 0.20, 0.27)
LEANING_COLLEAGUE = (0.05, 0.15, 0.30, 0.40)


def feed_boxes(
    g: ShoulderGuard,
    start: float,
    stop: float,
    faces: int | None,
    box: tuple[float, float, float, float] | None,
    dt: float = 0.25,
    last_input: float = -math.inf,
) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for i in range(round((stop - start) / dt) + 1):
        t = start + i * dt
        result = g.update(t, faces, box, last_input)
        if result is not None:
            out.append((t, result))
    return out


def test_guard_stays_up_when_the_user_leaves_and_the_onlooker_stays() -> None:
    g = guard(delay_s=1.0, clear_s=1.5)
    assert feed_boxes(g, 0, 1, faces=2, box=OWNER) == [(1.0, "trigger")]
    # The user walks away; only the onlooker's small face is left in view.
    assert feed_boxes(g, 1.25, 10, faces=1, box=ONLOOKER) == []
    assert g.active
    assert g.owner_missing
    # No face at all: the onlooker may just be missed, so the guard stays up...
    assert feed_boxes(g, 10.25, 20, faces=0, box=None) == []
    assert g.owner_missing
    # ...until the user is back at the desk.
    assert feed_boxes(g, 20.25, 22, faces=1, box=OWNER_MOVED) == [(21.75, "clear")]
    assert not g.owner_missing


def test_onlooker_missed_for_a_moment_keeps_the_guard_up() -> None:
    """r2-gaze-engine-02: detection gaps of a far face must not lift the curtain."""
    g = guard(delay_s=2.0, clear_s=1.5)
    assert feed_boxes(g, 0, 5, faces=2, box=OWNER, dt=0.5) == [(2.0, "trigger")]
    assert feed_boxes(g, 5.5, 30, faces=1, box=ONLOOKER, dt=0.5) == []
    # Four empty frames at the 2 fps "no face" rate, then the onlooker again.
    assert feed_boxes(g, 30.5, 32, faces=0, box=None, dt=0.5) == []
    assert feed_boxes(g, 32.5, 40, faces=1, box=ONLOOKER, dt=0.5) == []
    assert g.active
    assert g.owner_missing


def test_camera_gap_keeps_the_owner_missing() -> None:
    g = guard(delay_s=1.0, clear_s=1.5)
    feed_boxes(g, 0, 1, faces=2, box=OWNER)
    feed_boxes(g, 1.25, 3, faces=1, box=ONLOOKER)
    assert feed_boxes(g, 3.25, 10, faces=None, box=None) == []
    assert g.owner_missing
    assert feed_boxes(g, 10.25, 12, faces=0, box=None) == []
    assert g.active


def test_larger_colleague_leaning_in_never_becomes_the_owner() -> None:
    """r2-gaze-engine-01 / r2-controller-01: the owner is found by continuity."""
    g = guard(delay_s=2.0, clear_s=1.5)
    assert feed_boxes(g, 0, 10, faces=1, box=SEATED_USER) == []
    # The colleague is closer to the camera, so theirs is the primary face box.
    assert feed_boxes(g, 10.25, 14, faces=2, box=LEANING_COLLEAGUE) == [(12.25, "trigger")]
    # They straighten up and are gone from one frame to the next.
    assert feed_boxes(g, 14.25, 20, faces=1, box=SEATED_USER) == [(15.75, "clear")]
    assert not g.owner_missing
    assert not g.active


def test_lone_face_while_a_second_one_was_just_seen_is_not_taken_for_the_owner() -> None:
    g = guard(delay_s=2.0, clear_s=1.5)
    feed_boxes(g, 0, 5, faces=1, box=SEATED_USER)
    feed_boxes(g, 5.25, 6, faces=2, box=LEANING_COLLEAGUE)
    # The user turns to the colleague and is missed for a frame: only the
    # colleague's face is found. That must not make the colleague the owner.
    feed_boxes(g, 6.25, 6.25, faces=1, box=LEANING_COLLEAGUE)
    assert feed_boxes(g, 6.5, 9, faces=2, box=LEANING_COLLEAGUE) == [(7.25, "trigger")]
    assert feed_boxes(g, 9.25, 12, faces=1, box=SEATED_USER) == [(10.75, "clear")]
    assert not g.owner_missing


def test_user_moving_while_the_onlooker_is_there_stays_the_owner() -> None:
    g = guard(delay_s=1.0, clear_s=1.5)
    feed_boxes(g, 0, 2, faces=1, box=OWNER)
    shifted = (0.30, 0.22, 0.3, 0.4)  # overlaps OWNER: followed
    feed_boxes(g, 2.25, 4, faces=2, box=shifted)
    assert g.active
    assert feed_boxes(g, 4.25, 6, faces=1, box=shifted) == [(5.75, "clear")]


def test_largest_face_is_the_owner_only_while_none_is_known() -> None:
    # Switched on with two faces in view, the larger one the colleague's: the
    # guard cannot know better, so the user alone then counts as a stranger...
    g = guard(delay_s=1.0, clear_s=1.5)
    feed_boxes(g, 0, 2, faces=2, box=LEANING_COLLEAGUE)
    assert g.active
    assert feed_boxes(g, 2.25, 10, faces=1, box=SEATED_USER) == []
    assert g.owner_missing
    # ...until they use the computer: then their face is the owner's.
    typed = 10.0
    assert feed_boxes(g, 10.25, 12, faces=1, box=SEATED_USER, last_input=typed) == [
        (11.75, "clear")
    ]
    assert not g.owner_missing
    # And stays so: the colleague leaning in again does not take over.
    assert feed_boxes(g, 12.25, 14, faces=2, box=LEANING_COLLEAGUE) == [(13.25, "trigger")]
    assert feed_boxes(g, 14.25, 16, faces=1, box=SEATED_USER) == [(15.75, "clear")]


def test_input_right_after_the_user_left_does_not_adopt_the_onlooker() -> None:
    g = guard(delay_s=1.0, clear_s=1.5)
    feed_boxes(g, 0, 1, faces=2, box=OWNER)
    # The user's face has left the picture; their hand is still on the mouse.
    feed_boxes(g, 1.25, 1.25, faces=1, box=ONLOOKER)
    last_touch = 1.25 + INPUT_ADOPT_DELAY_S - 0.5
    assert feed_boxes(g, 1.5, 20, faces=1, box=ONLOOKER, last_input=last_touch) == []
    assert g.owner_missing


def test_input_while_no_face_is_in_view_releases_the_guard() -> None:
    g = guard(delay_s=1.0, clear_s=1.5)
    feed_boxes(g, 0, 1, faces=2, box=OWNER)
    feed_boxes(g, 1.25, 5, faces=1, box=ONLOOKER)
    assert feed_boxes(g, 5.25, 10, faces=0, box=None) == []
    # Someone works at the computer, but the camera sees no face (looking down).
    assert feed_boxes(g, 10.25, 12, faces=0, box=None, last_input=10.0) == [(11.75, "clear")]
    assert not g.owner_missing


def test_guard_clears_when_the_onlooker_leaves_and_the_user_stays() -> None:
    g = guard(delay_s=1.0, clear_s=1.5)
    feed_boxes(g, 0, 1, faces=2, box=OWNER)
    assert g.active
    # The user shifts in the chair while the onlooker walks off.
    assert feed_boxes(g, 1.25, 5, faces=1, box=OWNER_MOVED) == [(2.75, "clear")]
    assert not g.owner_missing


def test_user_returning_next_to_the_onlooker_is_not_owner_missing() -> None:
    g = guard(delay_s=1.0)
    feed_boxes(g, 0, 1, faces=2, box=OWNER)
    feed_boxes(g, 1.25, 3, faces=1, box=ONLOOKER)
    assert g.owner_missing
    feed_boxes(g, 3.25, 4, faces=2, box=OWNER)
    assert g.active
    assert not g.owner_missing


def test_without_face_boxes_the_guard_clears_as_before() -> None:
    g = guard(delay_s=1.0, clear_s=1.5)
    feed_boxes(g, 0, 1, faces=2, box=None)
    assert feed_boxes(g, 1.25, 5, faces=1, box=ONLOOKER) == [(2.75, "clear")]


def test_reset_forgets_the_owner() -> None:
    g = guard(delay_s=0.0)
    g.update(0.0, 2, OWNER)
    g.update(0.5, 1, ONLOOKER)
    assert g.owner_missing
    g.reset()
    assert not g.owner_missing
    assert not g.active


def test_config_from_settings() -> None:
    s = PrivacySettings(shoulder_guard=True, guard_delay_s=4.0)
    cfg = GuardConfig.from_settings(s)
    assert cfg.enabled
    assert cfg.delay_s == 4.0
    assert cfg.clear_s == GuardConfig().clear_s
    assert not GuardConfig.from_settings(PrivacySettings()).enabled
