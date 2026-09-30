"""Tests for eye_tracker.gaze.filters (One Euro filter)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eye_tracker.gaze.filters import OneEuroFilter, PointFilter


def run(filt: OneEuroFilter, values: np.ndarray, fps: float, t0: float = 0.0) -> np.ndarray:
    return np.array([filt(v, t0 + i / fps) for i, v in enumerate(values)])


# --------------------------------------------------------------------------- OneEuroFilter
def test_first_sample_passes_through_and_constant_stays_constant() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    assert f.value is None
    assert f(42.0, 10.0) == 42.0
    out = run(f, np.full(30, 42.0), fps=30, t0=10.1)
    assert np.allclose(out, 42.0)
    assert f.value == pytest.approx(42.0)


def test_jitter_is_reduced() -> None:
    rng = np.random.default_rng(0)
    noisy = 500.0 + rng.normal(0, 20.0, 300)
    out = run(OneEuroFilter(min_cutoff=1.0, beta=0.0), noisy, fps=30)
    assert np.std(out[30:]) < np.std(noisy) / 3


def test_beta_speeds_up_response_to_fast_moves() -> None:
    step = np.r_[np.zeros(10), np.full(20, 1000.0)]
    slow = run(OneEuroFilter(min_cutoff=0.5, beta=0.0), step, fps=15)
    fast = run(OneEuroFilter(min_cutoff=0.5, beta=0.01), step, fps=15)
    # Three frames after the jump the adaptive filter is much closer to the target.
    assert fast[12] > 900.0
    assert slow[12] < 700.0


def test_filter_is_time_aware() -> None:
    # The same step sampled at 5 fps and 30 fps must be at a similar place after 1 s.
    def after_one_second(fps: float) -> float:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        f(0.0, 0.0)
        value = 0.0
        for i in range(1, int(fps) + 1):
            value = f(100.0, i / fps)
        return value

    low, high = after_one_second(5), after_one_second(30)
    assert abs(low - high) < 10.0
    assert 50.0 < low < 100.0


def test_non_advancing_timestamp_is_ignored() -> None:
    f = OneEuroFilter(min_cutoff=1.0)
    f(0.0, 1.0)
    moved = f(100.0, 1.1)
    assert f(5000.0, 1.1) == moved  # duplicate timestamp
    assert f(5000.0, 1.0) == moved  # going back in time
    assert f(100.0, 1.2) > moved  # continues normally afterwards


def test_non_finite_input_holds_last_value() -> None:
    f = OneEuroFilter()
    assert math.isnan(f(math.nan, 0.0))  # nothing to hold yet
    f(10.0, 0.1)
    assert f(math.nan, 0.2) == 10.0
    assert f(math.inf, 0.3) == 10.0
    assert f(10.0, 0.4) == pytest.approx(10.0)


def test_reset_forgets_state() -> None:
    f = OneEuroFilter()
    f(0.0, 0.0)
    f(10.0, 0.1)
    f.reset()
    assert f.value is None
    assert f(77.0, 0.2) == 77.0


@pytest.mark.parametrize("kwargs", [{"min_cutoff": 0.0}, {"d_cutoff": -1.0}, {"beta": -0.1}])
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="min_cutoff and d_cutoff"):
        OneEuroFilter(**kwargs)


# --------------------------------------------------------------------------- PointFilter
def test_zero_smoothing_is_passthrough() -> None:
    pf = PointFilter(0.0)
    rng = np.random.default_rng(1)
    for i, (x, y) in enumerate(rng.uniform(0, 3840, (50, 2))):
        assert pf.update(x, y, i * 0.05) == (x, y)


def test_more_smoothing_means_less_jitter() -> None:
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 40.0, (200, 2)) + np.array([960.0, 540.0])

    def jitter(smoothing: float) -> float:
        pf = PointFilter(smoothing)
        out = np.array([pf.update(x, y, i / 12) for i, (x, y) in enumerate(noise)])
        return float(np.std(out[20:, 0]))

    raw = float(np.std(noise[20:, 0]))
    light, medium, heavy = jitter(0.1), jitter(0.5), jitter(1.0)
    assert raw > light > medium > heavy
    assert heavy < raw / 2


def test_monitor_jump_is_followed_quickly_at_default_smoothing() -> None:
    # A glance from the left to the right monitor at the balanced active rate (12 fps).
    pf = PointFilter(0.5)
    t = 0.0
    for _ in range(12):
        pf.update(960.0, 540.0, t)
        t += 1 / 12
    xs = []
    for _ in range(4):
        xs.append(pf.update(2880.0, 540.0, t)[0])
        t += 1 / 12
    assert xs[2] > 1920.0 + 0.8 * 960.0  # well inside the right monitor within 3 frames


def test_set_smoothing_clamps_and_keeps_state() -> None:
    pf = PointFilter(0.5)
    pf.update(100.0, 100.0, 0.0)
    pf.set_smoothing(7.0)
    assert pf.smoothing == 1.0
    pf.set_smoothing(-3.0)
    assert pf.smoothing == 0.0
    pf.set_smoothing(math.nan)
    assert pf.smoothing == 0.0
    pf.set_smoothing(0.8)
    x, _ = pf.update(110.0, 100.0, 0.1)
    assert 100.0 < x < 110.0  # continued from the earlier position, not from scratch


def test_point_filter_reset() -> None:
    pf = PointFilter(1.0)
    pf.update(0.0, 0.0, 0.0)
    pf.update(1000.0, 1000.0, 0.1)
    pf.reset()
    assert pf.update(1000.0, 1000.0, 0.2) == (1000.0, 1000.0)
