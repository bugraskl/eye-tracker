"""Tests for eye_tracker.gaze.learning (implicit samples, refit, drift detection)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from eye_tracker.gaze.calibration import CalibrationSample
from eye_tracker.gaze.learning import DriftMonitor, ImplicitLearner, refit_model
from eye_tracker.gaze.model import GazeModel
from eye_tracker.types import Monitor, Observation, Rect, monitor_at

MONITORS = [
    Monitor(0, "left", Rect(0, 0, 1920, 1080)),
    Monitor(1, "right", Rect(1920, 0, 1920, 1080)),
]


def lookup(x: float, y: float) -> int | None:
    m = monitor_at(MONITORS, x, y)
    return None if m is None else m.index


def obs(t: float = 0.0, features: np.ndarray | None = None, **kwargs: object) -> Observation:
    values = {"quality": 1.0, **kwargs}
    return Observation(
        timestamp=t,
        face_count=1,
        features=np.arange(8.0) if features is None else features,
        **values,  # type: ignore[arg-type]
    )


def move_and_settle(
    learner: ImplicitLearner, x: float, y: float, t: float
) -> CalibrationSample | None:
    """A short drag ending at (x, y) at time t, then an observation 0.4 s later."""
    learner.on_manual_cursor(x - 30, y - 10, t - 0.1)
    learner.on_manual_cursor(x, y, t)
    return learner.on_observation(obs(t + 0.4), t + 0.4, lookup)


# --------------------------------------------------------------------------- ImplicitLearner
def test_no_sample_without_a_manual_move() -> None:
    learner = ImplicitLearner()
    assert learner.on_observation(obs(), 10.0, lookup) is None
    assert learner.samples == []


def test_one_sample_per_settle() -> None:
    learner = ImplicitLearner(weight=0.5, settle_s=0.35)
    learner.on_manual_cursor(100.0, 100.0, 0.0)
    learner.on_manual_cursor(2500.0, 300.0, 0.2)
    assert learner.on_observation(obs(), 0.3, lookup) is None  # not settled yet
    sample = learner.on_observation(obs(features=np.full(8, 3.0)), 0.6, lookup)
    assert sample is not None
    assert (sample.x, sample.y, sample.monitor_index) == (2500.0, 300.0, 1)
    assert sample.weight == 0.5
    assert sample.point_id == -1
    assert np.all(sample.features == 3.0)
    # Further observations of the same settle are duplicates.
    assert learner.on_observation(obs(), 0.7, lookup) is None
    assert learner.on_observation(obs(), 1.0, lookup) is None
    assert len(learner.samples) == 1

    # A new move starts a new settle with its own (negative) point id.
    second = move_and_settle(learner, 500.0, 500.0, 3.0)
    assert second is not None
    assert second.point_id == -2
    assert second.monitor_index == 0
    assert len(learner.samples) == 2


def test_settle_expires_after_recent_window() -> None:
    learner = ImplicitLearner(settle_s=0.35)
    learner.on_manual_cursor(100.0, 100.0, 0.0)
    assert learner.on_observation(obs(), 1.6, lookup) is None  # > 1.5 s since the move
    assert learner.on_observation(obs(), 1.7, lookup) is None
    assert learner.samples == []


def test_waits_for_a_usable_observation_within_the_window() -> None:
    learner = ImplicitLearner()
    learner.on_manual_cursor(100.0, 100.0, 0.0)
    assert learner.on_observation(obs(blink=True), 0.4, lookup) is None
    assert learner.on_observation(obs(quality=0.0), 0.5, lookup) is None
    assert learner.on_observation(obs(skipped=True), 0.6, lookup) is None
    assert learner.on_observation(Observation(0.7, face_count=0), 0.7, lookup) is None
    assert learner.on_observation(obs(features=np.array([np.inf] * 8)), 0.8, lookup) is None
    assert learner.on_observation(obs(), 0.9, lookup) is not None


def test_cursor_outside_every_monitor_gives_no_sample() -> None:
    learner = ImplicitLearner()
    learner.on_manual_cursor(-500.0, -500.0, 0.0)
    assert learner.on_observation(obs(), 0.5, lookup) is None
    assert learner.on_observation(obs(), 0.6, lookup) is None  # settle consumed
    assert learner.samples == []


def test_moving_again_postpones_the_sample() -> None:
    learner = ImplicitLearner(settle_s=0.35)
    learner.on_manual_cursor(100.0, 100.0, 0.0)
    learner.on_manual_cursor(150.0, 100.0, 0.3)
    assert learner.on_observation(obs(), 0.5, lookup) is None  # only 0.2 s since last move
    sample = learner.on_observation(obs(), 0.7, lookup)
    assert sample is not None
    assert sample.x == 150.0


def test_capacity_is_shared_fairly_between_monitors() -> None:
    learner = ImplicitLearner(max_samples=4)
    t = 0.0
    for x in (100.0, 2000.0, 200.0, 300.0, 400.0, 500.0):  # mostly the left monitor
        assert move_and_settle(learner, x, 100.0, t) is not None
        t += 2.0
    xs = [s.x for s in learner.samples]
    assert len(xs) == 4
    assert 2000.0 in xs  # the lone right-monitor sample survives
    assert xs == [2000.0, 300.0, 400.0, 500.0]  # the oldest left samples went first


def test_zero_capacity_disables_learning() -> None:
    learner = ImplicitLearner(max_samples=0)
    assert move_and_settle(learner, 100.0, 100.0, 1.0) is None
    assert learner.samples == []


def test_refit_schedule() -> None:
    learner = ImplicitLearner(refit_every=3)
    t = 0.0
    for i in range(2):
        move_and_settle(learner, 100.0 + i, 100.0, t)
        t += 2.0
    assert not learner.should_refit()
    move_and_settle(learner, 300.0, 100.0, t)
    assert learner.should_refit()
    assert learner.new_since_refit == 3
    learner.mark_refit()
    assert not learner.should_refit()
    assert learner.new_since_refit == 0


def test_load_clear_and_ids_continue() -> None:
    learner = ImplicitLearner(max_samples=3)
    saved = [CalibrationSample(np.zeros(8), 10.0 * i, 10.0, 0, -(i + 1), 0.5) for i in range(5)]
    learner.load(saved)
    assert [s.point_id for s in learner.samples] == [-3, -4, -5]  # newest kept
    assert not learner.should_refit()
    sample = move_and_settle(learner, 2500.0, 100.0, 1.0)
    assert sample is not None
    assert sample.point_id == -6
    learner.set_max_samples(1)
    assert [s.point_id for s in learner.samples] == [-6]
    learner.clear()
    assert learner.samples == []
    sample = move_and_settle(learner, 100.0, 100.0, 5.0)
    assert sample is not None
    assert sample.point_id == -1


def test_feature_length_change_discards_old_samples(caplog: pytest.LogCaptureFixture) -> None:
    learner = ImplicitLearner()
    learner.load([CalibrationSample(np.zeros(6), 1.0, 1.0, 0, -1, 0.5)])
    with caplog.at_level(logging.WARNING, logger="eye_tracker.gaze.learning"):
        assert move_and_settle(learner, 100.0, 100.0, 1.0) is not None
    assert len(learner.samples) == 1
    assert learner.samples[0].features.shape == (8,)
    assert "discarding" in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [{"max_samples": -1}, {"weight": 0.0}, {"settle_s": -0.1}, {"refit_every": 0}],
)
def test_invalid_learner_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="max_samples"):
        ImplicitLearner(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_samples"):
        ImplicitLearner().set_max_samples(-5)


# --------------------------------------------------------------------------- refit_model
def test_refit_model_uses_template_settings_and_weights() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    Y = X @ np.array([[500.0, 0.0], [0.0, 300.0], [50.0, 50.0]]) + (1900.0, 540.0)
    base = [CalibrationSample(x, *y, 0, i // 4) for i, (x, y) in enumerate(zip(X, Y, strict=True))]
    template = GazeModel(degree=1, alpha=0.01).fit(X, Y, bounds=Rect(0, 0, 3840, 1080))

    # Learned samples that contradict the calibration pull the fit, but weakly.
    wrong = [CalibrationSample(x, 3500.0, 900.0, 1, -1 - i, 0.01) for i, x in enumerate(X[:10])]
    model = refit_model(base, wrong, template)
    assert (model.degree, model.alpha) == (1, 0.01)
    assert model.bounds == Rect(0, 0, 3840, 1080)
    err = np.hypot(*(model.predict(X) - Y).T)
    assert np.median(err) < 50.0

    heavy = [CalibrationSample(s.features, s.x, s.y, 1, s.point_id, 50.0) for s in wrong]
    pulled = refit_model(base, heavy, template, bounds=Rect(0, 0, 4000, 1200))
    assert pulled.bounds == Rect(0, 0, 4000, 1200)
    assert np.median(np.hypot(*(pulled.predict(X[:10]) - (3500.0, 900.0)).T)) < np.median(
        np.hypot(*(model.predict(X[:10]) - (3500.0, 900.0)).T)
    )


# --------------------------------------------------------------------------- DriftMonitor
def test_drift_error_rate_and_min_events() -> None:
    drift = DriftMonitor(window=40, alert_ratio=0.35, min_events=15)
    assert drift.error_rate == 0.0
    for _ in range(10):
        drift.record(0, 1)  # every prediction wrong
    assert drift.error_rate == 1.0
    assert not drift.should_alert(100.0)  # fewer than min_events
    drift.record(None, 1)  # no prediction: ignored
    assert drift.event_count == 10
    for _ in range(5):
        drift.record(1, 1)
    assert drift.error_rate == pytest.approx(10 / 15)
    assert drift.should_alert(100.0)


def test_drift_does_not_alert_when_accurate() -> None:
    drift = DriftMonitor(min_events=15)
    for i in range(40):
        drift.record(0, 1 if i % 5 == 0 else 0)  # 20 % disagreement
    assert drift.error_rate == pytest.approx(0.2)
    assert not drift.should_alert(0.0)


def test_drift_cooldown() -> None:
    drift = DriftMonitor(window=20, alert_ratio=0.5, min_events=5)
    for _ in range(5):
        drift.record_wrong_switch()
    assert drift.should_alert(1000.0)
    assert not drift.should_alert(1001.0)
    assert not drift.should_alert(1000.0 + 1799.0)
    assert drift.should_alert(1000.0 + 1800.0)
    assert not drift.should_alert(3000.0, cooldown_s=300.0)
    assert drift.should_alert(3100.0, cooldown_s=300.0)


def test_drift_window_rolls_over() -> None:
    drift = DriftMonitor(window=10, alert_ratio=0.5, min_events=5)
    for _ in range(10):
        drift.record(0, 1)
    for _ in range(10):
        drift.record(1, 1)
    assert drift.event_count == 10
    assert drift.error_rate == 0.0
    assert not drift.should_alert(0.0)


def test_drift_reset_clears_events_and_cooldown() -> None:
    drift = DriftMonitor(min_events=3, alert_ratio=0.5)
    for _ in range(3):
        drift.record(0, 1)
    assert drift.should_alert(10.0)
    drift.reset()
    assert drift.event_count == 0
    for _ in range(3):
        drift.record(0, 1)
    assert drift.should_alert(11.0)  # cooldown was reset too


@pytest.mark.parametrize(
    "kwargs", [{"window": 0}, {"alert_ratio": 0.0}, {"alert_ratio": 1.5}, {"min_events": 0}]
)
def test_invalid_drift_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="window"):
        DriftMonitor(**kwargs)  # type: ignore[arg-type]
