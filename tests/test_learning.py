"""Tests for eye_tracker.gaze.learning (implicit samples, refit, drift detection)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from eye_tracker.gaze.calibration import CalibrationSample, evaluate
from eye_tracker.gaze.learning import (
    IMPLICIT_MAX_ERROR,
    DriftMonitor,
    ImplicitLearner,
    plausible_label,
    refit_model,
)
from eye_tracker.gaze.model import GazeModel
from eye_tracker.types import Monitor, Observation, Rect, monitor_at, nearest_monitor
from gaze_synth import GAZE, TWO, calibration_samples, random_points, synth_features

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


def test_discard_last_takes_the_sample_back() -> None:
    learner = ImplicitLearner(refit_every=2)
    first = move_and_settle(learner, 100.0, 100.0, 1.0)
    second = move_and_settle(learner, 200.0, 100.0, 3.0)
    assert learner.should_refit()
    assert learner.discard_last() is second
    assert learner.samples == [first]
    assert learner.new_since_refit == 1
    assert not learner.should_refit()
    assert learner.discard_last() is None  # only the latest can be taken back
    assert learner.samples == [first]
    third = move_and_settle(learner, 300.0, 100.0, 5.0)
    assert third is not None
    assert third.point_id == -3  # ids are never reused


def test_discard_last_restores_the_sample_evicted_for_it() -> None:
    learner = ImplicitLearner(max_samples=2)
    t = 0.0
    for x in (100.0, 200.0, 2500.0):
        move_and_settle(learner, x, 100.0, t)
        t += 2.0
    assert [s.x for s in learner.samples] == [200.0, 2500.0]  # 100 made room
    learner.discard_last()
    assert [s.x for s in learner.samples] == [100.0, 200.0]


def test_discard_last_after_refit_or_reset() -> None:
    learner = ImplicitLearner(refit_every=1)
    sample = move_and_settle(learner, 100.0, 100.0, 1.0)
    learner.mark_refit()
    assert learner.discard_last() is sample  # still leaves the training set …
    assert learner.new_since_refit == 0  # … but was already accounted for
    for reset in (
        lambda: learner.load([]),
        learner.clear,
        lambda: learner.set_max_samples(0),
    ):
        learner.set_max_samples(400)
        move_and_settle(learner, 100.0, 100.0, 10.0)
        reset()
        assert learner.discard_last() is None


def test_set_max_samples_reports_dropped_samples() -> None:
    learner = ImplicitLearner(max_samples=5)
    for i in range(3):
        move_and_settle(learner, 100.0 + i, 100.0, 2.0 * i)
    assert not learner.set_max_samples(3)
    assert not learner.set_max_samples(10)
    assert learner.set_max_samples(1)
    assert len(learner.samples) == 1
    assert learner.set_max_samples(0)
    assert learner.samples == []
    assert not learner.set_max_samples(0)


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


def test_refit_without_learned_samples_restores_the_calibration_fit() -> None:
    # Lowering "max learned samples" (to 0) must be able to undo their influence.
    base = calibration_samples(TWO, np.random.default_rng(60), noise=1.0, per_point=8)
    calibrated, _ = evaluate(base, TWO, nonlinear=GAZE)
    rng = np.random.default_rng(61)
    wrong = [
        CalibrationSample(f, 3500.0, 900.0, 1, -1 - i, 0.5)
        for i, f in enumerate(synth_features(random_points(TWO[:1], rng, 80), rng, 1.0))
    ]
    polluted = refit_model(base, wrong, calibrated)
    assert polluted.nonlinear == GAZE
    learner = ImplicitLearner()
    learner.load(wrong)
    assert learner.set_max_samples(0)
    restored = refit_model(base, learner.samples, polluted)
    X = np.array([s.features for s in base])
    assert not np.allclose(polluted.predict(X), calibrated.predict(X), atol=1.0)
    assert np.allclose(restored.predict(X), calibrated.predict(X), atol=1e-6)
    assert (restored.degree, restored.alpha, restored.nonlinear, restored.bounds) == (
        calibrated.degree,
        calibrated.alpha,
        calibrated.nonlinear,
        calibrated.bounds,
    )
    assert restored.away_regions == calibrated.away_regions


def test_refit_learns_the_look_away_regions_of_the_calibrated_monitors() -> None:
    base = calibration_samples(TWO, np.random.default_rng(62), noise=1.0, per_point=8)
    calibrated, _ = evaluate(base, TWO, nonlinear=GAZE)
    assert [r.rect for r in calibrated.away_regions] == [m.rect for m in TWO]
    # A model saved before the regions existed gains them with the next refit...
    data = calibrated.to_dict()
    del data["away_regions"]
    legacy = GazeModel.from_dict(data)
    assert refit_model(base, [], legacy).away_regions == ()
    upgraded = refit_model(base, [], legacy, monitors=TWO)
    assert upgraded.away_regions == calibrated.away_regions
    # ...and a refit without monitors keeps those of the template.
    assert refit_model(base, [], upgraded).away_regions == calibrated.away_regions


# --------------------------------------------------------------------------- plausible labels
def _sample(x: float, y: float, monitor: int) -> CalibrationSample:
    return CalibrationSample(np.zeros(8), x, y, monitor, -1, 0.5)


def test_plausible_label() -> None:
    diagonal = MONITORS[1].rect.diagonal
    label = _sample(2300.0, 900.0, 1)
    assert plausible_label((2400.0, 800.0), label, MONITORS)
    # On the other monitor but close to the label: a drifted model the label corrects.
    assert plausible_label((1800.0, 900.0), label, MONITORS)
    # The cursor was parked on the right while the user read the left monitor.
    assert not plausible_label((500.0, 400.0), label, MONITORS)
    limit = IMPLICIT_MAX_ERROR * diagonal
    assert plausible_label((2300.0 - limit + 1, 900.0), label, MONITORS)
    assert not plausible_label((2300.0 - limit - 1, 900.0), label, MONITORS)
    assert plausible_label((500.0, 400.0), label, MONITORS, max_error=2.0)
    # No evidence either way.
    assert plausible_label((math.nan, 400.0), label, MONITORS)
    assert plausible_label((500.0, 400.0), label, [])
    # An index that is not in the list: the monitor under the cursor decides.
    small = [Monitor(0, "a", Rect(0, 0, 800, 600)), Monitor(1, "b", Rect(800, 0, 800, 600))]
    assert not plausible_label((700.0, 300.0), _sample(1500.0, 300.0, 9), small)


def test_gated_learning_resists_parked_cursor_labels() -> None:
    """Regression: every settle used to be learned, so a habit such as clicking
    "Run" on the right monitor and reading the output on the left one biased
    the model towards the right."""
    base = calibration_samples(TWO, np.random.default_rng(70), noise=1.0, per_point=12)
    model, _ = evaluate(base, TWO, nonlinear=GAZE)
    rng = np.random.default_rng(71)

    def learn(gate: bool) -> GazeModel:
        learner = ImplicitLearner(max_samples=400, refit_every=10_000)
        t = 0.0
        for i in range(400):
            gaze = random_points(TWO, rng, 1)[0]
            parked = i % 5 == 0  # 20 % of the settles
            if parked:
                gaze = random_points(TWO[:1], rng, 1)[0]  # reading the left monitor
            cursor = (3500.0, 900.0) if parked else tuple(gaze)
            features = synth_features(gaze[None, :], rng, 1.0)[0]
            learner.on_manual_cursor(*cursor, t)
            sample = learner.on_observation(obs(t + 0.4, features), t + 0.4, lookup)
            assert sample is not None
            if gate and not plausible_label(model.predict(sample.features), sample, TWO):
                assert learner.discard_last() is sample
            t += 2.0
        return refit_model(base, learner.samples, model)

    points = random_points(TWO, np.random.default_rng(72), 1500)
    X = synth_features(points, np.random.default_rng(73), 1.0)

    def accuracy(m: GazeModel) -> float:
        predicted = m.predict(X)
        return float(
            np.mean(
                [
                    nearest_monitor(TWO, *p)[0].index == nearest_monitor(TWO, *q)[0].index
                    for p, q in zip(predicted, points, strict=True)
                ]
            )
        )

    ungated, gated = accuracy(learn(gate=False)), accuracy(learn(gate=True))
    assert gated >= 0.99
    assert gated >= accuracy(model) - 0.005
    assert ungated < gated - 0.02


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


def test_switch_outcomes_count_both_ways() -> None:
    # With adaptive learning off only switches produce events. 5 % of 300 switches
    # undone is an accurate model, not a reason to alert.
    drift = DriftMonitor()
    alerts = 0
    for i in range(300):
        drift.record_switch(i % 20 != 0)
        alerts += drift.should_alert(float(i * 600))
    assert alerts == 0
    assert drift.error_rate == pytest.approx(0.05, abs=0.03)
    drift.record_switch(False)
    drift.record_wrong_switch()
    assert drift.event_count == 40


def test_mostly_undone_switches_still_alert() -> None:
    drift = DriftMonitor(window=20, alert_ratio=0.35, min_events=15)
    for i in range(20):
        drift.record_switch(i % 2 == 0)
    assert drift.error_rate == 0.5
    assert drift.should_alert(0.0)


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
