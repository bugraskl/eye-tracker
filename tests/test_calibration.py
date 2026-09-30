"""Tests for eye_tracker.gaze.calibration (plan, collector state machine, evaluation)."""

from __future__ import annotations

import itertools
import logging
import math

import numpy as np
import pytest

from eye_tracker.gaze.calibration import (
    CalibrationCollector,
    CalibrationReport,
    CalibrationSample,
    CalibrationTarget,
    evaluate,
    grade_for,
    make_plan,
    samples_to_arrays,
)
from eye_tracker.types import Monitor, Observation, Rect

# --------------------------------------------------------------------------- layouts
TWO = [
    Monitor(0, "left", Rect(0, 0, 1920, 1080), primary=True),
    Monitor(1, "right", Rect(1920, 0, 1920, 1080)),
]
THREE = [Monitor(i, f"m{i}", Rect(1920 * (i - 1), 0, 1920, 1080)) for i in range(3)]
STACKED = [
    Monitor(0, "top", Rect(0, -1080, 1920, 1080)),
    Monitor(1, "bottom", Rect(0, 0, 1920, 1080)),
]
# Three side by side plus a fourth stacked above the middle one.
THREE_PLUS_TOP = [
    Monitor(0, "centre", Rect(0, 0, 2560, 1440), primary=True),
    Monitor(1, "left", Rect(-1920, 180, 1920, 1080)),
    Monitor(2, "right", Rect(2560, 180, 1920, 1080)),
    Monitor(3, "top", Rect(320, -1080, 1920, 1080)),
]

# --------------------------------------------------------------------------- synthetic data
PX_CM = 53.0 / 1920  # 24" 1080p panel
NOISE_SCALE = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 0.01, 0.006])


def synth_features(
    points: np.ndarray, rng: np.random.Generator, noise: float = 0.0, camera_x: float = 1920.0
) -> np.ndarray:
    """(yaw, pitch, roll, tx, ty, tz, iris_h, iris_v) of a user ~65 cm away looking at points.

    The head turns part of the way towards the target (a random share, as people
    do) and the eyes cover the rest; flat screens make the mapping tan-shaped.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    target_x = (pts[:, 0] - camera_x) * PX_CM
    target_y = pts[:, 1] * PX_CM + 2.0
    hx, hy, hz = rng.normal(0, 2.5, n), rng.normal(0, 1.5, n), 65.0 + rng.normal(0, 3.0, n)
    gaze_yaw = np.degrees(np.arctan2(target_x - hx, hz))
    gaze_pitch = np.degrees(np.arctan2(target_y - hy, hz))
    head_share = np.clip(0.55 + rng.normal(0, 0.12, n), 0.1, 0.95)
    yaw = head_share * gaze_yaw + rng.normal(0, 2.0, n)
    pitch = 0.8 * head_share * gaze_pitch + rng.normal(0, 1.5, n)
    iris_h = 0.5 + 0.42 * np.sin(np.radians(gaze_yaw - yaw))
    iris_v = 0.05 + 0.20 * np.sin(np.radians(gaze_pitch - pitch))
    roll = rng.normal(0, 1.5, n)
    feats = np.column_stack([yaw, pitch, roll, hx, hy, -hz, iris_h, iris_v])
    return feats + rng.normal(size=feats.shape) * NOISE_SCALE * noise


def calibration_samples(
    monitors: list[Monitor],
    rng: np.random.Generator,
    noise: float = 0.0,
    *,
    per_point: int = 20,
    camera_x: float = 1920.0,
    points_per_monitor: int = 9,
) -> list[CalibrationSample]:
    samples = []
    for t in make_plan(monitors, points_per_monitor):
        feats = synth_features(np.tile((t.x, t.y), (per_point, 1)), rng, noise, camera_x)
        samples += [CalibrationSample(f, t.x, t.y, t.monitor_index, t.point_id) for f in feats]
    return samples


def obs(t: float = 0.0, features: np.ndarray | None = None, **kwargs: object) -> Observation:
    values = {"quality": 1.0, **kwargs}
    return Observation(
        timestamp=t,
        face_count=1,
        features=np.arange(8.0) if features is None else features,
        **values,  # type: ignore[arg-type]
    )


def nxy(plan: list[CalibrationTarget]) -> list[tuple[float, float]]:
    return [(round(t.nx, 3), round(t.ny, 3)) for t in plan]


# --------------------------------------------------------------------------- make_plan
def test_plan_nine_points_two_monitors_left_to_right_serpentine() -> None:
    plan = make_plan(list(reversed(TWO)))  # input order must not matter
    assert [t.point_id for t in plan] == list(range(18))
    assert [t.monitor_index for t in plan] == [0] * 9 + [1] * 9
    first = [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1), (0.9, 0.5), (0.5, 0.5), (0.1, 0.5),
             (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)]  # fmt: skip
    # The right monitor starts at its bottom-left corner, next to where the left one ended.
    second = [(0.1, 0.9), (0.5, 0.9), (0.9, 0.9), (0.9, 0.5), (0.5, 0.5), (0.1, 0.5),
              (0.1, 0.1), (0.5, 0.1), (0.9, 0.1)]  # fmt: skip
    assert nxy(plan[:9]) == first
    assert nxy(plan[9:]) == second
    assert (plan[0].x, plan[0].y) == pytest.approx((192.0, 108.0))
    assert (plan[9].x, plan[9].y) == pytest.approx((1920 + 192.0, 972.0))


def test_plan_stacked_monitors_top_first() -> None:
    plan = make_plan(STACKED)
    assert [t.monitor_index for t in plan] == [0] * 9 + [1] * 9
    # The top monitor ends bottom-right, so the bottom one starts top-right.
    assert nxy(plan[9:12]) == [(0.9, 0.1), (0.5, 0.1), (0.1, 0.1)]
    assert plan[9].y == pytest.approx(108.0)


def test_plan_three_plus_stacked_layout() -> None:
    plan = make_plan(THREE_PLUS_TOP)
    order = []
    for t in plan:
        if not order or order[-1] != t.monitor_index:
            order.append(t.monitor_index)
    assert order == [1, 0, 3, 2]  # by x, then y: left, centre, top (x=320), right
    assert [t.point_id for t in plan] == list(range(36))
    for t in plan:  # global coordinates match the monitor-relative ones
        rect = next(m.rect for m in THREE_PLUS_TOP if m.index == t.monitor_index)
        assert (t.x, t.y) == pytest.approx(rect.denormalize(t.nx, t.ny))
    # Within a monitor every step moves along one axis only (serpentine, no diagonals).
    for a, b in itertools.pairwise(plan):
        if a.monitor_index == b.monitor_index:
            assert a.nx == b.nx or a.ny == b.ny


def test_plan_orientation_shortens_total_travel() -> None:
    plan = make_plan(THREE)
    travel = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in itertools.pairwise(plan))
    naive = []
    for m in sorted(THREE, key=lambda m: m.rect.x):
        naive += [m.rect.denormalize(t.nx, t.ny) for t in make_plan([m])]
    naive_travel = sum(math.dist(a, b) for a, b in itertools.pairwise(naive))
    assert travel < naive_travel


def test_plan_other_sizes() -> None:
    five = make_plan(TWO[:1], points_per_monitor=5)
    assert nxy(five) == [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    assert nxy(make_plan(TWO[:1], points_per_monitor=1)) == [(0.5, 0.5)]
    sixteen = make_plan(TWO[:1], points_per_monitor=16, margin=0.05)
    assert len(sixteen) == 16
    assert nxy(sixteen)[:5] == [
        (0.05, 0.05),
        (0.35, 0.05),
        (0.65, 0.05),
        (0.95, 0.05),
        (0.95, 0.35),
    ]
    assert make_plan([], 9) == []


@pytest.mark.parametrize(("points", "margin"), [(0, 0.1), (3, 0.1), (7, 0.1), (9, 0.5), (9, -0.1)])
def test_plan_rejects_invalid_arguments(points: int, margin: float) -> None:
    with pytest.raises(ValueError, match=r"margin|points_per_monitor"):
        make_plan(TWO, points, margin)


# --------------------------------------------------------------------------- collector
def test_collector_happy_path() -> None:
    plan = make_plan(TWO)[:2]
    c = CalibrationCollector(plan, settle_s=0.8, collect_s=1.0, min_samples=5)
    assert (c.phase, c.current, c.progress) == ("idle", None, 0.0)
    assert c.update(0.0) == []
    assert not c.add(obs())

    c.start(10.0)
    assert c.phase == "settle"
    assert c.current == plan[0]
    assert c.update(10.0) == ["target"]
    assert not c.add(obs()), "nothing is recorded while the eyes settle"

    assert c.update(10.4) == []
    assert c.point_progress == pytest.approx(0.4 / 1.8)
    assert c.phase_progress == pytest.approx(0.5)
    assert c.update(10.8) == []
    assert c.phase == "collect"
    assert all(c.add(obs(10.8 + i * 0.1)) for i in range(6))
    assert c.current_sample_count == 6
    assert c.update(11.3) == []
    assert c.point_progress == pytest.approx(1.3 / 1.8)
    assert c.progress == pytest.approx((0 + 1.3 / 1.8) / 2)

    assert c.update(11.8) == ["target"]
    assert c.current == plan[1]
    assert c.phase == "settle"
    assert c.current_index == 1
    assert len(c.samples) == 6
    s = c.samples[0]
    assert (s.x, s.y, s.monitor_index, s.point_id, s.weight) == (
        plan[0].x,
        plan[0].y,
        plan[0].monitor_index,
        0,
        1.0,
    )

    c.update(12.6)
    assert c.phase == "collect"
    for i in range(5):
        c.add(obs(12.7 + i * 0.1))
    assert c.update(13.6) == ["finished"]
    assert c.phase == "done"
    assert c.current is None
    assert c.progress == 1.0
    assert len(c.samples) == 11
    assert c.skipped_points == []
    assert c.update(20.0) == []


def test_collector_retry_keeps_samples_then_succeeds() -> None:
    plan = make_plan(TWO)[:1]
    c = CalibrationCollector(plan, settle_s=0.5, collect_s=1.0, min_samples=5, max_retries=1)
    c.start(0.0)
    c.update(0.5)
    for i in range(3):
        c.add(obs(0.6 + i * 0.1))
    assert c.update(1.5) == ["retry"]
    assert c.phase == "collect"
    assert c.current_sample_count == 3
    assert c.phase_progress == 0.0
    c.add(obs(1.6))
    c.add(obs(1.7))
    assert c.update(2.5) == ["finished"]
    assert len(c.samples) == 5
    assert c.skipped_points == []


def test_collector_skips_after_retries() -> None:
    plan = make_plan(TWO)[:2]
    c = CalibrationCollector(plan, settle_s=0.5, collect_s=1.0, min_samples=5, max_retries=1)
    c.start(0.0)
    c.update(0.5)
    c.add(obs())  # only one sample: not enough
    assert c.update(1.5) == ["retry"]
    assert c.update(2.5) == ["target"]
    assert c.skipped_points == [0]
    assert c.samples == []  # samples of a skipped point are dropped
    assert c.current == plan[1]


def test_collector_without_retries_skips_immediately() -> None:
    c = CalibrationCollector(make_plan(TWO)[:1], settle_s=0.0, collect_s=0.5, max_retries=0)
    c.start(0.0)
    assert c.update(0.0) == ["target"]
    assert c.update(0.5) == ["finished"]
    assert c.skipped_points == [0]


def test_zero_settle_goes_straight_to_collect() -> None:
    c = CalibrationCollector(make_plan(TWO)[:1], settle_s=0.0, collect_s=1.0)
    c.start(5.0)
    assert c.update(5.0) == ["target"]
    assert c.phase == "collect"
    assert c.add(obs())


def test_stalled_ui_does_not_eat_the_next_phase() -> None:
    c = CalibrationCollector(make_plan(TWO)[:2], settle_s=0.8, collect_s=1.0)
    c.start(0.0)
    c.update(0.0)
    # The UI thread freezes for 10 s during settle: collecting starts when it
    # resumes instead of the point being retried or skipped.
    assert c.update(10.0) == []
    assert c.phase == "collect"
    assert c.phase_progress == 0.0
    assert c.skipped_points == []


def test_collector_rejects_unusable_observations() -> None:
    c = CalibrationCollector(make_plan(TWO)[:1], settle_s=0.0)
    c.start(0.0)
    c.update(0.0)
    assert not c.add(obs(blink=True))
    assert not c.add(obs(quality=0.1))
    assert not c.add(obs(skipped=True))  # motion-gate copy of an older frame
    assert not c.add(Observation(timestamp=0.0, face_count=0))
    assert not c.add(obs(features=np.array([1.0, math.nan, 2.0])))
    assert c.add(obs(features=np.ones(8)))
    assert not c.add(obs(features=np.ones(6)))  # backend changed mid-calibration
    assert c.current_sample_count == 1


def test_collector_samples_copy_features() -> None:
    c = CalibrationCollector(make_plan(TWO)[:1], settle_s=0.0, min_samples=1)
    c.start(0.0)
    c.update(0.0)
    features = np.ones(8)
    c.add(obs(features=features))
    features[:] = 99.0  # the backend may reuse its buffer
    c.update(1.0)
    assert np.all(c.samples[0].features == 1.0)


def test_empty_plan_finishes_immediately() -> None:
    c = CalibrationCollector([])
    c.start(0.0)
    assert c.update(0.0) == ["finished"]
    assert c.phase == "done"
    assert c.progress == 1.0


def test_restart_resets_everything() -> None:
    plan = make_plan(TWO)[:1]
    c = CalibrationCollector(plan, settle_s=0.0, min_samples=1)
    c.start(0.0)
    c.update(0.0)
    c.add(obs())
    c.update(1.0)
    assert len(c.samples) == 1
    c.start(2.0)
    assert c.samples == []
    assert c.phase == "settle"
    assert c.update(2.0) == ["target"]


@pytest.mark.parametrize(
    "kwargs",
    [{"settle_s": -1.0}, {"collect_s": 0.0}, {"min_samples": 0}, {"max_retries": -1}],
)
def test_collector_rejects_invalid_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"settle_s|min_samples"):
        CalibrationCollector(make_plan(TWO), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- evaluate
def test_evaluate_grades_clean_data_excellent() -> None:
    rng = np.random.default_rng(100)
    samples = calibration_samples(TWO, rng, noise=0.0)
    model, report = evaluate(samples, TWO)
    assert report.grade == "excellent", report
    assert report.monitor_accuracy >= 0.97
    assert report.mean_error_px < 100.0, report
    assert report.median_error_px <= report.mean_error_px * 1.5
    assert (report.n_samples, report.n_points) == (360, 18)
    assert set(report.per_monitor_accuracy) == {0, 1}
    assert report.alpha > 0
    assert report.degree in (1, 2, 3)
    assert model.is_fitted
    assert (model.degree, model.alpha) == (report.degree, report.alpha)

    # The final model generalises to gaze points it never saw.
    rng2 = np.random.default_rng(101)
    points = np.column_stack([rng2.uniform(100, 3740, 300), rng2.uniform(60, 1020, 300)])
    pred = model.predict(synth_features(points, rng2))
    assert np.median(np.hypot(*(pred - points).T)) < 100.0


def test_evaluate_grades_noisy_data_lower() -> None:
    clean = evaluate(calibration_samples(TWO, np.random.default_rng(102), noise=1.0), TWO)[1]
    noisy = evaluate(calibration_samples(TWO, np.random.default_rng(102), noise=12.0), TWO)[1]
    assert noisy.monitor_accuracy < clean.monitor_accuracy
    assert noisy.mean_error_px > 2 * clean.mean_error_px
    assert noisy.grade in ("good", "fair", "poor")
    assert clean.grade == "excellent"


def test_evaluate_wide_three_monitor_desk() -> None:
    rng = np.random.default_rng(103)
    _, report = evaluate(calibration_samples(THREE, rng, noise=1.0, camera_x=960.0), THREE)
    assert report.monitor_accuracy >= 0.95, report
    assert report.degree == 3  # the tan-shaped mapping needs the cubic terms
    assert set(report.per_monitor_accuracy) == {0, 1, 2}


def test_evaluate_stacked_layout() -> None:
    rng = np.random.default_rng(104)
    _, report = evaluate(calibration_samples(STACKED, rng, noise=1.0, camera_x=960.0), STACKED)
    assert report.grade == "excellent", report


def test_evaluate_with_forced_degree() -> None:
    samples = calibration_samples(TWO, np.random.default_rng(105), per_point=6)
    model, report = evaluate(samples, TWO, degree=2)
    assert model.degree == report.degree == 2


def test_evaluate_single_monitor_is_allowed() -> None:
    mon = [TWO[0]]
    _, report = evaluate(calibration_samples(mon, np.random.default_rng(106)), mon)
    assert report.monitor_accuracy == 1.0
    assert report.per_monitor_accuracy == {0: 1.0}
    assert math.isfinite(report.mean_error_px)


def test_evaluate_requires_enough_data() -> None:
    rng = np.random.default_rng(107)
    samples = calibration_samples(TWO, rng, per_point=5)
    with pytest.raises(ValueError, match="not enough calibration data"):
        evaluate(samples[:9], TWO)  # fewer than 10 samples
    with pytest.raises(ValueError, match="not enough calibration data"):
        evaluate([s for s in samples if s.point_id < 2], TWO)  # two points only
    with pytest.raises(ValueError, match="not enough calibration data"):
        evaluate([s for s in samples if s.monitor_index == 0], TWO)  # one of two monitors
    with pytest.raises(ValueError, match="no monitors"):
        evaluate(samples, [])


def test_evaluate_ignores_samples_of_unknown_monitors(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(108)
    samples = calibration_samples(TWO, rng, per_point=6)
    stray = CalibrationSample(np.zeros(8), 9999.0, 9999.0, monitor_index=7, point_id=99)
    with caplog.at_level(logging.WARNING, logger="eye_tracker.gaze.calibration"):
        _, report = evaluate([*samples, stray], TWO)
    assert report.n_samples == len(samples)
    assert "unknown monitors" in caplog.text


def test_end_to_end_with_collector_and_fake_clock() -> None:
    """Drive the collector like the calibration window does, with a 15 fps fake camera."""
    rng = np.random.default_rng(109)
    plan = make_plan(TWO)
    c = CalibrationCollector(plan, settle_s=0.8, collect_s=1.0)
    now = 0.0
    c.start(now)
    events: list[str] = []
    while c.phase != "done":
        events += c.update(now)
        target = c.current
        if target is not None:
            features = synth_features(np.array([[target.x, target.y]]), rng, noise=1.0)[0]
            c.add(obs(now, features=features))
        now += 1 / 15
        assert now < 100, "calibration never finished"
    assert events.count("target") == len(plan)
    assert events[-1] == "finished"
    assert c.skipped_points == []
    assert 14 * len(plan) <= len(c.samples) <= 16 * len(plan)  # ~15 fps x 1 s per point
    _, report = evaluate(c.samples, TWO)
    assert report.grade == "excellent", report


# --------------------------------------------------------------------------- report
@pytest.mark.parametrize(
    ("accuracy", "grade"),
    [(1.0, "excellent"), (0.97, "excellent"), (0.9699, "good"), (0.9, "good"),
     (0.8999, "fair"), (0.75, "fair"), (0.7499, "poor"), (0.0, "poor")],
)  # fmt: skip
def test_grade_thresholds(accuracy: float, grade: str) -> None:
    assert grade_for(accuracy) == grade


def _report(**overrides: object) -> CalibrationReport:
    values: dict[str, object] = {
        "monitor_accuracy": 0.996,
        "mean_error_px": 42.4,
        "median_error_px": 38.0,
        "per_monitor_accuracy": {0: 1.0, 1: 0.99},
        "n_samples": 180,
        "n_points": 18,
        "alpha": 1.0,
        "grade": "excellent",
        "degree": 3,
    }
    values.update(overrides)
    return CalibrationReport(**values)  # type: ignore[arg-type]


def test_report_summary() -> None:
    assert _report().summary() == "Excellent — 99% monitor accuracy, 180 samples, mean error 42 px"
    assert _report(monitor_accuracy=1.0).summary().startswith("Excellent — 100% ")
    assert _report(mean_error_px=math.nan).summary().endswith("180 samples")


def test_report_dict_round_trip() -> None:
    report = _report(median_error_px=math.nan)
    data = report.to_dict()
    assert data["per_monitor_accuracy"] == {"0": 1.0, "1": 0.99}
    assert data["median_error_px"] is None
    restored = CalibrationReport.from_dict(data)
    assert restored.per_monitor_accuracy == {0: 1.0, 1: 0.99}
    assert math.isnan(restored.median_error_px)
    assert restored.degree == 3
    assert restored.summary() == report.summary()
    with pytest.raises(ValueError, match="invalid calibration report"):
        CalibrationReport.from_dict({"grade": "excellent"})


def test_samples_to_arrays() -> None:
    samples = [
        CalibrationSample(np.array([1.0, 2.0]), 10.0, 20.0, 0, 0),
        CalibrationSample(np.array([3.0, 4.0]), 30.0, 40.0, 1, 1, weight=0.5),
    ]
    X, Y, W = samples_to_arrays(samples)
    assert X.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert Y.tolist() == [[10.0, 20.0], [30.0, 40.0]]
    assert W.tolist() == [1.0, 0.5]
    with pytest.raises(ValueError, match="no samples"):
        samples_to_arrays([])
