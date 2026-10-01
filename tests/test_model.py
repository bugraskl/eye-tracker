"""Tests for eye_tracker.gaze.model (ridge regression, model selection, persistence)."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np
import pytest

from eye_tracker.gaze.calibration import evaluate
from eye_tracker.gaze.model import (
    DEFAULT_ALPHAS,
    LOOK_AWAY_EXCESS,
    LOOK_AWAY_MARGIN,
    AwayRegion,
    GazeModel,
    gaze_feature_indices,
    lopo_predictions,
    select_alpha,
    select_model,
)
from eye_tracker.types import Monitor, Rect, nearest_monitor, virtual_bounds
from gaze_synth import (
    FEATURE_NAMES,
    GAZE,
    MIXED,
    MIXED_PANELS,
    STACKED,
    THREE,
    TWO,
    below_panels_cm,
    below_points,
    calibration_samples,
    features_at_cm,
    grid_dataset,
    pixel_errors,
    random_points,
    synth_features,
)


# --------------------------------------------------------------------------- basics
def test_unfitted_model_refuses_to_predict() -> None:
    model = GazeModel()
    assert not model.is_fitted
    assert model.n_features == 0
    with pytest.raises(RuntimeError):
        model.predict(np.zeros(8))


@pytest.mark.parametrize(("degree", "alpha"), [(0, 1.0), (4, 1.0), (2, -1.0), (2, math.nan)])
def test_invalid_hyper_parameters(degree: int, alpha: float) -> None:
    with pytest.raises(ValueError, match=r"degree|alpha"):
        GazeModel(degree=degree, alpha=alpha)


def test_input_validation() -> None:
    X = np.ones((5, 3))
    Y = np.zeros((5, 2))
    with pytest.raises(ValueError, match="non-finite"):
        GazeModel().fit(np.full((5, 3), np.nan), Y)
    with pytest.raises(ValueError, match="Y must have shape"):
        GazeModel().fit(X, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="weights must be"):
        GazeModel().fit(X, Y, weights=-np.ones(5))
    with pytest.raises(ValueError, match="weights must be"):
        GazeModel().fit(X, Y, weights=np.zeros(5))
    with pytest.raises(ValueError, match="non-empty"):
        GazeModel().fit(np.empty((0, 3)), np.empty((0, 2)))


def test_predict_shapes_and_feature_length_check() -> None:
    rng = np.random.default_rng(1)
    X, P, _ = grid_dataset(TWO, rng)
    model = GazeModel().fit(X, P, bounds=virtual_bounds(TWO))
    assert model.predict(X[0]).shape == (2,)
    assert model.predict(X[:7]).shape == (7, 2)
    with pytest.raises(ValueError, match="expected feature vector"):
        model.predict(np.zeros(5))


def test_linear_data_is_recovered_exactly() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4))
    true_map = rng.normal(size=(4, 2)) * 300
    Y = X @ true_map + (500.0, 250.0)
    model = GazeModel(degree=1, alpha=0.0).fit(X, Y)
    x_new = rng.normal(size=(20, 4)) * 0.5  # inside the training range
    assert np.allclose(model.predict(x_new), x_new @ true_map + (500.0, 250.0), atol=1e-6)


def test_intercept_is_not_penalised() -> None:
    # With an enormous alpha every slope shrinks to zero; an unpenalised
    # intercept then predicts the mean target, not the desktop origin.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 3))
    Y = np.column_stack([rng.uniform(2000, 3000, 50), rng.uniform(500, 800, 50)])
    model = GazeModel(degree=2, alpha=1e12).fit(X, Y, bounds=Rect(0, 0, 3840, 1080))
    assert np.allclose(model.predict(X), Y.mean(axis=0), atol=1e-3)


def test_feature_scale_does_not_matter() -> None:
    # Ridge on standardised features: rescaling or shifting a raw feature
    # (e.g. centimetres vs millimetres) must not change predictions.
    rng = np.random.default_rng(4)
    X, P, _ = grid_dataset(TWO, rng, noise=1.0)
    scale = np.array([1.0, 1000.0, 1.0, 0.01, 1.0, 10.0, 1.0, 1.0])
    shift = np.array([0.0, 5.0, 0.0, 100.0, 0.0, -50.0, 0.0, 0.0])
    a = GazeModel(alpha=1.0).fit(X, P, bounds=virtual_bounds(TWO))
    b = GazeModel(alpha=1.0).fit(X * scale + shift, P, bounds=virtual_bounds(TWO))
    assert np.allclose(a.predict(X), b.predict(X * scale + shift), atol=1e-6)


def test_zero_weight_samples_are_ignored() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 3))
    Y = X @ rng.normal(size=(3, 2)) * 100 + 900
    X_out = np.vstack([X, [[0.1, 0.2, 0.3]]])
    Y_out = np.vstack([Y, [[99999.0, -99999.0]]])  # absurd outlier
    w = np.r_[np.ones(40), 0.0]
    # alpha=0 makes the fit independent of the standardisation, which does see the outlier.
    clean = GazeModel(degree=2, alpha=0.0).fit(X, Y)
    weighted = GazeModel(degree=2, alpha=0.0).fit(X_out, Y_out, weights=w)
    assert np.allclose(clean.predict(X), weighted.predict(X), atol=1e-4)


def test_bounds_default_to_target_bounding_box() -> None:
    X = np.arange(10, dtype=float).reshape(5, 2)
    Y = np.array([[10.2, 20.0], [30.0, 40.7], [50.0, 25.0], [12.0, 33.0], [44.0, 21.0]])
    model = GazeModel(degree=1).fit(X, Y)
    assert model.bounds == Rect(10, 20, 40, 21)
    explicit = GazeModel(degree=1).fit(X, Y, bounds=Rect(0, 0, 100, 100))
    assert explicit.bounds == Rect(0, 0, 100, 100)
    with pytest.raises(ValueError, match="positive size"):
        GazeModel().fit(X, Y, bounds=Rect(0, 0, 0, 100))


# --------------------------------------------------------------------------- accuracy
@pytest.mark.parametrize(
    ("monitors", "camera_x", "max_median_px"),
    [(TWO, 1920.0, 60.0), (STACKED, 960.0, 40.0), (THREE, 960.0, 80.0)],
    ids=["two-side-by-side", "stacked", "three-wide"],
)
def test_model_recovers_gaze_points(
    monitors: list[Monitor], camera_x: float, max_median_px: float
) -> None:
    rng = np.random.default_rng(10)
    X, P, _ = grid_dataset(monitors, rng, noise=0.0, camera_x=camera_x)
    model = GazeModel(degree=3, alpha=1.0).fit(X, P, bounds=virtual_bounds(monitors))
    test_points = random_points(monitors, rng, 400)
    errors = pixel_errors(
        model.predict(synth_features(test_points, rng, 0.0, camera_x)), test_points
    )
    assert np.median(errors) < max_median_px, f"median error {np.median(errors):.1f} px"


def test_noise_degrades_accuracy_gracefully() -> None:
    rng = np.random.default_rng(11)
    medians = []
    for noise in (0.0, 1.0, 3.0):
        X, P, _ = grid_dataset(TWO, rng, noise=noise)
        model = GazeModel(degree=2, alpha=1.0).fit(X, P, bounds=virtual_bounds(TWO))
        test_points = random_points(TWO, rng, 300)
        pred = model.predict(synth_features(test_points, rng, noise))
        medians.append(float(np.median(pixel_errors(pred, test_points))))
    assert medians[0] < medians[1] < medians[2], medians
    assert medians[1] < 200.0, medians  # realistic webcam noise: still well within a monitor


def test_unseen_pose_is_clipped_not_extrapolated() -> None:
    rng = np.random.default_rng(12)
    X, P, _ = grid_dataset(TWO, rng)
    bounds = virtual_bounds(TWO)
    model = GazeModel(degree=3, alpha=0.1).fit(X, P, bounds=bounds)
    x = X[0].copy()
    x[2] = 60.0  # head roll far beyond anything seen during calibration (~ +-5 deg)
    px, py = model.predict(x)
    # Without clipping the cubic roll term would send this many screens away.
    assert bounds.x - bounds.w <= px <= bounds.right + bounds.w
    assert bounds.y - bounds.h <= py <= bounds.bottom + bounds.h


# --------------------------------------------------------------------------- persistence
def test_round_trip_through_json() -> None:
    rng = np.random.default_rng(13)
    X, P, _ = grid_dataset(STACKED, rng, noise=1.0, camera_x=960.0)
    model = GazeModel(degree=3, alpha=0.1).fit(X, P, bounds=virtual_bounds(STACKED))
    restored = GazeModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.degree == 3
    assert restored.alpha == 0.1
    assert restored.bounds == virtual_bounds(STACKED)
    assert restored.n_features == 8
    assert np.array_equal(restored.predict(X), model.predict(X))
    assert restored.has_linear_estimate
    away = [restored.looks_away(x, GAZE, monitors=STACKED) for x in X[::7]]
    assert away == [model.looks_away(x, GAZE, monitors=STACKED) for x in X[::7]]


def test_unfitted_round_trip() -> None:
    restored = GazeModel.from_dict(GazeModel(degree=1, alpha=3.0).to_dict())
    assert (restored.degree, restored.alpha, restored.is_fitted) == (1, 3.0, False)


def test_model_without_clip_range_still_loads() -> None:
    rng = np.random.default_rng(14)
    X, P, _ = grid_dataset(TWO, rng)
    data = GazeModel(degree=1).fit(X, P).to_dict()
    del data["lo"], data["hi"]
    assert GazeModel.from_dict(data).predict(X[:3]).shape == (3, 2)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(kind="neural-net"), "unsupported model kind"),
        (lambda d: d.update(degree=7), "degree must be one of"),
        (lambda d: d.update(coef=d["coef"][:-1]), "inconsistent shapes"),
        (lambda d: d.update(std=d["std"][:-1]), "inconsistent shapes"),
        (lambda d: d.update(mean=[math.nan] * len(d["mean"])), "non-finite"),
        (lambda d: d.update(std=[0.0] * len(d["std"])), "must be positive"),
        (lambda d: d.update(bounds=[0, 0, 0, 10]), "must be positive"),
        (lambda d: d.pop("lo"), "incomplete"),
        (lambda d: d.update(mean="oops"), "invalid model data"),
        (lambda d: d.pop("bounds"), "invalid model data"),
        (lambda d: d.update(linear_coef=d["linear_coef"][:-1]), "inconsistent shapes"),
        (lambda d: d.update(linear_coef=[[math.inf, 0.0]] * 9), "non-finite"),
        (lambda d: d.update(linear_coef="oops"), "invalid model data"),
    ],
)
def test_from_dict_rejects_malformed_data(mutate, message: str) -> None:
    rng = np.random.default_rng(15)
    X, P, _ = grid_dataset(TWO, rng, per_point=3)
    data = GazeModel(degree=2).fit(X, P).to_dict()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        GazeModel.from_dict(data)


def test_from_dict_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        GazeModel.from_dict([1, 2, 3])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- model selection
def test_select_alpha_returns_a_candidate_and_prefers_less_shrinkage_on_clean_data() -> None:
    rng = np.random.default_rng(20)
    X, P, groups = grid_dataset(TWO, rng, noise=0.0)
    alpha = select_alpha(X, P, groups, bounds=virtual_bounds(TWO))
    assert alpha in DEFAULT_ALPHAS
    assert alpha <= 1.0


def test_select_alpha_with_a_single_group_falls_back_to_the_middle() -> None:
    X = np.random.default_rng(21).normal(size=(12, 3))
    assert select_alpha(X, np.ones((12, 2)), np.zeros(12)) == 1.0
    assert select_alpha(X, np.ones((12, 2)), np.zeros(12), alphas=(0.5, 5.0, 50.0)) == 5.0


def test_select_alpha_rejects_bad_candidates() -> None:
    X = np.random.default_rng(22).normal(size=(12, 3))
    with pytest.raises(ValueError, match="must not be empty"):
        select_alpha(X, np.ones((12, 2)), np.arange(12) % 3, alphas=())
    with pytest.raises(ValueError, match="alpha must be"):
        select_alpha(X, np.ones((12, 2)), np.arange(12) % 3, alphas=(-1.0,))


def test_select_model_prefers_the_simplest_adequate_degree() -> None:
    rng = np.random.default_rng(23)
    X = rng.normal(size=(120, 3))
    Y = X @ np.array([[400.0, 30.0], [20.0, 300.0], [5.0, 5.0]]) + 1500 + rng.normal(0, 5, (120, 2))
    groups = np.repeat(np.arange(12), 10)
    selection = select_model(X, Y, groups)
    assert selection.degree == 1
    assert selection.predictions is not None
    assert selection.predictions.shape == (120, 2)


def test_select_model_uses_cubic_terms_for_wide_desks() -> None:
    # Three flat monitors span ~+-50 degrees; tan() bends too much for a quadratic.
    rng = np.random.default_rng(24)
    X, P, groups = grid_dataset(THREE, rng, noise=0.0, camera_x=960.0)
    selection = select_model(X, P, groups, bounds=virtual_bounds(THREE))
    assert selection.degree == 3
    assert selection.mean_error_px < 80.0, selection


def test_lopo_predictions_match_explicit_refits() -> None:
    # With alpha=0 the fit does not depend on standardisation, so the fast
    # downdating implementation must equal refitting without each group.
    rng = np.random.default_rng(25)
    X = rng.normal(size=(90, 3))
    Y = X @ rng.normal(size=(3, 2)) * 200 + rng.normal(0, 20, (90, 2)) + 1000
    groups = rng.permutation(np.repeat(np.arange(9), 10))
    fast = lopo_predictions(X, Y, groups, alpha=0.0, degree=1)
    for g in range(9):
        held = groups == g
        model = GazeModel(degree=1, alpha=0.0).fit(X[~held], Y[~held])
        assert np.allclose(fast[held], model.predict(X[held]), atol=1e-6)


def test_lopo_predictions_need_two_groups() -> None:
    X = np.random.default_rng(26).normal(size=(10, 2))
    with pytest.raises(ValueError, match="at least two groups"):
        lopo_predictions(X, np.zeros((10, 2)), np.zeros(10), alpha=1.0)


def test_lopo_rows_without_training_weight_are_nan() -> None:
    rng = np.random.default_rng(27)
    X = rng.normal(size=(20, 2))
    Y = rng.normal(size=(20, 2)) * 100
    groups = np.repeat([0, 1], 10)
    weights = np.r_[np.ones(10), np.zeros(10)]
    preds = lopo_predictions(X, Y, groups, alpha=1.0, degree=1, weights=weights)
    assert np.all(np.isnan(preds[:10]))  # its complement has zero weight
    assert np.all(np.isfinite(preds[10:]))


# --------------------------------------------------------------------------- nonlinear features
def monitor_accuracy(
    model: GazeModel, monitors: list[Monitor], points: np.ndarray, X: np.ndarray
) -> float:
    pred = model.predict(X)
    hits = [
        nearest_monitor(monitors, *p)[0].index == nearest_monitor(monitors, *t)[0].index
        for p, t in zip(pred, points, strict=True)
    ]
    return float(np.mean(hits))


def test_nonlinear_terms_only_for_the_selected_features() -> None:
    rng = np.random.default_rng(30)
    X, P, _ = grid_dataset(TWO, rng, noise=1.0)
    bounds = virtual_bounds(TWO)
    full = GazeModel(degree=3, alpha=1.0).fit(X, P, bounds=bounds)
    masked = GazeModel(degree=3, alpha=1.0, nonlinear=GAZE).fit(X, P, bounds=bounds)
    assert full.nonlinear is None
    assert masked.nonlinear == GAZE
    # 1 + 8 linear + 10 products + 4 cubes, instead of 1 + 8 + 36 + 8.
    assert np.asarray(masked.to_dict()["coef"]).shape == (23, 2)
    assert np.asarray(full.to_dict()["coef"]).shape == (53, 2)
    assert "nonlinear=[0, 1, 6, 7]" in repr(masked)

    # Head position (tx = feature 3) now enters linearly: equal steps move the
    # prediction by equal amounts. The gaze features still bend.
    x = X[40]
    step = np.zeros(8)
    step[3] = 0.5
    p0, p1, p2 = (masked.predict(x + k * step) for k in (-1, 0, 1))
    assert np.allclose(p2 - p1, p1 - p0, atol=1e-6)
    step = np.zeros(8)
    step[0] = 3.0
    q0, q1, q2 = (masked.predict(x + k * step) for k in (-1, 0, 1))
    assert not np.allclose(q2 - q1, q1 - q0, atol=1e-3)


def test_nonlinear_indices_are_normalised_and_validated() -> None:
    assert GazeModel(nonlinear=[7, 0, 1, 1]).nonlinear == (0, 1, 7)
    assert GazeModel(nonlinear=np.array([2, 1])).nonlinear == (1, 2)  # type: ignore[arg-type]
    for bad in ([-1], [1.5], [True], ["0"]):
        with pytest.raises(ValueError, match="feature indices"):
            GazeModel(nonlinear=bad)  # type: ignore[arg-type]
    X = np.random.default_rng(31).normal(size=(30, 3))
    Y = X[:, :2] * 100
    with pytest.raises(ValueError, match="out of range"):
        GazeModel(nonlinear=(3,)).fit(X, Y)
    with pytest.raises(ValueError, match="out of range"):
        select_model(X, Y, np.arange(30) % 5, nonlinear=(0, 5))
    with pytest.raises(ValueError, match="out of range"):
        lopo_predictions(X, Y, np.arange(30) % 5, 1.0, nonlinear=(9,))
    # No nonlinear features at all: every degree is the linear model.
    linear = GazeModel(degree=1, alpha=0.1).fit(X, Y)
    empty = GazeModel(degree=3, alpha=0.1, nonlinear=()).fit(X, Y)
    assert np.allclose(empty.predict(X), linear.predict(X))


def test_nonlinear_round_trip_and_legacy_models() -> None:
    rng = np.random.default_rng(32)
    X, P, _ = grid_dataset(TWO, rng, noise=1.0, per_point=4)
    model = GazeModel(degree=2, alpha=0.1, nonlinear=GAZE).fit(X, P)
    data = json.loads(json.dumps(model.to_dict()))
    assert data["nonlinear"] == [0, 1, 6, 7]
    restored = GazeModel.from_dict(data)
    assert restored.nonlinear == GAZE
    assert np.array_equal(restored.predict(X), model.predict(X))
    # Files written before the key existed expanded every feature.
    legacy = GazeModel(degree=2, alpha=0.1).fit(X, P).to_dict()
    assert "nonlinear" not in legacy
    assert GazeModel.from_dict(legacy).nonlinear is None
    assert GazeModel.from_dict({**legacy, "nonlinear": None}).nonlinear is None
    unfitted = GazeModel.from_dict(GazeModel(degree=3, nonlinear=(1,)).to_dict())
    assert (unfitted.degree, unfitted.nonlinear, unfitted.is_fitted) == (3, (1,), False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(nonlinear="yaw"), "nonlinear must be a list"),
        (lambda d: d.update(nonlinear=[0, 8]), "out of range"),
        (lambda d: d.update(nonlinear=[0, -1]), "feature indices"),
        (lambda d: d.update(nonlinear=[0, 1.0]), "feature indices"),
        (lambda d: d.update(nonlinear=[0, 1]), "inconsistent shapes"),  # coef has 4 of them
        # int(inf) raises OverflowError; it must surface as ValueError like the rest.
        (lambda d: d.update(bounds=[0, 0, 1920, math.inf]), "invalid model data"),
        (lambda d: d.update(degree=math.inf), "invalid model parameters"),
    ],
)
def test_from_dict_rejects_malformed_nonlinear_and_overflow(mutate, message: str) -> None:
    rng = np.random.default_rng(33)
    X, P, _ = grid_dataset(TWO, rng, per_point=3)
    data = GazeModel(degree=2, nonlinear=GAZE).fit(X, P).to_dict()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        GazeModel.from_dict(data)


def test_gaze_feature_indices(caplog: pytest.LogCaptureFixture) -> None:
    gaze_names = ("yaw", "pitch", "iris_h", "iris_v")
    assert gaze_feature_indices(FEATURE_NAMES, gaze_names) == GAZE
    assert gaze_feature_indices(FEATURE_NAMES, ("iris_v", "yaw")) == (0, 7)
    assert gaze_feature_indices(("nose_dx", "nose_dy", "roll"), ("nose_dx", "nose_dy")) == (0, 1)
    assert gaze_feature_indices(FEATURE_NAMES, ()) is None  # backend declares none
    with caplog.at_level(logging.WARNING, logger="eye_tracker.gaze.model"):
        assert gaze_feature_indices(FEATURE_NAMES, ("yaw", "blink")) == (0,)
        assert gaze_feature_indices(FEATURE_NAMES, ("blink",)) is None
    assert "blink" in caplog.text


@pytest.mark.parametrize("seed", [40, 41, 42])
@pytest.mark.parametrize(
    "head_offset",
    [(0.0, 8.0, 0.0), (0.0, -8.0, 0.0), (0.0, 0.0, 12.0), (10.0, 0.0, 0.0)],
    ids=["8cm-lower", "8cm-higher", "lean-back-12cm", "sideways-10cm"],
)
def test_posture_change_keeps_the_selected_model_accurate(
    seed: int, head_offset: tuple[float, float, float]
) -> None:
    # Regression for polynomial terms over head position: cross-validation holds
    # out dots, never postures, so it picked a cubic fit through noise-level
    # tx/ty/tz that broke as soon as the user sat differently.
    rng = np.random.default_rng(seed)
    X, P, groups = grid_dataset(TWO, rng, noise=1.0)
    bounds = virtual_bounds(TWO)
    selection = select_model(X, P, groups, bounds=bounds, nonlinear=GAZE)
    model = GazeModel(selection.degree, selection.alpha, nonlinear=GAZE).fit(X, P, bounds=bounds)
    points = random_points(TWO, rng, 1500)
    moved = synth_features(points, rng, 1.0, head_offset=head_offset)
    assert monitor_accuracy(model, TWO, points, moved) >= 0.99
    assert np.median(pixel_errors(model.predict(moved), points)) < 200.0


def test_posture_change_breaks_polynomials_over_head_position() -> None:
    # The same data with every feature expanded (the old design): documents why
    # the gaze-direction restriction exists.
    rng = np.random.default_rng(40)
    X, P, groups = grid_dataset(TWO, rng, noise=1.0)
    bounds = virtual_bounds(TWO)
    accuracies = {}
    for nonlinear in (None, GAZE):
        selection = select_model(X, P, groups, bounds=bounds, nonlinear=nonlinear)
        model = GazeModel(selection.degree, selection.alpha, nonlinear=nonlinear)
        model.fit(X, P, bounds=bounds)
        points = random_points(TWO, np.random.default_rng(1), 1500)
        moved = synth_features(points, np.random.default_rng(2), 1.0, head_offset=(0, -8, 0))
        accuracies[nonlinear] = monitor_accuracy(model, TWO, points, moved)
    assert accuracies[GAZE] >= 0.99
    assert accuracies[None] < accuracies[GAZE] - 0.02


def test_select_model_three_monitors_still_uses_cubic_gaze_terms() -> None:
    rng = np.random.default_rng(24)
    X, P, groups = grid_dataset(THREE, rng, noise=0.0, camera_x=960.0)
    selection = select_model(X, P, groups, bounds=virtual_bounds(THREE), nonlinear=GAZE)
    assert selection.degree == 3
    # Fewer terms fit the calibration dots marginally worse (~80 px vs ~75 px
    # with every feature expanded) and in exchange survive posture changes.
    assert selection.mean_error_px < 90.0, selection


# --------------------------------------------------------------------------- looking away
def test_extrapolation_measures_distance_beyond_the_calibrated_range() -> None:
    X = np.array([[0.0, 10.0], [2.0, 20.0], [1.0, 15.0]])
    Y = np.array([[0.0, 0.0], [100.0, 50.0], [50.0, 25.0]])
    model = GazeModel(degree=1, alpha=0.0).fit(X, Y)
    assert np.array_equal(model.extrapolation(np.array([1.0, 12.0])), [0.0, 0.0])
    assert np.allclose(model.extrapolation(np.array([2.5, 5.0])), [0.25, 0.5])
    batch = model.extrapolation(np.array([[-1.0, 20.0], [1.0, 22.0]]))
    assert np.allclose(batch, [[0.5, 0.0], [0.0, 0.2]])
    with pytest.raises(ValueError, match="expected feature vector"):
        model.extrapolation(np.zeros(3))
    with pytest.raises(RuntimeError):
        GazeModel().extrapolation(np.zeros(2))
    # Files without the calibrated range cannot tell.
    data = model.to_dict()
    del data["lo"], data["hi"]
    assert np.array_equal(GazeModel.from_dict(data).extrapolation(np.array([9.0, 99.0])), [0, 0])


def test_looks_away_uses_only_gaze_direction_features() -> None:
    X = np.array([[0.0, 10.0, 5.0], [2.0, 20.0, 6.0], [1.0, 15.0, 5.5]])
    Y = np.array([[0.0, 0.0], [100.0, 50.0], [50.0, 25.0]])
    model = GazeModel(degree=1, alpha=0.0, nonlinear=(0, 1)).fit(X, Y)
    assert not model.looks_away(np.array([1.0, 15.0, 5.5]))
    assert model.looks_away(np.array([1.0, 25.0, 5.5]))  # 0.5 of the range beyond
    assert not model.looks_away(np.array([1.0, 21.0, 5.5]))  # 0.1: a screen edge
    assert model.looks_away(np.array([1.0, 21.0, 5.5]), threshold=0.05)
    # Feature 2 (head position) is far out, but leaning back is not looking away.
    assert not model.looks_away(np.array([1.0, 15.0, 50.0]))
    assert model.looks_away(np.array([1.0, 15.0, 50.0]), features=[2])
    assert not model.looks_away(np.array([np.nan, 15.0, 5.5]))
    with pytest.raises(ValueError, match="one feature vector"):
        model.looks_away(X)
    with pytest.raises(ValueError, match="out of range"):
        model.looks_away(X[0], features=[3])
    # A model saved before gaze features were known cannot tell which features
    # are safe to use, unless the caller passes the backend's indices.
    legacy = GazeModel(degree=1, alpha=0.0).fit(X, Y)
    assert not legacy.looks_away(np.array([1.0, 99.0, 5.5]))
    assert legacy.looks_away(np.array([1.0, 99.0, 5.5]), features=(0, 1))
    assert LOOK_AWAY_EXCESS == 0.25


@pytest.mark.parametrize(
    ("monitors", "camera_x", "seed"),
    [(TWO, 1920.0, 50), (TWO, 1920.0, 51), (THREE, 960.0, 52)],
    ids=["two-a", "two-b", "three"],
)
def test_glances_below_the_monitors_look_away(
    monitors: list[Monitor], camera_x: float, seed: int
) -> None:
    # The per-feature fallback (no monitors passed, or a model saved without the
    # linear estimate); the combined test is checked further below.
    rng = np.random.default_rng(seed)
    X, P, groups = grid_dataset(monitors, rng, noise=1.0, camera_x=camera_x)
    bounds = virtual_bounds(monitors)
    selection = select_model(X, P, groups, bounds=bounds, nonlinear=GAZE)
    model = GazeModel(selection.degree, selection.alpha, nonlinear=GAZE).fit(X, P, bounds=bounds)

    def away_rate(points: np.ndarray, offset: tuple[float, float, float] = (0, 0, 0)) -> float:
        feats = synth_features(points, rng, 1.0, camera_x, head_offset=offset)
        return float(np.mean([model.looks_away(f) for f in feats]))

    # A phone or papers on the desk, 60-80 cm below the screens. Without the
    # check, clipping keeps these predictions next to the screens, where the
    # switch decider takes them for gaze at the nearest monitor.
    for cm, minimum in ((60, 0.7), (80, 0.9)):
        points = below_points(monitors, rng, 400, cm)
        predicted = model.predict(synth_features(points, rng, 1.0, camera_x))
        margin = 0.35 * monitors[0].rect.diagonal  # SwitchDecider's off-screen margin
        near = [nearest_monitor(monitors, *p)[1] <= margin for p in predicted]
        assert np.mean(near) > 0.9
        assert away_rate(points) >= minimum, cm
    # Gaze anywhere on the screens, edges and corners included, even after the
    # user moved, practically never counts as looking away.
    everywhere = random_points(monitors, rng, 1500, margin=0.0)
    for offset in ((0.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, -8.0, 0.0), (0.0, 0.0, 12.0)):
        assert away_rate(everywhere, offset) < 0.01, offset


# ------------------------------------------------------- looking away: combined direction
SCREEN = [Monitor(0, "screen", Rect(0, 0, 1000, 500))]


def simple_gaze_model() -> tuple[GazeModel, np.ndarray]:
    """Two gaze-direction features that map linearly onto one 1000x500 screen, and
    a head position (feature 2) that drifts with the gaze but is noisy."""
    rng = np.random.default_rng(60)
    xs, ys = np.meshgrid(np.linspace(100, 900, 20), np.linspace(50, 450, 10))
    P = np.repeat(np.column_stack([xs.ravel(), ys.ravel()]), 2, axis=0)
    head = 0.5 * P[:, 0] / 1000 + rng.normal(0, 0.05, len(P))
    X = np.column_stack([P[:, 0] / 1000, P[:, 1] / 500, head])
    model = GazeModel(degree=1, alpha=0.0, nonlinear=(0, 1)).fit(X, P, bounds=SCREEN[0].rect)
    return model, X


def gaze_at(x: float, y: float, head: float = 0.25) -> np.ndarray:
    return np.array([x / 1000, y / 500, head])


def test_looks_away_judges_the_combined_gaze_direction() -> None:
    model, X = simple_gaze_model()
    assert model.has_linear_estimate
    diagonal = SCREEN[0].rect.diagonal
    assert not model.looks_away(gaze_at(500, 250), monitors=SCREEN)
    assert not model.looks_away(gaze_at(0, 0), monitors=SCREEN)  # a corner
    assert not model.looks_away(gaze_at(1000 + 0.1 * diagonal, 250), monitors=SCREEN)
    assert model.looks_away(gaze_at(1000 + 0.4 * diagonal, 250), monitors=SCREEN)
    assert model.looks_away(gaze_at(500, 500 + 0.4 * diagonal), monitors=SCREEN)  # the desk
    assert not model.looks_away(gaze_at(500, 500 + 0.4 * diagonal), monitors=SCREEN, margin=0.5)
    # The head far from where it was: leaning back is not looking away (clipped),
    # unless the caller declares that feature a gaze direction.
    assert not model.looks_away(gaze_at(500, 250, head=50.0), monitors=SCREEN)
    assert model.looks_away(gaze_at(500, 250, head=50.0), [0, 1, 2], monitors=SCREEN)
    # Broken frames and unknown gaze features never read as "looking away".
    assert not model.looks_away(gaze_at(math.nan, 250), monitors=SCREEN)
    assert not model.looks_away(gaze_at(math.inf, 250), monitors=SCREEN)
    legacy = GazeModel(degree=1, alpha=0.0).fit(X, X[:, :2] * (1000, 500))
    assert not legacy.looks_away(gaze_at(5000, 250), monitors=SCREEN)
    assert legacy.looks_away(gaze_at(5000, 250), (0, 1), monitors=SCREEN)
    with pytest.raises(ValueError, match="out of range"):
        model.looks_away(gaze_at(500, 250), [3], monitors=SCREEN)
    with pytest.raises(ValueError, match="expected feature vector"):
        model.looks_away(np.zeros(2), monitors=SCREEN)


def test_models_saved_without_the_linear_estimate_use_the_per_feature_test() -> None:
    model, _ = simple_gaze_model()
    data = model.to_dict()
    del data["linear_coef"]
    old = GazeModel.from_dict(data)
    assert not old.has_linear_estimate
    # 0.15 of the diagonal beside the screen, but 0.3 of the calibrated x range
    # beyond it: the per-feature test calls that "away", the combined one does not.
    beside = gaze_at(1000 + 0.15 * SCREEN[0].rect.diagonal, 250)
    assert old.looks_away(beside, monitors=SCREEN)
    assert not model.looks_away(beside, monitors=SCREEN)
    # Without monitors the per-feature test is used as well.
    assert model.looks_away(beside)
    data["linear_coef"] = None
    assert not GazeModel.from_dict(data).has_linear_estimate


def taskbar_points(monitors: list[Monitor], rng: np.random.Generator, n: int) -> np.ndarray:
    """Points in the bottom 40 px of the monitors (the taskbar)."""
    idx = rng.integers(0, len(monitors), n)
    return np.array(
        [
            (rng.uniform(r.x, r.right), rng.uniform(r.bottom - 40, r.bottom))
            for r in (monitors[i].rect for i in idx)
        ]
    )


def edge_points(monitors: list[Monitor], rng: np.random.Generator, n: int) -> np.ndarray:
    """Points in the outer 3 % of the monitors, corners included."""
    out = []
    for _ in range(n):
        rect = monitors[rng.integers(0, len(monitors))].rect
        nx, ny = rng.uniform(0, 1, 2)
        side = rng.integers(0, 4)
        if side < 2:
            nx = rng.uniform(0, 0.03) if side == 0 else rng.uniform(0.97, 1.0)
        else:
            ny = rng.uniform(0, 0.03) if side == 2 else rng.uniform(0.97, 1.0)
        out.append(rect.denormalize(nx, ny))
    return np.array(out)


#: Users who share the work between head and eyes differently than while
#: calibrating, or sit elsewhere (keyword arguments of ``synth_features``).
ON_SCREEN_CHANGES: dict[str, dict[str, Any]] = {
    "as calibrated": {},
    "head turns more": {"head_share": 0.70},
    "head turns less": {"head_share": 0.40},
    "nods more, sits higher": {"pitch_share": 0.8 * 1.4, "head_offset": (0.0, -5.0, 0.0)},
    "sits lower": {"head_offset": (0.0, 8.0, 0.0)},
    "leans back": {"head_offset": (0.0, 0.0, 12.0)},
    "moves aside": {"head_offset": (8.0, 0.0, 0.0)},
}


def fitted_on_synthetic_calibration(
    monitors: list[Monitor], camera_x: float, rng: np.random.Generator
) -> GazeModel:
    X, P, groups = grid_dataset(monitors, rng, noise=1.0, camera_x=camera_x)
    bounds = virtual_bounds(monitors)
    selection = select_model(X, P, groups, bounds=bounds, nonlinear=GAZE)
    model = GazeModel(selection.degree, selection.alpha, nonlinear=GAZE)
    # As calibration.evaluate fits it: with the look-away regions of the monitors.
    return model.fit(X, P, bounds=bounds, regions=[m.rect for m in monitors])


@pytest.mark.parametrize(
    ("monitors", "camera_x", "seed"),
    [(TWO, 1920.0, 50), (TWO, 1920.0, 51), (THREE, 960.0, 52), (STACKED, 960.0, 53)],
    ids=["two-a", "two-b", "three", "stacked"],
)
def test_combined_look_away_test_keeps_every_screen_reachable(
    monitors: list[Monitor], camera_x: float, seed: int
) -> None:
    """r2-gaze-engine-03: edges, corners and the taskbar stay on screen even when
    the user moves the head more or less than while calibrating, while glances
    at the desk are still caught."""
    rng = np.random.default_rng(seed)
    model = fitted_on_synthetic_calibration(monitors, camera_x, rng)

    def away_rate(points: np.ndarray, **kwargs: Any) -> float:
        feats = synth_features(points, rng, 1.0, camera_x, **kwargs)
        return float(np.mean([model.looks_away(f, monitors=monitors) for f in feats]))

    for name, change in ON_SCREEN_CHANGES.items():
        for region in (taskbar_points, edge_points):
            rate = away_rate(region(monitors, rng, 500), **change)
            assert rate <= 0.01, (name, region.__name__, rate)
    assert away_rate(below_points(monitors, rng, 300, 30)) >= 0.9
    for cm in (40, 60, 80):
        assert away_rate(below_points(monitors, rng, 300, cm)) >= 0.95, cm


def test_per_feature_test_loses_the_taskbar_when_the_head_eye_split_changes() -> None:
    """Why the combined test exists: judged feature by feature, a user who nods
    more than while calibrating looks "away" at the bottom of the screens."""
    rng = np.random.default_rng(50)
    model = fitted_on_synthetic_calibration(TWO, 1920.0, rng)
    change = ON_SCREEN_CHANGES["nods more, sits higher"]
    feats = synth_features(taskbar_points(TWO, rng, 500), rng, 1.0, **change)
    assert np.mean([model.looks_away(f) for f in feats]) > 0.05
    assert np.mean([model.looks_away(f, monitors=TWO) for f in feats]) <= 0.01


# ------------------------------------------------- looking away: mixed pixel densities
def outer_strip(rect: Rect, rng: np.random.Generator, n: int, fraction: float) -> np.ndarray:
    """Points in the outer (left) ``fraction`` of ``rect``, top to bottom."""
    xs = rng.uniform(rect.x, rect.x + fraction * rect.w, n)
    return np.column_stack([xs, rng.uniform(rect.y, rect.bottom, n)])


def without_regions(model: GazeModel) -> GazeModel:
    """The same fit as the previous version made it (no look-away regions)."""
    data = model.to_dict()
    del data["away_regions"]
    return GazeModel.from_dict(data)


@pytest.mark.parametrize("seed", [140, 141, 142])
def test_mixed_density_desk_keeps_the_coarse_monitors_edge_on_screen(seed: int) -> None:
    """r3-logic-02: a 1080p panel next to a 4K one at 100 %. One linear map for the
    whole desk overshoots on the coarse monitor, and its outer edge read as
    "looking away", most of all while the user leans in (the per-feature rule
    used before flagged none of it)."""
    rng = np.random.default_rng(seed)
    samples = calibration_samples(MIXED, rng, 1.0, per_point=15, panels=MIXED_PANELS)
    model, _ = evaluate(samples, MIXED, nonlinear=GAZE)
    coarse, dense = model.away_regions
    assert coarse.rect == MIXED[0].rect
    assert coarse.diagonal > 1.2 * MIXED[0].rect.diagonal  # placed larger than it is
    assert dense == AwayRegion.of(MIXED[1].rect)  # placed smaller: the monitor itself
    previous = without_regions(model)

    def away_rate(m: GazeModel, points: np.ndarray, **change: Any) -> float:
        feats = synth_features(points, rng, 1.0, panels=MIXED_PANELS, **change)
        return float(np.mean([m.looks_away(f, monitors=MIXED) for f in feats]))

    leaning_in = {"head_offset": (0.0, 0.0, -10.0)}
    strip = outer_strip(MIXED[0].rect, rng, 500, 0.10)
    assert away_rate(model, strip, **leaning_in) <= 0.01
    assert away_rate(previous, strip, **leaning_in) >= 0.1  # the regression
    for name, change in {**ON_SCREEN_CHANGES, "leans in": leaning_in}.items():
        for region in (taskbar_points, edge_points):
            rate = away_rate(model, region(MIXED, rng, 400), **change)
            assert rate <= 0.01, (name, region.__name__, rate)
    # Glances at the desk are caught as well as before.
    for cm, minimum in ((40, 0.3), (60, 0.75)):
        x, y = below_panels_cm(MIXED_PANELS, rng, 300, cm)
        feats = features_at_cm(x, y, rng, 1.0)
        rate = float(np.mean([model.looks_away(f, monitors=MIXED) for f in feats]))
        before = float(np.mean([previous.looks_away(f, monitors=MIXED) for f in feats]))
        assert rate >= minimum, cm
        assert rate >= before - 0.02, cm


@pytest.mark.parametrize(
    ("monitors", "camera_x", "seed"),
    [(TWO, 1920.0, 50), (THREE, 960.0, 52), (STACKED, 960.0, 53)],
    ids=["two", "three", "stacked"],
)
def test_regions_never_make_the_look_away_test_stricter(
    monitors: list[Monitor], camera_x: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    model = fitted_on_synthetic_calibration(monitors, camera_x, rng)
    assert [r.rect for r in model.away_regions] == [m.rect for m in monitors]
    for region, monitor in zip(model.away_regions, monitors, strict=True):
        rect = monitor.rect
        assert region.x0 <= rect.x
        assert region.y0 <= rect.y
        assert region.x1 >= rect.right
        assert region.y1 >= rect.bottom
        assert region.diagonal >= rect.diagonal
    previous = without_regions(model)
    points = np.concatenate(
        [random_points(monitors, rng, 300, margin=0.0), below_points(monitors, rng, 300, 25)]
    )
    feats = synth_features(points, rng, 1.0, camera_x, head_offset=(0.0, 0.0, -8.0))
    for f in feats:
        if not previous.looks_away(f, monitors=monitors):
            assert not model.looks_away(f, monitors=monitors)


def test_away_regions_round_trip_and_validation() -> None:
    rng = np.random.default_rng(144)
    samples = calibration_samples(MIXED, rng, 1.0, per_point=10, panels=MIXED_PANELS)
    model, _ = evaluate(samples, MIXED, nonlinear=GAZE)
    data = json.loads(json.dumps(model.to_dict()))
    restored = GazeModel.from_dict(data)
    assert restored.away_regions == model.away_regions
    # Malformed regions make the model invalid, like every other field.
    for broken in (
        "not a list",
        [{"rect": [0, 0, 10, 10], "box": [0, 0, 10], "diagonal": 14.0}],
        [{"rect": [0, 0, 10, 10], "box": [5, 0, 0, 10], "diagonal": 14.0}],
        [{"rect": [0, 0, 10, 10], "box": [0, 0, 10, 10], "diagonal": float("nan")}],
        [{"rect": [0, 0, 0, 10], "box": [0, 0, 10, 10], "diagonal": 14.0}],
        [["rect"]],
    ):
        with pytest.raises(ValueError, match=r"model data|away region"):
            GazeModel.from_dict(dict(data, away_regions=broken))
    # Older files have none and still load.
    legacy = dict(data)
    del legacy["away_regions"]
    assert GazeModel.from_dict(legacy).away_regions == ()


def test_away_regions_widen_the_test_only_for_their_own_monitor() -> None:
    model, _ = simple_gaze_model()  # its linear estimate is the gaze point itself
    data = model.to_dict()
    # Where a coarser monitor would be placed: 400 px further left, and larger.
    data["away_regions"] = [
        {"rect": [0, 0, 1000, 500], "box": [-400, 0, 1000, 500], "diagonal": 2000}
    ]
    widened = GazeModel.from_dict(data)
    beside = gaze_at(-500, 250)  # 0.45 of the screen's diagonal to its left
    assert model.looks_away(beside, monitors=SCREEN)
    assert not widened.looks_away(beside, monitors=SCREEN)
    assert widened.looks_away(gaze_at(-1300, 250), monitors=SCREEN)  # beyond its margin too
    # The region of another layout (a profile of another desk) does not apply.
    taller = [Monitor(0, "screen", Rect(0, 0, 1000, 600))]
    assert widened.looks_away(beside, monitors=taller)
    assert LOOK_AWAY_MARGIN * Rect(0, 0, 1000, 600).diagonal < 500


def test_away_regions_need_targets_spread_over_the_monitor() -> None:
    rng = np.random.default_rng(145)
    samples = calibration_samples(TWO, rng, 1.0, per_point=10, points_per_monitor=1)
    X = np.array([s.features for s in samples])
    Y = np.array([(s.x, s.y) for s in samples])
    rects = [m.rect for m in TWO]
    model = GazeModel(1, 1.0, nonlinear=GAZE).fit(X, Y, bounds=virtual_bounds(TWO), regions=rects)
    # One dot per monitor cannot place its corners: the monitors themselves.
    assert model.away_regions == tuple(AwayRegion.of(r) for r in rects)
