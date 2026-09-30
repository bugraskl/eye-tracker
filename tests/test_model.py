"""Tests for eye_tracker.gaze.model (ridge regression, model selection, persistence)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from eye_tracker.gaze.model import (
    DEFAULT_ALPHAS,
    GazeModel,
    lopo_predictions,
    select_alpha,
    select_model,
)
from eye_tracker.types import Monitor, Rect, virtual_bounds

# --------------------------------------------------------------------------- synthetic data
# A user ~65 cm from 24" 1080p panels (53 cm wide). The head turns part of the way
# towards the target and the eyes do the rest; the webcam reports head pose, head
# position and iris offsets. Flat screens make pixels ~ tan(angle): non-linear.
PX_CM = 53.0 / 1920
NOISE_SCALE = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 0.01, 0.006])

TWO = [
    Monitor(0, "left", Rect(0, 0, 1920, 1080), primary=True),
    Monitor(1, "right", Rect(1920, 0, 1920, 1080)),
]
THREE = [Monitor(i, f"m{i}", Rect(1920 * (i - 1), 0, 1920, 1080)) for i in range(3)]
STACKED = [
    Monitor(0, "top", Rect(0, -1080, 1920, 1080)),
    Monitor(1, "bottom", Rect(0, 0, 1920, 1080)),
]


def synth_features(
    points: np.ndarray,
    rng: np.random.Generator,
    noise: float = 0.0,
    camera_x: float = 1920.0,
) -> np.ndarray:
    """8-D features (yaw, pitch, roll, tx, ty, tz, iris_h, iris_v) for gaze ``points``."""
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


def grid_dataset(
    monitors: list[Monitor],
    rng: np.random.Generator,
    noise: float = 0.0,
    per_point: int = 20,
    camera_x: float = 1920.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibration-like data: a 3x3 grid per monitor, ``per_point`` samples each."""
    points, groups = [], []
    for m in monitors:
        for ny in (0.1, 0.5, 0.9):
            for nx in (0.1, 0.5, 0.9):
                points += [m.rect.denormalize(nx, ny)] * per_point
                groups += [len(groups) // per_point] * per_point
    P = np.array(points)
    return synth_features(P, rng, noise, camera_x), P, np.array(groups)


def random_points(monitors: list[Monitor], rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform points on the monitors, avoiding the outer 5 % (like real use)."""
    idx = rng.integers(0, len(monitors), n)
    nx, ny = rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n)
    return np.array(
        [monitors[i].rect.denormalize(a, b) for i, a, b in zip(idx, nx, ny, strict=True)]
    )


def pixel_errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.hypot(pred[:, 0] - truth[:, 0], pred[:, 1] - truth[:, 1])


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
