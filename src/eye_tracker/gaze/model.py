"""Regression from a vision feature vector to a point on the virtual desktop.

The mapping between head pose / iris position and the point being looked at is
smooth but non-linear (perspective, eye-ball rotation, head/eye coordination) and
differs per person and per desk set-up. A ridge regression on standardised
features with low-order polynomial terms captures it well from a few hundred
calibration samples, fits in about a millisecond and predicts in microseconds,
which matters because prediction runs for every analysed frame.

Design matrices by ``degree``:

* ``1``: ``[1, z_i]``
* ``2``: ``[1, z_i, z_j*z_k (j <= k)]``
* ``3``: degree 2 plus the pure cubes ``z_j**3``. Looking at a flat screen from
  close up maps angles to pixels through ``tan``, which bends noticeably beyond
  about 30 degrees (wide or triple-monitor desks). Pure cubes model that odd,
  one-sided curvature at the cost of only ``k`` extra terms; the full cubic
  expansion would add ``O(k**3)`` terms and overfit.

``j`` and ``k`` run over the *nonlinear* features only (:attr:`GazeModel.nonlinear`;
all features when it is ``None``, the behaviour of models saved before it
existed). A backend declares which of its features encode gaze direction (head
rotation, iris offsets) and only those get products and cubes. Head position and
roll barely vary while the user calibrates, so cross-validation, which holds out
a dot but never a posture, cannot tell that a curved fit through them is noise,
and such a fit falls apart as soon as the user sits lower or leans back. Their
linear terms still correct the gaze for moderate posture changes.

Four details keep the model well behaved:

* Targets are normalised to the virtual-desktop bounds, so the regularisation
  strength ``alpha`` means the same thing on a 1080p laptop and a triple-4K desk.
* The intercept is not penalised, so shrinkage pulls predictions towards the
  mean gaze point rather than towards the desktop origin.
* Features are clipped to the calibrated range (widened by a margin) before the
  polynomial expansion. A pose never seen during calibration (for example a head
  roll the user did not make while calibrating) would otherwise be squared or
  cubed and could throw the prediction thousands of pixels off.
* Clipping hides how far outside the calibrated range a frame lies, and the
  clipped polynomial may even bend back onto the screens. :meth:`GazeModel.looks_away`
  therefore reports gaze far off every monitor (the user reads a phone on the
  desk or turns to a colleague), so such frames are treated as looking away
  instead of as gaze at the nearest monitor.

Looking away is judged from the *combined* gaze direction: a second, purely
linear ridge fit on the same standardised features, evaluated without clipping
the gaze-direction features (head position and roll stay clipped, so leaning
back or sitting lower cannot throw it off). Linear extrapolation is monotone,
so gaze well below or beside the screens lands well outside them. Head rotation
and eye-in-head offset are complementary - the same point can be looked at with
more head and less eye - and a fit on both keeps its estimate on the screens
when the user shares the work differently than while calibrating, where testing
each feature against its own calibrated range would call such gaze "away". The
strong ridge keeps the fit from leaning on either of two nearly collinear
features. Gaze at a monitor whose calibration dots were all skipped (which
``calibration.evaluate`` reports) is no longer "away" merely because it lies
beyond the calibrated range; the clipped model places it on or next to that
monitor. Models saved before this estimate existed fall back to the per-feature
range test (see :data:`LOOK_AWAY_EXCESS`).

One linear map serves the whole desk, but a turn of the gaze covers more pixels
on a dense monitor (a 4K panel next to a 1080p one: Windows and X11 report
native pixels) than on a coarse one. The map then overshoots on the coarse
monitor, whose outer edge it places well outside that monitor, most of all
when the user leans in. :meth:`GazeModel.fit` therefore also learns, from the
calibration samples of each monitor, where the linear estimate actually puts
that monitor (:class:`AwayRegion`); gaze near that place is never "away"
either. On a desk of equal monitors the place is the monitor itself, so the
test only ever becomes more lenient than with the monitor rectangles alone.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import Monitor, Rect, nearest_monitor

log = logging.getLogger(__name__)

#: Identifier written by :meth:`GazeModel.to_dict`; lets future model types coexist.
MODEL_KIND = "ridge-poly"

#: Standard deviations below this are treated as this value (constant features).
STD_FLOOR = 1e-6

#: Features are clipped to the training range widened by this fraction of the
#: range on each side. Calibration targets stop short of the screen edges, so a
#: generous margin is needed to still reach the edges and corners.
CLIP_MARGIN = 0.5

DEFAULT_ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
SUPPORTED_DEGREES: tuple[int, ...] = (1, 2, 3)

#: A frame looks away from every monitor when the linear gaze estimate (see the
#: module docstring) lies more than this fraction of the nearest monitor's
#: diagonal outside that monitor: about 12 cm beyond the edge of a 24" screen.
#: In synthetic tests (two, three and stacked monitors; posture changes of
#: 5-12 cm, and users who turn the head 25 % more or less than while
#: calibrating) at most 0.1 % of on-screen frames, edges and corners included,
#: exceed it; the worst case tried - nodding 40 % more while sitting 5 cm
#: higher - flags up to 5 % of the frames on the taskbar, where the per-feature
#: test flagged up to 80 %. Over 90 % of the frames of gaze 30 cm below the
#: screens and practically all at 40-80 cm (a phone or papers on the desk)
#: exceed it. Treated as "no gaze", every such frame restarts the switch dwell,
#: so even partial detection keeps a glance at the phone from switching monitors.
LOOK_AWAY_MARGIN = 0.2

#: Ridge strength of that linear estimate (standardised features, targets
#: normalised to the desktop). Stronger than the gaze model's usual choice:
#: head and eye features are nearly collinear during a calibration, and an
#: even split between them keeps the estimate on the screens when the user
#: later moves the head more or less, at a negligible cost in slope.
LOOK_AWAY_ALPHA = 10.0

#: Fallback for models saved without the linear estimate: a frame looks away
#: when one of its gaze-direction features lies more than this fraction of its
#: calibrated range beyond that range. Calibration dots stop 10 % short of the
#: screen edges, so gaze at an edge lies about 0.125 of the range beyond it.
#: This test is weaker (about half of the frames of gaze 40 cm below the
#: screens) and flags on-screen gaze once the user moves the head more or less
#: than while calibrating; recalibrating, or a refit by adaptive learning,
#: replaces it with the linear estimate.
LOOK_AWAY_EXCESS = 0.25

#: An :class:`AwayRegion` is learned from a monitor's samples only when their
#: targets (at least three distinct ones) span at least this fraction of the
#: monitor in both directions; otherwise the corners would be extrapolated from
#: too narrow a base.
REGION_MIN_SPAN = 0.25

#: ...and only when the image of the monitor is at most this many times its
#: diagonal: a larger one is a fit gone wrong, not a density difference (4K
#: next to 1080p scales by about 1.3).
REGION_MAX_SCALE = 3.0

# A more complex candidate (higher degree, smaller alpha) must beat the simpler
# one by this relative margin in cross-validation to be chosen.
_PREFER_SIMPLER = 1e-3


@dataclass(frozen=True, slots=True)
class AwayRegion:
    """Where the linear look-away estimate places one calibrated monitor.

    ``rect`` is the monitor as calibrated. ``x0, y0, x1, y1`` bound both the
    rectangle and its image under the linear estimate (an affine fit through
    the monitor's samples); ``diagonal`` is the larger of the two diagonals.
    Gaze within ``margin * diagonal`` of the box does not look away (see
    :meth:`GazeModel.looks_away`).
    """

    rect: Rect
    x0: float
    y0: float
    x1: float
    y1: float
    diagonal: float

    @classmethod
    def of(cls, rect: Rect) -> AwayRegion:
        """The region of a monitor whose image is unknown: the monitor itself."""
        return cls(rect, rect.x, rect.y, rect.right, rect.bottom, rect.diagonal)

    def distance_outside(self, px: float, py: float) -> float:
        """Euclidean distance from the point to the box (0 when inside)."""
        dx = max(self.x0 - px, 0.0, px - self.x1)
        dy = max(self.y0 - py, 0.0, py - self.y1)
        return math.hypot(dx, dy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rect": self.rect.to_list(),
            "box": [self.x0, self.y0, self.x1, self.y1],
            "diagonal": self.diagonal,
        }

    @classmethod
    def from_dict(cls, d: Any) -> AwayRegion:
        """Inverse of :meth:`to_dict`. Raises ``ValueError`` on malformed input."""
        if not isinstance(d, dict):
            raise ValueError("away region must be an object")
        try:
            rect = Rect.from_list(d["rect"])
            x0, y0, x1, y1 = (float(v) for v in d["box"])
            diagonal = float(d["diagonal"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid away region: {exc}") from exc
        values = (x0, y0, x1, y1, diagonal)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("away region contains non-finite numbers")
        if rect.w <= 0 or rect.h <= 0 or x1 < x0 or y1 < y0 or diagonal <= 0.0:
            raise ValueError("away region is inconsistent")
        return cls(rect, x0, y0, x1, y1, diagonal)


class GazeModel:
    """Maps a feature vector to a gaze point on the virtual desktop (ridge regression on
    standardised features with optional polynomial terms, see the module docstring).

    ``nonlinear`` lists the indices of the features that get quadratic and cubic
    terms (the backend's gaze-direction features, see :func:`gaze_feature_indices`);
    the others enter linearly. ``None`` expands every feature. Indices are sorted
    and de-duplicated; they are checked against the feature count when fitting.
    """

    def __init__(
        self,
        degree: int = 2,
        alpha: float = 1.0,
        nonlinear: Sequence[int] | None = None,
    ) -> None:
        if degree not in SUPPORTED_DEGREES:
            raise ValueError(f"degree must be one of {SUPPORTED_DEGREES}, got {degree!r}")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"alpha must be a finite number >= 0, got {alpha!r}")
        self.degree = degree
        self.alpha = alpha
        self._nonlinear = _as_indices(nonlinear)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._lo: np.ndarray | None = None
        self._hi: np.ndarray | None = None
        self._coef: np.ndarray | None = None  # (n_terms, 2), in normalised target space
        # (1 + d, 2): the linear estimate behind looks_away (None: saved without it).
        self._linear_coef: np.ndarray | None = None
        # Where that estimate places each calibrated monitor (empty: fitted without
        # the monitors, or saved before the regions existed).
        self._away_regions: tuple[AwayRegion, ...] = ()
        self._bounds: Rect | None = None

    # ---------------------------------------------------------------- properties
    @property
    def is_fitted(self) -> bool:
        return self._coef is not None

    @property
    def n_features(self) -> int:
        """Length of the feature vectors the model was fitted on (0 if unfitted)."""
        return 0 if self._mean is None else int(self._mean.shape[0])

    @property
    def nonlinear(self) -> tuple[int, ...] | None:
        """Indices of the features with polynomial terms (``None``: all of them)."""
        return self._nonlinear

    @property
    def bounds(self) -> Rect | None:
        """Virtual-desktop rectangle used to normalise targets (``None`` if unfitted)."""
        return self._bounds

    @property
    def away_regions(self) -> tuple[AwayRegion, ...]:
        """Where the linear look-away estimate places each monitor passed to
        :meth:`fit` as ``regions`` (see :class:`AwayRegion`)."""
        return self._away_regions

    def __repr__(self) -> str:
        state = f"fitted, {self.n_features} features" if self.is_fitted else "unfitted"
        nonlinear = "" if self._nonlinear is None else f", nonlinear={list(self._nonlinear)}"
        return f"GazeModel(degree={self.degree}, alpha={self.alpha:g}{nonlinear}, {state})"

    # ---------------------------------------------------------------- fitting
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        weights: np.ndarray | None = None,
        bounds: Rect | None = None,
        *,
        regions: Sequence[Rect] | None = None,
    ) -> GazeModel:
        """Fit the model and return ``self``.

        ``X`` is ``(n, d)`` features, ``Y`` is ``(n, 2)`` targets in global pixels and
        ``weights`` optional non-negative per-sample weights (e.g. 0.5 for samples
        learned from mouse use). ``bounds`` should be the virtual desktop; when
        omitted the bounding box of ``Y`` is used. ``regions`` are the monitor
        rectangles the samples were taken on; for each one, where the linear
        look-away estimate places it is learned (:attr:`away_regions`).
        """
        X_arr = _as_matrix(X, "X")
        Y_arr = _as_targets(Y, X_arr.shape[0])
        w = _as_weights(weights, X_arr.shape[0])
        bounds = _check_bounds(bounds, Y_arr)
        _check_indices(self._nonlinear, X_arr.shape[1])

        mean, std = _standardisation(X_arr)
        A = _design((X_arr - mean) / std, self.degree, self._nonlinear)
        AtW = A.T * w
        lhs, rhs = AtW @ A, AtW @ _normalise(Y_arr, bounds)
        self._coef = _solve_ridge(lhs, rhs, self.alpha)
        # Every design starts with the linear columns [1, z], so the linear fit
        # behind looks_away is the leading block of the same normal equations.
        k = 1 + X_arr.shape[1]
        self._linear_coef = _solve_ridge(lhs[:k, :k], rhs[:k], LOOK_AWAY_ALPHA)
        self._mean, self._std = mean, std
        self._lo, self._hi = X_arr.min(axis=0), X_arr.max(axis=0)
        self._bounds = bounds
        # The training samples lie within the calibrated range, so their linear
        # estimate needs no clipping: it is the leading block of the design.
        estimates = _denormalise(A[:, :k] @ self._linear_coef, bounds)
        self._away_regions = tuple(
            _away_region(rect, Y_arr, estimates, w) for rect in (regions or ())
        )
        return self

    # ---------------------------------------------------------------- prediction
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict gaze point(s) in global pixels: ``(d,) -> (2,)`` or ``(n, d) -> (n, 2)``."""
        if self._coef is None or self._mean is None or self._std is None or self._bounds is None:
            raise RuntimeError("GazeModel is not fitted")
        arr, X_arr = self._features(x)
        if self._lo is not None and self._hi is not None:
            span = self._hi - self._lo
            X_arr = np.clip(X_arr, self._lo - CLIP_MARGIN * span, self._hi + CLIP_MARGIN * span)
        Z = (X_arr - self._mean) / self._std
        Y = _denormalise(_design(Z, self.degree, self._nonlinear) @ self._coef, self._bounds)
        return Y[0] if arr.ndim == 1 else Y

    # ---------------------------------------------------------------- range checks
    def extrapolation(self, x: np.ndarray) -> np.ndarray:
        """How far each feature lies outside the calibrated range, in units of that range.

        ``0`` inside ``[lo, hi]`` (the minimum and maximum seen when fitting),
        ``0.25`` a quarter of the range below ``lo`` or above ``hi``, and so on;
        ``(d,) -> (d,)`` or ``(n, d) -> (n, d)``. All zeros for a model loaded
        from a file written before the range was stored, where it is unknown.
        """
        if self._mean is None:
            raise RuntimeError("GazeModel is not fitted")
        arr, X_arr = self._features(x)
        if self._lo is None or self._hi is None:
            excess = np.zeros_like(X_arr)
        else:
            span = np.maximum(self._hi - self._lo, STD_FLOOR)
            excess = np.maximum(np.maximum(self._lo - X_arr, X_arr - self._hi), 0.0) / span
        return excess[0] if arr.ndim == 1 else excess

    @property
    def has_linear_estimate(self) -> bool:
        """Whether :meth:`looks_away` can use the combined gaze direction (models
        fitted by this version; older files fall back to the per-feature test)."""
        return self._linear_coef is not None

    def looks_away(
        self,
        x: np.ndarray,
        features: Sequence[int] | None = None,
        threshold: float = LOOK_AWAY_EXCESS,
        *,
        monitors: Sequence[Monitor] | None = None,
        margin: float = LOOK_AWAY_MARGIN,
    ) -> bool:
        """Whether the feature vector ``x`` ``(d,)`` shows gaze away from every monitor.

        With ``monitors`` (the layout the model was calibrated for) and a model
        that has the linear estimate (:attr:`has_linear_estimate`): True when
        that estimate of the gaze point lies more than ``margin`` times the
        nearest monitor's diagonal outside it (see :data:`LOOK_AWAY_MARGIN` and
        the module docstring), and more than ``margin`` times the diagonal of
        every :attr:`away_regions` entry of those monitors outside that region.
        The gaze-direction ``features`` (indices; default :attr:`nonlinear`)
        extrapolate freely; the others are clipped to their calibrated range,
        since leaning back is not looking away.

        Otherwise the fallback: True when one of the gaze-direction features
        lies more than ``threshold`` of its calibrated range outside that range
        (see :data:`LOOK_AWAY_EXCESS`).

        Either way this is False when the gaze-direction features are unknown
        (``features`` and :attr:`nonlinear` both ``None``, as for models saved
        before ``nonlinear`` existed; pass the backend's indices for those), and
        for non-finite features: a broken frame never reads as "looking away".
        """
        indices = self._nonlinear if features is None else _as_indices(features)
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"expected one feature vector, got shape {arr.shape}")
        if monitors and self._linear_coef is not None:
            _, X_arr = self._features(arr)
            if not indices:
                return False
            _check_indices(indices, X_arr.shape[1])
            point = self._linear_estimate(X_arr[0], indices)
            if point is None:
                return False
            monitor, distance = nearest_monitor(monitors, *point)
            if distance <= margin * monitor.rect.diagonal:
                return False
            # Regions of other layouts (a profile used elsewhere) do not apply.
            rects = {m.rect for m in monitors}
            return not any(
                region.distance_outside(*point) <= margin * region.diagonal
                for region in self._away_regions
                if region.rect in rects
            )
        excess = self.extrapolation(arr)
        if not indices:
            return False
        _check_indices(indices, excess.shape[0])
        # NaN compares False, so a broken feature never reads as "looking away".
        return bool(np.max(excess[list(indices)]) > threshold)

    def _linear_estimate(
        self, x: np.ndarray, gaze_indices: tuple[int, ...]
    ) -> tuple[float, float] | None:
        """The linear gaze estimate for ``x`` ``(d,)`` in global pixels, with only the
        features other than ``gaze_indices`` clipped (``None`` if not finite)."""
        assert self._linear_coef is not None
        assert self._mean is not None
        assert self._std is not None
        assert self._bounds is not None
        if self._lo is not None and self._hi is not None:
            span = self._hi - self._lo
            clipped = np.clip(x, self._lo - CLIP_MARGIN * span, self._hi + CLIP_MARGIN * span)
            free = np.zeros(x.shape[0], dtype=bool)
            free[list(gaze_indices)] = True
            x = np.where(free, x, clipped)
        z = (x - self._mean) / self._std
        px, py = _denormalise(np.concatenate(([1.0], z)) @ self._linear_coef, self._bounds)
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        return (float(px), float(py))

    def _features(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(input as an array, input as an (n, d) matrix)``, with the length checked."""
        assert self._mean is not None
        arr = np.asarray(x, dtype=np.float64)
        X_arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
        if X_arr.ndim != 2 or X_arr.shape[1] != self._mean.shape[0]:
            raise ValueError(
                f"expected feature vector(s) of length {self._mean.shape[0]}, got shape {arr.shape}"
            )
        return arr, X_arr

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation (plain lists and numbers)."""
        out: dict[str, Any] = {"kind": MODEL_KIND, "degree": self.degree, "alpha": self.alpha}
        if self._nonlinear is not None:
            # Absent means "every feature", which is how older files are read.
            out["nonlinear"] = list(self._nonlinear)
        if (
            self._coef is not None
            and self._bounds is not None
            and self._mean is not None
            and self._std is not None
            and self._lo is not None
            and self._hi is not None
        ):
            out.update(
                {
                    "mean": self._mean.tolist(),
                    "std": self._std.tolist(),
                    "lo": self._lo.tolist(),
                    "hi": self._hi.tolist(),
                    "coef": self._coef.tolist(),
                    "bounds": self._bounds.to_list(),
                }
            )
            if self._linear_coef is not None:
                # Absent in older files, which then use the per-feature test.
                out["linear_coef"] = self._linear_coef.tolist()
            if self._away_regions:
                # Absent in older files: the monitor rectangles alone are used.
                out["away_regions"] = [region.to_dict() for region in self._away_regions]
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GazeModel:
        """Inverse of :meth:`to_dict`. Raises ``ValueError`` on malformed input."""
        if not isinstance(d, dict):
            raise ValueError("model must be an object")
        kind = d.get("kind", MODEL_KIND)
        if kind != MODEL_KIND:
            raise ValueError(f"unsupported model kind {kind!r}")
        try:
            model = cls(
                degree=int(d.get("degree", 2)),
                alpha=float(d.get("alpha", 1.0)),
                nonlinear=_index_list(d.get("nonlinear")),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid model parameters: {exc}") from exc
        if "coef" not in d:
            return model  # an unfitted model was saved

        try:
            mean = _vector(d["mean"], "mean")
            std = _vector(d["std"], "std")
            coef = np.asarray(d["coef"], dtype=np.float64)
            bounds = Rect.from_list(d["bounds"])
            lo = _vector(d["lo"], "lo") if "lo" in d else None
            hi = _vector(d["hi"], "hi") if "hi" in d else None
            linear = (
                np.asarray(d["linear_coef"], dtype=np.float64)
                if d.get("linear_coef") is not None
                else None
            )
            raw_regions = d.get("away_regions") or []
            if not isinstance(raw_regions, list):
                raise TypeError("away_regions must be a list")
            regions = tuple(AwayRegion.from_dict(r) for r in raw_regions)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            # OverflowError: int(inf) for a bounds value such as 1e400 or Infinity.
            raise ValueError(f"invalid model data: {exc}") from exc

        n = mean.shape[0]
        if n == 0 or std.shape != (n,):
            raise ValueError("model arrays have inconsistent shapes")
        _check_indices(model.nonlinear, n)
        if coef.shape != (_n_terms(n, model.degree, model.nonlinear), 2):
            raise ValueError("model arrays have inconsistent shapes")
        if (lo is None) != (hi is None):
            raise ValueError("model clip range is incomplete")
        arrays = [mean, std, coef]
        if lo is not None and hi is not None:
            if lo.shape != (n,) or hi.shape != (n,) or np.any(hi < lo):
                raise ValueError("model clip range is inconsistent")
            arrays += [lo, hi]
        if linear is not None:
            if linear.shape != (1 + n, 2):
                raise ValueError("model arrays have inconsistent shapes")
            arrays.append(linear)
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("model contains non-finite numbers")
        if np.any(std <= 0.0) or bounds.w <= 0 or bounds.h <= 0:
            raise ValueError("model scale parameters must be positive")

        model._mean, model._std, model._lo, model._hi = mean, std, lo, hi
        model._coef, model._bounds, model._linear_coef = coef, bounds, linear
        model._away_regions = regions
        return model


# ------------------------------------------------------------ model selection
def gaze_feature_indices(
    feature_names: Sequence[str], gaze_features: Sequence[str]
) -> tuple[int, ...] | None:
    """Indices of ``gaze_features`` within ``feature_names`` (a backend's
    ``VisionBackend.feature_names`` and ``gaze_features``).

    The result is what :class:`GazeModel`, :func:`select_model` and
    ``calibration.evaluate`` take as ``nonlinear`` and :meth:`GazeModel.looks_away`
    as ``features``. ``None`` (every feature is nonlinear, gaze direction unknown)
    when the backend declares no gaze features. Names missing from
    ``feature_names`` are ignored with a warning rather than raised on, so a
    backend declaration error degrades the model instead of stopping tracking.
    """
    names = list(feature_names)
    indices = []
    for name in gaze_features:
        if name in names:
            indices.append(names.index(name))
        else:
            log.warning("Gaze feature %r is not one of the features %s; ignored", name, names)
    return tuple(sorted(set(indices))) or None


def select_alpha(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    degree: int = 2,
    bounds: Rect | None = None,
    *,
    weights: np.ndarray | None = None,
    nonlinear: Sequence[int] | None = None,
) -> float:
    """Pick the ridge strength with the lowest leave-one-group-out mean pixel error.

    ``groups`` holds one id per sample (the calibration point id), so every fold
    predicts a point the model has never seen. That mirrors real use far better
    than per-sample cross-validation, where near-duplicate frames of the same point
    leak into the training set. Returns the middle alpha when fewer than two groups
    exist. Near-ties go to the larger alpha (the smoother model). ``nonlinear`` is
    passed to the candidate models (see :class:`GazeModel`).
    """
    selection = select_model(
        X,
        Y,
        groups,
        alphas=alphas,
        degrees=(degree,),
        bounds=bounds,
        weights=weights,
        nonlinear=nonlinear,
    )
    return selection.alpha


@dataclass(frozen=True, eq=False, slots=True)
class ModelSelection:
    """Outcome of :func:`select_model`."""

    degree: int
    alpha: float
    #: Mean leave-one-group-out error of the winner (``nan`` without cross-validation).
    mean_error_px: float
    #: Leave-one-group-out predictions ``(n, 2)`` of the winner (``None`` without CV).
    predictions: np.ndarray | None = field(default=None, repr=False)


def select_model(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    degrees: Sequence[int] = SUPPORTED_DEGREES,
    bounds: Rect | None = None,
    weights: np.ndarray | None = None,
    nonlinear: Sequence[int] | None = None,
) -> ModelSelection:
    """Choose ``degree`` and ``alpha`` jointly by leave-one-group-out cross-validation.

    Simpler candidates win near-ties (lower degree first, then larger alpha). With
    fewer than two groups no cross-validation is possible and the lowest listed
    degree (at most 2) with the middle alpha is returned. ``nonlinear`` selects
    the features with polynomial terms, as for :class:`GazeModel`; build the final
    model with the same value.
    """
    alpha_list = sorted({float(a) for a in alphas}, reverse=True)
    degree_list = sorted({int(d) for d in degrees})
    if not alpha_list or not degree_list:
        raise ValueError("alphas and degrees must not be empty")
    for a in alpha_list:
        if not math.isfinite(a) or a < 0.0:
            raise ValueError(f"alpha must be a finite number >= 0, got {a!r}")
    for deg in degree_list:
        if deg not in SUPPORTED_DEGREES:
            raise ValueError(f"degree must be one of {SUPPORTED_DEGREES}, got {deg!r}")

    X_arr = _as_matrix(X, "X")
    Y_arr = _as_targets(Y, X_arr.shape[0])
    g = _as_groups(groups, X_arr.shape[0])
    w = _as_weights(weights, X_arr.shape[0])
    indices = _as_indices(nonlinear)
    _check_indices(indices, X_arr.shape[1])
    # One normalisation for every fold, so alpha means the same in each of them
    # and in the final fit.
    bounds = _check_bounds(bounds, Y_arr)
    middle_alpha = sorted(alpha_list)[len(alpha_list) // 2]

    if np.unique(g).shape[0] < 2:
        degree = min(degree_list[0], 2)
        log.debug("select_model: fewer than two groups; degree=%d alpha=%g", degree, middle_alpha)
        return ModelSelection(degree, middle_alpha, math.nan, None)

    best: ModelSelection | None = None
    for degree in degree_list:
        folds = _LopoFolds(X_arr, Y_arr, g, w, degree=degree, bounds=bounds, nonlinear=indices)
        for alpha in alpha_list:
            preds = folds.predict(alpha)
            err = _mean_error(preds, Y_arr)
            log.debug("select_model: degree=%d alpha=%g -> %.1f px", degree, alpha, err)
            if not math.isfinite(err):
                continue
            if best is None or err < best.mean_error_px * (1.0 - _PREFER_SIMPLER):
                best = ModelSelection(degree, alpha, err, preds)
    if best is None:
        log.warning("select_model: every candidate diverged; using the defaults")
        return ModelSelection(min(degree_list[0], 2), middle_alpha, math.nan, None)
    return best


def lopo_predictions(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    *,
    degree: int = 2,
    bounds: Rect | None = None,
    weights: np.ndarray | None = None,
    nonlinear: Sequence[int] | None = None,
) -> np.ndarray:
    """Leave-one-group-out predictions ``(n, 2)`` in global pixels.

    Each group is predicted by a model trained on all other groups. Rows of a
    group whose complement carries no weight are ``nan``. Needs two groups.
    """
    X_arr = _as_matrix(X, "X")
    Y_arr = _as_targets(Y, X_arr.shape[0])
    g = _as_groups(groups, X_arr.shape[0])
    w = _as_weights(weights, X_arr.shape[0])
    indices = _as_indices(nonlinear)
    _check_indices(indices, X_arr.shape[1])
    if np.unique(g).shape[0] < 2:
        raise ValueError("leave-one-group-out needs at least two groups")
    if degree not in SUPPORTED_DEGREES:
        raise ValueError(f"degree must be one of {SUPPORTED_DEGREES}, got {degree!r}")
    folds = _LopoFolds(
        X_arr, Y_arr, g, w, degree=degree, bounds=_check_bounds(bounds, Y_arr), nonlinear=indices
    )
    return folds.predict(alpha)


class _LopoFolds:
    """Leave-one-group-out cross-validation that costs little more than one fit.

    The normal equations of a fold are the full ones minus the held-out group's
    contribution (``AᵀWA - A_gᵀW_gA_g``), so the design matrix is built once and
    each (fold, alpha) pair needs only a small ``p x p`` solve. Standardisation
    uses all samples; the resulting leakage of one mean/std estimate into the
    folds is negligible compared with the per-point effects being measured.
    Held-out rows are not clipped to the fold's feature range the way
    :meth:`GazeModel.predict` would; that only matters for a held-out point far
    outside every other point, where the estimate is then somewhat pessimistic.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        groups: np.ndarray,
        w: np.ndarray,
        *,
        degree: int,
        bounds: Rect,
        nonlinear: tuple[int, ...] | None = None,
    ) -> None:
        mean, std = _standardisation(X)
        A = _design((X - mean) / std, degree, nonlinear)
        Yn = _normalise(Y, bounds)
        AtW = A.T * w
        lhs, rhs = AtW @ A, AtW @ Yn
        total_w = float(w.sum())
        self._bounds = bounds
        self._n = X.shape[0]
        self._folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
        for gid in np.unique(groups):
            mask = groups == gid
            Ag, wg = A[mask], w[mask]
            AgtW = Ag.T * wg
            trainable = total_w - float(wg.sum()) > 1e-12
            self._folds.append((mask, lhs - AgtW @ Ag, rhs - AgtW @ Yn[mask], Ag, trainable))

    def predict(self, alpha: float) -> np.ndarray:
        preds = np.full((self._n, 2), np.nan)
        for mask, lhs, rhs, Ag, trainable in self._folds:
            if trainable:
                preds[mask] = Ag @ _solve_ridge(lhs, rhs, alpha)
        return _denormalise(preds, self._bounds)


# --------------------------------------------------------------------- internals
def _away_region(rect: Rect, Y: np.ndarray, estimates: np.ndarray, w: np.ndarray) -> AwayRegion:
    """Where the linear look-away estimate places the monitor ``rect``.

    ``Y`` are the training targets, ``estimates`` their linear estimates and ``w``
    the sample weights. The estimates of the samples on ``rect`` are fitted as an
    affine function of their targets (weighted least squares), which maps the
    corners of ``rect`` to its image. The region bounds the rectangle and that
    image, so it is never smaller than the monitor itself; without enough
    samples to place the corners reliably it is the monitor itself.
    """
    region = AwayRegion.of(rect)
    on_rect = (
        (Y[:, 0] >= rect.x)
        & (Y[:, 0] < rect.right)
        & (Y[:, 1] >= rect.y)
        & (Y[:, 1] < rect.bottom)
        & (w > 0.0)
    )
    targets = Y[on_rect]
    if np.unique(targets, axis=0).shape[0] < 3:
        return region
    span = targets.max(axis=0) - targets.min(axis=0)
    if span[0] < REGION_MIN_SPAN * rect.w or span[1] < REGION_MIN_SPAN * rect.h:
        return region
    root_w = np.sqrt(w[on_rect])[:, None]
    design = np.column_stack([targets, np.ones(targets.shape[0])])
    affine, _, rank, _ = np.linalg.lstsq(design * root_w, estimates[on_rect] * root_w, rcond=None)
    if rank < 3:
        return region
    corners = np.array(
        [
            [rect.x, rect.y, 1.0],
            [rect.right, rect.y, 1.0],
            [rect.right, rect.bottom, 1.0],
            [rect.x, rect.bottom, 1.0],
        ]
    )
    image = corners @ affine
    if not np.all(np.isfinite(image)):
        return region
    diagonal = max(float(np.hypot(*(image[2] - image[0]))), float(np.hypot(*(image[3] - image[1]))))
    if diagonal > REGION_MAX_SCALE * rect.diagonal:
        log.debug("Ignoring an implausible look-away region for %s", rect)
        return region
    x0, y0 = image.min(axis=0)
    x1, y1 = image.max(axis=0)
    return AwayRegion(
        rect,
        min(float(rect.x), float(x0)),
        min(float(rect.y), float(y0)),
        max(float(rect.right), float(x1)),
        max(float(rect.bottom), float(y1)),
        max(rect.diagonal, diagonal),
    )


def _n_terms(n_features: int, degree: int, nonlinear: tuple[int, ...] | None = None) -> int:
    k = n_features if nonlinear is None else len(nonlinear)
    n = 1 + n_features
    if degree >= 2:
        n += k * (k + 1) // 2
    if degree >= 3:
        n += k
    return n


@functools.lru_cache(maxsize=8)
def _pair_indices(d: int) -> tuple[np.ndarray, np.ndarray]:
    iu, ju = np.triu_indices(d)
    return iu, ju


def _design(Z: np.ndarray, degree: int, nonlinear: tuple[int, ...] | None = None) -> np.ndarray:
    n = Z.shape[0]
    cols = [np.ones((n, 1)), Z]
    if degree >= 2:
        # A list index selects columns; a tuple would index dimensions.
        Zn = Z if nonlinear is None else Z[:, list(nonlinear)]
        k = Zn.shape[1]
        if k:
            iu, ju = _pair_indices(k)
            cols.append(Zn[:, iu] * Zn[:, ju])
            if degree >= 3:
                cols.append(Zn**3)
    return np.hstack(cols)


def _standardisation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(axis=0), np.maximum(X.std(axis=0), STD_FLOOR)


def _solve_ridge(lhs: np.ndarray, rhs: np.ndarray, alpha: float) -> np.ndarray:
    """Solve ``(lhs + αI')β = rhs`` where ``I'`` is the identity without the intercept."""
    system = lhs.copy()
    idx = np.arange(1, system.shape[0])
    system[idx, idx] += alpha
    try:
        coef = np.linalg.solve(system, rhs)
        if np.all(np.isfinite(coef)):
            return coef
    except np.linalg.LinAlgError:
        pass
    # Singular system (alpha == 0 with collinear or too few samples): the
    # minimum-norm solution is the most conservative choice.
    log.debug("ridge system singular; falling back to least squares")
    return np.linalg.lstsq(system, rhs, rcond=None)[0]


def _mean_error(preds: np.ndarray, Y: np.ndarray) -> float:
    err = np.hypot(preds[:, 0] - Y[:, 0], preds[:, 1] - Y[:, 1])
    if np.all(np.isnan(err)):
        return math.nan
    return float(np.nanmean(err))


def _normalise(Y: np.ndarray, bounds: Rect) -> np.ndarray:
    return (Y - (bounds.x, bounds.y)) / (bounds.w, bounds.h)


def _denormalise(Yn: np.ndarray, bounds: Rect) -> np.ndarray:
    return Yn * (bounds.w, bounds.h) + (bounds.x, bounds.y)


def _check_bounds(bounds: Rect | None, Y: np.ndarray) -> Rect:
    if bounds is None:
        x0, y0 = np.floor(Y.min(axis=0))
        x1, y1 = np.ceil(Y.max(axis=0))
        return Rect(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0)))
    if bounds.w <= 0 or bounds.h <= 0:
        raise ValueError(f"bounds must have a positive size, got {bounds}")
    return bounds


def _as_matrix(X: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty (n, d) array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_targets(Y: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(Y, dtype=np.float64)
    if arr.shape != (n, 2):
        raise ValueError(f"Y must have shape ({n}, 2), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Y contains non-finite values")
    return arr


def _as_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be finite, non-negative and not all zero")
    return w


def _as_groups(groups: np.ndarray, n: int) -> np.ndarray:
    g = np.asarray(groups).reshape(-1)
    if g.shape != (n,):
        raise ValueError(f"groups must have shape ({n},), got {g.shape}")
    return g


def _vector(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a list of numbers")
    return arr


def _as_indices(indices: Sequence[int] | None) -> tuple[int, ...] | None:
    """Sorted, de-duplicated, non-negative feature indices (``None`` passes through)."""
    if indices is None:
        return None
    out = set()
    for i in indices:
        # bool is an int, and numpy integers are not; accept the latter only.
        if isinstance(i, bool) or not isinstance(i, int | np.integer):
            raise ValueError(f"feature indices must be integers, got {i!r}")
        if i < 0:
            raise ValueError(f"feature indices must be >= 0, got {i!r}")
        out.add(int(i))
    return tuple(sorted(out))


def _check_indices(indices: tuple[int, ...] | None, n_features: int) -> None:
    if indices and indices[-1] >= n_features:
        raise ValueError(f"feature index {indices[-1]} is out of range for {n_features} features")


def _index_list(value: Any) -> tuple[int, ...] | None:
    """``nonlinear`` as read from JSON: absent/null or a list of integers."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("nonlinear must be a list of feature indices")
    return _as_indices(value)
