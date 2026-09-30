"""Calibration: where to show targets, how to collect samples, how good the result is.

The flow driven by the calibration window is::

    plan = make_plan(monitors)                       # dots to show, in order
    collector = CalibrationCollector(plan)
    collector.start(now)
    ...every UI tick:   events = collector.update(now)
    ...every frame:     collector.add(observation)
    model, report = evaluate(collector.samples, monitors)

Everything here is plain Python/numpy and driven by an injected clock, so the
whole procedure is unit-testable without a camera or a screen.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..types import Monitor, Observation, nearest_monitor, virtual_bounds
from .model import DEFAULT_ALPHAS, SUPPORTED_DEGREES, GazeModel, lopo_predictions, select_model

log = logging.getLogger(__name__)

# Collector phases.
PHASE_IDLE = "idle"
PHASE_SETTLE = "settle"
PHASE_COLLECT = "collect"
PHASE_DONE = "done"

# Events returned by CalibrationCollector.update().
EVENT_TARGET = "target"
EVENT_RETRY = "retry"
EVENT_FINISHED = "finished"

#: Minimum monitor accuracy for each grade, best first.
GRADE_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("excellent", 0.97),
    ("good", 0.90),
    ("fair", 0.75),
)

MIN_SAMPLES = 10
MIN_POINTS = 3

# Timer ticks are sums of floats (11.8 + 0.8 = 12.599999…); without a little
# slack a phase could end one tick late.
_TIME_EPS = 1e-6


# ------------------------------------------------------------------------ plan
@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """One dot of the calibration sequence.

    ``nx``/``ny`` are relative to the monitor (0..1), ``x``/``y`` are global
    coordinates of the same point.
    """

    point_id: int
    monitor_index: int
    nx: float
    ny: float
    x: float
    y: float


def make_plan(
    monitors: Sequence[Monitor],
    points_per_monitor: int = 9,
    margin: float = 0.1,
) -> list[CalibrationTarget]:
    """Calibration targets for every monitor, in the order they should be shown.

    ``points_per_monitor``: 1 (centre), 5 (centre + 4 corners) or a square number
    ``k*k`` (a ``k x k`` grid spanning ``margin .. 1 - margin``). Monitors are
    visited left to right (then top to bottom). Within a monitor the grid is
    walked as a serpentine; the orientation of that walk is chosen so that it
    starts at the corner nearest to where the previous monitor ended, which keeps
    eye and head travel short. ``point_id`` counts up from 0.
    """
    if not 0.0 <= margin < 0.5:
        raise ValueError(f"margin must be in [0, 0.5), got {margin!r}")
    base = _monitor_path(points_per_monitor, margin)
    variants = [
        base,
        [(1.0 - nx, ny) for nx, ny in base],
        [(nx, 1.0 - ny) for nx, ny in base],
        [(1.0 - nx, 1.0 - ny) for nx, ny in base],
    ]

    plan: list[CalibrationTarget] = []
    last: tuple[float, float] | None = None
    for monitor in sorted(monitors, key=lambda m: (m.rect.x, m.rect.y, m.index)):
        rect = monitor.rect
        path = base
        if last is not None:
            distances = [math.dist(rect.denormalize(*v[0]), last) for v in variants]
            # index() returns the first of equally good variants, so the canonical
            # top-left start wins ties.
            path = variants[distances.index(min(distances))]
        for nx, ny in path:
            x, y = rect.denormalize(nx, ny)
            plan.append(CalibrationTarget(len(plan), monitor.index, nx, ny, x, y))
        last = (plan[-1].x, plan[-1].y)
    return plan


def _monitor_path(points: int, margin: float) -> list[tuple[float, float]]:
    lo, hi = margin, 1.0 - margin
    if points == 1:
        return [(0.5, 0.5)]
    if points == 5:
        # Centre first (easiest to find), then the corners as a loop.
        return [(0.5, 0.5), (lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    k = math.isqrt(points) if points > 0 else 0
    if k < 2 or k * k != points:
        raise ValueError(f"points_per_monitor must be 1, 5 or a square number >= 4, got {points!r}")
    coords = np.linspace(lo, hi, k).tolist()
    path: list[tuple[float, float]] = []
    for row, ny in enumerate(coords):
        xs = coords if row % 2 == 0 else coords[::-1]
        path.extend((nx, ny) for nx in xs)
    return path


# --------------------------------------------------------------------- samples
@dataclass(eq=False)
class CalibrationSample:
    """A feature vector labelled with the global point the user was looking at.

    ``point_id`` groups samples of one calibration dot (implicit samples learned
    from mouse use carry negative ids). ``weight`` scales the sample's influence
    on the fit.
    """

    features: np.ndarray
    x: float
    y: float
    monitor_index: int
    point_id: int
    weight: float = 1.0


def samples_to_arrays(
    samples: Iterable[CalibrationSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack samples into ``X (n, d)``, targets ``Y (n, 2)`` and ``weights (n,)``."""
    items = list(samples)
    if not items:
        raise ValueError("no samples")
    X = np.vstack([np.asarray(s.features, dtype=np.float64).reshape(1, -1) for s in items])
    Y = np.array([(s.x, s.y) for s in items], dtype=np.float64)
    W = np.array([s.weight for s in items], dtype=np.float64)
    return X, Y, W


# ------------------------------------------------------------------- collector
class CalibrationCollector:
    """Time-based state machine that shows each target and gathers its samples.

    Per target: ``settle`` (the eyes travel to the new dot; nothing is recorded)
    then ``collect`` (usable observations are recorded). A target that gathered
    fewer than ``min_samples`` when its collect window ends gets up to
    ``max_retries`` extra collect windows (samples so far are kept) and is then
    skipped, dropping its samples.

    Every phase starts at the ``now`` of the :meth:`update` call that entered it,
    so a stalled UI thread never eats into the next target's time. Progress
    properties describe the state as of the last :meth:`start`/:meth:`update`.
    """

    def __init__(
        self,
        plan: Sequence[CalibrationTarget],
        settle_s: float = 0.8,
        collect_s: float = 1.0,
        min_samples: int = 5,
        max_retries: int = 1,
    ) -> None:
        if settle_s < 0 or collect_s <= 0:
            raise ValueError("settle_s must be >= 0 and collect_s > 0")
        if min_samples < 1 or max_retries < 0:
            raise ValueError("min_samples must be >= 1 and max_retries >= 0")
        self._plan: tuple[CalibrationTarget, ...] = tuple(plan)
        self.settle_s = float(settle_s)
        self.collect_s = float(collect_s)
        self.min_samples = int(min_samples)
        self.max_retries = int(max_retries)
        self._reset()

    def _reset(self) -> None:
        self._phase = PHASE_IDLE
        self._index = 0
        self._phase_start = 0.0
        self._now = 0.0
        self._retries = 0
        self._pending: list[CalibrationSample] = []
        self._samples: list[CalibrationSample] = []
        self._skipped: list[int] = []
        self._events: list[str] = []
        self._n_features: int | None = None

    # ------------------------------------------------------------- control
    def start(self, now: float) -> None:
        """(Re)start from the first target. The next :meth:`update` reports ``"target"``
        (or ``"finished"`` for an empty plan)."""
        self._reset()
        self._now = now
        if not self._plan:
            self._phase = PHASE_DONE
            self._events.append(EVENT_FINISHED)
            return
        self._enter(PHASE_SETTLE, now)
        self._events.append(EVENT_TARGET)

    def update(self, now: float) -> list[str]:
        """Advance the timers; returns the events since the last call, in order:
        ``"target"`` (a new target is current), ``"retry"`` (the current target gets
        another collect window) and ``"finished"``."""
        if self._phase in (PHASE_SETTLE, PHASE_COLLECT):
            self._now = max(self._now, now)
            # Loop so zero-length phases (settle_s == 0) pass in a single call.
            while self._step(self._now):
                pass
        events, self._events = self._events, []
        return events

    def add(self, obs: Observation) -> bool:
        """Offer an observation; returns True if it was recorded for the current target.

        Only usable observations that were really analysed are accepted: motion-gate
        copies (``skipped``) may repeat a frame from before the target appeared.
        """
        target = self.current
        if self._phase != PHASE_COLLECT or target is None:
            return False
        if not obs.usable or obs.skipped or obs.features is None:
            return False
        features = np.array(obs.features, dtype=np.float64).reshape(-1)
        if features.size == 0 or not np.all(np.isfinite(features)):
            return False
        if self._n_features is None:
            self._n_features = features.size
        elif features.size != self._n_features:
            log.warning(
                "Ignoring observation with %d features (expected %d)",
                features.size,
                self._n_features,
            )
            return False
        self._pending.append(
            CalibrationSample(features, target.x, target.y, target.monitor_index, target.point_id)
        )
        return True

    # ------------------------------------------------------------- properties
    @property
    def plan(self) -> tuple[CalibrationTarget, ...]:
        return self._plan

    @property
    def phase(self) -> str:
        """``"idle"``, ``"settle"``, ``"collect"`` or ``"done"``."""
        return self._phase

    @property
    def current(self) -> CalibrationTarget | None:
        """Target being shown, or ``None`` before start and after the last target."""
        if self._phase in (PHASE_SETTLE, PHASE_COLLECT):
            return self._plan[self._index]
        return None

    @property
    def current_index(self) -> int:
        """Position of the current target in the plan."""
        return self._index

    @property
    def phase_progress(self) -> float:
        """0..1 through the current phase (a retry restarts the collect phase)."""
        elapsed = self._now - self._phase_start
        if self._phase == PHASE_SETTLE:
            return 1.0 if self.settle_s == 0 else _clamp01(elapsed / self.settle_s)
        if self._phase == PHASE_COLLECT:
            return _clamp01(elapsed / self.collect_s)
        return 1.0 if self._phase == PHASE_DONE else 0.0

    @property
    def point_progress(self) -> float:
        """0..1 through the current target, settle and collect time combined."""
        total = self.settle_s + self.collect_s
        if self._phase == PHASE_SETTLE:
            return self.phase_progress * self.settle_s / total
        if self._phase == PHASE_COLLECT:
            return (self.settle_s + self.phase_progress * self.collect_s) / total
        return 1.0 if self._phase == PHASE_DONE else 0.0

    @property
    def progress(self) -> float:
        """0..1 through the whole plan."""
        if self._phase == PHASE_DONE:
            return 1.0
        if self._phase == PHASE_IDLE or not self._plan:
            return 0.0
        return (self._index + self.point_progress) / len(self._plan)

    @property
    def current_sample_count(self) -> int:
        """Samples recorded so far for the current target."""
        return len(self._pending)

    @property
    def samples(self) -> list[CalibrationSample]:
        """Samples of all completed (not skipped) targets."""
        return list(self._samples)

    @property
    def skipped_points(self) -> list[int]:
        """Point ids that were skipped for lack of usable observations."""
        return list(self._skipped)

    # ------------------------------------------------------------- internals
    def _enter(self, phase: str, now: float) -> None:
        self._phase = phase
        self._phase_start = now

    def _step(self, now: float) -> bool:
        """Perform at most one transition; True if one happened."""
        elapsed = now - self._phase_start + _TIME_EPS
        if self._phase == PHASE_SETTLE:
            if elapsed < self.settle_s:
                return False
            self._enter(PHASE_COLLECT, now)
            return True
        if self._phase != PHASE_COLLECT or elapsed < self.collect_s:
            return False

        target = self._plan[self._index]
        if len(self._pending) >= self.min_samples:
            self._samples.extend(self._pending)
        elif self._retries < self.max_retries:
            self._retries += 1
            self._enter(PHASE_COLLECT, now)
            self._events.append(EVENT_RETRY)
            log.debug(
                "Calibration point %d: %d/%d samples, retrying",
                target.point_id,
                len(self._pending),
                self.min_samples,
            )
            return True
        else:
            self._skipped.append(target.point_id)
            log.info(
                "Calibration point %d skipped (%d usable samples)",
                target.point_id,
                len(self._pending),
            )

        self._pending = []
        self._retries = 0
        self._index += 1
        if self._index >= len(self._plan):
            self._enter(PHASE_DONE, now)
            self._events.append(EVENT_FINISHED)
            return False
        self._enter(PHASE_SETTLE, now)
        self._events.append(EVENT_TARGET)
        return True


# ------------------------------------------------------------------ evaluation
@dataclass
class CalibrationReport:
    """Quality of a calibration, estimated by leave-one-point-out cross-validation."""

    #: Fraction of samples whose held-out prediction is nearest to the right monitor.
    monitor_accuracy: float
    mean_error_px: float
    median_error_px: float
    per_monitor_accuracy: dict[int, float]
    n_samples: int
    n_points: int
    alpha: float
    grade: str
    #: Polynomial degree chosen for the model (see ``gaze.model``).
    degree: int = 2

    def summary(self) -> str:
        """One human-readable line, e.g. ``"Excellent — 99% monitor accuracy, 180 samples"``."""
        # Floor, so "100%" is only ever shown for a perfect result.
        pct = math.floor(self.monitor_accuracy * 100.0 + 1e-9)
        text = f"{self.grade.capitalize()} — {pct}% monitor accuracy, {self.n_samples} samples"
        if math.isfinite(self.mean_error_px):
            text += f", mean error {self.mean_error_px:.0f} px"
        return text

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (monitor keys become strings, non-finite numbers ``None``)."""
        out = asdict(self)
        out["per_monitor_accuracy"] = {str(k): v for k, v in self.per_monitor_accuracy.items()}
        return {k: _json_number(v) for k, v in out.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationReport:
        """Inverse of :meth:`to_dict`. Raises ``ValueError`` on malformed input."""
        try:
            per_monitor = {
                int(k): float(v) for k, v in dict(data.get("per_monitor_accuracy") or {}).items()
            }
            return cls(
                monitor_accuracy=_float(data["monitor_accuracy"]),
                mean_error_px=_float(data["mean_error_px"]),
                median_error_px=_float(data["median_error_px"]),
                per_monitor_accuracy=per_monitor,
                n_samples=int(data["n_samples"]),
                n_points=int(data["n_points"]),
                alpha=_float(data["alpha"]),
                grade=str(data["grade"]),
                degree=int(data.get("degree", 2)),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"invalid calibration report: {exc}") from exc


def grade_for(monitor_accuracy: float) -> str:
    """``"excellent"`` (>= 0.97), ``"good"`` (>= 0.9), ``"fair"`` (>= 0.75) or ``"poor"``."""
    for name, threshold in GRADE_THRESHOLDS:
        if monitor_accuracy >= threshold:
            return name
    return "poor"


def evaluate(
    samples: Sequence[CalibrationSample],
    monitors: Sequence[Monitor],
    degree: int | None = None,
) -> tuple[GazeModel, CalibrationReport]:
    """Fit the final model and estimate how well it will work.

    Hyper-parameters are chosen by leave-one-point-out cross-validation (every
    fold hides all samples of one calibration dot): ``alpha`` always, and the
    polynomial degree too unless ``degree`` is given. The report is computed from
    the held-out predictions of the chosen settings, so it reflects accuracy on
    gaze points the model has not seen. The returned model is fitted on all samples.

    Raises ``ValueError("not enough calibration data")`` with fewer than 10
    samples, fewer than 3 points, or points on fewer than two monitors (one on a
    single-monitor desk).
    """
    monitor_list = list(monitors)
    if not monitor_list:
        raise ValueError("no monitors")
    known = {m.index for m in monitor_list}
    usable = [s for s in samples if s.monitor_index in known]
    if len(usable) < len(samples):
        log.warning("Ignoring %d samples of unknown monitors", len(samples) - len(usable))

    point_ids = np.array([s.point_id for s in usable])
    covered = {s.monitor_index for s in usable}
    if (
        len(usable) < MIN_SAMPLES
        or np.unique(point_ids).shape[0] < MIN_POINTS
        or len(covered) < min(2, len(monitor_list))
    ):
        raise ValueError("not enough calibration data")

    X, Y, W = samples_to_arrays(usable)
    bounds = virtual_bounds(monitor_list)
    degrees = SUPPORTED_DEGREES if degree is None else (degree,)
    selection = select_model(
        X, Y, point_ids, alphas=DEFAULT_ALPHAS, degrees=degrees, bounds=bounds, weights=W
    )
    preds = selection.predictions
    if preds is None:  # every candidate diverged; still report honest held-out errors
        preds = lopo_predictions(
            X, Y, point_ids, selection.alpha, degree=selection.degree, bounds=bounds, weights=W
        )

    truth = np.array([s.monitor_index for s in usable])
    predicted = [_predicted_monitor(monitor_list, p) for p in preds]
    correct = np.array([p == t for p, t in zip(predicted, truth.tolist(), strict=True)])
    errors = np.hypot(preds[:, 0] - Y[:, 0], preds[:, 1] - Y[:, 1])
    finite = errors[np.isfinite(errors)]

    report = CalibrationReport(
        monitor_accuracy=float(np.mean(correct)),
        mean_error_px=float(np.mean(finite)) if finite.size else math.nan,
        median_error_px=float(np.median(finite)) if finite.size else math.nan,
        per_monitor_accuracy={
            int(idx): float(np.mean(correct[truth == idx])) for idx in sorted(covered)
        },
        n_samples=len(usable),
        n_points=int(np.unique(point_ids).shape[0]),
        alpha=selection.alpha,
        grade=grade_for(float(np.mean(correct))),
        degree=selection.degree,
    )
    model = GazeModel(degree=selection.degree, alpha=selection.alpha).fit(X, Y, W, bounds)
    log.info(
        "Calibration evaluated: %s (degree %d, alpha %g)",
        report.summary(),
        report.degree,
        report.alpha,
    )
    return model, report


def _predicted_monitor(monitors: Sequence[Monitor], point: np.ndarray) -> int | None:
    px, py = float(point[0]), float(point[1])
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    return nearest_monitor(monitors, px, py)[0].index


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _float(value: Any) -> float:
    return math.nan if value is None else float(value)


def _json_number(value: Any) -> Any:
    """JSON has no NaN/Infinity; represent them as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
