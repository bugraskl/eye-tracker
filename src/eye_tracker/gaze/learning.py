"""Keeping the calibration accurate from everyday use.

People look at the mouse pointer when they place it: after a manual cursor move
comes to rest (to click, select or read), the eyes are almost always near the
pointer. Each such moment is a free, weakly labelled calibration sample. Mixing
them into the fit (at a lower weight than deliberate calibration samples) lets
the model follow slow changes such as a different chair height or a moved
camera. The same moments reveal when the model has drifted: if its prediction
keeps disagreeing with where the user puts the cursor, it is time to recalibrate.

Not every settle is a good label: the user may click "Run" on one monitor and
already watch the output on the other. :func:`plausible_label` rejects labels
far from what the model currently predicts, and :meth:`ImplicitLearner.discard_last`
takes such a sample back out of the training set. The gate is generous on
purpose, so the samples that let the model follow a posture change still pass.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ..types import Monitor, Observation, Rect, nearest_monitor
from .calibration import CalibrationSample, samples_to_arrays
from .model import GazeModel

log = logging.getLogger(__name__)

#: A settle only counts if the cursor was moved within this many seconds.
RECENT_MOVE_S = 1.5

#: A learned label is implausible when the model's current prediction is farther
#: from it than this fraction of the diagonal of the cursor's monitor (about
#: 1100 px on a 1920x1080 monitor). In synthetic tests this removes every label
#: of "cursor parked on one monitor, reading the other" while keeping over 90 %
#: of the samples that correct an 8 cm change of posture; requiring the
#: prediction to be on the cursor's monitor would drop half of the latter.
IMPLICIT_MAX_ERROR = 0.5

MonitorLookup = Callable[[float, float], int | None]


class ImplicitLearner:
    """Adds weak labels from natural mouse use: when the user moves the cursor manually and it
    settles, they are very likely looking near it.

    Call :meth:`on_manual_cursor` for every poll that saw a user-initiated cursor
    move and :meth:`on_observation` for every observation. A *settle* is the
    moment the cursor has not moved for ``settle_s`` seconds after a move; the
    first usable observation within ``1.5 s`` of the last move becomes one sample
    (one sample per settle, never more).

    Implicit samples get negative ``point_id`` values (``-1, -2, …``), one per
    settle, so they never collide with calibration point ids. When full, the
    oldest sample of the monitor with the most samples is dropped, so one
    heavily used monitor cannot push the others out of the training set.
    """

    def __init__(
        self,
        max_samples: int = 400,
        weight: float = 0.5,
        settle_s: float = 0.35,
        refit_every: int = 25,
    ) -> None:
        if max_samples < 0 or weight <= 0 or settle_s < 0 or refit_every < 1:
            raise ValueError(
                "max_samples must be >= 0, weight > 0, settle_s >= 0 and refit_every >= 1"
            )
        self.max_samples = int(max_samples)
        self.weight = float(weight)
        self.settle_s = float(settle_s)
        self.refit_every = int(refit_every)
        self._samples: list[CalibrationSample] = []
        self._last_move: float | None = None
        self._cursor: tuple[float, float] = (0.0, 0.0)
        self._armed = False  # a settle is pending and has not produced a sample yet
        self._since_refit = 0
        self._next_id = -1
        # Undo information for discard_last(): the latest sample, whether it still
        # counts towards the next refit, and the (index, sample) pairs evicted to
        # make room for it, in eviction order.
        self._last_added: CalibrationSample | None = None
        self._last_counted = False
        self._last_evicted: list[tuple[int, CalibrationSample]] = []

    # -------------------------------------------------------------- events
    def on_manual_cursor(self, x: float, y: float, now: float) -> None:
        """Record a user-initiated cursor position (called on each manual move sample)."""
        self._last_move = now
        self._cursor = (float(x), float(y))
        self._armed = True

    def on_observation(
        self,
        obs: Observation,
        now: float,
        monitor_index_at: MonitorLookup,
    ) -> CalibrationSample | None:
        """Turn the observation into a sample if the cursor has just settled.

        ``monitor_index_at(x, y)`` maps the cursor position to a monitor index
        (``None`` outside every monitor). Returns the new sample, or ``None``.
        """
        if not self._armed or self._last_move is None or self.max_samples <= 0:
            return None
        since_move = now - self._last_move
        if since_move < self.settle_s:
            return None  # still moving (or about to move again)
        if since_move > RECENT_MOVE_S:
            self._armed = False  # this settle is too old to trust
            return None
        if not obs.usable or obs.skipped or obs.features is None:
            return None  # keep waiting for a good frame within the window
        features = np.array(obs.features, dtype=np.float64).reshape(-1)
        if features.size == 0 or not np.all(np.isfinite(features)):
            return None

        x, y = self._cursor
        monitor = monitor_index_at(x, y)
        self._armed = False
        if monitor is None:
            return None

        if self._samples and self._samples[0].features.shape != features.shape:
            log.warning(
                "Feature length changed (%d -> %d); discarding learned samples",
                self._samples[0].features.shape[0],
                features.shape[0],
            )
            self._samples.clear()
        sample = CalibrationSample(features, x, y, monitor, self._next_id, self.weight)
        self._next_id -= 1
        self._samples.append(sample)
        evicted = self._trim()
        self._since_refit += 1
        self._last_added, self._last_counted, self._last_evicted = sample, True, evicted
        return sample

    def discard_last(self) -> CalibrationSample | None:
        """Take back the sample most recently returned by :meth:`on_observation`.

        For a label the caller finds implausible (see :func:`plausible_label`): the
        sample leaves the training set, no longer counts towards the next refit,
        and the samples evicted to make room for it are restored. Returns the
        discarded sample, or ``None`` if there is nothing to undo (it was already
        discarded, or the samples were replaced, cleared or trimmed since).
        """
        sample = self._last_added
        if sample is None:
            return None
        self._samples = [s for s in self._samples if s is not sample]
        # Evictions shifted later indices; undoing them in reverse order restores
        # the list exactly (a victim always precedes the newest sample).
        for index, victim in reversed(self._last_evicted):
            if victim is not sample:
                self._samples.insert(index, victim)
        if self._last_counted:
            self._since_refit = max(0, self._since_refit - 1)
        self._forget_last()
        return sample

    # -------------------------------------------------------------- samples
    @property
    def samples(self) -> list[CalibrationSample]:
        """Learned samples, oldest first."""
        return list(self._samples)

    @property
    def new_since_refit(self) -> int:
        """Samples added since the last :meth:`mark_refit`."""
        return self._since_refit

    def load(self, samples: Iterable[CalibrationSample]) -> None:
        """Replace the learned samples (e.g. with those saved in the calibration file)."""
        self._samples = list(samples)
        self._trim()
        ids = [s.point_id for s in self._samples if s.point_id < 0]
        self._next_id = min(ids, default=0) - 1
        self._since_refit = 0
        self._forget_last()

    def clear(self) -> None:
        """Forget every learned sample (after a new calibration)."""
        self._samples.clear()
        self._since_refit = 0
        self._next_id = -1
        self._armed = False
        self._forget_last()

    def set_max_samples(self, max_samples: int) -> bool:
        """Change the capacity, dropping samples if needed (0 disables learning).

        Returns True if samples were dropped. The model fitted with them then
        still carries their influence, so the caller should refit it (with
        :func:`refit_model` and the remaining :attr:`samples`; with none left
        that reproduces the calibration-only fit) and save the result.
        """
        if max_samples < 0:
            raise ValueError("max_samples must be >= 0")
        self.max_samples = int(max_samples)
        dropped = bool(self._trim())
        if dropped:
            self._forget_last()
        return dropped

    def should_refit(self) -> bool:
        """True once ``refit_every`` new samples have arrived since the last refit."""
        return self._since_refit >= self.refit_every

    def mark_refit(self) -> None:
        self._since_refit = 0
        self._last_counted = False

    def _forget_last(self) -> None:
        self._last_added, self._last_counted, self._last_evicted = None, False, []

    def _trim(self) -> list[tuple[int, CalibrationSample]]:
        """Drop samples beyond the capacity; returns the ``(index, sample)`` evicted, in order."""
        evicted: list[tuple[int, CalibrationSample]] = []
        while len(self._samples) > self.max_samples:
            counts = Counter(s.monitor_index for s in self._samples)
            busiest = max(counts.values())
            # Samples are oldest first, so the first sample of a busiest monitor is
            # the oldest among them; ties between monitors go to the older sample.
            victim = next(
                i for i, s in enumerate(self._samples) if counts[s.monitor_index] == busiest
            )
            evicted.append((victim, self._samples.pop(victim)))
        return evicted


def refit_model(
    base: Sequence[CalibrationSample],
    implicit: Sequence[CalibrationSample],
    template: GazeModel,
    bounds: Rect | None = None,
    *,
    monitors: Sequence[Monitor] | None = None,
) -> GazeModel:
    """Fit a new model on calibration plus learned samples.

    The template supplies the degree, ``alpha`` and nonlinear features chosen at
    calibration time and, unless ``bounds`` is given, the target normalisation.
    Sample weights are honoured, so learned samples count less than calibration
    samples. Without learned samples this reproduces the calibration-only fit.
    The look-away regions (``GazeModel.away_regions``) are learned for
    ``monitors`` (the calibrated layout), or else for the monitors the template
    had them for.
    """
    items = [*base, *implicit]
    X, Y, W = samples_to_arrays(items)
    if monitors is not None:
        regions = [m.rect for m in monitors]
    else:
        regions = [region.rect for region in template.away_regions]
    model = GazeModel(degree=template.degree, alpha=template.alpha, nonlinear=template.nonlinear)
    return model.fit(
        X, Y, W, bounds=bounds if bounds is not None else template.bounds, regions=regions
    )


def plausible_label(
    predicted: Sequence[float] | np.ndarray,
    sample: CalibrationSample,
    monitors: Sequence[Monitor],
    max_error: float = IMPLICIT_MAX_ERROR,
) -> bool:
    """Whether a learned sample's label (the settled cursor) is plausible.

    ``predicted`` is the current model's gaze point for ``sample.features``. The
    label is implausible when it lies more than ``max_error`` times the diagonal
    of the cursor's monitor (``sample.monitor_index``, else the monitor nearest
    to the cursor) away from the prediction. The prediction need not be on the
    same monitor, so labels that correct a drifted model still count. A
    non-finite prediction or no monitors gives no evidence either way: True.
    """
    px, py = float(predicted[0]), float(predicted[1])
    if not (math.isfinite(px) and math.isfinite(py)) or not monitors:
        return True
    monitor = next((m for m in monitors if m.index == sample.monitor_index), None)
    if monitor is None:
        monitor = nearest_monitor(monitors, sample.x, sample.y)[0]
    return math.hypot(px - sample.x, py - sample.y) <= max_error * monitor.rect.diagonal


class DriftMonitor:
    """Tracks disagreement between prediction and manual cursor placement.

    Each event is either a comparison between the monitor the model predicts and
    the monitor where the user settled the cursor, or the outcome of an automatic
    switch: undone by the user within a moment (an error) or kept (correct).
    Recording both outcomes of switches matters: with adaptive learning off there
    are no settle comparisons, and counting only the undone switches would make
    every event an error. When at least ``min_events`` of the last ``window``
    events exist and at least ``alert_ratio`` of them are errors,
    :meth:`should_alert` fires (at most once per cooldown).
    """

    def __init__(self, window: int = 40, alert_ratio: float = 0.35, min_events: int = 15) -> None:
        if window < 1 or not 0.0 < alert_ratio <= 1.0 or min_events < 1:
            raise ValueError("window >= 1, 0 < alert_ratio <= 1 and min_events >= 1 required")
        self.alert_ratio = float(alert_ratio)
        self.min_events = int(min_events)
        self._events: deque[bool] = deque(maxlen=int(window))
        self._last_alert: float | None = None

    def record(self, predicted_monitor: int | None, actual_monitor: int) -> None:
        """Compare a prediction with where the user put the cursor.

        ``predicted_monitor`` ``None`` means there was no prediction (no face,
        blink); such events carry no information and are ignored.
        """
        if predicted_monitor is None:
            return
        self._events.append(predicted_monitor != actual_monitor)

    def record_switch(self, ok: bool) -> None:
        """Outcome of an automatic switch: ``ok`` False if the user undid it at once
        (moved the mouse back within the wrong-switch window), True once that
        window has passed without such a correction."""
        self._events.append(not ok)

    def record_wrong_switch(self) -> None:
        """The user moved the mouse back within 2 s of an automatic switch."""
        self.record_switch(False)

    @property
    def error_rate(self) -> float:
        """Fraction of recent events that were errors (0 without events)."""
        if not self._events:
            return 0.0
        return sum(self._events) / len(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def should_alert(self, now: float, cooldown_s: float = 1800) -> bool:
        """True if accuracy has dropped and no alert was given in the last ``cooldown_s``.

        Returning True starts the cooldown.
        """
        if len(self._events) < self.min_events or self.error_rate < self.alert_ratio:
            return False
        if self._last_alert is not None and now - self._last_alert < cooldown_s:
            return False
        self._last_alert = now
        log.info("Gaze accuracy dropped: %.0f%% recent disagreement", self.error_rate * 100)
        return True

    def reset(self) -> None:
        """Forget all events and the alert cooldown (after a new calibration)."""
        self._events.clear()
        self._last_alert = None
