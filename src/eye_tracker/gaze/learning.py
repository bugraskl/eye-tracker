"""Keeping the calibration accurate from everyday use.

People look at the mouse pointer when they place it: after a manual cursor move
comes to rest (to click, select or read), the eyes are almost always near the
pointer. Each such moment is a free, weakly labelled calibration sample. Mixing
them into the fit (at a lower weight than deliberate calibration samples) lets
the model follow slow changes such as a different chair height or a moved
camera. The same moments reveal when the model has drifted: if its prediction
keeps disagreeing with where the user puts the cursor, it is time to recalibrate.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ..types import Observation, Rect
from .calibration import CalibrationSample, samples_to_arrays
from .model import GazeModel

log = logging.getLogger(__name__)

#: A settle only counts if the cursor was moved within this many seconds.
RECENT_MOVE_S = 1.5

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
        self._trim()
        self._since_refit += 1
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

    def clear(self) -> None:
        """Forget every learned sample (after a new calibration)."""
        self._samples.clear()
        self._since_refit = 0
        self._next_id = -1
        self._armed = False

    def set_max_samples(self, max_samples: int) -> None:
        """Change the capacity, dropping samples if needed (0 disables learning)."""
        if max_samples < 0:
            raise ValueError("max_samples must be >= 0")
        self.max_samples = int(max_samples)
        self._trim()

    def should_refit(self) -> bool:
        """True once ``refit_every`` new samples have arrived since the last refit."""
        return self._since_refit >= self.refit_every

    def mark_refit(self) -> None:
        self._since_refit = 0

    def _trim(self) -> None:
        while len(self._samples) > self.max_samples:
            counts = Counter(s.monitor_index for s in self._samples)
            busiest = max(counts.values())
            # Samples are oldest first, so the first sample of a busiest monitor is
            # the oldest among them; ties between monitors go to the older sample.
            victim = next(
                i for i, s in enumerate(self._samples) if counts[s.monitor_index] == busiest
            )
            del self._samples[victim]


def refit_model(
    base: Sequence[CalibrationSample],
    implicit: Sequence[CalibrationSample],
    template: GazeModel,
    bounds: Rect | None = None,
) -> GazeModel:
    """Fit a new model on calibration plus learned samples.

    The template supplies the degree and ``alpha`` chosen at calibration time and,
    unless ``bounds`` is given, the target normalisation. Sample weights are
    honoured, so learned samples count less than calibration samples.
    """
    items = [*base, *implicit]
    X, Y, W = samples_to_arrays(items)
    return GazeModel(degree=template.degree, alpha=template.alpha).fit(
        X, Y, W, bounds=bounds if bounds is not None else template.bounds
    )


class DriftMonitor:
    """Tracks disagreement between prediction and manual cursor placement.

    Each event is either a comparison between the monitor the model predicts and
    the monitor where the user settled the cursor, or a wrong automatic switch
    that the user immediately undid. When at least ``min_events`` of the last
    ``window`` events exist and at least ``alert_ratio`` of them are errors,
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

    def record_wrong_switch(self) -> None:
        """The user moved the mouse back within 2 s of an automatic switch."""
        self._events.append(True)

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
