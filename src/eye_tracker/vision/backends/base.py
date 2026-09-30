"""Vision backend contract.

A backend turns one BGR camera frame into an :class:`~eye_tracker.types.Observation`.
Backends are used from a single worker thread and need not be thread-safe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ...types import Observation


class BackendUnavailable(RuntimeError):
    """Raised by a backend constructor when its runtime or model is missing."""


class VisionBackend(ABC):
    #: Short identifier stored with calibrations ("mediapipe", "opencv").
    name: ClassVar[str]
    #: Human-readable names of the entries of ``Observation.features``.
    feature_names: ClassVar[tuple[str, ...]]
    #: Changes whenever feature semantics change; calibrations with another
    #: version are rejected.
    feature_version: ClassVar[str]

    @abstractmethod
    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Observation:
        """Analyse one frame.

        ``timestamp`` is ``time.monotonic()`` seconds and strictly increases
        between calls. The returned observation carries the same timestamp.
        Must not raise for frames without faces (``face_count == 0``).
        """

    def set_max_faces(self, n: int) -> None:  # noqa: B027 - optional hook
        """Maximum number of faces to detect (2 enables the shoulder guard)."""

    def annotate(self, frame_bgr: np.ndarray, observation: Observation) -> np.ndarray:
        """Return a copy of the frame with landmarks drawn (camera preview only)."""
        return frame_bgr.copy()

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release native resources. Safe to call more than once."""
