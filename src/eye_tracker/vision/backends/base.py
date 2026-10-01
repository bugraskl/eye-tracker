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
    #: Short identifier stored with calibrations ("facemesh", "lite").
    name: ClassVar[str]
    #: Human-readable names of the entries of ``Observation.features``.
    feature_names: ClassVar[tuple[str, ...]]
    #: Changes whenever feature semantics change; calibrations with another
    #: version are rejected.
    feature_version: ClassVar[str]
    #: The subset of ``feature_names`` that encodes where the user looks (head
    #: rotation and eye direction, as opposed to head position or roll). The gaze
    #: model applies nonlinear terms only to these, and judges looking away from
    #: every monitor from the combined direction they give (``GazeModel.looks_away``).
    gaze_features: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Observation:
        """Analyse one frame.

        ``timestamp`` is ``time.monotonic()`` seconds and strictly increases
        between calls. The returned observation carries the same timestamp.
        Must not raise for frames without faces (``face_count == 0``).
        """

    def set_max_faces(self, n: int) -> None:  # noqa: B027 - optional hook
        """Maximum number of faces to detect (2 enables the shoulder guard)."""

    @property
    def settled(self) -> bool:
        """Whether the last observation is final for an unchanged picture.

        A tracking backend returns ``False`` when analysing the very same picture
        again would still give noticeably different features (its search window
        was still catching up with a moved face). The vision worker then analyses
        the next frame instead of letting the motion gate repeat the observation.
        """
        return True

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Forget what was tracked in earlier frames; the next frame starts afresh.

        Called when the camera was released or replaced: the next frame may show
        a different scene, so nothing may be carried over from before.
        """

    def annotate(self, frame_bgr: np.ndarray, observation: Observation) -> np.ndarray:
        """Return a copy of the frame with landmarks drawn (camera preview only)."""
        return frame_bgr.copy()

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release native resources. Safe to call more than once."""
