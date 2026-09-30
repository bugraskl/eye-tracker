"""Vision backends and backend selection.

``mediapipe`` (head pose + iris, accurate) is preferred; ``opencv`` (YuNet face
geometry, lightweight) is the fallback where MediaPipe is not installed, such as
on Intel Macs. Backend modules are imported lazily so that importing this
package stays cheap.
"""

from __future__ import annotations

import importlib.util
import logging

from ... import paths
from .base import BackendUnavailable, VisionBackend

log = logging.getLogger(__name__)

#: Backend names in order of preference.
BACKEND_NAMES: tuple[str, ...] = ("mediapipe", "opencv")

_MODEL_FILES = {
    "mediapipe": "face_landmarker.task",
    "opencv": "face_detection_yunet_2023mar.onnx",
}

__all__ = [
    "BACKEND_NAMES",
    "BackendUnavailable",
    "VisionBackend",
    "available_backends",
    "backend_class",
    "create_backend",
]


def _runtime_installed(name: str) -> bool:
    if name == "mediapipe":
        # find_spec only checks that the package exists; importing MediaPipe
        # takes about a second, so a broken install is only detected when the
        # backend is created (and "auto" then falls back to OpenCV).
        return importlib.util.find_spec("mediapipe") is not None
    import cv2

    return hasattr(cv2, "FaceDetectorYN")


def available_backends() -> list[str]:
    """Names of the backends whose runtime is installed and whose model is present."""
    return [
        name
        for name in BACKEND_NAMES
        if _runtime_installed(name) and paths.model_path(_MODEL_FILES[name]).is_file()
    ]


def backend_class(name: str = "auto") -> type[VisionBackend]:
    """The backend class ``create_backend(name)`` would try first, without creating it.

    Useful to learn ``feature_version`` (for calibration compatibility checks)
    before the worker thread has created the backend.

    Raises:
        BackendUnavailable: Unknown name, or ``"auto"`` with no backend available.
    """
    key = _normalise(name)
    if key == "auto":
        available = available_backends()
        if not available:
            raise BackendUnavailable(_nothing_available_message())
        key = available[0]
    if key == "mediapipe":
        from .mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend
    from .opencv_backend import OpenCVBackend

    return OpenCVBackend


def create_backend(name: str = "auto", max_faces: int = 1) -> VisionBackend:
    """Create a vision backend.

    ``"auto"`` tries MediaPipe first and falls back to OpenCV. Call this on the
    thread that will use the backend (MediaPipe objects are thread-bound).

    Raises:
        BackendUnavailable: The requested backend (or, for ``"auto"``, every
            backend) cannot be created. The message says why.
    """
    key = _normalise(name)
    if key != "auto":
        return _create(key, max_faces)
    reasons: list[str] = []
    for candidate in BACKEND_NAMES:
        try:
            backend = _create(candidate, max_faces)
        except BackendUnavailable as exc:
            log.info("Vision backend %s unavailable: %s", candidate, exc)
            reasons.append(f"{candidate}: {exc}")
            continue
        log.info("Using the %s vision backend", backend.name)
        return backend
    raise BackendUnavailable("No vision backend is available — " + "; ".join(reasons))


def _normalise(name: str) -> str:
    key = name.strip().lower()
    if key != "auto" and key not in BACKEND_NAMES:
        choices = ", ".join(("auto", *BACKEND_NAMES))
        raise BackendUnavailable(f"Unknown vision backend {name!r}; choose one of {choices}")
    return key


def _create(name: str, max_faces: int) -> VisionBackend:
    if name == "mediapipe":
        from .mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend(max_faces=max_faces)
    from .opencv_backend import OpenCVBackend

    return OpenCVBackend(max_faces=max_faces)


def _nothing_available_message() -> str:
    return (
        "No vision backend is available: install MediaPipe or OpenCV 4.5.4+ and make "
        "sure the models are present (scripts/fetch_models.py)"
    )
