"""Vision backends and backend selection.

``facemesh`` (478 landmarks with irises: head pose + eye direction) is preferred;
``lite`` (YuNet's five landmarks: head pose only, lowest CPU) is the fallback.
Both run entirely inside OpenCV, which the app ships anyway; no other inference
runtime is used, and nothing touches the network. Backend modules are imported
lazily so that importing this package stays cheap.
"""

from __future__ import annotations

import logging

from ... import paths
from .base import BackendUnavailable, VisionBackend

log = logging.getLogger(__name__)

#: Backend names in order of preference.
BACKEND_NAMES: tuple[str, ...] = ("facemesh", "lite")

#: Every bundled model file and its pinned SHA-256 (see ``models/NOTICE.md`` for
#: provenance; ``scripts/fetch_models.py`` pins the same values).
MODEL_FILES: dict[str, str] = {
    "face_landmarks_detector.tflite": (
        "c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5"
    ),
    "geometry_pipeline_metadata_landmarks.binarypb": (
        "bdbcda96dfcb7da883da124aaa2c55dee49770d934f0fcc71747f8c21bdc75b4"
    ),
    "face_detection_yunet_2023mar.onnx": (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    ),
}

#: The model files each backend needs.
BACKEND_MODEL_FILES: dict[str, tuple[str, ...]] = {
    "facemesh": (
        "face_landmarks_detector.tflite",
        "geometry_pipeline_metadata_landmarks.binarypb",
        "face_detection_yunet_2023mar.onnx",
    ),
    "lite": ("face_detection_yunet_2023mar.onnx",),
}

# Names used by Eye Tracker 0.1 settings files and scripts.
_LEGACY_NAMES = {"mediapipe": "facemesh", "opencv": "lite"}

__all__ = [
    "BACKEND_MODEL_FILES",
    "BACKEND_NAMES",
    "MODEL_FILES",
    "BackendUnavailable",
    "VisionBackend",
    "available_backends",
    "backend_class",
    "create_backend",
]


def _runtime_available(name: str) -> bool:
    import cv2

    if not hasattr(cv2, "FaceDetectorYN"):
        return False
    if name == "facemesh":
        return hasattr(cv2, "dnn") and hasattr(cv2.dnn, "readNetFromTFLite")
    return True


def available_backends() -> list[str]:
    """Names of the backends whose OpenCV features and model files are present.

    This is a cheap check; a backend listed here can still fail to load (for
    example a corrupt model), which :func:`create_backend` reports.
    """
    return [
        name
        for name in BACKEND_NAMES
        if _runtime_available(name)
        and all(paths.model_path(f).is_file() for f in BACKEND_MODEL_FILES[name])
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
    return _class(key)


def create_backend(name: str = "auto", max_faces: int = 1) -> VisionBackend:
    """Create a vision backend.

    ``"auto"`` tries ``facemesh`` first and falls back to ``lite``. Call this on
    the thread that will use the backend.

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
    key = _LEGACY_NAMES.get(key, key)
    if key != "auto" and key not in BACKEND_NAMES:
        choices = ", ".join(("auto", *BACKEND_NAMES))
        raise BackendUnavailable(f"Unknown vision backend {name!r}; choose one of {choices}")
    return key


def _class(name: str) -> type[VisionBackend]:
    if name == "facemesh":
        from .facemesh_backend import FaceMeshBackend

        return FaceMeshBackend
    from .lite_backend import LiteBackend

    return LiteBackend


def _create(name: str, max_faces: int) -> VisionBackend:
    if name == "facemesh":
        from .facemesh_backend import FaceMeshBackend

        return FaceMeshBackend(max_faces=max_faces)
    from .lite_backend import LiteBackend

    return LiteBackend(max_faces=max_faces)


def _nothing_available_message() -> str:
    return (
        "No vision backend is available: OpenCV 4.10 or newer with its DNN module is "
        "required, and the models must be present (scripts/fetch_models.py)"
    )
