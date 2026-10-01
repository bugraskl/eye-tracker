"""Tests for backend selection, the model files and the behaviour shared by all backends.

The face-mesh backend's own logic is tested in ``test_facemesh.py``. Tests here
run on drawn cartoon faces (see ``face_drawing.py``), so they need no photo;
the few that need a real face use the ``face_image`` fixture and are skipped
unless ``EYE_TRACKER_TEST_FACE`` points at a photo with one frontal face.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from eye_tracker import paths
from eye_tracker.types import Observation
from eye_tracker.vision import backends
from eye_tracker.vision.backends import (
    BACKEND_MODEL_FILES,
    BACKEND_NAMES,
    MODEL_FILES,
    BackendUnavailable,
    VisionBackend,
    available_backends,
    backend_class,
    create_backend,
)
from eye_tracker.vision.backends.facemesh_backend import FaceMeshBackend
from eye_tracker.vision.backends.lite_backend import LiteBackend, geometry_features
from face_drawing import draw_face, two_faces

AVAILABLE = available_backends()
needs_facemesh = pytest.mark.skipif("facemesh" not in AVAILABLE, reason="facemesh unavailable")
needs_lite = pytest.mark.skipif("lite" not in AVAILABLE, reason="YuNet is unavailable")
_BACKEND_PARAMS: list[object] = [*AVAILABLE] or [pytest.param("none", marks=pytest.mark.skip)]

log = logging.getLogger(__name__)


@pytest.fixture(params=_BACKEND_PARAMS)
def backend(request: pytest.FixtureRequest) -> Iterator[VisionBackend]:
    instance = create_backend(request.param)
    try:
        yield instance
    finally:
        instance.close()


def _run(backend: VisionBackend, frame: np.ndarray, start: float = 1.0, n: int = 3) -> Observation:
    """Process a frame a few times (trackers settle) and return the last result."""
    obs = None
    for i in range(n):
        obs = backend.process(frame, start + 0.05 * i)
    assert obs is not None
    return obs


# ------------------------------------------------------------------ selection
def test_both_backends_are_available_with_the_committed_models() -> None:
    assert AVAILABLE == ["facemesh", "lite"]


def test_available_backends_order() -> None:
    assert BACKEND_NAMES == ("facemesh", "lite")
    assert [name for name in BACKEND_NAMES if name in AVAILABLE] == AVAILABLE


def test_available_backends_needs_every_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths, "model_path", lambda name: tmp_path / name)
    assert available_backends() == []
    with pytest.raises(BackendUnavailable, match="No vision backend"):
        backend_class("auto")
    # Only YuNet present: the lite backend is available, facemesh is not.
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"x")
    assert available_backends() == ["lite"]
    assert backend_class("auto") is LiteBackend


def test_unknown_backend_raises() -> None:
    with pytest.raises(BackendUnavailable, match="Unknown vision backend"):
        create_backend("tensorflow")
    with pytest.raises(BackendUnavailable, match="auto, facemesh, lite"):
        backend_class("tensorflow")


@pytest.mark.parametrize(
    ("legacy", "cls"), [("mediapipe", FaceMeshBackend), ("opencv", LiteBackend)]
)
def test_legacy_names_map_to_the_new_backends(legacy: str, cls: type[VisionBackend]) -> None:
    assert backend_class(legacy) is cls
    assert backend_class(legacy.upper()) is cls


@pytest.mark.skipif(not AVAILABLE, reason="no backend available")
def test_auto_prefers_facemesh() -> None:
    instance = create_backend("auto")
    try:
        assert instance.name == AVAILABLE[0] == "facemesh"
        assert backend_class("auto") is type(instance)
        assert backend_class(" AUTO ").feature_version == instance.feature_version
    finally:
        instance.close()


@needs_lite
def test_auto_falls_back_to_lite(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: object, *args: object, **kwargs: object) -> None:
        raise BackendUnavailable("simulated landmark model failure")

    monkeypatch.setattr(FaceMeshBackend, "__init__", broken)
    instance = create_backend("auto", max_faces=2)
    try:
        assert isinstance(instance, LiteBackend)
    finally:
        instance.close()


def test_auto_reports_all_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: object, *args: object, **kwargs: object) -> None:
        raise BackendUnavailable(f"{type(self).__name__} broken")

    monkeypatch.setattr(FaceMeshBackend, "__init__", broken)
    monkeypatch.setattr(LiteBackend, "__init__", broken)
    with pytest.raises(BackendUnavailable) as info:
        create_backend("auto")
    assert "FaceMeshBackend broken" in str(info.value)
    assert "LiteBackend broken" in str(info.value)


def test_explicit_backend_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: object, *args: object, **kwargs: object) -> None:
        raise BackendUnavailable("facemesh broken")

    monkeypatch.setattr(FaceMeshBackend, "__init__", broken)
    with pytest.raises(BackendUnavailable, match="facemesh broken"):
        create_backend("facemesh")


def test_lite_missing_model(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailable, match="model not found"):
        LiteBackend(model_path=tmp_path / "missing.onnx")


def test_backend_metadata() -> None:
    assert FaceMeshBackend.name == "facemesh"
    assert FaceMeshBackend.feature_version == "facemesh-pose-iris-1"
    assert FaceMeshBackend.feature_names == (
        "yaw",
        "pitch",
        "roll",
        "tx",
        "ty",
        "tz",
        "iris_h",
        "iris_v",
    )
    assert FaceMeshBackend.gaze_features == ("yaw", "pitch", "iris_h", "iris_v")
    assert LiteBackend.name == "lite"
    assert LiteBackend.feature_version == "yunet-geom-1"
    assert LiteBackend.feature_names == (
        "nose_dx",
        "nose_dy",
        "roll",
        "face_cx",
        "face_cy",
        "scale",
    )
    assert LiteBackend.gaze_features == ("nose_dx", "nose_dy")
    for cls in (FaceMeshBackend, LiteBackend):
        assert set(cls.gaze_features) <= set(cls.feature_names)


# --------------------------------------------------------------- offline rule
def test_mediapipe_runtime_is_gone() -> None:
    """The MediaPipe runtime uploads usage logs; the app must not use or ship it."""
    package = Path(backends.__file__).parent
    assert not (package / "mediapipe_backend.py").exists()
    assert not (package.parent / "_mp_shim.py").exists()
    loaded = [name for name in sys.modules if name == "mediapipe" or name.startswith("mediapipe.")]
    assert loaded == []
    sources = "\n".join(p.read_text(encoding="utf-8") for p in package.parent.rglob("*.py"))
    assert "import mediapipe" not in sources
    assert "from mediapipe" not in sources


def test_mediapipe_is_not_a_dependency() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text("utf-8")
    dependencies = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "mediapipe" not in dependencies.lower()


# ---------------------------------------------------------------- model files
def test_model_files_are_committed_and_pinned() -> None:
    needed = {name for names in BACKEND_MODEL_FILES.values() for name in names}
    assert set(MODEL_FILES) == needed
    for name, sha256 in MODEL_FILES.items():
        path = paths.model_path(name)
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256, name
    assert not paths.model_path("face_landmarker.task").exists()


def test_model_notice_lists_every_file() -> None:
    notice = paths.model_path("NOTICE.md").read_text(encoding="utf-8")
    for name, sha256 in MODEL_FILES.items():
        assert name in notice
        assert sha256 in notice
    assert "Apache" in notice
    assert "MIT" in notice


def test_fetch_models_pins_the_same_checksums() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "fetch_models.py"
    spec = importlib.util.spec_from_file_location("fetch_models_for_backends", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    installed = dict(pair for model in module.MODELS for pair in model.installed())
    assert installed == MODEL_FILES


# ---------------------------------------------------------- no-face behaviour
def test_blank_frame_has_no_face(backend: VisionBackend) -> None:
    frame = np.full((480, 640, 3), 90, np.uint8)
    obs = backend.process(frame, 1.0)
    assert obs.timestamp == 1.0
    assert obs.face_count == 0
    assert obs.features is None
    assert obs.quality == 0.0
    assert not obs.usable
    assert not obs.face_present
    assert not obs.blind  # the worker, not the backend, judges blindness
    assert obs.frame_size == (640, 480)
    assert obs.inference_ms > 0.0


def test_repeated_timestamps_are_tolerated(backend: VisionBackend) -> None:
    frame = np.zeros((240, 320, 3), np.uint8)
    for ts in (5.0, 5.0, 4.0, 5.0005):
        assert backend.process(frame, ts).face_count == 0


@pytest.mark.parametrize("kind", ["gray", "bgra", "large", "tiny"])
def test_other_frame_formats(backend: VisionBackend, kind: str) -> None:
    frame = {
        "gray": np.zeros((240, 320), np.uint8),
        "bgra": np.zeros((240, 320, 4), np.uint8),
        "large": np.zeros((1080, 1920, 3), np.uint8),
        "tiny": np.zeros((8, 8, 3), np.uint8),
    }[kind]
    obs = backend.process(frame, 1.0)
    assert obs.face_count == 0
    assert obs.frame_size == (frame.shape[1], frame.shape[0])


def test_annotate_returns_a_copy(backend: VisionBackend) -> None:
    frame = np.full((240, 320, 3), 60, np.uint8)
    obs = backend.process(frame, 1.0)
    out = backend.annotate(frame, obs)
    assert out.shape == frame.shape
    assert out is not frame
    assert np.all(frame == 60)  # input untouched
    assert np.any(out != 60)  # "no face" label drawn


def test_close_is_idempotent_and_final(backend: VisionBackend) -> None:
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError):
        backend.process(np.zeros((10, 10, 3), np.uint8), 1.0)


def test_reset_starts_afresh_and_a_still_face_is_settled(backend: VisionBackend) -> None:
    """The worker calls reset() when the camera was released and asks ``settled``."""
    frame = draw_face()
    assert _run(backend, frame).usable
    assert backend.settled  # a still face: the motion gate may repeat it
    backend.reset()
    backend.reset()  # idempotent
    assert backend.settled
    obs = backend.process(frame, 5.0)
    assert obs.usable
    assert backend.settled


# --------------------------------------------------------------- drawn faces
def test_drawn_face_is_found(backend: VisionBackend) -> None:
    frame = draw_face()
    obs = _run(backend, frame)
    assert obs.face_count == 1
    assert obs.features is not None
    assert obs.features.shape == (len(backend.feature_names),)
    assert np.all(np.isfinite(obs.features))
    assert obs.quality >= 0.6
    assert not obs.blink
    assert obs.usable
    assert obs.head_yaw is not None
    assert abs(obs.head_yaw) < 8.0  # drawn frontal
    assert obs.face_box is not None
    x, y, w, h = obs.face_box
    assert x < 0.5 < x + w
    assert y < 0.5 < y + h
    assert 0.15 < w < 0.4
    log.info("%s: %.1f ms per frame", backend.name, obs.inference_ms)


def test_drawn_face_annotation(backend: VisionBackend) -> None:
    frame = draw_face()
    obs = _run(backend, frame)
    out = backend.annotate(frame, obs)
    assert out.shape == frame.shape
    changed = np.any(out != frame, axis=2)
    assert changed.sum() > 200
    # A stale observation (another timestamp) only gets the box and label.
    stale = backend.annotate(frame, Observation(timestamp=-1.0, face_count=0))
    assert np.any(stale != frame)


def test_head_turn_signs(backend: VisionBackend) -> None:
    """A face on the image left is turned towards image-right (it faces the lens)."""
    left = _run(backend, draw_face(cx=200), 1.0)
    right = _run(backend, draw_face(cx=440), 10.0)
    assert left.head_yaw is not None
    assert right.head_yaw is not None
    if backend.name == "facemesh":
        assert left.head_yaw > 3.0
        assert right.head_yaw < -3.0


def test_two_faces_are_counted(backend: VisionBackend) -> None:
    canvas = two_faces()
    backend.set_max_faces(2)
    obs = _run(backend, canvas)
    assert obs.face_count == 2
    assert obs.face_box is not None
    x, _, w, _ = obs.face_box
    assert x + w / 2 < 0.5  # the primary face is the user's, not the onlooker's
    out = backend.annotate(canvas, obs)
    assert out.shape == canvas.shape


# ------------------------------------------------------------- lite geometry
def _yunet_row(
    *,
    re: tuple[float, float] = (100.0, 100.0),
    le: tuple[float, float] = (140.0, 100.0),
    nose: tuple[float, float] = (120.0, 122.0),
    rm: tuple[float, float] = (105.0, 140.0),
    lm: tuple[float, float] = (135.0, 140.0),
    box: tuple[float, float, float, float] = (80.0, 60.0, 80.0, 100.0),
) -> np.ndarray:
    return np.array([*box, *re, *le, *nose, *rm, *lm, 0.9], dtype=np.float32)


def test_geometry_frontal() -> None:
    result = geometry_features(_yunet_row(), 320.0, 240.0)
    assert result is not None
    features, yaw, pitch = result
    nose_dx, nose_dy, roll, cx, cy, scale = features
    assert nose_dx == pytest.approx(0.0)
    assert nose_dy == pytest.approx(0.0)
    assert roll == pytest.approx(0.0)
    assert (cx, cy) == pytest.approx((120 / 320, 110 / 240))
    assert scale == pytest.approx(40 / 320)
    assert (yaw, pitch) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_geometry_signs() -> None:
    turned = geometry_features(_yunet_row(nose=(128.0, 122.0)), 320, 240)
    lowered = geometry_features(_yunet_row(nose=(120.0, 128.0)), 320, 240)
    tilted = geometry_features(_yunet_row(le=(140.0, 90.0)), 320, 240)
    assert turned is not None
    assert lowered is not None
    assert tilted is not None
    assert turned[0][0] > 0
    assert turned[1] > 0  # yaw
    assert lowered[0][1] > 0
    assert lowered[2] > 0  # pitch
    assert tilted[0][2] > 0  # counter-clockwise roll


def test_geometry_mirror_flips_horizontal_features() -> None:
    row = _yunet_row(nose=(127.0, 120.0), le=(140.0, 95.0))
    width = 320.0
    mirrored = row.copy()
    # Mirror every x coordinate; YuNet labels eyes/mouth corners by image side.
    mirrored[0] = width - (row[0] + row[2])
    mirrored[4:6] = (width - row[6], row[7])
    mirrored[6:8] = (width - row[4], row[5])
    mirrored[8] = width - row[8]
    mirrored[10:12] = (width - row[12], row[13])
    mirrored[12:14] = (width - row[10], row[11])
    a = geometry_features(row, width, 240)
    b = geometry_features(mirrored, width, 240)
    assert a is not None
    assert b is not None
    assert b[0][0] == pytest.approx(-a[0][0], abs=1e-6)  # nose_dx
    assert b[0][2] == pytest.approx(-a[0][2], abs=1e-6)  # roll
    assert b[0][1] == pytest.approx(a[0][1], abs=1e-6)  # nose_dy


def test_geometry_degenerate() -> None:
    assert geometry_features(_yunet_row(le=(100.0, 100.0)), 320, 240) is None
    assert geometry_features(_yunet_row(rm=(105.0, 100.0), lm=(135.0, 100.0)), 320, 240) is None


@needs_lite
def test_lite_reports_every_face_regardless_of_max_faces() -> None:
    lite = LiteBackend(max_faces=1)
    try:
        assert lite.process(two_faces(), 1.0).face_count == 2
    finally:
        lite.close()


# ------------------------------------------------------------------ real faces
def _load(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None, f"could not decode {path}"
    return image


def _camera_like(image: np.ndarray, longest: int = 640) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, longest / max(h, w))
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def test_real_face(face_image: Path, backend: VisionBackend) -> None:
    frame = _camera_like(_load(face_image))
    obs = _run(backend, frame, n=5)
    assert obs.face_count == 1
    assert obs.features is not None
    assert np.all(np.isfinite(obs.features))
    assert obs.quality >= 0.5
    assert obs.usable
    log.info("%s: %.1f ms per frame at %dx%d", backend.name, obs.inference_ms, *obs.frame_size)
    assert obs.inference_ms < 250.0


@needs_facemesh
def test_real_face_mirror_flips_yaw_roll_and_iris(face_image: Path) -> None:
    frame = _camera_like(_load(face_image))
    a = _run(FaceMeshBackend(), frame)
    b = _run(FaceMeshBackend(), cv2.flip(frame, 1))
    assert a.features is not None
    assert b.features is not None
    yaw, pitch, roll = 0, 1, 2
    assert a.features[yaw] + b.features[yaw] == pytest.approx(0.0, abs=2.0)
    assert a.features[roll] + b.features[roll] == pytest.approx(0.0, abs=2.0)
    assert a.features[pitch] == pytest.approx(b.features[pitch], abs=2.0)
    # Mirroring swaps the eyes and reverses the corner axes: iris_h -> 1 - iris_h.
    assert a.features[6] + b.features[6] == pytest.approx(1.0, abs=0.05)


@needs_facemesh
def test_real_face_iris_is_position_and_scale_invariant(face_image: Path) -> None:
    image = _load(face_image)
    base = _run(FaceMeshBackend(), _camera_like(image))
    h, w = image.shape[:2]
    shifted = _camera_like(image[: int(h * 0.92), int(w * 0.06) :])
    smaller = _camera_like(image, longest=420)
    for frame in (shifted, smaller):
        obs = _run(FaceMeshBackend(), frame)
        assert base.features is not None
        assert obs.features is not None
        assert obs.features[6] == pytest.approx(base.features[6], abs=0.05)  # iris_h
        assert obs.features[7] == pytest.approx(base.features[7], abs=0.05)  # iris_v
