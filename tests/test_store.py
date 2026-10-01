"""Tests for eye_tracker.gaze.store (calibration file, profiles, robustness)."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eye_tracker.gaze.calibration import CalibrationSample
from eye_tracker.gaze.model import GazeModel
from eye_tracker.gaze.store import (
    CALIBRATION_VERSION,
    MAX_PROFILES,
    CalibrationData,
    CalibrationLibrary,
    axis_error,
    load_calibration,
    same_aspect,
    save_calibration,
    utc_now_iso,
)
from eye_tracker.types import Monitor, Rect, layout_signature, virtual_bounds

MONITORS = [
    Monitor(0, "DELL U2720Q", Rect(0, 0, 3840, 2160), primary=True, scale=1.5),
    Monitor(1, "LG ÜltraGear", Rect(3840, 540, 1920, 1080), scale=1.0),
]
# A second desk: a laptop panel next to a 1080p monitor.
HOME = [
    Monitor(0, "laptop", Rect(0, 0, 1920, 1200), primary=True),
    Monitor(1, "monitor", Rect(1920, 0, 1920, 1080)),
]


def make_data(
    n_points: int = 12,
    per_point: int = 5,
    implicit: int = 4,
    *,
    monitors: list[Monitor] | None = None,
    camera: str = "0",
    frame_size: tuple[int, int] = (640, 480),
    created_at: str = "2026-09-30T08:15:00+00:00",
) -> CalibrationData:
    monitors = MONITORS if monitors is None else monitors
    rng = np.random.default_rng(7)
    samples = []
    for pid in range(n_points):
        monitor = monitors[pid % 2]
        x, y = monitor.rect.denormalize(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        for _ in range(per_point):
            samples.append(CalibrationSample(rng.normal(size=8), x, y, monitor.index, pid))
    learned = [
        CalibrationSample(rng.normal(size=8), 100.0 + i, 200.0, 0, -(i + 1), weight=0.5)
        for i in range(implicit)
    ]
    X = np.array([s.features for s in samples])
    Y = np.array([(s.x, s.y) for s in samples])
    model = GazeModel(degree=3, alpha=0.1, nonlinear=(0, 1, 6, 7)).fit(
        X, Y, bounds=virtual_bounds(monitors)
    )
    return CalibrationData(
        backend="facemesh",
        feature_version="facemesh-pose-iris-1",
        layout_signature=layout_signature(monitors),
        monitors=list(monitors),
        samples=samples,
        implicit_samples=learned,
        model=model,
        report={
            "grade": "excellent",
            "monitor_accuracy": 0.99,
            "mean_error_px": math.nan,
            "per_monitor_accuracy": {0: 1.0, 1: 0.98},
            "n_samples": np.int64(60),
        },
        created_at=created_at,
        camera=camera,
        frame_size=frame_size,
    )


def assert_samples_close(a: list[CalibrationSample], b: list[CalibrationSample]) -> None:
    assert len(a) == len(b)
    for s, t in zip(a, b, strict=True):
        assert np.allclose(s.features, t.features, atol=1e-6)
        assert (s.x, s.y) == pytest.approx((t.x, t.y), abs=1e-6)
        assert (s.monitor_index, s.point_id) == (t.monitor_index, t.point_id)
        assert s.weight == pytest.approx(t.weight)


def v1_doc(data: CalibrationData, tmp_path: Path) -> dict[str, Any]:
    """The single-calibration layout written by version 1 of the file format."""
    path = tmp_path / "v2.json"
    save_calibration(path, data)
    profile = json.loads(path.read_text(encoding="utf-8"))["profiles"][0]
    path.unlink()
    for key in ("camera", "frame_size"):
        profile.pop(key)
    return {"format": "eye-tracker-calibration", "version": 1, **profile}


# --------------------------------------------------------------------------- round trip
def test_round_trip(tmp_path: Path) -> None:
    data = make_data()
    path = tmp_path / "sub" / "calibration.json"
    save_calibration(path, data)
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.backend == "facemesh"
    assert loaded.feature_version == "facemesh-pose-iris-1"
    assert loaded.layout_signature == data.layout_signature
    assert loaded.monitors == MONITORS
    assert loaded.created_at == "2026-09-30T08:15:00+00:00"
    assert (loaded.camera, loaded.frame_size) == ("0", (640, 480))
    assert_samples_close(loaded.samples, data.samples)
    assert_samples_close(loaded.implicit_samples, data.implicit_samples)
    assert (loaded.model.degree, loaded.model.alpha) == (3, 0.1)
    assert loaded.model.nonlinear == (0, 1, 6, 7)
    X = np.array([s.features for s in data.samples])
    assert np.allclose(loaded.model.predict(X), data.model.predict(X), atol=1e-3)
    assert loaded.report["grade"] == "excellent"
    assert loaded.report["mean_error_px"] is None  # NaN is not valid JSON
    assert loaded.report["per_monitor_accuracy"] == {"0": 1.0, "1": 0.98}
    assert loaded.report["n_samples"] == 60
    assert loaded.grade == "excellent"
    assert loaded.is_compatible("facemesh", "facemesh-pose-iris-1", MONITORS) == (True, "")
    assert loaded.is_compatible("facemesh", "facemesh-pose-iris-1", MONITORS, "0", (640, 480)) == (
        True,
        "",
    )


def test_file_is_readable_json_without_images(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    save_calibration(path, make_data())
    text = path.read_text(encoding="utf-8")
    doc = json.loads(text)
    assert doc["format"] == "eye-tracker-calibration"
    assert doc["version"] == CALIBRATION_VERSION == 2
    assert set(doc) == {"format", "version", "profiles"}
    assert len(doc["profiles"]) == 1
    assert set(doc["profiles"][0]) == {
        "created_at", "backend", "feature_version", "layout_signature", "camera", "frame_size",
        "monitors", "report", "model", "samples", "implicit_samples",
    }  # fmt: skip
    profile = doc["profiles"][0]
    assert set(profile["samples"][0]) == {"features", "x", "y", "monitor", "point", "weight"}
    assert profile["frame_size"] == [640, 480]
    # One sample per line keeps the file diff-friendly and compact.
    assert text.count("\n") < 120
    assert "LG ÜltraGear" in text
    # Sample floats are rounded to six decimals.
    for value in profile["samples"][0]["features"]:
        assert round(value, 6) == value
    assert list(tmp_path.iterdir()) == [path]  # atomic write leaves no temp files


def test_non_finite_samples_are_not_saved(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    data = make_data()
    data.samples[0].features[3] = math.nan
    path = tmp_path / "c.json"
    with caplog.at_level(logging.WARNING, logger="eye_tracker.gaze.store"):
        save_calibration(path, data)
    loaded = load_calibration(path)
    assert loaded is not None
    assert len(loaded.samples) == len(data.samples) - 1
    assert "non-finite" in caplog.text


def test_defaults_and_created_at_format() -> None:
    data = make_data()
    minimal = CalibrationData(
        backend="opencv",
        feature_version="yunet-geom-1",
        layout_signature="abc",
        monitors=[],
        samples=[],
        implicit_samples=[],
        model=data.model,
    )
    assert minimal.report == {}
    assert minimal.grade is None
    assert (minimal.camera, minimal.frame_size) == ("", (0, 0))
    # The renamed OpenCV backend is keyed by its current name.
    assert minimal.key == ("abc", "lite", "yunet-geom-1", "")
    parsed = datetime.fromisoformat(minimal.created_at)
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0
    assert utc_now_iso().endswith("+00:00")


def test_unfitted_model_round_trip(tmp_path: Path) -> None:
    data = make_data()
    data.model = GazeModel(degree=1, alpha=2.0)
    path = tmp_path / "c.json"
    save_calibration(path, data)
    loaded = load_calibration(path)
    assert loaded is not None
    assert not loaded.model.is_fitted
    ok, reason = loaded.is_compatible("facemesh", "facemesh-pose-iris-1", MONITORS)
    assert not ok
    assert "no fitted model" in reason


def test_version_1_file_still_loads(tmp_path: Path) -> None:
    data = make_data()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(v1_doc(data, tmp_path)), encoding="utf-8")
    loaded = load_calibration(path)
    assert loaded is not None
    assert (loaded.camera, loaded.frame_size) == ("", (0, 0))  # unknown
    assert_samples_close(loaded.samples, data.samples)
    # An unknown camera never makes the calibration unusable.
    assert loaded.is_compatible("facemesh", "facemesh-pose-iris-1", MONITORS, "1", (1280, 720)) == (
        True,
        "",
    )


def test_model_without_nonlinear_key_loads_as_all_features(tmp_path: Path) -> None:
    data = make_data()
    data.model = GazeModel(degree=2, alpha=1.0).fit(
        np.array([s.features for s in data.samples]), np.array([(s.x, s.y) for s in data.samples])
    )
    path = tmp_path / "calibration.json"
    save_calibration(path, data)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert "nonlinear" not in doc["profiles"][0]["model"]
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.model.nonlinear is None


# --------------------------------------------------------------------------- compatibility
def test_is_compatible_reasons() -> None:
    data = make_data()
    ok, reason = data.is_compatible("lite", "facemesh-pose-iris-1", MONITORS)
    assert not ok
    assert "lite" in reason
    assert "facemesh" in reason
    ok, reason = data.is_compatible("facemesh", "facemesh-pose-iris-2", MONITORS)
    assert not ok
    assert "facemesh-pose-iris-2" in reason
    moved = [MONITORS[0], Monitor(1, "LG", Rect(-1920, 0, 1920, 1080))]
    ok, reason = data.is_compatible("facemesh", "facemesh-pose-iris-1", moved)
    assert not ok
    assert "layout" in reason
    # Monitor order and names do not matter, only the geometry.
    renamed = [Monitor(5, "x", MONITORS[1].rect), Monitor(6, "y", MONITORS[0].rect)]
    assert data.is_compatible("facemesh", "facemesh-pose-iris-1", renamed) == (True, "")


def _legacy_profile(backend: str, feature_version: str) -> CalibrationData:
    data = make_data()
    data.backend, data.feature_version = backend, feature_version
    return data


def test_opencv_calibrations_are_valid_for_the_renamed_lite_backend(tmp_path: Path) -> None:
    """The OpenCV/YuNet backend is now "lite" with unchanged measurements."""
    old = _legacy_profile("opencv", "yunet-geom-1")
    assert old.is_compatible("lite", "yunet-geom-1", MONITORS) == (True, "")
    assert old.key[1] == "lite"
    # Its measurement version still has to match.
    ok, reason = old.is_compatible("lite", "yunet-geom-2", MONITORS)
    assert not ok
    assert "yunet-geom-2" in reason
    # The library finds it for "lite", and a new "lite" calibration replaces it.
    library = CalibrationLibrary([old])
    assert library.best("lite", "yunet-geom-1", MONITORS) is old
    new = _legacy_profile("lite", "yunet-geom-1")
    library.put(new)
    assert library.profiles == [new]
    # Loading a file written by the old version stores the current name.
    path = tmp_path / "calibration.json"
    save_calibration(path, old)
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"][0]["backend"] == "opencv"
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.backend == "lite"
    assert loaded.is_compatible("lite", "yunet-geom-1", MONITORS) == (True, "")


def test_mediapipe_calibrations_are_not_used_by_facemesh() -> None:
    """The MediaPipe runtime backend measured differently from its successor."""
    old = _legacy_profile("mediapipe", "mp-pose-iris-1")
    ok, reason = old.is_compatible("facemesh", "facemesh-pose-iris-1", MONITORS)
    assert not ok
    assert "mediapipe" in reason
    assert "facemesh" in reason
    # Even under the same name, the measurement version differs.
    ok, _ = old.is_compatible("mediapipe", "facemesh-pose-iris-1", MONITORS)
    assert not ok
    library = CalibrationLibrary([old])
    assert library.best("facemesh", "facemesh-pose-iris-1", MONITORS) is None


def test_is_compatible_checks_the_camera() -> None:
    data = make_data(camera="0", frame_size=(640, 480))
    args = ("facemesh", "facemesh-pose-iris-1", MONITORS)
    assert data.is_compatible(*args, camera="1") == (False, "the camera changed")
    assert data.is_compatible(*args, camera=" 0 ") == (True, "")
    # Unknown on either side: no reason to reject.
    assert data.is_compatible(*args, camera="") == (True, "")
    assert data.is_compatible(*args, camera=None, frame_size=(0, 0)) == (True, "")
    assert make_data(camera="").is_compatible(*args, camera="1") == (True, "")
    # The resolution does not matter, the field of view (aspect ratio) does.
    assert data.is_compatible(*args, camera="0", frame_size=(1280, 960)) == (True, "")
    ok, reason = data.is_compatible(*args, camera="0", frame_size=(1280, 720))
    assert not ok
    assert "aspect ratio" in reason
    assert "640x480" in reason
    assert "1280x720" in reason
    assert data.camera_matches("1") == (False, "the camera changed")
    assert data.camera_matches(frame_size=(320, 240)) == (True, "")


def test_same_aspect() -> None:
    assert same_aspect((640, 480), (1280, 960))
    assert same_aspect((1920, 1080), (1280, 720))
    assert same_aspect((1920, 1080), (1920, 1088))  # within 1 % (encoder padding)
    assert not same_aspect((640, 480), (640, 360))
    assert same_aspect((0, 0), (640, 360))
    assert same_aspect((640, 480), (0, 0))


# --------------------------------------------------------------------------- robustness
def test_missing_file_returns_none_quietly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert load_calibration(tmp_path / "nope.json") is None
        assert len(CalibrationLibrary.load(tmp_path / "nope.json")) == 0
    assert caplog.text == ""


def test_unreadable_path_returns_none(tmp_path: Path) -> None:
    assert load_calibration(tmp_path) is None  # a directory, not a file


def _saved_doc(tmp_path: Path) -> tuple[Path, dict]:
    path = tmp_path / "calibration.json"
    save_calibration(path, make_data())
    return path, json.loads(path.read_text(encoding="utf-8"))


# Corruptions of one profile (in a version 2 file) or of a whole version 1 file.
PROFILE_CORRUPTIONS = [
    lambda doc: {k: v for k, v in doc.items() if k != "model"},
    lambda doc: {k: v for k, v in doc.items() if k != "backend"},
    lambda doc: {**doc, "backend": 42},
    lambda doc: {**doc, "monitors": "all of them"},
    lambda doc: {**doc, "monitors": [{"index": 0}]},
    lambda doc: {**doc, "samples": [{"features": [1, 2]}]},
    lambda doc: {**doc, "samples": [{**doc["samples"][0], "features": []}]},
    lambda doc: {**doc, "samples": [{**doc["samples"][0], "features": [1.0, 2.0]}]},
    lambda doc: {**doc, "samples": [{**doc["samples"][0], "weight": -1}]},
    lambda doc: {**doc, "implicit_samples": {"a": 1}},
    lambda doc: {**doc, "model": {**doc["model"], "coef": [[1.0, 2.0]]}},
    lambda doc: {**doc, "model": {**doc["model"], "nonlinear": [0, 99]}},
    # Values that overflow int()/float() used to escape as OverflowError.
    lambda doc: {**doc, "model": {**doc["model"], "bounds": [0, 0, 1920, math.inf]}},
    lambda doc: {**doc, "model": {**doc["model"], "degree": math.inf}},
    lambda doc: {**doc, "monitors": [{**doc["monitors"][0], "rect": [0, 0, 1920, 1e400]}]},
    lambda doc: {**doc, "monitors": [{**doc["monitors"][0], "index": -math.inf}]},
    lambda doc: {**doc, "monitors": [{**doc["monitors"][0], "scale": math.inf}]},
    lambda doc: {**doc, "samples": [{**doc["samples"][0], "point": math.inf}]},
    lambda doc: {**doc, "samples": [{**doc["samples"][0], "monitor": math.nan}]},
    lambda doc: {**doc, "samples": [{**doc["samples"][0], "x": 10**400}]},
    lambda doc: "not an object",
]


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda doc: "{ not json",
        lambda doc: "[]",
        lambda doc: "",
        lambda doc: "\x00\xff garbage",
        lambda doc: {**doc, "format": "something-else"},
        lambda doc: {**doc, "version": 0},
        lambda doc: {**doc, "version": "1"},
        lambda doc: {**doc, "profiles": "many"},
        lambda doc: {k: v for k, v in doc.items() if k != "profiles"},
        # Deep nesting makes the JSON parser recurse until RecursionError.
        lambda doc: (
            '{"format": "eye-tracker-calibration", "version": 2, "profiles": '
            + "[" * 100_000
            + "]" * 100_000
            + "}"
        ),
        *(
            lambda doc, c=c: {**doc, "profiles": [c(doc["profiles"][0])]}
            for c in PROFILE_CORRUPTIONS
        ),
    ],
)
def test_corrupt_files_return_none_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, corrupt
) -> None:
    path, doc = _saved_doc(tmp_path)
    content = corrupt(doc)
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="eye_tracker.gaze.store"):
        assert load_calibration(path) is None
    assert "calibration" in caplog.text.lower()


@pytest.mark.parametrize("corrupt", PROFILE_CORRUPTIONS)
def test_corrupt_version_1_files_return_none(tmp_path: Path, corrupt) -> None:
    path = tmp_path / "calibration.json"
    content = corrupt(v1_doc(make_data(), tmp_path))
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    assert load_calibration(path) is None


def test_file_with_byte_order_mark_loads(tmp_path: Path) -> None:
    # Windows PowerShell 5.1 `Set-Content -Encoding UTF8` writes a BOM.
    path, _ = _saved_doc(tmp_path)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.backend == "facemesh"


def test_newer_version_is_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path, doc = _saved_doc(tmp_path)
    doc["version"] = CALIBRATION_VERSION + 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="eye_tracker.gaze.store"):
        assert load_calibration(path) is None
    assert "newer" in caplog.text


def test_invalid_report_is_replaced_by_empty_dict(tmp_path: Path) -> None:
    path, doc = _saved_doc(tmp_path)
    doc["profiles"][0]["report"] = "excellent"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.report == {}


@pytest.mark.parametrize(
    ("camera", "frame_size", "expected"),
    [
        (7, [640, 480], ("", (640, 480))),
        (None, [640], ("", (0, 0))),
        ("0", "640x480", ("0", (0, 0))),
        ("0", [640.0, 480], ("0", (0, 0))),
        ("0", [-1, 480], ("0", (0, 0))),
        ("0", [True, 480], ("0", (0, 0))),
    ],
)
def test_malformed_camera_metadata_means_unknown(
    tmp_path: Path, camera: object, frame_size: object, expected: tuple[str, tuple[int, int]]
) -> None:
    path, doc = _saved_doc(tmp_path)
    doc["profiles"][0].update(camera=camera, frame_size=frame_size)
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = load_calibration(path)
    assert loaded is not None  # optional metadata never costs the calibration
    assert (loaded.camera, loaded.frame_size) == expected


def test_optional_fields_default(tmp_path: Path) -> None:
    path, doc = _saved_doc(tmp_path)
    doc.pop("format")
    profile = doc["profiles"][0]
    for key in ("implicit_samples", "report", "created_at", "camera", "frame_size"):
        profile.pop(key)
    for sample in profile["samples"]:
        sample.pop("weight")
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.implicit_samples == []
    assert all(s.weight == 1.0 for s in loaded.samples)
    assert (loaded.camera, loaded.frame_size) == ("", (0, 0))


# --------------------------------------------------------------------------- profiles
def test_save_replaces_the_same_setup_and_keeps_others(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    office = make_data(created_at="2026-09-01T08:00:00+00:00")
    save_calibration(path, office)
    home = make_data(monitors=HOME, created_at="2026-09-02T08:00:00+00:00")
    save_calibration(path, home)
    library = CalibrationLibrary.load(path)
    assert [p.created_at for p in library] == ["2026-09-02T08:00:00+00:00", office.created_at]
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.created_at == home.created_at  # the most recent one

    # A new calibration of the office replaces the old office profile only.
    again = make_data(n_points=4, created_at="2026-09-03T08:00:00+00:00")
    save_calibration(path, again)
    library = CalibrationLibrary.load(path)
    assert [p.created_at for p in library] == [again.created_at, home.created_at]
    assert len(library.profiles[0].samples) == 20
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calibration.json"]


def test_library_picks_the_profile_for_the_current_desk(tmp_path: Path) -> None:
    office, home = make_data(), make_data(monitors=HOME)
    library = CalibrationLibrary([home, office])
    args = ("facemesh", "facemesh-pose-iris-1")
    assert library.best(*args, MONITORS, camera="0") is office
    assert library.best(*args, HOME, camera="0") is home
    assert library.match(*args, HOME) == (home, "")
    # Another backend or camera: nothing fits, and the reason says why.
    assert library.match("lite", "yunet-geom-1", HOME)[0] is None
    assert "lite" in library.match("lite", "yunet-geom-1", HOME)[1]
    assert library.match(*args, HOME, camera="1") == (None, "the camera changed")
    third = [Monitor(0, "tv", Rect(0, 0, 3840, 2160))]
    assert library.match(*args, third) == (None, "the monitor layout changed")
    assert CalibrationLibrary().match(*args, HOME) == (None, "not calibrated yet")


def test_library_prefers_a_known_camera_then_the_most_recent() -> None:
    legacy = make_data(camera="", frame_size=(0, 0), created_at="legacy")
    webcam = make_data(camera="1", frame_size=(1280, 720), created_at="webcam")
    builtin = make_data(camera="0", frame_size=(640, 480), created_at="builtin")
    library = CalibrationLibrary([legacy, webcam, builtin])  # legacy most recent
    args = ("facemesh", "facemesh-pose-iris-1", MONITORS)
    assert library.best(*args, camera="1") is webcam
    assert library.best(*args, camera="0") is builtin
    assert library.best(*args, camera="2") is legacy  # unknown camera: may be right
    assert library.best(*args) is legacy  # camera unknown: most recently used
    assert library.best(*args, frame_size=(640, 480)) is builtin  # a known match wins
    assert library.best(*args, camera="0", frame_size=(1280, 720)) is legacy


def test_library_put_replaces_same_key_and_unknown_camera_profile() -> None:
    legacy = make_data(camera="", created_at="legacy")
    other_desk = make_data(monitors=HOME, camera="", created_at="home")
    library = CalibrationLibrary([legacy, other_desk])
    fresh = make_data(camera="0", created_at="fresh")
    replaced = library.put(fresh)
    # The unknown-camera profile of the same setup was most likely this camera.
    assert replaced == [legacy]
    assert library.profiles == [fresh, other_desk]
    newer = make_data(camera="0", created_at="newer")
    assert library.put(newer) == [fresh]
    assert library.profiles == [newer, other_desk]
    external = make_data(camera="1", created_at="external")
    assert library.put(external) == []
    assert library.profiles == [external, newer, other_desk]
    assert library.get(newer.key) is newer
    assert library.get(("nope", "", "", "")) is None
    # Putting a stored profile again (after a refit) just makes it the latest.
    assert library.put(other_desk) == []
    assert library.latest is other_desk
    assert len(library) == 3


def test_library_keeps_at_most_max_profiles_least_recently_used_first(tmp_path: Path) -> None:
    library = CalibrationLibrary(max_profiles=3)
    datas = [make_data(camera=str(i), created_at=f"c{i}") for i in range(4)]
    for d in datas[:3]:
        library.put(d)
    assert library.mark_used(datas[0])  # c0 used most recently now
    evicted = library.put(datas[3])
    assert evicted == [datas[1]]  # the least recently used
    assert [p.created_at for p in library] == ["c3", "c0", "c2"]
    path = tmp_path / "calibration.json"
    library.save(path)
    reloaded = CalibrationLibrary.load(path)
    assert [p.created_at for p in reloaded] == ["c3", "c0", "c2"]  # order persists
    assert len(CalibrationLibrary.load(path, max_profiles=2)) == 2
    assert MAX_PROFILES == 8
    with pytest.raises(ValueError, match="max_profiles"):
        CalibrationLibrary(max_profiles=0)


def test_library_remove_clear_and_membership() -> None:
    a, b = make_data(camera="0"), make_data(camera="1")
    library = CalibrationLibrary([a, b])
    assert a in library
    assert not library.mark_used(make_data(camera="2"))
    assert library.remove(a)
    assert not library.remove(a)
    assert a not in library
    assert list(library) == [b]
    library.clear()
    assert library.latest is None
    assert len(library) == 0


def test_corrupt_profile_is_skipped_and_the_others_kept(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "calibration.json"
    CalibrationLibrary([make_data(camera="0"), make_data(camera="1")]).save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["profiles"][0]["model"]["bounds"] = [0, 0, "wide", 10]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="eye_tracker.gaze.store"):
        library = CalibrationLibrary.load(path)
    assert [p.camera for p in library] == ["1"]
    assert "profile 1" in caplog.text


def test_duplicate_keys_in_a_file_keep_the_most_recent(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    CalibrationLibrary([make_data(created_at="new")]).save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["profiles"].append({**doc["profiles"][0], "created_at": "old"})
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert [p.created_at for p in CalibrationLibrary.load(path)] == ["new"]


def test_profiles_are_shared_by_reference_so_refits_persist(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    library = CalibrationLibrary([make_data(implicit=0)])
    data = library.best("facemesh", "facemesh-pose-iris-1", MONITORS)
    assert data is not None
    data.implicit_samples = [CalibrationSample(np.ones(8), 5.0, 6.0, 0, -1, 0.5)]
    library.save(path)
    reloaded = load_calibration(path)
    assert reloaded is not None
    assert len(reloaded.implicit_samples) == 1


def test_empty_library_saves_a_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    CalibrationLibrary().save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"] == []
    assert load_calibration(path) is None


# ------------------------------------------------------------ gaze accuracy
SIDE_BY_SIDE = [
    Monitor(0, "left", Rect(0, 0, 1920, 1080), primary=True),
    Monitor(1, "right", Rect(1920, 0, 1920, 1080)),
]


def _linear_data(report: dict[str, Any]) -> CalibrationData:
    """Two monitors; the features are the gaze point in kilo-pixels plus noise."""
    rng = np.random.default_rng(11)
    samples = []
    point = 0
    for m in SIDE_BY_SIDE:
        for nx in (0.1, 0.5, 0.9):
            for ny in (0.1, 0.5, 0.9):
                x, y = m.rect.denormalize(nx, ny)
                for _ in range(4):
                    noisy = np.array([x, y]) / 1000.0 + rng.normal(scale=0.02, size=2)
                    samples.append(CalibrationSample(noisy, x, y, m.index, point))
                point += 1
    X = np.array([s.features for s in samples])
    Y = np.array([(s.x, s.y) for s in samples])
    model = GazeModel(degree=1, alpha=0.01).fit(X, Y, bounds=virtual_bounds(SIDE_BY_SIDE))
    return CalibrationData(
        backend="lite",
        feature_version="v1",
        layout_signature=layout_signature(SIDE_BY_SIDE),
        monitors=list(SIDE_BY_SIDE),
        samples=samples,
        implicit_samples=[],
        model=model,
        report=report,
    )


def test_axis_error_prefers_the_stored_value() -> None:
    data = _linear_data({"per_monitor_error_px": {"0": [12.0, 34.0]}, "mean_error_px": 99.0})
    assert axis_error(data, data.monitors[0]) == (12.0, 34.0)
    # Matched by rectangle: the same screen under another number.
    assert axis_error(data, Monitor(5, "left", data.monitors[0].rect)) == (12.0, 34.0)
    # A monitor without a stored value falls back to the mean error / √2.
    assert axis_error(data, data.monitors[1]) == pytest.approx((70.0, 70.0), abs=0.01)


def test_axis_error_is_recomputed_for_old_profiles_and_saved(tmp_path: Path) -> None:
    data = _linear_data({"grade": "good", "alpha": 0.01, "degree": 1, "mean_error_px": 500.0})
    sx, sy = axis_error(data, data.monitors[1]) or (0.0, 0.0)
    # Feature noise of 0.02 kilo-pixels is some 20 px of gaze error, far below the
    # fallback from the (deliberately wrong) mean error.
    assert 5.0 < sx < 80.0
    assert 5.0 < sy < 80.0
    assert set(data.report["per_monitor_error_px"]) == {"0", "1"}
    path = tmp_path / "calibration.json"
    CalibrationLibrary([data]).save(path)
    reloaded = load_calibration(path)
    assert reloaded is not None
    assert axis_error(reloaded, reloaded.monitors[1]) == pytest.approx((sx, sy))


def test_axis_error_without_any_information() -> None:
    data = _linear_data({"mean_error_px": None})
    data.samples = data.samples[:4]  # a single dot: nothing can be held out
    assert axis_error(data, data.monitors[0]) is None
    data = _linear_data({"mean_error_px": 42.0})
    data.samples = []
    assert axis_error(data, data.monitors[0]) == pytest.approx((29.7, 29.7), abs=0.01)


def test_recomputing_an_old_profile_is_cheap_enough_for_the_gui_thread() -> None:
    """The first axis_error of an old profile runs on the GUI thread (see its docs):
    a realistic two-monitor calibration (16 dots per monitor, 30 samples each,
    degree 3, the face-mesh gaze features nonlinear) must take milliseconds."""
    import time

    from eye_tracker.gaze.calibration import evaluate
    from gaze_synth import GAZE, TWO, calibration_samples

    samples = calibration_samples(
        TWO, np.random.default_rng(3), noise=1.0, per_point=30, points_per_monitor=16
    )
    model, report = evaluate(samples, TWO, degree=3, nonlinear=GAZE)
    saved = report.to_dict()
    expected = saved.pop("per_monitor_error_px")  # as written by an earlier version
    data = CalibrationData(
        backend="facemesh",
        feature_version="v",
        layout_signature=layout_signature(TWO),
        monitors=list(TWO),
        samples=samples,
        implicit_samples=[],
        model=model,
        report=saved,
    )
    started = time.perf_counter()
    sigma = axis_error(data, TWO[1])
    elapsed = time.perf_counter() - started
    assert sigma == pytest.approx(tuple(expected["1"]), rel=1e-6)  # same as a new report
    # Measured: ~4 ms on a desktop. The bound leaves room for slow CI machines.
    assert elapsed < 0.25, f"{elapsed * 1000:.0f} ms"
