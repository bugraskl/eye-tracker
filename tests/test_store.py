"""Tests for eye_tracker.gaze.store (calibration file round trip and robustness)."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from eye_tracker.gaze.calibration import CalibrationSample
from eye_tracker.gaze.model import GazeModel
from eye_tracker.gaze.store import (
    CALIBRATION_VERSION,
    CalibrationData,
    load_calibration,
    save_calibration,
    utc_now_iso,
)
from eye_tracker.types import Monitor, Rect, layout_signature, virtual_bounds

MONITORS = [
    Monitor(0, "DELL U2720Q", Rect(0, 0, 3840, 2160), primary=True, scale=1.5),
    Monitor(1, "LG ÜltraGear", Rect(3840, 540, 1920, 1080), scale=1.0),
]


def make_data(n_points: int = 12, per_point: int = 5, implicit: int = 4) -> CalibrationData:
    rng = np.random.default_rng(7)
    samples = []
    for pid in range(n_points):
        monitor = MONITORS[pid % 2]
        x, y = monitor.rect.denormalize(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        for _ in range(per_point):
            samples.append(CalibrationSample(rng.normal(size=8), x, y, monitor.index, pid))
    learned = [
        CalibrationSample(rng.normal(size=8), 100.0 + i, 200.0, 0, -(i + 1), weight=0.5)
        for i in range(implicit)
    ]
    X = np.array([s.features for s in samples])
    Y = np.array([(s.x, s.y) for s in samples])
    model = GazeModel(degree=3, alpha=0.1).fit(X, Y, bounds=virtual_bounds(MONITORS))
    return CalibrationData(
        backend="mediapipe",
        feature_version="mp-pose-iris-1",
        layout_signature=layout_signature(MONITORS),
        monitors=list(MONITORS),
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
        created_at="2026-09-30T08:15:00+00:00",
    )


def assert_samples_close(a: list[CalibrationSample], b: list[CalibrationSample]) -> None:
    assert len(a) == len(b)
    for s, t in zip(a, b, strict=True):
        assert np.allclose(s.features, t.features, atol=1e-6)
        assert (s.x, s.y) == pytest.approx((t.x, t.y), abs=1e-6)
        assert (s.monitor_index, s.point_id) == (t.monitor_index, t.point_id)
        assert s.weight == pytest.approx(t.weight)


# --------------------------------------------------------------------------- round trip
def test_round_trip(tmp_path: Path) -> None:
    data = make_data()
    path = tmp_path / "sub" / "calibration.json"
    save_calibration(path, data)
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.backend == "mediapipe"
    assert loaded.feature_version == "mp-pose-iris-1"
    assert loaded.layout_signature == data.layout_signature
    assert loaded.monitors == MONITORS
    assert loaded.created_at == "2026-09-30T08:15:00+00:00"
    assert_samples_close(loaded.samples, data.samples)
    assert_samples_close(loaded.implicit_samples, data.implicit_samples)
    assert (loaded.model.degree, loaded.model.alpha) == (3, 0.1)
    X = np.array([s.features for s in data.samples])
    assert np.allclose(loaded.model.predict(X), data.model.predict(X), atol=1e-3)
    assert loaded.report["grade"] == "excellent"
    assert loaded.report["mean_error_px"] is None  # NaN is not valid JSON
    assert loaded.report["per_monitor_accuracy"] == {"0": 1.0, "1": 0.98}
    assert loaded.report["n_samples"] == 60
    assert loaded.grade == "excellent"
    assert loaded.is_compatible("mediapipe", "mp-pose-iris-1", MONITORS) == (True, "")


def test_file_is_readable_json_without_images(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    save_calibration(path, make_data())
    text = path.read_text(encoding="utf-8")
    doc = json.loads(text)
    assert doc["format"] == "eye-tracker-calibration"
    assert doc["version"] == CALIBRATION_VERSION
    assert set(doc) == {
        "format", "version", "created_at", "backend", "feature_version", "layout_signature",
        "monitors", "report", "model", "samples", "implicit_samples",
    }  # fmt: skip
    assert set(doc["samples"][0]) == {"features", "x", "y", "monitor", "point", "weight"}
    # One sample per line keeps the file diff-friendly and compact.
    assert text.count("\n") < 120
    assert "LG ÜltraGear" in text
    # Sample floats are rounded to six decimals.
    for value in doc["samples"][0]["features"]:
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
    ok, reason = loaded.is_compatible("mediapipe", "mp-pose-iris-1", MONITORS)
    assert not ok
    assert "no fitted model" in reason


# --------------------------------------------------------------------------- compatibility
def test_is_compatible_reasons() -> None:
    data = make_data()
    ok, reason = data.is_compatible("opencv", "mp-pose-iris-1", MONITORS)
    assert not ok
    assert "opencv" in reason
    assert "mediapipe" in reason
    ok, reason = data.is_compatible("mediapipe", "mp-pose-iris-2", MONITORS)
    assert not ok
    assert "mp-pose-iris-2" in reason
    moved = [MONITORS[0], Monitor(1, "LG", Rect(-1920, 0, 1920, 1080))]
    ok, reason = data.is_compatible("mediapipe", "mp-pose-iris-1", moved)
    assert not ok
    assert "layout" in reason
    # Monitor order and names do not matter, only the geometry.
    renamed = [Monitor(5, "x", MONITORS[1].rect), Monitor(6, "y", MONITORS[0].rect)]
    assert data.is_compatible("mediapipe", "mp-pose-iris-1", renamed) == (True, "")


# --------------------------------------------------------------------------- robustness
def test_missing_file_returns_none_quietly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert load_calibration(tmp_path / "nope.json") is None
    assert caplog.text == ""


def test_unreadable_path_returns_none(tmp_path: Path) -> None:
    assert load_calibration(tmp_path) is None  # a directory, not a file


def _saved_doc(tmp_path: Path) -> tuple[Path, dict]:
    path = tmp_path / "calibration.json"
    save_calibration(path, make_data())
    return path, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda doc: "{ not json",
        lambda doc: "[]",
        lambda doc: "",
        lambda doc: {**doc, "format": "something-else"},
        lambda doc: {**doc, "version": 0},
        lambda doc: {**doc, "version": "1"},
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
    ],
)
def test_corrupt_files_return_none_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, corrupt
) -> None:
    path, doc = _saved_doc(tmp_path)
    content = corrupt(doc)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="eye_tracker.gaze.store"):
        assert load_calibration(path) is None
    assert "calibration" in caplog.text.lower()


def test_newer_version_is_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path, doc = _saved_doc(tmp_path)
    doc["version"] = CALIBRATION_VERSION + 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="eye_tracker.gaze.store"):
        assert load_calibration(path) is None
    assert "newer" in caplog.text


def test_invalid_report_is_replaced_by_empty_dict(tmp_path: Path) -> None:
    path, doc = _saved_doc(tmp_path)
    doc["report"] = "excellent"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.report == {}


def test_optional_fields_default(tmp_path: Path) -> None:
    path, doc = _saved_doc(tmp_path)
    for key in ("format", "implicit_samples", "report", "created_at"):
        doc.pop(key)
    for sample in doc["samples"]:
        sample.pop("weight")
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.implicit_samples == []
    assert all(s.weight == 1.0 for s in loaded.samples)


def test_save_overwrites_atomically(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    first = make_data()
    save_calibration(path, first)
    second = make_data(n_points=4)
    second.backend = "opencv"
    save_calibration(path, second)
    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.backend == "opencv"
    assert len(loaded.samples) == 20
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calibration.json"]
