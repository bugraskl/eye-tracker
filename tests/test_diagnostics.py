"""Tests for eye_tracker.diagnostics (``doctor`` and ``bench``).

Nothing here opens a camera: camera probing is faked, and the benchmark runs on
a synthetic image written to a temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from eye_tracker import __version__, diagnostics, paths
from eye_tracker.config import Settings
from eye_tracker.gaze.calibration import CalibrationSample
from eye_tracker.gaze.model import GazeModel
from eye_tracker.gaze.store import CalibrationData, save_calibration
from eye_tracker.types import Monitor, Rect, layout_signature, virtual_bounds
from eye_tracker.vision.camera import CameraError

SECTIONS = {
    "generated_at",
    "app",
    "system",
    "qt",
    "monitors",
    "libraries",
    "backends",
    "cameras",
    "platform",
    "hotkeys",
    "autostart",
    "paths",
    "settings",
    "calibration",
    "problems",
}


@pytest.fixture
def env(qapp: Any, app_dirs: Path) -> Path:
    """A QApplication (so no QGuiApplication is created here) and private dirs."""
    return app_dirs


def _calibration(monitors: list[Monitor], backend: str, feature_version: str) -> CalibrationData:
    rng = np.random.default_rng(3)
    samples = []
    for pid in range(8):
        monitor = monitors[pid % len(monitors)]
        x, y = monitor.rect.denormalize(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        for _ in range(3):
            samples.append(CalibrationSample(rng.normal(size=6), x, y, monitor.index, pid))
    X = np.array([s.features for s in samples])
    Y = np.array([(s.x, s.y) for s in samples])
    model = GazeModel(degree=1, alpha=1.0).fit(X, Y, bounds=virtual_bounds(monitors))
    return CalibrationData(
        backend=backend,
        feature_version=feature_version,
        layout_signature=layout_signature(monitors),
        monitors=list(monitors),
        samples=samples,
        implicit_samples=[],
        model=model,
        report={
            "grade": "good",
            "monitor_accuracy": 0.93,
            "mean_error_px": 120.0,
            "median_error_px": 100.0,
            "per_monitor_accuracy": {"0": 0.93},
            "n_samples": len(samples),
            "n_points": 8,
            "alpha": 1.0,
        },
        created_at="2026-09-30T08:15:00+00:00",
    )


def _face_free_image(path: Path) -> Path:
    """A smooth synthetic picture (no face), written for the benchmark."""
    ys, xs = np.mgrid[0:240, 0:320]
    image = np.stack([xs % 256, ys % 256, (xs + ys) % 256], axis=-1).astype(np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


# ------------------------------------------------------------------------- doctor
def test_collect_report_structure(env: Path) -> None:
    report = diagnostics.collect_report(probe_cameras=False)
    assert set(report) == SECTIONS
    # The whole report is JSON-serialisable without a custom encoder.
    json.loads(json.dumps(report))

    assert report["app"]["version"] == __version__
    assert report["app"]["instance_running"] is False
    assert report["qt"]["platform_plugin"] == "offscreen"
    assert report["monitors"]["count"] >= 1
    assert report["monitors"]["virtual"] is True
    assert set(report["backends"]["models"]) == {"mediapipe", "opencv"}
    assert report["cameras"]["probed"] is False
    assert "devices" not in report["cameras"]
    assert set(report["platform"]["capabilities"]) >= {"lock", "cursor", "hotkeys"}
    assert report["settings"] == {"file_exists": False, "non_default": {}}
    assert report["calibration"] == {"exists": False}
    assert isinstance(report["autostart"]["command"], str)
    assert 'Not calibrated yet: run "eye-tracker calibrate".' in report["problems"]
    # The fake offscreen screen is not reported as a one-monitor setup.
    assert not any("Only one monitor" in p for p in report["problems"])


def test_report_paths_are_relative_to_home(env: Path) -> None:
    report = diagnostics.collect_report()
    settings_file = report["paths"]["settings_file"]
    home = str(Path.home())
    if str(paths.settings_file()).lower().startswith(home.lower()):
        assert settings_file.startswith("~")
        assert home not in settings_file
    assert report["paths"]["ipc_name"] == paths.ipc_name()


def test_camera_probe_is_opt_in(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eye_tracker.vision import camera

    calls: list[tuple[int, str]] = []

    def fake_list(max_index: int = 4, api: str = "auto") -> list[camera.CameraInfo]:
        calls.append((max_index, api))
        return [camera.CameraInfo(index=0, name="Camera 0", width=640, height=480)]

    monkeypatch.setattr(camera, "list_cameras", fake_list)
    diagnostics.collect_report(probe_cameras=False)
    assert calls == []
    report = diagnostics.collect_report(probe_cameras=True)
    assert calls == [(4, "auto")]
    assert report["cameras"]["devices"] == [
        {"index": 0, "name": "Camera 0", "width": 640, "height": 480}
    ]
    text = diagnostics.format_report(report)
    assert "camera 0" in text
    assert "Camera 0 (640x480)" in text

    monkeypatch.setattr(camera, "list_cameras", lambda max_index=4, api="auto": [])
    report = diagnostics.collect_report(probe_cameras=True)
    assert "No camera delivered a picture." in report["problems"]
    assert "(no camera found)" in diagnostics.format_report(report)


def test_settings_section_lists_changes_without_side_effects(env: Path) -> None:
    settings = Settings()
    settings.general.backend = "opencv"
    settings.presence.away_timeout_s = 120
    settings.save(paths.settings_file())
    report = diagnostics.collect_report()
    assert report["settings"]["valid"] is True
    assert report["settings"]["non_default"] == {
        "general.backend": "opencv",
        "presence.away_timeout_s": 120,
    }
    assert report["backends"]["configured"] == "opencv"
    assert report["backends"]["active"] == "opencv"

    paths.settings_file().write_text("{broken", encoding="utf-8")
    report = diagnostics.collect_report()
    assert report["settings"]["valid"] is False
    assert "The settings file is not valid JSON; defaults are used." in report["problems"]
    # Unlike Settings.load, the doctor never moves the broken file aside.
    assert paths.settings_file().read_text(encoding="utf-8") == "{broken"


def test_calibration_section(env: Path) -> None:
    settings = Settings()
    settings.general.backend = "opencv"
    settings.save(paths.settings_file())
    current = diagnostics.current_monitors()
    save_calibration(paths.calibration_file(), _calibration(current, "opencv", "yunet-geom-1"))

    calibration = diagnostics.collect_report()["calibration"]
    assert calibration["exists"] is True
    assert calibration["valid"] is True
    assert calibration["compatible"] is True
    assert calibration["grade"] == "good"
    assert calibration["samples"] == 24
    assert calibration["summary"].startswith("Good")

    other = [
        Monitor(0, "left", Rect(-1920, 0, 1920, 1080), primary=True),
        Monitor(1, "right", Rect(0, 0, 2560, 1440)),
    ]
    save_calibration(paths.calibration_file(), _calibration(other, "opencv", "yunet-geom-1"))
    report = diagnostics.collect_report()
    assert report["calibration"]["compatible"] is False
    assert "monitor layout" in report["calibration"]["reason"]
    assert any(p.startswith("The calibration no longer applies") for p in report["problems"])

    paths.calibration_file().write_text("not json", encoding="utf-8")
    report = diagnostics.collect_report()
    assert report["calibration"] == {"exists": True, "valid": False}
    assert "The calibration file is damaged; please recalibrate." in report["problems"]


def test_format_report(env: Path) -> None:
    report = diagnostics.collect_report()
    text = diagnostics.format_report(report)
    lines = text.splitlines()
    assert lines[0] == f"Eye Tracker {__version__} - diagnostics"
    assert lines[1] == "=" * len(lines[0])
    for heading in ("Problems", "App", "Qt", "Monitors", "Vision backends", "Paths"):
        assert heading in lines
    assert "virtual screen" in text
    assert text.endswith("\n")
    # Fixed texts are ASCII so the report survives any console code page.
    assert "—" not in text


def test_format_report_handles_errors_and_odd_values() -> None:
    report = {
        "app": {"version": "9.9"},
        "problems": [],
        "system": {"error": "OSError: nope"},
        "settings": {"non_default": {}, "flag": True, "missing": None, "ratio": 0.5},
        "cameras": {"probed": False, "devices": []},
        "monitors": {"count": 0, "items": []},
    }
    text = diagnostics.format_report(report)
    assert "Eye Tracker 9.9 - diagnostics" in text
    assert "none found" in text
    assert "error  OSError: nope" in text
    assert "flag         yes" in text
    assert "missing      unknown" in text
    assert "ratio        0.5" in text
    assert "non_default  -" in text


def test_problems_from_a_synthetic_report() -> None:
    report = {
        "app": {"instance_running": True},
        "system": {"error": "RuntimeError: x"},
        "backends": {
            "available": [],
            "models": {
                "mediapipe": {"file": "a.task", "present": False},
                "opencv": {"file": "b.onnx", "present": True, "sha256_ok": False},
            },
            "active_error": "No vision backend is available",
        },
        "monitors": {"count": 1},
        "cameras": {"probed": True, "devices": []},
        "platform": {
            "permissions": {"camera": False, "accessibility": False},
            "capabilities": {"cursor": False, "lock": False},
        },
        "hotkeys": {"enabled": True, "note": "Wayland", "invalid": {"recalibrate": "bad"}},
        "settings": {"valid": True},
        "calibration": {"exists": True, "valid": True, "compatible": True},
    }
    problems = diagnostics._problems(report)
    assert problems == [
        "Could not collect system information: RuntimeError: x",
        "No vision backend is available (MediaPipe or OpenCV with its model).",
        "The mediapipe model file a.task is missing.",
        "The opencv model file b.onnx is damaged (checksum).",
        "No vision backend is available",
        "Only one monitor detected: switching needs two or more "
        "(walk-away and privacy features still work).",
        "No camera delivered a picture (it may be in use by Eye Tracker itself).",
        "Camera access is blocked by the operating system's privacy settings.",
        "Accessibility permission is missing: keyboard focus cannot follow your gaze.",
        "This session does not allow moving the cursor (on Wayland install ydotool).",
        "Locking the screen is not supported here.",
        "Global hotkeys: Wayland",
        "Hotkey recalibrate is invalid: bad",
    ]


def test_redact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home" / "alice"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    sep = "\\" if "\\" in str(home) else "/"
    assert diagnostics._redact(None) is None
    assert diagnostics._redact(home) == "~"
    assert diagnostics._redact(f"{home}{sep}x{sep}y.log") == f"~{sep}x{sep}y.log"
    # A sibling whose name merely starts with the home path is left alone.
    assert diagnostics._redact(f"{home}2{sep}x") == f"{home}2{sep}x"
    assert diagnostics._redact("relative/path") == "relative/path"


def test_format_command() -> None:
    text = diagnostics.format_command(["/usr/bin/python 3", "-m", "eye_tracker"])
    assert "eye_tracker" in text
    assert text.count('"') + text.count("'") == 2  # only the path with a space is quoted


def test_describe_device() -> None:
    assert diagnostics._describe_device("1") == "camera 1"
    assert diagnostics._describe_device(" ") == "none"
    assert diagnostics._describe_device(str(Path("videos") / "me.mp4")) == "file me.mp4"


# -------------------------------------------------------------------------- bench
def test_run_bench_on_an_image(tmp_path: Path) -> None:
    image = _face_free_image(tmp_path / "frame.png")
    result = diagnostics.run_bench(0.25, str(image), "opencv", idle_fps=20)
    assert result["backend"] == "opencv"
    assert result["feature_version"] == "yunet-geom-1"
    assert result["device"] == "file frame.png"
    assert result["frame_size"] == [320, 240]
    assert set(result["modes"]) == {"max", "idle"}
    fast, idle = result["modes"]["max"], result["modes"]["idle"]
    assert fast["target_fps"] is None
    assert fast["frames"] > 0
    assert fast["analysed"] == fast["frames"]
    assert fast["face_ratio"] == 0.0
    assert fast["inference_ms"]["p50"] is not None
    assert idle["target_fps"] == 20
    # A still image: the motion gate skips almost every frame after the first.
    assert 1 <= idle["analysed"] < idle["frames"]
    assert idle["skip_ratio"] > 0
    json.dumps(result)

    text = diagnostics.format_bench(result)
    assert text.startswith("Eye Tracker benchmark - opencv backend (yunet-geom-1)")
    assert "max speed" in text
    assert "idle (20 fps)" in text
    assert "frames per second" in text


def test_run_bench_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        diagnostics.run_bench(0, "0", "opencv")
    with pytest.raises(CameraError, match="not found"):
        diagnostics.run_bench(0.1, str(tmp_path / "missing.png"), "opencv")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not an image")
    with pytest.raises(diagnostics.BenchError):
        diagnostics.run_bench(0.1, str(broken), "opencv")


def test_format_bench_with_missing_values() -> None:
    result = {
        "backend": "opencv",
        "feature_version": "v",
        "device": "camera 0",
        "frame_size": [640, 480],
        "cpu_count": 4,
        "modes": {
            "max": {
                "target_fps": None,
                "fps": 0.0,
                "analysed_fps": 0.0,
                "inference_ms": {"p50": None, "p95": None},
                "cpu_percent": None,
                "cpu_percent_core": None,
                "face_ratio": None,
            }
        },
    }
    text = diagnostics.format_bench(result)
    assert "max speed" in text
    assert "idle" not in text
    assert "inference p50 / p95   -" in text
