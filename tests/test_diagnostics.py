"""Tests for eye_tracker.diagnostics (``doctor`` and ``bench``).

Nothing here opens a camera: camera probing is faked, and the benchmark runs on
a synthetic image written to a temporary directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from eye_tracker import __version__, cli, diagnostics, ipc, paths
from eye_tracker.config import Settings
from eye_tracker.gaze.calibration import CalibrationSample
from eye_tracker.gaze.model import GazeModel
from eye_tracker.gaze.store import CalibrationData, save_calibration
from eye_tracker.platform.base import PlatformServices
from eye_tracker.platform.hotkeys import Hotkey, HotkeyManager, parse_hotkey
from eye_tracker.types import Monitor, Rect, layout_signature, virtual_bounds
from eye_tracker.vision.backends import MODEL_FILES
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
    assert set(report["backends"]["models"]) == set(MODEL_FILES)
    assert "mediapipe_installed" not in report["backends"]
    assert "mediapipe" not in report["libraries"]
    assert report["cameras"]["probed"] is False
    assert "devices" not in report["cameras"]
    assert set(report["platform"]["capabilities"]) >= {"lock", "cursor", "hotkeys"}
    assert report["settings"] == {"file_exists": False, "non_default": {}}
    assert report["calibration"] == {"exists": False}
    assert isinstance(report["autostart"]["command"], str)
    # The command is spelled as this copy is run (packages have no "eye-tracker").
    command = diagnostics._redacted_command(cli.cli_command("calibrate"))
    assert report["app"]["command_line"] == diagnostics._redacted_command(cli.cli_command())
    assert (
        f"Not calibrated yet: choose 'Calibrate...' in the tray menu or run '{command}'."
        in report["problems"]
    )
    # The fake offscreen screen is not reported as a one-monitor setup.
    assert not any("Only one monitor" in p for p in report["problems"])


def test_report_paths_are_relative_to_home(env: Path) -> None:
    report = diagnostics.collect_report()
    settings_file = report["paths"]["settings_file"]
    home = str(Path.home())
    if str(paths.settings_file()).lower().startswith(home.lower()):
        assert settings_file.startswith("~")
        assert home not in settings_file
    # The socket name is a hash of the user name: easy to reverse, so left out.
    assert "ipc_name" not in report["paths"]
    assert paths.ipc_name() not in json.dumps(report)
    assert report["paths"]["profile"] == "custom (--config-dir)"


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
    settings.general.backend = "lite"
    settings.presence.away_timeout_s = 120
    settings.save(paths.settings_file())
    report = diagnostics.collect_report()
    assert report["settings"]["valid"] is True
    assert report["settings"]["non_default"] == {
        "general.backend": "lite",
        "presence.away_timeout_s": 120,
    }
    assert report["backends"]["configured"] == "lite"
    assert report["backends"]["active"] == "lite"

    paths.settings_file().write_text("{broken", encoding="utf-8")
    report = diagnostics.collect_report()
    assert report["settings"]["valid"] is False
    assert "The settings file is not valid JSON; defaults are used." in report["problems"]
    # Unlike Settings.load, the doctor never moves the broken file aside.
    assert paths.settings_file().read_text(encoding="utf-8") == "{broken"


def test_calibration_section(env: Path) -> None:
    settings = Settings()
    settings.general.backend = "lite"
    settings.save(paths.settings_file())
    current = diagnostics.current_monitors()
    save_calibration(paths.calibration_file(), _calibration(current, "lite", "yunet-geom-1"))

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
    paths.calibration_file().unlink()
    save_calibration(paths.calibration_file(), _calibration(other, "lite", "yunet-geom-1"))
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
                "a.tflite": {"used_by": ["facemesh"], "present": False},
                "b.onnx": {"used_by": ["facemesh", "lite"], "present": True, "sha256_ok": False},
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
        "No vision backend is available (it needs OpenCV 4.10 or newer with its DNN "
        "module, and the bundled model files).",
        "The model file a.tflite (facemesh backend) is missing.",
        "The model file b.onnx (facemesh, lite backend) is damaged (checksum).",
        "No vision backend is available",
        "Only one monitor detected: switching needs two or more "
        "(walk-away and privacy features still work).",
        "No camera delivered a picture (it may be in use by Eye Tracker itself).",
        "Camera access is blocked by the operating system's privacy settings.",
        "Accessibility permission is missing: keyboard focus cannot follow your gaze.",
        "This session does not allow moving the cursor. On Wayland, sway and Hyprland "
        "work directly; elsewhere install ydotool 1.x and run ydotoold.",
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
    result = diagnostics.run_bench(0.25, str(image), "lite", idle_fps=20)
    assert result["backend"] == "lite"
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
    assert text.startswith("Eye Tracker benchmark - lite backend (yunet-geom-1)")
    assert "max speed" in text
    assert "idle (20 fps)" in text
    assert "frames per second" in text


def test_run_bench_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        diagnostics.run_bench(0, "0", "lite")
    with pytest.raises(CameraError, match="not found"):
        diagnostics.run_bench(0.1, str(tmp_path / "missing.png"), "lite")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not an image")
    with pytest.raises(diagnostics.BenchError):
        diagnostics.run_bench(0.1, str(broken), "lite")


def test_format_bench_with_missing_values() -> None:
    result = {
        "backend": "lite",
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


# -------------------------------------------------------------------- review fixes
def test_calibration_section_reports_every_stored_profile(env: Path) -> None:
    settings = Settings()
    settings.general.backend = "lite"
    settings.save(paths.settings_file())
    current = diagnostics.current_monitors()
    other = [
        Monitor(0, "left", Rect(-1920, 0, 1920, 1080), primary=True),
        Monitor(1, "right", Rect(0, 0, 2560, 1440)),
    ]
    home = _calibration(current, "lite", "yunet-geom-1")
    home.camera = "0"
    save_calibration(paths.calibration_file(), home)
    save_calibration(paths.calibration_file(), _calibration(other, "lite", "yunet-geom-1"))
    calibration = diagnostics.collect_report()["calibration"]
    assert calibration["profiles"] == 2
    # The most recent profile is for another desk, but this desk has one: usable.
    assert calibration["compatible"] is True
    assert calibration["matching_profile"].endswith("camera 0, good")
    assert set(calibration["profile_list"]) == {"1", "2"}


def test_settings_in_the_report_do_not_reveal_the_home_directory(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression (ui_app-09): a video file camera printed the full path."""
    home = tmp_path / "home" / "alice"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    settings = Settings()
    settings.camera.device = str(home / "Videos" / "me-at-desk.mp4")
    settings.privacy.pause_for_apps = [str(home / "apps" / "meeting.exe"), "zoom.exe"]
    settings.save(paths.settings_file())
    report = diagnostics.collect_report()
    non_default = report["settings"]["non_default"]
    assert non_default["camera.device"] == "file me-at-desk.mp4"
    apps = non_default["privacy.pause_for_apps"]
    assert apps[0].startswith("~")
    assert apps[0].endswith("meeting.exe")
    assert apps[1] == "zoom.exe"
    text = diagnostics.format_report(report)
    assert str(home) not in text
    assert "alice" not in text


class _FakeHotkeyManager:
    name = "fake"
    supported = True
    note = None

    def layout_conflict(self, hotkey: object) -> str | None:
        if str(hotkey) == "ctrl+alt+t":
            return "Ctrl+Alt+T is AltGr+T, which types a character on the Turkish Q layout"
        return None


def test_hotkeys_that_type_a_character_are_problems(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eye_tracker.platform import hotkeys

    monkeypatch.setattr(hotkeys, "create_hotkey_manager", _FakeHotkeyManager)
    settings = Settings()
    settings.hotkeys.toggle_tracking = "ctrl+alt+t"  # an Eye Tracker 0.1 default
    settings.save(paths.settings_file())
    report = diagnostics.collect_report()
    conflicts = report["hotkeys"]["layout_conflicts"]
    assert set(conflicts) == {"toggle_tracking"}
    assert (
        "Hotkey toggle_tracking cannot be used: Ctrl+Alt+T is AltGr+T, which types a "
        "character on the Turkish Q layout." in report["problems"]
    )
    settings.hotkeys.enabled = False  # not registered at all: not a problem
    settings.save(paths.settings_file())
    assert not any("cannot be used" in p for p in diagnostics.collect_report()["problems"])


def test_autostart_section_reports_the_status_and_registered_command(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eye_tracker.platform import autostart

    monkeypatch.setattr(autostart, "status", lambda config_dir=None: autostart.Status.STALE)
    monkeypatch.setattr(autostart, "is_enabled", lambda config_dir=None: False)
    monkeypatch.setattr(
        autostart, "registered_command", lambda: [str(Path.home() / "old" / "EyeTracker.exe")]
    )
    report = diagnostics.collect_report()
    section = report["autostart"]
    assert section["status"] == "stale"
    assert section["registered"].startswith("~")
    assert any(p.startswith("Start at login points to a copy") for p in report["problems"])


class _MacLikeServices(PlatformServices):
    name = "fake"

    def capabilities(self) -> dict[str, bool]:
        return dict.fromkeys(super().capabilities(), True)

    def permissions(self) -> dict[str, bool | None]:
        return {"camera": True, "accessibility": False}

    def accessibility_status(self) -> str:
        return "stale"


def test_a_stale_accessibility_grant_is_explained() -> None:
    section = diagnostics._platform_section(_MacLikeServices())
    assert section["accessibility"] == "stale"
    problems = diagnostics._problems({"platform": section})
    assert len(problems) == 1
    assert "granted to an earlier version" in problems[0]
    assert "remove Eye Tracker with '-' and add it again" in problems[0]
    assert problems[0].isascii()


class _MacCliServices(_MacLikeServices):
    """eye-tracker-cli on macOS: the terminal's permission says nothing about the app."""

    def permissions(self) -> dict[str, bool | None]:
        return {"camera": True, "accessibility": None}

    def accessibility_status(self) -> str:
        return "unknown"


def test_the_macos_cli_explains_why_accessibility_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "darwin")
    monkeypatch.setattr(diagnostics.paths, "is_frozen", lambda: True)
    macos = tmp_path / "Eye Tracker.app" / "Contents" / "MacOS"
    monkeypatch.setattr(diagnostics.sys, "executable", str(macos / "eye-tracker-cli"))
    section = diagnostics._platform_section(_MacCliServices())
    assert "accessibility" not in section  # unknown: no verdict either way
    note = section["notes"]["accessibility"]
    assert "the terminal" in note
    assert "Settings > Diagnostics" in note
    assert note.isascii()
    assert diagnostics._problems({"platform": section}) == []

    # The app itself (or a known permission) needs no such note.
    monkeypatch.setattr(diagnostics.sys, "executable", str(macos / "Eye Tracker"))
    assert "notes" not in diagnostics._platform_section(_MacCliServices())
    monkeypatch.setattr(diagnostics.sys, "executable", str(macos / "eye-tracker-cli"))
    assert "notes" not in diagnostics._platform_section(_MacLikeServices())


class _WaylandServices(PlatformServices):
    name = "fake-linux"

    @property
    def is_wayland(self) -> bool:
        return True


def test_linux_notes_explain_camera_release_and_wayland_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    notes = diagnostics._platform_section(_WaylandServices())["notes"]
    assert "PipeWire camera portal" in notes["camera_release"]
    assert "ydotool 1.x" in notes["cursor"]
    assert "hyprctl" in notes["cursor"]
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    assert "notes" not in diagnostics._platform_section(_WaylandServices())


def test_an_installed_mediapipe_is_reported_as_a_problem(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MediaPipe is not used (its models run in OpenCV) and ships a usage logger."""
    real_version = diagnostics.importlib.metadata.version

    def version(name: str) -> str:
        return "0.10.14" if name == "mediapipe" else real_version(name)

    monkeypatch.setattr(diagnostics.importlib.metadata, "version", version)
    report = diagnostics.collect_report()
    assert report["libraries"]["unwanted"] == {"mediapipe": "0.10.14"}
    # (The calibrate hint names this test's --config-dir, which contains "mediapipe".)
    (problem,) = [p for p in report["problems"] if p.startswith("The mediapipe package")]
    assert "does not use it" in problem
    assert "telemetry" in problem
    assert "pip uninstall mediapipe" in problem
    assert problem.isascii()
    # The text report shows it too.
    assert "mediapipe" in diagnostics.format_report(report)


# ------------------------------------------------------------- second review fixes
@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fake home directory whose account name is ``alice``."""
    fake = tmp_path / "home" / "alice"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake))
    return fake


def test_redact_replaces_the_home_directory_anywhere(home: Path) -> None:
    """Regression (r2-ui-app-03): only a leading home directory was replaced."""
    quoted = f'"{home / "AppData" / "Zoom.exe"}"'  # Explorer's "Copy as path"
    assert diagnostics._redact(quoted) == f'"{Path("~") / "AppData" / "Zoom.exe"}"'
    assert diagnostics._redact(f"see {home}") == "see ~"
    assert diagnostics._redact(home.as_posix() + "/x.log") == "~/x.log"  # Qt's slashes
    # A name that merely starts with the account name is someone else's.
    assert diagnostics._redact(f"{home}2") == f"{home}2"
    assert diagnostics._redact(f"{home}.old") == f"{home}.old"
    if sys.platform == "win32":  # case-insensitive file system
        assert diagnostics._redact(str(home).upper()) == "~"
    nested = diagnostics._redact_tree({"a": [str(home / "x"), {"b": (str(home),)}], "n": 3})
    assert nested == {"a": [str(Path("~") / "x"), {"b": ["~"]}], "n": 3}


def test_error_texts_do_not_reveal_the_home_directory(home: Path) -> None:
    def unreadable() -> None:
        # str() of an OSError shows the file name's repr (doubled backslashes).
        raise PermissionError(13, "Permission denied", str(home / "cfg" / "settings.json"))

    error = diagnostics._safe(unreadable)["error"]
    assert error.startswith("PermissionError: [Errno 13] Permission denied")
    assert "alice" not in error
    assert "~" in error


def test_quoted_app_paths_and_camera_serials_stay_out_of_the_report(env: Path, home: Path) -> None:
    settings = Settings()
    zoom = home / "AppData" / "Roaming" / "Zoom" / "bin" / "Zoom.exe"
    settings.privacy.pause_for_apps = [f'"{zoom}"', "obs64.exe"]
    settings.camera.device = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A1B2C3D4-video-index0"
    settings.save(paths.settings_file())
    report = diagnostics.collect_report()
    non_default = report["settings"]["non_default"]
    apps = non_default["privacy.pause_for_apps"]
    assert apps[0].startswith('"~')
    assert apps[0].endswith('Zoom.exe"')
    assert apps[1] == "obs64.exe"
    assert non_default["camera.device"] == "device usb-046d_HD_Pro_Webcam_C920_*-video-index0"
    text = diagnostics.format_report(report)
    assert "alice" not in text
    assert "A1B2C3D4" not in text


def test_describe_camera_devices() -> None:
    assert diagnostics._describe_device("/dev/video2") == "device video2"
    by_id = "/dev/v4l/by-id/usb-Generic_USB2.0_HD_UVC_WebCam_0x0001-video-index1"
    assert (
        diagnostics._describe_device(by_id)
        == "device usb-Generic_USB2.0_HD_UVC_WebCam_*-video-index1"
    )


class _LiveHotkeys(HotkeyManager):
    """The app's own manager: it knows what the OS accepted."""

    supported = True
    name = "live"

    def __init__(self, registered: dict[str, str], errors: dict[str, str]) -> None:
        super().__init__()
        self._live = {name: parse_hotkey(text) for name, text in registered.items()}
        self._errors.update(errors)

    @property
    def registered(self) -> dict[str, Hotkey]:
        return dict(self._live)


@pytest.fixture
def fake_hotkeys(monkeypatch: pytest.MonkeyPatch, env: Path) -> None:
    """Three valid hotkeys, checked by a manager that sees no layout conflict."""
    from eye_tracker.platform import hotkeys

    monkeypatch.setattr(hotkeys, "create_hotkey_manager", _FakeHotkeyManager)
    settings = Settings()
    settings.hotkeys.toggle_tracking = "ctrl+alt+meta+t"
    settings.hotkeys.toggle_privacy = "ctrl+alt+meta+p"
    settings.hotkeys.recalibrate = "ctrl+alt+meta+c"
    settings.save(paths.settings_file())


def test_doctor_shows_why_a_hotkey_was_not_registered(env: Path, fake_hotkeys: None) -> None:
    """Regression (r2-docs-12): doctor never knew what the OS refused."""
    live = _LiveHotkeys(
        {"toggle_tracking": "ctrl+alt+meta+t"},
        {"toggle_privacy": "Ctrl+Alt+Win+P is already used by another application"},
    )
    report = diagnostics.collect_report(hotkey_manager=live)
    assert report["hotkeys"]["registration"] == {
        "toggle_tracking": "registered",
        "toggle_privacy": "not registered: Ctrl+Alt+Win+P is already used by another application",
        "recalibrate": "not registered: reason unknown",
    }
    assert (
        "Hotkey toggle_privacy could not be registered: Ctrl+Alt+Win+P is already used by "
        "another application." in report["problems"]
    )
    text = diagnostics.format_report(report)
    assert "registration.toggle_tracking" in text


def test_doctor_asks_the_running_instance_about_its_hotkeys(env: Path, fake_hotkeys: None) -> None:
    # Nothing running: nobody can know what the OS would accept.
    report = diagnostics.collect_report()
    assert report["hotkeys"]["registration"] == diagnostics._HOTKEYS_NOT_RUNNING
    assert not any("could not be registered" in p for p in report["problems"])

    status: dict[str, Any] = {"state": "tracking"}
    server = ipc.InstanceServer(lambda command: json.dumps(status))
    assert server.listen()
    try:
        # An instance that does not report its hotkeys (an older version).
        report = diagnostics.collect_report()
        assert report["hotkeys"]["registration"] == diagnostics._HOTKEYS_NOT_REPORTED

        status["hotkeys"] = {
            "registered": ["toggle_tracking", "recalibrate"],
            "errors": {"toggle_privacy": "Ctrl+Alt+Win+P is already used by another application"},
        }
        report = diagnostics.collect_report()
        assert report["hotkeys"]["registration"] == {
            "toggle_tracking": "registered",
            "toggle_privacy": "not registered: Ctrl+Alt+Win+P is already used by another "
            "application",
            "recalibrate": "registered",
        }
    finally:
        server.close()


class _LinuxServices(PlatformServices):
    name = "fake-linux"

    def lock_methods(self) -> list[str]:
        return ["loginctl lock-session", "xdg-screensaver lock"]


def test_linux_report_says_which_lock_tools_exist_and_that_locking_is_untested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (r2-docs-13): "lock yes" read as "locking works"."""
    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    section = diagnostics._platform_section(_LinuxServices())
    assert section["lock_methods"] == ["loginctl lock-session", "xdg-screensaver lock"]
    note = section["notes"]["lock"]
    assert "known when it is tried" in note
    assert "Could not lock the screen" in note
    assert note.isascii()
    # Platforms without the query report the capability alone.
    assert "lock_methods" not in diagnostics._platform_section(PlatformServices())
