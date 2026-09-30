"""Tests for eye_tracker.cli.

The tray app itself is never started here (``run_app`` is replaced); the other
subcommands run for real against a temporary config directory. Autostart is
faked so the real login entry is never touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from eye_tracker import __version__, app, cli, ipc, paths
from eye_tracker.config import Settings
from eye_tracker.logging_setup import shutdown_logging
from eye_tracker.platform import autostart


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    shutdown_logging()


@pytest.fixture
def env(qapp: Any, app_dirs: Path) -> Path:
    """A QApplication exists (ipc/doctor would otherwise create a non-GUI one)."""
    return app_dirs


@pytest.fixture
def run_calls(monkeypatch: pytest.MonkeyPatch) -> list[argparse.Namespace]:
    calls: list[argparse.Namespace] = []

    def fake_run_app(args: argparse.Namespace) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(app, "run_app", fake_run_app)
    return calls


@pytest.fixture
def instance(env: Path) -> Iterator[Callable[[Callable[[str], str]], ipc.InstanceServer]]:
    servers: list[ipc.InstanceServer] = []

    def start(handler: Callable[[str], str]) -> ipc.InstanceServer:
        server = ipc.InstanceServer(handler)
        assert server.listen()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


class FakeAutostart:
    EXE = "C:\\Program Files\\Eye Tracker\\EyeTracker.exe"

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, supported: bool = True) -> None:
        self.enabled = False
        self.calls: list[str] = []
        self.supported = supported
        self.fail: str | None = None
        #: What status() reports while not enabled.
        self.idle_status = autostart.Status.DISABLED
        #: The ``config_dir`` of every status/enable/disable call.
        self.profiles: list[Path | None] = []
        #: What registered_command() reports (``None``: this copy when enabled).
        self.registered: list[str] | None = None
        monkeypatch.setattr(autostart, "is_supported", lambda: self.supported)
        monkeypatch.setattr(autostart, "is_enabled", lambda config_dir=None: self.enabled)
        monkeypatch.setattr(autostart, "status", self.status)
        monkeypatch.setattr(autostart, "enable", self.enable)
        monkeypatch.setattr(autostart, "disable", self.disable)
        monkeypatch.setattr(autostart, "location", lambda: "HKCU\\Run\\EyeTracker")
        monkeypatch.setattr(autostart, "registered_command", self.registered_command)
        monkeypatch.setattr(autostart, "launch_command", self.launch_command)

    def launch_command(self, background: bool = True, config_dir: Path | None = None) -> list[str]:
        profile = ["--config-dir", str(config_dir)] if config_dir else []
        return [self.EXE, *profile, *(["--background"] if background else [])]

    def registered_command(self) -> list[str] | None:
        if self.registered is not None:
            return self.registered
        return self.launch_command() if self.enabled else None

    def status(self, config_dir: Path | None = None) -> autostart.Status:
        self.profiles.append(config_dir)
        return autostart.Status.ENABLED if self.enabled else self.idle_status

    def enable(self, background: bool = True, config_dir: Path | None = None) -> None:
        self.calls.append(f"enable(background={background})")
        self.profiles.append(config_dir)
        if self.fail:
            raise autostart.AutostartError(self.fail)
        self.enabled = True

    def disable(self, config_dir: Path | None = None) -> None:
        self.calls.append("disable")
        self.profiles.append(config_dir)
        self.enabled = False


# ------------------------------------------------------------------------- basics
def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--version"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == f"Eye Tracker {__version__}"


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    for command in ("run", "calibrate", "doctor", "bench", "ctl", "autostart", "reset"):
        assert command in out
    for option in ("--config-dir", "--log-level", "--camera", "--backend", "--background"):
        assert option in out


def test_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["frobnicate"]) == cli.EXIT_USAGE
    assert cli.main(["ctl", "format-disk"]) == cli.EXIT_USAGE
    assert cli.main(["bench", "--seconds", "-1"]) == cli.EXIT_USAGE
    assert cli.main(["bench", "--seconds", "nan"]) == cli.EXIT_USAGE
    assert cli.main(["--log-level", "loud"]) == cli.EXIT_USAGE
    assert "usage:" in capsys.readouterr().err


def test_ctl_commands_mirror_ipc() -> None:
    assert set(cli.CTL_COMMANDS) == ipc.COMMANDS


def test_python_dash_m_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "eye_tracker", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"Eye Tracker {__version__}"


def test_gui_main_survives_missing_std_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    # pythonw / windowed builds: no console, sys.stdout and sys.stderr are None.
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    try:
        assert cli.gui_main(["--version"]) == cli.EXIT_OK
        assert sys.stdout is not None
        assert sys.stderr is not None
    finally:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.close()


# ---------------------------------------------------------------------------- run
def test_run_is_the_default(run_calls: list[argparse.Namespace], env: Path) -> None:
    assert cli.main([]) == 0
    assert cli.main(["--background", "--camera", "2", "--backend", "lite"]) == 0
    assert cli.main(["run", "--calibrate", "--log-level", "debug"]) == 0
    first, second, third = run_calls
    assert first.command is None
    assert not first.background
    assert not first.calibrate
    assert (second.background, second.camera, second.backend) == (True, "2", "lite")
    assert third.command == "run"
    assert third.calibrate
    assert third.log_level == "DEBUG"


def test_global_options_work_before_and_after_the_subcommand(
    run_calls: list[argparse.Namespace], env: Path
) -> None:
    assert cli.main(["--camera", "1", "run"]) == 0
    assert cli.main(["run", "--camera", "3"]) == 0
    assert [a.camera for a in run_calls] == ["1", "3"]


def test_calibrate_subcommand_runs_the_app_in_calibration_mode(
    run_calls: list[argparse.Namespace], env: Path
) -> None:
    assert cli.main(["calibrate", "--background"]) == 0
    (args,) = run_calls
    assert args.command == "calibrate"
    assert args.calibrate is True
    assert args.background is True


def test_config_dir_option(run_calls: list[argparse.Namespace], env: Path, tmp_path: Path) -> None:
    portable = tmp_path / "portable"
    assert cli.main(["--config-dir", str(portable)]) == 0
    assert paths.config_dir() == portable.resolve()


def test_keyboard_interrupt_exit_code(monkeypatch: pytest.MonkeyPatch, env: Path) -> None:
    def interrupted(args: argparse.Namespace) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(app, "run_app", interrupted)
    assert cli.main([]) == cli.EXIT_INTERRUPTED


# ------------------------------------------------------------------------- doctor
def test_doctor_json(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor", "--json"]) == cli.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    for key in ("app", "system", "qt", "monitors", "backends", "cameras", "paths", "problems"):
        assert key in report
    assert report["app"]["version"] == __version__
    assert report["cameras"]["probed"] is False
    assert isinstance(report["problems"], list)


def test_doctor_text(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor", "--log-level", "error"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith(f"Eye Tracker {__version__} - diagnostics")
    assert "Vision backends" in out


# -------------------------------------------------------------------------- bench
def _image(path: Path) -> Path:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[40:80, 60:100] = (40, 160, 220)
    assert cv2.imwrite(str(path), image)
    return path


def test_bench_json_on_an_image(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = _image(tmp_path / "still.png")
    code = cli.main(
        ["bench", "--seconds", "0.2", "--camera", str(image), "--backend", "lite", "--json"]
    )
    captured = capsys.readouterr()
    assert code == cli.EXIT_OK, captured.err
    result = json.loads(captured.out)
    assert result["backend"] == "lite"
    assert result["device"] == "file still.png"
    assert set(result["modes"]) == {"max", "idle"}
    assert "Benchmarking" in captured.err


def test_bench_text_uses_the_settings(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings()
    settings.camera.device = str(_image(tmp_path / "configured.png"))
    settings.general.backend = "lite"
    settings.save(paths.settings_file())
    assert cli.main(["bench", "--seconds", "0.2"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("Eye Tracker benchmark - lite backend")
    assert "file configured.png" in out


def test_bench_reports_a_missing_source(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.mp4"
    assert cli.main(["bench", "--seconds", "0.2", "--camera", str(missing)]) == cli.EXIT_ERROR
    assert "error: Video source not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------- ctl
def test_ctl_without_an_instance(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ctl", "status"]) == cli.EXIT_NOT_RUNNING
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Eye Tracker is not running" in captured.err


def test_ctl_with_an_instance(
    instance: Callable[[Callable[[str], str]], ipc.InstanceServer],
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: list[str] = []

    def handler(command: str) -> str:
        received.append(command)
        if command == "status":
            return json.dumps({"state": "tracking"})
        if command == "calibrate":
            return "error: calibration unavailable"
        return "ok"

    instance(handler)
    assert cli.main(["ctl", "status"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"state": "tracking"}
    assert cli.main(["ctl", "privacy-toggle"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == "ok"
    assert cli.main(["ctl", "calibrate"]) == cli.EXIT_ERROR
    assert "calibration unavailable" in capsys.readouterr().err
    assert received == ["status", "privacy-toggle", "calibrate"]


def test_ctl_with_an_instance_that_does_not_answer(
    instance: Callable[[Callable[[str], str]], ipc.InstanceServer],
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = instance(lambda command: "ok")
    # A slot that never replies (and no handler): the client times out.
    server._handler = None
    server.command_received.connect(lambda command, reply: None)
    assert cli.main(["ctl", "pause", "--timeout", "150"]) == cli.EXIT_ERROR
    assert "did not answer within 150 ms" in capsys.readouterr().err


# ---------------------------------------------------------------------- autostart
def test_autostart_status_enable_disable(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeAutostart(monkeypatch)
    assert cli.main(["autostart"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Start at login: disabled" in out
    assert "HKCU\\Run\\EyeTracker" in out
    assert "EyeTracker.exe" in out
    assert "--background" in out

    assert cli.main(["autostart", "enable"]) == cli.EXIT_OK
    assert fake.enabled
    assert "Start at login enabled" in capsys.readouterr().out
    assert cli.main(["autostart", "status"]) == cli.EXIT_OK
    assert "Start at login: enabled" in capsys.readouterr().out

    assert cli.main(["autostart", "disable"]) == cli.EXIT_OK
    assert not fake.enabled
    assert "Start at login disabled." in capsys.readouterr().out
    assert fake.calls == ["enable(background=True)", "disable"]


def test_autostart_failure_and_unsupported(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeAutostart(monkeypatch)
    fake.fail = "access denied"
    assert cli.main(["autostart", "enable"]) == cli.EXIT_ERROR
    assert "error: access denied" in capsys.readouterr().err

    fake.supported = False
    assert cli.main(["autostart", "status"]) == cli.EXIT_OK
    assert "not supported" in capsys.readouterr().out
    assert cli.main(["autostart", "enable"]) == cli.EXIT_ERROR
    assert "not supported" in capsys.readouterr().err
    assert fake.calls == ["enable(background=True)"]


def test_autostart_follows_the_config_dir(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression (ui_app-11): a portable instance registered the default profile."""
    fake = FakeAutostart(monkeypatch)
    portable = tmp_path / "portable"
    assert cli.main(["--config-dir", str(portable), "autostart", "enable"]) == cli.EXIT_OK
    assert fake.profiles == [portable]
    assert cli.main(["autostart", "status", "--config-dir", str(portable)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Start at login: enabled" in out
    assert f"--config-dir {portable}" in out.replace('"', "")
    assert cli.main(["autostart", "disable", "--config-dir", str(portable)]) == cli.EXIT_OK
    assert fake.profiles[-1] == portable
    # Without --config-dir the running process's profile applies (None).
    fake.profiles.clear()
    paths.set_base_override(None)
    try:
        assert cli.main(["autostart", "enable"]) == cli.EXIT_OK
    finally:
        paths.set_base_override(env)
    assert fake.profiles == [None]


def test_autostart_status_explains_entries_for_other_copies(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeAutostart(monkeypatch)
    fake.idle_status = autostart.Status.STALE
    fake.registered = ["/opt/old/eye-tracker", "--background"]
    assert cli.main(["autostart"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Start at login: stale\n" in out  # the status value itself
    assert "Broken" in out
    assert "Registered: /opt/old/eye-tracker --background" in out
    fake.idle_status = autostart.Status.OTHER_PROFILE
    assert cli.main(["autostart"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Start at login: other-profile\n" in out
    assert "another profile" in out
    assert cli.main(["autostart", "disable"]) == cli.EXIT_OK
    assert "left unchanged" in capsys.readouterr().out


def test_backend_option_takes_current_and_legacy_names(
    run_calls: list[argparse.Namespace], env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.BACKEND_CHOICES == ("auto", "facemesh", "lite")
    for given, expected in (
        ("facemesh", "facemesh"),
        ("LITE", "lite"),
        ("mediapipe", "facemesh"),  # Eye Tracker 0.1 names in old shortcuts
        ("opencv", "lite"),
    ):
        assert cli.main(["--backend", given]) == 0
        assert run_calls[-1].backend == expected
    assert cli.main(["--backend", "tflite"]) == cli.EXIT_USAGE
    assert "invalid choice" in capsys.readouterr().err


# -------------------------------------------------------------------------- reset
def _write_state() -> tuple[Path, Path]:
    settings_file = paths.settings_file()
    calibration_file = paths.calibration_file()
    Settings().save(settings_file)
    calibration_file.write_text("{}", encoding="utf-8")
    return settings_file, calibration_file


def test_reset_calibration_only(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    settings_file, calibration_file = _write_state()
    assert cli.main(["reset", "--calibration"]) == cli.EXIT_OK
    assert settings_file.exists()
    assert not calibration_file.exists()
    assert "Deleted" in capsys.readouterr().out


def test_reset_all(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    settings_file, calibration_file = _write_state()
    corrupt = settings_file.with_suffix(".json.corrupt")
    corrupt.write_text("{", encoding="utf-8")
    assert cli.main(["reset", "--all"]) == cli.EXIT_OK
    assert not settings_file.exists()
    assert not corrupt.exists()
    assert not calibration_file.exists()
    assert cli.main(["reset", "--settings"]) == cli.EXIT_OK
    assert "Nothing to reset." in capsys.readouterr().out


def test_reset_needs_a_target(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["reset"]) == cli.EXIT_USAGE
    assert "--calibration, --settings or --all" in capsys.readouterr().err


def test_reset_refuses_while_the_app_runs(
    instance: Callable[[Callable[[str], str]], ipc.InstanceServer],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _settings_file, calibration_file = _write_state()
    instance(lambda command: "ok")
    assert cli.main(["reset", "--all"]) == cli.EXIT_ERROR
    assert "Quit it first" in capsys.readouterr().err
    assert calibration_file.exists()
